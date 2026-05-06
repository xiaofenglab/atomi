#!/usr/bin/env python3
"""
Build a dataset with:
  - full backbone categories kept after split:
      neareq, phonopy, force_spread
  - prefail added adaptively by oversampling subgroup pools
  - realistic validation from held-out frames
  - manifest and build report

Main policy
-----------
1. Read and normalize all source extxyz files.
2. Split each source into train/valid first.
3. Keep ALL backbone train/valid frames:
      neareq, phonopy, force_spread
4. Build prefail training target as:
      prefail_train_target = prefail_train_fraction_of_backbone * backbone_train_total
5. Distribute prefail training target across prefail subgroups using subgroup weights.
6. Oversample prefail TRAIN only as needed.
7. Validation uses held-out prefail frames without aggressive oversampling.
   By default, prefail validation target is:
      prefail_valid_fraction_of_backbone * backbone_valid_total
   but capped by actual available held-out prefail frames.

So:
- backbone is mostly fixed and realistic
- prefail is the adjustable frontier
- validation remains a real held-out test

Example
-------
python3 build_v8_datasets.py \
  --neareq neareq_train.extxyz \
  --phonopy phonopy_V1.000.extxyz phonopy_V1.002.extxyz phonopy_V1.005.extxyz \
  --force-spread forces_train.extxyz \
  --prefail-group prefail450=prefail_450K.extxyz \
  --prefail-group prefail500=prefail_500K.extxyz \
  --prefail-group prefail600=mlacs_600K_train.extxyz \
  --prefail-group prefail700=prefail_700K.extxyz \
  --prefail-weight prefail450=1.0 \
  --prefail-weight prefail500=1.0 \
  --prefail-weight prefail600=2.0 \
  --prefail-weight prefail700=1.5 \
  --neareq-valid-ratio 0.12 \
  --phonopy-valid-ratio 0.12 \
  --force-spread-valid-ratio 0.12 \
  --prefail-valid-ratio 0.20 \
  --prefail-train-fraction-of-backbone 0.30 \
  --prefail-valid-fraction-of-backbone 0.15 \
  --allow-oversample-prefail \
  --train-out training_adaptive.extxyz \
  --valid-out validation_adaptive.extxyz \
  --manifest-out dataset_manifest_adaptive.csv \
  --report-out dataset_build_report_adaptive.txt \
  --seed 20260422

Notes
-----
- No manual top-level train-size / valid-size needed.
- Final train size is:
      backbone_train_total + prefail_train_target
- Final valid size is:
      backbone_valid_total + prefail_valid_target_actual
- If a prefail subgroup has no explicit weight, it defaults to 1.0.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


# =========================================================
# RAW EXTXYZ PARSING
# =========================================================

def read_logical_header(lines, i):
    parts = []
    line = lines[i].rstrip("\n")
    i += 1
    while True:
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            parts.append(stripped[:-1].strip())
            if i >= len(lines):
                break
            line = lines[i].rstrip("\n")
            i += 1
        else:
            parts.append(stripped)
            break
    return " ".join(p for p in parts if p), i


def parse_header_keyvals(header):
    pat = re.compile(r'(\S+)=(".*?"|\S+)')
    out = {}
    for m in pat.finditer(header):
        k = m.group(1)
        v = m.group(2)
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            v = v[1:-1]
        out[k] = v
    return out


def parse_frames_raw(path: str, top_category: str, subgroup: str):
    path_obj = Path(path)
    lines = path_obj.read_text(errors="ignore").splitlines()
    frames = []
    i = 0

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        natoms = int(lines[i].strip())
        i += 1

        header, i = read_logical_header(lines, i)
        if i + natoms > len(lines):
            raise ValueError(f"Unexpected EOF while reading frame from {path}")

        atom_lines = lines[i:i + natoms]
        i += natoms

        hdr = parse_header_keyvals(header)

        frames.append({
            "natoms": natoms,
            "header": header,
            "atom_lines": atom_lines,
            "dataset_tag": hdr.get("dataset_tag", ""),
            "run_dir": hdr.get("run_dir", ""),
            "config_tag": hdr.get("config_tag", ""),
            "config_id": hdr.get("config_id", ""),
            "source_file": str(path_obj),
            "top_category": top_category,
            "subgroup": subgroup,
            "occurrence_id": 1,
            "is_oversampled": False,
        })

    return frames


def write_frames_raw(path: str, frames):
    with open(path, "w") as f:
        for fr in frames:
            f.write(f"{fr['natoms']}\n")
            f.write(fr["header"].rstrip() + "\n")
            for line in fr["atom_lines"]:
                f.write(line.rstrip() + "\n")


# =========================================================
# LABEL NORMALIZATION
# =========================================================

def re_sub_dataset_tag(header: str, tag: str):
    return re.sub(r'dataset_tag=("[^"]*"|\S+)', f'dataset_tag={tag}', header)


def set_dataset_tag_in_header(header: str, tag: str):
    if "dataset_tag=" in header:
        return re_sub_dataset_tag(header, tag)
    return header + f" dataset_tag={tag}"


def rename_energy_key(header: str):
    if "REF_energy=" in header:
        return header
    return re.sub(r'(?<!\S)energy=("[^"]*"|\S+)', r'REF_energy=\1', header)


def rename_stress_key(header: str):
    if "REF_stress=" in header:
        return header
    return re.sub(r'(?<!\S)stress=("[^"]*"|\S+)', r'REF_stress=\1', header)


def rename_properties_field(header: str):
    m = re.search(r'Properties=("[^"]*"|\S+)', header)
    if not m:
        return header

    raw_val = m.group(1)
    quoted = raw_val.startswith('"') and raw_val.endswith('"')
    props = raw_val[1:-1] if quoted else raw_val

    toks = props.split(":")
    if len(toks) % 3 != 0:
        return header

    out = []
    for j in range(0, len(toks), 3):
        name = toks[j]
        kind = toks[j + 1]
        ncols = toks[j + 2]
        if name == "forces":
            name = "REF_forces"
        out.extend([name, kind, ncols])

    new_props = ":".join(out)
    new_raw = f'"{new_props}"' if quoted else new_props
    return header[:m.start(1)] + new_raw + header[m.end(1):]


def make_future_safe_header(header: str):
    header = rename_energy_key(header)
    header = rename_stress_key(header)
    header = rename_properties_field(header)
    return header


def normalize_frames(frames, dataset_tag: str):
    out = []
    for fr in frames:
        new_fr = dict(fr)
        new_fr["header"] = make_future_safe_header(set_dataset_tag_in_header(fr["header"], dataset_tag))
        new_fr["dataset_tag"] = dataset_tag
        out.append(new_fr)
    return out


# =========================================================
# HELPERS
# =========================================================

def deep_copy_frame(fr):
    out = dict(fr)
    out["header"] = str(fr["header"])
    out["atom_lines"] = list(fr["atom_lines"])
    return out


def split_frames(frames, valid_ratio: float, seed: int):
    if valid_ratio <= 0.0:
        return list(frames), []
    if valid_ratio >= 1.0:
        raise ValueError(f"valid_ratio must be < 1.0, got {valid_ratio}")
    n = len(frames)
    if n < 2:
        raise ValueError("Need at least 2 frames to split.")

    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_valid = max(1, int(round(n * valid_ratio)))
    n_valid = min(n_valid, n - 1)

    valid_idx = set(idx[:n_valid])
    train_frames = [frames[i] for i in range(n) if i not in valid_idx]
    valid_frames = [frames[i] for i in range(n) if i in valid_idx]
    return train_frames, valid_frames


def summarize_by_key(frames, key, title):
    c = Counter(fr.get(key, "UNKNOWN") or "UNKNOWN" for fr in frames)
    print(f"{title}: {len(frames)} frames")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")


def check_frame_has_ref_energy(header):
    return "REF_energy=" in header


def check_frame_has_ref_stress(header):
    return "REF_stress=" in header


def check_frame_has_ref_forces(header):
    m = re.search(r'Properties=("[^"]*"|\S+)', header)
    if not m:
        return False
    raw_val = m.group(1)
    props = raw_val[1:-1] if raw_val.startswith('"') and raw_val.endswith('"') else raw_val
    return "REF_forces:R:3" in props


def validate_frames(frames, name):
    missing_e = 0
    missing_f = 0
    missing_s = 0
    for fr in frames:
        if not check_frame_has_ref_energy(fr["header"]):
            missing_e += 1
        if not check_frame_has_ref_forces(fr["header"]):
            missing_f += 1
        if not check_frame_has_ref_stress(fr["header"]):
            missing_s += 1
    print(f"{name} label check:")
    print(f"  missing REF_energy : {missing_e}")
    print(f"  missing REF_forces : {missing_f}")
    print(f"  missing REF_stress : {missing_s}")
    if missing_e or missing_f or missing_s:
        raise RuntimeError(
            f"{name} contains unlabeled frames "
            f"(missing REF_energy={missing_e}, REF_forces={missing_f}, REF_stress={missing_s})"
        )


def sample_without_oversampling(frames, n_target, rng):
    n_have = len(frames)
    if n_target >= n_have:
        return [deep_copy_frame(fr) for fr in frames], {
            "mode": "keep_all",
            "n_input": n_have,
            "n_output": n_have,
            "factor": 1.0,
        }

    idx = list(range(n_have))
    rng.shuffle(idx)
    idx = idx[:n_target]
    chosen = [deep_copy_frame(frames[i]) for i in idx]

    counts = defaultdict(int)
    for fr in chosen:
        key = fr.get("run_dir", f"__no_run_dir__{id(fr)}")
        counts[key] += 1
        fr["occurrence_id"] = counts[key]
        fr["is_oversampled"] = False

    return chosen, {
        "mode": "downsample",
        "n_input": n_have,
        "n_output": len(chosen),
        "factor": len(chosen) / n_have if n_have else 0.0,
    }


def sample_with_optional_oversampling(frames, n_target, rng, allow_oversample):
    n_have = len(frames)
    if n_have == 0:
        return [], {
            "mode": "empty",
            "n_input": 0,
            "n_output": 0,
            "factor": 0.0,
        }

    if n_target <= n_have:
        return sample_without_oversampling(frames, n_target, rng)

    if not allow_oversample:
        chosen = [deep_copy_frame(fr) for fr in frames]
        counts = defaultdict(int)
        for fr in chosen:
            key = fr.get("run_dir", f"__no_run_dir__{id(fr)}")
            counts[key] += 1
            fr["occurrence_id"] = counts[key]
            fr["is_oversampled"] = False
        return chosen, {
            "mode": "keep_all_insufficient",
            "n_input": n_have,
            "n_output": len(chosen),
            "factor": 1.0,
        }

    chosen = [deep_copy_frame(fr) for fr in frames]
    counts = defaultdict(int)
    for fr in chosen:
        key = fr.get("run_dir", f"__no_run_dir__{id(fr)}")
        counts[key] += 1
        fr["occurrence_id"] = counts[key]
        fr["is_oversampled"] = False

    extras = []
    for _ in range(n_target - n_have):
        fr = deep_copy_frame(rng.choice(frames))
        key = fr.get("run_dir", f"__no_run_dir__{id(fr)}")
        counts[key] += 1
        fr["occurrence_id"] = counts[key]
        fr["is_oversampled"] = True
        extras.append(fr)

    chosen.extend(extras)
    return chosen, {
        "mode": "oversample",
        "n_input": n_have,
        "n_output": len(chosen),
        "factor": len(chosen) / n_have,
    }


# =========================================================
# PREFAIL DISTRIBUTION
# =========================================================

def parse_prefail_group_item(s: str):
    if "=" not in s:
        raise ValueError(f"Bad --prefail-group item, expected subgroup=path: {s}")
    subgroup, path = s.split("=", 1)
    subgroup = subgroup.strip()
    path = path.strip()
    if not subgroup or not path:
        raise ValueError(f"Bad --prefail-group item: {s}")
    return subgroup, path


def parse_prefail_weight_item(s: str):
    if "=" not in s:
        raise ValueError(f"Bad --prefail-weight item, expected subgroup=weight: {s}")
    subgroup, weight = s.split("=", 1)
    subgroup = subgroup.strip()
    weight = float(weight.strip())
    if weight <= 0:
        raise ValueError(f"Prefail weight must be > 0, got {weight} for {subgroup}")
    return subgroup, weight


def weighted_targets(total_target, subgroup_names, subgroup_weights):
    """
    Integer targets proportional to subgroup weights.
    """
    if total_target <= 0:
        return {sg: 0 for sg in subgroup_names}

    weights = [subgroup_weights.get(sg, 1.0) for sg in subgroup_names]
    total_w = sum(weights)
    raw = [total_target * w / total_w for w in weights]

    base = [int(math.floor(x)) for x in raw]
    remainder = total_target - sum(base)

    frac_order = sorted(
        range(len(subgroup_names)),
        key=lambda i: (raw[i] - base[i], subgroup_names[i]),
        reverse=True,
    )

    for i in frac_order[:remainder]:
        base[i] += 1

    return {sg: base[i] for i, sg in enumerate(subgroup_names)}


def build_prefail_train(prefail_train_groups, total_target, subgroup_weights, rng, allow_oversample):
    subgroup_names = sorted(prefail_train_groups.keys())
    targets = weighted_targets(total_target, subgroup_names, subgroup_weights)

    out = []
    summary = {}
    for sg in subgroup_names:
        chosen, info = sample_with_optional_oversampling(
            prefail_train_groups[sg],
            targets[sg],
            rng,
            allow_oversample=allow_oversample,
        )
        out.extend(chosen)
        summary[sg] = {"target": targets[sg], **info}

    rng.shuffle(out)
    return out, summary


def build_prefail_valid(prefail_valid_groups, desired_target, subgroup_weights, rng):
    """
    Validation should remain realistic:
    - no oversampling
    - if desired_target > available, just use all available
    - otherwise downsample proportionally by subgroup weights
    """
    subgroup_names = sorted(prefail_valid_groups.keys())
    available_total = sum(len(prefail_valid_groups[sg]) for sg in subgroup_names)
    actual_target = min(desired_target, available_total)

    targets = weighted_targets(actual_target, subgroup_names, subgroup_weights)

    out = []
    summary = {}
    for sg in subgroup_names:
        chosen, info = sample_without_oversampling(
            prefail_valid_groups[sg],
            targets[sg],
            rng,
        )
        out.extend(chosen)
        summary[sg] = {"target": targets[sg], **info}

    rng.shuffle(out)
    return out, summary, actual_target


def print_summary_block(title, summary):
    print(f"\n{title}")
    for k, row in summary.items():
        print(
            f"  {k:18s} target={row['target']:4d} "
            f"mode={row['mode']:20s} in={row['n_input']:4d} "
            f"out={row['n_output']:4d} factor={row['factor']:.3f}"
        )


# =========================================================
# MANIFEST / REPORT
# =========================================================

def write_manifest_csv(path, train_frames, valid_frames):
    fieldnames = [
        "split",
        "final_index",
        "top_category",
        "subgroup",
        "dataset_tag",
        "source_file",
        "run_dir",
        "config_tag",
        "config_id",
        "occurrence_id",
        "is_oversampled",
        "natoms",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for split_name, frames in [("train", train_frames), ("valid", valid_frames)]:
            for i, fr in enumerate(frames):
                w.writerow({
                    "split": split_name,
                    "final_index": i,
                    "top_category": fr.get("top_category", ""),
                    "subgroup": fr.get("subgroup", ""),
                    "dataset_tag": fr.get("dataset_tag", ""),
                    "source_file": fr.get("source_file", ""),
                    "run_dir": fr.get("run_dir", ""),
                    "config_tag": fr.get("config_tag", ""),
                    "config_id": fr.get("config_id", ""),
                    "occurrence_id": fr.get("occurrence_id", 1),
                    "is_oversampled": int(bool(fr.get("is_oversampled", False))),
                    "natoms": fr.get("natoms", ""),
                })


def write_report(path, args, backbone_train_total, backbone_valid_total,
                 prefail_train_target, prefail_valid_desired, prefail_valid_actual,
                 prefail_train_summary, prefail_valid_summary,
                 train_frames, valid_frames):
    lines = []
    lines.append("Adaptive prefail dataset build report")
    lines.append("")
    lines.append(f"train_out                        : {args.train_out}")
    lines.append(f"valid_out                        : {args.valid_out}")
    lines.append(f"manifest_out                     : {args.manifest_out}")
    lines.append(f"seed                             : {args.seed}")
    lines.append(f"prefail_train_fraction_backbone  : {args.prefail_train_fraction_of_backbone}")
    lines.append(f"prefail_valid_fraction_backbone  : {args.prefail_valid_fraction_of_backbone}")
    lines.append(f"allow_oversample_prefail         : {int(bool(args.allow_oversample_prefail))}")
    lines.append("")
    lines.append(f"backbone_train_total             : {backbone_train_total}")
    lines.append(f"backbone_valid_total             : {backbone_valid_total}")
    lines.append(f"prefail_train_target             : {prefail_train_target}")
    lines.append(f"prefail_valid_desired            : {prefail_valid_desired}")
    lines.append(f"prefail_valid_actual             : {prefail_valid_actual}")
    lines.append(f"final_train_size                 : {len(train_frames)}")
    lines.append(f"final_valid_size                 : {len(valid_frames)}")
    lines.append("")
    lines.append("Prefail train subgroup summary")
    for k, row in prefail_train_summary.items():
        lines.append(
            f"  {k:18s} target={row['target']:4d} mode={row['mode']:20s} "
            f"in={row['n_input']:4d} out={row['n_output']:4d} factor={row['factor']:.3f}"
        )
    lines.append("")
    lines.append("Prefail valid subgroup summary")
    for k, row in prefail_valid_summary.items():
        lines.append(
            f"  {k:18s} target={row['target']:4d} mode={row['mode']:20s} "
            f"in={row['n_input']:4d} out={row['n_output']:4d} factor={row['factor']:.3f}"
        )
    lines.append("")

    for frames, title1, title2 in [
        (train_frames, "Final train top_category counts", "Final train subgroup counts"),
        (valid_frames, "Final valid top_category counts", "Final valid subgroup counts"),
    ]:
        lines.append(title1)
        c = Counter(fr.get("top_category", "UNKNOWN") or "UNKNOWN" for fr in frames)
        for k, v in sorted(c.items()):
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(title2)
        c = Counter(fr.get("subgroup", "UNKNOWN") or "UNKNOWN" for fr in frames)
        for k, v in sorted(c.items()):
            lines.append(f"  {k}: {v}")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")


# =========================================================
# MAIN
# =========================================================

def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        prog="mace-build-dataset",
        description="Build adaptive train/valid datasets with fixed backbone and oversampled prefail frontier."
    )

    ap.add_argument("--neareq", nargs="+", required=True, help="One or more neareq extxyz files")
    ap.add_argument("--phonopy", nargs="+", required=True, help="One or more phonopy extxyz files")
    ap.add_argument("--force-spread", nargs="+", required=True, help="One or more force_spread extxyz files")
    ap.add_argument("--prefail-group", action="append", required=True,
                help="Repeat as subgroup=path, e.g. --prefail-group prefail450=file.extxyz")

    ap.add_argument("--prefail-weight", action="append", default=[],
                help="Repeat as subgroup=weight, e.g. --prefail-weight prefail600=2.5")

    ap.add_argument("--neareq-valid-ratio", type=float, default=0.12)
    ap.add_argument("--phonopy-valid-ratio", type=float, default=0.12)
    ap.add_argument("--force-spread-valid-ratio", type=float, default=0.12)
    ap.add_argument("--prefail-valid-ratio", type=float, default=0.20)

    ap.add_argument("--prefail-train-fraction-of-backbone", type=float, default=0.30,
                    help="Target prefail TRAIN size as a fraction of backbone TRAIN total")
    ap.add_argument("--prefail-valid-fraction-of-backbone", type=float, default=0.15,
                    help="Desired prefail VALID size as a fraction of backbone VALID total")

    ap.add_argument("--allow-oversample-prefail", action="store_true")

    ap.add_argument("--train-out", default="training_adaptive.extxyz")
    ap.add_argument("--valid-out", default="validation_adaptive.extxyz")
    ap.add_argument("--manifest-out", default="dataset_manifest_adaptive.csv")
    ap.add_argument("--report-out", default="dataset_build_report_adaptive.txt")
    ap.add_argument("--seed", type=int, default=12345)

    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    # subgroup weights
    subgroup_weights = {}
    for item in args.prefail_weight:
        sg, w = parse_prefail_weight_item(item)
        subgroup_weights[sg] = w

    # ------------------------------
    # Load/normalize backbone
    # ------------------------------
    neareq = []
    for f in args.neareq:
        neareq.extend(normalize_frames(parse_frames_raw(f, "neareq", "neareq"), "neareq"))

    phonopy = []
    for f in args.phonopy:
        subgroup = Path(f).stem
        phonopy.extend(normalize_frames(parse_frames_raw(f, "phonopy", subgroup), "phonopy"))

    force_spread = []
    for f in args.force_spread:
        subgroup = Path(f).stem
        force_spread.extend(normalize_frames(parse_frames_raw(f, "force_spread", subgroup), "force_spread"))

    # ------------------------------
    # Load/normalize prefail groups
    # ------------------------------
    prefail_groups = defaultdict(list)
    for item in args.prefail_group:
        subgroup, path = parse_prefail_group_item(item)
        prefail_groups[subgroup].extend(
            normalize_frames(parse_frames_raw(path, "prefail", subgroup), "prefail")
        )

    # ------------------------------
    # Split all categories first
    # ------------------------------
    neareq_train, neareq_valid = split_frames(neareq, args.neareq_valid_ratio, args.seed + 1)
    phonopy_train, phonopy_valid = split_frames(phonopy, args.phonopy_valid_ratio, args.seed + 2)
    force_train, force_valid = split_frames(force_spread, args.force_spread_valid_ratio, args.seed + 3)

    prefail_train_groups = {}
    prefail_valid_groups = {}
    for i, subgroup in enumerate(sorted(prefail_groups)):
        tr, va = split_frames(prefail_groups[subgroup], args.prefail_valid_ratio, args.seed + 100 + i)
        prefail_train_groups[subgroup] = tr
        prefail_valid_groups[subgroup] = va

    # ------------------------------
    # Keep full backbone
    # ------------------------------
    train_frames = [deep_copy_frame(fr) for fr in neareq_train]
    train_frames.extend(deep_copy_frame(fr) for fr in phonopy_train)
    train_frames.extend(deep_copy_frame(fr) for fr in force_train)

    valid_frames = [deep_copy_frame(fr) for fr in neareq_valid]
    valid_frames.extend(deep_copy_frame(fr) for fr in phonopy_valid)
    valid_frames.extend(deep_copy_frame(fr) for fr in force_valid)

    backbone_train_total = len(train_frames)
    backbone_valid_total = len(valid_frames)

    # ------------------------------
    # Prefail targets
    # ------------------------------
    prefail_train_target = int(round(args.prefail_train_fraction_of_backbone * backbone_train_total))
    prefail_valid_desired = int(round(args.prefail_valid_fraction_of_backbone * backbone_valid_total))

    prefail_train_frames, prefail_train_summary = build_prefail_train(
        prefail_train_groups,
        prefail_train_target,
        subgroup_weights,
        rng,
        allow_oversample=args.allow_oversample_prefail,
    )

    prefail_valid_frames, prefail_valid_summary, prefail_valid_actual = build_prefail_valid(
        prefail_valid_groups,
        prefail_valid_desired,
        subgroup_weights,
        rng,
    )

    train_frames.extend(prefail_train_frames)
    valid_frames.extend(prefail_valid_frames)

    rng.shuffle(train_frames)
    rng.shuffle(valid_frames)

    validate_frames(train_frames, "training")
    validate_frames(valid_frames, "validation")

    write_frames_raw(args.train_out, train_frames)
    write_frames_raw(args.valid_out, valid_frames)
    write_manifest_csv(args.manifest_out, train_frames, valid_frames)
    write_report(
        args.report_out,
        args,
        backbone_train_total,
        backbone_valid_total,
        prefail_train_target,
        prefail_valid_desired,
        prefail_valid_actual,
        prefail_train_summary,
        prefail_valid_summary,
        train_frames,
        valid_frames,
    )

    # console summary
    print("\n=== Backbone pools kept in full after split ===")
    summarize_by_key(neareq_train, "top_category", "neareq train")
    summarize_by_key(phonopy_train, "top_category", "phonopy train")
    summarize_by_key(force_train, "top_category", "force_spread train")
    summarize_by_key(neareq_valid, "top_category", "neareq valid")
    summarize_by_key(phonopy_valid, "top_category", "phonopy valid")
    summarize_by_key(force_valid, "top_category", "force_spread valid")

    all_prefail_train = []
    all_prefail_valid = []
    for sg in sorted(prefail_train_groups):
        all_prefail_train.extend(prefail_train_groups[sg])
        all_prefail_valid.extend(prefail_valid_groups[sg])

    summarize_by_key(all_prefail_train, "subgroup", "raw prefail train groups")
    summarize_by_key(all_prefail_valid, "subgroup", "raw prefail valid groups")

    print(f"\nbackbone_train_total      = {backbone_train_total}")
    print(f"backbone_valid_total      = {backbone_valid_total}")
    print(f"prefail_train_target      = {prefail_train_target}")
    print(f"prefail_valid_desired     = {prefail_valid_desired}")
    print(f"prefail_valid_actual      = {prefail_valid_actual}")

    print_summary_block("Prefail TRAIN subgroup allocation", prefail_train_summary)
    print_summary_block("Prefail VALID subgroup allocation", prefail_valid_summary)

    summarize_by_key(train_frames, "top_category", "FINAL training top_category")
    summarize_by_key(train_frames, "subgroup", "FINAL training subgroup")
    summarize_by_key(valid_frames, "top_category", "FINAL validation top_category")
    summarize_by_key(valid_frames, "subgroup", "FINAL validation subgroup")

    print(f"\nWrote: {args.train_out}")
    print(f"Wrote: {args.valid_out}")
    print(f"Wrote: {args.manifest_out}")
    print(f"Wrote: {args.report_out}")


if __name__ == "__main__":
    main()
