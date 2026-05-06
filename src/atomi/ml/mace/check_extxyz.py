#!/usr/bin/env python3
"""
Merged extxyz checker + optional MACE-safe rewriting.

Features
--------
- raw extxyz parsing (robust to ASE reserved-property hiding)
- train/valid summary stats
- per-tag summary
- histogram plots
- sanity checks for missing/non-finite labels
- composition / natoms consistency
- optional CSV report of bad frames
- optional rewrite of reserved keys:
    energy  -> REF_energy
    stress  -> REF_stress
    forces  -> REF_forces   (inside Properties=...)

Examples
--------
Check only:
python3 check_extxyz_v3.py \
  --train training_v5.extxyz \
  --valid validation_v5.extxyz \
  --show-tags \
  --write-bad-csv

Check + rewrite safe MACE training files:
python3 check_extxyz_v3.py \
  --train training_v5.extxyz \
  --valid validation_v5.extxyz \
  --show-tags \
  --write-bad-csv \
  --rewrite-refkeys \
  --train-out training_v5_ref.extxyz \
  --valid-out validation_v5_ref.extxyz
"""

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency numpy. Install/load numpy before running mace-check-extxyz."
    ) from exc


DEFAULT_TRAIN = "train.extxyz"
DEFAULT_VALID = "valid.extxyz"

ENERGY_KEYS = [
    "energy",
    "REF_energy",
    "ref_energy",
    "free_energy",
    "Energy",
    "MACE_energy",
]

FORCE_PROP_NAMES = {
    "forces",
    "force",
    "REF_forces",
    "ref_forces",
    "Forces",
}

DATASET_TAG_KEYS = [
    "dataset_tag",
    "tag",
    "source_tag",
    "config_type",
    "config_tag",
]


def unquote(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def read_logical_header(lines, i):
    """
    Read extxyz header line, supporting backslash-continued lines.
    Returns (header_string, next_index).
    """
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

    header = " ".join(p for p in parts if p)
    return header, i


def parse_keyval_header(header):
    """
    Parse key=value pairs from extxyz header.
    Handles quoted values with spaces.
    """
    pattern = re.compile(r'(\S+)=(".*?"|\S+)')
    out = {}
    for m in pattern.finditer(header):
        key = m.group(1)
        val = unquote(m.group(2))
        out[key] = val
    return out


def parse_properties(props_string):
    """
    Parse Properties=species:S:1:pos:R:3:forces:R:3 ...
    Returns list of (name, kind, ncols).
    """
    toks = props_string.split(":")
    if len(toks) % 3 != 0:
        raise ValueError(f"Could not parse Properties field: {props_string}")

    props = []
    for j in range(0, len(toks), 3):
        name = toks[j]
        kind = toks[j + 1]
        ncols = int(toks[j + 2])
        props.append((name, kind, ncols))
    return props


def extract_frame_data(header_dict, atom_lines, frame_idx):
    """
    Returns dict with:
      frame
      natoms
      energy
      dataset_tag
      config_id
      run_dir
      species_counter
      force_norms
      max_force
      has_finite_energy
      has_finite_forces
    """
    energy = None
    for k in ENERGY_KEYS:
        if k in header_dict:
            try:
                energy = float(header_dict[k])
            except Exception:
                energy = None
            break

    dataset_tag = None
    for k in DATASET_TAG_KEYS:
        if k in header_dict:
            dataset_tag = header_dict[k]
            break

    config_id = header_dict.get("config_id", "")
    run_dir = header_dict.get("run_dir", "")

    if "Properties" not in header_dict:
        raise KeyError("No Properties field found in extxyz header.")

    props = parse_properties(header_dict["Properties"])

    # Build column map
    col = 0
    species_slice = None
    force_slice = None

    for name, kind, ncols in props:
        if name == "species" and ncols == 1:
            species_slice = (col, col + 1)
        if name in FORCE_PROP_NAMES and ncols == 3:
            force_slice = (col, col + 3)
        col += ncols

    natoms = len(atom_lines)
    force_norms = np.array([], dtype=float)
    symbols = []

    for line in atom_lines:
        parts = line.split()

        if species_slice is not None:
            i0, _ = species_slice
            if len(parts) <= i0:
                raise ValueError(f"Atom line missing species column:\n{line}")
            symbols.append(parts[i0])

    if force_slice is not None:
        vals = []
        i0, i1 = force_slice
        for line in atom_lines:
            parts = line.split()
            if len(parts) < i1:
                raise ValueError(f"Atom line has too few columns for forces:\n{line}")
            try:
                f = np.array(
                    [float(parts[i0]), float(parts[i0 + 1]), float(parts[i0 + 2])],
                    dtype=float
                )
            except Exception:
                f = np.array([np.nan, np.nan, np.nan], dtype=float)
            vals.append(np.linalg.norm(f))
        force_norms = np.array(vals, dtype=float)

    has_finite_energy = energy is not None and math.isfinite(float(energy))
    has_finite_forces = force_norms.size > 0 and np.isfinite(force_norms).all()

    return {
        "frame": frame_idx,
        "natoms": natoms,
        "energy": energy,
        "dataset_tag": dataset_tag,
        "config_id": config_id,
        "run_dir": run_dir,
        "species_counter": Counter(symbols),
        "force_norms": force_norms,
        "max_force": float(np.nanmax(force_norms)) if force_norms.size else np.nan,
        "has_finite_energy": has_finite_energy,
        "has_finite_forces": has_finite_forces,
        "header_keys": list(header_dict.keys()),
    }


def load_extxyz_raw(filename):
    path = Path(filename)
    if not path.exists():
        return None

    lines = path.read_text(errors="ignore").splitlines()
    frames = []

    i = 0
    nlines = len(lines)
    frame_idx = 0

    while i < nlines:
        if not lines[i].strip():
            i += 1
            continue

        try:
            natoms = int(lines[i].strip())
        except ValueError:
            raise ValueError(f"Expected atom count at line {i+1}, got: {lines[i]!r}")
        i += 1

        if i >= nlines:
            raise ValueError("Unexpected end of file while reading header line.")

        header, i = read_logical_header(lines, i)
        header_dict = parse_keyval_header(header)

        if i + natoms > nlines:
            raise ValueError("Unexpected end of file while reading atom lines.")

        atom_lines = lines[i:i + natoms]
        i += natoms

        frame = extract_frame_data(header_dict, atom_lines, frame_idx)
        frames.append(frame)
        frame_idx += 1

    return frames


def energy_per_atom(frames):
    vals = []
    for fr in frames:
        if not fr["has_finite_energy"]:
            continue
        vals.append(fr["energy"] / fr["natoms"])
    return np.array(vals, dtype=float)


def all_force_magnitudes(frames):
    vals = []
    for fr in frames:
        if not fr["has_finite_forces"]:
            continue
        vals.extend(fr["force_norms"])
    return np.array(vals, dtype=float)


def max_force_per_structure(frames):
    vals = []
    for fr in frames:
        if not fr["has_finite_forces"]:
            continue
        vals.append(fr["max_force"])
    return np.array(vals, dtype=float)


def summarize(name, epa, fmag, fmax):
    print(f"\n{name}")

    if len(epa):
        print("Energy per atom (eV/atom)")
        print(f"  min  = {epa.min():.6f}")
        print(f"  max  = {epa.max():.6f}")
        print(f"  mean = {epa.mean():.6f}")
        print(f"  std  = {epa.std():.6f}")
    else:
        print("Energy per atom: no usable energy data found")

    if len(fmag):
        print("\nForce magnitude (all atoms)")
        print(f"  min  = {fmag.min():.6f}")
        print(f"  max  = {fmag.max():.6f}")
        print(f"  mean = {fmag.mean():.6f}")
        print(f"  std  = {fmag.std():.6f}")
    else:
        print("\nForce magnitude: no usable force data found")

    if len(fmax):
        print("\nMax force per structure")
        print(f"  min  = {fmax.min():.6f}")
        print(f"  max  = {fmax.max():.6f}")
        print(f"  mean = {fmax.mean():.6f}")
        print(f"  std  = {fmax.std():.6f}")
    else:
        print("\nMax force per structure: no usable force data found")


def summarize_by_dataset_tag(frames, name):
    buckets = {}
    for fr in frames:
        tag = fr["dataset_tag"]
        if tag is None:
            tag = "UNTAGGED"
        buckets.setdefault(tag, []).append(fr)

    if not buckets:
        print(f"\n{name}: no dataset_tag found, skipping per-tag summary.")
        return

    print(f"\n{name}: per-dataset_tag summary")
    for tag, subset in sorted(buckets.items()):
        epa = energy_per_atom(subset)
        fmax = max_force_per_structure(subset)

        epa_mean = f"{epa.mean(): .6f}" if len(epa) else "   n/a"
        fmax_mean = f"{fmax.mean(): .6f}" if len(fmax) else "   n/a"

        print(f"  {tag:20s} n={len(subset):4d}  E/atom mean={epa_mean}  Fmax mean={fmax_mean}")


def summarize_composition(frames, name):
    comp_counter = Counter()
    for fr in frames:
        comp = tuple(sorted(fr["species_counter"].items()))
        comp_counter[(fr["natoms"], comp)] += 1

    print(f"\n{name}: composition / natoms patterns")
    for (nat, comp), count in sorted(comp_counter.items(), key=lambda x: (-x[1], x[0][0])):
        comp_str = " ".join(f"{el}{n}" for el, n in comp)
        print(f"  {count:5d}  natoms={nat:4d}  {comp_str}")


def summarize_bad_frames(frames, name):
    n_bad_e = sum(not fr["has_finite_energy"] for fr in frames)
    n_bad_f = sum(not fr["has_finite_forces"] for fr in frames)
    n_bad_any = sum((not fr["has_finite_energy"]) or (not fr["has_finite_forces"]) for fr in frames)

    print(f"\n{name}: sanity checks")
    print(f"  Frames with missing/non-finite energy: {n_bad_e}")
    print(f"  Frames with missing/non-finite forces: {n_bad_f}")
    print(f"  Total bad frames: {n_bad_any}")


def write_bad_csv(frames, outfile):
    bad_rows = []
    for fr in frames:
        if fr["has_finite_energy"] and fr["has_finite_forces"]:
            continue
        bad_rows.append({
            "frame": fr["frame"],
            "natoms": fr["natoms"],
            "dataset_tag": fr["dataset_tag"],
            "config_id": fr["config_id"],
            "run_dir": fr["run_dir"],
            "has_finite_energy": fr["has_finite_energy"],
            "has_finite_forces": fr["has_finite_forces"],
            "header_keys": ";".join(fr["header_keys"]),
        })

    if not bad_rows:
        return None

    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bad_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bad_rows)

    return outfile


def save_hist(data, bins, xlabel, ylabel, title, outfile):
    if len(data) == 0:
        return
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def save_overlay_hist(data1, data2, bins, label1, label2, xlabel, ylabel, title, outfile):
    if len(data1) == 0 or len(data2) == 0:
        return
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist(data1, bins=bins, alpha=0.6, label=label1)
    plt.hist(data2, bins=bins, alpha=0.6, label=label2)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def print_first_frame_info(name, frames):
    if not frames:
        return
    fr = frames[0]
    print(f"\nFirst frame summary for {name}:")
    print(f"  natoms      : {fr['natoms']}")
    print(f"  energy      : {fr['energy']}")
    print(f"  dataset_tag : {fr['dataset_tag']}")
    print(f"  config_id   : {fr['config_id']}")
    print(f"  nforcevals  : {len(fr['force_norms'])}")
    print(f"  run_dir     : {fr['run_dir']}")


def rewrite_header_line(line: str) -> str:
    """
    Rewrite reserved ASE keys into explicit reference keys for safe MACE training:
      energy -> REF_energy
      stress -> REF_stress
      Properties ... forces:R:3 -> REF_forces:R:3
    """
    # key=value header fields
    line = re.sub(r'(^|\s)energy=', r'\1REF_energy=', line)
    line = re.sub(r'(^|\s)stress=', r'\1REF_stress=', line)

    # inside Properties=...
    m = re.search(r'Properties=("[^"]*"|\S+)', line)
    if m:
        props_token = m.group(1)
        props_unquoted = unquote(props_token)

        toks = props_unquoted.split(":")
        new_toks = []
        for i, tok in enumerate(toks):
            # property names are positions 0, 3, 6, ...
            if i % 3 == 0 and tok == "forces":
                tok = "REF_forces"
            new_toks.append(tok)

        props_new = ":".join(new_toks)

        if props_token.startswith('"') and props_token.endswith('"'):
            replacement = f'Properties="{props_new}"'
        else:
            replacement = f'Properties={props_new}'

        line = line[:m.start()] + replacement + line[m.end():]

    return line


def rewrite_extxyz_refkeys(infile: Path, outfile: Path):
    """
    Rewrite extxyz headers only; atom lines remain unchanged.
    """
    lines = infile.read_text(errors="ignore").splitlines()
    out_lines = []

    i = 0
    n = len(lines)

    while i < n:
        if not lines[i].strip():
            out_lines.append(lines[i])
            i += 1
            continue

        natoms = int(lines[i].strip())
        out_lines.append(lines[i])
        i += 1

        if i >= n:
            raise ValueError("Unexpected EOF after atom count")

        header, i_after_header = read_logical_header(lines, i)
        header_new = rewrite_header_line(header)
        out_lines.append(header_new)
        i = i_after_header

        for _ in range(natoms):
            if i >= n:
                raise ValueError("Unexpected EOF inside atom block")
            out_lines.append(lines[i])
            i += 1

    outfile.write_text("\n".join(out_lines) + "\n")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="mace-check-extxyz",
        description="Merged raw extxyz checker: stats + per-tag summary + bad-frame sanity checks + optional REF-key rewrite."
    )
    parser.add_argument("--train", default=DEFAULT_TRAIN, help="Training extxyz file")
    parser.add_argument("--valid", default=DEFAULT_VALID, help="Validation extxyz file")
    parser.add_argument("--show-tags", action="store_true", help="Print summary by dataset_tag")
    parser.add_argument("--write-bad-csv", action="store_true", help="Write CSV listing bad frames")

    parser.add_argument(
        "--rewrite-refkeys",
        action="store_true",
        help="Rewrite energy->REF_energy, stress->REF_stress, forces->REF_forces for MACE-safe training files"
    )
    parser.add_argument(
        "--train-out",
        default="training_ref.extxyz",
        help="Output rewritten training extxyz"
    )
    parser.add_argument(
        "--valid-out",
        default="validation_ref.extxyz",
        help="Output rewritten validation extxyz"
    )

    args = parser.parse_args(argv)

    train = load_extxyz_raw(args.train)
    if train is None:
        raise FileNotFoundError(f"{args.train} not found")

    print(f"\nLoaded {len(train)} training structures from {args.train}")
    print_first_frame_info("training", train)

    train_e_pa = energy_per_atom(train)
    train_fmag = all_force_magnitudes(train)
    train_fmax = max_force_per_structure(train)

    summarize("Training set", train_e_pa, train_fmag, train_fmax)
    summarize_bad_frames(train, "Training set")
    summarize_composition(train, "Training set")

    if args.show_tags:
        summarize_by_dataset_tag(train, "Training set")

    if args.write_bad_csv:
        out = write_bad_csv(train, Path("train_bad_frames.csv"))
        if out:
            print(f"\nWrote bad-frame report: {out}")

    save_hist(
        train_e_pa, bins=30,
        xlabel="Energy per atom (eV)",
        ylabel="Count",
        title="Training energy distribution",
        outfile="train_energy_hist.png"
    )

    save_hist(
        train_fmag, bins=50,
        xlabel="Force magnitude (eV/Å)",
        ylabel="Count",
        title="Training force distribution",
        outfile="train_force_hist.png"
    )

    save_hist(
        train_fmax, bins=30,
        xlabel="Max force per structure (eV/Å)",
        ylabel="Count",
        title="Training max force per structure",
        outfile="train_max_force_structure.png"
    )

    valid = load_extxyz_raw(args.valid)

    if valid is None:
        print(f"\n{args.valid} not found — skipping validation analysis.")
    else:
        print(f"\nLoaded {len(valid)} validation structures from {args.valid}")
        print_first_frame_info("validation", valid)

        valid_e_pa = energy_per_atom(valid)
        valid_fmag = all_force_magnitudes(valid)
        valid_fmax = max_force_per_structure(valid)

        summarize("Validation set", valid_e_pa, valid_fmag, valid_fmax)
        summarize_bad_frames(valid, "Validation set")
        summarize_composition(valid, "Validation set")

        if args.show_tags:
            summarize_by_dataset_tag(valid, "Validation set")

        if args.write_bad_csv:
            out = write_bad_csv(valid, Path("valid_bad_frames.csv"))
            if out:
                print(f"\nWrote bad-frame report: {out}")

        save_overlay_hist(
            train_e_pa, valid_e_pa, bins=30,
            label1="train", label2="valid",
            xlabel="Energy per atom (eV)",
            ylabel="Count",
            title="Energy comparison",
            outfile="energy_compare.png"
        )

        save_overlay_hist(
            train_fmag, valid_fmag, bins=50,
            label1="train", label2="valid",
            xlabel="Force magnitude (eV/Å)",
            ylabel="Count",
            title="Force comparison",
            outfile="force_compare.png"
        )

        save_overlay_hist(
            train_fmax, valid_fmax, bins=30,
            label1="train", label2="valid",
            xlabel="Max force per structure (eV/Å)",
            ylabel="Count",
            title="Max force per structure comparison",
            outfile="max_force_compare.png"
        )

    if args.rewrite_refkeys:
        print("\nRewriting extxyz files for MACE-safe keys...")
        train_out = Path(args.train_out)
        rewrite_extxyz_refkeys(Path(args.train), train_out)
        print(f"  wrote {train_out}")

        if Path(args.valid).exists():
            valid_out = Path(args.valid_out)
            rewrite_extxyz_refkeys(Path(args.valid), valid_out)
            print(f"  wrote {valid_out}")

        print("\nUse these rewritten files with:")
        print("  --energy_key=REF_energy")
        print("  --forces_key=REF_forces")
        print("  --stress_key=REF_stress")

    print("\nSaved plots:")
    print("  train_energy_hist.png")
    print("  train_force_hist.png")
    print("  train_max_force_structure.png")
    if Path(args.valid).exists():
        print("  energy_compare.png")
        print("  force_compare.png")
        print("  max_force_compare.png")


if __name__ == "__main__":
    main()
