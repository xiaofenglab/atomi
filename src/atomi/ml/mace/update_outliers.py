#!/usr/bin/env python3
"""
Update train/valid extxyz files using run_dir entries from an outlier report.

Features
--------
- parse run_dir=... entries from report.txt
- remove matching frames from train/valid extxyz
- optionally append replacement frames from one or more new extxyz files
- preserve raw extxyz headers/atom lines exactly

Typical workflow
----------------
1) Remove bad frames only:
python3 update_extxyz_outlier.py \
  --report energy_outliers_v5r3_train/report.txt \
  --train-in training_v5.extxyz \
  --valid-in validation_v5.extxyz \
  --train-out training_v5_clean.extxyz \
  --valid-out validation_v5_clean.extxyz

2) Remove bad frames, then append rerun results:
python3 update_extxyz_outlier.py \
  --report energy_outliers_v5r3_train/report.txt \
  --train-in training_v5.extxyz \
  --valid-in validation_v5.extxyz \
  --train-out training_v5_updated.extxyz \
  --valid-out validation_v5_updated.extxyz \
  --add-extxyz rerun_bad_energy.extxyz


Step 1

Remove old bad legacy frames first.

Step 2

Regenerate the rerun results into one extxyz file, for example:
    •   rerun_bad_energy.extxyz

Step 3

Run the update script in remove_and_append mode.

Step 4

Run your check_extxyz_v3.py again on the updated train/valid files.


"""

import argparse
import re
from pathlib import Path
from collections import Counter


# ---------------------------------------------------
# RAW EXTXYZ PARSING
# ---------------------------------------------------

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
    # supports quoted values with spaces
    pat = re.compile(r'(\S+)=(".*?"|\S+)')
    out = {}
    for m in pat.finditer(header):
        k = m.group(1)
        v = m.group(2)
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            v = v[1:-1]
        out[k] = v
    return out


def parse_frames_raw(path: str):
    lines = Path(path).read_text(errors="ignore").splitlines()
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
            "run_dir": hdr.get("run_dir", ""),
            "dataset_tag": hdr.get("dataset_tag", ""),
            "config_tag": hdr.get("config_tag", ""),
            "config_id": hdr.get("config_id", ""),
        })

    return frames


def write_frames_raw(path: str, frames):
    with open(path, "w") as f:
        for fr in frames:
            f.write(f"{fr['natoms']}\n")
            f.write(fr["header"].rstrip() + "\n")
            for line in fr["atom_lines"]:
                f.write(line.rstrip() + "\n")


# ---------------------------------------------------
# REPORT PARSING
# ---------------------------------------------------

def parse_run_dirs_from_report(report_path: str):
    pat = re.compile(r"run_dir=(\S+)")
    run_dirs = []
    with open(report_path, "r", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                run_dirs.append(m.group(1))
    return sorted(set(run_dirs))


# ---------------------------------------------------
# MAIN UPDATE LOGIC
# ---------------------------------------------------

def remove_frames_by_run_dir(frames, bad_run_dirs):
    keep = []
    removed = []
    bad = set(bad_run_dirs)

    for fr in frames:
        if fr["run_dir"] in bad:
            removed.append(fr)
        else:
            keep.append(fr)

    return keep, removed


def summarize(frames, name):
    tags = Counter(fr.get("dataset_tag", "") or "UNTAGGED" for fr in frames)
    print(f"{name}: {len(frames)} frames")
    for tag, n in sorted(tags.items()):
        print(f"  {tag}: {n}")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        prog="mace-update-outliers",
        description="Remove MACE energy outlier frames from train/valid extxyz files and optionally append reruns.",
    )
    ap.add_argument("--report", required=True, help="report.txt from outlier analysis")
    ap.add_argument("--train-in", required=True)
    ap.add_argument("--valid-in", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--valid-out", required=True)
    ap.add_argument(
        "--add-extxyz",
        nargs="*",
        default=[],
        help="Optional replacement extxyz files to append after removal"
    )
    ap.add_argument(
        "--mode",
        choices=["remove_only", "remove_and_append"],
        default="remove_only",
        help="Whether to only remove bad frames or also append replacement frames"
    )
    args = ap.parse_args(argv)
    if args.add_extxyz and args.mode == "remove_only":
        args.mode = "remove_and_append"

    bad_run_dirs = parse_run_dirs_from_report(args.report)
    if not bad_run_dirs:
        raise RuntimeError("No run_dir entries found in report.")

    print(f"Found {len(bad_run_dirs)} unique bad run_dir entries in report")

    train_frames = parse_frames_raw(args.train_in)
    valid_frames = parse_frames_raw(args.valid_in)

    summarize(train_frames, "Original training")
    summarize(valid_frames, "Original validation")

    train_keep, train_removed = remove_frames_by_run_dir(train_frames, bad_run_dirs)
    valid_keep, valid_removed = remove_frames_by_run_dir(valid_frames, bad_run_dirs)

    print(f"\nRemoved from training:   {len(train_removed)}")
    print(f"Removed from validation: {len(valid_removed)}")

    # Build maps of where each run_dir came from originally
    removed_from_train = {fr["run_dir"] for fr in train_removed}
    removed_from_valid = {fr["run_dir"] for fr in valid_removed}

    if args.mode == "remove_and_append":
        replacement_frames = []
        for path in args.add_extxyz:
            frames = parse_frames_raw(path)
            replacement_frames.extend(frames)

        print(f"Loaded replacement frames: {len(replacement_frames)}")

        appended_train = 0
        appended_valid = 0

        # only append rerun frames whose run_dir matches one removed from train/valid
        # if a run_dir was removed from both, prefer same split counts by first match
        for fr in replacement_frames:
            rd = fr["run_dir"]
            if rd in removed_from_train:
                train_keep.append(fr)
                appended_train += 1
                removed_from_train.remove(rd)
            elif rd in removed_from_valid:
                valid_keep.append(fr)
                appended_valid += 1
                removed_from_valid.remove(rd)

        print(f"Appended to training:   {appended_train}")
        print(f"Appended to validation: {appended_valid}")

        if removed_from_train:
            print(f"WARNING: {len(removed_from_train)} removed train run_dirs still missing replacements")
        if removed_from_valid:
            print(f"WARNING: {len(removed_from_valid)} removed valid run_dirs still missing replacements")

    write_frames_raw(args.train_out, train_keep)
    write_frames_raw(args.valid_out, valid_keep)

    print("\nUpdated datasets written:")
    print(f"  {args.train_out}")
    print(f"  {args.valid_out}")

    summarize(train_keep, "Updated training")
    summarize(valid_keep, "Updated validation")


if __name__ == "__main__":
    main()
