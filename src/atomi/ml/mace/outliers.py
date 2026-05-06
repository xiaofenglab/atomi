#!/usr/bin/env python3

"""
python3 find_energy_outliers.py \
  --extxyz validation_v5.extxyz \
  --model uo2_v5_lr1e3_r3_epoch268.model \
  --outdir energy_outliers_v5r3_valid \
  --device cuda \
  --dtype float32 \
  --top-n 30 \
  --write-poscars

python3 find_energy_outliers.py \
  --extxyz training_v5.extxyz \
  --model uo2_v5_lr1e3_r3_epoch268.model \
  --outdir energy_outliers_v5r3_train \
  --device cuda \
  --dtype float32 \
  --top-n 30 \
  --write-poscars

"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ENERGY_KEYS = ["REF_energy", "energy", "ref_energy", "free_energy", "Energy"]


def get_ref_energy(atoms):
    for key in ENERGY_KEYS:
        if key in atoms.info:
            return float(atoms.info[key])
    raise KeyError(f"No reference energy found. Available keys: {list(atoms.info.keys())}")


def safe_get(info, key, default=""):
    v = info.get(key, default)
    return "" if v is None else v


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        prog="mace-energy-outliers",
        description="Find high energy-error outliers for a MACE model on an extxyz dataset.",
    )
    ap.add_argument("--extxyz", required=True, help="Input extxyz")
    ap.add_argument("--model", required=True, help="MACE .model file")
    ap.add_argument("--outdir", default="energy_outliers")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--write-poscars", action="store_true")
    args = ap.parse_args(argv)

    try:
        import numpy as np
        from ase.io import read, write
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise SystemExit(
            "Missing MACE outlier dependencies. Install/load numpy, ase, and mace before running "
            "this command, for example in your GPU environment."
        ) from exc

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading frames from: {args.extxyz}")
    frames = read(args.extxyz, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    print(f"Loaded {len(frames)} frames")
    if not frames:
        raise RuntimeError(f"No frames found in {args.extxyz}")

    print(f"Loading model: {args.model}")
    calc = MACECalculator(
        model_paths=args.model,
        device=args.device,
        default_dtype=args.dtype,
    )

    rows = []
    per_tag = defaultdict(list)

    for i, atoms in enumerate(frames):
        natoms = len(atoms)
        e_ref = get_ref_energy(atoms)

        atoms.calc = calc
        e_pred = atoms.get_potential_energy()

        err = e_pred - e_ref
        err_pa = err / natoms
        abs_err_pa = abs(err_pa)

        tag = safe_get(atoms.info, "dataset_tag", "UNTAGGED")
        run_dir = safe_get(atoms.info, "run_dir", "")
        config_tag = safe_get(atoms.info, "config_tag", "")
        config_id = safe_get(atoms.info, "config_id", "")

        row = {
            "frame": i,
            "natoms": natoms,
            "dataset_tag": tag,
            "run_dir": run_dir,
            "config_tag": config_tag,
            "config_id": config_id,
            "E_ref_eV": e_ref,
            "E_pred_eV": e_pred,
            "err_eV": err,
            "err_meV_per_atom": 1000.0 * err_pa,
            "abs_err_meV_per_atom": 1000.0 * abs_err_pa,
        }
        rows.append(row)
        per_tag[tag].append(row)

    rows_sorted = sorted(rows, key=lambda r: r["abs_err_meV_per_atom"], reverse=True)

    # CSV of all frames
    csv_path = outdir / "energy_residuals_all.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    # summary by dataset_tag
    summary_rows = []
    for tag, vals in sorted(per_tag.items()):
        arr = np.array([v["err_meV_per_atom"] for v in vals], dtype=float)
        absarr = np.abs(arr)
        summary_rows.append({
            "dataset_tag": tag,
            "n": len(vals),
            "mean_signed_meV_per_atom": float(arr.mean()),
            "mean_abs_meV_per_atom": float(absarr.mean()),
            "max_abs_meV_per_atom": float(absarr.max()),
            "p95_abs_meV_per_atom": float(np.percentile(absarr, 95)),
        })

    summary_csv = outdir / "energy_residuals_by_tag.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_tag",
                "n",
                "mean_signed_meV_per_atom",
                "mean_abs_meV_per_atom",
                "max_abs_meV_per_atom",
                "p95_abs_meV_per_atom",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # write worst extxyz
    top_n = min(args.top_n, len(rows_sorted))
    worst_indices = [rows_sorted[j]["frame"] for j in range(top_n)]
    worst_frames = [frames[idx] for idx in worst_indices]
    write(outdir / "worst_energy_frames.extxyz", worst_frames, format="extxyz")

    # write per-frame POSCARs
    if args.write_poscars:
        poscar_dir = outdir / "worst_energy_poscars"
        poscar_dir.mkdir(exist_ok=True)
        for rank, idx in enumerate(worst_indices, start=1):
            fr = frames[idx]
            row = rows_sorted[rank - 1]
            sub = poscar_dir / f"rank_{rank:03d}_frame_{idx:06d}"
            sub.mkdir(exist_ok=True)
            write(sub / "POSCAR", fr, format="vasp")

            with open(sub / "meta.txt", "w") as f:
                for k, v in row.items():
                    f.write(f"{k}: {v}\n")

    # short text report
    report = outdir / "report.txt"
    with open(report, "w") as f:
        f.write(f"Input extxyz: {args.extxyz}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Frames: {len(frames)}\n\n")

        f.write("Top worst energy outliers (meV/atom):\n")
        for rank, row in enumerate(rows_sorted[:top_n], start=1):
            f.write(
                f"{rank:3d}  frame={row['frame']:6d}  "
                f"abs_err={row['abs_err_meV_per_atom']:10.3f}  "
                f"signed={row['err_meV_per_atom']:10.3f}  "
                f"tag={row['dataset_tag']}  "
                f"config_tag={row['config_tag']}  "
                f"run_dir={row['run_dir']}\n"
            )

        f.write("\nSummary by dataset_tag:\n")
        for r in summary_rows:
            f.write(
                f"{r['dataset_tag']:20s}  n={r['n']:4d}  "
                f"mean_abs={r['mean_abs_meV_per_atom']:8.3f}  "
                f"p95_abs={r['p95_abs_meV_per_atom']:8.3f}  "
                f"max_abs={r['max_abs_meV_per_atom']:8.3f}\n"
            )

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {outdir / 'worst_energy_frames.extxyz'}")
    print(f"Wrote: {report}")
    if args.write_poscars:
        print(f"Wrote POSCARs under: {outdir / 'worst_energy_poscars'}")


if __name__ == "__main__":
    main()
