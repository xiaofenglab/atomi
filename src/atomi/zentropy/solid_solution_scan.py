"""Plan compact MLIP solid-solution scans for defect thermodynamics."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from atomi.zentropy.stage_utils import parse_float_values, write_csv, write_json


SCHEMA = "atomi.zentropy.solid_solution_scan.v1"

FIELDS = [
    "candidate_id",
    "guest_fraction",
    "oxygen_delta",
    "compensation",
    "motif_family",
    "seed_structure",
    "priority",
    "suggested_stage",
    "notes",
]


DEFAULT_COMPENSATION = ["host_valence", "oxygen_vacancy", "mixed"]
DEFAULT_MOTIFS = ["isolated", "paired_near", "paired_far", "clustered"]


def _discover_seeds(seed_root: Path | None, glob_pattern: str) -> list[Path]:
    if seed_root is None:
        return []
    if seed_root.is_file():
        return [seed_root.resolve()]
    return sorted(path.resolve() for path in seed_root.glob(glob_pattern) if path.is_file())


def build_scan_rows(
    *,
    guest_fractions: list[float],
    oxygen_deltas: list[float],
    compensations: list[str],
    motif_families: list[str],
    seed_structures: list[Path],
    max_candidates: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_labels = [str(path) for path in seed_structures] or [""]
    for x_guest in guest_fractions:
        for delta in oxygen_deltas:
            for compensation in compensations:
                for family in motif_families:
                    for seed_structure in seed_labels:
                        rows.append(
                            {
                                "candidate_id": f"x{x_guest:g}_d{delta:g}_{compensation}_{family}_{len(rows) + 1:04d}",
                                "guest_fraction": x_guest,
                                "oxygen_delta": delta,
                                "compensation": compensation,
                                "motif_family": family,
                                "seed_structure": seed_structure,
                                "priority": "seed" if x_guest in (0.0, 0.5, 1.0) else "screen",
                                "suggested_stage": "MLIP relax -> motif-cluster -> selected DFT",
                                "notes": "manifest row; generate concrete structures with system-specific substitution/vacancy builder",
                            }
                        )
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_candidates is not None:
        rows = rows[: max(max_candidates, 0)]
    rows.sort(key=lambda row: (float(row["guest_fraction"]), float(row["oxygen_delta"]), row["compensation"], row["motif_family"]))
    for idx, row in enumerate(rows, start=1):
        row["candidate_id"] = f"scan_{idx:04d}_{row['candidate_id']}"
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlip-solid-solution-scan",
        description="Create a compact composition/defect-family manifest for MLIP solid-solution screening.",
    )
    parser.add_argument("--seed-root", type=Path, help="Optional root containing seed POSCAR/CONTCAR structures.")
    parser.add_argument("--seed-glob", default="**/POSCAR", help="Glob used inside seed-root.")
    parser.add_argument("--guest-fraction", action="append", help="Guest-cation fraction or start:stop:step. Repeatable.")
    parser.add_argument("--oxygen-delta", action="append", help="Oxygen nonstoichiometry delta or start:stop:step. Repeatable.")
    parser.add_argument("--compensation", action="append", default=[], help="Defect compensation family, repeatable.")
    parser.add_argument("--motif-family", action="append", default=[], help="Local motif family, repeatable.")
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--outdir", type=Path, default=Path("mlip_solid_solution_scan"))
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    guest_fractions = parse_float_values(args.guest_fraction, default=[0.0, 0.0625, 0.125, 0.25, 0.5])
    oxygen_deltas = parse_float_values(args.oxygen_delta, default=[0.0, 0.015625, 0.03125])
    compensations = args.compensation or DEFAULT_COMPENSATION
    motif_families = args.motif_family or DEFAULT_MOTIFS
    seed_structures = _discover_seeds(args.seed_root.resolve() if args.seed_root else None, args.seed_glob)
    rows = build_scan_rows(
        guest_fractions=guest_fractions,
        oxygen_deltas=oxygen_deltas,
        compensations=compensations,
        motif_families=motif_families,
        seed_structures=seed_structures,
        max_candidates=args.max_candidates,
        seed=args.seed,
    )
    outdir = args.outdir.resolve()
    csv_path = outdir / "solid_solution_scan_manifest.csv"
    json_path = outdir / "solid_solution_scan_manifest.json"
    write_csv(csv_path, rows, FIELDS)
    payload = {
        "schema": SCHEMA,
        "inputs": {
            "seed_root": str(args.seed_root.resolve()) if args.seed_root else "",
            "seed_glob": args.seed_glob,
        },
        "outputs": {"manifest_csv": str(csv_path), "manifest_json": str(json_path)},
        "guest_fractions": guest_fractions,
        "oxygen_deltas": oxygen_deltas,
        "compensations": compensations,
        "motif_families": motif_families,
        "seed_structures": [str(path) for path in seed_structures],
        "rows": rows,
    }
    write_json(json_path, payload)
    print(f"Scan candidates : {len(rows)}")
    print(f"Seed structures : {len(seed_structures)}")
    print(f"Wrote CSV       : {csv_path}")
    print(f"Wrote JSON      : {json_path}")
    return payload


if __name__ == "__main__":
    main()
