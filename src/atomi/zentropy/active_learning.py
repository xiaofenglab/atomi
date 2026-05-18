"""Rank zentropy motifs for follow-up DFT or MLIP sampling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from atomi.zentropy.stage_utils import finite_float, load_motif_records, load_table, write_csv, write_json


SCHEMA = "atomi.zentropy.active_learning.v1"

FIELDS = [
    "rank",
    "motif_id",
    "motif_family",
    "composition",
    "T_K",
    "probability",
    "delta_G_eV_per_fu",
    "uncertainty_eV_per_fu",
    "score",
    "reason",
    "run_dir",
    "suggested_action",
]


def _motif_lookup(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    _, records = load_motif_records(path)
    return {str(row.get("motif_id") or row.get("id") or row.get("name")): row for row in records}


def rank_candidates(
    probability_rows: list[dict[str, Any]],
    *,
    motif_lookup: dict[str, dict[str, Any]],
    min_probability: float,
    uncertainty_weight: float,
    max_delta_eV_per_fu: float | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in probability_rows:
        probability = finite_float(row.get("probability")) or 0.0
        delta_g = finite_float(row.get("delta_G_eV_per_fu"))
        if probability < min_probability and (max_delta_eV_per_fu is None or delta_g is None or delta_g > max_delta_eV_per_fu):
            continue
        motif_id = str(row.get("motif_id") or "")
        motif = motif_lookup.get(motif_id, {})
        uncertainty = finite_float(row.get("uncertainty_eV_per_fu") or motif.get("uncertainty_eV_per_fu")) or 0.0
        score = probability + uncertainty_weight * uncertainty
        reasons: list[str] = []
        if probability >= min_probability:
            reasons.append("high_probability")
        if uncertainty > 0.0:
            reasons.append("uncertain")
        if delta_g is not None and max_delta_eV_per_fu is not None and delta_g <= max_delta_eV_per_fu:
            reasons.append("near_ground_state")
        if not reasons:
            reasons.append("coverage")
        candidates.append(
            {
                "motif_id": motif_id,
                "motif_family": row.get("motif_family") or motif.get("motif_family") or "",
                "composition": row.get("composition") or "",
                "T_K": row.get("T_K") or "",
                "probability": probability,
                "delta_G_eV_per_fu": delta_g,
                "uncertainty_eV_per_fu": uncertainty,
                "score": score,
                "reason": ",".join(reasons),
                "run_dir": motif.get("run_dir") or "",
                "suggested_action": "DFT single point / short MLIP-MD / local defect-cloud expansion",
            }
        )
    candidates.sort(key=lambda row: (-(finite_float(row.get("score")) or 0.0), str(row.get("motif_id") or "")))
    for idx, row in enumerate(candidates, start=1):
        row["rank"] = idx
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy-active-learning",
        description="Rank motif states for the next DFT/MLIP sampling round from zentropy probabilities.",
    )
    parser.add_argument("--probability-csv", type=Path, required=True, help="ensemble_probabilities.csv from zentropy-solve.")
    parser.add_argument("--motif-db", type=Path, help="Optional motif DB for run paths and metadata.")
    parser.add_argument("--outdir", type=Path, default=Path("stage5_active_learning"))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-probability", type=float, default=0.02)
    parser.add_argument("--max-delta-eV-per-fu", type=float, help="Also keep states close to the minimum even if probability is low.")
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    probabilities = load_table(args.probability_csv.resolve())
    candidates = rank_candidates(
        probabilities,
        motif_lookup=_motif_lookup(args.motif_db.resolve() if args.motif_db else None),
        min_probability=args.min_probability,
        uncertainty_weight=args.uncertainty_weight,
        max_delta_eV_per_fu=args.max_delta_eV_per_fu,
    )[: max(args.top_n, 0)]
    outdir = args.outdir.resolve()
    csv_path = outdir / "active_learning_candidates.csv"
    manifest_path = outdir / "mlip_training_manifest.json"
    write_csv(csv_path, candidates, FIELDS)
    manifest = {
        "schema": SCHEMA,
        "inputs": {
            "probability_csv": str(args.probability_csv.resolve()),
            "motif_db": str(args.motif_db.resolve()) if args.motif_db else "",
        },
        "outputs": {"candidates": str(csv_path), "mlip_training_manifest": str(manifest_path)},
        "selection": {
            "top_n": args.top_n,
            "min_probability": args.min_probability,
            "max_delta_eV_per_fu": args.max_delta_eV_per_fu,
            "uncertainty_weight": args.uncertainty_weight,
        },
        "candidates": candidates,
    }
    write_json(manifest_path, manifest)
    write_json(outdir / "active_learning_metadata.json", {key: value for key, value in manifest.items() if key != "candidates"})
    print(f"Candidates : {len(candidates)}")
    print(f"Wrote CSV  : {csv_path}")
    print(f"Wrote JSON : {manifest_path}")
    return manifest


if __name__ == "__main__":
    main()
