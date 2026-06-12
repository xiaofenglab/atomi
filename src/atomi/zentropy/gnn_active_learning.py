"""GNN/MLIP active-learning bridge for defect thermodynamics.

The layer is deliberately backend-neutral.  It can consume predictions from
CHGNet/MACE/M3GNet/NequIP/other graph or equivariant MLIPs, but it does not
make those packages mandatory.  Its job is to turn low-Gd guarded motifs into a
candidate high-composition pool, merge surrogate predictions and uncertainty,
rank candidates for DFT, and hand selected structures/metadata to mode4.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from atomi.zentropy.backends.base import CETrainingRecord, CETrainingSet, read_ce_training_jsonl, write_ce_training_jsonl

SCHEMA = "atomi.zentropy.gnn_active_learning.v1"

CANDIDATE_FIELDS = [
    "candidate_id",
    "x_Gd",
    "delta_VO",
    "h_U5",
    "charge_residual",
    "compensation_family",
    "seed_motif_id",
    "seed_family",
    "motif_features_json",
    "metadata_json",
]

SCORED_FIELDS = CANDIDATE_FIELDS + [
    "predicted_G_eV_per_fu",
    "surrogate_uncertainty_eV",
    "data_distance",
    "redox_penalty",
    "acquisition_score",
    "selection_reason",
    "surrogate_source",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def parse_grid(values: list[str], *, default: list[float]) -> list[float]:
    if not values:
        return default
    out: list[float] = []
    for item in values:
        for token in str(item).split(","):
            text = token.strip()
            if not text:
                continue
            if ":" in text:
                start, stop, step = [float(part) for part in text.split(":")]
                if step == 0:
                    raise ValueError("Grid step cannot be zero.")
                current = start
                if step > 0:
                    while current <= stop + abs(step) * 1.0e-10:
                        out.append(round(current, 12))
                        current += step
                else:
                    while current >= stop - abs(step) * 1.0e-10:
                        out.append(round(current, 12))
                        current += step
            else:
                out.append(float(text))
    return list(dict.fromkeys(out))


def charge_residual(x_gd: float, delta_vo: float, h_u5: float) -> float:
    """Charge-neutral fluorite residual per cation: U5 + 2*VO - Gd."""

    return h_u5 + 2.0 * delta_vo - x_gd


def neutral_h_u5(x_gd: float, delta_vo: float) -> float | None:
    h_u5 = x_gd - 2.0 * delta_vo
    if h_u5 < -1.0e-10:
        return None
    return max(0.0, h_u5)


def compensation_family(x_gd: float, delta_vo: float, h_u5: float) -> str:
    if delta_vo <= 1.0e-10 and h_u5 > 1.0e-10:
        return "u5_compensated"
    if h_u5 <= 1.0e-10 and delta_vo > 1.0e-10:
        return "vacancy_compensated"
    if h_u5 > 1.0e-10 and delta_vo > 1.0e-10:
        return "mixed_u5_vacancy"
    return "parent_or_neutral_limit"


def _seed_rows(training: CETrainingSet | None) -> list[dict[str, Any]]:
    if training is None or not training.records:
        return [{"motif_id": "generic_fluorite_seed", "family": "generated", "features": {}}]
    rows: list[dict[str, Any]] = []
    for record in training.records:
        rows.append(
            {
                "motif_id": record.record_id,
                "family": record.metadata.get("motif_family") or record.metadata.get("family") or record.source,
                "features": record.motif_features,
            }
        )
    return rows


def training_ranges(training: CETrainingSet | None) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    if training is None:
        return ranges
    for axis in ("x_Gd", "delta_VO", "h_U5"):
        values = [record.composition[axis] for record in training.records if axis in record.composition]
        if values:
            ranges[axis] = (min(values), max(values))
    return ranges


def build_candidate_pool(
    *,
    x_grid: list[float],
    delta_grid: list[float],
    h_grid: list[float] | None = None,
    training: CETrainingSet | None = None,
    max_charge_residual: float = 1.0e-8,
    max_candidates_per_seed: int | None = None,
) -> list[dict[str, Any]]:
    seeds = _seed_rows(training)
    candidates: list[dict[str, Any]] = []
    count_by_seed: dict[str, int] = {}
    for seed in seeds:
        sid = str(seed["motif_id"])
        count_by_seed.setdefault(sid, 0)
        for x_gd in x_grid:
            for delta in delta_grid:
                h_values = h_grid if h_grid is not None else [neutral_h_u5(x_gd, delta)]
                for h_u5_maybe in h_values:
                    if h_u5_maybe is None:
                        continue
                    h_u5 = float(h_u5_maybe)
                    residual = charge_residual(x_gd, delta, h_u5)
                    if abs(residual) > max_charge_residual:
                        continue
                    if max_candidates_per_seed is not None and count_by_seed[sid] >= max_candidates_per_seed:
                        continue
                    cid = f"cand_{len(candidates)+1:06d}_x{x_gd:.5f}_d{delta:.5f}_h{h_u5:.5f}_{sid}"
                    features = dict(seed.get("features") or {})
                    features.update({"x_Gd": x_gd, "delta_VO": delta, "h_U5": h_u5})
                    candidates.append(
                        {
                            "candidate_id": cid,
                            "x_Gd": x_gd,
                            "delta_VO": delta,
                            "h_U5": h_u5,
                            "charge_residual": residual,
                            "compensation_family": compensation_family(x_gd, delta, h_u5),
                            "seed_motif_id": sid,
                            "seed_family": seed.get("family") or "",
                            "motif_features_json": json.dumps(features, sort_keys=True),
                            "metadata_json": json.dumps(
                                {
                                    "generated_by": "gnn_active_learning.build_candidate_pool",
                                    "role": "surrogate_screening_candidate",
                                },
                                sort_keys=True,
                            ),
                        }
                    )
                    count_by_seed[sid] += 1
    return candidates


def _prediction_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = _read_csv(path)
    return {str(row.get("candidate_id") or row.get("record_id") or ""): row for row in rows}


def _range_distance(value: float, axis_range: tuple[float, float] | None) -> float:
    if axis_range is None:
        return 1.0
    lo, hi = axis_range
    width = max(hi - lo, 1.0e-12)
    if lo <= value <= hi:
        return 0.0
    return min(abs(value - lo), abs(value - hi)) / width


def score_candidates(
    candidates: list[dict[str, Any]],
    *,
    prediction_rows: dict[str, dict[str, str]] | None = None,
    training: CETrainingSet | None = None,
    exploration_weight: float = 1.0,
    redox_weight: float = 0.25,
) -> list[dict[str, Any]]:
    prediction_rows = prediction_rows or {}
    ranges = training_ranges(training)
    scored: list[dict[str, Any]] = []
    for row in candidates:
        cid = str(row["candidate_id"])
        pred = prediction_rows.get(cid, {})
        x = float(row["x_Gd"])
        d = float(row["delta_VO"])
        h = float(row["h_U5"])
        data_distance = sum(
            _range_distance(value, ranges.get(axis))
            for axis, value in (("x_Gd", x), ("delta_VO", d), ("h_U5", h))
        )
        predicted = _finite(
            pred.get("predicted_G_eV_per_fu")
            or pred.get("predicted_energy_eV")
            or pred.get("energy_eV")
        )
        if predicted is None:
            # Deterministic placeholder: favors physically neutral low-energy-looking mixed states
            # only for ordering before real GNN/MLIP predictions are available.
            predicted = 0.5 * x + 0.25 * d + 0.1 * h
        uncertainty = _finite(pred.get("surrogate_uncertainty_eV") or pred.get("uncertainty_eV"))
        if uncertainty is None:
            uncertainty = 0.05 + 0.1 * data_distance
        redox_penalty = abs(charge_residual(x, d, h)) + (0.0 if h <= x + 1.0e-12 else 1.0)
        acquisition = predicted - exploration_weight * uncertainty + redox_weight * redox_penalty
        reason = "surrogate_low_G_high_uncertainty" if pred else "heuristic_no_prediction"
        out = dict(row)
        out.update(
            {
                "predicted_G_eV_per_fu": predicted,
                "surrogate_uncertainty_eV": uncertainty,
                "data_distance": data_distance,
                "redox_penalty": redox_penalty,
                "acquisition_score": acquisition,
                "selection_reason": reason,
                "surrogate_source": pred.get("source") or ("prediction_csv" if pred else "heuristic"),
            }
        )
        scored.append(out)
    return sorted(scored, key=lambda r: (float(r["acquisition_score"]), -float(r["surrogate_uncertainty_eV"])))


def select_candidates(rows: list[dict[str, Any]], *, top_n: int, min_data_distance: float = 0.0) -> list[dict[str, Any]]:
    filtered = [row for row in rows if float(row.get("data_distance") or 0.0) >= min_data_distance]
    return filtered[:top_n]


def selected_to_training_set(rows: list[dict[str, Any]], *, system_name: str, parent_structure: str) -> CETrainingSet:
    records: list[CETrainingRecord] = []
    for row in rows:
        features = json.loads(row.get("motif_features_json") or "{}")
        records.append(
            CETrainingRecord(
                record_id=str(row["candidate_id"]),
                structure_path=str(row.get("structure_path") or ""),
                composition={
                    "x_Gd": float(row["x_Gd"]),
                    "delta_VO": float(row["delta_VO"]),
                    "h_U5": float(row["h_U5"]),
                },
                motif_features={str(k): float(v) for k, v in features.items() if _finite(v) is not None},
                energy_eV=_finite(row.get("predicted_G_eV_per_fu")),
                uncertainty_eV=_finite(row.get("surrogate_uncertainty_eV")),
                source="gnn_active_learning_selected",
                metadata={
                    "selection_reason": row.get("selection_reason") or "",
                    "surrogate_source": row.get("surrogate_source") or "",
                    "requires_dft_label": True,
                    "use_predicted_energy_only_as_prior": True,
                },
            )
        )
    return CETrainingSet(
        system_name=system_name,
        parent_structure_path=parent_structure,
        composition_axes={"x_Gd": "N_Gd3/N_cation", "delta_VO": "N_VaO/N_cation", "h_U5": "N_U5/N_cation"},
        charge_constraints=["N_U5 + 2*N_VaO - N_Gd3 == 0"],
        records=records,
        metadata={"schema": SCHEMA, "stage": "gnn_active_learning_selected_candidates"},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy-gnn-active-learning",
        description="Generate, score, and select high-composition defect candidates using GNN/MLIP surrogate predictions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-candidates", help="Generate charge-neutral candidate states from composition grids and seed motifs.")
    build.add_argument("--seed-training-jsonl", type=Path)
    build.add_argument("--outdir", type=Path, default=Path("gnn_active_candidates"))
    build.add_argument("--output", type=Path, default=Path("candidate_pool.csv"))
    build.add_argument("--x-grid", action="append", default=[])
    build.add_argument("--delta-grid", action="append", default=[])
    build.add_argument("--h-u5-grid", action="append", default=[], help="Optional explicit h_U5 grid. Default enforces charge-neutral h_U5=x_Gd-2delta.")
    build.add_argument("--max-candidates-per-seed", type=int)

    score = sub.add_parser("score-candidates", help="Merge GNN/MLIP predictions and rank candidates for DFT labeling.")
    score.add_argument("--candidate-csv", type=Path, required=True)
    score.add_argument("--prediction-csv", type=Path)
    score.add_argument("--seed-training-jsonl", type=Path)
    score.add_argument("--outdir", type=Path, default=Path("gnn_active_scored"))
    score.add_argument("--output", type=Path, default=Path("scored_candidates.csv"))
    score.add_argument("--exploration-weight", type=float, default=1.0)
    score.add_argument("--redox-weight", type=float, default=0.25)

    select = sub.add_parser("select-dft", help="Select the most informative candidates for new DFT labels and mode4 handoff.")
    select.add_argument("--scored-csv", type=Path, required=True)
    select.add_argument("--outdir", type=Path, default=Path("gnn_active_selected"))
    select.add_argument("--top-n", type=int, default=24)
    select.add_argument("--min-data-distance", type=float, default=0.0)
    select.add_argument("--system", default="Gd-UO2")
    select.add_argument("--parent-structure", default="fluorite_Fm-3m")

    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.command == "build-candidates":
        training = read_ce_training_jsonl(args.seed_training_jsonl.resolve()) if args.seed_training_jsonl else None
        outdir = args.outdir.resolve()
        output = args.output if args.output.is_absolute() else outdir / args.output
        h_grid = parse_grid(args.h_u5_grid, default=[]) if args.h_u5_grid else None
        rows = build_candidate_pool(
            x_grid=parse_grid(args.x_grid, default=[0.0, 0.0625, 0.125]),
            delta_grid=parse_grid(args.delta_grid, default=[0.0, 0.03125, 0.0625]),
            h_grid=h_grid,
            training=training,
            max_candidates_per_seed=args.max_candidates_per_seed,
        )
        _write_csv(output, rows, CANDIDATE_FIELDS)
        meta = {"schema": SCHEMA, "stage": "build_candidates", "n_candidates": len(rows), "output": str(output)}
        _write_json(outdir / "candidate_pool.metadata.json", meta)
        print(f"Candidate rows: {len(rows)}")
        print(f"Wrote pool    : {output}")
        return meta
    if args.command == "score-candidates":
        training = read_ce_training_jsonl(args.seed_training_jsonl.resolve()) if args.seed_training_jsonl else None
        outdir = args.outdir.resolve()
        output = args.output if args.output.is_absolute() else outdir / args.output
        rows = score_candidates(
            _read_csv(args.candidate_csv.resolve()),
            prediction_rows=_prediction_index(args.prediction_csv.resolve()) if args.prediction_csv else None,
            training=training,
            exploration_weight=args.exploration_weight,
            redox_weight=args.redox_weight,
        )
        _write_csv(output, rows, SCORED_FIELDS)
        meta = {"schema": SCHEMA, "stage": "score_candidates", "n_scored": len(rows), "output": str(output)}
        _write_json(outdir / "scored_candidates.metadata.json", meta)
        print(f"Scored rows: {len(rows)}")
        print(f"Wrote score: {output}")
        return meta
    if args.command == "select-dft":
        outdir = args.outdir.resolve()
        rows = select_candidates(_read_csv(args.scored_csv.resolve()), top_n=args.top_n, min_data_distance=args.min_data_distance)
        selected_csv = outdir / "selected_dft_candidates.csv"
        _write_csv(selected_csv, rows, SCORED_FIELDS)
        training = selected_to_training_set(rows, system_name=args.system, parent_structure=args.parent_structure)
        training_jsonl = outdir / "selected_candidates_mode4_prior.jsonl"
        write_ce_training_jsonl(training_jsonl, training)
        meta = {
            "schema": SCHEMA,
            "stage": "select_dft",
            "n_selected": len(rows),
            "selected_csv": str(selected_csv),
            "mode4_prior_jsonl": str(training_jsonl),
            "reminder": "When this stage is reached, run VASP on selected candidates before using predicted energies as thermodynamic labels.",
        }
        _write_json(outdir / "selected_dft_candidates.metadata.json", meta)
        print(f"Selected rows : {len(rows)}")
        print(f"Wrote selected: {selected_csv}")
        print(f"Mode4 prior   : {training_jsonl}")
        return meta
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
