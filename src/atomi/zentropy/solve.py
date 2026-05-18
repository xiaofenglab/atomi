"""Lightweight zentropy ensemble solver for motif free-energy tables."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from atomi.zentropy.stage_utils import K_B_EV_PER_K, finite_float, load_table, write_csv, write_json


SCHEMA = "atomi.zentropy.solve.v1"

PROBABILITY_FIELDS = [
    "group_key",
    "T_K",
    "motif_id",
    "motif_family",
    "composition",
    "degeneracy",
    "G_eV_per_fu",
    "delta_G_eV_per_fu",
    "probability",
    "probability_percent",
    "weight",
    "accepted",
    "source",
]

THERMO_FIELDS = [
    "group_key",
    "T_K",
    "n_states",
    "G_ensemble_eV_per_fu",
    "G_min_eV_per_fu",
    "H_ensemble_eV_per_fu",
    "S_ensemble_eV_per_fuK",
    "Cp_ensemble_eV_per_fuK",
    "V_ensemble_A3_per_fu",
    "dominant_motif_id",
    "dominant_motif_family",
    "dominant_probability",
]


def _group_key(row: dict[str, Any], columns: list[str]) -> str:
    if not columns:
        return "all"
    return "|".join(f"{column}={row.get(column, '')}" for column in columns)


def _weighted_average(rows: list[dict[str, Any]], probs: list[float], field: str) -> float | None:
    total = 0.0
    norm = 0.0
    for row, prob in zip(rows, probs):
        value = finite_float(row.get(field))
        if value is None:
            continue
        total += prob * value
        norm += prob
    if norm == 0.0:
        return None
    return total / norm


def solve_rows(
    rows: list[dict[str, Any]],
    *,
    group_by: list[str],
    max_delta_eV_per_fu: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        temp = finite_float(row.get("T_K") or row.get("temperature_K"))
        energy = finite_float(row.get("G_eV_per_fu"))
        if temp is None or energy is None:
            continue
        grouped[(_group_key(row, group_by), temp)].append(row)

    probability_rows: list[dict[str, Any]] = []
    thermo_rows: list[dict[str, Any]] = []
    for (group_key, temp), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        energies = [float(finite_float(row.get("G_eV_per_fu"))) for row in group_rows]
        g_min = min(energies)
        accepted = [
            max_delta_eV_per_fu is None or (float(finite_float(row.get("G_eV_per_fu"))) - g_min) <= max_delta_eV_per_fu
            for row in group_rows
        ]
        beta = 1.0 / (K_B_EV_PER_K * temp)
        weights: list[float] = []
        for row, keep in zip(group_rows, accepted):
            if not keep:
                weights.append(0.0)
                continue
            degeneracy = finite_float(row.get("degeneracy")) or 1.0
            delta = float(finite_float(row.get("G_eV_per_fu"))) - g_min
            weights.append(max(degeneracy, 0.0) * math.exp(-beta * delta))
        z_value = sum(weights)
        if z_value <= 0.0:
            continue
        probs = [weight / z_value for weight in weights]
        g_ensemble = g_min - (1.0 / beta) * math.log(z_value)
        dominant_index = max(range(len(probs)), key=lambda idx: probs[idx])
        dominant = group_rows[dominant_index]

        for row, prob, weight, keep in zip(group_rows, probs, weights, accepted):
            energy = float(finite_float(row.get("G_eV_per_fu")))
            probability_rows.append(
                {
                    "group_key": group_key,
                    "T_K": temp,
                    "motif_id": row.get("motif_id") or "",
                    "motif_family": row.get("motif_family") or "",
                    "composition": row.get("composition") or "",
                    "degeneracy": finite_float(row.get("degeneracy")) or 1.0,
                    "G_eV_per_fu": energy,
                    "delta_G_eV_per_fu": energy - g_min,
                    "probability": prob,
                    "probability_percent": 100.0 * prob,
                    "weight": weight,
                    "accepted": "yes" if keep else "no",
                    "source": row.get("source") or "",
                }
            )

        thermo_rows.append(
            {
                "group_key": group_key,
                "T_K": temp,
                "n_states": len(group_rows),
                "G_ensemble_eV_per_fu": g_ensemble,
                "G_min_eV_per_fu": g_min,
                "H_ensemble_eV_per_fu": _weighted_average(group_rows, probs, "H_eV_per_fu"),
                "S_ensemble_eV_per_fuK": _weighted_average(group_rows, probs, "S_eV_per_fuK"),
                "Cp_ensemble_eV_per_fuK": _weighted_average(group_rows, probs, "Cp_eV_per_fuK"),
                "V_ensemble_A3_per_fu": _weighted_average(group_rows, probs, "V_A3_per_fu"),
                "dominant_motif_id": dominant.get("motif_id") or "",
                "dominant_motif_family": dominant.get("motif_family") or "",
                "dominant_probability": probs[dominant_index],
            }
        )
    return probability_rows, thermo_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy-solve",
        description="Solve motif probabilities from microstate G_i(T) and degeneracy tables.",
    )
    parser.add_argument("--free-energy-csv", type=Path, required=True, help="microstate_free_energy.csv from zentropy-free-energy.")
    parser.add_argument("--outdir", type=Path, default=Path("stage3_zentropy_solve"))
    parser.add_argument("--probability-csv", type=Path, help="Override ensemble probability CSV path.")
    parser.add_argument("--thermo-csv", type=Path, help="Override ensemble thermo CSV path.")
    parser.add_argument(
        "--group-by",
        action="append",
        default=[],
        help="Column used to solve independent ensembles, for example composition. Repeat for multiple columns.",
    )
    parser.add_argument(
        "--max-delta-eV-per-fu",
        type=float,
        help="Optional pruning window relative to the lowest G at each T/group.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    rows = load_table(args.free_energy_csv.resolve())
    probability_rows, thermo_rows = solve_rows(
        rows,
        group_by=args.group_by,
        max_delta_eV_per_fu=args.max_delta_eV_per_fu,
    )
    outdir = args.outdir.resolve()
    probability_csv = args.probability_csv.resolve() if args.probability_csv else outdir / "ensemble_probabilities.csv"
    thermo_csv = args.thermo_csv.resolve() if args.thermo_csv else outdir / "zentropy_thermo_functions.csv"
    write_csv(probability_csv, probability_rows, PROBABILITY_FIELDS)
    write_csv(thermo_csv, thermo_rows, THERMO_FIELDS)
    metadata = {
        "schema": SCHEMA,
        "inputs": {"free_energy_csv": str(args.free_energy_csv.resolve())},
        "outputs": {"probabilities": str(probability_csv), "thermo": str(thermo_csv)},
        "group_by": args.group_by,
        "max_delta_eV_per_fu": args.max_delta_eV_per_fu,
        "n_probability_rows": len(probability_rows),
        "n_thermo_rows": len(thermo_rows),
        "notes": [
            "This fallback solver treats motifs as discrete microstates with ideal Boltzmann weights and recorded degeneracy.",
            "Use group-by columns for fixed-composition or fixed-defect-family ensemble solves.",
        ],
    }
    write_json(outdir / "zentropy_solve_metadata.json", metadata)
    print(f"Probability rows : {len(probability_rows)}")
    print(f"Thermo rows      : {len(thermo_rows)}")
    print(f"Wrote probability: {probability_csv}")
    print(f"Wrote thermo     : {thermo_csv}")
    return metadata


if __name__ == "__main__":
    main()
