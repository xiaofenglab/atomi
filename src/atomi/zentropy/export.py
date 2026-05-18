"""Export zentropy ensemble results to downstream thermodynamic tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from atomi.zentropy.stage_utils import EV_PER_KJ_MOL, finite_float, load_table, write_csv, write_json


SCHEMA = "atomi.zentropy.export.v1"
KJ_MOL_PER_EV = 1.0 / EV_PER_KJ_MOL


CALPHAD_FIELDS = [
    "dataset_id",
    "phase",
    "group_key",
    "T_K",
    "G_kJ_mol",
    "H_kJ_mol",
    "S_J_molK",
    "Cp_J_molK",
    "V_A3_per_fu",
    "dominant_motif_id",
    "dominant_motif_family",
    "dominant_probability",
    "source",
]

POPULATION_FIELDS = [
    "group_key",
    "T_K",
    "motif_id",
    "motif_family",
    "composition",
    "probability",
    "probability_percent",
    "delta_G_eV_per_fu",
    "accepted",
]


def _ev_to_kj(value: Any) -> float | None:
    number = finite_float(value)
    if number is None:
        return None
    return number * KJ_MOL_PER_EV


def _ev_per_k_to_j(value: Any) -> float | None:
    number = finite_float(value)
    if number is None:
        return None
    return number * KJ_MOL_PER_EV * 1000.0


def build_calphad_rows(rows: list[dict[str, Any]], *, phase: str, dataset_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "dataset_id": dataset_id,
                "phase": phase,
                "group_key": row.get("group_key") or "",
                "T_K": row.get("T_K") or "",
                "G_kJ_mol": _ev_to_kj(row.get("G_ensemble_eV_per_fu")),
                "H_kJ_mol": _ev_to_kj(row.get("H_ensemble_eV_per_fu")),
                "S_J_molK": _ev_per_k_to_j(row.get("S_ensemble_eV_per_fuK")),
                "Cp_J_molK": _ev_per_k_to_j(row.get("Cp_ensemble_eV_per_fuK")),
                "V_A3_per_fu": row.get("V_ensemble_A3_per_fu") or "",
                "dominant_motif_id": row.get("dominant_motif_id") or "",
                "dominant_motif_family": row.get("dominant_motif_family") or "",
                "dominant_probability": row.get("dominant_probability") or "",
                "source": "zentropy-solve",
            }
        )
    return out


def build_moose_rows(rows: list[dict[str, Any]], *, material: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "material": material,
                "group_key": row.get("group_key") or "",
                "T_K": row.get("T_K") or "",
                "free_energy_kJ_mol": _ev_to_kj(row.get("G_ensemble_eV_per_fu")),
                "enthalpy_kJ_mol": _ev_to_kj(row.get("H_ensemble_eV_per_fu")),
                "entropy_J_molK": _ev_per_k_to_j(row.get("S_ensemble_eV_per_fuK")),
                "heat_capacity_J_molK": _ev_per_k_to_j(row.get("Cp_ensemble_eV_per_fuK")),
                "volume_A3_per_fu": row.get("V_ensemble_A3_per_fu") or "",
                "dominant_motif_id": row.get("dominant_motif_id") or "",
                "dominant_probability": row.get("dominant_probability") or "",
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy-export",
        description="Export zentropy ensemble thermo and motif populations to CALPHAD/MOOSE bridge tables.",
    )
    parser.add_argument("--thermo-csv", type=Path, required=True, help="zentropy_thermo_functions.csv from zentropy-solve.")
    parser.add_argument("--probability-csv", type=Path, help="ensemble_probabilities.csv from zentropy-solve.")
    parser.add_argument("--outdir", type=Path, default=Path("stage4_zentropy_export"))
    parser.add_argument("--material", default="material")
    parser.add_argument("--phase", default="defect_phase")
    parser.add_argument("--dataset-id", default="zentropy_dataset")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    thermo_rows = load_table(args.thermo_csv.resolve())
    probability_rows = load_table(args.probability_csv.resolve()) if args.probability_csv else []
    outdir = args.outdir.resolve()
    calphad_csv = outdir / "calphad_pseudodata.csv"
    moose_csv = outdir / "moose_material_table.csv"
    population_csv = outdir / "defect_population_table.csv"
    write_csv(calphad_csv, build_calphad_rows(thermo_rows, phase=args.phase, dataset_id=args.dataset_id), CALPHAD_FIELDS)
    write_csv(
        moose_csv,
        build_moose_rows(thermo_rows, material=args.material),
        [
            "material",
            "group_key",
            "T_K",
            "free_energy_kJ_mol",
            "enthalpy_kJ_mol",
            "entropy_J_molK",
            "heat_capacity_J_molK",
            "volume_A3_per_fu",
            "dominant_motif_id",
            "dominant_probability",
        ],
    )
    write_csv(population_csv, probability_rows, POPULATION_FIELDS)
    metadata = {
        "schema": SCHEMA,
        "inputs": {
            "thermo_csv": str(args.thermo_csv.resolve()),
            "probability_csv": str(args.probability_csv.resolve()) if args.probability_csv else "",
        },
        "outputs": {
            "calphad_pseudodata": str(calphad_csv),
            "moose_material_table": str(moose_csv),
            "defect_population_table": str(population_csv),
        },
        "material": args.material,
        "phase": args.phase,
        "dataset_id": args.dataset_id,
    }
    write_json(outdir / "zentropy_export_metadata.json", metadata)
    print(f"Wrote CALPHAD table : {calphad_csv}")
    print(f"Wrote MOOSE table   : {moose_csv}")
    print(f"Wrote populations   : {population_csv}")
    return metadata


if __name__ == "__main__":
    main()
