"""Assemble motif-resolved free-energy tables for zentropy solves."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from atomi.zentropy.stage_utils import (
    composition_label,
    energy_per_fu,
    finite_float,
    format_value,
    load_motif_records,
    load_table,
    motif_family,
    motif_id,
    parse_temperature_values,
    value_eV,
    volume_per_fu,
    write_csv,
    write_json,
)


SCHEMA = "atomi.zentropy.free_energy.v1"


FIELDS = [
    "motif_id",
    "motif_family",
    "defect_label",
    "composition",
    "T_K",
    "degeneracy",
    "formula_units",
    "G_eV_per_fu",
    "H_eV_per_fu",
    "S_eV_per_fuK",
    "Cp_eV_per_fuK",
    "V_A3_per_fu",
    "uncertainty_eV_per_fu",
    "source",
    "source_detail",
]


def _index_thermo_rows(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {"__global__": []}
    for path in paths:
        for row in load_table(path):
            clean = dict(row)
            clean["_source_path"] = str(path.resolve())
            keys = [
                str(clean.get("motif_id") or "").strip(),
                str(clean.get("motif_family") or clean.get("family") or "").strip(),
                str(clean.get("defect_label") or "").strip(),
            ]
            attached = False
            for key in keys:
                if key:
                    indexed.setdefault(key, []).append(clean)
                    attached = True
            if not attached:
                indexed["__global__"].append(clean)
    return indexed


def _matching_thermo_rows(index: dict[str, list[dict[str, Any]]], motif: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [motif_id(motif), motif_family(motif), str(motif.get("defect_label") or "").strip()]
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for key in keys:
        for row in index.get(key, []):
            marker = id(row)
            if marker not in seen:
                rows.append(row)
                seen.add(marker)
    if not rows:
        rows.extend(index.get("__global__", []))
    return rows


def _temperature(row: dict[str, Any]) -> float | None:
    return finite_float(row.get("T_K") or row.get("temperature_K") or row.get("temperature"))


def _row_for_temperature(rows: list[dict[str, Any]], temp: float) -> dict[str, Any] | None:
    exact = [row for row in rows if _temperature(row) is not None and abs(float(_temperature(row)) - temp) < 1.0e-8]
    if exact:
        return exact[0]
    if not rows:
        return None
    timeless = [row for row in rows if _temperature(row) is None]
    if timeless:
        return timeless[0]
    return None


def assemble_free_energy_rows(
    motif_records: list[dict[str, Any]],
    *,
    thermo_paths: list[Path],
    temperatures: list[float],
    reference_shift_eV_per_fu: float = 0.0,
) -> list[dict[str, Any]]:
    thermo_index = _index_thermo_rows(thermo_paths)
    out: list[dict[str, Any]] = []
    for idx, motif in enumerate(motif_records, start=1):
        mid = motif_id(motif, f"motif_{idx:04d}")
        family = motif_family(motif)
        static_g = energy_per_fu(motif)
        static_v = volume_per_fu(motif)
        matched = _matching_thermo_rows(thermo_index, motif)
        row_temperatures = sorted({float(t) for row in matched if (t := _temperature(row)) is not None})
        active_temperatures = row_temperatures or temperatures
        for temp in active_temperatures:
            thermo = _row_for_temperature(matched, temp) or {}
            correction = value_eV(
                thermo,
                "G_correction_eV_per_fu",
                "delta_G_eV_per_fu",
                kj_mol_keys=("G_correction_kJ_mol", "delta_G_kJ_mol"),
            )
            g_value = value_eV(
                thermo,
                "G_eV_per_fu",
                "free_energy_eV_per_fu",
                kj_mol_keys=("G_kJ_mol", "G_kJ_mol_fu", "free_energy_kJ_mol"),
            )
            if g_value is None and static_g is not None:
                g_value = static_g + (correction or 0.0) + reference_shift_eV_per_fu
            elif g_value is not None:
                g_value += reference_shift_eV_per_fu
            h_value = value_eV(thermo, "H_eV_per_fu", "enthalpy_eV_per_fu", kj_mol_keys=("H_kJ_mol", "enthalpy_kJ_mol"))
            s_value = value_eV(
                thermo,
                "S_eV_per_fuK",
                "entropy_eV_per_fuK",
                kj_mol_keys=("S_kJ_molK", "entropy_kJ_molK"),
            )
            cp_value = value_eV(
                thermo,
                "Cp_eV_per_fuK",
                "heat_capacity_eV_per_fuK",
                kj_mol_keys=("Cp_kJ_molK", "heat_capacity_kJ_molK"),
            )
            v_value = finite_float(
                thermo.get("V_A3_per_fu")
                or thermo.get("volume_A3_per_fu")
                or thermo.get("volume_per_formula_unit_A3")
            )
            out.append(
                {
                    "motif_id": mid,
                    "motif_family": family,
                    "defect_label": motif.get("defect_label") or "",
                    "composition": composition_label(motif),
                    "T_K": temp,
                    "degeneracy": finite_float(motif.get("degeneracy")) or 1.0,
                    "formula_units": finite_float(motif.get("formula_units"))
                    or finite_float((motif.get("size_normalization") or {}).get("formula_units"))
                    or "",
                    "G_eV_per_fu": g_value,
                    "H_eV_per_fu": h_value,
                    "S_eV_per_fuK": s_value,
                    "Cp_eV_per_fuK": cp_value,
                    "V_A3_per_fu": v_value if v_value is not None else static_v,
                    "uncertainty_eV_per_fu": value_eV(
                        thermo,
                        "uncertainty_eV_per_fu",
                        "sigma_G_eV_per_fu",
                        kj_mol_keys=("uncertainty_kJ_mol", "sigma_G_kJ_mol"),
                    ),
                    "source": "thermo_table" if thermo else "static_motif_energy",
                    "source_detail": thermo.get("_source_path") or motif.get("run_dir") or "",
                }
            )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy-free-energy",
        description="Assemble motif-resolved G_i(T) tables from a motif DB and optional QHA/MD thermodynamic tables.",
    )
    parser.add_argument("--motif-db", type=Path, required=True, help="defect_motif_db.json or compatible motif CSV.")
    parser.add_argument(
        "--thermo-csv",
        type=Path,
        action="append",
        default=[],
        help="Optional motif/family/global thermo table. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--temperature",
        action="append",
        help="Temperature in K or start:stop:step grid used when no thermo table temperatures are available.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("stage2_free_energy"))
    parser.add_argument("--output-csv", type=Path, help="Override microstate free-energy CSV path.")
    parser.add_argument("--reference-shift-eV-per-fu", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    payload, records = load_motif_records(args.motif_db)
    temperatures = parse_temperature_values(args.temperature)
    rows = assemble_free_energy_rows(
        records,
        thermo_paths=[path.resolve() for path in args.thermo_csv],
        temperatures=temperatures,
        reference_shift_eV_per_fu=args.reference_shift_eV_per_fu,
    )
    outdir = args.outdir.resolve()
    csv_path = args.output_csv.resolve() if args.output_csv else outdir / "microstate_free_energy.csv"
    json_path = outdir / "microstate_free_energy.json"
    write_csv(csv_path, rows, FIELDS)
    metadata = {
        "schema": SCHEMA,
        "motif_db_schema": payload.get("schema", ""),
        "inputs": {
            "motif_db": str(args.motif_db.resolve()),
            "thermo_csv": [str(path.resolve()) for path in args.thermo_csv],
        },
        "outputs": {"csv": str(csv_path), "json": str(json_path)},
        "n_motifs": len(records),
        "n_rows": len(rows),
        "temperature_grid_K": temperatures,
        "fields": FIELDS,
    }
    write_json(json_path, {"metadata": metadata, "rows": rows})
    write_json(outdir / "microstate_free_energy_metadata.json", metadata)
    print(f"Microstate rows : {len(rows)}")
    print(f"Wrote CSV       : {csv_path}")
    print(f"Wrote JSON      : {json_path}")
    if rows and all(not format_value(row.get("G_eV_per_fu")) for row in rows):
        print("Warning         : no G_eV_per_fu values were assembled; check motif and thermo inputs.")
    return metadata


if __name__ == "__main__":
    main()
