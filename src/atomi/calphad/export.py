"""Export CALPHAD/free-energy tables for MOOSE-facing workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from atomi.calphad.env import inspect_calphad_environment
from atomi.core.cell import cell_metadata, extensive_basis_factor, infer_formula_units


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value.strip()).strip("_") or "phase"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.12g}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def read_neighbor_metadata(property_csv: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in (
        "hybrid_cp_entropy_metadata.json",
        "normalization_metadata.json",
        "temperature_range_metadata.json",
    ):
        path = property_csv.parent / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        merged[name] = payload
        for key in (
            "basis",
            "energy_basis",
            "formula",
            "target_z_formula_units",
            "qha_formula_units",
            "md_formula_units",
            "n_formula_units",
            "cell_metadata",
        ):
            if key in payload and key not in merged:
                merged[key] = payload[key]
    return merged


def metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    cell = metadata.get("cell_metadata") if isinstance(metadata.get("cell_metadata"), dict) else {}
    for key in keys:
        if key in metadata:
            return metadata[key]
        if key in cell:
            return cell[key]
    return None


def source_basis_from_metadata(metadata: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    basis = metadata_value(metadata, "normalization_basis", "basis", "energy_basis")
    if basis in {"per-formula", "target-cell", "simulation-cell"}:
        return str(basis)
    return "per-formula"


def first_finite(row: dict[str, Any], names: tuple[str, ...]) -> tuple[float | None, str]:
    for name in names:
        value = finite_float(row.get(name))
        if value is not None:
            return value, name
    return None, ""


def add_calphad_canonical_fields(row: dict[str, Any], factor: float) -> None:
    cp, cp_source = first_finite(
        row,
        (
            "Cp",
            "Cp_used_for_integration_J_per_mol_UO2_K",
            "Cp_from_dH_J_per_mol_UO2_K",
            "Cp_J_per_mol_formula_K",
        ),
    )
    if cp is not None and "Cp_J_molK" not in row:
        row["Cp_J_molK"] = cp * factor
        row["Cp_source_column"] = cp_source
    entropy, entropy_source = first_finite(
        row,
        (
            "S_neel_corrected",
            "S_integrated",
            "S_rel_J_per_mol_UO2_K",
            "S_rel_J_mol_K",
            "S_rel_J_per_mol_formula_K",
        ),
    )
    if entropy is not None and "S_J_molK" not in row:
        row["S_J_molK"] = entropy * factor
        row["S_source_column"] = entropy_source
    h_kj, h_kj_source = first_finite(
        row,
        ("H_neel_corrected_kJ_mol", "H_integrated_kJ_mol", "H_abs_kJ_per_mol_UO2"),
    )
    h_j, h_j_source = first_finite(row, ("H_rel_J_per_mol_UO2", "H_rel_J_mol", "H_rel_J_per_mol_formula"))
    if "H_J_mol" not in row:
        if h_kj is not None:
            row["H_J_mol"] = h_kj * 1000.0 * factor
            row["H_source_column"] = h_kj_source
        elif h_j is not None:
            row["H_J_mol"] = h_j * factor
            row["H_source_column"] = h_j_source
    g_kj, g_kj_source = first_finite(
        row,
        ("G_neel_corrected_kJ_mol", "G_integrated_kJ_mol", "G_relative_kJ_mol"),
    )
    g_j, g_j_source = first_finite(row, ("G_rel_J_per_mol_UO2", "G_rel_J_mol", "G_rel_J_per_mol_formula"))
    if "G_J_mol" not in row:
        if g_kj is not None:
            row["G_J_mol"] = g_kj * 1000.0 * factor
            row["G_source_column"] = g_kj_source
        elif g_j is not None:
            row["G_J_mol"] = g_j * factor
            row["G_source_column"] = g_j_source


def normalize_rows(
    rows: list[dict[str, str]],
    *,
    phase: str | None,
    material: str,
    formula: str,
    basis_factor: float,
    output_basis: str,
    cell_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    extra_fields: list[str] = []
    for row in rows:
        temp = finite_float(row.get("T_K") or row.get("temperature_K") or row.get("T"))
        if temp is None:
            continue
        clean: dict[str, Any] = {
            "T_K": temp,
            "phase": row.get("phase") or phase or material,
            "composition": row.get("composition") or row.get("x") or formula or "",
            "formula": formula or row.get("formula") or "",
            "normalization_basis": output_basis,
            "n_formula_units": cell_meta.get("n_formula_units"),
            "target_z_formula_units": cell_meta.get("target_z_formula_units"),
        }
        for key, value in row.items():
            if key in {"T_K", "temperature_K", "T", "phase", "composition", "x"}:
                continue
            parsed = finite_float(value)
            if parsed is None:
                if value not in (None, ""):
                    clean[key] = value
            else:
                clean[key] = parsed
                if key not in extra_fields:
                    extra_fields.append(key)
        add_calphad_canonical_fields(clean, basis_factor)
        for key in ("G_J_mol", "H_J_mol", "S_J_molK", "Cp_J_molK"):
            if key in clean and key not in extra_fields:
                extra_fields.insert(0, key)
        normalized.append(clean)
    return sorted(normalized, key=lambda item: (str(item.get("phase")), float(item["T_K"]))), extra_fields


def series_by_field(rows: list[dict[str, Any]], field: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        value = finite_float(row.get(field))
        temp = finite_float(row.get("T_K"))
        if temp is not None and value is not None:
            points.append((temp, value))
    return sorted(points)


def write_moose_template(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    material: str,
    fields: list[str],
    temperature_variable: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = safe_name(material).lower()
    lines = [
        "# MOOSE phase-field/CALPHAD bridge generated by Atomi.",
        "# Units are inherited from the input table; use J/mol and consistent composition units for CALPHAD work.",
        "# This file is a starting template because exact free-energy material kernels depend on the MOOSE app.",
        "",
        "[Functions]",
    ]
    for field in fields:
        points = series_by_field(rows, field)
        if len(points) < 1:
            continue
        function = f"{prefix}_{safe_name(field)}"
        x_values = " ".join(format_value(temp) for temp, _ in points)
        y_values = " ".join(format_value(value) for _, value in points)
        lines.extend(
            [
                f"  [./{function}]",
                "    type = PiecewiseLinear",
                f"    x = '{x_values}'",
                f"    y = '{y_values}'",
                "  [../]",
            ]
        )
    lines.extend(
        [
            "[]",
            "",
            "[Materials]",
            f"  [./{prefix}_calphad_table_bridge]",
            "    type = GenericFunctionMaterial",
            f"    prop_names = '{' '.join(fields)}'",
            f"    prop_values = '{' '.join(f'{prefix}_{safe_name(field)}' for field in fields)}'",
            f"    # temperature variable expected by the functions: {temperature_variable}",
            "    # For phase-field, connect these table functions to DerivativeParsedMaterial",
            "    # or to an application-specific CALPHAD free-energy material.",
            "  [../]",
            "[]",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calphad_export",
        description="Normalize CALPHAD/pycalphad property tables and write MOOSE free-energy bridge templates.",
    )
    parser.add_argument("--tdb", type=Path, help="Optional TDB path to inspect with pycalphad.")
    parser.add_argument("--property-csv", type=Path, required=True, help="CSV containing T_K and G/mu/Cp/property columns.")
    parser.add_argument("--outdir", type=Path, default=Path("analysis/calphad_export"))
    parser.add_argument("--material", default="material")
    parser.add_argument("--formula", help="Formula label/composition, e.g. UO2. Defaults to --material.")
    parser.add_argument("--natoms", type=float, help="Atoms in the source simulation cell.")
    parser.add_argument("--atoms-per-formula-unit", type=float, help="Atoms per formula unit.")
    parser.add_argument("--formula-units", type=float, help="Formula units in the source simulation cell.")
    parser.add_argument("--target-z", type=float, help="Formula units in the target crystallographic cell, e.g. 4 for fluorite UO2.")
    parser.add_argument(
        "--input-basis",
        choices=("auto", "per-formula", "target-cell", "simulation-cell"),
        default="auto",
        help="Basis of extensive source columns. auto reads Atomi metadata when present.",
    )
    parser.add_argument(
        "--output-basis",
        choices=("per-formula", "target-cell", "simulation-cell"),
        default="per-formula",
        help="Basis for canonical CALPHAD fields G_J_mol/H_J_mol/S_J_molK/Cp_J_molK.",
    )
    parser.add_argument("--phase", help="Default phase label when the CSV does not include a phase column.")
    parser.add_argument("--component", action="append", default=[], help="Components to record in metadata.")
    parser.add_argument("--export-field", action="append", default=[], help="Fields to expose in the MOOSE include.")
    parser.add_argument("--temperature-variable", default="T")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    property_csv = args.property_csv.resolve()
    metadata = read_neighbor_metadata(property_csv)
    formula = args.formula or str(metadata_value(metadata, "formula") or args.material)
    formula_units = infer_formula_units(
        formula_units=args.formula_units
        or finite_float(metadata_value(metadata, "n_formula_units", "md_formula_units", "qha_formula_units")),
        natoms=args.natoms,
        atoms_per_formula_unit=args.atoms_per_formula_unit,
        formula=formula,
    )
    target_z = args.target_z or finite_float(metadata_value(metadata, "target_z_formula_units", "target_z"))
    input_basis = source_basis_from_metadata(metadata, args.input_basis)
    basis_factor = extensive_basis_factor(
        from_basis=input_basis,
        to_basis=args.output_basis,
        formula_units=formula_units,
        target_z=target_z,
    )
    cell_meta = cell_metadata(
        formula=formula,
        natoms=args.natoms,
        atoms_per_formula_unit=args.atoms_per_formula_unit,
        formula_units=formula_units,
        target_z=target_z,
        cell_role="calphad-source-cell",
        normalization_basis=args.output_basis,
    )
    source_rows = read_csv(property_csv)
    rows, numeric_fields = normalize_rows(
        source_rows,
        phase=args.phase,
        material=args.material,
        formula=formula,
        basis_factor=basis_factor,
        output_basis=args.output_basis,
        cell_meta=cell_meta,
    )
    export_fields = args.export_field or [
        field
        for field in numeric_fields
        if field.startswith(("G", "mu", "Cp", "H", "S", "mobility", "D_", "k_"))
    ]
    base_fields = ["T_K", "phase", "composition", "formula", "normalization_basis", "n_formula_units", "target_z_formula_units"]
    fields = [*base_fields, *[field for field in numeric_fields if field not in base_fields]]
    outdir = args.outdir.resolve()
    table = outdir / "calphad_property_table.csv"
    write_csv(table, rows, fields)
    include = outdir / f"{safe_name(args.material)}_phase_field_free_energy.i"
    write_moose_template(
        include,
        rows,
        material=args.material,
        fields=export_fields,
        temperature_variable=args.temperature_variable,
    )
    db_report = inspect_calphad_environment(
        database=args.tdb.resolve() if args.tdb else None,
        components=args.component,
        phases=[args.phase] if args.phase else None,
    )
    metadata = {
        "schema": "atomi.calphad.export.v1",
        "inputs": {
            "property_csv": str(property_csv),
            "tdb": str(args.tdb.resolve()) if args.tdb else "",
        },
        "outputs": {
            "property_table": str(table),
            "moose_template": str(include),
        },
        "material": args.material,
        "formula": formula,
        "phase": args.phase or "",
        "components": args.component,
        "cell_metadata": cell_meta,
        "unit_conversion": {
            "input_basis": input_basis,
            "output_basis": args.output_basis,
            "extensive_basis_factor": basis_factor,
            "canonical_fields": ["G_J_mol", "H_J_mol", "S_J_molK", "Cp_J_molK"],
        },
        "source_metadata": metadata,
        "export_fields": export_fields,
        "calphad_environment": db_report,
        "notes": [
            "This command exports tabulated CALPHAD quantities and a MOOSE bridge template.",
            "Exact phase-field free-energy material syntax should be specialized for the target MOOSE app.",
        ],
    }
    write_json(outdir / "calphad_export_metadata.json", metadata)
    print(f"Wrote CALPHAD property table: {table}")
    print(f"Wrote MOOSE phase-field template: {include}")
    return metadata


if __name__ == "__main__":
    main()
