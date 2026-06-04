"""Overlay SLUSCHI/LAMMPS entropy points against QHA-MD entropy curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEMPERATURE_COLUMNS = ("temperature_K", "T_K", "temperature", "T", "target_T_K")
SCONF_COLUMNS = (
    "Sconf_J_mol_formula_K",
    "sconf_J_mol_formula_K",
    "configurational_entropy_J_mol_formula_K",
    "mean_pair_sconfig_J_mol_formula_K",
    "mean_pair_sconfig_J_mol_atom_K",
    "sconfig_J_mol_atom_K",
    "configurational_entropy_J_mol_atom_K",
)
SVIB_COLUMNS = (
    "Svib_J_mol_formula_K",
    "svib_J_mol_formula_K",
    "vibrational_entropy_J_mol_formula_K",
    "Svib_J_mol_atom_K",
    "svib_J_mol_atom_K",
    "vibrational_entropy_J_mol_atom_K",
)
TOTAL_COLUMNS = (
    "Stotal_J_mol_formula_K",
    "stotal_J_mol_formula_K",
    "total_entropy_J_mol_formula_K",
    "entropy_J_mol_formula_K",
    "Stotal_J_mol_atom_K",
    "stotal_J_mol_atom_K",
    "total_entropy_J_mol_atom_K",
    "entropy_J_mol_atom_K",
)
STD_COLUMNS = (
    "stderr_J_mol_formula_K",
    "sem_J_mol_formula_K",
    "std_J_mol_formula_K",
    "entropy_stderr_J_mol_formula_K",
    "entropy_std_J_mol_formula_K",
    "sconfig_stderr_J_mol_formula_K",
    "sconfig_std_J_mol_formula_K",
    "svib_stderr_J_mol_formula_K",
    "svib_std_J_mol_formula_K",
    "stderr_J_mol_atom_K",
    "sem_J_mol_atom_K",
    "std_J_mol_atom_K",
    "entropy_stderr_J_mol_atom_K",
    "entropy_std_J_mol_atom_K",
    "sconfig_stderr_J_mol_atom_K",
    "sconfig_std_J_mol_atom_K",
    "sem_pair_sconfig_J_mol_atom_K",
    "std_pair_sconfig_J_mol_atom_K",
    "svib_stderr_J_mol_atom_K",
    "svib_std_J_mol_atom_K",
)
LOW_COLUMNS = ("uq_low_J_mol_formula_K", "lower_J_mol_formula_K", "entropy_low_J_mol_formula_K")
HIGH_COLUMNS = ("uq_high_J_mol_formula_K", "upper_J_mol_formula_K", "entropy_high_J_mol_formula_K")


@dataclass
class EntropyPoint:
    source: str
    label: str
    temperature_K: float
    entropy_J_mol_formula_K: float
    yerr_low_J_mol_formula_K: float | None = None
    yerr_high_J_mol_formula_K: float | None = None
    input_csv: str = ""
    value_column: str = ""
    unit_basis: str = ""


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    if not math.isfinite(value_float):
        return None
    return value_float


def _first_existing_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    fields = set(fieldnames)
    for candidate in candidates:
        if candidate in fields:
            return candidate
    return None


def _column_unit_basis(column: str, fallback: str) -> str:
    if "_J_mol_atom_K" in column:
        return "J/mol-atom/K"
    if "_J_mol_formula_K" in column:
        return "J/mol-formula/K"
    return fallback


def _convert_entropy(value: float, unit_basis: str, atoms_per_formula: float) -> float:
    if unit_basis == "J/mol-formula/K":
        return value
    if unit_basis == "kJ/mol-formula/K":
        return value * 1000.0
    if unit_basis == "J/mol-atom/K":
        return value * atoms_per_formula
    if unit_basis == "kJ/mol-atom/K":
        return value * 1000.0 * atoms_per_formula
    raise ValueError(f"Unsupported entropy unit/basis: {unit_basis}")


def load_qha_entropy(
    path: Path,
    *,
    qha_formula_units: float,
    qha_entropy_unit: str,
    t_min: float | None = None,
    t_max: float | None = None,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 2:
                continue
            temp = _float_or_none(parts[0])
            value = _float_or_none(parts[1])
            if temp is None or value is None:
                continue
            if t_min is not None and temp < t_min:
                continue
            if t_max is not None and temp > t_max:
                continue
            if qha_entropy_unit == "J/mol-cell/K":
                entropy = value / qha_formula_units
            elif qha_entropy_unit == "kJ/mol-cell/K":
                entropy = value * 1000.0 / qha_formula_units
            elif qha_entropy_unit == "J/mol-formula/K":
                entropy = value
            elif qha_entropy_unit == "kJ/mol-formula/K":
                entropy = value * 1000.0
            else:
                raise ValueError(f"Unsupported QHA entropy unit: {qha_entropy_unit}")
            points.append((temp, entropy))
    return sorted(points)


def _infer_value_columns(fieldnames: list[str], quantity: str) -> list[str]:
    if quantity == "sconfig":
        column = _first_existing_column(fieldnames, SCONF_COLUMNS)
        return [column] if column else []
    if quantity == "svib":
        column = _first_existing_column(fieldnames, SVIB_COLUMNS)
        return [column] if column else []
    if quantity == "total":
        column = _first_existing_column(fieldnames, TOTAL_COLUMNS)
        if column:
            return [column]
        sconf = _first_existing_column(fieldnames, SCONF_COLUMNS)
        svib = _first_existing_column(fieldnames, SVIB_COLUMNS)
        return [col for col in (svib, sconf) if col]
    for candidates in (TOTAL_COLUMNS, SVIB_COLUMNS, SCONF_COLUMNS):
        column = _first_existing_column(fieldnames, candidates)
        if column:
            return [column]
    return []


def _auto_error_from_row(
    row: dict[str, str],
    fieldnames: list[str],
    *,
    atoms_per_formula: float,
    sluschi_unit: str,
) -> tuple[float | None, float | None]:
    low_col = _first_existing_column(fieldnames, LOW_COLUMNS)
    high_col = _first_existing_column(fieldnames, HIGH_COLUMNS)
    if low_col and high_col:
        low = _float_or_none(row.get(low_col))
        high = _float_or_none(row.get(high_col))
        if low is not None and high is not None:
            basis = _column_unit_basis(low_col, sluschi_unit)
            return (
                abs(_convert_entropy(low, basis, atoms_per_formula)),
                abs(_convert_entropy(high, _column_unit_basis(high_col, sluschi_unit), atoms_per_formula)),
            )
    std_col = _first_existing_column(fieldnames, STD_COLUMNS)
    if std_col:
        std = _float_or_none(row.get(std_col))
        if std is not None:
            converted = abs(_convert_entropy(std, _column_unit_basis(std_col, sluschi_unit), atoms_per_formula))
            return converted, converted
    return None, None


def load_sluschi_entropy_csv(
    path: Path,
    *,
    atoms_per_formula: float,
    quantity: str,
    label: str | None,
    sluschi_unit: str,
    temperature_column: str | None = None,
    value_column: str | None = None,
    sum_columns: list[str] | None = None,
    stderr_column: str | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
) -> list[EntropyPoint]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"{path} has no CSV header")
        temp_col = temperature_column or _first_existing_column(fieldnames, TEMPERATURE_COLUMNS)
        if not temp_col:
            raise ValueError(f"Cannot find a temperature column in {path}")
        if sum_columns:
            value_cols = sum_columns
        elif value_column:
            value_cols = [value_column]
        else:
            value_cols = _infer_value_columns(fieldnames, quantity)
        missing = [column for column in value_cols if column not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing entropy column(s): {', '.join(missing)}")
        if not value_cols:
            raise ValueError(f"Cannot infer an entropy column in {path}; use --value-column or --sum-columns")
        points: list[EntropyPoint] = []
        point_label = label or path.parent.name or path.stem
        for row in reader:
            temp = _float_or_none(row.get(temp_col))
            if temp is None:
                continue
            if t_min is not None and temp < t_min:
                continue
            if t_max is not None and temp > t_max:
                continue
            values: list[float] = []
            bases: list[str] = []
            for column in value_cols:
                raw = _float_or_none(row.get(column))
                if raw is None:
                    continue
                basis = _column_unit_basis(column, sluschi_unit)
                values.append(_convert_entropy(raw, basis, atoms_per_formula))
                bases.append(basis)
            if not values:
                continue
            entropy = sum(values)
            if stderr_column:
                raw_stderr = _float_or_none(row.get(stderr_column))
                if raw_stderr is None:
                    yerr_low = yerr_high = None
                else:
                    converted = abs(_convert_entropy(raw_stderr, _column_unit_basis(stderr_column, sluschi_unit), atoms_per_formula))
                    yerr_low = yerr_high = converted
            else:
                yerr_low, yerr_high = _auto_error_from_row(
                    row,
                    fieldnames,
                    atoms_per_formula=atoms_per_formula,
                    sluschi_unit=sluschi_unit,
                )
            points.append(
                EntropyPoint(
                    source="SLUSCHI",
                    label=point_label,
                    temperature_K=temp,
                    entropy_J_mol_formula_K=entropy,
                    yerr_low_J_mol_formula_K=yerr_low,
                    yerr_high_J_mol_formula_K=yerr_high,
                    input_csv=str(path),
                    value_column="+".join(value_cols),
                    unit_basis="+".join(bases),
                )
            )
    return points


def write_overlay_csv(path: Path, qha_points: list[tuple[float, float]], sluschi_points: list[EntropyPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "label",
        "T_K",
        "entropy_J_mol_formula_K",
        "yerr_low_J_mol_formula_K",
        "yerr_high_J_mol_formula_K",
        "input_csv",
        "value_column",
        "unit_basis",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for temp, value in qha_points:
            writer.writerow(
                {
                    "source": "QHA",
                    "label": "QHA-MD entropy curve",
                    "T_K": temp,
                    "entropy_J_mol_formula_K": value,
                }
            )
        for point in sluschi_points:
            writer.writerow(
                {
                    "source": point.source,
                    "label": point.label,
                    "T_K": point.temperature_K,
                    "entropy_J_mol_formula_K": point.entropy_J_mol_formula_K,
                    "yerr_low_J_mol_formula_K": point.yerr_low_J_mol_formula_K,
                    "yerr_high_J_mol_formula_K": point.yerr_high_J_mol_formula_K,
                    "input_csv": point.input_csv,
                    "value_column": point.value_column,
                    "unit_basis": point.unit_basis,
                }
            )


def plot_overlay(path: Path, qha_points: list[tuple[float, float]], sluschi_points: list[EntropyPoint], title: str) -> None:
    cache = Path(tempfile.gettempdir()) / "atomi-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if qha_points:
        ax.plot(
            [temp for temp, _ in qha_points],
            [value for _, value in qha_points],
            color="#1f77b4",
            lw=2.2,
            label="QHA-MD entropy",
        )
    labels = sorted({point.label for point in sluschi_points})
    colors = plt.cm.tab10.colors
    for idx, label in enumerate(labels):
        group = [point for point in sluschi_points if point.label == label]
        yerr = [
            [point.yerr_low_J_mol_formula_K or 0.0 for point in group],
            [point.yerr_high_J_mol_formula_K or 0.0 for point in group],
        ]
        has_error = any(any(row) for row in yerr)
        ax.errorbar(
            [point.temperature_K for point in group],
            [point.entropy_J_mol_formula_K for point in group],
            yerr=yerr if has_error else None,
            fmt="o",
            ms=5,
            capsize=3 if has_error else 0,
            color=colors[idx % len(colors)],
            label=label,
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Entropy (J mol$^{-1}$ formula$^{-1}$ K$^{-1}$)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lammps-sconfig-qha-overlay",
        description=(
            "Read SLUSCHI/LAMMPS Svib/Sconf entropy CSV outputs and overlay them "
            "against a QHA-MD entropy-temperature.dat curve."
        ),
    )
    parser.add_argument("--qha-entropy", type=Path, required=True, help="QHA entropy-temperature.dat path.")
    parser.add_argument("--sluschi-csv", type=Path, action="append", required=True, help="SLUSCHI/Sconfig/Svib CSV; repeatable.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--atoms-per-formula", type=float, required=True, help="Atoms per formula unit for J/mol-atom/K conversion, e.g. UO2=3.")
    parser.add_argument("--qha-formula-units", type=float, default=1.0, help="Formula units represented by the QHA cell.")
    parser.add_argument(
        "--qha-entropy-unit",
        choices=("J/mol-cell/K", "kJ/mol-cell/K", "J/mol-formula/K", "kJ/mol-formula/K"),
        default="J/mol-cell/K",
    )
    parser.add_argument(
        "--quantity",
        choices=("auto", "sconfig", "svib", "total"),
        default="auto",
        help="Entropy quantity to infer from each SLUSCHI CSV.",
    )
    parser.add_argument(
        "--sluschi-unit",
        choices=("J/mol-atom/K", "kJ/mol-atom/K", "J/mol-formula/K", "kJ/mol-formula/K"),
        default="J/mol-atom/K",
        help="Fallback unit for SLUSCHI columns that do not encode their basis in the column name.",
    )
    parser.add_argument("--temperature-column", default=None)
    parser.add_argument("--value-column", default=None, help="Single SLUSCHI entropy column to plot.")
    parser.add_argument("--sum-columns", default=None, help="Comma-separated columns to sum, e.g. Svib_J_mol_formula_K,Sconf_J_mol_formula_K.")
    parser.add_argument("--stderr-column", default=None, help="Optional symmetric error column.")
    parser.add_argument("--label", action="append", default=None, help="Legend label for each --sluschi-csv, in the same order.")
    parser.add_argument("--title", default="SLUSCHI entropy overlay against QHA-MD")
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    qha_points = load_qha_entropy(
        args.qha_entropy,
        qha_formula_units=args.qha_formula_units,
        qha_entropy_unit=args.qha_entropy_unit,
        t_min=args.t_min,
        t_max=args.t_max,
    )
    sum_columns = [item.strip() for item in args.sum_columns.split(",") if item.strip()] if args.sum_columns else None
    labels = args.label or []
    sluschi_points: list[EntropyPoint] = []
    for idx, csv_path in enumerate(args.sluschi_csv):
        label = labels[idx] if idx < len(labels) else None
        sluschi_points.extend(
            load_sluschi_entropy_csv(
                csv_path,
                atoms_per_formula=args.atoms_per_formula,
                quantity=args.quantity,
                label=label,
                sluschi_unit=args.sluschi_unit,
                temperature_column=args.temperature_column,
                value_column=args.value_column,
                sum_columns=sum_columns,
                stderr_column=args.stderr_column,
                t_min=args.t_min,
                t_max=args.t_max,
            )
        )
    overlay_csv = args.outdir / "sluschi_qha_entropy_overlay.csv"
    metadata_json = args.outdir / "sluschi_qha_entropy_overlay_metadata.json"
    plot_png = args.outdir / "sluschi_qha_entropy_overlay.png"
    write_overlay_csv(overlay_csv, qha_points, sluschi_points)
    plot_error = None
    if not args.no_plot:
        try:
            plot_overlay(plot_png, qha_points, sluschi_points, args.title)
        except ModuleNotFoundError as exc:
            plot_error = f"Plot skipped because a plotting dependency is unavailable: {exc}"
            print(f"WARNING: {plot_error}")
    metadata = {
        "schema": "atomi.lammps.sluschi_qha_entropy_overlay.v1",
        "qha_entropy": str(args.qha_entropy.resolve()),
        "qha_entropy_unit": args.qha_entropy_unit,
        "qha_formula_units": args.qha_formula_units,
        "atoms_per_formula": args.atoms_per_formula,
        "sluschi_csv": [str(path.resolve()) for path in args.sluschi_csv],
        "quantity": args.quantity,
        "sluschi_unit_fallback": args.sluschi_unit,
        "n_qha_points": len(qha_points),
        "n_sluschi_points": len(sluschi_points),
        "outputs": {
            "overlay_csv": str(overlay_csv),
            "plot_png": str(plot_png) if not args.no_plot and plot_error is None else None,
            "metadata_json": str(metadata_json),
        },
        "plot_error": plot_error,
        "notes": [
            "All entropy values in the overlay CSV are normalized to J/mol-formula/K.",
            "SLUSCHI Sconfig outputs from lammps-sconfig are J/mol-atom/K and require --atoms-per-formula.",
            "Use --sum-columns to compare Svib+Sconf totals against QHA-MD entropy when both terms are available.",
        ],
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote overlay CSV : {overlay_csv}")
    if not args.no_plot and plot_error is None:
        print(f"Wrote overlay PNG : {plot_png}")
    print(f"Wrote metadata    : {metadata_json}")
    return metadata


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
