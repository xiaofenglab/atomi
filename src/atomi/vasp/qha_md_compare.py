import argparse
import csv
import json
import math
import sys
from pathlib import Path


EV_TO_KJ_PER_MOL = 96.48533212331002

MD_COLUMN_ALIASES = {
    "V": ("V_fit_A3", "V_mean_A3", "volume_A3"),
    "Cp": (
        "Cp_used_for_integration_J_per_mol_UO2_K",
        "Cp_from_dH_J_per_mol_UO2_K",
        "Cp_fluct_J_per_mol_UO2_K",
        "Cp_J_per_mol_formula_K",
    ),
    "S": (
        "S_rel_J_per_mol_UO2_K",
        "S_rel_J_mol_K",
        "S_rel_J_per_mol_formula_K",
    ),
    "G": (
        "G_rel_J_per_mol_UO2",
        "G_rel_J_mol",
        "G_rel_J_per_mol_formula",
    ),
    "H": (
        "H_rel_J_per_mol_UO2",
        "H_rel_J_mol",
        "H_rel_J_per_mol_formula",
    ),
    "alpha_V": ("alpha_V_micro_per_K", "alpha_V_1_per_K"),
    "K": ("KT_GPa_from_V_fluct", "bulk_modulus_GPa", "K_GPa"),
}


def finite_float(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_table(path: Path) -> list[list[float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            values = [float(part) for part in stripped.split()]
        except ValueError:
            continue
        if len(values) >= 2:
            rows.append(values)
    return rows


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def filter_range(points: list[tuple[float, float]], t_min, t_max) -> list[tuple[float, float]]:
    kept = []
    for temp, value in points:
        if not math.isfinite(temp) or not math.isfinite(value):
            continue
        if t_min is not None and temp < t_min:
            continue
        if t_max is not None and temp > t_max:
            continue
        kept.append((temp, value))
    return kept


def qha_series(path: Path, scale: float = 1.0) -> list[tuple[float, float]]:
    return [(row[0], row[1] * scale) for row in read_table(path)]


def interpolate(points: list[tuple[float, float]], temp: float) -> float | None:
    if not points:
        return None
    points = sorted(points)
    if temp < points[0][0] or temp > points[-1][0]:
        return None
    for point_temp, value in points:
        if abs(point_temp - temp) <= 1.0e-8:
            return value
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= temp <= t1 and t1 != t0:
            frac = (temp - t0) / (t1 - t0)
            return v0 + frac * (v1 - v0)
    return None


def qha_derived_enthalpy(args: argparse.Namespace) -> list[tuple[float, float]]:
    gibbs = qha_series(args.qha_dir / "gibbs-temperature.dat", energy_scale(args))
    entropy = qha_series(args.qha_dir / "entropy-temperature.dat", entropy_qha_scale(args) / 1000.0)
    points = []
    for temp, gibbs_value in gibbs:
        entropy_value = interpolate(entropy, temp)
        if entropy_value is not None:
            points.append((temp, gibbs_value + temp * entropy_value))
    return points


def resolve_column(path: Path, aliases: tuple[str, ...]) -> str | None:
    rows = read_csv_rows(path)
    if not rows:
        return None
    columns = set(rows[0])
    for column in aliases:
        if column in columns:
            return column
    return None


def md_series(
    path: Path,
    aliases: tuple[str, ...],
    scale: float = 1.0,
) -> tuple[list[tuple[float, float]], str]:
    column = resolve_column(path, aliases)
    if column is None:
        return [], ""
    points = []
    for row in read_csv_rows(path):
        temp = finite_float(row.get("T_K", row.get("target_T_K")))
        value = finite_float(row.get(column))
        points.append((temp, value * scale))
    return points, column


def reference_value(
    points: list[tuple[float, float]],
    args: argparse.Namespace,
    reference_temperature: float | None = None,
) -> float | None:
    if not points or args.energy_reference == "none":
        return None
    if reference_temperature is not None:
        target = reference_temperature
        return min(points, key=lambda item: abs(item[0] - target))[1]
    return points[0][1]


def relative_energy(
    points: list[tuple[float, float]],
    args: argparse.Namespace,
    reference_temperature: float | None = None,
) -> list[tuple[float, float]]:
    ref = reference_value(points, args, reference_temperature)
    if ref is None:
        return points
    return [(temp, value - ref) for temp, value in points]


def overlapping_reference_temperature(qha_points, md_points, args) -> float | None:
    if args.energy_reference == "none":
        return None
    if args.energy_reference == "temperature":
        return args.energy_reference_temperature
    if not qha_points or not md_points:
        return None
    qha_min = min(temp for temp, _value in qha_points)
    qha_max = max(temp for temp, _value in qha_points)
    md_min = min(temp for temp, _value in md_points)
    md_max = max(temp for temp, _value in md_points)
    overlap_min = max(qha_min, md_min)
    overlap_max = min(qha_max, md_max)
    if overlap_min <= overlap_max:
        return overlap_min
    return None


def energy_scale(args: argparse.Namespace) -> float:
    if args.qha_energy_unit == "eV-cell":
        per_formula = EV_TO_KJ_PER_MOL / args.qha_formula_units
    elif args.qha_energy_unit == "kJ/mol-formula":
        per_formula = 1.0
    else:
        per_formula = 1.0 / args.qha_formula_units
    if args.energy_basis == "target-cell":
        return per_formula * args.target_z
    return per_formula


def md_energy_scale(args: argparse.Namespace) -> float:
    # LAMMPS thermo-series relative energies are J/mol-formula in the grid CSV.
    per_formula = 1.0 / 1000.0
    if args.energy_basis == "target-cell":
        return per_formula * args.target_z
    return per_formula


def extensive_qha_scale(args: argparse.Namespace) -> float:
    return args.target_z / args.qha_formula_units


def extensive_md_scale(args: argparse.Namespace) -> float:
    return args.target_z / args.md_formula_units


def cp_qha_scale(args: argparse.Namespace) -> float:
    if args.qha_cp_unit == "eV-cell/K":
        per_formula = EV_TO_KJ_PER_MOL * 1000.0 / args.qha_formula_units
    elif args.qha_cp_unit == "J/mol-formula/K":
        per_formula = 1.0
    elif args.qha_cp_unit == "kJ/mol-formula/K":
        per_formula = 1000.0
    elif args.qha_cp_unit == "J/mol-cell/K":
        per_formula = 1.0 / args.qha_formula_units
    else:
        per_formula = 1000.0 / args.qha_formula_units
    if args.energy_basis == "target-cell":
        return per_formula * args.target_z
    return per_formula


def md_cp_scale(args: argparse.Namespace) -> float:
    # LAMMPS thermo-series Cp is J/mol-formula/K.
    return args.target_z if args.energy_basis == "target-cell" else 1.0


def entropy_qha_scale(args: argparse.Namespace) -> float:
    if args.qha_entropy_unit == "eV-cell/K":
        per_formula = EV_TO_KJ_PER_MOL * 1000.0 / args.qha_formula_units
    elif args.qha_entropy_unit == "J/mol-formula/K":
        per_formula = 1.0
    elif args.qha_entropy_unit == "kJ/mol-formula/K":
        per_formula = 1000.0
    elif args.qha_entropy_unit == "J/mol-cell/K":
        per_formula = 1.0 / args.qha_formula_units
    else:
        per_formula = 1000.0 / args.qha_formula_units
    if args.energy_basis == "target-cell":
        return per_formula * args.target_z
    return per_formula


def alpha_qha_scale(args: argparse.Namespace) -> float:
    return 1.0e6 if args.qha_alpha_unit == "1/K" else 1.0


def make_definitions(args: argparse.Namespace) -> list[dict]:
    qha = args.qha_dir
    md_grid = args.md_dir / "thermo_functions_grid.csv"
    md_summary = args.md_dir / "all_T_summary.csv"
    energy_label = "kJ/mol-target-cell" if args.energy_basis == "target-cell" else "kJ/mol-formula"
    cp_label = "J/mol-target-cell/K" if args.energy_basis == "target-cell" else "J/mol-formula/K"
    s_label = cp_label
    return [
        {
            "name": "volume",
            "ylabel": f"Volume (A3 per Z={args.target_z:g} cell)",
            "qha": (qha / "volume-temperature.dat", extensive_qha_scale(args)),
            "md": (md_grid, MD_COLUMN_ALIASES["V"], extensive_md_scale(args)),
        },
        {
            "name": "cp",
            "ylabel": f"Cp ({cp_label})",
            "qha": (qha / "Cp-temperature.dat", cp_qha_scale(args)),
            "md": (md_grid, MD_COLUMN_ALIASES["Cp"], md_cp_scale(args)),
        },
        {
            "name": "entropy",
            "ylabel": f"Entropy ({s_label})",
            "qha": (qha / "entropy-temperature.dat", entropy_qha_scale(args)),
            "md": (md_grid, MD_COLUMN_ALIASES["S"], md_cp_scale(args)),
        },
        {
            "name": "gibbs",
            "ylabel": f"Gibbs energy, relative ({energy_label})",
            "qha": (qha / "gibbs-temperature.dat", energy_scale(args)),
            "md": (md_grid, MD_COLUMN_ALIASES["G"], md_energy_scale(args)),
            "relative_energy": True,
        },
        {
            "name": "enthalpy",
            "ylabel": f"Enthalpy, relative ({energy_label})",
            "qha_derived": "enthalpy",
            "md": (md_grid, MD_COLUMN_ALIASES["H"], md_energy_scale(args)),
            "relative_energy": True,
        },
        {
            "name": "helmholtz",
            "ylabel": f"Helmholtz free energy, relative ({energy_label})",
            "qha": (qha / "helmholtz-temperature.dat", energy_scale(args)),
            "md": (None, "", 1.0),
            "relative_energy": True,
        },
        {
            "name": "cte",
            "ylabel": "Volumetric CTE (10^-6 K^-1)",
            "qha": (qha / "thermal_expansion.dat", alpha_qha_scale(args)),
            "md": (md_grid, MD_COLUMN_ALIASES["alpha_V"], 1.0),
        },
        {
            "name": "bulk_modulus",
            "ylabel": "Bulk modulus (GPa)",
            "qha": (qha / "bulk_modulus-temperature.dat", 1.0),
            "md": (md_summary, MD_COLUMN_ALIASES["K"], 1.0),
        },
    ]


def plot_overlay(path: Path, title: str, ylabel: str, qha_points, md_points, args) -> bool:
    if not qha_points and not md_points:
        return False
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.2, 4.8))
    if qha_points:
        x, y = zip(*qha_points)
        plt.plot(x, y, "-", color="#1f77b4", linewidth=2.0, label="QHA")
    if md_points:
        x, y = zip(*md_points)
        plt.plot(
            x,
            y,
            "o--",
            color="#d62728",
            markersize=3.5,
            linewidth=1.5,
            label="MD",
        )
    if args.t_min is not None or args.t_max is not None:
        plt.xlim(left=args.t_min, right=args.t_max)
    plt.xlabel("Temperature (K)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return True


def write_overlay_csv(path: Path, qha_points, md_points) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "T_K", "value"])
        for temp, value in qha_points:
            writer.writerow(["QHA", temp, value])
        for temp, value in md_points:
            writer.writerow(["MD", temp, value])


def write_metadata(path: Path, args: argparse.Namespace) -> None:
    metadata = {
        "qha_dir": str(args.qha_dir),
        "md_dir": str(args.md_dir),
        "target_z_formula_units": args.target_z,
        "qha_formula_units": args.qha_formula_units,
        "md_formula_units": args.md_formula_units,
        "temperature_min_requested_K": args.t_min,
        "temperature_max_requested_K": args.t_max,
        "energy_basis": args.energy_basis,
        "energy_reference": args.energy_reference,
        "energy_reference_temperature_K": args.energy_reference_temperature,
        "qha_energy_unit": args.qha_energy_unit,
        "qha_cp_unit": args.qha_cp_unit,
        "qha_entropy_unit": args.qha_entropy_unit,
        "qha_alpha_unit": args.qha_alpha_unit,
        "plot_style": {
            "QHA": "solid blue line",
            "MD": "red dashed line with circle markers",
        },
        "unit_notes": [
            "phonopy-qha energies are eV per QHA cell by default",
            "phonopy-qha Cp and entropy are treated as J/mol-cell/K by default",
            "lammps-thermo-series molar columns are treated as per mole of formula units",
            "volume is normalized to target_z formula units",
            "G/H/F are shifted at the minimal overlapping T unless --energy-reference changes it",
        ],
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_availability_report(path: Path, rows: list[dict]) -> None:
    fields = [
        "quantity",
        "comparison_type",
        "qha_source",
        "qha_points",
        "md_source",
        "md_column",
        "md_points",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-qha-md-compare",
        description="Overlay phonopy-QHA and LAMMPS thermo-series temperature functions.",
    )
    parser.add_argument("--qha-dir", type=Path, required=True)
    parser.add_argument("--md-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-z", type=float, default=4.0)
    parser.add_argument("--qha-formula-units", type=float, required=True)
    parser.add_argument("--md-formula-units", type=float, required=True)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument(
        "--energy-basis",
        choices=("per-formula", "target-cell"),
        default="per-formula",
    )
    parser.add_argument(
        "--qha-energy-unit",
        choices=("eV-cell", "kJ/mol-formula", "kJ/mol-cell"),
        default="eV-cell",
    )
    parser.add_argument(
        "--qha-cp-unit",
        choices=(
            "J/mol-cell/K",
            "kJ/mol-cell/K",
            "J/mol-formula/K",
            "kJ/mol-formula/K",
            "eV-cell/K",
        ),
        default="J/mol-cell/K",
    )
    parser.add_argument(
        "--qha-entropy-unit",
        choices=(
            "J/mol-cell/K",
            "kJ/mol-cell/K",
            "J/mol-formula/K",
            "kJ/mol-formula/K",
            "eV-cell/K",
        ),
        default="J/mol-cell/K",
    )
    parser.add_argument("--qha-alpha-unit", choices=("1/K", "micro/K"), default="1/K")
    parser.add_argument(
        "--energy-reference",
        choices=("overlap-min", "temperature", "none"),
        default="overlap-min",
        help="Reference used to shift G/H/F curves before overlay.",
    )
    parser.add_argument("--energy-reference-temperature", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.qha_dir = args.qha_dir.resolve()
    args.md_dir = args.md_dir.resolve()
    args.outdir = args.outdir.resolve()
    if args.qha_formula_units <= 0 or args.md_formula_units <= 0 or args.target_z <= 0:
        parser.error("formula-unit counts and --target-z must be positive")
    if not args.qha_dir.is_dir():
        parser.error(f"QHA directory not found: {args.qha_dir}")
    if not args.md_dir.is_dir():
        parser.error(f"MD directory not found: {args.md_dir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    availability_rows = []
    for item in make_definitions(args):
        qha_path = None
        if item.get("qha_derived") == "enthalpy":
            qha_path = args.qha_dir / "gibbs-temperature.dat"
            qha_points = qha_derived_enthalpy(args)
        else:
            qha_path, qha_scale = item["qha"]
            qha_points = qha_series(qha_path, qha_scale)
        md_path, md_aliases, md_scale = item["md"]
        if md_path is None:
            md_points, md_column = [], ""
        else:
            md_points, md_column = md_series(md_path, md_aliases, md_scale)
        qha_points = filter_range(qha_points, args.t_min, args.t_max)
        md_points = filter_range(md_points, args.t_min, args.t_max)
        if item.get("relative_energy"):
            ref_t = overlapping_reference_temperature(qha_points, md_points, args)
            qha_points = relative_energy(qha_points, args, ref_t)
            md_points = relative_energy(md_points, args, ref_t)
        comparison_type = "overlay" if qha_points and md_points else "single-source"
        note = ""
        if item.get("qha_derived") == "enthalpy" and not qha_points:
            note = "QHA H requires gibbs-temperature.dat and entropy-temperature.dat"
        elif not qha_points and not md_points:
            note = "No QHA data or matching MD column found"
        elif not qha_points:
            note = "QHA source missing or empty"
        elif not md_points:
            note = "MD source missing or no matching MD column"
        availability_rows.append(
            {
                "quantity": item["name"],
                "comparison_type": comparison_type if (qha_points or md_points) else "missing",
                "qha_source": qha_path.name if qha_path is not None else "G+T*S",
                "qha_points": len(qha_points),
                "md_source": md_path.name if md_path is not None else "",
                "md_column": md_column,
                "md_points": len(md_points),
                "note": note,
            }
        )
        if not qha_points and not md_points:
            continue
        png = args.outdir / f"{item['name']}_qha_md_overlay.png"
        csv_path = args.outdir / f"{item['name']}_qha_md_overlay.csv"
        plot_overlay(
            png,
            item["name"].replace("_", " ").title(),
            item["ylabel"],
            qha_points,
            md_points,
            args,
        )
        write_overlay_csv(csv_path, qha_points, md_points)
        index_rows.append(
            {
                "quantity": item["name"],
                "plot_png": png.name,
                "data_csv": csv_path.name,
                "qha_source": qha_path.name if qha_path is not None else "G+T*S",
                "md_source": md_path.name if md_path is not None else "",
                "md_column": md_column,
                "comparison_type": comparison_type,
            }
        )
        print(png)

    with (args.outdir / "overlay_index.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "quantity",
            "plot_png",
            "data_csv",
            "qha_source",
            "md_source",
            "md_column",
            "comparison_type",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)
    write_metadata(args.outdir / "normalization_metadata.json", args)
    write_availability_report(args.outdir / "availability_report.csv", availability_rows)
    if not index_rows:
        print("No matching QHA/MD quantities found to plot.")


if __name__ == "__main__":
    main(sys.argv[1:])
