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

LATTICE_PARAMETER_SPECS = [
    {
        "key": "a",
        "name": "lattice_a",
        "ylabel": "Lattice a (A)",
        "qha_names": ("a-temperature.dat", "lattice_a-temperature.dat", "lattice-temperature.dat"),
        "md_aliases": ("a_fit_A", "a_mean_A", "a_proxy_mean_A"),
    },
    {
        "key": "b",
        "name": "lattice_b",
        "ylabel": "Lattice b (A)",
        "qha_names": ("b-temperature.dat", "lattice_b-temperature.dat"),
        "md_aliases": ("b_fit_A", "b_mean_A", "ly_fit_A", "ly_mean_A"),
    },
    {
        "key": "c",
        "name": "lattice_c",
        "ylabel": "Lattice c (A)",
        "qha_names": ("c-temperature.dat", "lattice_c-temperature.dat"),
        "md_aliases": ("c_fit_A", "c_mean_A", "lz_fit_A", "lz_mean_A"),
    },
    {
        "key": "alpha",
        "name": "lattice_alpha",
        "ylabel": "Lattice alpha (deg)",
        "qha_names": ("alpha-temperature.dat", "lattice_alpha-temperature.dat"),
        "md_aliases": ("alpha_fit_deg", "alpha_mean_deg"),
    },
    {
        "key": "beta",
        "name": "lattice_beta",
        "ylabel": "Lattice beta (deg)",
        "qha_names": ("beta-temperature.dat", "lattice_beta-temperature.dat"),
        "md_aliases": ("beta_fit_deg", "beta_mean_deg"),
    },
    {
        "key": "gamma",
        "name": "lattice_gamma",
        "ylabel": "Lattice gamma (deg)",
        "qha_names": ("gamma-temperature.dat", "lattice_gamma-temperature.dat"),
        "md_aliases": ("gamma_fit_deg", "gamma_mean_deg"),
    },
]


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


def qha_first_available_series(
    qha_dir: Path,
    names: tuple[str, ...],
    scale: float = 1.0,
) -> tuple[list[tuple[float, float]], str]:
    for name in names:
        path = qha_dir / name
        points = qha_series(path, scale)
        if points:
            return points, name
    return [], ""


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


def md_actual_temperature_bounds(md_dir: Path) -> tuple[float, float] | None:
    rows = read_csv_rows(md_dir / "all_T_summary.csv")
    temperatures = []
    for row in rows:
        temp = finite_float(row.get("T_K", row.get("target_T_K")))
        if math.isfinite(temp):
            temperatures.append(temp)
    if not temperatures:
        return None
    return min(temperatures), max(temperatures)


def finite_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return sorted(
        (temp, value)
        for temp, value in points
        if math.isfinite(temp) and math.isfinite(value)
    )


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


def lattice_parameter_definitions(args: argparse.Namespace) -> list[dict]:
    md_grid = args.md_dir / "thermo_functions_grid.csv"
    definitions = []
    for spec in LATTICE_PARAMETER_SPECS:
        qha_points, qha_name = qha_first_available_series(args.qha_dir, spec["qha_names"])
        md_column = resolve_column(md_grid, spec["md_aliases"])
        if not qha_points and not md_column:
            continue
        definitions.append(
            {
                "name": spec["name"],
                "ylabel": spec["ylabel"],
                "qha": ((args.qha_dir / qha_name) if qha_name else None, 1.0),
                "md": (md_grid, spec["md_aliases"], 1.0),
            }
        )
    return definitions


def make_definitions(args: argparse.Namespace) -> list[dict]:
    qha = args.qha_dir
    md_grid = args.md_dir / "thermo_functions_grid.csv"
    md_summary = args.md_dir / "all_T_summary.csv"
    energy_label = "kJ/mol-target-cell" if args.energy_basis == "target-cell" else "kJ/mol-formula"
    cp_label = "J/mol-target-cell/K" if args.energy_basis == "target-cell" else "J/mol-formula/K"
    s_label = cp_label
    definitions = [
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
    definitions.extend(lattice_parameter_definitions(args))
    return definitions


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


def cp_switch_temperature(
    qha_points: list[tuple[float, float]],
    md_points: list[tuple[float, float]],
    requested: float | None = None,
    md_temperature_bounds: tuple[float, float] | None = None,
    min_switch_temperature: float | None = 50.0,
) -> tuple[float | None, str]:
    qha_points = finite_points(qha_points)
    md_points = finite_points(md_points)
    if requested is not None:
        return requested, "manual"
    if not qha_points or not md_points:
        return None, "missing-cp-source"

    qha_min = qha_points[0][0]
    qha_max = qha_points[-1][0]
    md_min = md_points[0][0]
    md_max = md_points[-1][0]
    if md_temperature_bounds is not None:
        md_min = max(md_min, md_temperature_bounds[0])
        md_max = min(md_max, md_temperature_bounds[1])
        if md_min > md_max:
            return None, "no-actual-md-grid-overlap"
    overlap_min = max(qha_min, md_min)
    overlap_max = min(qha_max, md_max)
    if min_switch_temperature is not None:
        overlap_min = max(overlap_min, float(min_switch_temperature))
    if overlap_min <= overlap_max:
        candidates = sorted(
            {
                temp
                for temp, _value in qha_points + md_points
                if overlap_min <= temp <= overlap_max
            }
        )
        if not candidates:
            candidates = [overlap_min, overlap_max]

        def cp_delta(temp: float) -> float:
            qha_value = interpolate(qha_points, temp)
            md_value = interpolate(md_points, temp)
            if qha_value is None or md_value is None:
                return math.inf
            return abs(qha_value - md_value)

        best = min(candidates, key=cp_delta)
        if md_temperature_bounds is not None:
            return best, "actual-md-overlap-closest-cp"
        return best, "overlap-closest-cp"

    if qha_max < md_min:
        switch = 0.5 * (qha_max + md_min)
        if min_switch_temperature is not None:
            switch = max(switch, float(min_switch_temperature))
        return switch, "gap-midpoint-qha-low-md-high"
    if md_max < qha_min:
        switch = 0.5 * (md_max + qha_min)
        if min_switch_temperature is not None and switch < float(min_switch_temperature):
            return None, "gap-switch-below-minimum"
        return switch, "gap-midpoint-md-low-qha-high"
    return None, "no-switch-found"


def smoothstep_weight(temp: float, blend_start: float, blend_end: float) -> float:
    if blend_end <= blend_start:
        return 1.0 if temp >= blend_end else 0.0
    x = min(max((temp - blend_start) / (blend_end - blend_start), 0.0), 1.0)
    return 3.0 * x * x - 2.0 * x * x * x


def default_blend_interval(switch_temp: float, qha_points, md_points) -> tuple[float, float]:
    qha_points = finite_points(qha_points)
    md_points = finite_points(md_points)
    if not qha_points or not md_points:
        return switch_temp, switch_temp
    qha_min, qha_max = qha_points[0][0], qha_points[-1][0]
    md_min, md_max = md_points[0][0], md_points[-1][0]
    overlap_min = max(qha_min, md_min)
    overlap_max = min(qha_max, md_max)
    half_width = 50.0
    if overlap_min <= overlap_max:
        half_width = min(50.0, max((overlap_max - overlap_min) / 4.0, 1.0))
        return (
            max(overlap_min, switch_temp - half_width),
            min(overlap_max, switch_temp + half_width),
        )
    return switch_temp, switch_temp


def build_hybrid_cp_rows(
    qha_points: list[tuple[float, float]],
    md_points: list[tuple[float, float]],
    blend_start: float,
    blend_end: float,
) -> list[dict]:
    temperatures = {
        temp
        for temp, _value in finite_points(qha_points) + finite_points(md_points)
        if (blend_start <= temp <= blend_end)
        or temp < blend_start
        or temp > blend_end
    }
    temperatures.update({blend_start, blend_end})
    rows = []
    for temp in sorted(temperatures):
        qha_value = interpolate(qha_points, temp)
        md_value = interpolate(md_points, temp)
        if temp < blend_start:
            if qha_value is None:
                continue
            cp_value = qha_value
            source = "QHA"
            weight = 0.0
        elif temp > blend_end:
            if md_value is None:
                continue
            cp_value = md_value
            source = "MD"
            weight = 1.0
        else:
            if qha_value is None or md_value is None:
                continue
            weight = smoothstep_weight(temp, blend_start, blend_end)
            cp_value = (1.0 - weight) * qha_value + weight * md_value
            source = "blend"
        rows.append(
            {
                "T_K": temp,
                "Cp": cp_value,
                "Cp_source": source,
                "blend_weight": weight,
            }
        )
    return rows


def cp_over_t(cp_value: float, temp: float) -> float:
    if temp <= 0.0:
        return 0.0
    return cp_value / temp


def add_integrated_thermo(
    rows: list[dict],
    qha_entropy: list[tuple[float, float]],
    qha_enthalpy: list[tuple[float, float]],
    blend_start: float,
) -> tuple[list[dict], str]:
    if not rows:
        return rows, "no-hybrid-cp"
    ref_temp = blend_start
    entropy_reference = interpolate(qha_entropy, ref_temp)
    if entropy_reference is None and qha_entropy:
        ref_temp, entropy_reference = min(qha_entropy, key=lambda item: abs(item[0] - blend_start))
    if entropy_reference is None or not math.isfinite(entropy_reference):
        entropy_reference = 0.0
        note = "S starts at 0 because QHA entropy was unavailable near blend_start"
    else:
        note = "S starts from QHA entropy near blend_start"
    enthalpy_reference = interpolate(qha_enthalpy, ref_temp)
    if enthalpy_reference is None and qha_enthalpy:
        _temp, enthalpy_reference = min(qha_enthalpy, key=lambda item: abs(item[0] - ref_temp))
    enthalpy_note = "QHA H reference at blend_start"
    if enthalpy_reference is None or not math.isfinite(enthalpy_reference):
        enthalpy_reference = 0.0
        enthalpy_note = "H_rel(blend_start)=0 because QHA H was unavailable"
    for row in rows:
        row["S_integrated"] = math.nan
        row["H_integrated_kJ_mol"] = math.nan
    ref_idx = min(range(len(rows)), key=lambda idx: abs(rows[idx]["T_K"] - ref_temp))
    rows[ref_idx]["S_integrated"] = entropy_reference
    rows[ref_idx]["H_integrated_kJ_mol"] = enthalpy_reference
    for previous, current in zip(rows[ref_idx:], rows[ref_idx + 1:]):
        delta_h = 0.5 * (previous["Cp"] + current["Cp"]) * (
            current["T_K"] - previous["T_K"]
        )
        delta_s = 0.5 * (
            cp_over_t(previous["Cp"], previous["T_K"])
            + cp_over_t(current["Cp"], current["T_K"])
        ) * (current["T_K"] - previous["T_K"])
        current["H_integrated_kJ_mol"] = previous["H_integrated_kJ_mol"] + delta_h / 1000.0
        current["S_integrated"] = previous["S_integrated"] + delta_s
    for current, previous in zip(reversed(rows[:ref_idx]), reversed(rows[1 : ref_idx + 1])):
        delta_h = 0.5 * (previous["Cp"] + current["Cp"]) * (
            previous["T_K"] - current["T_K"]
        )
        delta_s = 0.5 * (
            cp_over_t(previous["Cp"], previous["T_K"])
            + cp_over_t(current["Cp"], current["T_K"])
        ) * (previous["T_K"] - current["T_K"])
        current["H_integrated_kJ_mol"] = previous["H_integrated_kJ_mol"] - delta_h / 1000.0
        current["S_integrated"] = previous["S_integrated"] - delta_s
    for row in rows:
        row["G_integrated_kJ_mol"] = (
            row["H_integrated_kJ_mol"] - row["T_K"] * row["S_integrated"] / 1000.0
        )
    g0 = rows[ref_idx]["G_integrated_kJ_mol"]
    for row in rows:
        row["G_relative_kJ_mol"] = row["G_integrated_kJ_mol"] - g0
    return rows, f"{note}; {enthalpy_note}; reference_T={rows[ref_idx]['T_K']} K"


def relative_to_temperature(points, ref_t: float) -> list[tuple[float, float]]:
    ref = interpolate(points, ref_t)
    if ref is None or not math.isfinite(ref):
        ref = min(points, key=lambda item: abs(item[0] - ref_t))[1] if points else 0.0
    return [(temp, value - ref) for temp, value in points]


def apply_hybrid_y_limits(axis, hybrid_values: list[float]) -> None:
    values = [value for value in hybrid_values if math.isfinite(value)]
    if not values:
        return
    ymin = min(values)
    ymax = max(values)
    if abs(ymax - ymin) <= 1.0e-12:
        pad = max(abs(ymax) * 0.05, 1.0)
    else:
        pad = 0.08 * (ymax - ymin)
    axis.set_ylim(ymin - pad, ymax + pad)


def plot_hybrid_quantity(
    path: Path,
    title: str,
    ylabel: str,
    qha_points,
    md_points,
    hybrid_rows,
    hybrid_key: str,
    hybrid_label: str,
    blend_start,
    blend_end,
    args,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if qha_points:
        x, y = zip(*qha_points)
        ax.plot(x, y, "-.", color="#8c8c8c", linewidth=1.4, alpha=0.45, label="QHA reference")
    if md_points:
        x, y = zip(*md_points)
        ax.plot(x, y, "--", color="#5f5f5f", linewidth=1.3, alpha=0.45, label="MD reference")
    if hybrid_rows:
        ax.plot(
            [row["T_K"] for row in hybrid_rows],
            [row[hybrid_key] for row in hybrid_rows],
            "-",
            color="#111111",
            linewidth=2.2,
            label=hybrid_label,
        )
        apply_hybrid_y_limits(ax, [row[hybrid_key] for row in hybrid_rows])
    if blend_start == blend_end:
        ax.axvline(blend_start, color="#555555", linestyle=":", linewidth=1.2)
    else:
        ax.axvspan(blend_start, blend_end, color="#111111", alpha=0.08, label="blend interval")
    if args.t_min is not None or args.t_max is not None:
        ax.set_xlim(left=args.t_min, right=args.t_max)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def cp_overlap_diagnostics(qha_cp, md_cp, blend_start: float, blend_end: float) -> dict:
    temps = sorted(
        {
            temp
            for temp, _value in finite_points(qha_cp) + finite_points(md_cp)
            if blend_start <= temp <= blend_end
        }
        | {blend_start, blend_end}
    )
    rows = []
    deltas = []
    rels = []
    signs = []
    for temp in temps:
        qha_value = interpolate(qha_cp, temp)
        md_value = interpolate(md_cp, temp)
        if qha_value is None or md_value is None:
            continue
        delta = md_value - qha_value
        denom = max(abs(qha_value), 1.0e-12)
        rel = abs(delta) / denom
        rows.append((temp, qha_value, md_value, delta, rel))
        deltas.append(delta)
        rels.append(rel)
        signs.append(0 if abs(delta) <= 1.0e-12 else (1 if delta > 0 else -1))
    crossing = any(a * b < 0 for a, b in zip(signs, signs[1:]))
    if any(sign == 0 for sign in signs):
        crossing = True
    start_qha = interpolate(qha_cp, blend_start)
    start_md = interpolate(md_cp, blend_start)
    end_qha = interpolate(qha_cp, blend_end)
    end_md = interpolate(md_cp, blend_end)

    def relative_mismatch(qha_value, md_value):
        if qha_value is None or md_value is None:
            return None
        return abs(md_value - qha_value) / max(abs(qha_value), 1.0e-12)

    mean_abs = sum(abs(delta) for delta in deltas) / len(deltas) if deltas else math.nan
    rms = math.sqrt(sum(delta * delta for delta in deltas) / len(deltas)) if deltas else math.nan
    mean_rel = sum(rels) / len(rels) if rels else math.nan
    warning = any(value > 0.10 for value in rels)
    return {
        "rows": rows,
        "mean_absolute_cp_mismatch": mean_abs,
        "rms_cp_mismatch": rms,
        "mean_relative_cp_mismatch": mean_rel,
        "relative_mismatch_at_blend_start": relative_mismatch(start_qha, start_md),
        "relative_mismatch_at_blend_end": relative_mismatch(end_qha, end_md),
        "cp_curves_cross_in_blend_interval": crossing,
        "warning_cp_mismatch_exceeds_10_percent": warning,
    }


def plot_overlap_mismatch(path: Path, diagnostics: dict, args) -> None:
    import matplotlib.pyplot as plt

    rows = diagnostics.get("rows", [])
    if not rows:
        return
    temps = [row[0] for row in rows]
    rels = [row[4] * 100.0 for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(temps, rels, "o-", color="#111111", linewidth=1.8, markersize=3.5)
    ax.axhline(10.0, color="#b00020", linestyle="--", linewidth=1.2, label="10% warning")
    if args.t_min is not None or args.t_max is not None:
        ax.set_xlim(left=args.t_min, right=args.t_max)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("|Cp_MD - Cp_QHA| / |Cp_QHA| (%)")
    ax.set_title("QHA/MD Cp Mismatch In Blend Region")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def blend_series(qha_points, md_points, blend_start, blend_end) -> tuple[list[dict], str]:
    if not md_points:
        return [], "missing-md"
    if not qha_points:
        return [
            {"T_K": temp, "value": value, "source": "MD", "blend_weight": 1.0}
            for temp, value in finite_points(md_points)
        ], "md-only"
    temps = sorted(
        {temp for temp, _value in finite_points(qha_points) + finite_points(md_points)}
        | {blend_start, blend_end}
    )
    rows = []
    for temp in temps:
        qha_value = interpolate(qha_points, temp)
        md_value = interpolate(md_points, temp)
        if temp < blend_start:
            if qha_value is None:
                continue
            rows.append({"T_K": temp, "value": qha_value, "source": "QHA", "blend_weight": 0.0})
        elif temp > blend_end:
            if md_value is None:
                continue
            rows.append({"T_K": temp, "value": md_value, "source": "MD", "blend_weight": 1.0})
        elif qha_value is not None and md_value is not None:
            weight = smoothstep_weight(temp, blend_start, blend_end)
            rows.append(
                {
                    "T_K": temp,
                    "value": (1.0 - weight) * qha_value + weight * md_value,
                    "source": "blend",
                    "blend_weight": weight,
                }
            )
    return rows, "hybrid"


def selected_md_records(md_dir: Path) -> list[dict]:
    for name in ("used_stage_records.json", "discovered_stage_records.json"):
        path = md_dir / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []
    return []


def write_hybrid_outputs(args: argparse.Namespace) -> tuple[list[dict], dict]:
    qha_cp = filter_range(
        qha_series(args.qha_dir / "Cp-temperature.dat", cp_qha_scale(args)),
        args.t_min,
        args.t_max,
    )
    md_path = args.md_dir / "thermo_functions_grid.csv"
    md_cp, md_cp_column = md_series(md_path, MD_COLUMN_ALIASES["Cp"], md_cp_scale(args))
    md_cp = filter_range(md_cp, args.t_min, args.t_max)
    if not qha_cp or not md_cp:
        return [], {
            "note": "Hybrid Cp/S skipped because QHA Cp or MD Cp is unavailable",
            "qha_cp_points": len(qha_cp),
            "md_cp_points": len(md_cp),
            "md_cp_column": md_cp_column,
        }
    actual_md_bounds = md_actual_temperature_bounds(args.md_dir)
    switch_temp, switch_method = cp_switch_temperature(
        qha_cp,
        md_cp,
        args.hybrid_switch_temperature,
        actual_md_bounds,
        args.hybrid_min_switch_temperature,
    )
    if switch_temp is None:
        return [], {
            "note": "Hybrid Cp/S skipped because QHA and MD Cp were unavailable",
            "md_cp_column": md_cp_column,
        }
    if args.hybrid_blend_start is None and args.hybrid_blend_end is None:
        blend_start, blend_end = default_blend_interval(switch_temp, qha_cp, md_cp)
    elif args.hybrid_blend_start is not None and args.hybrid_blend_end is not None:
        blend_start = args.hybrid_blend_start
        blend_end = args.hybrid_blend_end
    else:
        raise ValueError("--hybrid-blend-start and --hybrid-blend-end must be used together")
    if blend_end < blend_start:
        raise ValueError("--hybrid-blend-end must be >= --hybrid-blend-start")

    qha_entropy = filter_range(
        qha_series(args.qha_dir / "entropy-temperature.dat", entropy_qha_scale(args)),
        args.t_min,
        args.t_max,
    )
    md_entropy, md_entropy_column = md_series(
        md_path,
        MD_COLUMN_ALIASES["S"],
        md_cp_scale(args),
    )
    md_entropy = filter_range(md_entropy, args.t_min, args.t_max)
    qha_enthalpy = filter_range(qha_derived_enthalpy(args), args.t_min, args.t_max)
    hybrid_rows = build_hybrid_cp_rows(qha_cp, md_cp, blend_start, blend_end)
    hybrid_rows, entropy_note = add_integrated_thermo(
        hybrid_rows,
        qha_entropy,
        qha_enthalpy,
        blend_start,
    )
    if not hybrid_rows:
        return [], {
            "note": "Hybrid Cp/S skipped because no points survived the switch",
            "md_cp_column": md_cp_column,
            "md_entropy_column": md_entropy_column,
        }

    csv_path = args.outdir / "hybrid_cp_entropy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "T_K",
            "Cp_source",
            "blend_weight",
            "Cp",
            "S_integrated",
            "H_integrated_kJ_mol",
            "G_integrated_kJ_mol",
            "G_relative_kJ_mol",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in hybrid_rows:
            writer.writerow(
                {
                    "T_K": row["T_K"],
                    "Cp_source": row["Cp_source"],
                    "blend_weight": row["blend_weight"],
                    "Cp": row["Cp"],
                    "S_integrated": row["S_integrated"],
                    "H_integrated_kJ_mol": row["H_integrated_kJ_mol"],
                    "G_integrated_kJ_mol": row["G_integrated_kJ_mol"],
                    "G_relative_kJ_mol": row["G_relative_kJ_mol"],
                }
            )

    diagnostics = cp_overlap_diagnostics(qha_cp, md_cp, blend_start, blend_end)
    mismatch_png = args.outdir / "overlap_mismatch_Cp.png"
    plot_overlap_mismatch(mismatch_png, diagnostics, args)

    first_hybrid_t = blend_start
    qha_enthalpy = relative_to_temperature(
        qha_enthalpy,
        first_hybrid_t,
    )
    md_enthalpy, md_enthalpy_column = md_series(
        md_path,
        MD_COLUMN_ALIASES["H"],
        md_energy_scale(args),
    )
    md_enthalpy = relative_to_temperature(
        filter_range(md_enthalpy, args.t_min, args.t_max),
        first_hybrid_t,
    )
    qha_gibbs = relative_to_temperature(
        filter_range(
            qha_series(args.qha_dir / "gibbs-temperature.dat", energy_scale(args)),
            args.t_min,
            args.t_max,
        ),
        first_hybrid_t,
    )
    md_gibbs, md_gibbs_column = md_series(
        md_path,
        MD_COLUMN_ALIASES["G"],
        md_energy_scale(args),
    )
    md_gibbs = relative_to_temperature(
        filter_range(md_gibbs, args.t_min, args.t_max),
        first_hybrid_t,
    )

    qha_volume = filter_range(
        qha_series(args.qha_dir / "volume-temperature.dat", extensive_qha_scale(args)),
        args.t_min,
        args.t_max,
    )
    md_volume, md_volume_column = md_series(
        md_path,
        MD_COLUMN_ALIASES["V"],
        extensive_md_scale(args),
    )
    md_volume = filter_range(md_volume, args.t_min, args.t_max)
    volume_rows, volume_mode = blend_series(qha_volume, md_volume, blend_start, blend_end)
    lattice_hybrids = []
    for spec in LATTICE_PARAMETER_SPECS:
        qha_lattice, qha_lattice_file = qha_first_available_series(
            args.qha_dir,
            spec["qha_names"],
        )
        qha_lattice = filter_range(qha_lattice, args.t_min, args.t_max)
        md_lattice, md_lattice_column = md_series(md_path, spec["md_aliases"], 1.0)
        md_lattice = filter_range(md_lattice, args.t_min, args.t_max)
        lattice_rows, lattice_mode = blend_series(
            qha_lattice,
            md_lattice,
            blend_start,
            blend_end,
        )
        if lattice_rows:
            lattice_hybrids.append(
                {
                    "key": spec["key"],
                    "ylabel": spec["ylabel"],
                    "qha_points": qha_lattice,
                    "qha_file": qha_lattice_file,
                    "md_points": md_lattice,
                    "md_column": md_lattice_column,
                    "rows": lattice_rows,
                    "mode": lattice_mode,
                }
            )
    if volume_rows or lattice_hybrids:
        va_csv = args.outdir / "hybrid_volume_lattice.csv"
        with va_csv.open("w", newline="", encoding="utf-8") as handle:
            fields = ["quantity", "T_K", "value", "source", "blend_weight"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in volume_rows:
                writer.writerow(
                    {
                        "quantity": "V_A3",
                        "T_K": row["T_K"],
                        "value": row["value"],
                        "source": row["source"],
                        "blend_weight": row["blend_weight"],
                    }
                )
            for item in lattice_hybrids:
                for row in item["rows"]:
                    writer.writerow(
                        {
                            "quantity": f"{item['key']}_lattice",
                            "T_K": row["T_K"],
                            "value": row["value"],
                            "source": row["source"],
                            "blend_weight": row["blend_weight"],
                        }
                    )
    else:
        va_csv = None

    cp_label = "J/mol-target-cell/K" if args.energy_basis == "target-cell" else "J/mol-formula/K"
    energy_label = "kJ/mol-target-cell" if args.energy_basis == "target-cell" else "kJ/mol-formula"
    cp_png = args.outdir / "hybrid_Cp_QHA_MD.png"
    entropy_png = args.outdir / "hybrid_S_QHA_MD.png"
    enthalpy_png = args.outdir / "hybrid_H_QHA_MD.png"
    gibbs_png = args.outdir / "hybrid_G_QHA_MD.png"
    volume_png = args.outdir / "hybrid_V_QHA_MD.png"
    plot_hybrid_quantity(
        cp_png,
        "Hybrid QHA+MD Cp",
        f"Cp ({cp_label})",
        qha_cp,
        md_cp,
        hybrid_rows,
        "Cp",
        "Hybrid Cp",
        blend_start,
        blend_end,
        args,
    )
    plot_hybrid_quantity(
        entropy_png,
        "Integrated Hybrid QHA+MD Entropy",
        f"S ({cp_label})",
        qha_entropy,
        md_entropy,
        hybrid_rows,
        "S_integrated",
        "Integrated hybrid S",
        blend_start,
        blend_end,
        args,
    )
    plot_hybrid_quantity(
        enthalpy_png,
        "Integrated Hybrid QHA+MD Enthalpy",
        f"H, relative ({energy_label})",
        qha_enthalpy,
        md_enthalpy,
        hybrid_rows,
        "H_integrated_kJ_mol",
        "Integrated hybrid H",
        blend_start,
        blend_end,
        args,
    )
    plot_hybrid_quantity(
        gibbs_png,
        "Integrated Hybrid QHA+MD Gibbs Energy",
        f"G, relative ({energy_label})",
        qha_gibbs,
        md_gibbs,
        hybrid_rows,
        "G_relative_kJ_mol",
        "Hybrid G = H - TS",
        blend_start,
        blend_end,
        args,
    )
    if volume_rows:
        plot_hybrid_quantity(
            volume_png,
            "Hybrid QHA+MD Volume",
            f"Volume (A3 per Z={args.target_z:g} cell)",
            qha_volume,
            md_volume,
            volume_rows,
            "value",
            "Hybrid V" if volume_mode == "hybrid" else "MD V",
            blend_start,
            blend_end,
            args,
        )
    for item in lattice_hybrids:
        lattice_png = args.outdir / f"hybrid_{item['key']}_QHA_MD.png"
        plot_hybrid_quantity(
            lattice_png,
            f"Hybrid QHA+MD Lattice {item['key']}",
            item["ylabel"],
            item["qha_points"],
            item["md_points"],
            item["rows"],
            "value",
            f"Hybrid {item['key']}" if item["mode"] == "hybrid" else f"MD {item['key']}",
            blend_start,
            blend_end,
            args,
        )
    metadata = {
        "switch_temperature_K": switch_temp,
        "switch_method": switch_method,
        "minimum_switch_temperature_K": args.hybrid_min_switch_temperature,
        "blend_start_K": blend_start,
        "blend_end_K": blend_end,
        "blend_function": "smoothstep w=3x^2-2x^3",
        "entropy_integration": "S(T) = S(T0) + integral_T0^T Cp(T')/T' dT'",
        "enthalpy_integration": "H(T) = H(T0) + integral_T0^T Cp(T') dT",
        "gibbs_integration": "G(T) = H(T) - T*S(T)",
        "entropy_reference_note": entropy_note,
        "cp_overlap_diagnostics": {
            key: value
            for key, value in diagnostics.items()
            if key != "rows"
        },
        "cp_overlap_diagnostics_rows": [
            {
                "T_K": row[0],
                "Cp_QHA": row[1],
                "Cp_MD": row[2],
                "Cp_MD_minus_QHA": row[3],
                "relative_mismatch": row[4],
            }
            for row in diagnostics.get("rows", [])
        ],
        "qha_cp_points": len(qha_cp),
        "md_cp_points": len(md_cp),
        "qha_file_paths": {
            "Cp": str((args.qha_dir / "Cp-temperature.dat").resolve()),
            "S": str((args.qha_dir / "entropy-temperature.dat").resolve()),
            "G": str((args.qha_dir / "gibbs-temperature.dat").resolve()),
            "V": str((args.qha_dir / "volume-temperature.dat").resolve()),
            "lattice_parameters": {
                item["key"]: str((args.qha_dir / item["qha_file"]).resolve())
                if item["qha_file"]
                else None
                for item in lattice_hybrids
            },
        },
        "md_dir": str(args.md_dir),
        "selected_md_configs_logs": selected_md_records(args.md_dir),
        "actual_md_temperature_min_K": actual_md_bounds[0] if actual_md_bounds else None,
        "actual_md_temperature_max_K": actual_md_bounds[1] if actual_md_bounds else None,
        "qha_entropy_points": len(qha_entropy),
        "md_entropy_points": len(md_entropy),
        "md_cp_column": md_cp_column,
        "md_entropy_column": md_entropy_column,
        "md_enthalpy_column": md_enthalpy_column,
        "md_gibbs_column": md_gibbs_column,
        "md_volume_column": md_volume_column,
        "md_lattice_columns": {
            item["key"]: item["md_column"]
            for item in lattice_hybrids
        },
        "volume_source_mode": volume_mode,
        "lattice_source_modes": {
            item["key"]: item["mode"]
            for item in lattice_hybrids
        },
        "volume_note": (
            "V(T) used the same QHA+MD smooth blend."
            if volume_mode == "hybrid"
            else "V(T) is MD-only because QHA volume was unavailable."
        ),
        "lattice_note": "Each detected lattice parameter gets its own QHA/MD hybrid plot.",
        "enthalpy_reference": entropy_note,
        "gibbs_reference": (
            "Hybrid G is recomputed from integrated H and S; QHA/MD references are "
            "shifted at blend_start for plotting."
        ),
        "basis": args.energy_basis,
        "cp_entropy_units": (
            "J/mol-target-cell/K"
            if args.energy_basis == "target-cell"
            else "J/mol-formula/K"
        ),
    }
    (args.outdir / "hybrid_cp_entropy_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(cp_png)
    print(entropy_png)
    print(enthalpy_png)
    print(gibbs_png)
    if volume_rows:
        print(volume_png)
    for item in lattice_hybrids:
        print(args.outdir / f"hybrid_{item['key']}_QHA_MD.png")
    index = [
        {
            "quantity": "hybrid_cp",
            "plot_png": cp_png.name,
            "data_csv": csv_path.name,
            "qha_source": "Cp-temperature.dat",
            "md_source": md_path.name,
            "md_column": md_cp_column,
            "comparison_type": "hybrid",
        },
        {
            "quantity": "hybrid_entropy",
            "plot_png": entropy_png.name,
            "data_csv": csv_path.name,
            "qha_source": "entropy-temperature.dat",
            "md_source": md_path.name,
            "md_column": md_entropy_column,
            "comparison_type": "integrated-hybrid",
        },
        {
            "quantity": "hybrid_enthalpy",
            "plot_png": enthalpy_png.name,
            "data_csv": csv_path.name,
            "qha_source": "G+T*S",
            "md_source": md_path.name,
            "md_column": md_enthalpy_column,
            "comparison_type": "integrated-hybrid",
        },
        {
            "quantity": "hybrid_gibbs",
            "plot_png": gibbs_png.name,
            "data_csv": csv_path.name,
            "qha_source": "gibbs-temperature.dat",
            "md_source": md_path.name,
            "md_column": md_gibbs_column,
            "comparison_type": "integrated-hybrid",
        },
    ]
    if volume_rows:
        index.append(
            {
                "quantity": "hybrid_volume",
                "plot_png": volume_png.name,
                "data_csv": va_csv.name if va_csv else "",
                "qha_source": "volume-temperature.dat",
                "md_source": md_path.name,
                "md_column": md_volume_column,
                "comparison_type": volume_mode,
            }
        )
    for item in lattice_hybrids:
        lattice_png = args.outdir / f"hybrid_{item['key']}_QHA_MD.png"
        index.append(
            {
                "quantity": f"hybrid_lattice_{item['key']}",
                "plot_png": lattice_png.name,
                "data_csv": va_csv.name if va_csv else "",
                "qha_source": item["qha_file"],
                "md_source": md_path.name,
                "md_column": item["md_column"],
                "comparison_type": item["mode"],
            }
        )
    return index, metadata


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
    parser.add_argument(
        "--no-hybrid-cp-s",
        action="store_true",
        help="Skip integrated hybrid QHA+MD Cp and entropy outputs.",
    )
    parser.add_argument(
        "--hybrid-switch-temperature",
        type=float,
        default=None,
        help="Override the automatic QHA-to-MD Cp switch temperature in K.",
    )
    parser.add_argument(
        "--hybrid-min-switch-temperature",
        type=float,
        default=50.0,
        help="Reject automatic QHA-to-MD Cp switches below this temperature in K.",
    )
    parser.add_argument(
        "--hybrid-blend-start",
        type=float,
        default=None,
        help="Start temperature for smooth QHA-to-MD Cp blending in K.",
    )
    parser.add_argument(
        "--hybrid-blend-end",
        type=float,
        default=None,
        help="End temperature for smooth QHA-to-MD Cp blending in K.",
    )
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
            qha_points = qha_series(qha_path, qha_scale) if qha_path is not None else []
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

    if not args.no_hybrid_cp_s:
        try:
            hybrid_index_rows, hybrid_metadata = write_hybrid_outputs(args)
        except ValueError as exc:
            parser.error(str(exc))
        index_rows.extend(hybrid_index_rows)
        hybrid_note = hybrid_metadata.get("note", "")
        if hybrid_metadata and "switch_temperature_K" in hybrid_metadata:
            hybrid_note = (
                f"switch={hybrid_metadata['switch_temperature_K']} K "
                f"({hybrid_metadata['switch_method']}); "
                f"blend={hybrid_metadata.get('blend_start_K')}--"
                f"{hybrid_metadata.get('blend_end_K')} K; "
                f"{hybrid_metadata['entropy_reference_note']}"
            )
        availability_rows.append(
            {
                "quantity": "hybrid_cp_entropy",
                "comparison_type": "integrated-hybrid" if hybrid_index_rows else "missing",
                "qha_source": "Cp-temperature.dat + entropy-temperature.dat",
                "qha_points": hybrid_metadata.get("qha_cp_points", 0),
                "md_source": "thermo_functions_grid.csv",
                "md_column": hybrid_metadata.get("md_cp_column", ""),
                "md_points": hybrid_metadata.get("md_cp_points", 0),
                "note": hybrid_note,
            }
        )

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
