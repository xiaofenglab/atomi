"""Thermal-conductivity table helpers for DFT/MD-to-MOOSE workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from atomi.core.cell import cell_metadata, infer_formula_units


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


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


def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def attach_cell_columns(row: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in (
        "formula",
        "natoms",
        "atoms_per_formula_unit",
        "n_formula_units",
        "target_z_formula_units",
        "cell_role",
        "normalization_basis",
    ):
        out[key] = meta.get(key)
    return out


def elastic_summary_rows(path: Path, select: str, meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        temp = finite_float(row.get("temperature_K") or row.get("T_K"))
        cahill = finite_float(row.get("k_min_cahill_W_mK"))
        clarke = finite_float(row.get("k_min_clarke_W_mK"))
        choices = [value for value in (cahill, clarke) if value is not None]
        if select == "cahill":
            selected = cahill
        elif select == "clarke":
            selected = clarke
        else:
            selected = sum(choices) / len(choices) if choices else None
        if temp is None or selected is None:
            continue
        rows.append(
            attach_cell_columns(
                {
                "T_K": temp,
                "k_W_mK": selected,
                "k_min_cahill_W_mK": cahill,
                "k_min_clarke_W_mK": clarke,
                "source": f"elastic_min_{select}",
                "source_file": str(path),
                },
                meta,
            )
        )
    return rows


def conductivity_table_rows(path: Path, label: str | None = None, meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta = meta or {}
    for row in read_csv(path):
        temp = finite_float(row.get("T_K") or row.get("temperature_K"))
        conductivity = finite_float(row.get("k_W_mK") or row.get("thermal_conductivity_W_mK"))
        if temp is None or conductivity is None:
            continue
        rows.append(
            attach_cell_columns(
                {
                "T_K": temp,
                "k_W_mK": conductivity,
                "k_std_W_mK": finite_float(row.get("k_std_W_mK")),
                "source": label or row.get("source") or "table",
                "source_file": str(path),
                },
                meta,
            )
        )
    return rows


def green_kubo_rows(
    path: Path,
    *,
    temperature_K: float | None,
    scale_to_W_mK: float,
    plateau_start_ps: float | None,
    plateau_fraction: float,
    label: str,
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(path)
    if not rows:
        return [], {"source_file": str(path), "error": "empty CSV"}
    if any("k_W_mK" in row and row["k_W_mK"] for row in rows):
        return conductivity_table_rows(path, label=label, meta=meta), {"source_file": str(path), "mode": "pre_integrated"}
    times = np.asarray(
        [
            value if (value := finite_float(row.get("time_ps"))) is not None else math.nan
            for row in rows
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(times)):
        return [], {"source_file": str(path), "error": "green-kubo CSV needs time_ps"}
    components: dict[str, np.ndarray] = {}
    for axis in ("x", "y", "z"):
        for name in (f"HCACF_{axis}", f"hfacf_{axis}", f"C{axis}{axis}"):
            if name in rows[0]:
                values = np.asarray(
                    [
                        value if (value := finite_float(row.get(name))) is not None else math.nan
                        for row in rows
                    ],
                    dtype=float,
                )
                if np.all(np.isfinite(values)):
                    components[axis] = values
                break
    if not components:
        return [], {"source_file": str(path), "error": "green-kubo CSV needs HCACF_x/y/z columns or k_W_mK"}
    start = plateau_start_ps
    if start is None:
        start = float(times[0] + (times[-1] - times[0]) * max(0.0, min(1.0, 1.0 - plateau_fraction)))
    plateau_mask = times >= start
    if not np.any(plateau_mask):
        plateau_mask[-1] = True
    k_components: dict[str, float] = {}
    for axis, values in components.items():
        cumulative = np.asarray(
            [trapz_compat(values[: idx + 1], times[: idx + 1]) for idx in range(len(times))],
            dtype=float,
        )
        k_components[f"k_{axis}_W_mK"] = float(np.mean(cumulative[plateau_mask]) * scale_to_W_mK)
    values = list(k_components.values())
    temp = temperature_K
    if temp is None:
        for row in rows:
            temp = finite_float(row.get("T_K") or row.get("temperature_K"))
            if temp is not None:
                break
    if temp is None:
        return [], {"source_file": str(path), "error": "temperature is required for green-kubo CSV"}
    result = attach_cell_columns(
        {
            "T_K": temp,
            "k_W_mK": float(np.mean(values)),
            "k_std_W_mK": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "source": label,
            "source_file": str(path),
            **k_components,
        },
        meta,
    )
    metadata = {
        "source_file": str(path),
        "mode": "raw_HCACF_integral",
        "scale_to_W_mK": scale_to_W_mK,
        "plateau_start_ps": start,
        "plateau_fraction": plateau_fraction,
        "component_count": len(components),
        "note": "HCACF units are user-defined; --green-kubo-scale converts the raw time integral to W/m/K.",
    }
    return [result], metadata


def first_finite(row: dict[str, str], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = finite_float(row.get(name))
        if value is not None:
            return value
    return None


def mode_family(row: dict[str, str], frequency_thz: float | None, optical_cutoff_thz: float | None) -> str:
    text = str(row.get("branch") or row.get("mode_family") or row.get("polarization") or "").strip().lower()
    if text in {"la", "ta", "ta1", "ta2", "acoustic"} or "acoustic" in text:
        return "acoustic"
    if text in {"optical", "op"} or "opt" in text:
        return "optical"
    if optical_cutoff_thz is not None and frequency_thz is not None:
        return "optical" if frequency_thz >= optical_cutoff_thz else "acoustic"
    return "unknown"


def nma_rows(
    path: Path,
    *,
    label: str | None,
    meta: dict[str, Any],
    optical_cutoff_thz: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(path)
    source = label or "nma"
    if not rows:
        return [], {"source_file": str(path), "error": "empty CSV"}
    if any("k_W_mK" in row and row["k_W_mK"] for row in rows):
        table_rows = conductivity_table_rows(path, label=source, meta=meta)
        return table_rows, {"source_file": str(path), "mode": "pre_integrated_nma", "rows": len(table_rows)}

    grouped: dict[float, list[dict[str, Any]]] = {}
    skipped = 0
    for row in rows:
        temp = first_finite(row, ("T_K", "temperature_K", "temperature"))
        if temp is None:
            skipped += 1
            continue
        tau_ps = first_finite(row, ("lifetime_ps", "tau_ps", "relaxation_time_ps"))
        heat_capacity = first_finite(
            row,
            (
                "mode_heat_capacity_J_m3K",
                "heat_capacity_J_m3K",
                "c_J_m3K",
                "Cv_J_m3K",
            ),
        )
        kx = first_finite(row, ("k_x_W_mK", "kappa_x_W_mK", "kx_W_mK"))
        ky = first_finite(row, ("k_y_W_mK", "kappa_y_W_mK", "ky_W_mK"))
        kz = first_finite(row, ("k_z_W_mK", "kappa_z_W_mK", "kz_W_mK"))
        if kx is None or ky is None or kz is None:
            vx = first_finite(row, ("vg_x_m_s", "group_velocity_x_m_s", "vx_m_s"))
            vy = first_finite(row, ("vg_y_m_s", "group_velocity_y_m_s", "vy_m_s"))
            vz = first_finite(row, ("vg_z_m_s", "group_velocity_z_m_s", "vz_m_s"))
            if tau_ps is None or heat_capacity is None or vx is None or vy is None or vz is None:
                skipped += 1
                continue
            tau_s = tau_ps * 1.0e-12
            kx = heat_capacity * vx * vx * tau_s
            ky = heat_capacity * vy * vy * tau_s
            kz = heat_capacity * vz * vz * tau_s
        frequency = first_finite(row, ("frequency_THz", "freq_THz", "omega_THz"))
        item = {
            "k_x_W_mK": kx,
            "k_y_W_mK": ky,
            "k_z_W_mK": kz,
            "frequency_THz": frequency,
            "family": mode_family(row, frequency, optical_cutoff_thz),
            "lifetime_ps": tau_ps,
        }
        grouped.setdefault(float(temp), []).append(item)

    out: list[dict[str, Any]] = []
    for temp, items in sorted(grouped.items()):
        kx = float(sum(float(item["k_x_W_mK"]) for item in items))
        ky = float(sum(float(item["k_y_W_mK"]) for item in items))
        kz = float(sum(float(item["k_z_W_mK"]) for item in items))
        values = [kx, ky, kz]
        family_k: dict[str, float] = {"acoustic": 0.0, "optical": 0.0, "unknown": 0.0}
        for item in items:
            contribution = float(item["k_x_W_mK"] + item["k_y_W_mK"] + item["k_z_W_mK"]) / 3.0
            family_k[item["family"]] = family_k.get(item["family"], 0.0) + contribution
        lifetimes = [float(item["lifetime_ps"]) for item in items if item.get("lifetime_ps") is not None]
        out.append(
            attach_cell_columns(
                {
                    "T_K": temp,
                    "k_W_mK": float(sum(values) / 3.0),
                    "k_std_W_mK": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "k_x_W_mK": kx,
                    "k_y_W_mK": ky,
                    "k_z_W_mK": kz,
                    "nma_acoustic_k_W_mK": family_k.get("acoustic", 0.0),
                    "nma_optical_k_W_mK": family_k.get("optical", 0.0),
                    "nma_unknown_branch_k_W_mK": family_k.get("unknown", 0.0),
                    "nma_mode_count": len(items),
                    "nma_lifetime_mean_ps": float(np.mean(lifetimes)) if lifetimes else None,
                    "nma_lifetime_min_ps": float(np.min(lifetimes)) if lifetimes else None,
                    "nma_lifetime_max_ps": float(np.max(lifetimes)) if lifetimes else None,
                    "source": source,
                    "source_file": str(path),
                },
                meta,
            )
        )
    return out, {
        "source_file": str(path),
        "mode": "mode_resolved_nma_summary",
        "rows": len(out),
        "skipped_mode_rows": skipped,
        "optical_cutoff_THz": optical_cutoff_thz,
        "note": (
            "Atomi aggregated NMA mode tables; full trajectory projection onto harmonic "
            "eigenvectors must be generated upstream."
        ),
    }


def is_gk_source(source: Any) -> bool:
    text = str(source or "").lower()
    return "green" in text or "gk" in text


def is_nma_source(source: Any) -> bool:
    return "nma" in str(source or "").lower()


def compare_gk_nma(
    rows: list[dict[str, Any]],
    *,
    temp_tolerance: float,
    warning_fraction: float,
    large_gap_fraction: float,
) -> list[dict[str, Any]]:
    gk_rows = [row for row in rows if is_gk_source(row.get("source"))]
    nma_rows_in = [row for row in rows if is_nma_source(row.get("source"))]
    comparisons: list[dict[str, Any]] = []
    for gk in gk_rows:
        gk_temp = finite_float(gk.get("T_K"))
        gk_k = finite_float(gk.get("k_W_mK"))
        if gk_temp is None or gk_k is None:
            continue
        matches = []
        for nma in nma_rows_in:
            nma_temp = finite_float(nma.get("T_K"))
            nma_k = finite_float(nma.get("k_W_mK"))
            if nma_temp is None or nma_k is None:
                continue
            if abs(nma_temp - gk_temp) <= temp_tolerance:
                matches.append((abs(nma_temp - gk_temp), nma, nma_k))
        if not matches:
            continue
        _delta_t, nma, nma_k = min(matches, key=lambda item: item[0])
        diff = gk_k - nma_k
        rel = abs(diff) / abs(gk_k) if abs(gk_k) > 1.0e-12 else None
        if rel is None:
            diagnostic = "undefined"
        elif rel <= warning_fraction:
            diagnostic = "phonon_like_consistent"
        elif rel <= large_gap_fraction:
            diagnostic = "moderate_gk_nma_gap"
        else:
            diagnostic = "large_gap_nonphonon_or_disorder_transport"
        comparisons.append(
            {
                "T_K": gk_temp,
                "gk_source": gk.get("source"),
                "nma_source": nma.get("source"),
                "k_gk_W_mK": gk_k,
                "k_nma_W_mK": nma_k,
                "delta_gk_minus_nma_W_mK": diff,
                "relative_abs_difference": rel,
                "diagnostic": diagnostic,
            }
        )
    return comparisons


def maybe_plot(path: Path, rows: list[dict[str, Any]]) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    if not rows:
        return ""
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row.get("source") or "series"), []).append(row)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    for source, items in by_source.items():
        items = sorted(items, key=lambda item: float(item["T_K"]))
        t = np.asarray([float(item["T_K"]) for item in items], dtype=float)
        k = np.asarray([float(item["k_W_mK"]) for item in items], dtype=float)
        err = np.asarray([finite_float(item.get("k_std_W_mK")) or 0.0 for item in items], dtype=float)
        ax.plot(t, k, marker="o", linewidth=1.6, label=source)
        if np.any(err > 0):
            ax.fill_between(t, k - err, k + err, alpha=0.18)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Thermal conductivity (W/m/K)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    target = path / "thermal_conductivity_vs_T.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    return str(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermal_k_lammps",
        description=(
            "Collect thermal conductivity from elastic lower-bound estimates, "
            "pre-integrated MD tables, scaled Green-Kubo HCACF integrals, "
            "or NMA mode/summary tables."
        ),
    )
    parser.add_argument("--elastic-dir", type=Path, help="Directory containing elastic_thermophysical_summary.csv.")
    parser.add_argument("--elastic-summary", type=Path, help="Explicit elastic_thermophysical_summary.csv path.")
    parser.add_argument("--formula", help="Formula, e.g. UO2, recorded in output metadata.")
    parser.add_argument("--natoms", type=float, help="Atoms in the MD/elastic simulation cell.")
    parser.add_argument("--atoms-per-formula-unit", type=float, help="Atoms per formula unit.")
    parser.add_argument("--formula-units", type=float, help="Formula units in the simulation cell.")
    parser.add_argument("--target-z", type=float, default=4.0, help="Formula units in the target crystallographic cell.")
    parser.add_argument(
        "--elastic-select",
        choices=("cahill", "clarke", "average"),
        default="cahill",
        help="Which elastic lower-bound conductivity estimate to export.",
    )
    parser.add_argument("--k-csv", type=Path, action="append", default=[], help="CSV with T_K,k_W_mK columns.")
    parser.add_argument("--k-label", action="append", default=[], help="Label for each --k-csv table.")
    parser.add_argument("--green-kubo-csv", type=Path, action="append", default=[], help="CSV with time_ps and HCACF_x/y/z columns.")
    parser.add_argument("--green-kubo-temperature-K", type=float, action="append", default=[], help="Temperature for each --green-kubo-csv.")
    parser.add_argument("--green-kubo-label", action="append", default=[], help="Label for each Green-Kubo table.")
    parser.add_argument(
        "--green-kubo-scale",
        type=float,
        default=1.0,
        help="Convert raw HCACF time integral to W/m/K. Use 1 for already scaled HCACF.",
    )
    parser.add_argument("--plateau-start-ps", type=float, help="Average cumulative Green-Kubo integral after this time.")
    parser.add_argument("--plateau-fraction", type=float, default=0.2, help="Tail fraction used when --plateau-start-ps is absent.")
    parser.add_argument(
        "--nma-csv",
        type=Path,
        action="append",
        default=[],
        help=(
            "NMA summary CSV with T_K,k_W_mK, or mode table with T_K,lifetime_ps,"
            "mode_heat_capacity_J_m3K,vg_x/y/z_m_s."
        ),
    )
    parser.add_argument("--nma-label", action="append", default=[], help="Label for each --nma-csv table.")
    parser.add_argument(
        "--nma-optical-cutoff-THz",
        type=float,
        help="Optional frequency threshold for acoustic/optical NMA contribution split.",
    )
    parser.add_argument(
        "--compare-gk-nma",
        action="store_true",
        help="Write a GK-vs-NMA diagnostic table when both sources are present.",
    )
    parser.add_argument(
        "--gk-nma-temperature-tolerance",
        type=float,
        default=1.0,
        help="Temperature tolerance in K for matching GK and NMA rows. Default: 1 K.",
    )
    parser.add_argument(
        "--gk-nma-warning-fraction",
        type=float,
        default=0.25,
        help="Relative GK/NMA difference below this is labeled phonon-like consistent.",
    )
    parser.add_argument(
        "--gk-nma-large-gap-fraction",
        type=float,
        default=0.50,
        help="Relative GK/NMA difference above this is labeled a large nonphonon/disorder gap.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("analysis/thermal_k_lammps"))
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    formula_units = infer_formula_units(
        formula_units=args.formula_units,
        natoms=args.natoms,
        atoms_per_formula_unit=args.atoms_per_formula_unit,
        formula=args.formula,
    )
    meta = cell_metadata(
        formula=args.formula,
        natoms=args.natoms,
        atoms_per_formula_unit=args.atoms_per_formula_unit,
        formula_units=formula_units,
        target_z=args.target_z,
        cell_role="thermal-conductivity-source-cell",
        normalization_basis="per-formula",
    )
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    summary = args.elastic_summary
    if summary is None and args.elastic_dir is not None:
        summary = args.elastic_dir / "elastic_thermophysical_summary.csv"
    if summary is not None:
        elastic_rows = elastic_summary_rows(summary.resolve(), args.elastic_select, meta)
        rows.extend(elastic_rows)
        sources.append({"kind": "elastic_lower_bound", "path": str(summary.resolve()), "rows": len(elastic_rows)})
    for idx, path in enumerate(args.k_csv):
        label = args.k_label[idx] if idx < len(args.k_label) else None
        table_rows = conductivity_table_rows(path.resolve(), label=label, meta=meta)
        rows.extend(table_rows)
        sources.append({"kind": "direct_table", "path": str(path.resolve()), "rows": len(table_rows)})
    for idx, path in enumerate(args.green_kubo_csv):
        temp = args.green_kubo_temperature_K[idx] if idx < len(args.green_kubo_temperature_K) else None
        label = args.green_kubo_label[idx] if idx < len(args.green_kubo_label) else f"green_kubo_{idx + 1}"
        gk_rows, gk_meta = green_kubo_rows(
            path.resolve(),
            temperature_K=temp,
            scale_to_W_mK=args.green_kubo_scale,
            plateau_start_ps=args.plateau_start_ps,
            plateau_fraction=args.plateau_fraction,
            label=label,
            meta=meta,
        )
        rows.extend(gk_rows)
        sources.append({"kind": "green_kubo", "path": str(path.resolve()), "rows": len(gk_rows), **gk_meta})
    for idx, path in enumerate(args.nma_csv):
        label = args.nma_label[idx] if idx < len(args.nma_label) else f"nma_{idx + 1}"
        mode_rows, nma_meta = nma_rows(
            path.resolve(),
            label=label,
            meta=meta,
            optical_cutoff_thz=args.nma_optical_cutoff_THz,
        )
        rows.extend(mode_rows)
        sources.append({"kind": "nma", "path": str(path.resolve()), "rows": len(mode_rows), **nma_meta})
    rows = sorted(rows, key=lambda row: (float(row["T_K"]), str(row.get("source") or "")))
    fields = [
        "T_K",
        "k_W_mK",
        "k_std_W_mK",
        "k_min_cahill_W_mK",
        "k_min_clarke_W_mK",
        "k_x_W_mK",
        "k_y_W_mK",
        "k_z_W_mK",
        "nma_acoustic_k_W_mK",
        "nma_optical_k_W_mK",
        "nma_unknown_branch_k_W_mK",
        "nma_mode_count",
        "nma_lifetime_mean_ps",
        "nma_lifetime_min_ps",
        "nma_lifetime_max_ps",
        "source",
        "source_file",
        "formula",
        "natoms",
        "atoms_per_formula_unit",
        "n_formula_units",
        "target_z_formula_units",
        "cell_role",
        "normalization_basis",
    ]
    csv_path = outdir / "thermal_conductivity_T.csv"
    write_csv(csv_path, rows, fields)
    comparisons = []
    comparison_path = ""
    should_compare = args.compare_gk_nma or (
        any(source.get("kind") == "green_kubo" for source in sources)
        and any(source.get("kind") == "nma" for source in sources)
    )
    if should_compare:
        comparisons = compare_gk_nma(
            rows,
            temp_tolerance=args.gk_nma_temperature_tolerance,
            warning_fraction=args.gk_nma_warning_fraction,
            large_gap_fraction=args.gk_nma_large_gap_fraction,
        )
        if comparisons:
            comparison_path = str(outdir / "gk_nma_comparison.csv")
            write_csv(
                Path(comparison_path),
                comparisons,
                [
                    "T_K",
                    "gk_source",
                    "nma_source",
                    "k_gk_W_mK",
                    "k_nma_W_mK",
                    "delta_gk_minus_nma_W_mK",
                    "relative_abs_difference",
                    "diagnostic",
                ],
            )
    plot = "" if args.no_plot else maybe_plot(outdir, rows)
    metadata = {
        "schema": "atomi.lammps.thermal_conductivity.v1",
        "outputs": {"csv": str(csv_path), "plot": plot, "gk_nma_comparison_csv": comparison_path},
        "cell_metadata": meta,
        "n_rows": len(rows),
        "sources": sources,
        "gk_nma_comparison": {
            "n_rows": len(comparisons),
            "temperature_tolerance_K": args.gk_nma_temperature_tolerance,
            "warning_fraction": args.gk_nma_warning_fraction,
            "large_gap_fraction": args.gk_nma_large_gap_fraction,
        },
        "notes": [
            "Elastic lower-bound estimates are DFT/elasticity screening values, not anharmonic transport.",
            "Green-Kubo raw HCACF integration requires a user-supplied unit scale unless the input is already in W/m/K units.",
            "NMA mode tables diagnose phonon quasiparticle lifetimes; full MD-to-mode projection is generated upstream.",
            "Large GK-NMA gaps can indicate mobile defects, coherent/off-diagonal transport, strong disorder, or an MLIP dynamics issue.",
        ],
    }
    write_json(outdir / "thermal_conductivity_metadata.json", metadata)
    print(f"Wrote thermal conductivity table: {csv_path}")
    if comparison_path:
        print(f"Wrote GK/NMA comparison: {comparison_path}")
    return metadata


if __name__ == "__main__":
    main()
