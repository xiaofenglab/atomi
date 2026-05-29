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


def finite_int(value: Any) -> int | None:
    number = finite_float(value)
    if number is None:
        return None
    return int(number)


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


def first_finite_value(row: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = finite_float(row.get(name))
        if value is not None:
            return value
    return None


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


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
        conductivity = first_finite_value(
            row,
            ("k_W_mK", "k_mean_W_mK", "thermal_conductivity_W_mK"),
        )
        if temp is None or conductivity is None:
            continue
        source = label or row.get("source") or "table"
        rows.append(
            attach_cell_columns(
                {
                "T_K": temp,
                "k_W_mK": conductivity,
                "k_std_W_mK": first_finite_value(row, ("k_std_W_mK", "k_seed_std_W_mK")),
                "k_sem_W_mK": first_finite_value(row, ("k_sem_W_mK", "k_seed_sem_W_mK")),
                "k_ci95_W_mK": first_finite_value(row, ("k_ci95_W_mK", "k_seed_ci95_W_mK")),
                "seed_count": first_finite_value(row, ("seed_count", "n_gk_seeds")),
                "ok_seed_count": first_finite_value(row, ("ok_seed_count", "n_gk_seeds")),
                "seed_cv_fraction": finite_float(row.get("seed_cv_fraction")),
                "axis_spread_fraction": finite_float(row.get("axis_spread_fraction")),
                "slope_disagreement_fraction": first_finite_value(
                    row,
                    ("slope_disagreement_fraction", "slope_disagreement_mean_fraction"),
                ),
                "source": source,
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


def read_json_optional(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"parse_error": str(path)}


def validation_reports_by_temperature(path: Path) -> dict[float, dict[str, Any]]:
    payload = read_json_optional(path)
    if not isinstance(payload, dict):
        return {}
    reports = payload.get("reports") or payload.get("temperatures") or []
    if not isinstance(reports, list):
        return {}
    out: dict[float, dict[str, Any]] = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        temp = finite_float(report.get("temperature_K") or report.get("T_K"))
        if temp is not None:
            out[temp] = report
    return out


def _validation_match(
    reports: dict[float, dict[str, Any]],
    temperature: float,
    tolerance: float = 1.0,
) -> dict[str, Any]:
    if not reports:
        return {}
    closest = min(reports, key=lambda temp: abs(temp - temperature))
    if abs(closest - temperature) <= tolerance:
        return reports[closest]
    return {}


def _route_label(route: str, fit_dir: Path, explicit: str | None, index: int) -> str:
    if explicit:
        return explicit
    name = fit_dir.parent.name or fit_dir.name
    return f"{route}_{name or index + 1}"


def _sem_from_std(std: float | None, count: int | None) -> float | None:
    if std is None:
        return None
    if count is None or count <= 1:
        return std
    return std / math.sqrt(count)


def _ci95_from_sem(sem: float | None) -> float | None:
    if sem is None:
        return None
    return 1.96 * sem


def route_summary_rows(
    fit_dir: Path,
    *,
    route: str,
    label: str,
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fit_dir = fit_dir.resolve()
    if route == "gk":
        table_path = fit_dir / "thermal_conductivity_T.csv"
        validation_path = fit_dir / "gk_validation_summary.json"
        seed_summary_path = fit_dir / "gk_seed_summary.csv"
    elif route == "rnemd":
        table_path = fit_dir / "thermal_conductivity_rnemd_T.csv"
        validation_path = fit_dir / "rnemd_validation_summary.json"
        seed_summary_path = fit_dir / "rnemd_seed_summary.csv"
    else:
        raise ValueError(f"Unknown transport route {route!r}")
    if not table_path.exists():
        return [], {
            "kind": f"{route}_fit",
            "fit_dir": str(fit_dir),
            "error": f"missing {table_path.name}",
        }
    validation = validation_reports_by_temperature(validation_path)
    rows: list[dict[str, Any]] = []
    for raw in read_csv(table_path):
        temp = finite_float(raw.get("T_K") or raw.get("temperature_K"))
        k_value = first_finite_value(raw, ("k_W_mK", "k_mean_W_mK", "thermal_conductivity_W_mK"))
        if temp is None or k_value is None:
            continue
        report = _validation_match(validation, temp)
        seed_count = finite_int(raw.get("seed_count") or raw.get("n_gk_seeds") or report.get("seed_count"))
        ok_seed_count = finite_int(raw.get("ok_seed_count") or raw.get("n_gk_seeds") or report.get("ok_seed_count"))
        std = first_not_none(
            first_finite_value(raw, ("k_std_W_mK", "k_seed_std_W_mK")),
            finite_float(report.get("k_seed_std_W_mK")),
        )
        sem = first_finite_value(raw, ("k_sem_W_mK", "k_seed_sem_W_mK"))
        if sem is None:
            sem = _sem_from_std(std, ok_seed_count or seed_count)
        ci95 = first_finite_value(raw, ("k_ci95_W_mK", "k_seed_ci95_W_mK"))
        if ci95 is None:
            ci95 = _ci95_from_sem(sem)
        row = attach_cell_columns(
            {
                "T_K": temp,
                "route": route,
                "k_W_mK": k_value,
                "k_std_W_mK": std,
                "k_sem_W_mK": sem,
                "k_ci95_W_mK": ci95,
                "seed_count": seed_count,
                "ok_seed_count": ok_seed_count,
                "validation_status": report.get("status", ""),
                "seed_cv_fraction": first_not_none(
                    first_finite_value(raw, ("seed_cv_fraction",)),
                    finite_float(report.get("seed_cv_fraction")),
                ),
                "axis_spread_fraction": first_not_none(
                    first_finite_value(raw, ("axis_spread_fraction",)),
                    finite_float(report.get("axis_spread_fraction")),
                ),
                "late_drift_fraction": first_not_none(
                    finite_float(report.get("late_integral_drift_mean_fraction")),
                    finite_float(report.get("late_drift_fraction")),
                ),
                "slope_disagreement_fraction": first_not_none(
                    first_finite_value(
                        raw,
                        ("slope_disagreement_fraction", "slope_disagreement_mean_fraction"),
                    ),
                    finite_float(report.get("slope_disagreement_fraction")),
                ),
                "source": label,
                "source_file": str(table_path),
                "fit_dir": str(fit_dir),
                "validation_json": str(validation_path) if validation_path.exists() else "",
            },
            meta,
        )
        rows.append(row)
    return rows, {
        "kind": f"{route}_fit",
        "label": label,
        "fit_dir": str(fit_dir),
        "temperature_table": str(table_path),
        "seed_summary": str(seed_summary_path) if seed_summary_path.exists() else "",
        "validation_json": str(validation_path) if validation_path.exists() else "",
        "rows": len(rows),
    }


def route_uncertainty(row: dict[str, Any]) -> float | None:
    for key in ("k_sem_W_mK", "k_ci95_W_mK", "k_std_W_mK"):
        value = finite_float(row.get(key))
        if value is not None and value > 0:
            if key == "k_ci95_W_mK":
                return value / 1.96
            return value
    return None


def transport_route_ok(row: dict[str, Any]) -> bool:
    k_value = finite_float(row.get("k_W_mK"))
    if k_value is None or k_value <= 0:
        return False
    status = str(row.get("validation_status") or "").lower()
    return status not in {"fail", "failed", "error"}


def combine_transport_routes(
    route_rows: list[dict[str, Any]],
    *,
    temperature_tolerance: float,
    route_disagreement_warn_fraction: float,
    route_disagreement_fail_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    remaining = sorted(route_rows, key=lambda row: float(row["T_K"]))
    clusters: list[list[dict[str, Any]]] = []
    for row in remaining:
        temp = float(row["T_K"])
        for cluster in clusters:
            center = float(np.mean([float(item["T_K"]) for item in cluster]))
            if abs(temp - center) <= temperature_tolerance:
                cluster.append(row)
                break
        else:
            clusters.append([row])

    combined_rows: list[dict[str, Any]] = []
    crosscheck_rows: list[dict[str, Any]] = []
    for cluster in clusters:
        temp = float(np.mean([float(row["T_K"]) for row in cluster]))
        usable = [row for row in cluster if transport_route_ok(row)]
        k_values = [float(row["k_W_mK"]) for row in usable]
        sigmas = [route_uncertainty(row) for row in usable]
        notes: list[str] = []
        status = "ok"
        if len(usable) < len(cluster):
            notes.append("one or more route rows were invalid or failed validation")
            status = "warn"
        if not usable:
            combined_k = math.nan
            within = math.nan
            between = math.nan
            combined_uq = math.nan
            status = "fail"
            notes.append("no usable route estimates")
        elif len(usable) == 1:
            combined_k = k_values[0]
            within = sigmas[0] if sigmas[0] is not None else math.nan
            between = math.nan
            combined_uq = within
            status = "warn"
            notes.append("single usable route; no cross-route check")
        elif all(sigma is not None and sigma > 0 for sigma in sigmas):
            weights = [1.0 / float(sigma) ** 2 for sigma in sigmas if sigma is not None]
            combined_k = float(sum(weight * value for weight, value in zip(weights, k_values)) / sum(weights))
            within = math.sqrt(1.0 / sum(weights))
            between = float(np.std(k_values, ddof=1))
            combined_uq = max(within, between)
        else:
            combined_k = float(np.mean(k_values))
            between = float(np.std(k_values, ddof=1)) if len(k_values) > 1 else math.nan
            within = between / math.sqrt(len(k_values)) if math.isfinite(between) else math.nan
            combined_uq = between if math.isfinite(between) else math.nan
            notes.append("missing per-route SEM for inverse-variance weighting; used arithmetic mean")
            status = "warn"
        spread = None
        if len(k_values) > 1 and abs(combined_k) > 1.0e-12 and math.isfinite(combined_k):
            spread = (max(k_values) - min(k_values)) / abs(combined_k)
            if spread >= route_disagreement_fail_fraction:
                status = "fail"
                notes.append("route disagreement exceeds fail threshold")
            elif spread >= route_disagreement_warn_fraction and status != "fail":
                status = "warn"
                notes.append("route disagreement exceeds warning threshold")
        z_score = None
        if len(usable) == 2 and all(sigma is not None and sigma > 0 for sigma in sigmas):
            denom = math.sqrt(float(sigmas[0]) ** 2 + float(sigmas[1]) ** 2)
            if denom > 0:
                z_score = abs(k_values[0] - k_values[1]) / denom
        by_route = {str(row.get("route")): row for row in cluster}
        gk = by_route.get("gk", {})
        rnemd = by_route.get("rnemd", {})
        common = {
            "T_K": temp,
            "route_count": len(cluster),
            "ok_route_count": len(usable),
            "gk_k_W_mK": finite_float(gk.get("k_W_mK")),
            "gk_uq_W_mK": route_uncertainty(gk) if gk else None,
            "gk_status": gk.get("validation_status", ""),
            "rnemd_k_W_mK": finite_float(rnemd.get("k_W_mK")),
            "rnemd_uq_W_mK": route_uncertainty(rnemd) if rnemd else None,
            "rnemd_status": rnemd.get("validation_status", ""),
            "route_spread_fraction": spread,
            "route_z_score": z_score,
            "status": status,
            "notes": "; ".join(dict.fromkeys(notes)),
        }
        combined_row = {
            "T_K": temp,
            "k_W_mK": combined_k,
            "k_std_W_mK": combined_uq,
            "k_sem_W_mK": within,
            "k_between_route_std_W_mK": between,
            "source": "combined_gk_rnemd",
            "source_tag": "combined_gk_rnemd",
            **common,
        }
        for key in (
            "formula",
            "natoms",
            "atoms_per_formula_unit",
            "n_formula_units",
            "target_z_formula_units",
            "cell_role",
            "normalization_basis",
        ):
            combined_row[key] = cluster[0].get(key)
        combined_rows.append(combined_row)
        crosscheck_rows.append(common | {"k_combined_W_mK": combined_k, "k_uq_W_mK": combined_uq})
    return combined_rows, crosscheck_rows


def write_moose_thermal_conductivity_csv(path: Path, rows: list[dict[str, Any]], source_tag: str) -> None:
    moose_rows = []
    for row in rows:
        temp = finite_float(row.get("T_K"))
        k_value = finite_float(row.get("k_W_mK"))
        if temp is None or k_value is None:
            continue
        moose_rows.append(
            {
                "T_K": temp,
                "k_W_mK": k_value,
                "k_std_W_mK": finite_float(row.get("k_std_W_mK")),
                "source_tag": row.get("source_tag") or source_tag,
            }
        )
    write_csv(path, moose_rows, ["T_K", "k_W_mK", "k_std_W_mK", "source_tag"])


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
    parser.add_argument(
        "--gk-fit-dir",
        type=Path,
        action="append",
        default=[],
        help="GK fit directory containing thermal_conductivity_T.csv and optional validation JSON.",
    )
    parser.add_argument("--gk-fit-label", action="append", default=[], help="Label for each --gk-fit-dir table.")
    parser.add_argument(
        "--rnemd-fit-dir",
        type=Path,
        action="append",
        default=[],
        help="rNEMD fit directory containing thermal_conductivity_rnemd_T.csv and optional validation JSON.",
    )
    parser.add_argument("--rnemd-fit-label", action="append", default=[], help="Label for each --rnemd-fit-dir table.")
    parser.add_argument(
        "--route-temperature-tolerance-K",
        type=float,
        default=1.0,
        help="Temperature tolerance for matching GK/rNEMD rows. Default: 1 K.",
    )
    parser.add_argument("--route-disagreement-warn-fraction", type=float, default=0.25)
    parser.add_argument("--route-disagreement-fail-fraction", type=float, default=0.50)
    parser.add_argument(
        "--moose-k-csv",
        type=Path,
        help="MOOSE-readable T_K,k_W_mK,k_std_W_mK source CSV. Default: outdir/moose_thermal_conductivity.csv when routes are combined.",
    )
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
    route_rows: list[dict[str, Any]] = []
    for idx, fit_dir in enumerate(args.gk_fit_dir):
        label = _route_label("gk", fit_dir, args.gk_fit_label[idx] if idx < len(args.gk_fit_label) else None, idx)
        extracted, source = route_summary_rows(fit_dir, route="gk", label=label, meta=meta)
        route_rows.extend(extracted)
        sources.append(source)
    for idx, fit_dir in enumerate(args.rnemd_fit_dir):
        label = _route_label(
            "rnemd",
            fit_dir,
            args.rnemd_fit_label[idx] if idx < len(args.rnemd_fit_label) else None,
            idx,
        )
        extracted, source = route_summary_rows(fit_dir, route="rnemd", label=label, meta=meta)
        route_rows.extend(extracted)
        sources.append(source)
    rows.extend(route_rows)
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
        "route",
        "k_W_mK",
        "k_std_W_mK",
        "k_sem_W_mK",
        "k_ci95_W_mK",
        "k_between_route_std_W_mK",
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
        "seed_count",
        "ok_seed_count",
        "seed_cv_fraction",
        "axis_spread_fraction",
        "late_drift_fraction",
        "slope_disagreement_fraction",
        "validation_status",
        "route_count",
        "ok_route_count",
        "gk_k_W_mK",
        "rnemd_k_W_mK",
        "route_spread_fraction",
        "route_z_score",
        "status",
        "notes",
        "source_tag",
        "source",
        "source_file",
        "fit_dir",
        "validation_json",
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
    combined_rows: list[dict[str, Any]] = []
    crosscheck_rows: list[dict[str, Any]] = []
    route_summary_path = ""
    combined_path = ""
    crosscheck_path = ""
    moose_k_csv = ""
    if route_rows:
        route_summary_path = str(outdir / "thermal_conductivity_route_summary.csv")
        write_csv(Path(route_summary_path), route_rows, fields)
        combined_rows, crosscheck_rows = combine_transport_routes(
            route_rows,
            temperature_tolerance=args.route_temperature_tolerance_K,
            route_disagreement_warn_fraction=args.route_disagreement_warn_fraction,
            route_disagreement_fail_fraction=args.route_disagreement_fail_fraction,
        )
        if combined_rows:
            combined_path = str(outdir / "thermal_conductivity_combined_T.csv")
            write_csv(Path(combined_path), combined_rows, fields)
            crosscheck_path = str(outdir / "thermal_conductivity_route_crosscheck.csv")
            write_csv(
                Path(crosscheck_path),
                crosscheck_rows,
                [
                    "T_K",
                    "k_combined_W_mK",
                    "k_uq_W_mK",
                    "route_count",
                    "ok_route_count",
                    "gk_k_W_mK",
                    "gk_uq_W_mK",
                    "gk_status",
                    "rnemd_k_W_mK",
                    "rnemd_uq_W_mK",
                    "rnemd_status",
                    "route_spread_fraction",
                    "route_z_score",
                    "status",
                    "notes",
                ],
            )
            target = args.moose_k_csv.resolve() if args.moose_k_csv else outdir / "moose_thermal_conductivity.csv"
            write_moose_thermal_conductivity_csv(target, combined_rows, "combined_gk_rnemd")
            moose_k_csv = str(target)
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
    plot_rows = rows + combined_rows
    plot = "" if args.no_plot else maybe_plot(outdir, plot_rows)
    metadata = {
        "schema": "atomi.lammps.thermal_conductivity.v1",
        "outputs": {
            "csv": str(csv_path),
            "plot": plot,
            "gk_nma_comparison_csv": comparison_path,
            "route_summary_csv": route_summary_path,
            "route_crosscheck_csv": crosscheck_path,
            "combined_csv": combined_path,
            "moose_thermal_conductivity_csv": moose_k_csv,
        },
        "cell_metadata": meta,
        "n_rows": len(rows),
        "sources": sources,
        "route_crosscheck": {
            "n_route_rows": len(route_rows),
            "n_combined_rows": len(combined_rows),
            "temperature_tolerance_K": args.route_temperature_tolerance_K,
            "route_disagreement_warn_fraction": args.route_disagreement_warn_fraction,
            "route_disagreement_fail_fraction": args.route_disagreement_fail_fraction,
            "uq_convention": (
                "Per-route SEM is preferred; route std is used when SEM is absent; "
                "the final combined uncertainty is the larger of inverse-variance "
                "within-route uncertainty and between-route spread."
            ),
        },
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
            "MOOSE can ingest moose_thermal_conductivity.csv through moose-qha-md-material --property-csv; it supplies T_K,k_W_mK,k_std_W_mK,source_tag.",
        ],
    }
    write_json(outdir / "thermal_conductivity_metadata.json", metadata)
    print(f"Wrote thermal conductivity table: {csv_path}")
    if route_summary_path:
        print(f"Wrote route summary: {route_summary_path}")
    if crosscheck_path:
        print(f"Wrote route crosscheck: {crosscheck_path}")
    if combined_path:
        print(f"Wrote combined k(T): {combined_path}")
    if moose_k_csv:
        print(f"Wrote MOOSE thermal-conductivity CSV: {moose_k_csv}")
    if comparison_path:
        print(f"Wrote GK/NMA comparison: {comparison_path}")
    return metadata


if __name__ == "__main__":
    main()
