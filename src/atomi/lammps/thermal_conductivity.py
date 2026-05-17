"""Thermal-conductivity table helpers for DFT/MD-to-MOOSE workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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


def elastic_summary_rows(path: Path, select: str) -> list[dict[str, Any]]:
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
            {
                "T_K": temp,
                "k_W_mK": selected,
                "k_min_cahill_W_mK": cahill,
                "k_min_clarke_W_mK": clarke,
                "source": f"elastic_min_{select}",
                "source_file": str(path),
            }
        )
    return rows


def conductivity_table_rows(path: Path, label: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        temp = finite_float(row.get("T_K") or row.get("temperature_K"))
        conductivity = finite_float(row.get("k_W_mK") or row.get("thermal_conductivity_W_mK"))
        if temp is None or conductivity is None:
            continue
        rows.append(
            {
                "T_K": temp,
                "k_W_mK": conductivity,
                "k_std_W_mK": finite_float(row.get("k_std_W_mK")),
                "source": label or row.get("source") or "table",
                "source_file": str(path),
            }
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(path)
    if not rows:
        return [], {"source_file": str(path), "error": "empty CSV"}
    if any("k_W_mK" in row and row["k_W_mK"] for row in rows):
        return conductivity_table_rows(path, label=label), {"source_file": str(path), "mode": "pre_integrated"}
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
    result = {
        "T_K": temp,
        "k_W_mK": float(np.mean(values)),
        "k_std_W_mK": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "source": label,
        "source_file": str(path),
        **k_components,
    }
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
            "pre-integrated MD tables, or scaled Green-Kubo HCACF integrals."
        ),
    )
    parser.add_argument("--elastic-dir", type=Path, help="Directory containing elastic_thermophysical_summary.csv.")
    parser.add_argument("--elastic-summary", type=Path, help="Explicit elastic_thermophysical_summary.csv path.")
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
    parser.add_argument("--outdir", type=Path, default=Path("analysis/thermal_k_lammps"))
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    summary = args.elastic_summary
    if summary is None and args.elastic_dir is not None:
        summary = args.elastic_dir / "elastic_thermophysical_summary.csv"
    if summary is not None:
        elastic_rows = elastic_summary_rows(summary.resolve(), args.elastic_select)
        rows.extend(elastic_rows)
        sources.append({"kind": "elastic_lower_bound", "path": str(summary.resolve()), "rows": len(elastic_rows)})
    for idx, path in enumerate(args.k_csv):
        label = args.k_label[idx] if idx < len(args.k_label) else None
        table_rows = conductivity_table_rows(path.resolve(), label=label)
        rows.extend(table_rows)
        sources.append({"kind": "direct_table", "path": str(path.resolve()), "rows": len(table_rows)})
    for idx, path in enumerate(args.green_kubo_csv):
        temp = args.green_kubo_temperature_K[idx] if idx < len(args.green_kubo_temperature_K) else None
        label = args.green_kubo_label[idx] if idx < len(args.green_kubo_label) else f"green_kubo_{idx + 1}"
        gk_rows, meta = green_kubo_rows(
            path.resolve(),
            temperature_K=temp,
            scale_to_W_mK=args.green_kubo_scale,
            plateau_start_ps=args.plateau_start_ps,
            plateau_fraction=args.plateau_fraction,
            label=label,
        )
        rows.extend(gk_rows)
        sources.append({"kind": "green_kubo", "path": str(path.resolve()), "rows": len(gk_rows), **meta})
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
        "source",
        "source_file",
    ]
    csv_path = outdir / "thermal_conductivity_T.csv"
    write_csv(csv_path, rows, fields)
    plot = "" if args.no_plot else maybe_plot(outdir, rows)
    metadata = {
        "schema": "atomi.lammps.thermal_conductivity.v1",
        "outputs": {"csv": str(csv_path), "plot": plot},
        "n_rows": len(rows),
        "sources": sources,
        "notes": [
            "Elastic lower-bound estimates are DFT/elasticity screening values, not anharmonic transport.",
            "Green-Kubo raw HCACF integration requires a user-supplied unit scale unless the input is already in W/m/K units.",
        ],
    }
    write_json(outdir / "thermal_conductivity_metadata.json", metadata)
    print(f"Wrote thermal conductivity table: {csv_path}")
    return metadata


if __name__ == "__main__":
    main()
