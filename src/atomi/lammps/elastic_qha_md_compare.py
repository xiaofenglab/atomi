#!/usr/bin/env python3
"""Compare LAMMPS finite-T elastic results against VASP/QHA structural or elastic data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Optional


ELASTIC_COMPONENTS = ("C11", "C22", "C33", "C12", "C13", "C23", "C44", "C55", "C66")
MODULI_COMPONENTS = ("K_H", "G_H", "E_H", "nu_H", "K_V", "G_V", "K_R", "G_R")
PLOT_COMPONENTS = ("C11", "C12", "C44", "K_H", "G_H", "E_H", "nu_H")

QHA_ELASTIC_CSV_CANDIDATES = (
    "elastic_moduli_T.csv",
    "elastic_constants_T.csv",
    "qha_elastic_moduli_T.csv",
    "qha_elastic_constants_T.csv",
    "static_elastic_moduli_T.csv",
    "static_elastic_constants_T.csv",
)

STRUCTURAL_QHA_CANDIDATES = {
    "V": ("volume-temperature.dat", "volume_temperature.dat"),
    "a": ("a-temperature.dat", "lattice_a-temperature.dat", "lattice-temperature.dat"),
    "b": ("b-temperature.dat", "lattice_b-temperature.dat"),
    "c": ("c-temperature.dat", "lattice_c-temperature.dat"),
}

MD_STRUCTURAL_ALIASES = {
    "V": ("V_mean_A3", "vol_mean_A3", "volume_A3_mean", "V_fit_A3", "V_target_cell_A3"),
    "a": ("a_mean_A", "a_A_mean", "a_fit_A", "a_proxy_mean_A"),
    "b": ("b_mean_A", "b_A_mean", "b_fit_A", "ly_mean_A"),
    "c": ("c_mean_A", "c_A_mean", "c_fit_A", "lz_mean_A"),
}


def finite_float(value, default=math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_table(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            values = [float(x) for x in stripped.replace(",", " ").split()]
        except ValueError:
            continue
        if len(values) >= 2 and math.isfinite(values[0]) and math.isfinite(values[1]):
            rows.append((values[0], values[1]))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_component_name(name: str) -> str:
    clean = name.strip()
    for suffix in ("_GPa", "_gpa"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
    aliases = {
        "K": "K_H",
        "G": "G_H",
        "E": "E_H",
        "nu": "nu_H",
        "poisson": "nu_H",
        "poisson_ratio": "nu_H",
        "bulk_modulus": "K_H",
        "shear_modulus": "G_H",
        "youngs_modulus": "E_H",
        "young_modulus": "E_H",
    }
    return aliases.get(clean, clean)


def temperature_from_row(row: dict[str, str]) -> float:
    for key in ("temperature_K", "T_K", "temperature", "T"):
        value = finite_float(row.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def component_value_from_row(row: dict[str, str], component: str) -> float:
    aliases = [component, f"{component}_GPa"]
    if component == "nu_H":
        aliases.extend(("nu", "poisson_ratio", "nu_H"))
    for key in aliases:
        value = finite_float(row.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def read_elastic_csv(path: Path, source: str) -> list[dict]:
    rows = []
    for row in read_csv(path):
        temp = temperature_from_row(row)
        if not math.isfinite(temp):
            continue
        for component in ELASTIC_COMPONENTS + MODULI_COMPONENTS:
            value = component_value_from_row(row, component)
            if math.isfinite(value):
                rows.append(
                    {
                        "source": source,
                        "temperature_K": temp,
                        "component": component,
                        "value": value,
                        "unit": "" if component == "nu_H" else "GPa",
                        "input_file": str(path),
                    }
                )
    return rows


def read_component_dat_files(root: Path, source: str) -> list[dict]:
    rows = []
    for component in ELASTIC_COMPONENTS + MODULI_COMPONENTS:
        for name in (
            f"{component}-temperature.dat",
            f"{component}_temperature.dat",
            f"{component.lower()}-temperature.dat",
            f"{component.lower()}_temperature.dat",
        ):
            path = root / name
            points = read_table(path)
            if not points:
                continue
            for temp, value in points:
                rows.append(
                    {
                        "source": source,
                        "temperature_K": temp,
                        "component": component,
                        "value": value,
                        "unit": "" if component == "nu_H" else "GPa",
                        "input_file": str(path),
                    }
                )
            break
    return rows


def discover_qha_elastic(qha_dir: Path, explicit: Optional[Path]) -> tuple[list[dict], dict]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(qha_dir / name for name in QHA_ELASTIC_CSV_CANDIDATES)
    rows: list[dict] = []
    used_files: list[str] = []
    for path in candidates:
        if path.exists():
            parsed = read_elastic_csv(path, "QHA/static")
            if parsed:
                rows.extend(parsed)
                used_files.append(str(path))
                break
    dat_rows = read_component_dat_files(qha_dir, "QHA/static")
    if dat_rows:
        rows.extend(dat_rows)
        used_files.extend(sorted({row["input_file"] for row in dat_rows}))
    components = sorted({row["component"] for row in rows})
    return rows, {
        "elasticity_ready": bool(rows),
        "files": used_files,
        "components": components,
        "note": (
            "QHA/static elastic data found."
            if rows
            else "Standard phonopy-QHA outputs do not contain Cij(T). Provide static/quasi-static elastic tables to compare elastic constants."
        ),
    }


def read_md_elastic(md_dir: Path) -> tuple[list[dict], Path]:
    path = md_dir / "elastic_moduli_T.csv"
    rows = read_elastic_csv(path, "MD elastic")
    if not rows:
        raise FileNotFoundError(f"No MD elastic rows found in {path}")
    return rows, path


def structural_series_from_qha(qha_dir: Path, key: str, qha_formula_units: float, target_z: float) -> tuple[list[dict], str]:
    for name in STRUCTURAL_QHA_CANDIDATES[key]:
        path = qha_dir / name
        points = read_table(path)
        if not points:
            continue
        scale = 1.0
        unit = "A"
        if key == "V":
            scale = target_z / qha_formula_units
            unit = "A^3 target-cell"
        rows = [
            {
                "source": "QHA",
                "temperature_K": temp,
                "quantity": key,
                "value": value * scale,
                "unit": unit,
                "input_file": str(path),
            }
            for temp, value in points
        ]
        return rows, str(path)
    if key == "a":
        volume_rows, path = structural_series_from_qha(qha_dir, "V", qha_formula_units, target_z)
        if volume_rows:
            rows = []
            for row in volume_rows:
                value = finite_float(row["value"])
                if value > 0:
                    rows.append({**row, "quantity": "a", "value": value ** (1.0 / 3.0), "unit": "A", "input_file": path + " (derived cubic a)"})
            return rows, path + " (derived cubic a)"
    return [], ""


def structural_series_from_md(md_dir: Path, key: str, md_formula_units: float, target_z: float) -> tuple[list[dict], str]:
    candidates = [md_dir / "elastic_moduli_T.csv", md_dir / "elastic_stage_summaries.json"]
    csv_path = candidates[0]
    rows = []
    if csv_path.exists():
        for row in read_csv(csv_path):
            temp = temperature_from_row(row)
            if not math.isfinite(temp):
                continue
            value = math.nan
            for alias in MD_STRUCTURAL_ALIASES[key]:
                value = finite_float(row.get(alias))
                if math.isfinite(value):
                    break
            if math.isfinite(value):
                if key == "V":
                    value = value * target_z / md_formula_units
                rows.append(
                    {
                        "source": "MD elastic",
                        "temperature_K": temp,
                        "quantity": key,
                        "value": value,
                        "unit": "A^3 target-cell" if key == "V" else "A",
                        "input_file": str(csv_path),
                    }
                )
    if rows:
        return rows, str(csv_path)
    return [], ""


def filter_t(rows: list[dict], t_min: Optional[float], t_max: Optional[float]) -> list[dict]:
    out = []
    for row in rows:
        temp = finite_float(row.get("temperature_K"))
        if not math.isfinite(temp):
            continue
        if t_min is not None and temp < t_min:
            continue
        if t_max is not None and temp > t_max:
            continue
        out.append(row)
    return out


def plot_component(path: Path, rows: list[dict], component: str, ylabel: str, title: str) -> bool:
    try:
        cache = Path(tempfile.gettempdir()) / "atomi-matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    selected = [row for row in rows if row.get("component", row.get("quantity")) == component]
    if not selected:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    styles = {
        "QHA/static": {"color": "tab:blue", "linestyle": "-", "marker": "o", "mfc": "white"},
        "QHA": {"color": "tab:blue", "linestyle": "-", "marker": "o", "mfc": "white"},
        "MD elastic": {"color": "tab:red", "linestyle": "--", "marker": "s", "mfc": "tab:red"},
    }
    for source in sorted({str(row["source"]) for row in selected}):
        series = sorted((finite_float(row["temperature_K"]), finite_float(row["value"])) for row in selected if row["source"] == source)
        series = [(t, v) for t, v in series if math.isfinite(t) and math.isfinite(v)]
        if not series:
            continue
        style = styles.get(source, {"linestyle": "-", "marker": "o"})
        ax.plot(
            [x for x, _ in series],
            [y for _, y in series],
            label=source,
            linewidth=1.5,
            markersize=4,
            markeredgewidth=1.2,
            **style,
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elastic_qha_md_compare",
        description="Check whether QHA outputs contain elastic data and compare QHA/static elasticity with LAMMPS MD elasticity.",
    )
    parser.add_argument("--qha-dir", type=Path, required=True, help="phonopy-qha output directory.")
    parser.add_argument("--elastic-md-dir", type=Path, required=True, help="elastic_lammps analyze output directory.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--qha-elastic-file", type=Path, help="Optional explicit QHA/static elastic CSV.")
    parser.add_argument("--qha-formula-units", type=float, required=True)
    parser.add_argument("--md-formula-units", type=float, required=True)
    parser.add_argument("--target-z", type=float, default=4.0)
    parser.add_argument("--t-min", type=float)
    parser.add_argument("--t-max", type=float)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    md_rows, md_file = read_md_elastic(args.elastic_md_dir)
    qha_elastic_rows, qha_elastic_meta = discover_qha_elastic(args.qha_dir, args.qha_elastic_file)
    elastic_rows = filter_t(md_rows + qha_elastic_rows, args.t_min, args.t_max)

    structural_rows: list[dict] = []
    structural_files: dict[str, dict[str, str]] = {}
    for key in ("V", "a", "b", "c"):
        qha_rows, qha_file = structural_series_from_qha(args.qha_dir, key, args.qha_formula_units, args.target_z)
        md_struct_rows, md_struct_file = structural_series_from_md(args.elastic_md_dir, key, args.md_formula_units, args.target_z)
        structural_rows.extend(filter_t(qha_rows + md_struct_rows, args.t_min, args.t_max))
        if qha_file or md_struct_file:
            structural_files[key] = {"qha": qha_file, "md": md_struct_file}

    if elastic_rows:
        write_csv(args.outdir / "elastic_qha_md_overlay.csv", elastic_rows)
    if structural_rows:
        write_csv(args.outdir / "structure_qha_md_overlay.csv", structural_rows)

    plot_files: list[str] = []
    if not args.no_plots:
        for component in PLOT_COMPONENTS:
            if plot_component(
                args.outdir / f"{component}_qha_md_overlay.png",
                elastic_rows,
                component,
                "dimensionless" if component == "nu_H" else "GPa",
                f"{component} QHA/static vs MD",
            ):
                plot_files.append(str(args.outdir / f"{component}_qha_md_overlay.png"))
        for key, ylabel in (("V", "Volume (A^3 target-cell)"), ("a", "a (A)"), ("b", "b (A)"), ("c", "c (A)")):
            if plot_component(
                args.outdir / f"{key}_qha_md_structure_overlay.png",
                structural_rows,
                key,
                ylabel,
                f"{key} QHA vs MD elastic-cell check",
            ):
                plot_files.append(str(args.outdir / f"{key}_qha_md_structure_overlay.png"))

    md_components = sorted({row["component"] for row in md_rows})
    metadata = {
        "qha_dir": str(args.qha_dir.resolve()),
        "elastic_md_dir": str(args.elastic_md_dir.resolve()),
        "md_elastic_file": str(md_file.resolve()),
        "qha_formula_units": args.qha_formula_units,
        "md_formula_units": args.md_formula_units,
        "target_z": args.target_z,
        "qha_elastic": qha_elastic_meta,
        "md_elastic_components": md_components,
        "structural_files": structural_files,
        "readiness": {
            "qha_has_structure": bool(structural_rows),
            "qha_has_elastic": bool(qha_elastic_rows),
            "md_has_elastic": bool(md_rows),
            "can_compare_elastic_constants": bool(qha_elastic_rows and md_rows),
            "recommendation": (
                "Compare Cij/moduli directly."
                if qha_elastic_rows
                else "QHA structure can cross-check MD equilibrium V/a, but phonopy-QHA alone is not enough for Cij. Add static elastic calculations at QHA volumes or Cij(T) tables."
            ),
        },
        "outputs": {
            "elastic_overlay_csv": str(args.outdir / "elastic_qha_md_overlay.csv") if elastic_rows else "",
            "structure_overlay_csv": str(args.outdir / "structure_qha_md_overlay.csv") if structural_rows else "",
            "plots": plot_files,
        },
    }
    write_json(args.outdir / "elastic_qha_md_metadata.json", metadata)

    print(f"QHA structural data: {'yes' if structural_rows else 'no'}")
    print(f"QHA/static elastic data: {'yes' if qha_elastic_rows else 'no'}")
    print(f"MD elastic data: yes ({len(md_components)} components)")
    if not qha_elastic_rows:
        print("QHA readiness: structure-only. Standard phonopy-QHA cannot provide Cij without additional static/quasi-static elastic calculations.")
    else:
        print("QHA readiness: elastic constants available for comparison.")
    print(f"Wrote metadata: {args.outdir / 'elastic_qha_md_metadata.json'}")
    return metadata


if __name__ == "__main__":
    main()
