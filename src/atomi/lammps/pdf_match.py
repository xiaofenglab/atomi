#!/usr/bin/env python3
"""Compare and reweight MD-derived PDF/S(Q)/F(Q) curves against experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from atomi.core.archive import archive_output_dir, default_archive_path
from atomi.lammps import rdf_pdf


AVOGADRO = 6.02214076e23


def write_json(path: Path, data: dict) -> None:
    def normalize(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        return value

    path.write_text(json.dumps(normalize(data), indent=2), encoding="utf-8")


def finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def atomic_mass_g_mol(symbol: str) -> float:
    try:
        from ase.data import atomic_masses, atomic_numbers

        return float(atomic_masses[atomic_numbers[symbol]])
    except Exception:
        fallback = {
            "H": 1.00794,
            "C": 12.0107,
            "N": 14.0067,
            "O": 15.9994,
            "Cl": 35.453,
            "Ga": 69.723,
            "U": 238.02891,
        }
        if symbol not in fallback:
            raise ValueError(f"Atomic mass for {symbol!r} is unavailable; install ASE or pass --density-g-cm3")
        return fallback[symbol]


def parse_formula_counts(formula: str) -> dict[str, float]:
    compact = formula.replace(" ", "")
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", compact)
    if not tokens:
        raise ValueError(f"Could not parse formula/composition {formula!r}")
    consumed = "".join(element + count for element, count in tokens)
    if consumed != compact:
        raise ValueError(f"Formula parser only supports simple formulas such as UO2; got {formula!r}")
    counts: dict[str, float] = {}
    for element, count in tokens:
        counts[element] = counts.get(element, 0.0) + (float(count) if count else 1.0)
    return counts


def formula_mass_g_mol(counts: dict[str, float]) -> float:
    return float(sum(atomic_mass_g_mol(symbol) * count for symbol, count in counts.items()))


def reduced_counts(counts: dict[str, float]) -> dict[str, float]:
    rounded = {symbol: int(round(value)) for symbol, value in counts.items()}
    if rounded and all(abs(counts[symbol] - rounded[symbol]) < 1e-6 and rounded[symbol] > 0 for symbol in counts):
        divisor = 0
        for value in rounded.values():
            divisor = math.gcd(divisor, value)
        if divisor > 0:
            return {symbol: value / divisor for symbol, value in rounded.items()}
    minimum = min(value for value in counts.values() if value > 0)
    return {symbol: value / minimum for symbol, value in counts.items()}


def formula_string_from_counts(counts: dict[str, float]) -> str:
    pieces = []
    for symbol in counts:
        value = counts[symbol]
        if abs(value - round(value)) < 1e-8:
            ivalue = int(round(value))
            pieces.append(symbol if ivalue == 1 else f"{symbol}{ivalue}")
        else:
            pieces.append(f"{symbol}{value:g}")
    return "".join(pieces)


def pdfgetx_composition_from_counts(counts: dict[str, float]) -> str:
    return " ".join(f"{symbol} {counts[symbol]:g}" for symbol in counts)


def read_md_candidate_summary(item: dict) -> dict:
    summary_path = item.get("summary_json")
    if not summary_path:
        return {}
    path = Path(summary_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def infer_md_counts_and_density(series_dir: Path, items: list[dict]) -> dict:
    if not items:
        return {}
    item = items[0]
    summary = read_md_candidate_summary(item)
    counts = summary.get("avg_counts") if isinstance(summary, dict) else None
    if not isinstance(counts, dict):
        counts = {}
    counts = {str(symbol): float(value) for symbol, value in counts.items() if finite_float(value) is not None}
    volume = finite_float(item.get("avg_volume_A3")) or finite_float(summary.get("avg_volume_A3") if isinstance(summary, dict) else None)
    density = None
    if counts and volume and volume > 0:
        mass_g = sum(atomic_mass_g_mol(symbol) * count for symbol, count in counts.items()) / AVOGADRO
        density = mass_g / (volume * 1.0e-24)
    reduced = reduced_counts(counts) if counts else {}
    return {
        "series_dir": str(series_dir.resolve()),
        "temperature": item.get("temperature"),
        "stage_name": item.get("stage_name"),
        "avg_counts": counts,
        "reduced_counts": reduced,
        "formula": formula_string_from_counts(reduced) if reduced else None,
        "pdfgetx_composition": pdfgetx_composition_from_counts(reduced) if reduced else None,
        "avg_volume_A3": volume,
        "density_g_cm3": density,
        "method": "MD selected-window composition and average volume",
    }


def load_api_key_json(path: Path, env_name: str) -> tuple[str | None, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = [
        payload.get(env_name),
        payload.get("materials_project_api_key"),
        payload.get("mp_api_key"),
    ]
    for key in ("materials_project", "materials-project", "mp"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.extend([nested.get("api_key"), nested.get(env_name), nested.get("materials_project_api_key")])
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip(), str(path)
    return None, str(path)


def fetch_materials_project_density(formula: str, api_key_env: str, api_key_json: Path | None) -> dict:
    try:
        from mp_api.client import MPRester  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Materials Project density lookup requires mp-api in the active environment.") from exc
    api_key = os.environ.get(api_key_env)
    api_key_source = f"env:{api_key_env}" if api_key else "none"
    if not api_key and api_key_json is not None:
        api_key, source = load_api_key_json(api_key_json, api_key_env)
        api_key_source = f"json:{source}"
    fields = ["material_id", "formula_pretty", "density"]
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(formula=formula, fields=fields)
    if not docs:
        raise RuntimeError(f"No Materials Project density result for {formula!r}")
    doc = docs[0]
    get = doc.get if isinstance(doc, dict) else lambda key, default=None: getattr(doc, key, default)
    density = finite_float(get("density"))
    if density is None:
        raise RuntimeError(f"Materials Project result for {formula!r} did not include density")
    return {
        "provider": "materials-project",
        "formula": formula,
        "material_id": str(get("material_id", "")),
        "formula_pretty": str(get("formula_pretty", "")),
        "density_g_cm3": density,
        "api_key_source": api_key_source,
    }


def fetch_aflow_density(formula: str, timeout: float) -> dict:
    compound = formula_string_from_counts(parse_formula_counts(formula))
    query = f"compound({compound}),auid,density,$paging(1),$format(json)"
    url = "https://aflow.org/API/aflux/v1.0/?" + urllib.parse.quote(query, safe="(),$:")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload:
        raise RuntimeError(f"No AFLOW density result for compound query {compound!r}")
    datum = payload[0]
    density = finite_float(datum.get("density"))
    if density is None:
        raise RuntimeError("AFLOW result did not include a numeric density field")
    return {
        "provider": "aflow",
        "formula": formula,
        "compound_query": compound,
        "material_id": str(datum.get("auid", "")),
        "density_g_cm3": density,
        "query_url": url,
    }


def load_xy_columns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                x_values.append(float(parts[0]))
                y_values.append(float(parts[1]))
            except ValueError:
                continue
    if not x_values:
        raise ValueError(f"No numeric two-column data found in {path}")
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def write_xy_csv(path: Path, xname: str, x: np.ndarray, columns: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([xname] + list(columns))
        for i, xi in enumerate(x):
            writer.writerow([xi] + [columns[key][i] for key in columns])


def infer_quantity(exp_path: Path, quantity: str) -> str:
    if quantity != "auto":
        return quantity
    suffix = exp_path.suffix.lower()
    if suffix == ".gr":
        return "G"
    if suffix in (".sq", ".sofq"):
        return "SQ"
    if suffix in (".fq", ".fofq"):
        return "FQ"
    if suffix in (".iq",):
        return "iQ"
    return "G"


def ensure_md_series(args: argparse.Namespace, mode: str) -> Path:
    if args.pdf_series is not None:
        series_dir = args.pdf_series.resolve()
        if not (series_dir / "series_summary.json").exists():
            raise FileNotFoundError(f"Missing series_summary.json in {series_dir}")
        return series_dir

    series_dir = args.outdir / f"{mode}_md_pdf_series"
    series_args = SimpleNamespace(
        config=args.config,
        md_root=args.md_root,
        config_dir=args.config_dir,
        config_glob=args.config_glob,
        duplicate_policy=args.duplicate_policy,
        t_min=args.t_min,
        t_max=args.t_max,
        dump_format=args.dump_format,
        type_map=args.type_map,
        dt=args.dt,
        dump_every=args.dump_every,
        window_ps=args.window_ps,
        frame_step=args.frame_step,
        outdir=series_dir,
        rmax=args.rmax,
        dr=args.dr,
        qmax=args.qmax,
        dq=args.dq,
        gr_rmax=args.gr_rmax,
        gr_dr=args.gr_dr,
        scattering=args.scattering,
        weights=args.weights,
        window_function=args.window_function,
        fitting_exports="auto",
        pdfgui_dr_uncertainty=0.0,
        pdfgui_dgr=1.0,
        frame_overlays=False,
        frame_overlay_step=1,
        frame_overlay_max=0,
        adp=False,
        no_plots=True,
        archive_path=None,
        no_archive_output=True,
        write_selected_extxyz=args.write_selected_extxyz,
    )
    rdf_pdf.run_series(series_args)
    return series_dir


def load_series_items(series_dir: Path) -> list[dict]:
    metadata = json.loads((series_dir / "series_summary.json").read_text(encoding="utf-8"))
    series = metadata.get("series", [])
    if not series:
        raise ValueError(f"No series entries found in {series_dir / 'series_summary.json'}")
    return series


def select_series_items(series: list[dict], md_temperature: float | None, tolerance: float) -> list[dict]:
    if md_temperature is None:
        return series
    if not series:
        return []
    best = min(series, key=lambda item: abs(float(item.get("temperature", math.inf)) - md_temperature))
    delta = abs(float(best.get("temperature", math.inf)) - md_temperature)
    if delta > tolerance:
        raise ValueError(
            f"No MD PDF candidate within {tolerance:g} K of requested --md-temperature {md_temperature:g}; "
            f"nearest is {best.get('temperature')} K."
        )
    return [best]


def read_model_curve(item: dict, quantity: str, g_source: str, fq_source: str) -> tuple[np.ndarray, np.ndarray, str]:
    if quantity == "G":
        key = "GofR_from_FQ" if g_source == "from-fq" else "GofR_direct"
        path = Path(item[key])
        x, y = load_xy_columns(path)
        return x, y, key
    if quantity == "SQ":
        path = Path(item["SofQ"])
        x, y = load_xy_columns(path)
        return x, y, "SofQ"
    if quantity == "FQ":
        key = "FofQ_windowed" if fq_source == "windowed" else "FofQ"
        path = Path(item[key])
        x, y = load_xy_columns(path)
        return x, y, key
    if quantity == "iQ":
        path = Path(item["SofQ"])
        x, sq = load_xy_columns(path)
        return x, sq - 1.0, "iQ_from_SofQ"
    raise ValueError(f"Unsupported quantity: {quantity}")


def interpolate_to_exp(
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    x_model: np.ndarray,
    y_model: np.ndarray,
    x_min: Optional[float],
    x_max: Optional[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = max(float(np.min(x_exp)), float(np.min(x_model)), float(x_min) if x_min is not None else -np.inf)
    hi = min(float(np.max(x_exp)), float(np.max(x_model)), float(x_max) if x_max is not None else np.inf)
    mask = (x_exp >= lo - 1e-12) & (x_exp <= hi + 1e-12)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Experiment/model overlap has fewer than three points.")
    x = x_exp[mask]
    exp = y_exp[mask]
    model = np.interp(x, x_model, y_model)
    return x, exp, model


def fit_nuisance(
    x: np.ndarray,
    y_exp: np.ndarray,
    y_model: np.ndarray,
    fit_scale: bool,
    baseline_order: int,
) -> tuple[np.ndarray, dict]:
    columns = []
    names = []
    fixed = np.zeros_like(y_exp)
    if fit_scale:
        columns.append(y_model)
        names.append("scale")
    else:
        fixed = y_model.copy()

    x_center = x - float(np.mean(x))
    if baseline_order >= 0:
        columns.append(np.ones_like(x))
        names.append("offset")
    if baseline_order >= 1:
        columns.append(x_center)
        names.append("slope_x_centered")
    if baseline_order >= 2:
        columns.append(x_center**2)
        names.append("quadratic_x_centered")

    if columns:
        design = np.column_stack(columns)
        coeff, *_ = np.linalg.lstsq(design, y_exp - fixed, rcond=None)
        corrected = fixed + design @ coeff
    else:
        coeff = np.asarray([], dtype=float)
        corrected = fixed

    params = {"fit_scale": fit_scale, "baseline_order": baseline_order}
    if not fit_scale:
        params["scale"] = 1.0
    for name, value in zip(names, coeff):
        params[name] = float(value)
    return corrected, params


def metrics(y_exp: np.ndarray, y_model: np.ndarray) -> dict:
    residual = y_model - y_exp
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sqrt(np.mean(y_exp**2)))
    rel_rmse = rmse / denom if denom > 0 else math.nan
    return {
        "rmse": rmse,
        "mae": mae,
        "relative_rmse": rel_rmse,
        "max_abs": float(np.max(np.abs(residual))),
        "n_points": int(len(y_exp)),
    }


def plot_compare(path: Path, x: np.ndarray, y_exp: np.ndarray, y_model: np.ndarray, title: str, xlabel: str, ylabel: str) -> bool:
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

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    marker_stride = max(1, len(x) // 220)
    axes[0].plot(
        x[::marker_stride],
        y_exp[::marker_stride],
        linestyle="none",
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.9,
        label="experiment",
    )
    axes[0].plot(x, y_model, color="#d62728", linewidth=1.4, label="MD model")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].legend(frameon=False)
    axes[1].plot(x, y_model - y_exp, color="#d62728", linewidth=1.2)
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("residual")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def plot_reweight(path: Path, x: np.ndarray, exp: np.ndarray, prior: np.ndarray, weighted: np.ndarray, corrected: np.ndarray, xlabel: str, ylabel: str) -> bool:
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

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(x, exp, color="black", linewidth=1.8, label="experiment")
    axes[0].plot(x, prior, color="0.6", linestyle="--", linewidth=1.2, label="uniform MD")
    axes[0].plot(x, weighted, color="#1f77b4", linewidth=1.4, label="reweighted MD")
    axes[0].plot(x, corrected, color="#d62728", linewidth=1.3, label="corrected reweighted")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    axes[1].plot(x, corrected - exp, color="#d62728", linewidth=1.2)
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("residual")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def quantity_labels(quantity: str) -> tuple[str, str]:
    if quantity == "G":
        return "r (A)", "G(r)"
    if quantity in ("SQ", "FQ", "iQ"):
        return "Q (A^-1)", quantity
    return "x", quantity


def quantity_extension(quantity: str) -> str:
    if quantity == "G":
        return "gr"
    if quantity == "SQ":
        return "sq"
    if quantity == "FQ":
        return "fq"
    if quantity == "iQ":
        return "iq"
    return "gr"


def raw_quantity(args: argparse.Namespace) -> str:
    return "G" if args.quantity == "auto" else args.quantity


def resolve_pdfgetx_composition_density(args: argparse.Namespace, series_dir: Path) -> dict:
    items = select_series_items(
        load_series_items(series_dir),
        getattr(args, "md_temperature", None),
        getattr(args, "md_temperature_tolerance", 1e-6),
    )
    md_info = infer_md_counts_and_density(series_dir, items)
    composition = args.composition
    composition_source = "user"
    formula = args.material_formula
    if not composition:
        composition = md_info.get("pdfgetx_composition")
        composition_source = "md-box"
    if not formula:
        formula = md_info.get("formula")

    api_checks = []
    if args.density_api_check in ("materials-project", "both") and formula:
        try:
            api_checks.append(fetch_materials_project_density(formula, args.materials_project_api_key_env, args.api_key_json))
        except Exception as exc:
            api_checks.append({"provider": "materials-project", "formula": formula, "error": str(exc)})
    if args.density_api_check in ("aflow", "both") and formula:
        try:
            api_checks.append(fetch_aflow_density(formula, args.api_timeout))
        except Exception as exc:
            api_checks.append({"provider": "aflow", "formula": formula, "error": str(exc)})

    density = None
    density_source = args.density_source
    if density_source == "auto":
        density_source = "user" if args.density_g_cm3 is not None else "md"
    if density_source == "user":
        density = args.density_g_cm3
    elif density_source == "md":
        density = md_info.get("density_g_cm3")
    elif density_source == "materials-project":
        density = fetch_materials_project_density(formula, args.materials_project_api_key_env, args.api_key_json)["density_g_cm3"] if formula else None
    elif density_source == "aflow":
        density = fetch_aflow_density(formula, args.api_timeout)["density_g_cm3"] if formula else None

    if not composition:
        raise ValueError("Could not infer PDFgetX composition from MD; pass --composition or --material-formula.")
    if density is None:
        raise ValueError("Could not determine PDFgetX density; pass --density-g-cm3 or use --density-source md with MD volume/counts.")
    if density <= 0:
        raise ValueError("--density-g-cm3/determined density must be positive")

    return {
        "composition": composition,
        "composition_source": composition_source,
        "formula": formula,
        "density_g_cm3": float(density),
        "density_source": density_source,
        "md_density": md_info,
        "api_density_checks": api_checks,
    }


def write_pdfgetx3_config(path: Path, args: argparse.Namespace, prep: dict, quantity: str) -> dict:
    exp_dir = path.parent
    qmax = args.pdfgetx_qmax if args.pdfgetx_qmax is not None else args.qmax
    qmin = args.pdfgetx_qmin if args.pdfgetx_qmin is not None else 0.0
    rmax = args.pdfgetx_rmax if args.pdfgetx_rmax is not None else (args.gr_rmax if args.gr_rmax is not None else args.rmax)
    rmin = args.pdfgetx_rmin if args.pdfgetx_rmin is not None else (args.x_min if args.x_min is not None and quantity == "G" else 0.5)
    rstep = args.pdfgetx_rstep if args.pdfgetx_rstep is not None else (args.gr_dr if args.gr_dr is not None else args.dr)
    qmaxinst = args.pdfgetx_qmaxinst if args.pdfgetx_qmaxinst is not None else qmax
    entries = {
        "dataformat": args.pdfgetx_dataformat,
        "datafile": str(args.exp_raw_sample.resolve()),
        "outputtype": "iq sq fq gr",
        "composition": prep["composition"],
        "density": f"{prep['density_g_cm3']:.10g}",
        "qmaxinst": f"{qmaxinst:.10g}",
        "qmin": f"{qmin:.10g}",
        "qmax": f"{qmax:.10g}",
        "rmin": f"{rmin:.10g}",
        "rmax": f"{rmax:.10g}",
        "rstep": f"{rstep:.10g}",
    }
    if args.wavelength is not None:
        entries["wavelength"] = f"{args.wavelength:.10g}"
    if args.exp_raw_empty is not None:
        entries["containerfile"] = str(args.exp_raw_empty.resolve())
    if args.exp_raw_background is not None:
        entries["backgroundfile"] = str(args.exp_raw_background.resolve())
    if args.exp_raw_reference is not None:
        entries["referencefile"] = str(args.exp_raw_reference.resolve())
    lines = [
        "# Generated by Atomi pdf_md_compare for pdfgetx3.",
        "# Inspect/cite PDFgetX3 settings before publication.",
        "[DEFAULT]",
        "",
    ]
    for key, value in entries.items():
        lines.append(f"{key} = {value}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    stem = args.exp_raw_sample.stem
    expected = exp_dir / f"{stem}.{quantity_extension(quantity)}"
    return {
        "config": str(path),
        "expected_output": str(expected),
        "pdfgetx3_parameters": entries,
        "q_range_A^-1": {"qmin": qmin, "qmax": qmax, "qmaxinst": qmaxinst},
        "r_range_A": {"rmin": rmin, "rmax": rmax, "rstep": rstep},
    }


def write_pdfgetx3_run_script(path: Path, pdfgetx3: str, cfg: Path) -> None:
    command = [pdfgetx3, "-c", cfg.name]
    text = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(cfg.parent.resolve()))}",
            " ".join(shlex.quote(part) for part in command),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def prepare_experiment_input(args: argparse.Namespace, series_dir: Path, quantity: str) -> tuple[Path, dict]:
    if args.exp is not None:
        return args.exp, {"mode": "reduced-data", "input": str(args.exp.resolve())}
    if args.exp_raw_sample is None:
        raise ValueError("Pass --exp for reduced data or --exp-raw-sample for raw .chi data.")
    if args.pdfgetx_dataformat == "twotheta" and args.wavelength is None:
        raise ValueError("--wavelength is required when --pdfgetx-dataformat twotheta")

    exp_dir = args.outdir / "pdfgetx3_exp"
    exp_dir.mkdir(parents=True, exist_ok=True)
    prep = resolve_pdfgetx_composition_density(args, series_dir)
    cfg = exp_dir / args.pdfgetx_cfg_name
    pdfgetx_info = write_pdfgetx3_config(cfg, args, prep, quantity)
    run_script = exp_dir / "run_pdfgetx3.sh"
    write_pdfgetx3_run_script(run_script, args.pdfgetx3, cfg)
    expected = Path(pdfgetx_info["expected_output"])
    run_result: dict[str, object] = {"run": False}
    if args.run_pdfgetx3:
        proc = subprocess.run([args.pdfgetx3, "-c", cfg.name], cwd=exp_dir, text=True, capture_output=True)
        run_result = {
            "run": True,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode != 0:
            raise RuntimeError(f"pdfgetx3 failed; see {run_script} and pdfgetx3 metadata in {exp_dir}")
    output_ready = expected.exists()
    if not output_ready and getattr(args, "prepare_pdfgetx3_only", False):
        pass
    elif not output_ready:
        raise FileNotFoundError(
            f"Expected PDFgetX3 output not found: {expected}. "
            f"Run {run_script} on the HPC, or pass --run-pdfgetx3 when pdfgetx3 is available."
        )
    info = {
        "mode": "raw-chi-pdfgetx3",
        "raw_sample": str(args.exp_raw_sample.resolve()),
        "raw_empty": str(args.exp_raw_empty.resolve()) if args.exp_raw_empty else None,
        "raw_background": str(args.exp_raw_background.resolve()) if args.exp_raw_background else None,
        "raw_reference": str(args.exp_raw_reference.resolve()) if args.exp_raw_reference else None,
        "run_script": str(run_script),
        "prep": prep,
        "pdfgetx3": pdfgetx_info,
        "run_result": run_result,
        "output_ready": output_ready,
    }
    write_json(exp_dir / "pdfgetx3_prep_metadata.json", info)
    return expected, info


def compare_candidates(
    exp_path: Path,
    series_items: list[dict],
    quantity: str,
    g_source: str,
    fq_source: str,
    x_min: Optional[float],
    x_max: Optional[float],
    fit_scale: bool,
    baseline_order: int,
) -> tuple[list[dict], dict]:
    x_exp, y_exp = load_xy_columns(exp_path)
    rows: list[dict] = []
    best_payload = None
    for item in series_items:
        x_model, y_model, model_key = read_model_curve(item, quantity, g_source, fq_source)
        x, exp, model = interpolate_to_exp(x_exp, y_exp, x_model, y_model, x_min, x_max)
        corrected, params = fit_nuisance(x, exp, model, fit_scale=fit_scale, baseline_order=baseline_order)
        raw_metrics = metrics(exp, model)
        corrected_metrics = metrics(exp, corrected)
        row = {
            "temperature": float(item["temperature"]),
            "stage_name": item.get("stage_name", ""),
            "model_key": model_key,
            "raw_rmse": raw_metrics["rmse"],
            "corrected_rmse": corrected_metrics["rmse"],
            "corrected_relative_rmse": corrected_metrics["relative_rmse"],
            "corrected_mae": corrected_metrics["mae"],
            "n_points": corrected_metrics["n_points"],
            "nuisance": params,
        }
        rows.append(row)
        if best_payload is None or row["corrected_rmse"] < best_payload["row"]["corrected_rmse"]:
            best_payload = {"row": row, "x": x, "exp": exp, "model": model, "corrected": corrected}

    rows.sort(key=lambda r: r["corrected_rmse"])
    if best_payload is None:
        raise ValueError("No MD candidates could be compared.")
    return rows, best_payload


def write_compare_outputs(
    args: argparse.Namespace,
    series_dir: Path,
    quantity: str,
    experiment_info: dict | None = None,
) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    series_items = select_series_items(
        load_series_items(series_dir),
        getattr(args, "md_temperature", None),
        getattr(args, "md_temperature_tolerance", 1e-6),
    )
    rows, best = compare_candidates(
        exp_path=args.exp,
        series_items=series_items,
        quantity=quantity,
        g_source=args.g_source,
        fq_source=args.fq_source,
        x_min=args.x_min,
        x_max=args.x_max,
        fit_scale=args.fit_scale,
        baseline_order=args.baseline_order,
    )

    with (args.outdir / "compare_rank.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "rank",
            "temperature",
            "stage_name",
            "model_key",
            "raw_rmse",
            "corrected_rmse",
            "corrected_relative_rmse",
            "corrected_mae",
            "n_points",
            "nuisance_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({**{key: row.get(key, "") for key in fieldnames if key not in ("rank", "nuisance_json")}, "rank": rank, "nuisance_json": json.dumps(row["nuisance"], sort_keys=True)})

    write_xy_csv(
        args.outdir / "best_compare_curve.csv",
        "x",
        best["x"],
        {
            "experiment": best["exp"],
            "md_raw": best["model"],
            "md_corrected": best["corrected"],
            "residual_corrected": best["corrected"] - best["exp"],
        },
    )
    xlabel, ylabel = quantity_labels(quantity)
    plots = []
    plot_path = args.outdir / "best_compare_overlay.png"
    if plot_compare(
        plot_path,
        best["x"],
        best["exp"],
        best["corrected"],
        f"Best MD-PDF Match ({quantity})",
        xlabel,
        ylabel,
    ):
        plots.append(str(plot_path))

    metadata = {
        "mode": "compare",
        "experiment": str(args.exp.resolve()),
        "experiment_info": experiment_info or {"mode": "reduced-data", "input": str(args.exp.resolve())},
        "quantity": quantity,
        "series_dir": str(series_dir.resolve()),
        "best": rows[0],
        "n_candidates": len(rows),
        "md_temperature_filter": {
            "md_temperature": getattr(args, "md_temperature", None),
            "md_temperature_tolerance": getattr(args, "md_temperature_tolerance", None),
        },
        "x_range": {"x_min": args.x_min, "x_max": args.x_max},
        "fit_scale": args.fit_scale,
        "baseline_order": args.baseline_order,
        "plots": plots,
        "archive": str(args.archive_path.resolve() if args.archive_path else default_archive_path(args.outdir).resolve())
        if not args.no_archive_output
        else None,
    }
    write_json(args.outdir / "compare_metadata.json", metadata)
    if not args.no_archive_output:
        archive = archive_output_dir(args.outdir, args.archive_path)
        metadata["archive"] = str(archive)
        write_json(args.outdir / "compare_metadata.json", metadata)
    return metadata


def softmax_weights(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp_values = np.exp(shifted)
    return exp_values / float(np.sum(exp_values))


def reweight_curves(curves: np.ndarray, exp: np.ndarray, kl_strength: float, max_iter: int, learning_rate: float) -> tuple[np.ndarray, dict]:
    n_models = curves.shape[0]
    prior = np.full(n_models, 1.0 / n_models)
    logits = np.zeros(n_models, dtype=float)
    last_loss = math.inf
    for _ in range(max_iter):
        weights = softmax_weights(logits)
        pred = weights @ curves
        residual = pred - exp
        grad_w = 2.0 * (curves @ residual) / len(exp)
        grad_w += kl_strength * (np.log(np.maximum(weights, 1e-300) / prior) + 1.0)
        grad_logits = weights * (grad_w - float(weights @ grad_w))
        proposal = logits - learning_rate * grad_logits
        new_weights = softmax_weights(proposal)
        new_pred = new_weights @ curves
        mse = float(np.mean((new_pred - exp) ** 2))
        kl = float(np.sum(new_weights * np.log(np.maximum(new_weights, 1e-300) / prior)))
        loss = mse + kl_strength * kl
        if loss <= last_loss or learning_rate < 1e-7:
            logits = proposal
            last_loss = loss
        else:
            learning_rate *= 0.5

    weights = softmax_weights(logits)
    kl = float(np.sum(weights * np.log(np.maximum(weights, 1e-300) / prior)))
    neff_inverse = float(1.0 / np.sum(weights**2))
    neff_entropy = float(np.exp(-np.sum(weights * np.log(np.maximum(weights, 1e-300)))))
    return weights, {"kl_divergence": kl, "neff_inverse_participation": neff_inverse, "neff_entropy": neff_entropy, "final_loss": last_loss}


def write_reweight_outputs(
    args: argparse.Namespace,
    series_dir: Path,
    quantity: str,
    experiment_info: dict | None = None,
) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    x_exp, y_exp = load_xy_columns(args.exp)
    series = select_series_items(
        load_series_items(series_dir),
        getattr(args, "md_temperature", None),
        getattr(args, "md_temperature_tolerance", 1e-6),
    )
    curves = []
    rows = []
    x_ref = None
    exp_ref = None
    for item in series:
        x_model, y_model, model_key = read_model_curve(item, quantity, args.g_source, args.fq_source)
        x, exp, model = interpolate_to_exp(x_exp, y_exp, x_model, y_model, args.x_min, args.x_max)
        if x_ref is None:
            x_ref = x
            exp_ref = exp
        else:
            if len(x) != len(x_ref) or np.max(np.abs(x - x_ref)) > 1e-9:
                model = np.interp(x_ref, x, model)
        curves.append(model)
        rows.append({"temperature": float(item["temperature"]), "stage_name": item.get("stage_name", ""), "model_key": model_key})

    if x_ref is None or exp_ref is None:
        raise ValueError("No MD candidates available for reweighting.")
    curve_matrix = np.asarray(curves, dtype=float)
    uniform_weights = np.full(len(curves), 1.0 / len(curves))
    uniform_curve = uniform_weights @ curve_matrix
    weights, diag = reweight_curves(curve_matrix, exp_ref, args.kl_strength, args.max_iter, args.learning_rate)
    weighted_curve = weights @ curve_matrix
    corrected_curve, nuisance = fit_nuisance(
        x_ref,
        exp_ref,
        weighted_curve,
        fit_scale=args.fit_scale,
        baseline_order=args.baseline_order,
    )

    with (args.outdir / "frame_window_weights.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["temperature", "stage_name", "model_key", "weight", "prior_weight"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, weight, prior in zip(rows, weights, uniform_weights):
            writer.writerow({**row, "weight": weight, "prior_weight": prior})

    write_xy_csv(
        args.outdir / "reweighted_curve.csv",
        "x",
        x_ref,
        {
            "experiment": exp_ref,
            "uniform_md": uniform_curve,
            "reweighted_md": weighted_curve,
            "corrected_reweighted_md": corrected_curve,
            "residual_corrected": corrected_curve - exp_ref,
        },
    )

    xlabel, ylabel = quantity_labels(quantity)
    plots = []
    plot_path = args.outdir / "reweighted_overlay.png"
    if plot_reweight(plot_path, x_ref, exp_ref, uniform_curve, weighted_curve, corrected_curve, xlabel, ylabel):
        plots.append(str(plot_path))

    metadata = {
        "mode": "reweight",
        "experiment": str(args.exp.resolve()),
        "experiment_info": experiment_info or {"mode": "reduced-data", "input": str(args.exp.resolve())},
        "quantity": quantity,
        "series_dir": str(series_dir.resolve()),
        "n_candidates": len(rows),
        "md_temperature_filter": {
            "md_temperature": getattr(args, "md_temperature", None),
            "md_temperature_tolerance": getattr(args, "md_temperature_tolerance", None),
        },
        "kl_strength": args.kl_strength,
        "diagnostics": diag,
        "uniform_metrics": metrics(exp_ref, uniform_curve),
        "reweighted_metrics": metrics(exp_ref, weighted_curve),
        "corrected_reweighted_metrics": metrics(exp_ref, corrected_curve),
        "nuisance": nuisance,
        "plots": plots,
        "archive": str(args.archive_path.resolve() if args.archive_path else default_archive_path(args.outdir).resolve())
        if not args.no_archive_output
        else None,
    }
    write_json(args.outdir / "reweight_metadata.json", metadata)
    if not args.no_archive_output:
        archive = archive_output_dir(args.outdir, args.archive_path)
        metadata["archive"] = str(archive)
        write_json(args.outdir / "reweight_metadata.json", metadata)
    return metadata


def add_common_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf-series", type=Path, help="Existing pdf_lammps_series output directory.")
    source.add_argument("--config", nargs="+", help="One or more production config JSON files.")
    source.add_argument("--md-root", type=Path, help="MD engine root. NPT folders are scanned; NVT folders are ignored.")
    exp = parser.add_argument_group("experimental input")
    exp_source = exp.add_mutually_exclusive_group(required=True)
    exp_source.add_argument("--exp", type=Path, help="Reduced experimental PDFgetX/PDFgui/RMC-style two-column data.")
    exp_source.add_argument("--exp-raw-sample", type=Path, help="Raw experimental sample .chi file for pdfgetx3 reduction.")
    exp.add_argument("--exp-raw-empty", type=Path, help="Optional empty container/cell .chi file for pdfgetx3.")
    exp.add_argument("--exp-raw-background", type=Path, help="Optional air/instrument background .chi file for pdfgetx3.")
    exp.add_argument("--exp-raw-reference", type=Path, help="Optional reference .chi file for pdfgetx3.")
    exp.add_argument("--pdfgetx3", default="pdfgetx3", help="pdfgetx3 executable name/path.")
    exp.add_argument("--run-pdfgetx3", action="store_true", help="Run pdfgetx3 after writing the generated config.")
    exp.add_argument("--prepare-pdfgetx3-only", action="store_true", help="Write pdfgetx3 config/run script and exit before comparison.")
    exp.add_argument("--pdfgetx-cfg-name", default="pdfgetx3.cfg")
    exp.add_argument("--pdfgetx-dataformat", choices=["twotheta", "QA", "Qnm"], default="QA")
    exp.add_argument("--wavelength", type=float, help="X-ray wavelength in Angstroms; required for twotheta input.")
    exp.add_argument("--composition", help="PDFgetX3 composition string, e.g. 'U 1 O 2'. Defaults to MD-inferred composition.")
    exp.add_argument("--material-formula", help="Simple formula for API density lookup/cross-checks, e.g. UO2.")
    exp.add_argument("--density-source", choices=["auto", "md", "user", "materials-project", "aflow"], default="auto")
    exp.add_argument("--density-g-cm3", type=float, help="User override density for PDFgetX3.")
    exp.add_argument("--density-api-check", choices=["none", "materials-project", "aflow", "both"], default="none")
    exp.add_argument("--materials-project-api-key-env", default="MP_API_KEY")
    exp.add_argument("--api-key-json", type=Path, help="Optional local JSON containing a Materials Project API key.")
    exp.add_argument("--api-timeout", type=float, default=20.0)
    exp.add_argument("--pdfgetx-qmin", type=float)
    exp.add_argument("--pdfgetx-qmax", type=float)
    exp.add_argument("--pdfgetx-qmaxinst", type=float)
    exp.add_argument("--pdfgetx-rmin", type=float)
    exp.add_argument("--pdfgetx-rmax", type=float)
    exp.add_argument("--pdfgetx-rstep", type=float)
    parser.add_argument("--quantity", choices=["auto", "G", "SQ", "FQ", "iQ"], default="auto")
    parser.add_argument("--g-source", choices=["from-fq", "direct"], default="from-fq")
    parser.add_argument("--fq-source", choices=["raw", "windowed"], default="raw")
    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--fit-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-order", type=int, choices=[-1, 0, 1, 2], default=0)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--config-glob", default="*.json")
    parser.add_argument("--duplicate-policy", choices=["highest_config_order", "first", "error"], default="highest_config_order")
    parser.add_argument("--dump-format", default="lammps-dump-text")
    parser.add_argument("--type-map", nargs="*", default=[])
    parser.add_argument("--dt", type=float)
    parser.add_argument("--dump-every", type=int)
    parser.add_argument("--window-ps", type=float, default=20.0)
    parser.add_argument("--md-temperature", type=float, help="Restrict comparison/reweighting to the nearest MD temperature.")
    parser.add_argument("--md-temperature-tolerance", type=float, default=1e-6)
    parser.add_argument("--t-min", type=float)
    parser.add_argument("--t-max", type=float)
    parser.add_argument("--frame-step", type=int)
    parser.add_argument("--rmax", type=float, default=12.0)
    parser.add_argument("--dr", type=float, default=0.02)
    parser.add_argument("--qmax", type=float, default=25.0)
    parser.add_argument("--dq", type=float, default=0.05)
    parser.add_argument("--gr-rmax", type=float)
    parser.add_argument("--gr-dr", type=float)
    parser.add_argument("--scattering", choices=("xray", "neutron", "custom"), default="xray")
    parser.add_argument("--weights", nargs="*", default=[])
    parser.add_argument("--window-function", choices=("lorch", "none"), default="lorch")
    parser.add_argument("--write-selected-extxyz", action="store_true")
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--no-archive-output", action="store_true")


def build_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_md_compare",
        description="Rank MD-derived PDF/S(Q)/F(Q) windows against experimental PDFgetX-style data.",
    )
    add_common_args(parser)
    return parser


def build_reweight_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_md_reweight",
        description="Maximum-entropy-style reweighting of MD PDF/S(Q)/F(Q) windows against experiment.",
    )
    add_common_args(parser)
    parser.add_argument("--kl-strength", type=float, default=1e-3, help="Penalty for moving away from uniform MD weights.")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    if (args.config or args.md_root) and not args.type_map:
        parser.error("--type-map is required when generating MD PDFs from --config or --md-root")
    if args.baseline_order < -1:
        parser.error("--baseline-order must be -1, 0, 1, or 2")
    exp_path = getattr(args, "exp", None)
    raw_sample = getattr(args, "exp_raw_sample", None)
    if exp_path is None and raw_sample is None:
        parser.error("pass --exp for reduced data or --exp-raw-sample for raw .chi data")
    if exp_path is not None:
        return infer_quantity(exp_path, args.quantity)
    return raw_quantity(args)


def compare_main(argv: Optional[list[str]] = None) -> None:
    parser = build_compare_parser()
    args = parser.parse_args(argv)
    quantity = validate_args(parser, args)
    series_dir = ensure_md_series(args, "compare")
    exp_path, experiment_info = prepare_experiment_input(args, series_dir, quantity)
    if getattr(args, "prepare_pdfgetx3_only", False):
        print(f"Wrote pdfgetx3 prep metadata: {args.outdir / 'pdfgetx3_exp' / 'pdfgetx3_prep_metadata.json'}")
        print(f"Run script: {experiment_info.get('run_script')}")
        print(f"Expected reduced data: {experiment_info.get('pdfgetx3', {}).get('expected_output')}")
        return
    args.exp = exp_path
    summary = write_compare_outputs(args, series_dir, quantity, experiment_info=experiment_info)
    best = summary["best"]
    print(f"Compared {summary['n_candidates']} MD candidates against {args.exp}")
    print(f"Best: T={best['temperature']:g} K, RMSE={best['corrected_rmse']:.6g}")
    print(f"Wrote compare outputs to: {args.outdir.resolve()}")
    if summary.get("archive"):
        print(f"Download archive written to: {summary['archive']}")


def reweight_main(argv: Optional[list[str]] = None) -> None:
    parser = build_reweight_parser()
    args = parser.parse_args(argv)
    quantity = validate_args(parser, args)
    series_dir = ensure_md_series(args, "reweight")
    exp_path, experiment_info = prepare_experiment_input(args, series_dir, quantity)
    if getattr(args, "prepare_pdfgetx3_only", False):
        print(f"Wrote pdfgetx3 prep metadata: {args.outdir / 'pdfgetx3_exp' / 'pdfgetx3_prep_metadata.json'}")
        print(f"Run script: {experiment_info.get('run_script')}")
        print(f"Expected reduced data: {experiment_info.get('pdfgetx3', {}).get('expected_output')}")
        return
    args.exp = exp_path
    summary = write_reweight_outputs(args, series_dir, quantity, experiment_info=experiment_info)
    diag = summary["diagnostics"]
    print(f"Reweighted {summary['n_candidates']} MD candidates against {args.exp}")
    print(f"Corrected RMSE: {summary['corrected_reweighted_metrics']['rmse']:.6g}")
    print(f"Effective frames/windows: {diag['neff_inverse_participation']:.3g}")
    print(f"Wrote reweight outputs to: {args.outdir.resolve()}")
    if summary.get("archive"):
        print(f"Download archive written to: {summary['archive']}")


def main(argv: Optional[list[str]] = None) -> None:
    compare_main(argv)


if __name__ == "__main__":
    main()
