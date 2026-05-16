#!/usr/bin/env python3
"""Compare and reweight MD-derived PDF/S(Q)/F(Q) curves against experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from atomi.core.archive import archive_output_dir, default_archive_path
from atomi.lammps import rdf_pdf


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
    axes[0].plot(x, y_exp, color="black", linewidth=1.8, label="experiment")
    axes[0].plot(x, y_model, color="#1f77b4", linewidth=1.5, label="MD model")
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


def compare_candidates(
    exp_path: Path,
    series_dir: Path,
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
    for item in load_series_items(series_dir):
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


def write_compare_outputs(args: argparse.Namespace, series_dir: Path, quantity: str) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows, best = compare_candidates(
        exp_path=args.exp,
        series_dir=series_dir,
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
        "quantity": quantity,
        "series_dir": str(series_dir.resolve()),
        "best": rows[0],
        "n_candidates": len(rows),
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


def write_reweight_outputs(args: argparse.Namespace, series_dir: Path, quantity: str) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    x_exp, y_exp = load_xy_columns(args.exp)
    series = load_series_items(series_dir)
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
        "quantity": quantity,
        "series_dir": str(series_dir.resolve()),
        "n_candidates": len(rows),
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
    parser.add_argument("--exp", type=Path, required=True, help="Experimental PDFgetX/PDFgui/RMC-style two-column data.")
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
    return infer_quantity(args.exp, args.quantity)


def compare_main(argv: Optional[list[str]] = None) -> None:
    parser = build_compare_parser()
    args = parser.parse_args(argv)
    quantity = validate_args(parser, args)
    series_dir = ensure_md_series(args, "compare")
    summary = write_compare_outputs(args, series_dir, quantity)
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
    summary = write_reweight_outputs(args, series_dir, quantity)
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
