#!/usr/bin/env python3
"""Manual PDF/RDF and powder-XRD inspection for CIF/POSCAR structures.

Run with the main Atomi environment, for example:

  ~/m_lammps_env/bin/python scripts/manual_static_pdf_xrd.py \
    --structure KCl=KCl.POSCAR --structure UCl3=UCl3.cif \
    --species-order K,Cl,U --outdir /tmp/static_xrd_pdf

The XRD path uses Atomi's phase-order guard backend. The PDF/RDF path is a
small PBC pair-distance calculator intended for quick manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def _import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "This script needs numpy. Run it with ~/m_lammps_env/bin/python or another Atomi environment."
        ) from exc
    return np


def _import_atomi_backend():
    try:
        from atomi.md.phase_order_guard import (  # type: ignore
            diffraction_form_factors,
            frame_from_structure_file,
            simulated_powder_xrd_from_frames,
            write_xrd_multi_overlay_plot,
        )
    except Exception as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "Could not import atomi.md.phase_order_guard. Run with ~/m_lammps_env/bin/python."
        ) from exc
    return {
        "diffraction_form_factors": diffraction_form_factors,
        "frame_from_structure_file": frame_from_structure_file,
        "simulated_powder_xrd_from_frames": simulated_powder_xrd_from_frames,
        "write_xrd_multi_overlay_plot": write_xrd_multi_overlay_plot,
    }


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
    elif ":" in value and not value.startswith("/"):
        label, path = value.split(":", 1)
    else:
        p = Path(value)
        label, path = p.stem, value
    return label.strip(), Path(path).expanduser()


def species_order_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def frame_cartesian(frame: dict[str, Any]):
    np = _import_numpy()
    frac = np.asarray(frame["frac"], dtype=float)
    cell = np.asarray(frame["cell"], dtype=float)
    return frac @ cell


def pbc_delta(frac_a, frac_b):
    np = _import_numpy()
    d = np.asarray(frac_b, dtype=float) - np.asarray(frac_a, dtype=float)
    return d - np.rint(d)


def compute_pdf_curves(
    frame: dict[str, Any],
    *,
    rmax: float,
    dr: float,
    partials: bool,
) -> dict[str, tuple[Any, Any]]:
    np = _import_numpy()
    symbols = list(frame["symbols"])
    frac = np.asarray(frame["frac"], dtype=float)
    cell = np.asarray(frame["cell"], dtype=float)
    volume = abs(float(np.linalg.det(cell)))
    natoms = len(symbols)
    r = np.arange(0.0, rmax + dr, dr)
    centers = 0.5 * (r[:-1] + r[1:])
    shell = 4.0 * math.pi * centers * centers * dr

    hist_total = np.zeros(len(centers), dtype=float)
    hist_partials: dict[str, Any] = {}
    for i in range(natoms):
        for j in range(i + 1, natoms):
            d_frac = pbc_delta(frac[i], frac[j])
            dist = float(np.linalg.norm(d_frac @ cell))
            if dist >= rmax:
                continue
            bin_idx = int(dist / dr)
            hist_total[bin_idx] += 2.0
            if partials:
                pair = "-".join(sorted((symbols[i], symbols[j])))
                hist_partials.setdefault(pair, np.zeros(len(centers), dtype=float))[bin_idx] += 2.0

    number_density = natoms / volume if volume > 0 else 1.0
    denom = natoms * number_density * shell
    denom[denom == 0.0] = 1.0
    curves = {"total_g_r": (centers, hist_total / denom)}
    if partials:
        for pair, hist in sorted(hist_partials.items()):
            curves[f"{pair}_g_r"] = (centers, hist / denom)
    return curves


def normalize_curve(y):
    np = _import_numpy()
    arr = np.asarray(y, dtype=float)
    arr = arr - float(np.min(arr))
    maxv = float(np.max(arr))
    return arr / maxv if maxv > 0 else arr


def write_curves_csv(path: Path, x, curves: dict[str, Any], x_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([x_label, *curves.keys()])
        for i, xv in enumerate(x):
            writer.writerow([float(xv), *[float(curves[name][i]) for name in curves]])


def write_overlay_svg(
    path: Path,
    *,
    title: str,
    x,
    curves: dict[str, Any],
    x_label: str,
    y_label: str,
) -> None:
    np = _import_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = [
        "#1f77b4",
        "#d55e00",
        "#009e73",
        "#cc79a7",
        "#7f7f7f",
        "#9467bd",
        "#8c564b",
        "#e69f00",
    ]
    x = np.asarray(x, dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x0, y0, width, height = 82.0, 54.0, 670.0, 355.0

    def polyline(y):
        y = normalize_curve(y)
        pts = []
        for xi, yi in zip(x, y):
            px = x0 + (float(xi) - x_min) / max(x_max - x_min, 1.0e-12) * width
            py = y0 + height - float(yi) * height
            pts.append(f"{px:.2f},{py:.2f}")
        return " ".join(pts)

    curve_markup = []
    legend_markup = []
    for idx, (label, y) in enumerate(curves.items()):
        color = colors[idx % len(colors)]
        curve_markup.append(
            f'<polyline points="{polyline(y)}" fill="none" stroke="{color}" stroke-width="2.0"/>'
        )
        ly = 28 + idx * 17
        legend_markup.append(f'<line x1="535" y1="{ly}" x2="570" y2="{ly}" stroke="{color}" stroke-width="2.2"/>')
        legend_markup.append(f'<text x="578" y="{ly + 4}" font-family="Arial" font-size="12">{label}</text>')

    ticks = np.linspace(x_min, x_max, 5)
    tick_markup = []
    for tick in ticks:
        px = x0 + (float(tick) - x_min) / max(x_max - x_min, 1.0e-12) * width
        tick_markup.append(
            f'<line x1="{px:.2f}" y1="{y0 + height:.2f}" x2="{px:.2f}" y2="{y0 + height + 5:.2f}" stroke="#444"/>'
        )
        tick_markup.append(
            f'<text x="{px:.2f}" y="{y0 + height + 22:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{tick:.1f}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="820" height="480" viewBox="0 0 820 480">
  <rect width="820" height="480" fill="white"/>
  <text x="410" y="30" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
  <rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="none" stroke="#222" stroke-width="1.2"/>
  <g opacity="0.22">
    <line x1="{x0}" y1="{y0 + height * 0.25:.2f}" x2="{x0 + width}" y2="{y0 + height * 0.25:.2f}" stroke="#999"/>
    <line x1="{x0}" y1="{y0 + height * 0.50:.2f}" x2="{x0 + width}" y2="{y0 + height * 0.50:.2f}" stroke="#999"/>
    <line x1="{x0}" y1="{y0 + height * 0.75:.2f}" x2="{x0 + width}" y2="{y0 + height * 0.75:.2f}" stroke="#999"/>
  </g>
  {''.join(curve_markup)}
  {''.join(tick_markup)}
  <text x="{x0 + width / 2:.2f}" y="462" text-anchor="middle" font-family="Arial" font-size="14">{x_label}</text>
  <text x="24" y="{y0 + height / 2:.2f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 24 {y0 + height / 2:.2f})">{y_label}</text>
  {''.join(legend_markup)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structure", action="append", required=True, help="LABEL=path/to/POSCAR_or_CIF. Repeatable.")
    ap.add_argument("--species-order", help="Comma-separated species order, e.g. Na,U,Cl.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="static_pdf_xrd")
    ap.add_argument("--two-theta-min", type=float, default=10.0)
    ap.add_argument("--two-theta-max", type=float, default=90.0)
    ap.add_argument("--two-theta-step", type=float, default=0.05)
    ap.add_argument("--wavelength-a", type=float, default=1.5406, help="Cu K-alpha default.")
    ap.add_argument("--coherence-radius-a", type=float, default=8.0)
    ap.add_argument("--smooth-sigma-deg", type=float, default=0.18)
    ap.add_argument("--rmax", type=float, default=10.0)
    ap.add_argument("--dr", type=float, default=0.02)
    ap.add_argument("--partials", action="store_true")
    args = ap.parse_args(argv)

    np = _import_numpy()
    backend = _import_atomi_backend()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    species_order = species_order_arg(args.species_order)

    frames: dict[str, dict[str, Any]] = {}
    for item in args.structure:
        label, path = parse_labeled_path(item)
        frame, meta = backend["frame_from_structure_file"](path, species_order=species_order)
        frames[label] = frame
        (outdir / f"{args.prefix}_{label}_metadata.json").write_text(str(meta) + "\n", encoding="utf-8")

    angles = np.arange(args.two_theta_min, args.two_theta_max + args.two_theta_step, args.two_theta_step)
    q = 4.0 * np.pi * np.sin(np.deg2rad(0.5 * angles)) / args.wavelength_a
    all_species = species_order or sorted({sym for frame in frames.values() for sym in frame["symbols"]})
    form_factors, _ = backend["diffraction_form_factors"](all_species, q, scattering="xraydb", custom=None)

    xrd_curves = {}
    for label, frame in frames.items():
        _, intensity = backend["simulated_powder_xrd_from_frames"](
            [frame],
            species_order=all_species,
            form_factors=form_factors,
            wavelength_a=args.wavelength_a,
            two_theta_min_deg=args.two_theta_min,
            two_theta_max_deg=args.two_theta_max,
            two_theta_step_deg=args.two_theta_step,
            coherence_radius_a=args.coherence_radius_a,
            smooth_sigma_deg=args.smooth_sigma_deg,
        )
        xrd_curves[label] = normalize_curve(intensity)
    write_curves_csv(outdir / f"{args.prefix}_xrd.csv", angles, xrd_curves, "two_theta_deg")
    xrd_png = outdir / f"{args.prefix}_xrd.png"
    try:
        backend["write_xrd_multi_overlay_plot"](xrd_png, angles, xrd_curves, title=f"{args.prefix}: simulated powder XRD")
    except Exception:
        pass
    write_overlay_svg(
        outdir / f"{args.prefix}_xrd.svg",
        title=f"{args.prefix}: simulated powder XRD",
        x=angles,
        curves=xrd_curves,
        x_label="2theta (deg, Cu K-alpha)",
        y_label="Normalized intensity",
    )

    total_pdf_curves = {}
    for label, frame in frames.items():
        pdf_curves = compute_pdf_curves(frame, rmax=args.rmax, dr=args.dr, partials=args.partials)
        for curve_label, (r, values) in pdf_curves.items():
            total_pdf_curves[f"{label}:{curve_label}"] = values
    write_curves_csv(outdir / f"{args.prefix}_pdf.csv", r, total_pdf_curves, "r_A")
    write_overlay_svg(
        outdir / f"{args.prefix}_pdf.svg",
        title=f"{args.prefix}: quick PBC RDF/PDF",
        x=r,
        curves=total_pdf_curves,
        x_label="r (A)",
        y_label="g(r), normalized for display",
    )
    print(outdir / f"{args.prefix}_xrd.svg")
    print(outdir / f"{args.prefix}_pdf.svg")


if __name__ == "__main__":
    main()
