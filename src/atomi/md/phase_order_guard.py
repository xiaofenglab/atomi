#!/usr/bin/env python3
"""XRD-first plus PDF/RDF long-range-order guards for MD liquid/solid screening.

When crystalline references are supplied, the primary guard is conventional
Bragg powder XRD from CIF/POSCAR references compared with the MD result frame.
The older early/tail trajectory PDF and finite-cell Debye diffraction signals
remain secondary local/medium-range and total-scattering order guards.  Finite
cell Debye scattering from small MD cells is useful, but it should not be
confused with a conventional CIF-derived powder XRD stick/broadened pattern.
This module is intentionally not a standalone liquid classifier: molten network
salts can retain sharp first-shell cation-anion peaks, so it focuses on damping
of long-range PDF/diffraction order.  Use its result together with
MSD/diffusion and thermodynamic stability before accepting SLUSCHI entropy
windows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from atomi.md.rdf_pdf import parse_csv_list, parse_weights, rdf_pdf_from_vasp_xdatcar, vasp_frames_from_xdatcar
from atomi.structure.adapters import read_vasp_poscar_basis


SCHEMA_PHASE_ORDER_GUARD = "atomi.md.phase_order_guard.v3"

ATOMIC_NUMBER_WEIGHTS = {
    "H": 1.0,
    "Li": 3.0,
    "C": 6.0,
    "O": 8.0,
    "Na": 11.0,
    "Cl": 17.0,
    "K": 19.0,
    "Ce": 58.0,
    "U": 92.0,
}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    def normalize(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        return value

    path.write_text(json.dumps(normalize(data), indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_xrd_csv(path: Path, two_theta_deg: np.ndarray, intensity: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["two_theta_deg", "intensity_normalized"])
        for angle, value in zip(two_theta_deg, intensity):
            writer.writerow([float(angle), float(value)])


def write_xrd_overlay_plot(
    path: Path,
    two_theta_deg: np.ndarray,
    early_intensity: np.ndarray,
    tail_intensity: np.ndarray,
    *,
    title: str,
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        svg_path = path.with_suffix(".svg")
        write_xrd_overlay_svg(
            svg_path,
            two_theta_deg,
            early_intensity,
            tail_intensity,
            title=title,
        )
        return str(svg_path.resolve())
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(two_theta_deg, early_intensity, color="#4a4a4a", linewidth=1.8, label="Early window")
    ax.plot(two_theta_deg, tail_intensity, color="#1f77b4", linewidth=1.8, label="Tail window")
    ax.set_title(title)
    ax.set_xlabel(r"2$\theta$ (deg, Cu K$\alpha$)")
    ax.set_ylabel("Normalized intensity")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path.resolve())


def _svg_polyline(x: np.ndarray, y: np.ndarray, *, x0: float, y0: float, width: float, height: float) -> str:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = 0.0
    y_max = max(float(np.max(y)), 1.0)
    points = []
    for xi, yi in zip(x, y):
        px = x0 + (float(xi) - x_min) / max(x_max - x_min, 1.0e-12) * width
        py = y0 + height - (float(yi) - y_min) / max(y_max - y_min, 1.0e-12) * height
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def write_xrd_overlay_svg(
    path: Path,
    two_theta_deg: np.ndarray,
    early_intensity: np.ndarray,
    tail_intensity: np.ndarray,
    *,
    title: str,
) -> None:
    x0, y0, width, height = 74.0, 54.0, 560.0, 300.0
    early_points = _svg_polyline(two_theta_deg, early_intensity, x0=x0, y0=y0, width=width, height=height)
    tail_points = _svg_polyline(two_theta_deg, tail_intensity, x0=x0, y0=y0, width=width, height=height)
    x_min = float(np.min(two_theta_deg))
    x_max = float(np.max(two_theta_deg))
    ticks = np.linspace(x_min, x_max, 5)
    tick_markup = []
    for tick in ticks:
        px = x0 + (tick - x_min) / max(x_max - x_min, 1.0e-12) * width
        tick_markup.append(
            f'<line x1="{px:.2f}" y1="{y0 + height:.2f}" x2="{px:.2f}" y2="{y0 + height + 5:.2f}" '
            'stroke="#444" stroke-width="1"/>'
        )
        tick_markup.append(
            f'<text x="{px:.2f}" y="{y0 + height + 22:.2f}" text-anchor="middle" '
            'font-family="Arial" font-size="12">'
            f"{tick:.0f}</text>"
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="700" height="430" viewBox="0 0 700 430">
  <rect width="700" height="430" fill="white"/>
  <text x="350" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
  <rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="none" stroke="#222" stroke-width="1.2"/>
  <g opacity="0.25">
    <line x1="{x0}" y1="{y0 + height * 0.25:.2f}" x2="{x0 + width}" y2="{y0 + height * 0.25:.2f}" stroke="#999" stroke-width="0.7"/>
    <line x1="{x0}" y1="{y0 + height * 0.50:.2f}" x2="{x0 + width}" y2="{y0 + height * 0.50:.2f}" stroke="#999" stroke-width="0.7"/>
    <line x1="{x0}" y1="{y0 + height * 0.75:.2f}" x2="{x0 + width}" y2="{y0 + height * 0.75:.2f}" stroke="#999" stroke-width="0.7"/>
  </g>
  <polyline points="{early_points}" fill="none" stroke="#4a4a4a" stroke-width="2.0"/>
  <polyline points="{tail_points}" fill="none" stroke="#1f77b4" stroke-width="2.0"/>
  {''.join(tick_markup)}
  <text x="{x0 + width / 2:.2f}" y="410" text-anchor="middle" font-family="Arial" font-size="14">2theta (deg, Cu K-alpha)</text>
  <text x="22" y="{y0 + height / 2:.2f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 22 {y0 + height / 2:.2f})">Normalized intensity</text>
  <line x1="500" y1="28" x2="535" y2="28" stroke="#4a4a4a" stroke-width="2"/>
  <text x="542" y="32" font-family="Arial" font-size="12">Early window</text>
  <line x1="500" y1="46" x2="535" y2="46" stroke="#1f77b4" stroke-width="2"/>
  <text x="542" y="50" font-family="Arial" font-size="12">Tail window</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_xrd_multi_overlay_svg(
    path: Path,
    two_theta_deg: np.ndarray,
    curves: dict[str, np.ndarray],
    *,
    title: str,
) -> None:
    colors = ["#404040", "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
    x0, y0, width, height = 82.0, 58.0, 610.0, 315.0
    x_min = float(np.min(two_theta_deg))
    x_max = float(np.max(two_theta_deg))
    tick_markup = []
    for tick in np.linspace(x_min, x_max, 5):
        px = x0 + (tick - x_min) / max(x_max - x_min, 1.0e-12) * width
        tick_markup.append(
            f'<line x1="{px:.2f}" y1="{y0 + height:.2f}" x2="{px:.2f}" y2="{y0 + height + 5:.2f}" '
            'stroke="#444" stroke-width="1"/>'
        )
        tick_markup.append(
            f'<text x="{px:.2f}" y="{y0 + height + 22:.2f}" text-anchor="middle" '
            f'font-family="Arial" font-size="12">{tick:.0f}</text>'
        )
    curve_markup = []
    legend_markup = []
    for idx, (label, values) in enumerate(curves.items()):
        color = colors[idx % len(colors)]
        curve_markup.append(
            f'<polyline points="{_svg_polyline(two_theta_deg, values, x0=x0, y0=y0, width=width, height=height)}" '
            f'fill="none" stroke="{color}" stroke-width="2.0"/>'
        )
        ly = 28 + idx * 18
        legend_markup.append(f'<line x1="520" y1="{ly}" x2="555" y2="{ly}" stroke="{color}" stroke-width="2.2"/>')
        legend_markup.append(f'<text x="563" y="{ly + 4}" font-family="Arial" font-size="12">{label}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="780" height="450" viewBox="0 0 780 450">
  <rect width="780" height="450" fill="white"/>
  <text x="390" y="30" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
  <rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="none" stroke="#222" stroke-width="1.2"/>
  <g opacity="0.22">
    <line x1="{x0}" y1="{y0 + height * 0.25:.2f}" x2="{x0 + width}" y2="{y0 + height * 0.25:.2f}" stroke="#999"/>
    <line x1="{x0}" y1="{y0 + height * 0.50:.2f}" x2="{x0 + width}" y2="{y0 + height * 0.50:.2f}" stroke="#999"/>
    <line x1="{x0}" y1="{y0 + height * 0.75:.2f}" x2="{x0 + width}" y2="{y0 + height * 0.75:.2f}" stroke="#999"/>
  </g>
  {''.join(curve_markup)}
  {''.join(tick_markup)}
  <text x="{x0 + width / 2:.2f}" y="430" text-anchor="middle" font-family="Arial" font-size="14">2theta (deg, Cu K-alpha)</text>
  <text x="24" y="{y0 + height / 2:.2f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 24 {y0 + height / 2:.2f})">Normalized intensity</text>
  {''.join(legend_markup)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_xrd_multi_overlay_plot(
    path: Path,
    two_theta_deg: np.ndarray,
    curves: dict[str, np.ndarray],
    *,
    title: str,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        svg_path = path.with_suffix(".svg")
        write_xrd_multi_overlay_svg(svg_path, two_theta_deg, curves, title=title)
        return str(svg_path.resolve())
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    colors = ["#404040", "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
    for idx, (label, values) in enumerate(curves.items()):
        ax.plot(two_theta_deg, values, linewidth=1.8, color=colors[idx % len(colors)], label=label)
    ax.set_title(title)
    ax.set_xlabel(r"2$\theta$ (deg, Cu K$\alpha$)")
    ax.set_ylabel("Normalized intensity")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path.resolve())


def read_total_pdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    r_values: list[float] = []
    g_values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            r_values.append(float(row["r_A"]))
            g_values.append(float(row["g_total_weighted"]))
    return np.asarray(r_values, dtype=float), np.asarray(g_values, dtype=float)


def _window_values(r: np.ndarray, g: np.ndarray, r_min: float, r_max: float | None) -> np.ndarray:
    upper = np.inf if r_max is None else r_max
    mask = (r >= r_min) & (r <= upper)
    if not np.any(mask):
        raise ValueError(f"No PDF bins in requested long-range window {r_min:g}-{upper:g} A.")
    return g[mask]


def order_metrics(r: np.ndarray, g: np.ndarray, *, r_min: float, r_max: float | None) -> dict[str, float]:
    values = _window_values(r, g, r_min, r_max)
    centered = values - float(np.mean(values))
    if len(values) > 2:
        curvature = np.diff(values, n=2)
        roughness = float(np.sqrt(np.mean(curvature * curvature)))
    else:
        roughness = 0.0
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "peak_to_peak": float(np.max(values) - np.min(values)),
        "p95_abs_centered": float(np.percentile(np.abs(centered), 95)),
        "roughness": roughness,
    }


def gaussian_smooth(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0.0:
        return values
    radius = max(int(np.ceil(4.0 * sigma_bins)), 1)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(values, kernel, mode="same")


def diffraction_form_factors(
    species_order: list[str],
    q_values: np.ndarray,
    *,
    scattering: str,
    custom: dict[str, float] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    requested_scattering = scattering
    if scattering == "custom":
        if not custom:
            raise ValueError("--xrd-scattering custom requires --xrd-weights.")
        missing = sorted(set(species_order) - set(custom))
        if missing:
            raise ValueError(f"Diffraction weights missing species: {','.join(missing)}")
        return (
            {symbol: np.full_like(q_values, float(custom[symbol]), dtype=float) for symbol in species_order},
            {
                "xray_form_factor_source": "custom_constant",
                "weights": {symbol: float(custom[symbol]) for symbol in species_order},
            },
        )
    if scattering == "xraydb":
        try:
            import xraydb  # type: ignore
        except ImportError:
            scattering = "atomic-number"
        else:
            q_xraydb = q_values / (4.0 * np.pi)
            return (
                {symbol: np.asarray(xraydb.f0(symbol, q_xraydb), dtype=float) for symbol in species_order},
                {
                    "xray_form_factor_source": "xraydb",
                    "xraydb_argument": "q / (4*pi), matching Atomi lammps.rdf_pdf X-ray S(Q) convention",
                    "weights_q0": {
                        symbol: float(np.asarray(xraydb.f0(symbol, np.asarray([0.0])), dtype=float)[0])
                        for symbol in species_order
                    },
                },
            )
    if scattering != "atomic-number":
        raise ValueError(f"Unsupported --xrd-scattering mode: {scattering}")
    missing_default = sorted(symbol for symbol in species_order if symbol not in ATOMIC_NUMBER_WEIGHTS)
    if missing_default:
        raise ValueError(
            "No default diffraction weights for species "
            f"{','.join(missing_default)}; pass --xrd-weights like Ce=58 Cl=17."
        )
    weights = {symbol: ATOMIC_NUMBER_WEIGHTS[symbol] for symbol in species_order}
    meta: dict[str, Any] = {
        "xray_form_factor_source": "atomic-number",
        "weights": weights,
    }
    if requested_scattering == "xraydb":
        meta["warning"] = "xraydb unavailable; fell back to constant atomic-number weights."
    else:
        meta["note"] = "Constant atomic-number weights approximate low-Q X-ray scattering."
    return ({symbol: np.full_like(q_values, weights[symbol], dtype=float) for symbol in species_order}, meta)


def xrd_two_theta_q(
    *,
    wavelength_a: float,
    two_theta_min_deg: float,
    two_theta_max_deg: float,
    two_theta_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    two_theta = np.arange(two_theta_min_deg, two_theta_max_deg + 0.5 * two_theta_step_deg, two_theta_step_deg)
    theta_rad = np.deg2rad(0.5 * two_theta)
    q = 4.0 * np.pi * np.sin(theta_rad) / wavelength_a
    return two_theta, q


def pair_distances_for_diffraction(
    frame: dict[str, Any],
    *,
    coherence_radius_a: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pair distances and atom indices including periodic images.

    This is a compact Debye-scattering approximation for powder-pattern
    screening.  It is not a replacement for a crystallographic refinement, but
    it is useful for asking whether strong finite-cell Bragg peakiness survives
    into the MD tail.
    """
    frac = np.asarray(frame["frac"], dtype=float)
    cell = np.asarray(frame["cell"], dtype=float)
    lengths = np.linalg.norm(cell, axis=1)
    repeats = [int(np.ceil(coherence_radius_a / max(length, 1.0))) for length in lengths]
    translations = np.asarray(
        [
            (i, j, k)
            for i in range(-repeats[0], repeats[0] + 1)
            for j in range(-repeats[1], repeats[1] + 1)
            for k in range(-repeats[2], repeats[2] + 1)
        ],
        dtype=float,
    )
    n_atoms = len(frac)
    pair_i: list[int] = []
    pair_j: list[int] = []
    distances: list[float] = []
    for t in translations:
        shifted = frac[None, :, :] + t[None, None, :]
        dfrac = shifted - frac[:, None, :]
        cart = dfrac @ cell
        dist = np.linalg.norm(cart, axis=2)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if np.allclose(t, 0.0) and j <= i:
                    continue
                r = float(dist[i, j])
                if 1.0e-8 < r <= coherence_radius_a:
                    pair_i.append(i)
                    pair_j.append(j)
                    distances.append(r)
    return np.asarray(distances, dtype=float), np.asarray(pair_i, dtype=int), np.asarray(pair_j, dtype=int)


def simulated_powder_xrd_from_frames(
    frames: list[dict[str, Any]],
    *,
    species_order: list[str],
    form_factors: dict[str, np.ndarray],
    wavelength_a: float,
    two_theta_min_deg: float,
    two_theta_max_deg: float,
    two_theta_step_deg: float,
    coherence_radius_a: float,
    smooth_sigma_deg: float,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    two_theta, q = xrd_two_theta_q(
        wavelength_a=wavelength_a,
        two_theta_min_deg=two_theta_min_deg,
        two_theta_max_deg=two_theta_max_deg,
        two_theta_step_deg=two_theta_step_deg,
    )
    intensity = np.zeros_like(q, dtype=float)
    for frame in frames:
        symbols = np.asarray(frame["symbols"])
        atom_f = np.vstack([form_factors[str(symbol)] for symbol in symbols]).T
        frame_i = np.sum(atom_f * atom_f, axis=1)
        distances, pair_i, pair_j = pair_distances_for_diffraction(frame, coherence_radius_a=coherence_radius_a)
        if len(distances):
            prefactors = atom_f[:, pair_i] * atom_f[:, pair_j]
            qr = q[:, None] * distances[None, :]
            sinc = np.ones_like(qr)
            mask = np.abs(qr) > 1.0e-12
            sinc[mask] = np.sin(qr[mask]) / qr[mask]
            frame_i += 2.0 * np.sum(prefactors * sinc, axis=1)
        intensity += frame_i
    intensity /= max(len(frames), 1)
    intensity = np.maximum(intensity, 0.0)
    sigma_bins = smooth_sigma_deg / two_theta_step_deg if two_theta_step_deg > 0.0 else 0.0
    intensity = gaussian_smooth(intensity, sigma_bins)
    max_i = float(np.max(intensity)) if len(intensity) else 0.0
    if normalize and max_i > 0.0:
        intensity = intensity / max_i
    return two_theta, intensity


def bragg_powder_xrd_from_frame(
    frame: dict[str, Any],
    *,
    species_order: list[str],
    form_factors: dict[str, np.ndarray],
    wavelength_a: float,
    two_theta_min_deg: float,
    two_theta_max_deg: float,
    two_theta_step_deg: float,
    smooth_sigma_deg: float,
    lorentz_polarization: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate conventional Bragg powder XRD from one periodic structure.

    This is the CIF/POSCAR reference path: generate reciprocal-lattice planes,
    apply Bragg's law, compute structure factors, optionally apply a standard
    Lorentz-polarization factor, and broaden the resulting peaks.  It differs
    from :func:`simulated_powder_xrd_from_frames`, which is a finite-cell Debye
    total-scattering approximation used for disordered MD snapshots.
    """
    two_theta, q_grid = xrd_two_theta_q(
        wavelength_a=wavelength_a,
        two_theta_min_deg=two_theta_min_deg,
        two_theta_max_deg=two_theta_max_deg,
        two_theta_step_deg=two_theta_step_deg,
    )
    intensity = np.zeros_like(two_theta, dtype=float)
    cell = np.asarray(frame["cell"], dtype=float)
    frac = np.asarray(frame["frac"], dtype=float) % 1.0
    symbols = np.asarray(frame["symbols"])
    missing = sorted(set(symbols) - set(species_order))
    if missing:
        raise ValueError(f"Reference species order is missing species present in frame: {','.join(missing)}")
    volume = abs(float(np.linalg.det(cell)))
    if volume <= 0.0:
        raise ValueError("Bragg XRD requires a non-zero periodic cell volume.")
    reciprocal = np.linalg.inv(cell).T
    theta_max = np.deg2rad(0.5 * two_theta_max_deg)
    min_d = wavelength_a / (2.0 * max(np.sin(theta_max), 1.0e-8))
    direct_lengths = np.linalg.norm(cell, axis=1)
    max_index = max(int(np.ceil(float(np.max(direct_lengths)) / max(min_d, 1.0e-8))) + 1, 1)
    sigma_bins = smooth_sigma_deg / two_theta_step_deg if two_theta_step_deg > 0.0 else 0.0
    sigma_deg = max(smooth_sigma_deg, two_theta_step_deg)
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkl = np.asarray([h, k, l], dtype=float)
                g_norm = float(np.linalg.norm(hkl @ reciprocal))
                if g_norm <= 1.0e-12:
                    continue
                d_hkl = 1.0 / g_norm
                sin_theta = wavelength_a / (2.0 * d_hkl)
                if sin_theta <= 0.0 or sin_theta >= 1.0:
                    continue
                angle = float(np.rad2deg(2.0 * np.arcsin(sin_theta)))
                if angle < two_theta_min_deg or angle > two_theta_max_deg:
                    continue
                q_peak = 2.0 * np.pi / d_hkl
                atom_f = np.asarray(
                    [
                        float(np.interp(q_peak, q_grid, form_factors[str(symbol)], left=form_factors[str(symbol)][0], right=form_factors[str(symbol)][-1]))
                        for symbol in symbols
                    ],
                    dtype=float,
                )
                phase = np.exp(2j * np.pi * (frac @ hkl))
                structure_factor = np.sum(atom_f * phase)
                peak_intensity = float(abs(structure_factor) ** 2)
                if lorentz_polarization:
                    theta = np.deg2rad(0.5 * angle)
                    two_theta_rad = np.deg2rad(angle)
                    lp = (1.0 + np.cos(two_theta_rad) ** 2) / max(
                        (np.sin(theta) ** 2) * max(np.cos(theta), 1.0e-8),
                        1.0e-8,
                    )
                    peak_intensity *= float(lp)
                if peak_intensity <= 0.0:
                    continue
                if sigma_bins <= 0.0:
                    idx = int(np.argmin(np.abs(two_theta - angle)))
                    intensity[idx] += peak_intensity
                else:
                    intensity += peak_intensity * np.exp(-0.5 * ((two_theta - angle) / sigma_deg) ** 2)
    max_i = float(np.max(intensity)) if len(intensity) else 0.0
    if normalize and max_i > 0.0:
        intensity = intensity / max_i
    return two_theta, intensity


def frame_from_vasp_poscar(path: Path, *, species_order: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a POSCAR/CONTCAR as one generic MD frame for reference XRD."""
    basis = read_vasp_poscar_basis(path)
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    coord_start = 7
    if lines[coord_start].lower().startswith("s"):
        coord_start += 1
    mode = lines[coord_start].strip().lower()
    coord_start += 1
    natoms = int(basis["natoms"])
    coords = []
    for row in lines[coord_start : coord_start + natoms]:
        parts = row.split()
        if len(parts) < 3:
            raise ValueError(f"Malformed coordinate row in {path}")
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if len(coords) != natoms:
        raise ValueError(f"Expected {natoms} coordinates in {path}, found {len(coords)}")
    cell = np.asarray(basis["lattice"], dtype=float)
    coord_array = np.asarray(coords, dtype=float)
    if mode.startswith("d"):
        frac = coord_array % 1.0
    elif mode.startswith(("c", "k")):
        frac = (coord_array @ np.linalg.inv(cell)) % 1.0
    else:
        raise ValueError(f"Unsupported POSCAR coordinate mode {mode!r} in {path}")
    order = species_order or list(basis["elements"])
    missing = sorted(set(basis["symbols"]) - set(order))
    if missing:
        raise ValueError(f"Reference species order is missing species present in {path}: {','.join(missing)}")
    frame = {
        "index": 0,
        "symbols": list(basis["symbols"]),
        "frac": frac,
        "cell": cell,
        "volume_A3": float(abs(np.linalg.det(cell))),
    }
    return frame, {
        "format": "vasp-poscar",
        "path": str(path.resolve()),
        "species_order": order,
        "elements": basis["elements"],
        "counts": dict(zip(basis["elements"], basis["counts"])),
        "natoms": natoms,
        "volume_A3": frame["volume_A3"],
    }


def frame_from_structure_file(path: Path, *, species_order: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read POSCAR/CONTCAR or CIF as one frame for reference XRD."""
    if path.suffix.lower() == ".cif":
        try:
            from ase.io import read  # type: ignore
        except Exception as exc:
            raise ValueError("Reading CIF references requires ASE.") from exc
        atoms = read(path)
        symbols = list(atoms.get_chemical_symbols())
        cell = np.asarray(atoms.cell.array, dtype=float)
        frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
        order = species_order or sorted(set(symbols), key=symbols.index)
        missing = sorted(set(symbols) - set(order))
        if missing:
            raise ValueError(f"Reference species order is missing species present in {path}: {','.join(missing)}")
        volume = float(abs(np.linalg.det(cell)))
        return (
            {
                "index": 0,
                "symbols": symbols,
                "frac": frac,
                "cell": cell,
                "volume_A3": volume,
            },
            {
                "format": "cif",
                "path": str(path.resolve()),
                "species_order": order,
                "elements": sorted(set(symbols), key=symbols.index),
                "counts": {symbol: symbols.count(symbol) for symbol in sorted(set(symbols), key=symbols.index)},
                "natoms": len(symbols),
                "volume_A3": volume,
            },
        )
    return frame_from_vasp_poscar(path, species_order=species_order)


def xrd_curve_similarity(reference: np.ndarray, target: np.ndarray, two_theta_deg: np.ndarray) -> dict[str, float]:
    """Compare normalized XRD curves against a crystalline reference pattern."""
    ref = np.asarray(reference, dtype=float)
    tgt = np.asarray(target, dtype=float)
    ref_c = ref - float(np.mean(ref))
    tgt_c = tgt - float(np.mean(tgt))
    denom = float(np.linalg.norm(ref_c) * np.linalg.norm(tgt_c))
    corr = float(np.dot(ref_c, tgt_c) / denom) if denom > 0.0 else 0.0
    rmse = float(np.sqrt(np.mean((ref - tgt) ** 2)))
    ref_metrics = diffraction_metrics(two_theta_deg, ref)
    target_metrics = diffraction_metrics(two_theta_deg, tgt)
    peak_shift = abs(ref_metrics["two_theta_at_max_deg"] - target_metrics["two_theta_at_max_deg"])
    return {
        "pearson_r": corr,
        "rmse": rmse,
        "std_target_over_reference": target_metrics["std"] / ref_metrics["std"] if ref_metrics["std"] else float("inf"),
        "p95_target_over_reference": target_metrics["p95_abs_centered"] / ref_metrics["p95_abs_centered"]
        if ref_metrics["p95_abs_centered"]
        else float("inf"),
        "max_peak_shift_deg": float(peak_shift),
        "reference_max_two_theta_deg": ref_metrics["two_theta_at_max_deg"],
        "target_max_two_theta_deg": target_metrics["two_theta_at_max_deg"],
    }


def xrd_reference_peak_score(
    reference: np.ndarray,
    target: np.ndarray,
    two_theta_deg: np.ndarray,
    *,
    min_reference_peak_height: float = 0.12,
    peak_window_deg: float = 0.30,
    retained_intensity_min: float = 0.25,
    max_peaks: int = 40,
) -> dict[str, float]:
    """Score how much target intensity remains at crystalline reference peaks."""
    ref = np.asarray(reference, dtype=float)
    tgt = np.asarray(target, dtype=float)
    angles = np.asarray(two_theta_deg, dtype=float)
    if len(ref) != len(tgt) or len(ref) != len(angles):
        raise ValueError("Reference, target, and 2theta arrays must have the same length.")
    if len(ref) < 3:
        return {
            "reference_peak_count": 0.0,
            "mean_target_intensity_near_reference_peaks": 0.0,
            "weighted_target_intensity_near_reference_peaks": 0.0,
            "strong_reference_peak_fraction_retained": 0.0,
        }
    peak_mask = (ref[1:-1] >= ref[:-2]) & (ref[1:-1] >= ref[2:]) & (ref[1:-1] >= min_reference_peak_height)
    peak_indices = np.nonzero(peak_mask)[0] + 1
    if len(peak_indices) > max_peaks:
        strongest = np.argsort(ref[peak_indices])[-max_peaks:]
        peak_indices = np.sort(peak_indices[strongest])
    if len(peak_indices) == 0:
        return {
            "reference_peak_count": 0.0,
            "mean_target_intensity_near_reference_peaks": 0.0,
            "weighted_target_intensity_near_reference_peaks": 0.0,
            "strong_reference_peak_fraction_retained": 0.0,
        }
    target_near: list[float] = []
    ref_heights: list[float] = []
    for idx in peak_indices:
        local = np.abs(angles - angles[idx]) <= peak_window_deg
        target_near.append(float(np.max(tgt[local])) if np.any(local) else 0.0)
        ref_heights.append(float(ref[idx]))
    target_array = np.asarray(target_near, dtype=float)
    ref_array = np.asarray(ref_heights, dtype=float)
    retained = target_array >= retained_intensity_min
    weighted = float(np.sum(ref_array * target_array) / max(float(np.sum(ref_array)), 1.0e-12))
    return {
        "reference_peak_count": float(len(peak_indices)),
        "mean_target_intensity_near_reference_peaks": float(np.mean(target_array)),
        "weighted_target_intensity_near_reference_peaks": weighted,
        "strong_reference_peak_fraction_retained": float(np.mean(retained)),
    }


def classify_bragg_reference_guard(
    similarity: dict[str, float],
    peak_score: dict[str, float],
    *,
    solid_corr_min: float,
    solid_peak_fraction_min: float,
    solid_weighted_intensity_min: float,
    liquid_corr_max: float,
    liquid_peak_fraction_max: float,
    liquid_weighted_intensity_max: float,
) -> tuple[str, list[str]]:
    """Classify XRD reference agreement for a liquid/solid structural gate."""
    corr = similarity["pearson_r"]
    retained_fraction = peak_score["strong_reference_peak_fraction_retained"]
    weighted_intensity = peak_score["weighted_target_intensity_near_reference_peaks"]
    solid_votes = sum(
        [
            corr >= solid_corr_min,
            retained_fraction >= solid_peak_fraction_min,
            weighted_intensity >= solid_weighted_intensity_min,
        ]
    )
    notes: list[str] = [
        "Bragg reference XRD is the primary long-range-order guard; use PDF/RDF and MSD as secondary checks."
    ]
    if solid_votes >= 2:
        notes.append("The target frame retains substantial intensity at crystalline reference Bragg positions.")
        return "bragg-like-solid-warning", notes
    if (
        corr <= liquid_corr_max
        and retained_fraction <= liquid_peak_fraction_max
        and weighted_intensity <= liquid_weighted_intensity_max
    ):
        notes.append("Crystalline reference Bragg peak agreement is strongly damped in the target frame.")
        return "bragg-order-lost", notes
    notes.append("Reference Bragg agreement is intermediate; hold for PDF/RDF, MSD, and thermodynamic stability checks.")
    return "mixed/needs-pdf-msd", notes


def _species_order_for_frames(frames: list[dict[str, Any]], explicit: list[str] | None) -> list[str]:
    if explicit:
        order = list(explicit)
    else:
        order = []
    for frame in frames:
        for symbol in frame["symbols"]:
            symbol = str(symbol)
            if symbol not in order:
                order.append(symbol)
    return order


def _load_reference_frames(
    references: list[list[str]] | None,
    *,
    species_order: list[str] | None,
) -> tuple[list[tuple[str, float, dict[str, Any]]], list[dict[str, Any]]]:
    reference_frames: list[tuple[str, float, dict[str, Any]]] = []
    metadata: list[dict[str, Any]] = []
    for raw in references or []:
        if len(raw) != 3:
            raise ValueError("--xrd-reference/--reference expects LABEL WEIGHT PATH.")
        label, weight_raw, path_raw = raw
        weight = float(weight_raw)
        frame, meta = frame_from_structure_file(Path(path_raw), species_order=species_order)
        reference_frames.append((label, weight, frame))
        metadata.append({"label": label, "weight": weight, **meta})
    return reference_frames, metadata


def bragg_reference_guard(
    target_frame: dict[str, Any],
    *,
    target_label: str,
    reference_frames: list[tuple[str, float, dict[str, Any]]],
    reference_metadata: list[dict[str, Any]],
    species_order: list[str],
    outdir: Path,
    xrd_scattering: str,
    xrd_weights: dict[str, float] | None,
    wavelength_a: float,
    two_theta_min_deg: float,
    two_theta_max_deg: float,
    two_theta_step_deg: float,
    coherence_radius_a: float,
    smooth_sigma_deg: float,
    min_reference_peak_height: float,
    peak_window_deg: float,
    retained_intensity_min: float,
    solid_corr_min: float,
    solid_peak_fraction_min: float,
    solid_weighted_intensity_min: float,
    liquid_corr_max: float,
    liquid_peak_fraction_max: float,
    liquid_weighted_intensity_max: float,
) -> dict[str, Any]:
    """Run the XRD-first long-range-order gate against crystalline references."""
    outdir.mkdir(parents=True, exist_ok=True)
    two_theta, q_values = xrd_two_theta_q(
        wavelength_a=wavelength_a,
        two_theta_min_deg=two_theta_min_deg,
        two_theta_max_deg=two_theta_max_deg,
        two_theta_step_deg=two_theta_step_deg,
    )
    form_factors, scattering_meta = diffraction_form_factors(
        species_order,
        q_values,
        scattering=xrd_scattering,
        custom=xrd_weights,
    )
    target_two_theta, target_intensity = bragg_powder_xrd_from_frame(
        target_frame,
        species_order=species_order,
        form_factors=form_factors,
        wavelength_a=wavelength_a,
        two_theta_min_deg=two_theta_min_deg,
        two_theta_max_deg=two_theta_max_deg,
        two_theta_step_deg=two_theta_step_deg,
        smooth_sigma_deg=smooth_sigma_deg,
    )
    ref_two_theta, reference_mixture, individual_references = reference_mixture_xrd_curve(
        reference_frames,
        species_order=species_order,
        form_factors=form_factors,
        wavelength_a=wavelength_a,
        two_theta_min_deg=two_theta_min_deg,
        two_theta_max_deg=two_theta_max_deg,
        two_theta_step_deg=two_theta_step_deg,
        coherence_radius_a=coherence_radius_a,
        smooth_sigma_deg=smooth_sigma_deg,
        method="bragg",
    )
    if not np.allclose(target_two_theta, ref_two_theta):
        raise ValueError("Target and reference XRD grids differ unexpectedly.")
    similarity = xrd_curve_similarity(reference_mixture, target_intensity, two_theta)
    peak_score = xrd_reference_peak_score(
        reference_mixture,
        target_intensity,
        two_theta,
        min_reference_peak_height=min_reference_peak_height,
        peak_window_deg=peak_window_deg,
        retained_intensity_min=retained_intensity_min,
    )
    label, notes = classify_bragg_reference_guard(
        similarity,
        peak_score,
        solid_corr_min=solid_corr_min,
        solid_peak_fraction_min=solid_peak_fraction_min,
        solid_weighted_intensity_min=solid_weighted_intensity_min,
        liquid_corr_max=liquid_corr_max,
        liquid_peak_fraction_max=liquid_peak_fraction_max,
        liquid_weighted_intensity_max=liquid_weighted_intensity_max,
    )
    write_xrd_csv(outdir / "target_bragg_xrd.csv", two_theta, target_intensity)
    write_xrd_csv(outdir / "reference_mixture_bragg_xrd.csv", two_theta, reference_mixture)
    for label_ref, intensity in individual_references.items():
        safe_label = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in label_ref)
        write_xrd_csv(outdir / f"reference_{safe_label}_bragg_xrd.csv", two_theta, intensity)
    curves = {
        "Reference mixture": reference_mixture,
        target_label: target_intensity,
        **{f"Reference {label_ref}": values for label_ref, values in individual_references.items()},
    }
    overlay = write_xrd_multi_overlay_plot(
        outdir / "bragg_xrd_reference_overlay.png",
        two_theta,
        curves,
        title="Bragg XRD reference guard",
    )
    summary = {
        "schema": SCHEMA_PHASE_ORDER_GUARD,
        "guard": "bragg_reference_xrd",
        "guard_role": "primary_long_range_order_guard",
        "phase_order_label": label,
        "target_label": target_label,
        "species_order": species_order,
        "references": reference_metadata,
        "method": "bragg_hkl_structure_factor_powder_xrd",
        "radiation": "Cu K-alpha" if abs(wavelength_a - 1.5406) < 1.0e-4 else "custom",
        "wavelength_A": wavelength_a,
        "two_theta_range_deg": {
            "min": two_theta_min_deg,
            "max": two_theta_max_deg,
            "step": two_theta_step_deg,
        },
        "smooth_sigma_deg": smooth_sigma_deg,
        "scattering": scattering_meta,
        "similarity_to_reference_mixture": similarity,
        "reference_peak_score": peak_score,
        "thresholds": {
            "min_reference_peak_height": min_reference_peak_height,
            "peak_window_deg": peak_window_deg,
            "retained_intensity_min": retained_intensity_min,
            "solid_corr_min": solid_corr_min,
            "solid_peak_fraction_min": solid_peak_fraction_min,
            "solid_weighted_intensity_min": solid_weighted_intensity_min,
            "liquid_corr_max": liquid_corr_max,
            "liquid_peak_fraction_max": liquid_peak_fraction_max,
            "liquid_weighted_intensity_max": liquid_weighted_intensity_max,
        },
        "outputs": {
            "target_bragg_xrd_csv": str((outdir / "target_bragg_xrd.csv").resolve()),
            "reference_mixture_bragg_xrd_csv": str((outdir / "reference_mixture_bragg_xrd.csv").resolve()),
            "overlay_plot": overlay,
            "overlay_png": overlay if overlay.endswith(".png") else None,
            "overlay_svg": overlay if overlay.endswith(".svg") else None,
        },
        "notes": notes,
    }
    _write_json(outdir / "bragg_xrd_reference_guard_summary.json", summary)
    _write_csv(
        outdir / "bragg_xrd_reference_guard_metrics.csv",
        [
            {"metric_group": "similarity", **similarity},
            {"metric_group": "reference_peak_score", **peak_score},
        ],
    )
    return summary


def reference_mixture_xrd_curve(
    reference_frames: list[tuple[str, float, dict[str, Any]]],
    *,
    species_order: list[str],
    form_factors: dict[str, np.ndarray],
    wavelength_a: float,
    two_theta_min_deg: float,
    two_theta_max_deg: float,
    two_theta_step_deg: float,
    coherence_radius_a: float,
    smooth_sigma_deg: float,
    method: str = "bragg",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return weighted reference-mixture XRD plus individual reference curves.

    Use ``method="bragg"`` for crystalline CIF/POSCAR references.  The older
    ``method="debye"`` path is retained for finite clusters/snapshots, but it
    should not be used as the default comparator for crystalline reference XRD.
    """
    mixture: np.ndarray | None = None
    curves: dict[str, np.ndarray] = {}
    total_weight = sum(max(weight, 0.0) for _, weight, _ in reference_frames)
    if total_weight <= 0.0:
        raise ValueError("Reference mixture weights must sum to a positive number.")
    two_theta_ref: np.ndarray | None = None
    for label, weight, frame in reference_frames:
        if method == "bragg":
            two_theta, intensity = bragg_powder_xrd_from_frame(
                frame,
                species_order=species_order,
                form_factors=form_factors,
                wavelength_a=wavelength_a,
                two_theta_min_deg=two_theta_min_deg,
                two_theta_max_deg=two_theta_max_deg,
                two_theta_step_deg=two_theta_step_deg,
                smooth_sigma_deg=smooth_sigma_deg,
                normalize=False,
            )
        elif method == "debye":
            two_theta, intensity = simulated_powder_xrd_from_frames(
                [frame],
                species_order=species_order,
                form_factors=form_factors,
                wavelength_a=wavelength_a,
                two_theta_min_deg=two_theta_min_deg,
                two_theta_max_deg=two_theta_max_deg,
                two_theta_step_deg=two_theta_step_deg,
                coherence_radius_a=coherence_radius_a,
                smooth_sigma_deg=smooth_sigma_deg,
                normalize=False,
            )
        else:
            raise ValueError(f"Unsupported reference XRD method: {method}")
        two_theta_ref = two_theta
        max_single = float(np.max(intensity))
        curves[label] = intensity / max_single if max_single > 0.0 else intensity
        weighted = intensity * (max(weight, 0.0) / total_weight)
        mixture = weighted if mixture is None else mixture + weighted
    if mixture is None or two_theta_ref is None:
        raise ValueError("No reference frames supplied.")
    max_i = float(np.max(mixture))
    if max_i > 0.0:
        mixture = mixture / max_i
    return two_theta_ref, mixture, curves


def diffraction_metrics(two_theta: np.ndarray, intensity: np.ndarray) -> dict[str, float]:
    centered = intensity - float(np.mean(intensity))
    if len(intensity) > 2:
        curvature = np.diff(intensity, n=2)
        roughness = float(np.sqrt(np.mean(curvature * curvature)))
    else:
        roughness = 0.0
    return {
        "mean": float(np.mean(intensity)),
        "std": float(np.std(intensity)),
        "peak_to_peak": float(np.max(intensity) - np.min(intensity)),
        "p95_abs_centered": float(np.percentile(np.abs(centered), 95)),
        "roughness": roughness,
        "max_intensity": float(np.max(intensity)),
        "two_theta_at_max_deg": float(two_theta[int(np.argmax(intensity))]),
    }


def compare_order_metrics(
    early: dict[str, float],
    tail: dict[str, float],
    *,
    max_tail_order_ratio: float,
    signal_name: str = "PDF",
) -> tuple[str, list[str], dict[str, float]]:
    ratios: dict[str, float] = {}
    for key in ("std", "peak_to_peak", "p95_abs_centered", "roughness"):
        denom = early[key]
        ratios[f"{key}_tail_over_early"] = tail[key] / denom if denom > 0.0 else float("inf")
    pass_keys = ("std_tail_over_early", "peak_to_peak_tail_over_early", "p95_abs_centered_tail_over_early")
    passed = sum(1 for key in pass_keys if ratios[key] <= max_tail_order_ratio)
    notes: list[str] = []
    if passed >= 2:
        label = "long-range-order-lost"
        notes.append(f"Tail {signal_name} long-range order is damped relative to the early window.")
    else:
        label = "long-range-order-retained-or-uncertain"
        notes.append(f"Tail {signal_name} does not show enough long-range-order damping by the configured ratio guard.")
    notes.append(
        f"Use this {signal_name} guard with MSD/diffusion and thermodynamic stability; "
        f"{signal_name} alone is not a liquid proof."
    )
    return label, notes, ratios


def combined_phase_order_label(pdf_label: str, diffraction_label: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    if pdf_label == "long-range-order-lost" and diffraction_label == "long-range-order-lost":
        return "long-range-order-lost", ["Both PDF and simulated diffraction show damped long-range order."]
    if pdf_label == "long-range-order-lost" or diffraction_label == "long-range-order-lost":
        notes.append("Only one of PDF or simulated diffraction passed; keep the combined guard cautious.")
        return "long-range-order-partly-lost", notes
    notes.append("Neither PDF nor simulated diffraction passed the configured long-range-order damping guard.")
    return "long-range-order-retained-or-uncertain", notes


def vasp_xdatcar_guard(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.resolve()
    early_dir = outdir / "early_pdf"
    tail_dir = outdir / "tail_pdf"
    early_xrd_dir = outdir / "early_xrd"
    tail_xrd_dir = outdir / "tail_xrd"
    species_order = parse_csv_list(args.species_order)
    weights = parse_weights(args.weights) if args.weights else None
    xrd_weights = parse_weights(args.xrd_weights) if args.xrd_weights else None
    early = rdf_pdf_from_vasp_xdatcar(
        poscar=args.poscar,
        xdatcar=args.xdatcar,
        outdir=early_dir,
        prefix="early",
        species_order=species_order,
        start_frame=args.early_start_frame,
        stop_frame=args.early_stop_frame,
        stride=args.stride,
        rmax=args.rmax,
        dr=args.dr,
        weights=weights,
    )
    tail = rdf_pdf_from_vasp_xdatcar(
        poscar=args.poscar,
        xdatcar=args.xdatcar,
        outdir=tail_dir,
        prefix="tail",
        species_order=species_order,
        start_frame=args.tail_start_frame,
        stop_frame=args.tail_stop_frame,
        stride=args.stride,
        rmax=args.rmax,
        dr=args.dr,
        weights=weights,
    )
    r_early, g_early = read_total_pdf(Path(early["outputs"]["total_pdf_csv"]))
    r_tail, g_tail = read_total_pdf(Path(tail["outputs"]["total_pdf_csv"]))
    early_metrics = order_metrics(r_early, g_early, r_min=args.long_r_min, r_max=args.long_r_max)
    tail_metrics = order_metrics(r_tail, g_tail, r_min=args.long_r_min, r_max=args.long_r_max)
    label, notes, ratios = compare_order_metrics(
        early_metrics,
        tail_metrics,
        max_tail_order_ratio=args.max_tail_order_ratio,
        signal_name="PDF",
    )
    early_frames, early_frame_meta = vasp_frames_from_xdatcar(
        poscar=args.poscar,
        xdatcar=args.xdatcar,
        start_frame=args.early_start_frame,
        stop_frame=args.early_stop_frame,
        stride=args.stride,
        species_order=species_order,
    )
    tail_frames, tail_frame_meta = vasp_frames_from_xdatcar(
        poscar=args.poscar,
        xdatcar=args.xdatcar,
        start_frame=args.tail_start_frame,
        stop_frame=args.tail_stop_frame,
        stride=args.stride,
        species_order=species_order,
    )
    xrd_two_theta, xrd_q = xrd_two_theta_q(
        wavelength_a=args.xrd_wavelength_a,
        two_theta_min_deg=args.xrd_two_theta_min,
        two_theta_max_deg=args.xrd_two_theta_max,
        two_theta_step_deg=args.xrd_two_theta_step,
    )
    xrd_form_factors, xrd_scattering_meta = diffraction_form_factors(
        list(early_frame_meta["species_order"]),
        xrd_q,
        scattering=args.xrd_scattering,
        custom=xrd_weights,
    )
    early_two_theta, early_intensity = simulated_powder_xrd_from_frames(
        early_frames,
        species_order=list(early_frame_meta["species_order"]),
        form_factors=xrd_form_factors,
        wavelength_a=args.xrd_wavelength_a,
        two_theta_min_deg=args.xrd_two_theta_min,
        two_theta_max_deg=args.xrd_two_theta_max,
        two_theta_step_deg=args.xrd_two_theta_step,
        coherence_radius_a=args.xrd_coherence_radius_a,
        smooth_sigma_deg=args.xrd_smooth_sigma_deg,
    )
    tail_two_theta, tail_intensity = simulated_powder_xrd_from_frames(
        tail_frames,
        species_order=list(tail_frame_meta["species_order"]),
        form_factors=xrd_form_factors,
        wavelength_a=args.xrd_wavelength_a,
        two_theta_min_deg=args.xrd_two_theta_min,
        two_theta_max_deg=args.xrd_two_theta_max,
        two_theta_step_deg=args.xrd_two_theta_step,
        coherence_radius_a=args.xrd_coherence_radius_a,
        smooth_sigma_deg=args.xrd_smooth_sigma_deg,
    )
    early_xrd_dir.mkdir(parents=True, exist_ok=True)
    tail_xrd_dir.mkdir(parents=True, exist_ok=True)
    early_xrd_csv = early_xrd_dir / "early_powder_xrd.csv"
    tail_xrd_csv = tail_xrd_dir / "tail_powder_xrd.csv"
    write_xrd_csv(early_xrd_csv, early_two_theta, early_intensity)
    write_xrd_csv(tail_xrd_csv, tail_two_theta, tail_intensity)
    xrd_overlay_png = write_xrd_overlay_plot(
        outdir / "simulated_powder_xrd_early_tail_overlay.png",
        xrd_two_theta,
        early_intensity,
        tail_intensity,
        title="Finite-cell Debye diffraction early vs tail",
    )
    early_diffraction_metrics = diffraction_metrics(early_two_theta, early_intensity)
    tail_diffraction_metrics = diffraction_metrics(tail_two_theta, tail_intensity)
    diffraction_label, diffraction_notes, diffraction_ratios = compare_order_metrics(
        early_diffraction_metrics,
        tail_diffraction_metrics,
        max_tail_order_ratio=args.max_tail_bragg_ratio,
        signal_name="finite-cell Debye diffraction",
    )
    combined_label, combined_notes = combined_phase_order_label(label, diffraction_label)
    bragg_reference_summary: dict[str, Any] | None = None
    workflow_phase_label = combined_label
    workflow_guard_role = "secondary_pdf_plus_finite_cell_debye_guard"
    if args.xrd_reference:
        reference_frames, reference_metadata = _load_reference_frames(
            args.xrd_reference,
            species_order=list(tail_frame_meta["species_order"]),
        )
        bragg_reference_summary = bragg_reference_guard(
            tail_frames[-1],
            target_label="Tail final frame",
            reference_frames=reference_frames,
            reference_metadata=reference_metadata,
            species_order=list(tail_frame_meta["species_order"]),
            outdir=outdir / "bragg_xrd_reference_guard",
            xrd_scattering=args.xrd_scattering,
            xrd_weights=xrd_weights,
            wavelength_a=args.xrd_wavelength_a,
            two_theta_min_deg=args.xrd_two_theta_min,
            two_theta_max_deg=args.xrd_two_theta_max,
            two_theta_step_deg=args.xrd_two_theta_step,
            coherence_radius_a=args.xrd_coherence_radius_a,
            smooth_sigma_deg=args.xrd_smooth_sigma_deg,
            min_reference_peak_height=args.xrd_min_reference_peak_height,
            peak_window_deg=args.xrd_peak_window_deg,
            retained_intensity_min=args.xrd_retained_intensity_min,
            solid_corr_min=args.xrd_solid_corr_min,
            solid_peak_fraction_min=args.xrd_solid_peak_fraction_min,
            solid_weighted_intensity_min=args.xrd_solid_weighted_intensity_min,
            liquid_corr_max=args.xrd_liquid_corr_max,
            liquid_peak_fraction_max=args.xrd_liquid_peak_fraction_max,
            liquid_weighted_intensity_max=args.xrd_liquid_weighted_intensity_max,
        )
        workflow_guard_role = "primary_bragg_reference_xrd_with_secondary_pdf_debye_guard"
        if bragg_reference_summary["phase_order_label"] == "bragg-order-lost":
            workflow_phase_label = "long-range-order-lost"
        elif bragg_reference_summary["phase_order_label"] == "bragg-like-solid-warning":
            workflow_phase_label = "long-range-order-retained-or-uncertain"
        else:
            workflow_phase_label = combined_label
    summary = {
        "schema": SCHEMA_PHASE_ORDER_GUARD,
        "engine": "vasp-xdatcar",
        "poscar": str(args.poscar.resolve()),
        "xdatcar": str(args.xdatcar.resolve()),
        "phase_order_label": workflow_phase_label,
        "primary_guard_role": workflow_guard_role,
        "secondary_pdf_debye_order_label": combined_label,
        "pdf_order_label": label,
        "diffraction_order_label": diffraction_label,
        "early_frames": {
            "start": args.early_start_frame,
            "stop": args.early_stop_frame,
            "n_selected": early["n_selected_frames"],
        },
        "tail_frames": {
            "start": args.tail_start_frame,
            "stop": args.tail_stop_frame,
            "n_selected": tail["n_selected_frames"],
        },
        "long_range_window_A": {"min": args.long_r_min, "max": args.long_r_max},
        "max_tail_order_ratio": args.max_tail_order_ratio,
        "early_order_metrics": early_metrics,
        "tail_order_metrics": tail_metrics,
        "ratios": ratios,
        "diffraction": {
            "method": "finite_cell_debye_total_scattering",
            "method_note": (
                "Early/tail MD diffraction is computed with a finite-cell Debye scattering approximation. "
                "Use Bragg hkl/structure-factor powder XRD for crystalline CIF/POSCAR references."
            ),
            "radiation": "Cu K-alpha" if abs(args.xrd_wavelength_a - 1.5406) < 1.0e-4 else "custom",
            "wavelength_A": args.xrd_wavelength_a,
            "two_theta_range_deg": {
                "min": args.xrd_two_theta_min,
                "max": args.xrd_two_theta_max,
                "step": args.xrd_two_theta_step,
            },
            "coherence_radius_A": args.xrd_coherence_radius_a,
            "smooth_sigma_deg": args.xrd_smooth_sigma_deg,
            "max_tail_bragg_ratio": args.max_tail_bragg_ratio,
            "scattering": xrd_scattering_meta,
            "early_order_metrics": early_diffraction_metrics,
            "tail_order_metrics": tail_diffraction_metrics,
            "ratios": diffraction_ratios,
            "outputs": {
                "early_powder_xrd_csv": str(early_xrd_csv.resolve()),
                "tail_powder_xrd_csv": str(tail_xrd_csv.resolve()),
                "overlay_plot": xrd_overlay_png,
                "overlay_png": xrd_overlay_png if xrd_overlay_png and xrd_overlay_png.endswith(".png") else None,
                "overlay_svg": xrd_overlay_png if xrd_overlay_png and xrd_overlay_png.endswith(".svg") else None,
            },
        },
        "pdf_outputs": {
            "early_total_pdf_csv": early["outputs"]["total_pdf_csv"],
            "tail_total_pdf_csv": tail["outputs"]["total_pdf_csv"],
            "early_partial_rdfs_csv": early["outputs"]["partial_rdfs_csv"],
            "tail_partial_rdfs_csv": tail["outputs"]["partial_rdfs_csv"],
        },
        "bragg_reference_guard": bragg_reference_summary,
        "notes": notes
        + diffraction_notes
        + combined_notes
        + (
            [
                "Crystalline Bragg reference XRD was supplied and used as the primary long-range-order guard."
            ]
            if bragg_reference_summary
            else [
                "No crystalline Bragg reference XRD was supplied; the command only used PDF plus finite-cell Debye secondary guards."
            ]
        ),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    _write_json(outdir / "phase_order_guard_summary.json", summary)
    _write_csv(
        outdir / "phase_order_guard_metrics.csv",
        [
            {"window": "early", **early_metrics},
            {"window": "tail", **tail_metrics},
            {"window": "tail_over_early", **ratios},
            {"window": "early_diffraction", **early_diffraction_metrics},
            {"window": "tail_diffraction", **tail_diffraction_metrics},
            {"window": "diffraction_tail_over_early", **diffraction_ratios},
        ],
    )
    print(json.dumps(summary, indent=2))
    return summary


def bragg_frame_guard(args: argparse.Namespace) -> dict[str, Any]:
    explicit_order = parse_csv_list(args.species_order) if args.species_order else None
    target_frame, target_meta = frame_from_structure_file(args.structure, species_order=explicit_order)
    preliminary_refs, _ = _load_reference_frames(args.reference, species_order=explicit_order)
    species_order = _species_order_for_frames(
        [target_frame] + [frame for _, _, frame in preliminary_refs],
        explicit_order,
    )
    if explicit_order is None:
        target_frame, target_meta = frame_from_structure_file(args.structure, species_order=species_order)
        reference_frames, reference_metadata = _load_reference_frames(args.reference, species_order=species_order)
    else:
        reference_frames, reference_metadata = preliminary_refs, _load_reference_frames(
            args.reference,
            species_order=species_order,
        )[1]
    xrd_weights = parse_weights(args.xrd_weights) if args.xrd_weights else None
    summary = bragg_reference_guard(
        target_frame,
        target_label=args.target_label,
        reference_frames=reference_frames,
        reference_metadata=reference_metadata,
        species_order=species_order,
        outdir=args.outdir.resolve(),
        xrd_scattering=args.xrd_scattering,
        xrd_weights=xrd_weights,
        wavelength_a=args.xrd_wavelength_a,
        two_theta_min_deg=args.xrd_two_theta_min,
        two_theta_max_deg=args.xrd_two_theta_max,
        two_theta_step_deg=args.xrd_two_theta_step,
        coherence_radius_a=args.xrd_coherence_radius_a,
        smooth_sigma_deg=args.xrd_smooth_sigma_deg,
        min_reference_peak_height=args.xrd_min_reference_peak_height,
        peak_window_deg=args.xrd_peak_window_deg,
        retained_intensity_min=args.xrd_retained_intensity_min,
        solid_corr_min=args.xrd_solid_corr_min,
        solid_peak_fraction_min=args.xrd_solid_peak_fraction_min,
        solid_weighted_intensity_min=args.xrd_solid_weighted_intensity_min,
        liquid_corr_max=args.xrd_liquid_corr_max,
        liquid_peak_fraction_max=args.xrd_liquid_peak_fraction_max,
        liquid_weighted_intensity_max=args.xrd_liquid_weighted_intensity_max,
    )
    summary["engine"] = "bragg-frame"
    summary["structure"] = target_meta
    _write_json(args.outdir.resolve() / "bragg_xrd_reference_guard_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def add_xrd_reference_guard_arguments(parser: argparse.ArgumentParser, *, reference_flag: str) -> None:
    parser.add_argument(
        reference_flag,
        dest="xrd_reference" if reference_flag == "--xrd-reference" else "reference",
        action="append",
        nargs=3,
        default=[],
        metavar=("LABEL", "WEIGHT", "PATH"),
        help=f"Crystalline reference for Bragg XRD mixture, e.g. {reference_flag} NaCl 18 NaCl.cif.",
    )
    parser.add_argument("--xrd-min-reference-peak-height", type=float, default=0.12)
    parser.add_argument("--xrd-peak-window-deg", type=float, default=0.30)
    parser.add_argument("--xrd-retained-intensity-min", type=float, default=0.25)
    parser.add_argument("--xrd-solid-corr-min", type=float, default=0.35)
    parser.add_argument("--xrd-solid-peak-fraction-min", type=float, default=0.50)
    parser.add_argument("--xrd-solid-weighted-intensity-min", type=float, default=0.40)
    parser.add_argument("--xrd-liquid-corr-max", type=float, default=0.20)
    parser.add_argument("--xrd-liquid-peak-fraction-max", type=float, default=0.30)
    parser.add_argument("--xrd-liquid-weighted-intensity-max", type=float, default=0.25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare early/tail MD PDFs as a long-range-order liquid/solid guard.")
    sub = parser.add_subparsers(dest="command", required=True)
    vasp = sub.add_parser("vasp-xdatcar", help="Run PDF order guard from a VASP POSCAR/XDATCAR pair.")
    vasp.add_argument("--poscar", type=Path, default=Path("POSCAR"))
    vasp.add_argument("--xdatcar", type=Path, default=Path("XDATCAR"))
    vasp.add_argument("--outdir", type=Path, default=Path("phase_order_guard"))
    vasp.add_argument("--species-order", help="Comma-separated species order, e.g. Na,U,Cl.")
    vasp.add_argument("--early-start-frame", type=int, default=0)
    vasp.add_argument("--early-stop-frame", type=int, default=50)
    vasp.add_argument("--tail-start-frame", type=int, required=True)
    vasp.add_argument("--tail-stop-frame", type=int)
    vasp.add_argument("--stride", type=int, default=1)
    vasp.add_argument("--rmax", type=float, default=9.0)
    vasp.add_argument("--dr", type=float, default=0.04)
    vasp.add_argument("--long-r-min", type=float, default=4.5)
    vasp.add_argument("--long-r-max", type=float)
    vasp.add_argument(
        "--max-tail-order-ratio",
        type=float,
        default=0.75,
        help="Tail/early ratio threshold for long-range-order damping. Two of std, peak-to-peak, p95 must pass.",
    )
    vasp.add_argument(
        "--weights",
        nargs="*",
        default=[],
        help="Optional total g(r) weights like U=92 Cl=17. Defaults to composition-count fallback.",
    )
    vasp.add_argument(
        "--xrd-scattering",
        choices=("xraydb", "atomic-number", "custom"),
        default="xraydb",
        help="Scattering factors for simulated powder XRD. Default uses xraydb f0(q) with atomic-number fallback.",
    )
    vasp.add_argument(
        "--xrd-weights",
        nargs="*",
        default=[],
        help="Optional custom constant XRD weights like U=92 Cl=17 when --xrd-scattering custom is used.",
    )
    vasp.add_argument(
        "--xrd-wavelength-a",
        type=float,
        default=1.5406,
        help="X-ray wavelength in Angstrom. Default is Cu K-alpha.",
    )
    vasp.add_argument("--xrd-two-theta-min", type=float, default=10.0)
    vasp.add_argument("--xrd-two-theta-max", type=float, default=90.0)
    vasp.add_argument("--xrd-two-theta-step", type=float, default=0.05)
    vasp.add_argument(
        "--xrd-coherence-radius-a",
        type=float,
        default=18.0,
        help="Finite periodic-image radius for Debye powder diffraction in Angstrom.",
    )
    vasp.add_argument(
        "--xrd-smooth-sigma-deg",
        type=float,
        default=0.12,
        help="Gaussian broadening sigma in 2theta degrees for simulated powder XRD.",
    )
    vasp.add_argument(
        "--max-tail-bragg-ratio",
        type=float,
        default=0.65,
        help="Tail/early ratio threshold for simulated-diffraction Bragg peak damping.",
    )
    add_xrd_reference_guard_arguments(vasp, reference_flag="--xrd-reference")
    bragg = sub.add_parser(
        "bragg-frame",
        help="Compare one POSCAR/CONTCAR/CIF frame against crystalline Bragg XRD references.",
    )
    bragg.add_argument("--structure", type=Path, required=True, help="Target POSCAR/CONTCAR/CIF frame to screen.")
    bragg.add_argument("--outdir", type=Path, default=Path("bragg_xrd_reference_guard"))
    bragg.add_argument("--target-label", default="Target frame")
    bragg.add_argument("--species-order", help="Comma-separated species order, e.g. Na,U,Cl.")
    bragg.add_argument(
        "--xrd-scattering",
        choices=("xraydb", "atomic-number", "custom"),
        default="xraydb",
        help="Scattering factors for Bragg XRD. Default uses xraydb f0(q) with atomic-number fallback.",
    )
    bragg.add_argument(
        "--xrd-weights",
        nargs="*",
        default=[],
        help="Optional custom constant XRD weights like U=92 Cl=17 when --xrd-scattering custom is used.",
    )
    bragg.add_argument("--xrd-wavelength-a", type=float, default=1.5406, help="Default is Cu K-alpha.")
    bragg.add_argument("--xrd-two-theta-min", type=float, default=10.0)
    bragg.add_argument("--xrd-two-theta-max", type=float, default=90.0)
    bragg.add_argument("--xrd-two-theta-step", type=float, default=0.05)
    bragg.add_argument(
        "--xrd-coherence-radius-a",
        type=float,
        default=18.0,
        help="Retained for API symmetry with Debye reference mode; Bragg mode does not use it.",
    )
    bragg.add_argument("--xrd-smooth-sigma-deg", type=float, default=0.12)
    add_xrd_reference_guard_arguments(bragg, reference_flag="--reference")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "vasp-xdatcar":
        return vasp_xdatcar_guard(args)
    if args.command == "bragg-frame":
        return bragg_frame_guard(args)
    return None


if __name__ == "__main__":
    main()
