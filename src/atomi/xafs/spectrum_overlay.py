"""Overlay XANES spectra with optional experimental white-line alignment.

This module is intentionally route-agnostic.  It can combine experimental
Larch/Athena data with simulated spectra from FDMNES, OCEAN, Molcas, or any
other workflow that can export a numeric two-column curve.  Raw input energies
are preserved in the output CSV; alignment is recorded as a derived
``energy_aligned`` column.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Spectrum:
    label: str
    kind: str
    energy: tuple[float, ...]
    intensity: tuple[float, ...]
    source: str


@dataclass(frozen=True)
class AlignedSpectrum:
    spectrum: Spectrum
    white_line_energy: float
    energy_shift: float
    energy_aligned: tuple[float, ...]


XANES_DEFAULT_ENERGY_WINDOW = (-200.0, 300.0)


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _split_data_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "!", ";")):
        return []
    return stripped.replace(",", " ").split()


def _read_delimited_curve(path: Path, energy_column: str, intensity_column: str) -> tuple[list[float], list[float]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for raw in lines:
        parts = _split_data_line(raw)
        if not parts:
            continue
        if header is None and any(not _is_number(part) for part in parts):
            header = parts
            continue
        if header is not None and len(parts) >= len(header):
            rows.append({name: parts[i] for i, name in enumerate(header)})
    if rows:
        if energy_column not in rows[0]:
            raise ValueError(f"Energy column {energy_column!r} not found in {path}")
        if intensity_column not in rows[0]:
            raise ValueError(f"Intensity column {intensity_column!r} not found in {path}")
        return (
            [float(row[energy_column]) for row in rows],
            [float(row[intensity_column]) for row in rows],
        )

    # Fallback for simple two-column whitespace tables.
    energies: list[float] = []
    intensities: list[float] = []
    for raw in lines:
        parts = _split_data_line(raw)
        if len(parts) < 2 or not (_is_number(parts[0]) and _is_number(parts[1])):
            continue
        energies.append(float(parts[0]))
        intensities.append(float(parts[1]))
    if not energies:
        raise ValueError(f"No numeric curve rows found in {path}")
    return energies, intensities


def read_spectrum(path: Path, *, label: str, kind: str, energy_column: str, intensity_column: str) -> Spectrum:
    energy, intensity = _read_delimited_curve(path.expanduser(), energy_column, intensity_column)
    if len(energy) != len(intensity) or not energy:
        raise ValueError(f"Invalid spectrum in {path}: energy/intensity lengths do not match")
    return Spectrum(
        label=label,
        kind=kind,
        energy=tuple(energy),
        intensity=tuple(intensity),
        source=str(path.expanduser()),
    )


def white_line_energy(spectrum: Spectrum, *, window: tuple[float, float] | None = None) -> float:
    candidates: list[tuple[float, float]] = []
    for energy, intensity in zip(spectrum.energy, spectrum.intensity):
        if window is not None and not (window[0] <= energy <= window[1]):
            continue
        candidates.append((energy, intensity))
    if not candidates:
        raise ValueError(f"No points from {spectrum.label!r} inside white-line window {window}")
    return max(candidates, key=lambda row: row[1])[0]


def align_spectra(
    spectra: list[Spectrum],
    *,
    experimental_index: int = 0,
    align: str = "white-line",
    white_line_window: tuple[float, float] | None = None,
) -> list[AlignedSpectrum]:
    if not spectra:
        raise ValueError("At least one spectrum is required")
    if not (0 <= experimental_index < len(spectra)):
        raise ValueError("experimental_index is outside the spectra list")
    reference = spectra[experimental_index]
    if align not in {"white-line", "none"}:
        raise ValueError("align must be 'white-line' or 'none'")
    reference_white = white_line_energy(reference, window=white_line_window) if align == "white-line" else 0.0
    aligned: list[AlignedSpectrum] = []
    for idx, spectrum in enumerate(spectra):
        wl = white_line_energy(spectrum, window=white_line_window) if align == "white-line" else 0.0
        shift = 0.0 if align == "none" or idx == experimental_index else reference_white - wl
        aligned.append(
            AlignedSpectrum(
                spectrum=spectrum,
                white_line_energy=wl,
                energy_shift=shift,
                energy_aligned=tuple(energy + shift for energy in spectrum.energy),
            )
        )
    return aligned


def write_overlay_csv(path: Path, aligned: list[AlignedSpectrum]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "kind",
                "source",
                "energy_raw",
                "energy_aligned",
                "intensity",
                "white_line_energy_raw",
                "energy_shift",
            ]
        )
        for item in aligned:
            spectrum = item.spectrum
            for raw, aligned_energy, intensity in zip(spectrum.energy, item.energy_aligned, spectrum.intensity):
                writer.writerow(
                    [
                        spectrum.label,
                        spectrum.kind,
                        spectrum.source,
                        f"{raw:.10g}",
                        f"{aligned_energy:.10g}",
                        f"{intensity:.10g}",
                        f"{item.white_line_energy:.10g}",
                        f"{item.energy_shift:.10g}",
                    ]
                )


def filter_aligned_spectra(
    aligned: list[AlignedSpectrum],
    *,
    energy_window: tuple[float, float] | None,
) -> list[AlignedSpectrum]:
    if energy_window is None:
        return aligned
    clipped: list[AlignedSpectrum] = []
    for item in aligned:
        keep = [
            (raw_energy, aligned_energy, intensity)
            for raw_energy, aligned_energy, intensity in zip(item.spectrum.energy, item.energy_aligned, item.spectrum.intensity)
            if energy_window[0] <= aligned_energy <= energy_window[1]
        ]
        if not keep:
            raise ValueError(f"No points from {item.spectrum.label!r} inside aligned energy window {energy_window}")
        raw_energy, energy_aligned, intensity = zip(*keep)
        clipped.append(
            AlignedSpectrum(
                spectrum=Spectrum(
                    label=item.spectrum.label,
                    kind=item.spectrum.kind,
                    energy=tuple(raw_energy),
                    intensity=tuple(intensity),
                    source=item.spectrum.source,
                ),
                white_line_energy=item.white_line_energy,
                energy_shift=item.energy_shift,
                energy_aligned=tuple(energy_aligned),
            )
        )
    return clipped


def _svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_overlay_svg(
    path: Path,
    aligned: list[AlignedSpectrum],
    *,
    title: str = "XANES overlay",
    x_label: str = "Aligned energy (eV)",
    y_label: str = "Normalized absorption",
    experimental_style: str = "dashed",
) -> None:
    if not aligned:
        raise ValueError("No aligned spectra to plot")
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [x for item in aligned for x in item.energy_aligned]
    y_values = [y for item in aligned for y in item.spectrum.intensity]
    xmin, xmax = min(x_values), max(x_values)
    ymin, ymax = min(y_values), max(y_values)
    if math.isclose(xmin, xmax):
        xmin -= 1.0
        xmax += 1.0
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    ypad = 0.08 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad
    width, height = 940, 580
    ml, mr, mt, mb = 86, 42, 64, 76
    pw, ph = width - ml - mr, height - mt - mb

    def sx(x: float) -> float:
        return ml + (x - xmin) / (xmax - xmin) * pw

    def sy(y: float) -> float:
        return mt + (ymax - y) / (ymax - ymin) * ph

    colors = ["#1f5a9d", "#c47a19", "#4f7f4f", "#8a4f9e", "#59636f"]
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{ml}" y="34" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700" fill="#2f3640">{_svg_escape(title)}</text>',
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = ymin + frac * (ymax - ymin)
        y = sy(yv)
        lines.append(f'<line x1="{ml}" x2="{width - mr}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e8ebef" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 10}" y="{y + 4:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="12" text-anchor="end" fill="#59636f">{yv:.2g}</text>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        xv = xmin + frac * (xmax - xmin)
        x = sx(xv)
        lines.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{mt}" y2="{height - mb}" stroke="#f1f3f5" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - mb + 24}" font-family="Arial, Helvetica, sans-serif" font-size="12" text-anchor="middle" fill="#59636f">{xv:.3g}</text>')
    lines.append(f'<line x1="{ml}" x2="{width - mr}" y1="{height - mb}" y2="{height - mb}" stroke="#303640" stroke-width="1.3"/>')
    lines.append(f'<line x1="{ml}" x2="{ml}" y1="{mt}" y2="{height - mb}" stroke="#303640" stroke-width="1.3"/>')
    lines.append(f'<text x="{ml + pw / 2:.1f}" y="{height - 24}" font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#2f3640">{_svg_escape(x_label)}</text>')
    lines.append(f'<text x="24" y="{mt + ph / 2:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="600" text-anchor="middle" fill="#2f3640" transform="rotate(-90 24 {mt + ph / 2:.1f})">{_svg_escape(y_label)}</text>')

    legend_y = mt + 10
    for idx, item in enumerate(aligned):
        color = colors[idx % len(colors)]
        spectrum = item.spectrum
        points = [(sx(x), sy(y)) for x, y in zip(item.energy_aligned, spectrum.intensity)]
        is_exp = spectrum.kind.lower() in {"exp", "experiment", "experimental"}
        dash = ' stroke-dasharray="7 5"' if is_exp and experimental_style == "dashed" else ""
        width_attr = "3.0" if is_exp else "2.2"
        lines.append(f'<polyline points="{_polyline(points)}" fill="none" stroke="{color}" stroke-width="{width_attr}" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        if is_exp and experimental_style == "hollow-points":
            step = max(1, len(points) // 24)
            for x, y in points[::step]:
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="#ffffff" stroke="{color}" stroke-width="1.4"/>')
        lx = width - 350
        ly = legend_y + 24 * idx
        lines.append(f'<line x1="{lx}" x2="{lx + 34}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="{width_attr}"{dash}/>')
        lines.append(f'<text x="{lx + 44}" y="{ly + 4}" font-family="Arial, Helvetica, sans-serif" font-size="12.5" fill="#2f3640">{_svg_escape(spectrum.label)} ({spectrum.kind}, shift {item.energy_shift:.3g} eV)</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_window(value: str) -> tuple[float, float] | None:
    if not value:
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Window must contain two numbers: MIN MAX")
    lo, hi = float(parts[0]), float(parts[1])
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def _effective_energy_window(args: argparse.Namespace) -> tuple[float, float] | None:
    if args.no_energy_window:
        return None
    if args.energy_window:
        return _parse_window(args.energy_window)
    if args.mode == "xanes":
        return XANES_DEFAULT_ENERGY_WINDOW
    return None


def _load_from_spec(spec: str, default_kind: str, energy_column: str, intensity_column: str) -> Spectrum:
    parts = spec.split(":", 2)
    if len(parts) == 1:
        path = Path(parts[0])
        return read_spectrum(path, label=path.stem, kind=default_kind, energy_column=energy_column, intensity_column=intensity_column)
    if len(parts) == 2:
        label, path_text = parts
        path = Path(path_text)
        return read_spectrum(path, label=label, kind=default_kind, energy_column=energy_column, intensity_column=intensity_column)
    label, kind, path_text = parts
    return read_spectrum(Path(path_text), label=label, kind=kind, energy_column=energy_column, intensity_column=intensity_column)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["xanes", "exafs"],
        default="xanes",
        help=(
            "Plotting domain policy. XANES clips aligned energy to -200..300 eV by default; "
            "EXAFS leaves the energy axis unclipped unless --energy-window is supplied."
        ),
    )
    parser.add_argument("--exp", required=True, help="Experimental spectrum as PATH or LABEL:PATH.")
    parser.add_argument(
        "--sim",
        action="append",
        default=[],
        help="Simulated spectrum as PATH, LABEL:PATH, or LABEL:KIND:PATH. Repeat for FDMNES/OCEAN/Molcas.",
    )
    parser.add_argument("--energy-column", default="energy_rel_eV")
    parser.add_argument("--intensity-column", default="intensity")
    parser.add_argument("--exp-energy-column", default="", help="Override energy column for --exp.")
    parser.add_argument("--exp-intensity-column", default="", help="Override intensity column for --exp.")
    parser.add_argument("--align", choices=["white-line", "none"], default="white-line")
    parser.add_argument("--white-line-window", default="", help="Optional raw-energy window used to find white-line maxima, e.g. '0 35'.")
    parser.add_argument(
        "--energy-window",
        default="",
        help=(
            "Aligned-energy output/plot window in eV, e.g. '-200 300'. "
            "Default for --mode xanes is '-200 300'; default for --mode exafs is no eV clipping."
        ),
    )
    parser.add_argument("--no-energy-window", action="store_true", help="Disable the default XANES aligned-energy clipping.")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-svg", type=Path)
    parser.add_argument("--title", default="XANES overlay")
    parser.add_argument("--exp-style", choices=["dashed", "hollow-points"], default="dashed")
    return parser


def overlay_main(args: argparse.Namespace) -> dict[str, Any]:
    exp = _load_from_spec(
        args.exp,
        "experiment",
        args.exp_energy_column or args.energy_column,
        args.exp_intensity_column or args.intensity_column,
    )
    spectra = [exp]
    spectra.extend(_load_from_spec(spec, "simulation", args.energy_column, args.intensity_column) for spec in args.sim)
    aligned = align_spectra(spectra, align=args.align, white_line_window=_parse_window(args.white_line_window))
    energy_window = _effective_energy_window(args)
    output_aligned = filter_aligned_spectra(aligned, energy_window=energy_window)
    write_overlay_csv(args.out_csv, output_aligned)
    if args.out_svg:
        write_overlay_svg(args.out_svg, output_aligned, title=args.title, experimental_style=args.exp_style)
    summary = {
        "schema": "atomi.xafs.xanes_overlay.v1",
        "mode": args.mode,
        "alignment": args.align,
        "white_line_window": args.white_line_window or None,
        "energy_window_aligned_eV": list(energy_window) if energy_window is not None else None,
        "out_csv": str(args.out_csv),
        "out_svg": str(args.out_svg) if args.out_svg else None,
        "spectra": [
            {
                "label": item.spectrum.label,
                "kind": item.spectrum.kind,
                "source": item.spectrum.source,
                "white_line_energy": item.white_line_energy,
                "energy_shift": item.energy_shift,
            }
            for item in aligned
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overlay_main(args)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
