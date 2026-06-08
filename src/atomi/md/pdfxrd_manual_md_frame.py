#!/usr/bin/env python3
"""Manual PDF/RDF and powder-XRD inspection for selected MD frames.

Examples:

  # VASP XDATCAR: compare first, middle, and last frames.
  ~/m_lammps_env/bin/python scripts/manual_md_frame_pdf_xrd.py \
    --engine vasp-xdatcar --poscar POSCAR --xdatcar XDATCAR \
    --species-order Na,U,Cl --frame 0 --frame 500 --frame -1 \
    --outdir /tmp/md_xrd_pdf --prefix nacl_ucl3_tail

  # LAMMPS dump with orthogonal/triclinic dump box from ITEM: BOX BOUNDS.
  ~/m_lammps_env/bin/python scripts/manual_md_frame_pdf_xrd.py \
    --engine lammps-dump --dump traj.dump --type-elements 1=K,2=Cl \
    --frame 0 --frame -1 --outdir /tmp/kcl_frames

  # CP2K XYZ needs an explicit cell, as three semicolon-separated vectors.
  ~/m_lammps_env/bin/python scripts/manual_md_frame_pdf_xrd.py \
    --engine cp2k-xyz --xyz project-pos-1.xyz \
    --cell "12.5,0,0;0,12.5,0;0,0,12.5" --frame -1 --outdir /tmp/cp2k_frame
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
    except Exception as exc:  # pragma: no cover
        raise SystemExit("This script needs numpy. Run with ~/m_lammps_env/bin/python.") from exc
    return np


def _import_atomi_backend():
    try:
        from atomi.md.phase_order_guard import (  # type: ignore
            diffraction_form_factors,
            simulated_powder_xrd_from_frames,
            vasp_frames_from_xdatcar,
            write_xrd_multi_overlay_plot,
        )
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Could not import atomi.md.phase_order_guard. Run with ~/m_lammps_env/bin/python.") from exc
    return {
        "diffraction_form_factors": diffraction_form_factors,
        "simulated_powder_xrd_from_frames": simulated_powder_xrd_from_frames,
        "vasp_frames_from_xdatcar": vasp_frames_from_xdatcar,
        "write_xrd_multi_overlay_plot": write_xrd_multi_overlay_plot,
    }


def species_order_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_type_elements(values: list[str] | None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for item in values or []:
        left, right = item.split("=", 1)
        mapping[int(left)] = right
    return mapping


def parse_cell(value: str):
    np = _import_numpy()
    rows = []
    for row in value.split(";"):
        rows.append([float(x) for x in row.split(",")])
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise SystemExit("--cell must look like 'a,b,c;d,e,f;g,h,i'")
    return np.asarray(rows, dtype=float)


def normalize_curve(y):
    np = _import_numpy()
    arr = np.asarray(y, dtype=float)
    arr = arr - float(np.min(arr))
    maxv = float(np.max(arr))
    return arr / maxv if maxv > 0 else arr


def pbc_delta(frac_a, frac_b):
    np = _import_numpy()
    d = np.asarray(frac_b, dtype=float) - np.asarray(frac_a, dtype=float)
    return d - np.rint(d)


def frame_from_cart(symbols: list[str], cart, cell, index: int) -> dict[str, Any]:
    np = _import_numpy()
    cell = np.asarray(cell, dtype=float)
    cart = np.asarray(cart, dtype=float)
    frac = cart @ np.linalg.inv(cell)
    frac = frac - np.floor(frac)
    return {
        "index": index,
        "symbols": symbols,
        "frac": frac,
        "cell": cell,
        "volume_A3": float(abs(np.linalg.det(cell))),
    }


def read_cp2k_xyz(path: Path, cell) -> list[dict[str, Any]]:
    np = _import_numpy()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    frames = []
    i = 0
    idx = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        nat = int(lines[i].strip())
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        symbols = []
        cart = []
        for line in lines[i + 2 : i + 2 + nat]:
            parts = line.split()
            symbols.append(parts[0])
            cart.append([float(parts[1]), float(parts[2]), float(parts[3])])
        frames.append(frame_from_cart(symbols, np.asarray(cart), cell, idx))
        idx += 1
        i += nat + 2
    return frames


def read_lammps_dump(path: Path, type_elements: dict[int, str]) -> list[dict[str, Any]]:
    np = _import_numpy()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        timestep = int(lines[i + 1].strip())
        nat = int(lines[i + 3].strip())
        bounds_header = lines[i + 4]
        bounds = [list(map(float, lines[i + 5 + k].split()[:3])) for k in range(3)]
        if "xy xz yz" in bounds_header:
            xlo_bound, xhi_bound, xy = bounds[0]
            ylo_bound, yhi_bound, xz = bounds[1]
            zlo_bound, zhi_bound, yz = bounds[2]
            xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
            xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
            ylo = ylo_bound - min(0.0, yz)
            yhi = yhi_bound - max(0.0, yz)
            zlo, zhi = zlo_bound, zhi_bound
            cell = np.asarray([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]], dtype=float)
            origin = np.asarray([xlo, ylo, zlo], dtype=float)
        else:
            xlo, xhi = bounds[0][:2]
            ylo, yhi = bounds[1][:2]
            zlo, zhi = bounds[2][:2]
            cell = np.asarray([[xhi - xlo, 0, 0], [0, yhi - ylo, 0], [0, 0, zhi - zlo]], dtype=float)
            origin = np.asarray([xlo, ylo, zlo], dtype=float)
        atom_header = lines[i + 8].split()[2:]
        rows = lines[i + 9 : i + 9 + nat]
        id_idx = atom_header.index("id") if "id" in atom_header else None
        type_idx = atom_header.index("type")
        coord_keys = ("x", "y", "z") if "x" in atom_header else ("xs", "ys", "zs")
        coord_idx = [atom_header.index(k) for k in coord_keys]
        atoms = []
        for row in rows:
            parts = row.split()
            atom_id = int(parts[id_idx]) if id_idx is not None else len(atoms)
            typ = int(parts[type_idx])
            coord = [float(parts[k]) for k in coord_idx]
            atoms.append((atom_id, typ, coord))
        atoms.sort(key=lambda item: item[0])
        symbols = [type_elements[typ] for _, typ, _ in atoms]
        coords = np.asarray([coord for _, _, coord in atoms], dtype=float)
        if coord_keys[0] == "xs":
            cart = coords @ cell
        else:
            cart = coords - origin
        frames.append(frame_from_cart(symbols, cart, cell, timestep))
        i += 9 + nat
    return frames


def compute_pdf_curves(frame: dict[str, Any], *, rmax: float, dr: float, partials: bool) -> dict[str, tuple[Any, Any]]:
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
            dist = float(np.linalg.norm(pbc_delta(frac[i], frac[j]) @ cell))
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


def select_frames(frames: list[dict[str, Any]], selected: list[int]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    n = len(frames)
    for raw in selected:
        idx = raw if raw >= 0 else n + raw
        if idx < 0 or idx >= n:
            raise SystemExit(f"Frame index {raw} out of range for {n} frames")
        out.append((f"frame_{raw}", frames[idx]))
    return out


def write_curves_csv(path: Path, x, curves: dict[str, Any], x_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([x_label, *curves.keys()])
        for i, xv in enumerate(x):
            writer.writerow([float(xv), *[float(curves[name][i]) for name in curves]])


def write_overlay_svg(path: Path, *, title: str, x, curves: dict[str, Any], x_label: str, y_label: str) -> None:
    np = _import_numpy()
    colors = ["#1f77b4", "#d55e00", "#009e73", "#cc79a7", "#7f7f7f", "#9467bd", "#8c564b", "#e69f00"]
    path.parent.mkdir(parents=True, exist_ok=True)
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
        curve_markup.append(f'<polyline points="{polyline(y)}" fill="none" stroke="{color}" stroke-width="2.0"/>')
        ly = 28 + idx * 17
        legend_markup.append(f'<line x1="535" y1="{ly}" x2="570" y2="{ly}" stroke="{color}" stroke-width="2.2"/>')
        legend_markup.append(f'<text x="578" y="{ly + 4}" font-family="Arial" font-size="12">{label}</text>')
    ticks = np.linspace(x_min, x_max, 5)
    tick_markup = []
    for tick in ticks:
        px = x0 + (float(tick) - x_min) / max(x_max - x_min, 1.0e-12) * width
        tick_markup.append(f'<line x1="{px:.2f}" y1="{y0 + height:.2f}" x2="{px:.2f}" y2="{y0 + height + 5:.2f}" stroke="#444"/>')
        tick_markup.append(f'<text x="{px:.2f}" y="{y0 + height + 22:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{tick:.1f}</text>')
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
    ap.add_argument("--engine", choices=["vasp-xdatcar", "lammps-dump", "cp2k-xyz"], required=True)
    ap.add_argument("--poscar")
    ap.add_argument("--xdatcar")
    ap.add_argument("--dump")
    ap.add_argument("--xyz")
    ap.add_argument("--cell", help="CP2K XYZ cell: 'a,b,c;d,e,f;g,h,i'.")
    ap.add_argument("--species-order", help="Comma-separated species order, e.g. Na,U,Cl.")
    ap.add_argument("--type-elements", action="append", help="LAMMPS type map, e.g. 1=K. Repeatable.")
    ap.add_argument("--frame", action="append", type=int, default=[-1], help="Frame index to plot. Negative allowed.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="md_frame_pdf_xrd")
    ap.add_argument("--two-theta-min", type=float, default=10.0)
    ap.add_argument("--two-theta-max", type=float, default=90.0)
    ap.add_argument("--two-theta-step", type=float, default=0.05)
    ap.add_argument("--wavelength-a", type=float, default=1.5406)
    ap.add_argument("--coherence-radius-a", type=float, default=8.0)
    ap.add_argument("--smooth-sigma-deg", type=float, default=0.18)
    ap.add_argument("--rmax", type=float, default=10.0)
    ap.add_argument("--dr", type=float, default=0.02)
    ap.add_argument("--partials", action="store_true")
    args = ap.parse_args(argv)

    np = _import_numpy()
    backend = _import_atomi_backend()
    species_order = species_order_arg(args.species_order)

    if args.engine == "vasp-xdatcar":
        if not args.poscar or not args.xdatcar:
            raise SystemExit("--engine vasp-xdatcar requires --poscar and --xdatcar")
        frames, meta = backend["vasp_frames_from_xdatcar"](
            poscar=Path(args.poscar).expanduser(),
            xdatcar=Path(args.xdatcar).expanduser(),
            species_order=species_order,
        )
        all_species = species_order or list(meta["species_order"])
    elif args.engine == "lammps-dump":
        if not args.dump:
            raise SystemExit("--engine lammps-dump requires --dump")
        frames = read_lammps_dump(Path(args.dump).expanduser(), parse_type_elements(args.type_elements))
        all_species = species_order or sorted({sym for frame in frames for sym in frame["symbols"]})
    else:
        if not args.xyz or not args.cell:
            raise SystemExit("--engine cp2k-xyz requires --xyz and --cell")
        frames = read_cp2k_xyz(Path(args.xyz).expanduser(), parse_cell(args.cell))
        all_species = species_order or sorted({sym for frame in frames for sym in frame["symbols"]})

    chosen = select_frames(frames, args.frame)
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    angles = np.arange(args.two_theta_min, args.two_theta_max + args.two_theta_step, args.two_theta_step)
    q = 4.0 * np.pi * np.sin(np.deg2rad(0.5 * angles)) / args.wavelength_a
    form_factors, _ = backend["diffraction_form_factors"](all_species, q, scattering="xraydb", custom=None)

    xrd_curves = {}
    for label, frame in chosen:
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
    try:
        backend["write_xrd_multi_overlay_plot"](
            outdir / f"{args.prefix}_xrd.png",
            angles,
            xrd_curves,
            title=f"{args.prefix}: selected-frame powder XRD",
        )
    except Exception:
        pass
    write_overlay_svg(
        outdir / f"{args.prefix}_xrd.svg",
        title=f"{args.prefix}: selected-frame powder XRD",
        x=angles,
        curves=xrd_curves,
        x_label="2theta (deg, Cu K-alpha)",
        y_label="Normalized intensity",
    )

    pdf_curves = {}
    r = None
    for label, frame in chosen:
        curves = compute_pdf_curves(frame, rmax=args.rmax, dr=args.dr, partials=args.partials)
        for curve_label, (r_values, values) in curves.items():
            r = r_values
            pdf_curves[f"{label}:{curve_label}"] = values
    write_curves_csv(outdir / f"{args.prefix}_pdf.csv", r, pdf_curves, "r_A")
    write_overlay_svg(
        outdir / f"{args.prefix}_pdf.svg",
        title=f"{args.prefix}: selected-frame RDF/PDF",
        x=r,
        curves=pdf_curves,
        x_label="r (A)",
        y_label="g(r), normalized for display",
    )
    print(outdir / f"{args.prefix}_xrd.svg")
    print(outdir / f"{args.prefix}_pdf.svg")


if __name__ == "__main__":
    main()
