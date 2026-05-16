#!/usr/bin/env python3
"""LAMMPS trajectory RDF/PDF/S(Q)/F(Q) analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from atomi.core.archive import archive_output_dir, default_archive_path
from atomi.lammps.thermo_series import (
    collect_config_paths,
    discover_npt_records_from_md_root,
    discover_production_records,
    filter_records_by_T,
)


def parse_type_map(items: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected type map item like 1=U, got: {item}")
        key, value = item.split("=", 1)
        out[int(key.strip())] = value.strip()
    return out


def parse_weights(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected weight item like U=92, got: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def require_ase():
    try:
        from ase.io import read, write
    except ImportError as exc:
        raise RuntimeError(
            "LAMMPS RDF/PDF analysis needs ASE. Install Atomi with the materials "
            "extra or install ase in the active environment."
        ) from exc
    return read, write


def get_lammps_types(atoms) -> np.ndarray:
    for key in ("type", "types", "atom_types"):
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=int)
    return np.asarray(atoms.get_atomic_numbers(), dtype=int)


def apply_type_map(frames: list, type_map: dict[int, str]) -> list:
    mapped = []
    for atoms in frames:
        item = atoms.copy()
        types = get_lammps_types(item)
        symbols = []
        for atom_type in types:
            atom_type = int(atom_type)
            if atom_type not in type_map:
                raise KeyError(f"LAMMPS atom type {atom_type} not found in --type-map")
            symbols.append(type_map[atom_type])
        item.set_chemical_symbols(symbols)
        mapped.append(item)
    return mapped


def minimum_image_deltas(frac_diff: np.ndarray) -> np.ndarray:
    return frac_diff - np.rint(frac_diff)


def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    integrator = getattr(np, "trapezoid", None)
    if integrator is None:
        integrator = np.trapz
    return float(integrator(y, x))


def average_frame(frames: list):
    if len(frames) == 1:
        return frames[0].copy()

    ref = frames[0]
    natoms = len(ref)
    symbols = ref.get_chemical_symbols()
    for atoms in frames:
        if len(atoms) != natoms:
            raise ValueError("All frames must have the same atom count for averaging.")
        if atoms.get_chemical_symbols() != symbols:
            raise ValueError("Species/order changed across frames; cannot average safely.")

    cells = np.array([atoms.cell.array for atoms in frames], dtype=float)
    cell_avg = np.mean(cells, axis=0)
    ref_cell = ref.cell.array
    ref_inv = np.linalg.inv(ref_cell)
    ref_frac = ref.get_positions() @ ref_inv

    fracs = []
    for atoms in frames:
        inv_cell = np.linalg.inv(atoms.cell.array)
        frac = atoms.get_positions() @ inv_cell
        fracs.append(ref_frac + minimum_image_deltas(frac - ref_frac))

    avg = ref.copy()
    avg.set_cell(cell_avg, scale_atoms=False)
    avg.set_positions(np.mean(np.asarray(fracs), axis=0) @ cell_avg)
    return avg


def read_frames_from_traj(path: Path, start: Optional[int], stop: Optional[int], step: Optional[int]) -> list:
    read, _ = require_ase()
    frames = read(path, index=slice(start, stop, step))
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise RuntimeError(f"No frames could be read from {path}")
    return frames


def read_frames_from_dump(
    dump: Path,
    dump_format: str,
    type_map: dict[int, str],
    dt_ps: float,
    dump_every: int,
    window_ps: Optional[float],
) -> tuple[list, dict]:
    read, _ = require_ase()
    frames = read(dump, index=":", format=dump_format)
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise RuntimeError(f"No frames could be read from {dump}")

    frames = apply_type_map(frames, type_map)
    dt_frame_ps = dt_ps * dump_every
    if dt_frame_ps <= 0:
        raise ValueError("--dt times --dump-every must be positive")

    total_time_ps = (len(frames) - 1) * dt_frame_ps if len(frames) > 1 else 0.0
    if window_ps is None or total_time_ps <= window_ps:
        selected = frames
    else:
        n_needed = min(int(math.ceil(window_ps / dt_frame_ps)) + 1, len(frames))
        selected = frames[-n_needed:]

    summary = {
        "dump_file": str(dump.resolve()),
        "dump_format": dump_format,
        "type_map": type_map,
        "dt_ps": dt_ps,
        "dump_every_steps": dump_every,
        "dt_frame_ps": dt_frame_ps,
        "window_ps_requested": window_ps,
        "n_total_frames": len(frames),
        "total_time_ps_available": total_time_ps,
        "n_selected_frames": len(selected),
        "window_ps_used": (len(selected) - 1) * dt_frame_ps if len(selected) > 1 else 0.0,
        "selected_frame_indices_0based": list(range(len(frames) - len(selected), len(frames))),
    }
    return selected, summary


def write_selected_frames(outdir: Path, prefix: str, frames: list) -> dict[str, str]:
    _, write = require_ase()
    out_multi = outdir / f"{prefix}_lastwindow.extxyz"
    out_last = outdir / f"{prefix}_lastframe.extxyz"
    out_avg = outdir / f"{prefix}_avgframe.extxyz"
    write(out_multi, frames, format="extxyz")
    write(out_last, frames[-1], format="extxyz")
    write(out_avg, average_frame(frames), format="extxyz")
    return {
        "multi_frame_extxyz": str(out_multi),
        "last_frame_extxyz": str(out_last),
        "avg_frame_extxyz": str(out_avg),
    }


def compute_partial_histograms(frames: list, species_order: list[str], r_edges: np.ndarray):
    nbins = len(r_edges) - 1
    pair_hist = {
        (a, b): np.zeros(nbins, dtype=float)
        for ia, a in enumerate(species_order)
        for b in species_order[ia:]
    }
    composition_sum: Counter[str] = Counter()
    volume_sum = 0.0

    for atoms in frames:
        pos = atoms.get_positions()
        cell = np.asarray(atoms.cell.array, dtype=float)
        inv_cell = np.linalg.inv(cell)
        symbols = np.asarray(atoms.get_chemical_symbols())
        n_atoms = len(symbols)

        frac = pos @ inv_cell
        dfrac = minimum_image_deltas(frac[:, None, :] - frac[None, :, :])
        distances = np.linalg.norm(dfrac @ cell, axis=2)
        iu = np.triu_indices(n_atoms, k=1)
        d = distances[iu]
        s1 = symbols[iu[0]]
        s2 = symbols[iu[1]]

        composition_sum.update(Counter(symbols))
        volume_sum += atoms.get_volume()

        for ia, a in enumerate(species_order):
            for b in species_order[ia:]:
                if a == b:
                    mask = (s1 == a) & (s2 == a)
                else:
                    mask = ((s1 == a) & (s2 == b)) | ((s1 == b) & (s2 == a))
                hist, _ = np.histogram(d[mask], bins=r_edges)
                pair_hist[(a, b)] += hist

    n_frames = len(frames)
    avg_counts = {s: composition_sum[s] / n_frames for s in species_order}
    return pair_hist, avg_counts, volume_sum / n_frames, n_frames


def normalize_partial_rdfs(
    pair_hist: dict[tuple[str, str], np.ndarray],
    avg_counts: dict[str, float],
    avg_volume: float,
    n_frames: int,
    r_edges: np.ndarray,
):
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = np.diff(r_edges)
    shell = 4.0 * np.pi * r**2 * dr
    partial = {}
    for (a, b), hist in pair_hist.items():
        na = avg_counts[a]
        nb = avg_counts[b]
        if a == b:
            denom = 0.5 * na * (na - 1.0)
        else:
            denom = na * nb
        partial[(a, b)] = hist * avg_volume / (n_frames * denom * shell)
    return r, partial


def concentrations(avg_counts: dict[str, float], species_order: list[str]) -> dict[str, float]:
    n_total = sum(avg_counts.values())
    return {s: avg_counts[s] / n_total for s in species_order}


def weighted_total_gr_constant(
    species_order: list[str],
    partial: dict[tuple[str, str], np.ndarray],
    avg_counts: dict[str, float],
    weights: dict[str, float],
):
    conc = concentrations(avg_counts, species_order)
    denom = sum(conc[s] * weights[s] for s in species_order) ** 2
    total = None
    for ia, a in enumerate(species_order):
        for b in species_order[ia:]:
            pref = conc[a] * weights[a] * conc[b] * weights[b] / denom
            if a != b:
                pref *= 2.0
            value = pref * partial[(a, b)]
            total = value if total is None else total + value
    return total, conc


def gr_to_gr_direct(r: np.ndarray, g_total: np.ndarray, rho0: float) -> np.ndarray:
    return 4.0 * np.pi * rho0 * r * (g_total - 1.0)


def partials_to_sq_constant(
    species_order: list[str],
    partial: dict[tuple[str, str], np.ndarray],
    avg_counts: dict[str, float],
    weights: dict[str, float],
    rho0: float,
    r: np.ndarray,
    q_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    conc = concentrations(avg_counts, species_order)
    denom = sum(conc[s] * weights[s] for s in species_order) ** 2
    sq = np.ones_like(q_values, dtype=float)
    for ia, a in enumerate(species_order):
        for b in species_order[ia:]:
            h_ab = partial[(a, b)] - 1.0
            pref = conc[a] * weights[a] * conc[b] * weights[b] / denom
            if a != b:
                pref *= 2.0
            part = np.zeros_like(q_values, dtype=float)
            for iq, q in enumerate(q_values):
                qr = q * r
                sinc = np.ones_like(r)
                nonzero = qr != 0.0
                sinc[nonzero] = np.sin(qr[nonzero]) / qr[nonzero]
                part[iq] = 4.0 * np.pi * rho0 * trapz_compat(r**2 * h_ab * sinc, r)
            sq += pref * part
    return sq, conc


def atomic_number_weights(species_order: list[str]) -> dict[str, float]:
    try:
        from ase.data import atomic_numbers
    except ImportError as exc:
        raise RuntimeError("X-ray fallback weights require ASE atomic numbers.") from exc
    return {s: float(atomic_numbers[s]) for s in species_order}


def neutron_weights(species_order: list[str]) -> dict[str, float]:
    try:
        import periodictable as pt
    except ImportError as exc:
        raise RuntimeError(
            "Neutron scattering mode needs periodictable. Install periodictable "
            "or use --scattering xray/custom."
        ) from exc
    weights = {}
    for symbol in species_order:
        b_coh = getattr(pt, symbol).neutron.b_c
        if b_coh is None:
            raise ValueError(f"No coherent neutron scattering length found for {symbol}")
        weights[symbol] = float(b_coh)
    return weights


def xray_form_factors(species_order: list[str], q_values: np.ndarray) -> Optional[dict[str, np.ndarray]]:
    try:
        import xraydb
    except ImportError:
        return None
    q_xraydb = q_values / (4.0 * np.pi)
    return {s: np.asarray(xraydb.f0(s, q_xraydb), dtype=float) for s in species_order}


def partials_to_sq_xray(
    species_order: list[str],
    partial: dict[tuple[str, str], np.ndarray],
    avg_counts: dict[str, float],
    form_factors: dict[str, np.ndarray],
    rho0: float,
    r: np.ndarray,
    q_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    conc = concentrations(avg_counts, species_order)
    f_bar = np.zeros_like(q_values, dtype=float)
    for symbol in species_order:
        f_bar += conc[symbol] * form_factors[symbol]
    denom = f_bar**2
    sq = np.ones_like(q_values, dtype=float)
    for ia, a in enumerate(species_order):
        for b in species_order[ia:]:
            h_ab = partial[(a, b)] - 1.0
            pref = conc[a] * conc[b] * form_factors[a] * form_factors[b] / denom
            if a != b:
                pref *= 2.0
            part = np.zeros_like(q_values, dtype=float)
            for iq, q in enumerate(q_values):
                qr = q * r
                sinc = np.ones_like(r)
                nonzero = qr != 0.0
                sinc[nonzero] = np.sin(qr[nonzero]) / qr[nonzero]
                part[iq] = 4.0 * np.pi * rho0 * trapz_compat(r**2 * h_ab * sinc, r)
            sq += pref * part
    return sq, conc


def apply_window(q_values: np.ndarray, fq: np.ndarray, mode: str, qmax: float):
    if mode == "none":
        return fq.copy(), np.ones_like(fq)
    if mode == "lorch":
        x = np.pi * q_values / qmax
        window = np.ones_like(fq)
        nonzero = x != 0.0
        window[nonzero] = np.sin(x[nonzero]) / x[nonzero]
        return fq * window, window
    raise ValueError(f"Unknown window: {mode}")


def fq_to_gr(q_values: np.ndarray, fq_windowed: np.ndarray, r_out: np.ndarray) -> np.ndarray:
    gr = np.zeros_like(r_out, dtype=float)
    for i, r_value in enumerate(r_out):
        gr[i] = (2.0 / np.pi) * trapz_compat(fq_windowed * np.sin(q_values * r_value), q_values)
    return gr


def write_xy(path: Path, x: np.ndarray, y: np.ndarray, xname: str, yname: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {xname} {yname}\n")
        for xi, yi in zip(x, y):
            handle.write(f"{xi:.10e} {yi:.10e}\n")


def write_xy_plain(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for xi, yi in zip(x, y):
            handle.write(f"{xi:.10e} {yi:.10e}\n")


def write_pdfgui_gr(path: Path, r: np.ndarray, gr: np.ndarray, dr_unc: float, dgr: float) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# PDFgui/PDFfit2 format: r_A G_r dr dG_r\n")
        handle.write("# dr and dG_r are placeholder uncertainties from command options.\n")
        for ri, gi in zip(r, gr):
            handle.write(f"{ri:.10e} {gi:.10e} {dr_unc:.10e} {dgr:.10e}\n")


def write_fitting_exports(
    outdir: Path,
    prefix: str,
    r: np.ndarray,
    gr_direct: np.ndarray,
    r_from_fq: np.ndarray,
    gr_from_fq: np.ndarray,
    q_values: np.ndarray,
    sq: np.ndarray,
    fq: np.ndarray,
    fq_windowed: np.ndarray,
    pdfgui_dr_uncertainty: float,
    pdfgui_dgr: float,
) -> dict[str, str]:
    """Write explicit convention files for PDFgui and RMC-style fitting tools."""
    iq = sq - 1.0
    rmc_gr_direct = np.divide(gr_direct, r, out=np.zeros_like(gr_direct), where=r != 0.0)
    rmc_gr_from_fq = np.divide(gr_from_fq, r_from_fq, out=np.zeros_like(gr_from_fq), where=r_from_fq != 0.0)

    paths = {
        "pdfgui_GofR_direct_4col": outdir / f"{prefix}_pdfgui_GofR_direct_4col.gr",
        "pdfgui_GofR_from_FQ_4col": outdir / f"{prefix}_pdfgui_GofR_from_FQ_4col.gr",
        "rmcprofile_SQ": outdir / f"{prefix}_rmcprofile_SQ.dat",
        "rmcprofile_iQ": outdir / f"{prefix}_rmcprofile_iQ_Sminus1.dat",
        "rmcprofile_pdfgetx_FQ": outdir / f"{prefix}_rmcprofile_pdfgetx_FQ_QSminus1.dat",
        "rmcprofile_pdfgetx_FQ_windowed": outdir / f"{prefix}_rmcprofile_pdfgetx_FQ_QSminus1_windowed.dat",
        "rmcprofile_GofR_direct_flat": outdir / f"{prefix}_rmcprofile_GofR_direct_flat.dat",
        "rmcprofile_GofR_from_FQ_flat": outdir / f"{prefix}_rmcprofile_GofR_from_FQ_flat.dat",
    }
    write_pdfgui_gr(paths["pdfgui_GofR_direct_4col"], r, gr_direct, pdfgui_dr_uncertainty, pdfgui_dgr)
    write_pdfgui_gr(paths["pdfgui_GofR_from_FQ_4col"], r_from_fq, gr_from_fq, pdfgui_dr_uncertainty, pdfgui_dgr)
    write_xy_plain(paths["rmcprofile_SQ"], q_values, sq)
    write_xy_plain(paths["rmcprofile_iQ"], q_values, iq)
    write_xy_plain(paths["rmcprofile_pdfgetx_FQ"], q_values, fq)
    write_xy_plain(paths["rmcprofile_pdfgetx_FQ_windowed"], q_values, fq_windowed)
    write_xy_plain(paths["rmcprofile_GofR_direct_flat"], r, rmc_gr_direct)
    write_xy_plain(paths["rmcprofile_GofR_from_FQ_flat"], r_from_fq, rmc_gr_from_fq)
    return {key: str(path) for key, path in paths.items()}


def write_multi_csv(path: Path, xname: str, x: np.ndarray, columns: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([xname] + list(columns))
        for i, xi in enumerate(x):
            writer.writerow([xi] + [columns[key][i] for key in columns])


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


def read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            x_values.append(float(parts[0]))
            y_values.append(float(parts[1]))
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def plot_outputs(
    outdir: Path,
    prefix: str,
    r: np.ndarray,
    partial: dict[tuple[str, str], np.ndarray],
    g_total: np.ndarray,
    gr_direct: np.ndarray,
    r_from_fq: np.ndarray,
    gr_from_fq: np.ndarray,
    q_values: np.ndarray,
    sq: np.ndarray,
    fq: np.ndarray,
    fq_windowed: np.ndarray,
) -> list[str]:
    try:
        cache = Path(tempfile.gettempdir()) / "atomi-matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for (a, b), values in partial.items():
        ax.plot(r, values, label=f"{a}-{b}")
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g_ab(r)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = outdir / f"{prefix}_partial_rdfs.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r, g_total, label="weighted g(r)")
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = outdir / f"{prefix}_gtot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r, gr_direct, label="direct G(r)")
    ax.plot(r_from_fq, gr_from_fq, label="G(r) from F(Q)", linestyle="--")
    ax.set_xlabel("r (A)")
    ax.set_ylabel("G(r)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = outdir / f"{prefix}_pdf_gr.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(q_values, sq)
    axes[0].set_ylabel("S(Q)")
    axes[1].plot(q_values, fq, label="F(Q)")
    axes[1].plot(q_values, fq_windowed, label="windowed F(Q)", linestyle="--")
    axes[1].set_xlabel("Q (A^-1)")
    axes[1].set_ylabel("F(Q)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path = outdir / f"{prefix}_sq_fq.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))
    return written


def plot_series_overlay(
    path: Path,
    series: list[dict],
    file_key: str,
    title: str,
    x_label: str,
    y_label: str,
) -> bool:
    try:
        cache = Path(tempfile.gettempdir()) / "atomi-matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        return False

    available = [item for item in series if item.get(file_key)]
    if not available:
        return False

    temperatures = np.asarray([item["temperature"] for item in available], dtype=float)
    norm = Normalize(vmin=float(np.min(temperatures)), vmax=float(np.max(temperatures)))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7, 4))
    for item in available:
        x, y = read_xy(Path(item[file_key]))
        ax.plot(x, y, color=cmap(norm(item["temperature"])), linewidth=1.7, alpha=0.95)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label("Temperature (K)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def find_record_dump(record: dict) -> Path:
    log_path = Path(record["log_path"])
    chunk_dir = log_path.parent
    stage_name = str(record.get("stage_name", ""))
    patterns = [
        "dump.*.lammpstrj",
        "*.lammpstrj",
        "dump.*",
    ]

    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(chunk_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                candidates.append(path)
                seen.add(path)

    if not candidates:
        stage_dir = chunk_dir.parent
        for pattern in patterns:
            for path in sorted(stage_dir.glob(pattern)):
                if path.is_file() and path not in seen:
                    candidates.append(path)
                    seen.add(path)

    if not candidates:
        raise FileNotFoundError(f"No LAMMPS dump trajectory found near {log_path}")

    if stage_name:
        named = [path for path in candidates if stage_name in path.name]
        if named:
            candidates = named

    return max(candidates, key=lambda path: path.stat().st_mtime)


def record_dump_every(record: dict, default_dump_every: Optional[int]) -> int:
    stage = record.get("stage") or {}
    if stage.get("dump_every") is not None:
        return int(stage["dump_every"])
    config_path = record.get("config_path")
    if config_path:
        try:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if cfg.get("dump_every") is not None:
                return int(cfg["dump_every"])
        except Exception:
            pass
    if default_dump_every is not None:
        return int(default_dump_every)
    return 500


def discover_series_records(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    if args.config:
        config_paths = collect_config_paths(args.config, args.config_dir, args.config_glob)
        records_all = discover_production_records(config_paths, duplicate_policy=args.duplicate_policy)
    else:
        if args.config_dir:
            raise ValueError("--config-dir can only be used with --config")
        records_all = discover_npt_records_from_md_root(
            args.md_root,
            duplicate_policy=args.duplicate_policy,
            timestep_ps=args.dt,
        )
    records = filter_records_by_T(records_all, args.t_min, args.t_max)
    return records_all, records


def run_series(args: argparse.Namespace) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    records_all, records = discover_series_records(args)

    discovered_rows = [
        {
            "temperature": record["temperature"],
            "stage_name": record["stage_name"],
            "log_path": str(record["log_path"]),
            "config_path": str(record.get("config_path") or ""),
            "md_root": str(record.get("md_root") or ""),
        }
        for record in records_all
    ]
    used_rows = [
        {
            "temperature": record["temperature"],
            "stage_name": record["stage_name"],
            "log_path": str(record["log_path"]),
            "config_path": str(record.get("config_path") or ""),
            "md_root": str(record.get("md_root") or ""),
        }
        for record in records
    ]
    write_json(args.outdir / "discovered_stage_records.json", {"records": discovered_rows})
    write_json(args.outdir / "used_stage_records.json", {"records": used_rows})

    series_items: list[dict] = []
    for record in records:
        temperature = float(record["temperature"])
        temp_label = f"{temperature:g}K".replace(".", "p")
        temp_dir = args.outdir / f"T_{temp_label}"
        prefix = f"T_{temp_label}"
        dump_path = find_record_dump(record)
        dump_every = record_dump_every(record, args.dump_every)
        timestep_ps = float(args.dt if args.dt is not None else record.get("timestep_ps", 0.0001))

        item_args = SimpleNamespace(
            dump=dump_path,
            traj=None,
            dump_format=args.dump_format,
            type_map=args.type_map,
            dt=timestep_ps,
            dump_every=dump_every,
            window_ps=args.window_ps,
            start=None,
            stop=None,
            step=args.frame_step,
            outdir=temp_dir,
            prefix=prefix,
            rmax=args.rmax,
            dr=args.dr,
            qmax=args.qmax,
            dq=args.dq,
            gr_rmax=args.gr_rmax,
            gr_dr=args.gr_dr,
            scattering=args.scattering,
            weights=args.weights,
            window_function=args.window_function,
            fitting_exports=args.fitting_exports,
            pdfgui_dr_uncertainty=args.pdfgui_dr_uncertainty,
            pdfgui_dgr=args.pdfgui_dgr,
            no_plots=args.no_plots,
            archive_path=None,
            no_archive_output=True,
            write_selected_extxyz=args.write_selected_extxyz,
        )
        summary = run(item_args)
        outputs = summary["outputs"]
        item = {
            "temperature": temperature,
            "stage_name": record["stage_name"],
            "log_path": str(record["log_path"]),
            "dump_path": str(dump_path),
            "dump_every": dump_every,
            "dt_ps": timestep_ps,
            "window_ps_used": summary["source"].get("window_ps_used"),
            "n_frames": summary["n_frames"],
            "avg_volume_A3": summary["avg_volume_A3"],
            "outdir": str(temp_dir),
            "summary_json": str(temp_dir / f"{prefix}_summary.json"),
            "gtot": outputs["total_g"],
            "GofR_direct": outputs["direct_GofR"],
            "SofQ": outputs["SofQ"],
            "FofQ": outputs["FofQ"],
            "FofQ_windowed": outputs["FofQ_windowed"],
            "GofR_from_FQ": outputs["GofR_from_FQ"],
            "pdfgui_GofR": outputs["pdfgui_GofR"],
            "rmcprofile_SofQ": outputs["rmcprofile_SofQ"],
            "rmcprofile_FofQ": outputs["rmcprofile_FofQ"],
            "fitting_exports": outputs.get("fitting_exports", {}),
        }
        series_items.append(item)

    with (args.outdir / "series_index.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "temperature",
            "stage_name",
            "dump_path",
            "n_frames",
            "window_ps_used",
            "avg_volume_A3",
            "outdir",
            "summary_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in series_items:
            writer.writerow({key: item.get(key, "") for key in fieldnames})

    overlay_plots: list[str] = []
    if not args.no_plots:
        overlay_defs = [
            ("gtot", "overlay_weighted_gr.png", "Weighted RDF", "r (A)", "g(r)"),
            ("GofR_direct", "overlay_pdf_GofR_direct.png", "Direct PDF G(r)", "r (A)", "G(r)"),
            ("GofR_from_FQ", "overlay_pdf_GofR_from_FQ.png", "PDF G(r) from F(Q)", "r (A)", "G(r)"),
            ("SofQ", "overlay_SofQ.png", "Total Scattering S(Q)", "Q (A^-1)", "S(Q)"),
            ("FofQ", "overlay_FofQ.png", "Reduced Structure Function F(Q)", "Q (A^-1)", "F(Q)"),
            (
                "FofQ_windowed",
                "overlay_FofQ_windowed.png",
                "Windowed Reduced Structure Function F(Q)",
                "Q (A^-1)",
                "F(Q)",
            ),
        ]
        for key, filename, title, xlabel, ylabel in overlay_defs:
            path = args.outdir / filename
            if plot_series_overlay(path, series_items, key, title, xlabel, ylabel):
                overlay_plots.append(str(path))

    metadata = {
        "mode": "series",
        "source": "config" if args.config else "md_root",
        "config": [str(Path(p).resolve()) for p in args.config] if args.config else [],
        "md_root": str(args.md_root.resolve()) if args.md_root else None,
        "npt_only": True,
        "nvt_ignored": True,
        "temperature_filter": {"t_min": args.t_min, "t_max": args.t_max},
        "window_ps": args.window_ps,
        "dump_format": args.dump_format,
        "type_map": args.type_map,
        "scattering": args.scattering,
        "series": series_items,
        "overlay_plots": overlay_plots,
        "archive": str(args.archive_path.resolve() if args.archive_path else default_archive_path(args.outdir).resolve())
        if not args.no_archive_output
        else None,
    }
    write_json(args.outdir / "series_summary.json", metadata)
    if not args.no_archive_output:
        archive = archive_output_dir(args.outdir, args.archive_path)
        metadata["archive"] = str(archive)
        write_json(args.outdir / "series_summary.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_lammps",
        description="Compute RDF/PDF/S(Q)/F(Q) from one LAMMPS trajectory or an NPT MD series.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dump", type=Path, help="LAMMPS dump trajectory")
    source.add_argument("--traj", type=Path, help="ASE-readable trajectory, usually extxyz")
    source.add_argument("--config", nargs="+", help="One or more production config JSON files for NPT series mode")
    source.add_argument(
        "--md-root",
        type=Path,
        help="Folder containing MD run folders. Series mode scans NPT folders only and ignores NVT folders.",
    )
    parser.add_argument("--config-dir", default=None, help="Optional directory containing more config JSON files")
    parser.add_argument("--config-glob", default="*.json", help="Glob pattern used with --config-dir")
    parser.add_argument(
        "--duplicate-policy",
        choices=["highest_config_order", "first", "error"],
        default="highest_config_order",
        help="How to handle duplicate temperatures in series mode.",
    )
    parser.add_argument("--dump-format", default="lammps-dump-text")
    parser.add_argument("--type-map", nargs="*", default=[], help="LAMMPS type map, e.g. 1=U 2=O")
    parser.add_argument("--dt", type=float, help="MD timestep in ps for --dump")
    parser.add_argument("--dump-every", type=int, help="LAMMPS steps between dump frames for --dump or --md-root")
    parser.add_argument("--window-ps", type=float, default=5.0, help="Last trajectory window for --dump")
    parser.add_argument("--t-min", type=float, help="Lowest NPT series temperature to include")
    parser.add_argument("--t-max", type=float, help="Highest requested NPT series temperature to include")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=None,
        help="Optional frame stride for each series dump after the last-window selection.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("rdf_pdf_analysis"))
    parser.add_argument("--prefix", default="lammps_pdf")
    parser.add_argument("--rmax", type=float, default=12.0)
    parser.add_argument("--dr", type=float, default=0.02)
    parser.add_argument("--qmax", type=float, default=25.0)
    parser.add_argument("--dq", type=float, default=0.05)
    parser.add_argument("--gr-rmax", type=float)
    parser.add_argument("--gr-dr", type=float)
    parser.add_argument("--scattering", choices=("xray", "neutron", "custom"), default="xray")
    parser.add_argument("--weights", nargs="*", default=[], help="Custom weights, e.g. U=92 O=8")
    parser.add_argument("--window-function", choices=("lorch", "none"), default="lorch")
    parser.add_argument(
        "--fitting-exports",
        choices=("auto", "none"),
        default="auto",
        help="Write explicit PDFgui and RMC-style fitting export files. Default: auto.",
    )
    parser.add_argument(
        "--pdfgui-dr-uncertainty",
        type=float,
        default=0.0,
        help="Placeholder dr uncertainty column for PDFgui four-column .gr exports.",
    )
    parser.add_argument(
        "--pdfgui-dgr",
        type=float,
        default=1.0,
        help="Placeholder dG uncertainty column for PDFgui four-column .gr exports.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--archive-path",
        type=Path,
        help="Optional tar.gz archive path. Default: <outdir>.tar.gz",
    )
    parser.add_argument(
        "--no-archive-output",
        action="store_true",
        help="Do not create a tar.gz archive of the output directory.",
    )
    parser.add_argument(
        "--no-selected-extxyz",
        dest="write_selected_extxyz",
        action="store_false",
        help="Do not write selected last-window/last-frame/average extxyz files.",
    )
    parser.set_defaults(write_selected_extxyz=True)
    return parser


def run(args: argparse.Namespace) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    source_summary = {}
    selected_outputs = {}
    if args.dump is not None:
        if not args.type_map:
            raise ValueError("--type-map is required when reading a LAMMPS dump")
        if args.dt is None or args.dump_every is None:
            raise ValueError("--dt and --dump-every are required when reading a LAMMPS dump")
        frames, source_summary = read_frames_from_dump(
            args.dump,
            args.dump_format,
            parse_type_map(args.type_map),
            args.dt,
            args.dump_every,
            args.window_ps,
        )
        if args.step is not None:
            frames = frames[:: args.step]
            source_summary["post_window_frame_step"] = args.step
            source_summary["n_selected_frames_after_step"] = len(frames)
    else:
        frames = read_frames_from_traj(args.traj, args.start, args.stop, args.step)
        source_summary = {
            "trajectory_file": str(args.traj.resolve()),
            "start": args.start,
            "stop": args.stop,
            "step": args.step,
            "n_selected_frames": len(frames),
        }

    if args.write_selected_extxyz:
        selected_outputs = write_selected_frames(args.outdir, args.prefix, frames)

    species_order = sorted(set(frames[0].get_chemical_symbols()))
    r_edges = np.arange(0.0, args.rmax + args.dr, args.dr)
    q_values = np.arange(args.dq, args.qmax + args.dq, args.dq)
    gr_rmax = args.gr_rmax if args.gr_rmax is not None else args.rmax
    gr_dr = args.gr_dr if args.gr_dr is not None else args.dr
    r_from_fq = np.arange(gr_dr, gr_rmax + gr_dr, gr_dr)

    pair_hist, avg_counts, avg_volume, n_frames = compute_partial_histograms(
        frames,
        species_order,
        r_edges,
    )
    r, partial = normalize_partial_rdfs(pair_hist, avg_counts, avg_volume, n_frames, r_edges)
    rho0 = sum(avg_counts.values()) / avg_volume

    scattering_meta: dict = {"mode": args.scattering}
    if args.scattering == "custom":
        weights = parse_weights(args.weights)
        missing = [s for s in species_order if s not in weights]
        if missing:
            raise ValueError(f"Missing --weights for species: {', '.join(missing)}")
        g_total, conc = weighted_total_gr_constant(species_order, partial, avg_counts, weights)
        sq, _ = partials_to_sq_constant(species_order, partial, avg_counts, weights, rho0, r, q_values)
        scattering_meta["weights"] = weights
    elif args.scattering == "neutron":
        weights = neutron_weights(species_order)
        g_total, conc = weighted_total_gr_constant(species_order, partial, avg_counts, weights)
        sq, _ = partials_to_sq_constant(species_order, partial, avg_counts, weights, rho0, r, q_values)
        scattering_meta["weights"] = weights
    else:
        form_factors = xray_form_factors(species_order, q_values)
        if form_factors is None:
            weights = atomic_number_weights(species_order)
            g_total, conc = weighted_total_gr_constant(species_order, partial, avg_counts, weights)
            sq, _ = partials_to_sq_constant(species_order, partial, avg_counts, weights, rho0, r, q_values)
            scattering_meta.update(
                {
                    "weights": weights,
                    "xray_form_factor_source": "atomic-number fallback",
                    "warning": "Install xraydb for Q-dependent X-ray form factors.",
                }
            )
        else:
            sq, conc = partials_to_sq_xray(
                species_order,
                partial,
                avg_counts,
                form_factors,
                rho0,
                r,
                q_values,
            )
            q0_weights = {s: float(form_factors[s][0]) for s in species_order}
            g_total, _ = weighted_total_gr_constant(species_order, partial, avg_counts, q0_weights)
            scattering_meta.update(
                {
                    "weights_q0": q0_weights,
                    "xray_form_factor_source": "xraydb",
                    "note": "S(Q), F(Q), and G(r) from F(Q) use Q-dependent X-ray form factors; direct g(r)/G(r) use Q~0 weights.",
                }
            )

    gr_direct = gr_to_gr_direct(r, g_total, rho0)
    fq = q_values * (sq - 1.0)
    fq_windowed, window_values = apply_window(q_values, fq, args.window_function, args.qmax)
    gr_from_fq = fq_to_gr(q_values, fq_windowed, r_from_fq)

    partial_cols = {f"g_{a}{b}": values for (a, b), values in partial.items()}
    write_multi_csv(args.outdir / f"{args.prefix}_partial_rdfs.csv", "r_A", r, partial_cols)
    write_xy(args.outdir / f"{args.prefix}_gtot.dat", r, g_total, "r_A", "g_total_weighted")
    write_xy(args.outdir / f"{args.prefix}_GofR_direct.dat", r, gr_direct, "r_A", "G_r_direct")
    write_xy(args.outdir / f"{args.prefix}_SofQ.dat", q_values, sq, "Q_A^-1", "S_Q")
    write_xy(args.outdir / f"{args.prefix}_FofQ.dat", q_values, fq, "Q_A^-1", "F_Q")
    write_xy(
        args.outdir / f"{args.prefix}_FofQ_windowed.dat",
        q_values,
        fq_windowed,
        "Q_A^-1",
        "F_Q_windowed",
    )
    write_xy(args.outdir / f"{args.prefix}_GofR_from_FQ.dat", r_from_fq, gr_from_fq, "r_A", "G_r_from_FQ")

    write_xy(args.outdir / f"{args.prefix}_pdfgui_GofR.gr", r_from_fq, gr_from_fq, "r_A", "G_r")
    write_xy(args.outdir / f"{args.prefix}_rmcprofile_SofQ.sq", q_values, sq, "Q_A^-1", "S_Q")
    write_xy(args.outdir / f"{args.prefix}_rmcprofile_FofQ.fq", q_values, fq_windowed, "Q_A^-1", "F_Q_windowed")
    fitting_exports = {}
    if args.fitting_exports == "auto":
        fitting_exports = write_fitting_exports(
            args.outdir,
            args.prefix,
            r,
            gr_direct,
            r_from_fq,
            gr_from_fq,
            q_values,
            sq,
            fq,
            fq_windowed,
            args.pdfgui_dr_uncertainty,
            args.pdfgui_dgr,
        )
    write_multi_csv(
        args.outdir / f"{args.prefix}_totals.csv",
        "r_A",
        r,
        {
            "g_total_weighted": g_total,
            "G_r_direct": gr_direct,
        },
    )
    write_multi_csv(
        args.outdir / f"{args.prefix}_sq_fq.csv",
        "Q_A^-1",
        q_values,
        {
            "S_Q": sq,
            "F_Q": fq,
            "window": window_values,
            "F_Q_windowed": fq_windowed,
        },
    )
    write_multi_csv(
        args.outdir / f"{args.prefix}_GfromFQ.csv",
        "r_A",
        r_from_fq,
        {"G_r_from_FQ": gr_from_fq},
    )

    plots = []
    if not args.no_plots:
        plots = plot_outputs(
            args.outdir,
            args.prefix,
            r,
            partial,
            g_total,
            gr_direct,
            r_from_fq,
            gr_from_fq,
            q_values,
            sq,
            fq,
            fq_windowed,
        )

    summary = {
        "source": source_summary,
        "selected_outputs": selected_outputs,
        "n_frames": n_frames,
        "avg_volume_A3": avg_volume,
        "rho0_atoms_per_A3": rho0,
        "avg_counts": avg_counts,
        "concentrations": conc,
        "species_order": species_order,
        "rmax_A": args.rmax,
        "dr_A": args.dr,
        "qmax_A^-1": args.qmax,
        "dq_A^-1": args.dq,
        "gr_rmax_A": gr_rmax,
        "gr_dr_A": gr_dr,
        "window_function": args.window_function,
        "scattering": scattering_meta,
        "fitting_export_notes": {
            "pdfgui_4col": "PDFgui/PDFfit2-style r, G(r), dr, dG(r). The dr and dG columns are placeholder uncertainties.",
            "rmcprofile_SQ": "Two-column normalized S(Q), expected to approach 1 at high Q.",
            "rmcprofile_iQ": "Two-column i(Q)=S(Q)-1, the flat-low-Q convention often used by RMCProfile/Keen-style inputs.",
            "pdfgetx_FQ": "Two-column F(Q)=Q[S(Q)-1], common in PDFgetX/PDFgui workflows; verify before using as RMCProfile F(Q).",
            "rmcprofile_GofR_flat": "Two-column G_pdf(r)/r, included as a flat-low-r real-space diagnostic for RMC-style convention checks.",
        },
        "plots": plots,
        "archive": str(args.archive_path.resolve() if args.archive_path else default_archive_path(args.outdir).resolve())
        if not args.no_archive_output
        else None,
        "outputs": {
            "partial_rdfs_csv": str(args.outdir / f"{args.prefix}_partial_rdfs.csv"),
            "total_g": str(args.outdir / f"{args.prefix}_gtot.dat"),
            "direct_GofR": str(args.outdir / f"{args.prefix}_GofR_direct.dat"),
            "SofQ": str(args.outdir / f"{args.prefix}_SofQ.dat"),
            "FofQ": str(args.outdir / f"{args.prefix}_FofQ.dat"),
            "FofQ_windowed": str(args.outdir / f"{args.prefix}_FofQ_windowed.dat"),
            "GofR_from_FQ": str(args.outdir / f"{args.prefix}_GofR_from_FQ.dat"),
            "pdfgui_GofR": str(args.outdir / f"{args.prefix}_pdfgui_GofR.gr"),
            "rmcprofile_SofQ": str(args.outdir / f"{args.prefix}_rmcprofile_SofQ.sq"),
            "rmcprofile_FofQ": str(args.outdir / f"{args.prefix}_rmcprofile_FofQ.fq"),
            "fitting_exports": fitting_exports,
        },
    }
    write_json(args.outdir / f"{args.prefix}_summary.json", summary)
    if not args.no_archive_output:
        archive = archive_output_dir(args.outdir, args.archive_path)
        summary["archive"] = str(archive)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config or args.md_root:
        if not args.type_map:
            parser.error("--type-map is required for series mode because LAMMPS dump files store atom types")
        if args.start is not None or args.stop is not None or args.step is not None:
            parser.error("Use --frame-step for series mode; --start/--stop/--step are single-trajectory options")
        summary = run_series(args)
        print(f"Series temperatures: {len(summary['series'])}")
        if summary["series"]:
            temps = [item["temperature"] for item in summary["series"]]
            print(f"Temperature range used: {min(temps):g} to {max(temps):g} K")
        print(f"Wrote series summary: {args.outdir / 'series_summary.json'}")
        print(f"Wrote series index: {args.outdir / 'series_index.csv'}")
        if summary.get("archive"):
            print(f"Download archive written to: {summary['archive']}")
        return
    summary = run(args)
    outputs = summary["outputs"]
    print(f"Frames used: {summary['n_frames']}")
    print(f"Average volume: {summary['avg_volume_A3']:.6f} A^3")
    print(f"rho0: {summary['rho0_atoms_per_A3']:.6f} atoms/A^3")
    print(f"Species: {', '.join(summary['species_order'])}")
    print(f"Wrote summary: {args.outdir / (args.prefix + '_summary.json')}")
    print(f"Wrote PDFgui G(r): {outputs['pdfgui_GofR']}")
    print(f"Wrote RMCProfile S(Q): {outputs['rmcprofile_SofQ']}")
    print(f"Wrote RMCProfile F(Q): {outputs['rmcprofile_FofQ']}")
    if outputs.get("fitting_exports"):
        print("Wrote explicit PDFgui/RMC-style fitting exports.")
    if summary.get("archive"):
        print(f"Download archive written to: {summary['archive']}")


if __name__ == "__main__":
    main()
