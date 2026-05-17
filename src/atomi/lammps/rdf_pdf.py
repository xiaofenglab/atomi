#!/usr/bin/env python3
"""LAMMPS trajectory RDF/PDF/S(Q)/F(Q) analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from atomi.core.archive import archive_output_dir, default_archive_path
from atomi.core.cell import cell_metadata, infer_formula_units
from atomi.lammps.box import (
    cell_parameters as box_cell_parameters,
    flatten_box_summary,
    format_box_summary,
    summarize_cells,
)
from atomi.lammps.thermo_series import (
    collect_config_paths,
    discover_npt_records_from_md_root,
    discover_production_records,
    filter_records_by_T,
)


def metadata_value(args: argparse.Namespace, name: str):
    return getattr(args, name, None)


def md_cell_metadata(args: argparse.Namespace, *, natoms: int, cell_role: str) -> dict:
    requested_natoms = metadata_value(args, "natoms")
    effective_natoms = float(requested_natoms) if requested_natoms is not None else float(natoms)
    formula_units = infer_formula_units(
        formula_units=metadata_value(args, "formula_units"),
        natoms=effective_natoms,
        atoms_per_formula_unit=metadata_value(args, "atoms_per_formula_unit"),
        formula=metadata_value(args, "formula"),
    )
    return cell_metadata(
        formula=metadata_value(args, "formula"),
        natoms=effective_natoms,
        atoms_per_formula_unit=metadata_value(args, "atoms_per_formula_unit"),
        formula_units=formula_units,
        target_z=metadata_value(args, "target_z"),
        cell_role=cell_role,
        normalization_basis="simulation-cell",
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


def summarize_array(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": std,
        f"{prefix}_sem": std / math.sqrt(n) if n > 1 else 0.0,
        f"{prefix}_p16": float(np.percentile(values, 16)) if n else math.nan,
        f"{prefix}_p84": float(np.percentile(values, 84)) if n else math.nan,
    }


def cell_parameters(cell: np.ndarray) -> dict[str, float]:
    return box_cell_parameters(cell)


def compute_structure_stats(frames: list) -> dict:
    params = [cell_parameters(np.asarray(atoms.cell.array, dtype=float)) for atoms in frames]
    out: dict[str, float | int] = {"n_frames": len(frames)}
    for key in ("volume_A3", "a_A", "b_A", "c_A", "alpha_deg", "beta_deg", "gamma_deg"):
        values = np.asarray([p[key] for p in params], dtype=float)
        out.update(summarize_array(values, key))
    box_summary = summarize_cells((np.asarray(atoms.cell.array, dtype=float) for atoms in frames))
    out["md_box"] = box_summary
    out.update(flatten_box_summary(box_summary))
    return out


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


def neutron_scattering_metadata(species_order: list[str]) -> dict:
    weights = neutron_weights(species_order)
    return {
        "mode": "neutron",
        "weights": weights,
        "weight_type": "coherent bound neutron scattering length",
        "weight_unit": "fm",
        "data_source": "periodictable neutron tables",
        "data_source_note": (
            "periodictable provides programmatic coherent neutron scattering "
            "lengths, commonly traceable to NIST/NCNR tabulations. ADDIE is an "
            "ORNL reduction/workflow tool, not the runtime table used here."
        ),
        "isotope_note": (
            "Element-average/natural-abundance values are used unless the user "
            "supplies custom weights with --scattering custom --weights."
        ),
        "normalization": "S(Q) uses concentration-weighted b_i b_j terms normalized by <b>^2.",
    }


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


def write_rows_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def compute_total_curve_bundle(
    frames: list,
    species_order: list[str],
    r_edges: np.ndarray,
    q_values: np.ndarray,
    r_from_fq: np.ndarray,
    scattering: str,
    weights_items: list[str],
    window_function: str,
    qmax: float,
    form_factors: Optional[dict[str, np.ndarray]] = None,
) -> dict:
    pair_hist, avg_counts, avg_volume, n_frames = compute_partial_histograms(frames, species_order, r_edges)
    r, partial = normalize_partial_rdfs(pair_hist, avg_counts, avg_volume, n_frames, r_edges)
    rho0 = sum(avg_counts.values()) / avg_volume
    if scattering == "custom":
        weights = parse_weights(weights_items)
        g_total, conc = weighted_total_gr_constant(species_order, partial, avg_counts, weights)
        sq, _ = partials_to_sq_constant(species_order, partial, avg_counts, weights, rho0, r, q_values)
    elif scattering == "neutron":
        weights = neutron_weights(species_order)
        g_total, conc = weighted_total_gr_constant(species_order, partial, avg_counts, weights)
        sq, _ = partials_to_sq_constant(species_order, partial, avg_counts, weights, rho0, r, q_values)
    else:
        if form_factors is None:
            form_factors = xray_form_factors(species_order, q_values)
        if form_factors is None:
            weights = atomic_number_weights(species_order)
            g_total, conc = weighted_total_gr_constant(species_order, partial, avg_counts, weights)
            sq, _ = partials_to_sq_constant(species_order, partial, avg_counts, weights, rho0, r, q_values)
        else:
            sq, conc = partials_to_sq_xray(species_order, partial, avg_counts, form_factors, rho0, r, q_values)
            q0_weights = {s: float(form_factors[s][0]) for s in species_order}
            g_total, _ = weighted_total_gr_constant(species_order, partial, avg_counts, q0_weights)
    gr_direct = gr_to_gr_direct(r, g_total, rho0)
    fq = q_values * (sq - 1.0)
    fq_windowed, _ = apply_window(q_values, fq, window_function, qmax)
    gr_from_fq = fq_to_gr(q_values, fq_windowed, r_from_fq)
    return {
        "r": r,
        "q": q_values,
        "g_total": g_total,
        "GofR_direct": gr_direct,
        "SofQ": sq,
        "FofQ": fq,
        "FofQ_windowed": fq_windowed,
        "r_from_fq": r_from_fq,
        "GofR_from_FQ": gr_from_fq,
        "avg_counts": avg_counts,
        "avg_volume_A3": avg_volume,
        "rho0_atoms_per_A3": rho0,
        "concentrations": conc,
    }


def select_frame_overlay_indices(n_frames: int, step: int, max_frames: int) -> list[int]:
    step = max(1, int(step))
    indices = list(range(0, n_frames, step))
    if max_frames > 0 and len(indices) > max_frames:
        stride = int(math.ceil(len(indices) / max_frames))
        indices = indices[::stride][:max_frames]
    return indices


def plot_frame_overlay(
    path: Path,
    x: np.ndarray,
    columns: dict[str, np.ndarray],
    average: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> bool:
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

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.08, 0.92, max(len(columns), 1)))
    for color, (label, values) in zip(colors, columns.items()):
        ax.plot(x, values, color=color, alpha=0.35, linewidth=0.8, label=label)
    ax.plot(x, average, color="black", linewidth=1.0, label="averaged structure")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 8:
        handles = [handles[0], handles[-2], handles[-1]]
        labels = [labels[0], labels[-2], labels[-1]]
    ax.legend(handles, labels, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def write_frame_overlay_outputs(
    outdir: Path,
    prefix: str,
    frames: list,
    species_order: list[str],
    r_edges: np.ndarray,
    q_values: np.ndarray,
    r_from_fq: np.ndarray,
    args: argparse.Namespace,
    averages: dict[str, np.ndarray],
) -> dict:
    indices = select_frame_overlay_indices(len(frames), args.frame_overlay_step, args.frame_overlay_max)
    form_factors = xray_form_factors(species_order, q_values) if args.scattering == "xray" else None
    bundles = []
    for idx in indices:
        bundles.append(
            (
                idx,
                compute_total_curve_bundle(
                    [frames[idx]],
                    species_order,
                    r_edges,
                    q_values,
                    r_from_fq,
                    args.scattering,
                    args.weights,
                    args.window_function,
                    args.qmax,
                    form_factors=form_factors,
                ),
            )
        )
    written: dict[str, object] = {
        "n_total_frames": len(frames),
        "n_overlay_frames": len(indices),
        "frame_indices_0based": indices,
        "csv": {},
        "plots": [],
    }
    curve_defs = [
        ("g_total", "r", "frame_overlay_gtot.csv", "frame_overlay_gtot.png", "r (A)", "g(r)", averages["g_total"]),
        (
            "GofR_direct",
            "r",
            "frame_overlay_GofR_direct.csv",
            "frame_overlay_GofR_direct.png",
            "r (A)",
            "G(r)",
            averages["GofR_direct"],
        ),
        (
            "GofR_from_FQ",
            "r_from_fq",
            "frame_overlay_GofR_from_FQ.csv",
            "frame_overlay_GofR_from_FQ.png",
            "r (A)",
            "G(r)",
            averages["GofR_from_FQ"],
        ),
        ("SofQ", "q", "frame_overlay_SofQ.csv", "frame_overlay_SofQ.png", "Q (A^-1)", "S(Q)", averages["SofQ"]),
        ("FofQ", "q", "frame_overlay_FofQ.csv", "frame_overlay_FofQ.png", "Q (A^-1)", "F(Q)", averages["FofQ"]),
    ]
    for key, x_key, csv_name, png_name, xlabel, ylabel, avg in curve_defs:
        columns = {f"frame_{idx:06d}": bundle[key] for idx, bundle in bundles}
        x = bundles[0][1][x_key]
        csv_path = outdir / f"{prefix}_{csv_name}"
        write_multi_csv(csv_path, x_key, x, columns)
        written["csv"][key] = str(csv_path)
        if not args.no_plots:
            png_path = outdir / f"{prefix}_{png_name}"
            if plot_frame_overlay(png_path, x, columns, avg, xlabel, ylabel, f"Per-frame {ylabel} overlay"):
                written["plots"].append(str(png_path))
    return written


def compute_adp_from_frames(frames: list) -> tuple[list[dict], list[dict]]:
    if len(frames) < 2:
        return [], []
    ref = frames[0]
    symbols = ref.get_chemical_symbols()
    natoms = len(symbols)
    ref_cell = np.asarray(ref.cell.array, dtype=float)
    ref_frac = ref.get_positions() @ np.linalg.inv(ref_cell)

    unwrapped_fracs = []
    cells = []
    for atoms in frames:
        if len(atoms) != natoms or atoms.get_chemical_symbols() != symbols:
            raise ValueError("ADP calculation needs fixed atom order and species across frames.")
        cell = np.asarray(atoms.cell.array, dtype=float)
        frac = atoms.get_positions() @ np.linalg.inv(cell)
        unwrapped_fracs.append(ref_frac + minimum_image_deltas(frac - ref_frac))
        cells.append(cell)

    fracs = np.asarray(unwrapped_fracs, dtype=float)
    mean_frac = np.mean(fracs, axis=0)
    atom_rows = []
    for i, symbol in enumerate(symbols):
        displacements = np.asarray([(fracs[j, i] - mean_frac[i]) @ cells[j] for j in range(len(frames))], dtype=float)
        cov = displacements.T @ displacements / len(frames)
        uiso = float(np.trace(cov) / 3.0)
        atom_rows.append(
            {
                "atom_index_1based": i + 1,
                "symbol": symbol,
                "U11_A2": cov[0, 0],
                "U22_A2": cov[1, 1],
                "U33_A2": cov[2, 2],
                "U12_A2": cov[0, 1],
                "U13_A2": cov[0, 2],
                "U23_A2": cov[1, 2],
                "Uiso_A2": uiso,
                "Biso_A2": 8.0 * math.pi**2 * uiso,
                "rms_displacement_A": math.sqrt(max(uiso * 3.0, 0.0)),
            }
        )

    species_rows = []
    for symbol in sorted(set(symbols)):
        values = np.asarray([row["Uiso_A2"] for row in atom_rows if row["symbol"] == symbol], dtype=float)
        species_rows.append(
            {
                "symbol": symbol,
                "n_atoms": len(values),
                "Uiso_mean_A2": float(np.mean(values)),
                "Uiso_std_A2": float(np.std(values)),
                "Uiso_min_A2": float(np.min(values)),
                "Uiso_max_A2": float(np.max(values)),
                "Biso_mean_A2": float(8.0 * math.pi**2 * np.mean(values)),
                "rms_displacement_mean_A": float(np.mean(np.sqrt(np.maximum(values * 3.0, 0.0)))),
            }
        )
    return atom_rows, species_rows


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_temperature_uq(
    path: Path,
    T: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
) -> bool:
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

    order = np.argsort(T)
    T = T[order]
    y = y[order]
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)[order]

    fig, ax = plt.subplots(figsize=(7, 4))
    if yerr is not None:
        ax.fill_between(T, y - yerr, y + yerr, color="#1f77b4", alpha=0.18, label="window UQ (SEM)")
        ax.errorbar(T, y, yerr=yerr, fmt="o", color="#1f77b4", capsize=3, linewidth=1.0)
    ax.plot(T, y, color="#1f77b4", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yerr is not None:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def plot_multi_temperature_uq(
    path: Path,
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
) -> bool:
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

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("tab10")
    for i, (label, (T, y, yerr)) in enumerate(sorted(series.items())):
        order = np.argsort(T)
        T = np.asarray(T, dtype=float)[order]
        y = np.asarray(y, dtype=float)[order]
        yerr = np.asarray(yerr, dtype=float)[order]
        color = cmap(i % 10)
        ax.fill_between(T, y - yerr, y + yerr, color=color, alpha=0.12)
        ax.errorbar(T, y, yerr=yerr, fmt="o", color=color, capsize=3, linewidth=1.0, label=label)
        ax.plot(T, y, color=color, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def write_series_structure_adp_outputs(outdir: Path, series_items: list[dict], no_plots: bool) -> dict:
    structure_rows = []
    adp_rows = []
    for item in series_items:
        stats = item.get("structure_stats", {})
        if stats:
            row = {
                "temperature": item["temperature"],
                "stage_name": item.get("stage_name", ""),
                "n_frames": stats.get("n_frames", item.get("n_frames", "")),
            }
            for key, value in stats.items():
                if key != "n_frames":
                    row[key] = value
            structure_rows.append(row)
        adp = item.get("adp", {})
        species_csv = adp.get("species_adp_csv") if isinstance(adp, dict) else None
        if species_csv and Path(species_csv).exists():
            for row in read_csv_dicts(Path(species_csv)):
                values = {k: float(v) if k not in ("symbol", "") and v != "" else v for k, v in row.items()}
                values["temperature"] = item["temperature"]
                values["stage_name"] = item.get("stage_name", "")
                adp_rows.append(values)

    outputs: dict[str, object] = {"plots": []}
    if structure_rows:
        structure_path = outdir / "series_structure_vs_T.csv"
        structure_fields = [
            "temperature",
            "stage_name",
            "n_frames",
            "volume_A3_mean",
            "volume_A3_std",
            "volume_A3_sem",
            "a_A_mean",
            "a_A_std",
            "a_A_sem",
            "b_A_mean",
            "b_A_std",
            "b_A_sem",
            "c_A_mean",
            "c_A_std",
            "c_A_sem",
            "alpha_deg_mean",
            "alpha_deg_sem",
            "beta_deg_mean",
            "beta_deg_sem",
            "gamma_deg_mean",
            "gamma_deg_sem",
            "box_symmetry",
            "box_status",
            "n_box_samples",
            "length_rel_tol",
            "angle_tol_deg",
            "tilt_source",
        ]
        write_rows_csv(structure_path, structure_rows, structure_fields)
        outputs["structure_csv"] = str(structure_path)
        if not no_plots:
            T = np.asarray([row["temperature"] for row in structure_rows], dtype=float)
            for key, ylabel, title, filename in [
                ("volume_A3", "Volume (A^3)", "Selected-window volume vs T", "series_volume_vs_T_UQ.png"),
                ("a_A", "a (A)", "Lattice a vs T", "series_lattice_a_vs_T_UQ.png"),
                ("b_A", "b (A)", "Lattice b vs T", "series_lattice_b_vs_T_UQ.png"),
                ("c_A", "c (A)", "Lattice c vs T", "series_lattice_c_vs_T_UQ.png"),
            ]:
                y = np.asarray([row.get(f"{key}_mean", math.nan) for row in structure_rows], dtype=float)
                yerr = np.asarray([row.get(f"{key}_sem", math.nan) for row in structure_rows], dtype=float)
                path = outdir / filename
                if plot_temperature_uq(path, T, y, yerr, "Temperature (K)", ylabel, title):
                    outputs["plots"].append(str(path))

    if adp_rows:
        adp_path = outdir / "series_adp_Uiso_vs_T.csv"
        adp_fields = [
            "temperature",
            "stage_name",
            "symbol",
            "n_atoms",
            "Uiso_mean_A2",
            "Uiso_std_A2",
            "Uiso_min_A2",
            "Uiso_max_A2",
            "Biso_mean_A2",
            "rms_displacement_mean_A",
        ]
        write_rows_csv(adp_path, adp_rows, adp_fields)
        outputs["adp_csv"] = str(adp_path)
        if not no_plots:
            combined: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for symbol in sorted({str(row["symbol"]) for row in adp_rows}):
                rows = [row for row in adp_rows if str(row["symbol"]) == symbol]
                T = np.asarray([row["temperature"] for row in rows], dtype=float)
                y = np.asarray([float(row["Uiso_mean_A2"]) for row in rows], dtype=float)
                yerr = np.asarray(
                    [
                        float(row["Uiso_std_A2"]) / math.sqrt(float(row["n_atoms"]))
                        if float(row["n_atoms"]) > 1
                        else 0.0
                        for row in rows
                    ],
                    dtype=float,
                )
                path = outdir / f"series_Uiso_{symbol}_vs_T_UQ.png"
                if plot_temperature_uq(path, T, y, yerr, "Temperature (K)", f"{symbol} Uiso (A^2)", f"{symbol} Uiso vs T"):
                    outputs["plots"].append(str(path))
                combined[symbol] = (T, y, yerr)
            combined_path = outdir / "series_Uiso_all_elements_vs_T_UQ.png"
            if combined and plot_multi_temperature_uq(
                combined_path,
                combined,
                "Temperature (K)",
                "Uiso (A^2)",
                "Element Uiso vs T",
            ):
                outputs["plots"].append(str(combined_path))
    if outputs.get("plots"):
        outputs["uncertainty_note"] = (
            "Volume/lattice UQ uses selected-frame SEM within each NPT window. "
            "Uiso UQ uses the species atom-to-atom SEM at each temperature."
        )
    return outputs


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


def strip_sbatch_generation_args(argv: list[str]) -> list[str]:
    value_flags = {
        "--run-script",
        "--sbatch-script",
        "--job-name",
        "--time",
        "--cpus",
        "--mem",
        "--module",
    }
    bool_flags = {"--write-sbatch", "--submit", "--module-purge"}
    stripped: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item in bool_flags:
            continue
        if item in value_flags:
            skip_next = True
            continue
        if any(item.startswith(flag + "=") for flag in value_flags):
            continue
        stripped.append(item)
    return stripped


def write_series_run_script(path: Path, argv: list[str], cwd: Path, module: str | None, module_purge: bool) -> None:
    command = ["python", "-m", "atomi.cli.main", "pdf_lammps_series", *strip_sbatch_generation_args(argv)]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(cwd))}",
    ]
    if module_purge:
        lines.append("module purge")
    if module:
        lines.append(f"module load {shlex.quote(module)}")
    lines.extend(
        [
            "",
            " ".join(shlex.quote(part) for part in command),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def write_series_sbatch_script(
    path: Path,
    run_script: Path,
    job_name: str,
    time_limit: str,
    cpus: int,
    mem: str,
) -> None:
    logs_dir = path.parent / "logs"
    content = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir.resolve()}/{job_name}_%j.out
#SBATCH --error={logs_dir.resolve()}/{job_name}_%j.err
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}

set -euo pipefail
mkdir -p {shlex.quote(str(logs_dir.resolve()))}
bash {shlex.quote(str(run_script.resolve()))}
"""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def prepare_series_sbatch(args: argparse.Namespace, argv: list[str], cwd: Path) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    run_script = args.outdir / args.run_script
    sbatch_script = args.outdir / args.sbatch_script
    write_series_run_script(run_script, argv, cwd, args.module, args.module_purge)
    write_series_sbatch_script(sbatch_script, run_script, args.job_name, args.time, args.cpus, args.mem)
    payload = {
        "run_script": str(run_script.resolve()),
        "sbatch_script": str(sbatch_script.resolve()),
        "submit_command": f"sbatch {shlex.quote(str(sbatch_script.resolve()))}",
        "cwd": str(cwd),
    }
    write_json(args.outdir / "series_sbatch_summary.json", payload)
    return payload


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
            formula=getattr(args, "formula", None),
            natoms=getattr(args, "natoms", None),
            atoms_per_formula_unit=getattr(args, "atoms_per_formula_unit", None),
            formula_units=getattr(args, "formula_units", None),
            target_z=getattr(args, "target_z", None),
            window_function=args.window_function,
            fitting_exports=args.fitting_exports,
            pdfgui_dr_uncertainty=args.pdfgui_dr_uncertainty,
            pdfgui_dgr=args.pdfgui_dgr,
            frame_overlays=args.frame_overlays,
            frame_overlay_step=args.frame_overlay_step,
            frame_overlay_max=args.frame_overlay_max,
            adp=args.adp,
            no_plots=args.no_plots,
            archive_path=None,
            no_archive_output=True,
            write_selected_extxyz=args.write_selected_extxyz,
        )
        summary = run(item_args)
        print("  " + format_box_summary(summary["structure_stats"]["md_box"], label=f"MD box {record['stage_name']}"))
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
            "cell_metadata": summary.get("cell_metadata", {}),
            "structure_stats": summary.get("structure_stats", {}),
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
            "adp": outputs.get("adp", {}),
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

    structure_adp_outputs = write_series_structure_adp_outputs(args.outdir, series_items, args.no_plots)

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
        "cell_metadata": series_items[0].get("cell_metadata", {}) if series_items else {},
        "scattering_note": (
            "Per-temperature summary JSON files include the actual scattering "
            "weights/form-factor metadata used for each averaged MD PDF."
        ),
        "series": series_items,
        "structure_adp_outputs": structure_adp_outputs,
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
    parser.add_argument("--formula", help="Formula label for shared cell metadata, e.g. UO2.")
    parser.add_argument("--natoms", type=float, help="Atoms in the source MD/simulation cell.")
    parser.add_argument("--atoms-per-formula-unit", type=float, help="Atoms per formula unit.")
    parser.add_argument("--formula-units", type=float, help="Formula units in the source MD/simulation cell.")
    parser.add_argument("--target-z", type=float, help="Formula units in the normalized target crystallographic cell.")
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
    parser.add_argument(
        "--frame-overlays",
        action="store_true",
        help="Write per-frame overlay CSV/PNG curves for the selected time window. In series mode this is done inside each T folder.",
    )
    parser.add_argument(
        "--frame-overlay-step",
        type=int,
        default=1,
        help="Use every Nth selected frame for --frame-overlays.",
    )
    parser.add_argument(
        "--frame-overlay-max",
        type=int,
        default=0,
        help="Maximum frames to overlay after stepping; 0 means all selected frames.",
    )
    parser.add_argument(
        "--adp",
        action="store_true",
        help="Estimate per-atom and per-species Uiso/Biso from selected-frame displacements.",
    )
    parser.add_argument(
        "--write-sbatch",
        action="store_true",
        help="For series mode, write run/Slurm scripts and exit instead of running analysis interactively.",
    )
    parser.add_argument("--run-script", default="run_pdf_lammps_series.sh")
    parser.add_argument("--sbatch-script", default="submit_pdf_lammps_series.sbatch")
    parser.add_argument("--job-name", default="pdf_lammps_series")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--mem", default="96G")
    parser.add_argument("--module", default=None, help="Optional environment module to load in the run script.")
    parser.add_argument("--module-purge", action="store_true", help="Write module purge before --module.")
    parser.add_argument("--submit", action="store_true", help="Write the Slurm script and submit it with sbatch.")
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

    cell_meta = md_cell_metadata(args, natoms=len(frames[0]), cell_role="md-pdf-source-cell")
    structure_stats = compute_structure_stats(frames)
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
        scattering_meta = neutron_scattering_metadata(species_order)
        weights = scattering_meta["weights"]
        g_total, conc = weighted_total_gr_constant(species_order, partial, avg_counts, weights)
        sq, _ = partials_to_sq_constant(species_order, partial, avg_counts, weights, rho0, r, q_values)
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

    frame_overlay_outputs = {}
    if args.frame_overlays:
        frame_overlay_outputs = write_frame_overlay_outputs(
            args.outdir,
            args.prefix,
            frames,
            species_order,
            r_edges,
            q_values,
            r_from_fq,
            args,
            {
                "g_total": g_total,
                "GofR_direct": gr_direct,
                "GofR_from_FQ": gr_from_fq,
                "SofQ": sq,
                "FofQ": fq,
            },
        )

    adp_outputs = {}
    if args.adp:
        atom_rows, species_rows = compute_adp_from_frames(frames)
        atom_path = args.outdir / f"{args.prefix}_adp_atoms.csv"
        species_path = args.outdir / f"{args.prefix}_adp_species.csv"
        atom_fields = [
            "atom_index_1based",
            "symbol",
            "U11_A2",
            "U22_A2",
            "U33_A2",
            "U12_A2",
            "U13_A2",
            "U23_A2",
            "Uiso_A2",
            "Biso_A2",
            "rms_displacement_A",
        ]
        species_fields = [
            "symbol",
            "n_atoms",
            "Uiso_mean_A2",
            "Uiso_std_A2",
            "Uiso_min_A2",
            "Uiso_max_A2",
            "Biso_mean_A2",
            "rms_displacement_mean_A",
        ]
        write_rows_csv(atom_path, atom_rows, atom_fields)
        write_rows_csv(species_path, species_rows, species_fields)
        adp_outputs = {
            "atom_adp_csv": str(atom_path),
            "species_adp_csv": str(species_path),
            "n_frames": len(frames),
            "note": "U values are in Angstrom^2 from selected-window Cartesian displacements; Biso = 8*pi^2*Uiso.",
        }

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
        "cell_metadata": cell_meta,
        "structure_stats": structure_stats,
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
        "window_average_note": "RDF/PDF/S(Q)/F(Q) outputs are averaged over all selected frames in the time window.",
        "frame_overlay_outputs": frame_overlay_outputs,
        "adp_outputs": adp_outputs,
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
            "frame_overlays": frame_overlay_outputs,
            "adp": adp_outputs,
        },
    }
    write_json(args.outdir / f"{args.prefix}_summary.json", summary)
    if not args.no_archive_output:
        archive = archive_output_dir(args.outdir, args.archive_path)
        summary["archive"] = str(archive)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.config or args.md_root:
        if not args.type_map:
            parser.error("--type-map is required for series mode because LAMMPS dump files store atom types")
        if args.start is not None or args.stop is not None or args.step is not None:
            parser.error("Use --frame-step for series mode; --start/--stop/--step are single-trajectory options")
        if args.write_sbatch or args.submit:
            scripts = prepare_series_sbatch(args, raw_argv, Path.cwd())
            print(f"Wrote run script    : {scripts['run_script']}")
            print(f"Wrote sbatch script : {scripts['sbatch_script']}")
            print(f"Submit with         : {scripts['submit_command']}")
            if args.submit:
                subprocess.run(["sbatch", scripts["sbatch_script"]], cwd=args.outdir, check=True)
            return
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
    print(format_box_summary(summary["structure_stats"]["md_box"]))
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
