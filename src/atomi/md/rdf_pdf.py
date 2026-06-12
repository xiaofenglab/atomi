#!/usr/bin/env python3
"""General MD RDF/PDF analysis utilities.

This module keeps trajectory parsing separate from RDF/PDF math so VASP,
CP2K, and LAMMPS workflows can share one thermodynamics-analysis path.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from atomi.structure.adapters import read_vasp_poscar_basis, read_vasp_xdatcar_frames


def parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_weights(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected weight item like U=92, got: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def write_multi_csv(path: Path, xname: str, x: np.ndarray, columns: dict[str, np.ndarray]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([xname] + list(columns))
        for i, xi in enumerate(x):
            writer.writerow([xi] + [columns[key][i] for key in columns])


def write_json(path: Path, data: dict[str, Any]) -> None:
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

    path.write_text(json.dumps(normalize(data), indent=2), encoding="utf-8")


def normalize_partial_rdfs(
    pair_hist: dict[tuple[str, str], np.ndarray],
    avg_counts: dict[str, float],
    avg_volume: float,
    n_frames: int,
    r_edges: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray]]:
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = np.diff(r_edges)
    shell = 4.0 * np.pi * r**2 * dr
    partial: dict[tuple[str, str], np.ndarray] = {}
    for (a, b), hist in pair_hist.items():
        na = avg_counts[a]
        nb = avg_counts[b]
        denom = 0.5 * na * (na - 1.0) if a == b else na * nb
        if denom <= 0.0:
            partial[(a, b)] = np.zeros_like(hist, dtype=float)
        else:
            partial[(a, b)] = hist * avg_volume / (n_frames * denom * shell)
    return r, partial


def concentrations(avg_counts: dict[str, float], species_order: list[str]) -> dict[str, float]:
    n_total = sum(avg_counts.values())
    return {symbol: avg_counts[symbol] / n_total for symbol in species_order}


def weighted_total_gr_constant(
    species_order: list[str],
    partial: dict[tuple[str, str], np.ndarray],
    avg_counts: dict[str, float],
    weights: dict[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    conc = concentrations(avg_counts, species_order)
    denom = sum(conc[symbol] * weights[symbol] for symbol in species_order) ** 2
    total: np.ndarray | None = None
    for ia, a in enumerate(species_order):
        for b in species_order[ia:]:
            pref = conc[a] * weights[a] * conc[b] * weights[b] / denom
            if a != b:
                pref *= 2.0
            value = pref * partial[(a, b)]
            total = value if total is None else total + value
    if total is None:
        raise ValueError("No species pairs available for weighted total g(r).")
    return total, conc


def vasp_frames_from_xdatcar(
    *,
    poscar: Path,
    xdatcar: Path,
    start_frame: int = 0,
    stop_frame: int | None = None,
    stride: int = 1,
    species_order: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read selected frames from a VASP XDATCAR/POSCAR pair.

    Coordinates are kept in fractional form for robust minimum-image distances
    under general periodic cells.
    """
    basis = read_vasp_poscar_basis(poscar)
    all_frames = read_vasp_xdatcar_frames(xdatcar, basis["natoms"])
    if not all_frames:
        raise ValueError(f"No Direct configuration frames found in XDATCAR: {xdatcar}")

    start = max(int(start_frame), 0)
    stop = len(all_frames) if stop_frame is None else min(int(stop_frame), len(all_frames))
    step = max(int(stride), 1)
    selected_indices = list(range(start, stop, step))
    if not selected_indices:
        raise ValueError(
            f"No XDATCAR frames selected: start={start_frame}, stop={stop_frame}, stride={stride}, "
            f"available={len(all_frames)}"
        )

    order = species_order or list(basis["elements"])
    symbols = list(basis["symbols"])
    missing = sorted(set(symbols) - set(order))
    if missing:
        raise ValueError(f"--species-order is missing species present in POSCAR/XDATCAR: {','.join(missing)}")

    lattice = np.asarray(basis["lattice"], dtype=float)
    volume = float(abs(np.linalg.det(lattice)))
    frames = [
        {
            "index": idx,
            "symbols": symbols,
            "frac": np.asarray(all_frames[idx], dtype=float),
            "cell": lattice,
            "volume_A3": volume,
        }
        for idx in selected_indices
    ]
    metadata = {
        "engine": "vasp-xdatcar",
        "poscar": str(poscar.resolve()),
        "xdatcar": str(xdatcar.resolve()),
        "n_total_frames": len(all_frames),
        "n_selected_frames": len(frames),
        "selected_frame_indices_0based": selected_indices,
        "species_order": order,
        "elements": basis["elements"],
        "counts": dict(zip(basis["elements"], basis["counts"])),
        "natoms": basis["natoms"],
        "volume_A3": volume,
    }
    return frames, metadata


def compute_partial_histograms_from_md_frames(
    frames: list[dict[str, Any]],
    species_order: list[str],
    r_edges: np.ndarray,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, float], float, int]:
    """Compute partial RDF histograms from generic frame dictionaries."""
    nbins = len(r_edges) - 1
    pair_hist = {
        (a, b): np.zeros(nbins, dtype=float)
        for ia, a in enumerate(species_order)
        for b in species_order[ia:]
    }
    composition_sum: Counter[str] = Counter()
    volume_sum = 0.0

    for frame in frames:
        symbols = np.asarray(frame["symbols"])
        cell = np.asarray(frame["cell"], dtype=float)
        frac = np.asarray(frame["frac"], dtype=float)
        n_atoms = len(symbols)

        dfrac = frac[:, None, :] - frac[None, :, :]
        dfrac -= np.rint(dfrac)
        distances = np.linalg.norm(dfrac @ cell, axis=2)
        iu = np.triu_indices(n_atoms, k=1)
        d = distances[iu]
        s1 = symbols[iu[0]]
        s2 = symbols[iu[1]]

        composition_sum.update(Counter(symbols))
        volume_sum += float(frame["volume_A3"])

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


def rdf_pdf_from_vasp_xdatcar(
    *,
    poscar: Path,
    xdatcar: Path,
    outdir: Path,
    prefix: str = "md",
    species_order: list[str] | None = None,
    start_frame: int = 0,
    stop_frame: int | None = None,
    stride: int = 1,
    rmax: float = 8.0,
    dr: float = 0.02,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Write partial RDF and weighted total g(r) for a VASP XDATCAR."""
    frames, metadata = vasp_frames_from_xdatcar(
        poscar=poscar,
        xdatcar=xdatcar,
        start_frame=start_frame,
        stop_frame=stop_frame,
        stride=stride,
        species_order=species_order,
    )
    order = list(metadata["species_order"])
    r_edges = np.arange(0.0, rmax + dr, dr)
    if len(r_edges) < 2:
        raise ValueError("--rmax/--dr produced no RDF bins")

    pair_hist, avg_counts, avg_volume, n_frames = compute_partial_histograms_from_md_frames(frames, order, r_edges)
    r, partial = normalize_partial_rdfs(pair_hist, avg_counts, avg_volume, n_frames, r_edges)

    if weights is None:
        weights = {symbol: float(avg_counts[symbol]) for symbol in order}
        weight_mode = "composition_count_fallback"
    else:
        missing_weights = sorted(set(order) - set(weights))
        if missing_weights:
            raise ValueError(f"--weights missing species: {','.join(missing_weights)}")
        weight_mode = "custom"
    g_total, concentrations = weighted_total_gr_constant(order, partial, avg_counts, weights)

    outdir.mkdir(parents=True, exist_ok=True)
    partial_cols: dict[str, np.ndarray] = {}
    for ia, a in enumerate(order):
        for b in order[ia:]:
            partial_cols[f"g_{a}_{b}"] = partial[(a, b)]
    partial_cols["g_total_weighted"] = g_total

    partial_csv = outdir / f"{prefix}_partial_rdfs.csv"
    total_csv = outdir / f"{prefix}_total_pdf.csv"
    meta_json = outdir / f"{prefix}_rdf_pdf_metadata.json"
    write_multi_csv(partial_csv, "r_A", r, partial_cols)
    write_multi_csv(total_csv, "r_A", r, {"g_total_weighted": g_total})

    metadata.update(
        {
            "schema": "atomi.md.rdf_pdf.v1",
            "prefix": prefix,
            "rmax_A": rmax,
            "dr_A": dr,
            "avg_counts": avg_counts,
            "avg_volume_A3": avg_volume,
            "rho0_atoms_per_A3": sum(avg_counts.values()) / avg_volume,
            "concentrations": concentrations,
            "weights": weights,
            "weight_mode": weight_mode,
            "outputs": {
                "partial_rdfs_csv": str(partial_csv.resolve()),
                "total_pdf_csv": str(total_csv.resolve()),
                "metadata_json": str(meta_json.resolve()),
            },
        }
    )
    write_json(meta_json, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="General MD RDF/PDF analysis.")
    sub = parser.add_subparsers(dest="command", required=True)
    vasp = sub.add_parser("vasp-xdatcar", help="Compute RDF/PDF from a VASP POSCAR/XDATCAR trajectory.")
    vasp.add_argument("--poscar", type=Path, default=Path("POSCAR"))
    vasp.add_argument("--xdatcar", type=Path, default=Path("XDATCAR"))
    vasp.add_argument("--outdir", type=Path, default=Path("md_rdf_pdf"))
    vasp.add_argument("--prefix", default="md")
    vasp.add_argument("--species-order", help="Comma-separated species order, e.g. U,C.")
    vasp.add_argument("--start-frame", type=int, default=0)
    vasp.add_argument("--stop-frame", type=int)
    vasp.add_argument("--stride", type=int, default=1)
    vasp.add_argument("--rmax", type=float, default=8.0)
    vasp.add_argument("--dr", type=float, default=0.02)
    vasp.add_argument(
        "--weights",
        nargs="*",
        default=[],
        help="Optional total g(r) weights like U=92 C=6. Defaults to composition-count fallback.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "vasp-xdatcar":
        metadata = rdf_pdf_from_vasp_xdatcar(
            poscar=args.poscar,
            xdatcar=args.xdatcar,
            outdir=args.outdir,
            prefix=args.prefix,
            species_order=parse_csv_list(args.species_order),
            start_frame=args.start_frame,
            stop_frame=args.stop_frame,
            stride=args.stride,
            rmax=args.rmax,
            dr=args.dr,
            weights=parse_weights(args.weights) if args.weights else None,
        )
        print(json.dumps(metadata, indent=2))
        return metadata
    return None


if __name__ == "__main__":
    main()
