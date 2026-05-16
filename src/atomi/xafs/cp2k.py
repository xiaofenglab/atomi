#!/usr/bin/env python3
"""Prepare absorber-centered XAFS clusters from CP2K AIMD XYZ trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

from atomi.cp2k.extract_frames import STEP_PATTERNS
from atomi.xafs.larch_md import (
    cluster_records_for_site,
    resolve_absorber,
    select_evenly,
    write_cluster_xyz,
    write_feff_input,
    write_feff_run_script,
    write_json,
    write_rows_csv,
    xray_edge_info,
)


@dataclass(frozen=True)
class Cp2kXyzFrame:
    index: int
    comment: str
    symbols: list[str]
    coords: np.ndarray
    step: int | None


def read_cp2k_xyz_trajectory(path: Path) -> list[Cp2kXyzFrame]:
    frames: list[Cp2kXyzFrame] = []
    with path.open("r", encoding="utf-8") as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            first = first.strip()
            if not first:
                continue
            try:
                natoms = int(first)
            except ValueError as exc:
                raise ValueError(f"Malformed XYZ atom-count line in {path}: {first!r}") from exc
            comment = handle.readline().rstrip("\n")
            symbols: list[str] = []
            coords: list[list[float]] = []
            for _ in range(natoms):
                parts = handle.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed XYZ atom line in {path}")
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            frames.append(
                Cp2kXyzFrame(
                    index=len(frames),
                    comment=comment,
                    symbols=symbols,
                    coords=np.asarray(coords, dtype=float),
                    step=parse_step(comment),
                )
            )
    if not frames:
        raise ValueError(f"No XYZ frames found in {path}")
    return frames


def parse_step(comment: str) -> int | None:
    for pattern in STEP_PATTERNS:
        match = pattern.search(comment)
        if match:
            return int(match.group(1))
    return None


def parse_cp2k_input_metadata(path: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "project": None,
        "temperature": None,
        "timestep_fs": None,
        "md_steps": None,
        "cell_abc_A": None,
        "coord_file_name": None,
    }
    stack: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("&END"):
            if stack:
                stack.pop()
            continue
        if upper.startswith("&"):
            stack.append(upper[1:].split()[0])
            continue
        if stack and stack[-1] == "GLOBAL":
            match = re.match(r"PROJECT\s+(.+)", line, re.IGNORECASE)
            if match:
                info["project"] = match.group(1).strip()
        if "CELL" in stack:
            match = re.match(
                r"ABC\s+([0-9Ee+\-.]+)\s+([0-9Ee+\-.]+)\s+([0-9Ee+\-.]+)",
                line,
                re.IGNORECASE,
            )
            if match:
                info["cell_abc_A"] = [float(match.group(i)) for i in range(1, 4)]
        if "TOPOLOGY" in stack:
            match = re.match(r"COORD_FILE_NAME\s+(.+)", line, re.IGNORECASE)
            if match:
                info["coord_file_name"] = match.group(1).strip()
        if "MD" in stack:
            match = re.match(r"TIMESTEP\s+([0-9Ee+\-.]+)", line, re.IGNORECASE)
            if match:
                info["timestep_fs"] = float(match.group(1))
            match = re.match(r"STEPS\s+([0-9]+)", line, re.IGNORECASE)
            if match:
                info["md_steps"] = int(match.group(1))
            match = re.match(r"TEMPERATURE\s+([0-9Ee+\-.]+)", line, re.IGNORECASE)
            if match:
                info["temperature"] = float(match.group(1))
    return info


def parse_cell(args: argparse.Namespace, input_info: dict[str, object]) -> tuple[list[float] | None, str]:
    if args.cell is not None:
        return [float(v) for v in args.cell], "command-line --cell"
    if args.box is not None:
        return [float(args.box), float(args.box), float(args.box)], "command-line --box"
    if args.cell_from_inp and isinstance(input_info.get("cell_abc_A"), list):
        return [float(v) for v in input_info["cell_abc_A"]], "CP2K input &CELL ABC"
    return None, "none"


def frame_time_ps(frame: Cp2kXyzFrame, timestep_fs: float | None, traj_every: int) -> float | None:
    if timestep_fs is None:
        return None
    step = frame.step if frame.step is not None else frame.index * traj_every
    return float(step) * float(timestep_fs) / 1000.0


def select_frames(
    args: argparse.Namespace,
    frames: list[Cp2kXyzFrame],
    input_info: dict[str, object],
) -> list[Cp2kXyzFrame]:
    selected = frames
    if args.start is not None or args.stop is not None:
        selected = selected[args.start : args.stop]
    if args.last_ps is not None:
        timestep_fs = args.timestep_fs
        if timestep_fs is None and isinstance(input_info.get("timestep_fs"), (float, int)):
            timestep_fs = float(input_info["timestep_fs"])
        if timestep_fs is None:
            raise ValueError("--last-ps requires --timestep-fs or a CP2K input with &MD TIMESTEP")
        times = [frame_time_ps(frame, timestep_fs, args.traj_every) for frame in selected]
        finite_times = [time for time in times if time is not None]
        if not finite_times:
            raise ValueError("Could not infer frame times for --last-ps")
        cutoff = max(finite_times) - float(args.last_ps)
        selected = [frame for frame, time in zip(selected, times) if time is not None and time >= cutoff - 1.0e-12]
    elif args.last_frames is not None and args.last_frames > 0:
        selected = selected[-args.last_frames :]
    if args.frame_step is not None and args.frame_step > 1:
        selected = selected[:: args.frame_step]
    selected = select_evenly(selected, args.max_frames)
    if not selected:
        raise ValueError("No CP2K AIMD frames selected for XAFS preparation.")
    return selected


def frames_to_atoms(frames: list[Cp2kXyzFrame], cell_abc: list[float] | None, pbc: bool) -> list[Atoms]:
    out = []
    for frame in frames:
        atoms = Atoms(symbols=frame.symbols, positions=frame.coords)
        if cell_abc is not None:
            atoms.set_cell(cell_abc)
            atoms.set_pbc(bool(pbc))
        else:
            atoms.set_pbc(False)
        out.append(atoms)
    return out


def write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(comment + "\n")
        for symbol, coord in zip(symbols, coords):
            handle.write(f"{symbol:2s} {coord[0]: .8f} {coord[1]: .8f} {coord[2]: .8f}\n")


def write_selected_frames(outdir: Path, frames: list[Cp2kXyzFrame]) -> dict[str, str]:
    multi = outdir / "xafs_cp2k_selected_lastwindow.xyz"
    last = outdir / "xafs_cp2k_selected_lastframe.xyz"
    avg = outdir / "xafs_cp2k_selected_avgframe.xyz"
    with multi.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(f"{len(frame.symbols)}\n")
            handle.write(frame.comment + "\n")
            for symbol, coord in zip(frame.symbols, frame.coords):
                handle.write(f"{symbol:2s} {coord[0]: .8f} {coord[1]: .8f} {coord[2]: .8f}\n")
    write_xyz(last, frames[-1].symbols, frames[-1].coords, frames[-1].comment)
    first_symbols = frames[0].symbols
    if all(frame.symbols == first_symbols for frame in frames):
        avg_coords = np.mean(np.asarray([frame.coords for frame in frames], dtype=float), axis=0)
        write_xyz(avg, first_symbols, avg_coords, f"Average of {len(frames)} selected CP2K AIMD frames")
    return {
        "multi_frame_xyz": str(multi),
        "last_frame_xyz": str(last),
        "avg_frame_xyz": str(avg) if avg.exists() else "",
    }


def summarize_selected_frames(
    frames: list[Cp2kXyzFrame],
    atoms_frames: list[Atoms],
    cell_abc: list[float] | None,
) -> dict:
    all_coords = np.vstack([frame.coords for frame in frames])
    span = np.ptp(all_coords, axis=0)
    summary: dict[str, object] = {
        "n_frames": len(frames),
        "n_atoms": len(frames[0].symbols),
        "first_frame_index": frames[0].index,
        "last_frame_index": frames[-1].index,
        "first_step": frames[0].step,
        "last_step": frames[-1].step,
        "composition": dict(Counter(frames[0].symbols)),
        "coordinate_span_A": {"x": float(span[0]), "y": float(span[1]), "z": float(span[2])},
        "cell_abc_A": cell_abc,
        "pbc_used": cell_abc is not None and bool(np.any(atoms_frames[0].pbc)),
    }
    if cell_abc is not None:
        summary["cell_volume_A3"] = float(np.prod(cell_abc))
    return summary


def run_prepare(args: argparse.Namespace) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    frames_all = read_cp2k_xyz_trajectory(args.xyz)
    input_info = parse_cp2k_input_metadata(args.inp) if args.inp is not None else {}
    selected_frames = select_frames(args, frames_all, input_info)
    cell_abc, cell_source = parse_cell(args, input_info)
    atoms_frames = frames_to_atoms(selected_frames, cell_abc, args.pbc)
    selected_outputs = write_selected_frames(args.outdir, selected_frames)

    symbols = selected_frames[0].symbols
    absorber = resolve_absorber(symbols, args.absorber)
    edge_info = xray_edge_info(absorber, args.edge)
    absorber_indices = [i for i, symbol in enumerate(symbols) if symbol == absorber]
    if not absorber_indices:
        raise ValueError(f"No absorber atoms with symbol {absorber!r} were found.")
    site_indices = absorber_indices[: args.max_absorber_sites] if args.max_absorber_sites > 0 else absorber_indices

    cluster_root = args.outdir / "clusters"
    cluster_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cluster_dirs: list[Path] = []
    for selected_order, (frame, atoms) in enumerate(zip(selected_frames, atoms_frames)):
        frame_dir = cluster_root / f"frame_{frame.index:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for site_order, absorber_index in enumerate(site_indices, start=1):
            records = cluster_records_for_site(atoms, absorber_index, absorber, args.cluster_radius)
            counts = Counter(row["symbol"] for row in records)
            site_dir = frame_dir / f"site_{absorber_index + 1:06d}_{absorber}"
            site_dir.mkdir(parents=True, exist_ok=True)
            title = (
                f"Atomi CP2K {absorber} {args.edge} frame={frame.index} "
                f"site={absorber_index + 1} radius={args.cluster_radius:g}A"
            )
            feff_meta = write_feff_input(
                site_dir / "feff.inp",
                records,
                absorber,
                args.edge,
                args.cluster_radius,
                args.s02,
                title,
            )
            write_cluster_xyz(site_dir / "cluster.xyz", records, title)
            site_meta = {
                "source": "cp2k_xyz",
                "selected_order": selected_order,
                "frame_index_0based": frame.index,
                "frame_comment": frame.comment,
                "step": frame.step,
                "absorber_index_1based": absorber_index + 1,
                "absorber": absorber,
                "edge": args.edge,
                "cluster_radius_A": args.cluster_radius,
                "n_cluster_atoms": len(records),
                "counts": dict(counts),
                "r_max_A": max(row["distance_A"] for row in records),
                "r_min_scatterer_A": min((row["distance_A"] for row in records[1:]), default=0.0),
                "potential_map": feff_meta["potential_map"],
                "cell_abc_A": cell_abc,
                "cell_source": cell_source,
                "pbc_used": cell_abc is not None and args.pbc,
            }
            write_json(site_dir / "cluster_metadata.json", site_meta)
            cluster_dirs.append(site_dir)
            rows.append(
                {
                    "selected_order": selected_order,
                    "frame_index_0based": frame.index,
                    "step": frame.step if frame.step is not None else "",
                    "site_order": site_order,
                    "absorber_index_1based": absorber_index + 1,
                    "absorber": absorber,
                    "edge": args.edge,
                    "cluster_dir": str(site_dir),
                    "feff_inp": str(site_dir / "feff.inp"),
                    "cluster_xyz": str(site_dir / "cluster.xyz"),
                    "n_cluster_atoms": len(records),
                    "r_max_A": site_meta["r_max_A"],
                    "r_min_scatterer_A": site_meta["r_min_scatterer_A"],
                    "formula_counts": " ".join(f"{k}:{v}" for k, v in sorted(counts.items())),
                }
            )

    cluster_list = args.outdir / "cluster_dirs.txt"
    cluster_list.write_text("\n".join(str(path) for path in cluster_dirs) + "\n", encoding="utf-8")
    write_feff_run_script(args.outdir / "run_feff_all.sh", cluster_list, args.feff_exe)
    fieldnames = [
        "selected_order",
        "frame_index_0based",
        "step",
        "site_order",
        "absorber_index_1based",
        "absorber",
        "edge",
        "cluster_dir",
        "feff_inp",
        "cluster_xyz",
        "n_cluster_atoms",
        "r_min_scatterer_A",
        "r_max_A",
        "formula_counts",
    ]
    write_rows_csv(args.outdir / "cluster_index.csv", rows, fieldnames)
    frame_summary = summarize_selected_frames(selected_frames, atoms_frames, cell_abc)
    metadata = {
        "mode": "xafs_cp2k_prepare",
        "source": {
            "xyz": str(args.xyz.resolve()),
            "inp": str(args.inp.resolve()) if args.inp is not None else None,
            "n_frames_total": len(frames_all),
            "selection": {
                "start": args.start,
                "stop": args.stop,
                "frame_step": args.frame_step,
                "last_frames": args.last_frames,
                "last_ps": args.last_ps,
                "max_frames": args.max_frames,
            },
            "input_metadata": input_info,
        },
        "selected_outputs": selected_outputs,
        "frame_summary": frame_summary,
        "cell_source": cell_source,
        "absorber_request": args.absorber,
        "absorber": absorber,
        "edge": args.edge,
        "edge_info": edge_info,
        "cluster_radius_A": args.cluster_radius,
        "s02": args.s02,
        "n_frames": len(selected_frames),
        "n_absorber_sites_per_frame": len(site_indices),
        "n_clusters": len(cluster_dirs),
        "cluster_index": str(args.outdir / "cluster_index.csv"),
        "cluster_dirs": str(cluster_list),
        "run_script": str(args.outdir / "run_feff_all.sh"),
    }
    write_json(args.outdir / "xafs_cp2k_prepare_metadata.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xafs_cp2k_prepare",
        description="Generate absorber-centered FEFF inputs from CP2K AIMD XYZ trajectories.",
    )
    parser.add_argument("--xyz", type=Path, required=True, help="Multi-frame CP2K AIMD XYZ, usually *-pos.xyz.")
    parser.add_argument("--inp", type=Path, help="Optional CP2K input for timestep/cell metadata.")
    parser.add_argument("--outdir", type=Path, default=Path("xafs_cp2k_prepare"))
    parser.add_argument("--absorber", default="metal", help="Absorber symbol, or 'metal' for first metal in the frame.")
    parser.add_argument("--metal", dest="absorber", help="Alias for --absorber when selecting a metal explicitly.")
    parser.add_argument("--edge", default="L3")
    parser.add_argument("--cluster-radius", type=float, default=6.0)
    parser.add_argument("--max-absorber-sites", type=int, default=0, help="0 means all absorber sites in each frame.")
    parser.add_argument("--s02", type=float, default=1.0)
    parser.add_argument("--feff-exe", default="feff8l")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--frame-step", type=int)
    parser.add_argument("--last-frames", type=int, default=100)
    parser.add_argument("--last-ps", type=float, help="Use only frames in the final N ps of AIMD.")
    parser.add_argument("--timestep-fs", type=float, help="CP2K MD timestep in fs, used with --last-ps.")
    parser.add_argument(
        "--traj-every",
        type=int,
        default=1,
        help="MD steps between XYZ frames when comments lack step numbers.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Evenly down-select selected frames; 0 keeps all.",
    )
    parser.add_argument("--box", type=float, help="Cubic CP2K box length in Angstrom for minimum-image clusters.")
    parser.add_argument(
        "--cell",
        nargs=3,
        type=float,
        metavar=("A", "B", "C"),
        help="Orthorhombic cell lengths in Angstrom.",
    )
    parser.add_argument(
        "--cell-from-inp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use &CELL ABC from --inp when available.",
    )
    parser.add_argument(
        "--pbc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use minimum-image distances when a cell is available.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.xyz.is_file():
        parser.error(f"XYZ trajectory not found: {args.xyz}")
    if args.inp is not None and not args.inp.is_file():
        parser.error(f"CP2K input not found: {args.inp}")
    summary = run_prepare(args)
    print(f"Wrote {summary['n_clusters']} CP2K XAFS FEFF clusters to: {args.outdir.resolve()}")
    if summary["frame_summary"].get("cell_abc_A") is None:
        print("NOTE: no periodic cell was used; clusters are based on direct Cartesian distances.")


if __name__ == "__main__":
    main()
