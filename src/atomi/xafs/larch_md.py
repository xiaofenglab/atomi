#!/usr/bin/env python3
"""Build and compare MD-ensemble XAFS calculations through Larch/FEFF."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
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
from atomi.lammps import rdf_pdf
from atomi.lammps.box import format_box_summary
from atomi.structure import atomic_number as shared_atomic_number
from atomi.xafs.status import configured_larch_python, probe_larch_python


METAL_SYMBOLS = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
}

FALLBACK_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ca": 20,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Zr": 40,
    "Mo": 42,
    "Ag": 47,
    "Cd": 48,
    "Sn": 50,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "W": 74,
    "Pt": 78,
    "Au": 79,
    "Pb": 82,
    "Th": 90,
    "U": 92,
    "Np": 93,
    "Pu": 94,
}


def write_json(path: Path, data: dict) -> None:
    def normalize(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        return value

    path.write_text(json.dumps(normalize(data), indent=2), encoding="utf-8")


def md_cell_metadata(args: argparse.Namespace, frames: list) -> dict:
    requested_natoms = getattr(args, "natoms", None)
    natoms = float(requested_natoms) if requested_natoms is not None else float(len(frames[0]))
    formula_units = infer_formula_units(
        formula_units=getattr(args, "formula_units", None),
        natoms=natoms,
        atoms_per_formula_unit=getattr(args, "atoms_per_formula_unit", None),
        formula=getattr(args, "formula", None),
    )
    return cell_metadata(
        formula=getattr(args, "formula", None),
        natoms=natoms,
        atoms_per_formula_unit=getattr(args, "atoms_per_formula_unit", None),
        formula_units=formula_units,
        target_z=getattr(args, "target_z", None),
        cell_role="md-xafs-source-cell",
        normalization_basis="simulation-cell",
    )


def atomic_number(symbol: str) -> int:
    try:
        return shared_atomic_number(symbol)
    except Exception:
        pass
    if symbol not in FALLBACK_Z:
        raise ValueError(f"Atomic number for {symbol!r} is unavailable; install xraydb or ASE.")
    return int(FALLBACK_Z[symbol])


def xray_edge_info(symbol: str, edge: str) -> dict:
    info = {
        "element": symbol,
        "edge": edge,
        "source": "unavailable",
        "energy_eV": None,
        "fluorescence_yield": None,
        "jump_ratio": None,
    }
    try:
        import xraydb  # type: ignore

        edge_obj = xraydb.xray_edge(symbol, edge)
        info.update(
            {
                "source": "xraydb",
                "energy_eV": float(getattr(edge_obj, "energy", math.nan)),
                "fluorescence_yield": float(getattr(edge_obj, "fyield", math.nan)),
                "jump_ratio": float(getattr(edge_obj, "jump_ratio", math.nan)),
            }
        )
    except Exception as exc:
        info["warning"] = f"xraydb edge lookup failed: {exc}"
    return info


def resolve_absorber(symbols: list[str], requested: str) -> str:
    if requested.lower() not in {"metal", "auto", "first-metal"}:
        return requested
    for symbol in symbols:
        if symbol in METAL_SYMBOLS:
            return symbol
    for symbol in symbols:
        if symbol != "H":
            return symbol
    return symbols[0]


def write_rows_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_numeric_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "*", ";")):
            continue
        pieces = stripped.replace(",", " ").split()
        if len(pieces) < 2:
            continue
        try:
            xs.append(float(pieces[0]))
            ys.append(float(pieces[1]))
        except ValueError:
            continue
    if not xs:
        raise ValueError(f"No numeric two-column data found in {path}")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def write_xy(path: Path, x: np.ndarray, y: np.ndarray, xname: str, yname: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {xname} {yname}\n")
        for xv, yv in zip(x, y):
            handle.write(f"{xv:.10g} {yv:.10g}\n")


def select_evenly(items: list, maximum: int | None) -> list:
    if maximum is None or maximum <= 0 or len(items) <= maximum:
        return items
    indices = np.linspace(0, len(items) - 1, maximum)
    selected = sorted({int(round(value)) for value in indices})
    return [items[i] for i in selected]


def load_pdf_frames(pdf_dir: Path, temperature: float | None) -> tuple[list, dict]:
    summary_path: Path | None = None
    series_summary = pdf_dir / "series_summary.json"
    if series_summary.exists():
        payload = json.loads(series_summary.read_text(encoding="utf-8"))
        series = payload.get("series", [])
        if not series:
            raise ValueError(f"No series items found in {series_summary}")
        if temperature is None and len(series) != 1:
            available = ", ".join(f"{float(item['temperature']):g}" for item in series)
            raise ValueError(f"--temperature is required for PDF series input. Available T: {available}")
        if temperature is None:
            item = series[0]
        else:
            item = min(series, key=lambda row: abs(float(row["temperature"]) - temperature))
        summary_path = Path(item["summary_json"])
    else:
        summaries = sorted(pdf_dir.glob("*_summary.json"))
        if not summaries:
            raise FileNotFoundError(f"No pdf_lammps *_summary.json found in {pdf_dir}")
        summary_path = summaries[0]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = summary.get("selected_outputs", {})
    traj = selected.get("multi_frame_extxyz") or selected.get("last_frame_extxyz")
    if not traj:
        raise ValueError(f"{summary_path} does not contain selected extxyz outputs.")
    frames = rdf_pdf.read_frames_from_traj(Path(traj), None, None, None)
    return frames, {"source": "pdf_dir", "pdf_dir": str(pdf_dir), "summary_json": str(summary_path)}


def choose_one_record(records: list[dict], temperature: float | None) -> dict:
    if not records:
        raise ValueError("No NPT production records found for XAFS preparation.")
    if temperature is None:
        if len(records) == 1:
            return records[0]
        available = ", ".join(f"{float(record['temperature']):g}" for record in records)
        raise ValueError(f"--temperature is required when multiple NPT stages exist. Available T: {available}")
    return min(records, key=lambda record: abs(float(record["temperature"]) - temperature))


def load_frames_for_prepare(args: argparse.Namespace) -> tuple[list, dict]:
    if args.pdf_dir is not None:
        frames, summary = load_pdf_frames(args.pdf_dir, args.temperature)
    elif args.dump is not None:
        if not args.type_map:
            raise ValueError("--type-map is required when reading a LAMMPS dump")
        if args.dt is None or args.dump_every is None:
            raise ValueError("--dt and --dump-every are required when reading a LAMMPS dump")
        frames, source = rdf_pdf.read_frames_from_dump(
            args.dump,
            args.dump_format,
            rdf_pdf.parse_type_map(args.type_map),
            args.dt,
            args.dump_every,
            args.window_ps,
        )
        summary = {"source": "dump", **source}
    elif args.traj is not None:
        frames = rdf_pdf.read_frames_from_traj(args.traj, args.start, args.stop, args.frame_step)
        summary = {"source": "traj", "trajectory_file": str(args.traj.resolve())}
    else:
        series_args = SimpleNamespace(
            config=args.config,
            md_root=args.md_root,
            config_dir=args.config_dir,
            config_glob=args.config_glob,
            duplicate_policy=args.duplicate_policy,
            t_min=args.t_min,
            t_max=args.t_max,
            dt=args.dt,
        )
        records_all, records = rdf_pdf.discover_series_records(series_args)
        record = choose_one_record(records, args.temperature)
        dump = rdf_pdf.find_record_dump(record)
        dump_every = rdf_pdf.record_dump_every(record, args.dump_every)
        timestep_ps = float(args.dt if args.dt is not None else record.get("timestep_ps", 0.0001))
        if not args.type_map:
            raise ValueError("--type-map is required for config/md-root LAMMPS dump discovery")
        frames, source = rdf_pdf.read_frames_from_dump(
            dump,
            args.dump_format,
            rdf_pdf.parse_type_map(args.type_map),
            timestep_ps,
            dump_every,
            args.window_ps,
        )
        summary = {
            "source": "config" if args.config else "md_root",
            "n_records_discovered": len(records_all),
            "selected_record": {
                "temperature": record["temperature"],
                "stage_name": record["stage_name"],
                "log_path": str(record["log_path"]),
                "dump_path": str(dump),
                "dump_every": dump_every,
                "timestep_ps": timestep_ps,
            },
            **source,
        }

    if args.frame_step is not None and args.traj is None:
        frames = frames[:: args.frame_step]
        summary["post_window_frame_step"] = args.frame_step
    frames = select_evenly(frames, args.max_frames)
    summary["n_frames_after_xafs_selection"] = len(frames)
    if not frames:
        raise ValueError("No frames selected for XAFS preparation.")
    return frames, summary


def minimum_image_relative_positions(atoms, absorber_index: int) -> tuple[np.ndarray, np.ndarray]:
    positions = atoms.get_positions()
    cell = np.asarray(atoms.cell.array, dtype=float)
    if not np.any(atoms.pbc) or abs(float(np.linalg.det(cell))) < 1.0e-12:
        rel = positions - positions[absorber_index]
        return rel, np.linalg.norm(rel, axis=1)
    inv_cell = np.linalg.inv(cell)
    frac = positions @ inv_cell
    dfrac = frac - frac[absorber_index]
    dfrac -= np.rint(dfrac)
    rel = dfrac @ cell
    distances = np.linalg.norm(rel, axis=1)
    return rel, distances


def cluster_records_for_site(
    atoms,
    absorber_index: int,
    absorber: str,
    cluster_radius: float,
) -> list[dict]:
    rel, distances = minimum_image_relative_positions(atoms, absorber_index)
    symbols = atoms.get_chemical_symbols()
    records = []
    for index, (symbol, vector, distance) in enumerate(zip(symbols, rel, distances)):
        if index == absorber_index or distance <= cluster_radius + 1.0e-8:
            records.append(
                {
                    "atom_index_1based": index + 1,
                    "symbol": symbol,
                    "x": float(vector[0]),
                    "y": float(vector[1]),
                    "z": float(vector[2]),
                    "distance_A": float(distance),
                    "is_absorber": index == absorber_index,
                }
            )
    records.sort(key=lambda row: (not row["is_absorber"], row["distance_A"], row["atom_index_1based"]))
    if records[0]["symbol"] != absorber or not records[0]["is_absorber"]:
        raise RuntimeError("Internal error while building absorber-centered cluster.")
    return records


def write_cluster_xyz(path: Path, records: list[dict], comment: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(records)}\n")
        handle.write(comment + "\n")
        for row in records:
            handle.write(
                f"{row['symbol']} {row['x']:.8f} {row['y']:.8f} {row['z']:.8f} "
                f"# src_index={row['atom_index_1based']} r={row['distance_A']:.6f}\n"
            )


def write_feff_input(
    path: Path,
    records: list[dict],
    absorber: str,
    edge: str,
    cluster_radius: float,
    s02: float,
    title: str,
) -> dict:
    scatter_symbols = sorted({row["symbol"] for row in records[1:]})
    potential_map = {symbol: i + 1 for i, symbol in enumerate(scatter_symbols)}
    lines = [
        f"TITLE {title}",
        "TITLE Generated by Atomi xafs_lammps_prepare from an MD ensemble cluster",
        f"EDGE {edge}",
        f"S02 {s02:.6g}",
        "CONTROL 1 1 1 1 1 1",
        "PRINT 1 0 0 0 0 0",
        f"RPATH {cluster_radius:.6g}",
        f"SCF {min(cluster_radius, 6.0):.6g}",
        f"FMS {min(cluster_radius, 8.0):.6g}",
        "",
        "POTENTIALS",
        "* ipot Z element label",
        f"  0 {atomic_number(absorber):3d} {absorber:2s} {absorber}_absorber",
    ]
    for symbol, ipot in potential_map.items():
        lines.append(f"  {ipot:d} {atomic_number(symbol):3d} {symbol:2s} {symbol}")
    lines.extend(["", "ATOMS", "* x y z ipot tag distance_A"])
    for row in records:
        ipot = 0 if row["is_absorber"] else potential_map[row["symbol"]]
        tag = f"{row['symbol']}{int(row['atom_index_1based'])}"
        lines.append(
            f"{row['x']:12.6f} {row['y']:12.6f} {row['z']:12.6f} "
            f"{ipot:2d} {tag:12s} {row['distance_A']:10.6f}"
        )
    lines.extend(["END", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"potential_map": potential_map}


def write_feff_run_script(path: Path, cluster_dirs_file: Path, feff_exe: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'FEFF_EXE="${{FEFF_EXE:-{feff_exe}}}"',
        f"CLUSTER_LIST={shlex.quote(str(cluster_dirs_file.name))}",
        "",
        'while IFS= read -r cluster_dir; do',
        '  [ -z "$cluster_dir" ] && continue',
        '  echo "Running FEFF in $cluster_dir"',
        '  (',
        '    cd "$cluster_dir"',
        '    if "$FEFF_EXE" feff.inp > feff.stdout.log 2> feff.stderr.log; then',
        "      exit 0",
        "    fi",
        '    "$FEFF_EXE" > feff.stdout.log 2> feff.stderr.log',
        "  )",
        "done < \"$CLUSTER_LIST\"",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def run_prepare(args: argparse.Namespace) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    frames, source_summary = load_frames_for_prepare(args)
    selected_outputs = rdf_pdf.write_selected_frames(args.outdir, "xafs_selected", frames)
    cell_meta = md_cell_metadata(args, frames)
    structure_stats = rdf_pdf.compute_structure_stats(frames)
    symbols = frames[0].get_chemical_symbols()
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
    for frame_index, atoms in enumerate(frames):
        frame_dir = cluster_root / f"frame_{frame_index:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for site_order, absorber_index in enumerate(site_indices, start=1):
            records = cluster_records_for_site(atoms, absorber_index, absorber, args.cluster_radius)
            counts = Counter(row["symbol"] for row in records)
            site_dir = frame_dir / f"site_{absorber_index + 1:06d}_{absorber}"
            site_dir.mkdir(parents=True, exist_ok=True)
            title = (
                f"Atomi {absorber} {args.edge} frame={frame_index} "
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
                "frame_index_0based": frame_index,
                "absorber_index_1based": absorber_index + 1,
                "absorber": absorber,
                "edge": args.edge,
                "cluster_radius_A": args.cluster_radius,
                "n_cluster_atoms": len(records),
                "counts": dict(counts),
                "r_max_A": max(row["distance_A"] for row in records),
                "r_min_scatterer_A": min((row["distance_A"] for row in records[1:]), default=0.0),
                "potential_map": feff_meta["potential_map"],
            }
            write_json(site_dir / "cluster_metadata.json", site_meta)
            cluster_dirs.append(site_dir)
            rows.append(
                {
                    "frame_index_0based": frame_index,
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
        "frame_index_0based",
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
    metadata = {
        "mode": "xafs_lammps_prepare",
        "source": source_summary,
        "selected_outputs": selected_outputs,
        "cell_metadata": cell_meta,
        "structure_stats": structure_stats,
        "absorber_request": args.absorber,
        "absorber": absorber,
        "edge": args.edge,
        "edge_info": edge_info,
        "cluster_radius_A": args.cluster_radius,
        "s02": args.s02,
        "n_frames": len(frames),
        "n_absorber_sites_per_frame": len(site_indices),
        "n_clusters": len(cluster_dirs),
        "cluster_index": str(args.outdir / "cluster_index.csv"),
        "cluster_dirs": str(cluster_list),
        "run_script": str(args.outdir / "run_feff_all.sh"),
        "roadmap": {
            "pdf_xafs_joint_analysis": (
                "Future Atomi workflow: combine pdf_md_compare/pdf_md_reweight with "
                "absorber-specific XAFS frame/site ranking so one MD ensemble can be "
                "constrained by both total scattering PDF and edge-specific XAFS."
            )
        },
    }
    write_json(args.outdir / "xafs_prepare_metadata.json", metadata)
    return metadata


def read_cluster_dirs(prepared_dir: Path) -> list[Path]:
    list_file = prepared_dir / "cluster_dirs.txt"
    if list_file.exists():
        return [Path(line.strip()) for line in list_file.read_text().splitlines() if line.strip()]
    index = prepared_dir / "cluster_index.csv"
    if index.exists():
        with index.open("r", newline="", encoding="utf-8") as handle:
            return [Path(row["cluster_dir"]) for row in csv.DictReader(handle) if row.get("cluster_dir")]
    raise FileNotFoundError(f"No cluster_dirs.txt or cluster_index.csv found in {prepared_dir}")


def run_feff_if_requested(cluster_dirs: list[Path], feff_exe: str, no_run: bool) -> list[dict]:
    rows = []
    executable = shutil.which(feff_exe) if not no_run else None
    for cluster_dir in cluster_dirs:
        row = {"cluster_dir": str(cluster_dir), "ran": False, "returncode": None, "status": "skipped"}
        if no_run:
            rows.append(row)
            continue
        if executable is None:
            row["status"] = f"missing executable: {feff_exe}"
            rows.append(row)
            continue
        stdout_log = cluster_dir / "feff.stdout.log"
        stderr_log = cluster_dir / "feff.stderr.log"
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            proc = subprocess.run([executable], cwd=cluster_dir, stdout=out, stderr=err, check=False)
        row.update({"ran": True, "returncode": proc.returncode, "status": "ok" if proc.returncode == 0 else "failed"})
        rows.append(row)
    return rows


def find_chi_file(cluster_dir: Path) -> Path | None:
    for name in ("chi.dat", "chi.dat1", "xmu.dat"):
        candidate = cluster_dir / name
        if candidate.exists():
            return candidate
    for candidate in sorted(cluster_dir.glob("chi*.dat")):
        if candidate.is_file():
            return candidate
    return None


def collect_chi_curves(cluster_dirs: list[Path], k_step: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    curves = []
    rows = []
    for cluster_dir in cluster_dirs:
        chi_file = find_chi_file(cluster_dir)
        if chi_file is None:
            rows.append({"cluster_dir": str(cluster_dir), "chi_file": "", "status": "missing"})
            continue
        try:
            k, chi = read_numeric_xy(chi_file)
        except Exception as exc:
            rows.append({"cluster_dir": str(cluster_dir), "chi_file": str(chi_file), "status": f"read-error: {exc}"})
            continue
        if len(k) < 2:
            rows.append({"cluster_dir": str(cluster_dir), "chi_file": str(chi_file), "status": "too-few-points"})
            continue
        order = np.argsort(k)
        curves.append((k[order], chi[order], chi_file))
        rows.append(
            {
                "cluster_dir": str(cluster_dir),
                "chi_file": str(chi_file),
                "status": "ok",
                "k_min": float(np.min(k)),
                "k_max": float(np.max(k)),
                "n_points": len(k),
            }
        )
    if not curves:
        raise ValueError("No usable FEFF chi.dat/xmu.dat files found in prepared clusters.")
    k_min = max(float(np.min(k)) for k, _, _ in curves)
    k_max = min(float(np.max(k)) for k, _, _ in curves)
    if k_max <= k_min:
        raise ValueError("FEFF chi curves have no overlapping k range.")
    grid = np.arange(k_min, k_max + 0.5 * k_step, k_step)
    matrix = np.asarray([np.interp(grid, k, chi) for k, chi, _ in curves], dtype=float)
    return grid, np.mean(matrix, axis=0), rows


def larch_xftf(
    k: np.ndarray,
    chi: np.ndarray,
    k_min: float,
    k_max: float,
    k_weight: int,
    dk: float,
    window: str,
) -> tuple[dict | None, str]:
    try:
        from larch import Group  # type: ignore
        from larch.xafs import xftf  # type: ignore
    except Exception as exc:
        external_python = configured_larch_python()
        if external_python:
            transform, status = external_larch_xftf(
                Path(external_python), k, chi, k_min, k_max, k_weight, dk, window
            )
            if transform is not None:
                return transform, status
            return None, f"xraylarch import failed in active env: {exc}; {status}"
        return None, f"xraylarch import failed: {exc}"
    try:
        group = Group(k=k, chi=chi)
        xftf(k, chi=chi, group=group, kmin=k_min, kmax=k_max, dk=dk, kweight=k_weight, window=window)
        return (
            {
                "r_A": np.asarray(group.r, dtype=float),
                "chir_mag": np.asarray(group.chir_mag, dtype=float),
                "chir_re": np.asarray(group.chir_re, dtype=float),
                "chir_im": np.asarray(group.chir_im, dtype=float),
            },
            "ok",
        )
    except Exception as exc:
        return None, f"xraylarch xftf failed: {exc}"


def external_larch_xftf(
    python_executable: Path,
    k: np.ndarray,
    chi: np.ndarray,
    k_min: float,
    k_max: float,
    k_weight: int,
    dk: float,
    window: str,
) -> tuple[dict | None, str]:
    probe = probe_larch_python(python_executable)
    if not probe.get("available"):
        return None, f"external Larch probe failed: {probe}"
    with tempfile.TemporaryDirectory(prefix="atomi-larch-xftf-") as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "chi_input.csv"
        output_path = tmpdir / "xftf_output.npz"
        np.savetxt(input_path, np.column_stack([k, chi]), delimiter=",", header="k,chi", comments="")
        script = (
            "import numpy as np, sys\n"
            "from larch import Group\n"
            "from larch.xafs import xftf\n"
            "inp, out = sys.argv[1], sys.argv[2]\n"
            "kmin, kmax, kweight, dk, window = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7]\n"
            "data = np.loadtxt(inp, delimiter=',', skiprows=1)\n"
            "k = data[:, 0]\n"
            "chi = data[:, 1]\n"
            "group = Group(k=k, chi=chi)\n"
            "xftf(k, chi=chi, group=group, kmin=kmin, kmax=kmax, dk=dk, kweight=kweight, window=window)\n"
            "np.savez(out, r_A=np.asarray(group.r), chir_mag=np.asarray(group.chir_mag), chir_re=np.asarray(group.chir_re), chir_im=np.asarray(group.chir_im))\n"
        )
        proc = subprocess.run(
            [
                str(python_executable.expanduser()),
                "-c",
                script,
                str(input_path),
                str(output_path),
                str(k_min),
                str(k_max),
                str(k_weight),
                str(dk),
                str(window),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not output_path.exists():
            return None, f"external xraylarch xftf failed: {proc.stderr.strip() or proc.stdout.strip()}"
        data = np.load(output_path)
        return (
            {
                "r_A": np.asarray(data["r_A"], dtype=float),
                "chir_mag": np.asarray(data["chir_mag"], dtype=float),
                "chir_re": np.asarray(data["chir_re"], dtype=float),
                "chir_im": np.asarray(data["chir_im"], dtype=float),
            },
            f"ok via external Larch python: {python_executable}",
        )


def plot_xy(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> bool:
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
    ax.plot(x, y, color="#d62728", linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def plot_compare(
    path: Path,
    k: np.ndarray,
    exp: np.ndarray,
    model: np.ndarray,
    k_weight: int,
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
    factor = np.power(k, k_weight)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k, model * factor, color="#d62728", linewidth=1.4, label="MD/Larch ensemble")
    ax.scatter(
        k,
        exp * factor,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        s=24,
        label="experiment",
    )
    ax.set_xlabel("k (A^-1)")
    ax.set_ylabel(f"k^{k_weight} chi(k)")
    ax.set_title("XAFS MD Ensemble vs Experiment")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def run_larch(args: argparse.Namespace) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "larch_python", None):
        os.environ["ATOMI_XAFS_LARCH_PYTHON"] = str(args.larch_python)
    cluster_dirs = read_cluster_dirs(args.prepared_dir)
    run_rows = run_feff_if_requested(cluster_dirs, args.feff_exe, args.no_run_feff)
    write_rows_csv(
        args.outdir / "feff_run_log.csv",
        run_rows,
        ["cluster_dir", "ran", "returncode", "status"],
    )
    k, chi, chi_rows = collect_chi_curves(cluster_dirs, args.k_step)
    write_rows_csv(
        args.outdir / "chi_curve_index.csv",
        chi_rows,
        ["cluster_dir", "chi_file", "status", "k_min", "k_max", "n_points"],
    )
    write_xy(args.outdir / "ensemble_chi_k.dat", k, chi, "k_A^-1", "chi")
    with (args.outdir / "ensemble_chi_k.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k_A^-1", "chi", f"k{args.k_weight}_chi"])
        writer.writeheader()
        for kv, cv in zip(k, chi):
            writer.writerow({"k_A^-1": kv, "chi": cv, f"k{args.k_weight}_chi": (kv**args.k_weight) * cv})

    plots = []
    if not args.no_plots and plot_xy(
        args.outdir / "ensemble_chi_k.png",
        k,
        np.power(k, args.k_weight) * chi,
        "k (A^-1)",
        f"k^{args.k_weight} chi(k)",
        "MD Ensemble XAFS",
    ):
        plots.append(str(args.outdir / "ensemble_chi_k.png"))

    transform, transform_status = larch_xftf(k, chi, args.k_min, args.k_max, args.k_weight, args.dk, args.window)
    transform_outputs = {}
    if transform is not None:
        r = transform["r_A"]
        mag = transform["chir_mag"]
        write_xy(args.outdir / "ensemble_chi_R_mag.dat", r, mag, "R_A", "chir_mag")
        transform_outputs["chir_mag"] = str(args.outdir / "ensemble_chi_R_mag.dat")
        if not args.no_plots and plot_xy(
            args.outdir / "ensemble_chi_R_mag.png",
            r,
            mag,
            "R (A)",
            "|chi(R)|",
            "MD Ensemble XAFS FT",
        ):
            plots.append(str(args.outdir / "ensemble_chi_R_mag.png"))

    metadata = {
        "mode": "xafs_larch_run",
        "prepared_dir": str(args.prepared_dir.resolve()),
        "feff_exe": args.feff_exe,
        "n_clusters": len(cluster_dirs),
        "n_chi_curves_used": sum(1 for row in chi_rows if row["status"] == "ok"),
        "k_grid": {"min": float(k[0]), "max": float(k[-1]), "step": args.k_step},
        "larch_transform_status": transform_status,
        "outputs": {
            "ensemble_chi_k_dat": str(args.outdir / "ensemble_chi_k.dat"),
            "ensemble_chi_k_csv": str(args.outdir / "ensemble_chi_k.csv"),
            **transform_outputs,
        },
        "plots": plots,
        "archive": str(
            args.archive_path.resolve() if args.archive_path else default_archive_path(args.outdir).resolve()
        )
        if not args.no_archive_output
        else None,
    }
    write_json(args.outdir / "xafs_larch_run_metadata.json", metadata)
    if not args.no_archive_output:
        archive = archive_output_dir(args.outdir, args.archive_path)
        metadata["archive"] = str(archive)
        write_json(args.outdir / "xafs_larch_run_metadata.json", metadata)
    return metadata


def resolve_model_chi(args: argparse.Namespace) -> Path:
    if args.md_chi is not None:
        return args.md_chi
    candidates = [
        args.xafs_dir / "ensemble_chi_k.dat",
        args.xafs_dir / "ensemble_chi_k.csv",
        args.xafs_dir / "xafs_larch_run" / "ensemble_chi_k.dat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No ensemble_chi_k.dat/csv found in {args.xafs_dir}; pass --md-chi")


def fit_scale_baseline(
    x: np.ndarray,
    exp: np.ndarray,
    model: np.ndarray,
    baseline_order: int,
) -> tuple[np.ndarray, dict]:
    columns = [model]
    if baseline_order >= 0:
        centered = x - np.mean(x)
        for order in range(baseline_order + 1):
            columns.append(centered**order)
    design = np.vstack(columns).T
    coeffs, *_ = np.linalg.lstsq(design, exp, rcond=None)
    fitted = design @ coeffs
    return fitted, {"scale": float(coeffs[0]), "baseline_coefficients": [float(v) for v in coeffs[1:]]}


def curve_metrics(exp: np.ndarray, model: np.ndarray) -> dict:
    resid = model - exp
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    denom = float(np.sum(exp**2))
    r_factor = float(np.sum(resid**2) / denom) if denom > 0 else math.nan
    return {"rmse": rmse, "mae": mae, "r_factor": r_factor}


def run_compare(args: argparse.Namespace) -> dict:
    args.outdir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_chi(args)
    k_model, chi_model = read_numeric_xy(model_path)
    k_exp, chi_exp = read_numeric_xy(args.exp_chi)
    k_min = args.k_min if args.k_min is not None else max(float(np.min(k_model)), float(np.min(k_exp)))
    k_max = args.k_max if args.k_max is not None else min(float(np.max(k_model)), float(np.max(k_exp)))
    mask = (k_exp >= k_min) & (k_exp <= k_max)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Too few experimental points in the requested/common k range.")
    k_ref = k_exp[mask]
    exp_ref = chi_exp[mask]
    model_ref = np.interp(k_ref, k_model, chi_model)
    if args.fit_scale:
        model_fit, fit_info = fit_scale_baseline(k_ref, exp_ref, model_ref, args.baseline_order)
    else:
        model_fit, fit_info = model_ref, {"scale": 1.0, "baseline_coefficients": []}
    metrics = curve_metrics(exp_ref, model_fit)
    with (args.outdir / "xafs_compare_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k_A^-1", "exp_chi", "md_chi", "fit_chi", "residual"])
        writer.writeheader()
        for kv, ev, mv, fv in zip(k_ref, exp_ref, model_ref, model_fit):
            writer.writerow({"k_A^-1": kv, "exp_chi": ev, "md_chi": mv, "fit_chi": fv, "residual": fv - ev})
    plots = []
    if not args.no_plots and plot_compare(
        args.outdir / "xafs_compare_chi_k.png",
        k_ref,
        exp_ref,
        model_fit,
        args.k_weight,
    ):
        plots.append(str(args.outdir / "xafs_compare_chi_k.png"))
    metadata = {
        "mode": "xafs_md_compare",
        "xafs_dir": str(args.xafs_dir.resolve()),
        "model_chi": str(model_path.resolve()),
        "exp_chi": str(args.exp_chi.resolve()),
        "k_range_A^-1": {"min": k_min, "max": k_max},
        "k_weight": args.k_weight,
        "fit": fit_info,
        "metrics": metrics,
        "outputs": {"compare_curve": str(args.outdir / "xafs_compare_curve.csv")},
        "plots": plots,
        "archive": str(
            args.archive_path.resolve() if args.archive_path else default_archive_path(args.outdir).resolve()
        )
        if not args.no_archive_output
        else None,
    }
    write_json(args.outdir / "xafs_compare_metadata.json", metadata)
    if not args.no_archive_output:
        archive = archive_output_dir(args.outdir, args.archive_path)
        metadata["archive"] = str(archive)
        write_json(args.outdir / "xafs_compare_metadata.json", metadata)
    return metadata


def build_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xafs_lammps_prepare",
        description="Generate absorber-centered FEFF inputs from LAMMPS/MD ensemble frames.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf-dir", type=Path, help="Existing pdf_lammps or pdf_lammps_series output directory.")
    source.add_argument("--dump", type=Path, help="LAMMPS dump trajectory.")
    source.add_argument("--traj", type=Path, help="ASE-readable trajectory, usually extxyz.")
    source.add_argument("--config", nargs="+", help="One or more md-engine production config JSON files.")
    source.add_argument(
        "--md-root",
        type=Path,
        help="MD engine root. NPT folders are scanned; NVT folders are ignored.",
    )
    parser.add_argument("--config-dir")
    parser.add_argument("--config-glob", default="*.json")
    parser.add_argument(
        "--duplicate-policy",
        choices=["highest_config_order", "first", "error"],
        default="highest_config_order",
    )
    parser.add_argument("--dump-format", default="lammps-dump-text")
    parser.add_argument("--type-map", nargs="*", default=[], help="LAMMPS type map, e.g. 1=O 2=U")
    parser.add_argument("--formula", help="Formula label for shared cell metadata, e.g. UO2.")
    parser.add_argument("--natoms", type=float, help="Atoms in the source MD/simulation cell.")
    parser.add_argument("--atoms-per-formula-unit", type=float, help="Atoms per formula unit.")
    parser.add_argument("--formula-units", type=float, help="Formula units in the source MD/simulation cell.")
    parser.add_argument("--target-z", type=float, help="Formula units in the normalized target crystallographic cell.")
    parser.add_argument("--dt", type=float, help="MD timestep in ps for dump/config discovery.")
    parser.add_argument("--dump-every", type=int, help="LAMMPS steps between dump frames.")
    parser.add_argument("--window-ps", type=float, default=10.0, help="Last trajectory window for MD dump input.")
    parser.add_argument("--temperature", type=float, help="Select nearest NPT/PDF-series temperature.")
    parser.add_argument("--t-min", type=float)
    parser.add_argument("--t-max", type=float)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--frame-step", type=int)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--absorber", default="metal", help="Absorber symbol, or 'metal' for first metal in the frame.")
    parser.add_argument("--edge", default="L3")
    parser.add_argument("--cluster-radius", type=float, default=6.0)
    parser.add_argument("--max-absorber-sites", type=int, default=0, help="0 means all absorber sites in each frame.")
    parser.add_argument("--s02", type=float, default=1.0)
    parser.add_argument("--feff-exe", default="feff8l")
    parser.add_argument("--outdir", type=Path, default=Path("xafs_prepare"))
    return parser


def build_larch_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xafs_larch_run",
        description="Run/collect FEFF-Larch XAFS curves from xafs_lammps_prepare clusters.",
    )
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("xafs_larch_run"))
    default_feff = os.environ.get("ATOMI_XAFS_FEFF_EXE") or os.environ.get("ATOMI_FEFF_EXE") or "feff8l"
    parser.add_argument("--feff-exe", default=default_feff)
    parser.add_argument(
        "--larch-python",
        type=Path,
        default=os.environ.get("ATOMI_XAFS_LARCH_PYTHON"),
        help="Python executable for an external Larch environment; defaults to ATOMI_XAFS_LARCH_PYTHON.",
    )
    parser.add_argument("--no-run-feff", action="store_true", help="Only collect existing chi.dat files.")
    parser.add_argument("--k-step", type=float, default=0.05)
    parser.add_argument("--k-min", type=float, default=2.5)
    parser.add_argument("--k-max", type=float, default=12.0)
    parser.add_argument("--k-weight", type=int, default=2)
    parser.add_argument("--dk", type=float, default=4.0)
    parser.add_argument("--window", default="kaiser")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--no-archive-output", action="store_true")
    return parser


def build_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xafs_md_compare",
        description="Compare MD/Larch ensemble chi(k) with experimental XAFS chi(k).",
    )
    parser.add_argument("--xafs-dir", type=Path, required=True)
    parser.add_argument("--md-chi", type=Path, help="Explicit MD/Larch chi(k) file.")
    parser.add_argument("--exp-chi", type=Path, required=True, help="Experimental two-column k chi(k) file.")
    parser.add_argument("--k-min", type=float)
    parser.add_argument("--k-max", type=float)
    parser.add_argument("--k-weight", type=int, default=2)
    parser.add_argument("--fit-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-order", type=int, choices=[-1, 0, 1, 2], default=0)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--no-archive-output", action="store_true")
    return parser


def prepare_main(argv: Optional[list[str]] = None) -> None:
    parser = build_prepare_parser()
    args = parser.parse_args(argv)
    summary = run_prepare(args)
    print(format_box_summary(summary["structure_stats"]["md_box"], label="XAFS MD box"))
    print(f"Wrote {summary['n_clusters']} FEFF clusters to: {args.outdir.resolve()}")


def larch_run_main(argv: Optional[list[str]] = None) -> None:
    parser = build_larch_run_parser()
    args = parser.parse_args(argv)
    summary = run_larch(args)
    print(f"Averaged {summary['n_chi_curves_used']} XAFS curves into: {args.outdir.resolve()}")
    if "failed" in str(summary.get("larch_transform_status", "")):
        print(f"Larch transform warning: {summary['larch_transform_status']}", file=sys.stderr)


def compare_main(argv: Optional[list[str]] = None) -> None:
    parser = build_compare_parser()
    args = parser.parse_args(argv)
    summary = run_compare(args)
    print(f"Wrote XAFS comparison to: {args.outdir.resolve()}")
    print(
        "Metrics: "
        f"RMSE={summary['metrics']['rmse']:.6g}, "
        f"MAE={summary['metrics']['mae']:.6g}, "
        f"R-factor={summary['metrics']['r_factor']:.6g}"
    )


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: xafs {prepare,run,compare} ...")
        print("")
        print("Convenience dispatcher for xafs_lammps_prepare, xafs_larch_run, and xafs_md_compare.")
        return
    mode, rest = argv[0], argv[1:]
    if mode == "prepare":
        prepare_main(rest)
    elif mode == "run":
        larch_run_main(rest)
    elif mode == "compare":
        compare_main(rest)
    else:
        raise SystemExit(f"Unknown XAFS mode {mode!r}; expected prepare, run, or compare.")


if __name__ == "__main__":
    main()
