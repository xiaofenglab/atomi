from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.io import read, write

from atomi.lammps.elastic import read_lammps_thermo_table
from atomi.vasp.prefail import (
    choose_reference,
    copy_vasp_template,
    frac_diff_pbc,
    iter_lammps_dump_frames,
    parse_atoms_block,
    parse_box_bounds,
    reorder_md_to_reference,
)
from atomi.vasp.prep import (
    resolve_input_poscar,
    species_order_from_atoms,
    summarize_atoms,
    template_poscar,
    validate_vasp_template,
)


@dataclass
class SnapshotRecord:
    run_dir: Path
    stage_name: str
    temperature_K: float | None
    chunk_name: str
    dump_path: Path
    timestep: int
    source_config: Path
    metadata: dict | None = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_from_base(path: str | Path, base: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def parse_atom_type_items(items: list[str] | None) -> dict[int, str]:
    mapping = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Bad atom type map item, expected type=Element: {item}")
        key, value = item.split("=", 1)
        mapping[int(key.strip())] = value.strip()
    return mapping


def infer_atom_type_map(cfg: dict, cli_items: list[str] | None) -> dict[int, str]:
    cli_map = parse_atom_type_items(cli_items)
    if cli_map:
        return cli_map

    for key in ("atom_type_map", "lammps_atom_type_map"):
        if key in cfg:
            return {int(k): str(v) for k, v in cfg[key].items()}

    if "species_order" in cfg:
        return {index + 1: species for index, species in enumerate(cfg["species_order"])}

    # Backward-compatible default for the current md-engine U/O input writer.
    if "mass_O" in cfg and "mass_U" in cfg:
        return {1: "O", 2: "U"}

    raise ValueError(
        "Could not infer LAMMPS atom types. Add --atom-type-map 1=O 2=U "
        "or put atom_type_map in the md-engine JSON."
    )


def reference_poscar_from_config(
    cfg: dict,
    config_path: Path,
    cli_poscar: Path | None,
    template_dir: Path | None,
) -> Path:
    if cli_poscar is not None:
        return cli_poscar.expanduser().resolve()
    if "reference_poscar" in cfg:
        return resolve_from_base(cfg["reference_poscar"], config_path.parent)
    templated = template_poscar(template_dir)
    if templated is not None:
        return templated.resolve()
    return resolve_input_poscar(None, template_dir)


def stage_passed(stage_dir: Path) -> bool:
    return (stage_dir / "PASS").exists()


def stage_temperature_k(stage: dict) -> float | None:
    for key in (
        "temperature",
        "temperature_end",
        "target_temperature",
        "target_temperature_K",
        "T",
    ):
        if key in stage:
            return float(stage[key])

    name = str(stage.get("name", ""))
    matches = re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*K\b", name, flags=re.IGNORECASE)
    if matches:
        return float(matches[-1])
    return None


def parse_temperature_values(values: Iterable[str] | None) -> set[float]:
    temperatures = set()
    for value in values or []:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if part.lower().endswith("k"):
                part = part[:-1]
            temperatures.add(float(part))
    return temperatures


def temperature_allowed(stage: dict, args: argparse.Namespace) -> bool:
    target_temperatures = parse_temperature_values(args.temperature)
    range_active = args.temperature_range is not None
    if not target_temperatures and not range_active:
        return True

    temperature = stage_temperature_k(stage)
    if temperature is None:
        print(f"[warning] Could not infer stage temperature; skipping {stage.get('name')}")
        return False

    tol = args.temperature_tolerance
    if any(abs(temperature - target) <= tol for target in target_temperatures):
        return True

    if range_active:
        tmin, tmax = sorted(float(value) for value in args.temperature_range)
        return (tmin - tol) <= temperature <= (tmax + tol)

    return False


def stage_allowed(stage_name: str, args: argparse.Namespace) -> bool:
    if args.stage and stage_name not in set(args.stage):
        return False
    if args.exclude_stage and stage_name in set(args.exclude_stage):
        return False
    if args.stage_regex and re.search(args.stage_regex, stage_name) is None:
        return False
    return True


def elasticity_stage_allowed(stage: dict, args: argparse.Namespace) -> bool:
    if not getattr(args, "elasticity_correction", False):
        return True
    name = str(stage.get("name", ""))
    deformation = stage.get("deformation") or {}
    if stage.get("elastic_run") or stage.get("thermo_stress") or deformation:
        return True
    return re.search(r"(elastic|stress|nvt_stress)", name, flags=re.IGNORECASE) is not None


def temperature_label(value: float | None) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    return str(int(round(value))) if abs(value - round(value)) < 1.0e-9 else f"{value:g}".replace(".", "p")


def strain_fraction_label(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    magnitude = abs(float(value))
    per_mille = magnitude * 1000.0
    if abs(per_mille - round(per_mille)) < 1.0e-9:
        return sign + f"{int(round(per_mille)):03d}"
    return sign + f"{int(round(magnitude * 10000)):04d}"


def safe_label(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._+-]+", "_", text).strip("_") or "case"


def stage_strain_metadata(stage: dict) -> dict:
    name = str(stage.get("name", ""))
    deformation = stage.get("deformation") or {}
    mode = str(deformation.get("mode", "ref") or "ref")
    strain = float(deformation.get("strain", 0.0) or 0.0)
    voigt = deformation.get("voigt_strain")
    if not isinstance(voigt, list) or len(voigt) != 6:
        voigt = [0.0] * 6
        if mode in ("hydro", "isotropic", "volumetric", "volume"):
            voigt[0] = strain
            voigt[1] = strain
            voigt[2] = strain
        elif mode in ("xx", "yy", "zz", "yz", "xz", "xy"):
            voigt[("xx", "yy", "zz", "yz", "xz", "xy").index(mode)] = strain

    lowered = f"{mode} {name}".lower()
    is_reference = abs(strain) < 1.0e-12 or mode in {"ref", "none", "unstrained"}
    if is_reference:
        family = "unstrained_ref"
        label = "unstrained_ref"
    elif any(key in lowered for key in ("hydro", "isotropic", "volumetric", "volume")):
        family = "hydrostatic"
        label = f"hydro_e_{strain_fraction_label(strain)}"
    elif any(key in lowered for key in ("ortho", "volume_conserving", "volume-conserving", "vc")):
        family = "orthorhombic_volume_conserving"
        axis = mode if mode in {"xx", "yy", "zz", "x", "y", "z"} else "vc"
        label = f"ortho_vc_{axis[-1]}_{strain_fraction_label(strain)}"
    elif mode in {"xx", "yy", "zz"}:
        family = "uniaxial_tetragonal"
        label = f"uniaxial_{mode[0]}_{strain_fraction_label(strain)}"
    elif mode in {"xy", "xz", "yz"}:
        family = "shear"
        label = f"shear_{mode}_{strain_fraction_label(strain)}"
    else:
        family = "strained"
        label = f"{safe_label(mode)}_{strain_fraction_label(strain)}"

    deformation_gradient = np.eye(3, dtype=float)
    deformation_gradient[0, 0] += float(voigt[0])
    deformation_gradient[1, 1] += float(voigt[1])
    deformation_gradient[2, 2] += float(voigt[2])
    deformation_gradient[1, 2] += float(voigt[3])
    deformation_gradient[0, 2] += float(voigt[4])
    deformation_gradient[0, 1] += float(voigt[5])

    return {
        "strain_family": family,
        "strain_mode": mode,
        "strain_amplitude": strain,
        "strain_label": label,
        "voigt_strain": [float(value) for value in voigt],
        "deformation_gradient": deformation_gradient.tolist(),
        "is_unstrained_reference": is_reference,
    }


def chunk_sort_key(path: Path) -> tuple[int, float, str]:
    match = re.match(r"chunk_(\d+)$", path.name)
    number = int(match.group(1)) if match else -1
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    return number, mtime, path.name


def find_latest_chunk_with_dump(
    stage: dict,
    stage_dir: Path,
    dump_glob: str,
) -> tuple[Path, Path] | None:
    candidates = []
    if "chunk_name" in stage:
        candidates.append(stage_dir / stage["chunk_name"])
    candidates.extend(stage_dir.glob("chunk_*"))
    candidates.append(stage_dir / "chunk_production")

    unique = []
    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        unique.append(candidate)

    for chunk_dir in sorted(unique, key=chunk_sort_key, reverse=True):
        dumps = sorted(chunk_dir.glob(dump_glob))
        if dumps:
            return chunk_dir, dumps[-1].resolve()
    return None


def passed_stages_with_dumps(
    cfg: dict,
    config_path: Path,
    root: Path,
    args: argparse.Namespace,
) -> list[tuple[dict, Path, Path, Path]]:
    stages_root = root / "stages"
    selected = []
    for stage in cfg.get("stages", []):
        stage_name = stage.get("name")
        if not stage_name or not stage_allowed(stage_name, args):
            continue
        if not elasticity_stage_allowed(stage, args):
            continue
        if not temperature_allowed(stage, args):
            continue
        stage_dir = stages_root / stage_name
        if not stage_passed(stage_dir):
            continue
        found = find_latest_chunk_with_dump(stage, stage_dir, args.dump_glob)
        if found is None:
            print(f"[warning] PASS stage has no dump matching {args.dump_glob}: {stage_dir}")
            continue
        chunk_dir, dump_path = found
        selected.append((stage, stage_dir, chunk_dir, dump_path))
    return selected


def select_dump_frames(
    dump_path: Path,
    explicit_timesteps: set[int],
    last_frames: int,
    stride: int,
) -> list[tuple[int, str, list[str], str, list[str]]]:
    if stride < 1:
        raise ValueError("--stride must be >= 1")

    if explicit_timesteps:
        selected = []
        found = set()
        for frame in iter_lammps_dump_frames(dump_path):
            timestep = frame[0]
            if timestep in explicit_timesteps:
                selected.append(frame)
                found.add(timestep)
        missing = sorted(explicit_timesteps - found)
        if missing:
            print(f"[warning] Timesteps not found in {dump_path}: {missing}")
        return selected

    tail: deque = deque(maxlen=max(last_frames * stride, last_frames, 1))
    for frame in iter_lammps_dump_frames(dump_path):
        tail.append(frame)
    return list(tail)[::stride][-last_frames:]


def config_timestep_ps(cfg: dict, args: argparse.Namespace) -> float | None:
    if getattr(args, "timestep_ps", None) is not None:
        return float(args.timestep_ps)
    for key in ("timestep_ps", "timestep"):
        if key in cfg:
            return float(cfg[key])
    return None


def select_elastic_frames(
    dump_path: Path,
    explicit_timesteps: set[int],
    frames_per_run: int,
    tail_fraction: float,
    min_separation_ps: float,
    timestep_ps: float | None,
) -> tuple[list[tuple[int, str, list[str], str, list[str]]], list[str]]:
    if explicit_timesteps:
        return select_dump_frames(dump_path, explicit_timesteps, frames_per_run, 1), []
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("--elastic-tail-fraction must be between 0 and 1.")
    all_frames = list(iter_lammps_dump_frames(dump_path))
    if not all_frames:
        return [], ["dump file has no frames"]
    start = max(0, int(math.floor(len(all_frames) * (1.0 - tail_fraction))))
    tail = all_frames[start:]
    if len(tail) <= frames_per_run:
        return tail, []

    min_step_separation = 0
    if timestep_ps and timestep_ps > 0.0 and min_separation_ps > 0.0:
        min_step_separation = int(round(min_separation_ps / timestep_ps))

    selected: list[tuple[int, str, list[str], str, list[str]]] = []
    for frame in reversed(tail):
        timestep = frame[0]
        if min_step_separation and any(abs(timestep - kept[0]) < min_step_separation for kept in selected):
            continue
        selected.append(frame)
        if len(selected) >= frames_per_run:
            break

    warnings = []
    if len(selected) < frames_per_run:
        warnings.append(
            "could not satisfy requested time separation; filled with evenly spaced stationary tail frames"
        )
        picked = {frame[0] for frame in selected}
        for index in np.linspace(0, len(tail) - 1, frames_per_run, dtype=int):
            frame = tail[int(index)]
            if frame[0] not in picked:
                selected.append(frame)
                picked.add(frame[0])
            if len(selected) >= frames_per_run:
                break

    return sorted(selected[:frames_per_run], key=lambda frame: frame[0]), warnings


def latest_log_file(chunk_dir: Path) -> Path | None:
    candidates = sorted(chunk_dir.glob("log.in.*"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        candidates = sorted(chunk_dir.glob("log.*"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def elastic_quality_warnings(chunk_dir: Path, args: argparse.Namespace) -> tuple[bool, list[str]]:
    log_path = latest_log_file(chunk_dir)
    if log_path is None:
        return True, ["no LAMMPS thermo log found; drift/stress-spike screening limited"]
    try:
        data = read_lammps_thermo_table(log_path)
    except Exception as exc:
        return True, [f"could not parse LAMMPS thermo log {log_path.name}: {exc}"]
    if "Step" not in data or len(data["Step"]) < 3:
        return True, [f"too few thermo rows in {log_path.name}; drift/stress-spike screening limited"]

    start = max(0, int(math.floor(len(data["Step"]) * (1.0 - args.elastic_tail_fraction))))
    warnings: list[str] = []
    ok = True
    if "Temp" in data:
        temp = np.asarray(data["Temp"][start:], dtype=float)
        mean_temp = float(np.mean(temp)) if len(temp) else 0.0
        if mean_temp and abs(float(temp[-1] - temp[0])) / abs(mean_temp) > args.elastic_max_temp_drift_fraction:
            ok = False
            warnings.append("temperature drift in stationary tail exceeds threshold")
    stress_cols = [key for key in ("Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy") if key in data]
    if len(stress_cols) == 6:
        pressure = np.vstack([np.asarray(data[key][start:], dtype=float) for key in stress_cols]).T
        norms = np.linalg.norm(pressure, axis=1)
        if len(norms) >= 3:
            median = float(np.median(norms))
            maximum = float(np.max(norms))
            std = float(np.std(norms))
            if median > 1.0e-12 and maximum > args.elastic_stress_spike_factor * median and maximum > median + 5.0 * std:
                ok = False
                warnings.append("abnormal pressure/stress spike detected in stationary tail")
    else:
        warnings.append("thermo log lacks full Pxx/Pyy/Pzz/Pyz/Pxz/Pxy stress columns")
    if args.allow_elastic_quality_warnings:
        ok = True
    return ok, warnings


def species_counts(atoms: Atoms) -> dict[str, int]:
    return dict(Counter(atoms.get_chemical_symbols()))


def local_distortion_score(atoms: Atoms, reference: Atoms) -> float:
    if len(atoms) != len(reference):
        return math.inf
    diff = frac_diff_pbc(atoms.get_scaled_positions(wrap=True), reference.get_scaled_positions(wrap=True))
    cart = diff @ atoms.cell.array
    return float(np.sqrt(np.mean(np.sum(cart * cart, axis=1))))


def reduce_large_frame_to_subcells(
    md_atoms: Atoms,
    reference_small: Atoms,
    replicate: tuple[int, int, int],
    species_order: tuple[str, ...],
    keep_all: bool,
    max_subcells: int,
) -> tuple[list[tuple[tuple[int, int, int], Atoms, float]], list[str]]:
    expected = species_counts(reference_small)
    frac = md_atoms.get_scaled_positions(wrap=True)
    symbols = np.array(md_atoms.get_chemical_symbols())
    parent_cell = np.asarray(md_atoms.cell.array, dtype=float)
    local_cell = parent_cell.copy()
    for axis, rep in enumerate(replicate):
        local_cell[axis, :] /= float(rep)

    candidates: list[tuple[tuple[int, int, int], Atoms, float]] = []
    warnings: list[str] = []
    for ix in range(replicate[0]):
        for iy in range(replicate[1]):
            for iz in range(replicate[2]):
                offset = np.asarray([ix, iy, iz], dtype=float)
                lower = offset / np.asarray(replicate, dtype=float)
                upper = (offset + 1.0) / np.asarray(replicate, dtype=float)
                lower_ok = frac >= lower - 1.0e-12
                upper_limit = np.where(
                    np.asarray([ix, iy, iz]) == np.asarray(replicate) - 1,
                    upper + 1.0e-12,
                    upper - 1.0e-12,
                )
                mask = np.all(lower_ok & (frac < upper_limit), axis=1)
                local_symbols = list(symbols[mask])
                local_frac = frac[mask] * np.asarray(replicate, dtype=float) - offset
                sub_atoms = Atoms(symbols=local_symbols, cell=local_cell, pbc=True)
                sub_atoms.set_scaled_positions(local_frac)
                counts = species_counts(sub_atoms)
                if counts != expected:
                    warnings.append(
                        f"skipped subcell {(ix, iy, iz)}: counts {counts} do not match reference {expected}"
                    )
                    continue
                try:
                    reordered = reorder_md_to_reference(reference_small, sub_atoms, species_order=species_order)
                except Exception as exc:
                    warnings.append(f"skipped subcell {(ix, iy, iz)}: could not reorder to reference ({exc})")
                    continue
                candidates.append(((ix, iy, iz), reordered, local_distortion_score(reordered, reference_small)))

    candidates.sort(key=lambda item: item[2], reverse=True)
    if keep_all:
        return candidates, warnings
    return candidates[: max(1, max_subcells)], warnings


def write_runlist(records: list[SnapshotRecord], runlist: Path) -> None:
    ensure_dir(runlist.parent)
    base = runlist.parent.resolve()
    lines = []
    for record in records:
        try:
            lines.append(str(record.run_dir.resolve().relative_to(base)))
        except ValueError:
            lines.append(str(record.run_dir.resolve()))
    runlist.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_index(records: list[SnapshotRecord], index_path: Path) -> None:
    ensure_dir(index_path.parent)
    fields = [
        "run_dir",
        "stage_name",
        "temperature_K",
        "chunk_name",
        "dump_path",
        "timestep",
        "source_config",
        "training_role",
        "strain_family",
        "strain_mode",
        "strain_amplitude",
        "is_unstrained_reference",
        "subcell_offset",
        "expected_labels",
        "warnings",
    ]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metadata = record.metadata or {}
            writer.writerow(
                {
                    "run_dir": str(record.run_dir.resolve()),
                    "stage_name": record.stage_name,
                    "temperature_K": record.temperature_K,
                    "chunk_name": record.chunk_name,
                    "dump_path": str(record.dump_path.resolve()),
                    "timestep": record.timestep,
                    "source_config": str(record.source_config.resolve()),
                    "training_role": metadata.get("intended_training_role", ""),
                    "strain_family": metadata.get("strain_family", ""),
                    "strain_mode": metadata.get("strain_mode", ""),
                    "strain_amplitude": metadata.get("strain_amplitude", ""),
                    "is_unstrained_reference": metadata.get("is_unstrained_reference", ""),
                    "subcell_offset": metadata.get("subcell_offset", ""),
                    "expected_labels": ",".join(metadata.get("expected_labels", [])),
                    "warnings": "; ".join(metadata.get("warnings", [])),
                }
            )


def parse_timesteps(values: Iterable[str] | None) -> set[int]:
    timesteps: set[int] = set()
    for value in values or []:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            timesteps.add(int(part))
    return timesteps


def prepare_snapshot_runs(args: argparse.Namespace) -> list[SnapshotRecord]:
    config_path = args.config.expanduser().resolve()
    cfg = read_json(config_path)
    root = args.project_root.expanduser().resolve() if args.project_root else config_path.parent
    if args.elasticity_correction and str(args.output_root) == "MD_SNAPSHOT_CANDIDATES":
        args.output_root = Path("UO2_ELASTIC_STRESS_CANDIDATES")
    output_root = ensure_dir(args.output_root.expanduser().resolve())
    template_dir = args.vasp_template.resolve() if args.vasp_template else None
    if args.reduce_large_md_to_2x2x2 and args.reference_poscar_2x2x2 is not None and args.poscar is None:
        reference_poscar = args.reference_poscar_2x2x2.expanduser().resolve()
    else:
        reference_poscar = reference_poscar_from_config(cfg, config_path, args.poscar, template_dir)
    reference = read(reference_poscar)
    reference_small = None
    if args.reduce_large_md_to_2x2x2:
        if args.reference_poscar_2x2x2 is None:
            raise ValueError("--reference-poscar-2x2x2 is required with --reduce-large-md-to-2x2x2")
        reference_poscar = args.reference_poscar_2x2x2.expanduser().resolve()
        reference_small = read(reference_poscar)
        reference = reference_small
    species_order = tuple(args.species_order) if args.species_order else species_order_from_atoms(reference)
    atom_type_map = infer_atom_type_map(cfg, args.atom_type_map)
    timestep_ps = config_timestep_ps(cfg, args)

    needs_template_poscar = (
        args.poscar is None and "reference_poscar" not in cfg and not args.reduce_large_md_to_2x2x2
    )
    validate_vasp_template(template_dir, atoms=reference, require_poscar=needs_template_poscar)
    explicit_timesteps = parse_timesteps(args.timesteps)
    selected_stage_info = passed_stages_with_dumps(cfg, config_path, root, args)
    if not selected_stage_info:
        raise RuntimeError("No PASS stages with LAMMPS dump files were found.")

    print(f"MD-engine config : {config_path}")
    print(f"Project root     : {root}")
    print(f"Reference POSCAR : {reference_poscar}")
    print(f"Reference        : {summarize_atoms(reference)}")
    print(f"Species order    : {', '.join(species_order)}")
    print(f"Atom type map    : {atom_type_map}")
    print(f"Output root      : {output_root}")
    print(f"PASS stages used : {len(selected_stage_info)}")
    if args.elasticity_correction:
        print("Mode             : elasticity_correction")
        print("DFT labels       : energy, forces, stress (MD stress is metadata only)")
        print("Stress warning   : DFT OUTCAR stress must be preserved during extxyz conversion.")
    if args.reduce_large_md_to_2x2x2:
        print(f"Reduction        : large MD -> reference subcells using replicate {tuple(args.large_to_small_replicate)}")

    records: list[SnapshotRecord] = []
    summary = []
    for stage, _stage_dir, chunk_dir, dump_path in selected_stage_info:
        stage_name = stage["name"]
        temperature = stage_temperature_k(stage)
        elastic_meta = stage_strain_metadata(stage)
        if args.elasticity_correction:
            if abs(float(elastic_meta["strain_amplitude"])) > args.elastic_max_strain and not args.include_large_elastic_strain:
                print(
                    f"[skip] {stage_name}: strain {elastic_meta['strain_amplitude']} exceeds "
                    f"--elastic-max-strain {args.elastic_max_strain:g}"
                )
                continue
            quality_ok, quality_warnings = elastic_quality_warnings(chunk_dir, args)
            if not quality_ok:
                print(f"[skip] {stage_name}: {'; '.join(quality_warnings)}")
                continue
            frames, selection_warnings = select_elastic_frames(
                dump_path,
                explicit_timesteps,
                args.elastic_frames_per_run,
                args.elastic_tail_fraction,
                args.elastic_min_separation_ps,
                timestep_ps,
            )
            quality_warnings.extend(selection_warnings)
        else:
            quality_warnings = []
            frames = select_dump_frames(
                dump_path,
                explicit_timesteps,
                args.last_frames,
                args.stride,
            )
        if args.max_frames_per_stage:
            frames = frames[-args.max_frames_per_stage :]
        print(f"[stage] {stage_name}: {len(frames)} frames from {chunk_dir.name}/{dump_path.name}")

        for frame_index, frame in enumerate(frames, start=1):
            timestep, box_header, box_lines, atoms_header, atom_lines = frame
            cell, origin = parse_box_bounds(box_header, box_lines)
            md_atoms = parse_atoms_block(atom_lines, atoms_header, atom_type_map, origin, cell)
            ref_for_match = choose_reference(reference, md_atoms, None)
            reordered = reorder_md_to_reference(ref_for_match, md_atoms, species_order=species_order)

            candidates: list[tuple[str, Atoms, dict]] = []
            if args.reduce_large_md_to_2x2x2:
                assert reference_small is not None
                reduced, reduce_warnings = reduce_large_frame_to_subcells(
                    reordered,
                    reference_small,
                    tuple(int(value) for value in args.large_to_small_replicate),
                    species_order,
                    keep_all=args.keep_all_subcells,
                    max_subcells=args.max_subcells_per_frame,
                )
                for offset, sub_atoms, distortion in reduced:
                    label = "subcell_" + "".join(str(value) for value in offset)
                    candidates.append(
                        (
                            label,
                            sub_atoms,
                            {
                                "subcell_offset": list(offset),
                                "local_distortion_score_A": distortion,
                                "reduced_cell_matrix": sub_atoms.cell.array.tolist(),
                                "large_to_small_replicate": list(args.large_to_small_replicate),
                                "reduction_note": (
                                    "Homogeneous elastic strain is preserved in the reduced subcell lattice. "
                                    "Long-wavelength correlations from the large MD cell are not fully retained."
                                ),
                                "warnings": reduce_warnings,
                            },
                        )
                    )
            else:
                candidates.append(("frame", reordered, {"warnings": []}))

            if not candidates:
                print(f"[warning] {stage_name} timestep {timestep}: no valid reduced subcells/candidates")
                continue

            for candidate_label, candidate_atoms, candidate_extra in candidates:
                if args.elasticity_correction:
                    t_label = f"T{temperature_label(temperature)}K"
                    case_label = safe_label(str(elastic_meta["strain_label"]))
                    run_dir = output_root / t_label / case_label / f"md_{timestep:010d}_{candidate_label}"
                else:
                    run_dir = output_root / stage_name / chunk_dir.name / f"md_{timestep:010d}"
                run_dir = ensure_dir(run_dir)
                write(run_dir / "POSCAR", candidate_atoms, format="vasp", direct=True, vasp5=True, sort=False)
                copy_vasp_template(template_dir, run_dir, copy_all=args.copy_template_all)

                warnings = [*quality_warnings, *candidate_extra.get("warnings", [])]
                info = {
                    "stage_name": stage_name,
                    "temperature_K": temperature,
                    "chunk_name": chunk_dir.name,
                    "dump_path": str(dump_path.resolve()),
                    "timestep": timestep,
                    "frame_index_in_selection": frame_index,
                    "source_md_run": str((root / "stages" / stage_name).resolve()),
                    "source_config": str(config_path.resolve()),
                    "reference_poscar": str(reference_poscar.resolve()),
                    "composition": summarize_atoms(candidate_atoms),
                    "intended_training_role": "elasticity_correction" if args.elasticity_correction else "md_snapshot",
                    "expected_labels": ["energy", "forces", "stress"] if args.elasticity_correction else ["energy", "forces"],
                    "md_stress_used_as_training_label": False,
                    "dft_stress_required": bool(args.elasticity_correction),
                    "warnings": warnings,
                    **(elastic_meta if args.elasticity_correction else {}),
                    **candidate_extra,
                }
                (run_dir / "case_info.json").write_text(
                    json.dumps(info, indent=2) + "\n",
                    encoding="utf-8",
                )
                summary.append(info)
                records.append(
                    SnapshotRecord(
                        run_dir=run_dir,
                        stage_name=stage_name,
                        temperature_K=temperature,
                        chunk_name=chunk_dir.name,
                        dump_path=dump_path,
                        timestep=timestep,
                        source_config=config_path,
                        metadata=info,
                    )
                )

    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-md-snapshot-candidates",
        description="Harvest successful md-engine LAMMPS frames into VASP-ready snapshot folders.",
    )
    parser.add_argument("--config", required=True, type=Path, help="md-engine JSON config.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Workflow root containing stages/. Defaults to the config directory.",
    )
    parser.add_argument(
        "--poscar",
        type=Path,
        default=None,
        help="Reference POSCAR. Defaults to reference_poscar in config, VASP_TEMPLATE/POSCAR, then ./POSCAR.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("MD_SNAPSHOT_CANDIDATES"))
    parser.add_argument("--vasp-template", type=Path, default=None)
    parser.add_argument(
        "--copy-template-all",
        action="store_true",
        help="Copy every template file/subdirectory except POSCAR.",
    )
    parser.add_argument("--runlist", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--dump-glob", default="dump.*.lammpstrj")
    parser.add_argument("--stage", action="append", default=None, help="Use only this stage; repeatable.")
    parser.add_argument("--exclude-stage", action="append", default=None)
    parser.add_argument("--stage-regex", default=None)
    parser.add_argument(
        "--temperature",
        "-T",
        action="append",
        default=None,
        help="Use only stages at this temperature in K; repeatable or comma-separated.",
    )
    parser.add_argument(
        "--temperature-range",
        nargs=2,
        type=float,
        metavar=("T_MIN", "T_MAX"),
        default=None,
        help="Use only stages with temperature between T_MIN and T_MAX K.",
    )
    parser.add_argument(
        "--temperature-tolerance",
        type=float,
        default=0.5,
        help="Tolerance in K for matching --temperature and range edges.",
    )
    parser.add_argument(
        "--last-frames",
        type=int,
        default=5,
        help="Number of tail frames to keep from each latest successful chunk.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Stride within the tail frame buffer.")
    parser.add_argument("--max-frames-per-stage", type=int, default=None)
    parser.add_argument(
        "--timesteps",
        nargs="*",
        default=None,
        help="Explicit timestep list, space or comma separated. Overrides --last-frames.",
    )
    parser.add_argument(
        "--atom-type-map",
        nargs="*",
        default=None,
        help="LAMMPS atom type map, e.g. --atom-type-map 1=O 2=U.",
    )
    parser.add_argument(
        "--species-order",
        nargs="*",
        default=None,
        help="Override POSCAR species order when writing snapshots.",
    )
    parser.add_argument(
        "--elasticity-correction",
        action="store_true",
        help="Select compact tail frames from NVT stress/strain runs for DFT stress relabeling.",
    )
    parser.add_argument(
        "--elastic-frames-per-run",
        type=int,
        default=3,
        help="Elasticity mode: target selected frames per strained/reference run. Default: 3.",
    )
    parser.add_argument(
        "--elastic-tail-fraction",
        type=float,
        default=0.5,
        help="Elasticity mode: only select from this final fraction of available frames. Default: 0.5.",
    )
    parser.add_argument(
        "--elastic-min-separation-ps",
        type=float,
        default=2.0,
        help="Elasticity mode: preferred minimum time separation between selected frames. Default: 2 ps.",
    )
    parser.add_argument(
        "--timestep-ps",
        type=float,
        help="Override MD timestep in ps for frame decorrelation. Defaults to config timestep/timestep_ps.",
    )
    parser.add_argument(
        "--elastic-max-strain",
        type=float,
        default=0.01,
        help="Elasticity mode: skip strained runs above this absolute strain unless --include-large-elastic-strain. Default: 0.01.",
    )
    parser.add_argument(
        "--include-large-elastic-strain",
        action="store_true",
        help="Elasticity mode: include large strains such as +/-2 percent.",
    )
    parser.add_argument(
        "--elastic-max-temp-drift-fraction",
        type=float,
        default=0.20,
        help="Elasticity mode: skip runs whose tail temperature drifts by more than this fraction. Default: 0.20.",
    )
    parser.add_argument(
        "--elastic-stress-spike-factor",
        type=float,
        default=4.0,
        help="Elasticity mode: skip runs with obvious tail pressure/stress spikes. Default max/median factor: 4.",
    )
    parser.add_argument(
        "--allow-elastic-quality-warnings",
        action="store_true",
        help="Elasticity mode: keep runs even when drift/spike checks warn.",
    )
    parser.add_argument(
        "--reduce-large-md-to-2x2x2",
        action="store_true",
        help="Split each selected large MD frame into representative 2x2x2 DFT subcells.",
    )
    parser.add_argument(
        "--reference-poscar-2x2x2",
        type=Path,
        help="Reference 2x2x2 POSCAR used to validate/reorder reduced subcells.",
    )
    parser.add_argument(
        "--large-to-small-replicate",
        nargs=3,
        type=int,
        default=[2, 2, 2],
        metavar=("NX", "NY", "NZ"),
        help="Large MD cell replicate relative to the target DFT cell. Default: 2 2 2.",
    )
    parser.add_argument(
        "--keep-all-subcells",
        action="store_true",
        help="Keep all valid reduced subcells instead of a compact representative subset.",
    )
    parser.add_argument(
        "--max-subcells-per-frame",
        type=int,
        default=1,
        help="Reduced mode: representative subcells kept per large MD frame. Default: 1.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    records = prepare_snapshot_runs(args)
    if not records:
        raise RuntimeError("No snapshot records were written.")

    output_root = args.output_root.resolve()
    runlist = args.runlist.resolve() if args.runlist else output_root / "runlist.txt"
    index = args.index.resolve() if args.index else output_root / "candidate_index.csv"
    write_runlist(records, runlist)
    write_index(records, index)

    by_stage: dict[str, int] = {}
    for record in records:
        by_stage[record.stage_name] = by_stage.get(record.stage_name, 0) + 1

    print("")
    print(f"Snapshot runs : {len(records)}")
    print("Stage counts:")
    for stage_name, count in sorted(by_stage.items()):
        print(f"  {stage_name:32s} {count:4d}")
    print(f"Wrote runlist : {runlist}")
    print(f"Wrote index   : {index}")


if __name__ == "__main__":
    main()
