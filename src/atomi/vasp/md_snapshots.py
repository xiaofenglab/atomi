from __future__ import annotations

import argparse
import csv
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ase.io import read, write

from atomi.vasp.prefail import (
    choose_reference,
    copy_vasp_template,
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
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "run_dir",
                "stage_name",
                "temperature_K",
                "chunk_name",
                "dump_path",
                "timestep",
                "source_config",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "run_dir": str(record.run_dir.resolve()),
                    "stage_name": record.stage_name,
                    "temperature_K": record.temperature_K,
                    "chunk_name": record.chunk_name,
                    "dump_path": str(record.dump_path.resolve()),
                    "timestep": record.timestep,
                    "source_config": str(record.source_config.resolve()),
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
    output_root = ensure_dir(args.output_root.resolve())
    template_dir = args.vasp_template.resolve() if args.vasp_template else None
    reference_poscar = reference_poscar_from_config(cfg, config_path, args.poscar, template_dir)
    reference = read(reference_poscar)
    species_order = tuple(args.species_order) if args.species_order else species_order_from_atoms(reference)
    atom_type_map = infer_atom_type_map(cfg, args.atom_type_map)

    needs_template_poscar = args.poscar is None and "reference_poscar" not in cfg
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

    records: list[SnapshotRecord] = []
    summary = []
    for stage, _stage_dir, chunk_dir, dump_path in selected_stage_info:
        stage_name = stage["name"]
        temperature = stage_temperature_k(stage)
        frames = select_dump_frames(
            dump_path,
            explicit_timesteps,
            args.last_frames,
            args.stride,
        )
        if args.max_frames_per_stage:
            frames = frames[-args.max_frames_per_stage :]
        print(f"[stage] {stage_name}: {len(frames)} frames from {chunk_dir.name}/{dump_path.name}")

        for frame in frames:
            timestep, box_header, box_lines, atoms_header, atom_lines = frame
            cell, origin = parse_box_bounds(box_header, box_lines)
            md_atoms = parse_atoms_block(atom_lines, atoms_header, atom_type_map, origin, cell)
            ref_for_match = choose_reference(reference, md_atoms, None)
            reordered = reorder_md_to_reference(ref_for_match, md_atoms, species_order=species_order)

            run_dir = ensure_dir(output_root / stage_name / chunk_dir.name / f"md_{timestep:010d}")
            write(run_dir / "POSCAR", reordered, format="vasp", direct=True, vasp5=True, sort=False)
            copy_vasp_template(template_dir, run_dir, copy_all=args.copy_template_all)

            info = {
                "stage_name": stage_name,
                "temperature_K": temperature,
                "chunk_name": chunk_dir.name,
                "dump_path": str(dump_path.resolve()),
                "timestep": timestep,
                "source_config": str(config_path.resolve()),
                "reference_poscar": str(reference_poscar.resolve()),
                "composition": summarize_atoms(reordered),
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
    parser.add_argument("--output-root", required=True, type=Path)
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
