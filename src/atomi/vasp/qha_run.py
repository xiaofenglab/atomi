import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

from atomi.vasp.qha_summary import is_finite, selected_volume_dirs, summarize_volume


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_summary_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scan_volume_rows(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> list[dict]:
    dirs = selected_volume_dirs(args, parser)
    summary_args = argparse.Namespace(
        parent_pattern=args.parent_pattern,
        disp_pattern=args.disp_pattern,
        atoms_per_fu=args.atoms_per_fu,
    )
    return [summarize_volume(directory, summary_args) for directory in dirs]


def source_rows(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[dict]:
    if args.summary_csv:
        return load_summary_csv(args.summary_csv.resolve())
    return scan_volume_rows(args, parser)


def sort_key(row: dict, sort_by: str) -> tuple:
    if sort_by == "volume":
        return (as_float(row.get("volume_A3")), row.get("volume_folder", ""))
    if sort_by == "folder":
        return (row.get("volume_folder", ""),)
    scale = as_float(row.get("scale_factor"))
    if not is_finite(scale):
        scale = math.inf
    return (scale, row.get("volume_folder", ""))


def qha_rows(rows: list[dict], sort_by: str) -> list[dict]:
    valid = [
        row
        for row in rows
        if is_finite(row.get("volume_A3")) and is_finite(row.get("energy_eV"))
    ]
    valid.sort(key=lambda row: sort_key(row, sort_by))
    return valid


def resolve_thermal_path(row: dict, thermal_file: str) -> Path:
    root = row.get("root")
    if not root:
        raise ValueError(f"row for {row.get('volume_folder', '<unknown>')} has no root path")
    return (Path(root) / thermal_file).resolve()


def thermal_paths(
    args: argparse.Namespace,
    rows: list[dict],
    parser: argparse.ArgumentParser,
) -> list[Path]:
    if args.thermal_yaml:
        paths = [path.resolve() for path in args.thermal_yaml]
        if len(paths) != len(rows):
            parser.error(
                f"--thermal-yaml count ({len(paths)}) does not match valid E-V points "
                f"({len(rows)}). phonopy-qha needs one thermal_properties.yaml per row."
            )
        missing = [path for path in paths if not path.exists()]
        if missing:
            parser.error(
                "missing explicit thermal YAML: " + ", ".join(str(path) for path in missing)
            )
        return paths

    paths = [resolve_thermal_path(row, args.thermal_file) for row in rows]
    missing = [path for path in paths if not path.exists()]
    if missing:
        parser.error("missing thermal YAML: " + ", ".join(str(path) for path in missing))
    return paths


def rel_or_abs(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def write_ev_dat(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            volume = as_float(row["volume_A3"])
            energy = as_float(row["energy_eV"])
            handle.write(f"{volume:.10f}  {energy:.10f}\n")


def write_manifest(path: Path, rows: list[dict], thermals: list[Path], outdir: Path) -> None:
    fields = ["volume_folder", "scale_factor", "volume_A3", "energy_eV", "thermal_yaml"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, thermal in zip(rows, thermals):
            writer.writerow(
                {
                    "volume_folder": row.get("volume_folder", ""),
                    "scale_factor": row.get("scale_factor", ""),
                    "volume_A3": row.get("volume_A3", ""),
                    "energy_eV": row.get("energy_eV", ""),
                    "thermal_yaml": rel_or_abs(thermal, outdir),
                }
            )


def write_run_script(
    path: Path,
    ev_path: Path,
    thermals: list[Path],
    outdir: Path,
    phonopy_module: str | None,
    phonopy_qha: str,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")
        if phonopy_module:
            handle.write(f"module load {shlex.quote(phonopy_module)}\n\n")
        handle.write(f"{shlex.quote(phonopy_qha)} {shlex.quote(rel_or_abs(ev_path, outdir))} \\\n")
        for index, thermal in enumerate(thermals):
            suffix = " \\" if index < len(thermals) - 1 else ""
            handle.write(f"  {shlex.quote(rel_or_abs(thermal, outdir))}{suffix}\n")
    path.chmod(0o755)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-qha-run",
        description="Prepare e-v.dat and a phonopy-qha run script from QHA volume folders.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="Folder containing QHA volume subfolders.")
    source.add_argument(
        "--volume-folder",
        type=Path,
        action="append",
        help="Explicit QHA volume folder. Repeat once per volume.",
    )
    source.add_argument("--summary-csv", type=Path, help="CSV from vasp-qha-summary.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--volume-pattern", default="V*")
    parser.add_argument("--disp-pattern", default="disp-*")
    parser.add_argument("--parent-pattern", default="parent*")
    parser.add_argument("--atoms-per-fu", type=float, default=3.0)
    parser.add_argument("--thermal-file", default="thermal_properties.yaml")
    parser.add_argument(
        "--thermal-yaml",
        type=Path,
        action="append",
        help="Explicit thermal YAML in E-V order. Repeat once per E-V row.",
    )
    parser.add_argument("--phonopy-module", default=os.environ.get("ATOMI_PHONOPY_MODULE"))
    parser.add_argument("--phonopy-qha", default="phonopy-qha")
    parser.add_argument("--ev-file", default="e-v.dat")
    parser.add_argument("--script-name", default="run_phonopy_qha.sh")
    parser.add_argument("--sort-by", choices=("scale", "volume", "folder"), default="scale")
    parser.add_argument("--execute", action="store_true", help="Run the generated shell script.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.atoms_per_fu <= 0:
        parser.error("--atoms-per-fu must be positive")

    try:
        rows = qha_rows(source_rows(args, parser), args.sort_by)
    except ValueError as exc:
        parser.error(str(exc))
    if not rows:
        parser.error("no valid QHA E-V rows found")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    thermals = thermal_paths(args, rows, parser)
    ev_path = outdir / args.ev_file
    script_path = outdir / args.script_name
    manifest_path = outdir / "qha_inputs.csv"

    write_ev_dat(ev_path, rows)
    write_manifest(manifest_path, rows, thermals, outdir)
    write_run_script(
        script_path,
        ev_path,
        thermals,
        outdir,
        args.phonopy_module,
        args.phonopy_qha,
    )

    print(ev_path)
    print(manifest_path)
    print(script_path)
    if args.execute:
        subprocess.run(["bash", str(script_path)], cwd=outdir, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
