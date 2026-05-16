#!/usr/bin/env python3
"""Prepare and analyze VASP static elastic tensor calculations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from atomi.lammps.elastic import reduce_tensor_by_symmetry, tensor_components, voigt_reuss_hill


VASP_TO_CANONICAL = {"XX": 0, "YY": 1, "ZZ": 2, "YZ": 3, "ZX": 4, "XZ": 4, "XY": 5}
REQUIRED_ELASTIC_TAGS = {
    "IBRION": "6",
    "ISIF": "3",
    "NSW": "1",
}
DEFAULT_ELASTIC_TAGS = {
    "NFREE": "2",
    "POTIM": "0.015",
    "LREAL": ".FALSE.",
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def sanitize_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", path.name.strip()) or "volume"


def discover_volume_folders(args: argparse.Namespace) -> list[Path]:
    folders = [path.resolve() for path in args.volume_folder]
    if args.root is not None:
        folders.extend(
            sorted(path.resolve() for path in args.root.glob(args.volume_pattern) if path.is_dir())
        )
    unique: list[Path] = []
    seen = set()
    for path in folders:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise ValueError("No volume folders provided. Use --volume-folder or --root.")
    return unique


def find_structure(folder: Path) -> Path:
    for name in ("CONTCAR", "POSCAR", "vasprun.xml", "OUTCAR"):
        path = folder / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No CONTCAR/POSCAR/vasprun.xml/OUTCAR found in {folder}")


def parse_volume_scale(label: str) -> float | None:
    match = re.search(r"V([0-9]+(?:\.[0-9]+)?)", label)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def strip_incar_comment(line: str) -> str:
    cut = len(line)
    for marker in ("!", "#"):
        idx = line.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    return line[:cut]


def update_incar_tags(text: str, updates: dict[str, str]) -> str:
    lines = text.splitlines()
    found: set[str] = set()
    out: list[str] = []
    for line in lines:
        body = strip_incar_comment(line)
        if "=" not in body:
            out.append(line)
            continue
        key = body.split("=", 1)[0].strip().upper()
        if key in updates:
            out.append(f"{key} = {updates[key]}")
            found.add(key)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in found:
            out.append(f"{key} = {value}")
    return "\n".join(out).rstrip() + "\n"


def template_incar(template: Path | None, extra_tags: dict[str, str]) -> str:
    if template is not None and (template / "INCAR").exists():
        text = read_text(template / "INCAR")
    else:
        text = "\n".join(
            [
                "SYSTEM = VASP static elastic tensor",
                "ENCUT = 520",
                "EDIFF = 1E-7",
                "PREC = Accurate",
                "ADDGRID = .TRUE.",
            ]
        )
    updates = dict(DEFAULT_ELASTIC_TAGS)
    updates.update(REQUIRED_ELASTIC_TAGS)
    updates.update(extra_tags)
    return update_incar_tags(text, updates)


def copy_optional_template_files(template: Path | None, run_dir: Path) -> None:
    if template is None:
        return
    for name in ("KPOINTS", "POTCAR"):
        src = template / name
        if src.exists():
            shutil.copy2(src, run_dir / name)


def parse_key_values(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected INCAR tag KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        out[key.strip().upper()] = value.strip()
    return out


def prepare_main(args: argparse.Namespace) -> dict[str, Any]:
    folders = discover_volume_folders(args)
    template = args.template.resolve() if args.template else None
    args.outdir.mkdir(parents=True, exist_ok=True)
    incar_text = template_incar(template, parse_key_values(args.incar_tag))
    rows: list[dict[str, Any]] = []
    runlist_lines: list[str] = []
    for index, folder in enumerate(folders, start=1):
        source = find_structure(folder)
        label = sanitize_name(folder)
        run_dir = args.outdir / f"{index:03d}_{label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        atoms = read(source, index=-1)
        from ase.io import write

        write(run_dir / "POSCAR", atoms, format="vasp", direct=True, sort=False)
        (run_dir / "INCAR").write_text(incar_text, encoding="utf-8")
        copy_optional_template_files(template, run_dir)
        volume = float(atoms.get_volume())
        row = {
            "index": index,
            "label": label,
            "volume_folder": str(folder),
            "source_structure": str(source),
            "run_dir": str(run_dir.resolve()),
            "volume_A3": volume,
            "volume_scale": parse_volume_scale(label),
            "incar_elastic_tags": "IBRION=6;ISIF=3;NSW=1",
        }
        rows.append(row)
        runlist_lines.append(str(run_dir.resolve()))
    write_csv(args.outdir / "vasp_elastic_manifest.csv", rows)
    (args.outdir / "runlist.txt").write_text("\n".join(runlist_lines) + "\n", encoding="utf-8")
    metadata = {
        "method": "VASP finite-difference elastic tensor",
        "recommended_tags": {
            "IBRION": "6",
            "ISIF": "3",
            "NSW": "1",
            "NFREE": DEFAULT_ELASTIC_TAGS["NFREE"],
            "POTIM": DEFAULT_ELASTIC_TAGS["POTIM"],
        },
        "note": (
            "Prepared from parent structures; run VASP in each run_dir and then "
            "use vasp_elastic analyze."
        ),
        "template": str(template) if template else "",
        "n_runs": len(rows),
    }
    write_json(args.outdir / "vasp_elastic_prepare_metadata.json", metadata)
    print(f"Wrote VASP elastic folders: {args.outdir.resolve()}")
    print(f"Wrote runlist: {args.outdir / 'runlist.txt'}")
    return metadata


def numeric_row(parts: list[str]) -> bool:
    if len(parts) < 7:
        return False
    return parts[0].upper() in VASP_TO_CANONICAL


def parse_elastic_tensor_kbar(outcar: Path) -> tuple[np.ndarray, dict[str, Any]]:
    lines = read_text(outcar).splitlines()
    blocks: list[tuple[str, np.ndarray, int]] = []
    for idx, line in enumerate(lines):
        upper = line.upper()
        if "ELASTIC MODULI" not in upper or "KBAR" not in upper:
            continue
        title = line.strip()
        columns: list[str] | None = None
        tensor = np.full((6, 6), np.nan, dtype=float)
        for offset in range(idx + 1, min(idx + 18, len(lines))):
            parts = lines[offset].split()
            upper_parts = [part.upper() for part in parts]
            labels = [part for part in upper_parts if part in VASP_TO_CANONICAL]
            if len(labels) >= 6 and not numeric_row(upper_parts):
                columns = labels[:6]
                continue
            if columns and numeric_row(upper_parts):
                row_label = upper_parts[0]
                values = []
                for raw in parts[1:7]:
                    try:
                        values.append(float(raw))
                    except ValueError:
                        values.append(math.nan)
                row_idx = VASP_TO_CANONICAL[row_label]
                for col_label, value in zip(columns, values):
                    tensor[row_idx, VASP_TO_CANONICAL[col_label]] = value * 0.1
        if np.isfinite(tensor).sum() >= 36:
            blocks.append((title, tensor, idx + 1))
    if not blocks:
        raise ValueError(f"No VASP elastic tensor block in kBar found in {outcar}")
    preferred = next((block for block in reversed(blocks) if "SYMMETR" in block[0].upper()), blocks[-1])
    title, tensor, line_no = preferred
    return tensor, {
        "source_file": str(outcar),
        "block_title": title,
        "line_number": line_no,
        "input_unit": "kBar",
        "output_unit": "GPa",
    }


def infer_symmetry_from_structure(path: Path, tolerance: float) -> str:
    try:
        atoms = read(path, index=-1)
    except Exception:
        return "full"
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    angles = np.asarray(atoms.cell.angles(), dtype=float)
    right = bool(np.all(np.abs(angles - 90.0) <= tolerance))
    same_ab = abs(lengths[0] - lengths[1]) <= tolerance
    same_bc = abs(lengths[1] - lengths[2]) <= tolerance
    if right and same_ab and same_bc:
        return "cubic"
    if right and same_ab:
        return "tetragonal"
    if right:
        return "orthorhombic"
    return "full"


def volume_from_run(run_dir: Path) -> tuple[float, str]:
    for name in ("CONTCAR", "POSCAR"):
        path = run_dir / name
        if path.exists():
            return float(read(path, index=-1).get_volume()), str(path)
    return math.nan, ""


def analyze_run(run_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    outcar = run_dir / args.outcar_name
    if not outcar.exists():
        raise FileNotFoundError(f"Missing {args.outcar_name}: {run_dir}")
    raw_c, tensor_meta = parse_elastic_tensor_kbar(outcar)
    structure_path = run_dir / "CONTCAR" if (run_dir / "CONTCAR").exists() else run_dir / "POSCAR"
    symmetry = args.symmetry
    if symmetry == "auto":
        symmetry = infer_symmetry_from_structure(structure_path, args.symmetry_tolerance)
    reduced = reduce_tensor_by_symmetry(raw_c, symmetry)
    moduli = voigt_reuss_hill(reduced)
    volume, volume_source = volume_from_run(run_dir)
    label = run_dir.name
    row: dict[str, Any] = {
        "temperature_K": args.temperature_K,
        "label": label,
        "volume_A3": volume,
        "volume_scale": parse_volume_scale(label),
        "symmetry": symmetry,
        "source": "VASP static elastic",
        "source_file": str(outcar),
        "volume_source": volume_source,
    }
    row.update(tensor_components(reduced))
    row.update(moduli)
    tensor_record = {
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "temperature_K": args.temperature_K,
        "symmetry": symmetry,
        "raw_tensor_GPa": raw_c.tolist(),
        "symmetry_reduced_tensor_GPa": reduced.tolist(),
        "parser": tensor_meta,
        "moduli": moduli,
    }
    return row, tensor_record


def discover_analyze_runs(args: argparse.Namespace) -> list[Path]:
    runs = [path.resolve() for path in args.run]
    if args.root is not None:
        runs.extend(sorted(path.resolve() for path in args.root.glob(args.run_glob) if path.is_dir()))
    unique: list[Path] = []
    seen = set()
    for path in runs:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise ValueError("No VASP elastic run folders provided. Use --run or --root.")
    return unique


def analyze_main(args: argparse.Namespace) -> dict[str, Any]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for run in discover_analyze_runs(args):
        try:
            row, tensor = analyze_run(run, args)
        except Exception as exc:
            failures.append({"run_dir": str(run), "error": str(exc)})
            if not args.allow_failures:
                raise
            continue
        rows.append(row)
        tensors.append(tensor)
    write_csv(args.outdir / "elastic_moduli_T.csv", rows)
    write_json(
        args.outdir / "elastic_tensors.json",
        {
            "schema": "atomi.vasp.static_elastic.v1",
            "unit": "GPa",
            "tensors": tensors,
            "failures": failures,
        },
    )
    metadata = {
        "method": "VASP IBRION=6 finite-difference static elastic tensor",
        "n_runs": len(rows),
        "n_failures": len(failures),
        "outputs": {
            "elastic_moduli_T.csv": str(args.outdir / "elastic_moduli_T.csv"),
            "elastic_tensors.json": str(args.outdir / "elastic_tensors.json"),
        },
    }
    write_json(args.outdir / "vasp_elastic_analysis_metadata.json", metadata)
    print(f"Wrote elastic moduli: {args.outdir / 'elastic_moduli_T.csv'}")
    print(f"Wrote elastic tensors: {args.outdir / 'elastic_tensors.json'}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp_elastic",
        description="Prepare and analyze VASP static elastic tensor calculations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Create IBRION=6/ISIF=3 elastic folders.")
    prep.add_argument("--volume-folder", action="append", type=Path, default=[])
    prep.add_argument("--root", type=Path, help="Root containing volume folders.")
    prep.add_argument("--volume-pattern", default="V*", help="Folder glob under --root.")
    prep.add_argument("--template", type=Path, help="Template folder with INCAR/KPOINTS/POTCAR.")
    prep.add_argument("--outdir", type=Path, default=Path("vasp_elastic"))
    prep.add_argument(
        "--incar-tag",
        action="append",
        default=[],
        help="Extra or overriding INCAR tag KEY=VALUE.",
    )

    ana = sub.add_parser("analyze", help="Parse OUTCAR elastic tensors and moduli.")
    ana.add_argument("--run", action="append", type=Path, default=[])
    ana.add_argument("--root", type=Path, help="Root containing completed VASP elastic run folders.")
    ana.add_argument("--run-glob", default="*", help="Run folder glob under --root.")
    ana.add_argument("--outdir", type=Path, default=Path("analysis/elastic_vasp"))
    ana.add_argument("--outcar-name", default="OUTCAR")
    ana.add_argument("--temperature-K", type=float, default=0.0)
    ana.add_argument(
        "--symmetry",
        choices=("auto", "full", "cubic", "tetragonal", "orthorhombic"),
        default="auto",
    )
    ana.add_argument("--symmetry-tolerance", type=float, default=1.0e-2)
    ana.add_argument("--allow-failures", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare_main(args)
    if args.command == "analyze":
        return analyze_main(args)
    parser.error(f"Unknown command: {args.command}")
    return None


if __name__ == "__main__":
    main(sys.argv[1:])
