from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any


BASE_FIELDS = ["motif_id", "run_dir", "structure", "outcar", "incar"]


def safe_id(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_.+-]+", "_", value.strip()).strip("_")
    return text or "motif"


def first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def format_path(path: Path | None, index_dir: Path, absolute: bool) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    if absolute:
        return str(resolved)
    return os.path.relpath(resolved, index_dir.resolve())


def motif_id_for_directory(directory: Path, root: Path, prefix: str | None) -> str:
    try:
        relative = directory.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(directory.name)
    raw = directory.name if str(relative) == "." else "__".join(relative.parts)
    if prefix:
        raw = f"{prefix}_{raw}"
    return safe_id(raw)


def candidate_directories(root: Path, pattern: str) -> list[Path]:
    root = root.resolve()
    directories = [root] if root.is_dir() else []
    if root.is_dir():
        directories.extend(path.resolve() for path in sorted(root.glob(pattern)) if path.is_dir())
    seen = set()
    unique = []
    for path in directories:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def scan_root(args: argparse.Namespace, root: Path, index_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    skipped: list[str] = []
    structure_names = ("POSCAR", "CONTCAR") if args.prefer_poscar else ("CONTCAR", "POSCAR")
    for directory in candidate_directories(root, args.glob):
        structure = first_existing(directory, structure_names)
        outcar = first_existing(directory, ("OUTCAR", "OUTCAR.gz"))
        incar = first_existing(directory, ("INCAR",))
        if structure is None:
            continue
        if outcar is None and not args.allow_missing_outcar:
            skipped.append(f"{directory}: has structure but no OUTCAR")
            continue
        rows.append(
            {
                "motif_id": motif_id_for_directory(directory, root, args.prefix),
                "run_dir": format_path(directory, index_dir, args.absolute),
                "structure": format_path(structure, index_dir, args.absolute),
                "outcar": format_path(outcar, index_dir, args.absolute),
                "incar": format_path(incar, index_dir, args.absolute),
            }
        )
    return rows, skipped


def read_existing_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], list(BASE_FIELDS)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or BASE_FIELDS)
    for field in BASE_FIELDS:
        if field not in fields:
            fields.append(field)
    return rows, fields


def row_key(row: dict[str, Any]) -> str:
    motif_id = str(row.get("motif_id") or "").strip()
    if motif_id:
        return f"motif_id:{motif_id}"
    run_dir = str(row.get("run_dir") or row.get("run") or "").strip()
    return f"run_dir:{run_dir}"


def merge_rows(
    existing: list[dict[str, str]],
    new_rows: list[dict[str, str]],
    replace_existing: bool,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    merged = list(existing)
    positions = {row_key(row): index for index, row in enumerate(merged) if row_key(row)}
    counts = {"new": 0, "replaced": 0, "skipped_existing": 0}
    for row in new_rows:
        key = row_key(row)
        if key in positions:
            if replace_existing:
                original = merged[positions[key]]
                merged[positions[key]] = {**original, **row}
                counts["replaced"] += 1
            else:
                counts["skipped_existing"] += 1
            continue
        positions[key] = len(merged)
        merged.append(row)
        counts["new"] += 1
    return merged, counts


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    extra = sorted({field for row in rows for field in row if field not in fields})
    fieldnames = fields + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="midx",
        description="Build or append a path-index CSV for zentropy_motif_db auto-metadata.",
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path(".")],
        help="Folder(s) to scan. Default is the current directory.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("motif_paths.csv"),
        help="CSV path index to create or append. Default: motif_paths.csv.",
    )
    parser.add_argument(
        "--glob",
        default="**/*",
        help="Directory glob under each root. Default recursively scans all subfolders.",
    )
    parser.add_argument("--absolute", action="store_true", help="Write absolute paths instead of relative paths.")
    parser.add_argument(
        "--prefer-poscar",
        action="store_true",
        help="Prefer POSCAR over CONTCAR when both are present.",
    )
    parser.add_argument(
        "--allow-missing-outcar",
        action="store_true",
        help="Include folders with POSCAR/CONTCAR but no OUTCAR.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace rows with matching motif_id instead of keeping existing rows.",
    )
    parser.add_argument("--prefix", help="Optional prefix added to generated motif_id values.")
    parser.add_argument("--dry-run", action="store_true", help="Print the scan summary without writing the CSV.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    index = args.index.resolve()
    index_dir = index.parent
    existing, fields = read_existing_rows(index)
    scanned: list[dict[str, str]] = []
    skipped: list[str] = []
    for root in args.roots:
        rows, missing = scan_root(args, root, index_dir)
        scanned.extend(rows)
        skipped.extend(missing)

    merged, counts = merge_rows(existing, scanned, args.replace_existing)
    if not args.dry_run:
        write_rows(index, merged, fields)

    print(f"Scanned folders     : {len(scanned) + len(skipped)}")
    print(f"Usable VASP rows    : {len(scanned)}")
    print(f"New rows appended   : {counts['new']}")
    print(f"Existing rows kept  : {counts['skipped_existing']}")
    print(f"Rows replaced       : {counts['replaced']}")
    print(f"Missing OUTCAR skip : {len(skipped)}")
    print(f"Index CSV           : {index}")
    if skipped:
        print("Skipped examples:")
        for item in skipped[:5]:
            print(f"  {item}")
    print("Next auto-metadata step:")
    print(f"  zentropy_motif_db auto-metadata --input-csv {index}")


if __name__ == "__main__":
    main()
