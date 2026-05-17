from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any


MOTIF_FIELDS = ["motif_id", "run_dir", "structure", "outcar", "incar"]
REFERENCE_FIELDS = [
    "reference_id",
    "formula",
    "path",
    "run_dir",
    "structure",
    "outcar",
    "incar",
    "energy_eV",
    "n_formula_units",
    "role",
    "endmember_kind",
    "is_true_endmember",
    "is_pseudo_endmember",
    "phase_model",
    "reference_basis",
    "thermo_role",
    "source",
    "notes",
]


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


def reference_id_for_directory(
    directory: Path,
    root: Path,
    reference_id: str | None,
    prefix: str | None,
) -> str:
    generated = motif_id_for_directory(directory, root, prefix)
    if not reference_id:
        return generated
    if directory.resolve() == root.resolve():
        return safe_id(reference_id)
    return safe_id(f"{reference_id}_{generated}")


def parse_poscar_counts(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    try:
        from atomi.vasp.magmom import read_poscar_species

        species = read_poscar_species(path)
    except Exception:
        return {}
    return {symbol: count for symbol, count in zip(species.symbols, species.counts)}


def gcd_counts(counts: dict[str, int]) -> int:
    import math

    divisor = 0
    for count in counts.values():
        divisor = count if divisor == 0 else math.gcd(divisor, count)
    return max(divisor, 1)


def reduced_formula(path: Path | None) -> str:
    counts = parse_poscar_counts(path)
    if not counts:
        return ""
    divisor = gcd_counts(counts)
    parts = []
    for element, count in counts.items():
        reduced = count // divisor
        parts.append(element if reduced == 1 else f"{element}{reduced}")
    return "".join(parts)


def normalize_endmember_kind(raw: str | None) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(raw or "").strip().lower()).strip("_")
    aliases = {
        "true": "true_endmember",
        "real": "true_endmember",
        "solution": "true_endmember",
        "solution_endmember": "true_endmember",
        "true_endmember": "true_endmember",
        "pseudo": "pseudo_endmember",
        "hypothetical": "pseudo_endmember",
        "same_lattice": "pseudo_endmember",
        "same_lattice_anchor": "pseudo_endmember",
        "pseudo_endmember": "pseudo_endmember",
        "stable": "stable_phase_reference",
        "stable_phase": "stable_phase_reference",
        "stable_phase_reference": "stable_phase_reference",
        "reservoir": "stable_phase_reference",
        "chemical_potential": "chemical_potential",
        "mu": "chemical_potential",
        "reference": "reference_only",
        "reference_only": "reference_only",
        "none": "",
    }
    return aliases.get(text, text)


def boolean_text(value: bool) -> str:
    return "true" if value else "false"


def reference_mode_labels(args: argparse.Namespace) -> tuple[str, str, str, str, str]:
    if args.true_endmember and args.pseudo_endmember:
        raise ValueError("Use only one of --true-endmember and --pseudo-endmember.")
    kind = normalize_endmember_kind(args.endmember_kind)
    if args.true_endmember:
        kind = "true_endmember"
    if args.pseudo_endmember:
        kind = "pseudo_endmember"
    reference_basis = args.reference_basis or ""
    thermo_role = args.thermo_role or ""
    if kind == "true_endmember":
        reference_basis = reference_basis or "same_lattice_anchor"
        thermo_role = thermo_role or "endmember"
    elif kind == "pseudo_endmember":
        reference_basis = reference_basis or "same_lattice_anchor"
        thermo_role = thermo_role or "pseudo_endmember"
    elif kind == "stable_phase_reference":
        reference_basis = reference_basis or "stable_phase"
        thermo_role = thermo_role or "reservoir"
    elif kind == "chemical_potential":
        reference_basis = reference_basis or "chemical_potential"
        thermo_role = thermo_role or "chemical_potential"
    return (
        kind,
        boolean_text(kind == "true_endmember"),
        boolean_text(kind == "pseudo_endmember"),
        reference_basis,
        thermo_role,
    )


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
        base_row = {
            "motif_id": motif_id_for_directory(directory, root, args.prefix),
            "run_dir": format_path(directory, index_dir, args.absolute),
            "structure": format_path(structure, index_dir, args.absolute),
            "outcar": format_path(outcar, index_dir, args.absolute),
            "incar": format_path(incar, index_dir, args.absolute),
        }
        if args.mode == "reference":
            reference_path = outcar or structure or directory
            (
                endmember_kind,
                is_true_endmember,
                is_pseudo_endmember,
                reference_basis,
                thermo_role,
            ) = reference_mode_labels(args)
            rows.append(
                {
                    "reference_id": reference_id_for_directory(directory, root, args.reference_id, args.prefix),
                    "formula": args.formula or reduced_formula(structure),
                    "path": format_path(reference_path, index_dir, args.absolute),
                    "run_dir": base_row["run_dir"],
                    "structure": base_row["structure"],
                    "outcar": base_row["outcar"],
                    "incar": base_row["incar"],
                    "energy_eV": args.energy_eV or "",
                    "n_formula_units": args.n_formula_units or "",
                    "role": args.role or "",
                    "endmember_kind": endmember_kind,
                    "is_true_endmember": is_true_endmember,
                    "is_pseudo_endmember": is_pseudo_endmember,
                    "phase_model": args.phase_model or "",
                    "reference_basis": reference_basis,
                    "thermo_role": thermo_role,
                    "source": args.source or "",
                    "notes": args.notes or "",
                }
            )
        else:
            rows.append(base_row)
    return rows, skipped


def base_fields_for_mode(mode: str) -> list[str]:
    return list(REFERENCE_FIELDS if mode == "reference" else MOTIF_FIELDS)


def read_existing_rows(path: Path, mode: str) -> tuple[list[dict[str, str]], list[str]]:
    base_fields = base_fields_for_mode(mode)
    if not path.exists():
        return [], base_fields
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or base_fields)
    for field in base_fields:
        if field not in fields:
            fields.append(field)
    return rows, fields


def row_key(row: dict[str, Any]) -> str:
    reference_id = str(row.get("reference_id") or "").strip()
    if reference_id:
        return f"reference_id:{reference_id}"
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
        description="Build or append path-index CSVs for motif metadata or reference phase tables.",
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
        help="Replace rows with matching motif_id/reference_id instead of keeping existing rows.",
    )
    parser.add_argument("--prefix", help="Optional prefix added to generated motif_id values.")
    parser.add_argument(
        "--mode",
        choices=("motif", "reference"),
        default="motif",
        help=(
            "Default motif mode preserves the original zentropy_motif_db index. "
            "Reference mode writes a defect-chem reference index."
        ),
    )
    parser.add_argument(
        "--reference-id",
        help="Reference id for a single reference folder, or prefix for multiple folders.",
    )
    parser.add_argument(
        "--formula",
        help="Formula for reference mode. If omitted, midx tries to infer a reduced formula.",
    )
    parser.add_argument("--energy-eV", help="Optional total reference energy to write directly into reference mode rows.")
    parser.add_argument("--n-formula-units", help="Optional number of formula units for reference mode rows.")
    parser.add_argument("--role", help="Optional reference role, e.g. parent, element, dopant_oxide.")
    parser.add_argument(
        "--endmember-kind",
        choices=(
            "true_endmember",
            "pseudo_endmember",
            "stable_phase_reference",
            "chemical_potential",
            "reference_only",
        ),
        help="Explicit endmember/reference classification for reference mode.",
    )
    parser.add_argument(
        "--true-endmember",
        action="store_true",
        help="Shortcut for --endmember-kind true_endmember.",
    )
    parser.add_argument(
        "--pseudo-endmember",
        action="store_true",
        help="Shortcut for --endmember-kind pseudo_endmember.",
    )
    parser.add_argument(
        "--phase-model",
        help="Optional phase/model label, e.g. fluorite, sesquioxide, metal.",
    )
    parser.add_argument(
        "--reference-basis",
        help="Optional basis label, e.g. stable_phase, same_lattice_anchor, chemical_potential.",
    )
    parser.add_argument(
        "--thermo-role",
        help="Optional thermodynamic role, e.g. parent, reservoir, pseudo_endmember.",
    )
    parser.add_argument("--source", help="Optional reference source label, e.g. dft.")
    parser.add_argument("--notes", help="Optional notes for reference mode rows.")
    parser.add_argument("--dry-run", action="store_true", help="Print the scan summary without writing the CSV.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    index = args.index.resolve()
    index_dir = index.parent
    existing, fields = read_existing_rows(index, args.mode)
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
    label = "reference" if args.mode == "reference" else "VASP"
    print(f"Usable {label} rows : {len(scanned)}")
    print(f"New rows appended   : {counts['new']}")
    print(f"Existing rows kept  : {counts['skipped_existing']}")
    print(f"Rows replaced       : {counts['replaced']}")
    print(f"Missing OUTCAR skip : {len(skipped)}")
    print(f"Index CSV           : {index}")
    if skipped:
        print("Skipped examples:")
        for item in skipped[:5]:
            print(f"  {item}")
    if args.mode == "reference":
        print("Next reference-energy step:")
        print(f"  defect-chem build-references --reference-index {index}")
    else:
        print("Next auto-metadata step:")
        print(f"  zentropy_motif_db auto-metadata --input-csv {index}")


if __name__ == "__main__":
    main()
