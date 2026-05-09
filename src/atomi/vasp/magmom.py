from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PoscarSpecies:
    symbols: list[str]
    counts: list[int]

    @property
    def total_atoms(self) -> int:
        return sum(self.counts)


@dataclass
class MagmomUpdate:
    species: PoscarSpecies
    selected_elements: list[str]
    moments: list[float]
    magmom_line: str
    backup_path: Path | None


def read_poscar_species(poscar: Path) -> PoscarSpecies:
    lines = poscar.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(f"POSCAR is too short: {poscar}")

    line5 = lines[5].split()
    line6 = lines[6].split()

    if _all_ints(line5):
        raise ValueError(
            "POSCAR appears to omit the VASP5 element-symbol line. "
            "Please use a POSCAR with explicit species names."
        )
    if not _all_ints(line6):
        raise ValueError(f"Could not parse POSCAR species counts from line 7 in {poscar}")

    return PoscarSpecies(symbols=line5, counts=[int(value) for value in line6])


def _all_ints(values: list[str]) -> bool:
    if not values:
        return False
    try:
        [int(value) for value in values]
    except ValueError:
        return False
    return True


def final_outcar_magnetization(outcar: Path, expected_atoms: int) -> list[float]:
    lines = outcar.read_text(encoding="utf-8", errors="replace").splitlines()
    block_start = None
    for index, line in enumerate(lines):
        if "magnetization" in line.lower() and "(x)" in line.lower():
            block_start = index
    if block_start is None:
        raise ValueError(f"No final 'magnetization (x)' table found in {outcar}")

    moments: list[float] = []
    for line in lines[block_start + 1 :]:
        stripped = line.strip()
        if not stripped:
            if moments:
                break
            continue
        parts = stripped.split()
        if not parts:
            continue
        if parts[0].lower().startswith("tot"):
            break
        try:
            int(parts[0])
        except ValueError:
            continue
        if len(parts) < 2:
            continue
        try:
            moments.append(float(parts[-1]))
        except ValueError:
            continue
        if len(moments) == expected_atoms:
            break

    if len(moments) != expected_atoms:
        raise ValueError(
            f"Parsed {len(moments)} magnetic moments from OUTCAR, "
            f"but POSCAR expects {expected_atoms} atoms."
        )
    return moments


def expand_magmom_tokens(tokens: list[str]) -> list[float]:
    values: list[float] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "*" in token:
            left, right = token.split("*", 1)
            values.extend([float(right)] * int(left))
        else:
            values.append(float(token))
    return values


def existing_magmom_values(incar: Path, total_atoms: int) -> list[float] | None:
    line_index, line = find_magmom_line(incar.read_text(encoding="utf-8", errors="replace").splitlines())
    if line_index is None or line is None:
        return None
    body = strip_incar_comment(line).split("=", 1)[-1]
    try:
        values = expand_magmom_tokens(body.split())
    except ValueError:
        return None
    if len(values) < total_atoms:
        values.extend([0.0] * (total_atoms - len(values)))
    return values[:total_atoms]


def strip_incar_comment(line: str) -> str:
    cut = len(line)
    for marker in ("!", "#"):
        index = line.find(marker)
        if index >= 0:
            cut = min(cut, index)
    return line[:cut]


def find_magmom_line(lines: list[str]) -> tuple[int | None, str | None]:
    pattern = re.compile(r"^\s*MAGMOM\s*=", re.IGNORECASE)
    for index, line in enumerate(lines):
        if pattern.match(line):
            return index, line
    return None, None


def selected_atom_indices(species: PoscarSpecies, selected_elements: set[str]) -> set[int]:
    selected = set()
    atom_start = 0
    for symbol, count in zip(species.symbols, species.counts):
        atom_end = atom_start + count
        if symbol in selected_elements:
            selected.update(range(atom_start, atom_end))
        atom_start = atom_end
    return selected


def updated_moments(
    species: PoscarSpecies,
    outcar_moments: list[float],
    selected_elements: list[str],
    previous_moments: list[float] | None,
    preserve_unselected: bool,
) -> list[float]:
    selected = selected_atom_indices(species, set(selected_elements))
    result = []
    for index, moment in enumerate(outcar_moments):
        if index in selected:
            result.append(moment)
        elif preserve_unselected and previous_moments is not None:
            result.append(previous_moments[index])
        else:
            result.append(0.0)
    return result


def format_magmom_line(
    species: PoscarSpecies,
    moments: list[float],
    selected_elements: list[str],
    decimals: int,
    compact_zero: bool,
) -> str:
    pieces = []
    atom_start = 0
    selected = set(selected_elements)
    for symbol, count in zip(species.symbols, species.counts):
        atom_end = atom_start + count
        block = moments[atom_start:atom_end]
        if symbol not in selected and compact_zero and all(abs(value) < 1.0e-12 for value in block):
            pieces.append(f"{count}*0")
        else:
            pieces.extend(_format_float(value, decimals) for value in block)
        atom_start = atom_end
    return "MAGMOM = " + " ".join(pieces)


def _format_float(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text


def replace_or_append_magmom(incar: Path, magmom_line: str) -> str:
    lines = incar.read_text(encoding="utf-8", errors="replace").splitlines()
    line_index, _line = find_magmom_line(lines)
    if line_index is None:
        lines.append(magmom_line)
    else:
        lines[line_index] = magmom_line
    return "\n".join(lines) + "\n"


def update_incar_magmom(
    outcar: Path,
    poscar: Path,
    incar: Path,
    elements: list[str],
    decimals: int = 3,
    compact_zero: bool = True,
    preserve_unselected: bool = False,
    dry_run: bool = False,
    backup: bool = True,
) -> MagmomUpdate:
    species = read_poscar_species(poscar)
    missing = [element for element in elements if element not in species.symbols]
    if missing:
        raise ValueError(
            f"Elements not present in POSCAR species order: {', '.join(missing)}. "
            f"POSCAR has: {', '.join(species.symbols)}"
        )

    outcar_moments = final_outcar_magnetization(outcar, species.total_atoms)
    previous = existing_magmom_values(incar, species.total_atoms) if preserve_unselected else None
    moments = updated_moments(
        species,
        outcar_moments,
        elements,
        previous_moments=previous,
        preserve_unselected=preserve_unselected,
    )
    magmom_line = format_magmom_line(
        species,
        moments,
        elements,
        decimals=decimals,
        compact_zero=compact_zero,
    )

    backup_path = None
    if not dry_run:
        if backup:
            backup_path = next_backup_path(incar)
            shutil.copy2(incar, backup_path)
        incar.write_text(replace_or_append_magmom(incar, magmom_line), encoding="utf-8")

    return MagmomUpdate(
        species=species,
        selected_elements=elements,
        moments=moments,
        magmom_line=magmom_line,
        backup_path=backup_path,
    )


def next_backup_path(path: Path) -> Path:
    candidate = path.with_name(path.name + ".bak")
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}.bak{index}")
        if not candidate.exists():
            return candidate
        index += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magit",
        description="Update INCAR MAGMOM from final OUTCAR moments for selected POSCAR elements.",
    )
    parser.add_argument("elements", nargs="+", help="Elements to update, e.g. Gd U.")
    parser.add_argument("--outcar", type=Path, default=Path("OUTCAR"))
    parser.add_argument("--poscar", type=Path, default=Path("POSCAR"))
    parser.add_argument("--incar", type=Path, default=Path("INCAR"))
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument(
        "--preserve-unselected",
        action="store_true",
        help="Keep existing MAGMOM values for elements not listed. Default: set them to 0.",
    )
    parser.add_argument(
        "--no-compact-zero",
        action="store_true",
        help="Write explicit zeros instead of compact entries such as 96*0.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the MAGMOM line only.")
    parser.add_argument("--no-backup", action="store_true", help="Do not write INCAR.bak.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    for path in (args.outcar, args.poscar, args.incar):
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {path}")

    result = update_incar_magmom(
        outcar=args.outcar,
        poscar=args.poscar,
        incar=args.incar,
        elements=args.elements,
        decimals=args.decimals,
        compact_zero=not args.no_compact_zero,
        preserve_unselected=args.preserve_unselected,
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )

    print(f"POSCAR species : {' '.join(result.species.symbols)}")
    print(f"POSCAR counts  : {' '.join(str(count) for count in result.species.counts)}")
    print(f"Updated elems  : {' '.join(result.selected_elements)}")
    print(result.magmom_line)
    if args.dry_run:
        print("Dry run: INCAR not modified.")
    else:
        if result.backup_path is not None:
            print(f"Backup        : {result.backup_path}")
        print(f"Updated       : {args.incar}")


if __name__ == "__main__":
    main()
