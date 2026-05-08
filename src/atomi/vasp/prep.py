from __future__ import annotations

import re
from pathlib import Path

from ase import Atoms


DEFAULT_TEMPLATE_FILES = ("INCAR", "KPOINTS", "POTCAR")


def unique_symbols(symbols: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    ordered = []
    for symbol in symbols:
        if symbol not in ordered:
            ordered.append(symbol)
    return tuple(ordered)


def species_order_from_atoms(atoms: Atoms) -> tuple[str, ...]:
    return unique_symbols(tuple(atoms.get_chemical_symbols()))


def summarize_atoms(atoms: Atoms) -> str:
    counts = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return ", ".join(f"{symbol}{counts[symbol]}" for symbol in counts)


def template_poscar(template_dir: Path | None) -> Path | None:
    if template_dir is None:
        return None
    candidate = template_dir / "POSCAR"
    return candidate if candidate.exists() else None


def resolve_input_poscar(poscar: Path | None, template_dir: Path | None) -> Path:
    if poscar is not None:
        return poscar.expanduser().resolve()
    templated = template_poscar(template_dir)
    if templated is not None:
        return templated.resolve()
    return Path("POSCAR").resolve()


def parse_potcar_symbols(potcar: Path) -> tuple[str, ...]:
    symbols = []
    if not potcar.exists():
        return tuple(symbols)

    for line in potcar.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        symbol = None
        if stripped.startswith("VRHFIN"):
            match = re.search(r"VRHFIN\s*=\s*([A-Za-z][A-Za-z0-9_]*)", stripped)
            if match:
                symbol = match.group(1).split(":", 1)[0]
        elif stripped.startswith("TITEL"):
            _, _, rhs = stripped.partition("=")
            parts = rhs.split()
            if len(parts) >= 2 and parts[0].startswith("PAW"):
                symbol = parts[1]
            elif parts:
                symbol = parts[0]

        if symbol:
            symbol = re.sub(r"[_-].*$", "", symbol)
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    return tuple(symbols)


def validate_vasp_template(
    template_dir: Path | None,
    atoms: Atoms | None = None,
    require_poscar: bool = False,
) -> None:
    if template_dir is None:
        print("[template] No VASP template supplied; writing POSCAR-only run folders.")
        return

    missing = [name for name in DEFAULT_TEMPLATE_FILES if not (template_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing template files in {template_dir}: {', '.join(missing)}")
    if require_poscar and template_poscar(template_dir) is None:
        raise FileNotFoundError(f"Missing POSCAR in template dir: {template_dir}")

    print(f"[template] VASP template: {template_dir.resolve()}")
    print("[template] Copying INCAR, KPOINTS, POTCAR into each run folder; POSCAR is generated.")
    if template_poscar(template_dir) is not None:
        print(f"[template] Template POSCAR: {(template_dir / 'POSCAR').resolve()}")

    potcar_symbols = parse_potcar_symbols(template_dir / "POTCAR")
    if potcar_symbols:
        print(f"[template] POTCAR symbols: {', '.join(potcar_symbols)}")

    if atoms is not None:
        poscar_symbols = species_order_from_atoms(atoms)
        print(f"[template] Structure: {summarize_atoms(atoms)}")
        print(f"[template] POSCAR symbol order: {', '.join(poscar_symbols)}")
        if potcar_symbols and tuple(poscar_symbols) != tuple(potcar_symbols):
            print(
                "[warning] POSCAR species order does not match POTCAR order: "
                f"POSCAR={list(poscar_symbols)} POTCAR={list(potcar_symbols)}"
            )
