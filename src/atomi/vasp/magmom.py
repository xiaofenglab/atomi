from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import shutil
import sys
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
class PoscarStructure:
    species: PoscarSpecies
    cell: list[list[float]]
    scaled_positions: list[list[float]]


@dataclass
class MagmomUpdate:
    species: PoscarSpecies
    selected_elements: list[str]
    moments: list[float]
    magmom_line: str
    backup_path: Path | None


@dataclass
class SpinRecord:
    run_dir: Path
    name: str
    dopant_mode: str
    host_mode: str
    moments: list[float]


def read_poscar_species(poscar: Path) -> PoscarSpecies:
    return read_poscar_structure(poscar).species


def read_poscar_structure(poscar: Path) -> PoscarStructure:
    lines = poscar.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(f"POSCAR is too short: {poscar}")

    scale = float(lines[1].split()[0])
    cell = [
        [float(value) * scale for value in lines[index].split()[:3]]
        for index in range(2, 5)
    ]
    line5 = lines[5].split()
    line6 = lines[6].split()

    if _all_ints(line5):
        raise ValueError(
            "POSCAR appears to omit the VASP5 element-symbol line. "
            "Please use a POSCAR with explicit species names."
        )
    if not _all_ints(line6):
        raise ValueError(f"Could not parse POSCAR species counts from line 7 in {poscar}")

    species = PoscarSpecies(symbols=line5, counts=[int(value) for value in line6])
    coord_index = 7
    if lines[coord_index].strip().lower().startswith("s"):
        coord_index += 1
    coord_mode = lines[coord_index].strip().lower()
    coord_index += 1

    scaled_positions = []
    for line in lines[coord_index : coord_index + species.total_atoms]:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Could not parse POSCAR coordinate line: {line}")
        xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
        if coord_mode.startswith(("c", "k")):
            xyz = cart_to_frac(xyz, cell)
        scaled_positions.append([value % 1.0 for value in xyz])
    if len(scaled_positions) != species.total_atoms:
        raise ValueError(
            f"Parsed {len(scaled_positions)} POSCAR positions, expected {species.total_atoms}."
        )
    return PoscarStructure(species=species, cell=cell, scaled_positions=scaled_positions)


def cart_to_frac(cart: list[float], cell: list[list[float]]) -> list[float]:
    # Solve row-vector cart = frac @ cell using Cramer's rule via explicit inverse.
    a, b, c = cell
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(det) < 1.0e-12:
        raise ValueError("POSCAR cell is singular.")
    inv = [
        [
            (b[1] * c[2] - b[2] * c[1]) / det,
            (a[2] * c[1] - a[1] * c[2]) / det,
            (a[1] * b[2] - a[2] * b[1]) / det,
        ],
        [
            (b[2] * c[0] - b[0] * c[2]) / det,
            (a[0] * c[2] - a[2] * c[0]) / det,
            (a[2] * b[0] - a[0] * b[2]) / det,
        ],
        [
            (b[0] * c[1] - b[1] * c[0]) / det,
            (a[1] * c[0] - a[0] * c[1]) / det,
            (a[0] * b[1] - a[1] * b[0]) / det,
        ],
    ]
    return [
        cart[0] * inv[0][j] + cart[1] * inv[1][j] + cart[2] * inv[2][j]
        for j in range(3)
    ]


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


def element_atom_indices(species: PoscarSpecies, element: str) -> list[int]:
    atom_start = 0
    for symbol, count in zip(species.symbols, species.counts):
        atom_end = atom_start + count
        if symbol == element:
            return list(range(atom_start, atom_end))
        atom_start = atom_end
    return []


def element_for_atom(species: PoscarSpecies, atom_index: int) -> str:
    atom_start = 0
    for symbol, count in zip(species.symbols, species.counts):
        atom_end = atom_start + count
        if atom_start <= atom_index < atom_end:
            return symbol
        atom_start = atom_end
    raise IndexError(atom_index)


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


def parse_element_list(values: list[str] | None) -> list[str]:
    result = []
    for value in values or []:
        for part in value.split(","):
            part = part.strip()
            if part and part not in result:
                result.append(part)
    return result


def parse_moment_specs(values: list[str] | None) -> dict[str, list[float]]:
    specs: dict[str, list[float]] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Bad moment spec, expected Element=value[,value]: {value}")
        element, raw = value.split("=", 1)
        moments = [abs(float(part)) for part in raw.replace(";", ",").split(",") if part.strip()]
        if not moments:
            raise ValueError(f"No moments provided for {element}")
        specs[element.strip()] = moments
    return specs


def infer_atom_magnitudes(
    species: PoscarSpecies,
    incar: Path,
    elements: list[str],
    moment_specs: dict[str, list[float]],
) -> list[float]:
    existing = existing_magmom_values(incar, species.total_atoms)
    magnitudes = [0.0] * species.total_atoms
    for element in elements:
        indices = element_atom_indices(species, element)
        if not indices:
            raise ValueError(f"Element {element} not found in POSCAR.")

        spec = moment_specs.get(element)
        if spec is not None:
            if existing is not None:
                existing_abs = [abs(existing[index]) for index in indices]
                for index, old_abs in zip(indices, existing_abs):
                    if old_abs > 1.0e-12:
                        magnitudes[index] = min(spec, key=lambda value: abs(value - old_abs))
                    else:
                        magnitudes[index] = spec[0]
            else:
                for index in indices:
                    magnitudes[index] = spec[0]
            continue

        if existing is None:
            raise ValueError(
                f"Could not infer moment magnitude for {element}; provide --moment {element}=VALUE."
            )
        for index in indices:
            old_abs = abs(existing[index])
            if old_abs <= 1.0e-12:
                raise ValueError(
                    f"Existing MAGMOM for {element} atom {index + 1} is zero; "
                    f"provide --moment {element}=VALUE."
                )
            magnitudes[index] = old_abs
    return magnitudes


def sign_patterns(natoms: int, mode: str) -> list[tuple[int, ...]]:
    if natoms <= 0:
        return [tuple()]
    if mode == "fm":
        return [tuple([1] * natoms), tuple([-1] * natoms)]
    if mode == "afm":
        pattern = tuple(1 if index % 2 == 0 else -1 for index in range(natoms))
        inverse = tuple(-value for value in pattern)
        return [pattern] if inverse == pattern else [pattern, inverse]
    if mode == "both":
        patterns = [tuple([1] * natoms), tuple([-1] * natoms)]
        afm = tuple(1 if index % 2 == 0 else -1 for index in range(natoms))
        for pattern in (afm, tuple(-value for value in afm)):
            if pattern not in patterns:
                patterns.append(pattern)
        return patterns
    if mode == "all":
        return list(itertools.product((1, -1), repeat=natoms))
    raise ValueError(f"Unknown spin mode: {mode}")


def apply_patterns(
    base: list[float],
    species: PoscarSpecies,
    dopants: list[str],
    hosts: list[str],
    dopant_pattern: tuple[int, ...],
    host_pattern_by_element: dict[str, tuple[int, ...]],
    host_magnitudes_by_element: dict[str, tuple[float, ...]],
) -> list[float]:
    result = list(base)
    cursor = 0
    for element in dopants:
        for atom_index in element_atom_indices(species, element):
            result[atom_index] = abs(result[atom_index]) * dopant_pattern[cursor]
            cursor += 1
    for element in hosts:
        pattern = host_pattern_by_element[element]
        magnitudes = host_magnitudes_by_element[element]
        for local_index, atom_index in enumerate(element_atom_indices(species, element)):
            result[atom_index] = abs(magnitudes[local_index]) * pattern[local_index]
    return result


def frac_distance_to_any_dopant(
    structure: PoscarStructure,
    host_atom: int,
    dopant_atoms: list[int],
) -> float:
    if not dopant_atoms:
        return 0.0
    host = structure.scaled_positions[host_atom]
    best = None
    for atom in dopant_atoms:
        dop = structure.scaled_positions[atom]
        diff = [host[i] - dop[i] for i in range(3)]
        diff = [value - round(value) for value in diff]
        cart = [
            diff[0] * structure.cell[0][j]
            + diff[1] * structure.cell[1][j]
            + diff[2] * structure.cell[2][j]
            for j in range(3)
        ]
        dist = sum(value * value for value in cart) ** 0.5
        if best is None or dist < best:
            best = dist
    return float(best or 0.0)


def host_magnitude_patterns(
    structure: PoscarStructure,
    element: str,
    magnitudes: list[float],
    dopant_atoms: list[int],
    mode: str,
    max_patterns: int,
) -> list[tuple[float, ...]]:
    indices = element_atom_indices(structure.species, element)
    values = [abs(magnitudes[index]) for index in indices]
    if not values:
        return [tuple()]
    if mode == "fixed" or len(set(values)) <= 1:
        return [tuple(values)]
    if mode not in ("enumerate", "near-dopant"):
        raise ValueError(f"Unknown host site mode: {mode}")

    template_pattern = tuple(values)
    majority = max(set(values), key=values.count)
    distances = [
        frac_distance_to_any_dopant(structure, atom_index, dopant_atoms)
        for atom_index in indices
    ]
    minority = [
        (magnitude, values.count(magnitude))
        for magnitude in sorted(set(values))
        if abs(magnitude - majority) > 1.0e-12
    ]
    minority.sort(key=lambda item: (item[1], item[0]))

    states = [(0.0, {}, tuple(range(len(values))))]
    for magnitude, count in minority:
        next_states = []
        for score, assignment, available in states:
            for chosen in bounded_position_combinations(available, count, distances, max_patterns):
                new_assignment = dict(assignment)
                for local_index in chosen:
                    new_assignment[local_index] = magnitude
                new_available = tuple(index for index in available if index not in set(chosen))
                new_score = score + sum(distances[index] for index in chosen)
                next_states.append((new_score, new_assignment, new_available))
        next_states.sort(key=lambda item: (item[0], tuple(sorted(item[1].items()))))
        states = next_states[:max_patterns]

    patterns = [template_pattern]
    for _score, assignment, _available in states[:max_patterns]:
        pattern = [majority] * len(values)
        for local_index, magnitude in assignment.items():
            pattern[local_index] = magnitude
        pattern = tuple(pattern)
        if pattern not in patterns:
            patterns.append(pattern)
        if len(patterns) >= max_patterns:
            break
    return patterns


def bounded_position_combinations(
    available: tuple[int, ...],
    count: int,
    distances: list[float],
    max_patterns: int,
) -> list[tuple[int, ...]]:
    if count <= 0:
        return [tuple()]
    if count >= len(available):
        return [tuple(available)]

    total = math.comb(len(available), count)
    if total <= max_patterns * 20:
        combos = list(itertools.combinations(available, count))
        combos.sort(key=lambda combo: (sum(distances[index] for index in combo), combo))
        return combos[:max_patterns]

    ranked = sorted(available, key=lambda index: (distances[index], index))
    candidates: list[tuple[int, ...]] = []

    def add(combo):
        combo = tuple(sorted(combo))
        if combo not in candidates:
            candidates.append(combo)

    add(ranked[:count])
    add(ranked[-count:])
    if count < len(ranked):
        for start in range(0, min(max_patterns * 2, len(ranked) - count + 1)):
            add(ranked[start : start + count])
            if len(candidates) >= max_patterns:
                break
    stride = max(1, len(ranked) // max(count, 1))
    add(ranked[::stride][:count])

    candidates.sort(key=lambda combo: (sum(distances[index] for index in combo), combo))
    return candidates[:max_patterns]


def copy_template_files(template_dir: Path, run_dir: Path, incar_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("POSCAR", "POTCAR", "KPOINTS"):
        src = template_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing template file: {src}")
        shutil.copy2(src, run_dir / name)
    (run_dir / "INCAR").write_text(incar_text, encoding="utf-8")


def write_spin_runlist(records: list[SpinRecord], runlist: Path) -> None:
    lines = []
    base = runlist.parent.resolve()
    for record in records:
        try:
            lines.append(str(record.run_dir.resolve().relative_to(base)))
        except ValueError:
            lines.append(str(record.run_dir.resolve()))
    runlist.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_spin_index(records: list[SpinRecord], index_path: Path, species: PoscarSpecies) -> None:
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run_dir", "name", "dopant_mode", "host_mode", "moments_by_atom"),
        )
        writer.writeheader()
        for record in records:
            moments = [
                {
                    "atom": index + 1,
                    "element": element_for_atom(species, index),
                    "magmom": record.moments[index],
                }
                for index in range(species.total_atoms)
            ]
            writer.writerow(
                {
                    "run_dir": str(record.run_dir.resolve()),
                    "name": record.name,
                    "dopant_mode": record.dopant_mode,
                    "host_mode": record.host_mode,
                    "moments_by_atom": json.dumps(moments),
                }
            )


def enumerate_spin_configs(args: argparse.Namespace) -> list[SpinRecord]:
    template = args.template.resolve()
    poscar = template / "POSCAR"
    incar = template / "INCAR"
    structure = read_poscar_structure(poscar)
    species = structure.species

    dopants = parse_element_list(args.dopant)
    hosts = parse_element_list(args.host)
    elements = []
    for element in dopants + hosts + parse_element_list(args.element):
        if element not in elements:
            elements.append(element)
    if not elements:
        raise ValueError("Provide at least one --dopant, --host, or --element.")
    for element in elements:
        if element not in species.symbols:
            raise ValueError(f"Element {element} not found in POSCAR species: {', '.join(species.symbols)}")

    if not dopants:
        dopants = elements[:1]
    if not hosts:
        hosts = [element for element in elements if element not in dopants]

    magnitudes = infer_atom_magnitudes(species, incar, elements, parse_moment_specs(args.moment))
    base_moments = existing_magmom_values(incar, species.total_atoms) or [0.0] * species.total_atoms
    for index, magnitude in enumerate(magnitudes):
        if magnitude > 0:
            base_moments[index] = magnitude
        elif not args.preserve_unselected:
            base_moments[index] = 0.0

    dopant_indices = [index for element in dopants for index in element_atom_indices(species, element)]
    dopant_patterns = sign_patterns(len(dopant_indices), args.dopant_mode)
    dopant_atom_indices = [index for element in dopants for index in element_atom_indices(species, element)]
    host_sign_patterns = {
        element: sign_patterns(len(element_atom_indices(species, element)), args.host_mode)
        for element in hosts
    }
    host_mag_patterns = {
        element: host_magnitude_patterns(
            structure,
            element,
            magnitudes,
            dopant_atom_indices,
            args.host_site_mode,
            args.max_site_patterns,
        )
        for element in hosts
    }

    combinations = []
    host_items = list(host_sign_patterns)
    host_configs = []
    host_sign_product = itertools.product(*(host_sign_patterns[element] for element in host_items))
    for host_sign_values in host_sign_product:
        host_sign_by_element = {
            element: pattern for element, pattern in zip(host_items, host_sign_values)
        }
        host_mag_product = itertools.product(*(host_mag_patterns[element] for element in host_items))
        for host_mag_values in host_mag_product:
            host_mag_by_element = {
                element: pattern for element, pattern in zip(host_items, host_mag_values)
            }
            host_configs.append((host_sign_by_element, host_mag_by_element))

    for host_sign_by_element, host_mag_by_element in host_configs:
        for dopant_pattern in dopant_patterns:
            combinations.append((dopant_pattern, host_sign_by_element, host_mag_by_element))

    if len(combinations) > args.max_configs:
        if args.truncate:
            combinations = combinations[: args.max_configs]
        else:
            raise ValueError(
                f"Spin enumeration would create {len(combinations)} configs, "
                f"above --max-configs {args.max_configs}. Use a smaller mode or --truncate."
            )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, (dopant_pattern, host_by_element, host_mag_by_element) in enumerate(
        combinations,
        start=1,
    ):
        moments = apply_patterns(
            base_moments,
            species,
            dopants,
            hosts,
            dopant_pattern,
            host_by_element,
            host_mag_by_element,
        )
        magmom_line = format_magmom_line(
            species,
            moments,
            selected_elements=elements,
            decimals=args.decimals,
            compact_zero=not args.no_compact_zero,
        )
        incar_text = replace_or_append_magmom(incar, magmom_line)
        name = f"spin_{idx:03d}"
        run_dir = output_root / name
        copy_template_files(template, run_dir, incar_text)
        records.append(
            SpinRecord(
                run_dir=run_dir,
                name=name,
                dopant_mode=args.dopant_mode,
                host_mode=args.host_mode,
                moments=moments,
            )
        )

    runlist = args.runlist.resolve() if args.runlist else output_root / "runlist.txt"
    index = args.index.resolve() if args.index else output_root / "spin_index.csv"
    write_spin_runlist(records, runlist)
    write_spin_index(records, index, species)
    print(f"POSCAR species : {' '.join(species.symbols)}")
    print(f"POSCAR counts  : {' '.join(str(count) for count in species.counts)}")
    print(f"Dopants        : {' '.join(dopants) if dopants else 'none'}")
    print(f"Hosts          : {' '.join(hosts) if hosts else 'none'}")
    if hosts:
        print("Host site pats : " + ", ".join(
            f"{element}={len(host_mag_patterns[element])}" for element in hosts
        ))
    print(f"Spin configs   : {len(records)}")
    print(f"Output root    : {output_root}")
    print(f"Runlist        : {runlist}")
    print(f"Index          : {index}")
    return records


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


def build_enum_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magit enum",
        description="Generate spin-arrangement VASP folders from template POSCAR/INCAR.",
    )
    parser.add_argument("--template", type=Path, default=Path("VASP_TEMPLATE"))
    parser.add_argument("--output-root", type=Path, default=Path("SPIN_CANDIDATES"))
    parser.add_argument("--runlist", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument(
        "--dopant",
        action="append",
        default=None,
        help="Dopant element to enumerate. Repeatable or comma-separated, e.g. --dopant Gd.",
    )
    parser.add_argument(
        "--host",
        action="append",
        default=None,
        help="Host magnetic element. Repeatable or comma-separated, e.g. --host U.",
    )
    parser.add_argument(
        "--element",
        action="append",
        default=None,
        help="Additional magnetic element when not distinguishing dopant/host.",
    )
    parser.add_argument(
        "--moment",
        action="append",
        default=None,
        help="Moment magnitude override, e.g. --moment Gd=7 --moment U=2,1.",
    )
    parser.add_argument(
        "--dopant-mode",
        choices=("all", "fm", "afm", "both"),
        default="all",
        help="Dopant sign patterns. Default all gives every +/- arrangement.",
    )
    parser.add_argument(
        "--host-mode",
        choices=("afm", "fm", "both", "all"),
        default="afm",
        help="Host sign patterns. Default afm gives alternating signs and the inverse.",
    )
    parser.add_argument(
        "--host-site-mode",
        choices=("enumerate", "near-dopant", "fixed"),
        default="enumerate",
        help=(
            "How to place different same-element magnitudes, such as U 2/1, among host sites. "
            "Default enumerate ranks low-count magnitudes near dopants."
        ),
    )
    parser.add_argument(
        "--max-site-patterns",
        type=int,
        default=6,
        help="Maximum magnitude-placement patterns per host element before global max-configs.",
    )
    parser.add_argument("--max-configs", type=int, default=50)
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="If enumeration exceeds --max-configs, keep the first max-configs.",
    )
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--no-compact-zero", action="store_true")
    parser.add_argument(
        "--preserve-unselected",
        action="store_true",
        help="Keep existing MAGMOM values for elements not selected. Default zeroes them.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("report", "spin-report", "analyze"):
        from atomi.vasp.spin_report import main as spin_report_main

        spin_report_main(argv[1:])
        return
    if argv and argv[0] in ("enum", "enumerate", "spins"):
        parser = build_enum_parser()
        args = parser.parse_args(argv[1:])
        enumerate_spin_configs(args)
        return

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
