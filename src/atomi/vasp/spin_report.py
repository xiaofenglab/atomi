"""Report VASP spin configurations, final moments, and energies."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from atomi.vasp.checks import array_indexed_output_candidates, collect_run_energies, iter_runlist
from atomi.vasp.magmom import (
    PoscarSpecies,
    existing_magmom_values,
    read_poscar_species,
)


MAG_HEADER_RE = re.compile(r"magnetization\s*\(x\)", re.IGNORECASE)
NIONS_RE = re.compile(r"\bNIONS\s*=\s*(\d+)")
ONSITE_ATOM_RE = re.compile(r"\batom\s*=\s*(\d+)\s+type\s*=\s*(\d+)\s+l\s*=\s*(\d+)", re.IGNORECASE)


@dataclass
class MagnetizationRow:
    ion: int
    s: float = 0.0
    p: float = 0.0
    d: float = 0.0
    f: float = 0.0
    tot: float = 0.0


@dataclass
class MagnetizationBlock:
    rows: list[MagnetizationRow]
    line_number: int
    expected_atoms: int | None = None
    warning: str | None = None
    source_kind: str = "magnetization_x"

    @property
    def moments(self) -> list[float]:
        return [row.tot for row in self.rows]


@dataclass
class AtomReport:
    index: int
    element: str
    initial: float | None
    final: float
    delta: float | None
    changed: bool
    mag_class: str
    physics_ok: bool | None = None
    physics_target: float | None = None
    physics_delta: float | None = None


@dataclass
class RunSpinReport:
    index: int
    run: Path
    resolved_run: Path
    output_run_dir: Path | None
    status: str
    energy_eV: float | None = None
    energy_kind: str = ""
    energy_source: Path | None = None
    mag_source: Path | None = None
    mag_status: str = "NO_OUTCAR"
    atoms: list[AtomReport] = field(default_factory=list)
    total_moment: float | None = None
    max_abs_moment: float | None = None
    changed_count: int = 0
    changed_by_element: dict[str, int] = field(default_factory=dict)
    initial_element_order: dict[str, str] = field(default_factory=dict)
    element_order: dict[str, str] = field(default_factory=dict)
    element_sum: dict[str, float] = field(default_factory=dict)
    summary_counts: dict[str, int] = field(default_factory=dict)
    warning: str = ""
    spin_index_name: str = ""
    dopant_mode: str = ""
    host_mode: str = ""
    physics_guard_status: str = "NOT_APPLIED"
    physics_guard_bad_count: int = 0
    physics_guard_bad_by_element: dict[str, int] = field(default_factory=dict)


def infer_nions_from_outcar(outcar: Path) -> int | None:
    opener = gzip.open if outcar.suffix == ".gz" else open
    with opener(outcar, "rt", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    matches = NIONS_RE.findall(text)
    return int(matches[-1]) if matches else None


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _parse_mag_rows(lines: list[str], start: int) -> list[MagnetizationRow]:
    rows: list[MagnetizationRow] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        parts = stripped.split()
        if not parts:
            continue
        if parts[0].lower().startswith("tot"):
            break
        try:
            ion = int(parts[0])
        except ValueError:
            continue
        values = [_parse_float(part) for part in parts[1:]]
        values = [value for value in values if value is not None]
        if len(values) < 4:
            continue
        if len(values) == 4:
            s, p, d, tot = values[:4]
            f = 0.0
        else:
            s, p, d, f, tot = values[:5]
        rows.append(MagnetizationRow(ion=ion, s=s, p=p, d=d, f=f, tot=tot))
    return rows


def _read_text_lines(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()


def extract_last_magnetization_block(outcar: Path, natoms: int | None = None) -> MagnetizationBlock:
    if not outcar.is_file():
        raise FileNotFoundError(f"OUTCAR not found: {outcar}")
    lines = _read_text_lines(outcar)
    expected = natoms if natoms is not None else infer_nions_from_outcar(outcar)
    candidates: list[MagnetizationBlock] = []
    for index, line in enumerate(lines):
        if not MAG_HEADER_RE.search(line):
            continue
        rows = _parse_mag_rows(lines, index)
        if rows:
            candidates.append(MagnetizationBlock(rows=rows, line_number=index + 1, expected_atoms=expected))
    if not candidates:
        onsite = extract_last_onsite_density_matrix_block(outcar, natoms=expected, lines=lines)
        if onsite is not None:
            return onsite
        raise ValueError(
            f"No 'magnetization (x)' table found in {outcar}. "
            "LORBIT may not have been enabled, or VASP stopped before writing moments."
        )
    if expected is not None:
        for block in reversed(candidates):
            if len(block.rows) == expected:
                return block
        onsite = extract_last_onsite_density_matrix_block(outcar, natoms=expected, lines=lines)
        if onsite is not None:
            return onsite
        block = candidates[-1]
        block.warning = (
            f"Extracted {len(block.rows)} moment rows, but expected {expected} atoms. "
            "Using the last available block."
        )
        return block
    return candidates[-1]


def extract_last_onsite_density_matrix_block(
    outcar: Path,
    natoms: int | None = None,
    lines: list[str] | None = None,
) -> MagnetizationBlock | None:
    lines = _read_text_lines(outcar) if lines is None else lines
    expected = natoms if natoms is not None else infer_nions_from_outcar(outcar)
    groups: list[tuple[int, dict[int, MagnetizationRow]]] = []
    current: dict[int, MagnetizationRow] = {}
    group_start = 0
    last_ion = 0
    for index, line in enumerate(lines):
        match = ONSITE_ATOM_RE.search(line)
        if not match:
            continue
        ion = int(match.group(1))
        angular_l = int(match.group(3))
        if current and (ion in current or ion < last_ion):
            groups.append((group_start, current))
            current = {}
        if not current:
            group_start = index + 1
        moment = _parse_onsite_atom_moment(lines, index, angular_l)
        if moment is None:
            continue
        current[ion] = _onsite_row(ion, angular_l, moment)
        last_ion = ion
    if current:
        groups.append((group_start, current))
    if not groups:
        return None
    line_number, rows_by_ion = groups[-1]
    if expected is None:
        rows = [rows_by_ion[ion] for ion in sorted(rows_by_ion)]
        expected = len(rows)
        missing = 0
    else:
        rows = [rows_by_ion.get(ion, MagnetizationRow(ion=ion)) for ion in range(1, expected + 1)]
        missing = expected - len(rows_by_ion)
    warning = (
        "No complete 'magnetization (x)' table was found; using onsite density matrix "
        "trace(spin component 1) - trace(spin component 2) fallback."
    )
    if missing > 0:
        warning += f" {missing} atoms without onsite matrices were filled with 0.0."
    return MagnetizationBlock(
        rows=rows,
        line_number=line_number,
        expected_atoms=expected,
        warning=warning,
        source_kind="onsite_density_matrix",
    )


def _parse_onsite_atom_moment(lines: list[str], atom_line_index: int, angular_l: int) -> float | None:
    dim = 2 * angular_l + 1
    first_index = _find_spin_component(lines, atom_line_index + 1, component=1)
    if first_index is None:
        return None
    first, _next = _parse_square_matrix(lines, first_index + 1, dim)
    second_index = _find_spin_component(lines, first_index + 1, component=2)
    if second_index is None:
        return None
    second, _next = _parse_square_matrix(lines, second_index + 1, dim)
    if len(first) != dim or len(second) != dim:
        return None
    return _matrix_trace(first) - _matrix_trace(second)


def _find_spin_component(lines: list[str], start: int, component: int) -> int | None:
    needle = f"spin component  {component}"
    for index in range(start, min(len(lines), start + 80)):
        stripped = lines[index].strip().lower()
        if ONSITE_ATOM_RE.search(lines[index]):
            return None
        if stripped == needle or stripped == f"spin component {component}":
            return index
    return None


def _parse_square_matrix(lines: list[str], start: int, dim: int) -> tuple[list[list[float]], int]:
    matrix: list[list[float]] = []
    index = start
    while index < len(lines) and len(matrix) < dim:
        stripped = lines[index].strip()
        index += 1
        if not stripped:
            continue
        values = [_parse_float(part) for part in stripped.split()]
        numeric = [value for value in values if value is not None]
        if len(numeric) < dim:
            if matrix:
                break
            continue
        matrix.append(numeric[:dim])
    return matrix, index


def _matrix_trace(matrix: list[list[float]]) -> float:
    return sum(row[index] for index, row in enumerate(matrix) if index < len(row))


def _onsite_row(ion: int, angular_l: int, moment: float) -> MagnetizationRow:
    row = MagnetizationRow(ion=ion, tot=moment)
    if angular_l == 0:
        row.s = moment
    elif angular_l == 1:
        row.p = moment
    elif angular_l == 2:
        row.d = moment
    elif angular_l == 3:
        row.f = moment
    return row


def read_poscar_species_auto(species_file: Path) -> PoscarSpecies:
    try:
        return read_poscar_species(species_file)
    except Exception as original_exc:
        try:
            from ase.io import read as ase_read
        except Exception as import_exc:  # pragma: no cover - depends on optional runtime state.
            raise original_exc from import_exc
        try:
            atoms = ase_read(str(species_file), format="vasp")
            symbols = list(atoms.get_chemical_symbols())
        except Exception as ase_exc:
            raise original_exc from ase_exc
        if not symbols:
            raise original_exc
        grouped_symbols: list[str] = []
        counts: list[int] = []
        for symbol in symbols:
            if grouped_symbols and grouped_symbols[-1] == symbol:
                counts[-1] += 1
            else:
                grouped_symbols.append(symbol)
                counts.append(1)
        return PoscarSpecies(symbols=grouped_symbols, counts=counts)


def read_species_labels(species_file: Path | None, natoms: int | None) -> tuple[list[str], PoscarSpecies | None, str]:
    if species_file is None:
        count = natoms or 0
        return ["X"] * count, None, "none"
    try:
        species = read_poscar_species_auto(species_file)
    except Exception as exc:  # pragma: no cover - defensive path, message is tested indirectly.
        count = natoms or 0
        return ["X"] * count, None, f"Could not parse species file {species_file}: {exc}"
    labels = []
    for symbol, count in zip(species.symbols, species.counts):
        labels.extend([symbol] * count)
    return labels, species, ""


def classify_moment(value: float, thresholds: tuple[float, float, float] = (0.5, 1.5, 5.0)) -> str:
    abs_value = abs(value)
    low, mid, high = thresholds
    if abs_value < low:
        return "near_zero"
    if low <= abs_value < mid:
        return "small_0p5_1p5"
    if mid <= abs_value < 2.5:
        return "medium_1p5_2p5"
    if abs_value > high:
        return "large_gt5"
    return "other"


def parse_moment_guards(values: list[str], default_tol: float) -> dict[str, tuple[list[float], float]]:
    guards: dict[str, tuple[list[float], float]] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --moment-guard {raw!r}; use Element=v1,v2 or Element=v1,v2@tol.")
        element, spec = raw.split("=", 1)
        element = element.strip()
        if not element:
            raise ValueError(f"Invalid --moment-guard {raw!r}; element is empty.")
        tolerance = default_tol
        if "@" in spec:
            spec, tol_text = spec.rsplit("@", 1)
            try:
                tolerance = float(tol_text)
            except ValueError as exc:
                raise ValueError(f"Invalid tolerance in --moment-guard {raw!r}.") from exc
        targets = []
        for item in re.split(r"[,;/]+", spec):
            item = item.strip()
            if not item:
                continue
            try:
                targets.append(float(item))
            except ValueError as exc:
                raise ValueError(f"Invalid moment value {item!r} in --moment-guard {raw!r}.") from exc
        if not targets:
            raise ValueError(f"Invalid --moment-guard {raw!r}; no target moments were given.")
        guards[element] = (targets, tolerance)
    return guards


def _append_unique_float(values: list[float], value: float, tolerance: float = 1.0e-9) -> None:
    if not any(abs(existing - value) <= tolerance for existing in values):
        values.append(value)


def _integer_like_magnitude(value: float, integer_tol: float) -> float:
    magnitude = abs(value)
    nearest = round(magnitude)
    if nearest > 0 and abs(magnitude - nearest) <= integer_tol:
        return float(nearest)
    return round(magnitude, 3)


def infer_moment_guards_from_initial_moments(
    species: PoscarSpecies,
    moments: list[float] | None,
    default_tol: float,
    integer_tol: float = 0.35,
    zero_tol: float = 0.25,
    zero_threshold: float = 0.25,
) -> dict[str, tuple[list[float], float]]:
    """Infer element moment guards from POSCAR order and initial INCAR MAGMOM."""

    if moments is None:
        return {}
    guards: dict[str, tuple[list[float], float]] = {}
    offset = 0
    for element, count in zip(species.symbols, species.counts):
        values = moments[offset : offset + count]
        offset += count
        if not values:
            continue
        targets: list[float] = []
        has_zero = False
        for value in values:
            if abs(value) <= zero_threshold:
                has_zero = True
                continue
            magnitude = _integer_like_magnitude(value, integer_tol=integer_tol)
            _append_unique_float(targets, magnitude)
            _append_unique_float(targets, -magnitude)
        if has_zero:
            _append_unique_float(targets, 0.0)
        if not targets:
            continue
        tolerance = zero_tol if targets == [0.0] else default_tol
        guards[element] = (targets, tolerance)
    return guards


def infer_moment_guards_from_files(
    species_file: Path | None,
    incar: Path | None,
    default_tol: float,
    integer_tol: float = 0.35,
    zero_tol: float = 0.25,
    zero_threshold: float = 0.25,
) -> dict[str, tuple[list[float], float]]:
    if species_file is None or incar is None or not species_file.is_file() or not incar.is_file():
        return {}
    try:
        species = read_poscar_species_auto(species_file)
        moments = existing_magmom_values(incar, species.total_atoms)
    except Exception:
        return {}
    return infer_moment_guards_from_initial_moments(
        species=species,
        moments=moments,
        default_tol=default_tol,
        integer_tol=integer_tol,
        zero_tol=zero_tol,
        zero_threshold=zero_threshold,
    )


def infer_moment_guards_from_run_dirs(
    run_dirs: list[Path],
    default_tol: float,
    species_override: Path | None = None,
    incar_override: Path | None = None,
    integer_tol: float = 0.35,
    zero_tol: float = 0.25,
    zero_threshold: float = 0.25,
) -> dict[str, tuple[list[float], float]]:
    if species_override is not None and incar_override is not None:
        guards = infer_moment_guards_from_files(
            species_override,
            incar_override,
            default_tol=default_tol,
            integer_tol=integer_tol,
            zero_tol=zero_tol,
            zero_threshold=zero_threshold,
        )
        if guards:
            return guards
    for run_dir in run_dirs:
        species_file = species_override or default_species_file(run_dir)
        incar = incar_override or default_incar(run_dir)
        guards = infer_moment_guards_from_files(
            species_file,
            incar,
            default_tol=default_tol,
            integer_tol=integer_tol,
            zero_tol=zero_tol,
            zero_threshold=zero_threshold,
        )
        if guards:
            return guards
    return {}


def merge_moment_guards(
    inferred: dict[str, tuple[list[float], float]],
    explicit: dict[str, tuple[list[float], float]],
) -> dict[str, tuple[list[float], float]]:
    guards = dict(inferred)
    guards.update(explicit)
    return guards


def runlist_dirs(runlist: Path) -> list[Path]:
    base = runlist.parent.resolve()
    dirs: list[Path] = []
    for run in iter_runlist(runlist):
        dirs.append(run if run.is_absolute() else (base / run).resolve())
    return dirs


def auto_moment_guard_notice(guards: dict[str, tuple[list[float], float]], enabled: bool) -> str:
    if not enabled:
        return "disabled"
    return guard_rule_text(guards) if guards else "not inferred"


def guard_rule_text(guards: dict[str, tuple[list[float], float]]) -> str:
    pieces = []
    for element in guards:
        targets, tolerance = guards[element]
        values = ",".join(format_float(value, decimals=3) for value in targets)
        pieces.append(f"{element}=[{values}] tol={tolerance:g}")
    return "; ".join(pieces)


def apply_moment_guards(atoms: list[AtomReport], guards: dict[str, tuple[list[float], float]]) -> tuple[str, int, dict[str, int]]:
    if not guards:
        return "NOT_APPLIED", 0, {}
    bad_by_element: dict[str, int] = {}
    checked = 0
    for atom in atoms:
        rule = guards.get(atom.element)
        if rule is None:
            atom.physics_ok = None
            atom.physics_target = None
            atom.physics_delta = None
            continue
        checked += 1
        targets, tolerance = rule
        target = min(targets, key=lambda value: abs(atom.final - value))
        delta = atom.final - target
        atom.physics_target = target
        atom.physics_delta = delta
        atom.physics_ok = abs(delta) <= tolerance
        if not atom.physics_ok:
            bad_by_element[atom.element] = bad_by_element.get(atom.element, 0) + 1
    if checked == 0:
        return "NO_MATCHED_ELEMENTS", 0, {}
    bad_count = sum(bad_by_element.values())
    return ("OK" if bad_count == 0 else "FAIL"), bad_count, bad_by_element


def summarize_counts(moments: list[float]) -> dict[str, int]:
    return {
        "abs_gt5": sum(1 for value in moments if abs(value) > 5.0),
        "abs_0p5_1p5": sum(1 for value in moments if 0.5 < abs(value) < 1.5),
        "abs_1p5_2p5": sum(1 for value in moments if 1.5 < abs(value) < 2.5),
    }


def magnetic_order(values: list[float], threshold: float = 0.2) -> str:
    active = [value for value in values if abs(value) >= threshold]
    if not active:
        return "nonmagnetic"
    if len(active) == 1:
        return "single_spin"
    positives = sum(1 for value in active if value > 0)
    negatives = sum(1 for value in active if value < 0)
    if positives == 0 or negatives == 0:
        return "FM"
    total = sum(active)
    max_abs = max(abs(value) for value in active)
    if abs(total) <= max(threshold, 0.25 * max_abs):
        return "AFM"
    return "AFM-like"


def format_float(value: float, decimals: int = 3, show_plus: bool = True) -> str:
    text = f"{value:+.{decimals}f}" if show_plus else f"{value:.{decimals}f}"
    if text.startswith("+0.") or text.startswith("-0."):
        if abs(value) < 0.5 * 10 ** (-decimals):
            text = "+" + f"{0.0:.{decimals}f}" if show_plus else f"{0.0:.{decimals}f}"
    return text


def magmom_line(moments: list[float], decimals: int = 3) -> str:
    return "MAGMOM = " + " ".join(format_float(value, decimals=decimals) for value in moments)


def compressed_magmom_line(moments: list[float], decimals: int = 3, tolerance: float = 0.05) -> str:
    if not moments:
        return "MAGMOM ="
    groups: list[tuple[int, float]] = []
    current = moments[0]
    count = 1
    for value in moments[1:]:
        if abs(value - current) <= tolerance:
            count += 1
            current = (current * (count - 1) + value) / count
        else:
            groups.append((count, current))
            current = value
            count = 1
    groups.append((count, current))
    pieces = []
    for count, value in groups:
        value_text = format_float(value, decimals=decimals)
        pieces.append(f"{count}*{value_text}" if count > 1 else value_text)
    return "MAGMOM = " + " ".join(pieces)


def write_single_outputs(
    outcar: Path,
    output_prefix: Path,
    block: MagnetizationBlock,
    labels: list[str],
    formats: str,
    decimals: int,
    compress_tol: float,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    block_path = output_prefix.with_name(output_prefix.name + "_last_magnetization_block.txt")
    expanded_path = output_prefix.with_name(output_prefix.name + "_MAGMOM_expanded.txt")
    vasp_path = output_prefix.with_name(output_prefix.name + "_MAGMOM_vasp.txt")
    labels = labels or ["X"] * len(block.rows)
    if len(labels) < len(block.rows):
        labels = labels + ["X"] * (len(block.rows) - len(labels))

    with block_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# source: {outcar}\n")
        handle.write(f"# magnetization block starts at line: {block.line_number}\n")
        if block.warning:
            handle.write(f"# warning: {block.warning}\n")
        handle.write("ion element s p d f tot\n")
        for row, element in zip(block.rows, labels):
            handle.write(
                f"{row.ion} {element} {row.s:.6f} {row.p:.6f} "
                f"{row.d:.6f} {row.f:.6f} {row.tot:.6f}\n"
            )

    if formats in {"expanded", "both"}:
        with expanded_path.open("w", encoding="utf-8") as handle:
            for row, element in zip(block.rows, labels):
                handle.write(f"{row.ion:<6d} {element:<4s} {format_float(row.tot, decimals)}\n")
    if formats in {"vasp", "both"}:
        moments = block.moments
        with vasp_path.open("w", encoding="utf-8") as handle:
            handle.write("# Expanded numerical line, safest for restart INCAR use\n")
            handle.write(magmom_line(moments, decimals=decimals) + "\n")
            handle.write("# Consecutive near-identical values compressed for readability\n")
            handle.write(compressed_magmom_line(moments, decimals=decimals, tolerance=compress_tol) + "\n")


def print_single_summary(block: MagnetizationBlock, labels: list[str]) -> None:
    moments = block.moments
    labels = labels or ["X"] * len(moments)
    counts = summarize_counts(moments)
    print(f"Moment rows        : {len(moments)}")
    print(f"Total moment       : {sum(moments):.6f}")
    print(f"Max |moment|       : {max((abs(value) for value in moments), default=0.0):.6f}")
    print(f"|moment| > 5       : {counts['abs_gt5']}")
    print(f"0.5 < |m| < 1.5    : {counts['abs_0p5_1p5']}")
    print(f"1.5 < |m| < 2.5    : {counts['abs_1p5_2p5']}")
    element_values: dict[str, list[float]] = {}
    for element, moment in zip(labels, moments):
        element_values.setdefault(element, []).append(moment)
    for element, values in element_values.items():
        print(f"{element} order        : {magnetic_order(values)}  sum={sum(values):.6f}")
    if block.warning:
        print(f"Warning            : {block.warning}")


def load_spin_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    rows: dict[str, dict[str, str]] = {}
    base = path.parent.resolve()
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_dir = row.get("run_dir", "")
            if not run_dir:
                continue
            resolved = Path(run_dir)
            if not resolved.is_absolute():
                resolved = base / resolved
            row["_resolved_run_dir"] = str(resolved.resolve())
            rows[str(resolved.resolve())] = row
            rows[resolved.name] = row
    return rows


def default_species_file(run_dir: Path) -> Path | None:
    for name in ("CONTCAR", "POSCAR"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def default_incar(run_dir: Path) -> Path | None:
    path = run_dir / "INCAR"
    return path if path.is_file() else None


def _context_dirs(intended_run_dir: Path, output_file: Path | None = None) -> list[Path]:
    candidates = [intended_run_dir]
    if output_file is not None:
        cursor = output_file.parent
        for _ in range(4):
            candidates.append(cursor)
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def first_species_file(intended_run_dir: Path, output_file: Path | None = None) -> Path | None:
    for directory in _context_dirs(intended_run_dir, output_file):
        species_file = default_species_file(directory)
        if species_file is not None:
            return species_file
    return None


def first_incar_file(intended_run_dir: Path, output_file: Path | None = None) -> Path | None:
    for directory in _context_dirs(intended_run_dir, output_file):
        incar = default_incar(directory)
        if incar is not None:
            return incar
    return None


def default_outcar(run_dir: Path) -> Path | None:
    path = run_dir / "OUTCAR"
    return path if path.is_file() else None


def magnetization_candidate_files(index: int, run_dir: Path, log_dir: Path | None) -> list[Path]:
    candidates: list[Path] = []
    for name in ("OUTCAR", "OUTCAR.gz"):
        path = run_dir / name
        if path.is_file():
            candidates.append(path)
    search_log_dir = run_dir.parent if log_dir is None else log_dir
    candidates.extend(array_indexed_output_candidates(index, search_log_dir, deep=True))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique, key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)


def extract_first_available_magnetization(
    candidates: list[Path],
    expected: int | None,
) -> tuple[Path | None, MagnetizationBlock | None, str]:
    warnings = []
    for path in candidates:
        try:
            return path, extract_last_magnetization_block(path, natoms=expected), ""
        except Exception as exc:
            warnings.append(f"{path}: {exc}")
    if not candidates:
        return None, None, "OUTCAR missing; run may have been deleted or not started."
    return None, None, "No usable magnetization block found in candidate files. " + " | ".join(warnings[:3])


def build_atom_reports(
    moments: list[float],
    labels: list[str],
    initial: list[float] | None,
    change_threshold: float,
) -> list[AtomReport]:
    reports = []
    labels = labels or ["X"] * len(moments)
    if len(labels) < len(moments):
        labels = labels + ["X"] * (len(moments) - len(labels))
    for idx, (element, final) in enumerate(zip(labels, moments), start=1):
        initial_value = initial[idx - 1] if initial is not None and idx - 1 < len(initial) else None
        delta = final - initial_value if initial_value is not None else None
        sign_changed = (
            initial_value is not None
            and abs(initial_value) >= change_threshold
            and abs(final) >= change_threshold
            and (initial_value > 0) != (final > 0)
        )
        changed = bool(delta is not None and abs(delta) > change_threshold) or sign_changed
        reports.append(
            AtomReport(
                index=idx,
                element=element,
                initial=initial_value,
                final=final,
                delta=delta,
                changed=changed,
                mag_class=classify_moment(final),
            )
        )
    return reports


def element_order_from_atoms(atoms: list[AtomReport], threshold: float) -> tuple[dict[str, str], dict[str, float]]:
    values: dict[str, list[float]] = {}
    for atom in atoms:
        values.setdefault(atom.element, []).append(atom.final)
    return (
        {element: magnetic_order(moments, threshold=threshold) for element, moments in values.items()},
        {element: sum(moments) for element, moments in values.items()},
    )


def initial_order_from_atoms(atoms: list[AtomReport], threshold: float) -> dict[str, str]:
    values: dict[str, list[float]] = {}
    for atom in atoms:
        if atom.initial is None:
            continue
        values.setdefault(atom.element, []).append(atom.initial)
    return {element: magnetic_order(moments, threshold=threshold) for element, moments in values.items()}


def changed_counts_by_element(atoms: list[AtomReport]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom in atoms:
        if atom.changed:
            counts[atom.element] = counts.get(atom.element, 0) + 1
    return counts


def build_run_reports(
    runlist: Path,
    spin_index: Path | None,
    log_dir: Path | None,
    energy_kind: str,
    stopped_after_minutes: float,
    dav_average_window: int,
    species_override: Path | None,
    natoms: int | None,
    change_threshold: float,
    order_threshold: float,
    moment_guards: dict[str, tuple[list[float], float]] | None = None,
) -> list[RunSpinReport]:
    energy_records = collect_run_energies(
        runlist=runlist,
        log_dir=log_dir,
        energy_kind=energy_kind,
        stopped_after_minutes=stopped_after_minutes,
        dav_average_window=dav_average_window,
        deep_artifacts=True,
    )
    spin_rows = load_spin_index(spin_index)
    run_entries = iter_runlist(runlist)
    reports: list[RunSpinReport] = []
    for entry, energy in zip(run_entries, energy_records):
        run_dir = entry if entry.is_absolute() else runlist.parent / entry
        run_dir = run_dir.resolve()
        report = RunSpinReport(
            index=energy.index,
            run=entry,
            resolved_run=run_dir,
            output_run_dir=None,
            status=energy.status,
            energy_eV=energy.energy_eV,
            energy_kind=energy.energy_kind,
            energy_source=energy.source,
        )
        spin_row = spin_rows.get(str(run_dir)) or spin_rows.get(run_dir.name)
        if spin_row:
            report.spin_index_name = spin_row.get("name", "")
            report.dopant_mode = spin_row.get("dopant_mode", "")
            report.host_mode = spin_row.get("host_mode", "")
            indexed_run_dir = Path(spin_row.get("_resolved_run_dir", ""))
            if indexed_run_dir.is_dir() and indexed_run_dir != run_dir:
                run_dir = indexed_run_dir
                report.resolved_run = indexed_run_dir
        if not run_dir.is_dir():
            candidates = magnetization_candidate_files(energy.index, run_dir, log_dir)
            if not candidates:
                report.mag_status = "NODIR"
                report.warning = f"Run directory missing: {run_dir}"
                reports.append(report)
                continue
        else:
            candidates = magnetization_candidate_files(energy.index, run_dir, log_dir)
        species_file = species_override or default_species_file(run_dir)
        expected = natoms
        if expected is None and species_file is not None:
            try:
                expected = read_poscar_species_auto(species_file).total_atoms
            except Exception:
                expected = None
        mag_source, block, mag_warning = extract_first_available_magnetization(candidates, expected=expected)
        if block is None or mag_source is None:
            report.mag_status = "NO_MAGNETIZATION"
            report.warning = mag_warning
            reports.append(report)
            continue
        report.mag_source = mag_source
        report.output_run_dir = mag_source.parent
        species_file = species_override or first_species_file(run_dir, mag_source)
        expected = natoms
        if expected is None and species_file is not None:
            try:
                expected = read_poscar_species_auto(species_file).total_atoms
            except Exception:
                expected = None
        if expected is not None and block.expected_atoms is not None and len(block.rows) != expected:
            try:
                mag_source, block, mag_warning = extract_first_available_magnetization(candidates, expected=expected)
            except Exception:
                pass
            if block is None or mag_source is None:
                report.mag_status = "NO_MAGNETIZATION"
                report.warning = mag_warning
                reports.append(report)
                continue
            report.mag_source = mag_source
            report.output_run_dir = mag_source.parent
        labels, species, species_warning = read_species_labels(species_file, len(block.rows))
        if species_warning:
            report.warning = species_warning
        if species is not None and len(labels) != len(block.rows):
            report.warning = (
                (report.warning + " " if report.warning else "")
                + f"Species count {len(labels)} does not match moment rows {len(block.rows)}."
            )
        initial = None
        incar = first_incar_file(run_dir, mag_source)
        if incar is not None and species is not None:
            initial = existing_magmom_values(incar, species.total_atoms)
        atoms = build_atom_reports(block.moments, labels, initial, change_threshold)
        report.atoms = atoms
        if block.source_kind == "onsite_density_matrix":
            report.mag_status = "ONSITE_MATRIX"
        else:
            report.mag_status = "OK" if block.warning is None else "WARN"
        report.total_moment = sum(block.moments)
        report.max_abs_moment = max((abs(value) for value in block.moments), default=0.0)
        report.summary_counts = summarize_counts(block.moments)
        report.changed_count = sum(1 for atom in atoms if atom.changed)
        report.changed_by_element = changed_counts_by_element(atoms)
        report.initial_element_order = initial_order_from_atoms(atoms, threshold=order_threshold)
        report.element_order, report.element_sum = element_order_from_atoms(atoms, threshold=order_threshold)
        if moment_guards:
            status, bad_count, bad_by_element = apply_moment_guards(atoms, moment_guards)
            report.physics_guard_status = status
            report.physics_guard_bad_count = bad_count
            report.physics_guard_bad_by_element = bad_by_element
        if block.warning:
            report.warning = (report.warning + " " if report.warning else "") + block.warning
        reports.append(report)
    return reports


def write_run_summary(reports: list[RunSpinReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "run",
        "status",
        "energy_eV",
        "energy_kind",
        "energy_source",
        "resolved_run",
        "output_run_dir",
        "mag_source",
        "mag_status",
        "total_moment",
        "max_abs_moment",
        "changed_count",
        "changed_by_element",
        "abs_gt5",
        "abs_0p5_1p5",
        "abs_1p5_2p5",
        "initial_element_order",
        "element_order",
        "element_sum",
        "physics_guard_status",
        "physics_guard_bad_count",
        "physics_guard_bad_by_element",
        "spin_index_name",
        "dopant_mode",
        "host_mode",
        "warning",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "index": report.index,
                    "run": str(report.run),
                    "status": report.status,
                    "energy_eV": "" if report.energy_eV is None else f"{report.energy_eV:.10f}",
                    "energy_kind": report.energy_kind,
                    "energy_source": "" if report.energy_source is None else str(report.energy_source),
                    "resolved_run": str(report.resolved_run),
                    "output_run_dir": "" if report.output_run_dir is None else str(report.output_run_dir),
                    "mag_source": "" if report.mag_source is None else str(report.mag_source),
                    "mag_status": report.mag_status,
                    "total_moment": "" if report.total_moment is None else f"{report.total_moment:.8f}",
                    "max_abs_moment": "" if report.max_abs_moment is None else f"{report.max_abs_moment:.8f}",
                    "changed_count": report.changed_count,
                    "changed_by_element": json.dumps(report.changed_by_element, sort_keys=True),
                    "abs_gt5": report.summary_counts.get("abs_gt5", ""),
                    "abs_0p5_1p5": report.summary_counts.get("abs_0p5_1p5", ""),
                    "abs_1p5_2p5": report.summary_counts.get("abs_1p5_2p5", ""),
                    "initial_element_order": json.dumps(report.initial_element_order, sort_keys=True),
                    "element_order": json.dumps(report.element_order, sort_keys=True),
                    "element_sum": json.dumps(report.element_sum, sort_keys=True),
                    "physics_guard_status": report.physics_guard_status,
                    "physics_guard_bad_count": report.physics_guard_bad_count,
                    "physics_guard_bad_by_element": json.dumps(report.physics_guard_bad_by_element, sort_keys=True),
                    "spin_index_name": report.spin_index_name,
                    "dopant_mode": report.dopant_mode,
                    "host_mode": report.host_mode,
                    "warning": report.warning,
                }
            )


def write_atom_table(reports: list[RunSpinReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_index",
        "run",
        "atom",
        "element",
        "initial_moment",
        "final_moment",
        "delta",
        "changed",
        "mag_class",
        "physics_ok",
        "physics_target",
        "physics_delta",
        "energy_eV",
        "mag_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for atom in report.atoms:
                writer.writerow(
                    {
                        "run_index": report.index,
                        "run": str(report.run),
                        "atom": atom.index,
                        "element": atom.element,
                        "initial_moment": "" if atom.initial is None else f"{atom.initial:.8f}",
                        "final_moment": f"{atom.final:.8f}",
                        "delta": "" if atom.delta is None else f"{atom.delta:.8f}",
                        "changed": "yes" if atom.changed else "no",
                        "mag_class": atom.mag_class,
                        "physics_ok": "" if atom.physics_ok is None else ("yes" if atom.physics_ok else "no"),
                        "physics_target": "" if atom.physics_target is None else f"{atom.physics_target:.8f}",
                        "physics_delta": "" if atom.physics_delta is None else f"{atom.physics_delta:.8f}",
                        "energy_eV": "" if report.energy_eV is None else f"{report.energy_eV:.10f}",
                        "mag_status": report.mag_status,
                    }
                )


def write_physics_filtered_tables(reports: list[RunSpinReport], paths: dict[str, Path]) -> None:
    passed = [report for report in reports if report.physics_guard_status == "OK"]
    write_run_summary(passed, paths["filtered_summary"])
    write_atom_table(passed, paths["filtered_atoms"])


def write_magmom_lines(reports: list[RunSpinReport], output_dir: Path, decimals: int, compress_tol: float) -> None:
    magmom_dir = output_dir / "magmom_lines"
    magmom_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        if not report.atoms:
            continue
        moments = [atom.final for atom in report.atoms]
        name = report.spin_index_name or report.resolved_run.name or f"run_{report.index:03d}"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = magmom_dir / f"{report.index:03d}_{safe}_MAGMOM.txt"
        path.write_text(
            "# Expanded numerical line, safest for restart INCAR use\n"
            + magmom_line(moments, decimals=decimals)
            + "\n# Consecutive near-identical values compressed for readability\n"
            + compressed_magmom_line(moments, decimals=decimals, tolerance=compress_tol)
            + "\n",
            encoding="utf-8",
        )


def write_markdown_report(reports: list[RunSpinReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok_with_energy = [report for report in reports if report.energy_eV is not None and report.atoms]
    lines = [
        "# VASP Spin-Energy Report",
        "",
        f"Total runlist rows: {len(reports)}",
        f"Runs with energy and magnetization: {len(ok_with_energy)}",
        "",
    ]
    guarded = [report for report in ok_with_energy if report.physics_guard_status not in {"", "NOT_APPLIED"}]
    if guarded:
        passed = [report for report in guarded if report.physics_guard_status == "OK"]
        failed = [report for report in guarded if report.physics_guard_status == "FAIL"]
        lines.extend(
            [
                "## Physics Guard Summary",
                "",
                f"- Guarded runs with energy and moments: {len(guarded)}",
                f"- Physics-accepted runs: {len(passed)}",
                f"- Physics-rejected runs: {len(failed)}",
                "",
            ]
        )
    if ok_with_energy:
        minimum = min(report.energy_eV for report in ok_with_energy if report.energy_eV is not None)
        best = min(ok_with_energy, key=lambda report: report.energy_eV or float("inf"))
        lines.extend(
            [
                "## Lowest Energy Configuration",
                "",
                f"- Run: `{best.run}`",
                f"- Energy: {best.energy_eV:.10f} eV",
                f"- Total moment: {best.total_moment:.6f}",
                f"- Max |moment|: {best.max_abs_moment:.6f}",
                f"- Element order: `{json.dumps(best.element_order, sort_keys=True)}`",
                "",
                "## Energy Table",
                "",
                "| index | run | dE_eV | total_moment | max_abs | changed | physics_guard | element_order | status |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for report in ok_with_energy:
            delta = (report.energy_eV or 0.0) - minimum
            lines.append(
                f"| {report.index} | `{report.run}` | {delta:.8f} | "
                f"{report.total_moment:.6f} | {report.max_abs_moment:.6f} | "
                f"{report.changed_count} | {report.physics_guard_status} | "
                f"`{json.dumps(report.element_order, sort_keys=True)}` | "
                f"{report.status}/{report.mag_status} |"
            )
        lines.append("")
    warnings = [report for report in reports if report.warning or report.status != "OK" or report.mag_status != "OK"]
    if warnings:
        lines.extend(["## Warnings And Incomplete Runs", ""])
        for report in warnings:
            lines.append(
                f"- `{report.run}`: status={report.status}, mag_status={report.mag_status}; "
                f"{report.warning or 'check run state'}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(
    reports: list[RunSpinReport],
    output_dir: Path,
    prefix: str,
    max_labels: int,
) -> list[Path]:
    usable = [report for report in reports if report.energy_eV is not None and report.atoms]
    if not usable:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    minimum = min(report.energy_eV for report in usable if report.energy_eV is not None)
    labels = [report.spin_index_name or report.resolved_run.name for report in usable]
    rel_e = [(report.energy_eV or 0.0) - minimum for report in usable]
    guard_applied = any(report.physics_guard_status not in {"", "NOT_APPLIED"} for report in usable)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def scatter(x_values: list[float], xlabel: str, name: str) -> None:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        if guard_applied:
            accepted = [idx for idx, report in enumerate(usable) if report.physics_guard_status == "OK"]
            rejected = [idx for idx, report in enumerate(usable) if report.physics_guard_status != "OK"]
            if rejected:
                ax.scatter(
                    [x_values[idx] for idx in rejected],
                    [rel_e[idx] for idx in rejected],
                    s=28,
                    c="0.72",
                    alpha=0.55,
                    label="physics rejected",
                )
            if accepted:
                ax.scatter(
                    [x_values[idx] for idx in accepted],
                    [rel_e[idx] for idx in accepted],
                    s=42,
                    c="#111111",
                    alpha=0.95,
                    label="physics accepted",
                )
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.scatter(x_values, rel_e, s=36)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("relative energy (eV)")
        ax.grid(True, alpha=0.3)
        if len(usable) <= max_labels:
            for x, y, label in zip(x_values, rel_e, labels):
                ax.annotate(label, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        fig.tight_layout()
        path = output_dir / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    scatter([float(report.total_moment or 0.0) for report in usable], "total magnetic moment", "energy_vs_total_moment")
    scatter([float(report.max_abs_moment or 0.0) for report in usable], "max |site moment|", "energy_vs_max_abs_moment")

    elements = sorted({atom.element for report in usable for atom in report.atoms})
    for element in elements:
        values = [float(report.element_sum.get(element, 0.0)) for report in usable]
        scatter(values, f"{element} summed moment", f"energy_vs_{element}_moment")
    return paths


def _short(text: str, width: int) -> str:
    if len(text) <= width:
        return text.ljust(width)
    if width <= 1:
        return text[:width]
    return (text[: width - 1] + "~").ljust(width)


def _fmt_float(value: float | None, width: int, precision: int = 3) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:.{precision}f}".rjust(width)


def _compact_count_map(values: dict[str, int], empty: str = "-") -> str:
    parts = [f"{key}:{values[key]}" for key in sorted(values) if values[key]]
    return ",".join(parts) if parts else empty


def _ordered_spin_elements(report: RunSpinReport) -> list[str]:
    elements = list(dict.fromkeys([*report.initial_element_order, *report.element_order]))

    def is_magnetic(element: str) -> bool:
        labels = [report.initial_element_order.get(element), report.element_order.get(element)]
        return any(label not in {None, "nonmagnetic"} for label in labels)

    magnetic = [element for element in elements if is_magnetic(element)]
    nonmagnetic = [element for element in elements if element not in magnetic]
    return magnetic + nonmagnetic


def compact_order_shift(report: RunSpinReport) -> str:
    parts: list[str] = []
    for element in _ordered_spin_elements(report):
        initial = report.initial_element_order.get(element)
        final = report.element_order.get(element)
        if final is None:
            continue
        if initial is not None and initial != final:
            parts.append(f"{element}:{initial}>{final}")
        else:
            parts.append(f"{element}:{final}")
    return ",".join(parts) if parts else "-"


def compact_spin_note(report: RunSpinReport) -> str:
    if report.physics_guard_bad_by_element:
        return f"bad={_compact_count_map(report.physics_guard_bad_by_element)}"
    if report.warning:
        return report.warning
    if report.status not in {"OK", "STOPPED"}:
        return report.status
    return "ok"


def print_batch_table(reports: list[RunSpinReport]) -> None:
    usable_energies = [report.energy_eV for report in reports if report.energy_eV is not None]
    best_energy = min(usable_energies) if usable_energies else None
    print()
    print("Atomi VASP Spin Report")
    print(
        "run  path                    status    energy      relE     total_m   mag       guard     chg       order             note"
    )
    print(
        "---  ----------------------  --------  ----------  -------  --------  --------  --------  --------  ----------------  ----------------"
    )
    for report in reports:
        relative = None
        if report.energy_eV is not None and best_energy is not None:
            relative = report.energy_eV - best_energy
        guard = report.physics_guard_status
        if guard in {"", "NOT_APPLIED"}:
            guard = report.mag_status
        print(
            f"{str(report.index).rjust(3)}  "
            f"{_short(Path(report.run).name or str(report.run), 22)}  "
            f"{_short(report.status, 8)}  "
            f"{_fmt_float(report.energy_eV, 10, 4)}  "
            f"{_fmt_float(relative, 7, 3)}  "
            f"{_fmt_float(report.total_moment, 8, 3)}  "
            f"{_short(report.mag_status, 8)}  "
            f"{_short(guard, 8)}  "
            f"{_short(_compact_count_map(report.changed_by_element), 8)}  "
            f"{_short(compact_order_shift(report), 16)}  "
            f"{_short(compact_spin_note(report), 16)}"
        )


def output_paths(prefix: Path) -> dict[str, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return {
        "summary": prefix.with_name(prefix.name + "_run_summary.csv"),
        "atoms": prefix.with_name(prefix.name + "_atom_moments.csv"),
        "filtered_summary": prefix.with_name(prefix.name + "_physics_filtered_run_summary.csv"),
        "filtered_atoms": prefix.with_name(prefix.name + "_physics_filtered_atom_moments.csv"),
        "report": prefix.with_name(prefix.name + "_report.md"),
        "plots": prefix.with_name(prefix.name + "_plots"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-spin-report",
        description="Extract final/last VASP magnetic moments and correlate them with energies.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--outcar", type=Path, help="Single OUTCAR to extract.")
    source.add_argument("--runlist", type=Path, help="Runlist for batch spin-energy report.")
    parser.add_argument("--output-prefix", type=Path, default=Path("spin_report"))
    parser.add_argument("--natoms", type=int, help="Expected atom count. Defaults to OUTCAR NIONS or POSCAR.")
    parser.add_argument("--species", type=Path, help="POSCAR/CONTCAR for atom species labels.")
    parser.add_argument("--incar", type=Path, help="INCAR for initial MAGMOM labels and automatic spin guards.")
    parser.add_argument("--format", choices=("expanded", "vasp", "both"), default="both")
    parser.add_argument("--spin-index", type=Path, help="Optional magit spin_index.csv.")
    parser.add_argument("--log-dir", type=Path, help="Directory containing array logs for energy fallback.")
    parser.add_argument(
        "--energy",
        choices=("toten", "without_entropy", "e0", "f", "dav"),
        default="toten",
        help="Preferred energy for batch mode. Falls back like checkeng.",
    )
    parser.add_argument("--stopped-after-min", type=float, default=15.0)
    parser.add_argument("--dav-average-window", type=int, default=10)
    parser.add_argument("--change-threshold", type=float, default=0.25)
    parser.add_argument("--order-threshold", type=float, default=0.2)
    parser.add_argument(
        "--moment-guard",
        action="append",
        default=[],
        help="Physics guard for final moments, repeatable: Element=v1,v2 or Element=v1,v2@tol.",
    )
    parser.add_argument("--moment-guard-tol", type=float, default=0.6, help="Default tolerance for --moment-guard.")
    parser.add_argument(
        "--no-auto-moment-guard",
        action="store_true",
        help="Disable default physics guards inferred from POSCAR/CONTCAR plus INCAR MAGMOM.",
    )
    parser.add_argument(
        "--auto-moment-integer-tol",
        type=float,
        default=0.35,
        help="Treat initial MAGMOM values within this distance of an integer as that integer.",
    )
    parser.add_argument(
        "--auto-zero-tol",
        type=float,
        default=0.25,
        help="Tolerance for automatically inferred near-zero/nonmagnetic elements.",
    )
    parser.add_argument("--compress-tol", type=float, default=0.05)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--max-plot-labels", type=int, default=50)
    return parser


def prepare_batch_moment_guards(args: argparse.Namespace, runlist: Path) -> dict[str, tuple[list[float], float]]:
    inferred: dict[str, tuple[list[float], float]] = {}
    if not args.no_auto_moment_guard:
        inferred = infer_moment_guards_from_run_dirs(
            runlist_dirs(runlist),
            default_tol=args.moment_guard_tol,
            species_override=args.species,
            incar_override=args.incar,
            integer_tol=args.auto_moment_integer_tol,
            zero_tol=args.auto_zero_tol,
        )
    explicit = parse_moment_guards(args.moment_guard, args.moment_guard_tol)
    return merge_moment_guards(inferred, explicit)


def prepare_single_moment_guards(
    args: argparse.Namespace,
    species_file: Path | None,
    incar: Path | None,
) -> dict[str, tuple[list[float], float]]:
    inferred: dict[str, tuple[list[float], float]] = {}
    if not args.no_auto_moment_guard:
        inferred = infer_moment_guards_from_files(
            species_file,
            incar,
            default_tol=args.moment_guard_tol,
            integer_tol=args.auto_moment_integer_tol,
            zero_tol=args.auto_zero_tol,
        )
    explicit = parse_moment_guards(args.moment_guard, args.moment_guard_tol)
    return merge_moment_guards(inferred, explicit)


def run_single(args: argparse.Namespace) -> None:
    block = extract_last_magnetization_block(args.outcar, natoms=args.natoms)
    species_file = args.species or default_species_file(args.outcar.parent)
    labels, species, warning = read_species_labels(species_file, args.natoms or len(block.rows))
    if warning:
        print(f"Warning            : {warning}")
    write_single_outputs(
        outcar=args.outcar,
        output_prefix=args.output_prefix,
        block=block,
        labels=labels,
        formats=args.format,
        decimals=args.decimals,
        compress_tol=args.compress_tol,
    )
    print_single_summary(block, labels)
    incar = args.incar or default_incar(args.outcar.parent)
    moment_guards = prepare_single_moment_guards(args, species_file, incar)
    if moment_guards:
        initial = None
        if incar is not None and species is not None:
            initial = existing_magmom_values(incar, species.total_atoms)
        atoms = build_atom_reports(block.moments, labels, initial, args.change_threshold)
        status, bad_count, bad_by_element = apply_moment_guards(atoms, moment_guards)
        print(f"Physics guard      : {guard_rule_text(moment_guards)}")
        print(f"Physics status     : {status}  bad_sites={bad_count}  by_element={bad_by_element or {}}")
    else:
        print(f"Physics guard      : {auto_moment_guard_notice(moment_guards, not args.no_auto_moment_guard)}")
    print(f"Wrote prefix       : {args.output_prefix}")


def run_batch(args: argparse.Namespace) -> None:
    runlist = args.runlist or Path("runlist.txt")
    moment_guards = prepare_batch_moment_guards(args, runlist)
    reports = build_run_reports(
        runlist=runlist,
        spin_index=args.spin_index,
        log_dir=args.log_dir,
        energy_kind=args.energy,
        stopped_after_minutes=args.stopped_after_min,
        dav_average_window=args.dav_average_window,
        species_override=args.species,
        natoms=args.natoms,
        change_threshold=args.change_threshold,
        order_threshold=args.order_threshold,
        moment_guards=moment_guards,
    )
    paths = output_paths(args.output_prefix)
    write_run_summary(reports, paths["summary"])
    write_atom_table(reports, paths["atoms"])
    if moment_guards:
        write_physics_filtered_tables(reports, paths)
    write_markdown_report(reports, paths["report"])
    write_magmom_lines(reports, args.output_prefix.parent, args.decimals, args.compress_tol)
    plot_paths: list[Path] = []
    if not args.no_plot:
        plot_paths = write_plots(reports, paths["plots"], args.output_prefix.name, args.max_plot_labels)
    ok = sum(1 for report in reports if report.energy_eV is not None and report.atoms)
    print_batch_table(reports)
    print()
    print(f"Runlist rows       : {len(reports)}")
    print(f"Energy+moments rows: {ok}")
    print(f"Energy status      : {dict(Counter(report.status for report in reports))}")
    print(f"Moment status      : {dict(Counter(report.mag_status for report in reports))}")
    print(f"Summary CSV        : {paths['summary']}")
    print(f"Atom CSV           : {paths['atoms']}")
    if moment_guards:
        print(f"Physics guard      : {guard_rule_text(moment_guards)}")
        print(f"Physics accepted   : {sum(1 for report in reports if report.physics_guard_status == 'OK')}")
        print(f"Physics rejected   : {sum(1 for report in reports if report.physics_guard_status == 'FAIL')}")
        print(f"Filtered summary   : {paths['filtered_summary']}")
        print(f"Filtered atom CSV  : {paths['filtered_atoms']}")
    print(f"Markdown report    : {paths['report']}")
    print(f"MAGMOM lines       : {args.output_prefix.parent / 'magmom_lines'}")
    if args.no_plot:
        print("Plots              : skipped")
    elif plot_paths:
        print(f"Plots              : {paths['plots']}")
    else:
        print("Plots              : not written (matplotlib unavailable or no usable rows)")
    if ok == 0 and reports:
        print("Hint               : check runlist root, --log-dir, and whether stopped array artifacts contain OUTCAR.")


def default_single_outcar(root: Path = Path(".")) -> Path | None:
    for name in ("OUTCAR", "OUTCAR.gz"):
        path = root / name
        if path.is_file():
            return path
    return None


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.outcar is None and args.runlist is None:
        default_runlist = Path("runlist.txt")
        default_outcar = default_single_outcar()
        if default_runlist.is_file():
            args.runlist = default_runlist
        elif default_outcar is not None:
            args.outcar = default_outcar
    if args.outcar is not None:
        run_single(args)
        return
    run_batch(args)


if __name__ == "__main__":
    main(sys.argv[1:])
