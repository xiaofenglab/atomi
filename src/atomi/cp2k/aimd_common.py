"""Shared parsers for CP2K AIMD copilot tools."""

from __future__ import annotations

import gzip
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


def strip_inline_comment(line: str) -> str:
    for marker in ("#", "!"):
        if marker in line:
            line = line.split(marker, 1)[0]
    return line.rstrip()


def _as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _text_lines(path: Path):
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
    else:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            yield from handle


def parse_cp2k_input(path: Path) -> dict[str, object]:
    """Parse reproducibility and restraint metadata from a CP2K input."""
    info: dict[str, object] = {
        "project": None,
        "run_type": None,
        "charge": None,
        "multiplicity": None,
        "ensemble": None,
        "steps": None,
        "timestep_fs": None,
        "temperature_K": None,
        "cell_abc_A": None,
        "coordinate_file": None,
        "trajectory_file": None,
        "trajectory_stride": None,
        "colvars": [],
        "restraints": [],
        "basis_set_file": None,
        "potential_file": None,
        "xc_functional": None,
        "vdw_type": None,
    }

    section_stack: list[str] = []
    current_colvar: dict[str, object] | None = None
    current_collective: dict[str, object] | None = None
    in_trajectory_each = False

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = strip_inline_comment(raw).strip()
        if not line:
            continue
        upper = line.upper()

        if upper.startswith("&END"):
            end_name = upper.split(maxsplit=1)[1].split()[0] if len(upper.split()) > 1 else ""
            if end_name == "COLVAR" and current_colvar is not None:
                current_colvar.setdefault("index", len(info["colvars"]) + 1)  # type: ignore[index]
                info["colvars"].append(current_colvar)  # type: ignore[union-attr]
                current_colvar = None
            if end_name == "COLLECTIVE" and current_collective is not None:
                info["restraints"].append(current_collective)  # type: ignore[union-attr]
                current_collective = None
            if end_name == "EACH" and in_trajectory_each:
                in_trajectory_each = False
            if section_stack:
                section_stack.pop()
            continue

        if upper.startswith("&"):
            section = upper[1:].split()[0]
            section_stack.append(section)
            if section == "COLVAR":
                current_colvar = {"kind": None, "atoms": []}
            elif section == "COLLECTIVE":
                current_collective = {}
            elif section == "EACH" and "TRAJECTORY" in section_stack:
                in_trajectory_each = True
            continue

        def match(pattern: str) -> re.Match[str] | None:
            return re.match(pattern, line, flags=re.IGNORECASE)

        if m := match(r"PROJECT\s+(.+)"):
            info["project"] = m.group(1).strip()
        elif m := match(r"RUN_TYPE\s+(\S+)"):
            info["run_type"] = m.group(1).strip()
        elif m := match(r"CHARGE\s+([-+]?\d+)"):
            info["charge"] = int(m.group(1))
        elif m := match(r"MULTIPLICITY\s+(\d+)"):
            info["multiplicity"] = int(m.group(1))
        elif m := match(r"BASIS_SET_FILE_NAME\s+(.+)"):
            info["basis_set_file"] = m.group(1).strip()
        elif m := match(r"POTENTIAL_FILE_NAME\s+(.+)"):
            info["potential_file"] = m.group(1).strip()
        elif m := match(r"ABC\s+([-+0-9.EedD]+)\s+([-+0-9.EedD]+)\s+([-+0-9.EedD]+)"):
            info["cell_abc_A"] = [_as_float(m.group(i)) for i in (1, 2, 3)]
        elif m := match(r"COORD_FILE_NAME\s+(.+)"):
            info["coordinate_file"] = m.group(1).strip()
        elif "MD" in section_stack and (m := match(r"ENSEMBLE\s+(\S+)")):
            info["ensemble"] = m.group(1).strip()
        elif "MD" in section_stack and (m := match(r"STEPS\s+(\d+)")):
            info["steps"] = int(m.group(1))
        elif "MD" in section_stack and (m := match(r"TIMESTEP\s+([-+0-9.EedD]+)")):
            info["timestep_fs"] = _as_float(m.group(1))
        elif "MD" in section_stack and (m := match(r"TEMPERATURE\s+([-+0-9.EedD]+)")):
            info["temperature_K"] = _as_float(m.group(1))
        elif "XC_FUNCTIONAL" in section_stack and line and not upper.startswith("&"):
            info["xc_functional"] = line.split()[0]
        elif "PAIR_POTENTIAL" in section_stack and (m := match(r"TYPE\s+(\S+)")):
            info["vdw_type"] = m.group(1).strip()
        elif current_colvar is not None and "DISTANCE" in section_stack and (m := match(r"ATOMS\s+(.+)")):
            current_colvar["kind"] = "distance"
            current_colvar["atoms"] = [int(item) for item in m.group(1).split()]
        elif current_collective is not None and (m := match(r"COLVAR\s+(\d+)")):
            current_collective["colvar"] = int(m.group(1))
        elif current_collective is not None and (m := match(r"TARGET(?:\s+\[.*?\])?\s+([-+0-9.EedD]+)")):
            current_collective["target_A"] = _as_float(m.group(1))
        elif current_collective is not None and "RESTRAINT" in section_stack and (
            m := match(r"K(?:\s+\[.*?\])?\s+([-+0-9.EedD]+)")
        ):
            current_collective["k_kcalmol"] = _as_float(m.group(1))
        elif "TRAJECTORY" in section_stack and (m := match(r"FILENAME\s*=?\s*(.+)")):
            info["trajectory_file"] = m.group(1).strip()
        elif in_trajectory_each and (m := match(r"MD\s+(\d+)")):
            info["trajectory_stride"] = int(m.group(1))

    return info


def parse_energy_file(path: Path) -> dict[str, object]:
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            rows.append([_as_float(item) for item in parts[:7]])
        except ValueError:
            continue

    if not rows:
        return {"record_count": 0}

    latest = rows[-1]
    used_times = [row[6] for row in rows[1:] if row[6] > 0.0]
    mean_step_seconds = sum(used_times) / len(used_times) if used_times else None
    return {
        "record_count": len(rows),
        "latest_step": int(round(latest[0])),
        "latest_time_fs": latest[1],
        "latest_temperature_K": latest[3],
        "latest_potential_au": latest[4],
        "latest_conserved_au": latest[5],
        "mean_step_seconds": mean_step_seconds,
    }


def scan_cp2k_log(path: Path) -> dict[str, object]:
    failure_patterns = re.compile(r"SCF run NOT converged|ERROR|ABORT", re.IGNORECASE)
    step_pattern = re.compile(r"\bMD\|\s*Step number\s+(\d+)", re.IGNORECASE)
    energy_pattern = re.compile(r"ENERGY\|.*?energy.*?:\s*([-+0-9.EedD]+)", re.IGNORECASE)
    warning_pattern = re.compile(r"\bWARNING\b", re.IGNORECASE)
    result = {
        "failure_count": 0,
        "warning_count": 0,
        "finished": False,
        "last_step": None,
        "energy_records": 0,
        "final_energy_au": None,
    }
    for raw in _text_lines(path):
        if failure_patterns.search(raw):
            result["failure_count"] += 1
        if warning_pattern.search(raw):
            result["warning_count"] += 1
        if "PROGRAM ENDED" in raw or "CP2K| run finished" in raw:
            result["finished"] = True
        if m := step_pattern.search(raw):
            result["last_step"] = int(m.group(1))
        if m := energy_pattern.search(raw):
            result["energy_records"] += 1
            result["final_energy_au"] = _as_float(m.group(1))
    return result


def iter_xyz_frames(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                natoms = int(line)
            except ValueError as exc:
                raise ValueError(f"{path}: expected atom count, got {line!r}") from exc
            comment = handle.readline().rstrip("\n")
            symbols: list[str] = []
            coords: list[tuple[float, float, float]] = []
            for _ in range(natoms):
                parts = handle.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"{path}: malformed XYZ atom line")
                symbols.append(parts[0])
                coords.append((_as_float(parts[1]), _as_float(parts[2]), _as_float(parts[3])))
            yield comment, symbols, coords


def xyz_frame_summary(path: Path, *, max_frames: int | None = None) -> dict[str, object]:
    count = 0
    atom_count = None
    symbols_counter: Counter[str] = Counter()
    last_comment = ""
    truncated = False
    for comment, symbols, _coords in iter_xyz_frames(path):
        count += 1
        atom_count = len(symbols)
        symbols_counter = Counter(symbols)
        last_comment = comment
        if max_frames is not None and count >= max_frames:
            truncated = True
            break
    return {
        "frame_count": count,
        "frame_count_truncated": truncated,
        "frame_count_limit": max_frames,
        "atom_count": atom_count,
        "composition": dict(symbols_counter),
        "last_comment": last_comment,
    }


def parse_step_from_comment(comment: str, fallback: int) -> int:
    for pattern in (
        r"\bi\s*=\s*([0-9]+)\b",
        r"\bstep\s*=\s*([0-9]+)\b",
        r"\bSTEP\s*=?\s*([0-9]+)\b",
    ):
        if m := re.search(pattern, comment, flags=re.IGNORECASE):
            return int(m.group(1))
    return fallback


def mic_delta(delta: float, length: float | None) -> float:
    if length is None or length <= 0:
        return delta
    return delta - length * round(delta / length)


def distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    cell_abc_A: Iterable[float] | None = None,
) -> float:
    cell = list(cell_abc_A or [])
    total = 0.0
    for idx in range(3):
        length = cell[idx] if idx < len(cell) else None
        delta = mic_delta(a[idx] - b[idx], length)
        total += delta * delta
    return math.sqrt(total)


def tail_values(values: list[float], fraction: float) -> list[float]:
    if not values:
        return []
    tail_n = max(1, int(math.ceil(len(values) * fraction)))
    return values[-tail_n:]


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def std(values: Iterable[float]) -> float | None:
    vals = list(values)
    if len(vals) < 2:
        return None
    avg = sum(vals) / len(vals)
    return math.sqrt(sum((value - avg) ** 2 for value in vals) / (len(vals) - 1))


def parse_lagrange_file(path: Path) -> dict[str, object]:
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Lagrangian Multipliers" not in raw:
            continue
        try:
            values = [_as_float(item) for item in raw.split(":", 1)[1].split()]
        except (IndexError, ValueError):
            continue
        if values:
            rows.append(values)
    if not rows:
        return {"record_count": 0, "column_means": []}
    width = max(len(row) for row in rows)
    means = []
    stdevs = []
    for idx in range(width):
        column = [row[idx] for row in rows if idx < len(row)]
        means.append(mean(column))
        stdevs.append(std(column))
    return {"record_count": len(rows), "column_means": means, "column_std": stdevs}


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None
