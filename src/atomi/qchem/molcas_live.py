"""Live, input-scoped monitoring for OpenMolcas production calculations.

``moclive`` treats the input deck as the authoritative RASSCF/CASPT2/RASSI
schedule and maps chronological output modules onto that plan.  The monitor is
deliberately conservative: numerical completion can pass automatically, while
orbital identity remains a review item unless the output contains direct
evidence that can be checked without guessing the intended chemistry.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO


SCHEMA = "atomi.molcas_live.v1"
MONITORED_KINDS = ("rasscf", "caspt2", "rassi")
MODULE_START_RE = re.compile(r"--- Start Module:\s*(rasscf|caspt2|rassi)\b", re.IGNORECASE)
MODULE_STOP_RE = re.compile(
    r"--- Stop Module:\s*(rasscf|caspt2|rassi)\b.*?/rc=([^\s]+)", re.IGNORECASE
)
MODULE_SPENT_RE = re.compile(r"--- Module\s+(rasscf|caspt2|rassi)\s+spent\b", re.IGNORECASE)
RASSCF_ROOT_RE = re.compile(r"RASSCF root number\s+(\d+)\b", re.IGNORECASE)
RASSCF_ROOT_REQUEST_RE = re.compile(r"Number of root\(s\) required\s+(\d+)\b", re.IGNORECASE)
CASPT2_ROOT_RE = re.compile(
    r"(?:XMS-|RMS-|MS-)?CASPT2\s+Root\s+(\d+)\s+Total energy", re.IGNORECASE
)
REFERENCE_WEIGHT_RE = re.compile(
    r"Reference weight:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)",
    re.IGNORECASE,
)
SO_RASSI_STATE_RE = re.compile(r"SO-RASSI State\s+(\d+)\b", re.IGNORECASE)
RASSCF_CONVERGENCE_RE = re.compile(r"Convergence after\s+(\d+)\s+iterations?", re.IGNORECASE)
COMPACT_MO_ROW_RE = re.compile(
    r"^\s*\d+\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)\b"
)
FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?")
ERROR_PATTERNS = (
    re.compile(r"_RC_(?!ALL_IS_WELL_)[A-Z0-9_]+_"),
    re.compile(r"\b(?:did not converge|not converged|convergence problem)\b", re.I),
    re.compile(r"\b(?:segmentation fault|out of memory|oom-kill|fatal error|aborted)\b", re.I),
    re.compile(r"\bRASSI\b.*\b(?:incompatible|mismatch)\b", re.I),
)
SUPSYM_WARNING_RE = re.compile(r"supersymmetr(?:y|ies).*?(?:warn|violat|break|mix)", re.I)
SPINNER_UNICODE = (
    "\u280b",
    "\u2819",
    "\u2839",
    "\u2838",
    "\u283c",
    "\u2834",
    "\u2826",
    "\u2827",
    "\u2807",
    "\u280f",
)
SPINNER_ASCII = ("|", "/", "-", "\\")


@dataclass(frozen=True)
class PlanBlock:
    index: int
    kind: str
    kind_index: int
    start_line: int
    end_line: int
    title: str = ""
    symmetry: int | None = None
    spin_multiplicity: int | None = None
    active_electrons: int | None = None
    expected_roots: int | None = None
    rassi_job_count: int | None = None
    rassi_state_counts: tuple[int, ...] = ()
    rassi_all_states: bool = False
    has_alter: bool = False
    has_supsym: bool = False
    has_cionly: bool = False
    has_tdm: bool = False


@dataclass
class RuntimeBlock:
    kind: str
    kind_index: int
    start_line: int
    end_line: int | None = None
    return_code: str | None = None
    saw_spent: bool = False
    roots: set[int] = field(default_factory=set)
    roots_requested: int | None = None
    active_electrons: int | None = None
    spin_quantum_number: float | None = None
    state_symmetry: int | None = None
    convergence_iterations: int | None = None
    caspt2_reference_weights: list[float] = field(default_factory=list)
    caspt2_selected_root: int | None = None
    so_states: set[int] = field(default_factory=set)
    rassi_so_table: bool = False
    rassi_transition_tables: int = 0
    stage: str = "initializing"
    mo_occupations: list[float] = field(default_factory=list)
    rasorb_roots: set[int] = field(default_factory=set)
    supersymmetry_warnings: int = 0
    error_lines: list[str] = field(default_factory=list)


@dataclass
class OutputParserState:
    blocks: list[RuntimeBlock] = field(default_factory=list)
    kind_counts: dict[str, int] = field(
        default_factory=lambda: {kind: 0 for kind in MONITORED_KINDS}
    )
    current_index: int | None = None
    line_count: int = 0
    happy_landing: bool = False
    global_errors: list[str] = field(default_factory=list)
    in_mo_section: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "line_count": self.line_count,
            "happy_landing": self.happy_landing,
            "global_errors": list(self.global_errors),
        }


def _clean_line(line: str) -> str:
    return line.strip()


def _next_value(lines: list[str], start: int, end: int) -> str | None:
    for idx in range(start, end):
        value = lines[idx].strip()
        if value and not value.startswith("*"):
            return value
    return None


def _keyword_value(lines: list[str], keyword: str) -> str | None:
    wanted = keyword.lower().replace(" ", "")
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            continue
        token, _, remainder = stripped.partition(" ")
        token = token.rstrip("=:").lower().replace(" ", "")
        if token != wanted:
            continue
        payload = remainder.strip().lstrip("=:").strip()
        if payload:
            return payload
        return _next_value(lines, idx + 1, len(lines))
    return None


def _first_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+", value)
    return int(match.group()) if match else None


def _block_title(lines: list[str]) -> str:
    value = _keyword_value(lines, "title")
    return value.strip() if value else ""


def _caspt2_roots(lines: list[str]) -> tuple[int | None, bool]:
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        match = re.match(r"^(?:X?MUL(?:T(?:ISTATE)?)?)\b\s*(.*)$", stripped, re.I)
        if not match:
            continue
        value = (
            match.group(1).strip().lstrip("=:").strip()
            or _next_value(lines, idx + 1, len(lines))
            or ""
        )
        if value.upper().startswith("ALL"):
            return None, True
        return _first_int(value), False
    return None, False


def _rassi_counts(lines: list[str]) -> tuple[int | None, tuple[int, ...], bool]:
    for idx, raw in enumerate(lines):
        if not re.match(r"^\s*(?:NROFJOBIPHS\b|Nr of JobIph files\s*:)", raw, re.I):
            continue
        value = _next_value(lines, idx + 1, len(lines)) or ""
        values = [int(item) for item in re.findall(r"\d+", value)]
        if not values:
            return None, (), "all" in value.lower()
        jobs = values[0]
        counts = tuple(values[1 : 1 + jobs])
        return jobs, counts, "all" in value.lower()
    return None, (), False


def parse_input_plan(text: str) -> list[PlanBlock]:
    """Return chronological RASSCF/CASPT2/RASSI blocks from an input deck."""

    lines = text.splitlines()
    raw_blocks: list[tuple[str, int, int, list[str]]] = []
    idx = 0
    while idx < len(lines):
        match = re.match(r"^\s*&\s*(RASSCF|CASPT2|RASSI)\b", lines[idx], re.I)
        if not match:
            idx += 1
            continue
        kind = match.group(1).lower()
        end = idx + 1
        while end < len(lines):
            if re.match(r"^\s*&\s*[A-Za-z0-9_]+\b", lines[end]):
                break
            if lines[end].strip().lower() == "end of input":
                end += 1
                break
            end += 1
        raw_blocks.append((kind, idx + 1, end, lines[idx:end]))
        idx = max(end, idx + 1)

    kind_counts = {kind: 0 for kind in MONITORED_KINDS}
    plan: list[PlanBlock] = []
    previous_rasscf_roots: int | None = None
    preceding_caspt2_roots: list[int] = []
    for index, (kind, start, end, body) in enumerate(raw_blocks, start=1):
        kind_counts[kind] += 1
        tokens = {
            re.split(r"[\s=:]+", line.strip(), maxsplit=1)[0].lower()
            for line in body
            if line.strip() and not line.lstrip().startswith("*")
        }
        expected: int | None = None
        jobs: int | None = None
        counts: tuple[int, ...] = ()
        all_states = False
        if kind == "rasscf":
            expected = _first_int(_keyword_value(body, "ciroots")) or 1
            previous_rasscf_roots = expected
        elif kind == "caspt2":
            expected, all_states = _caspt2_roots(body)
            if all_states and expected is None:
                expected = previous_rasscf_roots
            if expected is not None:
                preceding_caspt2_roots.append(expected)
        else:
            jobs, counts, all_states = _rassi_counts(body)
            if counts:
                expected = sum(counts)
            elif all_states and jobs:
                candidates = preceding_caspt2_roots[-jobs:]
                if len(candidates) == jobs:
                    expected = sum(candidates)
        plan.append(
            PlanBlock(
                index=index,
                kind=kind,
                kind_index=kind_counts[kind],
                start_line=start,
                end_line=end,
                title=_block_title(body),
                symmetry=_first_int(_keyword_value(body, "symmetry")) if kind == "rasscf" else None,
                spin_multiplicity=_first_int(_keyword_value(body, "spin"))
                if kind == "rasscf"
                else None,
                active_electrons=_first_int(_keyword_value(body, "nactel"))
                if kind == "rasscf"
                else None,
                expected_roots=expected,
                rassi_job_count=jobs,
                rassi_state_counts=counts,
                rassi_all_states=all_states,
                has_alter="alter" in tokens,
                has_supsym="supsym" in tokens,
                has_cionly="cionly" in tokens,
                has_tdm="tdm" in tokens,
            )
        )
    return plan


def _open_output(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _record_error(block: RuntimeBlock, line: str) -> None:
    if any(pattern.search(line) for pattern in ERROR_PATTERNS):
        value = _clean_line(line)
        if value and value not in block.error_lines and len(block.error_lines) < 8:
            block.error_lines.append(value)


def _update_stage(block: RuntimeBlock, line: str) -> None:
    lower = line.lower()
    if block.kind == "rasscf":
        if "rasscf iterations: energy and convergence statistics" in lower:
            block.stage = "optimizing orbitals / CI"
        elif "convergence after" in lower:
            block.stage = "writing root and orbital results"
        elif "pseudonatural active orbitals" in lower:
            block.stage = "auditing pseudonatural orbitals"
    elif block.kind == "caspt2":
        if "reference weight" in lower:
            block.stage = "evaluating CASPT2 roots"
        elif "total energy" in lower and "caspt2" in lower:
            block.stage = "collecting multistate energies"
    else:
        if "hamiltonian" in lower:
            block.stage = "building / diagonalizing SO Hamiltonian"
        elif "so state  total energy" in lower:
            block.stage = "writing SO-state composition table"
            block.rassi_so_table = True
        elif "so-rassi state" in lower:
            block.stage = "writing spin-orbit states"
        elif "dipole transition strengths" in lower:
            block.stage = "writing dipole transitions"
            block.rassi_transition_tables += 1
        elif "velocity transition strengths" in lower:
            block.stage = "writing velocity-gauge transitions"
            block.rassi_transition_tables += 1
        elif "gauge comparison" in lower:
            block.stage = "checking length / velocity gauges"


def _ingest_output_lines(lines: Iterable[str], state: OutputParserState) -> None:
    """Append complete output lines to a persistent parser state."""

    for line in lines:
        state.line_count += 1
        line_count = state.line_count
        current = state.blocks[state.current_index] if state.current_index is not None else None
        if "Happy landing" in line:
            state.happy_landing = True
        start = MODULE_START_RE.search(line)
        if start:
            kind = start.group(1).lower()
            if current is not None and current.end_line is None:
                current.end_line = line_count - 1
                current.error_lines.append("A later monitored module started before a stop marker")
            state.kind_counts[kind] += 1
            current = RuntimeBlock(
                kind=kind,
                kind_index=state.kind_counts[kind],
                start_line=line_count,
            )
            state.blocks.append(current)
            state.current_index = len(state.blocks) - 1
            state.in_mo_section = False
            continue
        if current is None:
            if any(pattern.search(line) for pattern in ERROR_PATTERNS):
                value = _clean_line(line)
                if value and value not in state.global_errors and len(state.global_errors) < 8:
                    state.global_errors.append(value)
            continue

        _record_error(current, line)
        _update_stage(current, line)
        if SUPSYM_WARNING_RE.search(line):
            current.supersymmetry_warnings += 1

        if current.kind == "rasscf":
            match = RASSCF_ROOT_REQUEST_RE.search(line)
            if match:
                current.roots_requested = int(match.group(1))
            match = RASSCF_ROOT_RE.search(line)
            if match:
                current.roots.add(int(match.group(1)))
            match = RASSCF_CONVERGENCE_RE.search(line)
            if match:
                current.convergence_iterations = int(match.group(1))
            match = re.search(r"Number of electrons in active shells\s+(\d+)", line, re.I)
            if match:
                current.active_electrons = int(match.group(1))
            match = re.search(r"Spin quantum number\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+))", line, re.I)
            if match:
                current.spin_quantum_number = float(match.group(1))
            match = re.search(r"State symmetry\s+(\d+)", line, re.I)
            if match:
                current.state_symmetry = int(match.group(1))
            match = re.search(r"Natural orbitals for root\s+(\d+)\s+are written", line, re.I)
            if match:
                current.rasorb_roots.add(int(match.group(1)))
            if "pseudonatural active orbitals" in line.lower():
                state.in_mo_section = True
            if state.in_mo_section and re.match(r"^\s*Occ\.\s*No\.", line, re.I):
                payload = re.sub(r"^\s*Occ\.\s*No\.\s*", "", line, flags=re.I)
                current.mo_occupations.extend(_float(value) for value in FLOAT_RE.findall(payload))
            elif state.in_mo_section:
                compact = COMPACT_MO_ROW_RE.match(line)
                if compact:
                    current.mo_occupations.append(_float(compact.group(1)))
        elif current.kind == "caspt2":
            match = CASPT2_ROOT_RE.search(line)
            if match:
                current.roots.add(int(match.group(1)))
            match = re.search(r"CASPT2 state passed to geometry opt\.\s+(\d+)", line, re.I)
            if match:
                current.caspt2_selected_root = int(match.group(1))
            if "FINAL CASPT2 RESULT:" in line.upper():
                current.roots.add(current.caspt2_selected_root or 1)
            match = REFERENCE_WEIGHT_RE.search(line)
            if match:
                current.caspt2_reference_weights.append(_float(match.group(1)))
        else:
            match = SO_RASSI_STATE_RE.search(line)
            if match:
                current.so_states.add(int(match.group(1)))

        stop = MODULE_STOP_RE.search(line)
        if stop and stop.group(1).lower() == current.kind:
            current.return_code = stop.group(2)
            current.end_line = line_count
            if current.return_code != "_RC_ALL_IS_WELL_":
                current.stage = "failed"
            elif current.stage == "initializing":
                current.stage = "finished"
            state.in_mo_section = False
            continue
        spent = MODULE_SPENT_RE.search(line)
        if spent and spent.group(1).lower() == current.kind:
            current.saw_spent = True
            if current.end_line is None:
                current.end_line = line_count
            if current.return_code is None and not current.error_lines:
                current.stage = "finished"
            state.in_mo_section = False


def parse_output_stream(lines: Iterable[str]) -> tuple[list[RuntimeBlock], dict[str, Any]]:
    """Scan output once with bounded memory and return monitored modules."""

    state = OutputParserState()
    _ingest_output_lines(lines, state)
    return state.blocks, state.metadata()


def _runtime_status(block: RuntimeBlock) -> str:
    if block.return_code and block.return_code != "_RC_ALL_IS_WELL_":
        return "failed"
    if block.error_lines and block.return_code != "_RC_ALL_IS_WELL_":
        return "failed"
    if block.return_code == "_RC_ALL_IS_WELL_" or block.saw_spent:
        return "finished"
    return "running"


def _guard(level: str, name: str, detail: str) -> dict[str, str]:
    return {"level": level, "name": name, "detail": detail}


def _block_guards(
    plan: PlanBlock, runtime: RuntimeBlock | None, status: str
) -> list[dict[str, str]]:
    if runtime is None:
        return [_guard("pending", "numerical status", "module has not started")]
    guards: list[dict[str, str]] = []
    if status == "failed":
        detail = runtime.return_code or (
            runtime.error_lines[0] if runtime.error_lines else "module failed"
        )
        guards.append(_guard("fail", "numerical status", detail))
    elif status == "finished":
        guards.append(
            _guard("pass", "numerical status", runtime.return_code or "spent marker found")
        )
    else:
        guards.append(_guard("pending", "numerical status", runtime.stage))

    total = runtime.roots_requested or plan.expected_roots
    if plan.kind in {"rasscf", "caspt2"} and total:
        done = len(runtime.roots)
        if plan.kind == "caspt2":
            # CASPT2 emits one reference weight per evaluated root before the
            # numbered XMS/RMS energy table appears at the end of large jobs.
            done = max(done, len(runtime.caspt2_reference_weights))
        if status == "finished" and done >= total:
            level = "pass"
            detail = f"{done}/{total} root energies found"
        elif status == "finished":
            # Large-root OpenMolcas jobs can truncate printed root-energy
            # tables even though the module returns normally and writes the
            # complete JobIph state space for later modules. Treat the
            # listing shortfall as an audit warning, not a numerical failure.
            level = "warn"
            detail = (
                f"{done}/{total} root energies printed; module returned successfully, "
                "so verify JobIph/RASSI state coverage"
            )
        elif status == "failed":
            level = "fail"
            detail = f"{done}/{total} root energies found before failure"
        else:
            level = "pending"
            detail = f"{done}/{total} root energies found"
        guards.append(_guard(level, "root coverage", detail))

    if plan.kind == "rasscf":
        expected_spin = (
            (plan.spin_multiplicity - 1) / 2.0 if plan.spin_multiplicity is not None else None
        )
        mismatches = []
        if (
            plan.symmetry is not None
            and runtime.state_symmetry is not None
            and plan.symmetry != runtime.state_symmetry
        ):
            mismatches.append(f"symmetry input={plan.symmetry}, output={runtime.state_symmetry}")
        if (
            expected_spin is not None
            and runtime.spin_quantum_number is not None
            and abs(expected_spin - runtime.spin_quantum_number) > 1.0e-6
        ):
            mismatches.append(
                f"spin input S={expected_spin:g}, output S={runtime.spin_quantum_number:g}"
            )
        if (
            plan.active_electrons is not None
            and runtime.active_electrons is not None
            and plan.active_electrons != runtime.active_electrons
        ):
            mismatches.append(
                f"active electrons input={plan.active_electrons}, output={runtime.active_electrons}"
            )
        if mismatches:
            guards.append(_guard("fail", "input/output signature", "; ".join(mismatches)))
        elif any(
            value is not None
            for value in (
                runtime.state_symmetry,
                runtime.spin_quantum_number,
                runtime.active_electrons,
            )
        ):
            guards.append(
                _guard(
                    "pass",
                    "input/output signature",
                    "available symmetry/spin/electron fields agree",
                )
            )
        else:
            level = "warn" if status == "finished" else "pending"
            guards.append(_guard(level, "input/output signature", "fingerprint not printed yet"))
        if runtime.supersymmetry_warnings:
            guards.append(
                _guard(
                    "warn",
                    "supersymmetry",
                    f"{runtime.supersymmetry_warnings} warning marker(s); inspect orbital mixing",
                )
            )
        elif plan.has_supsym:
            guards.append(
                _guard("pass", "supersymmetry", "SUPSYM requested; no warning marker found")
            )
        if plan.has_cionly:
            guards.append(
                _guard(
                    "warn",
                    "orbital optimization",
                    "CIONLY fixes input orbitals; validate their provenance",
                )
            )
        elif runtime.convergence_iterations is not None:
            guards.append(
                _guard(
                    "pass",
                    "orbital optimization",
                    f"converged after {runtime.convergence_iterations} iterations",
                )
            )
        else:
            level = "warn" if status == "finished" else "pending"
            guards.append(
                _guard(level, "orbital optimization", "no final convergence marker found")
            )
        if runtime.mo_occupations:
            minimum = min(runtime.mo_occupations)
            maximum = max(runtime.mo_occupations)
            finite = all(math.isfinite(value) for value in runtime.mo_occupations)
            level = "pass" if finite and minimum >= -0.05 and maximum <= 2.05 else "warn"
            guards.append(
                _guard(
                    level,
                    "MO occupations",
                    f"{len(runtime.mo_occupations)} printed values, range {minimum:.4f} to {maximum:.4f}",
                )
            )
        else:
            level = "warn" if status == "finished" else "pending"
            guards.append(
                _guard(level, "MO occupations", "pseudonatural occupations not available in output")
            )
        if status == "finished" and total:
            level = "pass" if len(runtime.rasorb_roots) >= total else "warn"
            guards.append(
                _guard(
                    level,
                    "orbital files",
                    f"RASORB written for {len(runtime.rasorb_roots)}/{total} roots",
                )
            )
        controls = []
        if plan.has_alter:
            controls.append("ALTER")
        if plan.has_supsym:
            controls.append("SUPSYM")
        guards.append(
            _guard(
                "review",
                "AO/MO identity",
                ("controls: " + "/".join(controls) + "; " if controls else "")
                + "generic monitoring cannot certify intended AO character; use moccheck/postanalysis",
            )
        )
    elif plan.kind == "caspt2":
        weights = runtime.caspt2_reference_weights
        if weights:
            minimum = min(weights)
            level = "pass" if minimum >= 0.50 else "warn"
            guards.append(
                _guard(
                    level,
                    "reference weights",
                    f"min={minimum:.4f}, max={max(weights):.4f}; 0.50 is a screening flag, not a universal cutoff",
                )
            )
        else:
            guards.append(_guard("pending", "reference weights", "not printed yet"))
    else:
        if runtime.so_states:
            guards.append(
                _guard("pass", "SO-state table", f"{len(runtime.so_states)} SO states printed")
            )
        elif runtime.rassi_so_table:
            guards.append(_guard("pass", "SO-state table", "SO-state composition table found"))
        else:
            level = "warn" if status == "finished" else "pending"
            guards.append(_guard(level, "SO-state table", runtime.stage))
        if runtime.rassi_transition_tables:
            guards.append(
                _guard(
                    "pass",
                    "transition tables",
                    f"{runtime.rassi_transition_tables} transition table marker(s) found",
                )
            )
        else:
            level = "warn" if status == "finished" else "pending"
            guards.append(_guard(level, "transition tables", runtime.stage))

    for error in runtime.error_lines:
        guards.append(_guard("fail", "output marker", error))
    return guards


def _progress(plan: PlanBlock, runtime: RuntimeBlock | None, status: str) -> dict[str, Any]:
    total = None if plan.kind == "rassi" else plan.expected_roots
    completed = 0
    label = "SO states" if plan.kind == "rassi" else "roots"
    if runtime is not None:
        if plan.kind == "rasscf":
            total = runtime.roots_requested or total
            completed = len(runtime.roots)
        elif plan.kind == "caspt2":
            completed = max(len(runtime.roots), len(runtime.caspt2_reference_weights))
        else:
            completed = len(runtime.so_states)
            total = None
            label = "SO states"
    if status == "finished" and total and completed < total:
        percent = 100.0 * completed / total
    elif total:
        percent = min(100.0, 100.0 * completed / total)
    else:
        percent = None
    return {"completed": completed, "total": total, "percent": percent, "label": label}


def _assemble_snapshot(
    inp: Path,
    output: Path,
    plan: list[PlanBlock],
    runtimes: list[RuntimeBlock],
    output_meta: dict[str, Any],
    *,
    incremental: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_kind = {(block.kind, block.kind_index): block for block in runtimes}
    block_rows: list[dict[str, Any]] = []
    for planned in plan:
        runtime = by_kind.get((planned.kind, planned.kind_index))
        status = "pending" if runtime is None else _runtime_status(runtime)
        guards = _block_guards(planned, runtime, status)
        if any(guard["level"] == "fail" for guard in guards):
            status = "failed"
        row = asdict(planned)
        row.update(
            {
                "status": status,
                "runtime": asdict(runtime) if runtime is not None else None,
                "progress": _progress(planned, runtime, status),
                "guards": guards,
            }
        )
        if runtime is not None:
            row["runtime"]["roots"] = sorted(runtime.roots)
            row["runtime"]["so_states"] = sorted(runtime.so_states)
            row["runtime"]["rasorb_roots"] = sorted(runtime.rasorb_roots)
        block_rows.append(row)

    statuses = [row["status"] for row in block_rows]
    if "failed" in statuses or output_meta["global_errors"]:
        overall = "failed"
    elif statuses and all(status == "finished" for status in statuses):
        overall = "complete"
    elif "running" in statuses:
        overall = "running"
    else:
        overall = "waiting"
    active = next((row["index"] for row in block_rows if row["status"] == "running"), None)
    if active is None:
        active = next((row["index"] for row in block_rows if row["status"] == "failed"), None)
    if active is None and overall != "complete":
        active = next((row["index"] for row in block_rows if row["status"] == "pending"), None)

    guard_counts = {level: 0 for level in ("pass", "warn", "fail", "pending", "review")}
    for row in block_rows:
        for guard in row["guards"]:
            guard_counts[guard["level"]] += 1

    stat = output.stat() if output.exists() else None
    unmatched = [
        asdict(runtime)
        for runtime in runtimes
        if (runtime.kind, runtime.kind_index) not in {(item.kind, item.kind_index) for item in plan}
    ]
    for item in unmatched:
        item["roots"] = sorted(item["roots"])
        item["so_states"] = sorted(item["so_states"])
        item["rasorb_roots"] = sorted(item["rasorb_roots"])
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(inp),
        "output_file": str(output),
        "output_exists": output.exists(),
        "output_size_bytes": stat.st_size if stat else 0,
        "output_mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        if stat
        else None,
        "output_age_seconds": max(0.0, time.time() - stat.st_mtime) if stat else None,
        "overall_status": overall,
        "active_block_index": active,
        "summary": {
            "scheduled": len(block_rows),
            "finished": statuses.count("finished"),
            "running": statuses.count("running"),
            "pending": statuses.count("pending"),
            "failed": statuses.count("failed"),
        },
        "guard_summary": guard_counts,
        "output": output_meta,
        "incremental": incremental,
        "blocks": block_rows,
        "unmatched_output_modules": unmatched,
        "interpretation_guard": (
            "moclive automates numerical and output-consistency checks. AO/MO chemical identity, "
            "state character, root homing, and spectral promotion still require project-specific review."
        ),
    }


def build_snapshot(inp_path: Path | str, output_path: Path | str) -> dict[str, Any]:
    """Build one structured monitoring snapshot by auditing the complete output."""

    inp = Path(inp_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    plan = parse_input_plan(inp.read_text(encoding="utf-8", errors="replace"))
    if not plan:
        raise ValueError("No &RASSCF, &CASPT2, or &RASSI blocks were found in the input")

    runtimes: list[RuntimeBlock] = []
    output_meta: dict[str, Any] = {"line_count": 0, "happy_landing": False, "global_errors": []}
    if output.exists():
        with _open_output(output) as handle:
            runtimes, output_meta = parse_output_stream(handle)
    return _assemble_snapshot(inp, output, plan, runtimes, output_meta)


class MocLiveSession:
    """Incrementally monitor one growing, uncompressed OpenMolcas output."""

    def __init__(self, inp_path: Path | str, output_path: Path | str) -> None:
        self.inp = Path(inp_path).expanduser().resolve()
        self.output = Path(output_path).expanduser().resolve()
        self.plan: list[PlanBlock] = []
        self.parser_state = OutputParserState()
        self.byte_offset = 0
        self.pending_bytes = b""
        self.last_read_bytes = 0
        self.total_read_bytes = 0
        self.reset_count = 0
        self._input_signature: tuple[int, int, int] | None = None
        self._output_identity: tuple[int, int] | None = None
        self._load_plan()

    @staticmethod
    def _file_signature(path: Path) -> tuple[int, int, int]:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _load_plan(self) -> None:
        self.plan = parse_input_plan(self.inp.read_text(encoding="utf-8", errors="replace"))
        if not self.plan:
            raise ValueError("No &RASSCF, &CASPT2, or &RASSI blocks were found in the input")
        self._input_signature = self._file_signature(self.inp)

    def _reset_output(self) -> None:
        self.parser_state = OutputParserState()
        self.byte_offset = 0
        self.pending_bytes = b""
        self._output_identity = None
        self.reset_count += 1

    def refresh(self) -> dict[str, Any]:
        """Read only newly appended output bytes and return the current snapshot."""

        input_signature = self._file_signature(self.inp)
        if input_signature != self._input_signature:
            self._load_plan()
            self._reset_output()

        self.last_read_bytes = 0
        if not self.output.exists():
            if self.byte_offset or self.parser_state.blocks:
                self._reset_output()
            return _assemble_snapshot(
                self.inp,
                self.output,
                self.plan,
                self.parser_state.blocks,
                self.parser_state.metadata(),
                incremental=self._incremental_metadata("waiting for output"),
            )

        if self.output.suffix.lower() == ".gz":
            snapshot = build_snapshot(self.inp, self.output)
            snapshot["incremental"] = self._incremental_metadata(
                "compressed output uses a full one-shot audit"
            )
            return snapshot

        stat = self.output.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self._output_identity is None:
            self._output_identity = identity
        elif identity != self._output_identity or stat.st_size < self.byte_offset:
            self._reset_output()
            self._output_identity = identity

        with self.output.open("rb") as handle:
            handle.seek(self.byte_offset)
            chunk = handle.read()
            self.byte_offset = handle.tell()
        self.last_read_bytes = len(chunk)
        self.total_read_bytes += len(chunk)

        data = self.pending_bytes + chunk
        newline = max(data.rfind(b"\n"), data.rfind(b"\r"))
        if newline >= 0:
            complete = data[: newline + 1]
            self.pending_bytes = data[newline + 1 :]
            text = complete.decode("utf-8", errors="replace")
            _ingest_output_lines(text.splitlines(keepends=True), self.parser_state)
        else:
            self.pending_bytes = data

        return _assemble_snapshot(
            self.inp,
            self.output,
            self.plan,
            self.parser_state.blocks,
            self.parser_state.metadata(),
            incremental=self._incremental_metadata("tailing appended bytes"),
        )

    def _incremental_metadata(self, mode: str) -> dict[str, Any]:
        return {
            "mode": mode,
            "byte_offset": self.byte_offset,
            "last_read_bytes": self.last_read_bytes,
            "total_read_bytes": self.total_read_bytes,
            "pending_bytes": len(self.pending_bytes),
            "reset_count": self.reset_count,
        }


def _bar(percent: float | None, width: int, *, ascii_only: bool) -> str:
    if percent is None:
        return "[" + ("." if ascii_only else "\u00b7") * width + "]"
    filled = min(width, max(0, round(width * percent / 100.0)))
    full = "#" if ascii_only else "\u2588"
    empty = "." if ascii_only else "\u2591"
    return "[" + full * filled + empty * (width - filled) + "]"


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"


def render_snapshot(
    snapshot: dict[str, Any],
    *,
    ascii_only: bool = False,
    spinner_frame: int = 0,
    bar_width: int = 24,
) -> str:
    """Render a stable terminal view for one snapshot."""

    icons = {
        "finished": "OK" if ascii_only else "\u2713",
        "failed": "X" if ascii_only else "\u2717",
        "pending": "." if ascii_only else "\u00b7",
    }
    spinner = (SPINNER_ASCII if ascii_only else SPINNER_UNICODE)[
        spinner_frame % len(SPINNER_ASCII if ascii_only else SPINNER_UNICODE)
    ]
    summary = snapshot["summary"]
    lines = [
        f"MOCLIVE | {snapshot['overall_status'].upper()} | {snapshot['generated_at']}",
        f"Input : {snapshot['input_file']}",
        f"Output: {snapshot['output_file']} ({_human_size(snapshot['output_size_bytes'])})",
        (
            f"Blocks: {summary['finished']}/{summary['scheduled']} finished, "
            f"{summary['running']} running, {summary['pending']} pending, {summary['failed']} failed"
        ),
        "",
        "Scheduled MOLCAS blocks",
    ]
    incremental = snapshot.get("incremental")
    if incremental:
        lines.insert(
            3,
            (
                f"Tail  : +{_human_size(incremental['last_read_bytes'])} this refresh, "
                f"offset {incremental['byte_offset']} B ({incremental['mode']})"
            ),
        )
    for row in snapshot["blocks"]:
        status = row["status"]
        icon = spinner if status == "running" else icons[status]
        progress = row["progress"]
        percent = progress["percent"]
        bar_percent = 100.0 if status == "finished" else percent
        title = row["title"] or "(untitled)"
        signature = ""
        if row["kind"] == "rasscf":
            parts = []
            if row["symmetry"] is not None:
                parts.append(f"sym={row['symmetry']}")
            if row["spin_multiplicity"] is not None:
                parts.append(f"spin={row['spin_multiplicity']}")
            signature = " " + " ".join(parts) if parts else ""
        if (
            status == "finished"
            and progress["total"]
            and progress["completed"] < progress["total"]
        ):
            progress_text = (
                f"finished; {progress['completed']}/{progress['total']} "
                f"{progress['label']} printed"
            )
        elif progress["total"]:
            progress_text = (
                f"{progress['completed']}/{progress['total']} {progress['label']} {percent:.1f}%"
            )
        elif progress["completed"]:
            progress_text = f"{progress['completed']} {progress['label']}"
            if row["kind"] == "rassi" and row["expected_roots"]:
                progress_text += f" ({row['expected_roots']} input spin-free states)"
        else:
            progress_text = status
        lines.append(
            f"[{icon}] {row['index']:02d} {row['kind'].upper():7}{signature:<15} "
            f"{_bar(bar_percent, bar_width, ascii_only=ascii_only)} {progress_text}"
        )
        lines.append(f"     {title}")

    active_index = snapshot.get("active_block_index")
    focus_index = active_index
    focus_label = "Active detail"
    if focus_index is None:
        completed_rasscf = [
            row["index"]
            for row in snapshot["blocks"]
            if row["kind"] == "rasscf" and row["status"] == "finished"
        ]
        if completed_rasscf:
            focus_index = completed_rasscf[-1]
            focus_label = "Latest completed RASSCF guard"
    if focus_index is not None:
        active = snapshot["blocks"][focus_index - 1]
        runtime = active.get("runtime") or {}
        lines.extend(["", f"{focus_label}: block {focus_index:02d} {active['kind'].upper()}"])
        if runtime:
            lines.append(f"  Stage: {runtime.get('stage', 'initializing')}")
            if active["kind"] == "rasscf" and runtime.get("convergence_iterations") is not None:
                lines.append(f"  RASSCF iterations: {runtime['convergence_iterations']}")
            if active["kind"] == "caspt2" and runtime.get("caspt2_reference_weights"):
                weights = runtime["caspt2_reference_weights"]
                lines.append(f"  Reference weights: {min(weights):.4f} to {max(weights):.4f}")
        lines.append("  Health / physical guards")
        guard_icons = {
            "pass": "+" if ascii_only else "\u2713",
            "warn": "!",
            "fail": "X" if ascii_only else "\u2717",
            "pending": "." if ascii_only else "\u00b7",
            "review": "?",
        }
        for guard in active["guards"]:
            lines.append(f"    [{guard_icons[guard['level']]}] {guard['name']}: {guard['detail']}")

    if snapshot["output"]["global_errors"]:
        lines.extend(["", "Global output warnings"])
        lines.extend(f"  ! {item}" for item in snapshot["output"]["global_errors"])
    if snapshot["unmatched_output_modules"]:
        lines.append(
            f"\nWarning: {len(snapshot['unmatched_output_modules'])} monitored output module(s) "
            "are outside the input schedule."
        )
    lines.extend(["", snapshot["interpretation_guard"]])
    return "\n".join(lines)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moclive",
        description="Monitor input-scoped OpenMolcas RASSCF/CASPT2/RASSI progress and guards.",
    )
    parser.add_argument("inp", type=Path, help="OpenMolcas input deck")
    parser.add_argument("output", type=Path, help="Matching live .log/.out file")
    parser.add_argument(
        "--interval", type=float, default=300.0, help="Refresh interval in seconds (default: 300)"
    )
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument(
        "--keep-watching", action="store_true", help="Keep refreshing after completion/failure"
    )
    parser.add_argument("--ascii", action="store_true", help="Use ASCII-only status symbols")
    parser.add_argument(
        "--no-clear", action="store_true", help="Do not clear the terminal between refreshes"
    )
    parser.add_argument(
        "--bar-width", type=int, default=24, help="Progress-bar width (default: 24)"
    )
    parser.add_argument("--json", action="store_true", help="Print one JSON snapshot and exit")
    parser.add_argument("--json-out", type=Path, help="Atomically update this JSON snapshot file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    if args.bar_width < 8:
        raise ValueError("--bar-width must be at least 8")
    once = args.once or args.json
    session = None if once else MocLiveSession(args.inp, args.output)
    frame = 0
    try:
        while True:
            snapshot = (
                build_snapshot(args.inp, args.output) if session is None else session.refresh()
            )
            if args.json_out:
                _write_json_atomic(args.json_out, snapshot)
            if args.json:
                print(json.dumps(snapshot, indent=2))
            else:
                if not args.no_clear and sys.stdout.isatty() and frame:
                    print("\033[2J\033[H", end="")
                print(
                    render_snapshot(
                        snapshot,
                        ascii_only=args.ascii,
                        spinner_frame=frame,
                        bar_width=min(
                            args.bar_width, max(8, shutil.get_terminal_size((120, 30)).columns // 3)
                        ),
                    ),
                    flush=True,
                )
            if once or (
                snapshot["overall_status"] in {"complete", "failed"} and not args.keep_watching
            ):
                return 1 if snapshot["overall_status"] == "failed" else 0
            frame += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmoclive stopped.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
