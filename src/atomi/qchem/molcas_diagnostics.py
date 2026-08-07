"""Human-readable OpenMolcas frontier-orbital diagnostics.

``moccheck`` pairs an OpenMolcas input deck with its output. It finds the
single last RASSCF reference immediately before the first ALTER/SUPSYM setup
block, verifies the corresponding output module, and reports occupied and
virtual orbitals from every symmetry table in that inherited orbital set.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "atomi.molcas_moccheck.v1"
COLLECTION_SCHEMA = "atomi.molcas_moccheck_collection.v1"
HANDOFF_SCHEMA = "atomi.molcas_moccheck_handoff.v1"
OPENMOLCAS_MAXALTER = 16
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
FLOAT_RE = re.compile(FLOAT_PATTERN)
RASSCF_START_RE = re.compile(r"^\s*&RASSCF\b", re.IGNORECASE)
LOG_RASSCF_START_RE = re.compile(r"--- Start Module:\s*rasscf\b", re.IGNORECASE)
LOG_RASSCF_END_RE = re.compile(r"--- Module rasscf spent\b", re.IGNORECASE)
MO_SYMMETRY_RE = re.compile(
    r"Molecular orbitals for symmetry species\s+(?P<symmetry>\d+)\s*:\s*(?P<label>\S+)",
    re.IGNORECASE,
)
AO_MATRIX_RE = re.compile(r"^\s*(?P<ao_index>\d+)\s+(?P<atom>\S+)\s+(?P<ao>\S+)\s+(?P<values>.*)$")
COMPACT_MO_ROW_RE = re.compile(
    rf"^\s*(?P<mo>\d+)\s*(?P<energy>{FLOAT_PATTERN})\s+"
    rf"(?P<occupation>{FLOAT_PATTERN})(?:\s+.*)?$"
)
COMPACT_AO_TERM_RE = re.compile(
    rf"(?P<ao_index>\d+)\s+(?P<atom>\S+)\s+(?P<ao>\S+)\s+"
    rf"\(\s*(?P<coefficient>{FLOAT_PATTERN})\s*\)"
)
TITLE_MULTIPLICITIES = {
    "singlet": 1,
    "doublet": 2,
    "triplet": 3,
    "quartet": 4,
    "quintet": 5,
    "sextet": 6,
}


@dataclass(frozen=True)
class RasscfInputBlock:
    index: int
    start_line: int
    end_line: int
    title: str
    symmetry: int | None
    spin_multiplicity: int | None
    nactel: tuple[int, ...]
    inactive: tuple[int, ...]
    ras1: tuple[int, ...]
    ras2: tuple[int, ...]
    ras3: tuple[int, ...]
    has_alter: bool
    has_supsym: bool
    alter_swaps: tuple[tuple[int, int, int], ...]
    supsym_groups: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class RasscfLogModule:
    index: int
    start_line: int
    end_line: int
    active_electrons: int | None
    spin_quantum_number: float | None
    state_symmetry: int | None
    text: str


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keyword_payload(lines: list[str], keyword: str) -> str | None:
    wanted = keyword.lower()
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            continue
        token, _, remainder = stripped.partition(" ")
        token = token.rstrip("=").lower()
        if token != wanted:
            continue
        payload = remainder.strip().lstrip("=").strip()
        if payload:
            return payload
        for following in lines[idx + 1 :]:
            value = following.strip()
            if not value or value.startswith("*"):
                continue
            return value
    return None


def _int_tuple(payload: str | None) -> tuple[int, ...]:
    if payload is None:
        return ()
    return tuple(int(value) for value in re.findall(r"[-+]?\d+", payload))


def _first_int(payload: str | None) -> int | None:
    values = _int_tuple(payload)
    return values[0] if values else None


def _keyword_line_index(lines: list[str], keyword: str) -> int | None:
    wanted = keyword.lower()
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            continue
        token = stripped.split(None, 1)[0].rstrip("=").lower()
        if token == wanted:
            return idx
    return None


def _data_lines_after_keyword(lines: list[str], keyword: str) -> list[str]:
    index = _keyword_line_index(lines, keyword)
    if index is None:
        return []
    data: list[str] = []
    for raw in lines[index + 1 :]:
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            continue
        token = stripped.split(None, 1)[0].rstrip("=").lower()
        if token in {
            "alter",
            "supsym",
            "thrs",
            "iterations",
            "levs",
            "orbl",
            "orba",
            "tdm",
            "end",
        }:
            break
        data.append(stripped)
    return data


def _parse_alter_swaps(lines: list[str]) -> tuple[tuple[int, int, int], ...]:
    data = _data_lines_after_keyword(lines, "alter")
    if not data:
        return ()
    count_values = _int_tuple(data[0])
    if not count_values:
        return ()
    swaps: list[tuple[int, int, int]] = []
    for line in data[1 : 1 + count_values[0]]:
        values = _int_tuple(line)
        if len(values) >= 3:
            swaps.append((values[0], values[1], values[2]))
    return tuple(swaps)


def _parse_supsym_groups(
    lines: list[str], *, symmetry_count: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    data = _data_lines_after_keyword(lines, "supsym")
    if not data:
        return ()
    groups_by_symmetry: list[tuple[tuple[int, ...], ...]] = []
    cursor = 0
    for _ in range(symmetry_count):
        if cursor >= len(data):
            groups_by_symmetry.append(())
            continue
        count_values = _int_tuple(data[cursor])
        cursor += 1
        group_count = count_values[0] if count_values else 0
        groups: list[tuple[int, ...]] = []
        for _ in range(group_count):
            if cursor >= len(data):
                break
            values = _int_tuple(data[cursor])
            cursor += 1
            if values:
                groups.append(tuple(values[1 : 1 + values[0]]))
        groups_by_symmetry.append(tuple(groups))
    return tuple(groups_by_symmetry)


def parse_rasscf_input_blocks(text: str) -> list[RasscfInputBlock]:
    """Parse chronological RASSCF blocks from an OpenMolcas input deck."""

    lines = text.splitlines()
    blocks: list[RasscfInputBlock] = []
    idx = 0
    while idx < len(lines):
        if not RASSCF_START_RE.match(lines[idx]):
            idx += 1
            continue
        end = idx + 1
        while end < len(lines) and lines[end].strip().lower() != "end of input":
            if RASSCF_START_RE.match(lines[end]):
                break
            end += 1
        if end < len(lines) and lines[end].strip().lower() == "end of input":
            end += 1
        body = lines[idx:end]
        tokens = {line.strip().split(None, 1)[0].lower() for line in body if line.strip()}
        orbital_vectors = [
            _int_tuple(_keyword_payload(body, keyword))
            for keyword in ("inactive", "ras1", "ras2", "ras3")
        ]
        symmetry_count = max((len(values) for values in orbital_vectors), default=0)
        blocks.append(
            RasscfInputBlock(
                index=len(blocks) + 1,
                start_line=idx + 1,
                end_line=end,
                title=(_keyword_payload(body, "title") or "").strip(),
                symmetry=_first_int(_keyword_payload(body, "symmetry")),
                spin_multiplicity=_first_int(_keyword_payload(body, "spin")),
                nactel=_int_tuple(_keyword_payload(body, "nactel")),
                inactive=_int_tuple(_keyword_payload(body, "inactive")),
                ras1=_int_tuple(_keyword_payload(body, "ras1")),
                ras2=_int_tuple(_keyword_payload(body, "ras2")),
                ras3=_int_tuple(_keyword_payload(body, "ras3")),
                has_alter="alter" in tokens,
                has_supsym="supsym" in tokens,
                alter_swaps=_parse_alter_swaps(body),
                supsym_groups=_parse_supsym_groups(body, symmetry_count=symmetry_count),
            )
        )
        idx = max(end, idx + 1)
    return blocks


def select_reference_block(
    blocks: list[RasscfInputBlock],
    *,
    symmetry: int = 1,
    reference_block: int | None = None,
) -> tuple[RasscfInputBlock, RasscfInputBlock | None]:
    """Select the one pre-ALTER/SUPSYM reference inherited by setup.

    ``symmetry`` remains accepted for API compatibility, but reference
    selection is intentionally independent of it: ALTER/SUPSYM inherits one
    JobIph from the immediately preceding RASSCF module, and every irrep must
    therefore be audited from that same module.
    """

    if not blocks:
        raise ValueError("No &RASSCF blocks were found in the input file")
    setup = next((block for block in blocks if block.has_alter or block.has_supsym), None)
    if reference_block is not None:
        if reference_block < 1 or reference_block > len(blocks):
            raise ValueError(f"--reference-block must be between 1 and {len(blocks)}")
        return blocks[reference_block - 1], setup
    if setup is None:
        raise ValueError(
            "No ALTER/SUPSYM setup block was found; select the reference explicitly with --reference-block"
        )
    candidates = [block for block in blocks if block.index < setup.index]
    if not candidates:
        raise ValueError(f"No RASSCF block occurs before setup block {setup.index}")
    return candidates[-1], setup


def available_mo_symmetries(module_text: str) -> list[int]:
    """Return MO-table symmetries in the final pseudonatural listing."""

    lowered = module_text.lower()
    marker = "pseudonatural active orbitals and approximate occupation numbers"
    anchor = lowered.rfind(marker)
    search_text = module_text[anchor:] if anchor >= 0 else module_text
    candidates = sorted(
        {int(match.group("symmetry")) for match in MO_SYMMETRY_RE.finditer(search_text)}
    )
    return [
        symmetry
        for symmetry in candidates
        if parse_full_mo_matrix(module_text, symmetry=symmetry)
        or parse_compact_mo_listing(module_text, symmetry=symmetry)
    ]


def available_reference_symmetries(
    blocks: list[RasscfInputBlock], *, reference_block: int | None = None
) -> list[int]:
    """Return symmetries that have a usable pre-setup reference block."""

    if not blocks:
        raise ValueError("No &RASSCF blocks were found in the input file")
    if reference_block is not None:
        if reference_block < 1 or reference_block > len(blocks):
            raise ValueError(f"--reference-block must be between 1 and {len(blocks)}")
        symmetry = blocks[reference_block - 1].symmetry
        if symmetry is None:
            raise ValueError(f"Input RASSCF block {reference_block} has no Symmetry value")
        return [symmetry]
    setup = next((block for block in blocks if block.has_alter or block.has_supsym), None)
    if setup is None:
        raise ValueError(
            "No ALTER/SUPSYM setup block was found; select the reference explicitly "
            "with --reference-block"
        )
    return sorted(
        {
            block.symmetry
            for block in blocks
            if block.index < setup.index and block.symmetry is not None
        }
    )


def _search_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _search_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1).replace("D", "E").replace("d", "e")) if match else None


def parse_rasscf_log_modules(text: str) -> list[RasscfLogModule]:
    """Parse chronological RASSCF modules and their identifying fingerprints."""

    lines = text.splitlines()
    starts = [idx for idx, line in enumerate(lines) if LOG_RASSCF_START_RE.search(line)]
    modules: list[RasscfLogModule] = []
    for ordinal, start in enumerate(starts, start=1):
        next_start = starts[ordinal] if ordinal < len(starts) else len(lines)
        end = next_start
        for idx in range(start + 1, next_start):
            if LOG_RASSCF_END_RE.search(lines[idx]):
                end = idx + 1
                break
        module_text = "\n".join(lines[start:end])
        modules.append(
            RasscfLogModule(
                index=ordinal,
                start_line=start + 1,
                end_line=end,
                active_electrons=_search_int(
                    r"Number of electrons in active shells\s+(\d+)", module_text
                ),
                spin_quantum_number=_search_float(
                    rf"Spin quantum number\s+({FLOAT_PATTERN})", module_text
                ),
                state_symmetry=_search_int(r"State symmetry\s+(\d+)", module_text),
                text=module_text,
            )
        )
    return modules


def _signature_matches(block: RasscfInputBlock, module: RasscfLogModule) -> bool:
    expected_active = block.nactel[0] if block.nactel else None
    expected_spin = (
        (block.spin_multiplicity - 1) / 2.0 if block.spin_multiplicity is not None else None
    )
    return (
        (block.symmetry is None or module.state_symmetry == block.symmetry)
        and (expected_active is None or module.active_electrons == expected_active)
        and (
            expected_spin is None
            or module.spin_quantum_number is not None
            and abs(module.spin_quantum_number - expected_spin) < 1.0e-6
        )
    )


def match_log_module(
    block: RasscfInputBlock, modules: list[RasscfLogModule]
) -> tuple[RasscfLogModule, str]:
    """Match an input block to output, preferring verified chronological order."""

    if not modules:
        raise ValueError("No 'Start Module: rasscf' sections were found in the output")
    if block.index <= len(modules):
        expected = modules[block.index - 1]
        if _signature_matches(block, expected):
            return expected, "chronological index with verified symmetry/spin/electron signature"
    candidates = [module for module in modules if _signature_matches(block, module)]
    if not candidates:
        raise ValueError(
            "Could not match the selected input RASSCF block to an output module by "
            "symmetry, spin, and active-electron count"
        )
    selected = min(candidates, key=lambda module: abs(module.index - block.index))
    return selected, "nearest matching symmetry/spin/electron signature"


def _floats(text: str) -> list[float]:
    return [float(value.replace("D", "E").replace("d", "e")) for value in FLOAT_RE.findall(text)]


def parse_full_mo_matrix(module_text: str, *, symmetry: int = 1) -> list[dict[str, Any]]:
    """Parse a full OpenMolcas MO coefficient matrix for one symmetry."""

    lines = module_text.splitlines()
    pseudonatural = [
        idx
        for idx, line in enumerate(lines)
        if "pseudonatural active orbitals and approximate occupation numbers" in line.lower()
    ]
    anchor = pseudonatural[-1] if pseudonatural else 0
    section_start: int | None = None
    section_end = len(lines)
    symmetry_label = ""
    for idx in range(anchor, len(lines)):
        match = MO_SYMMETRY_RE.search(lines[idx])
        if not match:
            continue
        found_symmetry = int(match.group("symmetry"))
        if section_start is None and found_symmetry == symmetry:
            section_start = idx + 1
            symmetry_label = match.group("label")
            continue
        if section_start is not None:
            section_end = idx
            break
    if section_start is None:
        return []

    rows: dict[int, dict[str, Any]] = {}
    idx = section_start
    while idx < section_end:
        stripped = lines[idx].strip()
        if not stripped.startswith("Orbital"):
            idx += 1
            continue
        orbitals = [int(value) for value in re.findall(r"\d+", stripped[len("Orbital") :])]
        if not orbitals:
            idx += 1
            continue
        energy_idx: int | None = None
        occupation_idx: int | None = None
        for lookahead in range(idx + 1, min(idx + 7, section_end)):
            label = lines[lookahead].strip().lower()
            if label.startswith("energy"):
                energy_idx = lookahead
            elif label.startswith("occ. no.") or label.startswith("occupation"):
                occupation_idx = lookahead
                break
        if energy_idx is None or occupation_idx is None:
            idx += 1
            continue
        energies = _floats(lines[energy_idx].split(None, 1)[1])
        occupations = _floats(lines[occupation_idx].split(None, 2)[-1])
        if len(energies) < len(orbitals) or len(occupations) < len(orbitals):
            raise ValueError(
                f"Incomplete MO header near output line {energy_idx + 1}: "
                f"{len(orbitals)} orbitals, {len(energies)} energies, {len(occupations)} occupations"
            )
        for offset, orbital in enumerate(orbitals):
            rows[orbital] = {
                "mo": orbital,
                "energy": energies[offset],
                "occupation": occupations[offset],
                "symmetry": symmetry,
                "symmetry_label": symmetry_label,
                "terms": [],
            }
        row_idx = occupation_idx + 1
        while row_idx < section_end and not lines[row_idx].strip().startswith("Orbital"):
            ao_match = AO_MATRIX_RE.match(lines[row_idx])
            if ao_match:
                coefficients = _floats(ao_match.group("values"))
                if len(coefficients) >= len(orbitals):
                    for offset, orbital in enumerate(orbitals):
                        coefficient = coefficients[offset]
                        rows[orbital]["terms"].append(
                            {
                                "ao_index": int(ao_match.group("ao_index")),
                                "atom": ao_match.group("atom"),
                                "ao": ao_match.group("ao"),
                                "coefficient": coefficient,
                                "coeff2": coefficient * coefficient,
                            }
                        )
            row_idx += 1
        idx = row_idx
    return [rows[orbital] for orbital in sorted(rows)]


def parse_compact_mo_listing(module_text: str, *, symmetry: int = 1) -> list[dict[str, Any]]:
    """Parse OpenMolcas ``ORBA COMP`` pseudonatural-orbital listings.

    Compact listings contain one orbital header followed by only the AO
    coefficients that pass OpenMolcas' print threshold. They are sufficient
    for frontier identity diagnostics, but they are not a complete coefficient
    matrix.
    """

    lines = module_text.splitlines()
    pseudonatural = [
        idx
        for idx, line in enumerate(lines)
        if "pseudonatural active orbitals and approximate occupation numbers" in line.lower()
    ]
    anchor = pseudonatural[-1] if pseudonatural else 0
    section_start: int | None = None
    section_end = len(lines)
    symmetry_label = ""
    for idx in range(anchor, len(lines)):
        match = MO_SYMMETRY_RE.search(lines[idx])
        if not match:
            continue
        found_symmetry = int(match.group("symmetry"))
        if section_start is None and found_symmetry == symmetry:
            section_start = idx + 1
            symmetry_label = match.group("label")
            continue
        if section_start is not None:
            section_end = idx
            break
    if section_start is None:
        return []

    rows: list[dict[str, Any]] = []
    idx = section_start
    while idx < section_end:
        match = COMPACT_MO_ROW_RE.match(lines[idx])
        if not match:
            idx += 1
            continue
        row: dict[str, Any] = {
            "mo": int(match.group("mo")),
            "energy": float(match.group("energy").replace("D", "E").replace("d", "e")),
            "occupation": float(match.group("occupation").replace("D", "E").replace("d", "e")),
            "symmetry": symmetry,
            "symmetry_label": symmetry_label,
            "terms": [],
        }
        idx += 1
        while idx < section_end and not COMPACT_MO_ROW_RE.match(lines[idx]):
            for term_match in COMPACT_AO_TERM_RE.finditer(lines[idx]):
                coefficient = float(
                    term_match.group("coefficient").replace("D", "E").replace("d", "e")
                )
                row["terms"].append(
                    {
                        "ao_index": int(term_match.group("ao_index")),
                        "atom": term_match.group("atom"),
                        "ao": term_match.group("ao"),
                        "coefficient": coefficient,
                        "coeff2": coefficient * coefficient,
                    }
                )
            idx += 1
        rows.append(row)
    return rows


def _title_spin_warning(block: RasscfInputBlock, role: str) -> str | None:
    title_lower = block.title.lower()
    for name, multiplicity in TITLE_MULTIPLICITIES.items():
        if name in title_lower and block.spin_multiplicity not in (None, multiplicity):
            return (
                f"{role} block title says {name}, but Spin {block.spin_multiplicity} "
                f"means multiplicity {block.spin_multiplicity}"
            )
    return None


def _infer_ras3_intent(input_text: str, inp_path: Path) -> dict[str, Any]:
    directive = re.search(
        r"^\s*\*\s*ATOMI\s+RAS3_TARGET\s+(.+?)\s*$",
        input_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if directive:
        targets = []
        for token in re.split(r"[\s,]+", directive.group(1).strip()):
            if not token:
                continue
            atom, separator, shell = token.partition(":")
            if not separator or not atom or not shell:
                continue
            targets.append({"atom": atom, "shell": shell.lower()})
        if targets:
            return {
                "status": "explicit",
                "targets": targets,
                "evidence": directive.group(0).strip(),
            }

    evidence_text = f"{inp_path.name}\n{input_text[:2500]}".lower()
    shells: list[str] = []
    if re.search(r"5f\s*[-+_/ ]\s*7s\s*[-+_/ ]\s*7p|5f7s7p", evidence_text):
        shells = ["5f", "7s", "7p"]
    elif re.search(r"5f\s*[-+_/ ]\s*7s|5f7s|7s\s+with\s+5f", evidence_text):
        shells = ["5f", "7s"]
    elif re.search(r"5f.?only|only\s+5f\s+for\s+ras3", evidence_text):
        shells = ["5f"]
    if shells:
        return {
            "status": "inferred",
            "targets": [{"atom": None, "shell": shell} for shell in shells],
            "evidence": "filename or leading input comments",
        }
    return {
        "status": "ambiguous",
        "targets": [],
        "evidence": (
            "No explicit '* ATOMI RAS3_TARGET Atom:Shell ...' annotation or recognized "
            "5f-only/5f+7s/5f+7s+7p label was found"
        ),
    }


def _shell_matches(ao: str, shell: str) -> bool:
    return ao.lower().startswith(shell.lower())


def _target_score(row: dict[str, Any], *, atom: str, shells: list[str]) -> float:
    target_coeff2 = sum(
        float(term["coeff2"])
        for term in row["terms"]
        if str(term["atom"]).lower() == atom.lower()
        and any(_shell_matches(str(term["ao"]), shell) for shell in shells)
    )
    printed_coeff2 = sum(float(term["coeff2"]) for term in row["terms"])
    return target_coeff2 / printed_coeff2 if printed_coeff2 > 0.0 else 0.0


def _component_score(row: dict[str, Any], *, atom: str, ao: str) -> float:
    component_coeff2 = sum(
        float(term["coeff2"])
        for term in row["terms"]
        if str(term["atom"]).lower() == atom.lower()
        and str(term["ao"]).lower() == ao.lower()
    )
    printed_coeff2 = sum(float(term["coeff2"]) for term in row["terms"])
    return component_coeff2 / printed_coeff2 if printed_coeff2 > 0.0 else 0.0


def _infer_target_atom(
    rows_by_symmetry: dict[int, list[dict[str, Any]]], shells: list[str]
) -> tuple[str | None, dict[str, float]]:
    totals: dict[str, float] = {}
    for rows in rows_by_symmetry.values():
        for row in rows:
            for term in row["terms"]:
                if any(_shell_matches(str(term["ao"]), shell) for shell in shells):
                    atom = str(term["atom"])
                    totals[atom] = totals.get(atom, 0.0) + float(term["coeff2"])
    if not totals:
        return None, totals
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[1][1] >= 0.8 * ranked[0][1]:
        return None, totals
    return ranked[0][0], totals


def _vector_value(values: tuple[int, ...], symmetry: int) -> int:
    return values[symmetry - 1] if symmetry <= len(values) else 0


def _ras3_slots(block: RasscfInputBlock, symmetry: int) -> list[int]:
    count = _vector_value(block.ras3, symmetry)
    start = (
        _vector_value(block.inactive, symmetry)
        + _vector_value(block.ras1, symmetry)
        + _vector_value(block.ras2, symmetry)
    )
    return list(range(start + 1, start + count + 1))


def _identity_positions_after_swaps(
    rows_by_symmetry: dict[int, list[dict[str, Any]]],
    swaps: tuple[tuple[int, int, int], ...],
) -> dict[int, dict[int, int]]:
    positions = {
        symmetry: {int(row["mo"]): int(row["mo"]) for row in rows}
        for symmetry, rows in rows_by_symmetry.items()
    }
    for symmetry, first, second in swaps:
        mapping = positions.get(symmetry)
        if mapping is None or first not in mapping or second not in mapping:
            continue
        mapping[first], mapping[second] = mapping[second], mapping[first]
    return positions


def _format_supsym_block(groups_by_symmetry: list[list[list[int]]]) -> list[str]:
    lines = ["SUPSYM"]
    for groups in groups_by_symmetry:
        lines.append(f" {len(groups)}")
        for group in groups:
            lines.append("  " + " ".join([str(len(group)), *(str(value) for value in group)]))
    return lines


def build_ras3_recommendation(
    input_text: str,
    inp_path: Path,
    *,
    setup: RasscfInputBlock | None,
    rows_by_symmetry: dict[int, list[dict[str, Any]]],
    ao_listing_format: str,
    occupancy_threshold: float,
    mixing_window_ha: float,
) -> dict[str, Any]:
    """Audit intended RAS3 identity and propose only evidence-backed homing changes."""

    intent = _infer_ras3_intent(input_text, inp_path)
    base: dict[str, Any] = {
        "status": "review_required",
        "intent": intent,
        "mixing_window_hartree": mixing_window_ha,
        "safety_note": (
            "ALTER changes orbital subspace membership. SUPSYM can prevent accidental "
            "same-symmetry exchange, but over-constraining RAS3 can suppress physical "
            "metal-ligand mixing. Validate the proposed setup in a short RASSCF-only probe."
        ),
        "transition_space_note": (
            "A chemically intended orbital that remains outside RAS3 is absent from the "
            "explicit RAS excitation manifold. A high starting SCF or pseudonatural-orbital "
            "energy does not make the orbital unphysical or exclude it from a chemically "
            "chosen active space. ALTER only homes the intended orbital into RAS3; SUPSYM "
            "does not create a transition or force an occupation, but restricts subsequent "
            "same-irrep orbital rotations. Validate the final AO-character subspace, RAS "
            "occupations, roots, and oscillator strengths together; weak intensity alone "
            "does not prove that a target orbital is missing."
        ),
    }
    if setup is None or not setup.ras3:
        return {**base, "status": "unavailable", "reason": "The setup block has no RAS3 vector"}
    if not rows_by_symmetry:
        return {**base, "status": "unavailable", "reason": "No MO rows were parsed"}
    if not intent["targets"]:
        return {
            **base,
            "status": "ambiguous_intent",
            "reason": intent["evidence"],
            "annotation_example": "* ATOMI RAS3_TARGET U1:5f U1:7s",
        }

    shells = list(dict.fromkeys(str(target["shell"]).lower() for target in intent["targets"]))
    explicit_atoms = {
        str(target["atom"])
        for target in intent["targets"]
        if target.get("atom") is not None
    }
    atom_scores: dict[str, float] = {}
    if len(explicit_atoms) == 1:
        target_atom = next(iter(explicit_atoms))
    elif len(explicit_atoms) > 1:
        return {
            **base,
            "status": "ambiguous_intent",
            "reason": "Multiple target atoms were declared; one central-atom target is required",
        }
    else:
        target_atom, atom_scores = _infer_target_atom(rows_by_symmetry, shells)
    if target_atom is None:
        return {
            **base,
            "status": "ambiguous_atom",
            "reason": "A unique central atom could not be inferred from the requested AO shells",
            "atom_scores": atom_scores,
            "annotation_example": "* ATOMI RAS3_TARGET U1:5f U1:7s",
        }

    positions = _identity_positions_after_swaps(rows_by_symmetry, setup.alter_swaps)
    per_symmetry: list[dict[str, Any]] = []
    safe_additions: list[list[int]] = []
    conditional_additions: list[list[int]] = []
    source_groups: list[list[int]] = []
    final_groups: list[list[int]] = []
    max_symmetry = max(
        len(setup.inactive), len(setup.ras1), len(setup.ras2), len(setup.ras3), 0
    )

    for symmetry in range(1, max_symmetry + 1):
        rows = rows_by_symmetry.get(symmetry, [])
        slots = _ras3_slots(setup, symmetry)
        final_groups.append(slots)
        if not slots or not rows:
            source_groups.append([])
            per_symmetry.append(
                {
                    "symmetry": symmetry,
                    "ras3_slots": slots,
                    "status": "no_ras3" if not slots else "missing_mo_table",
                }
            )
            continue
        row_by_id = {int(row["mo"]): row for row in rows}
        component_shell_order = {shell: index for index, shell in enumerate(shells)}
        component_labels = sorted(
            {
                str(term["ao"])
                for row in rows
                for term in row["terms"]
                if str(term["atom"]).lower() == target_atom.lower()
                and any(_shell_matches(str(term["ao"]), shell) for shell in shells)
            },
            key=lambda ao: (
                min(
                    component_shell_order[shell]
                    for shell in shells
                    if _shell_matches(ao, shell)
                ),
                ao.lower(),
            ),
        )
        component_candidates = []
        for component in component_labels:
            ranked = sorted(
                (
                    (_component_score(row, atom=target_atom, ao=component), int(row["mo"]), row)
                    for row in rows
                ),
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )
            if ranked and ranked[0][0] > 1.0e-4:
                component_candidates.append((component, ranked))
        component_candidates.sort(
            key=lambda item: (
                item[1][0][0] - (item[1][1][0] if len(item[1]) > 1 else 0.0),
                item[1][0][0],
            ),
            reverse=True,
        )
        selected: list[tuple[float, int, dict[str, Any], str]] = []
        used_mos: set[int] = set()
        for component, ranked in component_candidates:
            choice = next((item for item in ranked if item[1] not in used_mos), None)
            if choice is None:
                continue
            score, identity, row = choice
            selected.append((score, identity, row, component))
            used_mos.add(identity)
        selected = selected[: len(slots)]
        selected_ids = [item[1] for item in selected]
        source_groups.append(selected_ids)
        mapping = positions[symmetry]
        inverse = {identity: position for position, identity in mapping.items()}
        initial_occupants = list(slots)
        occupants = [mapping.get(slot, slot) for slot in slots]
        missing = [identity for identity in selected_ids if identity not in occupants]
        initially_present = [
            identity for identity in selected_ids if identity in initial_occupants
        ]
        present_after_existing_alter = [
            identity for identity in selected_ids if identity in occupants
        ]
        if selected_ids and len(initially_present) == len(selected_ids):
            residency_status = "already_resident"
            residency_note = (
                "Every intended target already occupies a final RAS3 slot before ALTER. "
                "No RAS3 homing swap is needed; any RAS3 SUPSYM group is only an identity "
                "lock during subsequent orbital optimization."
            )
        elif selected_ids and len(present_after_existing_alter) == len(selected_ids):
            residency_status = "homed_by_existing_alter"
            residency_note = (
                "The intended target reaches RAS3 through the existing ALTER map. Preserve "
                "that map when testing whether a RAS3 SUPSYM identity lock is necessary."
            )
        elif selected_ids:
            residency_status = "additional_homing_required"
            residency_note = (
                "One or more intended targets remain outside RAS3 after the existing ALTER "
                "map. Review the proposed homing swaps before interpreting transitions."
            )
        else:
            residency_status = "incomplete_target_selection"
            residency_note = (
                "The requested target-shell components could not be mapped completely to "
                "the available RAS3 slots."
            )
        replaceable_slots = [slot for slot in slots if mapping.get(slot, slot) not in selected_ids]
        additions: list[dict[str, Any]] = []
        for identity, target_slot in zip(missing, replaceable_slots):
            source_slot = inverse[identity]
            row = row_by_id[identity]
            occupied = float(row["occupation"]) > occupancy_threshold
            proposal = {
                "swap": [symmetry, source_slot, target_slot],
                "source_identity_mo": identity,
                "source_energy_hartree": float(row["energy"]),
                "source_occupation": float(row["occupation"]),
                "target_score": _target_score(row, atom=target_atom, shells=shells),
                "classification": (
                    "conditional_partition_change" if occupied else "virtual_homing"
                ),
            }
            additions.append(proposal)
            if occupied:
                conditional_additions.append(proposal["swap"])
            else:
                safe_additions.append(proposal["swap"])
            mapping[source_slot], mapping[target_slot] = mapping[target_slot], mapping[source_slot]
            inverse = {value: key for key, value in mapping.items()}

        close_pairs: list[dict[str, Any]] = []
        scored = sorted(
            (
                (_target_score(row, atom=target_atom, shells=shells), int(row["mo"]), row)
                for row in rows
            ),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        for score, identity, row, component in selected:
            neighbors = [
                other
                for other_score, other_id, other in scored
                if other_id not in selected_ids
                and abs(float(other["energy"]) - float(row["energy"])) <= mixing_window_ha
            ]
            if neighbors:
                closest = min(
                    neighbors,
                    key=lambda other: abs(float(other["energy"]) - float(row["energy"])),
                )
                close_pairs.append(
                    {
                        "selected_mo": identity,
                        "selected_target_score": score,
                        "target_component": component,
                        "neighbor_mo": int(closest["mo"]),
                        "delta_energy_hartree": abs(
                            float(closest["energy"]) - float(row["energy"])
                        ),
                        "neighbor_target_score": _target_score(
                            closest, atom=target_atom, shells=shells
                        ),
                    }
                )
        per_symmetry.append(
            {
                "symmetry": symmetry,
                "symmetry_label": rows[0].get("symmetry_label", ""),
                "ras3_slots": slots,
                "target_components": component_labels,
                "selected_source_mos": selected_ids,
                "initial_ras3_identity_mos_before_alter": initial_occupants,
                "ras3_residency_status": residency_status,
                "ras3_residency_note": residency_note,
                "selected_sources": [
                    {
                        "mo": identity,
                        "energy_hartree": float(row["energy"]),
                        "occupation": float(row["occupation"]),
                        "target_score": score,
                        "target_component": component,
                        "occupied_partition_change": (
                            float(row["occupation"]) > occupancy_threshold
                        ),
                    }
                    for score, identity, row, component in selected
                ],
                "current_ras3_identity_mos_after_existing_alter": occupants,
                "suggested_additions": additions,
                "close_energy_risks": close_pairs,
                "status": (
                    "complete"
                    if len(selected) == len(slots) == len(component_labels)
                    else "component_slot_mismatch"
                ),
            }
        )

    existing_groups = [
        [list(group) for group in groups] for groups in setup.supsym_groups
    ]
    while len(existing_groups) < max_symmetry:
        existing_groups.append([])
    existing_ras3_groups: list[list[list[int]]] = []
    mixing_preserving_groups: list[list[list[int]]] = []
    constraint_levels: list[str] = []
    for symmetry in range(max_symmetry):
        targets = set(source_groups[symmetry])
        matched = [
            list(group)
            for group in existing_groups[symmetry]
            if targets.intersection(group)
        ]
        covered = targets.intersection(value for group in matched for value in group)
        if not targets or not covered:
            level = "unconstrained"
        elif covered == targets:
            level = "fully_constrained"
        else:
            level = "partially_constrained"
        constraint_levels.append(level)
        existing_ras3_groups.append(matched)
        mixing_preserving_groups.append(
            [
                list(group)
                for group in existing_groups[symmetry]
                if not targets.intersection(group)
            ]
        )
    suggested_groups: list[list[list[int]]] = []
    for symmetry in range(max_symmetry):
        groups = [list(group) for group in existing_groups[symmetry]]
        # SUPSYM labels the input-orbital identities. ALTER then carries those
        # labels with the identities into their final active-space slots.
        target_group = source_groups[symmetry]
        if target_group and target_group not in groups:
            groups.append(target_group)
        suggested_groups.append(groups)

    alter_limit_exceeded = len(setup.alter_swaps) > OPENMOLCAS_MAXALTER
    return {
        **base,
        "status": (
            "invalid_alter_input"
            if alter_limit_exceeded
            else "review_required" if conditional_additions else "proposal_available"
        ),
        "alter_pair_count": len(setup.alter_swaps),
        "alter_pair_limit": OPENMOLCAS_MAXALTER,
        "alter_limit_exceeded": alter_limit_exceeded,
        "alter_limit_note": (
            "OpenMolcas 25.02 defines MAXALTER=16. Split a larger accepted permutation "
            "across sequential CIONLY homing blocks with at most 16 pairs each, pass the "
            "completed JobIph from one stage to the next, and define each later ALTER map "
            "against the post-previous-stage MO ordering. Do not optimize orbitals between "
            "homing stages. The final production probe should remove CIONLY and ALTER, then "
            "validate the active-space identity after real orbital optimization."
            if alter_limit_exceeded
            else "The setup ALTER pair count is within the OpenMolcas 25.02 limit."
        ),
        "target_atom": target_atom,
        "target_shells": shells,
        "ao_listing_format": ao_listing_format,
        "existing_alter_swaps": [list(swap) for swap in setup.alter_swaps],
        "safe_virtual_alter_additions": safe_additions,
        "conditional_occupied_alter_additions": conditional_additions,
        "occupied_source_note": (
            "An occupied source is not an automatic error: for an f^n ion it may need to enter "
            "the active space. Moving it from inactive/RAS2 into RAS3 changes the partition and "
            "must be checked against NACTEL, inactive counts, and the intended ground configuration."
        ),
        "per_symmetry": per_symmetry,
        "supsym": {
            "status": "review_required",
            "existing_groups": existing_groups,
            "existing_ras3_identity_groups": existing_ras3_groups,
            "ras3_constraint_levels": constraint_levels,
            "pre_alter_source_identity_groups": source_groups,
            "production_final_ras3_slot_groups": final_groups,
            "input_index_semantics": (
                "SUPSYM input indices are pre-ALTER source orbital identities; ALTER "
                "carries their labels to the final RAS3 slots."
            ),
            "alter_only_probe_block_preserving_existing_groups": _format_supsym_block(
                existing_groups
            ),
            "mixing_preserving_candidate_block": _format_supsym_block(
                mixing_preserving_groups
            ),
            "suggested_full_block_preserving_existing_groups": _format_supsym_block(
                suggested_groups
            ),
            "recommended_sequence": [
                "Apply the accepted ALTER homing swaps while preserving only existing "
                "SUPSYM constraints when physical metal-ligand mixing is desired.",
                "Run a short RASSCF-only probe and audit the final RAS3 AO character, "
                "occupations, roots, and convergence.",
                "Add the optional RAS3 source-identity SUPSYM group only if the probe "
                "shows destructive same-symmetry identity exchange or active-space drift.",
            ],
            "note": (
                "The optional identity-lock block labels pre-ALTER source identities. "
                "Confirm that the same labels appear at the reported final RAS3 slots after "
                "ALTER. A new RAS3 SUPSYM group is not required for homing and can suppress "
                "physical same-irrep metal-ligand mixing, so do not apply it solely because "
                "another orbital is near the LUMO. Existing RAS3 SUPSYM groups are an "
                "identity-locked control, not evidence that unconstrained active-external "
                "mixing would retain the same orbitals."
            ),
        },
    }


def _frontier_rows(
    rows: list[dict[str, Any]],
    *,
    occupied_count: int,
    virtual_count: int,
    occupancy_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    ordered = sorted(rows, key=lambda row: int(row["mo"]))
    occupied = [row for row in ordered if float(row["occupation"]) > occupancy_threshold]
    if not occupied:
        raise ValueError("No occupied orbitals were found with the requested occupancy threshold")
    homo = int(occupied[-1]["mo"])
    virtual = [
        row
        for row in ordered
        if int(row["mo"]) > homo and float(row["occupation"]) <= occupancy_threshold
    ]
    if not virtual:
        raise ValueError("No virtual orbitals were found after the highest occupied orbital")
    return occupied[-occupied_count:], virtual[:virtual_count], homo, int(virtual[0]["mo"])


def _decorate_orbital(
    row: dict[str, Any], *, frontier_label: str, top_aos: int, min_ao_weight: float
) -> dict[str, Any]:
    terms = list(row["terms"])
    coeff2_total = sum(float(term["coeff2"]) for term in terms) or 1.0
    decorated = []
    for term in terms:
        item = dict(term)
        item["display_weight_percent"] = 100.0 * float(item["coeff2"]) / coeff2_total
        decorated.append(item)
    decorated.sort(key=lambda term: float(term["display_weight_percent"]), reverse=True)
    kept = [term for term in decorated if float(term["display_weight_percent"]) >= min_ao_weight][
        :top_aos
    ]
    if not kept and decorated:
        kept = decorated[:1]
    result = {key: value for key, value in row.items() if key != "terms"}
    result["frontier_label"] = frontier_label
    result["ao_coeff2_total"] = coeff2_total
    result["dominant_aos"] = kept
    return result


def _build_diagnostic_from_parsed(
    inp_path: Path,
    output_path: Path,
    *,
    input_blocks: list[RasscfInputBlock],
    log_modules: list[RasscfLogModule],
    ras3_recommendation: dict[str, Any],
    symmetry: int = 1,
    reference_block: int | None = None,
    orbitals_each: int | None = None,
    occupied_orbitals: int = 6,
    virtual_orbitals: int = 10,
    top_aos: int = 6,
    min_ao_weight: float = 0.5,
    occupancy_threshold: float = 0.999,
) -> dict[str, Any]:
    """Build one structured diagnostic from previously parsed inputs."""

    reference, setup = select_reference_block(input_blocks, reference_block=reference_block)
    module, mapping_method = match_log_module(reference, log_modules)
    mo_rows = parse_full_mo_matrix(module.text, symmetry=symmetry)
    ao_listing_format = "full"
    if not mo_rows:
        mo_rows = parse_compact_mo_listing(module.text, symmetry=symmetry)
        ao_listing_format = "compact"
    if not mo_rows:
        raise ValueError(
            "No pseudonatural MO coefficient table was found for the selected symmetry. "
            "Ensure the reference RASSCF prints orbitals with ORBL ALL and ORBA FULL or COMP."
        )
    if orbitals_each is not None:
        occupied_orbitals = orbitals_each
        virtual_orbitals = orbitals_each
    occupied, virtual, homo, lumo = _frontier_rows(
        mo_rows,
        occupied_count=occupied_orbitals,
        virtual_count=virtual_orbitals,
        occupancy_threshold=occupancy_threshold,
    )
    occupied_items = []
    for offset, row in enumerate(reversed(occupied)):
        label = "HOMO" if offset == 0 else f"HOMO-{offset}"
        occupied_items.append(
            _decorate_orbital(
                row,
                frontier_label=label,
                top_aos=top_aos,
                min_ao_weight=min_ao_weight,
            )
        )
    occupied_items.reverse()
    virtual_items = []
    for offset, row in enumerate(virtual):
        label = "LUMO" if offset == 0 else f"LUMO+{offset}"
        virtual_items.append(
            _decorate_orbital(
                row,
                frontier_label=label,
                top_aos=top_aos,
                min_ao_weight=min_ao_weight,
            )
        )

    warnings: list[str] = []
    if ao_listing_format == "compact":
        warnings.append(
            "The selected block used OpenMolcas compact AO printing; coefficients below the "
            "print threshold are absent, and relative shares are normalized over printed terms only"
        )
    for block, role in [(reference, "reference"), (setup, "setup")]:
        if block is None:
            continue
        warning = _title_spin_warning(block, role)
        if warning:
            warnings.append(warning)
    if len(occupied_items) < occupied_orbitals:
        warnings.append(
            f"Only {len(occupied_items)} occupied orbitals were available; "
            f"{occupied_orbitals} requested"
        )
    if len(virtual_items) < virtual_orbitals:
        warnings.append(
            f"Only {len(virtual_items)} virtual orbitals were available; "
            f"{virtual_orbitals} requested"
        )
    if float(virtual_items[0]["energy"]) < float(occupied_items[-1]["energy"]):
        warnings.append(
            "The occupation-defined LUMO energy is below the HOMO energy; RASSCF active-space "
            "orbital order is not a canonical Aufbau energy ordering"
        )

    return {
        "schema": SCHEMA,
        "input_file": str(inp_path.resolve()),
        "output_file": str(output_path.resolve()),
        "selection_rule": (
            "symmetry-"
            f"{symmetry} orbital table from the final RASSCF block before first "
            "ALTER/SUPSYM setup"
            if reference_block is None
            else f"symmetry-{symmetry} orbital table from explicit input RASSCF block "
            f"{reference_block}"
        ),
        "reference_input_block": asdict(reference),
        "setup_input_block": asdict(setup) if setup else None,
        "matched_output_module": {
            "index": module.index,
            "start_line": module.start_line,
            "end_line": module.end_line,
            "active_electrons": module.active_electrons,
            "spin_quantum_number": module.spin_quantum_number,
            "state_symmetry": module.state_symmetry,
        },
        "mapping_method": mapping_method,
        "ao_listing": {
            "format": ao_listing_format,
            "complete_coefficient_matrix": ao_listing_format == "full",
        },
        "frontier": {
            "symmetry": symmetry,
            "symmetry_label": mo_rows[0].get("symmetry_label", ""),
            "occupation_threshold": occupancy_threshold,
            "definition": (
                "highest/lowest orbital index classified by occupation; values at or below "
                "the threshold, including partial occupations such as 0.5, are virtual-like"
            ),
            "homo": homo,
            "lumo": lumo,
            "occupied": occupied_items,
            "virtual": virtual_items,
        },
        "weight_definition": (
            "100*c_i^2/sum_j(c_j^2) over the AO coefficients present in the selected "
            f"{ao_listing_format} listing for each MO; "
            "diagnostic display share, not Mulliken or Loewdin population"
        ),
        "ras3_recommendation": ras3_recommendation,
        "warnings": warnings,
    }


def _reference_rows_for_all_symmetries(
    module: RasscfLogModule,
) -> tuple[dict[int, list[dict[str, Any]]], str]:
    rows_by_symmetry: dict[int, list[dict[str, Any]]] = {}
    formats: set[str] = set()
    for symmetry in available_mo_symmetries(module.text):
        rows = parse_full_mo_matrix(module.text, symmetry=symmetry)
        if rows:
            formats.add("full")
        else:
            rows = parse_compact_mo_listing(module.text, symmetry=symmetry)
            if rows:
                formats.add("compact")
        if rows:
            rows_by_symmetry[symmetry] = rows
    listing_format = "full" if formats == {"full"} else "compact" if formats == {"compact"} else "mixed"
    return rows_by_symmetry, listing_format


def build_diagnostic(
    inp_path: Path,
    output_path: Path,
    *,
    symmetry: int = 1,
    reference_block: int | None = None,
    orbitals_each: int | None = None,
    occupied_orbitals: int = 6,
    virtual_orbitals: int = 10,
    top_aos: int = 6,
    min_ao_weight: float = 0.5,
    occupancy_threshold: float = 0.999,
    mixing_window_ha: float = 0.15,
) -> dict[str, Any]:
    """Build a structured diagnostic for one symmetry."""

    input_text = _read_text(inp_path)
    input_blocks = parse_rasscf_input_blocks(input_text)
    log_modules = parse_rasscf_log_modules(_read_text(output_path))
    reference, setup = select_reference_block(input_blocks, reference_block=reference_block)
    module, _ = match_log_module(reference, log_modules)
    rows_by_symmetry, listing_format = _reference_rows_for_all_symmetries(module)
    recommendation = build_ras3_recommendation(
        input_text,
        inp_path,
        setup=setup,
        rows_by_symmetry=rows_by_symmetry,
        ao_listing_format=listing_format,
        occupancy_threshold=occupancy_threshold,
        mixing_window_ha=mixing_window_ha,
    )
    return _build_diagnostic_from_parsed(
        inp_path,
        output_path,
        input_blocks=input_blocks,
        log_modules=log_modules,
        ras3_recommendation=recommendation,
        symmetry=symmetry,
        reference_block=reference_block,
        orbitals_each=orbitals_each,
        occupied_orbitals=occupied_orbitals,
        virtual_orbitals=virtual_orbitals,
        top_aos=top_aos,
        min_ao_weight=min_ao_weight,
        occupancy_threshold=occupancy_threshold,
    )


def build_diagnostics(
    inp_path: Path,
    output_path: Path,
    *,
    symmetries: list[int] | None = None,
    reference_block: int | None = None,
    orbitals_each: int | None = None,
    occupied_orbitals: int = 6,
    virtual_orbitals: int = 10,
    top_aos: int = 6,
    min_ao_weight: float = 0.5,
    occupancy_threshold: float = 0.999,
    mixing_window_ha: float = 0.15,
) -> list[dict[str, Any]]:
    """Build diagnostics for all or selected symmetries with one file read."""

    input_text = _read_text(inp_path)
    input_blocks = parse_rasscf_input_blocks(input_text)
    log_modules = parse_rasscf_log_modules(_read_text(output_path))
    reference, setup = select_reference_block(input_blocks, reference_block=reference_block)
    module, _ = match_log_module(reference, log_modules)
    available = available_mo_symmetries(module.text)
    rows_by_symmetry, listing_format = _reference_rows_for_all_symmetries(module)
    recommendation = build_ras3_recommendation(
        input_text,
        inp_path,
        setup=setup,
        rows_by_symmetry=rows_by_symmetry,
        ao_listing_format=listing_format,
        occupancy_threshold=occupancy_threshold,
        mixing_window_ha=mixing_window_ha,
    )
    requested = available if symmetries is None else list(dict.fromkeys(symmetries))
    if not requested:
        raise ValueError("No MO symmetry tables were found in the selected reference module")
    invalid = [symmetry for symmetry in requested if symmetry < 1]
    if invalid:
        raise ValueError("--symmetry values must be positive")
    unavailable = [symmetry for symmetry in requested if symmetry not in available]
    if unavailable:
        choices = ", ".join(str(symmetry) for symmetry in available)
        missing = ", ".join(str(symmetry) for symmetry in unavailable)
        raise ValueError(
            f"No orbital table for symmetry {missing} in the selected reference module; "
            f"available: {choices}"
        )
    return [
        _build_diagnostic_from_parsed(
            inp_path,
            output_path,
            input_blocks=input_blocks,
            log_modules=log_modules,
            ras3_recommendation=recommendation,
            symmetry=symmetry,
            reference_block=reference_block,
            orbitals_each=orbitals_each,
            occupied_orbitals=occupied_orbitals,
            virtual_orbitals=virtual_orbitals,
            top_aos=top_aos,
            min_ao_weight=min_ao_weight,
            occupancy_threshold=occupancy_threshold,
        )
        for symmetry in requested
    ]


def _print_orbital(item: dict[str, Any]) -> None:
    print(
        f"  {item['frontier_label']:>7}  MO {item['mo']:>4}  "
        f"E={item['energy']:>10.4f} au  occ={item['occupation']:.4f}"
    )
    for term in item["dominant_aos"]:
        print(
            f"           AO {term['ao_index']:>4}  {term['atom']:<6} {term['ao']:<7} "
            f"c={term['coefficient']:+.4f}  rel|c|^2={term['display_weight_percent']:6.2f}%"
        )


def _print_ras3_recommendation(recommendation: dict[str, Any]) -> None:
    print("\nRAS3 intent and homing audit")
    print(f"Status: {recommendation['status']}")
    intent = recommendation.get("intent", {})
    if intent:
        print(
            f"Intent: {intent.get('status', 'unknown')} "
            f"({intent.get('evidence', 'no evidence')})"
        )
    if recommendation.get("target_atom"):
        shells = ", ".join(recommendation.get("target_shells", []))
        print(f"Target: {recommendation['target_atom']} shells [{shells}]")
    if recommendation.get("reason"):
        print(f"Reason: {recommendation['reason']}")
    if recommendation.get("alter_pair_count") is not None:
        print(
            f"ALTER pairs: {recommendation['alter_pair_count']} / "
            f"OpenMolcas limit {recommendation['alter_pair_limit']}"
        )
        if recommendation.get("alter_limit_exceeded"):
            print(f"ERROR: {recommendation['alter_limit_note']}")
    if recommendation.get("annotation_example"):
        print(f"Add explicit intent with: {recommendation['annotation_example']}")
    for item in recommendation.get("per_symmetry", []):
        slots = item.get("ras3_slots", [])
        if not slots:
            continue
        label = item.get("symmetry_label", "") or "unlabeled"
        print(
            f"- Symmetry {item['symmetry']} ({label}): RAS3 slots {slots}; "
            f"selected source MOs {item.get('selected_source_mos', [])}; "
            f"current identities {item.get('current_ras3_identity_mos_after_existing_alter', [])}"
        )
        if item.get("ras3_residency_status"):
            print(
                f"    residency: {item['ras3_residency_status']} - "
                f"{item.get('ras3_residency_note', '')}"
            )
        for source in item.get("selected_sources", []):
            marker = " OCCUPIED/PARTITION CHANGE" if source["occupied_partition_change"] else ""
            print(
                f"    source MO {source['mo']}: occ={source['occupation']:.4f}, "
                f"E={source['energy_hartree']:.4f} au, "
                f"component={source['target_component']}, "
                f"normalized printed-shell share={source['target_score']:.4f}"
                f"{marker}"
            )
        for proposal in item.get("suggested_additions", []):
            swap = " ".join(str(value) for value in proposal["swap"])
            print(f"    ALTER {swap}  [{proposal['classification']}]")
        for risk in item.get("close_energy_risks", []):
            print(
                f"    close-energy review: MO {risk['selected_mo']} vs "
                f"MO {risk['neighbor_mo']}, deltaE={risk['delta_energy_hartree']:.4f} au"
            )
    safe = recommendation.get("safe_virtual_alter_additions", [])
    conditional = recommendation.get("conditional_occupied_alter_additions", [])
    if safe:
        print("Safe virtual-homing ALTER additions")
        for swap in safe:
            print("  " + " ".join(str(value) for value in swap))
    if conditional:
        print("Conditional occupied-source ALTER additions - active-space review required")
        for swap in conditional:
            print("  " + " ".join(str(value) for value in swap))
        print("  " + recommendation["occupied_source_note"])
    supsym = recommendation.get("supsym")
    if supsym:
        levels = supsym.get("ras3_constraint_levels", [])
        constrained = [index + 1 for index, value in enumerate(levels) if value != "unconstrained"]
        if constrained:
            print(
                "Existing SUPSYM context: intended RAS3 source identities are constrained "
                f"in symmetries {constrained}"
            )
        else:
            print("Existing SUPSYM context: intended RAS3 source identities are unconstrained")
        print("Exact existing SUPSYM block")
        for line in supsym["alter_only_probe_block_preserving_existing_groups"]:
            print(f"  {line}")
        if constrained:
            print("Candidate probe without existing RAS3 identity locks")
            for line in supsym["mixing_preserving_candidate_block"]:
                print(f"  {line}")
        print("Optional RAS3 SUPSYM identity lock - use only after observed drift")
        for line in supsym["suggested_full_block_preserving_existing_groups"]:
            print(f"  {line}")
        print("  " + supsym["note"])
    print("Safety: " + recommendation["safety_note"])
    print("Transition-space guard: " + recommendation["transition_space_note"])


def print_diagnostic(
    result: dict[str, Any],
    *,
    show_heading: bool = True,
    show_paths: bool = True,
    show_ras3: bool = True,
) -> None:
    """Print a compact, human-readable diagnostic report."""

    reference = result["reference_input_block"]
    setup = result["setup_input_block"]
    module = result["matched_output_module"]
    frontier = result["frontier"]
    if show_heading:
        print("MOCHECK - OpenMolcas pre-ALTER/SUPSYM frontier audit")
    if show_paths:
        print(f"Input : {result['input_file']}")
        print(f"Output: {result['output_file']}")
    print(
        f"Reference input RASSCF #{reference['index']} "
        f"(lines {reference['start_line']}-{reference['end_line']}): {reference['title'] or '(untitled)'}"
    )
    if setup:
        flags = "/".join(
            name
            for name, enabled in [("ALTER", setup["has_alter"]), ("SUPSYM", setup["has_supsym"])]
            if enabled
        )
        print(
            f"First setup RASSCF     #{setup['index']} "
            f"(lines {setup['start_line']}-{setup['end_line']}): {flags}"
        )
    print(
        f"Matched output RASSCF  #{module['index']} "
        f"(lines {module['start_line']}-{module['end_line']})"
    )
    print(f"Mapping: {result['mapping_method']}")
    print(
        "AO listing: "
        f"{result['ao_listing']['format']} "
        f"(complete matrix={result['ao_listing']['complete_coefficient_matrix']})"
    )
    print(
        f"Reference-state signature: symmetry {module['state_symmetry']}, "
        f"S={module['spin_quantum_number']}, active electrons={module['active_electrons']}"
    )
    print(
        f"Orbital table: symmetry {frontier['symmetry']} "
        f"({frontier['symmetry_label'] or 'unlabeled'})"
    )
    print(
        f"Frontier: HOMO={frontier['homo']}, LUMO={frontier['lumo']}, "
        f"occupation threshold={frontier['occupation_threshold']:g}"
    )
    print("\nOccupied frontier orbitals")
    for item in frontier["occupied"]:
        _print_orbital(item)
    print("\nVirtual frontier orbitals")
    for item in frontier["virtual"]:
        _print_orbital(item)
    print("\nAO weight note: " + result["weight_definition"] + ".")
    if result["warnings"]:
        print("\nWarnings")
        for warning in result["warnings"]:
            print(f"- {warning}")
    if show_ras3:
        _print_ras3_recommendation(result["ras3_recommendation"])


def print_diagnostics(results: list[dict[str, Any]]) -> None:
    """Print one report or an all-symmetry report with stable separators."""

    if len(results) == 1:
        print_diagnostic(results[0])
        return
    labels = ", ".join(
        f"{result['frontier']['symmetry']} ({result['frontier']['symmetry_label']})"
        for result in results
    )
    print("MOCHECK - OpenMolcas all-symmetry pre-ALTER/SUPSYM frontier audit")
    print(f"Input : {results[0]['input_file']}")
    print(f"Output: {results[0]['output_file']}")
    print(f"Symmetries: {labels}")
    for result in results:
        frontier = result["frontier"]
        print("\n" + "=" * 78)
        print(f"SYMMETRY {frontier['symmetry']} ({frontier['symmetry_label'] or 'unlabeled'})")
        print_diagnostic(result, show_heading=False, show_paths=False, show_ras3=False)
    _print_ras3_recommendation(results[0]["ras3_recommendation"])


def build_moccheck_handoff(
    results: list[dict[str, Any]], inp_path: Path, output_path: Path
) -> dict[str, Any]:
    """Build the compact, provenance-checked baseline used by ``moccheckafter``."""

    if not results:
        raise ValueError("At least one moccheck diagnostic is required for a handoff")
    recommendation = results[0].get("ras3_recommendation", {})
    setup = results[0].get("setup_input_block")
    if setup is None:
        raise ValueError("A moccheck handoff requires an ALTER/SUPSYM setup block")
    targets = []
    for item in recommendation.get("per_symmetry", []):
        slots = [int(value) for value in item.get("ras3_slots", [])]
        sources = item.get("selected_sources", [])
        if not slots or not sources:
            continue
        targets.append(
            {
                "symmetry": int(item["symmetry"]),
                "symmetry_label": str(item.get("symmetry_label", "")),
                "final_ras3_slots": slots,
                "components": [
                    {
                        "ao": str(source["target_component"]),
                        "source_mo": int(source["mo"]),
                        "baseline_score": float(source["target_score"]),
                        "baseline_occupation": float(source["occupation"]),
                    }
                    for source in sources
                ],
                "setup_ras3_supsym_constraint": (
                    recommendation.get("supsym", {})
                    .get("ras3_constraint_levels", [])[int(item["symmetry"]) - 1]
                    if int(item["symmetry"]) - 1
                    < len(
                        recommendation.get("supsym", {}).get(
                            "ras3_constraint_levels", []
                        )
                    )
                    else "unknown"
                ),
                "baseline_ras3_residency_status": str(
                    item.get("ras3_residency_status", "unknown")
                ),
                "baseline_ras3_residency_note": str(
                    item.get("ras3_residency_note", "")
                ),
            }
        )
    stat = output_path.stat()
    return {
        "schema": HANDOFF_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(inp_path.resolve()),
            "sha256": _sha256(inp_path),
        },
        "output_snapshot": {
            "path": str(output_path.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "reference_input_block": results[0]["reference_input_block"],
        "setup_input_block": setup,
        "target_atom": recommendation.get("target_atom"),
        "target_shells": recommendation.get("target_shells", []),
        "ao_listing_format": recommendation.get("ao_listing_format"),
        "targets": targets,
        "interpretation_guard": (
            "Track the intended RAS3 subspace and AO components, not fixed pseudo-natural "
            "MO identities. Mixed metal-ligand character can be physical; SUPSYM is a "
            "fallback only after reproducible destructive identity drift. An orbital outside "
            "RAS3 is absent from the explicit RAS excitation manifold; SUPSYM does not add "
            "transitions or force occupations."
        ),
    }


def write_moccheck_handoff(
    results: list[dict[str, Any]], inp_path: Path, output_path: Path, handoff_path: Path
) -> dict[str, Any]:
    payload = build_moccheck_handoff(results, inp_path, output_path)
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moccheck",
        description=(
            "Inspect every symmetry table in the one RASSCF orbital set inherited by an "
            "OpenMolcas ALTER/SUPSYM setup block."
        ),
    )
    parser.add_argument("inp", type=Path, help="OpenMolcas .inp file")
    parser.add_argument(
        "output", type=Path, help="Matching OpenMolcas .log/.out file, optionally .gz"
    )
    parser.add_argument(
        "--symmetry",
        type=int,
        action="append",
        default=None,
        metavar="N",
        help=(
            "Target symmetry number; repeat for multiple symmetries. "
            "Default: all symmetry tables in the selected reference module"
        ),
    )
    parser.add_argument(
        "--reference-block",
        type=int,
        default=None,
        help="Use this 1-based input RASSCF block instead of automatic pre-ALTER/SUPSYM selection",
    )
    parser.add_argument(
        "--orbitals",
        type=int,
        default=None,
        help="Legacy symmetric override: show N occupied and N virtual orbitals",
    )
    parser.add_argument(
        "--occupied-orbitals",
        type=int,
        default=6,
        help="Occupied frontier orbitals to show (default: 6)",
    )
    parser.add_argument(
        "--virtual-orbitals",
        type=int,
        default=10,
        help="Virtual frontier orbitals to show (default: 10)",
    )
    parser.add_argument("--top-aos", type=int, default=6, help="Maximum dominant AOs shown per MO")
    parser.add_argument(
        "--min-ao-weight",
        type=float,
        default=0.5,
        help="Minimum relative coefficient-squared display share in percent",
    )
    parser.add_argument(
        "--occupancy-threshold",
        type=float,
        default=0.999,
        help=(
            "Occupation above which an orbital is treated as occupied; partial occupations "
            "such as 0.5 are virtual-like by default"
        ),
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="Optional structured diagnostic JSON"
    )
    parser.add_argument(
        "--handoff-out",
        type=Path,
        default=None,
        help="Optional compact baseline JSON for a later moccheckafter run",
    )
    parser.add_argument(
        "--mixing-window-ha",
        type=float,
        default=0.15,
        help="Energy window in Hartree for advisory same-symmetry mixing-risk flags",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.orbitals is not None and args.orbitals < 1:
        raise ValueError("--orbitals must be positive")
    if args.occupied_orbitals < 1:
        raise ValueError("--occupied-orbitals must be positive")
    if args.virtual_orbitals < 1:
        raise ValueError("--virtual-orbitals must be positive")
    if args.mixing_window_ha < 0:
        raise ValueError("--mixing-window-ha must be non-negative")
    if args.top_aos < 1:
        raise ValueError("--top-aos must be positive")
    results = build_diagnostics(
        args.inp,
        args.output,
        symmetries=args.symmetry,
        reference_block=args.reference_block,
        orbitals_each=args.orbitals,
        occupied_orbitals=args.occupied_orbitals,
        virtual_orbitals=args.virtual_orbitals,
        top_aos=args.top_aos,
        min_ao_weight=args.min_ao_weight,
        occupancy_threshold=args.occupancy_threshold,
        mixing_window_ha=args.mixing_window_ha,
    )
    print_diagnostics(results)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any]
        if len(results) == 1:
            payload = results[0]
        else:
            payload = {
                "schema": COLLECTION_SCHEMA,
                "input_file": results[0]["input_file"],
                "output_file": results[0]["output_file"],
                "symmetries": [result["frontier"]["symmetry"] for result in results],
                "diagnostics": results,
            }
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nWrote diagnostic JSON: {args.json_out}")
    if args.handoff_out is not None:
        write_moccheck_handoff(results, args.inp, args.output, args.handoff_out)
        print(f"Wrote moccheckafter handoff: {args.handoff_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
