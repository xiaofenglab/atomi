"""Human-readable OpenMolcas frontier-orbital diagnostics.

``moccheck`` pairs an OpenMolcas input deck with its output. It finds the
single last RASSCF reference immediately before the first ALTER/SUPSYM setup
block, verifies the corresponding output module, and reports occupied and
virtual orbitals from every symmetry table in that inherited orbital set.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "atomi.molcas_moccheck.v1"
COLLECTION_SCHEMA = "atomi.molcas_moccheck_collection.v1"
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


def _frontier_rows(
    rows: list[dict[str, Any]], *, count: int, occupancy_threshold: float
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
    return occupied[-count:], virtual[:count], homo, int(virtual[0]["mo"])


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
    symmetry: int = 1,
    reference_block: int | None = None,
    orbitals_each: int = 6,
    top_aos: int = 6,
    min_ao_weight: float = 0.5,
    occupancy_threshold: float = 1.0e-3,
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
    occupied, virtual, homo, lumo = _frontier_rows(
        mo_rows, count=orbitals_each, occupancy_threshold=occupancy_threshold
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
    if len(occupied_items) < orbitals_each:
        warnings.append(
            f"Only {len(occupied_items)} occupied orbitals were available; {orbitals_each} requested"
        )
    if len(virtual_items) < orbitals_each:
        warnings.append(
            f"Only {len(virtual_items)} virtual orbitals were available; {orbitals_each} requested"
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
            "definition": "highest/lowest orbital index classified by occupation, not canonical energy",
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
        "warnings": warnings,
    }


def build_diagnostic(
    inp_path: Path,
    output_path: Path,
    *,
    symmetry: int = 1,
    reference_block: int | None = None,
    orbitals_each: int = 6,
    top_aos: int = 6,
    min_ao_weight: float = 0.5,
    occupancy_threshold: float = 1.0e-3,
) -> dict[str, Any]:
    """Build a structured diagnostic for one symmetry."""

    input_blocks = parse_rasscf_input_blocks(_read_text(inp_path))
    log_modules = parse_rasscf_log_modules(_read_text(output_path))
    return _build_diagnostic_from_parsed(
        inp_path,
        output_path,
        input_blocks=input_blocks,
        log_modules=log_modules,
        symmetry=symmetry,
        reference_block=reference_block,
        orbitals_each=orbitals_each,
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
    orbitals_each: int = 6,
    top_aos: int = 6,
    min_ao_weight: float = 0.5,
    occupancy_threshold: float = 1.0e-3,
) -> list[dict[str, Any]]:
    """Build diagnostics for all or selected symmetries with one file read."""

    input_blocks = parse_rasscf_input_blocks(_read_text(inp_path))
    log_modules = parse_rasscf_log_modules(_read_text(output_path))
    reference, _ = select_reference_block(input_blocks, reference_block=reference_block)
    module, _ = match_log_module(reference, log_modules)
    available = available_mo_symmetries(module.text)
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
            symmetry=symmetry,
            reference_block=reference_block,
            orbitals_each=orbitals_each,
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


def print_diagnostic(
    result: dict[str, Any], *, show_heading: bool = True, show_paths: bool = True
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
        print_diagnostic(result, show_heading=False, show_paths=False)


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
        "--orbitals", type=int, default=6, help="Occupied and virtual orbitals to show"
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
        default=1.0e-3,
        help="Occupation above which an orbital is treated as occupied",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="Optional structured diagnostic JSON"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.orbitals < 1:
        raise ValueError("--orbitals must be positive")
    if args.top_aos < 1:
        raise ValueError("--top-aos must be positive")
    results = build_diagnostics(
        args.inp,
        args.output,
        symmetries=args.symmetry,
        reference_block=args.reference_block,
        orbitals_each=args.orbitals,
        top_aos=args.top_aos,
        min_ao_weight=args.min_ao_weight,
        occupancy_threshold=args.occupancy_threshold,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
