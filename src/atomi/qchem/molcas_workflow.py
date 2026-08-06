"""Frozen OpenMolcas postanalysis data and publication renderers.

``mocparse`` is the scientific boundary: it parses an input/output pair once,
freezes the reusable tables with source hashes, and records explicit caveats.
``mocxanes`` and ``mocmo`` verify and consume that bundle without reopening the
large OpenMolcas output.  This keeps later style revisions separate from state,
transition, and orbital selection.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from atomi.analysis.plot_dataset import (
    ColumnSpec,
    PlotDataset,
    SeriesSpec,
    TableSpec,
    ViewSpec,
    freeze_dataset,
    load_manifest,
)
from atomi.core.run_record import RunArtifact
from atomi.qchem.molcas_live import parse_input_plan, parse_output_stream
from atomi.qchem.molcas_postanalysis import parse_molcas_ao_sections
from atomi.viz.style import require_pyplot, save_figure
from atomi.xafs.molcas_xanes_spectrum import (
    HARTREE_TO_EV,
    Transition,
    broaden,
    xraydb_metadata,
)


MOLCAS_BUNDLE_SCHEMA = "atomi.molcas_postanalysis_bundle.v1"
MOCXANES_SCHEMA = "atomi.mocxanes_render.v1"
MOCMO_SCHEMA = "atomi.mocmo_render.v1"
POLLY_GAUSSIAN_FWHM_EV = 1.2894947
POLLY_LORENTZIAN_FWHM_EV = 0.3621724


@dataclass(frozen=True)
class EdgeSpec:
    label: str
    emin_ev: float
    emax_ev: float


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return cleaned or "molcas-run"


def _float(text: str) -> float:
    return float(text.replace("D", "E"))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _runtime_status(block: Any) -> str:
    if block.return_code and block.return_code != "_RC_ALL_IS_WELL_":
        return "failed"
    if block.error_lines and block.return_code != "_RC_ALL_IS_WELL_":
        return "failed"
    if block.return_code == "_RC_ALL_IS_WELL_" or block.saw_spent:
        return "finished"
    return "running"


def _parse_module_rows(inp_text: str | None, lines: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any], list[Any]]:
    runtime, metadata = parse_output_stream(lines)
    plan = parse_input_plan(inp_text) if inp_text else []
    runtime_by_key = {(item.kind, item.kind_index): item for item in runtime}
    plan_by_key = {(item.kind, item.kind_index): item for item in plan}
    keys = [(item.kind, item.kind_index) for item in plan]
    keys.extend(key for key in runtime_by_key if key not in plan_by_key)
    rows: list[dict[str, Any]] = []
    for sequence, key in enumerate(keys, start=1):
        planned = plan_by_key.get(key)
        observed = runtime_by_key.get(key)
        rows.append(
            {
                "sequence": sequence,
                "kind": key[0],
                "kind_index": key[1],
                "title": planned.title if planned else "",
                "symmetry": planned.symmetry if planned and planned.symmetry is not None else (observed.state_symmetry if observed else ""),
                "spin_multiplicity": planned.spin_multiplicity if planned and planned.spin_multiplicity is not None else "",
                "active_electrons": planned.active_electrons if planned and planned.active_electrons is not None else (observed.active_electrons if observed else ""),
                "expected_roots": planned.expected_roots if planned and planned.expected_roots is not None else (observed.roots_requested if observed else ""),
                "printed_roots": len(observed.roots) if observed else 0,
                "status": _runtime_status(observed) if observed else "pending",
                "return_code": observed.return_code if observed and observed.return_code else "",
                "elapsed_seconds": observed.elapsed_seconds if observed and observed.elapsed_seconds is not None else "",
                "start_line": observed.start_line if observed else "",
                "end_line": observed.end_line if observed and observed.end_line is not None else "",
                "stage": observed.stage if observed else "pending",
                "has_alter": bool(planned.has_alter) if planned else False,
                "has_supsym": bool(planned.has_supsym) if planned else False,
                "has_cionly": bool(planned.has_cionly) if planned else False,
                "has_tdm": bool(planned.has_tdm) if planned else False,
                "error_count": len(observed.error_lines) if observed else 0,
                "first_error": observed.error_lines[0] if observed and observed.error_lines else "",
            }
        )
    return rows, metadata, runtime


def _parse_spin_free_sections(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    section_index = -1
    row_re = re.compile(
        r"^\s*(\d+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)"
    )
    for idx, line in enumerate(lines):
        if "SPIN-FREE ENERGIES:" not in line:
            continue
        section_index += 1
        found = False
        quiet = 0
        for line_idx in range(idx + 1, len(lines)):
            raw = lines[line_idx]
            match = row_re.match(raw)
            if match:
                found = True
                quiet = 0
                rows.append(
                    {
                        "rassi_section": section_index,
                        "line": line_idx + 1,
                        "state": int(match.group(1)),
                        "relative_energy_au": _float(match.group(2)),
                        "relative_energy_ev": _float(match.group(3)),
                        "wavenumber_cm1": _float(match.group(4)),
                        "abs_m": _float(match.group(5)),
                    }
                )
                continue
            if found and raw.strip():
                quiet += 1
                if quiet >= 3:
                    break
    return rows


def _parse_so_sections(lines: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    state_rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    marker_lines: list[int] = []
    marker_re = re.compile(r"SO State\s+Total energy\s*\(au\)\s+Spin-free states, spin, and weights", re.I)
    row_re = re.compile(r"^\s*(\d+)\s+([-+0-9.EeDd]+)\s+(.*)$")
    triple_re = re.compile(r"(\d+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)")
    for idx, line in enumerate(lines):
        if not marker_re.search(line):
            continue
        section_index = len(marker_lines)
        marker_lines.append(idx + 1)
        local_states: list[dict[str, Any]] = []
        for line_idx in range(idx + 1, len(lines)):
            raw = lines[line_idx]
            if local_states and re.match(r"^\s*-{10,}\s*$", raw):
                break
            match = row_re.match(raw)
            if not match:
                continue
            so_state = int(match.group(1))
            energy_au = _float(match.group(2))
            local_states.append(
                {
                    "rassi_section": section_index,
                    "line": line_idx + 1,
                    "state": so_state,
                    "energy_au": energy_au,
                }
            )
            for rank, parent in enumerate(triple_re.finditer(match.group(3)), start=1):
                parent_rows.append(
                    {
                        "rassi_section": section_index,
                        "so_state": so_state,
                        "parent_rank": rank,
                        "spin_free_state": int(parent.group(1)),
                        "spin": _float(parent.group(2)),
                        "weight": _float(parent.group(3)),
                    }
                )
        if local_states:
            minimum = min(float(row["energy_au"]) for row in local_states)
            for row in local_states:
                row["relative_energy_ev"] = (float(row["energy_au"]) - minimum) * HARTREE_TO_EV
                state_rows.append(row)
    return state_rows, parent_rows, marker_lines


def _nearest_section(line_number: int, marker_lines: Sequence[int]) -> int:
    candidates = [idx for idx, marker in enumerate(marker_lines) if marker <= line_number]
    return candidates[-1] if candidates else -1


def _parse_transition_rows(
    lines: list[str],
    so_states: Sequence[dict[str, Any]],
    spin_free_states: Sequence[dict[str, Any]],
    so_marker_lines: Sequence[int],
) -> list[dict[str, Any]]:
    so_energy = {
        (int(row["rassi_section"]), int(row["state"])): float(row["energy_au"])
        for row in so_states
    }
    sf_energy = {
        (int(row["rassi_section"]), int(row["state"])): float(row["relative_energy_au"])
        for row in spin_free_states
    }
    marker_re = re.compile(r"^\s*\+\+\s*(.*?)transition strengths(?:\s*\((.*?)\))?\s*:", re.I)
    row_re = re.compile(r"^\s*(\d+)\s+(\d+)\s+([-+0-9.EeDd]+)\b")
    rows: list[dict[str, Any]] = []
    table_index = 0
    for idx, line in enumerate(lines):
        marker = marker_re.match(line)
        if not marker:
            continue
        table_index += 1
        title = line.strip()
        lower = title.lower()
        basis = "so" if "so states" in lower else "spin-free" if "spin-free states" in lower else "unknown"
        gauge = "velocity" if "velocity" in lower else "length" if "dipole" in lower else "second-order" if "second-order" in lower else "unknown"
        rassi_section = _nearest_section(idx + 1, so_marker_lines)
        found = False
        quiet = 0
        for line_idx in range(idx + 1, len(lines)):
            raw = lines[line_idx]
            if line_idx > idx + 1 and raw.strip().startswith("++"):
                break
            match = row_re.match(raw)
            if match:
                found = True
                quiet = 0
                state_from = int(match.group(1))
                state_to = int(match.group(2))
                energy_ev: float | str = ""
                if basis == "so":
                    e0 = so_energy.get((rassi_section, state_from))
                    e1 = so_energy.get((rassi_section, state_to))
                elif basis == "spin-free":
                    e0 = sf_energy.get((rassi_section, state_from))
                    e1 = sf_energy.get((rassi_section, state_to))
                else:
                    e0 = e1 = None
                if e0 is not None and e1 is not None:
                    energy_ev = (e1 - e0) * HARTREE_TO_EV
                rows.append(
                    {
                        "rassi_section": rassi_section,
                        "transition_table": table_index,
                        "line": line_idx + 1,
                        "state_basis": basis,
                        "gauge": gauge,
                        "state_from": state_from,
                        "state_to": state_to,
                        "oscillator_strength": _float(match.group(3)),
                        "energy_ev": energy_ev,
                        "source_title": title,
                    }
                )
                continue
            if found and raw.strip():
                quiet += 1
                if quiet >= 4:
                    break
    return rows


def _containing_rasscf(section_line: int, runtime: Sequence[Any]) -> Any | None:
    matches = [
        block
        for block in runtime
        if block.kind == "rasscf"
        and block.start_line <= section_line
        and (block.end_line is None or section_line <= block.end_line)
    ]
    return matches[-1] if matches else None


def _parse_orbital_rows(text: str, runtime: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    orbital_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    sections = parse_molcas_ao_sections(text, section_kind="all")
    for section in sections:
        block = _containing_rasscf(int(section["start_line"]), runtime)
        section_label = f"{section['kind']}_{section['source_index']}"
        if block is not None:
            section_label = f"rasscf_{block.kind_index}_{section_label}"
        for row in section["rows"]:
            # Numeric state/energy tables can follow an MO section before the
            # next ``++`` marker.  They are not orbitals unless an AO
            # coefficient record was actually printed.
            if not row["terms"]:
                continue
            terms = sorted(row["terms"], key=lambda item: float(item["abs_coefficient"]), reverse=True)
            compact = "; ".join(
                f"{term['label']} ({float(term['coefficient']):+.3f})" for term in terms[:8]
            )
            orbital_rows.append(
                {
                    "section_index": section["source_index"],
                    "section_label": section_label,
                    "section_kind": section["kind"],
                    "rasscf_index": block.kind_index if block is not None else "",
                    "symmetry": block.state_symmetry if block is not None and block.state_symmetry is not None else "",
                    "symmetry_species": row.get("symmetry_species") if row.get("symmetry_species") is not None else "",
                    "symmetry_label": row.get("symmetry_label") or "",
                    "spin_quantum_number": block.spin_quantum_number if block is not None and block.spin_quantum_number is not None else "",
                    "line": row["line"],
                    "mo": row["mo"],
                    "energy_au": row["energy"],
                    "energy_ev": float(row["energy"]) * HARTREE_TO_EV,
                    "occupation_electrons": row["occupation"],
                    "dominant_ao": terms[0]["label"] if terms else "",
                    "ao_composition": compact,
                    "printed_ao_components": len(terms),
                }
            )
            for rank, term in enumerate(terms, start=1):
                component_rows.append(
                    {
                        "section_index": section["source_index"],
                        "section_label": section_label,
                        "symmetry_species": row.get("symmetry_species") if row.get("symmetry_species") is not None else "",
                        "symmetry_label": row.get("symmetry_label") or "",
                        "mo": row["mo"],
                        "component_rank": rank,
                        "ao_index": term["ao_index"],
                        "atom": term["atom"],
                        "element": term["element"],
                        "ao": term["ao"],
                        "principal_n": term["principal_n"],
                        "shell": term["shell"],
                        "angular_label": term["angular_label"],
                        "coefficient": term["coefficient"],
                        "coefficient_squared": term["coeff2"],
                    }
                )
    return orbital_rows, component_rows


def _column(
    name: str,
    dtype: str,
    role: str,
    quantity: str = "",
    unit: str = "",
    description: str = "",
) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        dtype=dtype,
        role=role,
        quantity=quantity,
        unit=unit,
        description=description,
    )


TABLE_DEFINITIONS: dict[str, tuple[tuple[str, ...], tuple[ColumnSpec, ...], str]] = {
    "modules": (
        (
            "sequence", "kind", "kind_index", "title", "symmetry", "spin_multiplicity",
            "active_electrons", "expected_roots", "printed_roots", "status", "return_code",
            "elapsed_seconds", "start_line", "end_line", "stage", "has_alter", "has_supsym",
            "has_cionly", "has_tdm", "error_count", "first_error",
        ),
        (
            _column("sequence", "int", "category", "dimensionless"),
            _column("kind", "str", "category"),
            _column("kind_index", "int", "category", "dimensionless"),
            _column("title", "str", "label"),
            _column("symmetry", "int", "category", "dimensionless"),
            _column("spin_multiplicity", "int", "category", "dimensionless"),
            _column("active_electrons", "int", "auxiliary", "dimensionless"),
            _column("expected_roots", "int", "auxiliary", "dimensionless"),
            _column("printed_roots", "int", "auxiliary", "dimensionless"),
            _column("status", "str", "status"),
            _column("return_code", "str", "status"),
            _column("elapsed_seconds", "float", "auxiliary", "time", "s"),
            _column("start_line", "int", "auxiliary", "dimensionless"),
            _column("end_line", "int", "auxiliary", "dimensionless"),
            _column("stage", "str", "status"),
            _column("has_alter", "bool", "auxiliary"),
            _column("has_supsym", "bool", "auxiliary"),
            _column("has_cionly", "bool", "auxiliary"),
            _column("has_tdm", "bool", "auxiliary"),
            _column("error_count", "int", "auxiliary", "dimensionless"),
            _column("first_error", "str", "auxiliary"),
        ),
        "Input-scoped RASSCF/CASPT2/RASSI schedule and numerical status.",
    ),
    "spin_free_states": (
        ("rassi_section", "line", "state", "relative_energy_au", "relative_energy_ev", "wavenumber_cm1", "abs_m"),
        (
            _column("rassi_section", "int", "category", "dimensionless"),
            _column("line", "int", "auxiliary", "dimensionless"),
            _column("state", "int", "category", "dimensionless"),
            _column("relative_energy_au", "float", "auxiliary", "energy", "Eh"),
            _column("relative_energy_ev", "float", "y", "energy", "eV"),
            _column("wavenumber_cm1", "float", "auxiliary", "wavenumber", "cm^-1"),
            _column("abs_m", "float", "auxiliary", "dimensionless"),
        ),
        "Spin-free many-electron state energies printed by RASSI.",
    ),
    "spin_orbit_states": (
        ("rassi_section", "line", "state", "energy_au", "relative_energy_ev"),
        (
            _column("rassi_section", "int", "category", "dimensionless"),
            _column("line", "int", "auxiliary", "dimensionless"),
            _column("state", "int", "category", "dimensionless"),
            _column("energy_au", "float", "auxiliary", "energy", "Eh"),
            _column("relative_energy_ev", "float", "y", "energy", "eV"),
        ),
        "RASSI spin-orbit state energies relative to the section minimum.",
    ),
    "so_parent_weights": (
        ("rassi_section", "so_state", "parent_rank", "spin_free_state", "spin", "weight"),
        (
            _column("rassi_section", "int", "category", "dimensionless"),
            _column("so_state", "int", "category", "dimensionless"),
            _column("parent_rank", "int", "category", "dimensionless"),
            _column("spin_free_state", "int", "category", "dimensionless"),
            _column("spin", "float", "category", "dimensionless"),
            _column("weight", "float", "y", "dimensionless", description="Squared RASSI spin-free parent coefficient."),
        ),
        "Largest printed spin-free parent weights for each RASSI SO state.",
    ),
    "transitions": (
        (
            "rassi_section", "transition_table", "line", "state_basis", "gauge", "state_from",
            "state_to", "oscillator_strength", "energy_ev", "source_title",
        ),
        (
            _column("rassi_section", "int", "category", "dimensionless"),
            _column("transition_table", "int", "category", "dimensionless"),
            _column("line", "int", "auxiliary", "dimensionless"),
            _column("state_basis", "str", "category"),
            _column("gauge", "str", "category"),
            _column("state_from", "int", "category", "dimensionless"),
            _column("state_to", "int", "category", "dimensionless"),
            _column("oscillator_strength", "float", "y", "dimensionless"),
            _column("energy_ev", "float", "x", "energy", "eV"),
            _column("source_title", "str", "label"),
        ),
        "All parsed transition-strength rows; no edge, state, gauge, or intensity filtering.",
    ),
    "orbitals": (
        (
            "section_index", "section_label", "section_kind", "rasscf_index", "symmetry",
            "symmetry_species", "symmetry_label", "spin_quantum_number", "line", "mo", "energy_au", "energy_ev",
            "occupation_electrons", "dominant_ao", "ao_composition", "printed_ao_components",
        ),
        (
            _column("section_index", "int", "category", "dimensionless"),
            _column("section_label", "str", "category"),
            _column("section_kind", "str", "category"),
            _column("rasscf_index", "int", "category", "dimensionless"),
            _column("symmetry", "int", "category", "dimensionless"),
            _column("symmetry_species", "int", "category", "dimensionless"),
            _column("symmetry_label", "str", "category"),
            _column("spin_quantum_number", "float", "category", "dimensionless"),
            _column("line", "int", "auxiliary", "dimensionless"),
            _column("mo", "int", "x", "dimensionless"),
            _column("energy_au", "float", "auxiliary", "energy", "Eh"),
            _column("energy_ev", "float", "y", "energy", "eV"),
            _column("occupation_electrons", "float", "auxiliary", "electron_count", "electron"),
            _column("dominant_ao", "str", "label"),
            _column("ao_composition", "str", "auxiliary"),
            _column("printed_ao_components", "int", "auxiliary", "dimensionless"),
        ),
        "Printed molecular/pseudonatural orbital energies, occupations, and AO summaries.",
    ),
    "ao_components": (
        (
            "section_index", "section_label", "symmetry_species", "symmetry_label", "mo", "component_rank", "ao_index", "atom",
            "element", "ao", "principal_n", "shell", "angular_label", "coefficient", "coefficient_squared",
        ),
        (
            _column("section_index", "int", "category", "dimensionless"),
            _column("section_label", "str", "category"),
            _column("symmetry_species", "int", "category", "dimensionless"),
            _column("symmetry_label", "str", "category"),
            _column("mo", "int", "category", "dimensionless"),
            _column("component_rank", "int", "category", "dimensionless"),
            _column("ao_index", "int", "category", "dimensionless"),
            _column("atom", "str", "category"),
            _column("element", "str", "category"),
            _column("ao", "str", "label"),
            _column("principal_n", "str", "auxiliary"),
            _column("shell", "str", "auxiliary"),
            _column("angular_label", "str", "auxiliary"),
            _column("coefficient", "float", "y", "dimensionless"),
            _column("coefficient_squared", "float", "auxiliary", "dimensionless"),
        ),
        "All AO coefficients printed for each parsed MO row.",
    ),
}


def build_molcas_bundle(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    inp = args.inp.resolve() if args.inp else None
    text = _read_text(output)
    lines = text.splitlines()
    inp_text = _read_text(inp) if inp else None
    modules, stream_meta, runtime = _parse_module_rows(inp_text, lines)
    spin_free = _parse_spin_free_sections(lines)
    spin_orbit, parent_weights, so_markers = _parse_so_sections(lines)
    transitions = _parse_transition_rows(lines, spin_orbit, spin_free, so_markers)
    orbitals, ao_components = _parse_orbital_rows(text, runtime)
    rows_by_table = {
        "modules": modules,
        "spin_free_states": spin_free,
        "spin_orbit_states": spin_orbit,
        "so_parent_weights": parent_weights,
        "transitions": transitions,
        "orbitals": orbitals,
        "ao_components": ao_components,
    }
    available = {key: rows for key, rows in rows_by_table.items() if rows}
    if not available:
        raise ValueError(f"No supported MOLCAS postanalysis tables were found in {output}")

    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f"MOLCAS data bundle already exists: {outdir}; use a new revision directory.")
    with tempfile.TemporaryDirectory(prefix="mocparse-") as temp:
        tempdir = Path(temp)
        tables: list[TableSpec] = []
        for table_id, rows in available.items():
            fieldnames, columns, scope = TABLE_DEFINITIONS[table_id]
            source = tempdir / f"{table_id}.csv"
            _write_csv(source, rows, fieldnames)
            tables.append(
                TableSpec(
                    id=table_id,
                    path=str(source),
                    columns=columns,
                    row_key=(fieldnames[0],),
                    row_scope=scope,
                )
            )

        if "transitions" in available:
            view = ViewSpec(
                id="all-transitions",
                series=(SeriesSpec(id="all-transitions", table="transitions", x="energy_ev", y="oscillator_strength", label="Parsed transition strengths"),),
                xlabel="Excitation energy (eV)",
                ylabel="Oscillator strength",
                title="Unfiltered MOLCAS transition data",
                permitted_render_transforms=("axis_limits", "panel_layout", "legend_position"),
                forbidden_render_transforms=("infer_edge", "infer_initial_states", "silent_energy_shift", "silent_broadening", "silent_normalization"),
            )
        elif "spin_orbit_states" in available:
            view = ViewSpec(
                id="spin-orbit-levels",
                series=(SeriesSpec(id="spin-orbit-levels", table="spin_orbit_states", x="state", y="relative_energy_ev", label="SO states"),),
                xlabel="SO state index",
                ylabel="Energy relative to section minimum (eV)",
            )
        else:
            view = ViewSpec(
                id="orbital-levels",
                series=(SeriesSpec(id="orbital-levels", table="orbitals", x="mo", y="energy_ev", label="Printed orbitals"),),
                xlabel="MO index",
                ylabel="Orbital energy (eV)",
            )

        final_rassi = max((int(row["rassi_section"]) for row in spin_orbit), default=-1)
        failed_modules = [row["sequence"] for row in modules if row["status"] == "failed"]
        running_modules = [row["sequence"] for row in modules if row["status"] == "running"]
        sources = [RunArtifact(path=str(output), role="openmolcas_output")]
        if inp:
            sources.append(RunArtifact(path=str(inp), role="openmolcas_input"))
        dataset = PlotDataset(
            dataset_id=_safe_id(args.dataset_id or output.stem),
            revision=args.revision,
            project=args.project or output.parent.name,
            chart_family="molcas-postanalysis",
            scientific_question=args.scientific_question,
            science_owner=args.science_owner,
            canonical_report=args.canonical_report or "not-assigned",
            scientific_status=args.scientific_status,
            quality=args.quality,
            tables=tables,
            views=[view],
            sources=sources,
            transformations=[
                "Parse printed OpenMolcas RASSCF/CASPT2/RASSI tables without state or intensity filtering.",
                "Convert Hartree energy differences to eV using 27.211386245988 eV/Eh.",
            ],
            normalization="none",
            uncertainty={"meaning": "No numerical or experimental uncertainty inferred from the output."},
            caveats=[
                "Scientific acceptance remains with the MOLCAS project lead; parsing success is not physical validation.",
                "RASSI parent weights are squared many-electron state-mixing coefficients, not AO percentages or one-electron MO mixing.",
                "Only orbital coefficients printed in the OpenMolcas output are available; missing ORBL/ORBA output cannot be reconstructed.",
                "Edge identity, initial-state averaging, energy alignment, broadening, and normalization are intentionally deferred to mocxanes.",
            ],
            metadata={
                "molcas_parse_schema": MOLCAS_BUNDLE_SCHEMA,
                "source_output": str(output),
                "source_input": str(inp) if inp else "",
                "output_happy_landing": bool(stream_meta.get("happy_landing")),
                "global_errors": list(stream_meta.get("global_errors") or []),
                "failed_module_sequences": failed_modules,
                "running_module_sequences": running_modules,
                "final_rassi_section": final_rassi,
                "table_row_counts": {key: len(rows) for key, rows in available.items()},
                "table_semantics": {
                    "transitions": "unfiltered; renderer must select RASSI section, state basis, gauge, initial states, and edge windows",
                    "so_parent_weights": "weight is a squared RASSI parent coefficient",
                    "orbitals": "one-electron printed MO/pseudonatural data; not SO-state orbitals",
                },
            },
        )
        manifest = freeze_dataset(dataset, outdir)
    return manifest


def _bundle_table(manifest_path: Path, payload: dict[str, Any], table_id: str) -> Path:
    table = next((item for item in payload.get("tables", []) if item.get("id") == table_id), None)
    if table is None:
        raise ValueError(f"MOLCAS bundle does not contain required table {table_id!r}")
    return (manifest_path.resolve().parent / str(table["path"])).resolve()


def _load_molcas_bundle(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = load_manifest(manifest_path.resolve(), verify=True)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid plot-data bundle manifest: {manifest_path}") from exc
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("molcas_parse_schema") != MOLCAS_BUNDLE_SCHEMA:
        raise ValueError(f"Not a {MOLCAS_BUNDLE_SCHEMA} bundle: {manifest_path}")
    return payload


def _parse_ints(text: str) -> list[int]:
    values = [int(item) for item in re.split(r"[\s,]+", text.strip()) if item]
    if not values:
        raise ValueError("At least one state index is required")
    return values


def _select_section(rows: Sequence[dict[str, str]], policy: str) -> int:
    sections = sorted({int(row["rassi_section"]) for row in rows if str(row.get("rassi_section", "")).strip()})
    if not sections:
        raise ValueError("No RASSI section indices are present in the requested table")
    if policy == "last":
        return sections[-1]
    if policy == "first":
        return sections[0]
    section = int(policy)
    if section not in sections:
        raise ValueError(f"RASSI section {section} is unavailable; choices: {sections}")
    return section


def _parse_edge_spec(text: str) -> EdgeSpec:
    parts = text.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Edge must be LABEL:EMIN:EMAX, got {text!r}")
    label, emin, emax = parts
    spec = EdgeSpec(label=label.strip(), emin_ev=float(emin), emax_ev=float(emax))
    if not spec.label or spec.emax_ev <= spec.emin_ev:
        raise ValueError(f"Invalid edge specification: {text!r}")
    return spec


def _aggregate_transitions(
    rows: Sequence[dict[str, str]],
    *,
    rassi_section: int,
    gauge: str,
    initial_states: Sequence[int],
    aggregation: str,
) -> list[Transition]:
    selected = [
        row for row in rows
        if int(row["rassi_section"]) == rassi_section
        and row.get("state_basis") == "so"
        and (gauge == "any" or row.get("gauge") == gauge)
        and int(row["state_from"]) in initial_states
    ]
    missing_energy = [row for row in selected if not str(row.get("energy_ev", "")).strip()]
    if missing_energy:
        raise ValueError(
            f"Selected transition scope contains {len(missing_energy)} rows without excitation energies"
        )
    negative = [row for row in selected if float(row["oscillator_strength"]) < -1.0e-12]
    if negative:
        raise ValueError(
            f"Selected transition scope contains {len(negative)} negative oscillator strengths"
        )
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in selected:
        grouped.setdefault(int(row["state_to"]), []).append(row)
    transitions: list[Transition] = []
    for state_to, group in grouped.items():
        by_initial: dict[int, dict[str, str]] = {}
        for row in group:
            state_from = int(row["state_from"])
            if state_from in by_initial:
                raise ValueError(
                    "Selected transition scope contains duplicate SO transition rows for "
                    f"initial state {state_from} -> final state {state_to}; choose one RASSI table/gauge"
                )
            by_initial[state_from] = row
        missing = [state for state in initial_states if state not in by_initial]
        if missing:
            raise ValueError(
                f"Initial-state {aggregation} is incomplete for final state {state_to}; missing {missing}"
            )
        energies = [float(by_initial[state]["energy_ev"]) for state in initial_states]
        strengths = [float(by_initial[state]["oscillator_strength"]) for state in initial_states]
        intensity = float(np.mean(strengths)) if aggregation == "mean" else float(np.sum(strengths))
        if intensity <= 0.0:
            continue
        transitions.append(
            Transition(
                state_from=initial_states[0],
                state_to=state_to,
                oscillator_strength=intensity,
                energy_ev=float(np.mean(energies)),
                gauge=gauge,
                state_basis=f"so_initial_state_{aggregation}",
                source="mocparse bundle",
                section_index=rassi_section,
            )
        )
    return sorted(transitions, key=lambda item: float(item.energy_ev or 0.0))


def _normalize_strengths(rows: Sequence[dict[str, Any]]) -> list[float]:
    values = [float(row["oscillator_strength"]) for row in rows]
    maximum = max(values, default=0.0)
    return [value / maximum if maximum > 0 else 0.0 for value in values]


def _xanes_ylabel(normalization: str) -> str:
    return {
        "max": "Normalized intensity (a.u.)",
        "area": "Area-normalized intensity (eV$^{-1}$)",
        "none": "Unnormalized intensity (a.u.)",
    }[normalization]


def _resolve_xanes_broadening(
    args: argparse.Namespace,
    xray: dict[str, Any],
) -> dict[str, Any]:
    if args.broadening_profile == "u-m45-polly-high-resolution":
        if args.gaussian_fwhm is not None or args.lorentzian_fwhm is not None:
            raise ValueError(
                "The Polly preset fixes both widths; select --broadening-profile custom "
                "to use --gaussian-fwhm or --lorentzian-fwhm"
            )
        return {
            "preset": args.broadening_profile,
            "label": "Polly high-resolution theoretical comparison",
            "measurement_channel": "theoretical multiplet comparison",
            "width_provenance": "digitized Polly UO8 C2h M4 envelope regression",
            "gaussian_fwhm_ev": POLLY_GAUSSIAN_FWHM_EV,
            "lorentzian_fwhm_ev": POLLY_LORENTZIAN_FWHM_EV,
            "caveat": (
                "This U M-edge comparison profile is the Atomi default for continuity with "
                "recent Polly-style figures; it is not a universal element/edge lifetime model."
            ),
        }
    if args.broadening_profile == "xraydb-core-hole":
        if args.no_xraydb:
            raise ValueError("xraydb-core-hole cannot be combined with --no-xraydb")
        return {
            "preset": args.broadening_profile,
            "label": "element/edge-specific XrayDB core-hole profile",
            "measurement_channel": "ordinary XAS",
            "width_provenance": "XrayDB edge core-hole width plus Gaussian display resolution",
            "gaussian_fwhm_ev": args.gaussian_fwhm if args.gaussian_fwhm is not None else 1.0,
            "lorentzian_fwhm_ev": (
                args.lorentzian_fwhm
                if args.lorentzian_fwhm is not None
                else float(xray["core_hole_width_ev"])
            ),
            "caveat": "Use an experiment-specific instrumental resolution when one is documented.",
        }
    gaussian = args.gaussian_fwhm
    lorentzian = args.lorentzian_fwhm
    if args.broadening in {"voigt", "pseudo-voigt", "gaussian"} and gaussian is None:
        raise ValueError("Custom Gaussian/Voigt broadening requires --gaussian-fwhm")
    if args.broadening in {"voigt", "pseudo-voigt", "lorentzian"} and lorentzian is None:
        raise ValueError("Custom Lorentzian/Voigt broadening requires --lorentzian-fwhm")
    return {
        "preset": "custom",
        "label": "custom explicit broadening",
        "measurement_channel": "user-specified",
        "width_provenance": "explicit mocxanes CLI arguments",
        "gaussian_fwhm_ev": float(gaussian or 0.0),
        "lorentzian_fwhm_ev": float(lorentzian or 0.0),
        "caveat": "The MOLCAS project lead owns the physical justification for custom widths.",
    }


def render_xanes(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.bundle.resolve()
    payload = _load_molcas_bundle(manifest)
    transition_rows = _read_csv(_bundle_table(manifest, payload, "transitions"))
    section = _select_section(transition_rows, args.rassi_section)
    initial_states = _parse_ints(args.initial_states)
    all_transitions = _aggregate_transitions(
        transition_rows,
        rassi_section=section,
        gauge=args.gauge,
        initial_states=initial_states,
        aggregation=args.aggregation,
    )
    if not all_transitions:
        raise ValueError("No complete positive SO transition set survived the explicit section/gauge/initial-state selection")
    edges = [_parse_edge_spec(item) for item in args.edge]
    outdir = args.outdir.resolve()
    summary_path = outdir / "mocxanes_render.json"
    if summary_path.exists() and not args.force:
        raise FileExistsError(f"Render provenance already exists: {summary_path}; use a new outdir or --force")
    outdir.mkdir(parents=True, exist_ok=True)
    plt = require_pyplot()
    fig, axes = plt.subplots(len(edges), 1, figsize=(7.0, 3.65 * len(edges)), squeeze=False)
    edge_results: list[dict[str, Any]] = []
    for ax, edge in zip(axes[:, 0], edges):
        selected = [
            item for item in all_transitions
            if item.energy_ev is not None
            and edge.emin_ev <= float(item.energy_ev) + args.energy_shift_ev <= edge.emax_ev
        ]
        if not selected:
            raise ValueError(f"No transitions fall in explicit {edge.label} window {edge.emin_ev:g}:{edge.emax_ev:g} eV")
        xray = {"source": "disabled", "element": args.element, "edge": edge.label}
        if not args.no_xraydb:
            xray = xraydb_metadata(args.element, edge.label)
        profile = _resolve_xanes_broadening(args, xray)
        gaussian = float(profile["gaussian_fwhm_ev"])
        lorentzian = float(profile["lorentzian_fwhm_ev"])
        energy, intensity, used = broaden(
            selected,
            emin=edge.emin_ev,
            emax=edge.emax_ev,
            step=args.step,
            energy_shift=args.energy_shift_ev,
            gaussian_fwhm=gaussian,
            lorentzian_fwhm=lorentzian,
            mode=args.broadening,
            eta=args.pseudo_voigt_eta,
            normalize=args.normalize,
        )
        spectrum_path = outdir / f"{_safe_id(args.element.lower())}_{_safe_id(edge.label.lower())}_spectrum.csv"
        sticks_path = outdir / f"{_safe_id(args.element.lower())}_{_safe_id(edge.label.lower())}_sticks.csv"
        _write_csv(
            spectrum_path,
            [{"energy_ev": float(x), "intensity": float(y)} for x, y in zip(energy, intensity)],
            ("energy_ev", "intensity"),
        )
        _write_csv(
            sticks_path,
            used,
            ("state_from", "state_to", "energy_ev", "oscillator_strength", "gauge", "state_basis", "section_index"),
        )
        edge_color = "#1f4e79"
        ax.plot(energy, intensity, color=edge_color, linewidth=2.4, label=f"{args.element} {edge.label} edge XANES")
        relative = _normalize_strengths(used)
        for row, height in zip(used, relative):
            if height < args.stick_relative_threshold:
                continue
            ax.vlines(
                float(row["energy_ev"]),
                -args.stick_height * height,
                0.0,
                color=edge_color,
                linewidth=2.0,
                alpha=0.62,
            )
        ax.plot([], [], color=edge_color, linewidth=2.0, alpha=0.62, label="Dipole transitions")
        ax.axhline(0.0, color="#353535", linewidth=0.75)
        ax.set_xlim(edge.emin_ev, edge.emax_ev)
        curve_max = max(float(np.max(intensity)), 1.0e-12)
        ax.set_ylim(bottom=-1.12 * args.stick_height * curve_max, top=1.08 * curve_max)
        if args.normalize == "max":
            ax.set_yticks(np.linspace(0.0, 1.0, 5))
        ax.set_xlabel("Excitation energies (eV)")
        ax.set_ylabel(_xanes_ylabel(args.normalize))
        ax.legend(loc="upper right", frameon=False)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")
        peak = float(energy[int(np.argmax(intensity))])
        edge_results.append(
            {
                "edge": edge.label,
                "selection_window_ev": [edge.emin_ev, edge.emax_ev],
                "peak_energy_ev": peak,
                "n_transitions": len(used),
                "broadening_profile": profile,
                "gaussian_fwhm_ev": gaussian,
                "lorentzian_fwhm_ev": lorentzian,
                "xraydb": xray,
                "spectrum_csv": str(spectrum_path),
                "sticks_csv": str(sticks_path),
            }
        )
    fig.tight_layout(h_pad=1.7)
    figure_path = outdir / args.plot_name
    figures = save_figure(fig, figure_path, dpi=300, extra_formats=("pdf", "svg"))
    plt.close(fig)
    summary = {
        "schema": MOCXANES_SCHEMA,
        "bundle": str(manifest),
        "bundle_sha256": _sha256(manifest),
        "scientific_status": payload["dataset"]["scientific_status"],
        "element": args.element,
        "rassi_section": section,
        "gauge": args.gauge,
        "initial_states": initial_states,
        "initial_state_aggregation": args.aggregation,
        "energy_shift_ev": args.energy_shift_ev,
        "broadening_profile": args.broadening_profile,
        "broadening": args.broadening,
        "normalization": args.normalize,
        "stick_relative_threshold": args.stick_relative_threshold,
        "edges": edge_results,
        "figures": [str(path) for path in figures],
        "caveats": [
            "Edge windows, initial states, shift, broadening, and normalization are explicit renderer inputs.",
            "The element/edge label must match the core-excitation manifold prepared in the OpenMolcas input; a transition row alone does not prove central-atom character.",
            "As-computed cluster transition energies are not experimental calibration unless an explicit shift is supplied and documented.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _resolve_orbital_sections(rows: Sequence[dict[str, str]], requested: Sequence[int] | None) -> list[int]:
    available = sorted({int(row["section_index"]) for row in rows})
    if not available:
        raise ValueError("No orbital sections are present")
    if not requested:
        return [available[-1]]
    resolved: list[int] = []
    for raw in requested:
        value = available[raw] if raw < 0 and abs(raw) <= len(available) else raw
        if value not in available:
            raise ValueError(f"Orbital section {raw} is unavailable; choices: {available}")
        resolved.append(value)
    return resolved


def _occupation_color(value: float) -> str:
    if value >= 1.5:
        return "#1f4e79"
    if value > 0.05:
        return "#d17a22"
    return "#8f969d"


def _plot_mo_levels(
    rows: Sequence[dict[str, str]],
    sections: Sequence[int],
    path: Path,
    *,
    title: str,
    max_labels: int,
    energy_window_ev: tuple[float, float],
) -> tuple[list[Path], list[dict[str, Any]]]:
    plt = require_pyplot()
    plotted: list[dict[str, Any]] = []
    window_min, window_max = energy_window_ev
    references: dict[int, float] = {}
    groups: list[tuple[int, str, str, list[dict[str, str]]]] = []
    for section in sections:
        section_rows = [row for row in rows if int(row["section_index"]) == section]
        section_rows.sort(key=lambda row: float(row["energy_ev"]))
        occupied = [float(row["energy_ev"]) for row in section_rows if float(row["occupation_electrons"]) > 0.05]
        reference = max(occupied) if occupied else min(float(row["energy_ev"]) for row in section_rows)
        references[section] = reference
        species = sorted(
            {(row.get("symmetry_species", ""), row.get("symmetry_label", "")) for row in section_rows},
            key=lambda item: (int(item[0]) if str(item[0]).strip() else 0, str(item[1])),
        )
        for species_id, symmetry_label in species:
            selected = [
                row for row in section_rows
                if row.get("symmetry_species", "") == species_id
                and window_min <= float(row["energy_ev"]) - reference <= window_max
            ]
            if selected:
                groups.append((section, species_id, symmetry_label, selected))
    if not groups:
        raise ValueError(
            f"No orbitals fall in HOMO-relative window {window_min:g}:{window_max:g} eV"
        )
    fig, ax = plt.subplots(figsize=(max(8.0, 2.0 * len(groups) + 2.8), 6.2))
    labels: list[str] = []
    for column, (section, species_id, symmetry_label, selected) in enumerate(groups):
        reference = references[section]
        section_label = selected[0]["section_label"]
        labels.append(
            (symmetry_label or section_label)
            if len(sections) == 1
            else f"{section_label}\n{symmetry_label or 'all symmetries'}"
        )
        for row in selected:
            energy = float(row["energy_ev"]) - reference
            occupation = float(row["occupation_electrons"])
            color = _occupation_color(occupation)
            ax.hlines(energy, column - 0.34, column + 0.34, color=color, linewidth=2.2)
            plotted.append(
                {
                    "section_index": section,
                    "section_label": section_label,
                    "symmetry_species": species_id,
                    "symmetry_label": symmetry_label,
                    "mo": int(row["mo"]),
                    "relative_to_homo_ev": energy,
                    "occupation_electrons": occupation,
                    "dominant_ao": row["dominant_ao"],
                }
            )
        label_rows = sorted(selected, key=lambda row: abs(float(row["energy_ev"]) - reference))[:max_labels]
        desired = sorted(
            [(row, float(row["energy_ev"]) - reference) for row in label_rows],
            key=lambda item: item[1],
        )
        label_gap = min(1.1, max(0.22, (window_max - window_min) / max(2.0 * max_labels, 1.0)))
        placed: list[tuple[dict[str, str], float, float]] = []
        for row, energy in desired:
            label_y = energy if not placed else max(energy, placed[-1][2] + label_gap)
            placed.append((row, energy, label_y))
        if placed and placed[-1][2] > window_max:
            shift = placed[-1][2] - window_max
            placed = [(row, energy, label_y - shift) for row, energy, label_y in placed]
        for row, energy, label_y in placed:
            ax.plot(
                [column + 0.34, column + 0.37],
                [energy, label_y],
                color="#a8adb4",
                linewidth=0.55,
            )
            ax.text(
                column + 0.38,
                label_y,
                f"{row['mo']}  {row['dominant_ao']}",
                va="center",
                fontsize=7.5,
                clip_on=True,
            )
    ax.set_xticks(range(len(groups)), labels, rotation=0 if len(groups) <= 3 else 15, ha="center")
    ax.set_xlim(-0.55, len(groups) - 1 + 1.2)
    ax.set_ylim(window_min, window_max)
    ax.set_ylabel("Orbital energy relative to section HOMO (eV)")
    ax.set_title(title, pad=34)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot([], [], color="#1f4e79", linewidth=2.2, label="Occupied")
    ax.plot([], [], color="#d17a22", linewidth=2.2, label="Partially occupied")
    ax.plot([], [], color="#8f969d", linewidth=2.2, label="Virtual")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=3,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    figures = save_figure(fig, path, dpi=300, extra_formats=("pdf", "svg"))
    plt.close(fig)
    return figures, plotted


def _plot_state_correlation(
    sf_rows: Sequence[dict[str, str]],
    so_rows: Sequence[dict[str, str]],
    parent_rows: Sequence[dict[str, str]],
    section: int,
    path: Path,
    *,
    emin_ev: float,
    emax_ev: float,
    weight_threshold: float,
    max_states: int,
    title: str,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    sf = [row for row in sf_rows if int(row["rassi_section"]) == section and emin_ev <= float(row["relative_energy_ev"]) <= emax_ev]
    so = [row for row in so_rows if int(row["rassi_section"]) == section and emin_ev <= float(row["relative_energy_ev"]) <= emax_ev]
    sf.sort(key=lambda row: float(row["relative_energy_ev"]))
    so.sort(key=lambda row: float(row["relative_energy_ev"]))
    original_counts = {"spin_free": len(sf), "spin_orbit": len(so)}
    sf = sf[:max_states]
    so = so[:max_states]
    sf_by_state = {int(row["state"]): row for row in sf}
    so_by_state = {int(row["state"]): row for row in so}
    printed_sums: dict[int, float] = {}
    for row in parent_rows:
        if int(row["rassi_section"]) != section:
            continue
        state = int(row["so_state"])
        if state in so_by_state:
            printed_sums[state] = printed_sums.get(state, 0.0) + float(row["weight"])
    oversummed = {state: value for state, value in printed_sums.items() if value > 1.0 + 1.0e-3}
    if oversummed:
        raise ValueError(f"Printed RASSI parent weights exceed one: {oversummed}")
    remainders = {
        state: max(0.0, 1.0 - printed_sums.get(state, 0.0))
        for state in so_by_state
    }
    links = [
        row for row in parent_rows
        if int(row["rassi_section"]) == section
        and int(row["spin_free_state"]) in sf_by_state
        and int(row["so_state"]) in so_by_state
        and float(row["weight"]) >= weight_threshold
    ]
    if not sf or not so:
        raise ValueError(f"No matched spin-free/SO states in explicit {emin_ev:g}:{emax_ev:g} eV window")
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    for row in sf:
        y = float(row["relative_energy_ev"])
        ax.hlines(y, 0.0, 0.36, color="#4c566a", linewidth=1.5)
    for row in so:
        y = float(row["relative_energy_ev"])
        ax.hlines(y, 1.0, 1.36, color="#1f4e79", linewidth=1.8)
    for row in links:
        y0 = float(sf_by_state[int(row["spin_free_state"])]["relative_energy_ev"])
        y1 = float(so_by_state[int(row["so_state"])]["relative_energy_ev"])
        weight = float(row["weight"])
        ax.plot(
            [0.36, 1.0],
            [y0, y1],
            color="#7a8fa6",
            alpha=min(0.9, 0.15 + 0.9 * weight),
            linewidth=0.45 + 3.0 * weight,
            zorder=0,
        )

    def spread_label_positions(items: Sequence[dict[str, str]]) -> dict[int, float]:
        span = max(emax_ev - emin_ev, 1.0)
        gap = 0.022 * span
        ordered = sorted(
            ((int(row["state"]), float(row["relative_energy_ev"])) for row in items[:20]),
            key=lambda item: item[1],
        )
        placed: list[list[float]] = []
        for state, desired in ordered:
            y = desired if not placed else max(desired, placed[-1][1] + gap)
            placed.append([float(state), y])
        if placed and placed[-1][1] > emax_ev:
            shift = placed[-1][1] - emax_ev
            for item in placed:
                item[1] -= shift
        if placed and placed[0][1] < emin_ev:
            shift = emin_ev - placed[0][1]
            for item in placed:
                item[1] += shift
        return {int(state): y for state, y in placed}

    label_limit = 20
    sf_label_y = spread_label_positions(sf)
    so_label_y = spread_label_positions(so)
    for row in sf[:label_limit]:
        actual = float(row["relative_energy_ev"])
        label_y = sf_label_y[int(row["state"])]
        ax.plot([-0.04, 0.0], [label_y, actual], color="#a8adb4", linewidth=0.55)
        ax.text(-0.05, label_y, f"SF {row['state']}", ha="right", va="center", fontsize=7.2)
    for row in so[:label_limit]:
        actual = float(row["relative_energy_ev"])
        label_y = so_label_y[int(row["state"])]
        ax.plot([1.36, 1.40], [actual, label_y], color="#a8adb4", linewidth=0.55)
        ax.text(1.42, label_y, f"SO {row['state']}", ha="left", va="center", fontsize=7.2)
    ax.set_xlim(-0.45, 1.8)
    span = max(emax_ev - emin_ev, 1.0)
    ax.set_ylim(emin_ev - 0.055 * span, emax_ev + 0.015 * span)
    ax.set_xticks([0.18, 1.18], ["Spin-free states", "RASSI spin-orbit states"])
    ax.set_ylabel("Energy relative to each manifold minimum (eV)")
    ax.set_title(title)
    ax.text(
        0.5,
        -0.12,
        (
            f"Links: printed RASSI parent weights >= {100.0 * weight_threshold:.1f}%; "
            f"unprinted remainder {100.0 * min(remainders.values(), default=0.0):.1f}-"
            f"{100.0 * max(remainders.values(), default=0.0):.1f}% (not assigned)"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    figures = save_figure(fig, path, dpi=300, extra_formats=("pdf", "svg"))
    plt.close(fig)
    original_counts["unprinted_parent_remainder_by_so_state"] = {
        str(state): value for state, value in sorted(remainders.items())
    }
    return figures, [
        {
            "spin_free_state": int(row["spin_free_state"]),
            "so_state": int(row["so_state"]),
            "spin": float(row["spin"]),
            "weight": float(row["weight"]),
        }
        for row in links
    ], original_counts


def _parse_window(text: str) -> tuple[float, float]:
    parts = text.split(":", 1)
    if len(parts) != 2:
        raise ValueError("State window must be EMIN:EMAX in eV")
    emin, emax = float(parts[0]), float(parts[1])
    if emax <= emin:
        raise ValueError("State-window EMAX must exceed EMIN")
    return emin, emax


def render_mo(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.bundle.resolve()
    payload = _load_molcas_bundle(manifest)
    orbitals = _read_csv(_bundle_table(manifest, payload, "orbitals"))
    sf = _read_csv(_bundle_table(manifest, payload, "spin_free_states"))
    so = _read_csv(_bundle_table(manifest, payload, "spin_orbit_states"))
    parents = _read_csv(_bundle_table(manifest, payload, "so_parent_weights"))
    invalid_occupations = [
        row for row in orbitals
        if not -1.0e-6 <= float(row["occupation_electrons"]) <= 2.0 + 1.0e-6
    ]
    if invalid_occupations:
        raise ValueError(
            f"MOLCAS orbital table contains {len(invalid_occupations)} occupations outside [0, 2] electrons"
        )
    invalid_weights = [
        row for row in parents
        if not -1.0e-9 <= float(row["weight"]) <= 1.0 + 1.0e-9
    ]
    if invalid_weights:
        raise ValueError(
            f"RASSI parent table contains {len(invalid_weights)} weights outside [0, 1]"
        )
    sections = _resolve_orbital_sections(orbitals, args.orbital_section)
    rassi_section = _select_section(so, args.rassi_section)
    mo_window = _parse_window(args.mo_window)
    emin, emax = _parse_window(args.state_window)
    outdir = args.outdir.resolve()
    summary_path = outdir / "mocmo_render.json"
    if summary_path.exists() and not args.force:
        raise FileExistsError(f"Render provenance already exists: {summary_path}; use a new outdir or --force")
    outdir.mkdir(parents=True, exist_ok=True)
    mo_figures, plotted_orbitals = _plot_mo_levels(
        orbitals,
        sections,
        outdir / args.mo_plot_name,
        title=args.mo_title,
        max_labels=args.max_orbital_labels,
        energy_window_ev=mo_window,
    )
    state_figures, links, original_counts = _plot_state_correlation(
        sf,
        so,
        parents,
        rassi_section,
        outdir / args.state_plot_name,
        emin_ev=emin,
        emax_ev=emax,
        weight_threshold=args.parent_weight_threshold,
        max_states=args.max_states,
        title=args.state_title,
    )
    orbital_csv = outdir / "mocmo_selected_orbitals.csv"
    links_csv = outdir / "mocmo_state_correlation_links.csv"
    _write_csv(
        orbital_csv,
        plotted_orbitals,
        (
            "section_index", "section_label", "symmetry_species", "symmetry_label", "mo",
            "relative_to_homo_ev", "occupation_electrons", "dominant_ao",
        ),
    )
    _write_csv(links_csv, links, ("spin_free_state", "so_state", "spin", "weight"))
    summary = {
        "schema": MOCMO_SCHEMA,
        "bundle": str(manifest),
        "bundle_sha256": _sha256(manifest),
        "scientific_status": payload["dataset"]["scientific_status"],
        "orbital_sections": sections,
        "orbital_energy_reference": "per-section HOMO",
        "orbital_display_window_ev": list(mo_window),
        "rassi_section": rassi_section,
        "state_window_ev": [emin, emax],
        "state_energy_reference": "separate spin-free and spin-orbit manifold minima within the selected RASSI section",
        "parent_weight_threshold": args.parent_weight_threshold,
        "state_counts_before_cap": {
            "spin_free": original_counts["spin_free"],
            "spin_orbit": original_counts["spin_orbit"],
        },
        "unprinted_parent_remainder_by_so_state": original_counts.get(
            "unprinted_parent_remainder_by_so_state", {}
        ),
        "max_states_per_side": args.max_states,
        "n_links": len(links),
        "mo_figures": [str(path) for path in mo_figures],
        "state_correlation_figures": [str(path) for path in state_figures],
        "selected_orbitals_csv": str(orbital_csv),
        "state_links_csv": str(links_csv),
        "interpretation_guard": (
            "The MO panel contains one-electron printed orbitals. The correlation panel contains "
            "many-electron spin-free states and their RASSI spin-orbit parent weights; its links "
            "must not be described as AO percentages or literal MO mixing. The MO panel can be "
            "interpreted as ligand/crystal-field splitting only after its orbital identities and "
            "symmetry assignment are physically validated."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _add_mocparse_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("output", type=Path, help="OpenMolcas .out/.log or gzip-compressed output")
    parser.add_argument("--inp", type=Path, help="Authoritative OpenMolcas input deck")
    parser.add_argument("--outdir", type=Path, required=True, help="New immutable bundle directory")
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument("--project", default="")
    parser.add_argument("--science-owner", default="MOLCAS project lead")
    parser.add_argument("--canonical-report", default="")
    parser.add_argument("--scientific-question", default="Reusable full OpenMolcas postanalysis data layer")
    parser.add_argument("--scientific-status", choices=("accepted", "provisional", "screening", "rejected"), default="provisional")
    parser.add_argument("--quality", choices=("descriptor", "screening-prior", "production", "publication", "diagnostic"), default="diagnostic")


def _add_mocxanes_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("bundle", type=Path, help="mocparse plot_dataset.json")
    parser.add_argument("--element", required=True)
    parser.add_argument("--edge", action="append", required=True, help="Explicit LABEL:EMIN:EMAX window in eV; repeat for panels")
    parser.add_argument("--initial-states", required=True, help="Comma-separated SO initial states")
    parser.add_argument("--aggregation", choices=("mean", "sum"), default="mean")
    parser.add_argument("--rassi-section", default="last", help="first, last, or integer section")
    parser.add_argument("--gauge", choices=("length", "velocity", "any"), default="length")
    parser.add_argument("--energy-shift-ev", type=float, default=0.0)
    parser.add_argument(
        "--broadening-profile",
        "--profile-preset",
        choices=("u-m45-polly-high-resolution", "xraydb-core-hole", "custom"),
        default="u-m45-polly-high-resolution",
        help=(
            "Named width policy. The default reproduces the recent Polly-style U M-edge "
            "figures; preset widths override custom width flags."
        ),
    )
    parser.add_argument("--gaussian-fwhm", type=float)
    parser.add_argument("--lorentzian-fwhm", type=float)
    parser.add_argument("--broadening", choices=("voigt", "pseudo-voigt", "gaussian", "lorentzian"), default="voigt")
    parser.add_argument("--pseudo-voigt-eta", type=float, default=0.5)
    parser.add_argument("--normalize", choices=("max", "area", "none"), default="max")
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--stick-height", type=float, default=0.24)
    parser.add_argument("--stick-relative-threshold", type=float, default=0.01)
    parser.add_argument("--no-xraydb", action="store_true")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--plot-name", default="molcas_xanes.png")
    parser.add_argument("--force", action="store_true")


def _add_mocmo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("bundle", type=Path, help="mocparse plot_dataset.json")
    parser.add_argument("--orbital-section", type=int, action="append", help="Printed orbital section index; repeat; default last")
    parser.add_argument("--rassi-section", default="last", help="first, last, or integer section")
    parser.add_argument("--mo-window", default="-20:20", help="Explicit HOMO-relative MO EMIN:EMAX in eV")
    parser.add_argument("--state-window", default="0:5", help="Explicit spin-free/SO EMIN:EMAX in eV")
    parser.add_argument("--parent-weight-threshold", type=float, default=0.05)
    parser.add_argument("--max-states", type=int, default=80)
    parser.add_argument("--max-orbital-labels", type=int, default=24)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mo-plot-name", default="molcas_mo_levels.png")
    parser.add_argument("--state-plot-name", default="molcas_spin_free_to_so_states.png")
    parser.add_argument("--mo-title", default="OpenMolcas one-electron orbital levels")
    parser.add_argument("--state-title", default="Spin-free to RASSI spin-orbit state correlation")
    parser.add_argument("--force", action="store_true")


def mocparse_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze reusable OpenMolcas postanalysis tables with provenance.")
    _add_mocparse_arguments(parser)
    args = parser.parse_args(argv)
    manifest = build_molcas_bundle(args)
    print(f"Wrote MOLCAS data bundle: {manifest}")
    return 0


def mocxanes_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render publication XANES from a verified mocparse bundle.")
    _add_mocxanes_arguments(parser)
    args = parser.parse_args(argv)
    summary = render_xanes(args)
    print(f"Wrote XANES render: {Path(summary['figures'][0]).parent}")
    return 0


def mocmo_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render MO levels and spin-free-to-SO state correlation from mocparse data.")
    _add_mocmo_arguments(parser)
    args = parser.parse_args(argv)
    summary = render_mo(args)
    print(f"Wrote MO/state render: {Path(summary['mo_figures'][0]).parent}")
    return 0
