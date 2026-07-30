"""General Bagus-style covalency investigation after MOLCAS postanalysis.

The analysis is deliberately multi-measure.  It does not convert orbital
mixing, charge transfer, radial extent, or spectroscopy into one universal
"percent covalent" score.  A JSON specification maps project tables onto a
small canonical schema so the same layer can be used for actinides,
lanthanides, transition metals, and main-group systems.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_SPEC = "atomi.molcas_bagus_covalency_spec.v1"
SCHEMA_RESULT = "atomi.molcas_bagus_covalency_result.v1"
SCHEMA_PROVENANCE = "atomi.molcas_bagus_covalency_provenance.v1"

VALID_STATE_ROLES = {"ground", "excited", "reference", "so-parent"}
VALID_STATUSES = {"accepted", "provisional", "screening", "rejected", "unavailable"}
QUALITY_TO_STATUS = {
    "publication": "accepted",
    "production": "accepted",
    "accepted": "accepted",
    "provisional": "provisional",
    "diagnostic": "provisional",
    "screening": "screening",
    "screening-prior": "screening",
    "rejected": "rejected",
}
STATUS_PRIORITY = {
    "accepted": 0,
    "provisional": 1,
    "screening": 2,
    "unavailable": 3,
    "rejected": 4,
}
DEFAULT_THRESHOLDS = {
    "identity_projection_fraction_min": 0.80,
    "projection_coverage_fraction_min": 0.99,
    "radial_extent_ratio_min": 0.50,
    "radial_extent_ratio_max": 2.00,
    "configuration_norm_min": 0.95,
    "configuration_norm_max": 1.05,
    "projection_tolerance": 1.0e-6,
}
DEFAULT_REQUIRED_MEASURES = {
    "ground": [
        "relative_orbital_energy",
        "natural_occupation",
        "occupation_spatial_extent",
        "matched_radial_extent",
        "isolated_reference_projection",
        "ligand_hole_weight",
        "secondary_shell_participation",
    ],
    "excited": [
        "relative_orbital_energy",
        "natural_occupation",
        "occupation_spatial_extent",
        "matched_radial_extent",
        "isolated_reference_projection",
        "ligand_hole_weight",
        "secondary_shell_participation",
    ],
    "so-parent": [
        "relative_orbital_energy",
        "natural_occupation",
        "occupation_spatial_extent",
        "matched_radial_extent",
        "isolated_reference_projection",
        "ligand_hole_weight",
        "secondary_shell_participation",
    ],
    "spectrum": ["satellite_intensity"],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(
    value: Any,
    *,
    field: str,
    source: Path,
    row_number: int,
    required: bool = False,
) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        if required:
            raise ValueError(
                f"{source}:{row_number}: required numeric field {field!r} is blank"
            )
        return None
    try:
        result = float(text)
    except ValueError as exc:
        raise ValueError(
            f"{source}:{row_number}: field {field!r} is not numeric: {text!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(
            f"{source}:{row_number}: field {field!r} contains NaN or infinity"
        )
    return result


def _status_from_quality(value: str) -> str:
    key = str(value or "provisional").strip().lower()
    if key in VALID_STATUSES:
        return key
    if key not in QUALITY_TO_STATUS:
        raise ValueError(
            f"Unsupported quality/status {value!r}; use one of "
            f"{sorted(set(QUALITY_TO_STATUS) | VALID_STATUSES)}"
        )
    return QUALITY_TO_STATUS[key]


def _combine_status(*statuses: str) -> str:
    values = [status for status in statuses if status]
    if not values:
        return "unavailable"
    invalid = [status for status in values if status not in STATUS_PRIORITY]
    if invalid:
        raise ValueError(f"Unsupported scientific status: {invalid}")
    return max(values, key=lambda status: STATUS_PRIORITY[status])


def _source_list(inputs: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = inputs.get(key)
    if value is None:
        return []
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [dict(item) for item in value]
    raise ValueError(f"inputs.{key} must be an object or list of objects")


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _canonical_value(
    row: dict[str, str],
    source: dict[str, Any],
    name: str,
    default: Any = "",
) -> Any:
    columns = dict(source.get("columns") or {})
    actual = str(columns.get(name) or name)
    value = row.get(actual, "")
    if str(value).strip():
        value_maps = dict(source.get("value_maps") or {})
        mapping = dict(value_maps.get(name) or {})
        return mapping.get(str(value), value)
    defaults = dict(source.get("defaults") or {})
    return defaults.get(name, default)


def _read_source_rows(
    source: dict[str, Any],
    *,
    base: Path,
    required: tuple[str, ...],
    optional: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_id = str(source.get("id") or "").strip()
    if not source_id:
        raise ValueError("Every input source requires a stable id")
    path_text = str(source.get("path") or "").strip()
    if not path_text:
        raise ValueError(f"Input source {source_id!r} requires path")
    path = _resolve_path(path_text, base)
    if not path.is_file():
        raise FileNotFoundError(path)
    delimiter = str(source.get("delimiter") or ("\t" if path.suffix == ".tsv" else ","))
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        header = set(reader.fieldnames or [])
        if not header:
            raise ValueError(f"Input table has no header: {path}")
        columns = dict(source.get("columns") or {})
        defaults = dict(source.get("defaults") or {})
        missing = []
        for name in required:
            actual = str(columns.get(name) or name)
            if actual not in header and name not in defaults:
                missing.append(f"{name}->{actual}")
        if missing:
            raise ValueError(
                f"Input source {source_id!r} is missing required mappings: {missing}"
            )
        for row_number, raw in enumerate(reader, start=2):
            filters = dict(source.get("filters") or {})
            selected = True
            for name, expected in filters.items():
                actual = str(columns.get(name) or name)
                allowed = (
                    {str(item) for item in expected}
                    if isinstance(expected, list)
                    else {str(expected)}
                )
                if str(raw.get(actual, "")) not in allowed:
                    selected = False
                    break
            if not selected:
                continue
            row = {
                name: _canonical_value(raw, source, name)
                for name in (*required, *optional)
            }
            row["_source_id"] = source_id
            row["_source_path"] = str(path)
            row["_row_number"] = row_number
            row["_quality"] = str(
                source.get("quality")
                or source.get("scientific_status")
                or "provisional"
            )
            row["_identity_guard"] = bool(source.get("identity_guard", False))
            row["_scope"] = str(source.get("scope") or row.get("orbital_scope") or "all")
            rows.append(row)
    if not rows:
        raise ValueError(f"Input table contains no selected rows: {path}")
    record = {
        "id": source_id,
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "quality": str(source.get("quality") or "provisional"),
        "filters": dict(source.get("filters") or {}),
        "value_maps": dict(source.get("value_maps") or {}),
    }
    return rows, record


def template_spec() -> dict[str, Any]:
    """Return a chemistry-neutral starter specification."""

    return {
        "schema": SCHEMA_SPEC,
        "system": {
            "label": "metal_ligand_cluster",
            "central_element": "M",
            "central_atom_index": 1,
        },
        "method": {
            "program": "OpenMolcas",
            "basis": "record exact basis",
            "relativity": "record scalar/SOC/four-component treatment",
            "wavefunction": "record CASSCF/RASSCF/CASPT2/RASSI treatment",
        },
        "shells": {
            "primary": "metal valence shell, e.g. 5f/4f/3d",
            "secondary": "metal secondary shell, e.g. 6d/5d/4s",
            "core": "spectroscopic core shell when applicable",
            "ligand": "ligand donor manifold, e.g. O 2p",
        },
        "reference": {
            "label": "matched isolated-ion or fragment reference",
            "radial_extent_angstrom": {
                "metal valence shell, e.g. 5f/4f/3d": 1.0,
                "metal secondary shell, e.g. 6d/5d/4s": 1.5,
            },
            "basis_policy": "same basis, relativistic Hamiltonian, and projection metric",
        },
        "states": [
            {
                "id": "ground_1",
                "label": "Ground state",
                "role": "ground",
                "edge": "",
                "scientific_status": "accepted",
            },
            {
                "id": "excited_1",
                "label": "Selected excited state",
                "role": "excited",
                "edge": "EDGE",
                "scientific_status": "provisional",
            },
        ],
        "inputs": {
            "orbitals": [
                {
                    "id": "state_orbitals",
                    "path": "state_orbital_metrics.csv",
                    "scope": "active",
                    "quality": "production",
                    "identity_guard": True,
                    "columns": {
                        "state_id": "state_id",
                        "orbital_id": "orbital_id",
                        "occupation": "occupation",
                        "orbital_energy_ev": "orbital_energy_ev",
                        "radial_extent_angstrom": "radial_extent_angstrom",
                        "primary_projection": "primary_projection",
                        "secondary_projection": "secondary_projection",
                        "ligand_projection": "ligand_projection",
                        "orbital_lz_hbar": "orbital_lz_hbar",
                        "spin_sz_hbar": "spin_sz_hbar",
                        "total_jz_hbar": "total_jz_hbar",
                    },
                }
            ],
            "configurations": [
                {
                    "id": "state_configurations",
                    "path": "configuration_weights.csv",
                    "quality": "production",
                    "weight_scale": 1.0,
                    "columns": {
                        "state_id": "state_id",
                        "configuration_class": "configuration_class",
                        "weight": "weight",
                        "core_holes": "core_holes",
                        "ligand_holes": "ligand_holes",
                        "primary_electrons": "primary_electrons",
                        "secondary_electrons": "secondary_electrons",
                    },
                }
            ],
            "transitions": [
                {
                    "id": "dipole_transitions",
                    "path": "transitions.csv",
                    "quality": "production",
                    "assignment_basis": "state_resolved_configuration",
                    "satellite_values": [
                        "ligand_to_metal_charge_transfer",
                        "shake_up",
                    ],
                    "columns": {
                        "state_id": "final_state_id",
                        "edge": "edge",
                        "energy_ev": "energy_ev",
                        "oscillator_strength": "oscillator_strength",
                        "satellite_class": "satellite_class",
                        "assignment_status": "assignment_status",
                    },
                }
            ],
        },
        "spectral_sectors": [
            {
                "edge": "EDGE",
                "label": "high_energy_sideband",
                "kind": "descriptor",
                "emin_ev": 0.0,
                "emax_ev": 1.0,
            }
        ],
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "required_measures": dict(DEFAULT_REQUIRED_MEASURES),
        "sources": [],
        "caveats": [
            "Do not report one universal percent-covalent score.",
            "A spectral sideband is not a charge-transfer satellite without state-resolved configuration evidence.",
            "Ground and excited states require matched orbital identity, basis, Hamiltonian, and projection definitions.",
        ],
    }


def _load_spec(path: Path) -> dict[str, Any]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("schema") != SCHEMA_SPEC:
        raise ValueError(
            f"Expected schema {SCHEMA_SPEC!r}, got {body.get('schema')!r}"
        )
    system = dict(body.get("system") or {})
    if not str(system.get("label") or "").strip():
        raise ValueError("spec.system.label is required")
    if not str(system.get("central_element") or "").strip():
        raise ValueError("spec.system.central_element is required")
    shells = dict(body.get("shells") or {})
    if not str(shells.get("primary") or "").strip():
        raise ValueError("spec.shells.primary is required")
    states = list(body.get("states") or [])
    if not states:
        raise ValueError("spec.states must contain at least one state")
    seen = set()
    for state in states:
        state_id = str(state.get("id") or "").strip()
        if not state_id or state_id in seen:
            raise ValueError(f"State ids must be nonempty and unique: {state_id!r}")
        seen.add(state_id)
        role = str(state.get("role") or "").strip().lower()
        if role not in VALID_STATE_ROLES:
            raise ValueError(
                f"State {state_id!r} role must be one of {sorted(VALID_STATE_ROLES)}"
            )
        _status_from_quality(str(state.get("scientific_status") or "provisional"))
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(dict(body.get("thresholds") or {}))
    for key, value in thresholds.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Threshold {key!r} is not finite")
        thresholds[key] = number
    if thresholds["identity_projection_fraction_min"] < 0:
        raise ValueError("identity_projection_fraction_min must be nonnegative")
    if not 0 < thresholds["projection_coverage_fraction_min"] <= 1:
        raise ValueError(
            "projection_coverage_fraction_min must be in the interval (0, 1]"
        )
    if thresholds["radial_extent_ratio_min"] <= 0:
        raise ValueError("radial_extent_ratio_min must be positive")
    if (
        thresholds["radial_extent_ratio_max"]
        < thresholds["radial_extent_ratio_min"]
    ):
        raise ValueError("radial extent ratio bounds are reversed")
    if thresholds["configuration_norm_min"] <= 0:
        raise ValueError("configuration_norm_min must be positive")
    if (
        thresholds["configuration_norm_max"]
        < thresholds["configuration_norm_min"]
    ):
        raise ValueError("configuration norm bounds are reversed")
    body["thresholds"] = thresholds
    return body


def _state_index(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(state["id"]): dict(state) for state in spec["states"]}


def _state_status(state: dict[str, Any]) -> str:
    return _status_from_quality(str(state.get("scientific_status") or "provisional"))


def _load_orbitals(
    spec: dict[str, Any],
    *,
    base: Path,
    states: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = ("state_id", "occupation")
    optional = (
        "orbital_id",
        "orbital_scope",
        "orbital_energy_ev",
        "radial_extent_angstrom",
        "primary_projection",
        "secondary_projection",
        "ligand_projection",
        "orbital_lz_hbar",
        "spin_sz_hbar",
        "total_jz_hbar",
        "identity_status",
    )
    rows: list[dict[str, Any]] = []
    records = []
    tolerance = float(spec["thresholds"]["projection_tolerance"])
    for source in _source_list(dict(spec.get("inputs") or {}), "orbitals"):
        source_rows, record = _read_source_rows(
            source,
            base=base,
            required=required,
            optional=optional,
        )
        records.append(record)
        for row in source_rows:
            state_id = str(row["state_id"]).strip()
            if state_id not in states:
                raise ValueError(
                    f"{row['_source_path']}:{row['_row_number']}: "
                    f"unknown state_id {state_id!r}"
                )
            occupation = _finite_float(
                row["occupation"],
                field="occupation",
                source=Path(row["_source_path"]),
                row_number=int(row["_row_number"]),
                required=True,
            )
            if occupation is None or occupation < 0:
                raise ValueError("Orbital occupations must be nonnegative")
            parsed = dict(row)
            parsed["state_id"] = state_id
            parsed["occupation"] = occupation
            for key in (
                "orbital_energy_ev",
                "radial_extent_angstrom",
                "primary_projection",
                "secondary_projection",
                "ligand_projection",
                "orbital_lz_hbar",
                "spin_sz_hbar",
                "total_jz_hbar",
            ):
                parsed[key] = _finite_float(
                    row.get(key),
                    field=key,
                    source=Path(row["_source_path"]),
                    row_number=int(row["_row_number"]),
                )
            radial = parsed["radial_extent_angstrom"]
            if radial is not None and radial <= 0:
                raise ValueError("radial_extent_angstrom must be positive")
            for key in (
                "primary_projection",
                "secondary_projection",
                "ligand_projection",
            ):
                value = parsed[key]
                if value is not None and (value < -tolerance or value > 1 + tolerance):
                    raise ValueError(
                        f"{row['_source_path']}:{row['_row_number']}: "
                        f"{key}={value} is outside [0, 1]"
                    )
            parsed["orbital_scope"] = str(
                row.get("orbital_scope") or row["_scope"] or "all"
            )
            rows.append(parsed)
    return rows, records


def _reference_radius(spec: dict[str, Any], shell: str) -> float | None:
    reference = dict(spec.get("reference") or {})
    radial = dict(reference.get("radial_extent_angstrom") or {})
    if shell not in radial:
        return None
    value = float(radial[shell])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Reference radial extent for {shell!r} must be positive")
    return value


def _summarize_orbitals(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["state_id"],
                str(row["_source_id"]),
                str(row["orbital_scope"]),
            )
        ].append(row)
    primary_shell = str(spec["shells"]["primary"])
    reference_radius = _reference_radius(spec, primary_shell)
    threshold = float(spec["thresholds"]["identity_projection_fraction_min"])
    coverage_threshold = float(
        spec["thresholds"]["projection_coverage_fraction_min"]
    )
    ratio_min = float(spec["thresholds"]["radial_extent_ratio_min"])
    ratio_max = float(spec["thresholds"]["radial_extent_ratio_max"])
    output = []
    for (state_id, source_id, scope), selected in sorted(groups.items()):
        state = states[state_id]
        electron_count = sum(float(row["occupation"]) for row in selected)

        def projected(name: str) -> float | None:
            values = [
                (float(row["occupation"]), row[name])
                for row in selected
                if row[name] is not None
            ]
            if not values:
                return None
            return sum(occupation * float(value) for occupation, value in values)

        def weighted_expectation(name: str) -> float | None:
            values = [
                (float(row["occupation"]), row[name])
                for row in selected
                if row[name] is not None and float(row["occupation"]) > 0
            ]
            denominator = sum(occupation for occupation, _value in values)
            if denominator <= 0:
                return None
            return (
                sum(occupation * float(value) for occupation, value in values)
                / denominator
            )

        radial_rows = [
            row
            for row in selected
            if row["radial_extent_angstrom"] is not None
            and float(row["occupation"]) > 0
        ]
        radial_denominator = sum(float(row["occupation"]) for row in radial_rows)
        weighted_radius = (
            sum(
                float(row["occupation"]) * float(row["radial_extent_angstrom"])
                for row in radial_rows
            )
            / radial_denominator
            if radial_denominator > 0
            else None
        )
        energies = [
            float(row["orbital_energy_ev"])
            for row in selected
            if row["orbital_energy_ev"] is not None
        ]
        energy_centroid = weighted_expectation("orbital_energy_ev")
        primary = projected("primary_projection")
        secondary = projected("secondary_projection")
        ligand = projected("ligand_projection")
        primary_projection_available = any(
            row["primary_projection"] is not None for row in selected
        )
        primary_coverage_electrons = (
            sum(
                float(row["occupation"])
                for row in selected
                if row["primary_projection"] is not None
            )
            if primary_projection_available
            else None
        )
        primary_coverage_fraction = (
            primary_coverage_electrons / electron_count
            if primary_coverage_electrons is not None and electron_count > 0
            else None
        )
        identity_fraction = (
            primary / electron_count
            if primary is not None and electron_count > 0
            else None
        )
        ratio = (
            weighted_radius / reference_radius
            if weighted_radius is not None and reference_radius is not None
            else None
        )
        source_status = _status_from_quality(str(selected[0]["_quality"]))
        status = _combine_status(source_status, _state_status(state))
        guards = []
        if bool(selected[0]["_identity_guard"]):
            explicit_identity_statuses = [
                _status_from_quality(str(row["identity_status"]))
                for row in selected
                if str(row.get("identity_status") or "").strip()
            ]
            if explicit_identity_statuses:
                status = _combine_status(status, *explicit_identity_statuses)
            if (
                primary_coverage_fraction is None
                or primary_coverage_fraction < coverage_threshold
            ):
                guards.append(
                    {
                        "name": "primary_projection_coverage",
                        "status": "unavailable",
                        "value": primary_coverage_fraction,
                        "threshold": coverage_threshold,
                    }
                )
                status = _combine_status(status, "unavailable")
            else:
                guards.append(
                    {
                        "name": "primary_projection_coverage",
                        "status": "accepted",
                        "value": primary_coverage_fraction,
                        "threshold": coverage_threshold,
                    }
                )
            if (
                identity_fraction is None
                or primary_coverage_fraction is None
                or primary_coverage_fraction < coverage_threshold
            ):
                guards.append(
                    {
                        "name": "primary_shell_identity",
                        "status": "unavailable",
                        "value": None,
                        "threshold": threshold,
                    }
                )
                status = _combine_status(status, "unavailable")
            elif identity_fraction < threshold:
                guards.append(
                    {
                        "name": "primary_shell_identity",
                        "status": "rejected",
                        "value": identity_fraction,
                        "threshold": threshold,
                    }
                )
                status = "rejected"
            else:
                guards.append(
                    {
                        "name": "primary_shell_identity",
                        "status": "accepted",
                        "value": identity_fraction,
                        "threshold": threshold,
                    }
                )
        if ratio is not None:
            radial_status = (
                "accepted" if ratio_min <= ratio <= ratio_max else "rejected"
            )
            guards.append(
                {
                    "name": "matched_radial_extent",
                    "status": radial_status,
                    "value": ratio,
                    "threshold": [ratio_min, ratio_max],
                }
            )
            if radial_status == "rejected":
                status = "rejected"
        output.append(
            {
                "state_id": state_id,
                "state_label": str(state.get("label") or state_id),
                "state_role": str(state["role"]),
                "edge": str(state.get("edge") or ""),
                "source_id": source_id,
                "scope": scope,
                "electron_count": electron_count,
                "occupation_weighted_radial_extent_angstrom": weighted_radius,
                "primary_reference_radial_extent_angstrom": reference_radius,
                "radial_extent_ratio": ratio,
                "primary_projected_electrons": primary,
                "secondary_projected_electrons": secondary,
                "ligand_projected_electrons": ligand,
                "primary_projection_coverage_fraction": primary_coverage_fraction,
                "primary_identity_fraction": identity_fraction,
                "unprojected_primary_fraction": (
                    1.0 - identity_fraction
                    if identity_fraction is not None
                    else None
                ),
                "orbital_energy_min_ev": min(energies) if energies else None,
                "orbital_energy_max_ev": max(energies) if energies else None,
                "orbital_energy_centroid_ev": energy_centroid,
                "orbital_energy_span_ev": (
                    max(energies) - min(energies) if energies else None
                ),
                "orbital_lz_expectation_hbar": weighted_expectation(
                    "orbital_lz_hbar"
                ),
                "spin_sz_expectation_hbar": weighted_expectation("spin_sz_hbar"),
                "total_jz_expectation_hbar": weighted_expectation("total_jz_hbar"),
                "identity_guard_enabled": bool(selected[0]["_identity_guard"]),
                "guards": guards,
                "scientific_status": status,
            }
        )
    return output


def _load_configurations(
    spec: dict[str, Any],
    *,
    base: Path,
    states: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = ("state_id", "weight")
    optional = (
        "configuration_class",
        "core_holes",
        "ligand_holes",
        "primary_electrons",
        "secondary_electrons",
    )
    rows = []
    records = []
    for source in _source_list(dict(spec.get("inputs") or {}), "configurations"):
        weight_scale = float(source.get("weight_scale", 1.0))
        if not math.isfinite(weight_scale) or weight_scale <= 0:
            raise ValueError(
                f"Configuration source {source.get('id')!r} weight_scale "
                "must be finite and positive"
            )
        source_rows, record = _read_source_rows(
            source,
            base=base,
            required=required,
            optional=optional,
        )
        record["weight_scale"] = weight_scale
        records.append(record)
        for row in source_rows:
            state_id = str(row["state_id"]).strip()
            if state_id not in states:
                raise ValueError(
                    f"{row['_source_path']}:{row['_row_number']}: "
                    f"unknown state_id {state_id!r}"
                )
            parsed = dict(row)
            parsed["state_id"] = state_id
            raw_weight = _finite_float(
                row["weight"],
                field="weight",
                source=Path(row["_source_path"]),
                row_number=int(row["_row_number"]),
                required=True,
            )
            if raw_weight is None or raw_weight < 0:
                raise ValueError("Configuration weights must be nonnegative")
            parsed["weight"] = raw_weight * weight_scale
            for key in (
                "core_holes",
                "ligand_holes",
                "primary_electrons",
                "secondary_electrons",
            ):
                parsed[key] = _finite_float(
                    row.get(key),
                    field=key,
                    source=Path(row["_source_path"]),
                    row_number=int(row["_row_number"]),
                )
                if parsed[key] is not None and parsed[key] < 0:
                    raise ValueError(f"{key} must be nonnegative")
            rows.append(parsed)
    return rows, records


def _summarize_configurations(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["state_id"], str(row["_source_id"]))].append(row)
    norm_min = float(spec["thresholds"]["configuration_norm_min"])
    norm_max = float(spec["thresholds"]["configuration_norm_max"])
    output = []
    for (state_id, source_id), selected in sorted(groups.items()):
        state = states[state_id]
        raw_norm = sum(float(row["weight"]) for row in selected)
        if raw_norm <= 0:
            raise ValueError(
                f"Configuration source {source_id!r}, state {state_id!r} "
                "has zero total weight"
            )
        normalized = [
            (row, float(row["weight"]) / raw_norm) for row in selected
        ]

        def expected(name: str) -> float | None:
            if not any(row[name] is not None for row, _weight in normalized):
                return None
            return sum(
                (float(row[name]) if row[name] is not None else 0.0) * weight
                for row, weight in normalized
            )

        ligand_available = any(
            row["ligand_holes"] is not None for row, _weight in normalized
        )
        secondary_available = any(
            row["secondary_electrons"] is not None for row, _weight in normalized
        )
        ligand_weight = (
            sum(
                weight
                for row, weight in normalized
                if row["ligand_holes"] is not None
                and float(row["ligand_holes"]) > 0
            )
            if ligand_available
            else None
        )
        secondary_weight = (
            sum(
                weight
                for row, weight in normalized
                if row["secondary_electrons"] is not None
                and float(row["secondary_electrons"]) > 0
            )
            if secondary_available
            else None
        )
        coupled_weight = (
            sum(
                weight
                for row, weight in normalized
                if row["ligand_holes"] is not None
                and row["secondary_electrons"] is not None
                and float(row["ligand_holes"]) > 0
                and float(row["secondary_electrons"]) > 0
            )
            if ligand_available and secondary_available
            else None
        )
        status = _combine_status(
            _status_from_quality(str(selected[0]["_quality"])),
            _state_status(state),
        )
        norm_guard = (
            "accepted" if norm_min <= raw_norm <= norm_max else "rejected"
        )
        if norm_guard == "rejected":
            status = "rejected"
        output.append(
            {
                "state_id": state_id,
                "state_label": str(state.get("label") or state_id),
                "state_role": str(state["role"]),
                "edge": str(state.get("edge") or ""),
                "source_id": source_id,
                "raw_weight_norm": raw_norm,
                "configuration_norm_guard": norm_guard,
                "ligand_hole_weight": ligand_weight,
                "expected_ligand_holes": expected("ligand_holes"),
                "expected_primary_electrons": expected("primary_electrons"),
                "expected_secondary_electrons": expected("secondary_electrons"),
                "weight_with_secondary_occupation": secondary_weight,
                "coupled_ligand_hole_secondary_weight": coupled_weight,
                "expected_core_holes": expected("core_holes"),
                "scientific_status": status,
            }
        )
    return output


def _load_transitions(
    spec: dict[str, Any],
    *,
    base: Path,
    states: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = ("edge", "energy_ev", "oscillator_strength")
    optional = (
        "state_id",
        "satellite_class",
        "assignment_status",
    )
    rows = []
    records = []
    for source in _source_list(dict(spec.get("inputs") or {}), "transitions"):
        satellite_values = {
            str(value).strip().lower()
            for value in list(source.get("satellite_values") or [])
            if str(value).strip()
        }
        source_rows, record = _read_source_rows(
            source,
            base=base,
            required=required,
            optional=optional,
        )
        record["assignment_basis"] = str(source.get("assignment_basis") or "")
        record["satellite_values"] = sorted(satellite_values)
        records.append(record)
        for row in source_rows:
            state_id = str(row.get("state_id") or "").strip()
            if state_id and state_id not in states:
                raise ValueError(
                    f"{row['_source_path']}:{row['_row_number']}: "
                    f"unknown transition state_id {state_id!r}"
                )
            parsed = dict(row)
            parsed["state_id"] = state_id
            parsed["edge"] = str(row["edge"]).strip()
            if not parsed["edge"]:
                raise ValueError("Transition edge cannot be blank")
            parsed["energy_ev"] = _finite_float(
                row["energy_ev"],
                field="energy_ev",
                source=Path(row["_source_path"]),
                row_number=int(row["_row_number"]),
                required=True,
            )
            parsed["oscillator_strength"] = _finite_float(
                row["oscillator_strength"],
                field="oscillator_strength",
                source=Path(row["_source_path"]),
                row_number=int(row["_row_number"]),
                required=True,
            )
            if (
                parsed["oscillator_strength"] is None
                or parsed["oscillator_strength"] < 0
            ):
                raise ValueError("oscillator_strength must be nonnegative")
            parsed["_assignment_basis"] = str(
                source.get("assignment_basis") or ""
            )
            parsed["_satellite_values"] = satellite_values
            rows.append(parsed)
    return rows, records


def _summarize_transitions(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["_source_id"]), str(row["edge"]))].append(row)
    sectors = list(spec.get("spectral_sectors") or [])
    output = []
    for (source_id, edge), selected in sorted(groups.items()):
        total = sum(float(row["oscillator_strength"]) for row in selected)
        if total <= 0:
            raise ValueError(
                f"Transition source {source_id!r}, edge {edge!r} has "
                "zero positive oscillator strength"
            )
        source_status = _status_from_quality(str(selected[0]["_quality"]))
        for sector in sectors:
            sector_edge = str(sector.get("edge") or "")
            if sector_edge and sector_edge != edge:
                continue
            emin = float(sector["emin_ev"])
            emax = float(sector["emax_ev"])
            if emax < emin:
                raise ValueError(
                    f"Spectral sector {sector.get('label')!r} has reversed bounds"
                )
            intensity = sum(
                float(row["oscillator_strength"])
                for row in selected
                if emin <= float(row["energy_ev"]) <= emax
            )
            output.append(
                {
                    "source_id": source_id,
                    "edge": edge,
                    "metric": "spectral_sector",
                    "label": str(sector.get("label") or "sector"),
                    "kind": str(sector.get("kind") or "descriptor"),
                    "emin_ev": emin,
                    "emax_ev": emax,
                    "oscillator_strength": intensity,
                    "fraction_of_edge_intensity": intensity / total,
                    "assignment_basis": "energy_window_only",
                    "scientific_status": (
                        "screening"
                        if str(sector.get("kind") or "descriptor")
                        in {"descriptor", "sideband"}
                        else source_status
                    ),
                }
            )
        assignment_basis = str(selected[0]["_assignment_basis"])
        accepted_satellites = [
            row
            for row in selected
            if str(row.get("satellite_class") or "").strip().lower()
            in row["_satellite_values"]
            and str(row.get("assignment_status") or "").strip().lower()
            == "accepted"
        ]
        if (
            assignment_basis == "state_resolved_configuration"
            and accepted_satellites
        ):
            satellite_intensity = sum(
                float(row["oscillator_strength"]) for row in accepted_satellites
            )
            satellite_fraction = satellite_intensity / total
            satellite_status = source_status
            note = (
                "Satellite rows have accepted state-resolved configuration "
                "assignments."
            )
        else:
            satellite_intensity = None
            satellite_fraction = None
            satellite_status = "unavailable"
            note = (
                "No explicitly named and accepted state-resolved satellite "
                "assignment; energy sidebands alone are not charge-transfer "
                "satellites."
            )
        output.append(
            {
                "source_id": source_id,
                "edge": edge,
                "metric": "satellite_intensity",
                "label": "accepted_satellite",
                "kind": "charge_transfer_satellite",
                "emin_ev": None,
                "emax_ev": None,
                "oscillator_strength": satellite_intensity,
                "fraction_of_edge_intensity": satellite_fraction,
                "assignment_basis": assignment_basis or "none",
                "scientific_status": satellite_status,
                "note": note,
            }
        )
    return output


def _best_status_rows(
    rows: list[dict[str, Any]],
    *,
    state_id: str,
) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("state_id") == state_id]


def _measure_row(
    state: dict[str, Any],
    measure: str,
    status: str,
    value: float | None,
    unit: str,
    source_ids: list[str],
    decision: str,
) -> dict[str, Any]:
    return {
        "state_id": str(state.get("id") or ""),
        "state_label": str(state.get("label") or state.get("id") or ""),
        "state_role": str(state.get("role") or ""),
        "edge": str(state.get("edge") or ""),
        "measure": measure,
        "scientific_status": status,
        "value": value,
        "unit": unit,
        "source_ids": ";".join(sorted(set(source_ids))),
        "decision": decision,
    }


def _build_measure_status(
    states: dict[str, dict[str, Any]],
    orbital: list[dict[str, Any]],
    configurations: list[dict[str, Any]],
    spectral: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for state_id, state in states.items():
        orbital_rows = _best_status_rows(orbital, state_id=state_id)
        identity_rows = [
            row for row in orbital_rows if row["identity_guard_enabled"]
        ]
        identity = identity_rows[0] if identity_rows else None
        energy_rows = [
            row for row in orbital_rows if row["orbital_energy_span_ev"] is not None
        ]
        radial_rows = [
            row
            for row in orbital_rows
            if row["occupation_weighted_radial_extent_angstrom"] is not None
        ]
        config_rows = _best_status_rows(configurations, state_id=state_id)
        config = config_rows[0] if config_rows else None
        secondary_candidates = [
            row
            for row in orbital_rows
            if row["secondary_projected_electrons"] is not None
        ]

        if energy_rows:
            row = energy_rows[0]
            output.append(
                _measure_row(
                    state,
                    "relative_orbital_energy",
                    row["scientific_status"],
                    row["orbital_energy_span_ev"],
                    "eV",
                    [row["source_id"]],
                    "Energy span within the declared orbital set; a state-to-state shift requires matched orbital identities.",
                )
            )
        else:
            output.append(
                _measure_row(
                    state,
                    "relative_orbital_energy",
                    "unavailable",
                    None,
                    "eV",
                    [],
                    "No matched orbital energies were supplied.",
                )
            )
        occupation_row = identity or (orbital_rows[0] if orbital_rows else None)
        if occupation_row:
            output.append(
                _measure_row(
                    state,
                    "natural_occupation",
                    occupation_row["scientific_status"],
                    occupation_row["electron_count"],
                    "electrons",
                    [occupation_row["source_id"]],
                    "Occupation sum over the explicitly declared orbital scope.",
                )
            )
        else:
            output.append(
                _measure_row(
                    state,
                    "natural_occupation",
                    "unavailable",
                    None,
                    "electrons",
                    [],
                    "No state-resolved orbital occupations were supplied.",
                )
            )
        if radial_rows:
            row = radial_rows[0]
            output.append(
                _measure_row(
                    state,
                    "occupation_spatial_extent",
                    row["scientific_status"],
                    row["occupation_weighted_radial_extent_angstrom"],
                    "angstrom",
                    [row["source_id"]],
                    "Occupation-weighted extent; inspect the identity guard before comparison.",
                )
            )
        else:
            output.append(
                _measure_row(
                    state,
                    "occupation_spatial_extent",
                    "unavailable",
                    None,
                    "angstrom",
                    [],
                    "No radial-extent data were supplied.",
                )
            )
        matched_radial_rows = [
            row for row in radial_rows if row["radial_extent_ratio"] is not None
        ]
        if matched_radial_rows:
            row = matched_radial_rows[0]
            output.append(
                _measure_row(
                    state,
                    "matched_radial_extent",
                    row["scientific_status"],
                    row["radial_extent_ratio"],
                    "reference_ratio",
                    [row["source_id"]],
                    "Ratio to the declared matched isolated-ion or fragment reference.",
                )
            )
        else:
            output.append(
                _measure_row(
                    state,
                    "matched_radial_extent",
                    "unavailable",
                    None,
                    "reference_ratio",
                    [],
                    "A matched reference radius and identity orbital source are required.",
                )
            )
        if identity and identity["primary_identity_fraction"] is not None:
            output.append(
                _measure_row(
                    state,
                    "isolated_reference_projection",
                    identity["scientific_status"],
                    identity["primary_identity_fraction"],
                    "fraction",
                    [identity["source_id"]],
                    "Occupation-weighted projection into the declared primary-shell reference subspace.",
                )
            )
        else:
            output.append(
                _measure_row(
                    state,
                    "isolated_reference_projection",
                    "unavailable",
                    None,
                    "fraction",
                    [],
                    "No identity-guarded isolated-reference projection was supplied.",
                )
            )
        ligand_candidates = [
            row
            for row in orbital_rows
            if row["ligand_projected_electrons"] is not None
        ]
        if ligand_candidates:
            row = ligand_candidates[0]
            output.append(
                _measure_row(
                    state,
                    "ligand_projected_electrons",
                    row["scientific_status"],
                    row["ligand_projected_electrons"],
                    "electrons_orbital_projection",
                    [row["source_id"]],
                    "Occupation-weighted projection into the declared ligand reference subspace.",
                )
            )
        else:
            output.append(
                _measure_row(
                    state,
                    "ligand_projected_electrons",
                    "unavailable",
                    None,
                    "electrons_orbital_projection",
                    [],
                    "No ligand-subspace orbital projection was supplied.",
                )
            )
        if config and config["ligand_hole_weight"] is not None:
            output.append(
                _measure_row(
                    state,
                    "ligand_hole_weight",
                    config["scientific_status"],
                    config["ligand_hole_weight"],
                    "fraction",
                    [config["source_id"]],
                    "Normalized many-electron configuration weight with one or more ligand holes.",
                )
            )
        else:
            output.append(
                _measure_row(
                    state,
                    "ligand_hole_weight",
                    "unavailable",
                    None,
                    "fraction",
                    [],
                    "No ligand-hole-resolved configuration weights were supplied.",
                )
            )
        if config and config["expected_secondary_electrons"] is not None:
            output.append(
                _measure_row(
                    state,
                    "secondary_shell_participation",
                    config["scientific_status"],
                    config["expected_secondary_electrons"],
                    "electrons_active_configuration",
                    [config["source_id"]],
                    "Expected secondary-shell occupation in the declared configuration sector.",
                )
            )
        elif secondary_candidates:
            row = secondary_candidates[0]
            output.append(
                _measure_row(
                    state,
                    "secondary_shell_participation",
                    row["scientific_status"],
                    row["secondary_projected_electrons"],
                    "electrons_orbital_projection",
                    [row["source_id"]],
                    "Secondary-shell projection over the declared orbital scope; not an active-configuration occupation.",
                )
            )
        else:
            output.append(
                _measure_row(
                    state,
                    "secondary_shell_participation",
                    "unavailable",
                    None,
                    "electrons",
                    [],
                    "No secondary-shell projection or configuration occupation was supplied.",
                )
            )
        for field, measure in (
            ("orbital_lz_expectation_hbar", "orbital_lz_expectation"),
            ("spin_sz_expectation_hbar", "spin_sz_expectation"),
            ("total_jz_expectation_hbar", "total_jz_expectation"),
        ):
            candidates = [
                row for row in orbital_rows if row.get(field) is not None
            ]
            if candidates:
                row = candidates[0]
                output.append(
                    _measure_row(
                        state,
                        measure,
                        row["scientific_status"],
                        row[field],
                        "hbar",
                        [row["source_id"]],
                        "Occupation-weighted expectation from the supplied state-resolved spinor/orbital table.",
                    )
                )
            else:
                output.append(
                    _measure_row(
                        state,
                        measure,
                        "unavailable",
                        None,
                        "hbar",
                        [],
                        "Not supplied; scalar orbitals must not be interpreted as spinor angular-momentum observables.",
                    )
                )
    for row in spectral:
        if row["metric"] != "satellite_intensity":
            continue
        output.append(
            {
                "state_id": "",
                "state_label": "Spectrum",
                "state_role": "spectrum",
                "edge": row["edge"],
                "measure": "satellite_intensity",
                "scientific_status": row["scientific_status"],
                "value": row["fraction_of_edge_intensity"],
                "unit": "fraction_of_edge_oscillator_strength",
                "source_ids": row["source_id"],
                "decision": row["note"],
            }
        )
    return output


def _overall_status(
    spec: dict[str, Any],
    states: dict[str, dict[str, Any]],
    measures: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    required = dict(spec.get("required_measures") or DEFAULT_REQUIRED_MEASURES)
    checks = []
    statuses = []
    for role, role_measures in required.items():
        if role == "spectrum":
            continue
        if role not in VALID_STATE_ROLES:
            raise ValueError(
                f"required_measures contains unsupported state role {role!r}"
            )
        role_states = [
            state_id
            for state_id, state in states.items()
            if str(state["role"]) == role
        ]
        for state_id in role_states:
            for measure in list(role_measures or []):
                matches = [
                    row
                    for row in measures
                    if row["state_id"] == state_id and row["measure"] == measure
                ]
                status = (
                    _combine_status(*(row["scientific_status"] for row in matches))
                    if matches
                    else "unavailable"
                )
                checks.append(
                    {
                        "scope": role,
                        "state_id": state_id,
                        "measure": measure,
                        "status": status,
                    }
                )
                statuses.append(status)
    for measure in list(required.get("spectrum") or []):
        matches = [
            row
            for row in measures
            if row["state_role"] == "spectrum" and row["measure"] == measure
        ]
        if not matches:
            status = "unavailable"
            checks.append(
                {
                    "scope": "spectrum",
                    "state_id": "",
                    "measure": measure,
                    "status": status,
                }
            )
            statuses.append(status)
        else:
            for row in matches:
                checks.append(
                    {
                        "scope": "spectrum",
                        "state_id": row["edge"],
                        "measure": measure,
                        "status": row["scientific_status"],
                    }
                )
                statuses.append(row["scientific_status"])
    if any(status == "rejected" for status in statuses):
        return "rejected", checks
    if any(status in {"unavailable", "screening", "provisional"} for status in statuses):
        return "provisional", checks
    return "accepted", checks


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _flatten_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    return [
        {field: _csv_value(row.get(field)) for field in fields}
        for row in rows
    ]


def _render_plot(
    spec: dict[str, Any],
    orbital: list[dict[str, Any]],
    configurations: list[dict[str, Any]],
    spectral: list[dict[str, Any]],
    outdir: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    panels = []
    radial = [
        row
        for row in orbital
        if row["occupation_weighted_radial_extent_angstrom"] is not None
    ]
    if radial:
        panels.append(("radial", radial))
    projected = [
        row
        for row in orbital
        if row["primary_projected_electrons"] is not None
        or row["secondary_projected_electrons"] is not None
    ]
    if projected:
        panels.append(("projection", projected))
    if configurations:
        panels.append(("configuration", configurations))
    spectral_rows = [
        row
        for row in spectral
        if row["fraction_of_edge_intensity"] is not None
    ]
    if spectral_rows:
        panels.append(("spectral", spectral_rows))
    if not panels:
        return []
    panels = panels[:4]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 7.2),
        constrained_layout=True,
    )
    axes_flat = list(axes.flat)
    primary = str(spec["shells"]["primary"])
    secondary = str(spec["shells"].get("secondary") or "secondary")
    for axis, (kind, rows) in zip(axes_flat, panels):
        bar_groups = []
        if kind == "radial":
            labels = [row["state_label"] for row in rows]
            values = [
                float(row["occupation_weighted_radial_extent_angstrom"])
                for row in rows
            ]
            bars = axis.bar(range(len(rows)), values, color="#287c8e")
            bar_groups.append(bars)
            reference = rows[0]["primary_reference_radial_extent_angstrom"]
            if reference is not None:
                axis.axhline(
                    float(reference),
                    color="#d08b19",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"reference {primary}",
                )
                axis.legend(frameon=False)
            axis.set_ylabel("Occupation-weighted radial extent (angstrom)")
            axis.set_title("Spatial extent / matched reference")
        elif kind == "projection":
            labels = [row["state_label"] for row in rows]
            x_values = list(range(len(rows)))
            width = 0.36
            primary_values = [
                float(row["primary_projected_electrons"] or 0.0) for row in rows
            ]
            secondary_values = [
                float(row["secondary_projected_electrons"] or 0.0) for row in rows
            ]
            bars = axis.bar(
                [value - width / 2 for value in x_values],
                primary_values,
                width,
                label=primary,
                color="#287c8e",
            )
            secondary_bars = axis.bar(
                [value + width / 2 for value in x_values],
                secondary_values,
                width,
                label=secondary,
                color="#d08b19",
            )
            bar_groups.extend((bars, secondary_bars))
            axis.set_ylabel("Projected electrons")
            axis.set_title("Isolated-reference shell projection")
            axis.legend(frameon=False)
        elif kind == "configuration":
            labels = [row["state_label"] for row in rows]
            x_values = list(range(len(rows)))
            width = 0.36
            ligand = [100 * float(row["ligand_hole_weight"] or 0.0) for row in rows]
            secondary_weight = [
                100 * float(row["weight_with_secondary_occupation"] or 0.0)
                for row in rows
            ]
            bars = axis.bar(
                [value - width / 2 for value in x_values],
                ligand,
                width,
                label="ligand-hole weight",
                color="#287c8e",
            )
            secondary_bars = axis.bar(
                [value + width / 2 for value in x_values],
                secondary_weight,
                width,
                label=f"any {secondary} occupation",
                color="#d08b19",
            )
            bar_groups.extend((bars, secondary_bars))
            axis.set_ylabel("Configuration weight (%)")
            axis.set_title("Many-electron configuration sectors")
            axis.legend(frameon=False)
        else:
            labels = [f"{row['edge']}\n{row['label']}" for row in rows]
            values = [
                100 * float(row["fraction_of_edge_intensity"]) for row in rows
            ]
            bars = axis.bar(range(len(rows)), values, color="#287c8e")
            bar_groups.append(bars)
            axis.set_ylabel("Fraction of edge intensity (%)")
            axis.set_title("Spectral-sector intensity audit")
        for bar_group in bar_groups:
            for bar, row in zip(bar_group, rows):
                if row["scientific_status"] == "rejected":
                    bar.set_hatch("//")
                    bar.set_edgecolor("#9b2f2f")
                    bar.set_alpha(0.6)
                elif row["scientific_status"] in {
                    "screening",
                    "provisional",
                    "unavailable",
                }:
                    bar.set_alpha(0.55)
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.grid(False)
    for axis in axes_flat[len(panels) :]:
        axis.axis("off")
    system_label = str(spec["system"]["label"])
    figure.suptitle(
        f"{system_label}: Bagus-style multi-measure covalency audit",
        fontsize=15,
    )
    png = outdir / "bagus_covalency_audit.png"
    pdf = outdir / "bagus_covalency_audit.pdf"
    figure.savefig(png, dpi=240)
    figure.savefig(pdf)
    plt.close(figure)
    return [str(png), str(pdf)]


def _write_readme(
    path: Path,
    *,
    spec: dict[str, Any],
    overall_status: str,
    measures: list[dict[str, Any]],
    plots: list[str],
) -> None:
    system = dict(spec["system"])
    shells = dict(spec["shells"])
    counts = defaultdict(int)
    for row in measures:
        counts[row["scientific_status"]] += 1
    lines = [
        f"# {system['label']} Bagus-Style Covalency Investigation",
        "",
        f"- Central element: `{system['central_element']}`",
        f"- Primary shell: `{shells['primary']}`",
        f"- Secondary shell: `{shells.get('secondary', '')}`",
        f"- Overall status: **{overall_status}**",
        "",
        "## Interpretation Guard",
        "",
        "This packet reports complementary orbital, charge-transfer, and "
        "spectroscopic measures. It does not define one universal percent-"
        "covalent score. A failed orbital-identity or matched-radial guard "
        "rejects the affected state comparison without invalidating unrelated "
        "transition energies or intensities.",
        "",
        "An energy-window sideband remains a descriptor. It becomes a "
        "charge-transfer satellite measure only when the transition table "
        "contains an accepted state-resolved configuration assignment.",
        "",
        "## Measure Counts",
        "",
    ]
    for status in ("accepted", "provisional", "screening", "unavailable", "rejected"):
        lines.append(f"- {status}: `{counts[status]}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `bagus_covalency_orbital_metrics.csv`",
            "- `bagus_covalency_configuration_metrics.csv`",
            "- `bagus_covalency_spectral_metrics.csv`",
            "- `bagus_covalency_measure_status.csv`",
            "- `bagus_covalency_canonical_orbitals.csv`",
            "- `bagus_covalency_canonical_configurations.csv`",
            "- `bagus_covalency_canonical_transitions.csv`",
            "- `bagus_covalency_summary.json`",
            "- `bagus_covalency_provenance.json`",
        ]
    )
    for plot in plots:
        lines.append(f"- `{Path(plot).name}`")
    caveats = list(spec.get("caveats") or [])
    if caveats:
        lines.extend(["", "## Project Caveats", ""])
        for caveat in caveats:
            lines.append(f"- {caveat}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def investigate(spec_path: Path, outdir: Path, *, make_plot: bool = True) -> dict[str, Any]:
    """Run a complete covalency investigation from a mapping specification."""

    spec_path = spec_path.resolve()
    spec = _load_spec(spec_path)
    states = _state_index(spec)
    base = spec_path.parent
    outdir = outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {outdir}. Use a new revision directory."
        )
    outdir.mkdir(parents=True, exist_ok=True)

    orbital_rows, orbital_sources = _load_orbitals(
        spec,
        base=base,
        states=states,
    )
    configuration_rows, configuration_sources = _load_configurations(
        spec,
        base=base,
        states=states,
    )
    transition_rows, transition_sources = _load_transitions(
        spec,
        base=base,
        states=states,
    )
    orbital = _summarize_orbitals(spec, orbital_rows, states)
    configurations = _summarize_configurations(
        spec,
        configuration_rows,
        states,
    )
    spectral = _summarize_transitions(spec, transition_rows)
    measures = _build_measure_status(states, orbital, configurations, spectral)
    overall_status, required_checks = _overall_status(spec, states, measures)

    orbital_fields = [
        "state_id",
        "state_label",
        "state_role",
        "edge",
        "source_id",
        "scope",
        "electron_count",
        "occupation_weighted_radial_extent_angstrom",
        "primary_reference_radial_extent_angstrom",
        "radial_extent_ratio",
        "primary_projected_electrons",
        "secondary_projected_electrons",
        "ligand_projected_electrons",
        "primary_projection_coverage_fraction",
        "primary_identity_fraction",
        "unprojected_primary_fraction",
        "orbital_energy_min_ev",
        "orbital_energy_max_ev",
        "orbital_energy_centroid_ev",
        "orbital_energy_span_ev",
        "orbital_lz_expectation_hbar",
        "spin_sz_expectation_hbar",
        "total_jz_expectation_hbar",
        "identity_guard_enabled",
        "scientific_status",
    ]
    configuration_fields = [
        "state_id",
        "state_label",
        "state_role",
        "edge",
        "source_id",
        "raw_weight_norm",
        "configuration_norm_guard",
        "ligand_hole_weight",
        "expected_ligand_holes",
        "expected_primary_electrons",
        "expected_secondary_electrons",
        "weight_with_secondary_occupation",
        "coupled_ligand_hole_secondary_weight",
        "expected_core_holes",
        "scientific_status",
    ]
    spectral_fields = [
        "source_id",
        "edge",
        "metric",
        "label",
        "kind",
        "emin_ev",
        "emax_ev",
        "oscillator_strength",
        "fraction_of_edge_intensity",
        "assignment_basis",
        "scientific_status",
        "note",
    ]
    measure_fields = [
        "state_id",
        "state_label",
        "state_role",
        "edge",
        "measure",
        "scientific_status",
        "value",
        "unit",
        "source_ids",
        "decision",
    ]
    canonical_orbital_fields = [
        "source_id",
        "source_path",
        "source_row",
        "scientific_status",
        "identity_guard",
        "state_id",
        "orbital_id",
        "orbital_scope",
        "occupation",
        "orbital_energy_ev",
        "radial_extent_angstrom",
        "primary_projection",
        "secondary_projection",
        "ligand_projection",
        "orbital_lz_hbar",
        "spin_sz_hbar",
        "total_jz_hbar",
        "identity_status",
    ]
    canonical_configuration_fields = [
        "source_id",
        "source_path",
        "source_row",
        "scientific_status",
        "state_id",
        "configuration_class",
        "weight",
        "core_holes",
        "ligand_holes",
        "primary_electrons",
        "secondary_electrons",
    ]
    canonical_transition_fields = [
        "source_id",
        "source_path",
        "source_row",
        "scientific_status",
        "assignment_basis",
        "satellite_values",
        "state_id",
        "edge",
        "energy_ev",
        "oscillator_strength",
        "satellite_class",
        "assignment_status",
    ]
    orbital_csv = outdir / "bagus_covalency_orbital_metrics.csv"
    configuration_csv = outdir / "bagus_covalency_configuration_metrics.csv"
    spectral_csv = outdir / "bagus_covalency_spectral_metrics.csv"
    measure_csv = outdir / "bagus_covalency_measure_status.csv"
    canonical_orbital_csv = outdir / "bagus_covalency_canonical_orbitals.csv"
    canonical_configuration_csv = (
        outdir / "bagus_covalency_canonical_configurations.csv"
    )
    canonical_transition_csv = (
        outdir / "bagus_covalency_canonical_transitions.csv"
    )
    _write_csv(
        orbital_csv,
        _flatten_rows(orbital, orbital_fields),
        orbital_fields,
    )
    _write_csv(
        configuration_csv,
        _flatten_rows(configurations, configuration_fields),
        configuration_fields,
    )
    _write_csv(
        spectral_csv,
        _flatten_rows(spectral, spectral_fields),
        spectral_fields,
    )
    _write_csv(
        measure_csv,
        _flatten_rows(measures, measure_fields),
        measure_fields,
    )
    _write_csv(
        canonical_orbital_csv,
        [
            {
                "source_id": row["_source_id"],
                "source_path": row["_source_path"],
                "source_row": row["_row_number"],
                "scientific_status": _status_from_quality(row["_quality"]),
                "identity_guard": row["_identity_guard"],
                **{
                    field: _csv_value(row.get(field))
                    for field in canonical_orbital_fields[5:]
                },
            }
            for row in orbital_rows
        ],
        canonical_orbital_fields,
    )
    _write_csv(
        canonical_configuration_csv,
        [
            {
                "source_id": row["_source_id"],
                "source_path": row["_source_path"],
                "source_row": row["_row_number"],
                "scientific_status": _status_from_quality(row["_quality"]),
                **{
                    field: _csv_value(row.get(field))
                    for field in canonical_configuration_fields[4:]
                },
            }
            for row in configuration_rows
        ],
        canonical_configuration_fields,
    )
    _write_csv(
        canonical_transition_csv,
        [
            {
                "source_id": row["_source_id"],
                "source_path": row["_source_path"],
                "source_row": row["_row_number"],
                "scientific_status": _status_from_quality(row["_quality"]),
                "assignment_basis": row["_assignment_basis"],
                "satellite_values": ";".join(sorted(row["_satellite_values"])),
                **{
                    field: _csv_value(row.get(field))
                    for field in canonical_transition_fields[6:]
                },
            }
            for row in transition_rows
        ],
        canonical_transition_fields,
    )
    plots = _render_plot(spec, orbital, configurations, spectral, outdir) if make_plot else []
    source_records = orbital_sources + configuration_sources + transition_sources
    external_sources = []
    for source in list(spec.get("sources") or []):
        record = dict(source)
        path_text = str(record.get("path") or "")
        if path_text and "://" not in path_text:
            source_path = _resolve_path(path_text, base)
            record["resolved_path"] = str(source_path)
            if source_path.is_file():
                record["sha256"] = _sha256(source_path)
                record["bytes"] = source_path.stat().st_size
        external_sources.append(record)
    summary_path = outdir / "bagus_covalency_summary.json"
    provenance_path = outdir / "bagus_covalency_provenance.json"
    readme = outdir / "BAGUS_COVALENCY_README.md"
    _write_readme(
        readme,
        spec=spec,
        overall_status=overall_status,
        measures=measures,
        plots=plots,
    )
    summary = {
        "schema": SCHEMA_RESULT,
        "created_at": _utc_now(),
        "system": dict(spec["system"]),
        "method": dict(spec.get("method") or {}),
        "shells": dict(spec["shells"]),
        "reference": dict(spec.get("reference") or {}),
        "overall_scientific_status": overall_status,
        "states": list(spec["states"]),
        "required_measure_checks": required_checks,
        "measure_status_counts": {
            status: sum(
                row["scientific_status"] == status for row in measures
            )
            for status in VALID_STATUSES
        },
        "outputs": {
            "orbital_metrics_csv": str(orbital_csv),
            "configuration_metrics_csv": str(configuration_csv),
            "spectral_metrics_csv": str(spectral_csv),
            "measure_status_csv": str(measure_csv),
            "canonical_orbitals_csv": str(canonical_orbital_csv),
            "canonical_configurations_csv": str(canonical_configuration_csv),
            "canonical_transitions_csv": str(canonical_transition_csv),
            "summary_json": str(summary_path),
            "provenance_json": str(provenance_path),
            "readme": str(readme),
            "plots": plots,
        },
        "results": {
            "orbital_metrics": orbital,
            "configuration_metrics": configurations,
            "spectral_metrics": spectral,
            "measure_status": measures,
        },
        "interpretation_guard": (
            "No universal percent-covalent score is computed. State comparisons "
            "require matched orbital identity and reference definitions. "
            "Energy-window sidebands are not charge-transfer satellites without "
            "accepted state-resolved configuration assignments."
        ),
        "caveats": list(spec.get("caveats") or []),
    }
    _write_json(summary_path, summary)
    provenance = {
        "schema": SCHEMA_PROVENANCE,
        "created_at": _utc_now(),
        "spec": {
            "path": str(spec_path),
            "sha256": _sha256(spec_path),
        },
        "input_tables": source_records,
        "external_sources": external_sources,
        "transformations": [
            "occupation-weighted orbital aggregation",
            "matched radial-extent ratio",
            "isolated-reference shell projection aggregation",
            "configuration-weight normalization",
            "ligand-hole and secondary-shell configuration aggregation",
            "edge-wise oscillator-strength integration",
            "canonical plotting-table export before aggregation",
        ],
        "outputs": [
            str(orbital_csv),
            str(configuration_csv),
            str(spectral_csv),
            str(measure_csv),
            str(canonical_orbital_csv),
            str(canonical_configuration_csv),
            str(canonical_transition_csv),
            str(summary_path),
            str(provenance_path),
            str(readme),
            *plots,
        ],
    }
    _write_json(provenance_path, provenance)
    return summary


def run_from_args(args: argparse.Namespace) -> int:
    if args.write_template is not None:
        path = args.write_template.resolve()
        if path.exists() and not args.force:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, template_spec())
        print(f"Wrote Bagus covalency specification template: {path}")
        return 0
    if args.spec is None:
        raise ValueError("--spec is required unless --write-template is used")
    summary = investigate(
        args.spec,
        args.outdir,
        make_plot=not args.no_plot,
    )
    print(
        "Bagus covalency investigation status: "
        f"{summary['overall_scientific_status']}"
    )
    print(f"Wrote investigation packet: {args.outdir.resolve()}")
    return 0


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", type=Path)
    parser.add_argument(
        "--write-template",
        type=Path,
        help="Write a chemistry-neutral JSON specification template and exit.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("molcas_bagus_covalency"),
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting only the requested template file.",
    )
    parser.set_defaults(func=run_from_args)


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "bagus-covalency",
        help=(
            "Investigate matched ground/excited covalency using orbital, "
            "configuration, and spectral postanalysis tables."
        ),
    )
    _configure_parser(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "General Bagus-style covalency investigation downstream of "
            "OpenMolcas postanalysis."
        )
    )
    _configure_parser(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
