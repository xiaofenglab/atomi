"""Fetch and compare external material-property sources for MOOSE workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from atomi.moose.material_export import MOOSE_FIELDS, format_value, load_thermo_grid


SOURCE_FIELDS = MOOSE_FIELDS + ["provider", "material_id", "citation", "notes"]
COMPARE_FIELDS = [
    "source",
    "T_K",
    "field",
    "value",
    "reference_source",
    "reference_value",
    "delta",
    "relative_delta",
]
PLOT_FIELDS = [
    "k_W_mK",
    "Cp_J_kgK",
    "rho_kg_m3",
    "alpha_1_K",
    "dilatation",
    "E_Pa",
    "nu",
    "K_Pa",
    "G_Pa",
]
PREDICTION_REQUIREMENTS = {
    "thermal-stress": {
        "required": ["T_K", "k_W_mK", "Cp_J_kgK", "rho_kg_m3", "E_Pa", "nu"],
        "one_of": [["alpha_1_K", "dilatation"]],
        "description": "Heat conduction plus quasi-static thermal stress.",
    },
    "transient-thermal": {
        "required": ["T_K", "k_W_mK", "Cp_J_kgK", "rho_kg_m3"],
        "one_of": [],
        "description": "Transient heat conduction.",
    },
    "elastic-thermal": {
        "required": ["T_K", "E_Pa", "nu"],
        "one_of": [["alpha_1_K", "dilatation"]],
        "description": "Temperature-dependent elastic/thermal expansion material model.",
    },
    "phase-redistribution": {
        "required": ["T_K", "phase_free_energy", "chemical_potential", "mobility"],
        "one_of": [],
        "description": "Phase-field or diffusion-driven phase redistribution; requires thermodynamic and kinetic models.",
    },
}
FIELD_SOURCE_HINTS = {
    "T_K": ["thermo_qha_md", "user-csv"],
    "Cp_J_kgK": ["thermo_qha_md", "CALPHAD/TDB", "NIST-JANAF curated CSV"],
    "rho_kg_m3": ["thermo_qha_md density/volume", "experiment/handbook curated CSV"],
    "alpha_1_K": ["thermo_qha_md lattice/volume fits", "experiment/handbook curated CSV"],
    "dilatation": ["thermo_qha_md lattice/volume fits"],
    "k_W_mK": ["MLIP-MD transport workflow", "BISON/IAEA/literature curated CSV"],
    "E_Pa": ["MLIP-MD elastic workflow", "Materials Project", "AFLOW", "literature curated CSV"],
    "nu": ["MLIP-MD elastic workflow", "Materials Project", "AFLOW", "literature curated CSV"],
    "K_Pa": ["MLIP-MD elastic workflow", "Materials Project", "AFLOW"],
    "G_Pa": ["MLIP-MD elastic workflow", "Materials Project", "AFLOW"],
    "phase_free_energy": ["pycalphad/TDB", "Thermochimica", "custom MOOSE material model"],
    "chemical_potential": ["pycalphad/TDB", "Thermochimica", "custom MOOSE material model"],
    "mobility": ["diffusion database", "literature", "fitted phase-field kinetics"],
}
FIELD_INPUT_DETAILS = {
    "T_K": {"type": "temperature grid", "units": "K", "columns": ["T_K"]},
    "Cp_J_kgK": {
        "type": "specific heat capacity",
        "units": "J kg^-1 K^-1",
        "columns": ["T_K", "Cp_J_kgK"],
    },
    "rho_kg_m3": {
        "type": "mass density",
        "units": "kg m^-3",
        "columns": ["T_K", "rho_kg_m3"],
    },
    "alpha_1_K": {
        "type": "linear thermal expansion coefficient",
        "units": "K^-1",
        "columns": ["T_K", "alpha_1_K"],
    },
    "dilatation": {
        "type": "stress-free relative thermal strain",
        "units": "dimensionless",
        "columns": ["T_K", "dilatation"],
    },
    "k_W_mK": {
        "type": "thermal conductivity",
        "units": "W m^-1 K^-1",
        "columns": ["T_K", "k_W_mK"],
    },
    "E_Pa": {"type": "Young's modulus", "units": "Pa", "columns": ["T_K", "E_Pa"]},
    "nu": {"type": "Poisson ratio", "units": "dimensionless", "columns": ["T_K", "nu"]},
    "K_Pa": {"type": "bulk modulus", "units": "Pa", "columns": ["T_K", "K_Pa"]},
    "G_Pa": {"type": "shear modulus", "units": "Pa", "columns": ["T_K", "G_Pa"]},
    "phase_free_energy": {
        "type": "phase Gibbs/free energy model or tabulated function",
        "units": "J mol^-1 or MOOSE-consistent free-energy density",
        "columns": ["T_K", "composition", "phase", "phase_free_energy"],
    },
    "chemical_potential": {
        "type": "chemical potential model or table",
        "units": "J mol^-1",
        "columns": ["T_K", "composition", "component", "chemical_potential"],
    },
    "mobility": {
        "type": "phase-field mobility or diffusivity-derived kinetic coefficient",
        "units": "MOOSE model dependent",
        "columns": ["T_K", "composition", "phase", "mobility"],
    },
}
QHA_MD_NATIVE_FIELDS = {"T_K", "Cp_J_kgK", "rho_kg_m3", "alpha_1_K", "dilatation"}


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_source_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Source label is empty in {value!r}")
    return label, Path(path)


def row_has_field(rows: list[dict[str, Any]], field: str) -> bool:
    return any(finite_float(row.get(field)) is not None for row in rows)


def infer_qha_md_fields(qha_md_dir: Path | None) -> tuple[dict[str, bool], dict[str, Any]]:
    fields = {field: False for field in MOOSE_FIELDS}
    metadata: dict[str, Any] = {"available": False}
    if qha_md_dir is None:
        return fields, metadata
    try:
        rows, source_info = load_thermo_grid(qha_md_dir)
    except SystemExit as exc:
        metadata.update({"available": False, "error": str(exc), "qha_md_dir": str(qha_md_dir)})
        return fields, metadata
    fields["T_K"] = bool(rows)
    fields["Cp_J_kgK"] = any(row.get("Cp_J_molK") is not None for row in rows)
    fields["Cp_std_J_kgK"] = any(row.get("Cp_std_J_molK") is not None for row in rows)
    fields["rho_kg_m3"] = any(
        row.get("density_g_cm3") is not None or row.get("V_A3") is not None for row in rows
    )
    fields["alpha_1_K"] = any(row.get("alpha_1_K") is not None for row in rows)
    fields["dilatation"] = any(row.get("dilatation_source_value") is not None for row in rows)
    metadata.update(
        {
            "available": True,
            "qha_md_dir": str(qha_md_dir),
            "rows": len(rows),
            "temperature_range_K": [rows[0]["T_K"], rows[-1]["T_K"]] if rows else None,
            "source_info": source_info,
        }
    )
    return fields, metadata


def infer_csv_fields(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    rows = read_rows(path)
    fields = set(rows[0].keys()) if rows else set()
    result: dict[str, bool] = {}
    for field in set(MOOSE_FIELDS + list(FIELD_SOURCE_HINTS)):
        result[field] = field in fields and row_has_field(rows, field)
    return result


def merge_field_sources(
    qha_fields: dict[str, bool],
    csv_sources: list[tuple[str, Path]],
) -> dict[str, list[str]]:
    present: dict[str, list[str]] = {}
    for field, available in qha_fields.items():
        if available:
            present.setdefault(field, []).append("thermo_qha_md")
    for label, path in csv_sources:
        for field, available in infer_csv_fields(path).items():
            if available:
                present.setdefault(field, []).append(label)
    return present


def screen_plan(
    *,
    prediction: str,
    material: str,
    qha_md_dir: Path | None,
    csv_sources: list[tuple[str, Path]],
) -> dict[str, Any]:
    if prediction not in PREDICTION_REQUIREMENTS:
        choices = ", ".join(sorted(PREDICTION_REQUIREMENTS))
        raise SystemExit(f"Unknown prediction {prediction!r}. Choices: {choices}")
    qha_fields, qha_metadata = infer_qha_md_fields(qha_md_dir)
    present = merge_field_sources(qha_fields, csv_sources)
    spec = PREDICTION_REQUIREMENTS[prediction]
    required = list(spec["required"])
    present_required = [field for field in required if present.get(field)]
    missing_required = [field for field in required if not present.get(field)]
    one_of_status = []
    for group in spec["one_of"]:
        provided = [field for field in group if present.get(field)]
        one_of_status.append(
            {
                "options": group,
                "provided": provided,
                "missing": [] if provided else group,
                "satisfied": bool(provided),
            }
        )
    missing_structural = [
        field
        for group in one_of_status
        if not group["satisfied"]
        for field in group["missing"]
    ]
    missing_all = missing_required + missing_structural
    recommendations = []
    for field in missing_all:
        recommendations.append(
            {
                "field": field,
                **FIELD_INPUT_DETAILS.get(
                    field,
                    {"type": "material property", "units": "", "columns": ["T_K", field]},
                ),
                "candidate_sources": FIELD_SOURCE_HINTS.get(field, ["user-csv"]),
                "commands": recommendation_commands(field, material),
            }
        )
    qha_detected = [field for field in sorted(QHA_MD_NATIVE_FIELDS) if qha_fields.get(field)]
    qha_missing_relevant = [
        field for field in sorted(QHA_MD_NATIVE_FIELDS) if field in required and not qha_fields.get(field)
    ]
    for group in spec["one_of"]:
        native_options = [field for field in group if field in QHA_MD_NATIVE_FIELDS]
        if native_options and not any(qha_fields.get(field) for field in native_options):
            qha_missing_relevant.extend(
                field for field in native_options if not qha_fields.get(field)
            )
    needed_external_inputs = [
        item
        for item in recommendations
        if "thermo_qha_md" not in FIELD_SOURCE_HINTS.get(item["field"], [])
    ]
    result = {
        "material": material,
        "prediction": prediction,
        "description": spec["description"],
        "qha_md": qha_metadata,
        "qha_md_detected_fields": qha_detected,
        "qha_md_missing_relevant_fields": qha_missing_relevant,
        "qha_md_native_fields": sorted(QHA_MD_NATIVE_FIELDS),
        "external_sources_provided": bool(csv_sources),
        "needed_external_inputs": needed_external_inputs,
        "expected_external_csv_columns": sorted(
            {
                column
                for item in needed_external_inputs
                for column in item.get("columns", [])
                if column
            }
        ),
        "sources": [{"label": label, "path": str(path)} for label, path in csv_sources],
        "present_fields": {field: sources for field, sources in sorted(present.items())},
        "required_fields": required,
        "present_required": present_required,
        "missing_required": missing_required,
        "one_of": one_of_status,
        "ready": not missing_required and all(group["satisfied"] for group in one_of_status),
        "recommendations": recommendations,
    }
    result["next_commands"] = next_workflow_commands(result)
    return result


def next_workflow_commands(plan: dict[str, Any]) -> list[dict[str, str]]:
    material = str(plan["material"])
    material_key = material.lower()
    qha_dir = plan["qha_md"].get("qha_md_dir") or "analysis/thermo_qha_md"
    property_csv = f"{material_key}_external_properties.csv"
    out_csv = f"{material_key}_moose_material_properties.csv"
    out_meta = f"{material_key}_moose_material_properties.meta.json"
    include = f"{material_key}_material_functions.i"
    commands = []
    if not plan["qha_md"].get("available"):
        commands.append(
            {
                "step": "locate-qha-md",
                "command": (
                    "Run or locate the Atomi DFT-QHA/MD thermodynamic analysis directory, "
                    "then rerun this screen with --qha-md-dir <thermo_qha_md_dir>."
                ),
            }
        )
    if plan["needed_external_inputs"]:
        commands.append(
            {
                "step": "prepare-external-property-csv",
                "command": (
                    "Create a curated CSV with columns "
                    + ",".join(plan["expected_external_csv_columns"])
                    + f" and pass it as --property-csv {property_csv}."
                ),
            }
        )
        seen_source_commands = set()
        for item in plan["needed_external_inputs"]:
            for command in item.get("commands", []):
                if command in seen_source_commands:
                    continue
                seen_source_commands.add(command)
                commands.append({"step": f"source-{item['field']}", "command": command})
    if plan["prediction"] in {"thermal-stress", "transient-thermal", "elastic-thermal"}:
        commands.append(
            {
                "step": "compile-material-table",
                "command": (
                    "moose-qha-md-material "
                    f"--qha-md-dir {qha_dir} "
                    f"--property-csv {property_csv} "
                    f"--out-csv {out_csv} "
                    f"--out-meta {out_meta} "
                    f"--moose-include {include}"
                ),
            }
        )
    if plan["prediction"] == "thermal-stress":
        commands.append(
            {
                "step": "write-moose-input",
                "command": (
                    "moose-thermal-stress "
                    f"--material {material} "
                    f"--material-csv {out_csv} "
                    f"--material-meta {out_meta} "
                    f"--material-include {include}"
                ),
            }
        )
    elif plan["prediction"] == "phase-redistribution":
        commands.append(
            {
                "step": "build-phase-material-model",
                "command": (
                    "Use pycalphad/Thermochimica or a curated phase-thermodynamics table "
                    "to build MOOSE free-energy, chemical-potential, and mobility inputs."
                ),
            }
        )
    if plan["needed_external_inputs"] and plan["prediction"] != "phase-redistribution":
        commands.append(
            {
                "step": "compare-sources",
                "command": (
                    "moose-material-compare "
                    f"--source atomi={out_csv} "
                    f"--source external={property_csv} "
                    f"--outdir {material_key}_property_comparison"
                ),
            }
        )
    return commands


def recommendation_commands(field: str, material: str) -> list[str]:
    commands = []
    if field in {"E_Pa", "nu", "K_Pa", "G_Pa"}:
        commands.extend(
            [
                (
                    "moose-material-source --provider materials-project "
                    f"--material {material} --out-csv mp_{material.lower()}_elastic.csv"
                ),
                (
                    "moose-material-source --provider aflow "
                    f"--material {material} --out-csv aflow_{material.lower()}_elastic.csv"
                ),
            ]
        )
    if field in {"k_W_mK", "Cp_J_kgK", "rho_kg_m3", "alpha_1_K"}:
        commands.append(
            "moose-material-source --provider user-csv --input curated_properties.csv "
            "--source-label literature --citation 'fill citation'"
        )
    if field in {"phase_free_energy", "chemical_potential"}:
        commands.append(
            "Generate phase thermodynamics with pycalphad/Thermochimica, then pass as "
            "custom MOOSE functions/materials."
        )
    if field == "mobility":
        commands.append("Provide mobility/diffusivity from literature, MD, or a fitted kinetic model.")
    return commands


def render_screen_markdown(plan: dict[str, Any]) -> str:
    lines = [
        f"# MOOSE Material Screen: {plan['material']} / {plan['prediction']}",
        "",
        plan["description"],
        "",
        f"Ready: {'yes' if plan['ready'] else 'no'}",
        "",
        "## QHA/MD Folder",
    ]
    if plan["qha_md"].get("available"):
        lines.append(f"- available: yes ({plan['qha_md'].get('rows', 0)} rows)")
        temp_range = plan["qha_md"].get("temperature_range_K")
        if temp_range:
            lines.append(f"- temperature range: {temp_range[0]} to {temp_range[1]} K")
        if plan["qha_md_detected_fields"]:
            lines.append(
                "- detected fields: "
                + ", ".join(f"`{field}`" for field in plan["qha_md_detected_fields"])
            )
        else:
            lines.append("- detected fields: none")
        if plan["qha_md_missing_relevant_fields"]:
            lines.append(
                "- QHA/MD fields still missing for this prediction: "
                + ", ".join(f"`{field}`" for field in plan["qha_md_missing_relevant_fields"])
            )
    else:
        lines.append("- available: no")
        if plan["qha_md"].get("error"):
            lines.append(f"- reason: {plan['qha_md']['error']}")
    lines.extend(
        [
            "",
        "## Present Fields",
        ]
    )
    if plan["present_fields"]:
        for field, sources in plan["present_fields"].items():
            lines.append(f"- `{field}`: {', '.join(sources)}")
    else:
        lines.append("- none detected")
    lines.extend(["", "## Missing Required Fields"])
    if plan["missing_required"]:
        for field in plan["missing_required"]:
            lines.append(f"- `{field}`")
    else:
        lines.append("- none")
    for group in plan["one_of"]:
        if not group["satisfied"]:
            lines.append(f"- one of: {', '.join(f'`{field}`' for field in group['options'])}")
    lines.extend(["", "## Recommendations"])
    if plan["recommendations"]:
        for item in plan["recommendations"]:
            detail = item.get("type", "material property")
            units = item.get("units")
            columns = ", ".join(f"`{column}`" for column in item.get("columns", []))
            lines.append(f"- `{item['field']}` ({detail}; {units}): {', '.join(item['candidate_sources'])}")
            if columns:
                lines.append(f"  - expected columns: {columns}")
            for command in item["commands"]:
                lines.append(f"  - `{command}`")
    else:
        lines.append("- all required inputs are present")
    lines.extend(["", "## Next Commands"])
    if plan["next_commands"]:
        for item in plan["next_commands"]:
            lines.append(f"- {item['step']}: `{item['command']}`")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def formula_to_aflow_compound(formula: str) -> str:
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    if not tokens:
        raise SystemExit(f"Cannot parse formula for AFLOW query: {formula!r}")
    parts = []
    for element, count in sorted(tokens):
        parts.append(f"{element}{count or '1'}")
    return "".join(parts)


def load_api_key_json(path: Path, provider: str, env_name: str) -> tuple[str | None, str | None]:
    if not path.exists():
        raise SystemExit(f"API key JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse API key JSON {path}: {exc}") from exc

    candidates = [
        payload.get(env_name),
        payload.get("materials_project_api_key") if provider == "materials_project" else None,
        payload.get("materials_project_api_key") if provider == "materials-project" else None,
    ]
    for provider_key in {provider, provider.replace("-", "_")}:
        nested = payload.get(provider_key)
        if isinstance(nested, dict):
            candidates.extend(
                [
                    nested.get("api_key"),
                    nested.get(env_name),
                    nested.get("materials_project_api_key"),
                ]
            )

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip(), str(path)
    return None, str(path)


def resolve_materials_project_api_key(args: argparse.Namespace) -> tuple[str | None, str]:
    if args.api_key_env:
        value = os.environ.get(args.api_key_env)
        if value:
            return value, f"env:{args.api_key_env}"
    json_candidates: list[Path] = []
    if args.api_key_json is not None:
        json_candidates.append(args.api_key_json)
    if os.environ.get("ATOMI_API_KEYS_JSON"):
        json_candidates.append(Path(os.environ["ATOMI_API_KEYS_JSON"]))
    json_candidates.extend(
        [
            Path.home() / "atomi_hpc/atomi_hpc_config.kit.local.json",
            Path.home() / "hpc_atomi/atomi_hpc_config.kit.local.json",
        ]
    )
    seen: set[Path] = set()
    for json_path in json_candidates:
        json_path = json_path.expanduser()
        if json_path in seen or not json_path.exists():
            continue
        seen.add(json_path)
        key, source = load_api_key_json(json_path, "materials_project", args.api_key_env)
        if key:
            return key, f"json:{source}"
    return None, "none"


def elastic_row_from_kg(
    *,
    provider: str,
    material: str,
    material_id: str,
    k_gpa: float | None,
    g_gpa: float | None,
    nu: float | None,
    citation: str,
    notes: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {field: None for field in SOURCE_FIELDS}
    row["provider"] = provider
    row["material_id"] = material_id
    row["citation"] = citation
    row["notes"] = notes
    row["source_tag"] = f"{provider}:{material_id or material}"
    if k_gpa is not None:
        row["K_Pa"] = k_gpa * 1e9
    if g_gpa is not None:
        row["G_Pa"] = g_gpa * 1e9
    if k_gpa is not None and g_gpa is not None and (3.0 * k_gpa + g_gpa) != 0:
        row["E_Pa"] = 9.0 * k_gpa * g_gpa / (3.0 * k_gpa + g_gpa) * 1e9
    if nu is not None:
        row["nu"] = nu
    elif k_gpa is not None and g_gpa is not None and (2.0 * (3.0 * k_gpa + g_gpa)) != 0:
        row["nu"] = (3.0 * k_gpa - 2.0 * g_gpa) / (2.0 * (3.0 * k_gpa + g_gpa))
    return row


def doc_value(doc: Any, key: str, default: Any = None) -> Any:
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


def nested_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def serial_text(value: Any) -> str:
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return str(enum_value)
    return str(value)


def nested_number(obj: Any, *keys: str) -> float | None:
    current = obj
    for key in keys:
        current = nested_value(current, key, None)
        if current is None:
            return None
    return finite_float(current)


def materials_project_modulus_gpa(doc: Any, field: str) -> float | None:
    """Read MP elastic moduli from old flat or newer nested summary fields."""

    flat_key = {"bulk": "k_vrh", "shear": "g_vrh"}[field]
    flat = finite_float(doc_value(doc, flat_key))
    if flat is not None:
        return flat
    nested_key = {"bulk": "bulk_modulus", "shear": "shear_modulus"}[field]
    nested = doc_value(doc, nested_key, None)
    for key in ("vrh", "value", "mean"):
        value = nested_number(nested, key)
        if value is not None:
            return value
    return finite_float(nested)


def materials_project_poisson(doc: Any) -> float | None:
    value = finite_float(doc_value(doc, "homogeneous_poisson", None))
    if value is not None:
        return value
    value = finite_float(doc_value(doc, "poisson_ratio", None))
    if value is not None:
        return value
    elastic = doc_value(doc, "elasticity", None)
    return nested_number(elastic, "homogeneous_poisson")


def normalize_phase_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def materials_project_symmetry(doc: Any) -> dict[str, Any]:
    symmetry = doc_value(doc, "symmetry", {}) or {}
    return {
        "symbol": serial_text(
            nested_value(symmetry, "symbol", "") or nested_value(symmetry, "space_group_symbol", "")
        ),
        "number": nested_value(symmetry, "number", None) or nested_value(symmetry, "space_group_number", None),
        "crystal_system": serial_text(nested_value(symmetry, "crystal_system", "")),
    }


def materials_project_candidate_summary(doc: Any) -> dict[str, Any]:
    symmetry = materials_project_symmetry(doc)
    return {
        "material_id": str(doc_value(doc, "material_id", "")),
        "formula_pretty": doc_value(doc, "formula_pretty", ""),
        "k_vrh": materials_project_modulus_gpa(doc, "bulk"),
        "g_vrh": materials_project_modulus_gpa(doc, "shear"),
        "homogeneous_poisson": materials_project_poisson(doc),
        "symmetry_symbol": symmetry["symbol"],
        "symmetry_number": symmetry["number"],
        "crystal_system": symmetry["crystal_system"],
        "energy_above_hull": doc_value(doc, "energy_above_hull", None),
        "is_stable": doc_value(doc, "is_stable", None),
    }


def materials_project_phase_match(doc: Any, args: argparse.Namespace) -> bool:
    symmetry = materials_project_symmetry(doc)
    phase_tokens = [
        doc_value(doc, "material_id", ""),
        doc_value(doc, "formula_pretty", ""),
        symmetry["symbol"],
        symmetry["crystal_system"],
    ]
    requested_phase = normalize_phase_token(getattr(args, "phase", "") or "")
    requested_symbol = normalize_phase_token(getattr(args, "spacegroup_symbol", "") or "")
    if requested_phase and any(requested_phase == normalize_phase_token(token) for token in phase_tokens):
        return True
    if requested_symbol and requested_symbol == normalize_phase_token(symmetry["symbol"]):
        return True
    requested_number = getattr(args, "spacegroup_number", None)
    if requested_number is not None:
        try:
            if int(symmetry["number"]) == int(requested_number):
                return True
        except (TypeError, ValueError):
            return False
    return not (requested_phase or requested_symbol or requested_number is not None)


def materials_project_doc_score(doc: Any, args: argparse.Namespace) -> tuple[Any, ...]:
    symmetry = materials_project_symmetry(doc)
    requested_phase = normalize_phase_token(getattr(args, "phase", "") or "")
    requested_symbol = normalize_phase_token(getattr(args, "spacegroup_symbol", "") or "")
    requested_number = getattr(args, "spacegroup_number", None)
    phase_mismatch = 0
    if requested_phase:
        phase_tokens = [
            doc_value(doc, "material_id", ""),
            doc_value(doc, "formula_pretty", ""),
            symmetry["symbol"],
            symmetry["crystal_system"],
        ]
        phase_mismatch = 0 if any(requested_phase == normalize_phase_token(token) for token in phase_tokens) else 1
    symbol_mismatch = 0
    if requested_symbol:
        symbol_mismatch = 0 if requested_symbol == normalize_phase_token(symmetry["symbol"]) else 1
    number_mismatch = 0
    if requested_number is not None:
        try:
            number_mismatch = 0 if int(symmetry["number"]) == int(requested_number) else 1
        except (TypeError, ValueError):
            number_mismatch = 1
    missing_elastic = int(
        materials_project_modulus_gpa(doc, "bulk") is None
        or materials_project_modulus_gpa(doc, "shear") is None
    )
    prefer_stable = not getattr(args, "no_prefer_stable", False)
    unstable = 0
    if prefer_stable:
        is_stable = doc_value(doc, "is_stable", None)
        unstable = 0 if is_stable is True else 1 if is_stable is False else 0
    hull = finite_float(doc_value(doc, "energy_above_hull"))
    hull_score = hull if hull is not None else 1.0e9
    return (
        phase_mismatch,
        symbol_mismatch,
        number_mismatch,
        missing_elastic,
        unstable,
        hull_score,
        str(doc_value(doc, "material_id", "")),
    )


def select_materials_project_doc(docs: list[Any], args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if not docs:
        raise SystemExit(f"No Materials Project summary results for {args.material!r}")
    ranked = sorted(docs, key=lambda doc: materials_project_doc_score(doc, args))
    selected = ranked[0]
    exact_phase_matches = [doc for doc in docs if materials_project_phase_match(doc, args)]
    warnings: list[str] = []
    if (
        getattr(args, "phase", None)
        or getattr(args, "spacegroup_symbol", None)
        or getattr(args, "spacegroup_number", None) is not None
    ) and not exact_phase_matches:
        warnings.append(
            "No exact Materials Project phase/space-group match was found; "
            "selected the best scored formula match. Inspect candidate_summaries."
        )
    if (
        materials_project_modulus_gpa(selected, "bulk") is None
        or materials_project_modulus_gpa(selected, "shear") is None
    ):
        warnings.append(
            "Selected Materials Project entry does not contain both bulk and shear VRH elastic summaries."
        )
    selection = {
        "candidate_count": len(docs),
        "selected": materials_project_candidate_summary(selected),
        "candidate_summaries": [materials_project_candidate_summary(doc) for doc in ranked[:25]],
        "warnings": warnings,
        "selection_rules": [
            "prefer requested phase/space-group text or number",
            "prefer entries with bulk/shear VRH elastic summaries",
            "prefer stable / lower energy_above_hull entries unless --no-prefer-stable is set",
        ],
    }
    return selected, selection


def fetch_materials_project(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from mp_api.client import MPRester  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Materials Project fetching requires the optional mp-api package. "
            "Install mp-api and set MP_API_KEY, or provide --api-key-env."
        ) from exc
    api_key, api_key_source = resolve_materials_project_api_key(args)
    material_ids = [args.material_id] if args.material_id else None
    fields = [
        "material_id",
        "formula_pretty",
        "bulk_modulus",
        "shear_modulus",
        "homogeneous_poisson",
        "symmetry",
        "energy_above_hull",
        "is_stable",
    ]
    with MPRester(api_key) as mpr:
        if material_ids:
            docs = mpr.materials.summary.search(material_ids=material_ids, fields=fields)
        else:
            docs = mpr.materials.summary.search(formula=args.material, fields=fields)
    doc, selection = select_materials_project_doc(list(docs), args)
    get = doc.get if isinstance(doc, dict) else lambda key, default=None: getattr(doc, key, default)
    symmetry = materials_project_symmetry(doc)
    row = elastic_row_from_kg(
        provider="materials-project",
        material=args.material,
        material_id=str(get("material_id", "")),
        k_gpa=materials_project_modulus_gpa(doc, "bulk"),
        g_gpa=materials_project_modulus_gpa(doc, "shear"),
        nu=materials_project_poisson(doc),
        citation="Materials Project API; cite Materials Project and mp-api for retrieved data.",
        notes="0 K DFT-derived elastic summary; use as comparison/filler, not silent truth.",
    )
    metadata = {
        "provider": "materials-project",
        "material": args.material,
        "material_id": row["material_id"],
        "fields": ["bulk_modulus", "shear_modulus", "homogeneous_poisson"],
        "requested_phase": getattr(args, "phase", None),
        "requested_spacegroup_number": getattr(args, "spacegroup_number", None),
        "requested_spacegroup_symbol": getattr(args, "spacegroup_symbol", None),
        "selected_symmetry": symmetry,
        "selection": selection,
        "api_key_source": api_key_source,
        "source_url": "https://docs.materialsproject.org/",
    }
    return [row], metadata


def fetch_aflow(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compound = formula_to_aflow_compound(args.material)
    keywords = [
        "compound",
        "auid",
        "ael_bulk_modulus_vrh",
        "ael_shear_modulus_vrh",
        "ael_poisson_ratio",
    ]
    query = (
        f"compound({compound}),"
        + ",".join(keywords[1:])
        + ",$paging(1),$format(json)"
    )
    url = "https://aflow.org/API/aflux/v1.0/?" + urllib.parse.quote(query, safe="(),$:")
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise SystemExit(f"AFLOW AFLUX query failed: {url}\n{exc}") from exc
    if not payload:
        raise SystemExit(f"No AFLOW results for compound query {compound!r}")
    datum = payload[0]
    row = elastic_row_from_kg(
        provider="aflow",
        material=args.material,
        material_id=str(datum.get("auid", "")),
        k_gpa=finite_float(datum.get("ael_bulk_modulus_vrh")),
        g_gpa=finite_float(datum.get("ael_shear_modulus_vrh")),
        nu=finite_float(datum.get("ael_poisson_ratio")),
        citation="AFLOW AFLUX API; cite AFLOW/AFLOWLIB and the retrieved entry.",
        notes="AFLOW AEL elastic summary; use as comparison/filler with provenance.",
    )
    metadata = {
        "provider": "aflow",
        "material": args.material,
        "compound_query": compound,
        "query_url": url,
        "keywords": keywords,
    }
    return [row], metadata


def normalize_user_csv(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not args.input:
        raise SystemExit("--input is required for provider=user-csv")
    rows = []
    for source in read_rows(args.input):
        row = {field: source.get(field) for field in SOURCE_FIELDS}
        row["provider"] = source.get("provider") or args.source_label or "user-csv"
        row["citation"] = source.get("citation") or args.citation
        row["notes"] = source.get("notes") or args.notes
        row["source_tag"] = source.get("source_tag") or row["provider"]
        rows.append(row)
    metadata = {
        "provider": "user-csv",
        "input": str(args.input),
        "citation": args.citation,
        "notes": args.notes,
    }
    return rows, metadata


def source_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-material-source",
        description="Fetch or normalize external material properties for moose-qha-md-material.",
    )
    parser.add_argument(
        "--provider",
        choices=("materials-project", "aflow", "user-csv"),
        required=True,
    )
    parser.add_argument("--material", default="UO2")
    parser.add_argument("--material-id", help="Provider-specific material id, e.g. mp-1234.")
    parser.add_argument(
        "--phase",
        help=(
            "Optional Materials Project phase hint. Matched loosely against material id, "
            "formula, symmetry symbol, or crystal system; use --spacegroup-* for stricter selection."
        ),
    )
    parser.add_argument("--spacegroup-number", type=int, help="Prefer this Materials Project space-group number.")
    parser.add_argument("--spacegroup-symbol", help="Prefer this Materials Project space-group symbol, e.g. Fm-3m.")
    parser.add_argument(
        "--no-prefer-stable",
        action="store_true",
        help="Do not prefer stable / lower-energy-above-hull Materials Project entries during formula lookup.",
    )
    parser.add_argument("--input", type=Path, help="Curated CSV for provider=user-csv.")
    parser.add_argument("--out-csv", type=Path, default=Path("material_source_properties.csv"))
    parser.add_argument("--out-meta", type=Path, default=Path("material_source_properties.meta.json"))
    parser.add_argument("--api-key-env", default="MP_API_KEY")
    parser.add_argument(
        "--api-key-json",
        type=Path,
        help=(
            "Local-only JSON containing a Materials Project key. Also configurable "
            "with ATOMI_API_KEYS_JSON. The key is never written to output metadata."
        ),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--source-label")
    parser.add_argument("--citation", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    if args.provider == "materials-project":
        rows, metadata = fetch_materials_project(args)
    elif args.provider == "aflow":
        rows, metadata = fetch_aflow(args)
    else:
        rows, metadata = normalize_user_csv(args)
    write_rows(args.out_csv, rows, SOURCE_FIELDS)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_meta}")


def screen_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-material-screen",
        description=(
            "Inspect QHA/MD and external source coverage for a target MOOSE prediction "
            "before compiling material tables."
        ),
    )
    parser.add_argument("--prediction", choices=sorted(PREDICTION_REQUIREMENTS), default="thermal-stress")
    parser.add_argument("--material", default="UO2")
    parser.add_argument("--qha-md-dir", type=Path)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Existing source/material CSV to include in the screen.",
    )
    parser.add_argument("--out-json", type=Path, default=Path("moose_material_screen.json"))
    parser.add_argument("--out-md", type=Path, default=Path("moose_material_screen.md"))
    parser.add_argument("--json", action="store_true", help="Also print JSON to stdout.")
    args = parser.parse_args(argv)

    sources = [parse_source_arg(item) for item in args.source]
    plan = screen_plan(
        prediction=args.prediction,
        material=args.material,
        qha_md_dir=args.qha_md_dir,
        csv_sources=sources,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(render_screen_markdown(plan), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    if args.json:
        print(json.dumps(plan, indent=2))


def collect_series(label: str, rows: list[dict[str, str]]) -> dict[str, dict[float, float]]:
    series: dict[str, dict[float, float]] = {field: {} for field in PLOT_FIELDS}
    for row in rows:
        temp = finite_float(row.get("T_K"))
        if temp is None:
            continue
        for field in PLOT_FIELDS:
            value = finite_float(row.get(field))
            if value is not None:
                series[field][temp] = value
    return series


def compare_sources(
    sources: list[tuple[str, Path]],
    *,
    reference_label: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[float, float]]]]:
    all_series: dict[str, dict[str, dict[float, float]]] = {}
    for label, path in sources:
        all_series[label] = collect_series(label, read_rows(path))
    if reference_label is None:
        reference_label = sources[0][0]
    reference = all_series.get(reference_label)
    if reference is None:
        raise SystemExit(f"Reference source {reference_label!r} was not found.")
    comparison_rows: list[dict[str, Any]] = []
    for label, series_by_field in all_series.items():
        for field, values_by_temp in series_by_field.items():
            ref_by_temp = reference.get(field, {})
            for temp, value in sorted(values_by_temp.items()):
                ref = ref_by_temp.get(temp)
                delta = None if ref is None else value - ref
                rel = None if ref in (None, 0.0) else delta / ref
                comparison_rows.append(
                    {
                        "source": label,
                        "T_K": temp,
                        "field": field,
                        "value": value,
                        "reference_source": reference_label,
                        "reference_value": ref,
                        "delta": delta,
                        "relative_delta": rel,
                    }
                )
    return comparison_rows, all_series


def plot_comparisons(outdir: Path, series: dict[str, dict[str, dict[float, float]]]) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written = []
    for field in PLOT_FIELDS:
        plotted = False
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for label, by_field in series.items():
            values = by_field.get(field, {})
            if not values:
                continue
            temps = sorted(values)
            ax.plot(temps, [values[temp] for temp in temps], marker="o", linewidth=1.8, label=label)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("T (K)")
        ax.set_ylabel(field)
        ax.set_title(f"{field} comparison")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = outdir / f"{field}_comparison.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def compare_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-material-compare",
        description="Compare MOOSE material-property CSVs from Atomi and external sources.",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Material CSV to compare; first source is default reference.",
    )
    parser.add_argument("--reference", help="Reference source label. Defaults to first --source.")
    parser.add_argument("--outdir", type=Path, default=Path("moose_material_comparison"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    sources = [parse_source_arg(item) for item in args.source]
    rows, series = compare_sources(sources, reference_label=args.reference)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table = args.outdir / "material_property_comparison.csv"
    write_rows(table, rows, COMPARE_FIELDS)
    summary = {
        "sources": [{"label": label, "path": str(path)} for label, path in sources],
        "reference": args.reference or sources[0][0],
        "comparison_table": str(table),
    }
    if not args.no_plots:
        summary["plots"] = [str(path) for path in plot_comparisons(args.outdir, series)]
    meta = args.outdir / "material_property_comparison.meta.json"
    meta.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {table}")
    print(f"Wrote {meta}")
    for plot in summary.get("plots", []):
        print(f"Wrote {plot}")


if __name__ == "__main__":
    compare_main()
