"""Config-driven pycalphad workflow helpers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALWAYS_INCLUDE_BY_NAME = {"LIQUID", "GAS", "VAPOR", "FLUID"}
HALIDE_ELEMENTS = {"F", "CL", "BR", "I"}
SCHEMA_CONFIG = "atomi.calphad.workflow.config.v1"
SCHEMA_GRID = "atomi.calphad.workflow.grid.v1"
SCHEMA_REACTION = "atomi.calphad.workflow.reaction_summary.v1"
PERIODIC_SYMBOLS = {
    "H",
    "HE",
    "LI",
    "BE",
    "B",
    "C",
    "N",
    "O",
    "F",
    "NE",
    "NA",
    "MG",
    "AL",
    "SI",
    "P",
    "S",
    "CL",
    "AR",
    "K",
    "CA",
    "SC",
    "TI",
    "V",
    "CR",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "ZN",
    "GA",
    "GE",
    "AS",
    "SE",
    "BR",
    "KR",
    "RB",
    "SR",
    "Y",
    "ZR",
    "NB",
    "MO",
    "TC",
    "RU",
    "RH",
    "PD",
    "AG",
    "CD",
    "IN",
    "SN",
    "SB",
    "TE",
    "I",
    "XE",
    "CS",
    "BA",
    "LA",
    "CE",
    "PR",
    "ND",
    "PM",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
    "HF",
    "TA",
    "W",
    "RE",
    "OS",
    "IR",
    "PT",
    "AU",
    "HG",
    "TL",
    "PB",
    "BI",
    "PO",
    "AT",
    "RN",
    "FR",
    "RA",
    "AC",
    "TH",
    "PA",
    "U",
    "NP",
    "PU",
    "AM",
    "CM",
    "BK",
    "CF",
    "ES",
    "FM",
    "MD",
    "NO",
    "LR",
}


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_float(value: Any) -> float:
    number = finite_float(value)
    return number if number is not None else math.nan


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.12g}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_csv_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(parse_csv_list(item))
        return out
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


def frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("Grid step must be positive.")
    values: list[float] = []
    current = float(start)
    while current <= float(stop) + 1.0e-10:
        values.append(round(current, 12))
        current += step
    return values


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [float(start)]
    return [float(start) + (float(stop) - float(start)) * i / (count - 1) for i in range(count)]


def normalize_species_name(name: str) -> str:
    text = str(name).strip()
    text = re.sub(r"_POS\d+", "", text)
    text = re.sub(r"_NEG\d+", "", text)
    text = re.sub(r"[+-]\d+", "", text)
    return text


def parse_formula_counts(formula: str) -> dict[str, float]:
    text = normalize_species_name(formula)
    if text in {"", "/-", "-", "/"}:
        return {}
    if text.upper() == "VA":
        return {"VA": 1.0}
    counts: Counter[str] = Counter()
    i = 0
    while i < len(text):
        char = text[i]
        if not char.isalpha():
            i += 1
            continue
        if i + 1 < len(text) and text[i + 1].islower():
            symbol = (char + text[i + 1]).upper()
            i += 2
        else:
            two = text[i : i + 2].upper()
            if len(two) == 2 and two in PERIODIC_SYMBOLS:
                symbol = two
                i += 2
            else:
                symbol = char.upper()
                i += 1
        if symbol not in PERIODIC_SYMBOLS:
            continue
        start = i
        while i < len(text) and (text[i].isdigit() or text[i] == "."):
            i += 1
        amount = float(text[start:i] or 1.0)
        counts[symbol] += amount
    return dict(counts)


def species_elements_from_name(name: str) -> set[str]:
    text = normalize_species_name(name)
    if text in {"", "/-", "-", "/"}:
        return set()
    if text == "VA":
        return {"VA"}
    return set(parse_formula_counts(text))


def species_in_binary_subsystem(species_name: str, a: str, b: str) -> bool:
    return species_elements_from_name(species_name).issubset({a, b, "VA"})


def phase_feasible_in_binary_subsystem(phase_obj: Any, a: str, b: str) -> tuple[bool, list[dict[str, Any]], list[list[str]]]:
    return phase_feasible_in_multicomponent_subsystem(phase_obj, {a, b, "VA"})


def phase_feasible_in_multicomponent_subsystem(
    phase_obj: Any, allowed_elements: set[str]
) -> tuple[bool, list[dict[str, Any]], list[list[str]]]:
    allowed_somewhere = False
    bad_sublattices: list[dict[str, Any]] = []
    allowed_by_sublattice: list[list[str]] = []
    for index, sublattice in enumerate(phase_obj.constituents):
        allowed: list[str] = []
        rejected: list[str] = []
        for species in sublattice:
            name = str(species)
            elements = species_elements_from_name(name)
            if elements.issubset(allowed_elements):
                allowed.append(name)
                if name != "VA" and elements:
                    allowed_somewhere = True
            else:
                rejected.append(name)
        allowed_by_sublattice.append(allowed)
        if not allowed:
            bad_sublattices.append({"sublattice": index + 1, "rejected_species": rejected})
    return len(bad_sublattices) == 0 and allowed_somewhere, bad_sublattices, allowed_by_sublattice


def phase_formula_from_debug_entry(name: str, debug_entry: dict[str, Any] | None = None) -> dict[str, float]:
    for token in [re.split(r"[_\s(/]", name, maxsplit=1)[0], *re.split(r"[_\s()/]+", name)]:
        counts = parse_formula_counts(token)
        if counts and set(counts).intersection(HALIDE_ELEMENTS):
            return counts
    debug_formulas: list[dict[str, float]] = []
    if debug_entry:
        for sublattice in debug_entry.get("allowed_by_sublattice") or []:
            for species in sublattice:
                counts = parse_formula_counts(str(species))
                if counts:
                    debug_formulas.append(counts)
    if debug_formulas:
        debug_formulas.sort(key=lambda item: (len(item), sum(item.values())))
        return debug_formulas[0]
    for token in [re.split(r"[_\s(/]", name, maxsplit=1)[0], *re.split(r"[_\s()/]+", name)]:
        counts = parse_formula_counts(token)
        if counts:
            return counts
    return {}


def phase_name_formula_key(name: str) -> tuple[tuple[str, float], ...]:
    token = re.split(r"[_\s(/]", name, maxsplit=1)[0]
    counts = parse_formula_counts(token)
    if counts and set(counts).intersection(HALIDE_ELEMENTS):
        return formula_key(counts)
    return ()


def formula_key(counts: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((elem, round(float(amount), 8)) for elem, amount in counts.items() if elem != "VA" and amount))


def is_salt_formula_join(a: str, b: str) -> tuple[bool, str]:
    a_counts = parse_formula_counts(a)
    b_counts = parse_formula_counts(b)
    common_halides = sorted(set(a_counts).intersection(b_counts).intersection(HALIDE_ELEMENTS))
    return bool(common_halides), common_halides[0] if common_halides else ""


def is_generic_solution_phase(name: str) -> bool:
    upper = name.upper()
    return bool(re.fullmatch(r"SS[A-Z0-9_]*SOLN", upper))


def is_pure_liquid_phase(name: str) -> bool:
    upper = name.upper()
    return upper.endswith("_L1(LIQ)") or upper.endswith("_L1_LIQ") or "L1(LIQ)" in upper


def is_molten_salt_phase(name: str) -> bool:
    upper = name.upper()
    return bool(re.fullmatch(r"MS[A-Z0-9_]*", upper))


def recommend_salt_phase_subset(
    candidate_phases: list[str],
    a: str,
    b: str,
    debug: dict[str, Any] | None = None,
) -> list[str]:
    ok, anion = is_salt_formula_join(a, b)
    if not ok:
        return sorted(candidate_phases)
    endmember_keys = {formula_key(parse_formula_counts(a)), formula_key(parse_formula_counts(b))}
    cations = set(parse_formula_counts(a)).union(parse_formula_counts(b)).difference({anion, "VA"})
    selected: list[str] = []
    formula_by_phase: dict[str, tuple[tuple[str, float], ...]] = {}
    for phase in candidate_phases:
        upper = phase.upper()
        if is_molten_salt_phase(phase):
            selected.append(phase)
            continue
        if upper in {"GAS", "GAS_IDEAL"} or is_pure_liquid_phase(phase) or is_generic_solution_phase(phase):
            continue
        counts = phase_formula_from_debug_entry(phase, (debug or {}).get(phase))
        elements = set(counts).difference({"VA"})
        if not elements or not elements.issubset(cations | {anion}):
            continue
        if formula_key(counts) in endmember_keys:
            formula_by_phase[phase] = formula_key(counts)
            selected.append(phase)
            continue
        phase_cations = elements.difference({anion})
        if anion in elements and len(phase_cations.intersection(cations)) >= 2:
            formula_by_phase[phase] = formula_key(counts)
            selected.append(phase)
    deduped: list[str] = []
    by_formula: dict[tuple[tuple[str, float], ...], list[str]] = defaultdict(list)
    for phase in dict.fromkeys(selected):
        key = formula_by_phase.get(phase)
        if key:
            by_formula[key].append(phase)
        else:
            deduped.append(phase)
    for key, phases in by_formula.items():
        explicit = [phase for phase in phases if phase_name_formula_key(phase) == key]
        deduped.extend(explicit or phases)
    return sorted(dict.fromkeys(deduped))


def recommend_phase_subset(
    candidate_phases: list[str],
    a: str,
    b: str,
    debug: dict[str, Any] | None = None,
) -> list[str]:
    candidates = set(candidate_phases)
    if {a, b} == {"U", "O"}:
        preferred = [
            "ORTHORHOMBIC_A20",
            "TETRAGONAL_U",
            "BCC_A2",
            "FCC_A1",
            "LIQUID",
            "C1_MO2",
            "U4O9_S",
            "U4O9_S2",
            "U4O9_S3",
            "U3O8_S",
            "U3O8_S2",
            "U3O8_S3",
            "U3O8_S4",
            "UO3",
            "GAS",
        ]
        return [phase for phase in preferred if phase in candidates]
    if is_formula_binary(a, b):
        salt_selected = recommend_salt_phase_subset(candidate_phases, a, b, debug)
        if salt_selected:
            return salt_selected
    return sorted(candidate_phases)


def require_pycalphad():
    try:
        from pycalphad import Database, Model, calculate, equilibrium, variables as v
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pycalphad is required for this command. Run inside a CALPHAD environment "
            "or set up one with Atomi's calphad-doctor guidance."
        ) from exc
    return Database, Model, calculate, equilibrium, v


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute() or base is None:
        return p
    return (base / p).resolve()


@dataclass(frozen=True)
class CalphadConfig:
    path: Path | None
    tdb_file: Path
    components: list[str]
    selected_phases: list[str]
    x_axis_component: str
    component_a: str = ""
    component_b: str = ""
    pressure_pa: float = 101325.0
    total_moles: float = 1.0
    phase_labels: dict[str, str] | None = None
    formula_endmembers: dict[str, dict[str, float]] | None = None
    dependent_component: str = ""


def formula_endmember_config(a: str, b: str) -> tuple[list[str], dict[str, dict[str, float]], str]:
    counts = {a: parse_formula_counts(a), b: parse_formula_counts(b)}
    elements = sorted({elem for payload in counts.values() for elem in payload if elem != "VA"})
    common = sorted(set(counts[a]).intersection(counts[b]).difference({"VA"}))
    dependent = common[0] if common else (elements[-1] if elements else "")
    return [*elements, "VA"], counts, dependent


def pseudo_binary_element_fractions(
    x_value: float,
    formula_endmembers: dict[str, dict[str, float]],
    component_a: str,
    component_b: str,
) -> dict[str, float]:
    xa = 1.0 - float(x_value)
    xb = float(x_value)
    totals: Counter[str] = Counter()
    for elem, amount in formula_endmembers.get(component_a, {}).items():
        if elem != "VA":
            totals[elem] += xa * float(amount)
    for elem, amount in formula_endmembers.get(component_b, {}).items():
        if elem != "VA":
            totals[elem] += xb * float(amount)
    total_atoms = sum(totals.values())
    if total_atoms <= 0.0:
        raise ValueError("Pseudo-binary formula endmembers produced zero atoms.")
    return {elem: amount / total_atoms for elem, amount in totals.items()}


def is_formula_binary(a: str, b: str) -> bool:
    elements = set(parse_formula_counts(a)).union(parse_formula_counts(b)).difference({"VA"})
    return len(elements) > 2


def load_config(path: Path) -> CalphadConfig:
    data = read_json(path)
    base = path.parent
    tdb = resolve_path(data["tdb_file"], base)
    formula_endmembers = data.get("formula_endmembers")
    if formula_endmembers is not None:
        formula_endmembers = {
            str(name): {str(elem): float(amount) for elem, amount in payload.items()}
            for name, payload in formula_endmembers.items()
            if isinstance(payload, dict)
        }
    return CalphadConfig(
        path=path,
        tdb_file=tdb,
        components=parse_csv_list(data.get("components")),
        selected_phases=parse_csv_list(data.get("selected_phases") or data.get("phases")),
        x_axis_component=str(data.get("x_axis_component") or data.get("component_B") or ""),
        component_a=str(data.get("component_A") or ""),
        component_b=str(data.get("component_B") or ""),
        pressure_pa=float(data.get("pressure_pa") or 101325.0),
        total_moles=float(data.get("total_moles") or 1.0),
        phase_labels=data.get("phase_labels") if isinstance(data.get("phase_labels"), dict) else {},
        formula_endmembers=formula_endmembers,
        dependent_component=str(data.get("dependent_component") or ""),
    )


def inspect_binary_tdb(tdb_file: Path, a: str, b: str) -> dict[str, Any]:
    Database, *_ = require_pycalphad()
    dbf = Database(str(tdb_file))
    formula_mode = is_formula_binary(a, b)
    components, formula_endmembers, dependent = formula_endmember_config(a, b) if formula_mode else ([a, b, "VA"], {}, "")
    allowed_elements = set(components)
    candidate_phases: list[str] = []
    debug: dict[str, Any] = {}
    for name, phase in dbf.phases.items():
        upper = name.upper()
        if upper in ALWAYS_INCLUDE_BY_NAME or any(key in upper for key in ALWAYS_INCLUDE_BY_NAME):
            candidate_phases.append(name)
            debug[name] = {"status": "included", "reason": "forced inclusion by fluid-like phase name"}
            continue
        if formula_mode:
            ok, bad_sublattices, allowed_by_sublattice = phase_feasible_in_multicomponent_subsystem(phase, allowed_elements)
        else:
            ok, bad_sublattices, allowed_by_sublattice = phase_feasible_in_binary_subsystem(phase, a, b)
        if ok:
            candidate_phases.append(name)
            debug[name] = {
                "status": "included",
                "reason": f"phase feasible in subsystem {'-'.join(sorted(allowed_elements.difference({'VA'})))}",
                "allowed_by_sublattice": allowed_by_sublattice,
            }
        else:
            debug[name] = {
                "status": "excluded",
                "reason": f"no valid {'/'.join(sorted(allowed_elements))} choice on some sublattices",
                "bad_sublattices": bad_sublattices,
                "allowed_by_sublattice": allowed_by_sublattice,
            }
    candidate_phases = sorted(set(candidate_phases))
    config = {
        "schema": SCHEMA_CONFIG,
        "tdb_file": str(tdb_file),
        "component_A": a,
        "component_B": b,
        "components": components,
        "x_axis_component": b,
        "candidate_phases": candidate_phases,
        "selected_phases": recommend_phase_subset(candidate_phases, a, b, debug),
        "debug": debug,
    }
    if formula_mode:
        config["formula_endmembers"] = formula_endmembers
        config["dependent_component"] = dependent
        config["composition_mode"] = "pseudo_binary_formula_join"
        if is_salt_formula_join(a, b)[0]:
            config["phase_filter_note"] = (
                "Pseudo-binary halide salt join: selected phases keep molten-salt liquids and "
                "stoichiometric endmember/mixed salt solids, while excluding pure L1 liquids, "
                "gas phases, elemental phases, generic solution phases, and off-redox one-cation salts."
            )
    return config


def summarize_phases(phases: Any, npvals: Any, tol: float = 1.0e-10) -> list[tuple[str, float]]:
    kept: list[tuple[str, float]] = []
    for phase, amount in zip(list(phases), list(npvals)):
        name = str(phase)
        value = safe_float(amount)
        if name and name != "" and math.isfinite(value) and value > tol:
            kept.append((name, value))
    return kept


def stable_signature_from_kept(kept: list[tuple[str, float]]) -> str:
    return " + ".join(sorted(phase for phase, _ in kept)) if kept else "NONE"


def stable_detail_from_kept(kept: list[tuple[str, float]]) -> str:
    return "; ".join(f"{phase}:{amount:.6g}" for phase, amount in kept) if kept else "NONE"


def _first_finite(values: Any) -> float:
    try:
        import numpy as np

        array = np.asarray(values, dtype=float).ravel()
        finite = array[np.isfinite(array)]
        return float(finite[0]) if finite.size else math.nan
    except Exception:
        return math.nan


def _phase_amounts(eq: Any) -> list[tuple[str, float]]:
    try:
        import numpy as np

        phases = np.asarray(eq.Phase.values).ravel()
        amounts = np.asarray(eq.NP.values, dtype=float).ravel()
        n = min(len(phases), len(amounts))
        return summarize_phases(phases[:n], amounts[:n])
    except Exception:
        return []


def _component_values(eq: Any, variable: str) -> dict[str, float]:
    values = getattr(eq, variable).values
    comps = [str(item) for item in eq.component.values]
    try:
        import numpy as np

        array = np.asarray(values, dtype=float)
        reshaped = array.reshape(-1, len(comps))
        finite_rows = reshaped[np.isfinite(reshaped).any(axis=1)]
        row = finite_rows[0] if len(finite_rows) else reshaped[0]
        return {comp: safe_float(value) for comp, value in zip(comps, row)}
    except Exception:
        return {}


def deduplicate_mqmqa_parameters(dbf: Any, phases: list[str]) -> int:
    """Remove duplicate MQMQA records that make pycalphad model construction ambiguous."""
    table = getattr(dbf, "_parameters", None)
    if table is None or not hasattr(table, "all") or not hasattr(table, "remove"):
        return 0
    phase_set = set(phases)
    seen: set[tuple[str, str, str, Any]] = set()
    remove_ids: list[int] = []
    for row in table.all():
        ptype = str(row.get("parameter_type") or "")
        if row.get("phase_name") not in phase_set or not ptype.startswith("MQM"):
            continue
        key = (
            str(row.get("phase_name")),
            ptype,
            str(row.get("constituent_array")),
            row.get("parameter_order"),
        )
        if key in seen:
            doc_id = getattr(row, "doc_id", row.get("doc_id"))
            if doc_id is not None:
                remove_ids.append(doc_id)
        else:
            seen.add(key)
    if remove_ids:
        table.remove(doc_ids=remove_ids)
    return len(remove_ids)


def deduplicate_mqmqa_z_parameters(dbf: Any, phases: list[str]) -> int:
    """Backward-compatible alias for the broader MQMQA duplicate guard."""
    return deduplicate_mqmqa_parameters(dbf, phases)


def equilibrium_summary(
    *,
    dbf: Any,
    components: list[str],
    phases: list[str],
    conditions: dict[Any, Any],
    output: str = "GM",
    verbose: bool = False,
) -> tuple[dict[str, Any], Any]:
    _, _, _, equilibrium, _ = require_pycalphad()
    deduplicate_mqmqa_parameters(dbf, phases)
    eq = equilibrium(dbf, components, phases, conditions, output=output, verbose=verbose)
    kept = _phase_amounts(eq)
    summary = {
        "stable_signature": stable_signature_from_kept(kept),
        "stable_detail": stable_detail_from_kept(kept),
        "phase_amounts": [{"phase": phase, "amount": amount} for phase, amount in kept],
        "GM_J_mol": _first_finite(eq.GM.values) if hasattr(eq, "GM") else math.nan,
        "X": _component_values(eq, "X") if hasattr(eq, "X") else {},
        "MU": _component_values(eq, "MU") if hasattr(eq, "MU") else {},
    }
    return summary, eq


def tx_scan_rows(
    config: CalphadConfig,
    *,
    t_values: list[float],
    x_values: list[float],
    phases: list[str] | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    Database, _, _, _, v = require_pycalphad()
    dbf = Database(str(config.tdb_file))
    phase_list = phases or config.selected_phases
    rows: list[dict[str, Any]] = []
    for temp in t_values:
        for xval in x_values:
            row: dict[str, Any] = {"T_K": temp, f"X_{config.x_axis_component}": xval}
            conditions = {v.N: config.total_moles, v.T: temp, v.P: config.pressure_pa}
            if config.formula_endmembers:
                fractions = pseudo_binary_element_fractions(
                    xval,
                    config.formula_endmembers,
                    config.component_a,
                    config.component_b,
                )
                dependent = config.dependent_component or next(reversed(sorted(fractions)))
                for elem, fraction in fractions.items():
                    if elem != dependent:
                        conditions[v.X(elem)] = fraction
                    row[f"X_{elem}"] = fraction
            else:
                conditions[v.X(config.x_axis_component)] = xval
            try:
                summary, _ = equilibrium_summary(
                    dbf=dbf,
                    components=config.components,
                    phases=phase_list,
                    conditions=conditions,
                    verbose=verbose,
                )
                row.update(
                    {
                        "stable_signature": summary["stable_signature"],
                        "stable_detail": summary["stable_detail"],
                        "GM_J_mol": summary["GM_J_mol"],
                    }
                )
                for comp, value in summary["MU"].items():
                    row[f"mu_{comp}_J_mol"] = value
            except Exception as exc:
                row.update({"stable_signature": "FAILED", "stable_detail": str(exc), "GM_J_mol": math.nan})
            rows.append(row)
    return rows


def muo_scan_rows(
    config: CalphadConfig,
    *,
    t_values: list[float],
    mu_values: list[float],
    mu_component: str,
    phases: list[str] | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    Database, _, _, _, v = require_pycalphad()
    dbf = Database(str(config.tdb_file))
    phase_list = phases or config.selected_phases
    rows: list[dict[str, Any]] = []
    for temp in t_values:
        for mu in mu_values:
            row: dict[str, Any] = {"T_K": temp, f"MU_{mu_component}_J_mol": mu}
            try:
                summary, _ = equilibrium_summary(
                    dbf=dbf,
                    components=config.components,
                    phases=phase_list,
                    conditions={
                        v.N: config.total_moles,
                        v.T: temp,
                        v.P: config.pressure_pa,
                        v.MU(mu_component): mu,
                    },
                    verbose=verbose,
                )
                row.update(
                    {
                        "stable_signature": summary["stable_signature"],
                        "stable_detail": summary["stable_detail"],
                        "GM_J_mol": summary["GM_J_mol"],
                    }
                )
                for comp, value in summary["X"].items():
                    row[f"X_{comp}"] = value
            except Exception as exc:
                row.update({"stable_signature": "FAILED", "stable_detail": str(exc), "GM_J_mol": math.nan})
            rows.append(row)
    return rows


def infer_x_column(rows: list[dict[str, str]]) -> str:
    for key in ("X_O", "X", "x", "x_value"):
        if rows and key in rows[0]:
            return key
    for key in rows[0] if rows else []:
        if key.startswith("X_"):
            return key
    raise ValueError("Could not infer x/composition column from grid CSV.")


def infer_grid_axes(rows: list[dict[str, str]]) -> tuple[str, list[float], list[float], dict[tuple[float, float], str], dict[tuple[float, float], str]]:
    if not rows:
        raise ValueError("Grid CSV is empty.")
    x_col = infer_x_column(rows)
    t_values = sorted({float(row["T_K"]) for row in rows})
    x_values = sorted({float(row[x_col]) for row in rows})
    sig_map: dict[tuple[float, float], str] = {}
    detail_map: dict[tuple[float, float], str] = {}
    for row in rows:
        key = (float(row["T_K"]), float(row[x_col]))
        sig_map[key] = row.get("stable_signature") or row.get("signature") or "NONE"
        detail_map[key] = row.get("stable_detail") or ""
    return x_col, t_values, x_values, sig_map, detail_map


def build_signature_grid(t_values: list[float], x_values: list[float], sig_map: dict[tuple[float, float], str]) -> list[list[str]]:
    return [[sig_map.get((temp, x), "NONE") for x in x_values] for temp in t_values]


def neighbors4(i: int, j: int, nrows: int, ncols: int):
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < nrows and 0 <= nj < ncols:
            yield ni, nj


def connected_components(grid: list[list[str]]) -> dict[str, list[list[tuple[int, int]]]]:
    nrows = len(grid)
    ncols = len(grid[0]) if nrows else 0
    seen: set[tuple[int, int]] = set()
    components: dict[str, list[list[tuple[int, int]]]] = defaultdict(list)
    for i in range(nrows):
        for j in range(ncols):
            if (i, j) in seen:
                continue
            signature = grid[i][j]
            queue = deque([(i, j)])
            seen.add((i, j))
            coords: list[tuple[int, int]] = []
            while queue:
                ci, cj = queue.popleft()
                coords.append((ci, cj))
                for ni, nj in neighbors4(ci, cj, nrows, ncols):
                    if (ni, nj) not in seen and grid[ni][nj] == signature:
                        seen.add((ni, nj))
                        queue.append((ni, nj))
            components[signature].append(coords)
    return components


def summarize_fields(grid: list[list[str]], t_values: list[float], x_values: list[float]) -> list[dict[str, Any]]:
    comps = connected_components(grid)
    dx = abs(x_values[1] - x_values[0]) if len(x_values) > 1 else 1.0
    dt = abs(t_values[1] - t_values[0]) if len(t_values) > 1 else 1.0
    rows: list[dict[str, Any]] = []
    for signature, groups in comps.items():
        for region_id, coords in enumerate(groups, start=1):
            xs = [x_values[j] for _, j in coords]
            ts = [t_values[i] for i, _ in coords]
            phase_count = 0 if signature == "NONE" else len(signature.split(" + "))
            rows.append(
                {
                    "signature": signature,
                    "region_id": region_id,
                    "n_cells": len(coords),
                    "x_min": min(xs),
                    "x_max": max(xs),
                    "t_min": min(ts),
                    "t_max": max(ts),
                    "x_center": sum(xs) / len(xs),
                    "t_center": sum(ts) / len(ts),
                    "approx_area": len(coords) * dx * dt,
                    "phase_count": phase_count,
                    "is_single_phase": phase_count == 1,
                    "is_two_phase": phase_count == 2,
                }
            )
    return sorted(rows, key=lambda item: (item["phase_count"], item["signature"], item["region_id"]))


def summarize_boundaries(grid: list[list[str]], t_values: list[float], x_values: list[float]) -> list[dict[str, Any]]:
    nrows = len(grid)
    ncols = len(grid[0]) if nrows else 0
    pairs: dict[tuple[str, str], list[tuple[float, float, str]]] = defaultdict(list)
    for i in range(nrows):
        for j in range(ncols):
            here = grid[i][j]
            if j + 1 < ncols and here != grid[i][j + 1]:
                pairs[tuple(sorted((here, grid[i][j + 1])))].append((0.5 * (x_values[j] + x_values[j + 1]), t_values[i], "vertical_in_x"))
            if i + 1 < nrows and here != grid[i + 1][j]:
                pairs[tuple(sorted((here, grid[i + 1][j])))].append((x_values[j], 0.5 * (t_values[i] + t_values[i + 1]), "horizontal_in_T"))
    rows: list[dict[str, Any]] = []
    for (field_1, field_2), pts in pairs.items():
        xs = [item[0] for item in pts]
        ts = [item[1] for item in pts]
        orient = Counter(item[2] for item in pts).most_common(1)[0][0]
        rows.append(
            {
                "field_1": field_1,
                "field_2": field_2,
                "n_boundary_points": len(pts),
                "x_min": min(xs),
                "x_max": max(xs),
                "t_min": min(ts),
                "t_max": max(ts),
                "x_center": sum(xs) / len(xs),
                "t_center": sum(ts) / len(ts),
                "main_orientation": orient,
            }
        )
    return sorted(rows, key=lambda item: (item["t_center"], item["x_center"]))


def boundary_points_from_grid(
    grid: list[list[str]],
    t_values: list[float],
    x_values: list[float],
    *,
    include_none: bool = False,
) -> list[dict[str, Any]]:
    """Extract per-cell phase-boundary points from a categorical T-X grid."""
    nrows = len(grid)
    ncols = len(grid[0]) if nrows else 0
    rows: list[dict[str, Any]] = []
    for i in range(nrows):
        for j in range(ncols):
            here = grid[i][j]
            if not include_none and here == "NONE":
                continue
            if j + 1 < ncols and here != grid[i][j + 1]:
                other = grid[i][j + 1]
                if include_none or other != "NONE":
                    field_1, field_2 = sorted((here, other))
                    rows.append(
                        {
                            "x": 0.5 * (x_values[j] + x_values[j + 1]),
                            "T_K": t_values[i],
                            "field_1": field_1,
                            "field_2": field_2,
                            "orientation": "vertical_in_x",
                        }
                    )
            if i + 1 < nrows and here != grid[i + 1][j]:
                other = grid[i + 1][j]
                if include_none or other != "NONE":
                    field_1, field_2 = sorted((here, other))
                    rows.append(
                        {
                            "x": x_values[j],
                            "T_K": 0.5 * (t_values[i] + t_values[i + 1]),
                            "field_1": field_1,
                            "field_2": field_2,
                            "orientation": "horizontal_in_T",
                        }
                    )
    return sorted(rows, key=lambda item: (item["T_K"], item["x"], item["field_1"], item["field_2"]))


def summarize_candidate_invariants(grid: list[list[str]], t_values: list[float], x_values: list[float]) -> list[dict[str, Any]]:
    nrows = len(grid)
    ncols = len(grid[0]) if nrows else 0
    rows: list[dict[str, Any]] = []
    for i in range(nrows - 1):
        for j in range(ncols - 1):
            block = {grid[i][j], grid[i + 1][j], grid[i][j + 1], grid[i + 1][j + 1]}
            if len(block) >= 3:
                rows.append(
                    {
                        "x": 0.5 * (x_values[j] + x_values[j + 1]),
                        "T_K": 0.5 * (t_values[i] + t_values[i + 1]),
                        "n_neighboring_fields": len(block),
                        "neighboring_fields": " | ".join(sorted(block)),
                        "reaction_guess": " ; ".join(sorted(block)),
                    }
                )
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        unique[(round(row["x"], 10), round(row["T_K"], 8), row["neighboring_fields"])] = row
    return sorted(unique.values(), key=lambda item: (item["T_K"], item["x"]))


def write_reaction_report(path: Path, fields: list[dict[str, Any]], boundaries: list[dict[str, Any]], invariants: list[dict[str, Any]], x_col: str) -> None:
    lines = [
        "Reaction / transition summary from CALPHAD grid",
        "=" * 72,
        "",
        f"Composition axis: {x_col}",
        f"Detected connected phase fields: {len(fields)}",
        f"Detected boundary classes: {len(boundaries)}",
        f"Detected candidate invariant regions: {len(invariants)}",
        "",
        "Candidate invariants:",
    ]
    for row in invariants[:50]:
        lines.append(f"  T={row['T_K']:.6g} {x_col}={row['x']:.6g}: {row['neighboring_fields']}")
    lines.append("")
    lines.append("Boundary classes:")
    for row in boundaries[:80]:
        lines.append(
            f"  {row['field_1']} <-> {row['field_2']}: "
            f"T={row['t_min']:.6g}-{row['t_max']:.6g}, x={row['x_min']:.6g}-{row['x_max']:.6g}, n={row['n_boundary_points']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_grid_map(csv_path: Path, out_png: Path, title: str | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        raise RuntimeError("matplotlib and numpy are required for plot-map.") from exc
    rows = read_csv(csv_path)
    x_col, t_values, x_values, sig_map, _ = infer_grid_axes(rows)
    signatures = sorted({sig_map[(temp, x)] for temp in t_values for x in x_values})
    sig_to_idx = {sig: idx for idx, sig in enumerate(signatures)}
    grid = np.array([[sig_to_idx[sig_map.get((temp, x), "NONE")] for x in x_values] for temp in t_values])
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[min(x_values), max(x_values), min(t_values), max(t_values)],
        interpolation="nearest",
    )
    cbar = fig.colorbar(image, ax=ax, ticks=list(sig_to_idx.values()))
    cbar.ax.set_yticklabels(signatures)
    ax.set_xlabel(x_col)
    ax.set_ylabel("T (K)")
    ax.set_title(title or "CALPHAD phase map")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _phase_label(signature: str) -> str:
    if signature == "NONE":
        return ""
    labels = []
    for phase in signature.split(" + "):
        label = phase
        label = label.replace("_L1(LIQ)", "(liq)")
        label = label.replace("_S1(S)", "")
        label = label.replace("_NO.", " No.")
        label = label.replace("_", " ")
        labels.append(label)
    return " + ".join(labels)


def _composition_axis_label(x_col: str) -> str:
    if x_col.startswith("X_") and len(x_col) > 2:
        return f"x({x_col[2:]})"
    return x_col


def _axis_edges(values: list[float]) -> list[float]:
    if len(values) == 1:
        return [values[0] - 0.5, values[0] + 0.5]
    edges = [values[0] - 0.5 * (values[1] - values[0])]
    edges.extend(0.5 * (values[idx] + values[idx + 1]) for idx in range(len(values) - 1))
    edges.append(values[-1] + 0.5 * (values[-1] - values[-2]))
    return edges


def _boundary_segments(
    signature_grid: list[list[str]],
    x_values: list[float],
    t_values: list[float],
    *,
    include_none: bool = False,
) -> list[list[tuple[float, float]]]:
    nrows = len(signature_grid)
    ncols = len(signature_grid[0]) if nrows else 0
    x_edges = _axis_edges(x_values)
    t_edges = _axis_edges(t_values)
    segments: list[list[tuple[float, float]]] = []
    for i in range(nrows):
        for j in range(ncols):
            here = signature_grid[i][j]
            if not include_none and here == "NONE":
                continue
            if j + 1 < ncols and here != signature_grid[i][j + 1]:
                other = signature_grid[i][j + 1]
                if include_none or other != "NONE":
                    x_mid = 0.5 * (x_values[j] + x_values[j + 1])
                    segments.append([(x_mid, t_edges[i]), (x_mid, t_edges[i + 1])])
            if i + 1 < nrows and here != signature_grid[i + 1][j]:
                other = signature_grid[i + 1][j]
                if include_none or other != "NONE":
                    t_mid = 0.5 * (t_values[i] + t_values[i + 1])
                    segments.append([(x_edges[j], t_mid), (x_edges[j + 1], t_mid)])
    return segments


def _plot_grid_boundaries(
    ax: Any,
    signature_grid: list[list[str]],
    x_values: list[float],
    t_values: list[float],
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float = 1.0,
    label: str | None = None,
) -> None:
    from matplotlib.collections import LineCollection

    segments = _boundary_segments(signature_grid, x_values, t_values)
    if not segments:
        return
    collection = LineCollection(
        segments,
        colors=color,
        linestyles=linestyle,
        linewidths=linewidth,
        alpha=alpha,
        capstyle="butt",
        joinstyle="miter",
    )
    ax.add_collection(collection)
    if label:
        ax.plot([], [], color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)


def _typical_step(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    diffs = [abs(values[idx + 1] - values[idx]) for idx in range(len(values) - 1)]
    diffs = [item for item in diffs if item > 0]
    return min(diffs) if diffs else 1.0


def _centered_moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 3:
        return list(values)
    window = max(1, min(window, len(values)))
    half = window // 2
    smoothed: list[float] = []
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def _smoothed_boundary_paths(
    boundary_points: list[dict[str, Any]],
    x_values: list[float],
    t_values: list[float],
    *,
    window: int,
    mode: str = "average",
    max_gap_steps: float = 4.0,
) -> list[dict[str, Any]]:
    if window <= 1:
        return []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in boundary_points:
        grouped[(row["field_1"], row["field_2"], row["orientation"])].append(row)

    x_step = _typical_step(x_values)
    t_step = _typical_step(t_values)
    paths: list[dict[str, Any]] = []
    for (field_1, field_2, orientation), rows in grouped.items():
        independent_key = "T_K" if orientation == "vertical_in_x" else "x"
        dependent_key = "x" if orientation == "vertical_in_x" else "T_K"
        step = t_step if independent_key == "T_K" else x_step
        max_gap = 1.75 * step
        bridge_gap = max(max_gap, max_gap_steps * step)
        collapsed: dict[float, list[float]] = defaultdict(list)
        for row in rows:
            collapsed[float(row[independent_key])].append(float(row[dependent_key]))
        points = [(independent, sum(dependent) / len(dependent)) for independent, dependent in collapsed.items()]
        points.sort(key=lambda item: item[0])
        if len(points) < 3:
            continue
        if mode == "bridge-gaps":
            for left, right in zip(points[:-1], points[1:]):
                gap = abs(right[0] - left[0])
                if max_gap < gap <= bridge_gap:
                    if orientation == "vertical_in_x":
                        xy = [(left[1], left[0]), (right[1], right[0])]
                    else:
                        xy = [(left[0], left[1]), (right[0], right[1])]
                    paths.append(
                        {
                            "field_1": field_1,
                            "field_2": field_2,
                            "orientation": orientation,
                            "points": xy,
                        }
                    )
            continue
        if mode != "average":
            raise ValueError(f"Unsupported smooth mode: {mode}")
        chunks: list[list[tuple[float, float]]] = [[]]
        for point in points:
            if chunks[-1] and abs(point[0] - chunks[-1][-1][0]) > max_gap:
                chunks.append([])
            chunks[-1].append(point)
        for chunk in chunks:
            if len(chunk) < 3:
                continue
            independent = [point[0] for point in chunk]
            dependent = _centered_moving_average([point[1] for point in chunk], window)
            if orientation == "vertical_in_x":
                xy = list(zip(dependent, independent))
            else:
                xy = list(zip(independent, dependent))
            paths.append(
                {
                    "field_1": field_1,
                    "field_2": field_2,
                    "orientation": orientation,
                    "points": xy,
                }
            )
    return paths


def _plot_smoothed_boundaries(
    ax: Any,
    boundary_points: list[dict[str, Any]],
    x_values: list[float],
    t_values: list[float],
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    window: int,
    mode: str = "average",
    max_gap_steps: float = 4.0,
    alpha: float = 1.0,
    label: str | None = None,
) -> int:
    paths = _smoothed_boundary_paths(
        boundary_points,
        x_values,
        t_values,
        window=window,
        mode=mode,
        max_gap_steps=max_gap_steps,
    )
    for path in paths:
        points = path["points"]
        xs = [point[0] for point in points]
        ts = [point[1] for point in points]
        ax.plot(xs, ts, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    if paths and label:
        ax.plot([], [], color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)
    return len(paths)


def plot_phase_diagram_lines(
    csv_path: Path,
    out_png: Path,
    *,
    title: str | None = None,
    boundary_csv: Path | None = None,
    label_fields: bool = True,
    overlay_csvs: list[Path] | None = None,
    overlay_labels: list[str] | None = None,
    smooth_boundaries: bool = False,
    smooth_window: int = 5,
    smooth_mode: str = "average",
    smooth_max_gap_steps: float = 4.0,
    hide_raw_boundaries: bool = False,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        raise RuntimeError("matplotlib is required for plot-diagram.") from exc

    rows = read_csv(csv_path)
    x_col, t_values, x_values, sig_map, _ = infer_grid_axes(rows)
    signatures = sorted({sig_map[(temp, x)] for temp in t_values for x in x_values})
    signature_grid = build_signature_grid(t_values, x_values, sig_map)
    boundaries = boundary_points_from_grid(signature_grid, t_values, x_values)

    if boundary_csv is not None:
        fields = ["x", "T_K", "field_1", "field_2", "orientation"]
        write_csv(boundary_csv, boundaries, fields)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    n_smooth_paths = 0
    if smooth_boundaries and smooth_mode == "bridge-gaps":
        raw_color = "black"
        raw_linestyle = "-"
        raw_linewidth = 1.05
        raw_alpha = 0.92
    else:
        raw_color = "0.38" if smooth_boundaries else "black"
        raw_linestyle = "--" if smooth_boundaries else "-"
        raw_linewidth = 0.55 if smooth_boundaries else 1.15
        raw_alpha = 0.72 if smooth_boundaries else 1.0
    if not hide_raw_boundaries:
        _plot_grid_boundaries(
            ax,
            signature_grid,
            x_values,
            t_values,
            color=raw_color,
            linestyle=raw_linestyle,
            linewidth=raw_linewidth,
            alpha=raw_alpha,
            label="CALPHAD/MQMQA raw" if overlay_csvs or smooth_boundaries else None,
        )
    if smooth_boundaries:
        n_smooth_paths += _plot_smoothed_boundaries(
            ax,
            boundaries,
            x_values,
            t_values,
            color="black",
            linestyle="-",
            linewidth=1.4 if smooth_mode == "bridge-gaps" else 1.9,
            window=smooth_window,
            mode=smooth_mode,
            max_gap_steps=smooth_max_gap_steps,
            alpha=0.82 if smooth_mode == "bridge-gaps" else 1.0,
            label=(
                "CALPHAD/MQMQA gap bridges"
                if smooth_mode == "bridge-gaps" and (overlay_csvs or not hide_raw_boundaries)
                else "CALPHAD/MQMQA smoothed"
                if overlay_csvs or not hide_raw_boundaries
                else None
            ),
        )

    if overlay_csvs:
        overlay_labels = overlay_labels or []
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
        for idx, overlay_csv in enumerate(overlay_csvs):
            overlay_rows = read_csv(overlay_csv.resolve())
            _, overlay_t, overlay_x, overlay_sig_map, _ = infer_grid_axes(overlay_rows)
            overlay_grid = build_signature_grid(overlay_t, overlay_x, overlay_sig_map)
            label = overlay_labels[idx] if idx < len(overlay_labels) else overlay_csv.stem
            overlay_boundaries = boundary_points_from_grid(overlay_grid, overlay_t, overlay_x)
            if not hide_raw_boundaries:
                _plot_grid_boundaries(
                    ax,
                    overlay_grid,
                    overlay_x,
                    overlay_t,
                    color=colors[idx % len(colors)],
                    linestyle=":",
                    linewidth=0.9 if smooth_boundaries and smooth_mode == "bridge-gaps" else 0.6 if smooth_boundaries else 1.2,
                    alpha=0.72 if smooth_boundaries else 1.0,
                    label=f"{label} raw" if smooth_boundaries else label,
                )
            if smooth_boundaries:
                n_smooth_paths += _plot_smoothed_boundaries(
                    ax,
                    overlay_boundaries,
                    overlay_x,
                    overlay_t,
                    color=colors[idx % len(colors)],
                    linestyle="--",
                    linewidth=1.4 if smooth_mode == "bridge-gaps" else 1.9,
                    window=smooth_window,
                    mode=smooth_mode,
                    max_gap_steps=smooth_max_gap_steps,
                    alpha=0.82 if smooth_mode == "bridge-gaps" else 1.0,
                    label=(
                        f"{label} gap bridges"
                        if smooth_mode == "bridge-gaps" and not hide_raw_boundaries
                        else f"{label} smoothed"
                        if not hide_raw_boundaries
                        else label
                    ),
                )

    if label_fields:
        fields = summarize_fields(signature_grid, t_values, x_values)
        for field in fields:
            if field["signature"] == "NONE" or field["n_cells"] < 2:
                continue
            ax.text(
                field["x_center"],
                field["t_center"],
                _phase_label(field["signature"]),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )

    ax.set_xlim(min(x_values), max(x_values))
    ax.set_ylim(min(t_values), max(t_values))
    ax.set_xlabel(_composition_axis_label(x_col))
    ax.set_ylabel("T (K)")
    ax.set_title(title or "CALPHAD T-X phase diagram")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True, color="0.86", linewidth=0.5)
    if overlay_csvs:
        ax.legend(frameon=False, loc="best")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return {
        "plot_png": str(out_png),
        "boundary_csv": str(boundary_csv) if boundary_csv else None,
        "n_boundary_points": len(boundaries),
        "n_phase_fields": len(signatures),
        "n_smoothed_boundary_paths": n_smooth_paths,
        "smooth_window": smooth_window if smooth_boundaries else None,
        "smooth_mode": smooth_mode if smooth_boundaries else None,
        "smooth_max_gap_steps": smooth_max_gap_steps if smooth_boundaries else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calphad-workflow",
        description="Config-driven pycalphad inspection, equilibrium, scan, and phase-map helpers.",
    )
    sub = parser.add_subparsers(dest="command")

    init = sub.add_parser("init", help="Write a portable CALPHAD workflow folder skeleton.")
    init.add_argument("--outdir", type=Path, default=Path("calphad_workflow"))
    init.add_argument("--tdb", default="TDB/UPUOC.TDB")
    init.add_argument("--component-a", default="U")
    init.add_argument("--component-b", default="O")

    inspect = sub.add_parser("inspect", help="Inspect a TDB and write a binary subsystem config.")
    inspect.add_argument("--tdb", type=Path, required=True)
    inspect.add_argument("--A", "--component-a", dest="component_a", required=True)
    inspect.add_argument("--B", "--component-b", dest="component_b", required=True)
    inspect.add_argument("--xcomp", help="X-axis component. Defaults to --B.")
    inspect.add_argument("--out", type=Path)

    eq = sub.add_parser("eq", help="Run one equilibrium point and write JSON summary.")
    eq.add_argument("--config", type=Path, required=True)
    eq.add_argument("--temperature", type=float, required=True)
    eq.add_argument("--x", type=float, help="Mole fraction of config x-axis component.")
    eq.add_argument("--mu", type=float, help="Chemical potential in J/mol for --mu-component.")
    eq.add_argument("--mu-component", default=None)
    eq.add_argument("--phase", action="append", help="Override selected phases. Repeatable or comma-separated.")
    eq.add_argument("--out", type=Path, default=Path("equilibrium/eq_single.json"))
    eq.add_argument("--verbose-solver", action="store_true")

    scan_tx = sub.add_parser("scan-tx", help="Scan a T-X phase grid and write CSV.")
    scan_tx.add_argument("--config", type=Path, required=True)
    scan_tx.add_argument("--outdir", type=Path, default=Path("phase_diagram"))
    scan_tx.add_argument("--xmin", type=float, default=0.0)
    scan_tx.add_argument("--xmax", type=float, default=1.0)
    scan_tx.add_argument("--nx", type=int, default=51)
    scan_tx.add_argument("--tmin", type=float, required=True)
    scan_tx.add_argument("--tmax", type=float, required=True)
    scan_tx.add_argument("--nt", type=int, default=51)
    scan_tx.add_argument("--plot", action="store_true")
    scan_tx.add_argument(
        "--plot-style",
        choices=("map", "diagram", "both"),
        default="both",
        help="Plot diagnostic color map, paper-style boundary diagram, or both when --plot is set.",
    )
    scan_tx.add_argument("--smooth-boundaries", action="store_true", help="Overlay smoothed guide curves on paper-style phase diagrams.")
    scan_tx.add_argument("--smooth-window", type=int, default=5, help="Centered moving-average window for smoothed boundary guide curves.")
    scan_tx.add_argument(
        "--smooth-mode",
        choices=("average", "bridge-gaps"),
        default="average",
        help="Boundary guide mode: moving-average whole paths, or only bridge short gaps while preserving raw dense-grid features.",
    )
    scan_tx.add_argument(
        "--smooth-max-gap-steps",
        type=float,
        default=4.0,
        help="For --smooth-mode bridge-gaps, bridge missing boundary spans up to this many grid steps.",
    )
    scan_tx.add_argument("--hide-raw-boundaries", action="store_true", help="Hide raw grid-step boundaries when --smooth-boundaries is enabled.")

    scan_mu = sub.add_parser("scan-muo", help="Scan a T-mu(component) phase grid and write CSV.")
    scan_mu.add_argument("--config", type=Path, required=True)
    scan_mu.add_argument("--outdir", type=Path, default=Path("muO_maps"))
    scan_mu.add_argument("--mu-component", default=None)
    scan_mu.add_argument("--mu-min", type=float, required=True)
    scan_mu.add_argument("--mu-max", type=float, required=True)
    scan_mu.add_argument("--nmu", type=int, default=51)
    scan_mu.add_argument("--tmin", type=float, required=True)
    scan_mu.add_argument("--tmax", type=float, required=True)
    scan_mu.add_argument("--nt", type=int, default=51)
    scan_mu.add_argument("--plot", action="store_true")

    summary = sub.add_parser("reaction-summary", help="Summarize phase fields, boundaries, and invariant candidates from a grid CSV.")
    summary.add_argument("--grid-csv", type=Path, required=True)
    summary.add_argument("--outdir", type=Path, default=Path("reaction_summary"))

    plot = sub.add_parser("plot-map", help="Plot a phase map from a scan grid CSV.")
    plot.add_argument("--grid-csv", type=Path, required=True)
    plot.add_argument("--out", type=Path, default=Path("phase_map.png"))
    plot.add_argument("--title")
    plot.add_argument(
        "--style",
        choices=("map", "diagram", "both"),
        default="map",
        help="Plot diagnostic color map, paper-style boundary diagram, or both.",
    )
    plot.add_argument("--boundary-csv", type=Path, help="Optional CSV path for extracted boundary points in diagram mode.")
    plot.add_argument("--overlay-grid-csv", type=Path, action="append", default=[], help="Additional grid CSV to overlay as dashed boundaries.")
    plot.add_argument("--overlay-label", action="append", default=[], help="Legend label for each --overlay-grid-csv.")
    plot.add_argument("--no-label-fields", action="store_true", help="Suppress field labels in diagram mode.")
    plot.add_argument("--smooth-boundaries", action="store_true", help="Overlay smoothed guide curves in diagram mode.")
    plot.add_argument("--smooth-window", type=int, default=5, help="Centered moving-average window for smoothed boundary guide curves.")
    plot.add_argument(
        "--smooth-mode",
        choices=("average", "bridge-gaps"),
        default="average",
        help="Boundary guide mode: moving-average whole paths, or only bridge short gaps while preserving raw dense-grid features.",
    )
    plot.add_argument(
        "--smooth-max-gap-steps",
        type=float,
        default=4.0,
        help="For --smooth-mode bridge-gaps, bridge missing boundary spans up to this many grid steps.",
    )
    plot.add_argument("--hide-raw-boundaries", action="store_true", help="Hide raw grid-step boundaries when --smooth-boundaries is enabled.")

    diagram = sub.add_parser("plot-diagram", help="Plot a paper-style T-X phase-boundary diagram from a scan grid CSV.")
    diagram.add_argument("--grid-csv", type=Path, required=True)
    diagram.add_argument("--out", type=Path, default=Path("phase_diagram.png"))
    diagram.add_argument("--boundary-csv", type=Path, default=Path("phase_boundaries_points.csv"))
    diagram.add_argument("--title")
    diagram.add_argument("--overlay-grid-csv", type=Path, action="append", default=[], help="Additional grid CSV to overlay as dashed boundaries.")
    diagram.add_argument("--overlay-label", action="append", default=[], help="Legend label for each --overlay-grid-csv.")
    diagram.add_argument("--no-label-fields", action="store_true", help="Suppress field labels.")
    diagram.add_argument("--smooth-boundaries", action="store_true", help="Overlay smoothed guide curves.")
    diagram.add_argument("--smooth-window", type=int, default=5, help="Centered moving-average window for smoothed boundary guide curves.")
    diagram.add_argument(
        "--smooth-mode",
        choices=("average", "bridge-gaps"),
        default="average",
        help="Boundary guide mode: moving-average whole paths, or only bridge short gaps while preserving raw dense-grid features.",
    )
    diagram.add_argument(
        "--smooth-max-gap-steps",
        type=float,
        default=4.0,
        help="For --smooth-mode bridge-gaps, bridge missing boundary spans up to this many grid steps.",
    )
    diagram.add_argument("--hide-raw-boundaries", action="store_true", help="Hide raw grid-step boundaries when --smooth-boundaries is enabled.")
    return parser


def init_workflow(args: argparse.Namespace) -> dict[str, Any]:
    root = args.outdir.resolve()
    folders = ["TDB", "config", "inspect", "equilibrium", "phase_diagram", "muO_maps", "phase_probe", "phase_analysis", "reaction_summary"]
    for folder in folders:
        (root / folder).mkdir(parents=True, exist_ok=True)
    if is_formula_binary(args.component_a, args.component_b):
        components, formula_endmembers, dependent = formula_endmember_config(args.component_a, args.component_b)
    else:
        components, formula_endmembers, dependent = [args.component_a, args.component_b, "VA"], {}, ""
    config = {
        "schema": SCHEMA_CONFIG,
        "tdb_file": args.tdb,
        "component_A": args.component_a,
        "component_B": args.component_b,
        "components": components,
        "x_axis_component": args.component_b,
        "selected_phases": [],
        "notes": "Run calphad-workflow inspect to populate selected phases from the TDB.",
    }
    if formula_endmembers:
        config["formula_endmembers"] = formula_endmembers
        config["dependent_component"] = dependent
        config["composition_mode"] = "pseudo_binary_formula_join"
    config_path = root / "config" / f"{args.component_a}_{args.component_b}_phase_config.json"
    write_json(config_path, config)
    readme = root / "README_calphad_workflow.md"
    readme.write_text(
        textwrap.dedent(
            f"""\
            # Atomi CALPHAD workflow

            1. Put/copy your TDB at `{args.tdb}` or update `config/*_phase_config.json`.
            2. Inspect phases:
               `calphad-workflow inspect --tdb {args.tdb} --A {args.component_a} --B {args.component_b} --out config/{args.component_a}_{args.component_b}_phase_config.json`
            3. Run one point:
               `calphad-workflow eq --config config/{args.component_a}_{args.component_b}_phase_config.json --temperature 1500 --x 0.6666667`
            4. Scan T-X:
               `calphad-workflow scan-tx --config config/{args.component_a}_{args.component_b}_phase_config.json --tmin 300 --tmax 4500 --nt 61 --nx 81 --plot`
            5. Summarize reactions:
               `calphad-workflow reaction-summary --grid-csv phase_diagram/T_X_phase_grid.csv`
            """
        ),
        encoding="utf-8",
    )
    print(f"Wrote CALPHAD workflow skeleton: {root}")
    return {"root": str(root), "config": str(config_path), "readme": str(readme)}


def inspect_main(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out or Path(f"{args.component_a}_{args.component_b}_phase_config.json")
    config = inspect_binary_tdb(args.tdb.resolve(), args.component_a, args.component_b)
    if args.xcomp:
        config["x_axis_component"] = args.xcomp
    write_json(out, config)
    print(f"Binary subsystem: {args.component_a}-{args.component_b}")
    print(f"Candidate phases: {len(config['candidate_phases'])}")
    print(f"Selected phases : {', '.join(config['selected_phases'])}")
    print(f"Wrote config: {out}")
    return config


def eq_main(args: argparse.Namespace) -> dict[str, Any]:
    Database, _, _, _, v = require_pycalphad()
    config = load_config(args.config.resolve())
    dbf = Database(str(config.tdb_file))
    phases = parse_csv_list(args.phase) or config.selected_phases
    if args.x is None and args.mu is None:
        raise ValueError("eq requires either --x or --mu.")
    conditions: dict[Any, Any] = {v.N: config.total_moles, v.T: args.temperature, v.P: config.pressure_pa}
    if args.mu is not None:
        mu_component = args.mu_component or config.x_axis_component
        conditions[v.MU(mu_component)] = args.mu
    else:
        if config.formula_endmembers:
            fractions = pseudo_binary_element_fractions(args.x, config.formula_endmembers, config.component_a, config.component_b)
            dependent = config.dependent_component or next(reversed(sorted(fractions)))
            for elem, fraction in fractions.items():
                if elem != dependent:
                    conditions[v.X(elem)] = fraction
        else:
            conditions[v.X(config.x_axis_component)] = args.x
    summary, _ = equilibrium_summary(
        dbf=dbf,
        components=config.components,
        phases=phases,
        conditions=conditions,
        verbose=args.verbose_solver,
    )
    payload = {
        "schema": "atomi.calphad.workflow.eq.v1",
        "config": str(args.config.resolve()),
        "conditions": {str(key): format_value(value) for key, value in conditions.items()},
        "components": config.components,
        "phases": phases,
        "summary": summary,
    }
    write_json(args.out, payload)
    print(f"Stable phases: {summary['stable_signature']}")
    print(f"GM: {format_value(summary['GM_J_mol'])} J/mol")
    print(f"Wrote equilibrium summary: {args.out}")
    return payload


def scan_tx_main(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config.resolve())
    t_values = linspace(args.tmin, args.tmax, args.nt)
    x_values = linspace(args.xmin, args.xmax, args.nx)
    rows = tx_scan_rows(config, t_values=t_values, x_values=x_values)
    outdir = args.outdir.resolve()
    table = outdir / "T_X_phase_grid.csv"
    fields = ["T_K", f"X_{config.x_axis_component}", "stable_signature", "stable_detail", "GM_J_mol"]
    if config.formula_endmembers:
        for comp in config.components:
            if comp != "VA":
                fields.append(f"X_{comp}")
    for comp in config.components:
        fields.append(f"mu_{comp}_J_mol")
    write_csv(table, rows, fields)
    metadata = {
        "schema": SCHEMA_GRID,
        "mode": "T-X",
        "config": str(args.config.resolve()),
        "n_rows": len(rows),
        "outputs": {"grid_csv": str(table)},
    }
    if config.formula_endmembers:
        metadata["composition_mode"] = "pseudo_binary_formula_join"
        metadata["formula_endmembers"] = config.formula_endmembers
        metadata["dependent_component"] = config.dependent_component
    if args.plot and args.plot_style in {"map", "both"}:
        png = outdir / "T_X_phase_map.png"
        plot_grid_map(table, png, title="CALPHAD T-X phase map")
        metadata["outputs"]["plot_png"] = str(png)
    if args.plot and args.plot_style in {"diagram", "both"}:
        diagram_png = outdir / "T_X_phase_diagram.png"
        boundary_csv = outdir / "T_X_phase_boundary_points.csv"
        diagram_meta = plot_phase_diagram_lines(
            table,
            diagram_png,
            title="CALPHAD T-X phase diagram",
            boundary_csv=boundary_csv,
            smooth_boundaries=args.smooth_boundaries,
            smooth_window=args.smooth_window,
            smooth_mode=args.smooth_mode,
            smooth_max_gap_steps=args.smooth_max_gap_steps,
            hide_raw_boundaries=args.hide_raw_boundaries,
        )
        metadata["outputs"]["phase_diagram_png"] = str(diagram_png)
        metadata["outputs"]["phase_boundary_points"] = str(boundary_csv)
        metadata["n_boundary_points"] = diagram_meta["n_boundary_points"]
    write_json(outdir / "T_X_phase_grid_metadata.json", metadata)
    print(f"Wrote T-X phase grid: {table}")
    return metadata


def scan_muo_main(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config.resolve())
    mu_component = args.mu_component or config.x_axis_component
    t_values = linspace(args.tmin, args.tmax, args.nt)
    mu_values = linspace(args.mu_min, args.mu_max, args.nmu)
    rows = muo_scan_rows(config, t_values=t_values, mu_values=mu_values, mu_component=mu_component)
    outdir = args.outdir.resolve()
    table = outdir / f"T_mu{mu_component}_phase_grid.csv"
    fields = ["T_K", f"MU_{mu_component}_J_mol", "stable_signature", "stable_detail", "GM_J_mol"]
    for comp in config.components:
        fields.append(f"X_{comp}")
    write_csv(table, rows, fields)
    metadata = {
        "schema": SCHEMA_GRID,
        "mode": f"T-MU({mu_component})",
        "config": str(args.config.resolve()),
        "n_rows": len(rows),
        "outputs": {"grid_csv": str(table)},
    }
    if args.plot:
        png = outdir / f"T_mu{mu_component}_phase_map.png"
        plot_grid_map(table, png, title=f"CALPHAD T-mu({mu_component}) phase map")
        metadata["outputs"]["plot_png"] = str(png)
    write_json(outdir / f"T_mu{mu_component}_phase_grid_metadata.json", metadata)
    print(f"Wrote T-mu({mu_component}) phase grid: {table}")
    return metadata


def reaction_summary_main(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_csv(args.grid_csv.resolve())
    x_col, t_values, x_values, sig_map, _ = infer_grid_axes(rows)
    grid = build_signature_grid(t_values, x_values, sig_map)
    fields = summarize_fields(grid, t_values, x_values)
    boundaries = summarize_boundaries(grid, t_values, x_values)
    invariants = summarize_candidate_invariants(grid, t_values, x_values)
    outdir = args.outdir.resolve()
    fields_csv = outdir / "phase_fields_summary.csv"
    boundaries_csv = outdir / "phase_boundaries_summary.csv"
    invariants_csv = outdir / "candidate_invariants.csv"
    write_csv(fields_csv, fields, list(fields[0]) if fields else ["signature"])
    write_csv(boundaries_csv, boundaries, list(boundaries[0]) if boundaries else ["field_1", "field_2"])
    write_csv(invariants_csv, invariants, list(invariants[0]) if invariants else ["x", "T_K"])
    report = outdir / "reaction_detection_report.txt"
    write_reaction_report(report, fields, boundaries, invariants, x_col)
    metadata = {
        "schema": SCHEMA_REACTION,
        "grid_csv": str(args.grid_csv.resolve()),
        "x_column": x_col,
        "n_fields": len(fields),
        "n_boundaries": len(boundaries),
        "n_candidate_invariants": len(invariants),
        "outputs": {
            "phase_fields_summary": str(fields_csv),
            "phase_boundaries_summary": str(boundaries_csv),
            "candidate_invariants": str(invariants_csv),
            "report": str(report),
        },
    }
    write_json(outdir / "reaction_summary_metadata.json", metadata)
    print(f"Wrote reaction summary: {report}")
    return metadata


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        return init_workflow(args)
    if args.command == "inspect":
        return inspect_main(args)
    if args.command == "eq":
        return eq_main(args)
    if args.command == "scan-tx":
        return scan_tx_main(args)
    if args.command == "scan-muo":
        return scan_muo_main(args)
    if args.command == "reaction-summary":
        return reaction_summary_main(args)
    if args.command == "plot-map":
        outputs: dict[str, Any] = {}
        if args.style in {"map", "both"}:
            plot_grid_map(args.grid_csv.resolve(), args.out.resolve(), title=args.title)
            outputs["plot_png"] = str(args.out.resolve())
            print(f"Wrote phase map: {args.out.resolve()}")
        if args.style in {"diagram", "both"}:
            diagram_out = args.out.resolve()
            if args.style == "both":
                diagram_out = args.out.resolve().with_name(f"{args.out.resolve().stem}_diagram{args.out.resolve().suffix}")
            boundary_csv = args.boundary_csv.resolve() if args.boundary_csv else diagram_out.with_name(f"{diagram_out.stem}_boundary_points.csv")
            diagram_meta = plot_phase_diagram_lines(
                args.grid_csv.resolve(),
                diagram_out,
                title=args.title,
                boundary_csv=boundary_csv,
                label_fields=not args.no_label_fields,
                overlay_csvs=[item.resolve() for item in args.overlay_grid_csv],
                overlay_labels=args.overlay_label,
                smooth_boundaries=args.smooth_boundaries,
                smooth_window=args.smooth_window,
                smooth_mode=args.smooth_mode,
                smooth_max_gap_steps=args.smooth_max_gap_steps,
                hide_raw_boundaries=args.hide_raw_boundaries,
            )
            outputs.update(diagram_meta)
            print(f"Wrote phase diagram: {diagram_out}")
        return outputs
    if args.command == "plot-diagram":
        metadata = plot_phase_diagram_lines(
            args.grid_csv.resolve(),
            args.out.resolve(),
            title=args.title,
            boundary_csv=args.boundary_csv.resolve(),
            label_fields=not args.no_label_fields,
            overlay_csvs=[item.resolve() for item in args.overlay_grid_csv],
            overlay_labels=args.overlay_label,
            smooth_boundaries=args.smooth_boundaries,
            smooth_window=args.smooth_window,
            smooth_mode=args.smooth_mode,
            smooth_max_gap_steps=args.smooth_max_gap_steps,
            hide_raw_boundaries=args.hide_raw_boundaries,
        )
        print(f"Wrote phase diagram: {args.out.resolve()}")
        return metadata
    build_parser().print_help()
    return None


if __name__ == "__main__":
    main()
