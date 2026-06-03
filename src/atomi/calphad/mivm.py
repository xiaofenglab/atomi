"""Molecular Interaction Volume Model helpers and pycalphad bridge."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


R_J_MOLK = 8.31446261815324
SCHEMA = "atomi.calphad.mivm.parameters.v1"
SAMPLE_SCHEMA = "atomi.calphad.mivm.sample.v1"
DATABASE_SCHEMA = "atomi.copilot.mivm.parameter_database.v0.1"
BENCHMARK_UQ_PHASE_SCHEMA = "atomi.calphad.mivm.benchmark_uq_phase.v1"


MIVM_HELP_EPILOG = """\
Parameter guide:
  General MIVM/pycalphad needs:
    - phase/component basis and reference states
    - endmember or pseudo-endmember Gibbs energies
    - molar volumes Vm_i on the same component basis
    - coordination numbers Z_i or effective coordination numbers
    - directed pair parameters B_ij/B_ji, or epsilon/h parameters that define them
    - mixing enthalpy or activity data for fitting/validation

  Molten salts:
    - salt-component basis, e.g. LaCl3 and eutectic LiCl-KCl
    - liquid molar volumes from density/TMA/MD
    - cation-cation or physically chosen solvation coordination numbers from RDF/CN
    - pair potentials or PMF-derived B_ij, then calorimetry-refined B_ij if needed
    - charge/stoichiometry normalization for formula units and common-anion mixtures

  Solid/ceramic solutions, e.g. (Gd,U)O2:
    - substitutional/pseudo-component basis, e.g. U4+O2, Gd3+O1.5, U5+O2,
      and/or vacancy-compensated Gd-VO motifs
    - defect/charge-compensation model: U5+ compensation, oxygen vacancies, or mixed
    - molar volumes of endmembers and defect motifs from DFT, QHA, MD, or experiment
    - effective coordination numbers from fluorite/Ia-3 geometry, RDFs, or relaxed motifs
    - pair parameters for Gd-U4, Gd-U5, Gd-VO, U4-U5, and host-host interactions
    - DFT/experimental mixing enthalpies, defect-pair energies, activities, or solubility
      limits to fit/validate the MIVM excess Gibbs energy

Implementation rule:
  Atomi evaluates the MIVM excess Gibbs energy:
    Gex/RT = sum_i x_i ln(V_i / sum_j x_j V_j B_ji)
             - 1/2 sum_i Z_i x_i (sum_j x_j B_ji ln(B_ji) / sum_j x_j B_ji)
  The pair matrix is directed: B_ji is encoded as {"from": j, "to": i, ...}.
  Use the generated pycalphad bridge module as a custom Model helper, while keeping
  the parameter JSON as the single source of truth for fitting and provenance.
"""


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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


def load_parameter_database(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = data.get("schema")
    if schema != DATABASE_SCHEMA:
        raise ValueError(f"Expected MIVM database schema {DATABASE_SCHEMA!r}, got {schema!r}.")
    if data.get("mivm_parameter_schema") != SCHEMA:
        raise ValueError(f"MIVM database must reference parameter schema {SCHEMA!r}.")
    return data


def _database_table_rows(db_path: Path, database: dict[str, Any], table_key: str) -> list[dict[str, str]]:
    rel = database.get("tables", {}).get(table_key)
    if not rel:
        raise ValueError(f"MIVM database has no table entry for {table_key!r}.")
    table = db_path.parent / str(rel)
    return read_csv_rows(table)


def _emit_rows(rows: list[dict[str, Any]], fields: list[str], *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    print("\t".join(fields))
    for row in rows:
        print("\t".join(str(row.get(field, "")) for field in fields))


def _filter_parameter_sets(
    rows: list[dict[str, Any]],
    *,
    subgroup: str | None = None,
    component: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if subgroup and row.get("subgroup_id") != subgroup:
            continue
        components = row.get("components", [])
        if isinstance(components, str):
            components = [item for item in components.split(";") if item]
        if component and component not in components:
            continue
        out.append(row)
    return out


def database_action(args: argparse.Namespace) -> dict[str, Any] | None:
    database = load_parameter_database(args.db)
    if args.action == "summary":
        subgroups = database.get("subgroups", [])
        parameter_sets = database.get("parameter_sets", [])
        targets = _database_table_rows(args.db, database, "needed_parameter_checklist")
        summary = {
            "schema": database.get("schema"),
            "description": database.get("description", ""),
            "n_subgroups": len(subgroups),
            "n_parameter_sets": len(parameter_sets),
            "n_targets": len(targets),
            "tables": database.get("tables", {}),
        }
        if args.format == "json":
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"MIVM parameter database: {args.db}")
            print(f"Subgroups: {summary['n_subgroups']}")
            print(f"Parameter sets: {summary['n_parameter_sets']}")
            print(f"Target systems: {summary['n_targets']}")
        return summary
    if args.action == "list":
        rows = _filter_parameter_sets(
            database.get("parameter_sets", []),
            subgroup=args.subgroup,
            component=args.component,
        )
        fields = ["id", "subgroup_id", "confidence", "components", "parameter_file"]
        _emit_rows(rows, fields, output_format=args.format)
        return {"rows": rows}
    if args.action == "map":
        rows = _database_table_rows(args.db, database, "component_mstdb_map")
        out = []
        for row in rows:
            if args.subgroup and row.get("subgroup_id") != args.subgroup:
                continue
            if args.component and row.get("component") != args.component:
                continue
            out.append(row)
        fields = ["component", "mstdb_phase", "mstdb_species_aliases", "role", "parameter_set_id"]
        _emit_rows(out, fields, output_format=args.format)
        return {"rows": out}
    if args.action == "targets":
        rows = _database_table_rows(args.db, database, "needed_parameter_checklist")
        out = []
        for row in rows:
            if args.subgroup and row.get("subgroup_id") != args.subgroup:
                continue
            if args.priority and row.get("priority") != args.priority:
                continue
            out.append(row)
        fields = ["priority", "system", "status", "needed_data"]
        _emit_rows(out, fields, output_format=args.format)
        return {"rows": out}
    if args.action == "validate-all":
        rows = _filter_parameter_sets(
            database.get("parameter_sets", []),
            subgroup=args.subgroup,
            component=args.component,
        )
        results: list[dict[str, Any]] = []
        failures = 0
        for row in rows:
            params_path = args.db.parent / str(row["parameter_file"])
            try:
                params = load_parameters(params_path)
                warnings = validate_parameters(params)
                status = "PASS" if not warnings else "WARN"
            except Exception as exc:
                warnings = [str(exc)]
                status = "FAIL"
                failures += 1
            results.append({"id": row["id"], "status": status, "warnings": warnings})
        if args.format == "json":
            print(json.dumps(results, indent=2, sort_keys=True))
        else:
            for result in results:
                detail = "; ".join(result["warnings"])
                print(f"{result['status']}\t{result['id']}\t{detail}")
        if failures:
            raise ValueError(f"{failures} MIVM parameter set(s) failed validation.")
        return {"results": results}
    raise ValueError(f"Unsupported database action: {args.action}")


@dataclass(frozen=True)
class MIVMComponent:
    name: str
    molar_volume: float
    coordination: float
    reference_gibbs_j_mol: float = 0.0
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class MIVMPair:
    source: str
    target: str
    b: float | None = None
    ln_b: float | None = None
    ln_b_a: float | None = None
    ln_b_b_over_t: float | None = None
    delta_g_j_mol: float | None = None
    metadata: dict[str, Any] | None = None

    def ln_value(self, temperature_k: float) -> float:
        if temperature_k <= 0.0:
            raise ValueError("Temperature must be positive for MIVM pair evaluation.")
        parts: list[float] = []
        if self.b is not None:
            if self.b <= 0.0:
                raise ValueError(f"Pair B for {self.source}->{self.target} must be positive.")
            parts.append(math.log(self.b))
        if self.ln_b is not None:
            parts.append(self.ln_b)
        if self.ln_b_a is not None:
            parts.append(self.ln_b_a)
        if self.ln_b_b_over_t is not None:
            parts.append(self.ln_b_b_over_t / temperature_k)
        if self.delta_g_j_mol is not None:
            parts.append(-self.delta_g_j_mol / (R_J_MOLK * temperature_k))
        if not parts:
            raise ValueError(f"Pair {self.source}->{self.target} has no B/lnB/delta_g parameter.")
        return sum(parts)

    def value(self, temperature_k: float) -> float:
        return math.exp(self.ln_value(temperature_k))


@dataclass(frozen=True)
class MIVMParameters:
    phase: str
    components: dict[str, MIVMComponent]
    pairs: dict[tuple[str, str], MIVMPair]
    basis: str = "mole-fraction"
    description: str = ""

    @property
    def component_names(self) -> list[str]:
        return list(self.components)


def _component_from_mapping(name: str, payload: dict[str, Any]) -> MIVMComponent:
    volume = finite_float(
        payload.get("molar_volume")
        or payload.get("molar_volume_cm3_mol")
        or payload.get("Vm")
        or payload.get("V_m")
    )
    coordination = finite_float(payload.get("coordination") or payload.get("Z") or payload.get("z"))
    if volume is None or volume <= 0.0:
        raise ValueError(f"Component {name!r} needs a positive molar_volume.")
    if coordination is None or coordination <= 0.0:
        raise ValueError(f"Component {name!r} needs a positive coordination/Z value.")
    ref = finite_float(payload.get("reference_gibbs_j_mol") or payload.get("G0") or payload.get("g0")) or 0.0
    return MIVMComponent(
        name=name,
        molar_volume=volume,
        coordination=coordination,
        reference_gibbs_j_mol=ref,
        metadata={key: value for key, value in payload.items() if key not in {"molar_volume", "Vm", "Z", "G0"}},
    )


def _pair_from_mapping(payload: dict[str, Any]) -> MIVMPair:
    source = str(payload.get("from") or payload.get("source") or payload.get("j") or "").strip()
    target = str(payload.get("to") or payload.get("target") or payload.get("i") or "").strip()
    if not source or not target:
        raise ValueError(f"Pair entry needs directed from/to fields: {payload!r}")
    delta_g = (
        finite_float(payload.get("delta_g_j_mol"))
        if payload.get("delta_g_j_mol") is not None
        else finite_float(payload.get("interaction_j_mol") or payload.get("epsilon_j_mol"))
    )
    return MIVMPair(
        source=source,
        target=target,
        b=finite_float(payload.get("B") or payload.get("b")),
        ln_b=finite_float(payload.get("lnB") or payload.get("ln_b")),
        ln_b_a=finite_float(payload.get("lnB_A") or payload.get("ln_b_a")),
        ln_b_b_over_t=finite_float(payload.get("lnB_B_over_T") or payload.get("ln_b_b_over_t")),
        delta_g_j_mol=delta_g,
        metadata={key: value for key, value in payload.items() if key not in {"from", "to", "source", "target", "j", "i"}},
    )


def parameters_from_mapping(data: dict[str, Any]) -> MIVMParameters:
    raw_components = data.get("components")
    if not raw_components:
        raise ValueError("MIVM parameter file needs a non-empty 'components' section.")
    components: dict[str, MIVMComponent] = {}
    if isinstance(raw_components, dict):
        for name, payload in raw_components.items():
            if not isinstance(payload, dict):
                raise ValueError(f"Component {name!r} must be a mapping.")
            components[str(name)] = _component_from_mapping(str(name), payload)
    elif isinstance(raw_components, list):
        for payload in raw_components:
            if not isinstance(payload, dict) or not payload.get("name"):
                raise ValueError("Component list entries must be mappings with a name.")
            name = str(payload["name"])
            components[name] = _component_from_mapping(name, payload)
    else:
        raise ValueError("'components' must be a mapping or list.")

    pairs: dict[tuple[str, str], MIVMPair] = {}
    for payload in data.get("pairs", []) or []:
        if not isinstance(payload, dict):
            raise ValueError(f"Pair entries must be mappings, got {payload!r}")
        pair = _pair_from_mapping(payload)
        pairs[(pair.source, pair.target)] = pair
    for name in components:
        pairs.setdefault((name, name), MIVMPair(source=name, target=name, b=1.0))
    return MIVMParameters(
        phase=str(data.get("phase") or data.get("phase_name") or "MIVM_PHASE").upper(),
        components=components,
        pairs=pairs,
        basis=str(data.get("basis") or "mole-fraction"),
        description=str(data.get("description") or ""),
    )


def load_parameters(path: Path) -> MIVMParameters:
    return parameters_from_mapping(json.loads(path.read_text(encoding="utf-8")))


def validate_parameters(params: MIVMParameters) -> list[str]:
    warnings: list[str] = []
    names = params.component_names
    for source in names:
        for target in names:
            if (source, target) not in params.pairs:
                warnings.append(f"Missing directed pair B_{source},{target}; sampling will fail unless it is supplied.")
    if params.basis != "mole-fraction":
        warnings.append(f"Basis {params.basis!r} is recorded but Atomi sampling currently assumes mole fractions.")
    return warnings


def normalize_composition(composition: dict[str, float], params: MIVMParameters) -> dict[str, float]:
    clean = {name: float(composition.get(name, 0.0)) for name in params.component_names}
    missing = [name for name, value in composition.items() if name not in params.components and abs(value) > 0.0]
    if missing:
        raise ValueError(f"Composition contains unknown MIVM components: {', '.join(sorted(missing))}")
    if any(value < -1.0e-14 for value in clean.values()):
        raise ValueError("Composition mole fractions must be non-negative.")
    total = sum(max(value, 0.0) for value in clean.values())
    if total <= 0.0:
        raise ValueError("Composition must contain at least one positive component fraction.")
    return {name: max(value, 0.0) / total for name, value in clean.items()}


def pair_b(params: MIVMParameters, source: str, target: str, temperature_k: float) -> float:
    pair = params.pairs.get((source, target))
    if pair is None:
        raise ValueError(f"Missing directed MIVM pair parameter from={source!r} to={target!r}.")
    return pair.value(temperature_k)


def pair_ln_b(params: MIVMParameters, source: str, target: str, temperature_k: float) -> float:
    pair = params.pairs.get((source, target))
    if pair is None:
        raise ValueError(f"Missing directed MIVM pair parameter from={source!r} to={target!r}.")
    return pair.ln_value(temperature_k)


def excess_gibbs_j_mol(
    temperature_k: float,
    composition: dict[str, float],
    params: MIVMParameters,
) -> float:
    """Return MIVM molar excess Gibbs energy in J/mol on the component basis."""
    if temperature_k <= 0.0:
        raise ValueError("Temperature must be positive.")
    x = normalize_composition(composition, params)
    names = params.component_names
    size_term = 0.0
    pair_term = 0.0
    for target in names:
        x_i = x[target]
        if x_i <= 0.0:
            continue
        comp_i = params.components[target]
        volume_denom = 0.0
        b_denom = 0.0
        b_log_numer = 0.0
        for source in names:
            x_j = x[source]
            if x_j <= 0.0:
                continue
            comp_j = params.components[source]
            b_ji = pair_b(params, source, target, temperature_k)
            ln_b_ji = pair_ln_b(params, source, target, temperature_k)
            volume_denom += x_j * comp_j.molar_volume * b_ji
            b_denom += x_j * b_ji
            b_log_numer += x_j * b_ji * ln_b_ji
        if volume_denom <= 0.0 or b_denom <= 0.0:
            raise ValueError(f"Invalid MIVM denominator for component {target!r}.")
        size_term += x_i * math.log(comp_i.molar_volume / volume_denom)
        pair_term += -0.5 * comp_i.coordination * x_i * (b_log_numer / b_denom)
    return R_J_MOLK * temperature_k * (size_term + pair_term)


def reference_gibbs_j_mol(composition: dict[str, float], params: MIVMParameters) -> float:
    x = normalize_composition(composition, params)
    return sum(x[name] * params.components[name].reference_gibbs_j_mol for name in params.component_names)


def ideal_gibbs_j_mol(temperature_k: float, composition: dict[str, float], params: MIVMParameters) -> float:
    x = normalize_composition(composition, params)
    return R_J_MOLK * temperature_k * sum(value * math.log(value) for value in x.values() if value > 0.0)


def total_gibbs_j_mol(
    temperature_k: float,
    composition: dict[str, float],
    params: MIVMParameters,
) -> float:
    return (
        reference_gibbs_j_mol(composition, params)
        + ideal_gibbs_j_mol(temperature_k, composition, params)
        + excess_gibbs_j_mol(temperature_k, composition, params)
    )


def excess_enthalpy_gibbs_helmholtz_j_mol(
    temperature_k: float,
    composition: dict[str, float],
    params: MIVMParameters,
    rel_step: float = 1.0e-4,
) -> float:
    """Return H_ex from Gibbs-Helmholtz using the MIVM G_ex(T)."""
    dt = max(1.0e-3, abs(temperature_k) * rel_step)
    t0 = max(1.0e-6, temperature_k - dt)
    t1 = temperature_k + dt
    g_over_t0 = excess_gibbs_j_mol(t0, composition, params) / t0
    g_over_t1 = excess_gibbs_j_mol(t1, composition, params) / t1
    derivative = (g_over_t1 - g_over_t0) / (t1 - t0)
    return -(temperature_k**2) * derivative


def excess_enthalpy_j_mol(
    temperature_k: float,
    composition: dict[str, float],
    params: MIVMParameters,
) -> float:
    """Return the direct MIVM excess/mixing enthalpy in J/mol.

    This follows the molten-salt MIVM expression used for mixing-enthalpy
    curves, where constant directed B_ji parameters still carry pair-energy
    information and therefore produce a nonzero H_ex.
    """
    if temperature_k <= 0.0:
        raise ValueError("Temperature must be positive.")
    x = normalize_composition(composition, params)
    enthalpy_over_rt = 0.0
    for target in params.component_names:
        x_i = x[target]
        if x_i <= 0.0:
            continue
        comp_i = params.components[target]
        b_denom = 0.0
        b_log_numer = 0.0
        b_log_weighted = 0.0
        volume_denom = 0.0
        volume_log_numer = 0.0
        for source in params.component_names:
            x_j = x[source]
            if x_j <= 0.0:
                continue
            comp_j = params.components[source]
            b_ji = pair_b(params, source, target, temperature_k)
            ln_b_ji = pair_ln_b(params, source, target, temperature_k)
            b_weight = x_j * b_ji
            volume_weight = x_j * comp_j.molar_volume * b_ji
            b_denom += b_weight
            b_log_numer += b_weight * ln_b_ji
            b_log_weighted += (1.0 + ln_b_ji) * b_weight * ln_b_ji
            volume_denom += volume_weight
            volume_log_numer += volume_weight * ln_b_ji
        if b_denom <= 0.0 or volume_denom <= 0.0:
            raise ValueError(f"Invalid MIVM enthalpy denominator for component {target!r}.")
        avg_ln_b = b_log_numer / b_denom
        enthalpy_over_rt += 0.5 * comp_i.coordination * x_i * (
            avg_ln_b**2 - b_log_weighted / b_denom
        )
        enthalpy_over_rt += -x_i * (volume_log_numer / volume_denom)
    return R_J_MOLK * temperature_k * enthalpy_over_rt


def _molar_total_from_moles(
    temperature_k: float,
    moles: dict[str, float],
    params: MIVMParameters,
) -> float:
    total = sum(max(value, 0.0) for value in moles.values())
    if total <= 0.0:
        raise ValueError("Mole vector must contain at least one positive amount.")
    composition = {name: max(value, 0.0) / total for name, value in moles.items()}
    return total * total_gibbs_j_mol(temperature_k, composition, params)


def chemical_potentials_j_mol(
    temperature_k: float,
    composition: dict[str, float],
    params: MIVMParameters,
    step: float = 1.0e-6,
) -> dict[str, float]:
    x = normalize_composition(composition, params)
    mu: dict[str, float] = {}
    for name in params.component_names:
        plus = dict(x)
        plus[name] += step
        f_plus = _molar_total_from_moles(temperature_k, plus, params)
        if x[name] > step * 1.5:
            minus = dict(x)
            minus[name] -= step
            f_minus = _molar_total_from_moles(temperature_k, minus, params)
            mu[name] = (f_plus - f_minus) / (2.0 * step)
        else:
            f_base = _molar_total_from_moles(temperature_k, x, params)
            mu[name] = (f_plus - f_base) / step
    return mu


def activity_report(
    temperature_k: float,
    composition: dict[str, float],
    params: MIVMParameters,
) -> dict[str, dict[str, float]]:
    x = normalize_composition(composition, params)
    mu = chemical_potentials_j_mol(temperature_k, x, params)
    report: dict[str, dict[str, float]] = {}
    for name in params.component_names:
        ref = params.components[name].reference_gibbs_j_mol
        ln_activity = (mu[name] - ref) / (R_J_MOLK * temperature_k)
        activity = math.exp(max(min(ln_activity, 700.0), -700.0))
        gamma = activity / x[name] if x[name] > 0.0 else math.nan
        report[name] = {
            "mu_J_mol": mu[name],
            "activity": activity,
            "gamma": gamma,
            "ln_activity": ln_activity,
        }
    return report


def parameter_guide() -> dict[str, Any]:
    return {
        "general": [
            "phase_name and component basis used by pycalphad",
            "reference/endmember Gibbs energies on that basis",
            "molar volumes Vm_i in consistent units",
            "coordination numbers Z_i or effective coordination numbers",
            "directed pair parameters B_ij/B_ji or epsilon/h parameters",
            "temperature and composition ranges for fitting and validation",
            "mixing enthalpy, activity, chemical potential, or phase-equilibrium data",
        ],
        "molten_salt": [
            "define salt components or pseudo-binary components, e.g. LaCl3 and eutectic LiCl-KCl",
            "liquid molar volumes from density, TMA, or MD",
            "RDF/CN-derived solvation coordination numbers on the chosen component basis",
            "PMF-derived or fitted pair parameters B_ij/B_ji",
            "calorimetry Hmix and activity data for refining B_ij/B_ji",
            "explicit formula-unit normalization for common-anion or charge-asymmetric salts",
        ],
        "ceramic_solid": [
            "define substitutional/pseudo-components, e.g. U4+O2, Gd3+O1.5, U5+O2, Gd-VO motifs",
            "state the charge-compensation mechanism: U5+, oxygen vacancies, or mixed compensation",
            "molar volumes for endmembers and defect motifs from DFT, QHA, MD, or experiment",
            "effective coordination numbers from fluorite/Ia-3 geometry, RDFs, or relaxed motifs",
            "pair parameters for Gd-U4, Gd-U5, Gd-VO, U4-U5, and host-host interactions",
            "DFT/experimental mixing enthalpies, defect-pair energies, activities, or solubility limits",
            "magnetic/oxidation-state labels used to map DFT structures onto thermodynamic components",
        ],
        "pycalphad_strategy": [
            "evaluate the MIVM excess Gibbs energy before fitting or equilibrium use",
            "use the MIVM parameter JSON as the provenance source",
            "sample MIVM tables for Atomi/MOOSE screening and fitting",
            "use pycalphad-bridge to generate a custom Model class for equilibrium calculations",
            "record all parameter units and component normalization in the TDB or companion metadata",
        ],
    }


def text_guide(system: str) -> str:
    data = parameter_guide()
    sections = ["general"]
    if system in {"molten", "all"}:
        sections.append("molten_salt")
    if system in {"ceramic", "solid", "all"}:
        sections.append("ceramic_solid")
    sections.append("pycalphad_strategy")

    labels = {
        "general": "General MIVM/pycalphad parameters",
        "molten_salt": "Molten-salt MIVM parameters",
        "ceramic_solid": "Solid/ceramic MIVM parameters, e.g. (Gd,U)O2",
        "pycalphad_strategy": "Implementation strategy",
    }
    lines = ["MIVM parameter guide", "====================", ""]
    for section in sections:
        lines.append(labels[section])
        lines.append("-" * len(labels[section]))
        for item in data[section]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def template_parameters(system: str) -> dict[str, Any]:
    if system == "molten":
        return {
            "schema": SCHEMA,
            "phase": "LIQUID",
            "basis": "mole-fraction",
            "description": "Example molten-salt pseudo-binary MIVM parameter file.",
            "components": {
                "LaCl3": {"molar_volume": 58.0, "coordination": 8.0, "reference_gibbs_j_mol": 0.0},
                "LiKCl_eut": {"molar_volume": 32.0, "coordination": 6.0, "reference_gibbs_j_mol": 0.0},
            },
            "pairs": [
                {"from": "LaCl3", "to": "LiKCl_eut", "B": 0.85},
                {"from": "LiKCl_eut", "to": "LaCl3", "B": 1.10},
            ],
        }
    return {
        "schema": SCHEMA,
        "phase": "FLUORITE",
        "basis": "mole-fraction",
        "description": "Example ceramic MIVM parameter file for a charge-compensated (Gd,U)O2 basis.",
        "components": {
            "U4O2": {"molar_volume": 24.6, "coordination": 12.0, "reference_gibbs_j_mol": 0.0},
            "GdVO": {"molar_volume": 25.0, "coordination": 12.0, "reference_gibbs_j_mol": 5000.0},
            "U5O2": {"molar_volume": 24.2, "coordination": 12.0, "reference_gibbs_j_mol": 3000.0},
        },
        "pairs": [
            {"from": "GdVO", "to": "U4O2", "B": 0.95},
            {"from": "U4O2", "to": "GdVO", "B": 1.05},
            {"from": "U5O2", "to": "U4O2", "B": 1.02},
            {"from": "U4O2", "to": "U5O2", "B": 0.98},
            {"from": "GdVO", "to": "U5O2", "B": 1.08},
            {"from": "U5O2", "to": "GdVO", "B": 0.92},
        ],
    }


def parse_key_value_floats(items: list[str] | None) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in items or []:
        for part in item.replace(";", ",").split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(f"Expected NAME=VALUE, got {part!r}")
            key, value = part.split("=", 1)
            values[key.strip()] = float(value)
    return values


def parse_compositions(items: list[str] | None, params: MIVMParameters) -> list[dict[str, float]]:
    if not items:
        equal = 1.0 / len(params.components)
        return [{name: equal for name in params.component_names}]
    return [normalize_composition(parse_key_value_floats([item]), params) for item in items]


def binary_grid(spec: str, params: MIVMParameters) -> list[dict[str, float]]:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if len(parts) == 3:
        a, b, step_text = parts
        x_min, x_max, step = 0.0, 1.0, float(step_text)
    elif len(parts) == 5:
        a, b, x_min_text, x_max_text, step_text = parts
        x_min, x_max, step = float(x_min_text), float(x_max_text), float(step_text)
    else:
        raise ValueError("--binary-grid expects A,B,step or A,B,xmin,xmax,step")
    if a not in params.components or b not in params.components:
        raise ValueError("--binary-grid components must exist in the parameter file.")
    if step <= 0.0 or x_min < 0.0 or x_max > 1.0 or x_min > x_max:
        raise ValueError("Invalid binary grid range.")
    out: list[dict[str, float]] = []
    n_steps = int(round((x_max - x_min) / step))
    for index in range(n_steps + 1):
        xa = min(x_max, x_min + index * step)
        comp = {name: 0.0 for name in params.component_names}
        comp[a] = xa
        comp[b] = 1.0 - xa
        out.append(normalize_composition(comp, params))
    if not out or abs(out[-1][a] - x_max) > 1.0e-10:
        comp = {name: 0.0 for name in params.component_names}
        comp[a] = x_max
        comp[b] = 1.0 - x_max
        out.append(normalize_composition(comp, params))
    return out


def temperature_grid(args: argparse.Namespace) -> list[float]:
    values = [float(item) for item in args.temperature or []]
    if values:
        return sorted(dict.fromkeys(values))
    if args.T_min is None or args.T_max is None:
        return [1000.0]
    step = args.T_step or 100.0
    if step <= 0.0:
        raise ValueError("--T-step must be positive.")
    temps: list[float] = []
    current = float(args.T_min)
    while current <= float(args.T_max) + 1.0e-9:
        temps.append(round(current, 10))
        current += step
    return temps


def composition_label(composition: dict[str, float]) -> str:
    return ";".join(f"{name}={value:.8g}" for name, value in composition.items() if value > 0.0)


def sample_rows(
    params: MIVMParameters,
    temperatures: list[float],
    compositions: list[dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for temp in temperatures:
        for composition in compositions:
            x = normalize_composition(composition, params)
            activities = activity_report(temp, x, params)
            row: dict[str, Any] = {
                "T_K": temp,
                "composition": composition_label(x),
                "phase": params.phase,
                "G_reference_J_mol": reference_gibbs_j_mol(x, params),
                "G_ideal_J_mol": ideal_gibbs_j_mol(temp, x, params),
                "G_excess_MIVM_J_mol": excess_gibbs_j_mol(temp, x, params),
                "H_excess_MIVM_J_mol": excess_enthalpy_j_mol(temp, x, params),
                "G_total_MIVM_J_mol": total_gibbs_j_mol(temp, x, params),
            }
            for name in params.component_names:
                row[f"x_{name}"] = x[name]
                row[f"mu_{name}_J_mol"] = activities[name]["mu_J_mol"]
                row[f"activity_{name}"] = activities[name]["activity"]
                row[f"gamma_{name}"] = activities[name]["gamma"]
            rows.append(row)
    return rows


def sample_fields(params: MIVMParameters) -> list[str]:
    fields = [
        "T_K",
        "phase",
        "composition",
        "G_reference_J_mol",
        "G_ideal_J_mol",
        "G_excess_MIVM_J_mol",
        "H_excess_MIVM_J_mol",
        "G_total_MIVM_J_mol",
    ]
    for name in params.component_names:
        fields.extend([f"x_{name}", f"mu_{name}_J_mol", f"activity_{name}", f"gamma_{name}"])
    return fields


def _interp_linear(points: list[tuple[float, float]], x_value: float) -> float | None:
    if not points:
        return None
    sorted_points = sorted(points)
    if x_value < sorted_points[0][0] or x_value > sorted_points[-1][0]:
        return None
    for x0, y0 in sorted_points:
        if abs(x_value - x0) <= 1.0e-12:
            return y0
    for (x0, y0), (x1, y1) in zip(sorted_points, sorted_points[1:]):
        if x0 <= x_value <= x1:
            if abs(x1 - x0) <= 1.0e-15:
                return y0
            frac = (x_value - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return None


def _read_xy_points(
    path: Path,
    *,
    x_column: str,
    y_column: str,
    y_unit: str = "kJ/mol",
) -> list[tuple[float, float]]:
    scale = 1.0e-3 if y_unit == "J/mol" else 1.0
    rows = read_csv_rows(path)
    points: list[tuple[float, float]] = []
    for row in rows:
        x = finite_float(row.get(x_column))
        y = finite_float(row.get(y_column))
        if x is None or y is None:
            continue
        points.append((x, y * scale))
    return sorted(points)


def sanitize_mstdb_chemsage_text(text: str) -> tuple[str, dict[str, Any]]:
    """Make common MSTDB ChemSage charged species names safe for pycalphad parsing.

    Pycalphad's ChemSage reader can load MSTDB exports, but labels such as
    ``U[CN=VI]+3.0`` are parsed as if ``CN`` and ``VI`` were chemical elements.
    The sanitization keeps formal charge text and real-element stoichiometry:
    ``U[CN=VI]+3.0 -> U+3.0`` and ``U[DIMER]+6.0 -> U2+6.0``.
    """
    original = text
    replacements: list[dict[str, str]] = []

    def sub(pattern: str, repl: str, label: str) -> None:
        nonlocal text
        text, count = re.subn(pattern, repl, text)
        if count:
            replacements.append({"pattern": label, "replacement": repl, "count": str(count)})

    sub(r"\[CN=[A-Za-z0-9_+\-]+\]", "", "coordination tag [CN=*]")
    sub(r"\[([1-9][0-9]*)\+\]", "", "formal charge bracket [n+]")
    sub(
        r"\b([A-Z][A-Z]?)(\[DIMER\])(\+[0-9]+(?:\.[0-9]+)?)",
        r"\g<1>2\3",
        "metal dimer tag [DIMER]",
    )
    metadata = {
        "schema": "atomi.calphad.mivm.mstdb_chemsage_sanitizer.v1",
        "changed": text != original,
        "replacements": replacements,
        "notes": [
            "Use this for pycalphad parsing of ChemSage/MSTDB exports with bracketed charged-species labels.",
            "The original database remains the provenance source; validate sanitized outputs against known diagrams or mixing heats.",
        ],
    }
    return text, metadata


def write_sanitized_mstdb_chemsage(source: Path, out: Path, *, metadata_path: Path | None = None) -> dict[str, Any]:
    text = source.read_text(encoding="utf-8", errors="ignore")
    sanitized, metadata = sanitize_mstdb_chemsage_text(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sanitized, encoding="utf-8")
    metadata.update(
        {
            "source": str(source.resolve()),
            "output": str(out.resolve()),
            "source_size_bytes": source.stat().st_size,
            "output_size_bytes": out.stat().st_size,
        }
    )
    if metadata_path:
        write_json(metadata_path, metadata)
    return metadata


def parse_formula_counts(formula: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for element, count_text in re.findall(r"([A-Z][a-z]?)([0-9.]*)", formula):
        count = float(count_text) if count_text else 1.0
        counts[element.upper()] = counts.get(element.upper(), 0.0) + count
    if not counts:
        raise ValueError(f"Could not parse chemical formula {formula!r}.")
    return counts


def _float_grid(spec: str) -> list[float]:
    parts = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if len(parts) == 1:
        x_min, x_max, step = 0.0, 1.0, parts[0]
    elif len(parts) == 3:
        x_min, x_max, step = parts
    else:
        raise ValueError("--grid expects step or xmin,xmax,step")
    if step <= 0.0 or x_min < 0.0 or x_max > 1.0 or x_min > x_max:
        raise ValueError("Invalid grid range.")
    values: list[float] = []
    n_steps = int(round((x_max - x_min) / step))
    for index in range(n_steps + 1):
        values.append(min(x_max, x_min + index * step))
    if not values or abs(values[-1] - x_max) > 1.0e-10:
        values.append(x_max)
    return values


def _binary_formula_amounts(x_value: float, component_a: str, component_b: str, x_component: str) -> tuple[float, float]:
    if x_component == component_a:
        return x_value, 1.0 - x_value
    if x_component == component_b:
        return 1.0 - x_value, x_value
    raise ValueError("--x-component must be one of --component-a or --component-b.")


def _atom_fractions(amount_a: float, amount_b: float, counts_a: dict[str, float], counts_b: dict[str, float]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for element, count in counts_a.items():
        totals[element] = totals.get(element, 0.0) + amount_a * count
    for element, count in counts_b.items():
        totals[element] = totals.get(element, 0.0) + amount_b * count
    total_atoms = sum(totals.values())
    if total_atoms <= 0.0:
        raise ValueError("Binary formula amounts produce zero atoms.")
    return {element: count / total_atoms for element, count in totals.items()}


def _mqmqa_hm_atom(
    db: Any,
    phase: str,
    elements: list[str],
    atom_fractions: dict[str, float],
    temperature: float,
    pressure: float,
    *,
    dependent_element: str | None = None,
) -> float:
    try:
        from pycalphad import equilibrium, variables as v
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("pycalphad is required for mqmqa-binary.") from exc

    dependent = dependent_element.upper() if dependent_element else max(atom_fractions, key=lambda item: atom_fractions[item])
    if dependent not in atom_fractions:
        raise ValueError(f"Dependent element {dependent!r} is not present in this binary formula system.")
    cond: dict[Any, float] = {v.T: temperature, v.P: pressure}
    for element, fraction in atom_fractions.items():
        if element != dependent:
            cond[v.X(element)] = max(float(fraction), 1.0e-10)
    eq = equilibrium(db, elements, [phase], cond, output="HM", verbose=False)
    values = [float(value) for value in eq.HM.values.ravel()]
    for value in values:
        if math.isfinite(value):
            return value
    raise ValueError("pycalphad returned no finite HM value.")


def write_mqmqa_binary_curve(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from pycalphad import Database
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("pycalphad is required for mqmqa-binary.") from exc

    db = Database(str(args.tdb))
    counts_a = parse_formula_counts(args.component_a)
    counts_b = parse_formula_counts(args.component_b)
    elements = sorted(set(counts_a) | set(counts_b))
    unique_a = sorted(set(counts_a) - set(counts_b))
    dependent_element = args.dependent_element.upper() if args.dependent_element else (unique_a[0] if unique_a else None)
    grid = _float_grid(args.grid)
    eps = args.endpoint_epsilon
    ref_x_a = 1.0 - eps if args.x_component == args.component_a else eps
    ref_x_b = eps if args.x_component == args.component_a else 1.0 - eps
    amount_a_ref, amount_b_ref = _binary_formula_amounts(ref_x_a, args.component_a, args.component_b, args.x_component)
    h_a = _mqmqa_hm_atom(
        db,
        args.phase,
        elements,
        _atom_fractions(amount_a_ref, amount_b_ref, counts_a, counts_b),
        args.temperature,
        args.pressure,
        dependent_element=dependent_element,
    )
    amount_a_ref, amount_b_ref = _binary_formula_amounts(ref_x_b, args.component_a, args.component_b, args.x_component)
    h_b = _mqmqa_hm_atom(
        db,
        args.phase,
        elements,
        _atom_fractions(amount_a_ref, amount_b_ref, counts_a, counts_b),
        args.temperature,
        args.pressure,
        dependent_element=dependent_element,
    )
    unique_b = sorted(set(counts_b) - set(counts_a))
    pure_b_unique_fraction = sum(counts_b[element] for element in unique_b) / sum(counts_b.values()) if unique_b else None
    rows: list[dict[str, Any]] = []
    atoms_a = sum(counts_a.values())
    atoms_b = sum(counts_b.values())
    for x_value in grid:
        amount_a, amount_b = _binary_formula_amounts(x_value, args.component_a, args.component_b, args.x_component)
        atom_fractions = _atom_fractions(amount_a, amount_b, counts_a, counts_b)
        h_mix_atom = _mqmqa_hm_atom(
            db,
            args.phase,
            elements,
            atom_fractions,
            args.temperature,
            args.pressure,
            dependent_element=dependent_element,
        )
        if pure_b_unique_fraction and unique_b:
            alpha_b = sum(atom_fractions[element] for element in unique_b) / pure_b_unique_fraction
        else:
            alpha_b = amount_b
        alpha_b = min(1.0, max(0.0, alpha_b))
        h_excess_atom = (h_mix_atom - ((1.0 - alpha_b) * h_a + alpha_b * h_b)) / 1000.0
        total_atoms_per_formula_mix = amount_a * atoms_a + amount_b * atoms_b
        h_excess_formula = (
            h_mix_atom * total_atoms_per_formula_mix - (amount_a * h_a * atoms_a + amount_b * h_b * atoms_b)
        ) / 1000.0
        row: dict[str, Any] = {
            "T_K": args.temperature,
            "phase": args.phase,
            "x_component": args.x_component,
            "x": x_value,
            "component_a": args.component_a,
            "component_b": args.component_b,
            "Hmix_kJ_mol": h_excess_atom,
            "Hmix_kJ_per_mol_atom": h_excess_atom,
            "Hmix_kJ_per_mol_formula_mixture": h_excess_formula,
            "HM_atom_J_mol": h_mix_atom,
            "reference_HM_atom_a_J_mol": h_a,
            "reference_HM_atom_b_J_mol": h_b,
            "atom_reference_alpha_b": alpha_b,
        }
        for element in elements:
            row[f"X_{element}"] = atom_fractions[element]
        rows.append(row)
    outdir = args.outdir.resolve()
    table = outdir / "mqmqa_binary_curve.csv"
    fields = list(rows[0])
    write_csv(table, rows, fields)
    metadata = {
        "schema": "atomi.calphad.mivm.mqmqa_binary_curve.v1",
        "tdb": str(args.tdb.resolve()),
        "phase": args.phase,
        "temperature_K": args.temperature,
        "pressure_Pa": args.pressure,
        "component_a": args.component_a,
        "component_b": args.component_b,
        "x_component": args.x_component,
        "dependent_element": dependent_element,
        "grid": args.grid,
        "basis_note": (
            "Hmix_kJ_mol is an atom-molar endmember-linear excess enthalpy for compatibility with "
            "pycalphad/MSTDB HM curves. Hmix_kJ_per_mol_formula_mixture multiplies HM by formula-mixture atoms."
        ),
        "outputs": {"curve_csv": str(table)},
    }
    write_json(outdir / "mqmqa_binary_curve_metadata.json", metadata)
    return metadata


def tdb_sanity_check(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    upper = text.upper()
    lines = text.splitlines()
    counts = {
        "ELEMENT": sum(1 for line in lines if line.strip().upper().startswith("ELEMENT ")),
        "PHASE": sum(1 for line in lines if line.strip().upper().startswith("PHASE ")),
        "CONSTITUENT": sum(1 for line in lines if line.strip().upper().startswith("CONSTITUENT ")),
        "PARAMETER": sum(1 for line in lines if line.strip().upper().startswith("PARAMETER ")),
        "CHEMSAGE_SYSTEM": sum(1 for line in lines[:10] if line.strip().upper().startswith("SYSTEM ")),
        "MQMQA_LITERAL": upper.count("MQMQA"),
        "BRACKETED_CHARGED_SPECIES_HINTS": len(
            re.findall(r"\[[A-Z0-9_=+\-]+\]\+[0-9]+(?:\.[0-9]+)?|\[DIMER\]\+[0-9]+", text)
        ),
    }
    warnings: list[str] = []
    native_tdb_like = counts["PHASE"] > 0 and counts["PARAMETER"] > 0
    chemsage_like = bool(counts["CHEMSAGE_SYSTEM"])
    if counts["PHASE"] == 0 or counts["PARAMETER"] == 0:
        warnings.append(
            "File does not look like a native pycalphad TDB "
            "(missing PHASE or PARAMETER records)."
        )
    if counts["CHEMSAGE_SYSTEM"]:
        warnings.append(
            "File looks like a ChemSage/MSTDB export rather than a native TDB; "
            "pycalphad may parse it with the ChemSage reader, but the result must be benchmarked."
        )
    if counts["BRACKETED_CHARGED_SPECIES_HINTS"]:
        warnings.append(
            "Bracketed charged-species labels were detected; sanitize labels before pycalphad equilibrium "
            "if species such as U[CN=VI]+3.0 are parsed into pseudo-elements."
        )
    if "NO_FUNCTIONS" in path.name.upper():
        warnings.append("Filename indicates a no-functions/export subset; verify that all Gibbs functions survived conversion.")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "counts": counts,
        "warnings": warnings,
        "looks_like_pycalphad_tdb": native_tdb_like and not warnings,
        "native_tdb_like": native_tdb_like,
        "chemsage_like": chemsage_like,
        "needs_species_label_sanitization": counts["BRACKETED_CHARGED_SPECIES_HINTS"] > 0,
    }


def compare_binary_model(
    params: MIVMParameters,
    *,
    temperature_k: float,
    binary_grid_spec: str,
    x_component: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for composition in binary_grid(binary_grid_spec, params):
        x = normalize_composition(composition, params)
        rows.append(
            {
                "T_K": temperature_k,
                "x_component": x_component,
                "x": x[x_component],
                "composition": composition_label(x),
                "H_excess_MIVM_kJ_mol": excess_enthalpy_j_mol(temperature_k, x, params) / 1000.0,
                "G_excess_MIVM_kJ_mol": excess_gibbs_j_mol(temperature_k, x, params) / 1000.0,
            }
        )
    return rows


def comparison_metrics(
    model_points: list[tuple[float, float]],
    reference_points: list[tuple[float, float]],
    *,
    reference_label: str,
) -> dict[str, Any]:
    residuals: list[float] = []
    matched = 0
    for x, y_ref in reference_points:
        y_model = _interp_linear(model_points, x)
        if y_model is None:
            continue
        matched += 1
        residuals.append(y_model - y_ref)
    if not residuals:
        return {
            "reference": reference_label,
            "matched_points": 0,
            "rmse_kJ_mol": None,
            "mae_kJ_mol": None,
            "mean_bias_kJ_mol": None,
        }
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    mae = sum(abs(value) for value in residuals) / len(residuals)
    bias = sum(residuals) / len(residuals)
    return {
        "reference": reference_label,
        "matched_points": matched,
        "rmse_kJ_mol": rmse,
        "mae_kJ_mol": mae,
        "mean_bias_kJ_mol": bias,
    }


def _read_curve_family(
    path: Path,
    *,
    x_column: str,
    curve_columns: list[str],
    y_unit: str = "kJ/mol",
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    scale = 1.0e-3 if y_unit == "J/mol" else 1.0
    curves: list[dict[str, Any]] = []
    for index, column in enumerate(curve_columns):
        points: list[tuple[float, float]] = []
        for row in rows:
            x = finite_float(row.get(x_column))
            y = finite_float(row.get(column))
            if x is None or y is None:
                continue
            points.append((x, y * scale))
        if not points:
            raise ValueError(f"Curve column {column!r} in {path} has no finite points.")
        label = labels[index] if labels and index < len(labels) else column
        curves.append(
            {
                "label": label,
                "source": str(path),
                "x_column": x_column,
                "y_column": column,
                "points_kJ_mol": sorted(points),
            }
        )
    return curves


def _curve_minimum(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return (math.nan, math.nan)
    return min(points, key=lambda item: item[1])


def _curve_derivatives(points: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    sorted_points = sorted(points)
    out: list[tuple[float, float, float]] = []
    for index, (x, y) in enumerate(sorted_points):
        if index == 0 and len(sorted_points) > 1:
            x0, y0 = sorted_points[index]
            x1, y1 = sorted_points[index + 1]
        elif index == len(sorted_points) - 1 and len(sorted_points) > 1:
            x0, y0 = sorted_points[index - 1]
            x1, y1 = sorted_points[index]
        elif len(sorted_points) > 2:
            x0, y0 = sorted_points[index - 1]
            x1, y1 = sorted_points[index + 1]
        else:
            x0, y0 = x, y
            x1, y1 = x, y
        dydx = (y1 - y0) / (x1 - x0) if abs(x1 - x0) > 1.0e-15 else 0.0
        out.append((x, y, dydx))
    return out


def _fusion_gibbs_j_mol(temperature_k: float, *, tm_k: float, dhfus_kj_mol: float, dcp_j_mol_k: float = 0.0) -> float:
    if temperature_k <= 0.0 or tm_k <= 0.0:
        return math.nan
    dh_j_mol = dhfus_kj_mol * 1000.0
    return dh_j_mol * (1.0 - temperature_k / tm_k) + dcp_j_mol_k * (
        (temperature_k - tm_k) - temperature_k * math.log(temperature_k / tm_k)
    )


def _solve_liquidus_temperature(
    x: float,
    mu_ex_j_mol: float,
    *,
    tm_k: float,
    dhfus_kj_mol: float,
    dcp_j_mol_k: float = 0.0,
) -> float:
    if x <= 0.0:
        return math.nan

    def residual(temp: float) -> float:
        return _fusion_gibbs_j_mol(temp, tm_k=tm_k, dhfus_kj_mol=dhfus_kj_mol, dcp_j_mol_k=dcp_j_mol_k) + (
            R_J_MOLK * temp * math.log(max(x, 1.0e-15))
        ) + mu_ex_j_mol

    low = 1.0
    high = max(tm_k * 1.5, tm_k + 1000.0)
    f_low = residual(low)
    f_high = residual(high)
    expand = 0
    while math.isfinite(f_low) and math.isfinite(f_high) and f_low * f_high > 0.0 and expand < 12:
        high *= 1.5
        f_high = residual(high)
        expand += 1
    if not (math.isfinite(f_low) and math.isfinite(f_high)) or f_low * f_high > 0.0:
        return math.nan
    for _ in range(100):
        mid = 0.5 * (low + high)
        f_mid = residual(mid)
        if not math.isfinite(f_mid):
            return math.nan
        if abs(f_mid) < 1.0e-8 or abs(high - low) < 1.0e-8:
            return mid
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return 0.5 * (low + high)


def _line_compound_gform_j_mol(
    temperature_k: float,
    *,
    gform_ref_kj_mol: float,
    tref_k: float,
    dcp_form_j_mol_k: float = 0.0,
) -> float:
    if temperature_k <= 0.0 or tref_k <= 0.0:
        return math.nan
    return gform_ref_kj_mol * 1000.0 + dcp_form_j_mol_k * (
        (temperature_k - tref_k) - temperature_k * math.log(temperature_k / tref_k)
    )


def _solve_line_compound_liquidus_temperature(
    x_liquid_b: float,
    mu_a_ex_j_mol: float,
    mu_b_ex_j_mol: float,
    *,
    x_compound_b: float,
    tm_a_k: float,
    tm_b_k: float,
    dhfus_a_kj_mol: float,
    dhfus_b_kj_mol: float,
    dcp_a_j_mol_k: float,
    dcp_b_j_mol_k: float,
    gform_ref_kj_mol: float,
    gform_tref_k: float,
    dcp_form_j_mol_k: float = 0.0,
) -> float:
    x_a = 1.0 - x_liquid_b
    x_b = x_liquid_b
    if x_a <= 0.0 or x_b <= 0.0:
        return math.nan

    def residual(temp: float) -> float:
        term_a = _fusion_gibbs_j_mol(
            temp,
            tm_k=tm_a_k,
            dhfus_kj_mol=dhfus_a_kj_mol,
            dcp_j_mol_k=dcp_a_j_mol_k,
        ) + R_J_MOLK * temp * math.log(max(x_a, 1.0e-15)) + mu_a_ex_j_mol
        term_b = _fusion_gibbs_j_mol(
            temp,
            tm_k=tm_b_k,
            dhfus_kj_mol=dhfus_b_kj_mol,
            dcp_j_mol_k=dcp_b_j_mol_k,
        ) + R_J_MOLK * temp * math.log(max(x_b, 1.0e-15)) + mu_b_ex_j_mol
        gform = _line_compound_gform_j_mol(
            temp,
            gform_ref_kj_mol=gform_ref_kj_mol,
            tref_k=gform_tref_k,
            dcp_form_j_mol_k=dcp_form_j_mol_k,
        )
        return (1.0 - x_compound_b) * term_a + x_compound_b * term_b - gform

    low = 1.0
    high = max(tm_a_k, tm_b_k) * 1.5 + 100.0
    f_low = residual(low)
    f_high = residual(high)
    expand = 0
    while math.isfinite(f_low) and math.isfinite(f_high) and f_low * f_high > 0.0 and expand < 12:
        high *= 1.5
        f_high = residual(high)
        expand += 1
    if not (math.isfinite(f_low) and math.isfinite(f_high)) or f_low * f_high > 0.0:
        return math.nan
    for _ in range(100):
        mid = 0.5 * (low + high)
        f_mid = residual(mid)
        if not math.isfinite(f_mid):
            return math.nan
        if abs(f_mid) < 1.0e-8 or abs(high - low) < 1.0e-8:
            return mid
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return 0.5 * (low + high)


def _parse_line_compounds(specs: list[str] | None, *, default_tref_k: float) -> list[dict[str, Any]]:
    compounds: list[dict[str, Any]] = []
    for spec in specs or []:
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) not in {3, 4, 5}:
            raise ValueError("--line-compound expects label:x_B:gform_kJ_mol[:dCp_form_J_mol_K[:tref_K]].")
        label = parts[0]
        x_b = float(parts[1])
        gform = float(parts[2])
        dcp = float(parts[3]) if len(parts) >= 4 and parts[3] else 0.0
        tref = float(parts[4]) if len(parts) >= 5 and parts[4] else default_tref_k
        if not label:
            raise ValueError("Line compound label cannot be blank.")
        if x_b <= 0.0 or x_b >= 1.0:
            raise ValueError("Line compound x_B must be inside (0, 1).")
        compounds.append(
            {
                "label": label,
                "x_B": x_b,
                "gform_ref_kJ_mol": gform,
                "dCp_form_J_mol_K": dcp,
                "tref_K": tref,
            }
        )
    return compounds


def _binary_liquidus_from_gex_curve(
    points_kJ_mol: list[tuple[float, float]],
    *,
    tm_a_k: float,
    tm_b_k: float,
    dhfus_a_kj_mol: float,
    dhfus_b_kj_mol: float,
    dcp_a_j_mol_k: float = 0.0,
    dcp_b_j_mol_k: float = 0.0,
    line_compounds: list[dict[str, Any]] | None = None,
    x_min: float = 1.0e-6,
    x_max: float = 1.0 - 1.0e-6,
) -> list[dict[str, float]]:
    """Predict terminal liquidus branches from tabulated binary Gex/Hmix.

    The binary coordinate is x_B. ``points_kJ_mol`` are treated as a
    temperature-independent excess Gibbs proxy. For a binary molar excess
    function g(x_B), the excess chemical potentials are:
      mu_A^ex = g - x_B dg/dx_B
      mu_B^ex = g + (1 - x_B) dg/dx_B.
    """
    rows: list[dict[str, float]] = []
    for x_b, g_kj_mol, dgdx_kj_mol in _curve_derivatives(points_kJ_mol):
        if x_b <= x_min or x_b >= x_max:
            continue
        x_a = 1.0 - x_b
        g_j_mol = g_kj_mol * 1000.0
        dgdx_j_mol = dgdx_kj_mol * 1000.0
        mu_a_ex = g_j_mol - x_b * dgdx_j_mol
        mu_b_ex = g_j_mol + x_a * dgdx_j_mol
        t_a = _solve_liquidus_temperature(
            x_a,
            mu_a_ex,
            tm_k=tm_a_k,
            dhfus_kj_mol=dhfus_a_kj_mol,
            dcp_j_mol_k=dcp_a_j_mol_k,
        )
        t_b = _solve_liquidus_temperature(
            x_b,
            mu_b_ex,
            tm_k=tm_b_k,
            dhfus_kj_mol=dhfus_b_kj_mol,
            dcp_j_mol_k=dcp_b_j_mol_k,
        )
        compound_temperatures: dict[str, float] = {}
        for compound in line_compounds or []:
            t_c = _solve_line_compound_liquidus_temperature(
                x_b,
                mu_a_ex,
                mu_b_ex,
                x_compound_b=float(compound["x_B"]),
                tm_a_k=tm_a_k,
                tm_b_k=tm_b_k,
                dhfus_a_kj_mol=dhfus_a_kj_mol,
                dhfus_b_kj_mol=dhfus_b_kj_mol,
                dcp_a_j_mol_k=dcp_a_j_mol_k,
                dcp_b_j_mol_k=dcp_b_j_mol_k,
                gform_ref_kj_mol=float(compound["gform_ref_kJ_mol"]),
                gform_tref_k=float(compound["tref_K"]),
                dcp_form_j_mol_k=float(compound["dCp_form_J_mol_K"]),
            )
            compound_temperatures[f"line_compound_{compound['label']}_K"] = t_c
        candidates = [t_a, t_b, *compound_temperatures.values()]
        finite_candidates = [value for value in candidates if math.isfinite(value)]
        liquidus = max(finite_candidates) if finite_candidates else math.nan
        rows.append(
            {
                "x": x_b,
                "Gex_kJ_mol": g_kj_mol,
                "dGex_dx_kJ_mol": dgdx_kj_mol,
                "mu_A_ex_J_mol": mu_a_ex,
                "mu_B_ex_J_mol": mu_b_ex,
                "liquidus_A_K": t_a,
                "liquidus_B_K": t_b,
                "liquidus_K": liquidus,
                "dCp_A_liq_minus_solid_J_mol_K": dcp_a_j_mol_k,
                "dCp_B_liq_minus_solid_J_mol_K": dcp_b_j_mol_k,
                **compound_temperatures,
            }
        )
    return rows


def _predicted_eutectic(liquidus_rows: list[dict[str, float]]) -> dict[str, float]:
    finite_rows = [
        row
        for row in liquidus_rows
        if math.isfinite(row.get("liquidus_K", math.nan))
        and math.isfinite(row.get("liquidus_A_K", math.nan))
        and math.isfinite(row.get("liquidus_B_K", math.nan))
    ]
    if not finite_rows:
        return {"x": math.nan, "T_K": math.nan, "branch_gap_K": math.nan}
    best = min(finite_rows, key=lambda row: row["liquidus_K"])
    return {
        "x": best["x"],
        "T_K": best["liquidus_K"],
        "branch_gap_K": abs(best["liquidus_A_K"] - best["liquidus_B_K"]),
    }


def _softmax_from_chi2(rows: list[dict[str, Any]]) -> list[float]:
    if not rows:
        return []
    logs = []
    for row in rows:
        chi2 = finite_float(row.get("chi2_total"))
        logs.append(-0.5 * chi2 if chi2 is not None else -math.inf)
    max_log = max(logs)
    if not math.isfinite(max_log):
        return [1.0 / len(rows)] * len(rows)
    raw = [math.exp(value - max_log) for value in logs]
    total = sum(raw)
    return [value / total for value in raw] if total > 0.0 else [1.0 / len(raw)] * len(raw)


def _weighted_quantile(values: list[tuple[float, float]], quantile: float) -> float:
    finite = sorted((value, weight) for value, weight in values if math.isfinite(value) and weight > 0.0)
    if not finite:
        return math.nan
    total = sum(weight for _, weight in finite)
    threshold = quantile * total
    accum = 0.0
    for value, weight in finite:
        accum += weight
        if accum >= threshold:
            return value
    return finite[-1][0]


def _numeric_grid_spec(spec: str | None, *, default: float) -> list[float]:
    if spec is None or str(spec).strip() == "":
        return [float(default)]
    parts = [float(part.strip()) for part in str(spec).split(",") if part.strip()]
    if len(parts) == 1:
        return [parts[0]]
    if len(parts) != 3:
        raise ValueError("Grid spec expects one value or min,max,step.")
    start, stop, step = parts
    if step <= 0.0 or start > stop:
        raise ValueError("Grid spec has invalid min/max/step.")
    values: list[float] = []
    current = start
    while current <= stop + 1.0e-10:
        values.append(round(current, 12))
        current += step
    if not values or abs(values[-1] - stop) > 1.0e-10:
        values.append(stop)
    return values


def _write_benchmark_phase_plot(
    path: Path,
    *,
    diagram_rows: list[dict[str, Any]],
    envelope_rows: list[dict[str, Any]],
    benchmark: dict[str, float],
    title: str,
    x_label: str,
    t_min: float | None = None,
    t_max: float | None = None,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in diagram_rows:
        by_label[str(row["label"])].append(row)
    for label, rows in by_label.items():
        rows = sorted(rows, key=lambda item: float(item["x"]))
        filtered = [
            row
            for row in rows
            if math.isfinite(float(row["liquidus_K"]))
            and (t_min is None or float(row["liquidus_K"]) >= t_min)
            and (t_max is None or float(row["liquidus_K"]) <= t_max)
        ]
        if not filtered:
            continue
        xs = [float(row["x"]) for row in filtered]
        ys = [float(row["liquidus_K"]) for row in filtered]
        ax.plot(xs, ys, linewidth=1.0, alpha=0.35, label=label)
    if envelope_rows:
        filtered_env = [
            row
            for row in envelope_rows
            if math.isfinite(float(row["liquidus_p50_K"]))
            and (t_min is None or float(row["liquidus_p50_K"]) >= t_min)
            and (t_max is None or float(row["liquidus_p50_K"]) <= t_max)
        ]
        xs = [float(row["x"]) for row in filtered_env]
        p05 = [float(row["liquidus_p05_K"]) for row in filtered_env]
        p50 = [float(row["liquidus_p50_K"]) for row in filtered_env]
        p95 = [float(row["liquidus_p95_K"]) for row in filtered_env]
        if xs:
            ax.fill_between(xs, p05, p95, color="#1f77b4", alpha=0.18, label="posterior 5-95%")
            ax.plot(xs, p50, color="#1f77b4", linewidth=2.4, label="posterior median")
    if math.isfinite(benchmark.get("x", math.nan)) and math.isfinite(benchmark.get("T_K", math.nan)):
        ax.scatter(
            [benchmark["x"]],
            [benchmark["T_K"]],
            marker="*",
            s=180,
            color="#d62728",
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
            label="benchmark eutectic",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Temperature (K)")
    ax.set_title(title)
    if t_min is not None or t_max is not None:
        ax.set_ylim(bottom=t_min, top=t_max)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def write_benchmarked_uq_phase(args: argparse.Namespace) -> dict[str, Any]:
    curve_columns = parse_csv_list(args.curve_columns)
    if not curve_columns:
        raise ValueError("--curve-columns must name at least one candidate curve column.")
    labels = parse_csv_list(args.curve_labels)
    curves = _read_curve_family(
        args.curve_csv,
        x_column=args.x_column,
        curve_columns=curve_columns,
        y_unit=args.curve_y_unit,
        labels=labels,
    )
    if args.extra_curve_csv:
        for spec in args.extra_curve_csv:
            parts = [part.strip() for part in spec.split(":")]
            if len(parts) not in {3, 4}:
                raise ValueError("--extra-curve-csv expects path:x_column:y_column[:label].")
            path = Path(parts[0])
            label = parts[3] if len(parts) == 4 else parts[2]
            curves.extend(
                _read_curve_family(
                    path,
                    x_column=parts[1],
                    curve_columns=[parts[2]],
                    y_unit=args.curve_y_unit,
                    labels=[label],
                )
            )
    literature_points = (
        _read_xy_points(
            args.literature_csv,
            x_column=args.literature_x_column,
            y_column=args.literature_y_column,
            y_unit=args.literature_y_unit,
        )
        if args.literature_csv
        else []
    )
    outdir = args.outdir.resolve()
    diagram_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    per_curve_liquidus: dict[str, list[dict[str, float]]] = {}
    dcp_a_values = _numeric_grid_spec(args.dcp_a_grid, default=args.dcp_a)
    dcp_b_values = _numeric_grid_spec(args.dcp_b_grid, default=args.dcp_b)
    line_compounds = _parse_line_compounds(args.line_compound, default_tref_k=args.eutectic_t)
    for curve in curves:
        for dcp_a in dcp_a_values:
            for dcp_b in dcp_b_values:
                points = curve["points_kJ_mol"]
                liquidus = _binary_liquidus_from_gex_curve(
                    points,
                    tm_a_k=args.tm_a,
                    tm_b_k=args.tm_b,
                    dhfus_a_kj_mol=args.dhfus_a,
                    dhfus_b_kj_mol=args.dhfus_b,
                    dcp_a_j_mol_k=dcp_a,
                    dcp_b_j_mol_k=dcp_b,
                    line_compounds=line_compounds,
                )
                base_label = str(curve["label"])
                label = base_label
                if len(dcp_a_values) > 1 or len(dcp_b_values) > 1:
                    label = f"{base_label} dCpA={dcp_a:g} dCpB={dcp_b:g}"
                per_curve_liquidus[label] = liquidus
                for row in liquidus:
                    out = dict(row)
                    out["label"] = label
                    out["base_label"] = base_label
                    diagram_rows.append(out)
                eutectic = _predicted_eutectic(liquidus)
                model_points = [(float(x), float(y)) for x, y in points]
                hmix_metric = comparison_metrics(model_points, literature_points, reference_label=args.literature_label)
                rmse = finite_float(hmix_metric.get("rmse_kJ_mol"))
                chi2_hmix = (rmse / args.sigma_hmix) ** 2 if rmse is not None and args.sigma_hmix > 0 else 0.0
                if math.isfinite(eutectic["x"]) and math.isfinite(eutectic["T_K"]):
                    chi2_x = (
                        ((eutectic["x"] - args.eutectic_x) / args.sigma_eutectic_x) ** 2
                        if args.sigma_eutectic_x > 0
                        else 0.0
                    )
                    chi2_t = (
                        ((eutectic["T_K"] - args.eutectic_t) / args.sigma_eutectic_t) ** 2
                        if args.sigma_eutectic_t > 0
                        else 0.0
                    )
                    chi2_total = chi2_hmix + chi2_x + chi2_t
                else:
                    chi2_x = math.inf
                    chi2_t = math.inf
                    chi2_total = math.inf
                min_x, min_h = _curve_minimum(points)
                weight_rows.append(
                    {
                        "label": label,
                        "base_label": base_label,
                        "source": curve["source"],
                        "y_column": curve["y_column"],
                        "dCp_A_liq_minus_solid_J_mol_K": dcp_a,
                        "dCp_B_liq_minus_solid_J_mol_K": dcp_b,
                        "hmix_min_x": min_x,
                        "hmix_min_kJ_mol": min_h,
                        "hmix_rmse_kJ_mol": rmse,
                        "eutectic_x": eutectic["x"],
                        "eutectic_T_K": eutectic["T_K"],
                        "eutectic_branch_gap_K": eutectic["branch_gap_K"],
                        "chi2_hmix": chi2_hmix,
                        "chi2_eutectic_x": chi2_x,
                        "chi2_eutectic_T": chi2_t,
                        "chi2_total": chi2_total,
                    }
                )
    weights = _softmax_from_chi2(weight_rows)
    for row, weight in zip(weight_rows, weights):
        row["posterior_weight"] = weight
    envelope_rows: list[dict[str, Any]] = []
    x_values = sorted({round(float(row["x"]), 12) for row in diagram_rows})
    weight_by_label = {str(row["label"]): float(row["posterior_weight"]) for row in weight_rows}
    for x_value in x_values:
        samples: list[tuple[float, float]] = []
        for label, rows in per_curve_liquidus.items():
            point = _interp_linear([(float(row["x"]), float(row["liquidus_K"])) for row in rows], x_value)
            if point is not None:
                samples.append((point, weight_by_label.get(label, 0.0)))
        if samples:
            envelope_rows.append(
                {
                    "x": x_value,
                    "liquidus_p05_K": _weighted_quantile(samples, 0.05),
                    "liquidus_p50_K": _weighted_quantile(samples, 0.50),
                    "liquidus_p95_K": _weighted_quantile(samples, 0.95),
                }
            )
    weight_csv = outdir / "posterior_model_weights.csv"
    write_csv(
        weight_csv,
        weight_rows,
        [
            "label",
            "base_label",
            "source",
            "y_column",
            "dCp_A_liq_minus_solid_J_mol_K",
            "dCp_B_liq_minus_solid_J_mol_K",
            "hmix_min_x",
            "hmix_min_kJ_mol",
            "hmix_rmse_kJ_mol",
            "eutectic_x",
            "eutectic_T_K",
            "eutectic_branch_gap_K",
            "chi2_hmix",
            "chi2_eutectic_x",
            "chi2_eutectic_T",
            "chi2_total",
            "posterior_weight",
        ],
    )
    diagram_csv = outdir / "candidate_phase_diagrams.csv"
    diagram_fields = [
        "label",
        "base_label",
        "x",
        "Gex_kJ_mol",
        "dGex_dx_kJ_mol",
        "mu_A_ex_J_mol",
        "mu_B_ex_J_mol",
        "liquidus_A_K",
        "liquidus_B_K",
        "liquidus_K",
        "dCp_A_liq_minus_solid_J_mol_K",
        "dCp_B_liq_minus_solid_J_mol_K",
    ]
    for compound in line_compounds:
        diagram_fields.append(f"line_compound_{compound['label']}_K")
    write_csv(
        diagram_csv,
        diagram_rows,
        diagram_fields,
    )
    envelope_csv = outdir / "posterior_phase_envelope.csv"
    write_csv(envelope_csv, envelope_rows, ["x", "liquidus_p05_K", "liquidus_p50_K", "liquidus_p95_K"])
    plot = _write_benchmark_phase_plot(
        outdir / "uq_benchmarked_phase_diagram.png",
        diagram_rows=diagram_rows,
        envelope_rows=envelope_rows,
        benchmark={"x": args.eutectic_x, "T_K": args.eutectic_t},
        title=args.title or f"{args.component_a}-{args.component_b} UQ-benchmarked liquidus",
        x_label=f"x({args.x_component})",
        t_min=args.plot_t_min,
        t_max=args.plot_t_max,
    )
    best_hmix = min(weight_rows, key=lambda row: float(row["chi2_hmix"])) if weight_rows else {}
    best_eutectic = min(
        weight_rows,
        key=lambda row: float(row["chi2_eutectic_x"]) + float(row["chi2_eutectic_T"]),
    ) if weight_rows else {}
    best_joint = max(weight_rows, key=lambda row: float(row["posterior_weight"])) if weight_rows else {}
    tension = bool(best_hmix and best_eutectic and best_hmix.get("label") != best_eutectic.get("label"))
    branch_note = (
        "Note: this diagnostic liquidus model includes terminal solid-liquid branches and "
        f"{len(line_compounds)} specified line-compound branch(es). The line-compound formation "
        "terms are simplified pseudo-binary diagnostics, not a replacement for assessed "
        "compound Gibbs functions."
        if line_compounds
        else "Note: this diagnostic liquidus model includes only terminal solid-liquid branches."
    )
    report = outdir / "posterior_tension_report.md"
    report.write_text(
        textwrap.dedent(
            f"""\
            # {args.component_a}-{args.component_b} UQ-Benchmarked Mixing/Phase Report

This route weights candidate liquid excess-Gibbs curves against both mixing-enthalpy data
and a eutectic benchmark. It is a fast diagnostic layer before a full pycalphad refit.

            Benchmark eutectic: x({args.x_component}) = {args.eutectic_x:g}, T = {args.eutectic_t:g} K.
            Hmix sigma = {args.sigma_hmix:g} kJ/mol; eutectic sigmas = {args.sigma_eutectic_x:g} in x and {args.sigma_eutectic_t:g} K.

            Best Hmix candidate: {best_hmix.get('label', '')}
            Best eutectic candidate: {best_eutectic.get('label', '')}
            Best joint posterior candidate: {best_joint.get('label', '')}

            Posterior tension flag: {tension}

            Interpretation rule: if no candidate has simultaneously low Hmix and eutectic chi-square,
            the model family is under strain. That can indicate missing liquid excess entropy/T-dependence,
wrong Hmix basis conversion, inconsistent pure/fusion anchors, or incorrect solid/intermediate
compound Gibbs functions. This is useful information, not merely a failed fit.

{branch_note}
If a real or missing intermediate line compound participates in the invariant reaction,
its Gibbs energy can lift, split, or reshape the eutectic/peritectic region. In that case,
failure to match the eutectic does not uniquely implicate the liquid Hmix/Cp model.
"""
        ),
        encoding="utf-8",
    )
    metadata = {
        "schema": BENCHMARK_UQ_PHASE_SCHEMA,
        "component_a": args.component_a,
        "component_b": args.component_b,
        "x_component": args.x_component,
        "curve_csv": str(args.curve_csv.resolve()),
        "curve_columns": curve_columns,
        "literature_csv": str(args.literature_csv.resolve()) if args.literature_csv else "",
        "benchmark": {"eutectic_x": args.eutectic_x, "eutectic_T_K": args.eutectic_t},
        "pure_fusion_anchors": {
            "component_a": {"Tm_K": args.tm_a, "dHfus_kJ_mol": args.dhfus_a},
            "component_b": {"Tm_K": args.tm_b, "dHfus_kJ_mol": args.dhfus_b},
        },
        "dCp_grid_J_mol_K": {"component_a": dcp_a_values, "component_b": dcp_b_values},
        "line_compounds": line_compounds,
        "best_hmix_label": best_hmix.get("label", ""),
        "best_eutectic_label": best_eutectic.get("label", ""),
        "best_joint_label": best_joint.get("label", ""),
        "posterior_tension": tension,
        "outputs": {
            "posterior_model_weights": str(weight_csv),
            "candidate_phase_diagrams": str(diagram_csv),
            "posterior_phase_envelope": str(envelope_csv),
            "phase_plot": plot or "",
            "tension_report": str(report),
        },
    }
    write_json(outdir / "benchmark_uq_phase_metadata.json", metadata)
    return metadata


def write_comparison_plot(
    path: Path,
    *,
    model_points: list[tuple[float, float]],
    literature_points: list[tuple[float, float]],
    mqmqa_points: list[tuple[float, float]],
    title: str,
    model_label: str,
    literature_label: str,
    mqmqa_label: str,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if model_points:
        xs, ys = zip(*model_points)
        ax.plot(xs, ys, color="#1f77b4", linewidth=2.0, label=model_label)
    if mqmqa_points:
        xs, ys = zip(*mqmqa_points)
        ax.plot(xs, ys, color="#666666", linewidth=1.8, linestyle="--", label=mqmqa_label)
    if literature_points:
        xs, ys = zip(*literature_points)
        ax.scatter(xs, ys, color="#f39c12", edgecolor="black", linewidth=0.3, label=literature_label, zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Mole fraction")
    ax.set_ylabel("Excess/mixing enthalpy (kJ/mol)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def write_binary_comparison(args: argparse.Namespace) -> dict[str, Any]:
    params = load_parameters(args.params)
    if args.x_component not in params.components:
        raise ValueError("--x-component must be present in the MIVM parameter file.")
    rows = compare_binary_model(
        params,
        temperature_k=args.temperature,
        binary_grid_spec=args.binary_grid,
        x_component=args.x_component,
    )
    outdir = args.outdir.resolve()
    table = outdir / "mivm_binary_comparison.csv"
    write_csv(
        table,
        rows,
        ["T_K", "x_component", "x", "composition", "H_excess_MIVM_kJ_mol", "G_excess_MIVM_kJ_mol"],
    )
    model_points = [(float(row["x"]), float(row["H_excess_MIVM_kJ_mol"])) for row in rows]
    literature_points: list[tuple[float, float]] = []
    mqmqa_points: list[tuple[float, float]] = []
    metrics: list[dict[str, Any]] = []
    if args.literature_csv:
        literature_points = _read_xy_points(
            args.literature_csv,
            x_column=args.literature_x_column,
            y_column=args.literature_y_column,
            y_unit=args.literature_y_unit,
        )
        metrics.append(comparison_metrics(model_points, literature_points, reference_label=args.literature_label))
    if args.mqmqa_csv:
        mqmqa_points = _read_xy_points(
            args.mqmqa_csv,
            x_column=args.mqmqa_x_column,
            y_column=args.mqmqa_y_column,
            y_unit=args.mqmqa_y_unit,
        )
        metrics.append(comparison_metrics(mqmqa_points, literature_points, reference_label=args.mqmqa_label))
    metrics_table = outdir / "mivm_binary_comparison_metrics.csv"
    write_csv(
        metrics_table,
        metrics,
        ["reference", "matched_points", "rmse_kJ_mol", "mae_kJ_mol", "mean_bias_kJ_mol"],
    )
    tdb_sanity = tdb_sanity_check(args.tdb_sanity) if args.tdb_sanity else None
    plot = write_comparison_plot(
        outdir / "mivm_binary_comparison.png",
        model_points=model_points,
        literature_points=literature_points,
        mqmqa_points=mqmqa_points,
        title=args.title or f"MIVM binary comparison at {args.temperature:g} K",
        model_label=args.model_label,
        literature_label=args.literature_label,
        mqmqa_label=args.mqmqa_label,
    )
    metadata = {
        "schema": "atomi.calphad.mivm.binary_comparison.v1",
        "parameters": str(args.params.resolve()),
        "temperature_K": args.temperature,
        "binary_grid": args.binary_grid,
        "x_component": args.x_component,
        "outputs": {
            "comparison_table": str(table),
            "metrics_table": str(metrics_table),
            "plot": plot or "",
        },
        "metrics": metrics,
        "tdb_sanity": tdb_sanity,
    }
    write_json(outdir / "mivm_binary_comparison_metadata.json", metadata)
    return metadata


def build_pycalphad_bridge_text() -> str:
    return '''"""Generated Atomi MIVM/pycalphad bridge.

Usage:
    from pycalphad import Database, equilibrium, variables as v
    from mivm_pycalphad_bridge import MIVMModel, MIVM_PARAMETERS

    dbf = Database("base_references.tdb")
    models = {MIVM_PARAMETERS.phase: MIVMModel}
    ds = equilibrium(dbf, comps, [MIVM_PARAMETERS.phase], conditions, model=models)
"""

from pathlib import Path

from atomi.calphad.mivm import load_parameters, make_pycalphad_model_class


MIVM_PARAMETERS = load_parameters(Path(__file__).with_name("mivm_parameters.json"))
MIVMModel = make_pycalphad_model_class(MIVM_PARAMETERS)
models = {MIVM_PARAMETERS.phase: MIVMModel}
'''


def make_pycalphad_model_class(params: MIVMParameters):
    """Return a pycalphad Model subclass with MIVM as an extra Gibbs contribution.

    pycalphad is an optional dependency. The returned model uses the parameter JSON
    for B_ji, Vm_i, and Z_i, while the database still supplies phase constituents
    and reference/endmember terms.
    """
    try:
        from pycalphad import Model, variables as v
        from symengine import S, exp, log
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("pycalphad and symengine are required to build the MIVM Model class.") from exc

    class AtomiMIVMModel(Model):  # pragma: no cover - exercised only with optional pycalphad
        contributions = [
            ("ref", "reference_energy"),
            ("idmix", "ideal_mixing_energy"),
            ("xsmix", "excess_mixing_energy"),
            ("mivm", "mivm_excess_energy"),
            ("mag", "magnetic_energy"),
            ("2st", "twostate_energy"),
            ("ein", "einstein_energy"),
            ("vol", "volume_energy"),
            ("ord", "atomic_ordering_energy"),
        ]

        def _sym_pair_ln_b(self, source: str, target: str):
            pair = params.pairs.get((source, target))
            if pair is None:
                raise ValueError(f"Missing directed MIVM pair parameter from={source!r} to={target!r}.")
            expr = S.Zero
            if pair.b is not None:
                if pair.b <= 0:
                    raise ValueError(f"Pair B for {source}->{target} must be positive.")
                expr += log(S(pair.b))
            if pair.ln_b is not None:
                expr += S(pair.ln_b)
            if pair.ln_b_a is not None:
                expr += S(pair.ln_b_a)
            if pair.ln_b_b_over_t is not None:
                expr += S(pair.ln_b_b_over_t) / v.T
            if pair.delta_g_j_mol is not None:
                expr += -S(pair.delta_g_j_mol) / (v.R * v.T)
            return expr

        def mivm_excess_energy(self, dbe):
            phase = dbe.phases[self.phase_name]
            if len(phase.constituents) != 1:
                raise ValueError("Atomi MIVM pycalphad bridge currently expects a one-sublattice phase.")
            names = params.component_names
            sitefracs = {name: v.SiteFraction(self.phase_name, 0, name) for name in names}
            size_term = S.Zero
            pair_term = S.Zero
            for target in names:
                xi = sitefracs[target]
                comp_i = params.components[target]
                volume_denom = S.Zero
                b_denom = S.Zero
                b_log_numer = S.Zero
                for source in names:
                    xj = sitefracs[source]
                    comp_j = params.components[source]
                    ln_b = self._sym_pair_ln_b(source, target)
                    b_val = exp(ln_b)
                    volume_denom += xj * S(comp_j.molar_volume) * b_val
                    b_denom += xj * b_val
                    b_log_numer += xj * b_val * ln_b
                size_term += xi * log(S(comp_i.molar_volume) / volume_denom)
                pair_term += -S(0.5) * S(comp_i.coordination) * xi * (b_log_numer / b_denom)
            return v.R * v.T * (size_term + pair_term)

    AtomiMIVMModel.__name__ = f"AtomiMIVMModel_{params.phase}"
    return AtomiMIVMModel


def write_pycalphad_bridge(params_path: Path, outdir: Path) -> dict[str, Any]:
    params = load_parameters(params_path)
    outdir.mkdir(parents=True, exist_ok=True)
    copied_params = outdir / "mivm_parameters.json"
    shutil.copyfile(params_path, copied_params)
    bridge = outdir / "mivm_pycalphad_bridge.py"
    bridge.write_text(build_pycalphad_bridge_text(), encoding="utf-8")
    readme = outdir / "README_mivm_pycalphad_bridge.md"
    readme.write_text(
        textwrap.dedent(
            f"""\
            # Atomi MIVM pycalphad bridge

            Phase: `{params.phase}`
            Components: `{", ".join(params.component_names)}`

            This folder contains:
            - `mivm_parameters.json`: single source of truth for Vm, Z, and directed B_ji parameters.
            - `mivm_pycalphad_bridge.py`: imports Atomi and exposes `MIVMModel` plus `models`.

            Use the model with a pycalphad database that defines the same phase and constituents.
            The database should supply reference/endmember Gibbs energies; the generated custom
            model adds the MIVM excess Gibbs contribution to pycalphad's normal GM expression.
            """
        ),
        encoding="utf-8",
    )
    metadata = {
        "schema": "atomi.calphad.mivm.pycalphad_bridge.v1",
        "phase": params.phase,
        "components": params.component_names,
        "outputs": {
            "parameters": str(copied_params),
            "bridge": str(bridge),
            "readme": str(readme),
        },
    }
    write_json(outdir / "mivm_pycalphad_bridge_metadata.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calphad-mivm",
        description="Evaluate MIVM parameters and write Atomi/pycalphad bridge artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(MIVM_HELP_EPILOG),
    )
    subparsers = parser.add_subparsers(dest="command")
    guide = subparsers.add_parser(
        "guide",
        help="Print the MIVM parameter guide.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(MIVM_HELP_EPILOG),
    )
    guide.add_argument("--system", choices=("all", "molten", "ceramic", "solid"), default="all")
    guide.add_argument("--format", choices=("text", "json"), default="text")

    template = subparsers.add_parser("template", help="Write a starter MIVM parameter JSON.")
    template.add_argument("--system", choices=("molten", "ceramic", "solid"), default="ceramic")
    template.add_argument("--out", type=Path, default=Path("mivm_parameters.json"))

    validate = subparsers.add_parser("validate", help="Validate a MIVM parameter JSON.")
    validate.add_argument("--params", type=Path, required=True)

    sample = subparsers.add_parser("sample", help="Sample MIVM G/H/activity tables.")
    sample.add_argument("--params", type=Path, required=True)
    sample.add_argument("--outdir", type=Path, default=Path("analysis/mivm_sample"))
    sample.add_argument("--temperature", type=float, action="append", help="Temperature in K. Repeatable.")
    sample.add_argument("--T-min", type=float)
    sample.add_argument("--T-max", type=float)
    sample.add_argument("--T-step", type=float, default=100.0)
    sample.add_argument("--composition", action="append", help="Composition like A=0.25,B=0.75. Repeatable.")
    sample.add_argument("--binary-grid", help="Composition grid: A,B,step or A,B,xmin,xmax,step.")

    compare = subparsers.add_parser(
        "compare-binary",
        help="Compare a MIVM binary mixing curve with literature and optional MQMQA/MSTDB CSV data.",
    )
    compare.add_argument("--params", type=Path, required=True)
    compare.add_argument("--outdir", type=Path, default=Path("analysis/mivm_binary_comparison"))
    compare.add_argument("--temperature", type=float, required=True, help="Temperature in K.")
    compare.add_argument("--binary-grid", required=True, help="Composition grid: A,B,step or A,B,xmin,xmax,step.")
    compare.add_argument("--x-component", required=True, help="Component to use as the x-axis.")
    compare.add_argument("--model-label", default="MIVM")
    compare.add_argument("--title")
    compare.add_argument("--literature-csv", type=Path)
    compare.add_argument("--literature-x-column", default="x_UCl3")
    compare.add_argument("--literature-y-column", default="Hmix_kJ_mol")
    compare.add_argument("--literature-y-unit", choices=("kJ/mol", "J/mol"), default="kJ/mol")
    compare.add_argument("--literature-label", default="literature")
    compare.add_argument("--mqmqa-csv", type=Path, help="Optional pycalphad/MQMQA curve CSV to overlay.")
    compare.add_argument("--mqmqa-x-column", default="x")
    compare.add_argument("--mqmqa-y-column", default="Hmix_kJ_mol")
    compare.add_argument("--mqmqa-y-unit", choices=("kJ/mol", "J/mol"), default="kJ/mol")
    compare.add_argument("--mqmqa-label", default="MQMQA/MSTDB")
    compare.add_argument("--tdb-sanity", type=Path, help="Optional TDB/MSTDB file to sanity-check for pycalphad use.")

    benchmark = subparsers.add_parser(
        "benchmark-uq-phase",
        help=(
            "Weight UQ liquid Hmix/Gex curves by mixing-enthalpy and eutectic benchmarks, "
            "then write a fast posterior liquidus envelope."
        ),
    )
    benchmark.add_argument("--curve-csv", type=Path, required=True, help="CSV containing candidate Hmix/Gex curves.")
    benchmark.add_argument("--x-column", required=True, help="Composition column for x(component B).")
    benchmark.add_argument("--curve-columns", required=True, help="Comma-separated Hmix/Gex candidate columns.")
    benchmark.add_argument("--curve-labels", help="Optional comma-separated labels matching --curve-columns.")
    benchmark.add_argument("--curve-y-unit", choices=("kJ/mol", "J/mol"), default="kJ/mol")
    benchmark.add_argument(
        "--extra-curve-csv",
        action="append",
        help="Optional extra candidate as path:x_column:y_column[:label]. Repeatable.",
    )
    benchmark.add_argument("--literature-csv", type=Path, help="Optional Hmix benchmark CSV.")
    benchmark.add_argument("--literature-x-column", default="x_UCl3")
    benchmark.add_argument("--literature-y-column", default="Hmix_kJ_mol")
    benchmark.add_argument("--literature-y-unit", choices=("kJ/mol", "J/mol"), default="kJ/mol")
    benchmark.add_argument("--literature-label", default="literature Hmix")
    benchmark.add_argument("--component-a", required=True, help="Low-x terminal component, e.g. NaCl.")
    benchmark.add_argument("--component-b", required=True, help="High-x terminal component, e.g. UCl3.")
    benchmark.add_argument("--x-component", required=True, help="Name shown on the x-axis, usually component B.")
    benchmark.add_argument("--tm-a", type=float, required=True, help="Pure component A melting point in K.")
    benchmark.add_argument("--tm-b", type=float, required=True, help="Pure component B melting point in K.")
    benchmark.add_argument("--dhfus-a", type=float, required=True, help="Pure component A fusion enthalpy in kJ/mol.")
    benchmark.add_argument("--dhfus-b", type=float, required=True, help="Pure component B fusion enthalpy in kJ/mol.")
    benchmark.add_argument(
        "--dcp-a",
        type=float,
        default=0.0,
        help="Component A fusion heat-capacity jump Cp_liquid-Cp_solid in J/mol/K.",
    )
    benchmark.add_argument(
        "--dcp-b",
        type=float,
        default=0.0,
        help="Component B fusion heat-capacity jump Cp_liquid-Cp_solid in J/mol/K.",
    )
    benchmark.add_argument(
        "--dcp-a-grid",
        help="Optional component A dCp grid in J/mol/K: value or min,max,step. Overrides --dcp-a.",
    )
    benchmark.add_argument(
        "--dcp-b-grid",
        help="Optional component B dCp grid in J/mol/K: value or min,max,step. Overrides --dcp-b.",
    )
    benchmark.add_argument("--eutectic-x", type=float, required=True, help="Experimental/assessed eutectic x(component B).")
    benchmark.add_argument("--eutectic-t", type=float, required=True, help="Experimental/assessed eutectic temperature in K.")
    benchmark.add_argument("--sigma-hmix", type=float, default=0.5, help="Hmix likelihood sigma in kJ/mol RMSE units.")
    benchmark.add_argument("--sigma-eutectic-x", type=float, default=0.02, help="Eutectic composition likelihood sigma.")
    benchmark.add_argument("--sigma-eutectic-t", type=float, default=25.0, help="Eutectic temperature likelihood sigma in K.")
    benchmark.add_argument("--plot-t-min", type=float, help="Optional lower y-limit/filter for the phase plot in K.")
    benchmark.add_argument("--plot-t-max", type=float, help="Optional upper y-limit/filter for the phase plot in K.")
    benchmark.add_argument(
        "--line-compound",
        action="append",
        help=(
            "Optional line compound as label:x_component:gform_kJ_mol[:dCp_form_J_mol_K[:tref_K]]. "
            "Formation term is relative to terminal solids on the pseudo-binary basis."
        ),
    )
    benchmark.add_argument("--title")
    benchmark.add_argument("--outdir", type=Path, default=Path("analysis/mivm_benchmark_uq_phase"))

    sanitize = subparsers.add_parser(
        "mstdb-sanitize",
        help="Sanitize ChemSage/MSTDB charged-species labels for pycalphad equilibrium checks.",
    )
    sanitize.add_argument("--input", type=Path, required=True)
    sanitize.add_argument("--output", type=Path, required=True)
    sanitize.add_argument("--metadata", type=Path, help="Optional JSON metadata path.")

    mqmqa = subparsers.add_parser(
        "mqmqa-binary",
        help="Generate a pycalphad/MSTDB MQMQA binary mixing-enthalpy CSV for compare-binary overlays.",
    )
    mqmqa.add_argument("--tdb", type=Path, required=True, help="Sanitized pycalphad-readable MSTDB/ChemSage database.")
    mqmqa.add_argument("--phase", default="MSCL", help="MQMQA liquid phase, e.g. MSCL or MSFL.")
    mqmqa.add_argument("--component-a", required=True, help="Formula for endmember A, e.g. NaCl.")
    mqmqa.add_argument("--component-b", required=True, help="Formula for endmember B, e.g. UCl3.")
    mqmqa.add_argument("--x-component", required=True, help="Endmember to use as x-axis.")
    mqmqa.add_argument("--temperature", type=float, required=True, help="Temperature in K.")
    mqmqa.add_argument("--pressure", type=float, default=101325.0, help="Pressure in Pa.")
    mqmqa.add_argument("--grid", default="0.02,0.98,0.02", help="x grid: step or xmin,xmax,step.")
    mqmqa.add_argument("--endpoint-epsilon", type=float, default=1.0e-6)
    mqmqa.add_argument(
        "--dependent-element",
        help="Element to leave dependent in pycalphad composition conditions; defaults to an element unique to component A.",
    )
    mqmqa.add_argument("--outdir", type=Path, default=Path("analysis/mqmqa_binary"))

    bridge = subparsers.add_parser("pycalphad-bridge", help="Write a pycalphad custom-Model bridge module.")
    bridge.add_argument("--params", type=Path, required=True)
    bridge.add_argument("--outdir", type=Path, default=Path("analysis/mivm_pycalphad_bridge"))

    database = subparsers.add_parser("database", help="Inspect a subgrouped MIVM parameter database registry.")
    database.add_argument("--db", type=Path, required=True, help="Path to mivm_parameter_database.json.")
    database.add_argument(
        "action",
        choices=("summary", "list", "map", "targets", "validate-all"),
        help="Database inspection action.",
    )
    database.add_argument("--subgroup", help="Filter by subgroup id.")
    database.add_argument("--component", help="Filter by MIVM component name.")
    database.add_argument("--priority", help="Filter target systems by priority.")
    database.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "guide"
        args.system = "all"
        args.format = "text"
    if args.command == "guide":
        if args.format == "json":
            print(json.dumps(parameter_guide(), indent=2, sort_keys=True))
        else:
            print(text_guide(args.system), end="")
        return None
    if args.command == "template":
        system = "ceramic" if args.system == "solid" else args.system
        data = template_parameters(system)
        write_json(args.out, data)
        print(f"Wrote MIVM parameter template: {args.out}")
        return {"path": str(args.out)}
    if args.command == "validate":
        params = load_parameters(args.params)
        warnings = validate_parameters(params)
        print(f"MIVM parameter file: {args.params}")
        print(f"Phase: {params.phase}")
        print(f"Components: {', '.join(params.component_names)}")
        print(f"Directed pairs: {len(params.pairs)}")
        if warnings:
            for warning in warnings:
                print(f"WARNING: {warning}")
        else:
            print("MIVM parameter validation: PASS")
        return {"warnings": warnings}
    if args.command == "sample":
        params = load_parameters(args.params)
        warnings = validate_parameters(params)
        temperatures = temperature_grid(args)
        compositions = binary_grid(args.binary_grid, params) if args.binary_grid else parse_compositions(args.composition, params)
        rows = sample_rows(params, temperatures, compositions)
        outdir = args.outdir.resolve()
        table = outdir / "mivm_property_table.csv"
        write_csv(table, rows, sample_fields(params))
        metadata = {
            "schema": SAMPLE_SCHEMA,
            "parameters": str(args.params.resolve()),
            "phase": params.phase,
            "components": params.component_names,
            "n_temperatures": len(temperatures),
            "n_compositions": len(compositions),
            "n_rows": len(rows),
            "warnings": warnings,
            "outputs": {"property_table": str(table)},
        }
        write_json(outdir / "mivm_sample_metadata.json", metadata)
        print(f"Wrote MIVM property table: {table}")
        if warnings:
            for warning in warnings:
                print(f"WARNING: {warning}")
        return metadata
    if args.command == "compare-binary":
        metadata = write_binary_comparison(args)
        print(f"Wrote MIVM binary comparison: {metadata['outputs']['comparison_table']}")
        print(f"Wrote comparison metrics    : {metadata['outputs']['metrics_table']}")
        if metadata["outputs"].get("plot"):
            print(f"Wrote comparison plot       : {metadata['outputs']['plot']}")
        for metric in metadata["metrics"]:
            rmse = metric.get("rmse_kJ_mol")
            if rmse is not None:
                print(
                    f"{metric['reference']}: matched={metric['matched_points']} "
                    f"RMSE={rmse:.6g} kJ/mol MAE={metric['mae_kJ_mol']:.6g} kJ/mol"
                )
        if metadata.get("tdb_sanity"):
            sanity = metadata["tdb_sanity"]
            for warning in sanity.get("warnings", []):
                print(f"WARNING: {warning}")
        return metadata
    if args.command == "benchmark-uq-phase":
        metadata = write_benchmarked_uq_phase(args)
        print(f"Wrote posterior weights      : {metadata['outputs']['posterior_model_weights']}")
        print(f"Wrote candidate liquidus CSV : {metadata['outputs']['candidate_phase_diagrams']}")
        print(f"Wrote posterior envelope CSV : {metadata['outputs']['posterior_phase_envelope']}")
        if metadata["outputs"].get("phase_plot"):
            print(f"Wrote benchmarked phase plot : {metadata['outputs']['phase_plot']}")
        print(f"Wrote tension report         : {metadata['outputs']['tension_report']}")
        print(f"Best joint candidate         : {metadata['best_joint_label']}")
        if metadata["posterior_tension"]:
            print("WARNING: Hmix and eutectic benchmarks prefer different candidate curves.")
        return metadata
    if args.command == "mstdb-sanitize":
        metadata_path = args.metadata or args.output.with_suffix(args.output.suffix + ".metadata.json")
        metadata = write_sanitized_mstdb_chemsage(args.input, args.output, metadata_path=metadata_path)
        print(f"Wrote sanitized MSTDB/ChemSage file: {metadata['output']}")
        print(f"Wrote sanitizer metadata        : {metadata_path}")
        for replacement in metadata.get("replacements", []):
            print(
                f"replacement {replacement['pattern']}: "
                f"{replacement['count']} occurrence(s)"
            )
        return metadata
    if args.command == "mqmqa-binary":
        metadata = write_mqmqa_binary_curve(args)
        print(f"Wrote MQMQA binary curve: {metadata['outputs']['curve_csv']}")
        print("Basis note:", metadata["basis_note"])
        return metadata
    if args.command == "pycalphad-bridge":
        metadata = write_pycalphad_bridge(args.params, args.outdir.resolve())
        print(f"Wrote pycalphad bridge: {metadata['outputs']['bridge']}")
        print(f"Wrote parameter copy  : {metadata['outputs']['parameters']}")
        return metadata
    if args.command == "database":
        return database_action(args)
    parser.error(f"Unsupported command: {args.command}")
    return None


if __name__ == "__main__":
    main()
