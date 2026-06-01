"""Molecular Interaction Volume Model helpers and pycalphad bridge."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


R_J_MOLK = 8.31446261815324
SCHEMA = "atomi.calphad.mivm.parameters.v1"
SAMPLE_SCHEMA = "atomi.calphad.mivm.sample.v1"
DATABASE_SCHEMA = "atomi.copilot.mivm.parameter_database.v0.1"


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
    with table.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
