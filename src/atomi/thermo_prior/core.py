"""Schema and basis conversion utilities for thermodynamic ML priors."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

EV_PER_FORMULA_TO_KJ_PER_MOL = 96.4853321233
PRIOR_SCHEMA = "atomi.thermo_prior.v1"


def parse_formula_counts_case(formula: str) -> dict[str, float]:
    """Parse a compact chemical formula preserving element case.

    This intentionally handles simple formulas such as ``Na3U5Cl18`` and
    ``UCl3``. Hydrates, parentheses, and charged species are outside this
    first thermo-prior bridge.
    """
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", str(formula).strip())
    if not parts:
        raise ValueError(f"Could not parse formula: {formula}")
    counts: dict[str, float] = {}
    for element, raw_count in parts:
        counts[element] = counts.get(element, 0.0) + (float(raw_count) if raw_count else 1.0)
    return counts


def formula_unit_count(formula: str) -> float:
    return sum(parse_formula_counts_case(formula).values())


def solve_pseudobinary_coefficients(
    formula: str,
    component_a: str,
    component_b: str,
    *,
    tolerance: float = 1.0e-8,
) -> dict[str, float]:
    """Solve ``formula = coeff_a * component_a + coeff_b * component_b``.

    Returns coefficients and ``x_B = coeff_b / (coeff_a + coeff_b)`` on the
    pseudo-binary formula basis used by the fast MIVM phase diagnostic.
    """
    target = parse_formula_counts_case(formula)
    counts_a = parse_formula_counts_case(component_a)
    counts_b = parse_formula_counts_case(component_b)
    elements = sorted(set(target) | set(counts_a) | set(counts_b))
    equations = [(counts_a.get(el, 0.0), counts_b.get(el, 0.0), target.get(el, 0.0)) for el in elements]
    coeff_a: float | None = None
    coeff_b: float | None = None
    for i, (a1, b1, c1) in enumerate(equations):
        for a2, b2, c2 in equations[i + 1 :]:
            det = a1 * b2 - a2 * b1
            if abs(det) <= tolerance:
                continue
            cand_a = (c1 * b2 - c2 * b1) / det
            cand_b = (a1 * c2 - a2 * c1) / det
            if cand_a >= -tolerance and cand_b >= -tolerance:
                coeff_a, coeff_b = cand_a, cand_b
                break
        if coeff_a is not None and coeff_b is not None:
            break
    if coeff_a is None or coeff_b is None:
        raise ValueError(f"Could not express {formula} as a pseudo-binary combination of {component_a} and {component_b}.")
    coeff_a = 0.0 if abs(coeff_a) < tolerance else coeff_a
    coeff_b = 0.0 if abs(coeff_b) < tolerance else coeff_b
    for element, a_count, b_count in [(el, counts_a.get(el, 0.0), counts_b.get(el, 0.0)) for el in elements]:
        predicted = coeff_a * a_count + coeff_b * b_count
        actual = target.get(element, 0.0)
        if abs(predicted - actual) > tolerance:
            raise ValueError(
                f"{formula} is not on the {component_a}-{component_b} pseudo-binary join: "
                f"{element} expected {actual:g}, got {predicted:g}."
            )
    total = coeff_a + coeff_b
    if total <= 0.0:
        raise ValueError("Pseudo-binary coefficients sum to zero.")
    return {
        "coeff_a": coeff_a,
        "coeff_b": coeff_b,
        "x_B": coeff_b / total,
        "formula_units_per_compound": total,
    }


def salt_reference_gform_kj_mol(
    *,
    formula: str,
    component_a: str,
    component_b: str,
    formation_energy_ev_atom: float,
    component_a_formation_energy_ev_atom: float,
    component_b_formation_energy_ev_atom: float,
) -> float:
    """Convert elemental-basis formation energies to pseudo-binary salt basis.

    The returned value is kJ/mol of pseudo-binary formula units, i.e. divided
    by ``coeff_a + coeff_b`` for a compound like ``Na3U5Cl18 = 3 NaCl + 5 UCl3``.
    """
    coeffs = solve_pseudobinary_coefficients(formula, component_a, component_b)
    n_formula = formula_unit_count(formula)
    n_a = formula_unit_count(component_a)
    n_b = formula_unit_count(component_b)
    delta_ev_per_compound = (
        formation_energy_ev_atom * n_formula
        - coeffs["coeff_a"] * component_a_formation_energy_ev_atom * n_a
        - coeffs["coeff_b"] * component_b_formation_energy_ev_atom * n_b
    )
    return delta_ev_per_compound * EV_PER_FORMULA_TO_KJ_PER_MOL / coeffs["formula_units_per_compound"]


def write_line_compound_prior(
    *,
    out: Path,
    formula: str,
    component_a: str,
    component_b: str,
    label: str | None = None,
    gform_ref_kj_mol: float,
    dcp_form_j_mol_k: float = 0.0,
    tref_k: float = 298.15,
    temperature_min_k: float | None = None,
    temperature_max_k: float | None = None,
    uncertainty_kj_mol: float | None = None,
    source: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    coeffs = solve_pseudobinary_coefficients(formula, component_a, component_b)
    if temperature_min_k is not None and temperature_min_k <= 0.0:
        raise ValueError("temperature_min_k must be positive when provided.")
    if temperature_max_k is not None and temperature_max_k <= 0.0:
        raise ValueError("temperature_max_k must be positive when provided.")
    if temperature_min_k is not None and temperature_max_k is not None and temperature_min_k >= temperature_max_k:
        raise ValueError("temperature_min_k must be smaller than temperature_max_k.")
    prior = {
        "schema": PRIOR_SCHEMA,
        "kind": "line_compound",
        "formula": formula,
        "label": label or formula,
        "components": {"A": component_a, "B": component_b},
        "pseudo_binary": {
            "coeff_A": coeffs["coeff_a"],
            "coeff_B": coeffs["coeff_b"],
            "x_B": coeffs["x_B"],
            "basis": "kJ/mol pseudo-binary formula unit",
        },
        "thermo": {
            "gform_ref_kJ_mol": gform_ref_kj_mol,
            "dCp_form_J_mol_K": dcp_form_j_mol_k,
            "tref_K": tref_k,
            "temperature_min_K": temperature_min_k,
            "temperature_max_K": temperature_max_k,
        },
        "uncertainty": {"gform_sigma_kJ_mol": uncertainty_kj_mol},
        "source": source or {"method": "manual"},
        "notes": notes or [],
        "calphad_mivm": {
            "line_compound_spec": ":".join(
                [
                    str(label or formula),
                    f"{coeffs['x_B']:.12g}",
                    f"{gform_ref_kj_mol:.12g}",
                    f"{dcp_form_j_mol_k:.12g}",
                    f"{tref_k:.12g}",
                    "" if temperature_min_k is None else f"{temperature_min_k:.12g}",
                    "" if temperature_max_k is None else f"{temperature_max_k:.12g}",
                ]
            ).rstrip(":")
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return prior


def write_cp_prior(
    *,
    out: Path,
    formula: str,
    cp_j_mol_k: float,
    temperature_min_k: float,
    temperature_max_k: float,
    uncertainty_j_mol_k: float | None = None,
    source: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    prior = {
        "schema": PRIOR_SCHEMA,
        "kind": "cp_solid",
        "formula": formula,
        "thermo": {
            "model": "constant_cp_placeholder",
            "Cp_J_mol_K": cp_j_mol_k,
            "temperature_min_K": temperature_min_k,
            "temperature_max_K": temperature_max_k,
        },
        "uncertainty": {"Cp_sigma_J_mol_K": uncertainty_j_mol_k},
        "source": source or {"method": "manual_placeholder"},
        "notes": notes
        or [
            "Placeholder Cp prior. Replace with calibrated ML or experimental/phonopy data before final CALPHAD assessment."
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return prior


def read_prior(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != PRIOR_SCHEMA:
        raise ValueError(f"{path} is not an {PRIOR_SCHEMA} prior.")
    return data


def line_compound_spec_from_prior(data: dict[str, Any], *, default_tref_k: float) -> dict[str, Any]:
    if data.get("kind") != "line_compound":
        raise ValueError("Only line_compound priors can be passed to benchmark-uq-phase.")
    label = str(data.get("label") or data.get("formula") or "line_compound")
    pseudo = data.get("pseudo_binary", {})
    thermo = data.get("thermo", {})
    x_b = float(pseudo["x_B"])
    gform = float(thermo["gform_ref_kJ_mol"])
    dcp = float(thermo.get("dCp_form_J_mol_K", 0.0) or 0.0)
    tref = float(thermo.get("tref_K", default_tref_k) or default_tref_k)
    tmin = thermo.get("temperature_min_K")
    tmax = thermo.get("temperature_max_K")
    tmin_k = float(tmin) if tmin is not None else None
    tmax_k = float(tmax) if tmax is not None else None
    if not math.isfinite(x_b) or x_b <= 0.0 or x_b >= 1.0:
        raise ValueError(f"Invalid line-compound x_B in prior {label}.")
    if tmin_k is not None and (not math.isfinite(tmin_k) or tmin_k <= 0.0):
        raise ValueError(f"Invalid line-compound temperature_min_K in prior {label}.")
    if tmax_k is not None and (not math.isfinite(tmax_k) or tmax_k <= 0.0):
        raise ValueError(f"Invalid line-compound temperature_max_K in prior {label}.")
    if tmin_k is not None and tmax_k is not None and tmin_k >= tmax_k:
        raise ValueError(f"Invalid line-compound temperature window in prior {label}.")
    return {
        "label": label,
        "x_B": x_b,
        "gform_ref_kJ_mol": gform,
        "dCp_form_J_mol_K": dcp,
        "tref_K": tref,
        "tmin_K": tmin_k,
        "tmax_K": tmax_k,
        "prior_source": data.get("source", {}),
        "prior_uncertainty": data.get("uncertainty", {}),
    }


def load_line_compound_priors(paths: list[Path] | None, *, default_tref_k: float) -> list[dict[str, Any]]:
    compounds: list[dict[str, Any]] = []
    for path in paths or []:
        data = read_prior(Path(path))
        compounds.append(line_compound_spec_from_prior(data, default_tref_k=default_tref_k))
    return compounds
