"""Shared formula-unit and cell-normalization helpers."""

from __future__ import annotations

import math
import re
from typing import Any


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def formula_counts(formula: str | None) -> dict[str, float]:
    if not formula:
        return {}
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    if not tokens:
        return {}
    counts: dict[str, float] = {}
    for element, raw_count in tokens:
        counts[element] = counts.get(element, 0.0) + (float(raw_count) if raw_count else 1.0)
    return counts


def formula_atom_count(formula: str | None) -> float | None:
    counts = formula_counts(formula)
    if not counts:
        return None
    return float(sum(counts.values()))


def infer_formula_units(
    *,
    formula_units: float | None = None,
    natoms: float | None = None,
    atoms_per_formula_unit: float | None = None,
    formula: str | None = None,
) -> float | None:
    if formula_units is not None and formula_units > 0:
        return float(formula_units)
    apfu = atoms_per_formula_unit
    if apfu is None:
        apfu = formula_atom_count(formula)
    if natoms is None or apfu is None or apfu <= 0:
        return None
    return float(natoms) / float(apfu)


def cell_metadata(
    *,
    formula: str | None = None,
    natoms: float | None = None,
    atoms_per_formula_unit: float | None = None,
    formula_units: float | None = None,
    target_z: float | None = None,
    cell_role: str = "simulation-cell",
    normalization_basis: str = "per-formula",
) -> dict[str, Any]:
    apfu = atoms_per_formula_unit
    if apfu is None:
        apfu = formula_atom_count(formula)
    nfu = infer_formula_units(
        formula_units=formula_units,
        natoms=natoms,
        atoms_per_formula_unit=apfu,
        formula=formula,
    )
    meta = {
        "schema": "atomi.cell_metadata.v1",
        "formula": formula or "",
        "natoms": float(natoms) if natoms is not None else None,
        "atoms_per_formula_unit": float(apfu) if apfu is not None else None,
        "n_formula_units": float(nfu) if nfu is not None else None,
        "target_z_formula_units": float(target_z) if target_z is not None else None,
        "cell_role": cell_role,
        "normalization_basis": normalization_basis,
        "normalization_notes": [
            "per-formula means per mole of formula units",
            "target-cell means per mole of the target cell containing target_z formula units",
            "simulation-cell means per mole of the MD/DFT simulation cell containing n_formula_units",
        ],
    }
    return meta


def extensive_basis_factor(
    *,
    from_basis: str,
    to_basis: str,
    formula_units: float | None = None,
    target_z: float | None = None,
) -> float:
    """Return multiplier for extensive molar quantities between common bases."""

    aliases = {
        "formula": "per-formula",
        "per_formula": "per-formula",
        "mol-formula": "per-formula",
        "target": "target-cell",
        "target_cell": "target-cell",
        "mol-target-cell": "target-cell",
        "cell": "simulation-cell",
        "simulation": "simulation-cell",
        "simulation_cell": "simulation-cell",
        "mol-cell": "simulation-cell",
    }
    src = aliases.get(from_basis, from_basis)
    dst = aliases.get(to_basis, to_basis)
    if src == dst:
        return 1.0
    if src == "per-formula":
        per_formula = 1.0
    elif src == "target-cell":
        if target_z is None or target_z <= 0:
            raise ValueError("target_z is required for target-cell basis conversion")
        per_formula = 1.0 / float(target_z)
    elif src == "simulation-cell":
        if formula_units is None or formula_units <= 0:
            raise ValueError("formula_units is required for simulation-cell basis conversion")
        per_formula = 1.0 / float(formula_units)
    else:
        raise ValueError(f"Unknown extensive basis: {from_basis}")
    if dst == "per-formula":
        return per_formula
    if dst == "target-cell":
        if target_z is None or target_z <= 0:
            raise ValueError("target_z is required for target-cell basis conversion")
        return per_formula * float(target_z)
    if dst == "simulation-cell":
        if formula_units is None or formula_units <= 0:
            raise ValueError("formula_units is required for simulation-cell basis conversion")
        return per_formula * float(formula_units)
    raise ValueError(f"Unknown extensive basis: {to_basis}")
