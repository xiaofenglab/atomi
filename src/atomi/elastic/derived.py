"""Derived elastic, mechanical, and thermophysical properties."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from ase.data import atomic_masses, atomic_numbers


AVOGADRO = 6.02214076e23
BOLTZMANN = 1.380649e-23
PLANCK = 6.62607015e-34
GPA_TO_PA = 1.0e9
A3_TO_M3 = 1.0e-30
R_GAS = 8.31446261815324


def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def formula_counts(formula: str) -> dict[str, float]:
    """Parse a simple chemical formula such as UO2 or Gd0.1U0.9O2."""

    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    if not tokens:
        raise ValueError(f"Could not parse formula: {formula}")
    counts: dict[str, float] = {}
    for symbol, raw in tokens:
        if symbol not in atomic_numbers:
            raise ValueError(f"Unknown element in formula: {symbol}")
        counts[symbol] = counts.get(symbol, 0.0) + (float(raw) if raw else 1.0)
    return counts


def formula_molar_mass_g_mol(formula: str) -> float:
    counts = formula_counts(formula)
    return float(sum(atomic_masses[atomic_numbers[symbol]] * count for symbol, count in counts.items()))


def formula_atom_count(formula: str) -> float:
    return float(sum(formula_counts(formula).values()))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, str) and value.strip().lower() in {"nan", "none"}:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def row_volume_A3(row: dict[str, Any]) -> float | None:
    for key in ("V_mean_A3", "volume_A3", "volume_target_cell_A3", "volume_A3_mean"):
        value = _float_or_none(row.get(key))
        if value is not None and value > 0:
            return value
    return None


def density_from_formula_volume(
    *,
    formula: str,
    formula_units: float,
    volume_A3: float,
) -> float:
    """Return density in kg/m^3 from a formula, formula units in the cell, and cell volume."""

    molar_mass_kg_mol = formula_molar_mass_g_mol(formula) * 1.0e-3
    mass_kg = float(formula_units) * molar_mass_kg_mol / AVOGADRO
    return mass_kg / (float(volume_A3) * A3_TO_M3)


def complete_elastic_derived(row: dict[str, Any], tensor: np.ndarray | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    k_h = _float_or_none(row.get("K_H_GPa"))
    g_h = _float_or_none(row.get("G_H_GPa"))
    k_v = _float_or_none(row.get("K_V_GPa"))
    g_v = _float_or_none(row.get("G_V_GPa"))
    k_r = _float_or_none(row.get("K_R_GPa"))
    g_r = _float_or_none(row.get("G_R_GPa"))
    e_h = _float_or_none(row.get("E_H_GPa"))
    nu_h = _float_or_none(row.get("nu_H"))
    c11 = _float_or_none(row.get("C11_GPa"))
    c12 = _float_or_none(row.get("C12_GPa"))
    c44 = _float_or_none(row.get("C44_GPa"))
    if k_h is not None and g_h is not None and abs(g_h) > 1.0e-12:
        out["pugh_K_over_G"] = k_h / g_h
        out["ductility_pugh"] = "ductile" if k_h / g_h >= 1.75 else "brittle"
    if c12 is not None and c44 is not None:
        out["cauchy_pressure_GPa"] = c12 - c44
    if c11 is not None and c12 is not None and c44 is not None and abs(c11 - c12) > 1.0e-12:
        out["zener_anisotropy"] = 2.0 * c44 / (c11 - c12)
    if k_v is not None and g_v is not None and k_r is not None and g_r is not None:
        if abs(k_r) > 1.0e-12 and abs(g_r) > 1.0e-12:
            out["universal_anisotropy_AU"] = 5.0 * g_v / g_r + k_v / k_r - 6.0
    if e_h is not None:
        out["E_H_Pa"] = e_h * GPA_TO_PA
    if nu_h is not None:
        out["nu_H"] = nu_h
    if tensor is not None:
        eigenvalues = np.linalg.eigvalsh(0.5 * (np.asarray(tensor, dtype=float) + np.asarray(tensor, dtype=float).T))
        out["elastic_min_eigenvalue_GPa"] = float(np.min(eigenvalues))
        out["elastic_max_eigenvalue_GPa"] = float(np.max(eigenvalues))
        out["elastic_condition_number"] = (
            float(np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues)))
            if np.all(np.abs(eigenvalues) > 1.0e-12)
            else math.inf
        )
    return out


def complete_thermophysical_derived(
    row: dict[str, Any],
    *,
    formula: str | None = None,
    formula_units: float | None = None,
    density_kg_m3: float | None = None,
    molar_mass_g_mol: float | None = None,
    atoms_per_formula_unit: float | None = None,
) -> dict[str, Any]:
    """Compute sound velocities and Debye temperature when density information is available."""

    k_h = _float_or_none(row.get("K_H_GPa"))
    g_h = _float_or_none(row.get("G_H_GPa"))
    if k_h is None or g_h is None or k_h <= 0 or g_h <= 0:
        return {}
    volume_A3 = row_volume_A3(row)
    if formula:
        molar_mass_g_mol = formula_molar_mass_g_mol(formula)
        atoms_per_formula_unit = formula_atom_count(formula)
        if formula_units is not None and volume_A3 is not None:
            density_kg_m3 = density_from_formula_volume(
                formula=formula,
                formula_units=float(formula_units),
                volume_A3=volume_A3,
            )
    if density_kg_m3 is None or density_kg_m3 <= 0:
        return {}
    k_pa = k_h * GPA_TO_PA
    g_pa = g_h * GPA_TO_PA
    v_s = math.sqrt(g_pa / density_kg_m3)
    v_p = math.sqrt((k_pa + 4.0 * g_pa / 3.0) / density_kg_m3)
    v_m = (1.0 / 3.0 * (2.0 / v_s**3 + 1.0 / v_p**3)) ** (-1.0 / 3.0)
    out: dict[str, Any] = {
        "density_kg_m3": density_kg_m3,
        "density_g_cm3": density_kg_m3 / 1000.0,
        "v_s_km_s": v_s / 1000.0,
        "v_p_km_s": v_p / 1000.0,
        "v_m_km_s": v_m / 1000.0,
    }
    atom_number_density: float | None = None
    if volume_A3 is not None and formula_units is not None and atoms_per_formula_unit is not None:
        atom_number_density = float(formula_units) * float(atoms_per_formula_unit) / (volume_A3 * A3_TO_M3)
    elif molar_mass_g_mol is not None and atoms_per_formula_unit is not None:
        molar_mass_kg_mol = float(molar_mass_g_mol) * 1.0e-3
        atom_number_density = float(atoms_per_formula_unit) * density_kg_m3 * AVOGADRO / molar_mass_kg_mol
    if atom_number_density is not None and atom_number_density > 0:
        out["atom_number_density_m3"] = atom_number_density
        out["theta_D_K"] = PLANCK / BOLTZMANN * (3.0 * atom_number_density / (4.0 * math.pi)) ** (1.0 / 3.0) * v_m
    return out


def debye_cv_J_mol_formula_K(T_K: float, theta_D_K: float, atoms_per_formula_unit: float) -> float:
    """Debye constant-volume heat capacity per mole of formula units."""

    T_K = float(T_K)
    theta_D_K = float(theta_D_K)
    n = float(atoms_per_formula_unit)
    if T_K <= 0 or theta_D_K <= 0 or n <= 0:
        return 0.0
    x_max = theta_D_K / T_K
    if x_max > 250.0:
        return 0.0
    npts = max(200, int(min(5000, 50 * max(1.0, x_max))))
    x = np.linspace(1.0e-8, x_max, npts)
    ex = np.exp(np.clip(x, None, 700.0))
    integrand = x**4 * ex / np.expm1(x) ** 2
    integral = trapz_compat(integrand, x)
    return 9.0 * n * R_GAS * (T_K / theta_D_K) ** 3 * integral


def debye_thermal_table(
    theta_D_K: float,
    *,
    atoms_per_formula_unit: float,
    T_min: float,
    T_max: float,
    T_step: float,
) -> list[dict[str, float]]:
    """Return Cv plus H/S/F increments from integrating the Debye Cv curve."""

    if T_step <= 0:
        raise ValueError("T_step must be positive")
    grid = np.arange(float(T_min), float(T_max) + 0.5 * float(T_step), float(T_step))
    grid = grid[grid >= 0.0]
    if grid.size == 0 or grid[0] != 0.0:
        grid = np.insert(grid, 0, 0.0)
    cv = np.array([debye_cv_J_mol_formula_K(t, theta_D_K, atoms_per_formula_unit) for t in grid])
    h = np.zeros_like(grid)
    s = np.zeros_like(grid)
    for i in range(1, len(grid)):
        dt = grid[i] - grid[i - 1]
        h[i] = h[i - 1] + 0.5 * (cv[i] + cv[i - 1]) * dt
        left = cv[i - 1] / grid[i - 1] if grid[i - 1] > 0 else 0.0
        right = cv[i] / grid[i] if grid[i] > 0 else 0.0
        s[i] = s[i - 1] + 0.5 * (left + right) * dt
    f_kj = (h - grid * s) / 1000.0
    return [
        {
            "T_K": float(t),
            "Cv_J_mol_formula_K": float(c),
            "H_rel_kJ_mol_formula": float(hj / 1000.0),
            "S_J_mol_formula_K": float(sj),
            "F_rel_kJ_mol_formula": float(fj),
        }
        for t, c, hj, sj, fj in zip(grid, cv, h, s, f_kj)
    ]
