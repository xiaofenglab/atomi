"""Element-order helpers for generated LAMMPS inputs."""

from __future__ import annotations

import re
from typing import Any


_FALLBACK_ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Si": 28.085,
    "U": 238.0289,
    "Gd": 157.25,
}


def split_elements(value: Any, *, default: list[str] | None = None) -> list[str]:
    """Return element symbols from a list or comma/space-separated string."""
    if value in (None, ""):
        return list(default or [])
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,\s]+", value) if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def model_elements(cfg: dict[str, Any]) -> list[str]:
    """Return the LAMMPS atom-type/model element order.

    The historical default is O,U for existing UO2 workflows, but callers should
    pass ``model_elements`` explicitly for new systems such as UC2.
    """
    return split_elements(cfg.get("model_elements"), default=["O", "U"])


def atomic_mass(symbol: str) -> float:
    """Return an atomic mass in amu for a chemical symbol."""
    try:
        from ase.data import atomic_masses, atomic_numbers

        return float(atomic_masses[atomic_numbers[symbol]])
    except Exception:
        if symbol in _FALLBACK_ATOMIC_MASSES:
            return float(_FALLBACK_ATOMIC_MASSES[symbol])
        raise KeyError(
            f"No atomic mass configured for element {symbol!r}. "
            f"Set mass_{symbol} or element_masses.{symbol} in the LAMMPS config."
        ) from None


def _mass_from_config(cfg: dict[str, Any], symbol: str, type_index: int) -> float | None:
    for key in ("element_masses", "masses"):
        masses = cfg.get(key)
        if isinstance(masses, dict) and symbol in masses:
            return float(masses[symbol])
        if isinstance(masses, dict) and str(type_index) in masses:
            return float(masses[str(type_index)])
        if isinstance(masses, (list, tuple)) and type_index - 1 < len(masses):
            return float(masses[type_index - 1])
    legacy_key = f"mass_{symbol}"
    if legacy_key in cfg:
        return float(cfg[legacy_key])
    return None


def element_masses(cfg: dict[str, Any], elements: list[str] | None = None) -> dict[str, float]:
    """Return masses for the configured LAMMPS atom-type element order."""
    ordered = elements or model_elements(cfg)
    masses: dict[str, float] = {}
    for index, symbol in enumerate(ordered, start=1):
        configured = _mass_from_config(cfg, symbol, index)
        masses[symbol] = float(configured if configured is not None else atomic_mass(symbol))
    return masses


def lammps_mass_lines(cfg: dict[str, Any]) -> str:
    """Return LAMMPS ``mass`` commands matching ``model_elements`` order."""
    ordered = model_elements(cfg)
    masses = element_masses(cfg, ordered)
    return "\n".join(f"mass            {index} {masses[symbol]:.10g}" for index, symbol in enumerate(ordered, start=1))


def copy_mass_keys(template: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Copy generic and legacy mass settings from one config dict into another."""
    for key in ("element_masses", "masses"):
        if key in template:
            cfg[key] = template[key]
    for key, value in template.items():
        if key.startswith("mass_"):
            cfg[key] = value
