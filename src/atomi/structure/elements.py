"""Element metadata helpers shared by structure-centric workflows."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

VACANCY_TOKENS = {"", "va", "vac", "vacancy", "v_o", "vo", "x"}


@dataclass(frozen=True)
class ElementInfo:
    """Backend-neutral element metadata with source provenance."""

    symbol: str
    atomic_number: int | None = None
    atomic_mass_amu: float | None = None
    covalent_radius_A: float | None = None
    vdw_radius_A: float | None = None
    xray_edges_eV: dict[str, float] | None = None
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_element_symbol(symbol: str | None, *, allow_vacancy: bool = True) -> str | None:
    """Normalize element-like labels such as ``u``, ``U5+``, and ``Gd3+``.

    Vacancy labels return ``None`` when ``allow_vacancy`` is true.  This keeps
    defect-lattice labels such as ``Va`` from being mistaken for vanadium.
    """

    raw = str(symbol or "").strip()
    if allow_vacancy and raw.lower() in VACANCY_TOKENS:
        return None
    match = re.match(r"^\s*([A-Za-z]{1,2})", raw)
    if not match:
        raise ValueError(f"Cannot parse an element symbol from {symbol!r}.")
    candidate = match.group(1).capitalize()
    if allow_vacancy and candidate.lower() in VACANCY_TOKENS:
        return None
    try:
        from ase.data import atomic_numbers

        if candidate in atomic_numbers and int(atomic_numbers[candidate]) > 0:
            return candidate
    except Exception:
        pass
    try:
        import xraydb  # type: ignore

        if int(xraydb.atomic_number(candidate)) > 0:
            return candidate
    except Exception:
        pass
    raise ValueError(f"Unknown element symbol: {symbol!r}.")


def _ase_element_info(symbol: str) -> tuple[dict[str, Any], list[str]]:
    data: dict[str, Any] = {}
    sources: list[str] = []
    try:
        from ase.data import atomic_masses, atomic_numbers, covalent_radii, vdw_radii

        atomic_number = int(atomic_numbers[symbol])
        if atomic_number > 0:
            data["atomic_number"] = atomic_number
            data["atomic_mass_amu"] = float(atomic_masses[atomic_number])
            covalent = float(covalent_radii[atomic_number])
            if covalent > 0:
                data["covalent_radius_A"] = covalent
            vdw = float(vdw_radii[atomic_number])
            if vdw > 0:
                data["vdw_radius_A"] = vdw
            sources.append("ase")
    except Exception:
        pass
    return data, sources


def _xraydb_element_info(symbol: str, edges: Iterable[str]) -> tuple[dict[str, Any], list[str]]:
    data: dict[str, Any] = {}
    sources: list[str] = []
    try:
        import xraydb  # type: ignore

        atomic_number = int(xraydb.atomic_number(symbol))
        if atomic_number > 0:
            data.setdefault("atomic_number", atomic_number)
            sources.append("xraydb")
        edge_values: dict[str, float] = {}
        for edge in edges:
            try:
                edge_obj = xraydb.xray_edge(symbol, edge)
                energy = getattr(edge_obj, "energy", None)
                if energy is not None:
                    edge_values[str(edge)] = float(energy)
            except Exception:
                continue
        if edge_values:
            data["xray_edges_eV"] = edge_values
    except Exception:
        pass
    return data, sources


def _periodictable_element_info(symbol: str) -> tuple[dict[str, Any], list[str]]:
    data: dict[str, Any] = {}
    sources: list[str] = []
    try:
        import periodictable as pt  # type: ignore

        element = getattr(pt, symbol)
        if getattr(element, "number", None):
            data.setdefault("atomic_number", int(element.number))
        if getattr(element, "mass", None):
            data.setdefault("atomic_mass_amu", float(element.mass))
        if getattr(element, "covalent_radius", None):
            data.setdefault("covalent_radius_A", float(element.covalent_radius))
        sources.append("periodictable")
    except Exception:
        pass
    return data, sources


def element_info(
    symbol: str | None,
    *,
    include_xray_edges: bool = False,
    edges: Iterable[str] = ("K", "L3", "L2", "L1"),
    allow_vacancy: bool = True,
) -> ElementInfo | None:
    """Return normalized metadata for one element-like label.

    ASE is the primary metadata source because it covers the periodic table and
    is already part of Atomi's base dependency set.  ``periodictable`` and
    ``xraydb`` fill gaps and edge metadata when requested.
    """

    normalized = normalize_element_symbol(symbol, allow_vacancy=allow_vacancy)
    if normalized is None:
        return None
    merged: dict[str, Any] = {}
    sources: list[str] = []
    for payload, payload_sources in (
        _ase_element_info(normalized),
        _periodictable_element_info(normalized),
        _xraydb_element_info(normalized, edges) if include_xray_edges else ({}, []),
    ):
        for key, value in payload.items():
            if value is not None and key not in merged:
                merged[key] = value
        sources.extend(source for source in payload_sources if source not in sources)
    return ElementInfo(symbol=normalized, sources=tuple(sources), **merged)


def atomic_number(symbol: str | None) -> int:
    info = element_info(symbol, allow_vacancy=False)
    if info is None or info.atomic_number is None:
        raise ValueError(f"Atomic number for {symbol!r} is unavailable.")
    return int(info.atomic_number)


def atomic_mass_amu(symbol: str | None) -> float:
    info = element_info(symbol, allow_vacancy=False)
    if info is None or info.atomic_mass_amu is None:
        raise ValueError(f"Atomic mass for {symbol!r} is unavailable.")
    return float(info.atomic_mass_amu)


def element_table(
    symbols: Iterable[str],
    *,
    include_xray_edges: bool = False,
    edges: Iterable[str] = ("K", "L3", "L2", "L1"),
    skip_vacancies: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return metadata keyed by unique normalized element symbol."""

    table: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        info = element_info(symbol, include_xray_edges=include_xray_edges, edges=edges, allow_vacancy=skip_vacancies)
        if info is None:
            continue
        table.setdefault(info.symbol, info.to_dict())
    return dict(sorted(table.items(), key=lambda item: int(item[1].get("atomic_number") or 999)))


def annotate_symbols(
    symbols: Iterable[str],
    *,
    include_xray_edges: bool = False,
    edges: Iterable[str] = ("K", "L3", "L2", "L1"),
) -> list[dict[str, Any]]:
    """Return one metadata row per input symbol, preserving vacancies."""

    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(symbols):
        info = element_info(raw, include_xray_edges=include_xray_edges, edges=edges, allow_vacancy=True)
        row: dict[str, Any] = {"index": index, "raw_symbol": raw}
        if info is None:
            row.update({"symbol": None, "is_vacancy": True})
        else:
            row.update(info.to_dict())
            row["is_vacancy"] = False
        rows.append(row)
    return rows
