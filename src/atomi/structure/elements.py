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


@dataclass(frozen=True)
class ValenceMagmomInfo:
    """Curated initial-moment metadata for formal valence labels.

    The moment is an initialization/guard prior for collinear DFT+U workflows,
    not a measured ordered moment.  It follows formal open-shell electron
    counts and Hund spin-only ``2S`` where the valence/configuration is clear.
    """

    label: str
    element: str | None
    oxidation_state: int | None
    electron_configuration: str
    open_shell: str
    unpaired_electrons: int
    initial_magmom_abs_muB: float
    allowed_signs: tuple[int, ...] = (-1, 1)
    guard_abs_range_muB: tuple[float, float] | None = None
    sources: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LANTHANIDE_3PLUS_F_COUNTS: dict[str, int] = {
    "La": 0,
    "Ce": 1,
    "Pr": 2,
    "Nd": 3,
    "Pm": 4,
    "Sm": 5,
    "Eu": 6,
    "Gd": 7,
    "Tb": 8,
    "Dy": 9,
    "Ho": 10,
    "Er": 11,
    "Tm": 12,
    "Yb": 13,
    "Lu": 14,
}

LAN_THANIDE_SOURCE = (
    "Lanthanide chemistry: Ln3+ generally [Xe]4f^n; Ce/Pr/Tb also +4 and Sm/Eu/Yb also +2 can be stable; "
    "4f unpaired electrons and spin-orbit caveat are standard lanthanide-chemistry results."
)
ACTINIDE_SOURCE = (
    "Formal uranium oxide redox model: U3+/U4+/U5+/U6+ correspond to localized 5f3/5f2/5f1/5f0 priors; "
    "use as DFT+U initialization/guard, then verify OUTCAR moments and occupation matrices."
)
CLOSED_SHELL_ANION_SOURCE = "Closed-shell common anion/vacancy prior for DFT initial moments."


def _hund_unpaired_from_f_count(count: int) -> int:
    if count < 0 or count > 14:
        raise ValueError(f"f-shell count must be in [0, 14], got {count}")
    return count if count <= 7 else 14 - count


def _f_shell_magmom(
    *,
    label: str,
    element: str,
    oxidation_state: int,
    shell: str,
    count: int,
    sources: tuple[str, ...],
    notes: str,
) -> ValenceMagmomInfo:
    unpaired = _hund_unpaired_from_f_count(count)
    abs_mag = float(unpaired)
    upper = max(abs_mag + 1.0, 1.0)
    return ValenceMagmomInfo(
        label=label,
        element=element,
        oxidation_state=oxidation_state,
        electron_configuration=f"{shell}^{count}",
        open_shell=shell,
        unpaired_electrons=unpaired,
        initial_magmom_abs_muB=abs_mag,
        allowed_signs=(-1, 1) if unpaired else (0,),
        guard_abs_range_muB=(0.0, upper) if unpaired else (0.0, 0.25),
        sources=sources,
        notes=notes,
    )


def _build_valence_magmom_table() -> dict[str, ValenceMagmomInfo]:
    rows: dict[str, ValenceMagmomInfo] = {}

    def add(row: ValenceMagmomInfo) -> None:
        rows[row.label] = row
        if row.element and row.oxidation_state is not None:
            sign = "+" if row.oxidation_state >= 0 else "-"
            rows[f"{row.element}{abs(row.oxidation_state)}{sign}"] = row

    for element, f_count in LANTHANIDE_3PLUS_F_COUNTS.items():
        add(
            _f_shell_magmom(
                label=f"{element}3+",
                element=element,
                oxidation_state=3,
                shell="4f",
                count=f_count,
                sources=(LAN_THANIDE_SOURCE,),
                notes="Ln3+ spin-only initialization uses the number of unpaired 4f electrons; real effective moments include spin-orbit coupling.",
            )
        )

    for element, f_count, ox in (
        ("Sm", 6, 2),
        ("Eu", 7, 2),
        ("Yb", 14, 2),
        ("Ce", 0, 4),
        ("Pr", 1, 4),
        ("Tb", 7, 4),
    ):
        add(
            _f_shell_magmom(
                label=f"{element}{ox}+",
                element=element,
                oxidation_state=ox,
                shell="4f",
                count=f_count,
                sources=(LAN_THANIDE_SOURCE,),
                notes="Known lanthanide non-3+ stable-valence prior; validate for the host chemistry before using.",
            )
        )

    for ox, f_count in ((3, 3), (4, 2), (5, 1), (6, 0)):
        add(
            _f_shell_magmom(
                label=f"U{ox}+",
                element="U",
                oxidation_state=ox,
                shell="5f",
                count=f_count,
                sources=(ACTINIDE_SOURCE,),
                notes="Formal uranium redox prior for oxide DFT+U. U5+ 5f1 gives |MAGMOM_init|=1; use occupation-matrix/local-moment guards after SCF.",
            )
        )

    for label in ("O2-", "F-", "Cl-", "Br-", "I-", "S2-", "N3-", "Va", "vacancy"):
        add(
            ValenceMagmomInfo(
                label=label,
                element=None if label.lower() in VACANCY_TOKENS or label == "Va" else normalize_element_symbol(label),
                oxidation_state=None,
                electron_configuration="closed-shell/vacancy",
                open_shell="none",
                unpaired_electrons=0,
                initial_magmom_abs_muB=0.0,
                allowed_signs=(0,),
                guard_abs_range_muB=(0.0, 0.25),
                sources=(CLOSED_SHELL_ANION_SOURCE,),
                notes="Use zero initial MAGMOM unless explicit radical/hole-polaron chemistry is being modeled.",
            )
        )
    return rows


VALENCE_MAGMOM_TABLE: dict[str, ValenceMagmomInfo]


def valence_magmom_info(label: str | None, *, strict: bool = False) -> ValenceMagmomInfo | None:
    """Return a curated MAGMOM prior for a formal valence species.

    Examples include ``U4+``, ``U5+``, ``Gd3+``, ``O2-``, and ``Va``.  Unknown
    labels return ``None`` unless ``strict`` is true.
    """

    raw = str(label or "").strip()
    aliases = {raw, raw.replace(" ", ""), raw.capitalize()}
    normalized = None
    try:
        normalized = normalize_element_symbol(raw, allow_vacancy=True)
    except Exception:
        normalized = None
    match = re.match(r"^\s*([A-Za-z]{1,2})\s*([0-9]+)?\s*([+-])", raw)
    if match:
        element = normalize_element_symbol(match.group(1), allow_vacancy=False)
        magnitude = match.group(2) or "1"
        sign = match.group(3)
        aliases.add(f"{element}{magnitude}{sign}")
    if normalized and raw in VALENCE_MAGMOM_TABLE:
        aliases.add(raw)
    for alias in aliases:
        if alias in VALENCE_MAGMOM_TABLE:
            return VALENCE_MAGMOM_TABLE[alias]
    if strict:
        raise ValueError(f"No curated valence MAGMOM prior is available for {label!r}.")
    return None


def valence_magmom_table(
    labels: Iterable[str] | None = None,
    *,
    as_dict: bool = True,
) -> dict[str, dict[str, Any]] | list[ValenceMagmomInfo]:
    """Return curated valence MAGMOM priors.

    When labels are provided, unknown labels are skipped so callers can mix
    arbitrary species with the curated subset.
    """

    if labels is None:
        rows = list(dict.fromkeys(VALENCE_MAGMOM_TABLE.values()))
    else:
        rows = [info for label in labels if (info := valence_magmom_info(label)) is not None]
    if not as_dict:
        return rows
    return {row.label: row.to_dict() for row in rows}


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


VALENCE_MAGMOM_TABLE = _build_valence_magmom_table()


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
