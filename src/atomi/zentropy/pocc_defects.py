"""POCC/zentropy defect thermodynamics helpers.

This module keeps a clear boundary between configurational counting and
thermodynamic weighting.  POCC, enumlib, motif embedding, or VASP ingestion
provide configurations and degeneracies.  The zentropy layer then attaches
finite-temperature free energies and computes population vectors.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from atomi.vasp.qha_summary import parse_calc_folder
from atomi.zentropy.stage_utils import K_B_EV_PER_K, finite_float, format_value


SCHEMA = "atomi.zentropy.pocc_defects.v1"
R_J_MOLK = 8.31446261815324
EV_TO_KJ_MOL = 96.48533212331002

EnergyKind = Literal[
    "E_static_DFT",
    "F_static_approx",
    "F_harmonic",
    "F_QHA",
    "F_elec_mag_vib",
    "G_TP",
    "Omega_T_muO",
]
SurfaceKind = Literal[
    "bulk_F_from_static_logsum",
    "bulk_Omega_from_static_logsum",
    "bulk_F_from_pocc_cumulant",
    "bulk_F_from_pocc_cumulant_dvc",
    "bulk_F_from_zentropy",
    "bulk_Omega_from_zentropy",
]


@dataclass(frozen=True)
class DefectSpecies:
    name: str
    element: str | None
    sublattice: str
    oxidation: float | None
    effective_charge: int


@dataclass(frozen=True)
class Sublattice:
    name: str
    sites_per_formula_unit: float
    allowed_species: tuple[str, ...]


@dataclass
class DefectConfiguration:
    config_id: str
    phase: str
    species_counts: dict[str, int]
    sublattice_counts: dict[str, int]
    degeneracy: float = 1.0
    degeneracy_type: str = "unknown"
    degeneracy_basis: str = "finite_supercell"
    E_static_eV: float | None = None
    energy_kind: EnergyKind = "E_static_DFT"
    G_eV_T: dict[float, float] = field(default_factory=dict)
    motif_labels: list[str] = field(default_factory=list)
    motif_features: dict[str, float] = field(default_factory=dict)
    structure_path: str | None = None
    source: str = "unknown"
    energy_status: str = "static"
    uncertainty_eV: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PopulationRow:
    macrostate_id: str
    T_K: float
    mu_O_eV: float | None
    config_id: str
    motif_labels: str
    degeneracy: float
    G_eV: float
    omega_eV: float
    delta_omega_eV: float
    probability: float
    N_cation: int
    N_O: int
    N_VaO: int
    N_Gd3: int
    N_U5: int
    x_Gd: float
    delta: float
    h_U5: float


GDUO2_SPECIES = {
    "U4": DefectSpecies("U4", "U", "cation", 4.0, 0),
    "U5": DefectSpecies("U5", "U", "cation", 5.0, 1),
    "Gd3": DefectSpecies("Gd3", "Gd", "cation", 3.0, -1),
    "O": DefectSpecies("O", "O", "anion", -2.0, 0),
    "VaO": DefectSpecies("VaO", None, "anion", None, 2),
}

GDUO2_SUBLATTICES = {
    "cation": Sublattice("cation", 1.0, ("U4", "U5", "Gd3")),
    "anion": Sublattice("anion", 2.0, ("O", "VaO")),
}


def _as_int(value: Any, default: int = 0) -> int:
    number = finite_float(value)
    if number is None:
        return default
    return int(round(number))


def _as_float(value: Any, default: float = 0.0) -> float:
    number = finite_float(value)
    return default if number is None else float(number)


def gduo2_default_config() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "system": {
            "phase": "fluorite",
            "reference_formula": "UO2",
            "basis": "per_cation_formula_unit",
        },
        "species": {key: asdict(value) for key, value in GDUO2_SPECIES.items()},
        "sublattices": {key: asdict(value) for key, value in GDUO2_SUBLATTICES.items()},
        "constraints": {
            "charge_neutrality": "N_U5 + 2*N_VaO - N_Gd3 == 0",
            "composition": {
                "x_Gd": "N_Gd3 / N_cation",
                "h_U5": "N_U5 / N_cation",
                "delta": "N_VaO / N_cation",
            },
        },
        "zentropy": {
            "ensemble_mode": "oxygen_semi_grand",
            "surface_builder": {
                "mode": "pocc_static_logsum",
                "allowed_modes": [
                    "exact_logsum",
                    "pocc_static_logsum",
                    "pocc_cumulant",
                    "pocc_cumulant_dvc",
                    "zentropy_logsum",
                    "pocc_zentropy_logsum",
                ],
                "first_pass_recommendation": "Use pocc_static_logsum with E_static_DFT before adding vibrational/electronic/magnetic zentropy F_k(T) terms.",
            },
            "warnings": [
                "Degeneracy is a counting/symmetry quantity, not a probability.",
                "Population weights require both degeneracy and G_sigma(T) or Omega_sigma(T, mu_O).",
                "Do not call VASP electronic-smearing TOTEN the thermodynamic F_sigma(T).",
                "Do not silently infer U5; require metadata, Bader/charge, magnetic-polaron evidence, or manual review.",
            ],
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def _csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def _finite_or_none(value: Any) -> float | None:
    number = finite_float(value)
    return None if number is None or not math.isfinite(float(number)) else float(number)


def _jsonish(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _split_labels(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()]


def read_run_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows: dict[str, dict[str, str]] = {}
    for row in _csv_dict(path):
        keys = [
            row.get("run"),
            row.get("run_dir"),
            row.get("path"),
            row.get("structure_path"),
            row.get("config_id"),
            row.get("motif_id"),
            row.get("id"),
        ]
        clean = {key: value for key, value in row.items() if value not in (None, "")}
        for key in keys:
            if key:
                rows[str(key)] = clean
                rows[Path(str(key)).name] = clean
                try:
                    rows[str(Path(str(key)).expanduser().resolve())] = clean
                except OSError:
                    pass
    return rows


def metadata_for_run(run_dir: Path, metadata: dict[str, dict[str, str]]) -> dict[str, str]:
    aliases = {
        str(run_dir),
        run_dir.name,
        str(run_dir.expanduser()),
    }
    try:
        aliases.add(str(run_dir.expanduser().resolve()))
    except OSError:
        pass
    for alias in aliases:
        if alias in metadata:
            return metadata[alias]
    return {}


def parse_vasp_poscar_counts(path: Path) -> tuple[dict[str, int], float | None]:
    """Parse element counts and volume from a VASP POSCAR/CONTCAR-like file."""
    with _open_text(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) < 7:
        raise ValueError(f"Not enough POSCAR/CONTCAR lines in {path}")
    scale = float(lines[1].split()[0])
    lattice = [[float(value) * scale for value in lines[idx].split()[:3]] for idx in range(2, 5)]
    volume = abs(
        lattice[0][0] * (lattice[1][1] * lattice[2][2] - lattice[1][2] * lattice[2][1])
        - lattice[0][1] * (lattice[1][0] * lattice[2][2] - lattice[1][2] * lattice[2][0])
        + lattice[0][2] * (lattice[1][0] * lattice[2][1] - lattice[1][1] * lattice[2][0])
    )
    symbols_line = lines[5].split()
    if all(re.match(r"^[A-Z][a-z]?$", token) for token in symbols_line):
        symbols = symbols_line
        count_tokens = lines[6].split()
    else:
        raise ValueError(
            f"{path} appears to be VASP4-style without element symbols; provide metadata CSV counts instead."
        )
    counts = {symbol: int(round(float(raw))) for symbol, raw in zip(symbols, count_tokens)}
    return counts, volume


def find_vasp_run_dirs(roots: list[Path]) -> list[Path]:
    markers = ("vasprun.xml", "vasprun.xml.gz", "OUTCAR", "OUTCAR.gz")
    found: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue
        if root.is_file():
            found.add(root.parent.resolve())
            continue
        if any((root / marker).exists() for marker in markers):
            found.add(root.resolve())
        for marker in markers:
            for path in root.rglob(marker):
                found.add(path.parent.resolve())
    return sorted(found)


def _structure_path_for_run(run_dir: Path, meta: dict[str, str]) -> Path | None:
    explicit = meta.get("structure_path") or meta.get("contcar") or meta.get("poscar")
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
        candidate = run_dir / explicit
        if candidate.exists():
            return candidate
    for name in ("CONTCAR", "CONTCAR.gz", "POSCAR", "POSCAR.gz"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def _species_counts_from_structure(
    element_counts: dict[str, int],
    meta: dict[str, str],
    *,
    oxygen_sites_per_cation: float,
) -> tuple[dict[str, int], list[str]]:
    warnings: list[str] = []
    n_u_total = _as_int(meta.get("U_total"), element_counts.get("U", 0))
    n_gd = _as_int(meta.get("Gd3") or meta.get("Gd"), element_counts.get("Gd", 0))
    n_o = _as_int(meta.get("O"), element_counts.get("O", 0))
    n_cation = _as_int(meta.get("N_cation"), n_u_total + n_gd)
    n_anion_sites = _as_int(meta.get("N_anion_sites"), round(oxygen_sites_per_cation * n_cation))
    n_vo = _as_int(meta.get("VaO") or meta.get("VO") or meta.get("V_O"), max(n_anion_sites - n_o, 0))
    if n_anion_sites and n_o + n_vo != n_anion_sites:
        warnings.append("anion_count_inconsistent")
    explicit_u5 = meta.get("U5") not in (None, "")
    n_u5 = _as_int(meta.get("U5"), 0)
    n_u4 = _as_int(meta.get("U4"), n_u_total - n_u5)
    if n_u4 + n_u5 != n_u_total:
        warnings.append("uranium_count_inconsistent")
    if not explicit_u5 and n_u_total:
        warnings.append("u5_count_missing_assumed_zero_for_audit")
    return {"U4": n_u4, "U5": n_u5, "Gd3": n_gd, "O": n_o, "VaO": n_vo}, warnings


def ingest_vasp_runs(
    run_dirs: list[Path],
    *,
    metadata: dict[str, dict[str, str]] | None = None,
    phase: str = "fluorite",
    oxygen_sites_per_cation: float = 2.0,
    strict_oxidation: bool = False,
) -> tuple[list[DefectConfiguration], list[dict[str, Any]]]:
    metadata = metadata or {}
    configs: list[DefectConfiguration] = []
    audit_rows: list[dict[str, Any]] = []
    for idx, run_dir in enumerate(sorted(run_dirs), start=1):
        meta = metadata_for_run(run_dir, metadata)
        structure_path = _structure_path_for_run(run_dir, meta)
        warnings: list[str] = []
        element_counts: dict[str, int] = {}
        structure_volume = None
        if structure_path is None:
            warnings.append("missing_structure")
        else:
            try:
                element_counts, structure_volume = parse_vasp_poscar_counts(structure_path)
            except Exception as exc:
                warnings.append(f"structure_parse_failed:{exc}")
        counts, count_warnings = _species_counts_from_structure(
            element_counts,
            meta,
            oxygen_sites_per_cation=oxygen_sites_per_cation,
        )
        warnings.extend(count_warnings)
        calc = parse_calc_folder(run_dir)
        energy = _finite_or_none(meta.get("E_static_eV") or meta.get("energy_eV") or calc.get("energy_eV"))
        volume = _finite_or_none(meta.get("volume_A3") or calc.get("volume_A3") or structure_volume)
        config_id = str(meta.get("config_id") or meta.get("motif_id") or run_dir.name or f"vasp_{idx:04d}")
        degeneracy = _as_float(meta.get("degeneracy"), 1.0)
        oxidation_assignment = (
            meta.get("oxidation_assignment")
            or meta.get("u5_assignment")
            or meta.get("valence_assignment")
            or ""
        )
        if counts.get("U5", 0) and not oxidation_assignment:
            warnings.append("u5_assignment_not_declared")
        if strict_oxidation and counts.get("U4", 0) + counts.get("U5", 0) and not oxidation_assignment:
            warnings.append("strict_oxidation_missing")
        config_metadata: dict[str, Any] = {
            "run_dir": str(run_dir),
            "calc_parser": calc.get("parser_used", ""),
            "calc_source_file": calc.get("source_file", ""),
            "volume_A3": volume,
            "force_rms_eVA": _finite_or_none(calc.get("force_rms_eVA")),
            "force_max_eVA": _finite_or_none(calc.get("force_max_eVA")),
            "source_metadata": meta,
            "ingest_warnings": warnings,
        }
        if oxidation_assignment:
            config_metadata["oxidation_assignment"] = oxidation_assignment
        config = DefectConfiguration(
            config_id=config_id,
            phase=str(meta.get("phase") or phase),
            species_counts=counts,
            sublattice_counts={
                "cation": counts.get("U4", 0) + counts.get("U5", 0) + counts.get("Gd3", 0),
                "anion": counts.get("O", 0) + counts.get("VaO", 0),
            },
            degeneracy=degeneracy,
            degeneracy_type=str(meta.get("degeneracy_type") or "input_or_unity"),
            degeneracy_basis=str(meta.get("degeneracy_basis") or "finite_supercell"),
            E_static_eV=energy,
            energy_kind=str(meta.get("energy_kind") or "E_static_DFT"),  # type: ignore[arg-type]
            motif_labels=_split_labels(meta.get("motif_labels") or meta.get("motif_label") or meta.get("defect_label")),
            motif_features=dict(_jsonish(meta.get("motif_features"), {})),
            structure_path=str(structure_path) if structure_path else None,
            source="vasp_ingest",
            energy_status=str(meta.get("energy_status") or ("static" if energy is not None else "missing")),
            uncertainty_eV=finite_float(meta.get("uncertainty_eV")),
            metadata=config_metadata,
        )
        obs = gduo2_observables(counts)
        if obs["effective_charge"] != 0:
            warnings.append("non_neutral")
        if energy is None:
            warnings.append("missing_energy")
        configs.append(config)
        audit_rows.append(
            {
                "config_id": config_id,
                "run_dir": str(run_dir),
                "structure_path": str(structure_path) if structure_path else "",
                "energy_eV": energy,
                "volume_A3": volume,
                "U4": counts.get("U4", 0),
                "U5": counts.get("U5", 0),
                "Gd3": counts.get("Gd3", 0),
                "O": counts.get("O", 0),
                "VaO": counts.get("VaO", 0),
                "effective_charge": obs["effective_charge"],
                "x_Gd": obs["x_Gd"],
                "delta": obs["delta"],
                "h_U5": obs["h_U5"],
                "degeneracy": degeneracy,
                "oxidation_assignment": oxidation_assignment,
                "motif_labels": ";".join(config.motif_labels),
                "warnings": ";".join(warnings),
            }
        )
    if strict_oxidation:
        missing = [row for row in audit_rows if "strict_oxidation_missing" in str(row.get("warnings", ""))]
        if missing:
            raise ValueError(f"{len(missing)} VASP rows are missing explicit oxidation/U5 assignment.")
    return configs, audit_rows


def load_configurations(path: Path) -> list[DefectConfiguration]:
    if path.suffix.lower() == ".jsonl":
        records = read_jsonl(path)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", payload if isinstance(payload, list) else [])
    else:
        records = _csv_dict(path)
    configs: list[DefectConfiguration] = []
    for idx, row in enumerate(records, start=1):
        counts = row.get("species_counts")
        if isinstance(counts, str):
            counts = json.loads(counts)
        if not isinstance(counts, dict):
            counts = {
                key: _as_int(row.get(key))
                for key in ("U4", "U5", "Gd3", "O", "VaO")
                if row.get(key) not in (None, "")
            }
        counts = {str(key): _as_int(value) for key, value in counts.items()}
        n_cat = counts.get("U4", 0) + counts.get("U5", 0) + counts.get("Gd3", 0)
        n_an = counts.get("O", 0) + counts.get("VaO", 0)
        sub_counts = row.get("sublattice_counts")
        if isinstance(sub_counts, str):
            sub_counts = json.loads(sub_counts)
        if not isinstance(sub_counts, dict):
            sub_counts = {"cation": n_cat, "anion": n_an}
        g_t = row.get("G_eV_T") or row.get("free_energy")
        if isinstance(g_t, str):
            try:
                g_t = json.loads(g_t)
            except json.JSONDecodeError:
                g_t = {}
        parsed_g: dict[float, float] = {}
        if isinstance(g_t, dict):
            for key, value in g_t.items():
                t = finite_float(key)
                g = finite_float(value)
                if t is not None and g is not None:
                    parsed_g[float(t)] = float(g)
        e_static = finite_float(row.get("E_static_eV") or row.get("energy_eV") or row.get("E_eV"))
        motif_labels = row.get("motif_labels") or row.get("motif_label") or []
        if isinstance(motif_labels, str):
            motif_labels = [part for part in motif_labels.replace(";", ",").split(",") if part.strip()]
        motif_features = row.get("motif_features") or row.get("motif_feature_vector") or {}
        if isinstance(motif_features, str):
            try:
                motif_features = json.loads(motif_features)
            except json.JSONDecodeError:
                motif_features = {}
        configs.append(
            DefectConfiguration(
                config_id=str(row.get("config_id") or row.get("motif_id") or row.get("id") or f"config_{idx:04d}"),
                phase=str(row.get("phase") or "fluorite"),
                species_counts=counts,
                sublattice_counts={str(key): _as_int(value) for key, value in sub_counts.items()},
                degeneracy=max(_as_float(row.get("degeneracy"), 1.0), 0.0),
                degeneracy_type=str(row.get("degeneracy_type") or "input"),
                degeneracy_basis=str(row.get("degeneracy_basis") or "finite_supercell"),
                E_static_eV=e_static,
                energy_kind=str(row.get("energy_kind") or "E_static_DFT"),  # type: ignore[arg-type]
                G_eV_T=parsed_g,
                motif_labels=[str(item).strip() for item in motif_labels if str(item).strip()],
                motif_features={str(key): _as_float(value) for key, value in dict(motif_features).items()},
                structure_path=str(row.get("structure_path") or "") or None,
                source=str(row.get("source") or ""),
                energy_status=str(row.get("energy_status") or ("static" if e_static is not None else "missing")),
                uncertainty_eV=finite_float(row.get("uncertainty_eV")),
                metadata={key: value for key, value in row.items() if key not in {"species_counts", "motif_features"}},
            )
        )
    return configs


def effective_charge(counts: dict[str, int], species: dict[str, DefectSpecies] | None = None) -> int:
    model = species or GDUO2_SPECIES
    return int(sum(counts.get(name, 0) * spec.effective_charge for name, spec in model.items()))


def gduo2_observables(counts: dict[str, int]) -> dict[str, float | int]:
    n_cat = counts.get("U4", 0) + counts.get("U5", 0) + counts.get("Gd3", 0)
    n_an = counts.get("O", 0) + counts.get("VaO", 0)
    n_o = counts.get("O", 0)
    n_vo = counts.get("VaO", 0)
    n_gd = counts.get("Gd3", 0)
    n_u5 = counts.get("U5", 0)
    return {
        "N_cation": n_cat,
        "N_anion": n_an,
        "N_O": n_o,
        "N_VaO": n_vo,
        "N_Gd3": n_gd,
        "N_U5": n_u5,
        "effective_charge": effective_charge(counts),
        "x_Gd": n_gd / n_cat if n_cat else math.nan,
        "delta": n_vo / n_cat if n_cat else math.nan,
        "h_U5": n_u5 / n_cat if n_cat else math.nan,
    }


def multinomial_count(total: int, parts: list[int]) -> int:
    """Return total! / prod(parts!) using exact integer arithmetic."""
    if total < 0 or any(part < 0 for part in parts) or sum(parts) != total:
        return 0
    value = math.factorial(total)
    for part in parts:
        value //= math.factorial(part)
    return value


def raw_supercell_degeneracy(counts: dict[str, int]) -> int:
    """Raw decoration count for the Gd-UO2 finite supercell composition."""
    n_cat = counts.get("U4", 0) + counts.get("U5", 0) + counts.get("Gd3", 0)
    n_an = counts.get("O", 0) + counts.get("VaO", 0)
    cat = multinomial_count(n_cat, [counts.get("U4", 0), counts.get("U5", 0), counts.get("Gd3", 0)])
    an = multinomial_count(n_an, [counts.get("O", 0), counts.get("VaO", 0)])
    return cat * an


def _fluorite_supercell_positions(repeat: int = 2) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Return fractional cation/anion positions for a cubic fluorite conventional supercell."""
    if repeat <= 0:
        raise ValueError("repeat must be positive.")
    cation_basis = (
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    )
    anion_basis = tuple(
        (x, y, z)
        for x in (0.25, 0.75)
        for y in (0.25, 0.75)
        for z in (0.25, 0.75)
    )
    cation_positions: list[tuple[float, float, float]] = []
    anion_positions: list[tuple[float, float, float]] = []
    for i, j, k in itertools.product(range(repeat), repeat=3):
        offset = (float(i), float(j), float(k))
        for basis in cation_basis:
            cation_positions.append(tuple((basis[axis] + offset[axis]) / repeat for axis in range(3)))
        for basis in anion_basis:
            anion_positions.append(tuple((basis[axis] + offset[axis]) / repeat for axis in range(3)))
    return cation_positions, anion_positions


def _rounded_frac_key(position: Iterable[float], ndigits: int = 10) -> tuple[float, float, float]:
    return tuple(round(float(value) % 1.0, ndigits) for value in position)  # type: ignore[return-value]


def _fluorite_fm3m_site_permutations(repeat: int = 2, symprec: float = 1.0e-5) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Build parent Fm-3m symmetry permutations on fluorite cation/anion sites.

    The parent cell keeps U and O as distinct species, so the returned
    permutations preserve cation and anion sublattices.  This is the finite
    supercell symmetry group used to collapse raw decorations into POCC-like
    symmetry orbits.
    """
    try:
        import spglib  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional runtime guard
        raise RuntimeError("spglib is required for fluorite Fm-3m symmetry reduction.") from exc

    cation_positions, anion_positions = _fluorite_supercell_positions(repeat)
    positions = cation_positions + anion_positions
    numbers = [92] * len(cation_positions) + [8] * len(anion_positions)
    lattice = [[float(repeat), 0.0, 0.0], [0.0, float(repeat), 0.0], [0.0, 0.0, float(repeat)]]
    symmetry = spglib.get_symmetry((lattice, positions, numbers), symprec=symprec)
    rotations = symmetry["rotations"]
    translations = symmetry["translations"]
    cation_map = {_rounded_frac_key(pos): idx for idx, pos in enumerate(cation_positions)}
    anion_map = {_rounded_frac_key(pos): idx for idx, pos in enumerate(anion_positions)}
    operations: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for rotation, translation in zip(rotations, translations):
        cat_perm: list[int] = []
        an_perm: list[int] = []
        for pos in cation_positions:
            moved = [
                sum(float(rotation[row][col]) * pos[col] for col in range(3)) + float(translation[row])
                for row in range(3)
            ]
            key = _rounded_frac_key(moved)
            if key not in cation_map:
                raise RuntimeError(f"Symmetry operation did not map cation site onto cation sublattice: {key}")
            cat_perm.append(cation_map[key])
        for pos in anion_positions:
            moved = [
                sum(float(rotation[row][col]) * pos[col] for col in range(3)) + float(translation[row])
                for row in range(3)
            ]
            key = _rounded_frac_key(moved)
            if key not in anion_map:
                raise RuntimeError(f"Symmetry operation did not map anion site onto anion sublattice: {key}")
            an_perm.append(anion_map[key])
        op = (tuple(cat_perm), tuple(an_perm))
        if op not in seen:
            seen.add(op)
            operations.append(op)
    return operations


def _compose_permutation(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[index]] for index in range(len(right)))


def _combined_permutation(cat_perm: tuple[int, ...], an_perm: tuple[int, ...]) -> tuple[int, ...]:
    offset = len(cat_perm)
    return tuple(cat_perm) + tuple(offset + item for item in an_perm)


def _greedy_generators(group: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Find a compact generator set for a finite permutation group."""
    if not group:
        return []
    size = len(group[0])
    identity = tuple(range(size))
    target = set(group)
    closure: set[tuple[int, ...]] = {identity}
    generators: list[tuple[int, ...]] = []
    for candidate in group:
        if candidate in closure:
            continue
        trial_generators = [*generators, candidate]
        new_closure: set[tuple[int, ...]] = {identity}
        queue: deque[tuple[int, ...]] = deque([identity])
        while queue:
            current = queue.popleft()
            for generator in trial_generators:
                composed = _compose_permutation(generator, current)
                if composed not in new_closure:
                    new_closure.add(composed)
                    queue.append(composed)
        if len(new_closure) > len(closure):
            generators = trial_generators
            closure = new_closure
        if closure == target:
            break
    return generators


def _apply_orbit_generator(
    key: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    generator: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    cat_perm, an_perm = generator
    gd_sites, u5_sites, vo_sites = key
    return (
        tuple(sorted(cat_perm[index] for index in gd_sites)),
        tuple(sorted(cat_perm[index] for index in u5_sites)),
        tuple(sorted(an_perm[index] for index in vo_sites)),
    )


def _apply_operation_to_key(
    key: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    operation: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    cat_perm, an_perm = operation
    gd_sites, u5_sites, vo_sites = key
    return (
        tuple(sorted(cat_perm[index] for index in gd_sites)),
        tuple(sorted(cat_perm[index] for index in u5_sites)),
        tuple(sorted(an_perm[index] for index in vo_sites)),
    )


def _stabilizer_operations(
    key: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    operations: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [operation for operation in operations if _apply_operation_to_key(key, operation) == key]


def _apply_operation_to_subset(
    subset: tuple[int, ...],
    operation: tuple[tuple[int, ...], tuple[int, ...]],
    *,
    sublattice: str,
) -> tuple[int, ...]:
    perm = operation[0] if sublattice == "cation" else operation[1]
    return tuple(sorted(perm[index] for index in subset))


def _subset_orbit_representatives(
    universe: tuple[int, ...],
    size: int,
    operations: list[tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    sublattice: str,
) -> list[tuple[int, ...]]:
    if size == 0:
        return [()]
    visited: set[tuple[int, ...]] = set()
    representatives: list[tuple[int, ...]] = []
    for subset in itertools.combinations(universe, size):
        subset = tuple(subset)
        if subset in visited:
            continue
        orbit = {
            _apply_operation_to_subset(subset, operation, sublattice=sublattice)
            for operation in operations
        }
        visited.update(orbit)
        representatives.append(min(orbit))
    return sorted(representatives)


def _layered_fluorite_orbit_degeneracy(
    counts: dict[str, int],
    *,
    operations: list[tuple[tuple[int, ...], tuple[int, ...]]],
    motif_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_cat = counts.get("U4", 0) + counts.get("U5", 0) + counts.get("Gd3", 0)
    n_an = counts.get("O", 0) + counts.get("VaO", 0)
    n_gd = counts.get("Gd3", 0)
    n_u5 = counts.get("U5", 0)
    n_vo = counts.get("VaO", 0)
    all_cation_sites = tuple(range(n_cat))
    all_anion_sites = tuple(range(n_an))
    full_order = len(operations)
    partials: list[tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], list[Any]]] = [
        (((), (), ()), operations)
    ]
    if n_gd:
        next_partials = []
        for key, stabilizer in partials:
            for gd_sites in _subset_orbit_representatives(
                all_cation_sites, n_gd, stabilizer, sublattice="cation"
            ):
                new_key = (gd_sites, key[1], key[2])
                next_partials.append((new_key, _stabilizer_operations(new_key, stabilizer)))
        partials = next_partials
    if n_u5:
        next_partials = []
        for key, stabilizer in partials:
            remaining = tuple(site for site in all_cation_sites if site not in key[0])
            for u5_sites in _subset_orbit_representatives(remaining, n_u5, stabilizer, sublattice="cation"):
                new_key = (key[0], u5_sites, key[2])
                next_partials.append((new_key, _stabilizer_operations(new_key, stabilizer)))
        partials = next_partials
    if n_vo:
        next_partials = []
        for key, stabilizer in partials:
            for vo_sites in _subset_orbit_representatives(all_anion_sites, n_vo, stabilizer, sublattice="anion"):
                new_key = (key[0], key[1], vo_sites)
                next_partials.append((new_key, _stabilizer_operations(new_key, stabilizer)))
        partials = next_partials

    rows: list[dict[str, Any]] = []
    orbit_sizes: list[int] = []
    for key, stabilizer in sorted(partials, key=lambda item: item[0]):
        if not stabilizer:
            raise RuntimeError(f"Empty stabilizer for {motif_id} representative {key}")
        orbit_size = full_order // len(stabilizer)
        orbit_sizes.append(orbit_size)
        rows.append(
            {
                "motif_id": motif_id,
                "config_id": f"{motif_id}_orbit_{len(rows) + 1:04d}",
                "g_sigma": orbit_size,
                "representative_Gd3_sites": " ".join(str(site) for site in key[0]),
                "representative_U5_sites": " ".join(str(site) for site in key[1]),
                "representative_VaO_sites": " ".join(str(site) for site in key[2]),
                "symmetry_parent": "fluorite_Fm-3m",
                "symmetry_reduction_basis": "layered_stabilizer_orbit_enumeration",
                "symmetry_group_order": full_order,
            }
        )
    orbit_sum = sum(orbit_sizes)
    return (
        {
            "n_symmetry_distinct_configs": len(orbit_sizes),
            "g_sigma_sum": orbit_sum,
            "g_sigma_min": min(orbit_sizes) if orbit_sizes else 0,
            "g_sigma_max": max(orbit_sizes) if orbit_sizes else 0,
            "g_sigma_mean": orbit_sum / len(orbit_sizes) if orbit_sizes else math.nan,
            "g_sigma_orbit_sizes": ";".join(str(value) for value in sorted(set(orbit_sizes))),
            "symmetry_reduction_status": "exact_layered_stabilizer",
        },
        rows,
    )


def fluorite_fm3m_orbit_degeneracy(
    counts: dict[str, int],
    *,
    repeat: int = 2,
    symprec: float = 1.0e-5,
    max_raw_enumerate: int = 1_000_000,
    layered_when_large: bool = True,
    motif_id: str = "motif",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Enumerate symmetry-distinct fluorite decorations and orbit degeneracies.

    This exact path is intended for low-defect motif scans.  For larger Gd
    counts the raw decoration space becomes too large and should be handed to
    POCC/enumlib/ATAT or sampled by active learning.
    """
    n_cat = counts.get("U4", 0) + counts.get("U5", 0) + counts.get("Gd3", 0)
    n_an = counts.get("O", 0) + counts.get("VaO", 0)
    expected_cat = 4 * repeat**3
    expected_an = 8 * repeat**3
    raw_g = raw_supercell_degeneracy(counts)
    base_summary: dict[str, Any] = {
        "symmetry_parent": "fluorite_Fm-3m",
        "symmetry_reduction_basis": f"{repeat}x{repeat}x{repeat}_conventional_fluorite_supercell",
        "symmetry_tolerance": symprec,
        "g_raw_supercell": raw_g,
        "n_symmetry_distinct_configs": "",
        "g_sigma_sum": "",
        "g_sigma_min": "",
        "g_sigma_max": "",
        "g_sigma_mean": "",
        "g_sigma_orbit_sizes": "",
        "symmetry_reduction_status": "not_run",
    }
    if n_cat != expected_cat or n_an != expected_an:
        base_summary["symmetry_reduction_status"] = (
            f"skipped_site_count_mismatch_expected_{expected_cat}_cat_{expected_an}_anion"
        )
        return base_summary, []
    operations = _fluorite_fm3m_site_permutations(repeat=repeat, symprec=symprec)
    if raw_g > max_raw_enumerate and layered_when_large:
        layered_summary, layered_rows = _layered_fluorite_orbit_degeneracy(
            counts,
            operations=operations,
            motif_id=motif_id,
        )
        base_summary.update(
            {
                **layered_summary,
                "symmetry_group_order": len(operations),
                "symmetry_generator_count": "",
                "symmetry_reduction_status": layered_summary["symmetry_reduction_status"],
            }
        )
        if base_summary.get("g_sigma_sum") != raw_g:
            base_summary["symmetry_reduction_status"] = (
                f"invalid_layered_orbit_sum_{base_summary.get('g_sigma_sum')}_expected_{raw_g}"
            )
        return base_summary, layered_rows
    if raw_g > max_raw_enumerate:
        base_summary["symmetry_reduction_status"] = f"skipped_raw_count_{raw_g}_exceeds_limit_{max_raw_enumerate}"
        return base_summary, []
    combined_group = [_combined_permutation(cat, an) for cat, an in operations]
    combined_generators = _greedy_generators(combined_group)
    generator_set = [
        (tuple(gen[:n_cat]), tuple(index - n_cat for index in gen[n_cat:])) for gen in combined_generators
    ]
    n_gd = counts.get("Gd3", 0)
    n_u5 = counts.get("U5", 0)
    n_vo = counts.get("VaO", 0)
    all_cation_sites = tuple(range(n_cat))
    all_anion_sites = tuple(range(n_an))
    raw_keys: set[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = set()
    for gd_sites in itertools.combinations(all_cation_sites, n_gd):
        remaining_cations = tuple(site for site in all_cation_sites if site not in gd_sites)
        for u5_sites in itertools.combinations(remaining_cations, n_u5):
            for vo_sites in itertools.combinations(all_anion_sites, n_vo):
                raw_keys.add((tuple(gd_sites), tuple(u5_sites), tuple(vo_sites)))
    visited: set[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = set()
    config_rows: list[dict[str, Any]] = []
    orbit_sizes: list[int] = []
    for key in sorted(raw_keys):
        if key in visited:
            continue
        orbit: set[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {key}
        queue = deque([key])
        visited.add(key)
        while queue:
            current = queue.popleft()
            for generator in generator_set:
                moved = _apply_orbit_generator(current, generator)
                if moved not in orbit:
                    orbit.add(moved)
                    if moved in raw_keys:
                        visited.add(moved)
                    queue.append(moved)
        representative = min(orbit)
        orbit_size = len(orbit)
        orbit_sizes.append(orbit_size)
        config_rows.append(
            {
                "motif_id": motif_id,
                "config_id": f"{motif_id}_orbit_{len(config_rows) + 1:04d}",
                "g_sigma": orbit_size,
                "representative_Gd3_sites": " ".join(str(site) for site in representative[0]),
                "representative_U5_sites": " ".join(str(site) for site in representative[1]),
                "representative_VaO_sites": " ".join(str(site) for site in representative[2]),
                "symmetry_parent": "fluorite_Fm-3m",
                "symmetry_reduction_basis": base_summary["symmetry_reduction_basis"],
                "symmetry_group_order": len(operations),
            }
        )
    orbit_sum = sum(orbit_sizes)
    base_summary.update(
        {
            "n_symmetry_distinct_configs": len(orbit_sizes),
            "g_sigma_sum": orbit_sum,
            "g_sigma_min": min(orbit_sizes) if orbit_sizes else 0,
            "g_sigma_max": max(orbit_sizes) if orbit_sizes else 0,
            "g_sigma_mean": orbit_sum / len(orbit_sizes) if orbit_sizes else math.nan,
            "g_sigma_orbit_sizes": ";".join(str(value) for value in sorted(set(orbit_sizes))),
            "symmetry_group_order": len(operations),
            "symmetry_generator_count": len(generator_set),
            "symmetry_reduction_status": "exact",
        }
    )
    if orbit_sum != raw_g:
        base_summary["symmetry_reduction_status"] = f"invalid_orbit_sum_{orbit_sum}_expected_{raw_g}"
    return base_summary, config_rows


def gduo2_charge_neutral_motif_rows(
    *,
    n_cation: int = 32,
    gd_counts: list[int] | None = None,
    oxygen_sites_per_cation: float = 2.0,
    include_parent: bool = False,
) -> list[dict[str, Any]]:
    """Generate all charge-neutral Gd-UO2 compensation macrostates for a cell.

    This is a composition/motif-count grid, not a symmetry-reduced POCC
    enumeration.  It spans the charge-neutral relation
    N_U5 + 2*N_VaO - N_Gd3 = 0 for each requested Gd count.
    """
    if n_cation <= 0:
        raise ValueError("n_cation must be positive.")
    n_anion = int(round(n_cation * oxygen_sites_per_cation))
    if gd_counts is None:
        start = 0 if include_parent else 1
        gd_counts = list(range(start, n_cation + 1))
    rows: list[dict[str, Any]] = []
    for n_gd in sorted(set(int(value) for value in gd_counts)):
        if n_gd < 0 or n_gd > n_cation:
            raise ValueError(f"Gd count {n_gd} is outside [0, {n_cation}].")
        min_vo = max(0, math.ceil((2 * n_gd - n_cation) / 2))
        max_vo = min(n_anion, n_gd // 2)
        for n_vo in range(min_vo, max_vo + 1):
            n_u5 = n_gd - 2 * n_vo
            n_u4 = n_cation - n_gd - n_u5
            if n_u4 < 0 or n_u5 < 0:
                continue
            n_o = n_anion - n_vo
            if n_o < 0:
                continue
            if n_gd == 0 and n_u5 == 0 and n_vo == 0:
                family = "parent"
                motif_label = "UO2_parent"
            elif n_vo == 0:
                family = "u5_compensated"
                motif_label = f"{n_gd}Gd_{n_u5}U5"
            elif n_u5 == 0:
                family = "oxygen_vacancy_compensated"
                motif_label = f"{n_gd}Gd_{n_vo}VaO"
            else:
                family = "mixed_u5_vacancy_compensated"
                motif_label = f"{n_gd}Gd_{n_u5}U5_{n_vo}VaO"
            rows.append(
                {
                    "motif_id": f"{motif_label}_sc{n_cation}",
                    "motif_label": motif_label,
                    "motif_family": family,
                    "N_cation": n_cation,
                    "N_anion_sites": n_anion,
                    "U4": n_u4,
                    "U5": n_u5,
                    "Gd3": n_gd,
                    "O": n_o,
                    "VaO": n_vo,
                    "source": "generated_gduo2_charge_neutral_grid",
                    "notes": "Generated from N_U5 + 2*N_VaO - N_Gd3 = 0.",
                }
            )
    return rows


def _counts_from_motif_row(row: dict[str, Any], *, oxygen_sites_per_cation: float) -> dict[str, int]:
    n_cation = _as_int(row.get("N_cation") or row.get("n_cation") or row.get("cation_sites"), 32)
    n_anion = _as_int(
        row.get("N_anion_sites") or row.get("N_anion") or row.get("anion_sites"),
        int(round(n_cation * oxygen_sites_per_cation)),
    )
    n_gd = _as_int(row.get("N_Gd3") or row.get("Gd3") or row.get("N_Gd") or row.get("Gd"), 0)
    n_u5 = _as_int(row.get("N_U5") or row.get("U5"), 0)
    n_vo = _as_int(row.get("N_VaO") or row.get("VaO") or row.get("VO") or row.get("V_O"), 0)
    n_o = _as_int(row.get("N_O") or row.get("O"), n_anion - n_vo)
    n_u4 = _as_int(row.get("N_U4") or row.get("U4"), n_cation - n_gd - n_u5)
    return {"U4": n_u4, "U5": n_u5, "Gd3": n_gd, "O": n_o, "VaO": n_vo}


def build_degeneracy_table(
    rows: list[dict[str, Any]],
    *,
    oxygen_sites_per_cation: float = 2.0,
    symmetry_reduce_fluorite_fm3m: bool = False,
    fluorite_repeat: int = 2,
    symmetry_tolerance: float = 1.0e-5,
    max_raw_enumerate: int = 1_000_000,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Build an auditable g_k seed table from explicit motif count rows."""
    table: list[dict[str, Any]] = []
    symmetry_config_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        motif_id = str(row.get("motif_id") or row.get("config_id") or row.get("id") or f"motif_{idx:04d}")
        counts = _counts_from_motif_row(row, oxygen_sites_per_cation=oxygen_sites_per_cation)
        obs = gduo2_observables(counts)
        raw_g = raw_supercell_degeneracy(counts)
        symmetry_summary: dict[str, Any] = {}
        exact_config_rows: list[dict[str, Any]] = []
        if symmetry_reduce_fluorite_fm3m:
            symmetry_summary, exact_config_rows = fluorite_fm3m_orbit_degeneracy(
                counts,
                repeat=fluorite_repeat,
                symprec=symmetry_tolerance,
                max_raw_enumerate=max_raw_enumerate,
                motif_id=motif_id,
            )
            symmetry_config_rows.extend(exact_config_rows)
        supplied_g = finite_float(row.get("g_k") or row.get("degeneracy") or row.get("g_sigma"))
        exact_symmetry = str(symmetry_summary.get("symmetry_reduction_status", "")).startswith("exact")
        if supplied_g is None:
            g_value = raw_g
            if exact_symmetry:
                degeneracy_kind = "motif_sum_of_symmetry_orbits"
                degeneracy_status = "symmetry_reduced_audited_sum"
            else:
                degeneracy_kind = "raw_supercell_combinatorial"
                degeneracy_status = "raw_not_symmetry_reduced"
        else:
            g_value = supplied_g
            degeneracy_kind = str(row.get("degeneracy_kind") or row.get("degeneracy_type") or "supplied")
            degeneracy_status = str(row.get("degeneracy_status") or "supplied")
        warnings: list[str] = []
        if obs["effective_charge"] != 0:
            warnings.append("non_neutral")
        if raw_g <= 0:
            warnings.append("invalid_raw_degeneracy")
        if supplied_g is None and not exact_symmetry:
            warnings.append("symmetry_reduction_not_applied")
        if symmetry_summary.get("symmetry_reduction_status", "").startswith("invalid"):
            warnings.append("invalid_symmetry_orbit_sum")
        table.append(
            {
                "motif_id": motif_id,
                "motif_label": row.get("motif_label") or row.get("motif_labels") or motif_id,
                "motif_family": row.get("motif_family") or "",
                "N_cation": obs["N_cation"],
                "N_anion_sites": counts.get("O", 0) + counts.get("VaO", 0),
                "N_U4": counts.get("U4", 0),
                "N_U5": counts.get("U5", 0),
                "N_Gd3": counts.get("Gd3", 0),
                "N_O": counts.get("O", 0),
                "N_VaO": counts.get("VaO", 0),
                "x_Gd": obs["x_Gd"],
                "h_U5": obs["h_U5"],
                "delta": obs["delta"],
                "effective_charge": obs["effective_charge"],
                "charge_neutral": obs["effective_charge"] == 0,
                "g_k": g_value,
                "g_raw_supercell": raw_g,
                "ln_g_k": math.log(float(g_value)) if g_value and float(g_value) > 0 else math.nan,
                "log10_g_k": math.log10(float(g_value)) if g_value and float(g_value) > 0 else math.nan,
                "n_symmetry_distinct_configs": symmetry_summary.get("n_symmetry_distinct_configs", ""),
                "g_sigma_sum": symmetry_summary.get("g_sigma_sum", ""),
                "g_sigma_min": symmetry_summary.get("g_sigma_min", ""),
                "g_sigma_max": symmetry_summary.get("g_sigma_max", ""),
                "g_sigma_mean": symmetry_summary.get("g_sigma_mean", ""),
                "g_sigma_orbit_sizes": symmetry_summary.get("g_sigma_orbit_sizes", ""),
                "degeneracy_kind": degeneracy_kind,
                "degeneracy_basis": row.get("degeneracy_basis") or "finite_supercell",
                "degeneracy_status": degeneracy_status,
                "enumeration_method": row.get("enumeration_method")
                or (
                    "fluorite_fm3m_orbit_enumeration"
                    if exact_symmetry
                    else "composition_multinomial"
                ),
                "symmetry_parent": symmetry_summary.get("symmetry_parent", ""),
                "symmetry_reduction_status": symmetry_summary.get("symmetry_reduction_status", ""),
                "symmetry_group_order": symmetry_summary.get("symmetry_group_order", ""),
                "symmetry_generator_count": symmetry_summary.get("symmetry_generator_count", ""),
                "symmetry_tolerance": row.get("symmetry_tolerance") or symmetry_summary.get("symmetry_tolerance", ""),
                "coverage_warning": row.get("coverage_warning")
                or (
                    ""
                    if exact_symmetry
                    else "raw composition count; symmetry-reduced POCC/enumlib degeneracy not yet applied"
                ),
                "source": row.get("source") or "",
                "notes": row.get("notes") or "",
                "warnings": ";".join(warnings),
            }
        )
    metadata = {
        "schema": f"{SCHEMA}.degeneracy_table",
        "n_motifs": len(table),
        "notes": [
            "g_k is a counting quantity and is not a Boltzmann probability.",
            "Rows without supplied degeneracy use raw finite-supercell multinomial decoration counts.",
            "Raw finite-supercell counts are useful as a seed table but are not symmetry-reduced thermodynamic-limit densities of states.",
            "When fluorite_fm3m symmetry reduction is exact, g_k on motif rows is the sum over symmetry-distinct configuration orbit degeneracies, and symmetry_reduced_configurations.csv lists per-configuration g_sigma.",
        ],
    }
    return table, metadata, symmetry_config_rows


def validate_configurations(configs: list[DefectConfiguration]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad_charge = 0
    missing_energy = 0
    for config in configs:
        obs = gduo2_observables(config.species_counts)
        charge_ok = obs["effective_charge"] == 0
        bad_charge += 0 if charge_ok else 1
        missing_energy += 0 if (config.E_static_eV is not None or config.G_eV_T) else 1
        rows.append(
            {
                "config_id": config.config_id,
                "phase": config.phase,
                "charge_neutral": charge_ok,
                "effective_charge": obs["effective_charge"],
                "x_Gd": obs["x_Gd"],
                "delta": obs["delta"],
                "h_U5": obs["h_U5"],
                "N_cation": obs["N_cation"],
                "N_O": obs["N_O"],
                "N_VaO": obs["N_VaO"],
                "N_Gd3": obs["N_Gd3"],
                "N_U5": obs["N_U5"],
                "degeneracy": config.degeneracy,
                "has_energy": config.E_static_eV is not None or bool(config.G_eV_T),
                "motif_labels": ";".join(config.motif_labels),
                "warnings": ";".join(config_warnings(config)),
            }
        )
    return rows, {
        "schema": f"{SCHEMA}.validation",
        "n_configurations": len(configs),
        "n_non_neutral": bad_charge,
        "n_missing_energy": missing_energy,
        "notes": [
            "Charge neutrality for Gd-UO2 uses N_U5 + 2*N_VaO - N_Gd3 == 0.",
            "Rows with missing energy have known counting metadata but cannot be Boltzmann weighted yet.",
        ],
    }


def config_warnings(config: DefectConfiguration) -> list[str]:
    warnings: list[str] = []
    obs = gduo2_observables(config.species_counts)
    if obs["effective_charge"] != 0:
        warnings.append("non_neutral")
    if config.degeneracy <= 0:
        warnings.append("non_positive_degeneracy")
    if config.E_static_eV is None and not config.G_eV_T:
        warnings.append("missing_energy")
    if config.species_counts.get("U5", 0) and "oxidation_assignment" not in config.metadata:
        warnings.append("u5_assignment_not_declared")
    if config.degeneracy_basis == "finite_supercell":
        warnings.append("finite_supercell_degeneracy")
    return warnings


def _energy_at_temperature(config: DefectConfiguration, temperature: float) -> float | None:
    if config.G_eV_T:
        exact = config.G_eV_T.get(float(temperature))
        if exact is not None:
            return exact
        nearest = min(config.G_eV_T, key=lambda item: abs(float(item) - temperature))
        if abs(float(nearest) - temperature) <= 1.0e-8:
            return config.G_eV_T[nearest]
    return config.E_static_eV


def _site_ideal_entropy_j_molK(configs: list[DefectConfiguration], probs: list[float]) -> float:
    totals = {"U4": 0.0, "U5": 0.0, "Gd3": 0.0, "O": 0.0, "VaO": 0.0}
    n_cat = 0.0
    n_an = 0.0
    for config, prob in zip(configs, probs):
        counts = config.species_counts
        for key in totals:
            totals[key] += prob * counts.get(key, 0)
        n_cat += prob * (counts.get("U4", 0) + counts.get("U5", 0) + counts.get("Gd3", 0))
        n_an += prob * (counts.get("O", 0) + counts.get("VaO", 0))
    if n_cat <= 0:
        return 0.0
    cat_entropy = 0.0
    for key in ("U4", "U5", "Gd3"):
        y = totals[key] / n_cat
        if y > 0:
            cat_entropy -= y * math.log(y)
    an_entropy = 0.0
    if n_an > 0:
        for key in ("O", "VaO"):
            y = totals[key] / n_an
            if y > 0:
                an_entropy -= y * math.log(y)
    return R_J_MOLK * (cat_entropy + 2.0 * an_entropy)


def solve_static_zentropy(
    configs: list[DefectConfiguration],
    *,
    temperatures: list[float],
    mu_o_values: list[float | None],
    group_by_x_gd: bool = True,
    require_neutral: bool = True,
    surface_builder_mode: str = "pocc_static_logsum",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [cfg for cfg in configs if (not require_neutral or effective_charge(cfg.species_counts) == 0)]
    groups: dict[str, list[DefectConfiguration]] = {}
    for cfg in candidates:
        obs = gduo2_observables(cfg.species_counts)
        key = f"x_Gd={obs['x_Gd']:.8g}" if group_by_x_gd else "all"
        groups.setdefault(key, []).append(cfg)

    population_rows: list[dict[str, Any]] = []
    thermo_rows: list[dict[str, Any]] = []
    motif_rows: list[dict[str, Any]] = []
    for group_key, group_configs in sorted(groups.items()):
        for temperature in temperatures:
            for mu_o in mu_o_values:
                active: list[tuple[DefectConfiguration, float, float]] = []
                for cfg in group_configs:
                    g_value = _energy_at_temperature(cfg, temperature)
                    if g_value is None:
                        continue
                    obs = gduo2_observables(cfg.species_counts)
                    omega = g_value - (mu_o or 0.0) * int(obs["N_O"])
                    active.append((cfg, g_value, omega))
                if not active:
                    continue
                omega_min = min(item[2] for item in active)
                beta = 1.0 / (K_B_EV_PER_K * temperature)
                weights = [max(cfg.degeneracy, 0.0) * math.exp(-beta * (omega - omega_min)) for cfg, _, omega in active]
                z_value = sum(weights)
                if z_value <= 0:
                    continue
                probs = [weight / z_value for weight in weights]
                macrostate = f"{group_key}|T={temperature:g}|muO={mu_o if mu_o is not None else 'closed'}"
                pop_entropy = 0.0
                motif_pop: dict[str, float] = {}
                avg: dict[str, float] = {
                    "x_Gd": 0.0,
                    "delta": 0.0,
                    "h_U5": 0.0,
                    "N_O": 0.0,
                    "N_VaO": 0.0,
                    "N_U5": 0.0,
                }
                for (cfg, g_value, omega), prob in zip(active, probs):
                    obs = gduo2_observables(cfg.species_counts)
                    degeneracy = max(cfg.degeneracy, 1.0e-300)
                    pop_entropy -= R_J_MOLK * prob * math.log(max(prob / degeneracy, 1.0e-300))
                    for key in avg:
                        avg[key] += prob * float(obs[key])
                    labels = cfg.motif_labels or ["unlabeled"]
                    for label in labels:
                        motif_pop[label] = motif_pop.get(label, 0.0) + prob / len(labels)
                    population_rows.append(
                        asdict(
                            PopulationRow(
                                macrostate_id=macrostate,
                                T_K=temperature,
                                mu_O_eV=mu_o,
                                config_id=cfg.config_id,
                                motif_labels=";".join(labels),
                                degeneracy=cfg.degeneracy,
                                G_eV=g_value,
                                omega_eV=omega,
                                delta_omega_eV=omega - omega_min,
                                probability=prob,
                                N_cation=int(obs["N_cation"]),
                                N_O=int(obs["N_O"]),
                                N_VaO=int(obs["N_VaO"]),
                                N_Gd3=int(obs["N_Gd3"]),
                                N_U5=int(obs["N_U5"]),
                                x_Gd=float(obs["x_Gd"]),
                                delta=float(obs["delta"]),
                                h_U5=float(obs["h_U5"]),
                            )
                        )
                    )
                omega_ensemble = omega_min - (1.0 / beta) * math.log(z_value)
                surface_kind: SurfaceKind = (
                    "bulk_Omega_from_static_logsum" if mu_o is not None else "bulk_F_from_static_logsum"
                )
                site_entropy = _site_ideal_entropy_j_molK([item[0] for item in active], probs)
                dominant_index = max(range(len(probs)), key=lambda idx: probs[idx])
                dominant = active[dominant_index][0]
                thermo_rows.append(
                    {
                        "macrostate_id": macrostate,
                        "group_key": group_key,
                        "surface_builder_mode": surface_builder_mode,
                        "surface_kind": surface_kind,
                        "configuration_energy_kind": "E_static_DFT",
                        "T_K": temperature,
                        "mu_O_eV": mu_o,
                        "n_states": len(active),
                        "F_phase_eV": omega_ensemble if mu_o is None else math.nan,
                        "F_phase_kJ_mol": omega_ensemble * EV_TO_KJ_MOL if mu_o is None else math.nan,
                        "Omega_eV": omega_ensemble,
                        "Omega_kJ_mol": omega_ensemble * EV_TO_KJ_MOL,
                        "S_population_J_molK": pop_entropy,
                        "S_site_ideal_J_molK": site_entropy,
                        "S_excess_conf_J_molK": pop_entropy - site_entropy,
                        "avg_x_Gd": avg["x_Gd"],
                        "avg_delta": avg["delta"],
                        "avg_h_U5": avg["h_U5"],
                        "avg_N_O": avg["N_O"],
                        "avg_N_VaO": avg["N_VaO"],
                        "avg_N_U5": avg["N_U5"],
                        "dominant_config_id": dominant.config_id,
                        "dominant_motif_labels": ";".join(dominant.motif_labels or ["unlabeled"]),
                        "dominant_probability": probs[dominant_index],
                    }
                )
                for motif, probability in sorted(motif_pop.items()):
                    motif_rows.append(
                        {
                            "macrostate_id": macrostate,
                            "group_key": group_key,
                            "T_K": temperature,
                            "mu_O_eV": mu_o,
                            "motif_label": motif,
                            "probability": probability,
                        }
                    )
    return population_rows, thermo_rows, motif_rows


POP_FIELDS = [field.name for field in PopulationRow.__dataclass_fields__.values()]
THERMO_FIELDS = [
    "macrostate_id",
    "group_key",
    "surface_builder_mode",
    "surface_kind",
    "configuration_energy_kind",
    "T_K",
    "mu_O_eV",
    "n_states",
    "F_phase_eV",
    "F_phase_kJ_mol",
    "Omega_eV",
    "Omega_kJ_mol",
    "S_population_J_molK",
    "S_site_ideal_J_molK",
    "S_excess_conf_J_molK",
    "avg_x_Gd",
    "avg_delta",
    "avg_h_U5",
    "avg_N_O",
    "avg_N_VaO",
    "avg_N_U5",
    "dominant_config_id",
    "dominant_motif_labels",
    "dominant_probability",
]
MOTIF_FIELDS = ["macrostate_id", "group_key", "T_K", "mu_O_eV", "motif_label", "probability"]
VALIDATION_FIELDS = [
    "config_id",
    "phase",
    "charge_neutral",
    "effective_charge",
    "x_Gd",
    "delta",
    "h_U5",
    "N_cation",
    "N_O",
    "N_VaO",
    "N_Gd3",
    "N_U5",
    "degeneracy",
    "has_energy",
    "motif_labels",
    "warnings",
]
VASP_INGEST_FIELDS = [
    "config_id",
    "run_dir",
    "structure_path",
    "energy_eV",
    "volume_A3",
    "U4",
    "U5",
    "Gd3",
    "O",
    "VaO",
    "effective_charge",
    "x_Gd",
    "delta",
    "h_U5",
    "degeneracy",
    "oxidation_assignment",
    "motif_labels",
    "warnings",
]
DEGENERACY_FIELDS = [
    "motif_id",
    "motif_label",
    "motif_family",
    "N_cation",
    "N_anion_sites",
    "N_U4",
    "N_U5",
    "N_Gd3",
    "N_O",
    "N_VaO",
    "x_Gd",
    "h_U5",
    "delta",
    "effective_charge",
    "charge_neutral",
    "g_k",
    "g_raw_supercell",
    "ln_g_k",
    "log10_g_k",
    "n_symmetry_distinct_configs",
    "g_sigma_sum",
    "g_sigma_min",
    "g_sigma_max",
    "g_sigma_mean",
    "g_sigma_orbit_sizes",
    "degeneracy_kind",
    "degeneracy_basis",
    "degeneracy_status",
    "enumeration_method",
    "symmetry_parent",
    "symmetry_reduction_status",
    "symmetry_group_order",
    "symmetry_generator_count",
    "symmetry_tolerance",
    "coverage_warning",
    "source",
    "notes",
    "warnings",
]
SYMMETRY_CONFIG_FIELDS = [
    "motif_id",
    "config_id",
    "g_sigma",
    "representative_Gd3_sites",
    "representative_U5_sites",
    "representative_VaO_sites",
    "symmetry_parent",
    "symmetry_reduction_basis",
    "symmetry_group_order",
]
MAGNETIC_INIT_FIELDS = [
    "motif_id",
    "config_id",
    "spin_config_id",
    "g_sigma",
    "spin_policy",
    "spin_variant",
    "poscar_element_order",
    "potcar_order_guard",
    "ldau_order_guard",
    "recommended_ldau",
    "representative_Gd3_sites",
    "representative_U5_sites",
    "representative_VaO_sites",
    "n_Gd3_up",
    "n_Gd3_down",
    "n_U4_up",
    "n_U4_down",
    "n_U5_up",
    "n_U5_down",
    "net_initial_moment",
    "magmom_poscar_order",
    "time_reversal_of",
    "notes",
]


def _parse_site_indices(value: Any) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(sorted(int(item) for item in value))
    text = str(value).strip()
    if not text:
        return ()
    text = text.replace(",", " ")
    return tuple(sorted(int(part) for part in text.split() if part.strip()))


def _format_moment(value: float) -> str:
    if abs(value) < 1.0e-12:
        return "0"
    text = f"{value:.6g}"
    return text if text.startswith("-") else f"+{text}"


def _balanced_site_signs(sites: tuple[int, ...], *, flip: bool = False) -> dict[int, int]:
    """Return a deterministic low-net-moment sign pattern for a site set.

    This is an initialization policy, not a magnetic degeneracy enumeration.
    It keeps U4 and U5 families AFM-like by alternating signs over sorted
    parent fluorite cation site indices.  Time reversal is represented by
    flipping all signs.
    """
    signs: dict[int, int] = {}
    for idx, site in enumerate(sorted(sites)):
        sign = 1 if idx % 2 == 0 else -1
        signs[site] = -sign if flip else sign
    return signs


def build_magnetic_initialization_rows(
    config_rows: list[dict[str, Any]],
    *,
    n_cation: int = 32,
    n_anion: int = 64,
    gd_moment: float = 7.0,
    u4_moment: float = 2.0,
    u5_moment: float = 1.0,
    include_time_reversal: bool = True,
    spin_policy: str = "u4_u5_afm_like_gd_fm",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build explicit VASP MAGMOM initialization rows for POCC configs.

    The returned rows split each chemical POCC configuration into one or two
    magnetic initialization variants.  They do not multiply chemical g_sigma by
    a magnetic degeneracy by default; downstream static-logsum should treat
    computed spin variants as explicit sigma states once energies are known.
    """
    if n_cation <= 0 or n_anion < 0:
        raise ValueError("n_cation must be positive and n_anion must be non-negative.")
    if spin_policy != "u4_u5_afm_like_gd_fm":
        raise ValueError(f"Unsupported spin policy: {spin_policy}")
    all_cation_sites = tuple(range(n_cation))
    rows: list[dict[str, Any]] = []
    variants = [(False, "gd_fm_up")]
    if include_time_reversal:
        variants.append((True, "gd_fm_down_time_reverse"))
    for idx, row in enumerate(config_rows, start=1):
        motif_id = str(row.get("motif_id") or "motif")
        config_id = str(row.get("config_id") or f"{motif_id}_config_{idx:04d}")
        gd_sites = _parse_site_indices(row.get("representative_Gd3_sites"))
        u5_sites = _parse_site_indices(row.get("representative_U5_sites"))
        vo_sites = _parse_site_indices(row.get("representative_VaO_sites"))
        if set(gd_sites) & set(u5_sites):
            raise ValueError(f"{config_id} has overlapping Gd3 and U5 sites.")
        if any(site < 0 or site >= n_cation for site in (*gd_sites, *u5_sites)):
            raise ValueError(f"{config_id} has cation site outside [0, {n_cation}).")
        if any(site < 0 or site >= n_anion for site in vo_sites):
            raise ValueError(f"{config_id} has anion site outside [0, {n_anion}).")
        u_sites = tuple(site for site in all_cation_sites if site not in set(gd_sites))
        u4_sites = tuple(site for site in u_sites if site not in set(u5_sites))
        oxygen_count = n_anion - len(vo_sites)
        for flip, variant in variants:
            gd_sign = -1 if flip else 1
            u4_signs = _balanced_site_signs(u4_sites, flip=flip)
            u5_signs = _balanced_site_signs(u5_sites, flip=flip)
            gd_moments = [gd_sign * gd_moment for _site in gd_sites]
            u_moments: list[float] = []
            for site in u_sites:
                if site in u5_signs:
                    u_moments.append(u5_signs[site] * u5_moment)
                else:
                    u_moments.append(u4_signs[site] * u4_moment)
            oxygen_moments = [0.0] * oxygen_count
            moments = gd_moments + u_moments + oxygen_moments
            spin_config_id = f"{config_id}_{variant}"
            time_reversal_of = "" if not flip else f"{config_id}_gd_fm_up"
            rows.append(
                {
                    "motif_id": motif_id,
                    "config_id": config_id,
                    "spin_config_id": spin_config_id,
                    "g_sigma": row.get("g_sigma") or row.get("degeneracy") or "",
                    "spin_policy": spin_policy,
                    "spin_variant": variant,
                    "poscar_element_order": "Gd U O",
                    "potcar_order_guard": "POTCAR must be concatenated as Gd, U, O for these MAGMOM/LDAU arrays.",
                    "ldau_order_guard": "LDAUL/LDAUU/LDAUJ are in POSCAR species order: Gd U O.",
                    "recommended_ldau": "LDAUTYPE=1; LDAUL=3 3 -1; LDAUU=6.0 4.0 0.0; LDAUJ=0.0 0.0 0.0; LMAXMIX=6; LASPH=.TRUE.",
                    "representative_Gd3_sites": " ".join(str(site) for site in gd_sites),
                    "representative_U5_sites": " ".join(str(site) for site in u5_sites),
                    "representative_VaO_sites": " ".join(str(site) for site in vo_sites),
                    "n_Gd3_up": sum(1 for value in gd_moments if value > 0),
                    "n_Gd3_down": sum(1 for value in gd_moments if value < 0),
                    "n_U4_up": sum(1 for site in u4_sites if u4_signs[site] > 0),
                    "n_U4_down": sum(1 for site in u4_sites if u4_signs[site] < 0),
                    "n_U5_up": sum(1 for site in u5_sites if u5_signs[site] > 0),
                    "n_U5_down": sum(1 for site in u5_sites if u5_signs[site] < 0),
                    "net_initial_moment": sum(moments),
                    "magmom_poscar_order": " ".join(_format_moment(value) for value in moments),
                    "time_reversal_of": time_reversal_of,
                    "notes": (
                        "Chemical g_sigma is not multiplied by spin degeneracy here; "
                        "spin variants are explicit initialization states for static-F screening."
                    ),
                }
            )
    metadata = {
        "schema": f"{SCHEMA}.magnetic_initialization_table",
        "n_input_configurations": len(config_rows),
        "n_spin_initializations": len(rows),
        "spin_policy": spin_policy,
        "poscar_element_order": "Gd U O",
        "recommended_ldau_order": "Gd U O",
        "notes": [
            "This table initializes U4 and U5 as AFM-like low-net-moment subfamilies and Gd3 as FM.",
            "Time-reversal variants are written as explicit rows when requested.",
            "Use computed static energies to decide whether magnetic variants are distinct thermodynamic states before assigning magnetic degeneracy.",
        ],
    }
    return rows, metadata


def _parse_grid(values: list[str] | None, *, default: list[float | None]) -> list[float | None]:
    if not values:
        return list(default)
    out: list[float | None] = []
    for raw in values:
        text = str(raw).strip()
        if text.lower() in {"none", "closed"}:
            out.append(None)
        elif ":" in text:
            start, stop, step = [float(part) for part in text.split(":")]
            current = start
            if step == 0:
                raise ValueError("Grid step cannot be zero.")
            if step > 0:
                while current <= stop + abs(step) * 1.0e-9:
                    out.append(round(current, 12))
                    current += step
            else:
                while current >= stop - abs(step) * 1.0e-9:
                    out.append(round(current, 12))
                    current += step
        else:
            out.append(float(text))
    return out


def _add_common_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ensemble", type=Path, required=True, help="DefectConfiguration JSONL/JSON/CSV.")
    parser.add_argument("--outdir", type=Path, default=Path("pocc_zentropy_defects"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pocc-zentropy-defects",
        description="POCC/zentropy defect thermodynamic engine for Gd-doped UO2 and related fluorite oxides.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template", help="Write the default Gd-UO2 defect-engine YAML/JSON template.")
    template.add_argument("--output", type=Path, default=Path("gd_uo2.defect_engine.json"))

    validate = sub.add_parser("validate", help="Validate charge, degeneracy, and energy metadata.")
    _add_common_input(validate)

    degeneracy = sub.add_parser("degeneracy-table", help="Build an auditable motif degeneracy g_k table.")
    degeneracy.add_argument("--motif-csv", type=Path, help="CSV with motif_id and species/site counts.")
    degeneracy.add_argument("--outdir", type=Path, default=Path("pocc_zentropy_degeneracy"))
    degeneracy.add_argument("--output", type=Path, default=Path("motif_degeneracy_gk.csv"))
    degeneracy.add_argument("--oxygen-sites-per-cation", type=float, default=2.0)
    degeneracy.add_argument(
        "--gduo2-all-charge-neutral",
        action="store_true",
        help="Generate the Gd-UO2 charge-neutral compensation motif grid instead of reading --motif-csv.",
    )
    degeneracy.add_argument("--n-cation", type=int, default=32, help="Cation sites for generated Gd-UO2 grids.")
    degeneracy.add_argument(
        "--gd-count",
        action="append",
        type=int,
        default=[],
        help="Gd count to include in the generated grid. Repeatable. Default: all 1..N_cation.",
    )
    degeneracy.add_argument(
        "--include-parent",
        action="store_true",
        help="Include the undoped UO2 parent row in generated Gd-UO2 grids.",
    )
    degeneracy.add_argument(
        "--fluorite-fm3m-symmetry-reduce",
        action="store_true",
        help="Exactly reduce tractable 2x2x2 fluorite motif rows by parent Fm-3m symmetry and write per-orbit g_sigma rows.",
    )
    degeneracy.add_argument(
        "--fluorite-repeat",
        type=int,
        default=2,
        help="Conventional fluorite supercell repeat for symmetry reduction. repeat=2 gives 32 cation and 64 anion sites.",
    )
    degeneracy.add_argument("--symmetry-tolerance", type=float, default=1.0e-5)
    degeneracy.add_argument(
        "--max-raw-enumerate",
        type=int,
        default=1_000_000,
        help="Skip exact symmetry reduction when raw decoration count exceeds this limit.",
    )

    magnetic = sub.add_parser(
        "magnetic-init-table",
        help="Write spin-aware MAGMOM initialization rows on top of POCC symmetry configurations.",
    )
    magnetic.add_argument(
        "--config-csv",
        type=Path,
        required=True,
        help="CSV with motif_id/config_id and representative_Gd3/U5/VaO site columns.",
    )
    magnetic.add_argument("--outdir", type=Path, default=Path("pocc_zentropy_magnetic_init"))
    magnetic.add_argument("--output", type=Path, default=Path("magnetic_initialization_table.csv"))
    magnetic.add_argument("--n-cation", type=int, default=32)
    magnetic.add_argument("--n-anion", type=int, default=64)
    magnetic.add_argument("--gd-moment", type=float, default=7.0)
    magnetic.add_argument("--u4-moment", type=float, default=2.0)
    magnetic.add_argument("--u5-moment", type=float, default=1.0)
    magnetic.add_argument(
        "--single-time-direction",
        action="store_true",
        help="Only write the Gd-up/U-pattern row; default also writes its time-reversed partner.",
    )

    ingest = sub.add_parser("ingest-vasp", help="Scan VASP motif folders into a POCC defect ensemble JSONL.")
    ingest.add_argument("--root", type=Path, action="append", default=[], help="Root searched recursively for VASP runs.")
    ingest.add_argument("--run-dir", type=Path, action="append", default=[], help="Explicit VASP run directory.")
    ingest.add_argument("--metadata-csv", type=Path, help="CSV with run/config_id/counts/degeneracy/oxidation metadata.")
    ingest.add_argument("--outdir", type=Path, default=Path("pocc_zentropy_defects_ingest"))
    ingest.add_argument("--output", type=Path, default=Path("ensemble.jsonl"))
    ingest.add_argument("--phase", default="fluorite")
    ingest.add_argument("--oxygen-sites-per-cation", type=float, default=2.0)
    ingest.add_argument(
        "--strict-oxidation",
        action="store_true",
        help="Fail when U-bearing rows lack explicit oxidation/U5 assignment metadata.",
    )

    solve = sub.add_parser(
        "solve-static",
        aliases=("static-logsum", "solve-static-logsum"),
        help="Compute the cheap static-logsum F/Omega surface from E_static_DFT before expensive zentropy terms.",
    )
    _add_common_input(solve)
    solve.add_argument("--temperature", action="append", default=[], help="T in K or start:stop:step.")
    solve.add_argument("--mu-o", action="append", default=[], help="mu_O in eV/O, grid start:stop:step, or closed.")
    solve.add_argument("--no-group-by-x-gd", action="store_true")
    solve.add_argument("--allow-non-neutral", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.command == "template":
        write_json(args.output.resolve(), gduo2_default_config())
        print(f"Wrote Gd-UO2 defect-engine template: {args.output.resolve()}")
        return {"output": str(args.output.resolve())}

    if args.command == "degeneracy-table":
        if args.gduo2_all_charge_neutral:
            motif_rows = gduo2_charge_neutral_motif_rows(
                n_cation=args.n_cation,
                gd_counts=args.gd_count or None,
                oxygen_sites_per_cation=args.oxygen_sites_per_cation,
                include_parent=args.include_parent,
            )
        elif args.motif_csv:
            motif_rows = _csv_dict(args.motif_csv.resolve())
        else:
            raise ValueError("Provide --motif-csv or --gduo2-all-charge-neutral.")
        table, table_metadata, symmetry_config_rows = build_degeneracy_table(
            motif_rows,
            oxygen_sites_per_cation=args.oxygen_sites_per_cation,
            symmetry_reduce_fluorite_fm3m=args.fluorite_fm3m_symmetry_reduce,
            fluorite_repeat=args.fluorite_repeat,
            symmetry_tolerance=args.symmetry_tolerance,
            max_raw_enumerate=args.max_raw_enumerate,
        )
        outdir = args.outdir.resolve()
        table_path = args.output if args.output.is_absolute() else outdir / args.output
        write_csv(table_path, table, DEGENERACY_FIELDS)
        if symmetry_config_rows:
            write_csv(outdir / "symmetry_reduced_configurations.csv", symmetry_config_rows, SYMMETRY_CONFIG_FIELDS)
        write_json(outdir / "motif_degeneracy_gk.metadata.json", table_metadata)
        print(f"Motifs          : {len(table)}")
        if symmetry_config_rows:
            print(f"Symmetry configs: {len(symmetry_config_rows)}")
        print(f"Degeneracy table: {table_path}")
        return table_metadata

    if args.command == "magnetic-init-table":
        config_rows = _csv_dict(args.config_csv.resolve())
        rows, metadata = build_magnetic_initialization_rows(
            config_rows,
            n_cation=args.n_cation,
            n_anion=args.n_anion,
            gd_moment=args.gd_moment,
            u4_moment=args.u4_moment,
            u5_moment=args.u5_moment,
            include_time_reversal=not args.single_time_direction,
        )
        outdir = args.outdir.resolve()
        table_path = args.output if args.output.is_absolute() else outdir / args.output
        write_csv(table_path, rows, MAGNETIC_INIT_FIELDS)
        write_json(outdir / "magnetic_initialization_table.metadata.json", metadata)
        print(f"Input configs        : {len(config_rows)}")
        print(f"Spin initializations : {len(rows)}")
        print(f"Magnetic init table  : {table_path}")
        return metadata

    if args.command == "ingest-vasp":
        metadata = read_run_metadata(args.metadata_csv.resolve() if args.metadata_csv else None)
        explicit = [path.expanduser().resolve() for path in args.run_dir]
        discovered = find_vasp_run_dirs(args.root) if args.root else []
        run_dirs = sorted({*explicit, *discovered})
        if not run_dirs:
            raise ValueError("No VASP run directories found. Use --root or --run-dir.")
        configs, ingest_rows = ingest_vasp_runs(
            run_dirs,
            metadata=metadata,
            phase=args.phase,
            oxygen_sites_per_cation=args.oxygen_sites_per_cation,
            strict_oxidation=args.strict_oxidation,
        )
        outdir = args.outdir.resolve()
        ensemble_path = args.output if args.output.is_absolute() else outdir / args.output
        write_jsonl(ensemble_path, [asdict(config) for config in configs])
        write_csv(outdir / "vasp_ingest_audit.csv", ingest_rows, VASP_INGEST_FIELDS)
        validation_rows, validation_metadata = validate_configurations(configs)
        write_csv(outdir / "configuration_audit.csv", validation_rows, VALIDATION_FIELDS)
        metadata_payload = {
            "schema": f"{SCHEMA}.vasp_ingest",
            "inputs": {
                "roots": [str(path.expanduser()) for path in args.root],
                "run_dirs": [str(path) for path in args.run_dir],
                "metadata_csv": str(args.metadata_csv.resolve()) if args.metadata_csv else "",
            },
            "outputs": {
                "ensemble": str(ensemble_path),
                "vasp_ingest_audit": str(outdir / "vasp_ingest_audit.csv"),
                "configuration_audit": str(outdir / "configuration_audit.csv"),
            },
            "n_runs": len(run_dirs),
            "validation": validation_metadata,
            "notes": [
                "VASP POSCAR/CONTCAR can provide element counts, but U4/U5 oxidation must come from metadata, Bader, magnetic-polaron analysis, or manual review.",
                "Rows with u5_count_missing_assumed_zero_for_audit are counting placeholders until oxidation metadata is supplied.",
            ],
        }
        write_json(outdir / "vasp_ingest_metadata.json", metadata_payload)
        print(f"VASP runs       : {len(run_dirs)}")
        print(f"Ensemble        : {ensemble_path}")
        print(f"Ingest audit    : {outdir / 'vasp_ingest_audit.csv'}")
        return metadata_payload

    configs = load_configurations(args.ensemble.resolve())
    validation_rows, validation_metadata = validate_configurations(configs)
    outdir = args.outdir.resolve()
    write_csv(outdir / "configuration_audit.csv", validation_rows, VALIDATION_FIELDS)
    write_json(outdir / "configuration_audit.json", {"metadata": validation_metadata, "rows": validation_rows})

    if args.command == "validate":
        print(f"Configurations : {len(configs)}")
        print(f"Non-neutral    : {validation_metadata['n_non_neutral']}")
        print(f"Missing energy : {validation_metadata['n_missing_energy']}")
        print(f"Wrote audit    : {outdir / 'configuration_audit.csv'}")
        return validation_metadata

    temperatures = [float(value) for value in _parse_grid(args.temperature, default=[1000.0]) if value is not None]
    mu_values = _parse_grid(args.mu_o, default=[None])
    population_rows, thermo_rows, motif_rows = solve_static_zentropy(
        configs,
        temperatures=temperatures,
        mu_o_values=mu_values,
        group_by_x_gd=not args.no_group_by_x_gd,
        require_neutral=not args.allow_non_neutral,
    )
    write_csv(outdir / "population_vector.csv", population_rows, POP_FIELDS)
    write_csv(outdir / "static_logsum_surface.csv", thermo_rows, THERMO_FIELDS)
    write_csv(outdir / "zentropy_surface.csv", thermo_rows, THERMO_FIELDS)
    write_csv(outdir / "motif_populations.csv", motif_rows, MOTIF_FIELDS)
    metadata = {
        "schema": SCHEMA,
        "surface_builder_mode": "pocc_static_logsum",
        "surface_kind": "bulk_F_from_static_logsum_or_bulk_Omega_from_static_logsum",
        "configuration_energy_kind": "E_static_DFT",
        "inputs": {"ensemble": str(args.ensemble.resolve())},
        "outputs": {
            "configuration_audit": str(outdir / "configuration_audit.csv"),
            "population_vector": str(outdir / "population_vector.csv"),
            "static_logsum_surface": str(outdir / "static_logsum_surface.csv"),
            "zentropy_surface": str(outdir / "zentropy_surface.csv"),
            "motif_populations": str(outdir / "motif_populations.csv"),
        },
        "temperatures_K": temperatures,
        "mu_O_eV": mu_values,
        "n_population_rows": len(population_rows),
        "n_surface_rows": len(thermo_rows),
        "notes": [
            "This is the quick static-F layer: F_k(T) is approximated by E_static_DFT before adding vibrational/electronic/magnetic zentropy terms.",
            "Degeneracy is retained as counting metadata and only becomes a Boltzmann/logsum weight inside the static surface solve.",
            "S_population, S_site_ideal, and S_excess_conf are reported separately as a guard against ideal-mixing overuse.",
            "zentropy_surface.csv is kept as a compatibility alias; static_logsum_surface.csv is the explicit first-pass output.",
        ],
    }
    write_json(outdir / "pocc_zentropy_metadata.json", metadata)
    print(f"Population rows : {len(population_rows)}")
    print(f"Surface rows    : {len(thermo_rows)}")
    print(f"Wrote surface   : {outdir / 'static_logsum_surface.csv'}")
    return metadata


if __name__ == "__main__":
    main()
