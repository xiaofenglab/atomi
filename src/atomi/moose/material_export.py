"""Export QHA/MD thermodynamic data as MOOSE material-property inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import re
from bisect import bisect_left
from pathlib import Path
from typing import Any

try:
    from atomi import __version__
except Exception:  # pragma: no cover - import fallback for direct script execution
    __version__ = "unknown"


MOOSE_FIELDS = [
    "T_K",
    "k_W_mK",
    "k_std_W_mK",
    "Cp_J_kgK",
    "Cp_std_J_kgK",
    "rho_kg_m3",
    "rho_std_kg_m3",
    "alpha_1_K",
    "alpha_std_1_K",
    "dilatation",
    "E_Pa",
    "E_std_Pa",
    "nu",
    "nu_std",
    "K_Pa",
    "K_std_Pa",
    "G_Pa",
    "G_std_Pa",
    "source_tag",
]

REQUIRED_DEFAULT = ["T_K", "k_W_mK", "Cp_J_kgK", "rho_kg_m3", "E_Pa", "nu"]
STRUCTURAL_REQUIRED = ("alpha_1_K", "dilatation")
TDB_MERGE_FIELDS = ["Cp_J_kgK"]

ATOMIC_MASSES_G_MOL = {
    "H": 1.00784,
    "C": 12.011,
    "N": 14.0067,
    "O": 15.999,
    "F": 18.998403163,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.0983,
    "Ca": 40.078,
    "Fe": 55.845,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zr": 91.224,
    "Mo": 95.95,
    "Gd": 157.25,
    "U": 238.02891,
    "Pu": 244.0,
}

AVOGADRO = 6.02214076e23


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def moose_function_prefix(material: str) -> str:
    """Return a MOOSE-safe lowercase prefix for generated material functions."""
    prefix = re.sub(r"[^0-9A-Za-z]+", "_", material.strip()).strip("_").lower()
    return prefix or "material"


def moose_function_map(material: str, function_prefix: str | None = None) -> dict[str, str]:
    """Map MOOSE material-table fields to generated Function names."""
    prefix = function_prefix or moose_function_prefix(material)
    return {
        "k_W_mK": f"{prefix}_k",
        "Cp_J_kgK": f"{prefix}_Cp",
        "rho_kg_m3": f"{prefix}_rho",
        "alpha_1_K": f"{prefix}_alpha",
        "dilatation": f"{prefix}_dilatation",
        "E_Pa": f"{prefix}_E",
        "nu": f"{prefix}_nu",
        "K_Pa": f"{prefix}_K",
        "G_Pa": f"{prefix}_G",
    }


def formula_mass_g_mol(formula: str) -> float:
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    if not tokens:
        raise ValueError(f"Could not parse chemical formula {formula!r}")
    consumed = "".join(element + count for element, count in tokens)
    if consumed != formula:
        raise ValueError(
            f"Formula parser only supports simple formulas such as UO2; got {formula!r}"
        )
    total = 0.0
    for element, count_text in tokens:
        if element not in ATOMIC_MASSES_G_MOL:
            raise ValueError(
                f"No built-in atomic mass for {element!r}; pass --molar-mass-g-mol explicitly"
            )
        count = float(count_text) if count_text else 1.0
        total += ATOMIC_MASSES_G_MOL[element] * count
    return total


def formula_atom_fractions(formula: str) -> dict[str, float]:
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    counts: dict[str, float] = {}
    for element, count_text in tokens:
        counts[element.upper()] = counts.get(element.upper(), 0.0) + (
            float(count_text) if count_text else 1.0
        )
    total = sum(counts.values())
    if total <= 0.0:
        return {}
    return {element: count / total for element, count in counts.items()}


def sorted_series(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = [(float(x), float(y)) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    return sorted(clean, key=lambda item: item[0])


def interpolate(points: list[tuple[float, float]], x: float) -> float | None:
    points = sorted_series(points)
    if not points:
        return None
    if len(points) == 1:
        return points[0][1]
    if x < points[0][0] or x > points[-1][0]:
        return None
    idx = bisect_left([point[0] for point in points], x)
    if idx < len(points) and abs(points[idx][0] - x) < 1.0e-12:
        return points[idx][1]
    if idx == 0 or idx == len(points):
        return None
    x0, y0 = points[idx - 1]
    x1, y1 = points[idx]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def derivative_series(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points = sorted_series(points)
    if len(points) < 2:
        return []
    derived = []
    for idx, (temp, value) in enumerate(points):
        if idx == 0:
            t0, v0 = points[0]
            t1, v1 = points[1]
        elif idx == len(points) - 1:
            t0, v0 = points[-2]
            t1, v1 = points[-1]
        else:
            t0, v0 = points[idx - 1]
            t1, v1 = points[idx + 1]
        if t1 == t0 or value == 0.0:
            continue
        derived.append((temp, (v1 - v0) / (t1 - t0) / value))
    return derived


def nearest_or_interp(points: list[tuple[float, float]], x: float) -> float | None:
    value = interpolate(points, x)
    if value is not None:
        return value
    points = sorted_series(points)
    if not points:
        return None
    return min(points, key=lambda item: abs(item[0] - x))[1]


def first_float(row: dict[str, Any], names: list[str]) -> float | None:
    for name in names:
        value = finite_float(row.get(name))
        if value is not None:
            return value
    return None


def load_metadata(qha_md_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in (
        "hybrid_cp_entropy_metadata.json",
        "normalization_metadata.json",
        "temperature_range_metadata.json",
        "qha_low_t_splice_metadata.json",
    ):
        path = qha_md_dir / name
        if path.exists():
            try:
                metadata[name] = read_json(path)
            except json.JSONDecodeError:
                metadata[name] = {"parse_error": True}
    return metadata


def load_thermo_grid(qha_md_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hybrid = qha_md_dir / "hybrid_cp_entropy.csv"
    thermo_grid = qha_md_dir / "thermo_functions_grid.csv"
    volume_lattice = qha_md_dir / "hybrid_volume_lattice.csv"
    sources: dict[str, Any] = {"qha_md_dir": str(qha_md_dir)}

    if hybrid.exists():
        rows = []
        for raw in read_csv(hybrid):
            temp = finite_float(raw.get("T_K"))
            cp = finite_float(raw.get("Cp"))
            if temp is None:
                continue
            rows.append(
                {
                    "T_K": temp,
                    "Cp_J_molK": cp,
                    "Cp_source": raw.get("Cp_source", ""),
                    "source_file": hybrid.name,
                }
            )
        sources["primary_thermal_file"] = hybrid.name
    elif thermo_grid.exists():
        rows = []
        for raw in read_csv(thermo_grid):
            temp = finite_float(raw.get("T_K"))
            cp = first_float(
                raw,
                [
                    "Cp_used_for_integration_J_per_mol_UO2_K",
                    "Cp_from_dH_J_per_mol_UO2_K",
                    "Cp_fluct_J_per_mol_UO2_K",
                    "Cp",
                ],
            )
            if temp is None:
                continue
            row = {
                "T_K": temp,
                "Cp_J_molK": cp,
                "density_g_cm3": finite_float(raw.get("density_fit_g_cm3")),
                "alpha_1_K": first_float(raw, ["alpha_L_1_per_K", "alpha_1_K"]),
                "dilatation_source_value": first_float(raw, ["a_fit_A", "V_fit_A3"]),
                "V_A3": first_float(raw, ["V_target_cell_A3", "V_fit_A3"]),
                "source_file": thermo_grid.name,
            }
            cp_p16 = finite_float(raw.get("Cp_grid_p16"))
            cp_p84 = finite_float(raw.get("Cp_grid_p84"))
            if cp_p16 is not None and cp_p84 is not None:
                row["Cp_std_J_molK"] = abs(cp_p84 - cp_p16) / 2.0
            rows.append(row)
        sources["primary_thermal_file"] = thermo_grid.name
    else:
        raise SystemExit(
            f"No thermo_qha_md material source found in {qha_md_dir}. Expected "
            "hybrid_cp_entropy.csv or thermo_functions_grid.csv."
        )

    if volume_lattice.exists() and rows:
        structural = read_volume_lattice(volume_lattice)
        sources["structural_file"] = volume_lattice.name
        merge_structural_rows(rows, structural)
    elif thermo_grid.exists() and rows:
        sources.setdefault("structural_file", thermo_grid.name)
        add_derived_structural_from_grid(rows)
    return sorted(rows, key=lambda row: row["T_K"]), sources


def read_volume_lattice(path: Path) -> dict[str, list[tuple[float, float]]]:
    data: dict[str, list[tuple[float, float]]] = {}
    for row in read_csv(path):
        quantity = str(row.get("quantity", "")).strip()
        temp = finite_float(row.get("T_K"))
        value = finite_float(row.get("value"))
        if not quantity or temp is None or value is None:
            continue
        data.setdefault(quantity, []).append((temp, value))
    return {key: sorted_series(value) for key, value in data.items()}


def merge_structural_rows(
    rows: list[dict[str, Any]],
    structural: dict[str, list[tuple[float, float]]],
) -> None:
    lattice_keys = [key for key in ("a_lattice", "b_lattice", "c_lattice") if key in structural]
    lattice_key = lattice_keys[0] if lattice_keys else None
    alpha_lattice = derivative_series(structural[lattice_key]) if lattice_key else []
    volume = structural.get("V_A3", [])
    alpha_volume = [(temp, alpha_v / 3.0) for temp, alpha_v in derivative_series(volume)]

    length_series = structural[lattice_key] if lattice_key else []
    if not length_series and volume:
        length_series = [(temp, value ** (1.0 / 3.0)) for temp, value in volume if value > 0.0]

    for row in rows:
        temp = row["T_K"]
        if volume:
            row["V_A3"] = interpolate(volume, temp)
        alpha = interpolate(alpha_lattice, temp) if alpha_lattice else None
        if alpha is None:
            alpha = interpolate(alpha_volume, temp) if alpha_volume else None
        if alpha is not None:
            row["alpha_1_K"] = alpha
        length = interpolate(length_series, temp) if length_series else None
        if length is not None:
            row["dilatation_source_value"] = length


def add_derived_structural_from_grid(rows: list[dict[str, Any]]) -> None:
    if all(row.get("alpha_1_K") is not None for row in rows):
        return
    volume = [(row["T_K"], row["V_A3"]) for row in rows if row.get("V_A3") is not None]
    if not volume:
        return
    alpha_volume = [(temp, alpha_v / 3.0) for temp, alpha_v in derivative_series(volume)]
    for row in rows:
        if row.get("alpha_1_K") is None:
            row["alpha_1_K"] = interpolate(alpha_volume, row["T_K"])


def apply_temperature_window(
    rows: list[dict[str, Any]], t_min: float | None, t_max: float | None
) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        temp = row["T_K"]
        if t_min is not None and temp < t_min:
            continue
        if t_max is not None and temp > t_max:
            continue
        kept.append(row)
    return kept


def load_external_properties(
    csv_paths: list[Path], json_paths: list[Path], constants: list[str]
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, float], list[dict[str, Any]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    fixed: dict[str, float] = {}
    provenance: list[dict[str, Any]] = []

    for path in csv_paths:
        rows = read_csv(path)
        if not rows:
            continue
        fields = rows[0].keys()
        temp_field = "T_K" if "T_K" in fields else None
        if temp_field:
            for field in fields:
                if field == temp_field or field not in MOOSE_FIELDS:
                    continue
                points = []
                for row in rows:
                    temp = finite_float(row.get(temp_field))
                    value = finite_float(row.get(field))
                    if temp is not None and value is not None:
                        points.append((temp, value))
                if points:
                    series[field] = sorted_series(points)
        else:
            for field, value in rows[0].items():
                if field in MOOSE_FIELDS:
                    parsed = finite_float(value)
                    if parsed is not None:
                        fixed[field] = parsed
        provenance.append({"type": "property_csv", "path": str(path), "rows": len(rows)})

    for path in json_paths:
        payload = read_json(path)
        constants_payload = payload.get("constants", {}) if isinstance(payload, dict) else {}
        series_payload = payload.get("series", {}) if isinstance(payload, dict) else {}
        for field, value in constants_payload.items():
            if field in MOOSE_FIELDS:
                parsed = finite_float(value)
                if parsed is not None:
                    fixed[field] = parsed
        for field, values in series_payload.items():
            if field not in MOOSE_FIELDS or not isinstance(values, list):
                continue
            points = []
            for item in values:
                if isinstance(item, dict):
                    temp = finite_float(item.get("T_K"))
                    value = finite_float(item.get("value", item.get(field)))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    temp = finite_float(item[0])
                    value = finite_float(item[1])
                else:
                    continue
                if temp is not None and value is not None:
                    points.append((temp, value))
            if points:
                series[field] = sorted_series(points)
        for field, value in payload.items() if isinstance(payload, dict) else []:
            if field in MOOSE_FIELDS and field not in ("T_K", "source_tag"):
                parsed = finite_float(value)
                if parsed is not None:
                    fixed[field] = parsed
        provenance.append({"type": "property_json", "path": str(path)})

    for item in constants:
        if "=" not in item:
            raise SystemExit(f"Invalid --constant {item!r}; expected FIELD=VALUE")
        field, value = item.split("=", 1)
        field = field.strip()
        if field not in MOOSE_FIELDS or field in ("T_K", "source_tag"):
            raise SystemExit(f"Invalid MOOSE field for --constant: {field!r}")
        parsed = finite_float(value)
        if parsed is None:
            raise SystemExit(f"Invalid numeric value for --constant {item!r}")
        fixed[field] = parsed
        provenance.append({"type": "constant", "field": field})

    return series, fixed, provenance


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_tdb_composition(values: list[str], material: str) -> dict[str, float]:
    composition = formula_atom_fractions(material)
    for item in values:
        if "=" not in item:
            raise SystemExit(
                f"Invalid --tdb-composition {item!r}; expected ELEMENT=ATOMIC_FRACTION"
            )
        element, value = item.split("=", 1)
        parsed = finite_float(value)
        if parsed is None:
            raise SystemExit(f"Invalid numeric value in --tdb-composition {item!r}")
        composition[element.strip().upper()] = parsed
    return composition


def load_tdb_table(path: Path) -> dict[str, list[tuple[float, float]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        series, _fixed, _provenance = load_external_properties([path], [], [])
        return {key: value for key, value in series.items() if key in TDB_MERGE_FIELDS}
    if suffix == ".json":
        series, _fixed, _provenance = load_external_properties([], [path], [])
        return {key: value for key, value in series.items() if key in TDB_MERGE_FIELDS}
    return {}


def _finite_difference_cp_from_gibbs(
    temps: list[float], gibbs_j_mol_formula: list[float]
) -> list[tuple[float, float]]:
    points = sorted_series(list(zip(temps, gibbs_j_mol_formula)))
    if len(points) < 3:
        return []
    cp_points = []
    for idx in range(1, len(points) - 1):
        t0, g0 = points[idx - 1]
        temp, g = points[idx]
        t1, g1 = points[idx + 1]
        if t1 == temp or temp == t0:
            continue
        d1 = (g - g0) / (temp - t0)
        d2 = (g1 - g) / (t1 - temp)
        curvature = 2.0 * (d2 - d1) / (t1 - t0)
        cp_points.append((temp, -temp * curvature))
    return cp_points


def _extract_xarray_property(dataset: Any, name: str) -> list[float]:
    if not hasattr(dataset, "data_vars") or name not in dataset.data_vars:
        return []
    values = dataset[name].values
    flat = []
    try:
        import numpy as np

        array = np.asarray(values, dtype=float)
        for temp_index in range(array.shape[0]):
            candidates = array[temp_index].ravel()
            finite = candidates[np.isfinite(candidates)]
            flat.append(float(finite[0]) if finite.size else math.nan)
    except Exception:
        return []
    return flat


def _pycalphad_tdb_worker(
    queue: Any,
    tdb_path: str,
    components: list[str],
    phases: list[str],
    temps: list[float],
    composition: dict[str, float],
    molar_mass_g_mol: float,
) -> None:
    try:
        from pycalphad import Database, equilibrium, variables as v

        dbf = Database(tdb_path)
        conditions: dict[Any, Any] = {v.T: temps, v.P: 101325}
        for element, fraction in composition.items():
            if element in {"VA", "/-"}:
                continue
            conditions[v.X(element)] = fraction
        ds = equilibrium(dbf, components, phases, conditions, output="GM")
        gm_values = _extract_xarray_property(ds, "GM")
        cp_molar = _finite_difference_cp_from_gibbs(temps, gm_values)
        cp_mass = [
            (temp, cp_j_mol_k / (molar_mass_g_mol / 1000.0))
            for temp, cp_j_mol_k in cp_molar
            if math.isfinite(cp_j_mol_k)
        ]
        queue.put({"series": {"Cp_J_kgK": cp_mass}, "error": None})
    except Exception as exc:  # pragma: no cover - depends on optional pycalphad/TDBs
        queue.put({"series": {}, "error": str(exc)})


def sample_tdb_properties(
    *,
    path: Path | None,
    temps: list[float],
    material: str,
    molar_mass_g_mol: float,
    components: list[str],
    phases: list[str],
    composition: dict[str, float],
    timeout_s: float,
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, Any]]:
    if path is None:
        return {}, {"enabled": False}
    path = path.expanduser()
    metadata: dict[str, Any] = {
        "enabled": True,
        "path": str(path),
        "priority_fields": TDB_MERGE_FIELDS,
        "components": components,
        "phases": phases,
        "composition_atomic_fraction": composition,
    }
    if not path.is_file():
        metadata["error"] = "TDB/property file does not exist"
        return {}, metadata
    table_series = load_tdb_table(path)
    if table_series:
        metadata["mode"] = f"precomputed_{path.suffix.lower().lstrip('.')}"
        metadata["fields"] = sorted(table_series.keys())
        return table_series, metadata

    if path.suffix.lower() != ".tdb":
        metadata["error"] = "Unsupported TDB source; use .tdb, .csv, or .json"
        return {}, metadata
    if not phases:
        metadata["error"] = "Direct .tdb sampling requires --tdb-phases"
        return {}, metadata
    if not components:
        metadata["error"] = "Direct .tdb sampling requires --tdb-components or formula inference"
        return {}, metadata
    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    process = ctx.Process(
        target=_pycalphad_tdb_worker,
        args=(
            queue,
            str(path),
            components,
            phases,
            sorted(set(float(temp) for temp in temps)),
            composition,
            molar_mass_g_mol,
        ),
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(2.0)
        metadata["error"] = f"pycalphad sampling timed out after {timeout_s:g} s"
        return {}, metadata
    if queue.empty():
        metadata["error"] = "pycalphad sampling returned no result"
        return {}, metadata
    result = queue.get()
    metadata["mode"] = "pycalphad_finite_difference_GM"
    if result.get("error"):
        metadata["error"] = result["error"]
    series = result.get("series", {}) or {}
    metadata["fields"] = sorted(series.keys())
    return series, metadata


def merge_external(
    row: dict[str, Any], series: dict[str, list[tuple[float, float]]], fixed: dict[str, float]
) -> None:
    temp = row["T_K"]
    for field, value in fixed.items():
        row[field] = value
    for field, points in series.items():
        value = interpolate(points, temp)
        if value is not None:
            row[field] = value


def merge_tdb(
    row: dict[str, Any],
    series: dict[str, list[tuple[float, float]]],
    priority: str,
) -> None:
    if priority == "off":
        return
    temp = row["T_K"]
    for field, points in series.items():
        if field not in TDB_MERGE_FIELDS:
            continue
        if priority == "fill-missing" and row.get(field) is not None:
            continue
        value = interpolate(points, temp)
        if value is not None:
            row[field] = value


def cp_to_mass_basis(cp_j_mol_k: float | None, molar_mass_g_mol: float) -> float | None:
    if cp_j_mol_k is None:
        return None
    return cp_j_mol_k / (molar_mass_g_mol / 1000.0)


def density_from_volume(
    volume_a3: float | None, target_z: float | None, molar_mass_g_mol: float
) -> float | None:
    if volume_a3 is None or target_z is None or volume_a3 <= 0.0 or target_z <= 0.0:
        return None
    mass_kg = target_z * (molar_mass_g_mol / 1000.0) / AVOGADRO
    volume_m3 = volume_a3 * 1.0e-30
    return mass_kg / volume_m3


def infer_target_z(metadata: dict[str, Any], explicit: float | None) -> float | None:
    if explicit is not None:
        return explicit
    for name in ("normalization_metadata.json", "temperature_range_metadata.json"):
        payload = metadata.get(name, {})
        for key in ("target_z_formula_units", "target_z"):
            value = finite_float(payload.get(key)) if isinstance(payload, dict) else None
            if value is not None:
                return value
    return None


def add_dilatation(rows: list[dict[str, Any]], stress_free_t: float) -> None:
    points = [
        (row["T_K"], row["dilatation_source_value"])
        for row in rows
        if row.get("dilatation_source_value") is not None
    ]
    reference = nearest_or_interp(points, stress_free_t)
    if reference is None or reference == 0.0:
        return
    for row in rows:
        value = row.get("dilatation_source_value")
        if value is not None:
            row["dilatation"] = value / reference - 1.0


def build_moose_rows(
    qha_md_dir: Path,
    *,
    material: str,
    molar_mass_g_mol: float,
    target_z: float | None,
    stress_free_t: float,
    t_min: float | None,
    t_max: float | None,
    property_csv: list[Path],
    property_json: list[Path],
    constants: list[str],
    source_tag: str,
    tdb_path: Path | None,
    tdb_components: list[str],
    tdb_phases: list[str],
    tdb_composition: dict[str, float],
    tdb_priority: str,
    tdb_timeout_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = load_metadata(qha_md_dir)
    inferred_target_z = infer_target_z(metadata, target_z)
    thermo_rows, source_info = load_thermo_grid(qha_md_dir)
    thermo_rows = apply_temperature_window(thermo_rows, t_min, t_max)
    if not thermo_rows:
        raise SystemExit("No thermo_qha_md rows survived the requested temperature window.")

    series, fixed, external_provenance = load_external_properties(
        property_csv, property_json, constants
    )
    inferred_components = tdb_components
    if tdb_path and not inferred_components:
        inferred_components = sorted(formula_atom_fractions(material).keys()) + ["VA"]
    tdb_series, tdb_metadata = sample_tdb_properties(
        path=tdb_path,
        temps=[row["T_K"] for row in thermo_rows],
        material=material,
        molar_mass_g_mol=molar_mass_g_mol,
        components=inferred_components,
        phases=tdb_phases,
        composition=tdb_composition,
        timeout_s=tdb_timeout_s,
    )
    add_dilatation(thermo_rows, stress_free_t)

    rows: list[dict[str, Any]] = []
    for source in thermo_rows:
        row = {field: None for field in MOOSE_FIELDS}
        row["T_K"] = source["T_K"]
        row["source_tag"] = source_tag
        cp = cp_to_mass_basis(source.get("Cp_J_molK"), molar_mass_g_mol)
        cp_std = cp_to_mass_basis(source.get("Cp_std_J_molK"), molar_mass_g_mol)
        row["Cp_J_kgK"] = cp
        row["Cp_std_J_kgK"] = cp_std
        density = None
        density_g_cm3 = source.get("density_g_cm3")
        if density_g_cm3 is not None:
            density = density_g_cm3 * 1000.0
        if density is None:
            density = density_from_volume(source.get("V_A3"), inferred_target_z, molar_mass_g_mol)
        row["rho_kg_m3"] = density
        row["alpha_1_K"] = source.get("alpha_1_K")
        row["dilatation"] = source.get("dilatation")
        merge_external(row, series, fixed)
        merge_tdb(row, tdb_series, tdb_priority)
        rows.append(row)

    metadata_out = {
        "material": material,
        "units": "SI",
        "atomi_version": __version__,
        "temperature_range_K": [rows[0]["T_K"], rows[-1]["T_K"]],
        "stress_free_T_K": stress_free_t,
        "density_convention": (
            "thermo_qha_md density_fit_g_cm3"
            if any(row.get("density_g_cm3") is not None for row in thermo_rows)
            else "computed from QHA/MD volume, target_z_formula_units, and molar mass"
        ),
        "molar_mass_g_mol": molar_mass_g_mol,
        "target_z_formula_units": inferred_target_z,
        "interpolation": {"default": "piecewise_linear", "extrapolation": "error"},
        "moose_functions": moose_function_map(material),
        "sources": {
            "thermo_qha_md": source_info,
            "metadata_files": sorted(metadata.keys()),
            "external_properties": external_provenance,
            "tdb": tdb_metadata | {"priority": tdb_priority},
        },
        "columns": {
            "k_W_mK": "thermal conductivity",
            "Cp_J_kgK": "specific heat capacity",
            "rho_kg_m3": "mass density",
            "alpha_1_K": "instantaneous linear thermal expansion coefficient",
            "dilatation": "linear strain relative to stress_free_T_K",
            "E_Pa": "Young's modulus",
            "nu": "Poisson ratio",
            "K_Pa": "bulk modulus",
            "G_Pa": "shear modulus",
        },
    }
    return rows, metadata_out


def validate_rows(rows: list[dict[str, Any]], allow_partial: bool) -> list[str]:
    problems = []
    temps = [row["T_K"] for row in rows]
    if temps != sorted(temps) or len(set(temps)) != len(temps):
        problems.append("T_K must be strictly increasing with no duplicates")
    for row in rows:
        temp = row["T_K"]
        for field in REQUIRED_DEFAULT:
            if row.get(field) is None:
                problems.append(f"missing {field} at T={temp:g} K")
        if all(row.get(field) is None for field in STRUCTURAL_REQUIRED):
            problems.append(f"missing alpha_1_K or dilatation at T={temp:g} K")
        for field, value in row.items():
            if field in ("source_tag",):
                continue
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                problems.append(f"non-finite {field} at T={temp:g} K")
    if problems and not allow_partial:
        preview = "\n".join(f"- {problem}" for problem in problems[:20])
        extra = "" if len(problems) <= 20 else f"\n... {len(problems) - 20} more"
        raise SystemExit(
            "MOOSE material export is incomplete. Provide missing literature/user "
            f"properties or rerun with --allow-partial.\n{preview}{extra}"
        )
    return problems


def function_xy(rows: list[dict[str, Any]], field: str) -> tuple[list[float], list[float]]:
    pairs = [(row["T_K"], row.get(field)) for row in rows if row.get(field) is not None]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def render_moose_functions(
    rows: list[dict[str, Any]],
    *,
    material: str = "UO2",
    function_prefix: str | None = None,
) -> str:
    function_map = moose_function_map(material, function_prefix)
    lines = [
        "# Generated by atomi moose-qha-md-material",
        "# Temperature is in K; property values are SI.",
        "[Functions]",
    ]
    for field, name in function_map.items():
        temps, values = function_xy(rows, field)
        if not temps:
            continue
        x = " ".join(format_value(value) for value in temps)
        y = " ".join(format_value(value) for value in values)
        lines.extend(
            [
                f"  [{name}]",
                "    type = PiecewiseLinear",
                f"    x = '{x}'",
                f"    y = '{y}'",
                "  []",
            ]
        )
    lines.append("[]")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moose-qha-md-material",
        description=(
            "Merge thermo_qha_md outputs with user/literature properties and write "
            "SI MOOSE material-property tables."
        ),
    )
    parser.add_argument("--qha-md-dir", type=Path, required=True, help="thermo_qha_md output dir.")
    parser.add_argument("--out-csv", type=Path, default=Path("uo2_moose_material_properties.csv"))
    parser.add_argument(
        "--out-meta",
        type=Path,
        default=Path("uo2_moose_material_properties.meta.json"),
    )
    parser.add_argument("--moose-include", type=Path, help="Optional MOOSE [Functions] include.")
    parser.add_argument("--material", default="UO2")
    parser.add_argument("--molar-mass-g-mol", type=float)
    parser.add_argument("--target-z", type=float, help="Formula units in the exported volume cell.")
    parser.add_argument("--stress-free-T", type=float, default=300.0)
    parser.add_argument("--t-min", type=float)
    parser.add_argument("--t-max", type=float)
    parser.add_argument(
        "--property-csv",
        type=Path,
        action="append",
        default=[],
        help="CSV with T_K plus MOOSE fields, or one-row constants without T_K.",
    )
    parser.add_argument(
        "--property-json",
        type=Path,
        action="append",
        default=[],
        help="JSON with constants/series for MOOSE fields.",
    )
    parser.add_argument(
        "--constant",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Constant literature/user value, e.g. E_Pa=2.05e11.",
    )
    parser.add_argument(
        "--tdb",
        type=Path,
        help=(
            "Optional CALPHAD source. .csv/.json may contain precomputed MOOSE fields; "
            ".tdb is sampled with pycalphad when available."
        ),
    )
    parser.add_argument(
        "--tdb-priority",
        choices=("fill-missing", "prefer-tdb", "off"),
        default="fill-missing",
        help="How TDB/precomputed CALPHAD values merge with DFT/QHA/MD values.",
    )
    parser.add_argument(
        "--tdb-components",
        help=(
            "Comma-separated pycalphad components, e.g. U,O,VA. "
            "Defaults to formula elements + VA."
        ),
    )
    parser.add_argument(
        "--tdb-phases",
        help="Comma-separated pycalphad phases for direct .tdb sampling, e.g. UO2_FCC.",
    )
    parser.add_argument(
        "--tdb-composition",
        action="append",
        default=[],
        metavar="ELEMENT=ATOMIC_FRACTION",
        help="Override formula-inferred atomic fractions for direct .tdb sampling.",
    )
    parser.add_argument(
        "--tdb-timeout",
        type=float,
        default=30.0,
        help="Seconds allowed for direct pycalphad .tdb sampling.",
    )
    parser.add_argument("--source-tag", default="qha_md_moose_bridge")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write rows even when required MOOSE fields are missing.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    molar_mass = args.molar_mass_g_mol
    if molar_mass is None:
        molar_mass = formula_mass_g_mol(args.material)
    rows, metadata = build_moose_rows(
        args.qha_md_dir,
        material=args.material,
        molar_mass_g_mol=molar_mass,
        target_z=args.target_z,
        stress_free_t=args.stress_free_T,
        t_min=args.t_min,
        t_max=args.t_max,
        property_csv=args.property_csv,
        property_json=args.property_json,
        constants=args.constant,
        source_tag=args.source_tag,
        tdb_path=args.tdb,
        tdb_components=parse_csv_list(args.tdb_components),
        tdb_phases=parse_csv_list(args.tdb_phases),
        tdb_composition=parse_tdb_composition(args.tdb_composition, args.material),
        tdb_priority=args.tdb_priority,
        tdb_timeout_s=args.tdb_timeout,
    )
    problems = validate_rows(rows, args.allow_partial)
    metadata["validation"] = {
        "allow_partial": args.allow_partial,
        "problems": problems,
        "required_fields": REQUIRED_DEFAULT,
        "structural_requirement": "alpha_1_K or dilatation",
    }
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_csv, rows, MOOSE_FIELDS)
    args.out_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_meta}")
    if args.moose_include:
        args.moose_include.parent.mkdir(parents=True, exist_ok=True)
        args.moose_include.write_text(
            render_moose_functions(rows, material=args.material),
            encoding="utf-8",
        )
        print(f"Wrote {args.moose_include}")


if __name__ == "__main__":
    main()
