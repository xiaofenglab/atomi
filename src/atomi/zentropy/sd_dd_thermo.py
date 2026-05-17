"""Single-defect and double-defect thermodynamics cross-checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


KB_EV_K = 8.617333262145e-5
EV_PER_DEFECT_TO_KJ_MOL = 96.48533212331002
J_PER_MOL_PER_EV = EV_PER_DEFECT_TO_KJ_MOL * 1000.0
SCHEMA = "atomi.zentropy.sd_dd_thermo.v1"


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


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_+-]+", "_", value.strip()).strip("_") or "item"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_key_values(items: list[str] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        result[key.strip()] = float(value)
    return result


def temperature_grid(args: argparse.Namespace) -> list[float]:
    if args.temperature:
        values = []
        for item in args.temperature:
            values.extend(float(part) for part in item.replace(";", ",").split(",") if part.strip())
        return sorted(dict.fromkeys(values))
    if args.T_min is None or args.T_max is None:
        return [1000.0]
    step = args.T_step or 100.0
    if step <= 0:
        raise ValueError("--T-step must be positive.")
    values = []
    current = float(args.T_min)
    while current <= float(args.T_max) + 1.0e-9:
        values.append(round(current, 10))
        current += step
    return values


def defect_kind(row: dict[str, Any]) -> str:
    raw = str(row.get("model") or row.get("kind") or row.get("defect_model") or "").strip().upper()
    if raw in {"SD", "SINGLE", "SINGLE_DEFECT"}:
        return "SD"
    if raw in {"DD", "DOUBLE", "PAIR", "DOUBLE_DEFECT"}:
        return "DD"
    n_defects = finite_float(row.get("n_defects") or row.get("order"))
    if n_defects is not None and n_defects >= 2:
        return "DD"
    text = " ".join(str(row.get(key) or "") for key in ("defect_id", "motif_family", "notes")).lower()
    return "DD" if "pair" in text or "double" in text else "SD"


def delta_species(row: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key, value in row.items():
        if not key.startswith("delta_"):
            continue
        number = finite_float(value)
        if number is None:
            continue
        species = key.split("delta_", 1)[1]
        if species:
            deltas[species] = number
    return deltas


def normalize_defect_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    clean_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        defect_id = row.get("defect_id") or row.get("motif_id") or row.get("name") or f"defect_{index}"
        formation_e = finite_float(
            row.get("formation_energy_eV")
            or row.get("E_form_eV")
            or row.get("G_form_eV")
            or row.get("formation_free_energy_eV")
        )
        entropy = finite_float(row.get("formation_entropy_J_molK") or row.get("S_form_J_molK"))
        clean = {key: value for key, value in row.items()}
        clean.update(
            {
                "defect_id": defect_id,
                "model": defect_kind(row),
                "formation_energy_eV": formation_e,
                "formation_entropy_J_molK": entropy,
                "degeneracy": finite_float(row.get("degeneracy")) or 1.0,
                "capacity_per_formula": (
                    finite_float(
                        row.get("capacity_per_formula")
                        or row.get("site_capacity")
                        or row.get("site_fraction_capacity")
                        or row.get("available_sites_per_formula")
                    )
                    or 1.0
                ),
                "charge": finite_float(row.get("charge")) or 0.0,
                "sublattice": row.get("sublattice") or "",
                "site_species": row.get("site_species") or row.get("composition") or "",
                "source": row.get("source") or "",
                "notes": row.get("notes") or "",
                "delta_species": delta_species(row),
            }
        )
        clean_rows.append(clean)
    return clean_rows


def build_pair_rows(single_rows: list[dict[str, Any]], pair_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_id = {str(row["defect_id"]): row for row in single_rows}
    out: list[dict[str, Any]] = []
    for index, pair in enumerate(pair_rows, start=1):
        defect_a = pair.get("defect_a") or pair.get("single_a") or pair.get("component_a") or ""
        defect_b = pair.get("defect_b") or pair.get("single_b") or pair.get("component_b") or ""
        row = {key: value for key, value in pair.items()}
        pair_id = pair.get("pair_id") or pair.get("defect_id") or f"{safe_name(defect_a)}__{safe_name(defect_b)}"
        formation_e = finite_float(pair.get("formation_energy_eV") or pair.get("E_form_eV"))
        entropy = finite_float(pair.get("formation_entropy_J_molK") or pair.get("S_form_J_molK"))
        deltas = delta_species(pair)
        charge = finite_float(pair.get("charge"))
        if formation_e is None:
            if defect_a not in by_id or defect_b not in by_id:
                raise ValueError(f"Pair row {index} references unknown defects: {defect_a}, {defect_b}")
            energy_a = by_id[defect_a].get("formation_energy_eV")
            energy_b = by_id[defect_b].get("formation_energy_eV")
            if energy_a is None or energy_b is None:
                raise ValueError(f"Pair row {index} cannot infer formation energy from missing single-defect energies.")
            binding = finite_float(pair.get("binding_energy_eV")) or 0.0
            formation_e = float(energy_a) + float(energy_b) + binding
            if entropy is None:
                entropy = (by_id[defect_a].get("formation_entropy_J_molK") or 0.0) + (
                    by_id[defect_b].get("formation_entropy_J_molK") or 0.0
                )
            if not deltas:
                for source in (by_id[defect_a], by_id[defect_b]):
                    for species, value in source.get("delta_species", {}).items():
                        deltas[species] = deltas.get(species, 0.0) + float(value)
            if charge is None:
                charge = float(by_id[defect_a].get("charge") or 0.0) + float(by_id[defect_b].get("charge") or 0.0)
        row.update(
            {
                "defect_id": pair_id,
                "model": "DD",
                "formation_energy_eV": formation_e,
                "formation_entropy_J_molK": entropy,
                "degeneracy": finite_float(pair.get("degeneracy")) or 1.0,
                "capacity_per_formula": finite_float(pair.get("capacity_per_formula") or pair.get("site_capacity")) or 1.0,
                "charge": charge or 0.0,
                "defect_a": defect_a,
                "defect_b": defect_b,
                "binding_energy_eV": finite_float(pair.get("binding_energy_eV")),
                "sublattice": pair.get("sublattice") or "paired_defect_site",
                "site_species": pair.get("site_species") or f"{defect_a}+{defect_b}",
                "source": pair.get("source") or "pair_csv",
                "notes": pair.get("notes") or "Double-defect row inferred from pair definition.",
                "delta_species": deltas,
            }
        )
        out.append(row)
    return out


def effective_formation_energy_eV(
    row: dict[str, Any],
    temperature: float,
    chemical_potentials: dict[str, float],
    electron_mu_eV: float | None,
) -> float | None:
    formation_e = row.get("formation_energy_eV")
    if formation_e is None:
        return None
    entropy = row.get("formation_entropy_J_molK")
    entropy_eV_K = float(entropy) / J_PER_MOL_PER_EV if entropy is not None else 0.0
    value = float(formation_e) - temperature * entropy_eV_K
    for species, delta in row.get("delta_species", {}).items():
        if species in chemical_potentials:
            value -= float(delta) * chemical_potentials[species]
    if electron_mu_eV is not None:
        value += float(row.get("charge") or 0.0) * electron_mu_eV
    return value


def log1pexp(value: float) -> float:
    if value > 50:
        return value
    if value < -50:
        return math.exp(value)
    return math.log1p(math.exp(value))


def equilibrium_population(
    effective_g_eV: float,
    temperature: float,
    degeneracy: float,
    capacity: float,
) -> dict[str, float]:
    kbt = KB_EV_K * temperature
    if kbt <= 0:
        raise ValueError("Temperature must be positive.")
    capacity = max(float(capacity), 0.0)
    degeneracy = max(float(degeneracy), 1.0e-300)
    z = math.log(degeneracy) - effective_g_eV / kbt
    site_fraction = 1.0 / (1.0 + math.exp(-z)) if -700 < z < 700 else (1.0 if z >= 700 else 0.0)
    concentration = capacity * site_fraction
    free_energy = -capacity * kbt * log1pexp(z)
    dilute = capacity * math.exp(z) if z < 700 else math.inf
    return {
        "site_fraction_of_capacity": site_fraction,
        "concentration_per_formula": concentration,
        "dilute_concentration_per_formula": dilute,
        "free_energy_lowering_eV_per_formula": free_energy,
        "log_activity": z,
    }


def evaluate_rows(
    rows: list[dict[str, Any]],
    temperatures: list[float],
    chemical_potentials: dict[str, float],
    electron_mu_eV: float | None,
    dilute_warning_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    cef_rows: list[dict[str, Any]] = []
    for temperature in temperatures:
        summary = {
            "T_K": temperature,
            "single_defect_concentration_per_formula": 0.0,
            "double_defect_concentration_per_formula": 0.0,
            "net_charge_per_formula": 0.0,
            "oxygen_delta_per_formula": 0.0,
            "free_energy_lowering_eV_per_formula": 0.0,
            "n_active_rows": 0,
            "warnings": [],
        }
        for row in rows:
            effective_g = effective_formation_energy_eV(row, temperature, chemical_potentials, electron_mu_eV)
            if effective_g is None:
                continue
            pop = equilibrium_population(
                effective_g,
                temperature,
                float(row.get("degeneracy") or 1.0),
                float(row.get("capacity_per_formula") or 1.0),
            )
            concentration = pop["concentration_per_formula"]
            model = str(row.get("model") or "SD")
            if model == "DD":
                summary["double_defect_concentration_per_formula"] += concentration
            else:
                summary["single_defect_concentration_per_formula"] += concentration
            summary["net_charge_per_formula"] += concentration * float(row.get("charge") or 0.0)
            summary["oxygen_delta_per_formula"] += concentration * float(row.get("delta_species", {}).get("O", 0.0))
            summary["free_energy_lowering_eV_per_formula"] += pop["free_energy_lowering_eV_per_formula"]
            summary["n_active_rows"] += 1
            if pop["site_fraction_of_capacity"] > dilute_warning_fraction:
                summary["warnings"].append(f"{row['defect_id']} exceeds dilute fraction")
            detail_rows.append(
                {
                    "T_K": temperature,
                    "defect_id": row["defect_id"],
                    "model": model,
                    "defect_a": row.get("defect_a", ""),
                    "defect_b": row.get("defect_b", ""),
                    "formation_energy_eV": row.get("formation_energy_eV"),
                    "effective_formation_energy_eV": effective_g,
                    "degeneracy": row.get("degeneracy"),
                    "capacity_per_formula": row.get("capacity_per_formula"),
                    "site_fraction_of_capacity": pop["site_fraction_of_capacity"],
                    "concentration_per_formula": concentration,
                    "dilute_concentration_per_formula": pop["dilute_concentration_per_formula"],
                    "free_energy_lowering_eV_per_formula": pop["free_energy_lowering_eV_per_formula"],
                    "charge": row.get("charge"),
                    "net_charge_per_formula": concentration * float(row.get("charge") or 0.0),
                    "delta_O": row.get("delta_species", {}).get("O"),
                    "oxygen_delta_per_formula": concentration * float(row.get("delta_species", {}).get("O", 0.0)),
                    "sublattice": row.get("sublattice"),
                    "site_species": row.get("site_species"),
                    "source": row.get("source"),
                }
            )
            cef_rows.append(
                {
                    "T_K": temperature,
                    "phase": "DEFECT_FLUORITE",
                    "defect_id": row["defect_id"],
                    "model": model,
                    "sublattice": row.get("sublattice") or ("pair" if model == "DD" else "defect_site"),
                    "site_species": row.get("site_species") or row["defect_id"],
                    "site_fraction_seed": pop["site_fraction_of_capacity"],
                    "G_kJ_mol_defect": effective_g * EV_PER_DEFECT_TO_KJ_MOL,
                    "cef_role": "seed_site_fraction_or_endmember_energy_for_future_CEF_assessment",
                }
            )
        summary["warnings"] = ";".join(summary["warnings"])
        summary_rows.append(summary)
    return detail_rows, summary_rows, cef_rows


def write_model_notes(path: Path) -> None:
    text = """# SD/DD Defect Thermodynamics Notes

This module is a dilute-defect cross-check beside the zentropy-ML workflow.

- SD rows are independent single-defect species.
- DD rows are paired/double-defect species, either supplied directly or inferred from two SD rows plus a binding energy.
- Equilibrium populations use an ideal lattice-gas expression from effective formation free energy, degeneracy, and capacity.
- Chemical-potential shifts use `G_eff = G_form - sum(delta_i * mu_i) + q * mu_e`.
- High site fractions are flagged because dilute SD/DD assumptions then become weak.

Use this as a fast thermodynamic sanity check and a seed table for later CEF/CALPHAD assessment, not as a replacement for a fitted sublattice model or zentropy microstate ensemble.
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sd-dd-thermo",
        description="Single-defect/double-defect dilute thermodynamics cross-check for zentropy and CEF workflows.",
    )
    parser.add_argument("--defect-csv", type=Path, required=True, help="CSV of SD or explicit DD defect species.")
    parser.add_argument("--pair-csv", type=Path, help="Optional DD pair definitions built from SD rows plus binding energy.")
    parser.add_argument("--outdir", type=Path, default=Path("analysis/sd_dd_thermo"))
    parser.add_argument("--material", default="material")
    parser.add_argument("--formula", default="")
    parser.add_argument("--temperature", action="append", help="Temperature list, e.g. 800,1000,1200. Repeatable.")
    parser.add_argument("--T-min", type=float)
    parser.add_argument("--T-max", type=float)
    parser.add_argument("--T-step", type=float, default=100.0)
    parser.add_argument(
        "--chemical-potential",
        action="append",
        default=[],
        help="Species chemical potential in eV/atom for formation shifts, e.g. O=-5.0.",
    )
    parser.add_argument("--electron-chemical-potential", type=float, help="Electron chemical potential/Fermi term in eV.")
    parser.add_argument("--dilute-warning-fraction", type=float, default=0.05)
    parser.add_argument("--json", action="store_true", help="Print metadata JSON.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    temperatures = temperature_grid(args)
    chemical_potentials = parse_key_values(args.chemical_potential)
    rows = normalize_defect_rows(read_csv(args.defect_csv.resolve()))
    if args.pair_csv:
        rows.extend(build_pair_rows(rows, read_csv(args.pair_csv.resolve())))
    detail_rows, summary_rows, cef_rows = evaluate_rows(
        rows,
        temperatures,
        chemical_potentials,
        args.electron_chemical_potential,
        args.dilute_warning_fraction,
    )
    outdir = args.outdir.resolve()
    detail_path = outdir / "sd_dd_defect_populations.csv"
    summary_path = outdir / "sd_dd_summary.csv"
    cef_path = outdir / "sd_dd_cef_seed.csv"
    notes_path = outdir / "sd_dd_model_notes.md"
    metadata_path = outdir / "sd_dd_metadata.json"
    write_csv(
        detail_path,
        detail_rows,
        [
            "T_K",
            "defect_id",
            "model",
            "defect_a",
            "defect_b",
            "formation_energy_eV",
            "effective_formation_energy_eV",
            "degeneracy",
            "capacity_per_formula",
            "site_fraction_of_capacity",
            "concentration_per_formula",
            "dilute_concentration_per_formula",
            "free_energy_lowering_eV_per_formula",
            "charge",
            "net_charge_per_formula",
            "delta_O",
            "oxygen_delta_per_formula",
            "sublattice",
            "site_species",
            "source",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "T_K",
            "single_defect_concentration_per_formula",
            "double_defect_concentration_per_formula",
            "net_charge_per_formula",
            "oxygen_delta_per_formula",
            "free_energy_lowering_eV_per_formula",
            "n_active_rows",
            "warnings",
        ],
    )
    write_csv(
        cef_path,
        cef_rows,
        [
            "T_K",
            "phase",
            "defect_id",
            "model",
            "sublattice",
            "site_species",
            "site_fraction_seed",
            "G_kJ_mol_defect",
            "cef_role",
        ],
    )
    write_model_notes(notes_path)
    metadata = {
        "schema": SCHEMA,
        "material": args.material,
        "formula": args.formula,
        "inputs": {
            "defect_csv": str(args.defect_csv.resolve()),
            "pair_csv": str(args.pair_csv.resolve()) if args.pair_csv else "",
        },
        "temperatures_K": temperatures,
        "chemical_potentials_eV": chemical_potentials,
        "electron_chemical_potential_eV": args.electron_chemical_potential,
        "outputs": {
            "populations": str(detail_path),
            "summary": str(summary_path),
            "cef_seed": str(cef_path),
            "notes": str(notes_path),
        },
        "model_scope": [
            "dilute independent SD/DD lattice-gas cross-check",
            "DD pairs may be explicit species or inferred from SD rows plus binding energy",
            "CEF seed output is a starting table, not a fitted CALPHAD assessment",
        ],
        "literature_context": [
            "Curti and Kulik used sublattice solid-solution thermodynamics and Gibbs energy minimization for UO2 fuels.",
            "Hillert's CEF treats solution phases with sublattices using site fractions and constitutional entropy.",
            "This module is deliberately simpler and is meant to flag trends before zentropy or CEF fitting.",
        ],
    }
    write_json(metadata_path, metadata)
    if args.json:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    else:
        print(f"Wrote SD/DD populations : {detail_path}")
        print(f"Wrote SD/DD summary     : {summary_path}")
        print(f"Wrote CEF seed table    : {cef_path}")
    return metadata


if __name__ == "__main__":
    main()
