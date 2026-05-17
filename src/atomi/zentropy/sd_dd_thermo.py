"""Single-defect and double-defect thermodynamics cross-checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from atomi.zentropy.motif_db import parse_formula


KB_EV_K = 8.617333262145e-5
EV_PER_DEFECT_TO_KJ_MOL = 96.48533212331002
J_PER_MOL_PER_EV = EV_PER_DEFECT_TO_KJ_MOL * 1000.0
SCHEMA = "atomi.zentropy.sd_dd_thermo.v1"
WORKFLOW_SCHEMA = "atomi.zentropy.sd_dd_workflow.v1"


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def workflow_directories(root: Path) -> dict[str, Path]:
    return {
        "references": root / "00_references",
        "seeds": root / "01_seed_structures",
        "dft": root / "02_sd_dd_dft",
        "metadata": root / "03_motif_metadata",
        "defects": root / "04_defect_tables",
        "thermo": root / "05_sd_dd_thermo",
        "solution": root / "06_solution_model",
        "calphad": root / "07_calphad_seed",
    }


def init_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="defect-chem init",
        description="Create an SD/DD defect thermodynamics workflow skeleton.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("sd_dd_workflow"))
    parser.add_argument("--system", default="(Gd,U)O2")
    parser.add_argument("--parent-formula", default="UO2")
    parser.add_argument("--dopant", default="Gd")
    parser.add_argument("--host", default="U")
    parser.add_argument("--oxygen", default="O")
    args = parser.parse_args(argv)
    root = args.outdir.resolve()
    dirs = workflow_directories(root)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    (dirs["references"] / "reference_energies_template.csv").write_text(
        "element,mu_eV_per_atom,reference_id,energy_eV,formula,source,notes\n"
        "U,,,,,,element or reservoir chemical potential in eV/atom\n"
        "O,,,,,,oxygen chemical potential in eV/atom\n"
        "Gd,,,,,,dopant reservoir chemical potential in eV/atom\n"
        ",,parent_UO2,,UO2,,energy per parent formula unit or primitive reference\n",
        encoding="utf-8",
    )
    (dirs["seeds"] / "sd_dd_seed_index_template.csv").write_text(
        "case_id,model,seed_poscar,template,defect_a,defect_b,charge,delta_O,degeneracy,capacity_per_formula,notes\n"
        "Gd_U_seed,SD,seed_POSCARs/Gd_U/POSCAR,VASP_TEMPLATE,,,,,2,1.0,Gd substitution seed\n"
        "V_O_seed,SD,seed_POSCARs/V_O/POSCAR,VASP_TEMPLATE,,,,,-1,1,2.0,O vacancy seed\n"
        "GdU_VO_pair,DD,seed_POSCARs/GdU_VO_pair/POSCAR,VASP_TEMPLATE,Gd_U_seed,V_O_seed,,,-1,1.0,Gd-vacancy pair seed\n",
        encoding="utf-8",
    )
    (dirs["defects"] / "defect_pairs_template.csv").write_text(
        "pair_id,defect_a,defect_b,binding_energy_eV,capacity_per_formula,notes\n"
        f"{args.dopant}_{args.host}_VO_pair,{args.dopant}_{args.host},V_O,,1.0,negative binding stabilizes the pair\n",
        encoding="utf-8",
    )
    (dirs["solution"] / "solution_points_template.csv").write_text(
        "x,G_mix_eV_per_formula,T_K,source,notes\n"
        "0.0,0.0,,,\n"
        "0.5,,,,mixed or ordered endmember motif\n"
        "1.0,0.0,,,\n",
        encoding="utf-8",
    )
    commands = [
        f"midx 02_sd_dd_dft --index {dirs['metadata'] / 'motif_paths.csv'}",
        (
            "zentropy_motif_db auto-metadata "
            f"--input-csv {dirs['metadata'] / 'motif_paths.csv'} "
            f"--materialize-root {dirs['metadata'] / 'materialized_seeds'} "
            f"--metadata-csv {dirs['metadata'] / 'motif_metadata.csv'} "
            f"--site-state-csv {dirs['metadata'] / 'site_states.csv'}"
        ),
        (
            "zentropy_motif_db index "
            f"--root {dirs['metadata'] / 'materialized_seeds'} "
            f"--metadata-csv {dirs['metadata'] / 'motif_metadata.csv'} "
            f"--site-state-csv {dirs['metadata'] / 'site_states.csv'} "
            f"--db {dirs['metadata'] / 'defect_motif_db.json'}"
        ),
        (
            "defect-chem build-defects "
            f"--motif-db-json {dirs['metadata'] / 'defect_motif_db.json'} "
            f"--reference-csv {dirs['references'] / 'reference_energies_template.csv'} "
            f"--out {dirs['defects'] / 'defects.csv'}"
        ),
        (
            "defect-chem run "
            f"--defect-csv {dirs['defects'] / 'defects.csv'} "
            f"--pair-csv {dirs['defects'] / 'defect_pairs_template.csv'} "
            f"--outdir {dirs['thermo']}"
        ),
    ]
    readme = [
        f"# SD/DD Defect Thermodynamics Workflow: {args.system}",
        "",
        "This skeleton keeps SD/DD point-defect chemistry separate from the zentropy-ML path.",
        "Use it to prepare DFT runs, turn completed motif energies into defect species,",
        "fit simple CALPHAD-style solution seeds, and run dilute SD/DD cross-checks.",
        "",
        "## Stage Order",
        "",
        "1. Put seed POSCARs and a VASP_TEMPLATE into `01_seed_structures`.",
        "2. Use `defect-chem prepare-runs` to create array-DFT-ready folders in `02_sd_dd_dft`.",
        "3. Use `midx`, `auto-metadata`, and `zentropy_motif_db index` after VASP finishes.",
        "4. Use `defect-chem build-defects` to build `defects.csv` from DFT energies.",
        "5. Use `defect-chem run` for SD/DD thermodynamics.",
        "6. Use `defect-chem fit-solution` to seed regular/Redlich-Kister CALPHAD parameters.",
        "",
        "## Useful Commands",
        "",
        *[f"- `{command}`" for command in commands],
        "",
    ]
    (root / "README_SD_DD_WORKFLOW.md").write_text("\n".join(readme), encoding="utf-8")
    metadata = {
        "schema": WORKFLOW_SCHEMA,
        "system": args.system,
        "parent_formula": args.parent_formula,
        "host": args.host,
        "dopant": args.dopant,
        "oxygen": args.oxygen,
        "directories": {key: str(path) for key, path in dirs.items()},
        "suggested_commands": commands,
        "model_path": [
            "SD/DD DFT motifs",
            "defects.csv formation energies",
            "dilute defect thermodynamics",
            "solution-model seed parameters",
            "CALPHAD/pycalphad assessment",
        ],
    }
    write_json(root / "sd_dd_workflow.json", metadata)
    print(f"Wrote SD/DD workflow skeleton: {root}")
    print(f"Readme: {root / 'README_SD_DD_WORKFLOW.md'}")
    return metadata


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


def copy_vasp_template(template: Path, destination: Path, poscar: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(poscar, destination / "POSCAR")
    for name in ("INCAR", "KPOINTS", "POTCAR"):
        source = template / name
        if source.exists():
            shutil.copy2(source, destination / name)


def prepare_runs_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="defect-chem prepare-runs",
        description="Create SD/DD VASP run folders from seed POSCARs and a VASP template.",
    )
    parser.add_argument("--seed-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("02_sd_dd_dft"))
    parser.add_argument("--default-template", type=Path, default=Path("VASP_TEMPLATE"))
    parser.add_argument("--runlist", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    base = args.seed_csv.resolve().parent
    outdir = args.outdir.resolve()
    runlist = args.runlist.resolve() if args.runlist else outdir / "runlist.txt"
    index = args.index.resolve() if args.index else outdir / "sd_dd_seed_index.csv"
    manifest_rows: list[dict[str, Any]] = []
    run_dirs: list[Path] = []
    for row_number, row in enumerate(read_csv(args.seed_csv.resolve()), start=1):
        case_id = row.get("case_id") or row.get("defect_id") or row.get("motif_id") or f"case_{row_number:04d}"
        case_id = safe_name(case_id)
        seed_raw = row.get("seed_poscar") or row.get("poscar") or row.get("structure")
        if not seed_raw:
            raise ValueError(f"Seed row {row_number} is missing seed_poscar/poscar/structure.")
        seed_poscar = Path(seed_raw)
        if not seed_poscar.is_absolute():
            seed_poscar = base / seed_poscar
        template = Path(row.get("template") or args.default_template)
        if not template.is_absolute():
            template = base / template
        if not seed_poscar.exists():
            raise FileNotFoundError(f"Seed POSCAR not found for {case_id}: {seed_poscar}")
        if not template.exists():
            raise FileNotFoundError(f"VASP template not found for {case_id}: {template}")
        run_dir = outdir / case_id
        if run_dir.exists() and args.replace:
            shutil.rmtree(run_dir)
        if run_dir.exists() and not args.replace:
            raise FileExistsError(f"Run folder already exists: {run_dir}. Use --replace to overwrite.")
        copy_vasp_template(template, run_dir, seed_poscar)
        case_info = {
            "case_name": case_id,
            "model": row.get("model") or defect_kind(row),
            "defect_a": row.get("defect_a") or "",
            "defect_b": row.get("defect_b") or "",
            "seed_poscar": str(seed_poscar.resolve()),
            "template": str(template.resolve()),
            "source_row": {key: value for key, value in row.items() if value not in (None, "")},
        }
        write_json(run_dir / "case_info.json", case_info)
        run_dirs.append(run_dir)
        manifest = dict(row)
        manifest.update(
            {
                "case_id": case_id,
                "run_dir": str(run_dir.resolve()),
                "seed_poscar": str(seed_poscar.resolve()),
                "template": str(template.resolve()),
            }
        )
        manifest_rows.append(manifest)
    write_csv(index, manifest_rows, sorted({key for row in manifest_rows for key in row}))
    runlist.parent.mkdir(parents=True, exist_ok=True)
    runlist.write_text("\n".join(str(path.resolve()) for path in run_dirs) + ("\n" if run_dirs else ""), encoding="utf-8")
    metadata = {
        "schema": "atomi.zentropy.sd_dd_prepare_runs.v1",
        "seed_csv": str(args.seed_csv.resolve()),
        "outdir": str(outdir),
        "runlist": str(runlist),
        "index": str(index),
        "n_runs": len(run_dirs),
        "next_step": f"submit array DFT with {runlist}",
    }
    write_json(outdir / "sd_dd_prepare_metadata.json", metadata)
    print(f"Prepared SD/DD VASP runs : {len(run_dirs)}")
    print(f"Runlist                  : {runlist}")
    print(f"Seed index               : {index}")
    return metadata


def reference_data(path: Path | None) -> tuple[dict[str, float], dict[str, dict[str, str]]]:
    chemical_potentials: dict[str, float] = {}
    references: dict[str, dict[str, str]] = {}
    if path is None:
        return chemical_potentials, references
    if path.suffix.lower() == ".json":
        data = read_json(path)
        for key, value in (data.get("chemical_potentials_eV") or data.get("chemical_potentials") or {}).items():
            chemical_potentials[str(key)] = float(value)
        references = {str(key): value for key, value in (data.get("references") or {}).items()}
        return chemical_potentials, references
    for row in read_csv(path):
        element = row.get("element") or row.get("species")
        mu = finite_float(row.get("mu_eV_per_atom") or row.get("chemical_potential_eV"))
        if element and mu is not None:
            chemical_potentials[element] = mu
        reference_id = row.get("reference_id") or row.get("id")
        if reference_id:
            references[reference_id] = row
    return chemical_potentials, references


def record_model(record: dict[str, Any]) -> str:
    text = " ".join(
        str(record.get(key) or "")
        for key in ("motif_id", "motif_family", "motif_type", "defect_label")
    ).lower()
    if "pair" in text or "double" in text or "complex" in text:
        return "DD"
    meta = record.get("motif_metadata", {})
    text = " ".join(str(meta.get(key) or "") for key in ("motif_family", "motif_type", "defect_label")).lower()
    return "DD" if "pair" in text or "double" in text or "complex" in text else "SD"


def formation_energy_from_record(
    record: dict[str, Any],
    parent_counts: dict[str, float],
    parent_energy_eV: float | None,
    chemical_potentials: dict[str, float],
) -> tuple[float | None, str]:
    energy = finite_float(record.get("energy_eV"))
    if energy is None:
        return None, "missing_energy"
    counts = {str(key): float(value) for key, value in record.get("counts", {}).items()}
    norm = record.get("size_normalization", {})
    formula_units = finite_float(norm.get("formula_units"))
    if parent_energy_eV is not None and formula_units is not None:
        value = energy - formula_units * parent_energy_eV
        for element, parent_count in parent_counts.items():
            delta = counts.get(element, 0.0) - formula_units * float(parent_count)
            if abs(delta) > 1.0e-12 and element in chemical_potentials:
                value -= delta * chemical_potentials[element]
        for element, count in counts.items():
            if element not in parent_counts and element in chemical_potentials:
                value -= count * chemical_potentials[element]
        return value, "parent_reference_plus_delta_mu"
    if chemical_potentials:
        value = energy
        for element, count in counts.items():
            if element in chemical_potentials:
                value -= count * chemical_potentials[element]
        return value, "absolute_mu_sum"
    return None, "need_parent_reference_or_chemical_potentials"


def build_defects_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="defect-chem build-defects",
        description="Build defects.csv from a zentropy motif DB and reference chemical potentials.",
    )
    parser.add_argument("--motif-db-json", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--out", type=Path, default=Path("defects.csv"))
    parser.add_argument("--parent-formula", default="UO2")
    parser.add_argument("--parent-reference-energy-eV", type=float)
    parser.add_argument("--chemical-potential", action="append", default=[])
    parser.add_argument("--default-capacity-per-formula", type=float)
    args = parser.parse_args(argv)
    db = read_json(args.motif_db_json.resolve())
    parent_counts = parse_formula(args.parent_formula)
    mu_from_file, references = reference_data(args.reference_json or args.reference_csv)
    mu = {**mu_from_file, **parse_key_values(args.chemical_potential)}
    parent_energy = args.parent_reference_energy_eV
    if parent_energy is None:
        for ref in references.values():
            if ref.get("formula") == args.parent_formula:
                parent_energy = finite_float(ref.get("energy_eV") or ref.get("energy_eV_per_formula"))
                break
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for record in db.get("records", []):
        eform, method = formation_energy_from_record(record, parent_counts, parent_energy, mu)
        if eform is None:
            skipped.append({"motif_id": record.get("motif_id", ""), "reason": method})
            continue
        norm = record.get("size_normalization", {})
        formula_units = finite_float(norm.get("formula_units")) or 1.0
        capacity = args.default_capacity_per_formula
        if capacity is None:
            capacity = 1.0 / formula_units if formula_units else 1.0
        meta = record.get("motif_metadata", {})
        charge = finite_float(record.get("charge_state"))
        if charge is None:
            charge = finite_float(record.get("charge", {}).get("nominal_charge_total")) or 0.0
        row = {
            "defect_id": record.get("motif_id"),
            "model": record_model(record),
            "formation_energy_eV": eform,
            "formation_energy_method": method,
            "degeneracy": record.get("degeneracy", 1.0),
            "capacity_per_formula": capacity,
            "charge": charge,
            "delta_O": norm.get("oxygen_delta_per_formula_unit", ""),
            "sublattice": meta.get("sublattice") or "",
            "site_species": record.get("defect_label") or meta.get("defect_label") or record.get("motif_id"),
            "motif_family": record.get("motif_family", ""),
            "spin_order_host": meta.get("spin_order_host", ""),
            "spin_order_all": meta.get("spin_order_all", ""),
            "source": record.get("run_dir", ""),
            "notes": "Built from defect_motif_db.json; verify reference chemical potentials before publication.",
        }
        rows.append(row)
    fields = [
        "defect_id",
        "model",
        "formation_energy_eV",
        "formation_energy_method",
        "degeneracy",
        "capacity_per_formula",
        "charge",
        "delta_O",
        "sublattice",
        "site_species",
        "motif_family",
        "spin_order_host",
        "spin_order_all",
        "source",
        "notes",
    ]
    write_csv(args.out.resolve(), rows, fields)
    metadata = {
        "schema": "atomi.zentropy.sd_dd_defects_builder.v1",
        "motif_db_json": str(args.motif_db_json.resolve()),
        "reference_file": str((args.reference_json or args.reference_csv).resolve()) if (args.reference_json or args.reference_csv) else "",
        "parent_formula": args.parent_formula,
        "parent_reference_energy_eV": parent_energy,
        "chemical_potentials_eV": mu,
        "n_rows": len(rows),
        "skipped": skipped,
        "out": str(args.out.resolve()),
    }
    write_json(args.out.resolve().with_suffix(".metadata.json"), metadata)
    print(f"Wrote defects.csv rows : {len(rows)}")
    print(f"Output                : {args.out.resolve()}")
    if skipped:
        print(f"Skipped records       : {len(skipped)}")
    return metadata


def fit_solution_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="defect-chem fit-solution",
        description="Fit simple CALPHAD seed parameters from a binary solution energy curve.",
    )
    parser.add_argument("--solution-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("analysis/sd_dd_solution_model"))
    parser.add_argument("--model", choices=("regular", "redlich-kister"), default="regular")
    parser.add_argument("--material", default="material")
    parser.add_argument("--phase", default="DEFECT_FLUORITE")
    parser.add_argument("--component-a", default="UO2")
    parser.add_argument("--component-b", default="GdO1.5")
    args = parser.parse_args(argv)
    raw_rows = read_csv(args.solution_csv.resolve())
    points = []
    for row in raw_rows:
        x = finite_float(row.get("x") or row.get("x_B") or row.get("mole_fraction"))
        gmix = finite_float(row.get("G_mix_eV_per_formula") or row.get("G_excess_eV_per_formula"))
        if x is None or gmix is None or x <= 0 or x >= 1:
            continue
        points.append((x, gmix))
    if not points:
        raise ValueError("No usable solution points found. Need x and G_mix_eV_per_formula for 0 < x < 1.")
    x_arr = np.asarray([item[0] for item in points], dtype=float)
    y_arr = np.asarray([item[1] for item in points], dtype=float)
    if args.model == "regular":
        basis = (x_arr * (1.0 - x_arr))[:, None]
        labels = ["L0_eV_per_formula"]
    else:
        basis = np.column_stack(
            [
                x_arr * (1.0 - x_arr),
                x_arr * (1.0 - x_arr) * (2.0 * x_arr - 1.0),
            ]
        )
        labels = ["L0_eV_per_formula", "L1_eV_per_formula"]
    coeffs, *_ = np.linalg.lstsq(basis, y_arr, rcond=None)
    predicted = basis @ coeffs
    outdir = args.outdir.resolve()
    param_rows = [
        {
            "model": args.model,
            "parameter": label,
            "value_eV_per_formula": value,
            "value_kJ_mol_formula": value * EV_PER_DEFECT_TO_KJ_MOL,
            "phase": args.phase,
            "component_a": args.component_a,
            "component_b": args.component_b,
        }
        for label, value in zip(labels, coeffs)
    ]
    fit_rows = [
        {
            "x": x,
            "G_mix_eV_per_formula": y,
            "G_fit_eV_per_formula": fit,
            "residual_eV_per_formula": y - fit,
        }
        for x, y, fit in zip(x_arr, y_arr, predicted)
    ]
    write_csv(outdir / "solution_model_parameters.csv", param_rows, list(param_rows[0]))
    write_csv(outdir / "solution_model_fit.csv", fit_rows, list(fit_rows[0]))
    metadata = {
        "schema": "atomi.zentropy.sd_dd_solution_fit.v1",
        "model": args.model,
        "material": args.material,
        "phase": args.phase,
        "component_a": args.component_a,
        "component_b": args.component_b,
        "n_points": len(points),
        "notes": [
            "Regular/Redlich-Kister coefficients are CALPHAD seed parameters, not a full assessment.",
            "For a true CEF model, map these parameters onto the selected sublattice endmembers and refit with pycalphad/TDB constraints.",
        ],
        "outputs": {
            "parameters": str(outdir / "solution_model_parameters.csv"),
            "fit": str(outdir / "solution_model_fit.csv"),
        },
    }
    write_json(outdir / "solution_model_metadata.json", metadata)
    print(f"Wrote solution parameters: {outdir / 'solution_model_parameters.csv'}")
    return metadata


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


def build_run_parser() -> argparse.ArgumentParser:
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


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_run_parser().parse_args(argv)
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


def build_dispatch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="defect-chem",
        description="SD/DD defect chemistry workflow helpers and thermodynamics cross-checks.",
    )
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("init", help="Create a workflow skeleton.")
    sub.add_parser("prepare-runs", help="Create VASP folders from seed POSCARs.")
    sub.add_parser("build-defects", help="Build defects.csv from a motif DB.")
    sub.add_parser("fit-solution", help="Fit regular/Redlich-Kister CALPHAD seed parameters.")
    sub.add_parser("run", help="Run SD/DD dilute thermodynamics.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    raw = sys.argv[1:] if argv is None else list(argv)
    actions = {
        "init": init_main,
        "prepare-runs": prepare_runs_main,
        "prepare": prepare_runs_main,
        "build-defects": build_defects_main,
        "build": build_defects_main,
        "fit-solution": fit_solution_main,
        "fit": fit_solution_main,
        "run": run_main,
    }
    if raw and raw[0] in actions:
        return actions[raw[0]](raw[1:])
    if raw and raw[0] in ("-h", "--help"):
        build_dispatch_parser().parse_args(raw)
        return {}
    return run_main(raw)


if __name__ == "__main__":
    main()
