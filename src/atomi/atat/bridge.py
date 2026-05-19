"""Bridge ATAT configuration tools into Atomi defect workflows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "atomi.atat.bridge.v1"
STATUS_SCHEMA = "atomi.atat.status.v1"
QUICK_OPT_SCHEMA = "atomi.atat.quick_opt.v1"

ATAT_TOOLS: dict[str, list[str]] = {
    "cluster_expansion": ["maps", "mmaps", "corrdump", "genstr", "emc2"],
    "sqs": ["mcsqs", "corrdump"],
    "structure_conversion": ["str2poscar", "poscar2str"],
    "diagnostics": ["mapsrep", "checkcell", "cellcvrt"],
}

WORKFLOW_STAGES: list[dict[str, str]] = [
    {
        "stage_id": "01_motif_search",
        "atat_role": "Enumerate low-energy occupational/spin-valence configurations.",
        "atomi_role": "Prepare VASP folders, run fail-fast branch screening, then keep physics-accepted low-energy motifs.",
        "typical_inputs": "lat.in, clusters.out, pseudo_species_map.csv, VASP_TEMPLATE",
        "typical_outputs": "atat_candidate_index.csv, atomi_runlist.txt, stage1_branch_summary.csv",
    },
    {
        "stage_id": "02_sd_dd_cluster_expansion",
        "atat_role": "Fit effective interactions from accepted DFT energies and accelerate motif/defect arrangement search.",
        "atomi_role": "Convert accepted motif energies into SD/DD defect tables and CALPHAD-ready interaction summaries.",
        "typical_inputs": "energy-labeled structures, defects.csv, reference_energies.csv",
        "typical_outputs": "ce_fit_manifest.json, pair_interactions.csv, sd_dd_defects_from_ce.csv",
    },
    {
        "stage_id": "03_sqs_mid_high_concentration",
        "atat_role": "Generate SQS or representative disordered cells for mid/high composition defect chemistry.",
        "atomi_role": "Use SQS structures as seed motifs for spin branching, MLIP/DFT screening, and zentropy microstates.",
        "typical_inputs": "rndstr.in, target composition, pseudo_species_map.csv",
        "typical_outputs": "bestsqs.out, sqs_candidate_index.csv, VASP-ready runlist",
    },
    {
        "stage_id": "04_feedback_to_atomi",
        "atat_role": "Provide ranked structures, correlations, CE uncertainties, and MC population hints.",
        "atomi_role": "Feed candidates to vasp-branch-live, vasp-spin-report, zentropy motif DB, SD/DD, and MLIP active learning.",
        "typical_inputs": "accepted DFT results, spin_energy_run_summary.csv, stage1_branch_summary.csv",
        "typical_outputs": "defect_motif_db.json, active_learning_candidates.csv, calphad_pseudodata.csv",
    },
]

SPECIES_FIELDS = [
    "pseudo_species",
    "element",
    "role",
    "spin_value",
    "charge_state",
    "sublattice",
    "moment_guard",
    "vasp_element",
    "notes",
]

CE_TRAINING_FIELDS = [
    "training_id",
    "source_csv",
    "source_row",
    "run",
    "resolved_run",
    "output_run_dir",
    "structure_path",
    "energy_eV",
    "energy_eV_per_fu",
    "relative_energy_eV_per_fu",
    "physics_status",
    "mag_status",
    "decision",
    "element_order",
    "changed_by_element",
    "atat_candidate_id",
    "motif_family",
    "use_for",
    "notes",
]

QUICK_COMMAND_FIELDS = ["step", "purpose", "command"]


@dataclass
class PseudoSpecies:
    pseudo_species: str
    element: str
    role: str
    spin_value: str = ""
    charge_state: str = ""
    sublattice: str = ""
    moment_guard: str = ""
    vasp_element: str = ""
    notes: str = ""

    def row(self) -> dict[str, str]:
        return {
            "pseudo_species": self.pseudo_species,
            "element": self.element,
            "role": self.role,
            "spin_value": self.spin_value,
            "charge_state": self.charge_state,
            "sublattice": self.sublattice,
            "moment_guard": self.moment_guard,
            "vasp_element": self.vasp_element or self.element,
            "notes": self.notes,
        }


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_+-]+", "_", value.strip()).strip("_") or "item"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def split_items(items: list[str] | None) -> list[str]:
    values: list[str] = []
    for raw in items or []:
        for part in str(raw).split(","):
            text = part.strip()
            if text:
                values.append(text)
    return values


def shell_join(parts: list[str]) -> str:
    return " \\\n  ".join(parts)


def parse_moment_specs(items: list[str] | None) -> dict[str, list[float]]:
    moments: dict[str, list[float]] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --moment {raw!r}; use Element=value[,value].")
        element, values = raw.split("=", 1)
        element = element.strip()
        if not element:
            raise ValueError(f"Invalid --moment {raw!r}; missing element.")
        parsed: list[float] = []
        for value in values.split(","):
            text = value.strip()
            if not text:
                continue
            try:
                parsed.append(abs(float(text)))
            except ValueError as exc:
                raise ValueError(f"Invalid magnetic moment value {text!r} in {raw!r}.") from exc
        if not parsed:
            raise ValueError(f"Invalid --moment {raw!r}; no values were provided.")
        existing = moments.setdefault(element, [])
        for value in parsed:
            if value not in existing:
                existing.append(value)
    return moments


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1.0e-10:
        return str(int(round(value)))
    return f"{value:g}"


def build_guard_specs(
    magnetic_elements: list[str],
    nonmagnetic_elements: list[str],
    moment_specs: dict[str, list[float]],
    guard_tol: float,
    nonmagnetic_tol: float,
    explicit_guards: list[str] | None,
) -> list[str]:
    if explicit_guards:
        return split_items(explicit_guards)
    guards: list[str] = []
    for element in magnetic_elements:
        magnitudes = moment_specs.get(element) or [1.0]
        targets: list[str] = []
        for magnitude in magnitudes:
            targets.append(format_number(magnitude))
            targets.append(format_number(-magnitude))
        guards.append(f"{element}={','.join(targets)}@{format_number(guard_tol)}")
    for element in nonmagnetic_elements:
        guards.append(f"{element}=0@{format_number(nonmagnetic_tol)}")
    return guards


def quick_pseudo_species_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    magnetic_elements = split_items(args.magnetic_element)
    nonmagnetic_elements = split_items(args.nonmagnetic_element)
    moment_specs = parse_moment_specs(args.moment)
    rows: list[dict[str, str]] = []
    for element in magnetic_elements:
        for magnitude in moment_specs.get(element) or [1.0]:
            mag = format_number(magnitude)
            rows.append(
                PseudoSpecies(
                    f"{safe_name(element)}{safe_name(mag)}p",
                    element,
                    "spin_state",
                    f"+{mag}",
                    "",
                    "magnetic_sublattice",
                    f"{element}={mag},{format_number(-magnitude)}@{format_number(args.moment_guard_tol)}",
                    element,
                    "Quick optimization pseudo-species for spin-up branch.",
                ).row()
            )
            rows.append(
                PseudoSpecies(
                    f"{safe_name(element)}{safe_name(mag)}m",
                    element,
                    "spin_state",
                    f"{format_number(-magnitude)}",
                    "",
                    "magnetic_sublattice",
                    f"{element}={mag},{format_number(-magnitude)}@{format_number(args.moment_guard_tol)}",
                    element,
                    "Quick optimization pseudo-species for spin-down branch.",
                ).row()
            )
    for element in nonmagnetic_elements:
        rows.append(
            PseudoSpecies(
                safe_name(element),
                element,
                "nonmagnetic_species",
                "0",
                "",
                "nonmagnetic_sublattice",
                f"{element}=0@{format_number(args.nonmagnetic_tolerance)}",
                element,
                "Nonmagnetic guard used during fail-fast screening.",
            ).row()
        )
    return rows


def tool_status(tool: str) -> dict[str, str | bool]:
    path = shutil.which(tool)
    return {"available": bool(path), "path": path or ""}


def inspect_atat_environment() -> dict[str, Any]:
    grouped = {
        group: {tool: tool_status(tool) for tool in tools}
        for group, tools in ATAT_TOOLS.items()
    }
    flat = {tool: tool_status(tool) for tools in ATAT_TOOLS.values() for tool in tools}
    ready = {
        "can_generate_sqs": bool(flat.get("mcsqs", {}).get("available")),
        "can_fit_cluster_expansion": bool(flat.get("maps", {}).get("available") or flat.get("mmaps", {}).get("available")),
        "can_compute_correlations": bool(flat.get("corrdump", {}).get("available")),
        "can_convert_structures": bool(
            flat.get("str2poscar", {}).get("available") or flat.get("poscar2str", {}).get("available")
        ),
    }
    suggestions = []
    if not ready["can_generate_sqs"]:
        suggestions.append("mcsqs not found on PATH; load or install ATAT before SQS generation.")
    if not ready["can_fit_cluster_expansion"]:
        suggestions.append("maps/mmaps not found on PATH; cluster-expansion fitting is not ready.")
    if not ready["can_convert_structures"]:
        suggestions.append("str2poscar/poscar2str not found; keep conversion manual or add ATAT tools to PATH.")
    if all(ready.values()):
        suggestions.append("ATAT core tools are visible; Atomi can use this environment as an external workflow bridge.")
    return {
        "schema": STATUS_SCHEMA,
        "path": os.environ.get("PATH", ""),
        "executables": grouped,
        "ready": ready,
        "suggestions": suggestions,
    }


def print_status(report: dict[str, Any]) -> None:
    print("Atomi ATAT bridge status")
    for group, tools in report["executables"].items():
        print(f"{group}:")
        for tool, info in tools.items():
            label = "OK" if info["available"] else "missing"
            print(f"  {tool:<12} {label:<8} {info['path']}")
    print("Ready:")
    for key, value in report["ready"].items():
        print(f"  {key:<28} {value}")
    if report["suggestions"]:
        print("Notes:")
        for item in report["suggestions"]:
            print(f"  - {item}")


def parse_pseudo_species(items: list[str] | None) -> list[PseudoSpecies]:
    species = []
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(
                "Use --pseudo-species LABEL=element,role,spin,charge,sublattice,guard,notes"
            )
        label, spec = raw.split("=", 1)
        parts = [part.strip() for part in spec.split(",")]
        while len(parts) < 7:
            parts.append("")
        species.append(
            PseudoSpecies(
                pseudo_species=label.strip(),
                element=parts[0],
                role=parts[1] or "occupational_state",
                spin_value=parts[2],
                charge_state=parts[3],
                sublattice=parts[4],
                moment_guard=parts[5],
                vasp_element=parts[0],
                notes=parts[6],
            )
        )
    return species


def default_pseudo_species(args: argparse.Namespace) -> list[PseudoSpecies]:
    rows: list[PseudoSpecies] = []
    host = args.host
    dopants = args.dopant or []
    oxygen = args.oxygen
    if host:
        if host == "U":
            rows.extend(
                [
                    PseudoSpecies("U4p", "U", "host_valence_spin", "+2", "4+", "cation", "U=2,-2,1,-1@0.7"),
                    PseudoSpecies("U4m", "U", "host_valence_spin", "-2", "4+", "cation", "U=2,-2,1,-1@0.7"),
                    PseudoSpecies("U5p", "U", "host_valence_spin", "+1", "5+", "cation", "U=2,-2,1,-1@0.7"),
                    PseudoSpecies("U5m", "U", "host_valence_spin", "-1", "5+", "cation", "U=2,-2,1,-1@0.7"),
                ]
            )
        else:
            rows.extend(
                [
                    PseudoSpecies(f"{safe_name(host)}p", host, "host_spin_state", "+", "", "cation"),
                    PseudoSpecies(f"{safe_name(host)}m", host, "host_spin_state", "-", "", "cation"),
                ]
            )
    for dopant in dopants:
        if dopant == "Gd":
            rows.extend(
                [
                    PseudoSpecies("Gdp", "Gd", "dopant_spin_state", "+7", "3+", "cation", "Gd=7,-7@0.6"),
                    PseudoSpecies("Gdm", "Gd", "dopant_spin_state", "-7", "3+", "cation", "Gd=7,-7@0.6"),
                ]
            )
        else:
            label = safe_name(dopant)
            rows.extend(
                [
                    PseudoSpecies(f"{label}p", dopant, "dopant_spin_state", "+", "", "cation"),
                    PseudoSpecies(f"{label}m", dopant, "dopant_spin_state", "-", "", "cation"),
                ]
            )
    if oxygen:
        rows.append(PseudoSpecies(oxygen, oxygen, "anion", "0", "2-", "anion", f"{oxygen}=0@0.25"))
        rows.append(PseudoSpecies(args.vacancy_label, oxygen, "vacancy", "0", "", "anion", "", "oxygen vacancy"))
    return rows


def build_species_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    species = default_pseudo_species(args)
    species.extend(parse_pseudo_species(args.pseudo_species))
    seen: set[str] = set()
    rows = []
    for item in species:
        if item.pseudo_species in seen:
            continue
        seen.add(item.pseudo_species)
        rows.append(item.row())
    return rows


def write_stage_map(path: Path) -> None:
    write_csv(
        path,
        WORKFLOW_STAGES,
        ["stage_id", "atat_role", "atomi_role", "typical_inputs", "typical_outputs"],
    )


def write_bridge_readme(path: Path, args: argparse.Namespace) -> None:
    text = f"""# Atomi-ATAT Bridge

System: {args.system}

This workspace treats ATAT as the lattice-configuration engine and Atomi as the
DFT preparation, spin guard, fail-fast screening, SD/DD, zentropy, and MLIP
bookkeeping layer.

Core logic:

1. Encode cation/anion/defect/spin-valence states as pseudo-species in
   `pseudo_species_map.csv`.
2. Use ATAT to enumerate, fit cluster expansions, or generate SQS structures.
3. Convert selected ATAT structures to VASP-ready seed folders.
4. Use Atomi spin branching and fail-fast VASP screening to reject unphysical
   or high-energy configurations.
5. Feed accepted structures and energies to SD/DD, zentropy motif databases,
   MLIP training, or CALPHAD pseudo-data export.

Important: pseudo-species such as U5p or Gdp are bookkeeping labels. Always use
`vasp-spin-report` or `vasp-branch-live` moment guards to verify that DFT
preserved the intended local moment/valence character.
"""
    path.write_text(text, encoding="utf-8")


def init_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="atat-bridge init",
        description="Create an ATAT-to-Atomi bridge workspace for defect thermodynamics.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("atat_bridge"))
    parser.add_argument("--system", default="(Gd,U)O2-x")
    parser.add_argument("--host", default="U")
    parser.add_argument("--dopant", action="append", default=["Gd"])
    parser.add_argument("--oxygen", default="O")
    parser.add_argument("--vacancy-label", default="V_O")
    parser.add_argument(
        "--pseudo-species",
        action="append",
        default=[],
        help="Add/override pseudo species: LABEL=element,role,spin,charge,sublattice,guard,notes.",
    )
    args = parser.parse_args(argv)
    root = args.outdir.resolve()
    for stage in WORKFLOW_STAGES:
        (root / stage["stage_id"]).mkdir(parents=True, exist_ok=True)
    species_rows = build_species_rows(args)
    write_csv(root / "pseudo_species_map.csv", species_rows, SPECIES_FIELDS)
    write_stage_map(root / "atat_atomi_stage_map.csv")
    write_json(
        root / "atat_bridge_plan.json",
        {
            "schema": SCHEMA,
            "system": args.system,
            "host": args.host,
            "dopants": args.dopant,
            "oxygen": args.oxygen,
            "vacancy_label": args.vacancy_label,
            "workflow_stages": WORKFLOW_STAGES,
            "pseudo_species": species_rows,
            "atat_status": inspect_atat_environment(),
            "handoff_commands": {
                "index_atat_candidates": "atat-bridge index --root 01_motif_search --out atat_candidate_index.csv",
                "build_ce_training_set": "atat-bridge ce-handoff --summary-csv stage1_screen/stage1_branch_summary.csv --outdir 02_sd_dd_cluster_expansion/ce_handoff",
                "fail_fast_screen": "vasp-branch-live --runlist runlist.txt --log-dir . --outdir stage1_screen --moment-guard ...",
                "spin_report": "vasp-spin-report --runlist runlist.txt --log-dir . --output-prefix spin_energy --moment-guard ...",
                "sd_dd": "defect-chem build-defects ... ; defect-chem run ...",
                "zentropy": "zentropy_motif_db index ... ; zentropy-free-energy ... ; zentropy-solve ...",
            },
        },
    )
    write_bridge_readme(root / "ATAT_ATOMI_BRIDGE_NOTES.md", args)
    (root / "01_motif_search" / "README.md").write_text(
        "Place ATAT lat.in, clusters.out, enum/fit outputs, or converted structures here.\n",
        encoding="utf-8",
    )
    (root / "03_sqs_mid_high_concentration" / "README.md").write_text(
        "Place rndstr.in, bestsqs.out, and SQS composition notes here.\n",
        encoding="utf-8",
    )
    print(f"Wrote ATAT bridge workspace : {root}")
    print(f"Pseudo-species map          : {root / 'pseudo_species_map.csv'}")
    print(f"Stage map                   : {root / 'atat_atomi_stage_map.csv'}")
    print(f"Plan                        : {root / 'atat_bridge_plan.json'}")
    print("Next                        : run atat-doctor, then fill lat.in/rndstr.in or index ATAT candidates.")


def classify_candidate(path: Path) -> str:
    name = path.name.lower()
    if "bestsqs" in name or "rndstr" in name:
        return "sqs"
    if name.startswith("str") or "enum" in name:
        return "enumerated_structure"
    if "lat" in name:
        return "lattice_definition"
    return "atat_structure"


def index_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="atat-bridge index",
        description="Index ATAT-generated structure/candidate files for Atomi handoff.",
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("atat_candidate_index.csv"))
    parser.add_argument("--pattern", action="append", default=["bestsqs*.out", "str*.out", "rndstr*.out", "lat.in"])
    parser.add_argument("--target-stage", default="fail_fast")
    args = parser.parse_args(argv)
    candidates: list[Path] = []
    for pattern in args.pattern:
        candidates.extend(path for path in args.root.rglob(pattern) if path.is_file())
    unique = sorted({path.resolve() for path in candidates})
    rows = []
    for index, path in enumerate(unique, start=1):
        kind = classify_candidate(path)
        rows.append(
            {
                "candidate_id": f"atat_{index:04d}",
                "path": str(path),
                "source_kind": kind,
                "target_stage": args.target_stage,
                "atomi_next": "convert_to_vasp_then_vasp-branch-live",
                "notes": "Use pseudo_species_map.csv to restore element/spin/MAGMOM semantics.",
            }
        )
    write_csv(
        args.out,
        rows,
        ["candidate_id", "path", "source_kind", "target_stage", "atomi_next", "notes"],
    )
    print(f"ATAT candidates indexed : {len(rows)}")
    print(f"Candidate index         : {args.out.resolve()}")


def first_present(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return ""


def row_physics_status(row: dict[str, str]) -> str:
    return first_present(row, ["physics_guard_status", "guard", "spin_status"]) or "NOT_APPLIED"


def row_decision(row: dict[str, str]) -> str:
    return first_present(row, ["action", "decision", "status"]) or "unknown"


def row_is_accepted(row: dict[str, str], include_warning: bool, include_unchecked: bool) -> bool:
    physics = row_physics_status(row)
    if physics in {"FAIL", "NO_MATCHED_ELEMENTS"}:
        return False
    if physics in {"NOT_APPLIED", ""} and not include_unchecked:
        return False
    decision = row_decision(row).lower()
    if decision in {"stop", "bad", "error", "missing", "nodir"}:
        return False
    if decision in {"warning", "warn"}:
        return include_warning
    return True


def infer_structure_path(row: dict[str, str]) -> str:
    for key in ("structure_path", "poscar", "POSCAR"):
        if row.get(key):
            return row[key]
    for key in ("output_run_dir", "resolved_run", "run_dir", "run"):
        value = row.get(key)
        if value:
            return str(Path(value) / "POSCAR")
    return ""


def candidate_lookup(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    lookup: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        candidate_id = row.get("candidate_id") or ""
        for key in ("path", "run", "run_dir", "structure_path"):
            value = row.get(key)
            if value:
                lookup[str(Path(value).resolve())] = row
        if candidate_id:
            lookup[candidate_id] = row
    return lookup


def match_candidate(row: dict[str, str], lookup: dict[str, dict[str, str]]) -> dict[str, str] | None:
    for key in ("atat_candidate_id", "candidate_id"):
        value = row.get(key)
        if value and value in lookup:
            return lookup[value]
    for key in ("structure_path", "output_run_dir", "resolved_run", "run_dir", "run"):
        value = row.get(key)
        if not value:
            continue
        path = Path(value)
        keys = [str(path.resolve())]
        if path.is_dir():
            keys.append(str((path / "POSCAR").resolve()))
        for item in keys:
            if item in lookup:
                return lookup[item]
    return None


def ce_training_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    lookup = candidate_lookup(args.candidate_index)
    rows: list[dict[str, Any]] = []
    for source in args.summary_csv:
        for source_row, row in enumerate(read_csv(source), start=1):
            if not row_is_accepted(row, args.include_warning, args.include_unchecked):
                continue
            energy = finite_float(first_present(row, ["energy_eV", "energy", "E_eV"]))
            if energy is None:
                continue
            energy_per_fu = energy / args.formula_units
            candidate = match_candidate(row, lookup)
            structure_path = infer_structure_path(row)
            rows.append(
                {
                    "training_id": f"ce_{len(rows) + 1:05d}",
                    "source_csv": str(source),
                    "source_row": source_row,
                    "run": first_present(row, ["run", "run_dir", "branch_id", "resolved_run"]),
                    "resolved_run": first_present(row, ["resolved_run", "run_dir", "run"]),
                    "output_run_dir": row.get("output_run_dir", ""),
                    "structure_path": structure_path,
                    "energy_eV": f"{energy:.12g}",
                    "energy_eV_per_fu": f"{energy_per_fu:.12g}",
                    "relative_energy_eV_per_fu": "",
                    "physics_status": row_physics_status(row),
                    "mag_status": row.get("mag_status", ""),
                    "decision": row_decision(row),
                    "element_order": row.get("element_order", ""),
                    "changed_by_element": row.get("changed_by_element", ""),
                    "atat_candidate_id": "" if candidate is None else candidate.get("candidate_id", ""),
                    "motif_family": first_present(row, ["motif_family", "frame_id", "family"]),
                    "use_for": args.use_for,
                    "notes": "accepted by Atomi physics/decision filters; convert structure_path to ATAT str.out for CE fitting",
                }
            )
    if rows:
        minimum = min(float(row["energy_eV_per_fu"]) for row in rows)
        for row in rows:
            row["relative_energy_eV_per_fu"] = f"{float(row['energy_eV_per_fu']) - minimum:.12g}"
    return rows


def write_ce_command_notes(path: Path, args: argparse.Namespace) -> None:
    text = f"""# ATAT CE / MC handoff generated by Atomi

# Inputs prepared by Atomi:
#   {args.outdir / 'ce_training_set.csv'}
#   {args.outdir / 'atat_ce_manifest.json'}
#
# Recommended flow:
# 1. Convert accepted structure_path entries to ATAT str.out-like structures.
# 2. Use ATAT maps/mmaps/corrdump to fit a cluster expansion from the accepted
#    DFT energies.
# 3. Use ATAT emc2 or your preferred Monte Carlo runner to estimate
#    composition/T-dependent motif populations.
# 4. Normalize MC/population outputs into:
#      {args.outdir / 'atat_mc_population_handoff_template.csv'}
#    and pass them to zentropy-free-energy / zentropy-solve as motif-family
#    priors or pseudo-data.
#
# These command sketches intentionally avoid fixed flags because ATAT lat.in,
# composition variables, and pseudo-species encoding are system-specific.
# Keep pseudo-species labels synchronized with pseudo_species_map.csv and
# validate them with vasp-branch-live / vasp-spin-report moment guards.
"""
    path.write_text(text, encoding="utf-8")


def ce_handoff_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="atat-bridge ce-handoff",
        description="Build an ATAT cluster-expansion training handoff from accepted Atomi DFT rows.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        action="append",
        required=True,
        help="Atomi stage1_branch_summary.csv or spin_energy_run_summary.csv; repeatable.",
    )
    parser.add_argument("--candidate-index", type=Path, help="Optional atat_candidate_index.csv to link structures.")
    parser.add_argument("--outdir", type=Path, default=Path("atat_ce_handoff"))
    parser.add_argument("--formula-units", type=float, default=1.0, help="Formula units represented by each energy.")
    parser.add_argument("--include-warning", action="store_true", help="Keep Atomi warning rows in the CE candidate set.")
    parser.add_argument(
        "--include-unchecked",
        action="store_true",
        help="Keep rows where no physics moment guard was applied.",
    )
    parser.add_argument("--use-for", default="ce_fit,sd_dd,zentropy", help="Free-form downstream role label.")
    args = parser.parse_args(argv)
    if args.formula_units <= 0:
        raise ValueError("--formula-units must be positive.")
    rows = ce_training_rows(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    training_csv = args.outdir / "ce_training_set.csv"
    write_csv(training_csv, rows, CE_TRAINING_FIELDS)
    interaction_template = args.outdir / "sd_dd_interaction_template.csv"
    write_csv(
        interaction_template,
        [
            {
                "interaction_id": "atat_ce_pair_or_cluster_001",
                "source": "ATAT_CE",
                "species_or_cluster": "",
                "effective_interaction_eV": "",
                "temperature_K": "",
                "composition": "",
                "notes": "Fill from fitted CE/ECI or coarse-grained pair binding values.",
            }
        ],
        ["interaction_id", "source", "species_or_cluster", "effective_interaction_eV", "temperature_K", "composition", "notes"],
    )
    population_template = args.outdir / "atat_mc_population_handoff_template.csv"
    write_csv(
        population_template,
        [
            {
                "T_K": "",
                "composition": "",
                "oxygen_delta": "",
                "motif_family": "",
                "motif_id": "",
                "probability": "",
                "G_eV_per_fu": "",
                "source": "ATAT_MC",
                "notes": "Use as zentropy motif prior or pseudo-data after MC normalization.",
            }
        ],
        ["T_K", "composition", "oxygen_delta", "motif_family", "motif_id", "probability", "G_eV_per_fu", "source", "notes"],
    )
    command_notes = args.outdir / "atat_ce_commands.md"
    write_ce_command_notes(command_notes, args)
    manifest = {
        "schema": "atomi.atat.ce_handoff.v1",
        "summary_csv": [str(path) for path in args.summary_csv],
        "candidate_index": "" if args.candidate_index is None else str(args.candidate_index),
        "formula_units": args.formula_units,
        "n_training_rows": len(rows),
        "filters": {
            "include_warning": args.include_warning,
            "include_unchecked": args.include_unchecked,
            "rejected_physics_status": ["FAIL", "NO_MATCHED_ELEMENTS"],
            "rejected_decisions": ["stop", "bad", "error", "missing", "nodir"],
        },
        "outputs": {
            "ce_training_set": str(training_csv),
            "sd_dd_interaction_template": str(interaction_template),
            "atat_mc_population_handoff_template": str(population_template),
            "atat_ce_commands": str(command_notes),
        },
        "downstream": {
            "sd_dd": "Use interaction template or coarse-grained CE interactions in defect-chem build-defects / solution-model fitting.",
            "zentropy": "Use MC population handoff as motif-family priors or pseudo-data beside microstate G_i(T).",
            "fail_fast": "Return new ATAT-selected structures to vasp-branch-live before expensive DFT.",
        },
    }
    write_json(args.outdir / "atat_ce_manifest.json", manifest)
    print(f"CE training rows       : {len(rows)}")
    print(f"CE training set        : {training_csv.resolve()}")
    print(f"CE manifest            : {(args.outdir / 'atat_ce_manifest.json').resolve()}")
    print(f"MC population template : {population_template.resolve()}")


def prepare_quick_template(args: argparse.Namespace, root: Path) -> tuple[Path, list[str]]:
    template = args.template.expanduser().resolve()
    if not template.is_dir():
        raise FileNotFoundError(f"VASP template directory not found: {template}")
    source_poscar = args.poscar.expanduser().resolve() if args.poscar else template / "POSCAR"
    if not source_poscar.is_file():
        raise FileNotFoundError(
            "POSCAR not found. Pass --poscar, or provide POSCAR inside --template."
        )

    target = root / "00_vasp_template"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_poscar, target / "POSCAR")
    missing: list[str] = []
    for name in ("INCAR", "KPOINTS", "POTCAR"):
        src = template / name
        if src.is_file():
            shutil.copy2(src, target / name)
        else:
            missing.append(name)
    return target, missing


def quick_command_rows(args: argparse.Namespace, guards: list[str]) -> list[dict[str, str]]:
    dopants = split_items(args.dopant)
    hosts = split_items(args.host)
    magnetic_elements = split_items(args.magnetic_element)
    spin_parts = [
        "magit enum",
        "--template 00_vasp_template",
        "--output-root 02_spin_candidates",
        "--runlist runlist.txt",
        "--index spin_index.csv",
    ]
    if dopants or hosts:
        for element in dopants:
            spin_parts.append(f"--dopant {element}")
        for element in hosts:
            spin_parts.append(f"--host {element}")
    else:
        for element in magnetic_elements:
            spin_parts.append(f"--element {element}")
    for raw in args.moment or []:
        spin_parts.append(f"--moment {raw}")
    spin_parts.extend(
        [
            f"--dopant-mode {args.spin_mode}",
            f"--max-configs {args.max_configs}",
        ]
    )
    if args.truncate:
        spin_parts.append("--truncate")

    guard_parts = [f"--moment-guard {guard}" for guard in guards]
    live_parts = [
        "vasp-branch-live",
        "--runlist runlist.txt",
        "--log-dir .",
        "--outdir 03_fail_fast",
        f"--single-frame-id {safe_name(args.system)}_{safe_name(args.supercell)}",
        f"--keep-per-frame {args.keep_per_frame}",
        f"--stopped-after-min {format_number(args.stopped_after_min)}",
        f"--refresh {format_number(args.refresh)}",
    ] + guard_parts
    screen_parts = [
        "vasp-branch-screen",
        "--runlist runlist.txt",
        "--log-dir .",
        "--outdir 03_fail_fast",
        f"--single-frame-id {safe_name(args.system)}_{safe_name(args.supercell)}",
        f"--keep-per-frame {args.keep_per_frame}",
        f"--stopped-after-min {format_number(args.stopped_after_min)}",
    ] + guard_parts
    report_parts = [
        "vasp-spin-report",
        "--runlist runlist.txt",
        "--spin-index spin_index.csv",
        "--log-dir .",
        "--output-prefix 04_final_report/spin_energy",
        f"--stopped-after-min {format_number(args.stopped_after_min)}",
    ] + guard_parts
    ce_parts = [
        "atat-bridge ce-handoff",
        "--summary-csv 03_fail_fast/stage1_branch_summary.csv",
        "--outdir 05_atat_ce_handoff",
        "--include-warning",
    ]
    return [
        {
            "step": "00_atat_status",
            "purpose": "Check whether external ATAT tools are visible on this machine.",
            "command": "atat-doctor",
        },
        {
            "step": "01_optional_atat_index",
            "purpose": "After external ATAT enumeration/SQS files are placed in 01_atat_candidates, index them for Atomi.",
            "command": "atat-bridge index --root 01_atat_candidates --out atat_candidate_index.csv",
        },
        {
            "step": "02_spin_candidates",
            "purpose": "Generate compact spin/local-moment branches from the VASP template.",
            "command": shell_join(spin_parts),
        },
        {
            "step": "03_array_dft",
            "purpose": "Submit your VASP array workflow against runlist.txt; Atomi leaves scheduler submission to your local script.",
            "command": "# sbatch your_vasp_array_script.sbatch runlist.txt",
        },
        {
            "step": "04_live_fail_fast",
            "purpose": "Scan running/stopped branches, show energy and spin health, and write survivor tables.",
            "command": shell_join(live_parts),
        },
        {
            "step": "05_one_shot_screen",
            "purpose": "Run the same branch ranking once after jobs stop or finish.",
            "command": shell_join(screen_parts),
        },
        {
            "step": "06_spin_energy_report",
            "purpose": "Generate final energy vs magnetic-moment tables and plots.",
            "command": shell_join(report_parts),
        },
        {
            "step": "07_atat_ce_handoff",
            "purpose": "Prepare accepted low-energy rows for ATAT cluster expansion or MC population work.",
            "command": shell_join(ce_parts),
        },
    ]


def write_quick_commands(path: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    lines = [
        f"# Quick Materials Optimization: {args.system}",
        "",
        f"Formula: {args.formula}",
        f"Starting cell: {args.supercell}",
        "",
        "Run these commands from this workspace directory.",
        "The spin-only path is ready immediately after this scaffold is created.",
        "The ATAT path becomes active after you add ATAT lat.in/rndstr.in or structure outputs to 01_atat_candidates.",
        "",
    ]
    for row in rows:
        lines.extend([f"## {row['step']}", row["purpose"], "", "```bash", row["command"], "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quick_shell(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated command sketch. Review before running on an HPC login node.",
        "",
    ]
    for row in rows:
        lines.extend([f"# {row['step']}: {row['purpose']}", row["command"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def quick_opt_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="materials-opt",
        description="Create a compact ATAT/Atomi quick optimization scaffold for spin and lattice candidates.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("materials_opt_quick"))
    parser.add_argument("--system", default="UC2")
    parser.add_argument("--formula", default="UC2")
    parser.add_argument("--supercell", default="2x1x1")
    parser.add_argument("--poscar", type=Path, help="Starting POSCAR. If omitted, use POSCAR from --template.")
    parser.add_argument("--template", type=Path, default=Path("VASP_TEMPLATE"))
    parser.add_argument("--magnetic-element", action="append", default=[], help="Magnetic element to branch, repeatable.")
    parser.add_argument("--nonmagnetic-element", action="append", default=[], help="Nonmagnetic element guard, repeatable.")
    parser.add_argument("--dopant", action="append", default=[], help="Optional dopant-style magnetic element for magit enum.")
    parser.add_argument("--host", action="append", default=[], help="Optional host magnetic element for magit enum.")
    parser.add_argument("--moment", action="append", default=[], help="Moment magnitude, e.g. U=2 or U=2,1.")
    parser.add_argument("--moment-guard", action="append", default=[], help="Explicit moment guard, e.g. U=2,-2@0.7.")
    parser.add_argument("--moment-guard-tol", type=float, default=0.7)
    parser.add_argument("--nonmagnetic-tolerance", type=float, default=0.25)
    parser.add_argument("--spin-mode", choices=("all", "fm", "afm", "both"), default="all")
    parser.add_argument("--max-configs", type=int, default=16)
    parser.add_argument("--no-truncate", dest="truncate", action="store_false", default=True)
    parser.add_argument("--keep-per-frame", type=int, default=2)
    parser.add_argument("--stopped-after-min", type=float, default=15.0)
    parser.add_argument("--refresh", type=float, default=10.0)
    args = parser.parse_args(argv)

    moment_specs = parse_moment_specs(args.moment)
    magnetic_elements = split_items(args.magnetic_element)
    for element in split_items(args.dopant) + split_items(args.host):
        if element not in magnetic_elements:
            magnetic_elements.append(element)
    if not magnetic_elements:
        magnetic_elements = list(moment_specs)
    if not magnetic_elements:
        raise ValueError("Provide --magnetic-element or --moment Element=value for spin branching.")
    args.magnetic_element = magnetic_elements
    nonmagnetic_elements = split_items(args.nonmagnetic_element)

    root = args.outdir.expanduser().resolve()
    for name in (
        "01_atat_candidates",
        "02_spin_candidates",
        "03_fail_fast",
        "04_final_report",
        "05_atat_ce_handoff",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    template_path, missing_template_files = prepare_quick_template(args, root)
    guards = build_guard_specs(
        magnetic_elements,
        nonmagnetic_elements,
        moment_specs,
        args.moment_guard_tol,
        args.nonmagnetic_tolerance,
        args.moment_guard,
    )
    species_rows = quick_pseudo_species_rows(args)
    command_rows = quick_command_rows(args, guards)

    write_csv(root / "pseudo_species_map.csv", species_rows, SPECIES_FIELDS)
    write_csv(root / "quick_opt_commands.csv", command_rows, QUICK_COMMAND_FIELDS)
    write_quick_commands(root / "QUICK_OPT_COMMANDS.md", command_rows, args)
    write_quick_shell(root / "quick_opt_commands.sh", command_rows)
    write_json(
        root / "quick_opt_plan.json",
        {
            "schema": QUICK_OPT_SCHEMA,
            "system": args.system,
            "formula": args.formula,
            "supercell": args.supercell,
            "magnetic_elements": magnetic_elements,
            "nonmagnetic_elements": nonmagnetic_elements,
            "moment_specs": {key: [format_number(value) for value in values] for key, values in moment_specs.items()},
            "moment_guards": guards,
            "max_configs": args.max_configs,
            "spin_mode": args.spin_mode,
            "template": str(template_path),
            "missing_template_files": missing_template_files,
            "outputs": {
                "commands": str(root / "QUICK_OPT_COMMANDS.md"),
                "shell_commands": str(root / "quick_opt_commands.sh"),
                "runlist_after_spin_enum": str(root / "runlist.txt"),
                "spin_index_after_spin_enum": str(root / "spin_index.csv"),
                "fail_fast_summary": str(root / "03_fail_fast" / "stage1_branch_summary.csv"),
                "stage2_survivors": str(root / "03_fail_fast" / "stage2_survivors_runlist.txt"),
                "spin_report": str(root / "04_final_report" / "spin_energy_run_summary.csv"),
            },
            "atat_status": inspect_atat_environment(),
            "notes": [
                "For a stoichiometric UC2 2x1x1 demo, the spin-only path can start with magit enum.",
                "Use ATAT in 01_atat_candidates when occupational, vacancy, or SQS lattice candidates are needed.",
                "Lowest-energy selection should use physics-accepted rows from 03_fail_fast or 04_final_report.",
            ],
        },
    )
    (root / "01_atat_candidates" / "README.md").write_text(
        "Optional ATAT area. Put lat.in/rndstr.in and ATAT structure outputs here, then run:\n"
        "  atat-bridge index --root 01_atat_candidates --out atat_candidate_index.csv\n",
        encoding="utf-8",
    )

    print(f"Quick optimization workspace : {root}")
    print(f"VASP template copy           : {template_path}")
    print(f"Pseudo-species map           : {root / 'pseudo_species_map.csv'}")
    print(f"Command guide                : {root / 'QUICK_OPT_COMMANDS.md'}")
    print(f"Plan                         : {root / 'quick_opt_plan.json'}")
    if missing_template_files:
        print(f"Template warning             : missing {', '.join(missing_template_files)}")
    print("Next                         : cd into the workspace and run the magit enum command from QUICK_OPT_COMMANDS.md.")


def status_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="atat-doctor", description="Report ATAT tool availability for Atomi bridges.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = inspect_atat_environment()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_status(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atat-bridge", description="Bridge ATAT lattice workflows to Atomi.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("status", help="Report ATAT executable availability.")
    subparsers.add_parser("doctor", help="Alias for status.")
    init_parser = subparsers.add_parser("init", help="Create an ATAT/Atomi bridge workspace.")
    init_parser.add_argument("args", nargs=argparse.REMAINDER)
    index_parser = subparsers.add_parser("index", help="Index ATAT candidate structure files.")
    index_parser.add_argument("args", nargs=argparse.REMAINDER)
    ce_parser = subparsers.add_parser("ce-handoff", help="Build a CE/MC handoff from accepted Atomi DFT rows.")
    ce_parser.add_argument("args", nargs=argparse.REMAINDER)
    quick_parser = subparsers.add_parser("quick-opt", help="Create a quick materials optimization scaffold.")
    quick_parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    if not argv:
        status_main([])
        return
    command, rest = argv[0], argv[1:]
    if command in {"status", "doctor"}:
        status_main(rest)
    elif command == "init":
        init_main(rest)
    elif command == "index":
        index_main(rest)
    elif command in {"ce-handoff", "ce", "ce-plan"}:
        ce_handoff_main(rest)
    elif command in {"quick-opt", "quick", "materials-opt"}:
        quick_opt_main(rest)
    else:
        parser = build_parser()
        parser.parse_args(argv)


def doctor_main(argv: list[str] | None = None) -> None:
    status_main(argv)


if __name__ == "__main__":
    main()
