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
    else:
        parser = build_parser()
        parser.parse_args(argv)


def doctor_main(argv: list[str] | None = None) -> None:
    status_main(argv)


if __name__ == "__main__":
    main()
