"""Stage framework for zentropy-guided defect thermodynamics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCHEMA = "atomi.zentropy.workflow.v1"

STAGES: list[dict[str, Any]] = [
    {
        "id": "stage1_motif_db",
        "title": "Defect motif database",
        "goal": (
            "Index relaxed DFT motifs from different cell sizes with formula-unit "
            "normalization, magnetic moments, valence labels, degeneracy, and "
            "source provenance."
        ),
        "primary_command": "zentropy_motif_db index",
        "typical_outputs": ["defect_motif_db.json", "defect_motif_db.csv", "mlip_inputs/"],
        "status": "implemented",
    },
    {
        "id": "stage2_free_energy",
        "title": "Microstate free-energy assembly",
        "goal": (
            "Attach G_i(T), H_i(T), S_i(T), volume/lattice, defect chemistry, "
            "and uncertainty metadata to each motif using static DFT, QHA, MD, "
            "or MLIP-derived data."
        ),
        "primary_command": "planned: zentropy_free_energy",
        "typical_outputs": ["microstate_free_energy.csv", "microstate_free_energy.json"],
        "status": "framework",
    },
    {
        "id": "stage3_zentropy_solve",
        "title": "Zentropy ensemble solve",
        "goal": (
            "Combine motif free energies and degeneracies into equilibrium "
            "probabilities and macroscopic thermodynamic functions under "
            "user-selected composition, T, P, and oxygen-potential constraints."
        ),
        "primary_command": "planned: zentropy_solve",
        "typical_outputs": ["ensemble_probabilities.csv", "zentropy_thermo_functions.csv"],
        "status": "framework",
    },
    {
        "id": "stage4_calphad_export",
        "title": "CALPHAD and MOOSE bridge",
        "goal": (
            "Export zentropy-derived thermodynamic functions, defect populations, "
            "and property surfaces to CALPHAD/MOOSE-friendly tables for downstream "
            "fuel-performance and phase-stability workflows."
        ),
        "primary_command": "planned: zentropy_export",
        "typical_outputs": ["calphad_pseudodata.csv", "moose_material_table.csv"],
        "status": "framework",
    },
    {
        "id": "stage5_active_learning",
        "title": "MLIP/GNN active-learning loop",
        "goal": (
            "Select high-impact motifs, temperatures, and local environments for "
            "additional DFT/MLIP sampling when ensemble probabilities or "
            "experimental benchmarks show gaps."
        ),
        "primary_command": "planned: zentropy_active_learning",
        "typical_outputs": ["active_learning_candidates.csv", "mlip_training_manifest.json"],
        "status": "framework",
    },
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_stage_manifest_csv(path: Path, stages: list[dict[str, Any]]) -> None:
    fields = ["id", "title", "status", "primary_command", "goal", "typical_outputs"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage in stages:
            row = dict(stage)
            row["typical_outputs"] = ";".join(stage.get("typical_outputs", []))
            writer.writerow({field: row.get(field, "") for field in fields})


def default_stage_config(stage: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    base = {
        "schema": "atomi.zentropy.stage_config.v1",
        "stage": stage,
        "material": args.material,
        "phase": args.phase,
        "parent_formula": args.parent_formula,
        "host_cation": args.host_cation,
        "guest_cations": args.guest_cation,
        "oxygen": args.oxygen,
    }
    if stage["id"] == "stage1_motif_db":
        base["inputs"] = {
            "dft_run_roots": [],
            "metadata_csv": "motif_metadata.csv",
            "site_state_csv": "site_states.csv",
        }
        base["outputs"] = {
            "db": "defect_motif_db.json",
            "csv": "defect_motif_db.csv",
            "mlip_inputs": "mlip_inputs/",
        }
        base["notes"] = [
            "Use metadata_csv/site_state_csv to carry defect labels, site valence, "
            "MAGMOM, charge state, degeneracy, and motif-family tags.",
            "DFT motifs from different supercell sizes should be compared through "
            "formula-unit-normalized energies and compositions.",
        ]
    elif stage["id"] == "stage2_free_energy":
        base["inputs"] = {
            "motif_db": "../stage1_motif_db/defect_motif_db.json",
            "qha_dirs": [],
            "thermo_qha_md_dirs": [],
            "mlip_md_dirs": [],
        }
        base["outputs"] = {
            "microstate_free_energy": "microstate_free_energy.csv",
            "uncertainty": "microstate_uncertainty.json",
        }
        base["notes"] = [
            "Attach G_i(T) per motif before the ensemble solve; do not mix cell "
            "sizes without formula-unit normalization.",
            "Magnetic and valence labels remain motif metadata so they can become "
            "ensemble-state variables later.",
        ]
    elif stage["id"] == "stage3_zentropy_solve":
        base["inputs"] = {
            "microstate_free_energy": "../stage2_free_energy/microstate_free_energy.csv",
            "runtime": "pyzentropy if available; otherwise Atomi planned fallback/export bridge",
        }
        base["outputs"] = {
            "probabilities": "ensemble_probabilities.csv",
            "thermo": "zentropy_thermo_functions.csv",
        }
        base["notes"] = [
            "Use high-probability motif pruning only after recording why "
            "low-probability states are safe to omit.",
            "Keep experimental benchmark data separate from fitted state data to "
            "avoid hidden double counting.",
        ]
    elif stage["id"] == "stage4_calphad_export":
        base["inputs"] = {
            "zentropy_thermo": "../stage3_zentropy_solve/zentropy_thermo_functions.csv",
            "defect_populations": "../stage3_zentropy_solve/ensemble_probabilities.csv",
        }
        base["outputs"] = {
            "calphad_pseudodata": "calphad_pseudodata.csv",
            "moose_material_table": "moose_material_table.csv",
        }
    elif stage["id"] == "stage5_active_learning":
        base["inputs"] = {
            "motif_db": "../stage1_motif_db/defect_motif_db.json",
            "ensemble_probabilities": "../stage3_zentropy_solve/ensemble_probabilities.csv",
            "experimental_residuals": [],
        }
        base["outputs"] = {
            "candidate_structures": "active_learning_candidates.csv",
            "mlip_manifest": "mlip_training_manifest.json",
        }
    return base


def build_workflow_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "material": args.material,
        "phase": args.phase,
        "parent_formula": args.parent_formula,
        "host_cation": args.host_cation,
        "guest_cations": args.guest_cation,
        "oxygen": args.oxygen,
        "stage_order": [stage["id"] for stage in STAGES],
        "stages": STAGES,
        "runtime_strategy": {
            "preferred_package": "pyzentropy",
            "base_install": "not required",
            "optional_extra": "atomi[zentropy]",
            "external_configuration": {
                "python": "ATOMI_ZENTROPY_PYTHON",
                "environment": "ATOMI_ZENTROPY_ENV",
                "executable": "ATOMI_ZENTROPY_EXE",
            },
            "note": (
                "Atomi can use pyzentropy from the active Python environment, "
                "or record/export data for a separate zentropy environment."
            ),
        },
    }


def init_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="zentropy_workflow init",
        description="Create a staged zentropy-guided defect-thermodynamics workspace.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("zentropy_workflow"))
    parser.add_argument("--material", default="(Gd,U)O2")
    parser.add_argument("--phase", default="defect-fluorite")
    parser.add_argument("--parent-formula", default="UO2")
    parser.add_argument("--host-cation", default="U")
    parser.add_argument("--guest-cation", action="append", default=None)
    parser.add_argument("--oxygen", default="O")
    args = parser.parse_args(argv)
    args.guest_cation = args.guest_cation or ["Gd"]

    args.outdir.mkdir(parents=True, exist_ok=True)
    workflow = build_workflow_payload(args)
    write_json(args.outdir / "zentropy_workflow.json", workflow)
    write_stage_manifest_csv(args.outdir / "stage_manifest.csv", STAGES)
    for stage in STAGES:
        stage_dir = args.outdir / stage["id"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        write_json(stage_dir / "stage_config.json", default_stage_config(stage, args))

    print(f"Wrote zentropy workflow : {args.outdir.resolve()}")
    print(f"Stage manifest          : {(args.outdir / 'stage_manifest.csv').resolve()}")
    print("Next stage              : stage1_motif_db with zentropy_motif_db index")


def stages_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="zentropy_workflow stages",
        description="Print the staged Atomi zentropy workflow map.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)
    if args.json:
        print(json.dumps({"schema": SCHEMA, "stages": STAGES}, indent=2))
        return
    for stage in STAGES:
        print(f"{stage['id']}: {stage['title']} [{stage['status']}]")
        print(f"  command: {stage['primary_command']}")
        print(f"  goal   : {stage['goal']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy_workflow",
        description="Create and inspect staged zentropy-guided defect thermodynamics workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Create a staged workspace.")
    subparsers.add_parser("stages", help="Print the stage map.")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    if not argv:
        build_parser().parse_args(argv)
    command, rest = argv[0], argv[1:]
    if command == "init":
        init_main(rest)
        return
    if command == "stages":
        stages_main(rest)
        return
    build_parser().error(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
