"""Engine-neutral MD-to-entropy workflow for CP2K and LAMMPS.

This module coordinates the preparation, physical-guard, and entropy layers.
It does not replace the simulation engines or SLUSCHI.  Instead, it records
the state lineage and prevents an unguarded trajectory from being promoted to
a Route-C entropy row.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Any

from atomi.sluschi import route_c


SCHEMA_MD_ENTROPY_PLAN = "atomi.md.entropy.plan.v1"
SCHEMA_MD_ENTROPY_GATE = "atomi.md.entropy.gate.v1"
SCHEMA_MD_ENTROPY_RESULT = "atomi.md.entropy.result.v1"

GUARD_STATES = ("pass", "warning", "fail", "missing")
QUALITY_LEVELS = ("descriptor", "screening-prior", "production")
QUALITY_RANK = {name: idx for idx, name in enumerate(QUALITY_LEVELS)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def guard_policy(phase: str) -> dict[str, Any]:
    phase = phase.lower()
    if phase not in {"solid", "liquid"}:
        raise ValueError(f"Unsupported phase {phase!r}; expected solid or liquid.")
    expectation = {
        "solid": {
            "coordination": "Narrow, persistent chemically relevant CN modes.",
            "msd": "Bounded species-resolved MSD with no sustained diffusion.",
            "xrd": "Crystalline reference peaks should remain, allowing thermal broadening.",
        },
        "liquid": {
            "coordination": "Broad, stationary chemically relevant CN distributions.",
            "msd": "Sustained species-resolved diffusion over the accepted tail.",
            "xrd": "Long-range Bragg order should be strongly damped or lost.",
        },
    }[phase]
    return {
        "phase": phase,
        "prerequisites": [
            "numerical_completion",
            "trajectory_continuity",
            "temperature_stability",
            "energy_stability",
            "density_or_volume_anchor",
            "tail_stationarity",
        ],
        "primary_phase_guards": [
            "coordination_distribution",
            "msd_or_diffusion",
        ],
        "supporting_phase_guards": [
            "xrd_or_bragg",
            "rdf_or_pdf",
        ],
        "priority_note": (
            "Coordination distributions and MSD/diffusion are the primary phase-identity "
            "decision. XRD/Bragg and RDF/PDF are supporting long-range-order evidence; "
            "finite-cell high-temperature XRD alone must not overturn consistent CN+MSD."
        ),
        "expectation": expectation,
    }


def preparation_stages(phase: str) -> list[dict[str, Any]]:
    phase = phase.lower()
    common_tail = [
        {
            "stage": "npt_density_anchor",
            "ensemble": "NPT",
            "purpose": "Establish target-T/P density and cell before entropy production.",
            "required": "unless a documented accepted density/cell restart is reused",
        },
        {
            "stage": "fixed_cell_nvt_production",
            "ensemble": "NVT",
            "purpose": "Generate the stationary fixed-cell production trajectory.",
            "required": True,
        },
        {
            "stage": "tail_selection",
            "ensemble": "NVT tail",
            "purpose": "Select dense, uniformly spaced, stationary frames using the actual production cell.",
            "required": True,
        },
        {
            "stage": "physical_guard",
            "ensemble": "postprocess",
            "purpose": "Apply numerical prerequisites, primary CN+MSD guards, and supporting XRD/PDF evidence.",
            "required": True,
        },
        {
            "stage": "route_c_entropy",
            "ensemble": "postprocess",
            "purpose": "Produce separately reported S_vib and S_conf, then S_total.",
            "required": True,
        },
    ]
    if phase == "liquid":
        return [
            {
                "stage": "seed",
                "ensemble": "structure/restart",
                "purpose": "Use an accepted liquid restart or a structure suitable for melting.",
                "required": True,
            },
            {
                "stage": "nvt_melt_and_hold",
                "ensemble": "NVT",
                "purpose": "Erase crystalline memory and decorrelate at a chemistry-appropriate high temperature.",
                "required": "unless an accepted liquid seed is reused",
            },
            {
                "stage": "controlled_cooling",
                "ensemble": "NVT or NPT",
                "purpose": "Cool or anneal to the target temperature without quenching into an unverified state.",
                "required": "unless an accepted target-temperature liquid restart is reused",
            },
            *common_tail,
        ]
    if phase == "solid":
        return [
            {
                "stage": "seed",
                "ensemble": "structure/restart",
                "purpose": "Use the correct relaxed crystal or an accepted solid restart.",
                "required": True,
            },
            {
                "stage": "nvt_solid_heating",
                "ensemble": "NVT",
                "purpose": "Heat/equilibrate the crystal at target temperature without melting.",
                "required": "unless an accepted target-temperature solid restart is reused",
            },
            *common_tail,
        ]
    raise ValueError(f"Unsupported phase {phase!r}; expected solid or liquid.")


def engine_bridge(engine: str, phase: str, temperature_k: float) -> dict[str, Any]:
    label = f"{phase}_{int(round(temperature_k))}K"
    if engine == "cp2k":
        return {
            "status": "atomi cp2k-aimd-status RUN_DIR --json-out cp2k_aimd_status.json",
            "phase_order_guard": (
                "atomi md-phase-order-guard cp2k-xyz --xyz PRODUCTION-pos-1.xyz "
                "--inp PRODUCTION.restart --outdir phase_order_guard"
            ),
            "tail_to_sluschi": (
                f"atomi sluschi-cp2k-prep --xyz PRODUCTION-pos-1.xyz --inp PRODUCTION.restart "
                f"--phase {label} --start-frame TAIL_START --outdir sluschi_prepared"
            ),
            "cell_rule": (
                "Use the actual restart/extxyz cell. Do not silently substitute the nominal input cell "
                "after an NPT or RESTART_CELL trajectory."
            ),
            "trajectory": "CP2K multi-frame position XYZ plus restart/input cell",
        }
    if engine == "lammps":
        return {
            "status": "atomi lammps-summary log.lammps --last-fraction 0.5",
            "phase_order_guard": (
                "atomi md-phase-order-guard lammps-dump --dump production.dump "
                "--type-element 1=ELEMENT1 --type-element 2=ELEMENT2 --outdir phase_order_guard"
            ),
            "tail_to_sluschi": (
                f"atomi sluschi-bridge lammps-prep --trajectory production.dump "
                f"--type-elements 1=ELEMENT1,2=ELEMENT2 --phase {phase} "
                f"--temperature-k {temperature_k:g} --start-frame TAIL_START --outdir sluschi_prepared"
            ),
            "cell_rule": "Use the per-frame LAMMPS box and prefer unwrapped coordinates for S_vib.",
            "trajectory": "LAMMPS custom dump with atom id/type and wrapped plus preferably unwrapped coordinates",
        }
    raise ValueError(f"Unsupported engine {engine!r}; expected cp2k or lammps.")


def prepare_main(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.resolve()
    stages = preparation_stages(args.phase)
    policy = guard_policy(args.phase)
    bridge = engine_bridge(args.engine, args.phase, args.temperature_k)
    plan = {
        "schema": SCHEMA_MD_ENTROPY_PLAN,
        "system": args.system,
        "formula": args.formula,
        "engine": args.engine,
        "phase": args.phase,
        "temperature_K": args.temperature_k,
        "source_seed": str(args.source_seed.resolve()) if args.source_seed else "",
        "requested_quality": args.quality,
        "stages": stages,
        "guard_policy": policy,
        "engine_bridge": bridge,
        "tail_policy": {
            "fixed_cell": True,
            "stationary": True,
            "dense_uniform_spacing": True,
            "tail_fraction": args.tail_fraction,
            "minimum_independent_tails_for_production": 2,
            "actual_restart_cell_required": True,
        },
        "entropy_contract": {
            "route": "SLUSCHI-zentropy Route C",
            "required_components": ["S_vib_J_mol_atom_K", "S_conf_J_mol_atom_K"],
            "solid_S_conf_default": "numeric zero only after an accepted ordered-solid guard",
            "liquid_S_conf_default": "selected coordination/pair-channel distribution",
            "S_total": "S_vib + S_conf + optional S_elec",
        },
    }
    write_json(outdir / "md_entropy_plan.json", plan)
    stage_rows = [
        {
            "order": idx,
            "stage": stage["stage"],
            "ensemble": stage["ensemble"],
            "purpose": stage["purpose"],
            "required": stage["required"],
            "status": "planned",
        }
        for idx, stage in enumerate(stages, 1)
    ]
    write_csv(
        outdir / "md_entropy_stage_runlist.csv",
        stage_rows,
        ["order", "stage", "ensemble", "purpose", "required", "status"],
    )
    (outdir / "README_MD_ENTROPY.md").write_text(
        textwrap.dedent(
            f"""\
            # Atomi MD entropy workflow

            `{args.system or args.formula}`: `{args.engine}` `{args.phase}` at
            `{args.temperature_k:g} K`.

            Prepare or reuse a documented seed, establish the target density/cell,
            and generate a stationary fixed-cell NVT tail. Numerical completion and
            trajectory continuity are prerequisites. Coordination distributions and
            MSD/diffusion are the primary phase guards; XRD/PDF are supporting
            evidence. Only a passing gate may enter Route C, which reports `S_vib`
            and `S_conf` separately before assembling `S_total`.
            """
        ),
        encoding="utf-8",
    )
    print(f"Wrote MD-entropy plan: {outdir / 'md_entropy_plan.json'}")
    return plan


def _quality_cap(requested: str, cap: str) -> str:
    return QUALITY_LEVELS[min(QUALITY_RANK[requested], QUALITY_RANK[cap])]


def evaluate_gate(
    *,
    phase: str,
    statuses: dict[str, str],
    requested_quality: str,
    independent_tails: int,
) -> dict[str, Any]:
    policy = guard_policy(phase)
    prerequisites = policy["prerequisites"]
    primary = policy["primary_phase_guards"]
    supporting = policy["supporting_phase_guards"]
    reasons: list[str] = []
    warnings: list[str] = []

    failed_essential = [name for name in [*prerequisites, *primary] if statuses[name] == "fail"]
    missing_essential = [name for name in [*prerequisites, *primary] if statuses[name] == "missing"]
    warning_essential = [name for name in [*prerequisites, *primary] if statuses[name] == "warning"]
    failed_supporting = [name for name in supporting if statuses[name] == "fail"]
    warning_supporting = [name for name in supporting if statuses[name] == "warning"]
    missing_supporting = [name for name in supporting if statuses[name] == "missing"]

    if failed_essential or missing_essential:
        decision = "rejected"
        reasons.extend(f"essential guard failed: {name}" for name in failed_essential)
        reasons.extend(f"essential guard missing: {name}" for name in missing_essential)
    elif warning_essential:
        decision = "accepted-with-warning"
        warnings.extend(f"essential guard is provisional: {name}" for name in warning_essential)
    else:
        decision = "accepted"

    if failed_supporting:
        warnings.extend(f"supporting guard disagrees: {name}" for name in failed_supporting)
        if decision == "accepted":
            decision = "accepted-with-warning"
    if warning_supporting:
        warnings.extend(f"supporting guard is provisional: {name}" for name in warning_supporting)
        if decision == "accepted":
            decision = "accepted-with-warning"
    if missing_supporting:
        warnings.extend(f"supporting guard missing: {name}" for name in missing_supporting)
        if decision == "accepted":
            decision = "accepted-with-warning"

    effective_quality = requested_quality
    if decision == "rejected":
        effective_quality = "descriptor"
    elif decision == "accepted-with-warning":
        effective_quality = _quality_cap(effective_quality, "screening-prior")
    if independent_tails < 2:
        effective_quality = _quality_cap(effective_quality, "screening-prior")
        warnings.append("Production quality requires at least two independent accepted tails or replicas.")

    return {
        "decision": decision,
        "promotable_to_entropy": decision != "rejected",
        "requested_quality": requested_quality,
        "effective_quality": effective_quality,
        "independent_tails": independent_tails,
        "reasons": reasons,
        "warnings": warnings,
    }


def gate_main(args: argparse.Namespace) -> dict[str, Any]:
    statuses = {
        "numerical_completion": args.numerical_completion,
        "trajectory_continuity": args.trajectory_continuity,
        "temperature_stability": args.temperature_stability,
        "energy_stability": args.energy_stability,
        "density_or_volume_anchor": args.density_or_volume_anchor,
        "tail_stationarity": args.tail_stationarity,
        "coordination_distribution": args.coordination_distribution,
        "msd_or_diffusion": args.msd_or_diffusion,
        "xrd_or_bragg": args.xrd_or_bragg,
        "rdf_or_pdf": args.rdf_or_pdf,
    }
    evaluation = evaluate_gate(
        phase=args.phase,
        statuses=statuses,
        requested_quality=args.quality,
        independent_tails=args.independent_tails,
    )
    payload = {
        "schema": SCHEMA_MD_ENTROPY_GATE,
        "system": args.system,
        "formula": args.formula,
        "engine": args.engine,
        "phase": args.phase,
        "temperature_K": args.temperature_k,
        "tail_manifest": str(args.tail_manifest.resolve()) if args.tail_manifest else "",
        "guard_policy": guard_policy(args.phase),
        "guards": statuses,
        **evaluation,
        "notes": args.note,
    }
    outdir = args.outdir.resolve()
    write_json(outdir / "md_entropy_gate.json", payload)
    print(f"MD-entropy gate: {payload['decision']} ({payload['effective_quality']})")
    print(f"Wrote gate: {outdir / 'md_entropy_gate.json'}")
    return payload


def entropy_main(args: argparse.Namespace) -> dict[str, Any]:
    gate = json.loads(args.gate_json.read_text(encoding="utf-8"))
    if gate.get("schema") != SCHEMA_MD_ENTROPY_GATE:
        raise ValueError(f"{args.gate_json} is not an Atomi MD-entropy gate.")
    if not gate.get("promotable_to_entropy"):
        raise ValueError("MD-entropy gate rejected this tail; Route-C entropy production is blocked.")
    phase = str(gate["phase"]).lower()
    if phase != args.phase:
        raise ValueError(f"Gate phase {phase!r} does not match requested phase {args.phase!r}.")

    sconf = args.sconf_j_mol_atom_k
    if phase == "solid" and sconf is None and args.solid_sconf_policy == "zero":
        sconf = 0.0
    if args.svib_j_mol_atom_k is None and args.mds_summary is None:
        raise ValueError("Entropy handoff requires --svib-j-mol-atom-k or --mds-summary.")
    if phase == "liquid" and sconf is None and args.coordination_csv is None and args.mds_summary is None:
        raise ValueError("Liquid entropy requires S_conf from coordination data, MDS output, or an explicit value.")

    route_args = argparse.Namespace(
        outdir=args.outdir,
        phase=phase,
        temperature_k=float(gate["temperature_K"]),
        formula=args.formula or gate.get("formula", ""),
        coordination_csv=args.coordination_csv,
        mds_summary=args.mds_summary,
        thermo_csv=args.thermo_csv,
        h_kj_mol_atom=args.h_kj_mol_atom,
        svib_j_mol_atom_k=args.svib_j_mol_atom_k,
        sconf_j_mol_atom_k=sconf,
        selec_j_mol_atom_k=args.selec_j_mol_atom_k,
        pair_policy=args.pair_policy,
        quality=gate["effective_quality"],
    )
    route_result = route_c.analyze_main(route_args)
    row = route_result["summary"]
    if row.get("Svib_J_mol_atom_K") is None:
        raise ValueError("Route C did not produce S_vib; entropy decomposition is incomplete.")
    if row.get("Sconf_J_mol_atom_K") is None:
        raise ValueError("Route C did not produce S_conf; entropy decomposition is incomplete.")

    result = {
        "schema": SCHEMA_MD_ENTROPY_RESULT,
        "system": gate.get("system", ""),
        "formula": row.get("formula", ""),
        "engine": gate.get("engine", ""),
        "phase": phase,
        "temperature_K": gate["temperature_K"],
        "gate_json": str(args.gate_json.resolve()),
        "gate_decision": gate["decision"],
        "quality": gate["effective_quality"],
        "decomposition": {
            "S_vib_J_mol_atom_K": row["Svib_J_mol_atom_K"],
            "S_conf_J_mol_atom_K": row["Sconf_J_mol_atom_K"],
            "S_elec_J_mol_atom_K": row["Selec_J_mol_atom_K"],
            "S_total_J_mol_atom_K": row["Stotal_J_mol_atom_K"],
        },
        "thermodynamics": {
            "H_kJ_mol_atom": row["H_kJ_mol_atom"],
            "G_kJ_mol_atom": row["G_kJ_mol_atom"],
        },
        "route_c_outputs": {
            "summary_csv": str((args.outdir.resolve() / "route_c_summary.csv")),
            "summary_json": str((args.outdir.resolve() / "route_c_summary.json")),
            "phase_health_json": str((args.outdir.resolve() / "phase_health_route_c.json")),
        },
        "solid_sconf_policy": args.solid_sconf_policy if phase == "solid" else "",
    }
    write_json(args.outdir.resolve() / "md_entropy_result.json", result)
    print(f"Wrote guarded MD entropy: {args.outdir.resolve() / 'md_entropy_result.json'}")
    return result


def status_main(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    files = {
        "plan": root / "md_entropy_plan.json",
        "gate": root / "md_entropy_gate.json",
        "result": root / "md_entropy_result.json",
    }
    payload: dict[str, Any] = {"root": str(root), "files": {}, "stage": "not-started"}
    for name, path in files.items():
        payload["files"][name] = str(path) if path.is_file() else ""
    if files["result"].is_file():
        payload["stage"] = "entropy-complete"
        payload["result"] = json.loads(files["result"].read_text(encoding="utf-8"))
    elif files["gate"].is_file():
        payload["stage"] = "guard-complete"
        payload["gate"] = json.loads(files["gate"].read_text(encoding="utf-8"))
    elif files["plan"].is_file():
        payload["stage"] = "planned"
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def add_guard_argument(parser: argparse.ArgumentParser, flag: str, help_text: str) -> None:
    parser.add_argument(flag, choices=GUARD_STATES, default="missing", help=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="md-entropy",
        description="Guarded CP2K/LAMMPS MD-to-SLUSCHI Route-C entropy workflow.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Write an engine-specific MD entropy plan and stage runlist.")
    prepare.add_argument("--outdir", type=Path, default=Path("md_entropy_plan"))
    prepare.add_argument("--system", default="")
    prepare.add_argument("--formula", default="")
    prepare.add_argument("--engine", choices=("cp2k", "lammps"), required=True)
    prepare.add_argument("--phase", choices=("solid", "liquid"), required=True)
    prepare.add_argument("--temperature-k", type=float, required=True)
    prepare.add_argument("--source-seed", type=Path)
    prepare.add_argument("--tail-fraction", type=float, default=0.5)
    prepare.add_argument("--quality", choices=QUALITY_LEVELS, default="screening-prior")

    gate = sub.add_parser("gate", help="Decide whether a fixed-cell NVT tail may enter Route C.")
    gate.add_argument("--outdir", type=Path, default=Path("md_entropy_guard"))
    gate.add_argument("--system", default="")
    gate.add_argument("--formula", default="")
    gate.add_argument("--engine", choices=("cp2k", "lammps"), required=True)
    gate.add_argument("--phase", choices=("solid", "liquid"), required=True)
    gate.add_argument("--temperature-k", type=float, required=True)
    gate.add_argument("--tail-manifest", type=Path)
    gate.add_argument("--quality", choices=QUALITY_LEVELS, default="screening-prior")
    gate.add_argument("--independent-tails", type=int, default=1)
    gate.add_argument("--note", action="append", default=[])
    add_guard_argument(gate, "--numerical-completion", "Engine completed the intended production steps.")
    add_guard_argument(gate, "--trajectory-continuity", "No missing, duplicated, or reordered tail steps.")
    add_guard_argument(gate, "--temperature-stability", "Tail temperature is stable around target.")
    add_guard_argument(gate, "--energy-stability", "Tail energy has no runaway or material secular drift.")
    add_guard_argument(gate, "--density-or-volume-anchor", "Density/cell is accepted and provenance is recorded.")
    add_guard_argument(gate, "--tail-stationarity", "Selected fixed-cell NVT tail is stationary.")
    add_guard_argument(gate, "--coordination-distribution", "CN distributions support the declared phase.")
    add_guard_argument(gate, "--msd-or-diffusion", "MSD/diffusion supports the declared phase.")
    add_guard_argument(gate, "--xrd-or-bragg", "Supporting XRD/Bragg evidence supports the declared phase.")
    add_guard_argument(gate, "--rdf-or-pdf", "Supporting RDF/PDF evidence supports the declared phase.")

    entropy = sub.add_parser("entropy", help="Run Route-C analysis only after an accepted MD-entropy gate.")
    entropy.add_argument("--gate-json", type=Path, required=True)
    entropy.add_argument("--outdir", type=Path, default=Path("md_entropy_result"))
    entropy.add_argument("--phase", choices=("solid", "liquid"), required=True)
    entropy.add_argument("--formula", default="")
    entropy.add_argument("--coordination-csv", type=Path)
    entropy.add_argument("--mds-summary", type=Path)
    entropy.add_argument("--thermo-csv", type=Path)
    entropy.add_argument("--h-kj-mol-atom", type=float)
    entropy.add_argument("--svib-j-mol-atom-k", type=float)
    entropy.add_argument("--sconf-j-mol-atom-k", type=float)
    entropy.add_argument("--selec-j-mol-atom-k", type=float, default=0.0)
    entropy.add_argument(
        "--solid-sconf-policy",
        choices=("zero", "coordination"),
        default="zero",
        help="Use numerical zero only for a physically accepted ordered solid, or derive S_conf from CN data.",
    )
    entropy.add_argument(
        "--pair-policy",
        choices=("auto", "all", "same-species", "heavy-heavy", "light-heavy"),
        default="auto",
    )

    status = sub.add_parser("status", help="Summarize plan, guard, and entropy artifacts.")
    status.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return prepare_main(args)
    if args.command == "gate":
        return gate_main(args)
    if args.command == "entropy":
        return entropy_main(args)
    if args.command == "status":
        return status_main(args)
    return None


if __name__ == "__main__":
    main()
