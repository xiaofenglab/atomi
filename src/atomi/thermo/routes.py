from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "atomi.thermo.condensed_routes.v1"


@dataclass
class RouteInput:
    system: str
    formula: str
    structure: str | None
    qha_dir: str | None
    aimd_temperatures_K: list[float]
    route: str
    notes: str | None = None


@dataclass
class ThermoRoutePlan:
    schema: str
    input: RouteInput
    route_1_mlip_md: dict
    route_2_vasp_qha_aimd_sluschi: dict
    enthalpy_and_calphad_handoff: dict
    recommended_uc2_start: dict


def parse_temperatures(value: str) -> list[float]:
    temps = [float(item) for item in value.replace(";", ",").split(",") if item.strip()]
    if not temps:
        raise ValueError("At least one temperature is required.")
    return temps


def route_1() -> dict:
    return {
        "name": "MLIP-MD thermodynamics route",
        "purpose": "Fast production thermodynamics once an MLIP is trusted.",
        "low_temperature": [
            "Use QHA/phonopy when available for low-temperature Cp, entropy, free energy, and thermal expansion.",
            "Use thermo_qha_md to blend QHA with MD-derived high-temperature Cp/structure curves.",
        ],
        "high_temperature": [
            "Run MLIP-MD NPT/NVT series for Cp, enthalpy increments, density/lattice, and phase health.",
            "For entropy, either integrate Cp/T with an experimental or SLUSCHI anchor, or overlay SLUSCHI entropy rows from NVT trajectories.",
        ],
        "strengths": [
            "Large cells, many temperatures, repeat seeds, defect/disorder sampling.",
            "Cheap enough for uncertainty ranges and composition/defect scans.",
        ],
        "main_risks": [
            "MLIP bias outside the training domain.",
            "Absolute formation enthalpy still needs a database/DFT reference.",
            "Entropy anchor quality controls absolute S(T).",
        ],
    }


def route_2(temps: list[float]) -> dict:
    return {
        "name": "VASP-QHA + AIMD + SLUSCHI entropy route",
        "purpose": "Higher-fidelity condensed-matter thermodynamics route for benchmark compounds.",
        "low_temperature": [
            "Relax reference cell with VASP.",
            "Run volume grid phonons and phonopy-QHA for low-temperature Cp, S, G, V(T), and anisotropic expansion when supported.",
            "Use vasp-qha-summary / thermo_qha_md to normalize formula units and produce corrected thermodynamic curves.",
        ],
        "high_temperature": [
            "Run selected VASP AIMD NVT trajectories at representative high-T points.",
            "For each AIMD point, prepare separate phase/state trajectories if phase comparison is needed.",
            "Convert CP2K/VASP/XYZ-like trajectories or XDATCAR-derived frames to SLUSCHI-compatible inputs where possible.",
            "Run SLUSCHI entropy-prior analysis to obtain Svib, Sconf, and Stotal with uncertainty/error bars.",
            "Overlay SLUSCHI entropy markers on QHA/MD entropy curves and optionally use a 300 K SLUSCHI point as an entropy anchor when experiment is unavailable.",
        ],
        "initial_aimd_temperatures_K": temps,
        "minimum_outputs_per_temperature": [
            "AIMD trajectory frames with cell metadata.",
            "Mean AIMD internal energy/enthalpy proxy relative to the reference cell.",
            "SLUSCHI Svib/Sconf/Stotal CSV+JSON.",
            "Phase-health/structure-health diagnostics.",
        ],
        "strengths": [
            "Does not depend on MLIP transferability for benchmark points.",
            "Gives independent entropy terms comparable to the MLIP-MD route.",
            "Useful for validating MLIP route before large-scale scans.",
        ],
        "main_risks": [
            "AIMD is expensive, so temperature coverage is sparse.",
            "Finite-size and short-trajectory entropy estimates need UQ and phase-health checks.",
            "Absolute formation enthalpy still needs an external database/DFT reference.",
        ],
    }


def enthalpy_handoff() -> dict:
    return {
        "current_gap": "Neither route by itself supplies a standard formation enthalpy database.",
        "what_is_available_now": [
            "QHA gives temperature-dependent vibrational free energy/entropy relative to the chosen DFT reference cell.",
            "AIMD/MLIP-MD gives temperature-dependent internal energy/enthalpy increments and Cp-like slopes.",
            "SLUSCHI gives entropy decomposition terms from trajectory ensembles.",
        ],
        "needed_for_calphad": [
            "Standard enthalpy of formation for the reference compound/phase, or a consistent DFT-derived reference scheme.",
            "Heat capacity/entropy/free-energy increments on a declared basis: per formula, per atom, or per CALPHAD constituent.",
            "Uncertainty metadata for fitted Cp, entropy anchors, and formation enthalpy priors.",
        ],
        "pycalphad_handoff": [
            "Store route outputs as CSV/JSON with basis metadata.",
            "Use thermo-prior JSON entries for formation enthalpy, Cp functions, entropy anchors, and melting/transition points.",
            "Do not publish CALPHAD parameters until the enthalpy reference basis is explicitly declared.",
        ],
    }


def uc2_start(system: str, formula: str, structure: str | None, qha_dir: str | None, temps: list[float]) -> dict:
    return {
        "system": system,
        "formula": formula,
        "example_role": "first VASP-QHA + AIMD + SLUSCHI benchmark case",
        "structure_source": structure,
        "qha_source": qha_dir,
        "suggested_first_aimd_temperatures_K": temps,
        "suggested_steps": [
            "Confirm relaxed UC2 2x2x1 structure and formula-unit count.",
            "Use existing UC2 QHA output as low-temperature baseline.",
            "Prepare short VASP AIMD NVT smoke runs at the listed temperatures from QHA-corrected or relaxed finite-T cells.",
            "If smoke trajectories are healthy, extend to production AIMD and parse entropy through SLUSCHI.",
            "Overlay SLUSCHI entropy points with thermo_qha_md using --sluschi-entropy-csv.",
            "Defer pycalphad phase Gibbs functions until a formation enthalpy source is supplied.",
        ],
    }


def write_csv_runlist(path: Path, temps: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "T_K", "ensemble", "purpose", "status"])
        writer.writeheader()
        for temp in temps:
            writer.writerow(
                {
                    "case": f"aimd_T{int(round(temp))}",
                    "T_K": temp,
                    "ensemble": "NVT",
                    "purpose": "AIMD trajectory for SLUSCHI entropy-prior extraction",
                    "status": "planned",
                }
            )


def _path_string(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.expanduser())


def _bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_markdown(plan: ThermoRoutePlan) -> str:
    route2 = plan.route_2_vasp_qha_aimd_sluschi
    temps = ", ".join(f"{t:g}" for t in plan.input.aimd_temperatures_K)
    sections = [
        f"# Condensed-Matter Thermodynamics Route Plan: {plan.input.system}",
        "## Route Summary",
        "Atomi tracks two major condensed-matter thermodynamics routes.",
        "1. **MLIP-MD thermodynamics route**: fast production route for broad temperature, defect, and composition coverage after MLIP validation.\n"
        "2. **VASP-QHA + AIMD + SLUSCHI route**: higher-fidelity benchmark route using VASP QHA for low-temperature Cp and sparse AIMD plus SLUSCHI for high-temperature entropy terms.",
        "## Route 1: MLIP-MD",
        f"Purpose: {plan.route_1_mlip_md['purpose']}",
        "Strengths:\n" + _bullet(plan.route_1_mlip_md["strengths"]),
        "Main risks:\n" + _bullet(plan.route_1_mlip_md["main_risks"]),
        "## Route 2: VASP-QHA + AIMD + SLUSCHI",
        f"Purpose: {route2['purpose']}",
        "Low-temperature lane:\n" + _bullet(route2["low_temperature"]),
        "High-temperature lane:\n" + _bullet(route2["high_temperature"]),
        f"Planned AIMD temperatures: `{temps} K`",
        "Required outputs per AIMD temperature:\n" + _bullet(route2["minimum_outputs_per_temperature"]),
        "## Enthalpy And CALPHAD Handoff",
        f"Current gap: {plan.enthalpy_and_calphad_handoff['current_gap']}",
        "Needed before robust pycalphad handoff:\n"
        + _bullet(plan.enthalpy_and_calphad_handoff["needed_for_calphad"]),
        f"## {plan.input.formula} Start",
        f"Structure source: `{plan.recommended_uc2_start['structure_source']}`",
        f"QHA source: `{plan.recommended_uc2_start['qha_source']}`",
        "Suggested first steps:\n" + _bullet(plan.recommended_uc2_start["suggested_steps"]),
    ]
    return "\n\n".join(sections) + "\n"


def plan_main(args: argparse.Namespace) -> dict:
    temps = parse_temperatures(args.aimd_temperatures)
    route_input = RouteInput(
        system=args.system,
        formula=args.formula,
        structure=_path_string(args.structure),
        qha_dir=_path_string(args.qha_dir),
        aimd_temperatures_K=temps,
        route=args.route,
        notes=args.notes,
    )
    plan = ThermoRoutePlan(
        schema=SCHEMA,
        input=route_input,
        route_1_mlip_md=route_1(),
        route_2_vasp_qha_aimd_sluschi=route_2(temps),
        enthalpy_and_calphad_handoff=enthalpy_handoff(),
        recommended_uc2_start=uc2_start(args.system, args.formula, route_input.structure, route_input.qha_dir, temps),
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    payload = asdict(plan)
    json_path = args.outdir / "thermodynamics_route_plan.json"
    md_path = args.outdir / "THERMODYNAMICS_ROUTE_PLAN.md"
    runlist_path = args.outdir / "aimd_sluschi_runlist.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(plan), encoding="utf-8")
    write_csv_runlist(runlist_path, temps)
    print(md_path)
    return {"markdown": str(md_path), "json": str(json_path), "runlist": str(runlist_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermo-routes",
        description="Plan condensed-matter thermodynamics routes and CALPHAD handoff metadata.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Write a route plan Markdown/JSON/runlist scaffold.")
    plan.add_argument("--system", required=True, help="Human-readable system label, e.g. UC2.")
    plan.add_argument("--formula", required=True, help="Formula basis, e.g. UC2.")
    plan.add_argument("--structure", type=Path, help="Reference relaxed structure path.")
    plan.add_argument("--qha-dir", type=Path, help="Existing QHA output directory.")
    plan.add_argument(
        "--aimd-temperatures",
        default="900,1500,2100",
        help="Comma-separated AIMD temperatures in K for the initial SLUSCHI entropy route.",
    )
    plan.add_argument(
        "--route",
        choices=("both", "mlip-md", "vasp-qha-aimd-sluschi"),
        default="both",
        help="Route focus for this plan.",
    )
    plan.add_argument("--outdir", type=Path, required=True)
    plan.add_argument("--notes", help="Optional notes stored in the JSON manifest.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan_main(args)
        return
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
