"""Molten-salt liquid AIMD route planning for SLUSCHI/MIVM workflows.

This module keeps liquid-initialization and entropy-handoff recommendations in
Atomi instead of project-local scripts.  It does not run VASP/CP2K/LAMMPS by
itself; it emits structured guidance that project generators and reports can
reuse before launching MD.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LiquidSeedRoute:
    route: str
    best_use: str
    advantages: str
    limits: str
    acceptance_tests: str
    recommendation: str


@dataclass(frozen=True)
class WorkflowStep:
    stage: str
    action: str
    guard: str
    output: str


@dataclass(frozen=True)
class SluschiEntropyRule:
    rule: str
    method: str
    guard: str
    failure_mode: str


@dataclass(frozen=True)
class SaltMonitoringTrack:
    project: str
    route: str
    monitor_checks: str
    advance_when: str
    hold_when: str
    report_destination: str


LITERATURE_ANCHORS: list[dict[str, str]] = [
    {
        "key": "packmol-2009",
        "label": "Martinez et al., PACKMOL, J. Comput. Chem. 2009",
        "url": "https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.21224",
        "use": "PACKMOL builds clash-controlled random initial configurations; it is not an equilibration substitute.",
    },
    {
        "key": "beeler-nacl-ucl3-2022",
        "label": "Andersson and Beeler, NaCl-UCl3 molten-salt AIMD, J. Nucl. Mater. 2022",
        "url": "https://laro.lanl.gov/esploro/outputs/journalArticle/Ab-initio-molecular-dynamics-AIMD-simulations/9916362369503761",
        "use": "NaCl-UCl3 VASP/DFT+U AIMD reference for density, RDF/CN, network, and liquid thermodynamics checks.",
    },
    {
        "key": "li-ucln-nacl-2019",
        "label": "Li, Dai, and Jiang, UCln-NaCl first-principles MD, ACS Appl. Energy Mater. 2019",
        "url": "https://doi.org/10.1021/acsaem.8b02157",
        "use": "Structural warning for UCl3-rich liquids: persistent U-Cl coordination and shared-Cl networks can be molten-liquid descriptors, not solidness vetoes.",
    },
    {
        "key": "li-pim-nacl-ucl3-2020",
        "label": "Li, Dai, and Jiang, NaCl-UCl3 polarizable-ion MD, J. Mol. Liq. 2020",
        "url": "https://impact.ornl.gov/en/publications/molecular-dynamics-simulations-of-structural-and-transport-proper/",
        "use": "Transport warning for x(UCl3) > ~0.25: Cl motion is coupled to the U-Cl network, so liquid acceptance should use MSD/diffusion plus thermodynamic stability.",
    },
]


def liquid_seed_routes(system: str = "molten chloride") -> list[LiquidSeedRoute]:
    return [
        LiquidSeedRoute(
            route="PACKMOL/random atom packing",
            best_use="Independent liquid seed, composition scan, and hysteresis/path-bias check.",
            advantages="Composition-general, low crystalline memory, and fast to generate for multicomponent salts.",
            limits="May create high-energy local contacts if tolerance/volume are poor; not equilibrated by construction.",
            acceptance_tests="Minimum pair distance, short relax/melt stability, target NPT density, RDF/PDF, MSD, cation-anion CN, and actinide-cation network metrics when relevant.",
            recommendation="Use as one required independent seed near key liquid compositions; follow with high-T melt/pre-equilibration plus target NPT.",
        ),
        LiquidSeedRoute(
            route="Crystal/order-derived high-temperature premelt",
            best_use="Melting-path studies, metastable branch tests, and systems with reliable crystalline precursors.",
            advantages="Avoids random close contacts and gives a physically interpretable melt history.",
            limits="Can retain crystalline/network memory if the melt is too short; path dependent.",
            acceptance_tests="Loss of framework order, liquid MSD, stable target density, and RDF/PDF agreement with a random-seed route.",
            recommendation="Use as the companion route to PACKMOL; do not accept as sole proof of liquid entropy.",
        ),
        LiquidSeedRoute(
            route="PACKMOL -> high-T melt -> target NPT -> fixed-volume NVT",
            best_use=f"Production SLUSCHI entropy and MIVM-prior trajectories for {system}.",
            advantages="Combines low initial memory, density equilibration, and clean fixed-volume entropy sampling.",
            limits="Most expensive and requires explicit stage bookkeeping.",
            acceptance_tests="NPT density plateau, NVT temperature/energy stability, network-aware phase-window liquid label when actinide/lanthanide chlorides are present, block uncertainty, and unit/basis guard.",
            recommendation="Preferred production route for NaCl-UCl3 and future LiCl-KCl-UCl3 entropy/MIVM workflows.",
        ),
        LiquidSeedRoute(
            route="Direct target-temperature NVT from random/grid seed",
            best_use="Smoke tests for inputs, queue scripts, parser wiring, and SLUSCHI handoff.",
            advantages="Cheap and stable for infrastructure checks.",
            limits="Not production liquid thermodynamics; artificial density/order can dominate short runs.",
            acceptance_tests="Use only for parser validation unless independent density and phase-window checks pass.",
            recommendation="Keep as a diagnostic lane; do not feed MIVM entropy fits from this route.",
        ),
    ]


def sluschi_entropy_steps() -> list[WorkflowStep]:
    return [
        WorkflowStep(
            "MD trajectory",
            "Generate equilibrated tail windows from VASP XDATCAR, CP2K XYZ, or LAMMPS dump.",
            "Record timestep, coordinate basis, lattice convention, composition, temperature, phase, and frame stride.",
            "Trajectory-window manifest and raw selected frames.",
        ),
        WorkflowStep(
            "Phase-window gate",
            "Run RDF/PDF plus MSD/local-order checks before entropy.",
            "RDF alone is insufficient; for UCl3/CeCl3-rich liquids use network-aware checks so high cation-Cl coordination is not treated as a solid veto.",
            "Accepted frame ranges with phase labels.",
        ),
        WorkflowStep(
            "SLUSCHI/MDS handoff",
            "Use atomi sluschi-bridge vasp-prep/cp2k-prep/lammps-prep and mds-entropy-run.",
            "Confirm Cartesian Angstrom positions, Angstrom lattice, prepared ps units, and legacy MDS fs conversion.",
            "SLUSCHI workdir with unit manifest.",
        ),
        WorkflowStep(
            "Entropy parse",
            "Parse constrained Svib from collect.stdout and pair/channel Sconf with explicit stoichiometry and a documented reduction rule.",
            "Reject rows without valid vib.out/entropy.out/collect.stdout, phase-health acceptance, dense NVT windows, and explicit unit/basis conversion.",
            "Svib/Sconf/Stotal CSV+JSON with quality tier.",
        ),
        WorkflowStep(
            "Thermo handoff",
            "Convert to J mol-formula^-1 K^-1 or chosen CALPHAD/MIVM basis.",
            "Guard atom/formula/mixture/pseudo-component basis and ideal/configurational double counting.",
            "QHA/literature overlay and MIVM-ready Sconf table.",
        ),
    ]


def sluschi_entropy_method_rules() -> list[SluschiEntropyRule]:
    return [
        SluschiEntropyRule(
            "Phase first",
            "Run XRD/PDF/RDF/MSD/order guards before entropy and only parse stable single-phase tail windows.",
            "For network liquids, persistent first-shell cation-Cl peaks are descriptors; Bragg-like long-range order is the solid warning.",
            "Running SLUSCHI on mixed or crystallizing windows collapses liquid/solid entropy contrast.",
        ),
        SluschiEntropyRule(
            "Svib source",
            "Use the constrained/use-this-value Svib line from collect.stdout; it is a vibrational entropy from SLUSCHI MDS, not RDF/PDF integration.",
            "Reject zero fallback, NaN vib.out, empty entropy/vib support files, or frame counts that fail onephase_v6 preflight.",
            "Parsing entropy.out alone or accepting zero fallback makes Svib appear missing or artificially low.",
        ),
        SluschiEntropyRule(
            "Sconf source",
            "Use SLUSCHI pair-channel configurational entropy recommendations from collect.stdout/Sconf files with an explicit chemical reduction.",
            "For pure binary KCl liquid vs Hong/Shang Fig. 3, use same-species-liquid: liquid 1-1 and 2-2 channels. For mixed salts, document the species-pair selector; keep all-pair mean as diagnostic only unless justified.",
            "Blind all-pair averaging can depress liquid Sconf by mixing same-species and cross-species channels.",
        ),
        SluschiEntropyRule(
            "Units and basis",
            "Track coordinate units, lattice units, prepared timestep units, legacy SLUSCHI fs conversion, and entropy basis.",
            "Always state J mol-atom^-1 K^-1, J mol-formula^-1 K^-1, or MIVM pseudo-component basis before plotting or pycalphad handoff.",
            "Atom/formula/pseudo-binary confusion can create factor-of-two or composition-dependent entropy errors.",
        ),
        SluschiEntropyRule(
            "Production quality",
            "Use longer equilibrated NVT tails and multiple block windows; report mean and uncertainty with quality tier.",
            "Screening-prior rows can guide workflow choices, but production rows need phase guards, stable T/E/density, and block consistency.",
            "Short target trajectories can look liquid-like but still under-sample Sconf and bias Svib.",
        ),
    ]


def mivm_handoff_steps() -> list[WorkflowStep]:
    return [
        WorkflowStep(
            "Structural validation",
            "Compare RDF/CN/PMF/network descriptors with AIMD/literature references.",
            "Do not fit MIVM from phase-mixed or non-equilibrated windows.",
            "Validated rows by composition and temperature.",
        ),
        WorkflowStep(
            "Entropy prior",
            "Use accepted SLUSCHI liquid Sconf as concentration-dependent prior or excess correction.",
            "Keep reference state, basis, and uncertainty explicit.",
            "Sconf(x,T) table with quality flags.",
        ),
        WorkflowStep(
            "MIVM sampling",
            "Fit/sample B_ij priors with PMF/RDF constraints plus Hmix/Gex/phase-boundary data.",
            "Treat PMF-derived B values as priors, not final thermodynamic truth.",
            "MIVM parameter ensemble and diagnostics.",
        ),
        WorkflowStep(
            "pycalphad bridge",
            "Use calphad-mivm/pycalphad bridge or config-driven scans for phase-equilibrium checks.",
            "Reject parameter sets that improve entropy but break enthalpy or phase-boundary constraints.",
            "Publication plots and candidate parameter JSON/TDB-side bridge.",
        ),
    ]


def salt_monitoring_tracks() -> list[SaltMonitoringTrack]:
    """Reusable monitor checklist for current molten-salt projects.

    Keep this data compact and project-facing.  Heartbeat automation should use
    it as a durable checklist, while concrete job ids remain in the live monitor
    instructions or project state files.
    """

    return [
        SaltMonitoringTrack(
            project="KCl Hong/Shang-style benchmark",
            route="LAMMPS demonstration, CP2K AIMD demonstration, and metastable solid/liquid branch probes.",
            monitor_checks="squeue/sacct, log tails, completed dump/final-data pairs, CP2K XYZ trajectories, phase-window health, SLUSCHI S_vib/S_conf outputs, and Hong/Shang overlay tables.",
            advance_when="A branch has complete trajectories with clean liquid-like or solid-like phase windows and valid entropy outputs on an explicit formula/unit basis.",
            hold_when="Timeout without complete tail windows, phase-mixed windows, missing trajectory/data pairs, or SLUSCHI outputs without valid collect/vib/entropy files.",
            report_destination="Google Drive KCl project bridge only; no reports on HPC.",
        ),
        SaltMonitoringTrack(
            project="NaCl-UCl3 liquid entropy / MIVM prior",
            route="VASP melt-target route from high-T melt to 1250 K target NPT/NVT tail windows; future PACKMOL and premelt companion routes.",
            monitor_checks="melt CONTCAR handoff, target-stage XDATCAR/OSZICAR/OUTCAR/vasprun health, density/temperature stability, network-aware RDF/PDF/MSD, U-Cl coordination/network descriptors, phase-window liquid label, and SLUSCHI entropy basis.",
            advance_when="Target liquid windows are stable and diffusive, with U-Cl network coordination reported as a descriptor rather than a solid veto, phase-health acceptance, and S_conf suitable for MIVM/pycalphad with uncertainty and basis metadata.",
            hold_when="Melt did not complete, target POSCAR was not seeded from melt CONTCAR, windows are phase mixed, or entropy lacks unit/reference-state metadata.",
            report_destination="Google Drive molten_salt_mivm project bridge; MIVM handoff tables must keep S_conf basis explicit.",
        ),
        SaltMonitoringTrack(
            project="CeCl3 superionic / partial-sublattice melting probe",
            route="CP2K AIMD falsification study across sub-melting, near-melting, and molten benchmark temperatures.",
            monitor_checks="CP2K SCF stability, pos/vel XYZ trajectories, restart health, Ce and Cl MSD, RDF, Ce-Ce framework order, and phase-window labels.",
            advance_when="Trajectory completes cleanly and diagnostics distinguish normal solid, liquid, or possible Cl-sublattice diffusion with Ce framework retained.",
            hold_when="SCF instability, missing trajectories, RDF-only evidence, or no MSD/Ce-framework support for the assigned phase.",
            report_destination="Google Drive CeCl3 subproject under molten_salt_mivm; no HPC Markdown.",
        ),
        SaltMonitoringTrack(
            project="LiCl-KCl-UCl3 pseudo-binary MIVM",
            route="Literature/AIMD-assisted parameter extraction plus future production liquid entropy windows.",
            monitor_checks="source inventory, density/molar-volume basis, PMF/RDF/CN anchors, pseudo-component definition, entropy prior provenance, and CALPHAD/pycalphad consistency.",
            advance_when="Parameters or entropy priors have clear reference states, uncertainty, and do not break phase-boundary or enthalpy constraints.",
            hold_when="PMF-derived B values are treated as final truth, density signs/units are ambiguous, or LiCl-KCl eutectic pseudo-component basis is missing.",
            report_destination="Google Drive licl_kcl_ucl3_pseudobinary_mivm subproject.",
        ),
    ]


def guidance_payload(system: str = "molten chloride") -> dict[str, Any]:
    return {
        "schema": "atomi.md.molten_salt_liquid_guidance.v1",
        "system": system,
        "recommended_production_route": "PACKMOL/random seed -> high-T melt/pre-equilibration -> target NPT density -> fixed-volume NVT tail -> phase-window gate -> SLUSCHI entropy -> MIVM/pycalphad handoff",
        "liquid_seed_routes": [asdict(row) for row in liquid_seed_routes(system)],
        "sluschi_entropy_steps": [asdict(row) for row in sluschi_entropy_steps()],
        "sluschi_entropy_method_rules": [asdict(row) for row in sluschi_entropy_method_rules()],
        "mivm_handoff_steps": [asdict(row) for row in mivm_handoff_steps()],
        "salt_monitoring_tracks": [asdict(row) for row in salt_monitoring_tracks()],
        "literature_anchors": LITERATURE_ANCHORS,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_guidance(outdir: Path, system: str) -> dict[str, Any]:
    payload = guidance_payload(system)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "molten_salt_liquid_guidance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(outdir / "liquid_seed_routes.csv", payload["liquid_seed_routes"])
    _write_csv(outdir / "sluschi_entropy_steps.csv", payload["sluschi_entropy_steps"])
    _write_csv(outdir / "sluschi_entropy_method_rules.csv", payload["sluschi_entropy_method_rules"])
    _write_csv(outdir / "mivm_handoff_steps.csv", payload["mivm_handoff_steps"])
    _write_csv(outdir / "salt_monitoring_tracks.csv", payload["salt_monitoring_tracks"])
    md = [
        f"# {system} molten-salt liquid AIMD, SLUSCHI, and MIVM guidance",
        "",
        payload["recommended_production_route"],
        "",
        "## Liquid Seed Routes",
        "",
        "| Route | Best use | Advantages | Limits | Acceptance tests | Recommendation |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["liquid_seed_routes"]:
        md.append(
            f"| {row['route']} | {row['best_use']} | {row['advantages']} | {row['limits']} | {row['acceptance_tests']} | {row['recommendation']} |"
        )
    for title, key in (("SLUSCHI Entropy Steps", "sluschi_entropy_steps"), ("MIVM Handoff Steps", "mivm_handoff_steps")):
        md += ["", f"## {title}", "", "| Stage | Action | Guard | Output |", "|---|---|---|---|"]
        for row in payload[key]:
            md.append(f"| {row['stage']} | {row['action']} | {row['guard']} | {row['output']} |")
    md += [
        "",
        "## SLUSCHI Entropy Method Rules",
        "",
        "| Rule | Method | Guard | Failure mode |",
        "|---|---|---|---|",
    ]
    for row in payload["sluschi_entropy_method_rules"]:
        md.append(f"| {row['rule']} | {row['method']} | {row['guard']} | {row['failure_mode']} |")
    md += [
        "",
        "## Salt Monitoring Tracks",
        "",
        "| Project | Route | Monitor checks | Advance when | Hold when | Report destination |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["salt_monitoring_tracks"]:
        md.append(
            f"| {row['project']} | {row['route']} | {row['monitor_checks']} | {row['advance_when']} | {row['hold_when']} | {row['report_destination']} |"
        )
    (outdir / "molten_salt_liquid_guidance.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Write molten-salt liquid AIMD -> SLUSCHI -> MIVM workflow guidance.")
    parser.add_argument("--system", default="molten chloride")
    parser.add_argument("--outdir", type=Path, default=Path("molten_salt_liquid_guidance"))
    args = parser.parse_args(argv)
    payload = write_guidance(args.outdir, args.system)
    print(f"Wrote molten-salt guidance: {args.outdir}")
    return payload


if __name__ == "__main__":
    main()
