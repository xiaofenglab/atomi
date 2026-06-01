"""MIVM parameter guidance and future pycalphad integration hooks."""

from __future__ import annotations

import argparse
import json
import textwrap
from typing import Any


MIVM_HELP_EPILOG = """\
Parameter guide:
  General MIVM/pycalphad needs:
    - phase/component basis and reference states
    - endmember or pseudo-endmember Gibbs energies
    - molar volumes Vm_i on the same component basis
    - coordination numbers Z_i or effective coordination numbers
    - directed pair parameters B_ij/B_ji, or epsilon/h parameters that define them
    - mixing enthalpy or activity data for fitting/validation

  Molten salts:
    - salt-component basis, e.g. LaCl3 and eutectic LiCl-KCl
    - liquid molar volumes from density/TMA/MD
    - cation-cation or physically chosen solvation coordination numbers from RDF/CN
    - pair potentials or PMF-derived B_ij, then calorimetry-refined B_ij if needed
    - charge/stoichiometry normalization for formula units and common-anion mixtures

  Solid/ceramic solutions, e.g. (Gd,U)O2:
    - substitutional/pseudo-component basis, e.g. U4+O2, Gd3+O1.5, U5+O2,
      and/or vacancy-compensated Gd-VO motifs
    - defect/charge-compensation model: U5+ compensation, oxygen vacancies, or mixed
    - molar volumes of endmembers and defect motifs from DFT, QHA, MD, or experiment
    - effective coordination numbers from fluorite/Ia-3 geometry, RDFs, or relaxed motifs
    - pair parameters for Gd-U4, Gd-U5, Gd-VO, U4-U5, and host-host interactions
    - DFT/experimental mixing enthalpies, defect-pair energies, activities, or solubility
      limits to fit/validate the MIVM excess Gibbs energy

Implementation rule:
  Use MIVM excess Gibbs energy as the pycalphad GM contribution. Use the corrected
  MIVM enthalpy expression only for calorimetry/DFT fitting and validation.
"""


def parameter_guide() -> dict[str, Any]:
    return {
        "general": [
            "phase_name and component basis used by pycalphad",
            "reference/endmember Gibbs energies on that basis",
            "molar volumes Vm_i in consistent units",
            "coordination numbers Z_i or effective coordination numbers",
            "directed pair parameters B_ij/B_ji or epsilon/h parameters",
            "temperature and composition ranges for fitting and validation",
            "mixing enthalpy, activity, chemical potential, or phase-equilibrium data",
        ],
        "molten_salt": [
            "define salt components or pseudo-binary components, e.g. LaCl3 and eutectic LiCl-KCl",
            "liquid molar volumes from density, TMA, or MD",
            "RDF/CN-derived solvation coordination numbers on the chosen component basis",
            "PMF-derived or fitted pair parameters B_ij/B_ji",
            "calorimetry ΔHmix and activity data for refining B_ij/B_ji",
            "explicit formula-unit normalization for common-anion or charge-asymmetric salts",
        ],
        "ceramic_solid": [
            "define substitutional/pseudo-components, e.g. U4+O2, Gd3+O1.5, U5+O2, Gd-VO motifs",
            "state the charge-compensation mechanism: U5+, oxygen vacancies, or mixed compensation",
            "molar volumes for endmembers and defect motifs from DFT, QHA, MD, or experiment",
            "effective coordination numbers from fluorite/Ia-3 geometry, RDFs, or relaxed motifs",
            "pair parameters for Gd-U4, Gd-U5, Gd-VO, U4-U5, and host-host interactions",
            "DFT/experimental mixing enthalpies, defect-pair energies, activities, or solubility limits",
            "magnetic/oxidation-state labels used to map DFT structures onto thermodynamic components",
        ],
        "pycalphad_strategy": [
            "implement the MIVM excess Gibbs energy as the GM contribution",
            "use corrected MIVM enthalpy expressions for fitting and validation, not as GM directly",
            "start with pseudo-binary evaluators before full multicomponent pycalphad minimization",
            "record all parameter units and component normalization in the TDB or companion metadata",
        ],
    }


def text_guide(system: str) -> str:
    data = parameter_guide()
    sections = ["general"]
    if system in {"molten", "all"}:
        sections.append("molten_salt")
    if system in {"ceramic", "solid", "all"}:
        sections.append("ceramic_solid")
    sections.append("pycalphad_strategy")

    labels = {
        "general": "General MIVM/pycalphad parameters",
        "molten_salt": "Molten-salt MIVM parameters",
        "ceramic_solid": "Solid/ceramic MIVM parameters, e.g. (Gd,U)O2",
        "pycalphad_strategy": "Implementation strategy",
    }
    lines = ["MIVM parameter guide", "====================", ""]
    for section in sections:
        lines.append(labels[section])
        lines.append("-" * len(labels[section]))
        for item in data[section]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calphad-mivm",
        description=(
            "Print Molecular Interaction Volume Model parameter guidance for molten salts "
            "and solid/ceramic solutions before pycalphad integration."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(MIVM_HELP_EPILOG),
    )
    subparsers = parser.add_subparsers(dest="command")
    guide = subparsers.add_parser(
        "guide",
        help="Print the MIVM parameter guide.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(MIVM_HELP_EPILOG),
    )
    guide.add_argument(
        "--system",
        choices=("all", "molten", "ceramic", "solid"),
        default="all",
        help="Which parameter guide to print. Default: all.",
    )
    guide.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "guide"
        args.system = "all"
        args.format = "text"
    if args.command == "guide":
        if args.format == "json":
            print(json.dumps(parameter_guide(), indent=2, sort_keys=True))
        else:
            print(text_guide(args.system), end="")
        return
    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
