"""Route map for Atomi XAFS simulation workflows.

This module keeps the XAFS method choice explicit:

Route A
    VASP/static or MD structures -> absorber-centered clusters -> FEFF ->
    Larch/xraydb postprocessing and comparison.

Route B
    VASP/DFT+U parent context -> QE/OCEAN periodic-solid XANES/BSE workflow.

The route map is intentionally method-level and material-agnostic so project
students can apply it to UC2, U4O9, molten salts, aqueous clusters, or future
systems without rewriting the policy.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


ROUTE_A = {
    "id": "route_a_vasp_feff_larch",
    "label": "Route A: VASP/static-or-MD -> FEFF -> Larch",
    "purpose": (
        "Local absorber-centered XAFS/EXAFS comparison from a VASP-relaxed "
        "structure, AIMD/MD trajectory, or averaged cluster ensemble."
    ),
    "commands": [
        "atomi xafs_vasp_feff_prepare --traj CONTCAR --absorber U --edge L3 --cluster-radius 6.0 --outdir xafs_routeA_prepare",
        "atomi xafs_larch_run --prepared-dir xafs_routeA_prepare --outdir xafs_routeA_larch",
        "atomi xafs_md_compare --xafs-dir xafs_routeA_larch --exp-chi experiment.chik --outdir xafs_routeA_compare",
    ],
    "strengths": [
        "Fast local-structure screen and ensemble averaging.",
        "Works naturally with VASP-relaxed structures, MD frames, and finite clusters.",
        "Good companion to PDF/RDF and cluster-family analysis.",
    ],
    "guards": [
        "Validate absorber identity, edge, cluster radius, and FEFF potential map.",
        "Record whether FEFF was actually run or only pre/postprocessed.",
        "Use Larch/xraydb versions and k/R ranges in the report.",
    ],
    "limits": [
        "Local-cluster approximation; not a full periodic BSE/core-hole calculation.",
        "FEFF executable and Larch runtime are optional external dependencies.",
    ],
}


ROUTE_B = {
    "id": "route_b_qe_ocean",
    "label": "Route B: QE/OCEAN periodic-solid XANES",
    "purpose": (
        "Periodic-solid XANES/XAS screen using OCEAN's QE/Shirley/BSE pipeline, "
        "with VASP/DFT+U used as structure/provenance context rather than a "
        "direct wavefunction backend."
    ),
    "commands": [
        "atomi ocean-xanes-status",
        "atomi ocean-xanes-bridge prepare --structure CONTCAR --vasp-dir vasp_dftu_scf --absorber U --edge M4 --dft-engine quantum_espresso --outdir ocean_U_M4",
        "sbatch ocean_U_M4/submit_ocean_xanes.sbatch",
        "atomi ocean-xanes-bridge collect --ocean-dir ocean_U_M4 --write ocean_U_M4/ocean_xanes_summary.json",
    ],
    "strengths": [
        "Periodic solid route with screening/core-hole/BSE-style physics.",
        "Better suited to band-structure-sensitive XANES than finite clusters.",
    ],
    "guards": [
        "Use native OCEAN dft{ qe } / --dft-engine quantum_espresso on JUSTUS2.",
        "Use OCEAN 2.9.7 keywords nstep/toldfe/mixing for SCF controls.",
        "Validate pseudo+OPF, absorber site, edge, k-grid, bands, and broadening before science claims.",
    ],
    "limits": [
        "Current Atomi bridge does not pass VASP WAVECAR/CHGCAR directly into OCEAN.",
        "U/C/O pseudo+OPF choices remain method-development diagnostics until validated.",
    ],
}


def build_xafs_route_status(check_runtime: bool = False) -> dict[str, Any]:
    """Return the Atomi XAFS route map and, optionally, runtime probes."""
    status: dict[str, Any] = {
        "schema": "atomi.xafs.routes.v1",
        "route_order": ["route_a_vasp_feff_larch", "route_b_qe_ocean"],
        "routes": [ROUTE_A, ROUTE_B],
        "project_policy": {
            "comparison_rule": (
                "For UC2 and U4O9, run Route A FEFF/Larch alongside Route B "
                "QE/OCEAN when possible; record both in student reports and "
                "promote only after route-specific guards pass."
            ),
            "portfolio_owner": "Sarah",
            "atomi_owner": "Anna",
        },
    }
    if check_runtime:
        from atomi.xafs.ocean import build_status_report as build_ocean_status_report
        from atomi.xafs.status import build_xafs_status

        status["runtime"] = {
            "route_a_larch_feff": build_xafs_status(),
            "route_b_ocean": build_ocean_status_report(argparse.Namespace(executable=None, root=None, bin=None)),
        }
    return status


def print_route_status(status: dict[str, Any]) -> None:
    print("Atomi XAFS route map")
    for route in status["routes"]:
        print(f"  {route['label']}")
        print(f"    id      : {route['id']}")
        print(f"    purpose : {route['purpose']}")
        print("    main commands:")
        for command in route["commands"]:
            print(f"      - {command}")
        print("    guards:")
        for guard in route["guards"]:
            print(f"      - {guard}")
    if "runtime" in status:
        route_a = status["runtime"]["route_a_larch_feff"]
        route_b = status["runtime"]["route_b_ocean"]["ocean"]
        print("  runtime summary:")
        print(f"    Route A Larch mode : {route_a.get('larch_mode')}")
        print(f"    Route A FEFF env   : {route_a.get('feff_executable') or 'not configured'}")
        print(f"    Route B OCEAN      : {'available' if route_b.get('available') else 'missing'} ({route_b.get('resolved_executable') or route_b.get('executable')})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xafs_routes",
        description="Print Atomi XAFS Route A/B policy: VASP-FEFF/Larch vs QE-OCEAN.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--check-runtime", action="store_true", help="Also probe optional FEFF/Larch and OCEAN runtime status.")
    args = parser.parse_args(argv)
    status = build_xafs_route_status(check_runtime=args.check_runtime)
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print_route_status(status)


if __name__ == "__main__":
    main()
