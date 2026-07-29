"""Plan reproducible OpenMolcas natural-orbital replay and GRID_IT rendering."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_NATURAL_ORBITAL_PLAN = "atomi.molcas_natural_orbital_plan.v1"


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not label:
        raise ValueError("--label must contain at least one letter or number")
    return label


def _orbital_pairs(specs: list[str]) -> list[tuple[int, int]]:
    if not specs:
        raise ValueError("At least one --orbital SYM:ORB or SYM:FIRST-LAST is required")
    pairs: list[tuple[int, int]] = []
    for spec in specs:
        for token in spec.replace(",", " ").split():
            if ":" not in token:
                raise ValueError(f"Orbital selection must be SYM:ORB or SYM:FIRST-LAST, not {token!r}")
            symmetry_text, orbital_text = token.split(":", 1)
            symmetry = int(symmetry_text)
            if symmetry < 1:
                raise ValueError(f"Symmetry index must be positive: {token!r}")
            if "-" in orbital_text:
                first_text, last_text = orbital_text.split("-", 1)
                first, last = int(first_text), int(last_text)
                if first < 1 or last < first:
                    raise ValueError(f"Invalid orbital range: {token!r}")
                requested = [(symmetry, orbital) for orbital in range(first, last + 1)]
            else:
                orbital = int(orbital_text)
                if orbital < 1:
                    raise ValueError(f"Orbital index must be positive: {token!r}")
                requested = [(symmetry, orbital)]
            for pair in requested:
                if pair not in pairs:
                    pairs.append(pair)
    return pairs


def _relativistic_lines(method: str, amfi: bool) -> list[str]:
    if method == "x2c":
        lines = ["RX2C"]
    elif method == "dkh2":
        lines = ["RELATIVISTIC", "R02O02"]
    else:  # pragma: no cover - argparse enforces the choices
        raise ValueError(f"Unsupported relativistic method: {method}")
    lines.append("AMFI" if amfi else "NOAMFI")
    return lines


def _gateway_block(args: argparse.Namespace) -> str:
    relativistic = "\n".join(_relativistic_lines(args.relativity, not args.no_amfi))
    return f"""\
&GATEWAY
COORD={args.coord.expanduser()}
BASIS
{args.basis}
Group
{args.group}
NOMOVE
{relativistic}
ANGM
{args.angular_momentum_origin[0]:g} {args.angular_momentum_origin[1]:g} {args.angular_momentum_origin[2]:g}
BSSH
End of input

&SEWARD
ONEONLY
End of input
"""


def _state_note(args: argparse.Namespace) -> str:
    if args.state_role == "ground":
        return "Ground-state spin-free natural orbitals from the selected RASSCF root."
    if args.state_role == "spin-free-excited":
        return "Spin-free excited-state natural orbitals from the selected RASSCF root."
    return (
        f"Spin-free parent natural orbitals used to interpret SO state {args.so_state}; "
        "these are not a unique spin-orbit orbital or transition orbital."
    )


def natural_orbital_plan(args: argparse.Namespace) -> int:
    if args.root < 1:
        raise ValueError("--root must be positive")
    if args.state_role == "so-parent" and (
        args.so_state is None or args.spin_free_state is None
    ):
        raise ValueError("--state-role so-parent requires --so-state and --spin-free-state")
    pairs = _orbital_pairs(args.orbital)
    label = _safe_label(args.label)
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    replay_input = outdir / f"{label}.inp"
    grid_input = outdir / f"{label}_grid.inp"
    expected_siorb = outdir / f"{label}.SiOrb.1"
    expected_grid = outdir / f"{label}_grid.grid"
    manifest_path = outdir / f"{label}_natural_orbital_plan.json"

    metadata_comments = [
        f"* Atomi natural-orbital replay: {args.state_role}.",
        f"* JobIph root {args.root}. {_state_note(args)}",
    ]
    if args.spin_free_state is not None:
        metadata_comments.append(f"* Production spin-free state label: {args.spin_free_state}.")
    if args.parent_weight_percent is not None:
        metadata_comments.append(
            f"* Printed parent weight in the selected SO state: {args.parent_weight_percent:.4g}%."
        )
    replay_text = "\n".join(metadata_comments) + "\n\n" + _gateway_block(args)
    replay_text += f"""\

>>COPY {args.jobiph.expanduser()} JOB001

&RASSI
EJOB
NROFJOBIPH
1 1
{args.root}
NATORB
1
TRD1
End of input
"""

    orbital_rows = "\n".join(f"{symmetry} {orbital}" for symmetry, orbital in pairs)
    axis_length = args.grid_axis_length
    grid_text = f"""\
* Atomi GRID_IT input for {args.state_role} natural orbitals.
* {_state_note(args)}

{_gateway_block(args)}

&GRID_IT
TITLE
{args.title or label}
FILEORB
{expected_siorb}
NAME
{label}_natural_orbitals
NOLUSCUS
ASCII
NPOINTS
{args.grid_points} {args.grid_points} {args.grid_points}
GORI
{args.grid_origin[0]:g} {args.grid_origin[1]:g} {args.grid_origin[2]:g}
{axis_length:g} 0.0 0.0
0.0 {axis_length:g} 0.0
0.0 0.0 {axis_length:g}
ORBITAL
{len(pairs)}
{orbital_rows}
MULLIKEN
End of input
"""
    replay_input.write_text(replay_text, encoding="utf-8")
    grid_input.write_text(grid_text, encoding="utf-8")

    selector_flags = " ".join(
        f"--orbital {symmetry}:{orbital}" for symmetry, orbital in pairs
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA_NATURAL_ORBITAL_PLAN,
        "label": label,
        "state_role": args.state_role,
        "state_interpretation": _state_note(args),
        "source_jobiph": str(args.jobiph.expanduser()),
        "jobiph_root": args.root,
        "spin_free_state": args.spin_free_state,
        "so_state": args.so_state,
        "printed_parent_weight_percent": args.parent_weight_percent,
        "relativity": {
            "scalar_hamiltonian": "X2C" if args.relativity == "x2c" else "DKH2",
            "property_picture_change": (
                "X2C transformed property integrals"
                if args.relativity == "x2c"
                else "DKH2 R02O02"
            ),
            "spin_orbit_integrals": "AMFI" if not args.no_amfi else "disabled",
            "input_is_explicit": True,
        },
        "basis": args.basis,
        "group": args.group,
        "coordinate_source": str(args.coord.expanduser()),
        "selected_orbitals": [
            {"symmetry": symmetry, "orbital": orbital}
            for symmetry, orbital in pairs
        ],
        "files": {
            "replay_input": str(replay_input),
            "expected_siorb": str(expected_siorb),
            "grid_input": str(grid_input),
            "expected_ascii_grid": str(expected_grid),
        },
        "commands": [
            f"pymolcas {replay_input.name} > {label}.out",
            f"pymolcas {grid_input.name} > {label}_grid.out",
            (
                "molcas-postanalysis orbital-grid-render "
                f"--grid {expected_grid.name} {selector_flags} "
                f"--outdir {label}_figures"
            ),
        ],
        "guards": [
            "Replay must finish with successful RASSI EJOB, NATORB, and TRD1 output.",
            "The SiOrb file must have nonzero natural occupations and a nonzero one-particle density.",
            "GRID_IT output must pass finite-value and positive/negative phase guards.",
            "Use one common absolute isovalue for quantitative comparisons.",
            "For an SO-parent plan, report the RASSI spin-free parent weights separately.",
            "Do not label a spin-free parent natural orbital as a unique SO orbital.",
            "Use RASSI SONT for spin-orbit natural transition orbitals.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote natural-orbital replay input: {replay_input}")
    print(f"Wrote GRID_IT input: {grid_input}")
    print(f"Wrote natural-orbital plan: {manifest_path}")
    return 0


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "natural-orbital-plan",
        help="Write matched RASSI EJOB/NATORB/TRD1 replay and GRID_IT inputs.",
    )
    parser.add_argument("--jobiph", type=Path, required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--coord", type=Path, required=True)
    parser.add_argument("--basis", default="ANO-RCC-VDZP")
    parser.add_argument("--group", default="XY XYZ")
    parser.add_argument(
        "--relativity",
        choices=("x2c", "dkh2"),
        default="x2c",
        help="Explicit scalar-relativistic model; DKH2 uses R02O02 property correction.",
    )
    parser.add_argument("--no-amfi", action="store_true")
    parser.add_argument(
        "--state-role",
        choices=("ground", "spin-free-excited", "so-parent"),
        default="spin-free-excited",
    )
    parser.add_argument("--spin-free-state", type=int)
    parser.add_argument("--so-state", type=int)
    parser.add_argument("--parent-weight-percent", type=float)
    parser.add_argument("--orbital", action="append", default=[], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--outdir", type=Path, default=Path("molcas_natural_orbitals"))
    parser.add_argument("--grid-points", type=int, default=81)
    parser.add_argument("--grid-origin", type=float, nargs=3, default=(-7.0, -7.0, -7.0))
    parser.add_argument("--grid-axis-length", type=float, default=14.0)
    parser.add_argument(
        "--angular-momentum-origin",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
    )
    parser.set_defaults(func=natural_orbital_plan)
    return parser
