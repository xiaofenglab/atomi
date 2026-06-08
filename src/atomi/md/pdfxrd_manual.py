"""Manual PDF/RDF and powder-XRD inspection entry points.

This module is the public Atomi front door for one-off structure inspection.
Use ``static`` for CIF/POSCAR structures and ``md-frame`` for selected MD
frames from VASP XDATCAR, LAMMPS dump, or CP2K XYZ trajectories.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "static",
        help="Manual PDF/XRD overlay for one or more CIF/POSCAR/CONTCAR structures.",
    )
    sub.add_parser(
        "md-frame",
        aliases=("md_frame", "frame"),
        help="Manual PDF/XRD overlay for selected MD trajectory frames.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = list(argv or [])
    if not args or args[0] in {"-h", "--help"}:
        build_parser().parse_args(args)
        return
    command, rest = args[0], args[1:]
    if command == "static":
        from atomi.md.pdfxrd_manual_static import main as static_main

        static_main(rest)
        return
    if command in {"md-frame", "md_frame", "frame"}:
        from atomi.md.pdfxrd_manual_md_frame import main as md_frame_main

        md_frame_main(rest)
        return
    build_parser().parse_args(args)


if __name__ == "__main__":
    main()
