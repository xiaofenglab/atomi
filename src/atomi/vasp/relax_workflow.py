"""General VASP relaxation workflow entry point.

This module is a VASP-facing wrapper around the shared materials optimization
relaxation engine. It intentionally accepts the same arguments as
``materials-opt relax-seeds`` so any POSCAR-producing front end can hand off to
one volume-scan and promotion workflow.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    from atomi.atat.bridge import relax_seeds_main

    relax_seeds_main(argv, prog="vasp-relax-workflow")
