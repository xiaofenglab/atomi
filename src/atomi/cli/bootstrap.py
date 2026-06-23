"""Lightweight Atomi CLI bootstrap.

Registered pass-through commands can dispatch before importing the legacy CLI,
which keeps small utilities such as ``atomi local-structure doctor`` responsive
even in environments with heavy optional scientific IO plugins.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from atomi.cli.registry import dispatch_registered_command


def main(argv: Sequence[str] | None = None) -> object:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if dispatch_registered_command(raw_args):
        return None

    from atomi.cli.main import main as legacy_main

    return legacy_main(raw_args)


if __name__ == "__main__":  # pragma: no cover
    main()
