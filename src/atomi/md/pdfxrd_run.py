"""Workflow/batch PDF-XRD liquid/solid guard entry point.

This is a stable public alias for :mod:`atomi.md.phase_order_guard`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    from atomi.md.phase_order_guard import main as phase_order_guard_main

    return phase_order_guard_main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    main()
