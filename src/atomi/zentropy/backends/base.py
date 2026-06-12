"""Shared interfaces for defect thermodynamic large-state backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from atomi.thermo.schemas import (
    CETrainingRecord,
    CETrainingSet,
    ThermoSurface,
    read_ce_training_jsonl,
    write_ce_training_jsonl,
)


class ThermoBackend(Protocol):
    """Protocol for optional large-state thermodynamics backends."""

    name: str

    def available(self) -> bool:
        ...

    def capability_report(self) -> dict[str, Any]:
        ...

    def prepare(self, config: dict[str, Any], ensemble: Any) -> dict[str, Any]:
        ...

    def run(self, config: dict[str, Any], ensemble: Any, output_dir: Path) -> dict[str, Any]:
        ...

    def collect(self, output_dir: Path) -> dict[str, Any]:
        ...


def unavailable_run_result(backend: str, reason: str, install_hint: str = "") -> dict[str, Any]:
    return {
        "backend": backend,
        "status": "not_run",
        "available": False,
        "reason": reason,
        "install_hint": install_hint,
    }
