"""Shared interfaces for defect thermodynamic large-state backends."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class CETrainingRecord:
    """Backend-neutral CE/MC training-state record."""

    record_id: str
    structure_path: str
    mapped_parent_structure_path: str | None = None
    species_counts: dict[str, int] = field(default_factory=dict)
    composition: dict[str, float] = field(default_factory=dict)
    motif_features: dict[str, float] = field(default_factory=dict)
    energy_eV: float | None = None
    free_energy_terms: dict[str, Any] | None = None
    weight: float | None = None
    source: str = "unknown"
    uncertainty_eV: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CETrainingSet:
    """Backend-neutral CE/MC training set exchanged with smol or CASM."""

    system_name: str
    parent_structure_path: str
    sublattice_model: dict[str, Any] = field(default_factory=dict)
    species: dict[str, Any] = field(default_factory=dict)
    charge_constraints: list[str] = field(default_factory=list)
    composition_axes: dict[str, Any] = field(default_factory=dict)
    records: list[CETrainingRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl_rows(self) -> list[dict[str, Any]]:
        header = {
            "record_type": "metadata",
            "system_name": self.system_name,
            "parent_structure_path": self.parent_structure_path,
            "sublattice_model": self.sublattice_model,
            "species": self.species,
            "charge_constraints": self.charge_constraints,
            "composition_axes": self.composition_axes,
            "metadata": self.metadata,
        }
        rows = [header]
        for record in self.records:
            row = asdict(record)
            row["record_type"] = "training_record"
            rows.append(row)
        return rows

    @classmethod
    def from_jsonl_rows(cls, rows: list[dict[str, Any]]) -> "CETrainingSet":
        if not rows:
            raise ValueError("CETrainingSet JSONL is empty.")
        header = rows[0]
        if header.get("record_type") != "metadata":
            raise ValueError("First CETrainingSet JSONL row must be record_type=metadata.")
        records = [
            CETrainingRecord(
                record_id=str(row["record_id"]),
                structure_path=str(row.get("structure_path") or ""),
                mapped_parent_structure_path=row.get("mapped_parent_structure_path"),
                species_counts={str(key): int(value) for key, value in dict(row.get("species_counts") or {}).items()},
                composition={str(key): float(value) for key, value in dict(row.get("composition") or {}).items()},
                motif_features={str(key): float(value) for key, value in dict(row.get("motif_features") or {}).items()},
                energy_eV=row.get("energy_eV"),
                free_energy_terms=row.get("free_energy_terms"),
                weight=row.get("weight"),
                source=str(row.get("source") or "unknown"),
                uncertainty_eV=row.get("uncertainty_eV"),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in rows[1:]
            if row.get("record_type") == "training_record"
        ]
        return cls(
            system_name=str(header.get("system_name") or ""),
            parent_structure_path=str(header.get("parent_structure_path") or ""),
            sublattice_model=dict(header.get("sublattice_model") or {}),
            species=dict(header.get("species") or {}),
            charge_constraints=list(header.get("charge_constraints") or []),
            composition_axes=dict(header.get("composition_axes") or {}),
            records=records,
            metadata=dict(header.get("metadata") or {}),
        )


@dataclass
class ThermoSurface:
    """Common output surface schema for pycalphad/CALPHAD fitting routes."""

    phase_name: str
    backend: str
    t_grid: list[float] = field(default_factory=list)
    composition_grid: list[dict[str, float]] = field(default_factory=list)
    chemical_potential_grid: list[dict[str, float]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    motif_observables: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    convergence_metadata: dict[str, Any] = field(default_factory=dict)


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


def write_ce_training_jsonl(path: Path, training_set: CETrainingSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in training_set.to_jsonl_rows():
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_ce_training_jsonl(path: Path) -> CETrainingSet:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return CETrainingSet.from_jsonl_rows(rows)


def unavailable_run_result(backend: str, reason: str, install_hint: str = "") -> dict[str, Any]:
    return {
        "backend": backend,
        "status": "not_run",
        "available": False,
        "reason": reason,
        "install_hint": install_hint,
    }
