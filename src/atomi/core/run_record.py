"""Shared run/evidence records for cross-module workflow tracking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunArtifact:
    """A file or directory produced by a computation or post-processing step."""

    path: str
    role: str
    exists: bool | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicalGuardResult:
    """A single physical/convergence guard attached to a run record."""

    name: str
    status: str
    value: Any = None
    threshold: Any = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    """Backend-neutral evidence record for DFT/MD/ML/post-processing runs."""

    record_id: str
    engine: str
    run_type: str
    path: str
    status: str = "unknown"
    job_id: str | None = None
    system_name: str | None = None
    composition: dict[str, float] = field(default_factory=dict)
    structure_path: str | None = None
    energy_eV: float | None = None
    free_energy_terms: dict[str, Any] = field(default_factory=dict)
    moments: dict[str, Any] = field(default_factory=dict)
    artifacts: list[RunArtifact] = field(default_factory=list)
    guards: list[PhysicalGuardResult] = field(default_factory=list)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def passed_required_guards(self) -> bool:
        """Return true when no guard is explicitly failed."""

        return all(guard.status.lower() not in {"fail", "failed", "error"} for guard in self.guards)


def run_record_to_dict(record: RunRecord) -> dict[str, Any]:
    row = asdict(record)
    row["record_type"] = "run_record"
    return row


def run_record_from_dict(row: dict[str, Any]) -> RunRecord:
    artifacts = [RunArtifact(**dict(item)) for item in row.get("artifacts") or []]
    guards = [PhysicalGuardResult(**dict(item)) for item in row.get("guards") or []]
    return RunRecord(
        record_id=str(row["record_id"]),
        engine=str(row.get("engine") or "unknown"),
        run_type=str(row.get("run_type") or "unknown"),
        path=str(row.get("path") or ""),
        status=str(row.get("status") or "unknown"),
        job_id=row.get("job_id"),
        system_name=row.get("system_name"),
        composition={str(key): float(value) for key, value in dict(row.get("composition") or {}).items()},
        structure_path=row.get("structure_path"),
        energy_eV=row.get("energy_eV"),
        free_energy_terms=dict(row.get("free_energy_terms") or {}),
        moments=dict(row.get("moments") or {}),
        artifacts=artifacts,
        guards=guards,
        uncertainty=dict(row.get("uncertainty") or {}),
        metadata=dict(row.get("metadata") or {}),
    )


def write_run_records_jsonl(path: Path, records: list[RunRecord]) -> None:
    """Write run records to JSONL for project registries and monitoring."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(run_record_to_dict(record), sort_keys=True) + "\n")


def read_run_records_jsonl(path: Path) -> list[RunRecord]:
    """Read run records from JSONL."""

    records: list[RunRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_type", "run_record") == "run_record":
            records.append(run_record_from_dict(row))
    return records
