"""Validated orbital-entanglement datasets for OpenMolcas/QCMaquis plots."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "atomi.molcas_orbital_entanglement.v1"
MI_CONVENTIONS = {
    "half": 0.5,
    "full": 1.0,
}


@dataclass(frozen=True)
class EntanglementDataset:
    metadata: dict[str, Any]
    orbitals: tuple[dict[str, str], ...]
    pairs: tuple[dict[str, str], ...]
    metadata_path: Path
    orbitals_path: Path
    pairs_path: Path


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def _required_columns(
    rows: tuple[dict[str, str], ...],
    columns: set[str],
    path: Path,
) -> None:
    missing = sorted(columns - set(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _finite_nonnegative(value: str, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative, got {value!r}")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_entanglement_dataset(
    metadata_json: Path,
    orbitals_csv: Path,
    pairs_csv: Path,
) -> EntanglementDataset:
    metadata_path = metadata_json.expanduser().resolve()
    orbitals_path = orbitals_csv.expanduser().resolve()
    pairs_path = pairs_csv.expanduser().resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = EntanglementDataset(
        metadata=metadata,
        orbitals=_read_csv(orbitals_path),
        pairs=_read_csv(pairs_path),
        metadata_path=metadata_path,
        orbitals_path=orbitals_path,
        pairs_path=pairs_path,
    )
    validate_entanglement_dataset(dataset)
    return dataset


def validate_entanglement_dataset(
    dataset: EntanglementDataset,
    *,
    tolerance: float = 1.0e-7,
) -> None:
    metadata = dataset.metadata
    required_metadata = {
        "schema",
        "wavefunction_method",
        "state_id",
        "active_space",
        "orbital_basis",
        "orbital_order",
        "selected_root",
        "dmrg_bond_dimension",
        "dmrg_converged",
        "log_base",
        "mutual_information_convention",
        "source_files",
    }
    missing = sorted(required_metadata - set(metadata))
    if missing:
        raise ValueError(
            f"{dataset.metadata_path} is missing required metadata: {', '.join(missing)}"
        )
    if metadata["schema"] != SCHEMA:
        raise ValueError(f"Unsupported entanglement schema: {metadata['schema']!r}")
    method = str(metadata["wavefunction_method"])
    if "dmrg" not in method.lower() and "qcmaquis" not in method.lower():
        raise ValueError("wavefunction_method must identify DMRG or QCMaquis")
    if metadata["dmrg_converged"] is not True:
        raise ValueError("dmrg_converged must be true before entanglement plotting")
    if int(metadata["dmrg_bond_dimension"]) <= 0:
        raise ValueError("dmrg_bond_dimension must be positive")
    log_base = float(metadata["log_base"])
    if not math.isfinite(log_base) or log_base <= 0.0 or math.isclose(log_base, 1.0):
        raise ValueError("log_base must be finite, positive, and different from one")
    convention = str(metadata["mutual_information_convention"])
    if convention not in MI_CONVENTIONS:
        raise ValueError(
            "mutual_information_convention must be one of: "
            + ", ".join(sorted(MI_CONVENTIONS))
        )

    _required_columns(
        dataset.orbitals,
        {
            "state_id",
            "node_id",
            "label",
            "family",
            "orbital_order",
            "symmetry",
            "one_orbital_entropy",
        },
        dataset.orbitals_path,
    )
    _required_columns(
        dataset.pairs,
        {
            "state_id",
            "source",
            "target",
            "two_orbital_entropy",
            "mutual_information",
            "value",
        },
        dataset.pairs_path,
    )

    state_id = str(metadata["state_id"])
    orbital_ids: set[str] = set()
    entropies: dict[str, float] = {}
    declared_order = [str(item) for item in metadata["orbital_order"]]
    for row in dataset.orbitals:
        if row["state_id"] != state_id:
            raise ValueError("Orbital rows mix states or disagree with metadata state_id")
        node_id = row["node_id"]
        if node_id in orbital_ids:
            raise ValueError(f"Duplicate orbital node_id: {node_id}")
        orbital_ids.add(node_id)
        entropies[node_id] = _finite_nonnegative(
            row["one_orbital_entropy"],
            f"one_orbital_entropy[{node_id}]",
        )
    if declared_order != [
        row["node_id"]
        for row in sorted(dataset.orbitals, key=lambda item: int(item["orbital_order"]))
    ]:
        raise ValueError("metadata orbital_order disagrees with orbitals_csv ordering")

    seen_pairs: set[tuple[str, str]] = set()
    factor = MI_CONVENTIONS[convention]
    for row in dataset.pairs:
        if row["state_id"] != state_id:
            raise ValueError("Pair rows mix states or disagree with metadata state_id")
        source = row["source"]
        target = row["target"]
        if source == target:
            raise ValueError(f"Self-edge is not allowed: {source}")
        if source not in orbital_ids or target not in orbital_ids:
            raise ValueError(f"Pair references unknown orbital: {source}, {target}")
        pair = tuple(sorted((source, target)))
        if pair in seen_pairs:
            raise ValueError(f"Duplicate unordered orbital pair: {pair[0]}, {pair[1]}")
        seen_pairs.add(pair)
        two_entropy = _finite_nonnegative(
            row["two_orbital_entropy"],
            f"two_orbital_entropy[{source},{target}]",
        )
        mutual_information = _finite_nonnegative(
            row["mutual_information"],
            f"mutual_information[{source},{target}]",
        )
        plotted_value = _finite_nonnegative(
            row["value"],
            f"value[{source},{target}]",
        )
        expected = factor * (entropies[source] + entropies[target] - two_entropy)
        if expected < -tolerance:
            raise ValueError(
                f"Entropies imply negative mutual information for {source},{target}"
            )
        expected = max(0.0, expected)
        if not math.isclose(mutual_information, expected, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(
                f"Mutual information mismatch for {source},{target}: "
                f"declared {mutual_information:g}, expected {expected:g}"
            )
        if not math.isclose(plotted_value, mutual_information, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(
                f"Plot value must equal mutual_information for {source},{target}"
            )

    source_files = metadata["source_files"]
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("source_files must be a nonempty list of path/hash records")
    for record in source_files:
        if not isinstance(record, dict) or not {"path", "sha256"} <= set(record):
            raise ValueError("Each source_files record requires path and sha256")
        source_path = Path(record["path"]).expanduser()
        if not source_path.is_absolute():
            source_path = (dataset.metadata_path.parent / source_path).resolve()
        if not source_path.is_file():
            raise ValueError(f"Entanglement source file does not exist: {source_path}")
        if _sha256(source_path) != record["sha256"]:
            raise ValueError(f"Entanglement source hash mismatch: {source_path}")


def to_orbital_network_tables(
    dataset: EntanglementDataset,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return validated copies suitable for the generic orbital-network renderer."""

    validate_entanglement_dataset(dataset)
    nodes = [dict(row) for row in dataset.orbitals]
    edges = [dict(row) for row in dataset.pairs]
    return nodes, edges
