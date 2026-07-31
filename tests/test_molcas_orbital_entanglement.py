from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from atomi.qchem.molcas_orbital_entanglement import (
    SCHEMA,
    load_entanglement_dataset,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "qcmaquis_results.h5"
    source.write_bytes(b"synthetic validated test fixture")
    orbitals = tmp_path / "orbitals.csv"
    pairs = tmp_path / "pairs.csv"
    metadata = tmp_path / "metadata.json"
    _write_csv(
        orbitals,
        [
            "state_id",
            "node_id",
            "label",
            "family",
            "orbital_order",
            "symmetry",
            "one_orbital_entropy",
            "status",
        ],
        [
            {
                "state_id": "sf1",
                "node_id": "o1",
                "label": "Au19",
                "family": "U 5f",
                "orbital_order": 1,
                "symmetry": "Au",
                "one_orbital_entropy": 0.4,
                "status": "accepted",
            },
            {
                "state_id": "sf1",
                "node_id": "o2",
                "label": "Bu25",
                "family": "U 5f",
                "orbital_order": 2,
                "symmetry": "Bu",
                "one_orbital_entropy": 0.3,
                "status": "accepted",
            },
        ],
    )
    _write_csv(
        pairs,
        [
            "state_id",
            "source",
            "target",
            "two_orbital_entropy",
            "mutual_information",
            "value",
        ],
        [
            {
                "state_id": "sf1",
                "source": "o1",
                "target": "o2",
                "two_orbital_entropy": 0.5,
                "mutual_information": 0.2,
                "value": 0.2,
            }
        ],
    )
    metadata.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "wavefunction_method": "QCMaquis DMRG-CI",
                "state_id": "sf1",
                "active_space": "CAS(2,7)",
                "orbital_basis": "state-averaged active molecular orbitals",
                "orbital_order": ["o1", "o2"],
                "selected_root": 1,
                "dmrg_bond_dimension": 1024,
                "dmrg_converged": True,
                "log_base": 2.718281828459045,
                "mutual_information_convention": "full",
                "source_files": [{"path": str(source), "sha256": _sha256(source)}],
            }
        ),
        encoding="utf-8",
    )
    return metadata, orbitals, pairs


def test_loads_complete_qcmaquis_entanglement_dataset(tmp_path: Path) -> None:
    metadata, orbitals, pairs = _fixture(tmp_path)
    dataset = load_entanglement_dataset(metadata, orbitals, pairs)
    assert dataset.metadata["state_id"] == "sf1"
    assert len(dataset.orbitals) == 2
    assert len(dataset.pairs) == 1


def test_rejects_method_label_without_convergence_and_hash_provenance(
    tmp_path: Path,
) -> None:
    metadata, orbitals, pairs = _fixture(tmp_path)
    record = json.loads(metadata.read_text(encoding="utf-8"))
    record["dmrg_converged"] = False
    metadata.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="dmrg_converged"):
        load_entanglement_dataset(metadata, orbitals, pairs)


def test_rejects_mutual_information_inconsistent_with_entropies(
    tmp_path: Path,
) -> None:
    metadata, orbitals, pairs = _fixture(tmp_path)
    rows = list(csv.DictReader(pairs.open(newline="", encoding="utf-8")))
    rows[0]["mutual_information"] = "0.1"
    rows[0]["value"] = "0.1"
    _write_csv(pairs, list(rows[0]), rows)
    with pytest.raises(ValueError, match="Mutual information mismatch"):
        load_entanglement_dataset(metadata, orbitals, pairs)


def test_rejects_duplicate_unordered_pair(tmp_path: Path) -> None:
    metadata, orbitals, pairs = _fixture(tmp_path)
    rows = list(csv.DictReader(pairs.open(newline="", encoding="utf-8")))
    rows.append(dict(rows[0]) | {"source": "o2", "target": "o1"})
    _write_csv(pairs, list(rows[0]), rows)
    with pytest.raises(ValueError, match="Duplicate unordered orbital pair"):
        load_entanglement_dataset(metadata, orbitals, pairs)
