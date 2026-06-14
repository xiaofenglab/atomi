from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms
from ase.io import write

from atomi.ml.crystal_graph_dataset import (
    SCHEMA,
    build_graph_dataset,
    build_graph_dataset_from_ce_training_jsonl,
    console_main,
    validate_graph_jsonl,
)
from atomi.zentropy.backends.base import CETrainingRecord, CETrainingSet, write_ce_training_jsonl


def _write_nacl(path: Path) -> None:
    atoms = Atoms("NaCl", positions=[[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]], cell=[6.0, 6.0, 6.0], pbc=True)
    write(path, atoms, format="extxyz")


def test_build_graph_dataset_from_structure_file(tmp_path: Path) -> None:
    structure = tmp_path / "nacl.extxyz"
    _write_nacl(structure)

    output = tmp_path / "graphs.jsonl"
    summary = build_graph_dataset(
        [structure],
        output,
        cutoff=3.0,
        labels={"nacl": {"energy_eV": -1.25}},
        metadata={"project": "unit"},
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary.n_records == 1
    assert rows[0]["schema"] == SCHEMA
    assert rows[0]["species_counts"] == {"Cl": 1, "Na": 1}
    assert rows[0]["labels"]["energy_eV"] == -1.25
    assert rows[0]["metadata"]["project"] == "unit"
    assert len(rows[0]["node_features"]) == 2
    assert len(rows[0]["edges"]) > 0

    validation = validate_graph_jsonl(output)
    assert validation.n_records == 1
    assert validation.n_edges_total == summary.n_edges_total


def test_build_graph_dataset_uses_branch_aware_vasp_record_ids(tmp_path: Path) -> None:
    first = tmp_path / "ideal_exact_1x1x3" / "01_vacancy_separated" / "POSCAR"
    second = tmp_path / "refined_Na_vac_1x3x5" / "01_vacancy_separated" / "POSCAR"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    atoms = Atoms("NaCl", positions=[[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]], cell=[6.0, 6.0, 6.0], pbc=True)
    write(first, atoms, format="vasp")
    write(second, atoms, format="vasp")

    output = tmp_path / "graphs.jsonl"
    build_graph_dataset([first, second], output, cutoff=3.0)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    record_ids = [row["record_id"] for row in rows]

    assert record_ids == [
        "ideal_exact_1x1x3__01_vacancy_separated__POSCAR",
        "refined_Na_vac_1x3x5__01_vacancy_separated__POSCAR",
    ]
    assert len(set(record_ids)) == 2


def test_crystal_graph_dataset_console_main_returns_none(tmp_path: Path, capsys) -> None:
    structure = tmp_path / "nacl.extxyz"
    _write_nacl(structure)
    output = tmp_path / "graphs.jsonl"

    assert console_main(["build", str(structure), "--out", str(output), "--cutoff", "3.0"]) is None
    printed = json.loads(capsys.readouterr().out)

    assert printed["n_records"] == 1
    assert output.exists()


def test_build_graph_dataset_from_ce_training_jsonl_resolves_relative_paths(tmp_path: Path) -> None:
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()
    structure = structure_dir / "seed.extxyz"
    _write_nacl(structure)
    training = CETrainingSet(
        system_name="NaCl",
        parent_structure_path="rocksalt",
        records=[
            CETrainingRecord(
                record_id="seed",
                structure_path="structures/seed.extxyz",
                composition={"x_NaCl": 1.0},
                motif_features={"nn_Na_Cl": 1.0},
                energy_eV=-1.0,
                uncertainty_eV=0.05,
                source="unit",
            )
        ],
    )
    training_jsonl = tmp_path / "training.jsonl"
    write_ce_training_jsonl(training_jsonl, training)

    output = tmp_path / "ce_graphs.jsonl"
    summary = build_graph_dataset_from_ce_training_jsonl(training_jsonl, output, cutoff=3.0)
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

    assert summary.n_records == 1
    assert summary.n_skipped == 0
    assert row["record_id"] == "seed"
    assert row["labels"]["energy_eV"] == -1.0
    assert row["labels"]["composition"]["x_NaCl"] == 1.0
    assert row["metadata"]["system_name"] == "NaCl"
