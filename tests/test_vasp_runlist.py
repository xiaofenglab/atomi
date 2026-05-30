from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read, write

from atomi.cli.main import main as atomi_main
from atomi.vasp.poscar_repeat import main as repeat_poscar_main
from atomi.vasp.runlist import main as listvasp_main


def touch_poscar(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "POSCAR").write_text("POSCAR\n", encoding="utf-8")


def test_listvasp_writes_relative_poscar_folder_runlist(tmp_path: Path, capsys) -> None:
    touch_poscar(tmp_path)
    touch_poscar(tmp_path / "randomized_candidates" / "random_001")
    touch_poscar(tmp_path / "randomized_candidates" / "random_002")
    touch_poscar(tmp_path / ".hidden" / "skip_me")
    (tmp_path / "no_poscar").mkdir()

    listvasp_main([str(tmp_path), "--output", "runlist.txt"])

    assert (tmp_path / "runlist.txt").read_text(encoding="utf-8").splitlines() == [
        ".",
        "randomized_candidates/random_001",
        "randomized_candidates/random_002",
    ]
    output = capsys.readouterr().out
    assert "Wrote 3 POSCAR folder(s)" in output


def test_listvasp_absolute_paths_and_atomi_alias(tmp_path: Path) -> None:
    touch_poscar(tmp_path / "candidate_a")
    touch_poscar(tmp_path / "candidate_b")
    runlist = tmp_path / "absolute_runlist.txt"

    atomi_main(["listvasp", str(tmp_path), "--output", str(runlist), "--absolute", "--quiet"])

    assert runlist.read_text(encoding="utf-8").splitlines() == [
        str((tmp_path / "candidate_a").resolve()),
        str((tmp_path / "candidate_b").resolve()),
    ]


def test_listvasp_can_include_hidden_and_limit_depth(tmp_path: Path) -> None:
    touch_poscar(tmp_path / ".hidden")
    touch_poscar(tmp_path / "a" / "b")
    runlist = tmp_path / "runlist.txt"

    listvasp_main([str(tmp_path), "--include-hidden", "--max-depth", "1", "-o", str(runlist), "--quiet"])

    assert runlist.read_text(encoding="utf-8").splitlines() == [".hidden"]


def test_repeat_poscar_writes_supercell_and_metadata(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    atoms = Atoms(
        ["U", "C", "C"],
        scaled_positions=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)],
        cell=[3.0, 4.0, 5.0],
        pbc=True,
    )
    write(poscar, atoms, format="vasp", direct=True, sort=False, vasp5=True)
    (tmp_path / "INCAR").write_text("MAGMOM = 2 0 0\n", encoding="utf-8")
    outdir = tmp_path / "UC2_2x1x1"

    repeat_poscar_main([str(poscar), "--repeat", "2x1x1", "--outdir", str(outdir), "--copy-inputs"])

    repeated = read(outdir / "POSCAR", format="vasp")
    assert len(repeated) == 6
    assert repeated.cell.lengths()[0] == pytest.approx(6.0)
    assert repeated.cell.lengths()[1] == pytest.approx(4.0)
    assert repeated.cell.lengths()[2] == pytest.approx(5.0)
    assert (outdir / "INCAR").read_text(encoding="utf-8") == "MAGMOM = 2 0 0\n"
    metadata = json.loads((outdir / "POSCAR.repeat_metadata.json").read_text(encoding="utf-8"))
    assert metadata["repeat"] == [2, 1, 1]
    assert metadata["input_atoms"] == 3
    assert metadata["output_atoms"] == 6


def test_repeat_poscar_atomi_alias_accepts_three_repeat_values(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    atoms = Atoms(
        ["U", "C", "C"],
        scaled_positions=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)],
        cell=[3.0, 4.0, 5.0],
        pbc=True,
    )
    write(poscar, atoms, format="vasp", direct=True, sort=False, vasp5=True)
    output = tmp_path / "POSCAR_1x2x1"

    atomi_main(["poscar-repeat", str(poscar), "--repeat", "1", "2", "1", "--output", str(output)])

    repeated = read(output, format="vasp")
    assert len(repeated) == 6
    assert repeated.cell.lengths()[0] == pytest.approx(3.0)
    assert repeated.cell.lengths()[1] == pytest.approx(8.0)
    assert repeated.cell.lengths()[2] == pytest.approx(5.0)
