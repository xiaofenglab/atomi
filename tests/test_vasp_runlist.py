from __future__ import annotations

from pathlib import Path

from atomi.cli.main import main as atomi_main
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
