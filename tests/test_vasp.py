from pathlib import Path

from atomi.codes.vasp import missing_inputs


def test_missing_inputs_reports_absent_files(tmp_path: Path) -> None:
    (tmp_path / "INCAR").write_text("SYSTEM = test\n", encoding="utf-8")

    assert missing_inputs(tmp_path) == ["POSCAR", "POTCAR", "KPOINTS"]
