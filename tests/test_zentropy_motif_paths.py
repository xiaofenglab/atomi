from __future__ import annotations

import csv
from pathlib import Path

from atomi.zentropy import motif_paths


def write_vasp_run(path: Path, structure_name: str = "CONTCAR", outcar: bool = True) -> None:
    path.mkdir(parents=True)
    (path / structure_name).write_text("structure\n", encoding="utf-8")
    if outcar:
        (path / "OUTCAR").write_text("outcar\n", encoding="utf-8")
    (path / "INCAR").write_text("MAGMOM = 1 -1\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_motif_paths_builds_relative_auto_metadata_index(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    write_vasp_run(root / "gd" / "spin_001")
    write_vasp_run(root / "gd" / "spin_002", structure_name="POSCAR")
    write_vasp_run(root / "missing_outcar", outcar=False)
    index = tmp_path / "motif_paths.csv"

    motif_paths.main([str(root), "--index", str(index)])

    rows = read_rows(index)
    assert [row["motif_id"] for row in rows] == ["gd__spin_001", "gd__spin_002"]
    assert rows[0]["run_dir"] == "runs/gd/spin_001"
    assert rows[0]["structure"] == "runs/gd/spin_001/CONTCAR"
    assert rows[0]["outcar"] == "runs/gd/spin_001/OUTCAR"
    assert rows[0]["incar"] == "runs/gd/spin_001/INCAR"


def test_motif_paths_appends_without_duplicating_existing_rows(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    write_vasp_run(root / "spin_001")
    index = tmp_path / "motif_paths.csv"

    motif_paths.main([str(root), "--index", str(index)])
    motif_paths.main([str(root), "--index", str(index)])
    write_vasp_run(root / "spin_002")
    motif_paths.main([str(root), "--index", str(index)])

    rows = read_rows(index)
    assert [row["motif_id"] for row in rows] == ["spin_001", "spin_002"]
