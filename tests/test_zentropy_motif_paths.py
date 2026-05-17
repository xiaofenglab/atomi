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


def write_poscar(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "U O",
                "1.0",
                "4 0 0",
                "0 4 0",
                "0 0 4",
                "U O",
                "2 4",
                "Direct",
                "0 0 0",
                "0.5 0.5 0.5",
                "0.25 0.25 0.25",
                "0.75 0.75 0.75",
                "0.25 0.75 0.25",
                "0.75 0.25 0.75",
                "",
            ]
        ),
        encoding="utf-8",
    )


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


def test_motif_paths_reference_mode_builds_defect_chem_reference_index(tmp_path: Path) -> None:
    reference = tmp_path / "references" / "UO2"
    reference.mkdir(parents=True)
    write_poscar(reference / "CONTCAR")
    (reference / "OUTCAR").write_text("free  energy   TOTEN  =  -120.0 eV\n", encoding="utf-8")
    (reference / "INCAR").write_text("INCAR\n", encoding="utf-8")
    index = tmp_path / "reference_phase_index.csv"

    motif_paths.main(
        [
            str(reference),
            "--mode",
            "reference",
            "--index",
            str(index),
            "--reference-id",
            "parent_UO2",
            "--role",
            "parent",
            "--source",
            "dft",
        ]
    )

    reference_rows = read_rows(index)
    assert reference_rows[0]["reference_id"] == "parent_UO2"
    assert reference_rows[0]["formula"] == "UO2"
    assert reference_rows[0]["path"] == "references/UO2/OUTCAR"
    assert reference_rows[0]["run_dir"] == "references/UO2"
    assert reference_rows[0]["role"] == "parent"
    assert reference_rows[0]["source"] == "dft"


def test_motif_paths_reference_mode_replaces_by_reference_id(tmp_path: Path) -> None:
    reference = tmp_path / "references" / "Gd2O3"
    write_vasp_run(reference)
    index = tmp_path / "reference_phase_index.csv"

    motif_paths.main(
        [
            str(reference),
            "--mode",
            "reference",
            "--index",
            str(index),
            "--reference-id",
            "Gd2O3",
            "--formula",
            "Gd2O3",
        ]
    )
    motif_paths.main(
        [
            str(reference),
            "--mode",
            "reference",
            "--index",
            str(index),
            "--reference-id",
            "Gd2O3",
            "--formula",
            "Gd2O3",
            "--role",
            "dopant_oxide",
            "--replace-existing",
        ]
    )

    reference_rows = read_rows(index)
    assert len(reference_rows) == 1
    assert reference_rows[0]["reference_id"] == "Gd2O3"
    assert reference_rows[0]["role"] == "dopant_oxide"
