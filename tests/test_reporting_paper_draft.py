from __future__ import annotations

import json
from pathlib import Path

from atomi.cli.main import main as atomi_main
from atomi.reporting import paper_draft


def write_vasp_run(root: Path) -> None:
    root.mkdir()
    (root / "POSCAR").write_text(
        "\n".join(
            [
                "UO2 test",
                "1.0",
                "5 0 0",
                "0 5 0",
                "0 0 5",
                "U O",
                "1 2",
                "Direct",
                "0 0 0",
                "0.25 0.25 0.25",
                "0.75 0.75 0.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "INCAR").write_text(
        "ENCUT = 520\nEDIFF = 1E-6\nISPIN = 2\nMAGMOM = 2 0 0\n",
        encoding="utf-8",
    )
    (root / "KPOINTS").write_text("mesh\n0\nGamma\n3 3 3\n0 0 0\n", encoding="utf-8")
    (root / "OUTCAR").write_text(
        "\n".join(
            [
                " NIONS =      3 ions",
                " volume of cell :      41.234",
                " free  energy   TOTEN  =      -25.125 eV",
                " Elapsed time (sec):  12.5",
                " General timing and accounting informations for this job:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_lammps_run(root: Path) -> None:
    root.mkdir()
    (root / "log.lammps").write_text(
        "\n".join(
            [
                "LAMMPS test",
                "Step Temp Press PotEng Volume",
                "0 300 1 -100 1000",
                "100 305 2 -99 1001",
                "Loop time of 1.0 on 1 procs for 100 steps",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_calphad_run(root: Path) -> None:
    root.mkdir()
    (root / "test.TDB").write_text(
        "ELEMENT U BLANK 0 0 0 !\nELEMENT O BLANK 0 0 0 !\nPHASE FLUORITE % 2 1 1 !\n",
        encoding="utf-8",
    )


def write_defect_cloud_run(root: Path) -> None:
    root.mkdir()
    (root / "defect_cloud_summary.json").write_text(
        json.dumps(
            {
                "schema": "atomi.vasp.defect_cloud.summary.v1",
                "n_seed_motifs": 2,
                "n_candidate_runs": 16,
                "per_motif_requested": 8,
                "seed": 20260518,
                "families_by_motif": {
                    "GdUO2_seed_01": {
                        "base": 1,
                        "random_displacement": 3,
                        "isotropic_strain": 2,
                        "species_biased_displacement": 1,
                        "mixed_displacement": 1,
                    },
                    "GdUO2_seed_02": {
                        "base": 1,
                        "random_displacement": 3,
                        "isotropic_strain": 2,
                        "species_biased_displacement": 1,
                        "mixed_displacement": 1,
                    },
                },
                "defaults": {
                    "random_amp_A": 0.02,
                    "structured_amp_A": 0.01,
                    "bias_species": "O",
                    "bias_amp_A": 0.05,
                    "mixed_amp_A": 0.04,
                    "iso_strains": [-0.01, 0.01],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "defect_cloud_index.csv").write_text(
        "\n".join(
            [
                "motif_id,family,run_dir",
                "GdUO2_seed_01,base,GdUO2_seed_01/base",
                "GdUO2_seed_01,random_displacement,GdUO2_seed_01/random_001",
                "GdUO2_seed_02,species_biased_displacement,GdUO2_seed_02/bias_O_001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "runlist.txt").write_text("GdUO2_seed_01/base\nGdUO2_seed_01/random_001\n", encoding="utf-8")


def test_paper_draft_scans_and_appends(tmp_path: Path) -> None:
    vasp = tmp_path / "vasp"
    lammps = tmp_path / "md"
    calphad = tmp_path / "calphad"
    write_vasp_run(vasp)
    write_lammps_run(lammps)
    write_calphad_run(calphad)
    document = tmp_path / "paper" / "working.md"
    evidence = tmp_path / "paper" / "evidence.json"

    paper_draft.main(
        [
            "--used",
            "DFT",
            "MD",
            "CALPHAD",
            "--run",
            str(vasp),
            "--run",
            str(lammps),
            "--run",
            str(calphad),
            "--document",
            str(document),
            "--evidence-json",
            str(evidence),
            "--title",
            "Methods seed",
            "--material",
            "UO2",
        ]
    )

    text = document.read_text(encoding="utf-8")
    assert "Methods seed" in text
    assert "Electronic-structure calculations" in text
    assert "ENCUT=520" in text
    assert "final DFT energy -25.125" in text
    assert "MD thermo summary" in text
    assert "phase_count=1" in text

    parsed = json.loads(evidence.read_text(encoding="utf-8"))
    assert parsed[0]["detected_modules"] == ["DFT"]
    assert "dft_outcar" in parsed[0]["facts"]


def test_paper_draft_describes_vasp_defect_candidate_generation(tmp_path: Path) -> None:
    prep = tmp_path / "defect_prep"
    write_defect_cloud_run(prep)
    document = tmp_path / "draft.md"
    evidence = tmp_path / "evidence.json"

    paper_draft.main(
        [
            "--used",
            "defect-cloud",
            "DFT",
            "--run",
            str(prep),
            "--document",
            str(document),
            "--evidence-json",
            str(evidence),
            "--mode",
            "overwrite",
            "--no-style-note",
            "--title",
            "Defect candidate preparation",
            "--material",
            "(Gd,U)O2-x",
        ]
    )

    text = document.read_text(encoding="utf-8")
    assert "Requested modules: VASP_PREP, DFT" in text
    assert "Defect-seed and candidate electronic-structure folders" in text
    assert "2 seed motifs" in text
    assert "16 candidate VASP folders" in text
    assert "bias_species=O" in text
    assert "array-run index runlist.txt with 2 entries" in text

    parsed = json.loads(evidence.read_text(encoding="utf-8"))
    assert parsed[0]["detected_modules"] == ["VASP_PREP"]
    assert parsed[0]["facts"]["defect_cloud_summary"]["family_totals"]["base"] == 2


def test_paper_draft_top_level_cli(tmp_path: Path) -> None:
    vasp = tmp_path / "vasp"
    write_vasp_run(vasp)
    document = tmp_path / "draft.md"

    atomi_main(
        [
            "paper-draft",
            "--used",
            "DFT",
            "--run",
            str(vasp),
            "--document",
            str(document),
            "--mode",
            "overwrite",
            "--no-style-note",
        ]
    )

    text = document.read_text(encoding="utf-8")
    assert "Requested modules: DFT" in text
    assert "Style Notes" not in text
    assert "UO2 test" in text


def test_normalize_modules_keeps_unknown_keyword() -> None:
    assert paper_draft.normalize_modules(["dft, mlip", "custom"]) == ["DFT", "MLIP", "CUSTOM"]


def test_normalize_modules_accepts_defect_cloud_alias() -> None:
    assert paper_draft.normalize_modules(["defect-cloud"]) == ["VASP_PREP"]
