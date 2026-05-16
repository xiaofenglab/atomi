from __future__ import annotations

import csv
import json
from pathlib import Path

from atomi.vasp import elastic


def write_poscar(path: Path, scale: float = 5.0) -> None:
    path.write_text(
        "\n".join(
            [
                "UO2 cubic",
                "1.0",
                f"{scale} 0.0 0.0",
                f"0.0 {scale} 0.0",
                f"0.0 0.0 {scale}",
                "U O",
                "1 2",
                "Direct",
                "0.0 0.0 0.0",
                "0.25 0.25 0.25",
                "0.75 0.75 0.75",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_outcar(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                " SYMMETRIZED ELASTIC MODULI (kBar)",
                " Direction      XX        YY        ZZ        XY        YZ        ZX",
                " -------------------------------------------------------------------",
                " XX        3000.0    1000.0    1000.0       0.0       0.0       0.0",
                " YY        1000.0    3000.0    1000.0       0.0       0.0       0.0",
                " ZZ        1000.0    1000.0    3000.0       0.0       0.0       0.0",
                " XY           0.0       0.0       0.0     800.0       0.0       0.0",
                " YZ           0.0       0.0       0.0       0.0     800.0       0.0",
                " ZX           0.0       0.0       0.0       0.0       0.0     800.0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_vasp_elastic_prepare_from_volume_folders(tmp_path: Path) -> None:
    volume = tmp_path / "V1.000"
    volume.mkdir()
    write_poscar(volume / "CONTCAR")
    template = tmp_path / "template"
    template.mkdir()
    (template / "INCAR").write_text("ENCUT = 500\nIBRION = 2\n", encoding="utf-8")
    outdir = tmp_path / "elastic_vasp"

    elastic.main(
        [
            "prepare",
            "--volume-folder",
            str(volume),
            "--template",
            str(template),
            "--outdir",
            str(outdir),
        ]
    )

    incar = next(outdir.glob("*/INCAR")).read_text(encoding="utf-8")
    assert "IBRION = 6" in incar
    assert "ISIF = 3" in incar
    assert "NSW = 1" in incar
    manifest = list(csv.DictReader((outdir / "vasp_elastic_manifest.csv").open(encoding="utf-8")))
    assert manifest[0]["volume_scale"] == "1.0"


def test_vasp_elastic_analyze_parses_gpa_and_moduli(tmp_path: Path) -> None:
    run = tmp_path / "elastic_V1.000"
    run.mkdir()
    write_poscar(run / "POSCAR")
    write_outcar(run / "OUTCAR")
    outdir = tmp_path / "analysis"

    elastic.main(["analyze", "--run", str(run), "--outdir", str(outdir), "--symmetry", "cubic"])

    rows = list(csv.DictReader((outdir / "elastic_moduli_T.csv").open(encoding="utf-8")))
    assert float(rows[0]["C11_GPa"]) == 300.0
    assert float(rows[0]["C12_GPa"]) == 100.0
    assert float(rows[0]["C44_GPa"]) == 80.0
    assert float(rows[0]["K_H_GPa"]) > 0.0
    tensors = json.loads((outdir / "elastic_tensors.json").read_text(encoding="utf-8"))
    assert tensors["tensors"][0]["parser"]["input_unit"] == "kBar"
