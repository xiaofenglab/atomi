from __future__ import annotations

import csv
import json
from pathlib import Path

from ase.io import read

from atomi.zentropy import motif_db


def write_poscar(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Gd U O defect motif",
                "1.0",
                "4.0 0.0 0.0",
                "0.0 4.0 0.0",
                "0.0 0.0 4.0",
                "Gd U O",
                "1 1 4",
                "Direct",
                "0.00 0.00 0.00",
                "0.50 0.50 0.50",
                "0.25 0.25 0.25",
                "0.75 0.75 0.75",
                "0.25 0.75 0.25",
                "0.75 0.25 0.75",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_outcar(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                " NIONS =      6 ions",
                " free  energy   TOTEN  =       -100.000000 eV",
                " volume of cell :       64.000000",
                " magnetization (x)",
                " # of ion      s       p       d       tot",
                " ----------------------------------------",
                " 1             0.0     0.0     0.0     7.1",
                " 2             0.0     0.0     0.0     2.2",
                " 3             0.0     0.0     0.0     0.0",
                " 4             0.0     0.0     0.0     0.0",
                " 5             0.0     0.0     0.0     0.0",
                " 6             0.0     0.0     0.0     0.0",
                " tot           0.0     0.0     0.0     9.3",
                "",
            ]
        ),
        encoding="utf-8",
    )


def make_run(tmp_path: Path) -> Path:
    run = tmp_path / "gd_uo2_motif"
    run.mkdir()
    write_poscar(run / "CONTCAR")
    write_outcar(run / "OUTCAR")
    return run


def test_motif_db_indexes_size_normalized_defect_motif(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "run,motif_id,motif_family,motif_type,defect_label,tags,degeneracy\n"
        "gd_uo2_motif,gd_uo2_test,gadolinium_defect,Gd_on_U,Gd dilute,gd;dilute,2\n",
        encoding="utf-8",
    )
    site_states = tmp_path / "site_states.csv"
    site_states.write_text(
        "motif_id,atom_index_1based,element,valence,spin_label,role\n"
        "gd_uo2_test,2,U,3,U3+,charge_compensation\n",
        encoding="utf-8",
    )
    db = tmp_path / "defect_motif_db.json"
    index = tmp_path / "defect_motif_index.csv"

    motif_db.main(
        [
            "index",
            "--run",
            str(run),
            "--db",
            str(db),
            "--csv",
            str(index),
            "--metadata-csv",
            str(metadata),
            "--site-state-csv",
            str(site_states),
        ]
    )

    payload = json.loads(db.read_text(encoding="utf-8"))
    record = payload["records"][0]
    norm = record["size_normalization"]
    assert record["motif_id"] == "gd_uo2_test"
    assert norm["formula_units"] == 2.0
    assert norm["guest_cation_fraction"] == 0.5
    assert norm["oxygen_delta_per_formula_unit"] == 0.0
    assert record["energy_per_formula_unit_eV"] == -50.0
    assert record["magmom"]["by_element"]["Gd"]["mean"] == 7.1
    assert record["site_states"][0]["spin_label"] == "U3+"
    assert "energy_per_formula_unit_eV" in index.read_text(encoding="utf-8")


def test_motif_db_imports_magit_spin_index_as_site_states(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    spin_index = tmp_path / "spin_index.csv"
    moments = [
        {"atom": 1, "element": "Gd", "magmom": 7.0},
        {"atom": 2, "element": "U", "magmom": -2.0},
        {"atom": 3, "element": "O", "magmom": 0.0},
    ]
    with spin_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run_dir", "name", "dopant_mode", "host_mode", "moments_by_atom"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_dir": str(run.resolve()),
                "name": "spin_001",
                "dopant_mode": "all",
                "host_mode": "afm",
                "moments_by_atom": json.dumps(moments),
            }
        )
    db = tmp_path / "defect_motif_db.json"

    motif_db.main(
        [
            "index",
            "--run",
            str(run),
            "--db",
            str(db),
            "--csv",
            str(tmp_path / "index.csv"),
            "--spin-index",
            str(spin_index),
            "--moment-state",
            "Gd:7=Gd3+",
            "--moment-state",
            "U:2=U4+",
        ]
    )

    record = json.loads(db.read_text(encoding="utf-8"))["records"][0]
    assert record["motif_id"] == "spin_001"
    assert record["motif_family"] == "magit_spin_variant"
    assert "magit" in record["tags"]
    assert record["site_states"][0]["spin_label"] == "Gd3+_up"
    assert record["site_states"][1]["spin_label"] == "U4+_down"


def test_motif_db_exports_repeated_mlip_poscar_and_magmom(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    db = tmp_path / "defect_motif_db.json"
    motif_db.main(["index", "--run", str(run), "--db", str(db), "--csv", str(tmp_path / "i.csv")])
    out = tmp_path / "mlip_inputs"

    motif_db.main(["export-mlip", "--db", str(db), "--outdir", str(out), "--repeat", "2", "1", "1"])

    motif_dir = out / "gd_uo2_motif"
    atoms = read(motif_dir / "POSCAR")
    assert len(atoms) == 12
    magmom = (motif_dir / "INCAR.magmom").read_text(encoding="utf-8")
    assert magmom.startswith("MAGMOM =")
    assert len(magmom.split("=")[1].split()) == 12
    manifest = (out / "mlip_export_manifest.csv").read_text(encoding="utf-8")
    assert "gd_uo2_motif" in manifest
