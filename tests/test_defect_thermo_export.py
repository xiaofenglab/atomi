from __future__ import annotations

import csv
import json
from pathlib import Path

from atomi.zentropy.defect_thermo import main


def test_defect_thermo_export_keeps_energy_volume_and_spin_labels(tmp_path: Path) -> None:
    source = tmp_path / "defects.csv"
    source.write_text(
        "defect_id,composition,charge,formation_energy_eV,migration_barrier_eV,volume_change_A3,magmom_label,prefactor_cm2_s\n"
        "GdU_VO_nn,Gd0.03125U0.96875O1.984375,2,1.5,0.8,3.2,Gd3+_U5+,0.01\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    main(["--defect-csv", str(source), "--outdir", str(out), "--material", "Gd_U_O2"])

    rows = list(csv.DictReader((out / "defect_thermo_table.csv").open(encoding="utf-8")))
    assert rows[0]["magmom_label"] == "Gd3+_U5+"
    assert float(rows[0]["formation_energy_kJ_mol"]) > 140.0
    assert float(rows[0]["defect_volume_m3_mol"]) > 0.0
    species = list(csv.DictReader((out / "calphad_defect_species.csv").open(encoding="utf-8")))
    assert species[0]["defect_id"] == "GdU_VO_nn"
    assert "GenericConstantMaterial" in (out / "Gd_U_O2_defect_thermo.i").read_text(encoding="utf-8")
    metadata = json.loads((out / "defect_thermo_metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_defects"] == 1
