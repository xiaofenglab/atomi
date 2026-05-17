from __future__ import annotations

import csv
import json
from pathlib import Path

from atomi.calphad.export import main


def test_calphad_export_writes_table_and_moose_template(tmp_path: Path) -> None:
    source = tmp_path / "free_energy.csv"
    source.write_text(
        "T_K,phase,composition,G_J_mol,mu_U_J_mol,Cp_J_molK\n"
        "300,FLUORITE,UO2,-1000,-400,65\n"
        "600,FLUORITE,UO2,-1200,-420,75\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    main(["--property-csv", str(source), "--outdir", str(out), "--material", "UO2", "--component", "U"])

    rows = list(csv.DictReader((out / "calphad_property_table.csv").open(encoding="utf-8")))
    assert rows[0]["phase"] == "FLUORITE"
    include = (out / "UO2_phase_field_free_energy.i").read_text(encoding="utf-8")
    assert "PiecewiseLinear" in include
    assert "GenericFunctionMaterial" in include
    metadata = json.loads((out / "calphad_export_metadata.json").read_text(encoding="utf-8"))
    assert "G_J_mol" in metadata["export_fields"]


def test_calphad_export_converts_qha_md_target_cell_to_per_formula(tmp_path: Path) -> None:
    source = tmp_path / "hybrid_cp_entropy.csv"
    source.write_text(
        "T_K,Cp,S_neel_corrected,H_neel_corrected_kJ_mol,G_neel_corrected_kJ_mol\n"
        "300,40,80,4,2\n",
        encoding="utf-8",
    )
    (tmp_path / "hybrid_cp_entropy_metadata.json").write_text(
        json.dumps(
            {
                "formula": "UO2",
                "basis": "target-cell",
                "cell_metadata": {
                    "formula": "UO2",
                    "n_formula_units": 32,
                    "target_z_formula_units": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    main(["--property-csv", str(source), "--outdir", str(out), "--material", "UO2"])

    rows = list(csv.DictReader((out / "calphad_property_table.csv").open(encoding="utf-8")))
    assert float(rows[0]["Cp_J_molK"]) == 10.0
    assert float(rows[0]["S_J_molK"]) == 20.0
    assert float(rows[0]["H_J_mol"]) == 1000.0
    assert float(rows[0]["G_J_mol"]) == 500.0
    metadata = json.loads((out / "calphad_export_metadata.json").read_text(encoding="utf-8"))
    assert metadata["unit_conversion"]["input_basis"] == "target-cell"
    assert metadata["unit_conversion"]["extensive_basis_factor"] == 0.25


def test_calphad_export_maps_thermo_lammps_grid_columns(tmp_path: Path) -> None:
    source = tmp_path / "thermo_functions_grid.csv"
    source.write_text(
        "T_K,Cp_used_for_integration_J_per_mol_UO2_K,S_rel_J_per_mol_UO2_K,H_rel_J_per_mol_UO2,G_rel_J_per_mol_UO2\n"
        "300,65,77,1200,900\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    main(
        [
            "--property-csv",
            str(source),
            "--outdir",
            str(out),
            "--material",
            "UO2",
            "--formula-units",
            "32",
            "--target-z",
            "4",
        ]
    )
    rows = list(csv.DictReader((out / "calphad_property_table.csv").open(encoding="utf-8")))
    assert float(rows[0]["Cp_J_molK"]) == 65.0
    assert float(rows[0]["S_J_molK"]) == 77.0
    assert float(rows[0]["H_J_mol"]) == 1200.0
    assert float(rows[0]["G_J_mol"]) == 900.0
