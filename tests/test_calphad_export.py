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
