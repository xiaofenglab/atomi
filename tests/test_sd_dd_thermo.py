from __future__ import annotations

import csv
import json
from pathlib import Path

from atomi.zentropy import sd_dd_thermo


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_sd_dd_thermo_writes_populations_summary_and_cef_seed(tmp_path: Path) -> None:
    defects = tmp_path / "defects.csv"
    defects.write_text(
        "defect_id,model,formation_energy_eV,degeneracy,capacity_per_formula,charge,delta_O,sublattice,site_species\n"
        "Gd_U,SD,1.0,2,1.0,-1,0,cation,Gd_U\n"
        "V_O,SD,2.0,1,2.0,2,-1,anion,V_O\n",
        encoding="utf-8",
    )
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "pair_id,defect_a,defect_b,binding_energy_eV,capacity_per_formula\n"
        "GdU_VO_pair,Gd_U,V_O,-0.4,1.0\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "out"

    sd_dd_thermo.main(
        [
            "--defect-csv",
            str(defects),
            "--pair-csv",
            str(pairs),
            "--outdir",
            str(outdir),
            "--temperature",
            "600,1200",
            "--chemical-potential",
            "O=-5.0",
            "--material",
            "Gd_UO2",
            "--formula",
            "UO2",
        ]
    )

    pop = rows(outdir / "sd_dd_defect_populations.csv")
    pair_rows = [row for row in pop if row["defect_id"] == "GdU_VO_pair"]
    assert pair_rows
    assert float(pair_rows[0]["formation_energy_eV"]) == 2.6
    summary = rows(outdir / "sd_dd_summary.csv")
    assert float(summary[1]["single_defect_concentration_per_formula"]) > float(
        summary[0]["single_defect_concentration_per_formula"]
    )
    cef = rows(outdir / "sd_dd_cef_seed.csv")
    assert cef[0]["cef_role"] == "seed_site_fraction_or_endmember_energy_for_future_CEF_assessment"
    metadata = json.loads((outdir / "sd_dd_metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == sd_dd_thermo.SCHEMA


def test_sd_dd_thermo_cli_json_reports_outputs(tmp_path: Path, capsys) -> None:
    defects = tmp_path / "defects.csv"
    defects.write_text(
        "defect_id,formation_energy_eV,degeneracy\n"
        "V_O,1.5,1\n",
        encoding="utf-8",
    )

    sd_dd_thermo.main(["--defect-csv", str(defects), "--outdir", str(tmp_path / "out"), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert "populations" in payload["outputs"]
