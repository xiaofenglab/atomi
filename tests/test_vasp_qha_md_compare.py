import csv
from pathlib import Path

import pytest

from atomi.vasp.qha_md_compare import main


def write_qha_dat(path: Path, rows: list[tuple[float, float]]) -> None:
    path.write_text(
        "\n".join(f"{temp} {value}" for temp, value in rows) + "\n",
        encoding="utf-8",
    )


def test_qha_md_compare_normalizes_to_target_cell_and_units(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    qha = tmp_path / "qha"
    md = tmp_path / "md"
    out = tmp_path / "overlay"
    qha.mkdir()
    md.mkdir()

    write_qha_dat(qha / "volume-temperature.dat", [(300.0, 800.0)])
    write_qha_dat(qha / "gibbs-temperature.dat", [(300.0, 32.0)])
    write_qha_dat(qha / "Cp-temperature.dat", [(300.0, 80.0)])
    (md / "thermo_functions_grid.csv").write_text(
        "T_K,V_fit_A3,Cp_used_for_integration_J_per_mol_UO2_K,"
        "S_rel_J_per_mol_UO2_K,G_rel_J_per_mol_UO2,H_rel_J_per_mol_UO2,"
        "alpha_V_micro_per_K\n"
        "300,800,80,40,96485.33212331002,1000,20\n",
        encoding="utf-8",
    )
    (md / "all_T_summary.csv").write_text(
        "T_K,KT_GPa_from_V_fluct\n300,200\n",
        encoding="utf-8",
    )

    main(
        [
            "--qha-dir",
            str(qha),
            "--md-dir",
            str(md),
            "--outdir",
            str(out),
            "--qha-formula-units",
            "32",
            "--md-formula-units",
            "32",
            "--target-z",
            "4",
            "--t-min",
            "0",
            "--t-max",
            "500",
        ]
    )

    volume_rows = list(csv.DictReader((out / "volume_qha_md_overlay.csv").open()))
    assert volume_rows[0]["source"] == "QHA"
    assert float(volume_rows[0]["value"]) == 100.0
    assert volume_rows[1]["source"] == "MD"
    assert float(volume_rows[1]["value"]) == 100.0

    gibbs_rows = list(csv.DictReader((out / "gibbs_qha_md_overlay.csv").open()))
    assert round(float(gibbs_rows[0]["value"]), 6) == round(96.48533212331002, 6)
    assert round(float(gibbs_rows[1]["value"]), 6) == round(96.48533212331002, 6)
    assert (out / "overlay_index.csv").exists()
