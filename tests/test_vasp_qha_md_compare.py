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

    write_qha_dat(qha / "volume-temperature.dat", [(300.0, 800.0), (500.0, 832.0)])
    write_qha_dat(qha / "gibbs-temperature.dat", [(300.0, 32.0), (500.0, 33.0)])
    write_qha_dat(qha / "entropy-temperature.dat", [(300.0, 0.0), (500.0, 0.0)])
    write_qha_dat(qha / "Cp-temperature.dat", [(300.0, 320.0), (500.0, 320.0)])
    (md / "thermo_functions_grid.csv").write_text(
        "T_K,V_fit_A3,Cp_used_for_integration_J_per_mol_UO2_K,"
        "S_rel_J_per_mol_UO2_K,G_rel_J_per_mol_UO2,H_rel_J_per_mol_UO2,"
        "alpha_V_micro_per_K\n"
        "300,800,10,10,0,0,20\n"
        "500,832,10,10,3015.166003853438,3015.166003853438,22\n",
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
    assert volume_rows[2]["source"] == "MD"
    assert float(volume_rows[2]["value"]) == 100.0

    cp_rows = list(csv.DictReader((out / "cp_qha_md_overlay.csv").open()))
    assert float(cp_rows[0]["value"]) == 10.0
    assert float(cp_rows[2]["value"]) == 10.0

    gibbs_rows = list(csv.DictReader((out / "gibbs_qha_md_overlay.csv").open()))
    assert float(gibbs_rows[0]["value"]) == 0.0
    assert float(gibbs_rows[1]["value"]) == pytest.approx(3.015166628853436)
    assert float(gibbs_rows[2]["value"]) == 0.0
    assert float(gibbs_rows[3]["value"]) == pytest.approx(3.015166003853438)

    enthalpy_rows = list(csv.DictReader((out / "enthalpy_qha_md_overlay.csv").open()))
    assert float(enthalpy_rows[1]["value"]) == pytest.approx(3.015166628853436)
    assert float(enthalpy_rows[3]["value"]) == pytest.approx(3.015166003853438)
    assert (out / "overlay_index.csv").exists()
    assert (out / "normalization_metadata.json").exists()
    assert (out / "availability_report.csv").exists()


def test_qha_md_compare_shifts_energy_at_minimal_overlap(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    qha = tmp_path / "qha"
    md = tmp_path / "md"
    out = tmp_path / "overlay"
    qha.mkdir()
    md.mkdir()

    write_qha_dat(qha / "gibbs-temperature.dat", [(0.0, 10.0), (300.0, 11.0)])
    (md / "thermo_functions_grid.csv").write_text(
        "T_K,G_rel_J_per_mol_UO2,H_rel_J_per_mol_UO2\n"
        "300,0,0\n"
        "500,1000,1000\n",
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
            "1",
            "--md-formula-units",
            "1",
            "--target-z",
            "1",
            "--t-min",
            "0",
            "--t-max",
            "500",
        ]
    )

    rows = list(csv.DictReader((out / "gibbs_qha_md_overlay.csv").open()))
    assert rows[0]["source"] == "QHA"
    assert float(rows[0]["value"]) == pytest.approx(-96.48533212331002)
    assert float(rows[1]["value"]) == pytest.approx(0.0)
    assert rows[2]["source"] == "MD"
    assert float(rows[2]["value"]) == pytest.approx(0.0)


def test_qha_md_compare_uses_md_column_aliases_and_interpolates_entropy(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    qha = tmp_path / "qha"
    md = tmp_path / "md"
    out = tmp_path / "overlay"
    qha.mkdir()
    md.mkdir()

    write_qha_dat(qha / "gibbs-temperature.dat", [(300.0, 10.0), (400.0, 11.0)])
    write_qha_dat(qha / "entropy-temperature.dat", [(300.0, 0.0), (500.0, 64.0)])
    (md / "thermo_functions_grid.csv").write_text(
        "T_K,S_rel_J_mol_K,H_rel_J_mol,G_rel_J_mol\n"
        "300,0,0,0\n"
        "400,32,1000,1000\n",
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
            "300",
            "--t-max",
            "400",
        ]
    )

    enthalpy_rows = list(csv.DictReader((out / "enthalpy_qha_md_overlay.csv").open()))
    assert len(enthalpy_rows) == 4
    assert float(enthalpy_rows[1]["value"]) > 0.0
    report = list(csv.DictReader((out / "availability_report.csv").open()))
    entropy_row = next(row for row in report if row["quantity"] == "entropy")
    assert entropy_row["md_column"] == "S_rel_J_mol_K"
    assert entropy_row["comparison_type"] == "overlay"
