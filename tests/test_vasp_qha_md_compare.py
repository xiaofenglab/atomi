import csv
import json
from pathlib import Path

import pytest

import atomi.vasp.qha_md_compare as qha_md_compare
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


def test_qha_md_compare_writes_hybrid_cp_entropy(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    qha = tmp_path / "qha"
    md = tmp_path / "md"
    out = tmp_path / "overlay"
    qha.mkdir()
    md.mkdir()

    write_qha_dat(qha / "Cp-temperature.dat", [(300.0, 10.0), (500.0, 20.0)])
    write_qha_dat(qha / "entropy-temperature.dat", [(300.0, 1.0), (500.0, 8.0)])
    write_qha_dat(qha / "volume-temperature.dat", [(300.0, 100.0), (500.0, 110.0)])
    write_qha_dat(qha / "a-temperature.dat", [(300.0, 4.0), (500.0, 4.1)])
    (md / "thermo_functions_grid.csv").write_text(
        "T_K,Cp_used_for_integration_J_per_mol_UO2_K,S_rel_J_mol_K,V_fit_A3,a_fit_A\n"
        "500,22,9,111,4.11\n"
        "700,40,18,130,4.3\n",
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
            "300",
            "--t-max",
            "700",
            "--structure-reference-temperature",
            "500",
            "--lattice-reference",
            "a=4.2",
            "--volume-reference",
            "115",
            "--structure-correction",
            "shift",
        ]
    )

    rows = list(csv.DictReader((out / "hybrid_cp_entropy.csv").open()))
    assert [row["Cp_source"] for row in rows] == ["QHA", "blend", "MD"]
    assert float(rows[1]["T_K"]) == pytest.approx(500.0)
    assert float(rows[1]["Cp"]) == pytest.approx(22.0)
    assert "blend_weight" in rows[1]
    assert float(rows[2]["S_integrated"]) > float(rows[1]["S_integrated"])
    assert float(rows[2]["H_integrated_kJ_mol"]) > float(rows[1]["H_integrated_kJ_mol"])
    assert "G_relative_kJ_mol" in rows[0]
    metadata = json.loads((out / "hybrid_cp_entropy_metadata.json").read_text())
    assert metadata["switch_method"] == "overlap-closest-cp"
    assert metadata["switch_temperature_K"] == pytest.approx(500.0)
    assert metadata["blend_function"] == "smoothstep w=3x^2-2x^3"
    assert "cp_overlap_diagnostics" in metadata
    assert (out / "hybrid_Cp_QHA_MD.png").exists()
    assert (out / "hybrid_S_QHA_MD.png").exists()
    assert (out / "hybrid_H_QHA_MD.png").exists()
    assert (out / "hybrid_G_QHA_MD.png").exists()
    assert (out / "hybrid_V_QHA_MD.png").exists()
    assert (out / "hybrid_a_QHA_MD.png").exists()
    assert (out / "hybrid_alpha_V_QHA_MD.png").exists()
    assert (out / "hybrid_alpha_L_a_QHA_MD.png").exists()
    assert (out / "lattice_a_qha_md_overlay.png").exists()
    lattice_rows = list(csv.DictReader((out / "lattice_a_qha_md_overlay.csv").open()))
    assert {row["source"] for row in lattice_rows} == {"QHA", "MD"}
    assert (out / "overlap_mismatch_Cp.png").exists()
    assert metadata["structural_hybrid"]["correction_type"] == "shift"
    assert metadata["structural_hybrid"]["lattice_references"]["a"] == 4.2


def test_hybrid_cp_switch_ignores_extrapolated_low_t_md_grid(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    qha = tmp_path / "qha"
    md = tmp_path / "md"
    out = tmp_path / "overlay"
    qha.mkdir()
    md.mkdir()

    write_qha_dat(qha / "Cp-temperature.dat", [(0.0, 0.0), (300.0, 60.0), (600.0, 70.0)])
    write_qha_dat(qha / "entropy-temperature.dat", [(0.0, 0.0), (300.0, 20.0), (600.0, 30.0)])
    (md / "thermo_functions_grid.csv").write_text(
        "T_K,Cp_used_for_integration_J_per_mol_UO2_K,S_rel_J_mol_K\n"
        "0,0,0\n"
        "300,90,20\n"
        "600,72,30\n"
        "900,80,40\n",
        encoding="utf-8",
    )
    (md / "all_T_summary.csv").write_text(
        "target_T_K\n300\n600\n900\n",
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
            "900",
        ]
    )

    rows = list(csv.DictReader((out / "hybrid_cp_entropy.csv").open()))
    assert rows[0]["Cp_source"] == "QHA"
    assert float(rows[1]["T_K"]) == pytest.approx(300.0)
    assert rows[1]["Cp_source"] == "QHA"
    metadata = json.loads((out / "hybrid_cp_entropy_metadata.json").read_text())
    assert metadata["switch_method"] == "actual-md-overlap-closest-cp"
    assert metadata["switch_temperature_K"] == pytest.approx(600.0)


def test_qha_md_compare_derives_cubic_lattice_a_from_volume(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    qha = tmp_path / "qha"
    md = tmp_path / "md"
    out = tmp_path / "overlay"
    qha.mkdir()
    md.mkdir()

    write_qha_dat(qha / "Cp-temperature.dat", [(300.0, 10.0), (500.0, 20.0)])
    write_qha_dat(qha / "entropy-temperature.dat", [(300.0, 1.0), (500.0, 8.0)])
    write_qha_dat(qha / "volume-temperature.dat", [(300.0, 3200.0), (500.0, 3456.0)])
    (md / "thermo_functions_grid.csv").write_text(
        "T_K,Cp_used_for_integration_J_per_mol_UO2_K,S_rel_J_mol_K,V_fit_A3,a_fit_A\n"
        "500,22,9,432,4.2\n"
        "700,40,18,500,4.4\n",
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
            "4",
            "--target-z",
            "4",
            "--t-min",
            "300",
            "--t-max",
            "700",
        ]
    )

    lattice_rows = list(csv.DictReader((out / "lattice_a_qha_md_overlay.csv").open()))
    qha_rows = [row for row in lattice_rows if row["source"] == "QHA"]
    assert qha_rows
    assert float(qha_rows[0]["value"]) == pytest.approx(400.0 ** (1.0 / 3.0))
    metadata = json.loads((out / "hybrid_cp_entropy_metadata.json").read_text())
    assert (
        metadata["qha_file_paths"]["lattice_parameter_sources"]["a"]
        == "derived_from_volume_cubic"
    )
    assert metadata["structural_hybrid"]["lattice_parameters"]["a"]["qha_source"] == (
        "derived_from_volume_cubic"
    )


def test_qha_md_hybrid_entropy_is_zero_at_zero_kelvin(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    qha = tmp_path / "qha"
    md = tmp_path / "md"
    out = tmp_path / "overlay"
    qha.mkdir()
    md.mkdir()

    write_qha_dat(qha / "Cp-temperature.dat", [(0.0, 0.0), (300.0, 60.0), (600.0, 70.0)])
    write_qha_dat(qha / "entropy-temperature.dat", [(0.0, 0.0), (300.0, 20.0), (600.0, 30.0)])
    (md / "thermo_functions_grid.csv").write_text(
        "T_K,Cp_used_for_integration_J_per_mol_UO2_K,S_rel_J_mol_K\n"
        "0,0,0\n"
        "300,90,20\n"
        "600,72,30\n"
        "900,80,40\n",
        encoding="utf-8",
    )
    (md / "all_T_summary.csv").write_text(
        "target_T_K\n300\n600\n900\n",
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
            "900",
        ]
    )

    rows = list(csv.DictReader((out / "hybrid_cp_entropy.csv").open()))
    zero_row = next(row for row in rows if float(row["T_K"]) == pytest.approx(0.0))
    assert float(zero_row["S_integrated"]) == pytest.approx(0.0)
    assert min(float(row["S_integrated"]) for row in rows) >= -1.0e-12
    metadata = json.loads((out / "hybrid_cp_entropy_metadata.json").read_text())
    assert "S(0 K)=0" in metadata["entropy_reference_note"]


def test_qha_md_hybrid_enthalpy_anchor_shifts_gibbs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    qha = tmp_path / "qha"
    md = tmp_path / "md"
    out = tmp_path / "overlay"
    qha.mkdir()
    md.mkdir()

    write_qha_dat(qha / "Cp-temperature.dat", [(0.0, 0.0), (300.0, 60.0), (600.0, 70.0)])
    write_qha_dat(qha / "entropy-temperature.dat", [(0.0, 0.0), (300.0, 20.0), (600.0, 30.0)])
    (md / "thermo_functions_grid.csv").write_text(
        "T_K,Cp_used_for_integration_J_per_mol_UO2_K,S_rel_J_mol_K\n"
        "0,0,0\n"
        "300,60,20\n"
        "600,70,30\n",
        encoding="utf-8",
    )
    (md / "all_T_summary.csv").write_text("target_T_K\n300\n600\n", encoding="utf-8")

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
            "600",
            "--enthalpy-anchor-temperature",
            "300",
            "--enthalpy-anchor-value",
            "-1084.49",
            "--enthalpy-anchor-unit",
            "kJ/mol-formula",
        ]
    )

    rows = list(csv.DictReader((out / "hybrid_cp_entropy.csv").open()))
    by_t = {float(row["T_K"]): row for row in rows}
    h_300 = float(by_t[300.0]["H_integrated_kJ_mol"])
    s_300 = float(by_t[300.0]["S_integrated"])
    g_300 = float(by_t[300.0]["G_integrated_kJ_mol"])
    assert h_300 == pytest.approx(-1084.49)
    assert g_300 == pytest.approx(h_300 - 300.0 * s_300 / 1000.0)
    metadata = json.loads((out / "hybrid_cp_entropy_metadata.json").read_text())
    assert metadata["enthalpy_anchor"]["value_kJ_mol_basis"] == pytest.approx(-1084.49)


def test_qha_md_can_fill_enthalpy_anchor_from_jaea(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    qha = tmp_path / "qha"
    md = tmp_path / "md"
    out = tmp_path / "overlay"
    qha.mkdir()
    md.mkdir()

    write_qha_dat(qha / "Cp-temperature.dat", [(0.0, 0.0), (300.0, 60.0), (600.0, 70.0)])
    write_qha_dat(qha / "entropy-temperature.dat", [(0.0, 0.0), (300.0, 20.0), (600.0, 30.0)])
    (md / "thermo_functions_grid.csv").write_text(
        "T_K,Cp_used_for_integration_J_per_mol_UO2_K,S_rel_J_mol_K\n"
        "0,0,0\n"
        "300,60,20\n"
        "600,70,30\n",
        encoding="utf-8",
    )
    (md / "all_T_summary.csv").write_text("target_T_K\n300\n600\n", encoding="utf-8")

    def fake_jaea_anchor(formula, temperature, *, phase="solid"):
        return {
            "database": "jaea",
            "formula": formula,
            "phase": phase,
            "temperature_value_K": temperature,
            "H_J_mol_formula": -1084490.0,
            "S_J_mol_formula_K": 77.8,
            "G_J_mol_formula": -1107840.0,
            "Cp_J_mol_formula_K": 64.0,
            "url": "https://example.test/UO2.html",
        }

    monkeypatch.setattr(qha_md_compare, "jaea_anchor", fake_jaea_anchor)
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
            "600",
            "--thermo-db",
            "jaea",
            "--thermo-formula",
            "UO2",
            "--thermo-phase",
            "solid",
        ]
    )

    rows = list(csv.DictReader((out / "hybrid_cp_entropy.csv").open()))
    by_t = {float(row["T_K"]): row for row in rows}
    assert float(by_t[300.0]["H_integrated_kJ_mol"]) == pytest.approx(-1084.49)
    metadata = json.loads((out / "hybrid_cp_entropy_metadata.json").read_text())
    assert metadata["enthalpy_anchor"]["thermo_db_anchor"]["database"] == "jaea"
    assert metadata["enthalpy_anchor"]["thermo_db_anchor"]["formula"] == "UO2"


def test_hybrid_cp_switch_rejects_extreme_low_t_match(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    qha = tmp_path / "qha"
    md = tmp_path / "md"
    out = tmp_path / "overlay"
    qha.mkdir()
    md.mkdir()

    write_qha_dat(qha / "Cp-temperature.dat", [(0.0, 0.0), (40.0, 10.0), (200.0, 50.0)])
    write_qha_dat(qha / "entropy-temperature.dat", [(0.0, 0.0), (40.0, 3.0), (200.0, 20.0)])
    (md / "thermo_functions_grid.csv").write_text(
        "T_K,Cp_used_for_integration_J_per_mol_UO2_K,S_rel_J_mol_K\n"
        "0,0,0\n"
        "40,10,3\n"
        "100,30,10\n"
        "200,52,20\n",
        encoding="utf-8",
    )
    (md / "all_T_summary.csv").write_text("target_T_K\n40\n100\n200\n", encoding="utf-8")

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
            "200",
        ]
    )

    metadata = json.loads((out / "hybrid_cp_entropy_metadata.json").read_text())
    assert metadata["switch_temperature_K"] >= 50.0
    assert metadata["minimum_switch_temperature_K"] == 50.0
