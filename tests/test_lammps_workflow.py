import csv
import json

import numpy as np
import pytest

import atomi.lammps.thermo_series as thermo_series
from atomi.lammps.workflow import effective_max_chunks, is_nvt_ramp_stage
from atomi.lammps.thermo_series import (
    build_combined_thermo,
    choose_qha_md_cp_switch,
    fill_missing_anchors_from_qha,
    integrate_qha_cp_anchor,
    qha_cp_thermo_curve,
)


def test_nvt_ramp_stage_is_limited_to_one_chunk() -> None:
    cfg = {"max_chunks_small": 5, "max_chunks_large": 8}
    stage = {
        "name": "lc_nvt_ramp_400K",
        "type": "nvt",
        "temperature_start": 300,
        "temperature_end": 400,
        "max_chunks": 4,
    }

    assert is_nvt_ramp_stage(stage)
    assert effective_max_chunks(cfg, stage) == 1


def test_fixed_temperature_stage_can_use_configured_chunks() -> None:
    cfg = {"max_chunks_small": 5, "max_chunks_large": 8}
    stage = {
        "name": "lc_nvt_eqm_400K",
        "type": "nvt",
        "temperature_start": 400,
        "temperature_end": 400,
        "max_chunks": 3,
    }

    assert not is_nvt_ramp_stage(stage)
    assert effective_max_chunks(cfg, stage) == 3


def test_fixed_step_stage_still_defaults_to_single_chunk() -> None:
    cfg = {"max_chunks_small": 5, "max_chunks_large": 8}
    stage = {
        "name": "lc_nvt_relax_400K",
        "type": "nvt",
        "temperature": 400,
        "steps": 1000,
    }

    assert effective_max_chunks(cfg, stage) == 1


def test_qha_cp_can_generate_thermo_anchor_values(tmp_path) -> None:
    qha = tmp_path / "qha"
    qha.mkdir()
    (qha / "Cp-temperature.dat").write_text(
        "0 0\n100 20\n300 60\n",
        encoding="utf-8",
    )

    anchor = integrate_qha_cp_anchor(
        qha_dir=qha,
        anchor_T=300.0,
        qha_formula_units=2.0,
        qha_cp_unit="J/mol-cell/K",
    )

    assert anchor["Cp_J_mol_formula_K"] == 30.0
    assert anchor["H_J_mol_formula"] == 4500.0
    assert anchor["S_J_mol_formula_K"] == 25.0


def test_qha_anchor_fills_only_missing_manual_values(tmp_path) -> None:
    qha = tmp_path / "qha"
    qha.mkdir()
    (qha / "Cp-temperature.dat").write_text(
        "0 0\n100 10\n300 30\n",
        encoding="utf-8",
    )

    T, S, Cp, H, metadata = fill_missing_anchors_from_qha(
        thermo_anchor_T=None,
        thermo_anchor_S_J_mol_K=99.0,
        thermo_anchor_Cp_J_mol_K=None,
        thermo_anchor_H_J_mol=None,
        qha_anchor_dir=qha,
        qha_anchor_formula_units=1.0,
        qha_anchor_cp_unit="J/mol-cell/K",
    )

    assert T == 300.0
    assert S == 99.0
    assert Cp == 30.0
    assert H == 4500.0
    assert "thermo_anchor_S_J_mol_K" not in metadata["filled_fields"]


def test_jaea_anchor_can_fill_lammps_enthalpy_anchor(monkeypatch) -> None:
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

    monkeypatch.setattr(thermo_series, "jaea_anchor", fake_jaea_anchor)
    anchor_t, anchor_h, metadata = thermo_series.fill_enthalpy_anchor_from_thermo_db(
        thermo_db="jaea",
        thermo_formula="UO2",
        thermo_phase="solid",
        thermo_db_temperature=None,
        thermo_anchor_T=None,
        thermo_anchor_H_J_mol=None,
        anchor_metadata=None,
    )

    assert anchor_t == 300.0
    assert anchor_h == -1084490.0
    assert metadata["thermo_db_filled_fields"] == ["thermo_anchor_H_J_mol"]
    assert metadata["thermo_db_anchor"]["formula"] == "UO2"

    anchor_t, anchor_h, metadata = thermo_series.fill_enthalpy_anchor_from_thermo_db(
        thermo_db="jaea",
        thermo_formula="UO2",
        thermo_phase="solid",
        thermo_db_temperature=None,
        thermo_anchor_T=300.0,
        thermo_anchor_H_J_mol=4500.0,
        anchor_metadata={"filled_fields": ["thermo_anchor_H_J_mol"]},
    )
    assert anchor_h == -1084490.0

    anchor_t, anchor_h, metadata = thermo_series.fill_enthalpy_anchor_from_thermo_db(
        thermo_db="jaea",
        thermo_formula="UO2",
        thermo_phase="solid",
        thermo_db_temperature=None,
        thermo_anchor_T=300.0,
        thermo_anchor_H_J_mol=123.0,
        anchor_metadata={"filled_fields": []},
    )
    assert anchor_h == 123.0


def test_qha_md_cp_switch_rejects_low_temperature_match() -> None:
    switch, method = choose_qha_md_cp_switch(
        qha_T=np.array([0.0, 40.0, 200.0]),
        qha_Cp=np.array([0.0, 10.0, 50.0]),
        md_T=np.array([0.0, 40.0, 200.0]),
        md_Cp=np.array([0.0, 10.0, 52.0]),
        minimum=50.0,
    )

    assert switch >= 50.0
    assert method == "overlap-closest-cp"


def test_qha_curve_derives_cubic_lattice_a_from_volume(tmp_path) -> None:
    qha = tmp_path / "qha"
    qha.mkdir()
    (qha / "Cp-temperature.dat").write_text("0 0\n100 20\n", encoding="utf-8")
    (qha / "volume-temperature.dat").write_text("0 256\n100 500\n", encoding="utf-8")

    curve = qha_cp_thermo_curve(qha, 4.0)
    a_param = curve["lattice_parameters"]["a"]

    assert a_param["source"] == "derived_from_volume_cubic_z4"
    assert np.isclose(a_param["values"][0], 256.0 ** (1.0 / 3.0))


def test_build_combined_thermo_can_use_qha_low_t_splice(tmp_path) -> None:
    qha = tmp_path / "qha"
    qha.mkdir()
    (qha / "Cp-temperature.dat").write_text(
        "0 0\n100 20\n200 40\n",
        encoding="utf-8",
    )
    (qha / "volume-temperature.dat").write_text(
        "0 99\n100 100\n200 101\n",
        encoding="utf-8",
    )
    (qha / "a-temperature.dat").write_text(
        "0 3.9\n100 4.0\n200 4.1\n",
        encoding="utf-8",
    )
    summaries = [
        {
            "target_T_K": 300.0,
            "n_formula_units": 1.0,
            "V_mean_A3": 100.0,
            "a_mean_A": 4.0,
            "density_mean_g_cm3": 10.0,
            "H_mean_eV_cell": 1.0,
            "Cp_fluct_J_per_mol_UO2_K": 100.0,
            "KT_GPa_from_V_fluct": 200.0,
        },
        {
            "target_T_K": 400.0,
            "n_formula_units": 1.0,
            "V_mean_A3": 101.0,
            "a_mean_A": 4.01,
            "density_mean_g_cm3": 9.9,
            "H_mean_eV_cell": 2.0,
            "Cp_fluct_J_per_mol_UO2_K": 120.0,
            "KT_GPa_from_V_fluct": 190.0,
        },
    ]

    build_combined_thermo(
        summaries=summaries,
        outdir=tmp_path / "out",
        plot_T_min=0,
        plot_T_max=400,
        plot_T_step=100,
        n_bootstrap=0,
        qha_low_t_curve=qha_cp_thermo_curve(qha, 1.0),
        structure_reference_temperature=200.0,
        volume_reference=104.0,
        lattice_references={"a": 4.2},
        structure_correction="shift",
        thermo_anchor_T=300.0,
        thermo_anchor_H_J_mol=-1084490.0,
        anchor_metadata={
            "thermo_db_anchor": {
                "database": "jaea",
                "formula": "UO2",
                "temperature_value_K": 300.0,
                "S_J_mol_formula_K": 77.8,
                "H_J_mol_formula": -1084490.0,
                "G_J_mol_formula": -1107840.0,
            }
        },
        plot_thermo_db_points=True,
    )

    rows = list(csv.DictReader((tmp_path / "out" / "thermo_functions_grid.csv").open()))
    by_t = {float(row["T_K"]): row for row in rows}
    assert float(by_t[100.0]["Cp_used_for_integration_J_per_mol_UO2_K"]) == 20.0
    assert float(by_t[200.0]["Cp_used_for_integration_J_per_mol_UO2_K"]) == 40.0
    assert float(by_t[300.0]["Cp_used_for_integration_J_per_mol_UO2_K"]) == 100.0
    assert float(by_t[100.0]["qha_md_blend_weight"]) == 0.0
    assert float(by_t[0.0]["S_rel_J_per_mol_UO2_K"]) == 0.0
    assert min(float(row["S_rel_J_per_mol_UO2_K"]) for row in rows) >= -1.0e-12
    assert float(by_t[300.0]["H_rel_J_per_mol_UO2"]) == -1084490.0
    assert float(by_t[300.0]["G_rel_J_per_mol_UO2"]) == pytest.approx(
        -1084490.0 - 300.0 * float(by_t[300.0]["S_rel_J_per_mol_UO2_K"])
    )
    assert (tmp_path / "out" / "qha_low_t_splice_metadata.json").exists()
    assert (tmp_path / "out" / "hybrid_Cp_QHA_MD.png").exists()
    assert (tmp_path / "out" / "hybrid_S_QHA_MD.png").exists()
    assert (tmp_path / "out" / "hybrid_H_QHA_MD.png").exists()
    assert (tmp_path / "out" / "hybrid_G_QHA_MD.png").exists()
    assert (tmp_path / "out" / "hybrid_V_QHA_MD.png").exists()
    assert (tmp_path / "out" / "hybrid_a_QHA_MD.png").exists()
    assert (tmp_path / "out" / "hybrid_alpha_V_QHA_MD.png").exists()
    assert (tmp_path / "out" / "hybrid_alpha_L_QHA_MD.png").exists()
    assert (tmp_path / "out" / "volume_QHA_MD_overlap.png").exists()
    assert (tmp_path / "out" / "lattice_a_QHA_MD_overlap.png").exists()
    metadata = json.loads((tmp_path / "out" / "qha_low_t_splice_metadata.json").read_text())
    assert metadata["blend_function"] == "smoothstep w=3x^2-2x^3"
    assert metadata["qha_volume_mode"] == "hybrid"
    assert metadata["qha_lattice_modes"]["a"] == "hybrid"
    assert metadata["structural_hybrid"]["correction_type"] == "shift"
    assert metadata["structural_hybrid"]["lattice_references"]["a"] == 4.2
    assert metadata["entropy_reference"]["S_J_mol_formula_K"] == 0.0
    assert metadata["enthalpy_anchor_shift"]["anchor_H_J_mol_formula"] == -1084490.0


def test_lammps_entropy_anchor_calibrates_qha_splice_blend_start() -> None:
    T_grid = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
    qha_T = np.array([0.0, 300.0, 600.0, 700.0])
    qha_Cp = np.array([0.0, 30.0, 40.0, 50.0])
    md_Cp = np.array([0.0, 250.0, 280.0, 300.0, 300.0, 300.0, 300.0, 300.0])

    blend_start, metadata = thermo_series.calibrate_blend_start_for_entropy_grid(
        T_grid,
        qha_T,
        qha_Cp,
        md_Cp,
        original_start=550.0,
        blend_end=650.0,
        entropy_temperature=300.0,
        entropy_target=28.0,
        minimum_start=200.0,
    )

    assert metadata["enabled"] is True
    assert blend_start < 550.0
    assert blend_start >= 200.0
    assert metadata["S_at_calibrated_blend_start_J_mol_formula_K"] == pytest.approx(28.0, abs=0.5)
