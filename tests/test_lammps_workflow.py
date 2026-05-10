import csv
import json

import numpy as np

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
    )

    rows = list(csv.DictReader((tmp_path / "out" / "thermo_functions_grid.csv").open()))
    by_t = {float(row["T_K"]): row for row in rows}
    assert float(by_t[100.0]["Cp_used_for_integration_J_per_mol_UO2_K"]) == 20.0
    assert float(by_t[200.0]["Cp_used_for_integration_J_per_mol_UO2_K"]) == 40.0
    assert float(by_t[300.0]["Cp_used_for_integration_J_per_mol_UO2_K"]) == 100.0
    assert float(by_t[100.0]["qha_md_blend_weight"]) == 0.0
    assert float(by_t[0.0]["S_rel_J_per_mol_UO2_K"]) == 0.0
    assert min(float(row["S_rel_J_per_mol_UO2_K"]) for row in rows) >= -1.0e-12
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
