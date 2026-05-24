import csv
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

import atomi.lammps.thermo_series as thermo_series
from atomi.lammps import production_array
from atomi.lammps.workflow import (
    create_stage_wrapper,
    check_not_exploded_for_max_chunk,
    effective_max_chunks,
    lammps_wrapper_text,
    is_npt_equilibration_stage,
    is_nvt_ramp_stage,
    warn_if_ramp_max_chunks_ignored,
)
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


def test_nvt_ramp_stage_warns_when_configured_chunks_are_ignored(capsys) -> None:
    stage = {
        "name": "lc_nvt_ramp_100K",
        "type": "nvt",
        "temperature_start": 50,
        "temperature_end": 100,
        "max_chunks": 3,
    }

    warn_if_ramp_max_chunks_ignored(stage, max_chunks=1)

    captured = capsys.readouterr()
    assert "ignoring configured max_chunks=3" in captured.out
    assert "using max_chunks=1" in captured.out


def test_uploaded_large_cell_ramp_pattern_is_effectively_one_chunk() -> None:
    cfg = {"max_chunks_small": 5, "max_chunks_large": 8}
    stages = [
        {"name": "lc_nvt_ramp_100K", "type": "nvt", "temperature_start": 50, "temperature_end": 100, "max_chunks": 3},
        {"name": "lc_nvt_ramp_150K", "type": "nvt", "temperature_start": 100, "temperature_end": 150, "max_chunks": 3},
        {"name": "lc_nvt_ramp_300K", "type": "nvt", "temperature_start": 250, "temperature_end": 300, "max_chunks": 3},
        {"name": "lc_nvt_ramp_700K", "type": "nvt", "temperature_start": 600, "temperature_end": 700, "max_chunks": 1},
    ]

    assert all(is_nvt_ramp_stage(stage) for stage in stages)
    assert [effective_max_chunks(cfg, stage) for stage in stages] == [1, 1, 1, 1]


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


def test_npt_stage_honors_global_max_chunks() -> None:
    cfg = {"max_chunks": 3, "max_chunks_small": 5, "max_chunks_large": 8}
    stage = {
        "name": "lc_npt_eqm_400K",
        "type": "npt",
        "temperature": 400,
    }

    assert effective_max_chunks(cfg, stage) == 3


def test_npt_eqm_stage_defaults_to_three_chunks_not_large_cell_limit() -> None:
    cfg = {"max_chunks_small": 5, "max_chunks_large": 8}
    stage = {
        "name": "lc_npt_eqm_1200K",
        "type": "npt",
        "temperature": 1200,
        "large_cell": True,
    }

    assert is_npt_equilibration_stage(stage)
    assert effective_max_chunks(cfg, stage) == 3


def test_npt_eqm_stage_can_use_named_configured_limit() -> None:
    cfg = {"max_chunks_small": 5, "max_chunks_large": 8, "max_chunks_npt_eqm": 2}
    stage = {
        "name": "lc_npt_eqm_1200K",
        "type": "npt",
        "temperature": 1200,
        "large_cell": True,
    }

    assert effective_max_chunks(cfg, stage) == 2


def test_npt_eqm_stage_explicit_stage_limit_still_wins() -> None:
    cfg = {"max_chunks_small": 5, "max_chunks_large": 8, "max_chunks_npt_eqm": 2}
    stage = {
        "name": "lc_npt_eqm_1200K",
        "type": "npt",
        "temperature": 1200,
        "large_cell": True,
        "max_chunks": 4,
    }

    assert effective_max_chunks(cfg, stage) == 4


def test_final_npt_chunk_can_force_pass_when_not_exploded() -> None:
    cfg = {
        "equilibrium_rules": {
            "temperature_tol_fraction": 0.1,
            "temperature_tol_min": 10.0,
            "volume_slope_tol": 1e-12,
            "energy_slope_tol": 1e-12,
            "runaway_temperature_factor": 3.0,
            "stable_temperature_tol_fraction": 0.25,
            "stable_temperature_tol_min": 20.0,
            "stable_volume_slope_tol": 1e-12,
            "stable_energy_slope_tol": 1e-12,
            "stable_runaway_temperature_factor": 3.0,
        }
    }
    stage = {"name": "lc_npt_eqm_400K", "type": "npt", "temperature": 400}
    steps = list(range(20))
    T = [398.0 + 0.1 * i for i in range(20)]
    P = [1000.0 - 10.0 * i for i in range(20)]
    V = [1000.0 + 0.01 * i for i in range(20)]
    PE = [-100.0 + 0.001 * i for i in range(20)]

    ok, msg = check_not_exploded_for_max_chunk(cfg, stage, steps, T, P, V, PE)

    assert ok
    assert "not exploded" in msg


def test_fixed_step_stage_still_defaults_to_single_chunk() -> None:
    cfg = {"max_chunks_small": 5, "max_chunks_large": 8}
    stage = {
        "name": "lc_nvt_relax_400K",
        "type": "nvt",
        "temperature": 400,
        "steps": 1000,
    }

    assert effective_max_chunks(cfg, stage) == 1


def test_stage_wrapper_rewrites_sbatch_resources_from_environment(tmp_path, monkeypatch) -> None:
    template = tmp_path / "run_lammps_gpu.sh"
    template.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "##SBATCH --partition=old",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                "##SBATCH --gres=gpu:1",
                "#SBATCH --cpus-per-task=1",
                "#SBATCH --mem-per-cpu=3500M",
                "#SBATCH --time=1:00:00",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ATOMI_LAMMPS_PARTITION", "gpu")
    monkeypatch.setenv("ATOMI_LAMMPS_GRES", "gpu:1")
    monkeypatch.setenv("ATOMI_LAMMPS_CPUS_PER_TASK", "4")
    chunk_dir = tmp_path / "chunk_01"
    chunk_dir.mkdir()

    wrapper = create_stage_wrapper(
        {"wrapper_script": str(template)},
        chunk_dir,
        "05:36:00",
    )
    text = wrapper.read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --time=05:36:00" in text


def test_lammps_wrapper_fail_fast_when_gk_exe_missing() -> None:
    template = (
        Path("src")
        / "atomi"
        / "templates"
        / "lammps_workflow"
        / "run_lammps_gpu.sh"
    ).read_text(encoding="utf-8")

    assert "GK_REQUESTED=0" in template
    assert "ATOMI_LMP_GK_EXE is not set" in template
    assert 'ATOMI_LMP_EXE="${ATOMI_LMP_GK_EXE}"' in template
    assert 'confighpc --dir "$ATOMI_HPC_DIR" --no-env-var --shell' in template
    assert 'confighpc --config "$ATOMI_HPC_CONFIG" --shell' in template
    assert "*.local.json" in template
    assert "atomi_hpc_env.sh" in template
    assert "ATOMI_LMP_INSTALL_DIR" in template
    assert 'ATOMI_LMP_INSTALL_DIR/lib64' in template
    assert "ATOMI_LAMMPS_PYTHONPATH" in template
    assert "LAMMPS_PYTHONPATH" in template
    assert "PYTHON_LIBDIRS" in template
    assert "ATOMI_DETECTED_PYTHON_LIBDIRS" in template
    assert "ATOMI_PYTHON_LIBDIRS" in template
    assert "ATOMI_TORCH_LIBDIRS" in template
    assert "ATOMI_DETECTED_TORCH_LIBDIRS" in template
    assert "TORCH_LIBDIRS" in template
    assert "MACE_ALLOW_CPU" in template
    assert "LDLIBRARY" in template
    assert "TORCH_SHOW_CPP_STACKTRACES" in template
    assert "ATOMI_MLIP_MODEL_PATH" in template
    assert "torch.jit.load model" in template
    assert "forward_exchange API" in template
    assert "/src/lammps/python" in template
    assert "/build_mliap/cython" in template
    assert "source \"$ATOMI_LAMMPS_ENV/bin/activate\"" in template
    assert "source \"$ATOMI_LAMMPS_GK_ENV/bin/activate\"" in template
    assert "LAMMPS_GK_ENV" in template
    assert "mliap_unified_couple" in template
    assert "Atomi GK/ML-IAP preflight: PASS" in template
    assert "selected GK executable does not expose the ML-IAP mliap pair style" in template
    assert "ML-IAP model file not found" in template
    assert "required ML-IAP Python modules could not be imported" in template
    assert 'required = ("lammps", "lammps.mliap", "torch", "cupy")' in template
    assert 'optional = ("mliap_unified_couple", "mace")' in template


def test_mliap_config_refreshes_stale_lammps_wrapper(tmp_path) -> None:
    stale = tmp_path / "run_lammps_gpu.sh"
    stale.write_text(
        "#!/bin/bash\n"
        "#SBATCH --time=01:00:00\n"
        'if [ -n "${ATOMI_LMP_GK_EXE:-}" ]; then\n'
        '  ATOMI_LMP_EXE="${ATOMI_LMP_GK_EXE}"\n'
        "fi\n",
        encoding="utf-8",
    )

    text = lammps_wrapper_text(
        {
            "wrapper_script": str(stale),
            "pair_style_backend": "mliap",
            "runtime_profile": "lammps_gk_mliap",
        }
    )

    assert "GK_REQUESTED=0" in text
    assert 'confighpc --dir "$ATOMI_HPC_DIR" --no-env-var --shell' in text
    assert "ATOMI_LMP_INSTALL_DIR" in text
    assert "ATOMI_LAMMPS_PYTHONPATH" in text
    assert "/build_mliap/cython" in text
    assert "Atomi GK/ML-IAP preflight: PASS" in text


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


def test_lammps_neel_correction_grid_updates_s_h_g() -> None:
    T_grid = np.array([0.0, 30.8, 50.0, 300.0])
    S_grid = np.array([0.0, 2.0, 5.0, 70.0])
    H_grid = np.array([0.0, 10.0, 20.0, 1000.0])

    S_corr, H_corr, G_corr, weights, metadata = thermo_series.apply_neel_correction_grid(
        T_grid=T_grid,
        S_grid=S_grid,
        H_grid=H_grid,
        neel_correction="on",
        neel_t=30.8,
        neel_entropy=8.4,
        neel_enthalpy="auto",
        neel_apply_above_t=50.0,
        entropy_anchor_is_direct=False,
        anchor_metadata={},
        thermo_formula="UO2",
    )

    assert metadata["applied"] is True
    assert weights[-1] == pytest.approx(1.0)
    assert S_corr[-1] == pytest.approx(78.4)
    assert H_corr[-1] == pytest.approx(1258.72)
    assert G_corr[-1] == pytest.approx(1258.72 - 300.0 * 78.4)
    assert metadata["delta_S_gap_before_J_mol_formula_K"] == pytest.approx(7.81270)
    assert abs(metadata["delta_S_gap_after_J_mol_formula_K"]) < 1.0

    S_skip, H_skip, _G_skip, _weights_skip, skipped = thermo_series.apply_neel_correction_grid(
        T_grid=T_grid,
        S_grid=S_grid,
        H_grid=H_grid,
        neel_correction="on",
        neel_t=30.8,
        neel_entropy=8.4,
        neel_enthalpy="auto",
        neel_apply_above_t=50.0,
        entropy_anchor_is_direct=True,
        anchor_metadata={},
        thermo_formula="UO2",
    )
    assert skipped["applied"] is False
    assert np.allclose(S_skip, S_grid)
    assert np.allclose(H_skip, H_grid)


def test_lammps_direct_entropy_anchor_shifts_s_and_recomputes_g() -> None:
    T_grid = np.array([0.0, 300.0])
    S_grid = np.array([0.0, 70.0])
    H_grid = np.array([0.0, 1000.0])

    S_out, G_out, metadata = thermo_series.apply_entropy_anchor_grid(
        T_grid=T_grid,
        S_grid=S_grid,
        H_grid=H_grid,
        entropy_anchor_T=300.0,
        entropy_anchor_S=77.81270,
        source="jaea",
    )

    assert metadata["applied"] is True
    assert S_out[-1] == pytest.approx(77.81270)
    assert G_out[-1] == pytest.approx(1000.0 - 300.0 * 77.81270)


def test_lammps_neel_adjusted_entropy_benchmark_subtracts_neel_entropy() -> None:
    db_anchor = {
        "database": "jaea",
        "formula": "UO2",
        "temperature_value_K": 300.0,
        "S_J_mol_formula_K": 77.81270,
    }

    target, metadata = thermo_series.neel_adjusted_entropy_benchmark(
        db_anchor,
        neel_t=30.8,
        neel_entropy=8.4,
        neel_apply_above_t=50.0,
    )

    assert target == pytest.approx(77.81270 - 8.4)
    assert metadata["used_as_entropy_anchor"] is False
    assert metadata["used_for_blend_calibration"] is True


def test_lammps_md_root_discovery_uses_npt_and_ignores_nvt(tmp_path) -> None:
    npt_chunk = tmp_path / "stages" / "npt_prod_300K" / "chunk_production"
    nvt_chunk = tmp_path / "stages" / "nvt_ramp_300K" / "chunk_01"
    npt_chunk.mkdir(parents=True)
    nvt_chunk.mkdir(parents=True)
    (npt_chunk / "log.in.npt_prod_300K_production").write_text("npt log\n", encoding="utf-8")
    (nvt_chunk / "log.in.nvt_ramp_300K_c01").write_text("nvt log\n", encoding="utf-8")

    records = thermo_series.discover_npt_records_from_md_root(tmp_path)

    assert len(records) == 1
    assert records[0]["stage_name"] == "npt_prod_300K"
    assert records[0]["temperature"] == 300.0
    assert records[0]["log_path"].name == "log.in.npt_prod_300K_production"


def test_lammps_config_discovery_uses_fixed_npt_and_ignores_nvt(tmp_path) -> None:
    config = tmp_path / "config_lc_800K.json"
    config.write_text(
        json.dumps(
            {
                "timestep": 0.001,
                "stages": [
                    {
                        "name": "lc_nvt_ramp_300K",
                        "type": "nvt",
                        "temperature_start": 0,
                        "temperature_end": 300,
                    },
                    {
                        "name": "lc_npt_eqm_300K",
                        "type": "npt",
                        "temperature": 300,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    nvt_chunk = tmp_path / "stages" / "lc_nvt_ramp_300K" / "chunk_01"
    npt_chunk_1 = tmp_path / "stages" / "lc_npt_eqm_300K" / "chunk_01"
    npt_chunk_2 = tmp_path / "stages" / "lc_npt_eqm_300K" / "chunk_02"
    nvt_chunk.mkdir(parents=True)
    npt_chunk_1.mkdir(parents=True)
    npt_chunk_2.mkdir(parents=True)
    (nvt_chunk / "log.in.lc_nvt_ramp_300K_c01").write_text("nvt log\n", encoding="utf-8")
    (npt_chunk_1 / "log.in.lc_npt_eqm_300K_c01").write_text("npt old\n", encoding="utf-8")
    (npt_chunk_2 / "log.in.lc_npt_eqm_300K_c02").write_text("npt new\n", encoding="utf-8")

    records = thermo_series.discover_production_records([config])

    assert len(records) == 1
    assert records[0]["stage_name"] == "lc_npt_eqm_300K"
    assert records[0]["temperature"] == 300.0
    assert records[0]["timestep_ps"] == 0.001
    assert records[0]["log_path"].name == "log.in.lc_npt_eqm_300K_c02"


def test_lammps_cli_rejects_qha_hybrid_flags(tmp_path) -> None:
    with pytest.raises(SystemExit):
        thermo_series.main([
            "--manual-analysis-root",
            str(tmp_path),
            "--qha-anchor-dir",
            str(tmp_path / "qha"),
            "--qha-low-t-splice",
        ])


def test_lammps_tail_window_selects_last_requested_ps() -> None:
    data = {
        "step": np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float),
        "temp": np.linspace(300, 310, 11),
        "press_GPa": np.zeros(11),
        "vol_A3": np.linspace(100, 110, 11),
        "pe": np.linspace(-10, 0, 11),
        "enthalpy_eV": np.linspace(-9, 1, 11),
    }

    mask, metrics, _table = thermo_series.select_tail_window(
        data,
        target_T=300.0,
        timestep_ps=1.0,
        min_window_ps=20.0,
    )

    assert data["step"][mask].tolist() == [30.0, 35.0, 40.0, 45.0, 50.0]
    assert metrics["selection_method"] == "tail_last_window"


def test_lammps_compare_series_normalizes_different_box_sizes(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    small = tmp_path / "small"
    large = tmp_path / "large"
    out = tmp_path / "compare"
    small.mkdir()
    large.mkdir()

    (small / "thermo_functions_grid.csv").write_text(
        "T_K,n_formula_units,V_fit_A3,a_fit_A,Cp_used_for_integration_J_per_mol_UO2_K,"
        "S_rel_J_per_mol_UO2_K,H_rel_J_per_mol_UO2,G_rel_J_per_mol_UO2,"
        "alpha_V_micro_per_K,alpha_L_micro_per_K,qha_md_blend_weight\n"
        "300,32,800,5.0,60,70,1000,-2000,30,10,1\n"
        "500,32,832,5.1,70,80,2000,-3000,32,11,1\n",
        encoding="utf-8",
    )
    (small / "all_T_summary.csv").write_text("target_T_K,n_formula_units\n300,32\n", encoding="utf-8")
    (large / "thermo_functions_grid.csv").write_text(
        "T_K,n_formula_units,V_fit_A3,a_fit_A,Cp_used_for_integration_J_per_mol_UO2_K,"
        "S_rel_J_per_mol_UO2_K,H_rel_J_per_mol_UO2,G_rel_J_per_mol_UO2,"
        "alpha_V_micro_per_K,alpha_L_micro_per_K,qha_md_blend_weight\n"
        "300,256,6400,5.0,61,71,1100,-2100,31,10.5,1\n"
        "500,256,6656,5.1,71,81,2100,-3100,33,11.5,1\n",
        encoding="utf-8",
    )
    (large / "all_T_summary.csv").write_text("target_T_K,n_formula_units\n300,256\n", encoding="utf-8")

    thermo_series.compare_existing_lammps_series(
        [small, large],
        outdir=out,
        labels=["small", "large"],
        target_z=4.0,
    )

    metadata = json.loads((out / "compare_metadata.json").read_text())
    assert metadata["series"][0]["n_formula_units"] == 32.0
    assert metadata["series"][1]["n_formula_units"] == 256.0
    assert (out / "compare_volume_target_cell.png").exists()
    assert (out / "compare_Cp.png").exists()
    index = list(csv.DictReader((out / "compare_index.csv").open()))
    assert next(row for row in index if row["quantity"] == "V_target_cell_A3")["written"] == "True"


def test_lammps_compare_series_cli_writes_download_archive(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    small = tmp_path / "small"
    large = tmp_path / "large"
    out = tmp_path / "compare_cli"
    small.mkdir()
    large.mkdir()

    header = (
        "T_K,n_formula_units,V_fit_A3,a_fit_A,Cp_used_for_integration_J_per_mol_UO2_K,"
        "S_rel_J_per_mol_UO2_K,H_rel_J_per_mol_UO2,G_rel_J_per_mol_UO2,"
        "alpha_V_micro_per_K,alpha_L_micro_per_K,qha_md_blend_weight\n"
    )
    (small / "thermo_functions_grid.csv").write_text(
        header + "300,32,800,5.0,60,70,1000,-2000,30,10,1\n",
        encoding="utf-8",
    )
    (small / "all_T_summary.csv").write_text("target_T_K,n_formula_units\n300,32\n", encoding="utf-8")
    (large / "thermo_functions_grid.csv").write_text(
        header + "300,256,6400,5.0,61,71,1100,-2100,31,10.5,1\n",
        encoding="utf-8",
    )
    (large / "all_T_summary.csv").write_text("target_T_K,n_formula_units\n300,256\n", encoding="utf-8")

    thermo_series.main(
        [
            "--compare-series",
            str(small),
            str(large),
            "--compare-label",
            "small",
            "--compare-label",
            "large",
            "--outdir",
            str(out),
        ]
    )

    archive = out.with_name(f"{out.name}.tar.gz")
    assert archive.exists()
    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
    assert f"{out.name}/compare_index.csv" in names
    assert f"{out.name}/compare_metadata.json" in names


def test_md_engine_array_generates_selected_production_manifest_and_script(tmp_path) -> None:
    wrapper = tmp_path / "run_lammps_gpu.sh"
    wrapper.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=md-engine\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks=1\n"
        "#SBATCH --time=01:00:00\n"
        'echo "$1"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    model = tmp_path / "model.pt"
    model.write_text("model\n", encoding="utf-8")
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    for temp in (300, 400, 500):
        eqm = stages_dir / f"npt_eqm_{temp}K"
        eqm.mkdir()
        (eqm / f"npt_eqm_{temp}K.restart").write_text("restart\n", encoding="utf-8")

    config = tmp_path / "config_production.json"
    stages = [
        {
            "name": f"npt_prod_{temp}K",
            "type": "npt",
            "temperature": temp,
            "input_structure": f"stages/npt_eqm_{temp}K/npt_eqm_{temp}K.restart",
            "fixed_steps": 1000,
            "production_run": True,
        }
        for temp in (300, 400, 500)
    ]
    config.write_text(
        json.dumps(
            {
                "wrapper_script": str(wrapper),
                "model_file": str(model),
                "timestep": 0.001,
                "mass_O": 15.999,
                "mass_U": 238.0289,
                "performance": {
                    "reference_atoms": 96,
                    "reference_steps": 100000,
                    "reference_hours": 0.75,
                    "safety_factor": 1.0,
                },
                "stages": stages,
            }
        ),
        encoding="utf-8",
    )

    production_array.main(
        [
            "--root",
            str(tmp_path),
            "--config",
            str(config),
            "--T-range",
            "350:550",
            "--array-limit",
            "2",
        ]
    )

    outdir = tmp_path / "analysis" / "md_engine_array"
    manifest = outdir / "md_engine_array_manifest.tsv"
    script = outdir / "run_md_production_array.sh"
    rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
    assert [row["stage_name"] for row in rows] == ["npt_prod_400K", "npt_prod_500K"]
    assert [row["task_id"] for row in rows] == ["1", "2"]
    script_text = script.read_text(encoding="utf-8")
    assert "#SBATCH --array=1-2%2" in script_text
    assert "--run-task" in script_text
