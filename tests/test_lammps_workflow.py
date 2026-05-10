from atomi.lammps.workflow import effective_max_chunks, is_nvt_ramp_stage
from atomi.lammps.thermo_series import (
    fill_missing_anchors_from_qha,
    integrate_qha_cp_anchor,
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
