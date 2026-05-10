from atomi.lammps.workflow import effective_max_chunks, is_nvt_ramp_stage


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
