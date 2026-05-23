import json

from atomi.lammps import green_kubo
from atomi.lammps.workflow import generate_production_input, green_kubo_settings, set_project_root


def base_cfg(tmp_path):
    model = tmp_path / "model.model"
    model.write_text("model\n", encoding="utf-8")
    wrapper = tmp_path / "run_lammps.sh"
    wrapper.write_text("#!/bin/bash\n", encoding="utf-8")
    return {
        "wrapper_script": str(wrapper),
        "model_file": str(model),
        "timestep": 0.001,
        "mass_O": 15.999,
        "mass_U": 238.029,
        "velocity_seed": 12345,
        "thermostat": {"tdamp": 0.1},
        "barostat": {"pdamp": 1.0},
        "performance": {"reference_atoms": 96, "reference_steps": 1000, "reference_hours": 0.1},
    }


def test_green_kubo_stage_generates_nve_heat_flux_input(tmp_path):
    cfg = base_cfg(tmp_path)
    data = tmp_path / "start.data"
    data.write_text("data\n", encoding="utf-8")
    stage = {
        "name": "gk_T300K_s01",
        "type": "nve",
        "temperature": 300,
        "fixed_steps": 1000,
        "green_kubo_run": True,
        "recreate_velocity": True,
        "velocity_seed": 9001,
        "green_kubo_settings": {
            "sample_interval_ps": 0.01,
            "correlation_time_ps": 0.05,
            "nvt_preequilibration_ps": 0.02,
        },
    }

    text, _data, _restart, steps = generate_production_input(cfg, stage, data, "gk_test")

    assert steps == 1000
    assert "velocity        all create 300 9001" in text
    assert "fix             pre all nvt" in text
    assert "fix             1 all nve" in text
    assert "compute         atomi_flux all heat/flux" in text
    assert "fix             JJ all ave/correlate 10 5 50" in text
    assert "heatflux_hcacf.dat" in text
    assert "v_atomi_Jx v_atomi_Jy v_atomi_Jz" in text


def test_green_kubo_settings_rounds_sampling_to_timestep(tmp_path):
    settings = green_kubo_settings(
        {},
        {"green_kubo_settings": {"sample_interval_ps": 0.012, "correlation_time_ps": 0.06}},
        0.005,
    )

    assert settings["nevery"] == 2
    assert settings["effective_sample_interval_ps"] == 0.01
    assert settings["nrepeat"] == 6


def test_green_kubo_prepare_writes_multi_seed_config(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["stages"] = [
        {
            "name": "npt_prod_300K",
            "type": "npt",
            "temperature": 300,
            "production_run": True,
            "chunk_name": "chunk_production",
        }
    ]
    config = tmp_path / "config_production.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    stage_dir = tmp_path / "stages" / "npt_prod_300K"
    chunk = stage_dir / "chunk_production"
    chunk.mkdir(parents=True)
    (chunk / "log.in.npt_prod_300K_production").write_text("LAMMPS log\n", encoding="utf-8")
    (stage_dir / "npt_prod_300K.restart").write_text("restart\n", encoding="utf-8")
    set_project_root(tmp_path)

    green_kubo.main(
        [
            "prepare",
            "--config",
            str(config),
            "--outdir",
            "analysis/gk",
            "--config-out",
            "config_gk.json",
            "--n-seeds",
            "2",
            "--seed-start",
            "100",
            "--seed-step",
            "5",
            "--nve-time-ps",
            "1",
        ]
    )

    out = json.loads((tmp_path / "config_gk.json").read_text(encoding="utf-8"))
    assert [stage["velocity_seed"] for stage in out["stages"]] == [100, 105]
    assert all(stage["type"] == "nve" for stage in out["stages"])
    assert all(stage["green_kubo_run"] for stage in out["stages"])
    assert (tmp_path / "analysis" / "gk" / "gk_manifest.csv").exists()


def test_green_kubo_analyze_integrates_lammps_hcacf(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["stages"] = [
        {
            "name": "gk_T300K_s01",
            "type": "nve",
            "temperature": 300,
            "green_kubo_run": True,
            "chunk_name": "chunk_gk",
            "velocity_seed": 100,
        }
    ]
    config = tmp_path / "config_gk.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    chunk = tmp_path / "stages" / "gk_T300K_s01" / "chunk_gk"
    chunk.mkdir(parents=True)
    (chunk / "heatflux_hcacf.dat").write_text(
        "\n".join(
            [
                "# TimeStep Number-of-time-windows",
                "# Index TimeDelta c_flux[1]*c_flux[1] c_flux[2]*c_flux[2] c_flux[3]*c_flux[3]",
                "100 1",
                "1 0 1.0 1.0 1.0",
                "2 10 0.5 0.5 0.5",
                "3 20 0.25 0.25 0.25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    green_kubo.main(
        [
            "analyze",
            "--gk-config",
            str(config),
            "--outdir",
            str(tmp_path / "analysis" / "gk_fit"),
            "--scale-mode",
            "raw",
            "--plateau-start-ps",
            "0",
        ]
    )

    table = (tmp_path / "analysis" / "gk_fit" / "thermal_conductivity_T.csv").read_text(encoding="utf-8")
    assert "GK_MD_T300K" in table
    assert "0.00625" in table
