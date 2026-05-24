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
    assert "suffix          off" in text
    assert "Atomi GK phase: heat-flux compatibility preflight" in text
    assert "run             0" in text
    assert "Atomi GK phase: short NVT pre-equilibration" in text
    assert "thermo_modify   flush yes" in text
    assert "Atomi GK phase: NVE heat-current production" in text
    assert "fix             pre all nvt" in text
    assert "fix             1 all nve" in text
    assert "compute         atomi_flux all heat/flux" in text
    assert text.index("compute         atomi_flux all heat/flux") < text.index("fix             pre all nvt")
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
    (stage_dir / "npt_prod_300K.data").write_text("data\n", encoding="utf-8")
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
    assert all(stage["green_kubo_settings"]["disable_accelerated_suffix_for_heat_flux"] for stage in out["stages"])
    assert all(stage["green_kubo_settings"]["heat_flux_preflight"] for stage in out["stages"])
    assert all(stage["input_structure"].endswith("npt_prod_300K.data") for stage in out["stages"])
    assert all(stage["input_restart_fallback"].endswith("npt_prod_300K.restart") for stage in out["stages"])
    assert (tmp_path / "analysis" / "gk" / "gk_manifest.csv").exists()
    manifest = (tmp_path / "analysis" / "gk" / "gk_manifest.csv").read_text(encoding="utf-8")
    assert "input_kind" in manifest
    assert "data" in manifest


def test_green_kubo_prepare_can_write_mliap_backend(tmp_path, monkeypatch):
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
    (stage_dir / "npt_prod_300K.data").write_text("data\n", encoding="utf-8")
    model = tmp_path / "model-mliap_lammps.pt"
    model.write_text("model\n", encoding="utf-8")
    monkeypatch.setenv("ATOMI_LAMMPS_PARTITION", "gpu")
    monkeypatch.setenv("ATOMI_LAMMPS_GRES", "gpu:1")
    set_project_root(tmp_path)

    green_kubo.main(
        [
            "prepare",
            "--config",
            str(config),
            "--config-out",
            "config_gk_mliap.json",
            "--model-file",
            str(model),
            "--pair-style-backend",
            "mliap",
            "--model-elements",
            "O",
            "U",
            "--n-seeds",
            "1",
            "--nve-time-ps",
            "1",
        ]
    )

    out = json.loads((tmp_path / "config_gk_mliap.json").read_text(encoding="utf-8"))
    assert out["runtime_profile"] == "lammps_gk_mliap"
    assert out["pair_style_backend"] == "mliap"
    assert out["model_file"].endswith("model-mliap_lammps.pt")
    assert out["model_elements"] == ["O", "U"]
    assert out["slurm_resources"]["partition"] == "gpu"
    assert out["slurm_resources"]["gres"] == "gpu:1"
    text, _data, _restart, _steps = generate_production_input(
        out,
        out["stages"][0],
        tmp_path / out["stages"][0]["input_structure"],
        "gk_mliap",
    )
    assert "pair_style      mliap unified" in text
    assert "pair_coeff      * * O U" in text
    assert "suffix          off" not in text
    assert not out["stages"][0]["green_kubo_settings"]["disable_accelerated_suffix_for_heat_flux"]


def test_green_kubo_mliap_probe_forces_gk_binary(tmp_path):
    cfg = base_cfg(tmp_path)
    data = tmp_path / "start.data"
    data.write_text("data\n", encoding="utf-8")
    cfg.update(
        {
            "pair_style_backend": "mliap",
            "runtime_profile": "lammps_gk_mliap",
            "model_elements": ["O", "U"],
            "stages": [
                {
                    "name": "gk_T300K_s01",
                    "type": "nve",
                    "temperature": 300,
                    "input_structure": "start.data",
                    "green_kubo_run": True,
                }
            ],
        }
    )
    config = tmp_path / "config_gk_mliap.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")

    green_kubo.main(
        [
            "probe",
            "--config",
            str(config),
            "--stage",
            "gk_T300K_s01",
            "--outdir",
            str(tmp_path / "probe_mliap"),
        ]
    )

    probe_input = (tmp_path / "probe_mliap" / "gk_heatflux_probe.in").read_text(encoding="utf-8")
    sbatch_runner = (tmp_path / "probe_mliap" / "run_probe_sbatch.sh").read_text(encoding="utf-8")
    submitter = (tmp_path / "probe_mliap" / "submit_probe.sh").read_text(encoding="utf-8")
    assert "suffix          kk" in probe_input
    assert "pair_style      mliap unified" in probe_input
    assert "GK_REQUESTED=0" in sbatch_runner
    assert 'confighpc --dir "$ATOMI_HPC_DIR" --no-env-var --shell' in sbatch_runner
    assert "Atomi GK/ML-IAP preflight: PASS" in sbatch_runner
    assert "mliap_unified_couple" in sbatch_runner
    assert "export ATOMI_LAMMPS_USE_GK_EXE=1" in submitter


def test_green_kubo_probe_writes_heat_flux_preflight(tmp_path, monkeypatch):
    cfg = base_cfg(tmp_path)
    wrapper = tmp_path / "run_lammps_gpu.sh"
    wrapper.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "##SBATCH --partition=your_gpu_partition",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                "##SBATCH --gres=gpu:1",
                "#SBATCH --cpus-per-task=1",
                "#SBATCH --time=01:00:00",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cfg["wrapper_script"] = str(wrapper)
    monkeypatch.setenv("ATOMI_LAMMPS_PARTITION", "gpu")
    monkeypatch.setenv("ATOMI_LAMMPS_GRES", "gpu:1")
    data = tmp_path / "start.data"
    data.write_text("data\n", encoding="utf-8")
    cfg["stages"] = [
        {
            "name": "gk_T300K_s01",
            "type": "nve",
            "temperature": 300,
            "input_structure": "start.data",
            "green_kubo_run": True,
        }
    ]
    config = tmp_path / "config_gk.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")

    green_kubo.main(
        [
            "probe",
            "--config",
            str(config),
            "--stage",
            "gk_T300K_s01",
            "--outdir",
            str(tmp_path / "probe"),
            "--suffix",
            "kk",
        ]
    )

    probe_input = (tmp_path / "probe" / "gk_heatflux_probe.in").read_text(encoding="utf-8")
    runner = (tmp_path / "probe" / "run_probe.sh").read_text(encoding="utf-8")
    sbatch_runner = (tmp_path / "probe" / "run_probe_sbatch.sh").read_text(encoding="utf-8")
    submitter = (tmp_path / "probe" / "submit_probe.sh").read_text(encoding="utf-8")
    report = json.loads((tmp_path / "probe" / "gk_heatflux_probe_report.json").read_text(encoding="utf-8"))
    assert "suffix          kk" in probe_input
    assert "compute         atomi_flux all heat/flux" in probe_input
    assert "run             0" in probe_input
    assert "Atomi GK probe: PASS heat/flux preflight completed" in probe_input
    assert "eval \"$LMP_CMD -in gk_heatflux_probe.in\"" in runner
    assert "#SBATCH --partition=gpu" in sbatch_runner
    assert "#SBATCH --gres=gpu:1" in sbatch_runner
    assert "#SBATCH --time=00:05:00" in sbatch_runner
    assert "sbatch run_probe_sbatch.sh gk_heatflux_probe.in" in submitter
    assert report["stage"] == "gk_T300K_s01"
    assert report["suffix"] == "kk"
    assert report["sbatch_runner"].endswith("run_probe_sbatch.sh")
    assert not report["executed"]
    assert "selected GK binary" in report["notes"][0]


def test_green_kubo_probe_classifies_common_mliap_failures():
    assert "ATOMI_LMP_GK_EXE" in green_kubo.classify_probe_log(
        "ERROR: Atomi LAMMPS preflight failed: selected GK executable does not expose the ML-IAP mliap pair style."
    )
    assert "Python coupling" in green_kubo.classify_probe_log(
        "ERROR: Atomi LAMMPS preflight failed: required ML-IAP Python modules could not be imported."
    )
    assert "unified Python module" in green_kubo.classify_probe_log("ERROR: Loading mliappy unified module failure.")
    assert "CUDA driver" in green_kubo.classify_probe_log("libcuda.so.1: cannot open shared object file")
    assert "libpython" in green_kubo.classify_probe_log("OSError: Unable to locate python shared library")
    assert "incompatible Torch/C10" in green_kubo.classify_probe_log("ImportError: torch/lib/libshm.so: undefined symbol: c10")
    assert "loaded but failed" in green_kubo.classify_probe_log("ERROR: Running mliappy unified module failure.")
    assert "API mismatch" in green_kubo.classify_probe_log(
        "ERROR: Running mliappy unified compute_forces failure. AttributeError: MLIAPDataPy object has no attribute forward_exchange"
    )
    assert "per-atom energy/virial" in green_kubo.classify_probe_log("ERROR: pair_mace does not support vflag_atom.")


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
