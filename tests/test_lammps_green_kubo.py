import json

from atomi.lammps import green_kubo
from atomi.lammps.workflow import generate_production_input, green_kubo_settings, set_project_root
from atomi.viz.gk import read_gk_run_plan, read_hcacf_rows, summarize_gk_status


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
    assert out["timestep"] == 0.00025
    assert out["timestep_ps"] == 0.00025
    assert out["stages"][0]["fixed_steps"] == 4000
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
    assert "suffix          kk" in text
    assert "timestep        0.00025" in text
    assert "suffix          off" not in text
    assert out["green_kubo_settings"]["heat_flux_suffix"] == "kk"
    assert not out["stages"][0]["green_kubo_settings"]["disable_accelerated_suffix_for_heat_flux"]


def test_green_kubo_prepare_applies_timestep_to_generated_input(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["timestep"] = 0.0001
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
    set_project_root(tmp_path)

    green_kubo.main(
        [
            "prepare",
            "--config",
            str(config),
            "--config-out",
            "config_gk_mliap_timestep.json",
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
            "0.02",
            "--nvt-preequilibration-ps",
            "0.002",
            "--timestep-ps",
            "0.00025",
            "--sample-interval-ps",
            "0.001",
            "--correlation-time-ps",
            "0.005",
        ]
    )

    out = json.loads((tmp_path / "config_gk_mliap_timestep.json").read_text(encoding="utf-8"))
    text, _data, _restart, steps = generate_production_input(
        out,
        out["stages"][0],
        tmp_path / out["stages"][0]["input_structure"],
        "gk_mliap_timestep",
    )
    assert out["timestep"] == 0.00025
    assert out["timestep_ps"] == 0.00025
    assert steps == 80
    assert "timestep        0.00025" in text
    assert "run             8" in text
    assert "fix             JJ all ave/correlate 4 5 20" in text
    assert "run             80" in text
    assert "timestep        0.0001" not in text


def test_green_kubo_prepare_records_mliap_runtime_estimate(tmp_path):
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
    set_project_root(tmp_path)

    green_kubo.main(
        [
            "prepare",
            "--config",
            str(config),
            "--outdir",
            "analysis/gk_mliap_estimate",
            "--config-out",
            "config_gk_mliap_estimate.json",
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
            "1.0",
            "--nvt-preequilibration-ps",
            "0.25",
            "--timestep-ps",
            "0.00025",
            "--gk-steps-per-hour",
            "3000",
            "--gk-walltime-safety-factor",
            "1.2",
            "--gk-reference-atoms",
            "768",
            "--array-limit",
            "1",
        ]
    )

    out = json.loads((tmp_path / "config_gk_mliap_estimate.json").read_text(encoding="utf-8"))
    plan = json.loads((tmp_path / "analysis" / "gk_mliap_estimate" / "gk_plan.json").read_text(encoding="utf-8"))
    estimate = plan["runtime_estimate"]
    assert estimate["timestep_fs"] == 0.25
    assert estimate["nve_steps_per_stage"] == 4000
    assert estimate["nvt_preequilibration_steps_per_stage"] == 1000
    assert estimate["estimated_total_md_steps_per_stage"] == 5000
    assert estimate["estimated_walltime_hours_per_stage"] == 2.0
    assert out["stages"][0]["walltime_hours"] == 2.0
    assert out["green_kubo_settings"]["runtime_estimate"] == estimate
    assert out["performance"]["model"] == "observed_gk_mliap_steps_per_hour"
    assert out["performance"]["reference_atoms"] == 768
    assert out["performance"]["reference_steps"] == 3000.0


def test_green_kubo_prepare_reads_mliap_timing_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOMI_LAMMPS_GK_STEPS_PER_HOUR", "2600")
    monkeypatch.setenv("ATOMI_LAMMPS_GK_WALLTIME_SAFETY_FACTOR", "1.25")
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
    set_project_root(tmp_path)

    green_kubo.main(
        [
            "prepare",
            "--config",
            str(config),
            "--outdir",
            "analysis/gk_mliap_env_estimate",
            "--config-out",
            "config_gk_mliap_env_estimate.json",
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
            "1.0",
            "--nvt-preequilibration-ps",
            "0.25",
            "--timestep-ps",
            "0.00025",
            "--array-limit",
            "1",
        ]
    )

    out = json.loads((tmp_path / "config_gk_mliap_env_estimate.json").read_text(encoding="utf-8"))
    estimate = out["green_kubo_settings"]["runtime_estimate"]
    assert estimate["estimated_total_md_steps_per_stage"] == 5000
    assert estimate["observed_steps_per_hour"] == 2600.0
    assert estimate["walltime_safety_factor"] == 1.25
    assert estimate["estimated_walltime_hours_per_stage"] == 2.4038461538461537
    assert out["stages"][0]["walltime_hours"] == 2.4038461538461537


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
    assert "align Python torch" in green_kubo.classify_probe_log(
        "ERROR: Running mliappy unified module failure. AttributeError: partially initialized module 'torch' has no attribute 'fx'"
    )
    assert "CuPy" in green_kubo.classify_probe_log("NameError: name 'cupy' is not defined")
    assert "CuPy" in green_kubo.classify_probe_log("ModuleNotFoundError: No module named 'cupy'")
    assert "MACE_ALLOW_CPU" in green_kubo.classify_probe_log("ValueError: GPU requested but tensor is on CPU")
    assert "cuequivariance_torch" in green_kubo.classify_probe_log(
        "AttributeError: module 'torch.compiler' has no attribute 'is_compiling'"
    )
    assert "cuequivariance_torch" in green_kubo.classify_probe_log(
        "AttributeError: module 'torch.fx._symbolic_trace' has no attribute 'is_fx_symbolic_tracing'"
    )
    assert "MPI_Init" in green_kubo.classify_probe_log("WARNING: Atomi LAMMPS -h preflight failed before input execution. MPI_Init")
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


def test_green_kubo_validate_reports_seed_and_axis_warnings(tmp_path, capsys):
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
    fit = tmp_path / "analysis" / "gk_fit"
    fit.mkdir(parents=True)
    chunk = tmp_path / "stages" / "gk_T300K_s01" / "chunk_gk"
    chunk.mkdir(parents=True)
    hcacf = chunk / "heatflux_hcacf.dat"
    hcacf.write_text(
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
    (fit / "gk_seed_summary.csv").write_text(
        "stage_name,temperature_K,seed,status,hcacf_path,k_seed_W_mK\n"
        f"gk_T300K_s01,300,100,ok,{hcacf},1.0\n",
        encoding="utf-8",
    )
    (fit / "thermal_conductivity_T.csv").write_text(
        "T_K,k_W_mK,k_x_W_mK,k_y_W_mK,k_z_W_mK,n_gk_seeds\n"
        "300,1.0,0.5,1.0,2.0,1\n",
        encoding="utf-8",
    )

    green_kubo.main(
        [
            "validate",
            "--gk-config",
            str(config),
            "--fit-dir",
            str(fit),
            "--min-seeds",
            "3",
        ]
    )

    output = capsys.readouterr().out
    assert "GK Validation Summary" in output
    assert "T=300 K" in output
    assert "only 1 ok seed" in output
    assert "axis spread high" in output
    report = json.loads((fit / "gk_validation_summary.json").read_text(encoding="utf-8"))
    assert report["temperatures"][0]["k_seed_sem_W_mK"] == 0.0


def test_green_kubo_hcacf_parser_handles_lammps_count_column(tmp_path):
    hcacf = tmp_path / "heatflux_hcacf.dat"
    hcacf.write_text(
        "\n".join(
            [
                "# TimeStep Number-of-time-windows",
                "# Index TimeDelta Ncount c_flux[1]*c_flux[1] c_flux[2]*c_flux[2] c_flux[3]*c_flux[3]",
                "8000 200",
                "192 7640 10 75123.5 -1.23732e+06 -2.99007e+06",
                "193 7680 9 586779 -798807 -2.05871e+06",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = read_hcacf_rows(hcacf, 0.00025)

    assert abs(rows[0]["time_ps"] - 1.91) < 1.0e-12
    assert rows[0]["count"] == 10
    assert rows[0]["HCACF_x"] == 75123.5
    assert rows[0]["HCACF_y"] == -1.23732e6
    assert rows[0]["HCACF_z"] == -2.99007e6


def test_green_kubo_status_reports_nve_progress_after_reset(tmp_path):
    input_file = tmp_path / "in.gk_T300K_s01_production"
    input_file.write_text(
        "\n".join(
            [
                "timestep        0.00025",
                'print           "Atomi GK phase: short NVT pre-equilibration before NVE heat-current production"',
                "run             8000",
                "reset_timestep  0",
                'print           "Atomi GK phase: NVE heat-current production and HCACF accumulation"',
                "fix             JJ all ave/correlate 40 200 8000 c_atomi_flux[1] c_atomi_flux[2] c_atomi_flux[3] type auto file heatflux_hcacf.dat ave running",
                "run             80000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log_file = tmp_path / "log.in.gk_T300K_s01_production"
    log_file.write_text(
        "\n".join(
            [
                'print           "Atomi GK phase: short NVT pre-equilibration before NVE heat-current production"',
                "Step Temp PotEng",
                "0 300 -1",
                "8000 301 -1",
                "Loop time of 1 on 1 procs for 8000 steps",
                'print           "Atomi GK phase: NVE heat-current production and HCACF accumulation"',
                "Step Temp PotEng",
                "0 301 -1",
                "3000 300 -1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plan = read_gk_run_plan(input_file)
    status = summarize_gk_status(log_file, plan)

    assert plan.nvt_steps == 8000
    assert plan.nve_steps == 80000
    assert plan.nevery == 40
    assert plan.nrepeat == 200
    assert plan.nfreq == 8000
    assert status.phase == "nve"
    assert status.current_steps == 3000
    assert status.expected_steps == 80000
    assert status.current_ps == 0.75
