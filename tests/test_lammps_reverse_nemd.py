import json
from pathlib import Path

from atomi.lammps import reverse_nemd
from atomi.lammps.workflow import set_project_root


def base_cfg(tmp_path: Path) -> dict:
    model = tmp_path / "model.model"
    model.write_text("model\n", encoding="utf-8")
    wrapper = tmp_path / "run_lammps.sh"
    wrapper.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH --job-name=md-engine",
                "#SBATCH --output=lammps.%j.out",
                "#SBATCH --error=lammps.%j.err",
                "##SBATCH --partition=your_gpu_partition",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                "##SBATCH --gres=gpu:1",
                "#SBATCH --cpus-per-task=1",
                "#SBATCH --mem-per-cpu=3500M",
                "#SBATCH --time=01:00:00",
                "set -euo pipefail",
                'INPUT="${1:?input}"',
                'echo "would run ${INPUT}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
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


def write_completed_npt(tmp_path: Path, cfg: dict) -> Path:
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
    return config


def test_reverse_nemd_prepare_writes_replicated_inputs_and_array(tmp_path, monkeypatch):
    cfg = base_cfg(tmp_path)
    config = write_completed_npt(tmp_path, cfg)
    monkeypatch.setenv("ATOMI_LAMMPS_PARTITION", "gpu")
    monkeypatch.setenv("ATOMI_LAMMPS_GRES", "gpu:1")
    set_project_root(tmp_path)

    reverse_nemd.main(
        [
            "prepare",
            "--config",
            str(config),
            "--outdir",
            "analysis/rnemd",
            "--config-out",
            "config_rnemd.json",
            "--T-min",
            "300",
            "--T-max",
            "300",
            "--n-seeds",
            "1",
            "--run-time-ps",
            "2",
            "--replicate",
            "1x3x3",
            "--direction",
            "z",
            "--rnemd-steps-per-hour",
            "3000",
            "--rnemd-walltime-safety-factor",
            "1.5",
            "--array-limit",
            "1",
        ]
    )

    generated = json.loads((tmp_path / "config_rnemd.json").read_text(encoding="utf-8"))
    assert generated["runtime_profile"] == "lammps_rnemd"
    assert generated["rnemd_settings"]["replicate"] == "1x3x3"
    assert generated["rnemd_settings"]["runtime_estimate"]["run_steps_per_stage"] == 2000
    assert generated["rnemd_settings"]["runtime_estimate"]["estimated_walltime_hours_per_stage"] == 1.0
    assert generated["stages"][0]["walltime_hours"] == 1.0
    assert generated["slurm_resources"]["partition"] == "gpu"
    assert generated["slurm_resources"]["gres"] == "gpu:1"

    chunk = tmp_path / "analysis" / "rnemd" / "rnemd_T300K_s01" / "chunk_rnemd"
    input_text = (chunk / "in.rnemd_T300K_s01_production").read_text(encoding="utf-8")
    assert "read_data" in input_text
    assert "replicate       1 3 3" in input_text
    assert "suffix          kk" in input_text
    assert "pair_style      mace no_domain_decomposition" in input_text
    assert "fix             rnemd_flux all thermal/conductivity 100 z 20" in input_text
    assert "compute         rnemd_layers all chunk/atom bin/1d z lower 0.05 units reduced" in input_text
    assert "fix             rnemd_profile all ave/chunk 100 100 10000" in input_text
    assert "run             2000" in input_text

    wrapper = (chunk / "run_stage.sh").read_text(encoding="utf-8")
    assert "#SBATCH --time=01:00:00" in wrapper
    assert "#SBATCH --partition=gpu" in wrapper
    assert "#SBATCH --gres=gpu:1" in wrapper
    manifest = (tmp_path / "analysis" / "rnemd" / "rnemd_manifest.tsv").read_text(encoding="utf-8")
    assert "rnemd_T300K_s01" in manifest
    assert "1x3x3" in manifest
    array_script = (tmp_path / "analysis" / "rnemd" / "array" / "run_rnemd_array.sh").read_text(encoding="utf-8")
    assert "#SBATCH --array=1-1%1" in array_script
    assert "#SBATCH --partition=gpu" in array_script
    assert "./run_stage.sh" in array_script


def test_reverse_nemd_prepare_scales_template_timing_by_replicated_atoms(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["performance"] = {"reference_atoms": 768, "steps_per_hour": 23256}
    config = write_completed_npt(tmp_path, cfg)
    set_project_root(tmp_path)

    reverse_nemd.main(
        [
            "prepare",
            "--config",
            str(config),
            "--outdir",
            "analysis/rnemd_scaled",
            "--config-out",
            "config_rnemd_scaled.json",
            "--T-min",
            "300",
            "--T-max",
            "300",
            "--n-seeds",
            "1",
            "--run-time-ps",
            "1",
            "--replicate",
            "1x3x3",
            "--array-limit",
            "1",
        ]
    )

    generated = json.loads((tmp_path / "config_rnemd_scaled.json").read_text(encoding="utf-8"))
    estimate = generated["rnemd_settings"]["runtime_estimate"]
    assert estimate["throughput_source"] == "template_performance_scaled_by_atoms"
    assert estimate["base_atoms"] == 768
    assert estimate["target_atoms"] == 6912
    assert estimate["estimated_steps_per_hour"] == 2584.0
    assert estimate["run_steps_per_stage"] == 1000
