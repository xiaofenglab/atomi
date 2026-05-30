import json
from pathlib import Path

from atomi.ml.mace import train


def test_mace_train_prepare_writes_profile_script(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "atomi_hpc_config.kit.local.json"
    config_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "mace_training_gpu": {
                        "partition": "gpu",
                        "gres": "gpu:1",
                        "cpus_per_task": 8,
                        "mem": "32G",
                        "time": "16:00:00",
                        "env_path": "/home/user/mlip_env",
                        "command": "mace_run_train",
                        "default_training_parameters": {
                            "epochs": 200,
                            "batch_size": 16,
                            "energy_key": "REF_energy",
                            "forces_key": "REF_forces",
                            "stress_key": "REF_stress",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    train.main(
        [
            "prepare",
            "--run-name",
            "uc2_lr1e3",
            "--mode",
            "new",
            "--no-2e",
            "--train-file",
            "datasets/training.extxyz",
            "--valid-file",
            "datasets/validation.extxyz",
            "--e0-mode",
            "explicit",
            "--e0s",
            "{6: -8.0, 92: -6.0}",
            "--epochs",
            "400",
        ]
    )

    script = tmp_path / "train_uc2_lr1e3.sbatch"
    plan = tmp_path / "train_uc2_lr1e3.plan.json"
    text = script.read_text(encoding="utf-8")
    payload = json.loads(plan.read_text(encoding="utf-8"))

    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "source /home/user/mlip_env/bin/activate" in text
    assert "--name=uc2_lr1e3" in text
    assert "--train_file=datasets/training.extxyz" in text
    assert "--valid_file=datasets/validation.extxyz" in text
    assert "--E0s={6: -8.0, 92: -6.0}" in text
    assert "--hidden_irreps=128x0e + 128x1o" in text
    assert payload["training_parameters"]["epochs"] == 400
    assert payload["e0_mode"] == "explicit"


def test_mace_train_retrain_uses_foundation_without_hidden_irreps() -> None:
    parser = train.build_parser()
    args = parser.parse_args(
        [
            "--run-name",
            "retrain_demo",
            "--mode",
            "retrain",
            "--train-file",
            "train.extxyz",
            "--valid-file",
            "valid.extxyz",
            "--foundation-model",
            "old.model",
            "--e0-mode",
            "estimated",
        ]
    )

    script, plan = train.render_training_sbatch(args, {"env_path": "/env"})

    assert "--foundation_model=old.model" in script
    assert "--multiheads_finetuning=False" in script
    assert "--hidden_irreps=" not in script
    assert plan["mode"] == "retrain"
    assert plan["e0_mode"] == "estimated"
