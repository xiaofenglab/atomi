from pathlib import Path

from atomi.ml.mace import convert


def test_convertmace_local_can_pass_mliap_format(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model.model"
    model.write_text("model\n", encoding="utf-8")
    calls = []

    def fake_run(command, check=False, **_kwargs):
        calls.append((command, check))

    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    convert.convert_mace_model_local(model, model_format="mliap")

    assert calls == [
        (
            ["python", "-m", "mace.cli.create_lammps_model", str(model), "--format=mliap"],
            True,
        )
    ]


def test_convertmace_slurm_script_can_pass_mliap_format(tmp_path: Path) -> None:
    model = tmp_path / "model.model"
    env = tmp_path / "env"

    script = convert.build_slurm_script(
        model=model,
        env_path=env,
        partition="gpu",
        gres="gpu:1",
        time_limit="00:15:00",
        model_format="mliap",
    )

    assert "Format    : mliap" in script
    assert f'python -m mace.cli.create_lammps_model "{model}" --format=mliap' in script
