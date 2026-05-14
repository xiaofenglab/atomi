from pathlib import Path

from atomi.calphad.env import _parse_csv, inspect_calphad_environment
from atomi.moose.env import inspect_moose_environment
from atomi.moose.workflow import (
    build_info,
    load_moose_profile,
    render_slurm_submit,
    run_smoke,
)


def test_moose_environment_reports_requested_app(monkeypatch, tmp_path: Path) -> None:
    app = tmp_path / "my_moose_app-opt"
    app.write_text("#!/bin/sh\n", encoding="utf-8")
    app.chmod(0o755)

    monkeypatch.setenv("MOOSE_DIR", "/apps/moose")
    monkeypatch.setattr("atomi.moose.env.shutil.which", lambda name: None)

    report = inspect_moose_environment(app=str(app))

    assert report["module"] == "moose"
    assert report["app"] == str(app)
    assert report["executables"][str(app)]["available"] is True
    assert report["executables"][str(app)]["path"] == str(app)
    assert report["environment"]["MOOSE_DIR"] == "/apps/moose"


def test_calphad_environment_without_database_records_requested_scope() -> None:
    report = inspect_calphad_environment(
        components=_parse_csv("U,O,VA"),
        phases=_parse_csv("FLUORITE,LIQUID"),
    )

    assert report["module"] == "calphad"
    assert "available" in report["pycalphad"]
    assert report["database"] is None
    assert report["requested_components"] == ["U", "O", "VA"]
    assert report["requested_phases"] == ["FLUORITE", "LIQUID"]


def test_calphad_missing_database_is_reported(tmp_path: Path) -> None:
    report = inspect_calphad_environment(database=tmp_path / "missing.tdb")

    assert report["database"]["path"].endswith("missing.tdb")
    assert report["database"]["exists"] is False
    assert report["database"]["parsed"] is False


def test_moose_profile_loads_from_private_hpc_config(tmp_path: Path) -> None:
    config = tmp_path / "atomi_hpc_config.json"
    config.write_text(
        """
{
  "profiles": {
    "moose_gpu_kokkos": {
      "status": "ready",
      "scheduler": "slurm",
      "partition": "gpu",
      "gres": "gpu:1",
      "test_executable": "/private/app/moose_test-opt",
      "python_env": "/private/env",
      "calphad_work": "/private/calphad"
    }
  }
}
""",
        encoding="utf-8",
    )

    profile = load_moose_profile(config)
    info = build_info(profile, "moose_gpu_kokkos")

    assert info["status"] == "ready"
    assert info["partition"] == "gpu"
    assert info["test_executable"] == "/private/app/moose_test-opt"


def test_moose_submit_script_uses_profile_without_private_defaults() -> None:
    profile = {
        "partition": "gpu",
        "gres": "gpu:1",
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 8,
        "time": "00:10:00",
        "activation_script": "/private/activate.sh",
        "build_environment_exports": {"PSM2_CUDA": "1"},
        "test_executable": "/private/moose_test-opt",
    }

    script = render_slurm_submit(profile, job_name="smoke")

    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "source /private/activate.sh" in script
    assert "/private/moose_test-opt --help" in script


def test_moose_smoke_runs_configured_app(tmp_path: Path) -> None:
    app = tmp_path / "fake-moose-opt"
    app.write_text("#!/bin/sh\necho fake moose help\n", encoding="utf-8")
    app.chmod(0o755)

    report = run_smoke({"test_executable": str(app)})

    assert report["returncode"] == 0
    assert "fake moose help" in report["output"]
