import json
from pathlib import Path

from atomi.core import doctor


def test_build_report_can_include_hpc_probe(monkeypatch, tmp_path: Path) -> None:
    def fake_which(name: str) -> str | None:
        if name in {"python3", "git"}:
            return f"/usr/bin/{name}"
        return None

    def fake_shell_probe(command: str, timeout: int = 20) -> dict[str, object]:
        return {"command": command, "returncode": 0, "output": f"ran: {command}"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(doctor.shutil, "which", fake_which)
    monkeypatch.setattr(doctor, "_run_shell_probe", fake_shell_probe)

    report = doctor.build_report(include_hpc_probe=True)

    probe = report["hpc_probe"]
    assert probe["pwd"] == str(tmp_path)
    assert probe["which"]["python3"] == "/usr/bin/python3"
    assert probe["which"]["git"] == "/usr/bin/git"
    assert probe["which"]["sbatch"] is None
    assert probe["which"]["nvcc"] is None
    assert probe["commands"]["module_avail_gcc_head60"]["output"].startswith("ran: module avail gcc")
    assert probe["commands"]["module_avail_cuda_head80"]["output"].startswith("ran: module avail cuda")
    assert probe["commands"]["nvidia_smi_query"]["command"].startswith("nvidia-smi --query-gpu")
    assert probe["commands"]["home_scratch_df"]["command"] == 'df -h "$HOME" "${SCRATCH:-$HOME}" 2>/dev/null'

    assert report["profiles"]["lammps_md_engine"]["module_commands"][0] == "module purge"
    assert report["profiles"]["gpu_lammps"]["modules"] == []
    assert "privately" in report["profiles"]["gpu_lammps"]["note"]


def test_hpc_config_report_redacts_private_values(tmp_path: Path) -> None:
    config_path = tmp_path / "kit.local.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "site": "KIT",
                "profiles": {
                    "lammps_md_engine": {
                        "modules": ["private/compiler", "private/cuda"],
                        "lammps_executable": "/private/lmp",
                        "partition": "private_gpu",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = doctor.build_report(hpc_config_path=config_path)

    config = report["hpc_config"]
    assert config["found"] is True
    assert config["site"] == "KIT"
    assert config["profile_names"] == ["lammps_md_engine"]
    profile = config["profiles"]["lammps_md_engine"]
    assert "modules" in profile["private_keys_redacted"]
    assert "lammps_executable" in profile["private_keys_redacted"]
    assert "partition" in profile["private_keys_redacted"]
    assert "private/compiler" not in json.dumps(config)


def test_hpc_config_report_can_include_private_values(tmp_path: Path) -> None:
    config_path = tmp_path / "kit.local.json"
    config_path.write_text(
        json.dumps({"profiles": {"mace_lammps": {"env_path": "/private/env"}}}),
        encoding="utf-8",
    )

    report = doctor.build_report(hpc_config_path=config_path, include_private_config=True)

    assert report["hpc_config"]["config"]["profiles"]["mace_lammps"]["env_path"] == "/private/env"


def test_write_private_template_and_discovery_script(tmp_path: Path) -> None:
    config_path = tmp_path / "atomi_hpc_config.new.local.json"
    script_path = tmp_path / "atomi_hpc_discover.sh"

    doctor.write_private_template(config_path, site="new_cluster")
    doctor.write_discovery_script(script_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    script = script_path.read_text(encoding="utf-8")

    assert config["site"] == "new_cluster"
    assert config["privacy"] == "local-only; do not commit or push"
    assert "vasp_cpu" in config["profiles"]
    assert "lammps_md_engine" in config["profiles"]
    assert config["profiles"]["lammps_md_engine"]["modules"] == []
    assert "ATOMI_PROBE_LAMMPS_GPU_MODULES" in script
    assert "ATOMI_DISCOVERY_INTERACTIVE=1" in script
    assert "ask_stack" in script
    assert "module spider" in script
    assert script_path.stat().st_mode & 0o111


def test_env_script_and_auto_setup_with_existing_config(tmp_path: Path) -> None:
    config_path = tmp_path / "atomi_hpc_config.existing.local.json"
    env_path = tmp_path / "atomi_hpc_env.sh"
    config_path.write_text(
        json.dumps(
            {
                "site": "existing",
                "profiles": {
                    "lammps_md_engine": {
                        "env_path": "/private/env",
                        "partition": "gpu",
                        "gres": "gpu:1",
                        "cpus_per_task": 4,
                        "modules": ["compiler/private", "cuda/private"],
                        "lammps_executable": "/private/lmp",
                        "lammps_prefix": "/private/lammps",
                    },
                    "cp2k": {
                        "data_dir": "/private/cp2k/data",
                        "executable": "/private/cp2k.psmp",
                        "runtime_library_path": "/private/intel/lib",
                        "environment": {
                            "CP2K_DATA_DIR": "/private/cp2k/data",
                            "OMP_NUM_THREADS": "${SLURM_CPUS_PER_TASK}",
                        },
                    },
                    "mace_training_gpu": {"env_path": "/private/mlip_env"},
                    "phonopy": {"module": "private/phonopy"},
                    "xafs_larch": {
                        "python": "/private/larch/bin/python",
                        "feff_executable": "/private/feff8l",
                    },
                    "zentropy": {
                        "python": "/private/zentropy/bin/python",
                        "zentropy_executable": "/private/zentropy/bin/pyzentropy",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = doctor.auto_setup_hpc(hpc_config_path=config_path, env_path=env_path)
    env_text = env_path.read_text(encoding="utf-8")

    assert result["config_found"] is True
    assert result["profile_names"] == [
        "cp2k",
        "lammps_md_engine",
        "mace_training_gpu",
        "phonopy",
        "xafs_larch",
        "zentropy",
    ]
    assert "source" in result["next_steps"][0]
    assert "export ATOMI_HPC_CONFIG=" in env_text
    assert "export ATOMI_LAMMPS_ENV=/private/env" in env_text
    assert "export ATOMI_LAMMPS_PARTITION=gpu" in env_text
    assert "export ATOMI_LAMMPS_GRES=gpu:1" in env_text
    assert "export ATOMI_LAMMPS_CPUS_PER_TASK=4" in env_text
    assert "export ATOMI_LAMMPS_MODULES='compiler/private cuda/private'" in env_text
    assert "export ATOMI_LMP_EXE=/private/lmp" in env_text
    assert "export ATOMI_CP2K_DATA_DIR=/private/cp2k/data" in env_text
    assert "export ATOMI_CP2K_EXE=/private/cp2k.psmp" in env_text
    assert "export ATOMI_CP2K_INTEL_RUNTIME_LIB=/private/intel/lib" in env_text
    assert "export ATOMI_MACE_TRAIN_ENV=/private/mlip_env" in env_text
    assert "export ATOMI_PHONOPY_MODULE=private/phonopy" in env_text
    assert "export ATOMI_XAFS_LARCH_PYTHON=/private/larch/bin/python" in env_text
    assert "export ATOMI_XAFS_FEFF_EXE=/private/feff8l" in env_text
    assert "export ATOMI_ZENTROPY_PYTHON=/private/zentropy/bin/python" in env_text
    assert "export ATOMI_ZENTROPY_EXE=/private/zentropy/bin/pyzentropy" in env_text
    assert "export CP2K_DATA_DIR=/private/cp2k/data" in env_text
    assert "OMP_NUM_THREADS" not in env_text


def test_auto_setup_without_config_writes_helpers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = doctor.auto_setup_hpc(site="new cluster")

    assert result["config_found"] is False
    assert (tmp_path / "atomi_hpc_config.new_cluster.local.json").exists()
    assert (tmp_path / "atomi_hpc_discover.sh").exists()
    assert "ATOMI_DISCOVERY_INTERACTIVE=1" in result["next_steps"][1]


def test_confighpc_applies_default_local_config(tmp_path: Path) -> None:
    config_path = tmp_path / "atomi_hpc_config.kit.local.json"
    env_path = tmp_path / "atomi_hpc_env.sh"
    config_path.write_text(
        json.dumps(
            {
                "site": "KIT",
                "environment_exports": {"ATOMI_CUSTOM": "yes"},
                "profiles": {
                    "lammps_md_engine": {"lammps_executable": "/private/lmp"},
                    "cp2k": {"data_dir": "/private/cp2k/data"},
                },
            }
        ),
        encoding="utf-8",
    )

    result = doctor.configure_hpc_environment(directory=tmp_path, env_path=env_path)
    env_text = env_path.read_text(encoding="utf-8")

    assert result["config_found"] is True
    assert result["config_path"] == str(config_path.resolve())
    assert result["site"] == "KIT"
    assert result["export_count"] == 4
    assert "export ATOMI_CUSTOM=yes" in env_text
    assert "export ATOMI_CP2K_DATA_DIR=/private/cp2k/data" in env_text
    assert "export ATOMI_LMP_EXE=/private/lmp" in env_text
    assert f"export ATOMI_HPC_CONFIG={config_path}" in env_text


def test_confighpc_prefers_named_local_config_and_warns_on_multiple(tmp_path: Path) -> None:
    less_preferred = tmp_path / "other.local.json"
    preferred = tmp_path / "atomi_hpc_config.kit.local.json"
    less_preferred.write_text(json.dumps({"site": "other"}), encoding="utf-8")
    preferred.write_text(json.dumps({"site": "kit"}), encoding="utf-8")

    selected, warnings = doctor.select_confighpc_config(directory=tmp_path, use_env=False)

    assert selected == preferred.resolve()
    assert warnings
    assert "Multiple *.local.json" in warnings[0]


def test_confighpc_reports_missing_config_without_discovery(tmp_path: Path) -> None:
    result = doctor.configure_hpc_environment(directory=tmp_path)

    assert result["config_found"] is False
    assert result["searched_patterns"] == list(doctor.LOCAL_CONFIG_PATTERNS)
    assert "atomi-doctor --auto-setup" in result["next_steps"][-1]
