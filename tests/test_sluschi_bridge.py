import csv
import json
from pathlib import Path

from atomi.cli.main import main as atomi_main
from atomi.core import doctor
from atomi.sluschi import bridge


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_sluschi_bridge_init_writes_kcl_licl_handoff(tmp_path: Path):
    model = tmp_path / "supersalt.model"
    model.write_text("model", encoding="utf-8")
    out = tmp_path / "sluschi_kcl_licl"

    atomi_main(
        [
            "sluschi-bridge",
            "init",
            "--outdir",
            str(out),
            "--system",
            "KCl-LiCl",
            "--temperatures",
            "900,1000",
            "--compositions",
            "LiCl=0.25,KCl=0.75;LiCl=0.50,KCl=0.50",
            "--mlip-model",
            str(model),
        ]
    )

    plan = json.loads((out / "sluschi_bridge_plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == bridge.SCHEMA_PLAN
    assert plan["components"] == ["LiCl", "KCl"]
    assert plan["composition_grid"] == ["LiCl=0.25,KCl=0.75", "LiCl=0.50,KCl=0.50"]
    assert plan["temperature_grid_K"] == [900.0, 1000.0]
    assert plan["mlip"]["provider"] == "SuperSalt"
    assert (out / "sluschi_inputs" / "job.in").exists()
    manifest = json.loads((out / "mlip" / "sluschi_mlip_manifest.json").read_text(encoding="utf-8"))
    assert manifest["elements"] == ["Li", "K", "Cl"]
    assert manifest["provider_metadata"]["doi"] == bridge.SUPERSALT_DOI
    assert manifest["model_info"]["exists"] is True
    assert manifest["model_info"]["sha256"]
    assert "composition coverage" in manifest["validation_required"][0]
    assert (out / "sluschi_inputs" / "in.supersalt_probe").exists()


def test_sluschi_status_reads_hpc_profile(tmp_path: Path, monkeypatch):
    config = tmp_path / "atomi_hpc_config.kit.local.json"
    lmp = tmp_path / "lmp"
    lmp.write_text("#!/bin/sh\n", encoding="utf-8")
    config.write_text(
        json.dumps(
            {
                "profiles": {
                    "sluschi": {
                        "root": "/home/user/SLUSCHI",
                        "bin": "/home/user/SLUSCHI/bin",
                        "mlip_model": "/models/supersalt.pt",
                        "mlip_provider": "SuperSalt",
                        "lammps_executable": str(lmp),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH", str(tmp_path))

    status = bridge.inspect_environment(config)

    assert status["root"] == "/home/user/SLUSCHI"
    assert status["bin"] == "/home/user/SLUSCHI/bin"
    assert status["mlip_model"] == "/models/supersalt.pt"
    assert status["mlip_provider"] == "SuperSalt"
    assert status["executables"]["lmp"] == str(lmp)
    assert status["supersalt"]["doi"] == bridge.SUPERSALT_DOI
    assert status["ready_for_bridge"] is True


def test_sluschi_supersalt_example_uses_profile_model(tmp_path: Path):
    model = tmp_path / "SuperSalt-swa.model"
    model.write_text("model", encoding="utf-8")
    lmp = tmp_path / "lmp"
    lmp.write_text("#!/bin/sh\n", encoding="utf-8")
    config = tmp_path / "atomi_hpc_config.kit.local.json"
    config.write_text(
        json.dumps(
            {
                "profiles": {
                    "sluschi": {
                        "root": "/home/user/SLUSCHI",
                        "bin": "/home/user/SLUSCHI/src",
                        "mlip_model": str(model),
                        "mlip_provider": "SuperSalt",
                        "lammps_executable": str(lmp),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "demo"

    result = bridge.main(["supersalt-example", "--hpc-config", str(config), "--outdir", str(out)])

    assert result is not None
    assert result["mlip_model"] == str(model)
    assert result["lammps_executable"] == str(lmp)
    manifest = json.loads((out / "mlip" / "sluschi_mlip_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "SuperSalt"
    assert manifest["model_path"] == str(model)
    assert manifest["provider_metadata"]["covered_elements"] == bridge.SUPERSALT_ELEMENTS
    assert (out / "README_KCL_LICL_SUPERSALT_DEMO.md").exists()
    probe = (out / "sluschi_inputs" / "run_supersalt_probe.sbatch").read_text(encoding="utf-8")
    assert str(lmp) in probe


def test_sluschi_parse_collects_calphad_handoff_values(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "sluschi.out").write_text(
        "\n".join(
            [
                "melting temperature = 973.15 K",
                "heat of fusion = 14500",
                "liquid Cp = 96.5 J/mol/K at temperature = 1100 K",
                "solid entropy = 72.1 J/mol/K",
                "",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "results"

    result = bridge.main(
        [
            "parse",
            "--root",
            str(root),
            "--outdir",
            str(out),
            "--system",
            "NaCl-UCl3",
            "--components",
            "NaCl,UCl3",
            "--composition",
            "x_UCl3=0.50",
        ]
    )

    data = rows(out / "sluschi_parsed_results.csv")
    assert result["n_results"] >= 2
    assert {row["observable"] for row in data} >= {"melting_temperature_K", "heat_of_fusion_J_mol"}
    cp_rows = [row for row in data if row["observable"] == "heat_capacity_J_mol_K"]
    assert cp_rows
    assert cp_rows[0]["unit"] == "J/mol/K"
    assert cp_rows[0]["phase"] == "liquid"
    assert cp_rows[0]["composition"] == ""
    assert (out / "sluschi_parsed_results.json").exists()
    prior = json.loads((out / "sluschi_thermo_prior.json").read_text(encoding="utf-8"))
    assert prior["schema"] == "atomi.thermo_prior.v1"
    assert prior["kind"] == "sluschi_phase_observable_set"
    assert prior["system"] == "NaCl-UCl3"
    assert prior["components"] == ["NaCl", "UCl3"]
    assert any(item["observable"] == "heat_capacity_J_mol_K" for item in prior["thermo"]["observables"])


def test_confighpc_exports_sluschi_profile_values(tmp_path: Path):
    config = tmp_path / "atomi_hpc_config.kit.local.json"
    config.write_text(
        json.dumps(
            {
                "profiles": {
                    "sluschi": {
                        "root": "/home/user/SLUSCHI",
                        "bin": "/home/user/SLUSCHI/src",
                        "mlip_model": "/models/supersalt.pt",
                        "mlip_provider": "SuperSalt",
                        "lammps_executable": "/apps/lammps/bin/lmp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    env_text = doctor.render_env_script(doctor.load_hpc_config(config))

    assert "ATOMI_SLUSCHI_ROOT=/home/user/SLUSCHI" in env_text
    assert "ATOMI_SLUSCHI_BIN=/home/user/SLUSCHI/src" in env_text
    assert "ATOMI_SUPERSALT_MODEL=/models/supersalt.pt" in env_text
    assert "ATOMI_MLIP_PROVIDER=SuperSalt" in env_text
    assert "ATOMI_LMP_EXE=/apps/lammps/bin/lmp" in env_text
