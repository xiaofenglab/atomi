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
            "/models/supersalt.pt",
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
    assert "composition coverage" in manifest["validation_required"][0]


def test_sluschi_status_reads_hpc_profile(tmp_path: Path, monkeypatch):
    config = tmp_path / "atomi_hpc_config.kit.local.json"
    config.write_text(
        json.dumps(
            {
                "profiles": {
                    "sluschi": {
                        "root": "/home/user/SLUSCHI",
                        "bin": "/home/user/SLUSCHI/bin",
                        "mlip_model": "/models/supersalt.pt",
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
    assert status["ready_for_bridge"] is True


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
