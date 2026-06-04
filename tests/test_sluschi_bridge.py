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
                        "env_path": "/envs/m_lammps_env",
                        "lammps_prefix": "/apps/lammps",
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
    assert status["env_path"] == "/envs/m_lammps_env"
    assert status["lammps_prefix"] == "/apps/lammps"
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
                        "env_path": "/envs/m_lammps_env",
                        "lammps_prefix": "/apps/lammps",
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
    assert result["env_path"] == "/envs/m_lammps_env"
    assert result["lammps_prefix"] == "/apps/lammps"
    manifest = json.loads((out / "mlip" / "sluschi_mlip_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "SuperSalt"
    assert manifest["model_path"] == str(model)
    assert manifest["provider_metadata"]["covered_elements"] == bridge.SUPERSALT_ELEMENTS
    assert (out / "README_KCL_LICL_SUPERSALT_DEMO.md").exists()
    probe = (out / "sluschi_inputs" / "run_supersalt_probe.sbatch").read_text(encoding="utf-8")
    assert str(lmp) in probe
    assert 'source "/envs/m_lammps_env/bin/activate"' in probe
    assert 'export LD_LIBRARY_PATH="/apps/lammps/lib:/apps/lammps/lib64:${LD_LIBRARY_PATH:-}"' in probe


def test_sluschi_lammps_prep_scripts_use_requested_type_basis(tmp_path: Path):
    out = tmp_path / "kcl_prep"

    result = bridge.main(
        [
            "lammps-prep-scripts",
            "--outdir",
            str(out),
            "--type-elements",
            "1=K,2=Cl",
        ]
    )

    pos_script = (out / "lmp_pos.py").read_text(encoding="utf-8")
    prep_script = (out / "lmp_prep.csh").read_text(encoding="utf-8")
    manifest = json.loads((out / "sluschi_lammps_prep_manifest.json").read_text(encoding="utf-8"))
    assert result["type_elements"] == {"1": "K", "2": "Cl"}
    assert manifest["elements"] == ["K", "Cl"]
    assert "symbols_by_type = {1: 'K', 2: 'Cl'}" in pos_script
    assert "@ nelms = 2" in prep_script
    assert "echo K >> param" in prep_script
    assert "echo Cl >> param" in prep_script
    assert "echo Li >> param" not in prep_script


def test_sluschi_cp2k_prep_writes_native_entropy_inputs(tmp_path: Path):
    xyz = tmp_path / "kcl-pos.xyz"
    xyz.write_text(
        "\n".join(
            [
                "4",
                "i = 0",
                "K 0.0 0.0 0.0",
                "Cl 1.0 0.0 0.0",
                "K 0.0 1.0 0.0",
                "Cl 1.0 1.0 0.0",
                "4",
                "i = 1",
                "K 0.1 0.0 0.0",
                "Cl 1.1 0.0 0.0",
                "K 0.0 1.1 0.0",
                "Cl 1.0 1.1 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    inp = tmp_path / "kcl.inp"
    inp.write_text("&FORCE_EVAL\n&SUBSYS\n&CELL\nABC 2.0 2.0 2.0\n&END CELL\n&END SUBSYS\n&END FORCE_EVAL\n", encoding="utf-8")
    out = tmp_path / "cp2k_prep"

    result = bridge.main(
        [
            "cp2k-prep",
            "--xyz",
            str(xyz),
            "--inp",
            str(inp),
            "--outdir",
            str(out),
            "--elements",
            "K,Cl",
            "--timestep-fs",
            "3",
            "--frame-stride-md-steps",
            "10",
            "--phase",
            "liquid",
        ]
    )

    assert result["schema"] == bridge.SCHEMA_CP2K_PREP
    assert result["n_selected_frames"] == 2
    assert result["sluschi_step_ps"] == 0.03
    assert (out / "param").read_text(encoding="utf-8").splitlines()[:7] == [
        "2",
        "2 2",
        "39.0983",
        "35.453",
        "0.03",
        "4",
        "K",
    ]
    assert (out / "phase_temp").read_text(encoding="utf-8").strip() == "liquid"
    assert len((out / "latt").read_text(encoding="utf-8").splitlines()) == 6
    pos_lines = (out / "pos").read_text(encoding="utf-8").splitlines()
    assert len(pos_lines) == 8
    assert pos_lines[0].startswith("0 ")
    assert pos_lines[2].startswith("0.5 ")
    assert pos_lines[4].startswith("0.05 ")
    manifest = json.loads((out / "sluschi_cp2k_prep_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_engine"] == "cp2k"
    assert manifest["counts"] == {"Cl": 2, "K": 2}


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


def test_lammps_sconfig_parses_sluschi_pair_recommendations(tmp_path: Path):
    root = tmp_path / "solid_T900"
    root.mkdir()
    (root / "collect.stdout").write_text(
        "\n".join(
            [
                "The pair between element 1-1 appears to be solid. I suggest that you take the mean:  1.25",
                "The pair between element 1-2 appears to be liquid. I suggest that you take the median:  2.50",
                "The pair between element 2-2 appears to be solid. I suggest that you take the mean:  0.75",
            ]
        ),
        encoding="utf-8",
    )
    (root / "Sconf.txt").write_text("1.0 2.0 3.0\n", encoding="utf-8")
    (root / "Sconf_min.txt").write_text("0.5 0.7\n", encoding="utf-8")
    out = tmp_path / "sconfig"

    result = bridge.main(
        [
            "sconfig",
            "--root",
            str(root),
            "--outdir",
            str(out),
            "--system",
            "UO2",
            "--formula",
            "UO2",
            "--phase",
            "fluorite",
            "--temperature-k",
            "900",
            "--quality",
            "screening-prior",
        ]
    )

    pair_rows = rows(out / "lammps_sconfig_pairs.csv")
    assert result["n_pair_recommendations"] == 3
    assert {row["pair"] for row in pair_rows} == {"1-1", "1-2", "2-2"}
    summary = json.loads((out / "lammps_sconfig_summary.json").read_text(encoding="utf-8"))
    assert summary["n_liquid_like_pairs"] == 1
    assert summary["n_solid_like_pairs"] == 2
    assert summary["mean_pair_sconfig_J_mol_atom_K"] == 1.5
    prior = json.loads((out / "lammps_sconfig_thermo_prior.json").read_text(encoding="utf-8"))
    assert prior["kind"] == "sluschi_lammps_sconfig"
    assert prior["thermo"]["observables"][0]["observable"] == "configurational_entropy_J_mol_atom_K"
    assert prior["thermo"]["observables"][0]["quality"] == "screening-prior"


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
                        "env_path": "/envs/m_lammps_env",
                        "lammps_prefix": "/apps/lammps",
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
    assert "ATOMI_SLUSCHI_ENV=/envs/m_lammps_env" in env_text
    assert "ATOMI_SLUSCHI_LAMMPS_PREFIX=/apps/lammps" in env_text


def test_sluschi_entropy_summary_combines_svib_and_sconf(tmp_path: Path):
    root = tmp_path / "run01"
    root.mkdir()
    (root / "collect.stdout").write_text(
        "\n".join(
            [
                "Svib:  13.1601 47.1196  J/K/mol atom. Do NOT use this value.",
                "Svib:  13.0397 46.4997  Constrained by ideal gas entropy. Use this value, not the line above.",
                "The pair between element 1-1 appears to be solid. I suggest that you take the minimum: -4.157e-05",
                "The pair between element 1-2 appears to be solid. I suggest that you take the minimum: -4.157e-05",
                "The pair between element 2-1 appears to be solid. I suggest that you take the minimum: -4.157e-05",
                "The pair between element 2-2 appears to be solid. I suggest that you take the minimum: -4.157e-05",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "entropy"

    result = bridge.main(
        [
            "entropy-summary",
            "--root",
            str(root),
            "--outdir",
            str(out),
            "--system",
            "UO2",
            "--formula",
            "UO2",
            "--phase",
            "fluorite",
            "--temperature-k",
            "300",
            "--type-stoich",
            "1=2,2=1",
        ]
    )

    data = rows(out / "sluschi_entropy_summary.csv")
    assert result["schema"] == bridge.SCHEMA_ENTROPY_SUMMARY
    assert result["workflow_lane"] == "entropy_prior"
    assert data[0]["Svib_J_mol_formula_K"] == "72.5791"
    assert data[0]["Sconf_J_mol_formula_K"] == "-0.00012471"
    assert data[0]["Stotal_J_mol_formula_K"] == "72.57897529"
    assert data[0]["Svib_type1_J_mol_atom_K"] == "13.0397"
    assert data[0]["type1_stoich"] == "2.0"


def test_sluschi_phase_health_flags_mixed_solid_entropy_row(tmp_path: Path):
    summary = {
        "schema": bridge.SCHEMA_ENTROPY_SUMMARY,
        "summary": {
            "system": "KCl",
            "formula": "KCl",
            "phase": "solid",
            "temperature_K": 1100.0,
            "composition": "x_KCl=1.0",
        },
        "sconfig_summary": {
            "system": "KCl",
            "formula": "KCl",
            "phase": "solid",
            "temperature_K": 1100.0,
            "composition": "x_KCl=1.0",
            "n_pair_recommendations": 4,
            "n_liquid_like_pairs": 2,
            "n_solid_like_pairs": 2,
        },
    }
    summary_json = tmp_path / "sluschi_entropy_summary.json"
    summary_json.write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "phase_health"

    result = bridge.main(["phase-health", "--summary-json", str(summary_json), "--outdir", str(out)])

    assert result["schema"] == bridge.SCHEMA_PHASE_HEALTH
    assert result["phase_health_label"] == "mixed"
    assert result["accepted_for_phase_label"] is False
    assert result["recommended_use"] == "screening-prior"
    data = rows(out / "sluschi_phase_health.csv")
    assert data[0]["phase_health_label"] == "mixed"
    assert "Solid-labeled trajectory" in data[0]["warnings"]


def test_sluschi_phase_health_accepts_solid_like_entropy_row(tmp_path: Path):
    root = tmp_path / "run01"
    root.mkdir()
    (root / "collect.stdout").write_text(
        "\n".join(
            [
                "The pair between element 1-1 appears to be solid. I suggest that you take the minimum: 0.1",
                "The pair between element 1-2 appears to be solid. I suggest that you take the minimum: 0.2",
                "The pair between element 2-1 appears to be solid. I suggest that you take the minimum: 0.2",
                "The pair between element 2-2 appears to be solid. I suggest that you take the minimum: 0.1",
            ]
        ),
        encoding="utf-8",
    )

    result = bridge.main(
        [
            "phase-health",
            "--root",
            str(root),
            "--outdir",
            str(tmp_path / "health"),
            "--expected-phase",
            "solid",
            "--system",
            "KCl",
            "--formula",
            "KCl",
        ]
    )

    assert result["phase_health_label"] == "solid-like"
    assert result["accepted_for_phase_label"] is True


def test_sluschi_workflow_guide_writes_two_lane_semantics(tmp_path: Path):
    out = tmp_path / "guide"

    result = bridge.main(["workflow-guide", "--system", "KCl", "--outdir", str(out)])

    assert result["schema"] == bridge.SCHEMA_WORKFLOW_GUIDE
    assert "coexistence" in result["lanes"]
    assert "entropy_prior" in result["lanes"]
    text = (out / "SLUSCHI_WORKFLOW_GUIDE.md").read_text(encoding="utf-8")
    assert "small-cell solid-liquid coexistence" in text


def test_sluschi_melting_anchor_parses_mpfit_output(tmp_path: Path):
    root = tmp_path / "coex"
    root.mkdir()
    (root / "SLUSCHI.out").write_text(
        "\n".join(
            [
                "=== running MPFit ===",
                "Melting temperature and std error: 1044.0 12.5",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "anchor"

    result = bridge.main(
        [
            "melting-anchor",
            "--root",
            str(root),
            "--outdir",
            str(out),
            "--system",
            "KCl",
            "--formula",
            "KCl",
            "--components",
            "KCl",
            "--composition",
            "x_KCl=1.0",
        ]
    )

    assert result["schema"] == bridge.SCHEMA_MELTING_ANCHOR
    data = rows(out / "sluschi_melting_anchor.csv")
    assert data[0]["melting_temperature_K"] == "1044.0"
    assert data[0]["temperature_std_error_K"] == "12.5"
    assert data[0]["method"] == "sluschi_mpfit"
    prior = json.loads((out / "sluschi_melting_anchor_thermo_prior.json").read_text(encoding="utf-8"))
    assert prior["kind"] == "sluschi_melting_anchor"
    assert prior["thermo"]["observables"][0]["observable"] == "melting_temperature_K"


def test_sluschi_melting_anchor_can_use_phase_health_bracket(tmp_path: Path):
    paths = []
    for name, temp, label in [
        ("solid_T1000", 1000.0, "single-phase-like"),
        ("solid_T1100", 1100.0, "coexistence-like"),
        ("solid_T1200", 1200.0, "single-phase-like"),
    ]:
        path = tmp_path / name / "sluschi_phase_health.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "schema": bridge.SCHEMA_PHASE_HEALTH,
                    "system": "KCl",
                    "formula": "KCl",
                    "temperature_K": temp,
                    "phase_health_label": label,
                }
            ),
            encoding="utf-8",
        )
        paths.extend(["--phase-health-json", str(path)])
    out = tmp_path / "anchor"

    result = bridge.main(
        [
            "melting-anchor",
            "--root",
            str(tmp_path),
            "--outdir",
            str(out),
            "--system",
            "KCl",
            "--formula",
            "KCl",
            "--components",
            "KCl",
            "--composition",
            "x_KCl=1.0",
            "--quality",
            "screening-prior",
            *paths,
        ]
    )

    data = rows(out / "sluschi_melting_anchor.csv")
    assert result["n_anchors"] == 1
    assert data[0]["melting_temperature_K"] == "1100.0"
    assert data[0]["temperature_std_error_K"] == "100.0"
    assert data[0]["method"] == "phase_health_bracket"
    assert data[0]["quality"] == "screening-prior"
