from pathlib import Path

import pytest

from atomi.calphad.env import _parse_csv, inspect_calphad_environment
from atomi.moose.env import inspect_moose_environment
from atomi.moose.material_export import main as moose_material_main
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


def test_moose_qha_md_material_export_merges_hybrid_outputs(tmp_path: Path) -> None:
    qha_md = tmp_path / "qha_md"
    qha_md.mkdir()
    (qha_md / "hybrid_cp_entropy.csv").write_text(
        "T_K,Cp_source,blend_weight,Cp,S_integrated,H_integrated_kJ_mol,G_integrated_kJ_mol\n"
        "300,QHA,0,235,0,0,0\n"
        "600,MD,1,300,20,6,0\n",
        encoding="utf-8",
    )
    (qha_md / "hybrid_volume_lattice.csv").write_text(
        "quantity,T_K,value,source,blend_weight\n"
        "V_A3,300,163.6,QHA,0\n"
        "V_A3,600,168.6,MD,1\n",
        encoding="utf-8",
    )
    (qha_md / "normalization_metadata.json").write_text(
        '{"target_z_formula_units": 4, "qha_formula_units": 32}\n',
        encoding="utf-8",
    )
    out_csv = tmp_path / "uo2_moose_material_properties.csv"
    out_meta = tmp_path / "uo2_moose_material_properties.meta.json"
    include = tmp_path / "uo2_material_functions.i"

    moose_material_main(
        [
            "--qha-md-dir",
            str(qha_md),
            "--out-csv",
            str(out_csv),
            "--out-meta",
            str(out_meta),
            "--moose-include",
            str(include),
            "--constant",
            "k_W_mK=7.8",
            "--constant",
            "E_Pa=2.05e11",
            "--constant",
            "nu=0.316",
        ]
    )

    import csv
    import json

    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == 2
    assert float(rows[0]["Cp_J_kgK"]) > 800.0
    assert float(rows[0]["rho_kg_m3"]) == pytest.approx(10970.0, rel=0.02)
    assert float(rows[0]["alpha_1_K"]) > 0.0
    assert float(rows[0]["dilatation"]) == pytest.approx(0.0)
    assert rows[0]["k_W_mK"] == "7.8"
    assert "uo2_k" in include.read_text(encoding="utf-8")
    metadata = json.loads(out_meta.read_text(encoding="utf-8"))
    assert metadata["units"] == "SI"
    assert metadata["target_z_formula_units"] == 4


def test_moose_qha_md_material_export_interpolates_property_csv(tmp_path: Path) -> None:
    qha_md = tmp_path / "qha_md"
    qha_md.mkdir()
    (qha_md / "thermo_functions_grid.csv").write_text(
        "T_K,Cp_used_for_integration_J_per_mol_UO2_K,density_fit_g_cm3,alpha_L_1_per_K\n"
        "300,240,10.9,1.0e-5\n"
        "600,300,10.6,1.2e-5\n",
        encoding="utf-8",
    )
    props = tmp_path / "literature.csv"
    props.write_text(
        "T_K,k_W_mK,E_Pa,nu\n"
        "300,8.0,2.0e11,0.31\n"
        "600,5.0,1.8e11,0.32\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "out.csv"

    moose_material_main(
        [
            "--qha-md-dir",
            str(qha_md),
            "--property-csv",
            str(props),
            "--out-csv",
            str(out_csv),
            "--out-meta",
            str(tmp_path / "out.json"),
        ]
    )

    import csv

    rows = list(csv.DictReader(out_csv.open()))
    assert float(rows[1]["k_W_mK"]) == pytest.approx(5.0)
    assert float(rows[0]["rho_kg_m3"]) == pytest.approx(10900.0)


def test_moose_qha_md_material_export_requires_literature_fields(tmp_path: Path) -> None:
    qha_md = tmp_path / "qha_md"
    qha_md.mkdir()
    (qha_md / "thermo_functions_grid.csv").write_text(
        "T_K,Cp_used_for_integration_J_per_mol_UO2_K,density_fit_g_cm3,alpha_L_1_per_K\n"
        "300,240,10.9,1.0e-5\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="missing k_W_mK"):
        moose_material_main(
            [
                "--qha-md-dir",
                str(qha_md),
                "--out-csv",
                str(tmp_path / "out.csv"),
                "--out-meta",
                str(tmp_path / "out.json"),
            ]
        )
