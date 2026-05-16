from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from atomi.lammps import elastic


def write_fake_log(path, stress_gpa, lx=5.47, ly=5.47, lz=5.47):
    pressure_bar = -np.asarray(stress_gpa, dtype=float) / elastic.BAR_TO_GPA
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("LAMMPS fake log\n")
        handle.write("Step Temp PotEng TotEng Press Volume Lx Ly Lz Pxx Pyy Pzz Pyz Pxz Pxy Xy Xz Yz\n")
        for step in (0, 1000, 2000, 3000):
            handle.write(
                f"{step} 300 -1 -1 0 {lx*ly*lz:.8f} {lx} {ly} {lz} "
                + " ".join(f"{x:.8f}" for x in pressure_bar)
                + " 0 0 0\n"
            )


def test_prepare_writes_elastic_config_from_npt_config(tmp_path):
    root = tmp_path
    wrapper = root / "run_lammps_gpu.sh"
    wrapper.write_text("#!/bin/bash\n#SBATCH --time=01:00:00\n", encoding="utf-8")
    model = root / "model.pt"
    model.write_text("model\n", encoding="utf-8")
    stages = []
    for temp in (100, 200, 300):
        name = f"npt_prod_{temp}K"
        stage_dir = root / "stages" / name
        stage_dir.mkdir(parents=True)
        (stage_dir / f"{name}.restart").write_text("restart\n", encoding="utf-8")
        write_fake_log(stage_dir / "chunk_production" / f"log.in.{name}", np.zeros(6))
        stages.append({"name": name, "type": "npt", "temperature": temp, "production_run": True})
    cfg = {
        "wrapper_script": str(wrapper),
        "model_file": str(model),
        "timestep": 0.1,
        "mass_O": 15.999,
        "mass_U": 238.0289,
        "thermostat": {"tdamp": 0.8},
        "equilibrium_rules": {},
        "stages": stages,
    }
    config = root / "config_production.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    args = SimpleNamespace(
        config=[str(config)],
        md_root=None,
        template_config=None,
        config_dir=None,
        config_glob="*.json",
        duplicate_policy="highest_config_order",
        outdir=root / "analysis" / "elastic_lammps",
        config_out=root / "config_elastic.json",
        T_min=None,
        T_max=None,
        temperature_start=100.0,
        temperature_step=200.0,
        temperature_tol=1.0,
        include_all_temperatures=False,
        strain=None,
        mode=None,
        run_time_ps=20.0,
        analysis_window_ps=10.0,
        timestep_ps=None,
        dump_every=500,
        symmetry="auto",
    )
    plan = elastic.prepare_main(args)
    out = json.loads((root / "config_elastic.json").read_text(encoding="utf-8"))

    assert plan["temperatures_K"] == [100.0, 300.0]
    assert out["elastic_settings"]["run_time_ps"] == 20.0
    assert any(stage["type"] == "nvt" and stage["elastic_run"] for stage in out["stages"])
    assert any(stage["name"] == "elastic_T100K_ref" for stage in out["stages"])


def test_prepare_writes_elastic_config_from_md_root(tmp_path):
    root = tmp_path
    wrapper = root / "run_lammps_gpu.sh"
    wrapper.write_text("#!/bin/bash\n#SBATCH --time=01:00:00\n", encoding="utf-8")
    model = root / "model.pt"
    model.write_text("model\n", encoding="utf-8")
    for temp in (100, 200, 300):
        name = f"npt_prod_{temp}K"
        stage_dir = root / "stages" / name
        stage_dir.mkdir(parents=True)
        (stage_dir / f"{name}.restart").write_text("restart\n", encoding="utf-8")
        write_fake_log(stage_dir / "chunk_production" / f"log.in.{name}", np.zeros(6))
    cfg = {
        "wrapper_script": str(wrapper),
        "model_file": str(model),
        "timestep": 0.1,
        "mass_O": 15.999,
        "mass_U": 238.0289,
        "thermostat": {"tdamp": 0.8},
        "equilibrium_rules": {},
    }
    config = root / "config_production.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    args = SimpleNamespace(
        config=None,
        md_root=root,
        template_config=config,
        config_dir=None,
        config_glob="*.json",
        duplicate_policy="highest_config_order",
        outdir=root / "analysis" / "elastic_lammps",
        config_out=root / "config_elastic.json",
        T_min=None,
        T_max=None,
        temperature_start=100.0,
        temperature_step=200.0,
        temperature_tol=1.0,
        include_all_temperatures=False,
        strain=None,
        mode=None,
        run_time_ps=20.0,
        analysis_window_ps=10.0,
        timestep_ps=None,
        dump_every=500,
        symmetry="auto",
    )
    plan = elastic.prepare_main(args)

    assert plan["temperatures_K"] == [100.0, 300.0]
    assert (root / "config_elastic.json").exists()


def test_analyze_fits_cubic_elastic_moduli(tmp_path):
    root = tmp_path
    c11, c12, c44 = 300.0, 100.0, 80.0
    c = np.zeros((6, 6), dtype=float)
    c[:3, :3] = c12
    np.fill_diagonal(c[:3, :3], c11)
    c[3, 3] = c[4, 4] = c[5, 5] = c44
    stages = []
    cases = [("ref", 0.0, np.zeros(6))]
    for mode in elastic.DEFAULT_MODES:
        for strain in (0.01, -0.01, 0.02, -0.02):
            cases.append((mode, strain, np.asarray(elastic.voigt_strain(mode, strain), dtype=float)))
    for mode, strain, eps in cases:
        name = "elastic_T300K_ref" if mode == "ref" else f"elastic_T300K_{mode}_{elastic.strain_label(strain)}"
        stress = c @ eps
        stage = {
            "name": name,
            "type": "nvt",
            "temperature": 300.0,
            "chunk_name": "chunk_elastic",
            "elastic_run": True,
            "deformation": {
                "mode": mode,
                "strain": strain,
                "voigt_strain": eps.tolist(),
            },
        }
        stages.append(stage)
        write_fake_log(root / "stages" / name / "chunk_elastic" / f"log.in.{name}", stress)
    config = root / "config_elastic.json"
    config.write_text(json.dumps({"timestep": 0.1, "stages": stages}), encoding="utf-8")
    args = SimpleNamespace(
        elastic_config=config,
        outdir=root / "analysis" / "elastic_lammps" / "fit",
        window_ps=10.0,
        timestep_ps=None,
        symmetry="auto",
        symmetry_tolerance=0.01,
    )
    result = elastic.analyze_main(args)
    row = result["rows"][0]

    assert row["symmetry"] == "cubic"
    assert abs(row["C11_GPa"] - c11) < 1e-6
    assert abs(row["C12_GPa"] - c12) < 1e-6
    assert abs(row["C44_GPa"] - c44) < 1e-6
    expected_bulk = (c11 + 2.0 * c12) / 3.0
    expected_shear_voigt = (c11 - c12 + 3.0 * c44) / 5.0
    expected_shear_reuss = 5.0 * (c11 - c12) * c44 / (4.0 * c44 + 3.0 * (c11 - c12))
    expected_shear_hill = 0.5 * (expected_shear_voigt + expected_shear_reuss)
    expected_young = 9.0 * expected_bulk * expected_shear_hill / (3.0 * expected_bulk + expected_shear_hill)
    assert abs(row["E_H_GPa"] - expected_young) < 1e-6
    assert (args.outdir / "elastic_moduli_T.csv").exists()
