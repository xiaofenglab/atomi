from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from atomi.elastic.derived import complete_thermophysical_derived, formula_atom_count
from atomi.elastic.viz import main
from atomi.lammps.elastic import tensor_components, voigt_reuss_hill


def cubic_tensor() -> np.ndarray:
    c11, c12, c44 = 300.0, 100.0, 80.0
    c = np.zeros((6, 6), dtype=float)
    c[:3, :3] = c12
    np.fill_diagonal(c[:3, :3], c11)
    c[3, 3] = c[4, 4] = c[5, 5] = c44
    return c


def test_derived_debye_temperature_from_formula_volume() -> None:
    c = cubic_tensor()
    row = {"V_mean_A3": 1310.0, **tensor_components(c), **voigt_reuss_hill(c)}
    derived = complete_thermophysical_derived(row, formula="UO2", formula_units=32)

    assert formula_atom_count("UO2") == 3.0
    assert derived["density_g_cm3"] > 9.0
    assert derived["v_s_km_s"] > 0.0
    assert derived["theta_D_K"] > 0.0


def test_elastic_viz_reads_lammps_tensor_payload(tmp_path: Path) -> None:
    c = cubic_tensor()
    moduli = voigt_reuss_hill(c)
    elastic_dir = tmp_path / "fit"
    elastic_dir.mkdir()
    (elastic_dir / "elastic_tensors.json").write_text(
        json.dumps(
            {
                "300.0": {
                    "temperature_K": 300.0,
                    "symmetry": "cubic",
                    "C_symmetry_reduced_GPa": c.tolist(),
                    "moduli": moduli,
                    "md_box": {"volume_A3_mean": 1310.0, "a_A_mean": 10.94},
                }
            }
        ),
        encoding="utf-8",
    )
    with (elastic_dir / "elastic_moduli_T.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["temperature_K", "V_mean_A3", *tensor_components(c).keys(), *moduli.keys()],
        )
        writer.writeheader()
        writer.writerow({"temperature_K": 300.0, "V_mean_A3": 1310.0, **tensor_components(c), **moduli})
    outdir = tmp_path / "viz"
    main(
        [
            "--elastic-dir",
            str(elastic_dir),
            "--outdir",
            str(outdir),
            "--formula",
            "UO2",
            "--formula-units",
            "32",
            "--backend",
            "native",
            "--plot-3d",
            "--surface-npoints",
            "5",
            "--write-debye-thermal",
            "--debye-T-max",
            "20",
            "--debye-T-step",
            "10",
        ]
    )

    summary = list(csv.DictReader((outdir / "elastic_thermophysical_summary.csv").open(encoding="utf-8")))
    assert summary[0]["source"] == "LAMMPS/MD elastic"
    assert float(summary[0]["theta_D_K"]) > 0.0
    assert (outdir / "elate_inputs" / "T300K_tensor_GPa.txt").exists()
    assert (outdir / "elate_surfaces" / "T300K_young_3d.html").exists()
    assert (outdir / "debye_thermal_functions.csv").exists()


def test_elastic_viz_reads_vasp_tensor_payload(tmp_path: Path) -> None:
    c = cubic_tensor()
    elastic_dir = tmp_path / "vasp"
    elastic_dir.mkdir()
    (elastic_dir / "elastic_tensors.json").write_text(
        json.dumps(
            {
                "schema": "atomi.vasp.static_elastic.v1",
                "unit": "GPa",
                "tensors": [
                    {
                        "label": "001_V1.000",
                        "temperature_K": 0.0,
                        "symmetry": "cubic",
                        "symmetry_reduced_tensor_GPa": c.tolist(),
                        "moduli": voigt_reuss_hill(c),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with (elastic_dir / "elastic_moduli_T.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "temperature_K", "volume_A3"])
        writer.writeheader()
        writer.writerow({"label": "001_V1.000", "temperature_K": 0.0, "volume_A3": 1310.0})
    outdir = tmp_path / "viz"
    main(["--elastic-dir", str(elastic_dir), "--outdir", str(outdir), "--formula", "UO2", "--formula-units", "32"])

    summary = list(csv.DictReader((outdir / "elastic_thermophysical_summary.csv").open(encoding="utf-8")))
    assert summary[0]["source"] == "VASP/static elastic"
    assert float(summary[0]["universal_anisotropy_AU"]) >= 0.0
    assert (outdir / "elastic_viz_metadata.json").exists()
