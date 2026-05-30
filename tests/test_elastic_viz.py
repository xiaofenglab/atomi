from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from atomi.elastic.derived import complete_thermophysical_derived, formula_atom_count
from atomi.elastic.viz import main
from atomi.lammps.elastic import tensor_components, voigt_reuss_hill
from atomi.moose.material_export import load_external_properties


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
    assert derived["rho_kg_m3"] == derived["density_kg_m3"]
    assert derived["v_s_km_s"] > 0.0
    assert derived["theta_D_K"] > 0.0
    assert derived["k_min_cahill_W_mK"] > 0.0
    assert derived["k_min_clarke_W_mK"] > 0.0


def test_elastic_viz_reads_lammps_tensor_payload(tmp_path: Path, capsys) -> None:
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
            "--surface-energy-J-m2",
            "1.5",
            "--write-debye-thermal",
            "--debye-T-max",
            "20",
            "--debye-T-step",
            "10",
        ]
    )

    summary = list(csv.DictReader((outdir / "elastic_thermophysical_summary.csv").open(encoding="utf-8")))
    assert summary[0]["source"] == "LAMMPS/MD elastic"
    assert float(summary[0]["T_K"]) == 300.0
    assert float(summary[0]["E_Pa"]) > 0.0
    assert float(summary[0]["K_Pa"]) > 0.0
    assert float(summary[0]["G_Pa"]) > 0.0
    assert float(summary[0]["nu"]) > 0.0
    assert float(summary[0]["rho_kg_m3"]) > 0.0
    assert float(summary[0]["theta_D_K"]) > 0.0
    assert float(summary[0]["hardness_teter_GPa"]) > 0.0
    assert float(summary[0]["strain_energy_density_1pct_x_MJ_m3"]) > 0.0
    assert float(summary[0]["fracture_toughness_griffith_MPa_sqrt_m"]) > 0.0
    moose_rows = list(csv.DictReader((outdir / "moose_elastic_properties.csv").open(encoding="utf-8")))
    assert moose_rows[0]["source_tag"] == "LAMMPS_MD_elastic"
    assert float(moose_rows[0]["E_Pa"]) == float(summary[0]["E_Pa"])
    series, _, _ = load_external_properties([outdir / "moose_elastic_properties.csv"], [], [])
    assert {"E_Pa", "nu", "K_Pa", "G_Pa", "rho_kg_m3"}.issubset(series)
    assert (outdir / "elate_inputs" / "T300K_tensor_GPa.txt").exists()
    assert (outdir / "elate_surfaces" / "T300K_young_3d.html").exists()
    assert (outdir / "debye_thermal_functions.csv").exists()
    captured = capsys.readouterr().out
    assert "Elastic tensor summary" in captured
    assert "C_ij tensor : GPa, Voigt order xx yy zz yz xz xy" in captured
    assert "Bulk modulus K (GPa): Voigt upper=" in captured
    assert "Shear modulus G (GPa): Voigt upper=" in captured
    assert "Elastic moduli vs temperature" in captured


def test_elastic_viz_writes_benchmark_corrected_outputs(tmp_path: Path) -> None:
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
                    "md_box": {"volume_A3_mean": 1310.0},
                },
                "600.0": {
                    "temperature_K": 600.0,
                    "symmetry": "cubic",
                    "C_symmetry_reduced_GPa": (0.9 * c).tolist(),
                    "moduli": voigt_reuss_hill(0.9 * c),
                    "md_box": {"volume_A3_mean": 1310.0},
                },
            }
        ),
        encoding="utf-8",
    )
    with (elastic_dir / "elastic_moduli_T.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["temperature_K", "V_mean_A3", *tensor_components(c).keys(), *moduli.keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"temperature_K": 300.0, "V_mean_A3": 1310.0, **tensor_components(c), **moduli})
    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text(
        "T_K,K_GPa,G_GPa,E_GPa,nu\n"
        f"300,{1.25 * moduli['K_H_GPa']},{1.25 * moduli['G_H_GPa']},{1.25 * moduli['E_H_GPa']},{moduli['nu_H']}\n",
        encoding="utf-8",
    )
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
            "--benchmark-elastic-csv",
            str(benchmark),
            "--benchmark-correction-mode",
            "scalar-ratio",
            "--benchmark-anchor-T",
            "300",
        ]
    )

    corrected = list(csv.DictReader((outdir / "elastic_thermophysical_summary_corrected.csv").open(encoding="utf-8")))
    raw = list(csv.DictReader((outdir / "elastic_thermophysical_summary.csv").open(encoding="utf-8")))
    assert len(corrected) == 2
    assert float(corrected[0]["E_H_GPa"]) == pytest.approx(1.25 * float(raw[0]["E_H_GPa"]))
    assert (outdir / "elastic_tensors_corrected.json").exists()
    assert (outdir / "elastic_benchmark_correction.json").exists()
    series, _, _ = load_external_properties([outdir / "moose_elastic_properties_corrected.csv"], [], [])
    assert {"E_Pa", "nu", "K_Pa", "G_Pa", "rho_kg_m3"}.issubset(series)


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
