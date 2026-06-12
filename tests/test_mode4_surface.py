from __future__ import annotations

import csv
import json
from pathlib import Path

from atomi.zentropy.backends.base import CETrainingRecord, CETrainingSet
from atomi.zentropy.mode4_surface import (
    build_training_set_from_csv,
    fit_linear_model,
    main,
    sample_surface,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_build_training_set_from_csv_extracts_composition_and_features(tmp_path: Path) -> None:
    source = tmp_path / "motifs.csv"
    source.write_text(
        "record_id,x_Gd,delta_VO,h_U5,energy_eV,feature_Gd_U5_nn,species_Gd3,species_U5,species_VaO\n"
        "a,0.0625,0,0.0625,-10.0,2,2,2,0\n",
        encoding="utf-8",
    )

    training = build_training_set_from_csv(source, system_name="Gd-UO2", parent_structure="fluorite")

    assert len(training.records) == 1
    rec = training.records[0]
    assert rec.composition["x_Gd"] == 0.0625
    assert rec.composition["h_U5"] == 0.0625
    assert rec.motif_features["Gd_U5_nn"] == 2.0
    assert rec.species_counts["Gd3"] == 2


def test_linear_mode4_fit_samples_dense_surface_with_confidence_labels(tmp_path: Path) -> None:
    records = []
    for i, (x, d, h) in enumerate(
        [
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.1),
        ]
    ):
        energy = -5.0 + 2.0 * x + 3.0 * d + 0.5 * h
        records.append(
            CETrainingRecord(
                record_id=f"r{i}",
                structure_path="",
                composition={"x_Gd": x, "delta_VO": d, "h_U5": h},
                energy_eV=energy,
            )
        )
    training = CETrainingSet(system_name="Gd-UO2", parent_structure_path="fluorite", records=records)

    model = fit_linear_model(
        training,
        phase="FLUORITE",
        feature_names=["x_Gd", "delta_VO", "h_U5"],
        ridge_lambda=0.0,
    )
    rows = sample_surface(model, temperatures=[1000.0], x_grid=[0.05, 0.3], delta_grid=[0.025], h_grid=[0.05])

    assert len(rows) == 2
    assert abs(rows[0]["G_eV_per_fu"] - (-5.0 + 2 * 0.05 + 3 * 0.025 + 0.5 * 0.05)) < 1.0e-8
    assert rows[0]["confidence_label"] == "interpolated"
    assert rows[1]["confidence_label"] in {"edge", "extrapolated"}


def test_mode4_surface_cli_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "training.csv"
    source.write_text(
        "record_id,x_Gd,delta_VO,h_U5,energy_eV\n"
        "r0,0,0,0,-5.0\n"
        "r1,0.1,0,0,-4.80\n"
        "r2,0,0.05,0,-4.85\n"
        "r3,0,0,0.1,-4.95\n",
        encoding="utf-8",
    )
    training = tmp_path / "training.jsonl"
    main(["build-training", "--input-csv", str(source), "--output", str(training)])
    fit_dir = tmp_path / "fit"
    main(
        [
            "fit",
            "--training-jsonl",
            str(training),
            "--outdir",
            str(fit_dir),
            "--feature",
            "x_Gd",
            "--feature",
            "delta_VO",
            "--feature",
            "h_U5",
            "--ridge-lambda",
            "0",
        ]
    )
    model = json.loads((fit_dir / "mode4_linear_model.json").read_text(encoding="utf-8"))
    assert model["schema"].endswith("linear_motif_hamiltonian.v1")

    surface_dir = tmp_path / "surface"
    main(
        [
            "sample-surface",
            "--model-json",
            str(fit_dir / "mode4_linear_model.json"),
            "--outdir",
            str(surface_dir),
            "--temperature",
            "300,600",
            "--x-grid",
            "0:0.2:0.1",
            "--delta-grid",
            "0,0.05",
            "--h-u5-grid",
            "0,0.1",
        ]
    )
    rows = read_csv(surface_dir / "mode4_dense_gibbs_surface.csv")
    assert len(rows) == 24

    export_dir = tmp_path / "export"
    main(["export-parameterized", "--surface-csv", str(surface_dir / "mode4_dense_gibbs_surface.csv"), "--outdir", str(export_dir)])
    assert (export_dir / "pycalphad_parameterized_g_surface.csv").exists()
    assert (export_dir / "moose_parameterized_g_surface.csv").exists()
    assert (export_dir / "parameterized_gibbs_surface_handoff.json").exists()
