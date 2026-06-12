from __future__ import annotations

from pathlib import Path

from atomi.thermo import CETrainingRecord, CETrainingSet, ThermoSurface, read_ce_training_jsonl, write_ce_training_jsonl


def test_shared_thermo_schema_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    training = CETrainingSet(
        system_name="Gd-UO2",
        parent_structure_path="fluorite",
        records=[
            CETrainingRecord(
                record_id="case_01",
                structure_path="case_01/CONTCAR",
                composition={"x_Gd": 1 / 32, "delta_VO": 0.0, "h_U5": 1 / 32},
                motif_features={"Gd_U5_nn": 1.0},
                energy_eV=-10.0,
                source="vasp",
            )
        ],
    )

    write_ce_training_jsonl(path, training)
    loaded = read_ce_training_jsonl(path)

    assert loaded.system_name == "Gd-UO2"
    assert loaded.records[0].composition["x_Gd"] == 1 / 32
    assert loaded.records[0].motif_features["Gd_U5_nn"] == 1.0


def test_shared_thermo_surface_schema() -> None:
    surface = ThermoSurface(
        phase_name="FLUORITE_GD_U_O_DEFECT",
        backend="mode4_surface",
        t_grid=[300.0],
        composition_grid=[{"x_Gd": 0.03125, "delta_VO": 0.0}],
        rows=[{"G_eV_per_fu": -5.0, "confidence_label": "interpolated"}],
    )

    assert surface.rows[0]["confidence_label"] == "interpolated"
