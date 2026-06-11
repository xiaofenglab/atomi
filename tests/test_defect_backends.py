from __future__ import annotations

import json
from pathlib import Path

from atomi.zentropy.backends import (
    CETrainingRecord,
    CETrainingSet,
    ThermoSurface,
    backend_names,
    build_backend_doctor_report,
)
from atomi.zentropy.backends.base import read_ce_training_jsonl, write_ce_training_jsonl
from atomi.zentropy.pocc_defects import main as pocc_main


def test_backend_doctor_reports_optional_backends_without_failure() -> None:
    report = build_backend_doctor_report()
    names = {row["backend"] for row in report["backends"]}

    assert {"pocc_gqca_population_vector", "pocc_motif_mc", "smol_ce_mc", "casm_ce_mc_ti"} <= names
    assert report["backend_count"] == len(backend_names())
    by_name = {row["backend"]: row for row in report["backends"]}
    assert by_name["pocc_gqca_population_vector"]["available"] is True
    assert "install_hint" in by_name["smol_ce_mc"]
    assert "install_hint" in by_name["casm_ce_mc_ti"]


def test_cli_backend_doctor_can_emit_json(capsys) -> None:
    payload = pocc_main(["backend", "doctor", "--backend", "smol_ce_mc", "--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)

    assert payload["backend_count"] == 1
    assert parsed["backends"][0]["backend"] == "smol_ce_mc"


def test_ce_training_set_jsonl_round_trip(tmp_path: Path) -> None:
    training_set = CETrainingSet(
        system_name="Gd-UO2",
        parent_structure_path="parent/POSCAR",
        sublattice_model={"cation": ["U4", "U5", "Gd3"], "anion": ["O", "VaO"]},
        species={"U4": {"charge": 0}, "Gd3": {"charge": -1}},
        charge_constraints=["N_U5 + 2*N_VaO - N_Gd3 == 0"],
        composition_axes={"x_Gd": "N_Gd3/N_cation"},
        records=[
            CETrainingRecord(
                record_id="2Gd_2U5_orbit_0001",
                structure_path="structures/2Gd_2U5.vasp",
                species_counts={"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0},
                composition={"x_Gd": 2 / 32, "h_U5": 2 / 32, "delta": 0.0},
                motif_features={"Gd_U5_nn": 2.0},
                energy_eV=-100.0,
                source="vasp",
            )
        ],
    )
    path = tmp_path / "gd_uo2.ce_training.jsonl"

    write_ce_training_jsonl(path, training_set)
    loaded = read_ce_training_jsonl(path)

    assert loaded.system_name == "Gd-UO2"
    assert loaded.records[0].record_id == "2Gd_2U5_orbit_0001"
    assert loaded.records[0].species_counts["Gd3"] == 2
    assert loaded.records[0].motif_features["Gd_U5_nn"] == 2.0


def test_thermo_surface_schema_accepts_backend_metadata() -> None:
    surface = ThermoSurface(
        phase_name="FLUORITE_GD_U_O",
        backend="pocc_gqca_population_vector",
        t_grid=[1000.0],
        composition_grid=[{"x_Gd": 0.0625}],
        rows=[{"G_excess_eV": -0.01}],
        convergence_metadata={"mode4_trigger": "near_degenerate_motifs"},
    )

    assert surface.backend == "pocc_gqca_population_vector"
    assert surface.rows[0]["G_excess_eV"] < 0


def test_backend_doctor_text_command(capsys) -> None:
    pocc_main(["backend", "doctor", "--backend", "pocc_motif_mc"])
    out = capsys.readouterr().out

    assert "backend: pocc_motif_mc" in out
    assert "available: true" in out
