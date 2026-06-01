import csv
import json
from pathlib import Path

from atomi.calphad import workflow
from atomi.cli.main import main as atomi_main


class FakePhase:
    def __init__(self, constituents):
        self.constituents = constituents


def test_species_filtering_and_uo_preferred_phase_order():
    assert workflow.normalize_species_name("U_POS4") == "U"
    assert workflow.species_elements_from_name("UO2") == {"U", "O"}
    assert workflow.species_in_binary_subsystem("U_POS4", "U", "O")
    assert workflow.species_in_binary_subsystem("VA", "U", "O")
    assert not workflow.species_in_binary_subsystem("ZRO2", "U", "O")

    phases = ["GAS", "C1_MO2", "U4O9_S", "ZRO2_TETR", "BCC_A2"]
    assert workflow.recommend_phase_subset(phases, "U", "O") == ["BCC_A2", "C1_MO2", "U4O9_S", "GAS"]


def test_phase_feasible_in_binary_subsystem_reports_allowed_sublattices():
    ok_phase = FakePhase([["U_POS4", "PU"], ["O_NEG2", "VA"]])
    ok, bad, allowed = workflow.phase_feasible_in_binary_subsystem(ok_phase, "U", "O")

    assert ok
    assert bad == []
    assert allowed == [["U_POS4"], ["O_NEG2", "VA"]]

    bad_phase = FakePhase([["ZR"], ["O_NEG2", "VA"]])
    ok, bad, allowed = workflow.phase_feasible_in_binary_subsystem(bad_phase, "U", "O")

    assert not ok
    assert bad[0]["sublattice"] == 1
    assert allowed == [[], ["O_NEG2", "VA"]]


def test_init_workflow_writes_portable_layout(tmp_path: Path):
    result = workflow.main(
        [
            "init",
            "--outdir",
            str(tmp_path / "calphad_workflow"),
            "--tdb",
            "TDB/UPUOC.TDB",
            "--component-a",
            "U",
            "--component-b",
            "O",
        ]
    )

    root = Path(result["root"])
    config = json.loads((root / "config" / "U_O_phase_config.json").read_text(encoding="utf-8"))
    assert (root / "TDB").is_dir()
    assert (root / "phase_diagram").is_dir()
    assert (root / "muO_maps").is_dir()
    assert config["tdb_file"] == "TDB/UPUOC.TDB"
    assert config["components"] == ["U", "O", "VA"]


def test_atomi_cli_forwards_calphad_workflow_init(tmp_path: Path):
    outdir = tmp_path / "forwarded"

    atomi_main(["calphad-workflow", "init", "--outdir", str(outdir)])

    assert (outdir / "README_calphad_workflow.md").exists()
    assert (outdir / "config" / "U_O_phase_config.json").exists()


def test_reaction_summary_writes_grid_diagnostics(tmp_path: Path):
    grid_csv = tmp_path / "T_X_phase_grid.csv"
    rows = [
        {"T_K": 1000, "X_O": 0.1, "stable_signature": "A", "stable_detail": "A:1", "GM_J_mol": 0},
        {"T_K": 1000, "X_O": 0.2, "stable_signature": "A", "stable_detail": "A:1", "GM_J_mol": 0},
        {"T_K": 1000, "X_O": 0.3, "stable_signature": "B", "stable_detail": "B:1", "GM_J_mol": 0},
        {"T_K": 1100, "X_O": 0.1, "stable_signature": "A", "stable_detail": "A:1", "GM_J_mol": 0},
        {"T_K": 1100, "X_O": 0.2, "stable_signature": "C", "stable_detail": "C:1", "GM_J_mol": 0},
        {"T_K": 1100, "X_O": 0.3, "stable_signature": "B", "stable_detail": "B:1", "GM_J_mol": 0},
    ]
    with grid_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outdir = tmp_path / "reaction_summary"

    result = workflow.main(["reaction-summary", "--grid-csv", str(grid_csv), "--outdir", str(outdir)])

    assert result["n_fields"] >= 3
    assert result["n_boundaries"] >= 2
    assert (outdir / "phase_fields_summary.csv").exists()
    assert (outdir / "phase_boundaries_summary.csv").exists()
    assert (outdir / "candidate_invariants.csv").exists()
    report = (outdir / "reaction_detection_report.txt").read_text(encoding="utf-8")
    assert "Candidate invariants" in report
