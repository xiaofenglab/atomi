import csv
import json
from pathlib import Path

from atomi.calphad import workflow
from atomi.cli.main import main as atomi_main


class FakePhase:
    def __init__(self, constituents):
        self.constituents = constituents


class FakeParameter(dict):
    def __init__(self, doc_id, **kwargs):
        super().__init__(**kwargs)
        self.doc_id = doc_id


class FakeParameterTable:
    def __init__(self, rows):
        self.rows = rows
        self.removed = []

    def all(self):
        return list(self.rows)

    def remove(self, *, doc_ids):
        self.removed.extend(doc_ids)
        self.rows = [row for row in self.rows if row.doc_id not in set(doc_ids)]


class FakeDatabase:
    def __init__(self, rows):
        self._parameters = FakeParameterTable(rows)


def test_species_filtering_and_uo_preferred_phase_order():
    assert workflow.normalize_species_name("U_POS4") == "U"
    assert workflow.species_elements_from_name("UO2") == {"U", "O"}
    assert workflow.species_elements_from_name("NA2UCL6") == {"NA", "U", "CL"}
    assert workflow.parse_formula_counts("NaCl") == {"NA": 1.0, "CL": 1.0}
    assert workflow.parse_formula_counts("UCl3") == {"U": 1.0, "CL": 3.0}
    assert workflow.species_in_binary_subsystem("U_POS4", "U", "O")
    assert workflow.species_in_binary_subsystem("VA", "U", "O")
    assert not workflow.species_in_binary_subsystem("ZRO2", "U", "O")

    phases = ["GAS", "C1_MO2", "U4O9_S", "ZRO2_TETR", "BCC_A2"]
    assert workflow.recommend_phase_subset(phases, "U", "O") == ["BCC_A2", "C1_MO2", "U4O9_S", "GAS"]


def test_halide_salt_phase_subset_excludes_pure_liquids_and_solution_aliases():
    phases = [
        "MSFL",
        "LIF_L1(LIQ)",
        "UF4_L1(LIQ)",
        "LI4UF8_S1(S)",
        "LIUF5_S1(S)",
        "LIU4F17_S1(S)",
        "LIU2F9_S1(S)",
        "UF3_P3C1_NO.158(S)",
        "UF4_C2/C_NO.15(S)",
        "SSAESOLN",
        "GAS_IDEAL",
    ]

    selected = workflow.recommend_phase_subset(phases, "LiF", "UF4")

    assert selected == [
        "LI4UF8_S1(S)",
        "LIU2F9_S1(S)",
        "LIU4F17_S1(S)",
        "LIUF5_S1(S)",
        "MSFL",
        "UF4_C2/C_NO.15(S)",
    ]


def test_halide_salt_phase_subset_keeps_structural_alias_endmembers_from_debug():
    phases = ["C2_C", "FM3M", "MSCL", "NACL_L1(LIQ)", "NA2UCL6_S1(S)", "NAU2CL7_S1(S)", "UCL4_L1(LIQ)"]
    debug = {
        "C2_C": {"allowed_by_sublattice": [["UCL3"]]},
        "FM3M": {"allowed_by_sublattice": [["NACL"]]},
    }

    selected = workflow.recommend_phase_subset(phases, "NaCl", "UCl3", debug)

    assert selected == ["C2_C", "FM3M", "MSCL", "NA2UCL6_S1(S)", "NAU2CL7_S1(S)"]


def test_halide_salt_phase_subset_prefers_formula_named_aliases():
    phases = ["C2_C", "FM3M", "LIF_FM3M_NO.225(S)", "MSFL", "UF4_C2/C_NO.15(S)"]
    debug = {
        "C2_C": {"allowed_by_sublattice": [["UF4"]]},
        "FM3M": {"allowed_by_sublattice": [["LIF"]]},
    }

    selected = workflow.recommend_phase_subset(phases, "LiF", "UF4", debug)

    assert selected == ["LIF_FM3M_NO.225(S)", "MSFL", "UF4_C2/C_NO.15(S)"]


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


def test_pseudo_binary_formula_join_constraints():
    components, endmembers, dependent = workflow.formula_endmember_config("NaCl", "UCl3")
    fractions = workflow.pseudo_binary_element_fractions(0.5, endmembers, "NaCl", "UCl3")

    assert components == ["CL", "NA", "U", "VA"]
    assert dependent == "CL"
    assert fractions == {"NA": 1.0 / 6.0, "CL": 4.0 / 6.0, "U": 1.0 / 6.0}


def test_deduplicate_mqmqa_parameters_removes_duplicate_keys():
    dbf = FakeDatabase(
        [
            FakeParameter(
                1,
                phase_name="MSFL",
                parameter_type="MQMZ",
                constituent_array=(("LI", "U"), ("F", "F")),
                parameter_order=None,
            ),
            FakeParameter(
                2,
                phase_name="MSFL",
                parameter_type="MQMZ",
                constituent_array=(("LI", "U"), ("F", "F")),
                parameter_order=None,
            ),
            FakeParameter(
                3,
                phase_name="MSFL",
                parameter_type="MQMG",
                constituent_array=(("LI",), ("F",)),
                parameter_order=None,
            ),
        ]
    )

    removed = workflow.deduplicate_mqmqa_parameters(dbf, ["MSFL"])

    assert removed == 1
    assert dbf._parameters.removed == [2]
    assert [row.doc_id for row in dbf._parameters.rows] == [1, 3]


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


def test_init_workflow_formula_endmembers_use_elemental_components(tmp_path: Path):
    result = workflow.main(
        [
            "init",
            "--outdir",
            str(tmp_path / "salt_workflow"),
            "--tdb",
            "TDB/MSTDB.dat",
            "--component-a",
            "LiF",
            "--component-b",
            "UF4",
        ]
    )

    root = Path(result["root"])
    config = json.loads((root / "config" / "LiF_UF4_phase_config.json").read_text(encoding="utf-8"))
    assert config["components"] == ["F", "LI", "U", "VA"]
    assert config["formula_endmembers"]["LiF"] == {"F": 1.0, "LI": 1.0}
    assert config["formula_endmembers"]["UF4"] == {"F": 4.0, "U": 1.0}
    assert config["dependent_component"] == "F"


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


def test_boundary_points_from_grid_skips_none_by_default():
    grid = [
        ["A", "A", "B"],
        ["A", "NONE", "B"],
        ["C", "C", "B"],
    ]

    rows = workflow.boundary_points_from_grid(grid, [900, 1000, 1100], [0.1, 0.2, 0.3])

    assert rows
    assert all(row["field_1"] != "NONE" and row["field_2"] != "NONE" for row in rows)
    assert {row["orientation"] for row in rows} == {"vertical_in_x", "horizontal_in_T"}


def test_plot_diagram_cli_writes_boundary_csv(tmp_path: Path):
    grid_csv = tmp_path / "T_X_phase_grid.csv"
    rows = [
        {"T_K": 900, "X_UCL3": 0.1, "stable_signature": "FM3M", "stable_detail": "FM3M:1", "GM_J_mol": 0},
        {"T_K": 900, "X_UCL3": 0.2, "stable_signature": "FM3M", "stable_detail": "FM3M:1", "GM_J_mol": 0},
        {"T_K": 900, "X_UCL3": 0.3, "stable_signature": "MSCL", "stable_detail": "MSCL:1", "GM_J_mol": 0},
        {"T_K": 900, "X_UCL3": 0.4, "stable_signature": "MSCL", "stable_detail": "MSCL:1", "GM_J_mol": 0},
        {"T_K": 1000, "X_UCL3": 0.1, "stable_signature": "FM3M", "stable_detail": "FM3M:1", "GM_J_mol": 0},
        {"T_K": 1000, "X_UCL3": 0.2, "stable_signature": "MSCL", "stable_detail": "MSCL:1", "GM_J_mol": 0},
        {"T_K": 1000, "X_UCL3": 0.3, "stable_signature": "MSCL", "stable_detail": "MSCL:1", "GM_J_mol": 0},
        {"T_K": 1000, "X_UCL3": 0.4, "stable_signature": "MSCL", "stable_detail": "MSCL:1", "GM_J_mol": 0},
        {"T_K": 1100, "X_UCL3": 0.1, "stable_signature": "FM3M", "stable_detail": "FM3M:1", "GM_J_mol": 0},
        {"T_K": 1100, "X_UCL3": 0.2, "stable_signature": "MSCL", "stable_detail": "MSCL:1", "GM_J_mol": 0},
        {"T_K": 1100, "X_UCL3": 0.3, "stable_signature": "MSCL", "stable_detail": "MSCL:1", "GM_J_mol": 0},
        {"T_K": 1100, "X_UCL3": 0.4, "stable_signature": "MSCL", "stable_detail": "MSCL:1", "GM_J_mol": 0},
        {"T_K": 1200, "X_UCL3": 0.1, "stable_signature": "FM3M", "stable_detail": "FM3M:1", "GM_J_mol": 0},
        {"T_K": 1200, "X_UCL3": 0.2, "stable_signature": "MSCL", "stable_detail": "MSCL:1", "GM_J_mol": 0},
        {"T_K": 1200, "X_UCL3": 0.3, "stable_signature": "MSCL", "stable_detail": "MSCL:1", "GM_J_mol": 0},
        {"T_K": 1200, "X_UCL3": 0.4, "stable_signature": "MSCL", "stable_detail": "MSCL:1", "GM_J_mol": 0},
    ]
    with grid_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "phase_diagram.png"
    boundary_csv = tmp_path / "boundary_points.csv"

    metadata = workflow.main(
        [
            "plot-diagram",
            "--grid-csv",
            str(grid_csv),
            "--out",
            str(out),
            "--boundary-csv",
            str(boundary_csv),
            "--smooth-boundaries",
            "--smooth-window",
            "3",
        ]
    )

    assert metadata and metadata["n_boundary_points"] >= 2
    assert metadata["n_smoothed_boundary_paths"] >= 1
    assert metadata["smooth_window"] == 3
    assert out.exists()
    boundary_rows = list(csv.DictReader(boundary_csv.open(encoding="utf-8")))
    assert boundary_rows


def test_plot_diagram_bridge_gaps_mode_preserves_raw_boundaries(tmp_path: Path):
    grid_csv = tmp_path / "T_X_phase_grid.csv"
    rows = [
        {"T_K": 900, "X_O": 0.1, "stable_signature": "A", "stable_detail": "A:1", "GM_J_mol": 0},
        {"T_K": 900, "X_O": 0.2, "stable_signature": "B", "stable_detail": "B:1", "GM_J_mol": 0},
        {"T_K": 1000, "X_O": 0.1, "stable_signature": "NONE", "stable_detail": "", "GM_J_mol": 0},
        {"T_K": 1000, "X_O": 0.2, "stable_signature": "NONE", "stable_detail": "", "GM_J_mol": 0},
        {"T_K": 1100, "X_O": 0.1, "stable_signature": "A", "stable_detail": "A:1", "GM_J_mol": 0},
        {"T_K": 1100, "X_O": 0.2, "stable_signature": "B", "stable_detail": "B:1", "GM_J_mol": 0},
        {"T_K": 1200, "X_O": 0.1, "stable_signature": "A", "stable_detail": "A:1", "GM_J_mol": 0},
        {"T_K": 1200, "X_O": 0.2, "stable_signature": "B", "stable_detail": "B:1", "GM_J_mol": 0},
    ]
    with grid_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = workflow.main(
        [
            "plot-diagram",
            "--grid-csv",
            str(grid_csv),
            "--out",
            str(tmp_path / "phase_diagram.png"),
            "--boundary-csv",
            str(tmp_path / "boundary_points.csv"),
            "--smooth-boundaries",
            "--smooth-mode",
            "bridge-gaps",
            "--smooth-max-gap-steps",
            "3",
        ]
    )

    assert metadata["smooth_mode"] == "bridge-gaps"
    assert metadata["n_boundary_points"] == 3
    assert metadata["n_smoothed_boundary_paths"] == 1
