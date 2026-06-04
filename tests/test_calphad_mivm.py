import csv
import json
from pathlib import Path

import pytest

from atomi.calphad.mivm import (
    excess_enthalpy_j_mol,
    excess_gibbs_j_mol,
    load_parameters,
    main as mivm_main,
    parse_formula_counts,
    sanitize_mstdb_chemsage_text,
    tdb_sanity_check,
)
from atomi.cli.main import main as atomi_main


def test_mivm_default_guide_mentions_ceramic_parameters(capsys):
    mivm_main([])

    out = capsys.readouterr().out
    assert "(Gd,U)O2" in out
    assert "Gd-VO" in out
    assert "MIVM excess Gibbs energy" in out


def test_mivm_ceramic_json_contains_charge_compensation(capsys):
    mivm_main(["guide", "--system", "ceramic", "--format", "json"])

    data = json.loads(capsys.readouterr().out)
    ceramic_text = " ".join(data["ceramic_solid"])
    assert "U5+O2" in ceramic_text
    assert "oxygen vacancies" in ceramic_text


def test_atomi_cli_forwards_calphad_mivm(capsys):
    atomi_main(["calphad-mivm", "guide", "--system", "ceramic"])

    out = capsys.readouterr().out
    assert "Solid/ceramic" in out
    assert "(Gd,U)O2" in out


def write_simple_params(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "atomi.calphad.mivm.parameters.v1",
                "phase": "LIQUID",
                "components": {
                    "A": {"molar_volume": 10.0, "coordination": 10.0},
                    "B": {"molar_volume": 10.0, "coordination": 10.0},
                },
                "pairs": [
                    {"from": "A", "to": "B", "B": 1.0},
                    {"from": "B", "to": "A", "B": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_mivm_equal_volume_unit_pairs_has_zero_excess_gibbs(tmp_path: Path):
    path = tmp_path / "params.json"
    write_simple_params(path)
    params = load_parameters(path)

    assert excess_gibbs_j_mol(1000.0, {"A": 0.25, "B": 0.75}, params) == pytest.approx(0.0, abs=1.0e-10)
    assert excess_enthalpy_j_mol(1000.0, {"A": 0.25, "B": 0.75}, params) == pytest.approx(0.0, abs=1.0e-10)


def test_mivm_direct_enthalpy_uses_constant_pair_parameters(tmp_path: Path):
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps(
            {
                "schema": "atomi.calphad.mivm.parameters.v1",
                "phase": "LIQUID",
                "components": {
                    "LaCl3": {"molar_volume": 70.27, "coordination": 8.76},
                    "LiKCl_eut": {"molar_volume": 32.51, "coordination": 8.39},
                },
                "pairs": [
                    {"from": "LaCl3", "to": "LiKCl_eut", "B": 1.38},
                    {"from": "LiKCl_eut", "to": "LaCl3", "B": 1.04},
                ],
            }
        ),
        encoding="utf-8",
    )
    params = load_parameters(path)

    assert excess_enthalpy_j_mol(873.0, {"LaCl3": 0.42, "LiKCl_eut": 0.58}, params) == pytest.approx(
        -4590.0,
        abs=15.0,
    )


def test_mivm_sample_writes_bridge_table(tmp_path: Path):
    path = tmp_path / "params.json"
    write_simple_params(path)
    outdir = tmp_path / "sample"

    mivm_main(
        [
            "sample",
            "--params",
            str(path),
            "--outdir",
            str(outdir),
            "--temperature",
            "1000",
            "--binary-grid",
            "A,B,0,1,0.5",
        ]
    )

    rows = list(csv.DictReader((outdir / "mivm_property_table.csv").open(encoding="utf-8")))
    assert len(rows) == 3
    assert rows[1]["composition"] == "A=0.5;B=0.5"
    assert float(rows[1]["G_excess_MIVM_J_mol"]) == pytest.approx(0.0, abs=1.0e-10)
    assert float(rows[1]["H_excess_MIVM_J_mol"]) == pytest.approx(0.0, abs=1.0e-10)
    metadata = json.loads((outdir / "mivm_sample_metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == "atomi.calphad.mivm.sample.v1"


def test_mivm_sample_can_replace_ideal_entropy_with_sluschi_sconf(tmp_path: Path):
    path = tmp_path / "params.json"
    write_simple_params(path)
    sconf = tmp_path / "sconf.csv"
    sconf.write_text("x,Sconf_J_mol_formula_K\n0,0\n0.5,10\n1,0\n", encoding="utf-8")
    outdir = tmp_path / "sample_sconf"

    mivm_main(
        [
            "sample",
            "--params",
            str(path),
            "--outdir",
            str(outdir),
            "--temperature",
            "1000",
            "--binary-grid",
            "A,B,0,1,0.5",
            "--sconf-csv",
            str(sconf),
            "--sconf-x-column",
            "x",
            "--sconf-mode",
            "replace-ideal",
            "--sconf-x-component",
            "B",
        ]
    )

    rows = list(csv.DictReader((outdir / "mivm_property_table.csv").open(encoding="utf-8")))
    mid = rows[1]
    assert float(mid["Sconf_SLUSCHI_J_mol_K"]) == pytest.approx(10.0)
    assert float(mid["G_config_SLUSCHI_J_mol"]) == pytest.approx(-10000.0)
    assert float(mid["G_total_MIVM_SLUSCHI_J_mol"]) == pytest.approx(-10000.0)
    assert mid["Sconf_mode"] == "replace-ideal"
    metadata = json.loads((outdir / "mivm_sample_metadata.json").read_text(encoding="utf-8"))
    assert metadata["sconf"]["mode"] == "replace-ideal"


def test_mivm_compare_binary_writes_metrics_and_plot(tmp_path: Path):
    params_path = tmp_path / "params.json"
    write_simple_params(params_path)
    literature = tmp_path / "lit.csv"
    literature.write_text("x_A,Hmix_kJ_mol\n0,0\n0.5,0\n1,0\n", encoding="utf-8")
    outdir = tmp_path / "compare"

    metadata = mivm_main(
        [
            "compare-binary",
            "--params",
            str(params_path),
            "--outdir",
            str(outdir),
            "--temperature",
            "1000",
            "--binary-grid",
            "A,B,0,1,0.5",
            "--x-component",
            "A",
            "--literature-csv",
            str(literature),
            "--literature-x-column",
            "x_A",
            "--literature-y-column",
            "Hmix_kJ_mol",
            "--literature-label",
            "toy literature",
        ]
    )

    assert metadata is not None
    rows = list(csv.DictReader((outdir / "mivm_binary_comparison.csv").open(encoding="utf-8")))
    assert len(rows) == 3
    metrics = list(csv.DictReader((outdir / "mivm_binary_comparison_metrics.csv").open(encoding="utf-8")))
    assert metrics[0]["reference"] == "toy literature"
    assert float(metrics[0]["rmse_kJ_mol"]) == pytest.approx(0.0, abs=1.0e-12)
    assert (outdir / "mivm_binary_comparison_metadata.json").exists()


def test_benchmark_uq_phase_weights_hmix_and_eutectic(tmp_path: Path):
    curves = tmp_path / "curves.csv"
    curves.write_text(
        "\n".join(
            [
                "x_B,hmix_left,hmix_center",
                "0.05,-1.0,-0.2",
                "0.20,-4.0,-2.0",
                "0.35,-3.0,-5.0",
                "0.50,-1.0,-4.0",
                "0.65,-0.3,-2.0",
                "0.80,-0.1,-0.5",
                "0.95,0.0,-0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    literature = tmp_path / "hmix_lit.csv"
    literature.write_text("x_B,Hmix_kJ_mol\n0.20,-4.0\n0.35,-3.0\n0.50,-1.0\n", encoding="utf-8")
    outdir = tmp_path / "bench"

    metadata = mivm_main(
        [
            "benchmark-uq-phase",
            "--curve-csv",
            str(curves),
            "--x-column",
            "x_B",
            "--curve-columns",
            "hmix_left,hmix_center",
            "--curve-labels",
            "left,center",
            "--literature-csv",
            str(literature),
            "--literature-x-column",
            "x_B",
            "--literature-y-column",
            "Hmix_kJ_mol",
            "--component-a",
            "A",
            "--component-b",
            "B",
            "--x-component",
            "B",
            "--tm-a",
            "1000",
            "--tm-b",
            "1100",
            "--dhfus-a",
            "20",
            "--dhfus-b",
            "22",
            "--eutectic-x",
            "0.45",
            "--eutectic-t",
            "760",
            "--sigma-hmix",
            "0.5",
            "--sigma-eutectic-x",
            "0.2",
            "--sigma-eutectic-t",
            "200",
            "--outdir",
            str(outdir),
        ]
    )

    assert metadata is not None
    assert metadata["schema"] == "atomi.calphad.mivm.benchmark_uq_phase.v1"
    assert (outdir / "posterior_model_weights.csv").exists()
    assert (outdir / "candidate_phase_diagrams.csv").exists()
    assert (outdir / "posterior_phase_envelope.csv").exists()
    assert (outdir / "posterior_tension_report.md").exists()
    rows = list(csv.DictReader((outdir / "posterior_model_weights.csv").open(encoding="utf-8")))
    assert {row["label"] for row in rows} == {"left", "center"}
    assert sum(float(row["posterior_weight"]) for row in rows) == pytest.approx(1.0)
    assert any(float(row["hmix_rmse_kJ_mol"]) < 1.0 for row in rows)
    assert all("dCp_B_liq_minus_solid_J_mol_K" in row for row in rows)


def test_benchmark_uq_phase_accepts_sluschi_sconf_replace_ideal(tmp_path: Path):
    curves = tmp_path / "curves.csv"
    curves.write_text("x_B,hmix\n0.1,-1\n0.3,-3\n0.5,-2\n0.7,-1\n0.9,0\n", encoding="utf-8")
    sconf = tmp_path / "sconf.csv"
    sconf.write_text("x,Sconf_J_mol_formula_K\n0.1,1\n0.5,6\n0.9,1\n", encoding="utf-8")
    outdir = tmp_path / "bench_sconf"

    metadata = mivm_main(
        [
            "benchmark-uq-phase",
            "--curve-csv",
            str(curves),
            "--x-column",
            "x_B",
            "--curve-columns",
            "hmix",
            "--component-a",
            "A",
            "--component-b",
            "B",
            "--x-component",
            "B",
            "--tm-a",
            "1000",
            "--tm-b",
            "1100",
            "--dhfus-a",
            "20",
            "--dhfus-b",
            "22",
            "--eutectic-x",
            "0.35",
            "--eutectic-t",
            "800",
            "--sconf-csv",
            str(sconf),
            "--sconf-x-column",
            "x",
            "--sconf-mode",
            "replace-ideal",
            "--outdir",
            str(outdir),
        ]
    )

    assert metadata is not None
    assert metadata["sconf"]["mode"] == "replace-ideal"
    rows = list(csv.DictReader((outdir / "candidate_phase_diagrams.csv").open(encoding="utf-8")))
    assert rows[0]["Sconf_mode"] == "replace-ideal"
    assert any(float(row["Sconf_SLUSCHI_J_mol_K"]) > 1.0 for row in rows)
    weights = list(csv.DictReader((outdir / "posterior_model_weights.csv").open(encoding="utf-8")))
    assert weights[0]["Sconf_mode"] == "replace-ideal"


def test_benchmark_uq_phase_scans_dcp_grid(tmp_path: Path):
    curves = tmp_path / "curves.csv"
    curves.write_text("x_B,hmix\n0.1,-1\n0.3,-3\n0.5,-2\n0.7,-1\n0.9,0\n", encoding="utf-8")
    outdir = tmp_path / "bench_dcp"

    metadata = mivm_main(
        [
            "benchmark-uq-phase",
            "--curve-csv",
            str(curves),
            "--x-column",
            "x_B",
            "--curve-columns",
            "hmix",
            "--component-a",
            "A",
            "--component-b",
            "B",
            "--x-component",
            "B",
            "--tm-a",
            "1000",
            "--tm-b",
            "1100",
            "--dhfus-a",
            "20",
            "--dhfus-b",
            "22",
            "--dcp-b-grid=-10,10,10",
            "--eutectic-x",
            "0.35",
            "--eutectic-t",
            "800",
            "--outdir",
            str(outdir),
        ]
    )

    assert metadata is not None
    assert metadata["dCp_grid_J_mol_K"]["component_b"] == [-10.0, 0.0, 10.0]
    rows = list(csv.DictReader((outdir / "posterior_model_weights.csv").open(encoding="utf-8")))
    assert len(rows) == 3
    assert {float(row["dCp_B_liq_minus_solid_J_mol_K"]) for row in rows} == {-10.0, 0.0, 10.0}


def test_benchmark_uq_phase_accepts_line_compound(tmp_path: Path):
    curves = tmp_path / "curves.csv"
    curves.write_text("x_B,hmix\n0.1,-1\n0.3,-3\n0.5,-2\n0.7,-1\n0.9,0\n", encoding="utf-8")
    outdir = tmp_path / "bench_compound"

    metadata = mivm_main(
        [
            "benchmark-uq-phase",
            "--curve-csv",
            str(curves),
            "--x-column",
            "x_B",
            "--curve-columns",
            "hmix",
            "--component-a",
            "A",
            "--component-b",
            "B",
            "--x-component",
            "B",
            "--tm-a",
            "1000",
            "--tm-b",
            "1100",
            "--dhfus-a",
            "20",
            "--dhfus-b",
            "22",
            "--line-compound",
            "A3B5:0.625:-10:0:800",
            "--eutectic-x",
            "0.35",
            "--eutectic-t",
            "800",
            "--outdir",
            str(outdir),
        ]
    )

    assert metadata is not None
    assert metadata["line_compounds"][0]["label"] == "A3B5"
    rows = list(csv.DictReader((outdir / "candidate_phase_diagrams.csv").open(encoding="utf-8")))
    assert "line_compound_A3B5_K" in rows[0]


def test_tdb_sanity_warns_on_chemsage_style_export(tmp_path: Path):
    path = tmp_path / "MSTDB-No-Functions.dat"
    path.write_text(
        " System Na-U-Cl L\n"
        " Na U Cl\n"
        " NaCl\n"
        "  1000.0 0.0 0.0\n",
        encoding="utf-8",
    )

    sanity = tdb_sanity_check(path)

    assert not sanity["looks_like_pycalphad_tdb"]
    assert not sanity["native_tdb_like"]
    assert sanity["chemsage_like"]
    assert sanity["counts"]["CHEMSAGE_SYSTEM"] == 1
    assert sanity["warnings"]


def test_mstdb_sanitizer_preserves_real_species_stoichiometry():
    text = "U[CN=VI]+3.0 CL-1.0 U[DIMER]+6.0 NA[1+]+1.0"

    sanitized, metadata = sanitize_mstdb_chemsage_text(text)

    assert "U+3.0" in sanitized
    assert "U2+6.0" in sanitized
    assert "NA+1.0" in sanitized
    assert "[CN=VI]" not in sanitized
    assert "[DIMER]" not in sanitized
    assert metadata["changed"]


def test_mstdb_sanitize_cli_writes_metadata(tmp_path: Path):
    source = tmp_path / "MSTDB.dat"
    out = tmp_path / "MSTDB.sanitized.dat"
    metadata = tmp_path / "MSTDB.sanitized.metadata.json"
    source.write_text(" System Na-U-Cl L\nU[CN=VI]+3.0 U[DIMER]+6.0\n", encoding="utf-8")

    result = mivm_main(["mstdb-sanitize", "--input", str(source), "--output", str(out), "--metadata", str(metadata)])

    assert result is not None
    assert out.exists()
    assert metadata.exists()
    assert "U+3.0" in out.read_text(encoding="utf-8")
    assert "U2+6.0" in out.read_text(encoding="utf-8")


def test_parse_formula_counts_for_halide_endmembers():
    assert parse_formula_counts("NaCl") == {"NA": 1.0, "CL": 1.0}
    assert parse_formula_counts("UCl3") == {"U": 1.0, "CL": 3.0}


def test_mivm_pycalphad_bridge_writes_model_helper(tmp_path: Path):
    path = tmp_path / "params.json"
    write_simple_params(path)
    outdir = tmp_path / "bridge"

    mivm_main(["pycalphad-bridge", "--params", str(path), "--outdir", str(outdir)])

    helper = (outdir / "mivm_pycalphad_bridge.py").read_text(encoding="utf-8")
    assert "make_pycalphad_model_class" in helper
    assert (outdir / "mivm_parameters.json").exists()


def write_mivm_database(root: Path) -> Path:
    params_dir = root / "data" / "parameter_sets"
    params_dir.mkdir(parents=True)
    write_simple_params(params_dir / "toy.json")
    rows = {
        "component_mstdb_map.csv": [
            {
                "parameter_set_id": "toy",
                "subgroup_id": "toy_group",
                "component": "A",
                "mstdb_phase": "MSCL",
                "mstdb_species_aliases": "ACl",
                "role": "toy",
            }
        ],
        "needed_parameter_checklist.csv": [
            {
                "subgroup_id": "toy_group",
                "system": "A-B",
                "priority": "high",
                "status": "needs_fit",
                "needed_data": "Hmix",
            }
        ],
    }
    for filename, table_rows in rows.items():
        with (root / "data" / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
    database = {
        "schema": "atomi.copilot.mivm.parameter_database.v0.1",
        "mivm_parameter_schema": "atomi.calphad.mivm.parameters.v1",
        "subgroups": [{"id": "toy_group", "halide": "Cl", "status": "test", "label": "toy"}],
        "parameter_sets": [
            {
                "id": "toy",
                "subgroup_id": "toy_group",
                "confidence": "test",
                "components": ["A", "B"],
                "parameter_file": "data/parameter_sets/toy.json",
            }
        ],
        "tables": {
            "component_mstdb_map": "data/component_mstdb_map.csv",
            "needed_parameter_checklist": "data/needed_parameter_checklist.csv",
        },
    }
    db_path = root / "mivm_parameter_database.json"
    db_path.write_text(json.dumps(database), encoding="utf-8")
    return db_path


def test_mivm_database_lists_parameter_sets(tmp_path: Path, capsys):
    db_path = write_mivm_database(tmp_path)

    mivm_main(["database", "--db", str(db_path), "list", "--component", "A"])

    out = capsys.readouterr().out
    assert "toy_group" in out
    assert "data/parameter_sets/toy.json" in out


def test_mivm_database_maps_targets_and_validates(tmp_path: Path, capsys):
    db_path = write_mivm_database(tmp_path)

    mivm_main(["database", "--db", str(db_path), "map", "--component", "A"])
    out = capsys.readouterr().out
    assert "ACl" in out
    assert "MSCL" in out

    mivm_main(["database", "--db", str(db_path), "targets", "--priority", "high"])
    out = capsys.readouterr().out
    assert "A-B" in out
    assert "Hmix" in out

    mivm_main(["database", "--db", str(db_path), "validate-all"])
    out = capsys.readouterr().out
    assert "PASS\ttoy" in out
