import csv
import json
from pathlib import Path

import pytest

from atomi.calphad.mivm import (
    excess_enthalpy_j_mol,
    excess_gibbs_j_mol,
    load_parameters,
    main as mivm_main,
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
    assert sanity["counts"]["CHEMSAGE_SYSTEM"] == 1
    assert sanity["warnings"]


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
