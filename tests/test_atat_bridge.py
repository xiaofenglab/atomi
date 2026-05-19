from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from atomi.cli.main import main as atomi_main
from atomi.atat import bridge


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_atat_status_detects_fake_tools(tmp_path: Path, monkeypatch) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for tool in ("mcsqs", "corrdump", "str2poscar"):
        path = fake_bin / tool
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    report = bridge.inspect_atat_environment()

    assert report["executables"]["sqs"]["mcsqs"]["available"] is True
    assert report["ready"]["can_generate_sqs"] is True
    assert report["ready"]["can_convert_structures"] is True
    assert report["ready"]["can_fit_cluster_expansion"] is False


def test_atat_bridge_init_writes_stage_and_species_maps(tmp_path: Path) -> None:
    out = tmp_path / "atat"

    atomi_main(["atat-bridge", "init", "--outdir", str(out), "--system", "(Gd,U)O2-x"])

    species = rows(out / "pseudo_species_map.csv")
    labels = {row["pseudo_species"] for row in species}
    assert {"U4p", "U4m", "U5p", "U5m", "Gdp", "Gdm", "O", "V_O"}.issubset(labels)
    gd = [row for row in species if row["pseudo_species"] == "Gdp"][0]
    assert gd["moment_guard"] == "Gd=7,-7@0.6"
    stages = rows(out / "atat_atomi_stage_map.csv")
    assert stages[0]["stage_id"] == "01_motif_search"
    plan = json.loads((out / "atat_bridge_plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == bridge.SCHEMA
    assert "sqs" in plan["workflow_stages"][2]["stage_id"]
    assert (out / "ATAT_ATOMI_BRIDGE_NOTES.md").exists()


def test_atat_bridge_index_collects_candidate_files(tmp_path: Path) -> None:
    root = tmp_path / "atat_runs"
    (root / "sqs").mkdir(parents=True)
    (root / "sqs" / "bestsqs.out").write_text("structure\n", encoding="utf-8")
    (root / "enum").mkdir()
    (root / "enum" / "str0001.out").write_text("structure\n", encoding="utf-8")
    out = tmp_path / "atat_candidate_index.csv"

    bridge.index_main(["--root", str(root), "--out", str(out)])

    data = rows(out)
    assert len(data) == 2
    assert {row["source_kind"] for row in data} == {"sqs", "enumerated_structure"}
    assert all(row["atomi_next"] == "convert_to_vasp_then_vasp-branch-live" for row in data)


def test_atat_doctor_json_cli(capsys, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", os.fspath(tmp_path))

    atomi_main(["atat-doctor", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == bridge.STATUS_SCHEMA
    assert "can_generate_sqs" in payload["ready"]


def test_atat_ce_handoff_filters_physics_accepted_rows(tmp_path: Path) -> None:
    summary = tmp_path / "stage1_branch_summary.csv"
    fields = [
        "branch_id",
        "run_dir",
        "output_run_dir",
        "energy_eV",
        "physics_guard_status",
        "mag_status",
        "action",
        "element_order",
        "changed_by_element",
        "frame_id",
    ]
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "branch_id": "spin_good",
                "run_dir": str(tmp_path / "runs" / "spin_good"),
                "output_run_dir": str(tmp_path / "runs" / "spin_good"),
                "energy_eV": "-10.0",
                "physics_guard_status": "OK",
                "mag_status": "OK",
                "action": "continue",
                "element_order": '{"Gd": "FM"}',
                "changed_by_element": '{"U": 1}',
                "frame_id": "frame0",
            }
        )
        writer.writerow(
            {
                "branch_id": "spin_bad",
                "run_dir": str(tmp_path / "runs" / "spin_bad"),
                "output_run_dir": str(tmp_path / "runs" / "spin_bad"),
                "energy_eV": "-11.0",
                "physics_guard_status": "FAIL",
                "mag_status": "OK",
                "action": "continue",
                "frame_id": "frame0",
            }
        )
        writer.writerow(
            {
                "branch_id": "spin_stop",
                "run_dir": str(tmp_path / "runs" / "spin_stop"),
                "output_run_dir": str(tmp_path / "runs" / "spin_stop"),
                "energy_eV": "-12.0",
                "physics_guard_status": "OK",
                "mag_status": "OK",
                "action": "stop",
                "frame_id": "frame0",
            }
        )
    candidate_index = tmp_path / "atat_candidate_index.csv"
    candidate_index.write_text(
        "candidate_id,path,source_kind,target_stage,atomi_next,notes\n"
        f"atat_0001,{(tmp_path / 'runs' / 'spin_good').resolve()},enumerated_structure,fail_fast,next,\n",
        encoding="utf-8",
    )
    out = tmp_path / "ce"

    bridge.ce_handoff_main(
        [
            "--summary-csv",
            str(summary),
            "--candidate-index",
            str(candidate_index),
            "--outdir",
            str(out),
            "--formula-units",
            "2",
        ]
    )

    training = rows(out / "ce_training_set.csv")
    assert len(training) == 1
    assert training[0]["training_id"] == "ce_00001"
    assert training[0]["energy_eV_per_fu"] == "-5"
    assert training[0]["relative_energy_eV_per_fu"] == "0"
    assert training[0]["physics_status"] == "OK"
    assert training[0]["atat_candidate_id"] == "atat_0001"
    manifest = json.loads((out / "atat_ce_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "atomi.atat.ce_handoff.v1"
    assert manifest["n_training_rows"] == 1
    assert (out / "atat_mc_population_handoff_template.csv").exists()
    assert (out / "sd_dd_interaction_template.csv").exists()


def test_atat_ce_handoff_cli_accepts_unchecked_rows(tmp_path: Path, capsys) -> None:
    summary = tmp_path / "spin_energy_run_summary.csv"
    summary.write_text(
        "index,run,status,energy_eV,physics_guard_status,mag_status,element_order,changed_by_element\n"
        "1,spin_001,OK,-3.0,NOT_APPLIED,OK,\"{}\",\"{}\"\n",
        encoding="utf-8",
    )
    out = tmp_path / "ce"

    atomi_main(
        [
            "atat-bridge",
            "ce-handoff",
            "--summary-csv",
            str(summary),
            "--outdir",
            str(out),
            "--include-unchecked",
        ]
    )

    assert "CE training rows" in capsys.readouterr().out
    training = rows(out / "ce_training_set.csv")
    assert len(training) == 1
    assert training[0]["mag_status"] == "OK"
