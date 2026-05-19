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
