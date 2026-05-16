from __future__ import annotations

import json

from atomi.zentropy import workflow


def test_zentropy_workflow_init_writes_stage_framework(tmp_path) -> None:
    outdir = tmp_path / "gduo2_zentropy"

    workflow.main(
        [
            "init",
            "--outdir",
            str(outdir),
            "--material",
            "(Gd,U)O2",
            "--parent-formula",
            "UO2",
            "--guest-cation",
            "Gd",
        ]
    )

    manifest = json.loads((outdir / "zentropy_workflow.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == workflow.SCHEMA
    assert manifest["runtime_strategy"]["preferred_package"] == "pyzentropy"
    assert manifest["stage_order"][0] == "stage1_motif_db"
    assert (outdir / "stage_manifest.csv").exists()
    for stage_id in manifest["stage_order"]:
        stage_config = json.loads((outdir / stage_id / "stage_config.json").read_text(encoding="utf-8"))
        assert stage_config["stage"]["id"] == stage_id


def test_zentropy_workflow_stages_json(capsys) -> None:
    workflow.main(["stages", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"][0]["id"] == "stage1_motif_db"
    assert any(stage["id"] == "stage5_active_learning" for stage in payload["stages"])
