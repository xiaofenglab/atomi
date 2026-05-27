import json
import subprocess

import pytest

from atomi.reporting import prismml_export


def test_prismml_export_scans_and_classifies_artifacts(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    files = {
        "plot.png": "plot",
        "table.csv": "a,b\n1,2\n",
        "POSCAR": "poscar\n",
        "notes.md": "# note\n",
        "ignore.bin": "raw",
    }
    for name, text in files.items():
        (run / name).write_text(text, encoding="utf-8")

    artifacts = prismml_export.scan_artifacts([run], limit=10)

    by_name = {artifact.path.name: artifact.kind for artifact in artifacts}
    assert by_name == {
        "POSCAR": "structure",
        "notes.md": "note",
        "plot.png": "plot",
        "table.csv": "table",
    }


def test_prismml_export_writes_deterministic_jsonl(tmp_path):
    plot = tmp_path / "plot.svg"
    plot.write_text("<svg />\n", encoding="utf-8")
    artifacts = prismml_export.scan_artifacts([plot], limit=10)
    records = prismml_export.build_prompt_records(
        artifacts,
        project_title="UO2 Transport",
        material="uranium oxide",
        style="clean talk visual",
        size="1024x1024",
        steps=4,
        seed_start=4201,
    )
    out = tmp_path / "prompts.jsonl"

    prismml_export.write_jsonl(out, records)

    parsed = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [record["seed"] for record in parsed] == [4201, 4202, 4203, 4204, 4205]
    assert parsed[0]["producer"] == "atomi-prismml-export"
    assert parsed[0]["output"] == "outputs/atomi_bridge/uo2_transport_title_visual.png"
    assert parsed[0]["atomi_artifacts"][0]["kind"] == "plot"
    assert "fake plot axes" in parsed[0]["negative_prompt"]


def test_prismml_export_missing_run_path_has_clean_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Atomi run path does not exist"):
        prismml_export.scan_artifacts([tmp_path / "missing"], limit=10)


def test_prismml_export_writes_shell_batch(tmp_path):
    records = prismml_export.build_prompt_records(
        [],
        project_title="Atomi Talk",
        material="UO2",
        style="conference",
        size="512x512",
        steps=2,
        seed_start=10,
    )
    batch = tmp_path / "run_prismml_batch.sh"

    prismml_export.write_prismml_batch(batch, records, tmp_path / "Bonsai-Image-Demo")

    assert batch.read_text(encoding="utf-8").startswith("#!/bin/sh\nset -eu\n")
    subprocess.run(["sh", "-n", str(batch)], check=True)


def test_prismml_export_standard_library_only():
    source = prismml_export.__loader__.get_source(prismml_export.__name__)
    assert source is not None
    forbidden = ["torch", "cuda", "transformers", "huggingface", "prismml", "node"]
    lowered = source.lower()
    assert not any(f"import {name}" in lowered or f"from {name}" in lowered for name in forbidden)
