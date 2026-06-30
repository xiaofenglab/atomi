import argparse
import json
import subprocess
import sys
from pathlib import Path

from atomi.md import gsasii_bridge


def test_probe_gsasii_parses_scriptable_status(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='startup line\n{"available": true, "origin": "/private/GSASII/GSASIIscriptable.py", "has_G2Project": true}\n',
            stderr="",
        )

    monkeypatch.setattr(gsasii_bridge.subprocess, "run", fake_run)

    report = gsasii_bridge.probe_gsasii("/private/gsas2/bin/python", "/private/gsas2")

    assert report["available"] is True
    assert report["has_G2Project"] is True
    assert report["python"] == "/private/gsas2/bin/python"
    assert report["gsasii_root"] == "/private/gsas2"
    assert report["returncode"] == 0


def test_write_project_writes_editable_script_and_metadata(tmp_path: Path) -> None:
    histogram = tmp_path / "exp.xye"
    instrument = tmp_path / "inst.prm"
    phase = tmp_path / "seed.cif"
    histogram.write_text("10 1 0.1\n", encoding="utf-8")
    instrument.write_text("# instrument\n", encoding="utf-8")
    phase.write_text("data_seed\n", encoding="utf-8")
    outdir = tmp_path / "gsasii_fit"

    args = argparse.Namespace(
        histogram=histogram,
        instrument=instrument,
        phase=phase,
        outdir=outdir,
        python="/private/gsas2/bin/python",
        gsasii_root="/private/gsas2",
        script_name="fit.py",
        project_name="fit.gpx",
        summary_name="fit_summary.json",
        phase_name="phase_a",
        histogram_format="",
        phase_format="",
        background_type="chebyschev-1",
        background_coeffs=4,
        tth_min=15.0,
        tth_max=90.0,
        refine_cell=True,
        refine_profile=True,
        refine_zero=True,
        refine_sample_displacement=False,
    )

    metadata = gsasii_bridge.write_project_main(args)
    script = (outdir / "fit.py").read_text(encoding="utf-8")
    run_script = (outdir / "run_gsasii_xrd_fit.sh").read_text(encoding="utf-8")
    project_json = json.loads((outdir / "gsasii_bridge_project.json").read_text(encoding="utf-8"))

    assert "GSASII.GSASIIscriptable" in script
    assert "do_refinements" in script
    assert '"Cell": True' in script
    assert '"Instrument Parameters": ["U", "V", "W", "X", "Y"]' in script
    assert "ATOMI_GSASII_ROOT=/private/gsas2" in run_script
    assert metadata["script"] == project_json["script"]
    assert project_json["backend"].startswith("GSAS-II")


def test_install_plan_recommends_external_runtime(capsys) -> None:
    args = argparse.Namespace(json=False)

    payload = gsasii_bridge.install_plan_main(args)
    captured = capsys.readouterr()

    assert "separate" in payload["recommendation"]
    assert "GSAS-II runtime" in payload["recommendation"]
    assert any("No GPU" in item for item in payload["why"])
    assert "GSAS-II / Atomi HPC install plan" in captured.out


def test_standalone_status_cli_returns_zero(monkeypatch, capsys) -> None:
    def fake_probe(python=None, gsasii_root=None):
        return {
            "available": True,
            "python": "/private/gsas2/bin/python",
            "gsasii_root": "/private/gsas2",
            "origin": "/private/gsas2/GSAS-II/GSASII/GSASIIscriptable.py",
        }

    monkeypatch.setattr(gsasii_bridge, "probe_gsasii", fake_probe)
    monkeypatch.setattr(sys, "argv", ["gsasii-status"])

    assert gsasii_bridge.status_cli() == 0
    captured = capsys.readouterr()
    assert "GSAS-II bridge status" in captured.out
    assert "import      : ok" in captured.out
