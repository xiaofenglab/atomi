from pathlib import Path

from atomi.calphad.env import _parse_csv, inspect_calphad_environment
from atomi.moose.env import inspect_moose_environment


def test_moose_environment_reports_requested_app(monkeypatch, tmp_path: Path) -> None:
    app = tmp_path / "my_moose_app-opt"
    app.write_text("#!/bin/sh\n", encoding="utf-8")
    app.chmod(0o755)

    monkeypatch.setenv("MOOSE_DIR", "/apps/moose")
    monkeypatch.setattr("atomi.moose.env.shutil.which", lambda name: None)

    report = inspect_moose_environment(app=str(app))

    assert report["module"] == "moose"
    assert report["app"] == str(app)
    assert report["executables"][str(app)]["available"] is True
    assert report["executables"][str(app)]["path"] == str(app)
    assert report["environment"]["MOOSE_DIR"] == "/apps/moose"


def test_calphad_environment_without_database_records_requested_scope() -> None:
    report = inspect_calphad_environment(
        components=_parse_csv("U,O,VA"),
        phases=_parse_csv("FLUORITE,LIQUID"),
    )

    assert report["module"] == "calphad"
    assert "available" in report["pycalphad"]
    assert report["database"] is None
    assert report["requested_components"] == ["U", "O", "VA"]
    assert report["requested_phases"] == ["FLUORITE", "LIQUID"]


def test_calphad_missing_database_is_reported(tmp_path: Path) -> None:
    report = inspect_calphad_environment(database=tmp_path / "missing.tdb")

    assert report["database"]["path"].endswith("missing.tdb")
    assert report["database"]["exists"] is False
    assert report["database"]["parsed"] is False
