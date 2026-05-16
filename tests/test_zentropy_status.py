from __future__ import annotations

import json
import sys

from atomi.zentropy import status


def test_zentropy_status_reports_active_environment(monkeypatch) -> None:
    monkeypatch.delenv("ATOMI_ZENTROPY_PYTHON", raising=False)
    monkeypatch.delenv("ATOMI_ZENTROPY_ENV", raising=False)
    monkeypatch.delenv("ATOMI_ZENTROPY_EXE", raising=False)

    report = status.build_zentropy_status()

    assert report["active_environment"]["python"] == sys.executable
    assert report["zentropy_mode"] in {"active-python", "external-python", "executable", "missing"}
    assert report["suggestions"]


def test_zentropy_external_python_probe_returns_json() -> None:
    report = status.probe_zentropy_python(sys.executable, timeout=20)

    assert report["configured"] is True
    assert report["requested_python"] == sys.executable
    assert "pyzentropy" in report
    assert "zentropy" in report


def test_zentropy_status_json_cli(capsys, monkeypatch) -> None:
    monkeypatch.delenv("ATOMI_ZENTROPY_PYTHON", raising=False)
    monkeypatch.delenv("ATOMI_ZENTROPY_ENV", raising=False)
    monkeypatch.delenv("ATOMI_ZENTROPY_EXE", raising=False)

    status.main(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert "active_environment" in payload
    assert "ready_for_zentropy_runtime" in payload
