from __future__ import annotations

import pytest

from atomi.md import pdfxrd_manual, pdfxrd_run


def test_pdfxrd_manual_static_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        pdfxrd_manual.main(["static", "--help"])
    assert exc.value.code == 0
    assert "--structure" in capsys.readouterr().out


def test_pdfxrd_manual_md_frame_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        pdfxrd_manual.main(["md-frame", "--help"])
    assert exc.value.code == 0
    assert "--engine" in capsys.readouterr().out


def test_pdfxrd_run_forwards_to_phase_order_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from atomi.md import phase_order_guard

    captured: dict[str, list[str]] = {}

    def fake_main(argv: list[str] | None = None) -> dict[str, list[str]]:
        captured["argv"] = list(argv or [])
        return captured

    monkeypatch.setattr(phase_order_guard, "main", fake_main)
    result = pdfxrd_run.main(["bragg-frame", "--help"])
    assert result == {"argv": ["bragg-frame", "--help"]}
