from __future__ import annotations

from atomi.elastic.status import format_status, main, probe


def test_elate_status_reports_native_backend(capsys) -> None:
    data = main([])
    captured = capsys.readouterr()

    assert data["native_backend_available"] is True
    assert "Atomi Elastic Visualization status" in captured.out
    assert "native backend: yes" in captured.out


def test_elate_status_json(capsys) -> None:
    data = main(["--json"])
    captured = capsys.readouterr()

    assert data["python"]
    assert '"native_backend_available": true' in captured.out


def test_elate_status_formatter_handles_missing_elate() -> None:
    data = probe()
    data["elate"]["available"] = False
    data["elate"]["elastic_class_available"] = False
    text = format_status(data)

    assert "ELATE: missing" in text
    assert "--backend auto" in text
