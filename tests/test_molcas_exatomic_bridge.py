from __future__ import annotations

from atomi.qchem import exatomic_bridge


def test_install_plan_keeps_exatomic_optional() -> None:
    plan = exatomic_bridge.install_plan()

    assert plan["schema"] == "atomi.molcas_exatomic_install_plan.v1"
    assert "optional" in plan["recommendation"].lower()
    assert "nbo_program" in plan["roles"]
    assert "kit_justus2_m_lammps_env" in plan["commands"]
    assert "wsu_kamiak_private_molcas_tools" in plan["commands"]


def test_probe_missing_python_reports_unavailable() -> None:
    report = exatomic_bridge.probe_exatomic("/definitely/not/a/python")

    assert report["schema"] == "atomi.molcas_exatomic_status.v1"
    assert report["available"] is False
    assert report["resolved_python"] == ""


def test_status_cli_json_runs_without_exatomic(capsys) -> None:
    rc = exatomic_bridge.status_cli(["--python", "/definitely/not/a/python", "--json"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "atomi.molcas_exatomic_status.v1" in captured.out
    assert "/definitely/not/a/python" in captured.out
