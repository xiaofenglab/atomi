from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomi.qchem import molcas_live


INPUT_TEXT = """
&RASSCF
Title
 UO9 U4+ C2v ground A1
Symmetry
 1
Spin
 3
nActEl
 2 0 0
CIROOTS
 3 3 1
SUPSYM
 1
  2 34 35
ORBL
 ALL
ORBA
 FULL
End of input

&CASPT2
Title
 UO9 U4+ C2v ground A1 PT2
MULTISTATE
 3 1 2 3
End of input

&RASSCF
Title = UO9 U4+ C2v M5 excited A1
Symmetry = 1
Spin = 3
nActEl = 2 0 0
CIROOTS = 4 4 1
ORBL
 ALL
ORBA
 FULL
End of input

&CASPT2
Title = UO9 U4+ C2v M5 excited A1 PT2
XMUL = ALL
End of input

&RASSI
Title
 UO9 U4+ C2v M5 spin-orbit coupling
SpinOrbit
Nr of JobIph files:
 2 3 4
 1 2 3
 1 2 3 4
End of input
""".lstrip()


def _rasscf_started(title: str, roots: int, completed_roots: int) -> str:
    root_lines = "".join(
        f"::    RASSCF root number {root:2d} Total energy: {-4100.0 - root:.8f}\n"
        for root in range(1, completed_roots + 1)
    )
    return (
        "--- Start Module: rasscf at Thu Aug 6 00:00:00 2026 ---\n"
        f"      Title: {title}\n"
        "      Number of electrons in active shells       2\n"
        "      Spin quantum number                      1.0\n"
        "      State symmetry                             1\n"
        f"      Number of root(s) required                 {roots}\n"
        "      RASSCF iterations: Energy and convergence statistics\n"
        "      Convergence after  8 iterations\n"
        f"{root_lines}"
    )


def _caspt2_started(title: str, completed_roots: int, weight: float = 0.62) -> str:
    rows = "".join(
        f"      Reference weight:           {weight:.5f}\n"
        f"::    XMS-CASPT2 Root {root:3d} Total energy: {-4100.0 - root:.8f}\n"
        for root in range(1, completed_roots + 1)
    )
    return f"--- Start Module: caspt2 at Thu Aug 6 00:10:00 2026 ---\n      Title: {title}\n{rows}"


def _finished(module: str, rc: str = "_RC_ALL_IS_WELL_") -> str:
    return (
        f"--- Stop Module: {module} at Thu Aug 6 00:20:00 2026 /rc={rc} ---\n"
        f"--- Module {module} spent 5 seconds ---\n"
    )


def _write_case(tmp_path: Path, output_text: str) -> tuple[Path, Path]:
    inp = tmp_path / "uo9.inp"
    output = tmp_path / "uo9.log"
    inp.write_text(INPUT_TEXT, encoding="utf-8")
    output.write_text(output_text, encoding="utf-8")
    return inp, output


def _block(snapshot: dict, index: int) -> dict:
    return next(item for item in snapshot["blocks"] if item["index"] == index)


def test_parse_input_plan_preserves_order_and_roots() -> None:
    plan = molcas_live.parse_input_plan(INPUT_TEXT)

    assert [item.kind for item in plan] == ["rasscf", "caspt2", "rasscf", "caspt2", "rassi"]
    assert [item.index for item in plan] == [1, 2, 3, 4, 5]
    assert plan[0].title == "UO9 U4+ C2v ground A1"
    assert plan[0].expected_roots == 3
    assert plan[1].expected_roots == 3
    assert plan[2].expected_roots == 4
    assert plan[3].expected_roots == 4
    assert plan[4].rassi_state_counts == (3, 4)
    assert plan[4].expected_roots == 7


def test_snapshot_tracks_active_rasscf_roots_and_signature(tmp_path: Path) -> None:
    output_text = (
        _rasscf_started("ground", 3, 3)
        + _finished("rasscf")
        + _caspt2_started("ground PT2", 3)
        + _finished("caspt2")
        + _rasscf_started("M5", 4, 2)
    )
    inp, output = _write_case(tmp_path, output_text)

    snapshot = molcas_live.build_snapshot(inp, output)

    assert snapshot["schema"] == "atomi.molcas_live.v1"
    assert snapshot["overall_status"] == "running"
    assert snapshot["active_block_index"] == 3
    assert _block(snapshot, 1)["status"] == "finished"
    assert _block(snapshot, 2)["status"] == "finished"
    active = _block(snapshot, 3)
    assert active["progress"] == {
        "completed": 2,
        "total": 4,
        "percent": 50.0,
        "label": "roots",
    }
    signature = next(item for item in active["guards"] if item["name"] == "input/output signature")
    assert signature["level"] == "pass"


def test_snapshot_tracks_active_caspt2_and_reference_weight(tmp_path: Path) -> None:
    output_text = (
        _rasscf_started("ground", 3, 3)
        + _finished("rasscf")
        + _caspt2_started("ground PT2", 2, weight=0.49)
    )
    inp, output = _write_case(tmp_path, output_text)

    snapshot = molcas_live.build_snapshot(inp, output)
    active = _block(snapshot, 2)

    assert active["progress"]["completed"] == 2
    assert active["progress"]["total"] == 3
    assert active["progress"]["percent"] == pytest.approx(66.6666666667)
    weight_guard = next(item for item in active["guards"] if item["name"] == "reference weights")
    assert weight_guard["level"] == "warn"


def test_snapshot_reports_rassi_stage_without_fake_so_percentage(tmp_path: Path) -> None:
    output_text = (
        _rasscf_started("ground", 3, 3)
        + _finished("rasscf")
        + _caspt2_started("ground PT2", 3)
        + _finished("caspt2")
        + _rasscf_started("M5", 4, 4)
        + _finished("rasscf")
        + _caspt2_started("M5 PT2", 4)
        + _finished("caspt2")
        + "--- Start Module: rassi at Thu Aug 6 00:30:00 2026 ---\n"
        + "      Constructing the spin-orbit Hamiltonian\n"
        + "::    SO-RASSI State    1     Total energy: -4100.0\n"
    )
    inp, output = _write_case(tmp_path, output_text)

    snapshot = molcas_live.build_snapshot(inp, output)
    rassi = _block(snapshot, 5)

    assert rassi["status"] == "running"
    assert rassi["runtime"]["stage"] == "writing spin-orbit states"
    assert rassi["expected_roots"] == 7
    assert rassi["progress"] == {
        "completed": 1,
        "total": None,
        "percent": None,
        "label": "SO states",
    }


def test_pending_rassi_does_not_use_spin_free_states_as_so_denominator(tmp_path: Path) -> None:
    inp, output = _write_case(tmp_path, "")

    snapshot = molcas_live.build_snapshot(inp, output)
    rassi = _block(snapshot, 5)

    assert rassi["status"] == "pending"
    assert rassi["expected_roots"] == 7
    assert rassi["progress"] == {
        "completed": 0,
        "total": None,
        "percent": None,
        "label": "SO states",
    }


def test_duplicate_root_records_are_counted_once(tmp_path: Path) -> None:
    output_text = _rasscf_started("ground", 3, 2)
    output_text += "::    RASSCF root number  2 Total energy: -4102.00000000\n"
    inp, output = _write_case(tmp_path, output_text)

    snapshot = molcas_live.build_snapshot(inp, output)

    assert _block(snapshot, 1)["progress"]["completed"] == 2


def test_live_session_reads_only_appended_bytes(tmp_path: Path) -> None:
    initial = _rasscf_started("ground", 3, 1)
    inp, output = _write_case(tmp_path, initial)
    session = molcas_live.MocLiveSession(inp, output)

    first = session.refresh()
    assert first["incremental"]["last_read_bytes"] == len(initial.encode())
    assert _block(first, 1)["progress"]["completed"] == 1

    appended = "::    RASSCF root number  2 Total energy: -4102.00000000\n"
    with output.open("a", encoding="utf-8") as handle:
        handle.write(appended)
    second = session.refresh()

    assert second["incremental"]["last_read_bytes"] == len(appended.encode())
    assert second["incremental"]["total_read_bytes"] == output.stat().st_size
    assert _block(second, 1)["progress"]["completed"] == 2

    third = session.refresh()
    assert third["incremental"]["last_read_bytes"] == 0
    assert _block(third, 1)["progress"]["completed"] == 2


def test_live_session_resets_after_output_truncation(tmp_path: Path) -> None:
    inp, output = _write_case(tmp_path, _rasscf_started("ground", 3, 2))
    session = molcas_live.MocLiveSession(inp, output)
    session.refresh()

    replacement = _rasscf_started("ground restarted", 3, 1)
    output.write_text(replacement, encoding="utf-8")
    snapshot = session.refresh()

    assert snapshot["incremental"]["reset_count"] == 1
    assert snapshot["incremental"]["last_read_bytes"] == len(replacement.encode())
    assert _block(snapshot, 1)["progress"]["completed"] == 1


def test_snapshot_promotes_success_and_surfaces_failure(tmp_path: Path) -> None:
    complete_text = (
        _rasscf_started("ground", 3, 3)
        + _finished("rasscf")
        + _caspt2_started("ground PT2", 3)
        + _finished("caspt2")
        + _rasscf_started("M5", 4, 4)
        + _finished("rasscf")
        + _caspt2_started("M5 PT2", 4)
        + _finished("caspt2")
        + "--- Start Module: rassi at Thu Aug 6 00:30:00 2026 ---\n"
        + "++ Dipole transition strengths (SO states):\n"
        + _finished("rassi")
        + ".# Happy landing! #.\n"
    )
    inp, output = _write_case(tmp_path, complete_text)
    completed = molcas_live.build_snapshot(inp, output)

    assert completed["overall_status"] == "complete"
    assert completed["active_block_index"] is None

    failed_output = tmp_path / "failed.log"
    failed_output.write_text(
        _rasscf_started("ground", 3, 1) + _finished("rasscf", "_RC_NOT_CONVERGED_"),
        encoding="utf-8",
    )
    failed = molcas_live.build_snapshot(inp, failed_output)

    assert failed["overall_status"] == "failed"
    assert failed["active_block_index"] == 1
    assert _block(failed, 1)["status"] == "failed"


def test_render_snapshot_has_unicode_and_ascii_status_modes(tmp_path: Path) -> None:
    inp, output = _write_case(
        tmp_path,
        _rasscf_started("ground", 3, 3) + _finished("rasscf") + _caspt2_started("ground PT2", 2),
    )
    snapshot = molcas_live.build_snapshot(inp, output)

    unicode_report = molcas_live.render_snapshot(snapshot, spinner_frame=0, bar_width=12)
    ascii_report = molcas_live.render_snapshot(
        snapshot, ascii_only=True, spinner_frame=0, bar_width=12
    )

    assert "MOCLIVE" in unicode_report
    assert "\u2713" in unicode_report
    assert "\u280b" in unicode_report
    assert "2/3 roots 66.7%" in unicode_report
    assert "Health / physical guards" in unicode_report
    assert "[OK]" in ascii_report
    assert "[|]" in ascii_report
    assert "\u2713" not in ascii_report


def test_main_supports_one_shot_json_and_json_out(tmp_path: Path, capsys) -> None:
    inp, output = _write_case(tmp_path, _rasscf_started("ground", 3, 2))
    json_out = tmp_path / "moclive.json"

    rc = molcas_live.main([str(inp), str(output), "--once", "--json", "--json-out", str(json_out)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "atomi.molcas_live.v1"
    assert payload["overall_status"] == "running"
    assert payload["active_block_index"] == 1
    assert json.loads(json_out.read_text(encoding="utf-8"))["summary"]["running"] == 1
