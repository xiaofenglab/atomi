from __future__ import annotations

from pathlib import Path

from atomi.core.run_record import (
    PhysicalGuardResult,
    RunArtifact,
    RunRecord,
    read_run_records_jsonl,
    write_run_records_jsonl,
)


def test_run_record_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    record = RunRecord(
        record_id="pocc_stage2_case01",
        engine="vasp",
        run_type="fixed_cell_relax",
        path="cases/case_01",
        status="complete",
        job_id="21767924",
        system_name="Gd-UO2",
        composition={"x_Gd": 1 / 32, "h_U5": 1 / 32},
        energy_eV=-100.0,
        artifacts=[RunArtifact(path="case_01.tgz", role="vasp_archive", exists=True)],
        guards=[PhysicalGuardResult(name="gd_moment", status="pass", value=7.05, threshold="near +/-7")],
    )

    write_run_records_jsonl(path, [record])
    loaded = read_run_records_jsonl(path)

    assert loaded[0].job_id == "21767924"
    assert loaded[0].artifacts[0].role == "vasp_archive"
    assert loaded[0].guards[0].status == "pass"
    assert loaded[0].passed_required_guards()


def test_run_record_failed_guard_is_not_promotable() -> None:
    record = RunRecord(
        record_id="bad_spin",
        engine="vasp",
        run_type="static",
        path="bad",
        guards=[PhysicalGuardResult(name="u5_moment", status="failed", value=0.0)],
    )

    assert not record.passed_required_guards()
