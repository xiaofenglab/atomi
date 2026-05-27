from pathlib import Path
import os
import time

import pytest

from atomi.codes.vasp import missing_inputs, summarize_outcar
from atomi.vasp.checks import (
    check_runs,
    clean_stopped_energy_outputs,
    collect_run_energies,
    vasp_energies,
)
from atomi.viz.lammps import (
    read_thermo_rows,
    summarize_lammps_run_progress,
    summarize_recent_runtime_fraction,
    summarize_thermo,
)
from atomi.viz.vasp_live import _run_gnuplot, count_dav_steps, dav_timing_path, update_dav_timing


def test_missing_inputs_reports_absent_files(tmp_path: Path) -> None:
    (tmp_path / "INCAR").write_text("SYSTEM = test\n", encoding="utf-8")

    assert missing_inputs(tmp_path) == ["POSCAR", "POTCAR", "KPOINTS"]


def test_count_dav_steps(tmp_path: Path) -> None:
    output = tmp_path / "vasp.out"
    output.write_text(
        "running\n"
        "DAV:   1    -0.100000E+02   -0.100E+01   -0.100E+01  10  0.1E-02\n"
        "RMM:   2    -0.110000E+02   -0.100E+00   -0.100E+00  10  0.1E-03\n"
        "DAV:   3    -0.111000E+02   -0.100E-01   -0.100E-01  10  0.1E-04\n",
        encoding="utf-8",
    )

    assert count_dav_steps(output) == 2


def test_count_dav_steps_accepts_indented_vasp_lines(tmp_path: Path) -> None:
    output = tmp_path / "vasp.out"
    output.write_text(
        " running\n"
        "  DAV:   1    -0.100000E+02   -0.100E+01   -0.100E+01\n"
        "\tDAV:   2    -0.110000E+02   -0.100E+00   -0.100E+00\n",
        encoding="utf-8",
    )

    assert count_dav_steps(output) == 2


def test_run_gnuplot_reports_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = tmp_path / "plot.gp"
    script.write_text("bad command\n", encoding="utf-8")

    class Result:
        stdout = "partial plot\n"
        stderr = "line 12: all points y value undefined\n"
        returncode = 1

    def fake_run(*_args, **_kwargs):
        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="all points y value undefined"):
        _run_gnuplot(["file='vasp.out'"], script)


def test_update_dav_timing_excludes_initialization_and_batches_new_dav_steps(tmp_path: Path) -> None:
    output = tmp_path / "vasp.out"
    output.write_text(
        "initialization text\n"
        "DAV:   1    -0.100000E+02   -0.100E+01   -0.100E+01  10  0.1E-02\n",
        encoding="utf-8",
    )

    first = update_dav_timing(output, now=100.0)
    assert first.dav_count == 1
    assert first.timed_steps == 0
    assert first.latest_seconds is None

    output.write_text(
        output.read_text(encoding="utf-8")
        + "DAV:   2    -0.110000E+02   -0.100E+00   -0.100E+00  10  0.1E-03\n",
        encoding="utf-8",
    )
    second = update_dav_timing(output, now=112.0)
    assert second.dav_count == 2
    assert second.timed_steps == 1
    assert second.latest_seconds == pytest.approx(12.0)

    output.write_text(
        output.read_text(encoding="utf-8")
        + "DAV:   3    -0.111000E+02   -0.100E-01   -0.100E-01  10  0.1E-04\n"
        + "DAV:   4    -0.112000E+02   -0.100E-02   -0.100E-02  10  0.1E-05\n",
        encoding="utf-8",
    )
    third = update_dav_timing(output, now=132.0)
    assert third.timed_steps == 3
    assert third.latest_seconds == pytest.approx(10.0)
    assert third.mean_seconds == pytest.approx((12.0 + 10.0 + 10.0) / 3.0)
    timing_text = dav_timing_path(output).read_text(encoding="utf-8")
    assert "first observed DAV count is a baseline" in timing_text
    assert "2 112.000000 12.000000 1" in timing_text
    assert "3 132.000000 10.000000 2" in timing_text
    assert "4 132.000000 10.000000 2" in timing_text


def test_check_runs_marks_stale_output_as_stopped(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "run_A"
    run_b = tmp_path / "run_B"
    run_a.mkdir()
    run_b.mkdir()
    runlist.write_text(f"{run_a}\n{run_b}\n", encoding="utf-8")
    stale = run_a / "vasp.out"
    fresh = run_b / "vasp.out"
    stale.write_text("DAV: 1 -1.0 0 0\n", encoding="utf-8")
    fresh.write_text("DAV: 1 -1.0 0 0\n", encoding="utf-8")
    old_time = time.time() - 20 * 60
    os.utime(stale, (old_time, old_time))

    counts = check_runs(runlist, stopped_after_minutes=10)

    assert counts.stopped == 1
    assert counts.running == 1


def test_check_runs_uses_array_logs_for_stopped_status(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "spin_001"
    run_b = tmp_path / "spin_002"
    run_a.mkdir()
    run_b.mkdir()
    runlist.write_text("spin_001\nspin_002\n", encoding="utf-8")
    stale = tmp_path / "vasp.out_std.12345.1"
    fresh = tmp_path / "vasp.out_std.12346.2"
    stale.write_text("DAV: 1 -1.0 0 0\n", encoding="utf-8")
    fresh.write_text("DAV: 1 -1.0 0 0\n", encoding="utf-8")
    old_time = time.time() - 11 * 60
    os.utime(stale, (old_time, old_time))

    counts = check_runs(runlist)

    assert counts.stopped == 1
    assert counts.running == 1
    assert counts.not_started == 0


def test_summarize_outcar_uses_selected_file(tmp_path: Path) -> None:
    outcar = tmp_path / "OUTCAR.test"
    outcar.write_text(
        " free energy    TOTEN  =       -10.0000 eV\n"
        " total energy-change (2. order) :-0.1\n"
        " E-fermi :  5.4321     XC(G=0):\n"
        " direct lattice vectors                 reciprocal lattice vectors\n"
        " 1.0 0.0 0.0\n"
        " 0.0 1.0 0.0\n"
        " 0.0 0.0 1.0\n"
        " TOTAL-FORCE (eV/Angst)\n"
        " -----------------------------------------------------------------------------------\n"
        " 0.0 0.0 0.0  1.0 0.0 0.0\n"
        " 0.0 0.0 0.0  0.0 2.0 0.0\n"
        " total drift: 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    summary = summarize_outcar(outcar)

    assert summary.final_total_energy_line is not None
    assert "-10.0000" in summary.final_total_energy_line
    assert summary.fermi_energy_line is not None
    assert "5.4321" in summary.fermi_energy_line
    assert summary.max_force == 2.0


def test_collect_run_energies_uses_array_logs(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "run_A"
    run_b = tmp_path / "run_B"
    run_a.mkdir()
    run_b.mkdir()
    runlist.write_text(f"{run_a}\n{run_b}\n", encoding="utf-8")
    (tmp_path / "vasp.out.1").write_text(
        " free  energy   TOTEN  =       -10.000000 eV\n"
        " free  energy   TOTEN  =       -11.500000 eV\n",
        encoding="utf-8",
    )
    (tmp_path / "vasp.out.2").write_text(
        " 1 F= -.12000000E+02 E0= -.12100000E+02 d E =-.1E-01\n",
        encoding="utf-8",
    )

    records = collect_run_energies(runlist, log_dir=tmp_path)

    assert [record.status for record in records] == ["OK", "OK"]
    assert records[0].energy_eV == -11.5
    assert records[0].energy_kind == "toten"
    assert records[1].energy_eV == -12.1
    assert records[1].energy_kind == "e0"


def test_collect_run_energies_uses_dav_energy_from_active_log(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "spin_001"
    run_a.mkdir()
    runlist.write_text("spin_001\n", encoding="utf-8")
    (tmp_path / "vasp.out_std.21317022.1").write_text(
        "DAV: 217    -0.218550013889E+04    0.15528E-02   -0.28211E-02  7104\n"
        "DAV: 218    -0.218548523694E+04    0.14902E-01   -0.12478E-02  6864\n"
        "DAV: 219    -0.218545944662E+04    0.25790E-01   -0.30214E-01  6768\n",
        encoding="utf-8",
    )

    records = collect_run_energies(runlist, log_dir=tmp_path)

    assert records[0].status == "OK"
    expected = (-2185.50013889 - 2185.48523694 - 2185.45944662) / 3
    assert records[0].energy_eV == pytest.approx(expected)
    assert records[0].energy_kind == "dav_avg3"


def test_collect_run_energies_averages_last_ten_dav_steps(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "spin_001"
    run_a.mkdir()
    runlist.write_text("spin_001\n", encoding="utf-8")
    lines = [
        f"DAV: {idx:3d} {-100.0 - idx:.8E} 0.1E-02 -0.1E-02 100\n"
        for idx in range(1, 13)
    ]
    (tmp_path / "vasp.out_std.21317022.1").write_text("".join(lines), encoding="utf-8")

    records = collect_run_energies(runlist, log_dir=tmp_path)

    expected = sum(-100.0 - idx for idx in range(3, 13)) / 10
    assert records[0].status == "OK"
    assert records[0].energy_eV == pytest.approx(expected)
    assert records[0].energy_kind == "dav_avg10"


def test_collect_run_energies_marks_stale_active_log_stopped(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "spin_001"
    run_a.mkdir()
    runlist.write_text("spin_001\n", encoding="utf-8")
    log = tmp_path / "vasp.out_std.21317022.1"
    log.write_text(
        "DAV: 219    -0.218545944662E+04    0.25790E-01   -0.30214E-01  6768\n",
        encoding="utf-8",
    )
    old_time = time.time() - 16 * 60
    os.utime(log, (old_time, old_time))

    records = collect_run_energies(runlist)

    assert records[0].status == "STOPPED"
    assert records[0].energy_eV == -2185.45944662
    assert records[0].energy_kind == "dav_avg1"


def test_collect_run_energies_keeps_recent_active_log_ok(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "spin_001"
    run_a.mkdir()
    runlist.write_text("spin_001\n", encoding="utf-8")
    log = tmp_path / "vasp.out_std.21317022.1"
    log.write_text(
        "DAV: 219    -0.218545944662E+04    0.25790E-01   -0.30214E-01  6768\n",
        encoding="utf-8",
    )
    old_time = time.time() - 11 * 60
    os.utime(log, (old_time, old_time))

    records = collect_run_energies(runlist)

    assert records[0].status == "OK"
    assert records[0].energy_eV == -2185.45944662
    assert records[0].energy_kind == "dav_avg1"


def test_collect_run_energies_falls_back_to_run_folder(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "run_A"
    run_a.mkdir()
    runlist.write_text("run_A\n", encoding="utf-8")
    (run_a / "OUTCAR").write_text(
        " energy  without entropy=      -20.250000  energy(sigma->0) = -20.260000\n",
        encoding="utf-8",
    )

    records = collect_run_energies(runlist, log_dir=tmp_path)

    assert records[0].status == "OK"
    assert records[0].energy_eV == -20.25
    assert records[0].source == run_a / "OUTCAR"


def test_collect_run_energies_prefers_array_stdout_over_completed_outcar(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "spin_001"
    run_a.mkdir()
    runlist.write_text("spin_001\n", encoding="utf-8")
    outcar = run_a / "OUTCAR"
    outcar.write_text(
        " free  energy   TOTEN  =       2200.000000 eV\n",
        encoding="utf-8",
    )
    stdout = tmp_path / "vasp.out_std.21444132.1"
    stdout.write_text(
        " free  energy   TOTEN  =       -11.500000 eV\n",
        encoding="utf-8",
    )
    now = time.time()
    os.utime(stdout, (now - 60.0, now - 60.0))
    os.utime(outcar, (now, now))

    records = collect_run_energies(runlist, log_dir=tmp_path)

    assert records[0].status == "OK"
    assert records[0].energy_eV == -11.5
    assert records[0].source == stdout


def test_collect_run_energies_keeps_default_checkeng_path_shallow(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "spin_001"
    run_a.mkdir()
    runlist.write_text("spin_001\n", encoding="utf-8")
    artifact_run = tmp_path / "bwforcluster-vasp_array.sbatch.12345.1.260518_030213" / "run"
    artifact_run.mkdir(parents=True)
    nested_outcar = artifact_run / "OUTCAR"
    nested_outcar.write_text(" free  energy   TOTEN  =       -7.000000 eV\n", encoding="utf-8")

    shallow = collect_run_energies(runlist, log_dir=tmp_path)
    deep = collect_run_energies(runlist, log_dir=tmp_path, deep_artifacts=True)

    assert shallow[0].status == "NOLOG"
    assert shallow[0].source is None
    assert deep[0].status == "OK"
    assert deep[0].source == nested_outcar
    assert deep[0].energy_eV == -7.0


def test_vasp_energies_prints_tsv(tmp_path: Path, capsys) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "run_A"
    run_a.mkdir()
    runlist.write_text(f"{run_a}\n", encoding="utf-8")
    (tmp_path / "vasp.out.1").write_text(
        " free  energy   TOTEN  =       -5.000000 eV\n",
        encoding="utf-8",
    )

    vasp_energies([str(runlist), "--log-dir", str(tmp_path), "--delimiter", "tab"])

    output = capsys.readouterr().out
    assert "index\trun\tenergy_eV\tkind\tstatus\tsource" in output
    assert "1\t" in output
    assert "-5.0000000000" in output


def test_checkeng_clean_stopped_artifacts_dry_run(tmp_path: Path, capsys) -> None:
    runlist = tmp_path / "runlist.txt"
    run_dir = tmp_path / "spin_001"
    run_dir.mkdir()
    runlist.write_text("spin_001\n", encoding="utf-8")
    artifact_run = tmp_path / "bwforcluster-vasp_array.sbatch.12345.1.260518_030213" / "run"
    artifact_run.mkdir(parents=True)
    outcar = artifact_run / "OUTCAR"
    outcar.write_text(" free  energy   TOTEN  =       -5.000000 eV\n", encoding="utf-8")
    keep = artifact_run / "POSCAR"
    keep.write_text("keep\n", encoding="utf-8")
    old_time = time.time() - 20 * 60
    os.utime(outcar, (old_time, old_time))

    vasp_energies(
        [
            str(runlist),
            "--log-dir",
            str(tmp_path),
            "--stopped-after-min",
            "15",
            "--clean-stopped",
            "--clean-pattern",
            "OUTCAR",
        ]
    )

    output = capsys.readouterr().out
    assert "STOPPED-CLEAN run 1" in output
    assert "would remove:" in output
    assert outcar.exists()
    assert keep.exists()


def test_clean_stopped_energy_outputs_execute_only_stopped_artifacts(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "spin_001"
    run_b = tmp_path / "spin_002"
    run_a.mkdir()
    run_b.mkdir()
    runlist.write_text("spin_001\nspin_002\n", encoding="utf-8")
    stopped_run = tmp_path / "bwforcluster-vasp_array.sbatch.12345.1.260518_030213" / "run"
    active_run = tmp_path / "bwforcluster-vasp_array.sbatch.12346.2.260518_030213" / "run"
    stopped_run.mkdir(parents=True)
    active_run.mkdir(parents=True)
    stopped_outcar = stopped_run / "OUTCAR"
    active_outcar = active_run / "OUTCAR"
    stopped_outcar.write_text(" free  energy   TOTEN  =       -5.000000 eV\n", encoding="utf-8")
    active_outcar.write_text(" free  energy   TOTEN  =       -4.000000 eV\n", encoding="utf-8")
    stopped_keep = stopped_run / "INCAR"
    stopped_keep.write_text("keep\n", encoding="utf-8")
    old_time = time.time() - 20 * 60
    os.utime(stopped_outcar, (old_time, old_time))

    records = collect_run_energies(
        runlist,
        log_dir=tmp_path,
        stopped_after_minutes=15,
        deep_artifacts=True,
    )
    counts = clean_stopped_energy_outputs(
        runlist=runlist,
        records=records,
        log_dir=tmp_path,
        patterns=["OUTCAR"],
        execute=True,
    )

    assert counts.stopped_runs == 1
    assert counts.removed_files == 1
    assert not stopped_outcar.exists()
    assert stopped_keep.exists()
    assert active_outcar.exists()


def test_read_lammps_thermo_rows(tmp_path: Path) -> None:
    log = tmp_path / "log.lammps"
    log.write_text(
        "LAMMPS\n"
        "Step Temp PotEng TotEng Press Volume\n"
        "0 300 -10 -9 100 1000\n"
        "100 310 -11 -10 50 1001\n"
        "Loop time of 1 on 1 procs\n",
        encoding="utf-8",
    )

    rows = read_thermo_rows(log)
    summary = summarize_thermo(rows, last_fraction=1.0)

    assert len(rows) == 2
    assert rows[-1].step == 100
    assert summary.npoints == 2
    assert summary.temp_avg == 305


def test_lammps_recent_runtime_fraction_uses_last_step_span(tmp_path: Path) -> None:
    log = tmp_path / "log.lammps"
    log.write_text(
        "Step Temp PotEng TotEng Press Volume\n"
        "0 300 -10 -9 100 1000\n"
        "100 320 -12 -11 80 1010\n"
        "200 340 -14 -13 60 1020\n"
        "400 380 -18 -17 20 1040\n",
        encoding="utf-8",
    )

    summary = summarize_recent_runtime_fraction(read_thermo_rows(log), fraction=0.2)

    assert summary["npoints"] == 1.0
    assert summary["step_min"] == 400
    assert summary["temp"] == 380
    assert summary["temp_std"] == 0.0


def test_lammps_recent_runtime_fraction_reports_error_percent(tmp_path: Path) -> None:
    log = tmp_path / "log.lammps"
    log.write_text(
        "Step Temp PotEng TotEng Press Volume\n"
        "0 300 -10 -9 100 1000\n"
        "100 320 -12 -11 80 1010\n"
        "200 340 -14 -13 60 1020\n"
        "400 380 -18 -17 20 1040\n",
        encoding="utf-8",
    )

    summary = summarize_recent_runtime_fraction(read_thermo_rows(log), fraction=0.5)

    assert summary["npoints"] == 2.0
    assert summary["temp"] == 360
    assert summary["temp_std"] == 20
    assert summary["temp_std_percent"] == pytest.approx(5.5555556)
    assert summary["potential_energy"] == -16
    assert summary["potential_energy_std"] == 2
    assert summary["potential_energy_std_percent"] == pytest.approx(12.5)


def test_lammps_run_progress_uses_timestep_and_latest_run_block(tmp_path: Path) -> None:
    log = tmp_path / "log.gk"
    log.write_text(
        "timestep        0.00025\n"
        "run             10000\n"
        "Step Temp PotEng TotEng Press Volume\n"
        "0 300 -10 -9 100 1000\n"
        "10000 301 -11 -10 90 1001\n"
        "Loop time of 1 on 1 procs\n"
        "run             80000\n"
        "Step Temp PotEng TotEng Press Volume\n"
        "10000 300 -10 -9 100 1000\n"
        "11000 300 -10 -9 100 1000\n"
        "13000 300 -10 -9 100 1000\n",
        encoding="utf-8",
    )

    progress = summarize_lammps_run_progress(log, read_thermo_rows(log))

    assert progress is not None
    assert progress.timestep_ps == pytest.approx(0.00025)
    assert progress.current_steps == pytest.approx(3000)
    assert progress.expected_steps == pytest.approx(80000)
    assert progress.current_ps == pytest.approx(0.75)
    assert progress.expected_ps == pytest.approx(20.0)
    assert progress.percent == pytest.approx(3.75)


def test_lammps_run_progress_can_override_timestep(tmp_path: Path) -> None:
    log = tmp_path / "log.old-md"
    log.write_text(
        "run 50000\n"
        "Step Temp PotEng TotEng Press Volume\n"
        "0 300 -10 -9 100 1000\n"
        "5000 300 -10 -9 100 1000\n",
        encoding="utf-8",
    )

    progress = summarize_lammps_run_progress(log, read_thermo_rows(log), timestep_ps=0.0001)

    assert progress is not None
    assert progress.timestep_ps == pytest.approx(0.0001)
    assert progress.current_steps == pytest.approx(5000)
    assert progress.expected_steps == pytest.approx(50000)
    assert progress.current_ps == pytest.approx(0.5)
    assert progress.expected_ps == pytest.approx(5.0)
