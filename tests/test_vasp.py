from pathlib import Path

from atomi.codes.vasp import missing_inputs, summarize_outcar
from atomi.vasp.checks import collect_run_energies, vasp_energies
from atomi.viz.lammps import read_thermo_rows, summarize_thermo
from atomi.viz.vasp_live import count_dav_steps


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
    assert records[0].energy_eV == -2185.45944662
    assert records[0].energy_kind == "dav"


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
