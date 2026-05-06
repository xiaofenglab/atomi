from pathlib import Path

from atomi.codes.vasp import missing_inputs
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
