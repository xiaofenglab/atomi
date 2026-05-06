from pathlib import Path

from atomi.codes.vasp import missing_inputs
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
