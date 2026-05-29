import gzip
from pathlib import Path

import pytest

from atomi.vasp.magmom import default_outcar_path, final_outcar_magnetization, update_incar_magmom


def write_poscar(path: Path) -> None:
    path.write_text(
        "test\n"
        "1.0\n"
        "1 0 0\n"
        "0 1 0\n"
        "0 0 1\n"
        "U O\n"
        "2 1\n"
        "Direct\n"
        "0 0 0\n"
        "0.5 0.5 0.5\n"
        "0.25 0.25 0.25\n",
        encoding="utf-8",
    )


def outcar_text() -> str:
    return (
        " NIONS =      3 ions\n"
        " magnetization (x)\n"
        " # of ion       s       p       d       tot\n"
        " -------------------------------------------\n"
        "    1        0.0     0.0     0.0     2.100\n"
        "    2        0.0     0.0     0.0    -2.200\n"
        "    3        0.0     0.0     0.0     0.010\n"
        " tot         0.0     0.0     0.0    -0.090\n"
    )


def test_final_outcar_magnetization_reads_gzip(tmp_path: Path) -> None:
    outcar = tmp_path / "OUTCAR.gz"
    with gzip.open(outcar, "wt", encoding="utf-8") as handle:
        handle.write(outcar_text())

    assert final_outcar_magnetization(outcar, 3) == pytest.approx([2.1, -2.2, 0.01])


def test_magit_update_accepts_gzip_outcar(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    incar = tmp_path / "INCAR"
    outcar = tmp_path / "OUTCAR.gz"
    write_poscar(poscar)
    incar.write_text("ISPIN = 2\nMAGMOM = 3*0\n", encoding="utf-8")
    with gzip.open(outcar, "wt", encoding="utf-8") as handle:
        handle.write(outcar_text())

    result = update_incar_magmom(
        outcar=outcar,
        poscar=poscar,
        incar=incar,
        elements=["U"],
        backup=False,
    )

    assert result.moments == pytest.approx([2.1, -2.2, 0.0])
    assert "MAGMOM = 2.1 -2.2 1*0" in incar.read_text(encoding="utf-8")


def test_magit_default_outcar_prefers_gzip_when_plain_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "OUTCAR.gz").write_bytes(b"not real gzip for path selection only\n")

    assert default_outcar_path() == Path("OUTCAR.gz")
