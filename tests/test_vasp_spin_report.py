from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from atomi.cli.main import main as atomi_main
from atomi.vasp.spin_report import extract_last_magnetization_block, main


def write_poscar(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "spin test",
                "1.0",
                "5 0 0",
                "0 5 0",
                "0 0 5",
                "Gd U O",
                "2 2 2",
                "Direct",
                "0 0 0",
                "0.1 0.1 0.1",
                "0.2 0.2 0.2",
                "0.3 0.3 0.3",
                "0.4 0.4 0.4",
                "0.5 0.5 0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_incar(path: Path) -> None:
    path.write_text("MAGMOM = 7 -7 2 -2 2*0\n", encoding="utf-8")


def mag_block(values: list[float], include_total: bool = True) -> str:
    lines = [
        " magnetization (x)",
        " # of ion       s       p       d       f       tot",
        " -----------------------------------------------",
    ]
    for idx, value in enumerate(values, start=1):
        lines.append(f" {idx:5d} 0.000 0.000 0.000 0.000 {value: .3f}")
    if include_total:
        lines.append(" ------------------------------------------------")
        lines.append(f" tot 0.000 0.000 0.000 0.000 {sum(values): .3f}")
    return "\n".join(lines) + "\n"


def write_outcar(path: Path, complete_last: bool = True) -> None:
    first = [7.0, -7.0, 2.0, -2.0, 0.0, 0.0]
    final = [7.1, 7.0, 1.0, -2.1, 0.02, -0.02]
    partial = [7.2, 7.1]
    text = " NIONS =      6 ions\n" + mag_block(first)
    text += " free  energy   TOTEN  =       -10.000000 eV\n"
    if complete_last:
        text += mag_block(final)
    else:
        text += mag_block(final)
        text += " magnetization (x)\n # of ion s p d f tot\n"
        for idx, value in enumerate(partial, start=1):
            text += f" {idx} 0 0 0 0 {value}\n"
    text += " free  energy   TOTEN  =       -11.250000 eV\n"
    path.write_text(text, encoding="utf-8")


def write_outcar_gz(path: Path, complete_last: bool = True) -> None:
    tmp = path.with_suffix("")
    write_outcar(tmp, complete_last=complete_last)
    text = tmp.read_text(encoding="utf-8")
    tmp.unlink()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def test_extract_last_complete_magnetization_block(tmp_path: Path) -> None:
    outcar = tmp_path / "OUTCAR"
    write_outcar(outcar, complete_last=False)

    block = extract_last_magnetization_block(outcar)

    assert len(block.rows) == 6
    assert block.rows[0].tot == pytest.approx(7.1)
    assert block.warning is None


def test_single_outcar_writes_magmom_files(tmp_path: Path, capsys) -> None:
    outcar = tmp_path / "OUTCAR"
    poscar = tmp_path / "POSCAR"
    write_outcar(outcar)
    write_poscar(poscar)
    prefix = tmp_path / "single" / "spin"

    main(
        [
            "--outcar",
            str(outcar),
            "--species",
            str(poscar),
            "--output-prefix",
            str(prefix),
        ]
    )

    output = capsys.readouterr().out
    assert "Total moment" in output
    assert (tmp_path / "single" / "spin_last_magnetization_block.txt").exists()
    expanded = (tmp_path / "single" / "spin_MAGMOM_expanded.txt").read_text(encoding="utf-8")
    assert "1      Gd   +7.100" in expanded
    vasp_line = (tmp_path / "single" / "spin_MAGMOM_vasp.txt").read_text(encoding="utf-8")
    assert "MAGMOM = +7.100 +7.000 +1.000 -2.100 +0.020 -0.020" in vasp_line


def test_single_outcar_gz_writes_magmom_files(tmp_path: Path) -> None:
    outcar = tmp_path / "OUTCAR.gz"
    poscar = tmp_path / "POSCAR"
    write_outcar_gz(outcar)
    write_poscar(poscar)
    prefix = tmp_path / "single" / "spin_gz"

    main(
        [
            "--outcar",
            str(outcar),
            "--species",
            str(poscar),
            "--output-prefix",
            str(prefix),
        ]
    )

    expanded = (tmp_path / "single" / "spin_gz_MAGMOM_expanded.txt").read_text(encoding="utf-8")
    assert "1      Gd   +7.100" in expanded


def test_batch_spin_energy_report(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_a = tmp_path / "spin_001"
    run_b = tmp_path / "spin_002"
    run_a.mkdir()
    run_b.mkdir()
    for run in (run_a, run_b):
        write_poscar(run / "POSCAR")
        write_incar(run / "INCAR")
        write_outcar(run / "OUTCAR")
    (run_b / "OUTCAR").write_text("NIONS = 6\n no moments yet\n", encoding="utf-8")
    runlist.write_text("spin_001\nspin_002\nmissing_spin\n", encoding="utf-8")
    spin_index = tmp_path / "spin_index.csv"
    spin_index.write_text(
        "run_dir,name,dopant_mode,host_mode,moments_by_atom\n"
        f"{run_a},spin_001,all,afm,[]\n"
        f"{run_b},spin_002,all,afm,[]\n",
        encoding="utf-8",
    )
    prefix = tmp_path / "reports" / "spin_energy"

    atomi_main(
        [
            "vasp-spin-report",
            "--runlist",
            str(runlist),
            "--spin-index",
            str(spin_index),
            "--output-prefix",
            str(prefix),
            "--no-plot",
        ]
    )

    summary_path = tmp_path / "reports" / "spin_energy_run_summary.csv"
    atom_path = tmp_path / "reports" / "spin_energy_atom_moments.csv"
    report_path = tmp_path / "reports" / "spin_energy_report.md"
    assert summary_path.exists()
    assert atom_path.exists()
    assert report_path.exists()
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    assert rows[0]["mag_status"] == "OK"
    assert rows[0]["energy_eV"] == "-11.2500000000"
    assert rows[0]["changed_count"] == "2"
    assert '"Gd": "FM"' in rows[0]["element_order"]
    assert rows[1]["mag_status"] == "NO_MAGNETIZATION"
    assert rows[2]["mag_status"] == "NODIR"
    atom_rows = list(csv.DictReader(atom_path.open(encoding="utf-8")))
    assert atom_rows[0]["element"] == "Gd"
    assert atom_rows[0]["changed"] == "no"
    assert (tmp_path / "reports" / "magmom_lines" / "001_spin_001_MAGMOM.txt").exists()


def test_batch_uses_array_artifact_directory_for_stopped_spin_run(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_dir = tmp_path / "SPIN_CANDIDATES" / "spin_001"
    run_dir.mkdir(parents=True)
    write_poscar(run_dir / "POSCAR")
    write_incar(run_dir / "INCAR")
    runlist.write_text("spin_001\n", encoding="utf-8")
    spin_index = tmp_path / "SPIN_CANDIDATES" / "spin_index.csv"
    spin_index.write_text(
        "run_dir,name,dopant_mode,host_mode,moments_by_atom\n"
        f"{run_dir},spin_001,all,afm,[]\n",
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "bwforcluster-phonopy_array_96.sbatch.21488960.1.260518_030213"
    artifact_dir.mkdir()
    write_outcar(artifact_dir / "OUTCAR")
    prefix = tmp_path / "SPIN_CANDIDATES" / "spin_energy"

    atomi_main(
        [
            "vasp-spin-report",
            "--runlist",
            str(runlist),
            "--spin-index",
            str(spin_index),
            "--log-dir",
            str(tmp_path),
            "--output-prefix",
            str(prefix),
            "--no-plot",
        ]
    )

    rows = list(csv.DictReader((tmp_path / "SPIN_CANDIDATES" / "spin_energy_run_summary.csv").open()))
    assert rows[0]["mag_status"] == "OK"
    assert rows[0]["energy_eV"] == "-11.2500000000"
    assert rows[0]["mag_source"].endswith("OUTCAR")
    atom_rows = list(csv.DictReader((tmp_path / "SPIN_CANDIDATES" / "spin_energy_atom_moments.csv").open()))
    assert atom_rows[0]["element"] == "Gd"


def test_batch_uses_outcar_gz_for_moments_and_vasp_stdout_for_latest_energy(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_dir = tmp_path / "spin_001"
    run_dir.mkdir()
    write_poscar(run_dir / "POSCAR")
    write_incar(run_dir / "INCAR")
    write_outcar_gz(run_dir / "OUTCAR.gz")
    (tmp_path / "vasp.out_std.21488960.1").write_text(
        "DAV: 1 -12.500000 0.0 0.0\n"
        " free  energy   TOTEN  =       -12.750000 eV\n",
        encoding="utf-8",
    )
    runlist.write_text("spin_001\n", encoding="utf-8")
    prefix = tmp_path / "reports" / "spin_energy"

    atomi_main(
        [
            "vasp-spin-report",
            "--runlist",
            str(runlist),
            "--log-dir",
            str(tmp_path),
            "--output-prefix",
            str(prefix),
            "--no-plot",
        ]
    )

    rows = list(csv.DictReader((tmp_path / "reports" / "spin_energy_run_summary.csv").open()))
    assert rows[0]["mag_status"] == "OK"
    assert rows[0]["mag_source"].endswith("OUTCAR.gz")
    assert rows[0]["energy_source"].endswith("vasp.out_std.21488960.1")
    assert rows[0]["energy_eV"] == "-12.7500000000"


def test_missing_magnetization_error_is_clear(tmp_path: Path) -> None:
    outcar = tmp_path / "OUTCAR"
    outcar.write_text("NIONS = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="LORBIT"):
        extract_last_magnetization_block(outcar)
