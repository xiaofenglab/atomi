from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from atomi.cli.main import main as atomi_main
from atomi.vasp.spin_report import (
    extract_last_magnetization_block,
    guard_rule_text,
    infer_moment_guards_from_files,
    main,
)


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


def write_outcar_with_final(path: Path, final: list[float], energy: float = -11.25) -> None:
    text = " NIONS =      6 ions\n" + mag_block([7.0, -7.0, 2.0, -2.0, 0.0, 0.0])
    text += f" free  energy   TOTEN  =       {energy:.6f} eV\n"
    text += mag_block(final)
    text += f" free  energy   TOTEN  =       {energy - 0.5:.6f} eV\n"
    path.write_text(text, encoding="utf-8")


def write_outcar_gz(path: Path, complete_last: bool = True) -> None:
    tmp = path.with_suffix("")
    write_outcar(tmp, complete_last=complete_last)
    text = tmp.read_text(encoding="utf-8")
    tmp.unlink()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def onsite_block(atom: int, angular_l: int, moment: float) -> str:
    dim = 2 * angular_l + 1
    spin1 = max(moment, 0.0)
    spin2 = max(-moment, 0.0)

    def matrix(first_diagonal: float) -> list[str]:
        rows = []
        for row_index in range(dim):
            values = ["0.0000"] * dim
            if row_index == 0:
                values[0] = f"{first_diagonal:.4f}"
            rows.append(" " + " ".join(values))
        return rows

    lines = [
        f"atom = {atom:4d}  type =  2  l = {angular_l}",
        "",
        " onsite density matrix",
        "",
        "spin component  1",
        "",
        *matrix(spin1),
        "",
        "spin component  2",
        "",
        *matrix(spin2),
        "",
        " occupancies and eigenvectors",
    ]
    return "\n".join(lines) + "\n"


def write_onsite_outcar(path: Path) -> None:
    text = " NIONS =      6 ions\n"
    text += onsite_block(atom=1, angular_l=3, moment=7.0)
    text += onsite_block(atom=3, angular_l=3, moment=-2.0)
    text += " free  energy   TOTEN  =       -13.500000 eV\n"
    path.write_text(text, encoding="utf-8")


def test_extract_last_complete_magnetization_block(tmp_path: Path) -> None:
    outcar = tmp_path / "OUTCAR"
    write_outcar(outcar, complete_last=False)

    block = extract_last_magnetization_block(outcar)

    assert len(block.rows) == 6
    assert block.rows[0].tot == pytest.approx(7.1)
    assert block.warning is None


def test_extract_onsite_density_matrix_fallback(tmp_path: Path) -> None:
    outcar = tmp_path / "OUTCAR"
    write_onsite_outcar(outcar)

    block = extract_last_magnetization_block(outcar)

    assert block.source_kind == "onsite_density_matrix"
    assert len(block.rows) == 6
    assert block.rows[0].tot == pytest.approx(7.0)
    assert block.rows[1].tot == pytest.approx(0.0)
    assert block.rows[2].tot == pytest.approx(-2.0)
    assert "onsite density matrix" in (block.warning or "")


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


def test_spincheck_no_args_uses_current_outcar(tmp_path: Path, monkeypatch, capsys) -> None:
    write_outcar(tmp_path / "OUTCAR")
    write_poscar(tmp_path / "POSCAR")
    write_incar(tmp_path / "INCAR")
    monkeypatch.chdir(tmp_path)

    main([])

    output = capsys.readouterr().out
    assert "Moment rows" in output
    assert "Physics guard" in output
    assert (tmp_path / "spin_report_MAGMOM_vasp.txt").exists()


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


def test_batch_spin_energy_report(tmp_path: Path, capsys) -> None:
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

    output = capsys.readouterr().out
    assert "Atomi VASP Spin Report" in output
    assert "guard" in output
    assert "chg" in output
    assert "order" in output
    assert "Gd:AFM>FM" in output
    assert "U:AFM" in output
    assert "Physics guard" in output
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


def test_auto_moment_guard_infers_multiple_integer_states_from_incar(tmp_path: Path) -> None:
    run = tmp_path / "spin_u4_u5"
    run.mkdir()
    write_poscar(run / "POSCAR")
    (run / "INCAR").write_text("MAGMOM = 7 -7 2 1 2*0\n", encoding="utf-8")

    guards = infer_moment_guards_from_files(run / "POSCAR", run / "INCAR", default_tol=0.6)

    text = guard_rule_text(guards)
    assert "Gd=[+7.000,-7.000] tol=0.6" in text
    assert "U=[+2.000,-2.000,+1.000,-1.000] tol=0.6" in text
    assert "O=[+0.000] tol=0.25" in text


def test_batch_spin_physics_guard_writes_filtered_tables(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    run_ok = tmp_path / "spin_001"
    run_bad = tmp_path / "spin_002"
    run_ok.mkdir()
    run_bad.mkdir()
    for run in (run_ok, run_bad):
        write_poscar(run / "POSCAR")
        write_incar(run / "INCAR")
    write_outcar_with_final(run_ok / "OUTCAR", [7.1, -7.0, 1.0, -2.1, 0.02, -0.02], energy=-12.0)
    write_outcar_with_final(run_bad / "OUTCAR", [5.8, -7.0, 1.0, -2.1, 0.02, -0.02], energy=-13.0)
    runlist.write_text("spin_001\nspin_002\n", encoding="utf-8")
    prefix = tmp_path / "reports" / "spin_energy"

    atomi_main(
        [
            "vasp-spin-report",
            "--runlist",
            str(runlist),
            "--output-prefix",
            str(prefix),
            "--moment-guard",
            "Gd=7,-7@0.5",
            "--moment-guard",
            "U=2,-2,1,-1@0.5",
            "--moment-guard",
            "O=0@0.2",
            "--no-plot",
        ]
    )

    rows = list(csv.DictReader((tmp_path / "reports" / "spin_energy_run_summary.csv").open()))
    assert rows[0]["physics_guard_status"] == "OK"
    assert rows[1]["physics_guard_status"] == "FAIL"
    assert rows[1]["physics_guard_bad_by_element"] == '{"Gd": 1}'
    filtered = list(csv.DictReader((tmp_path / "reports" / "spin_energy_physics_filtered_run_summary.csv").open()))
    assert len(filtered) == 1
    assert filtered[0]["run"] == "spin_001"
    atom_rows = list(csv.DictReader((tmp_path / "reports" / "spin_energy_atom_moments.csv").open()))
    bad_gd = [row for row in atom_rows if row["run"] == "spin_002" and row["atom"] == "1"][0]
    assert bad_gd["physics_ok"] == "no"
    assert bad_gd["physics_target"] == "7.00000000"


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


def test_batch_uses_nested_array_artifact_onsite_matrix_fallback(tmp_path: Path) -> None:
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
    artifact_run = tmp_path / "bwforcluster-phonopy_array_96.sbatch.21488960.1.260518_030213" / "run"
    artifact_run.mkdir(parents=True)
    write_onsite_outcar(artifact_run / "OUTCAR")
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
    assert rows[0]["mag_status"] == "ONSITE_MATRIX"
    assert rows[0]["energy_eV"] == "-13.5000000000"
    assert rows[0]["mag_source"].endswith("run/OUTCAR")
    atom_rows = list(csv.DictReader((tmp_path / "SPIN_CANDIDATES" / "spin_energy_atom_moments.csv").open()))
    assert atom_rows[0]["final_moment"] == "7.00000000"
    assert atom_rows[2]["final_moment"] == "-2.00000000"


def test_batch_uses_nested_artifact_poscar_and_incar_when_run_folder_missing(tmp_path: Path) -> None:
    runlist = tmp_path / "runlist.txt"
    runlist.write_text("spin_001\n", encoding="utf-8")
    artifact_run = tmp_path / "bwforcluster-phonopy_array_96.sbatch.21488960.1.260518_030213" / "scratch" / "run"
    artifact_run.mkdir(parents=True)
    write_poscar(artifact_run / "POSCAR")
    write_incar(artifact_run / "INCAR")
    write_outcar(artifact_run / "OUTCAR")
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
    assert rows[0]["run"] == "spin_001"
    assert rows[0]["resolved_run"].endswith("spin_001")
    assert rows[0]["output_run_dir"].endswith("scratch/run")
    assert rows[0]["mag_source"].endswith("scratch/run/OUTCAR")
    assert rows[0]["changed_count"] == "2"
    atom_rows = list(csv.DictReader((tmp_path / "reports" / "spin_energy_atom_moments.csv").open()))
    assert atom_rows[0]["element"] == "Gd"
    assert atom_rows[0]["initial_moment"] == "7.00000000"


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


def test_single_spincheck_suggests_neutral_u4o9_corrected_incar(tmp_path: Path, capsys) -> None:
    poscar = tmp_path / "POSCAR"
    incar = tmp_path / "INCAR"
    outcar = tmp_path / "OUTCAR"
    corrected = tmp_path / "INCAR.corrected"
    prefix = tmp_path / "reports" / "u4o9"
    poscar.write_text(
        "\n".join(
            [
                "U4O9 spin test",
                "1.0",
                "8 0 0",
                "0 8 0",
                "0 0 8",
                "U O",
                "4 9",
                "Direct",
                "0.00 0.00 0.00",
                "0.10 0.10 0.10",
                "0.20 0.20 0.20",
                "0.30 0.30 0.30",
                "0.40 0.40 0.40",
                "0.45 0.45 0.45",
                "0.50 0.50 0.50",
                "0.55 0.55 0.55",
                "0.60 0.60 0.60",
                "0.65 0.65 0.65",
                "0.70 0.70 0.70",
                "0.75 0.75 0.75",
                "0.80 0.80 0.80",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    incar.write_text("ENCUT = 520\nMAGMOM = 2 -2 2 -2 9*0\n", encoding="utf-8")
    initial = [2.0, -2.0, 2.0, -2.0] + [0.0] * 9
    final = [2.1, -0.1, 1.1, 2.2] + [0.01] * 9
    text = " NIONS =      13 ions\n" + mag_block(initial)
    text += " free  energy   TOTEN  =       -20.000000 eV\n"
    text += mag_block(final)
    text += " free  energy   TOTEN  =       -21.000000 eV\n"
    outcar.write_text(text, encoding="utf-8")

    main(
        [
            "--outcar",
            str(outcar),
            "--species",
            str(poscar),
            "--incar",
            str(incar),
            "--output-prefix",
            str(prefix),
            "--magmom-oxidation",
            "U:2=4,U:1=5,O:0=-2",
            "--correction-magnetic-order",
            "afm",
            "--corrected-incar",
            str(corrected),
        ]
    )

    output = capsys.readouterr().out
    assert "Charge check       : total=0 neutral=True" in output
    corrected_text = corrected.read_text(encoding="utf-8")
    assert "MAGMOM = +2.000 -1.000 +1.000 -2.000" in corrected_text
    rows = list(csv.DictReader((tmp_path / "reports" / "u4o9_spin_corrections.csv").open()))
    atom2 = rows[1]
    atom4 = rows[3]
    assert atom2["suggested_oxidation"] == "5"
    assert "unphysical_or_drifted" in atom2["label"]
    assert atom4["suggested_moment"] == "-2.00000000"
    assert "sign_flipped" in atom4["label"]


def test_missing_magnetization_error_is_clear(tmp_path: Path) -> None:
    outcar = tmp_path / "OUTCAR"
    outcar.write_text("NIONS = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="LORBIT"):
        extract_last_magnetization_block(outcar)
