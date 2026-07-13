import argparse
import json
from pathlib import Path

from atomi.xafs import fdmnes


def test_fdmnes_install_plan_explains_route_c(capsys) -> None:
    payload = fdmnes.install_plan_main(argparse.Namespace(json=False))
    captured = capsys.readouterr()

    assert payload["schema"] == "atomi.fdmnes_xanes_install_plan.v1"
    assert "Route C FDMNES" in payload["bridge_roles"]
    assert "separate external runtime" in payload["recommendation"]
    assert "FDMNES / Atomi route-C install plan" in captured.out


def test_fdmnes_main_returns_shell_success(capsys) -> None:
    assert fdmnes.main(["install-plan"]) == 0
    assert "route-C install plan" in capsys.readouterr().out


def test_fdmnes_prepare_writes_vasp_connected_workspace(tmp_path: Path) -> None:
    vasp_dir = tmp_path / "vasp_relax"
    vasp_dir.mkdir()
    (vasp_dir / "CONTCAR").write_text(
        "CeO2 test\n"
        "1\n"
        "5.4 0 0\n"
        "0 5.4 0\n"
        "0 0 5.4\n"
        "Ce O\n"
        "1 2\n"
        "Direct\n"
        "0 0 0\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n",
        encoding="utf-8",
    )
    (vasp_dir / "INCAR").write_text(
        "ENCUT = 600\nLDAU = .TRUE.\nLDAUL = 3 -1\nLDAUU = 5.0 0\nLASPH = .TRUE.\n",
        encoding="utf-8",
    )
    (vasp_dir / "KPOINTS").write_text("auto\n0\nGamma\n4 4 4\n0 0 0\n", encoding="utf-8")
    (vasp_dir / "OUTCAR").write_text(" free  energy   TOTEN  =     -123.456 eV\n", encoding="utf-8")
    outdir = tmp_path / "fdmnes_ce_l3"
    args = argparse.Namespace(
        structure=None,
        vasp_dir=vasp_dir,
        absorber="Ce",
        edge="L3",
        absorber_index=0,
        outdir=outdir,
        output_prefix="ce_l3_quick",
        radius=7.5,
        energy_range="-15 70 0.5",
        green=True,
        scf=True,
        quadrupole=True,
        spinorbit=True,
        convolution=True,
        extra=["Density"],
        executable="/private/fdmnes/bin/fdmnes",
        root="/private/fdmnes",
        bin="/private/fdmnes/bin",
        module="chem/fdmnes/test",
        job_name="fdmnes-ce-l3",
        ntasks=2,
        cpus_per_task=4,
        mem="12G",
        time="06:00:00",
    )

    metadata = fdmnes.prepare_main(args)
    fdmnes_input = (outdir / "fdmnes.inp").read_text(encoding="utf-8")
    run_script = (outdir / "run_fdmnes_xanes.sh").read_text(encoding="utf-8")
    sbatch_script = (outdir / "submit_fdmnes_xanes.sbatch").read_text(encoding="utf-8")
    project = json.loads((outdir / "fdmnes_xanes_project.json").read_text(encoding="utf-8"))
    fdmfile = (outdir / "fdmfile.txt").read_text(encoding="utf-8")

    assert "Filout\nce_l3_quick" in fdmnes_input
    assert "Range\n-15 70 0.5" in fdmnes_input
    assert "Radius\n7.500000" in fdmnes_input
    assert "Edge\nL3" in fdmnes_input
    assert "Absorber\n1" in fdmnes_input
    assert "Quadrupole" in fdmnes_input
    assert "Spinorbit" in fdmnes_input
    assert "Density" in fdmnes_input
    assert "Crystal\n5.4000000000 5.4000000000 5.4000000000 90.00000000 90.00000000 90.00000000" in fdmnes_input
    assert " 58 0.000000000000 0.000000000000 0.000000000000 ! absorber" in fdmnes_input
    assert "  8 0.250000000000 0.250000000000 0.250000000000" in fdmnes_input
    assert "module load \"${ATOMI_FDMNES_MODULE}\"" in run_script
    assert "ATOMI_FDMNES_EXE=/private/fdmnes/bin/fdmnes" in run_script
    assert "cat > fdmfile.txt" in run_script
    assert "FDMNES did not consume the generated fdmfile.txt" in run_script
    assert f"#SBATCH --chdir={outdir.resolve()}" in sbatch_script
    assert "bash run_fdmnes_xanes.sh" in sbatch_script
    assert fdmfile == "1\nfdmnes.inp\n"
    assert metadata["schema"] == project["schema"] == "atomi.fdmnes_xanes_project.v1"
    assert project["vasp"]["incar_tags"]["ENCUT"] == "600"
    assert project["vasp"]["incar_tags"]["LDAUU"] == "5.0 0"
    assert project["vasp"]["final_toten_ev"] == -123.456
    assert project["absorber"] == "Ce"
    assert project["edge"] == "L3"


def test_fdmnes_prepare_accepts_vasp_split_species_labels(tmp_path: Path) -> None:
    vasp_dir = tmp_path / "vasp_uo2_lr_reference"
    vasp_dir.mkdir()
    (vasp_dir / "CONTCAR").write_text(
        "UO2 split-species test\n"
        "1\n"
        "5.47 0 0\n"
        "0 5.47 0\n"
        "0 0 5.47\n"
        "U_ Ubulk O\n"
        "1 1 4\n"
        "Direct\n"
        "0 0 0\n"
        "0.5 0.5 0.5\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.25\n"
        "0.75 0.25 0.75\n"
        "0.25 0.75 0.75\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "fdmnes_u_l3"
    args = argparse.Namespace(
        structure=None,
        vasp_dir=vasp_dir,
        absorber="U",
        edge="L3",
        absorber_index=0,
        outdir=outdir,
        output_prefix="uo2_u_l3",
        radius=6.0,
        energy_range="-20 80 0.5",
        green=True,
        scf=False,
        quadrupole=False,
        spinorbit=True,
        convolution=True,
        extra=[],
        executable="fdmnes",
        root="",
        bin="",
        module="",
        job_name="fdmnes-u-l3",
        ntasks=1,
        cpus_per_task=1,
        mem="8G",
        time="02:00:00",
    )

    metadata = fdmnes.prepare_main(args)
    text = (outdir / "fdmnes.inp").read_text(encoding="utf-8")

    assert " 92 0.000000000000 0.000000000000 0.000000000000 ! absorber" in text
    assert " 92 0.500000000000 0.500000000000 0.500000000000" in text
    assert metadata["raw_elements"] == ["U_", "Ubulk", "O"]
    assert metadata["elements"] == ["U", "U", "O"]


def test_fdmnes_collect_summarizes_numeric_spectrum(tmp_path: Path) -> None:
    spectrum = tmp_path / "xanes.dat"
    spectrum.write_text("# e mu\n0.0 0.1\n1.0 2.0\n2.0 0.8\n", encoding="utf-8")
    out = tmp_path / "summary.json"
    args = argparse.Namespace(fdmnes_dir=tmp_path, spectrum=None, write=out)

    summary = fdmnes.collect_main(args)

    assert summary["schema"] == "atomi.fdmnes_xanes_summary.v1"
    assert summary["n_points"] == 3
    assert summary["peak_energy"] == 1.0
    assert json.loads(out.read_text(encoding="utf-8"))["intensity_max"] == 2.0


def test_fdmnes_collect_skips_metadata_before_xanes_table(tmp_path: Path) -> None:
    spectrum = tmp_path / "fdmnes_out.txt"
    spectrum.write_text(
        " 17166.300   92  2  3  5.3441817E-01 = E_edge, Z, n_edge\n"
        "    Energy    <xanes>\n"
        "  -20.0000  1.0483774E-05\n"
        "   0.50000  1.6542545E-03\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(fdmnes_dir=tmp_path, spectrum=spectrum, write=None)

    summary = fdmnes.collect_main(args)

    assert summary["n_points"] == 2
    assert summary["energy_min"] == -20.0
    assert summary["energy_max"] == 0.5
    assert summary["peak_energy"] == 0.5
