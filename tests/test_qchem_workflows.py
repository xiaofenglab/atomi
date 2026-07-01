import argparse
from pathlib import Path

from atomi.qchem.molcas import MolcasClusterOptions, read_poscar, write_cluster_workspace
from atomi.qchem.turbomole import DefineOptions, render_define_stdin, write_define_workspace


def test_turbomole_define_preserves_blank_lines(tmp_path: Path) -> None:
    text = render_define_stdin(DefineOptions(charge=0, symmetry="ci"))
    lines = text.splitlines()
    assert lines[:10] == ["a coord", "sy ci", "ired", "*", "bl", "ecpl", "*", "eht", "", "0"]
    assert "shift\n \n0.25" not in text
    assert "shift\n\n0.25" in text
    assert text.endswith("*\n*\n*\n")

    xyz = tmp_path / "ga_3h2o.xyz"
    xyz.write_text("1\nGa\nGa 0 0 0\n", encoding="utf-8")
    outputs = write_define_workspace(tmp_path / "tm", xyz, DefineOptions(charge=1), overwrite=False)
    assert Path(outputs["define_stdin"]).read_text(encoding="utf-8").splitlines()[9] == "1"
    assert "x2t" in Path(outputs["run_script"]).read_text(encoding="utf-8")
    sbatch = Path(outputs["relax_sbatch"]).read_text(encoding="utf-8")
    assert "jobex -ri -c 200" in sbatch
    assert "chem/turbomole/7.5" in sbatch
    policy = Path(outputs["basis_policy"]).read_text(encoding="utf-8")
    assert "OpenMolcas" in policy
    assert "ANO-RCC" in policy
    assert "preconditioner" in policy


def _write_uo_poscar(path: Path) -> None:
    path.write_text(
        """UO test
1.0
10 0 0
0 10 0
0 0 10
U O
2 4
Direct
0.0 0.0 0.0
0.5 0.5 0.5
0.12 0.0 0.0
0.0 0.12 0.0
0.0 0.0 0.12
0.5 0.5 0.62
""",
        encoding="utf-8",
    )


def test_molcas_cluster_writes_xfield_and_templates(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    _write_uo_poscar(poscar)
    data = read_poscar(poscar)
    assert data["species"] == ["U", "U", "O", "O", "O", "O"]

    outdir = tmp_path / "molcas"
    meta = write_cluster_workspace(
        poscar,
        outdir,
        MolcasClusterOptions(center_indices=(1,), ligand_cutoff_A=1.5, point_charge_cutoff_A=8.0, charge_mode="u5-like"),
        label="u01",
    )
    assert meta["qm_atom_count"] == 4
    assert meta["template_charge"] == -1
    xfield = (outdir / "u01.xfield").read_text(encoding="utf-8").splitlines()
    assert xfield[0].endswith("Angstrom 0")
    ground = (outdir / "u01.ground.inp").read_text(encoding="utf-8")
    assert "Group = NoSym" in ground
    assert "XField = u01.xfield" in ground
    medge = (outdir / "u01.Medge_template.inp").read_text(encoding="utf-8")
    assert "Do not use an ECP" in medge



def test_openmolcas_bridge_prepare_and_collect(tmp_path: Path) -> None:
    from atomi.qchem import openmolcas_bridge

    xyz = tmp_path / "u_o_cluster.xyz"
    xyz.write_text("2\nUO\nU 0 0 0\nO 2.2 0 0\n", encoding="utf-8")
    outdir = tmp_path / "molcas_bridge"
    metadata = openmolcas_bridge.prepare_main(
        argparse.Namespace(
            xyz=xyz,
            outdir=outdir,
            label="u4o9_u01",
            xyz_name="",
            copy_xyz=True,
            charge=2,
            spin=3,
            basis="ANO-RCC-VDZP",
            group="NoSym",
            recipe="actinide-m45-xanes",
            nactel="2 0 0",
            inactive="40 0",
            ras1="1 0",
            ras2="7 0",
            ras3="0 5",
            ciroots="1 1 1",
            iterations="300 100",
            levs="5.0",
            frozen="20 20",
            ipea="0",
            imag="5.0",
            threshold="1.0E-09 1.0E-07",
            no_caspt2=False,
            multistate=False,
            no_orbital_prep=False,
            no_partner=False,
            partner_spin=5,
            partner_ciroots="3 3 1",
            sonorb="1,2,3",
            no_bssh=False,
            no_amfi=False,
            core_hole_note="U M4/M5 scaffold",
            extra_rasscf_line=[],
            executable="pymolcas",
            module="chem/openmolcas/test",
            job_name="u4o9-u01",
            ntasks=4,
            mem_per_cpu_mb=4000,
            time="02:00:00",
            scratch_gb="50",
        )
    )
    text = (outdir / "u4o9_u01.inp").read_text(encoding="utf-8")
    run_script = (outdir / "run_openmolcas.sh").read_text(encoding="utf-8")
    assert metadata["schema"] == "atomi.openmolcas_bridge_project.v1"
    assert "Actinide M4,5-edge scaffold" in text
    assert "&RASSCF" in text
    assert "&CASPT2" in text
    assert "&RASSI" in text
    assert "SpinOrbit" in text
    assert "ATOMI_MOLCAS_MODULE=chem/openmolcas/test" in run_script

    output = tmp_path / "molcas.out"
    output.write_text(
        "--- Start Module: caspt2 at now ---\n"
        "::    XMS-CASPT2 Root  1     Total energy:  -4104.79129157\n"
        "--- Stop Module: caspt2 at now /rc=_RC_ALL_IS_WELL_ ---\n"
        "--- Start Module: rassi at now ---\n"
        "--- Stop Module: rassi at now /rc=_RC_ALL_IS_WELL_ ---\n",
        encoding="utf-8",
    )
    summary = openmolcas_bridge.collect_main(argparse.Namespace(output=output, write=None))
    assert summary["caspt2_roots"][0]["energy_hartree"] == -4104.79129157
    assert summary["rassi_module_count"] == 1
    assert not summary["has_error_marker"]


def test_openmolcas_status_cli_accepts_registry_argv(capsys) -> None:
    from atomi.qchem import openmolcas_bridge

    assert openmolcas_bridge.status_cli(["--json"]) == 0
    assert "atomi.openmolcas_status.v1" in capsys.readouterr().out
    assert openmolcas_bridge.install_plan_cli([]) == 0
    assert "OpenMolcas / Atomi HPC install plan" in capsys.readouterr().out


def test_pegamoid_bridge_status_and_prepare(tmp_path: Path, capsys) -> None:
    from atomi.qchem import pegamoid_bridge

    assert pegamoid_bridge.status_cli(["--json"]) == 0
    assert "atomi.pegamoid_status.v1" in capsys.readouterr().out
    assert pegamoid_bridge.install_plan_cli([]) == 0
    assert "separate GUI/runtime environment" in capsys.readouterr().out

    h5 = tmp_path / "Ga_6h2o.rasscf.h5"
    orb = tmp_path / "Ga_6h2o.RasOrb"
    h5.write_text("placeholder", encoding="utf-8")
    orb.write_text("placeholder", encoding="utf-8")
    meta = pegamoid_bridge.prepare_main(
        argparse.Namespace(file=[str(h5), str(orb)], outdir=tmp_path / "pegamoid", label="ga", module="", maxscratch="100MB")
    )
    run_script = Path(meta["run_script"]).read_text(encoding="utf-8")
    assert "PEGAMOID_MAXSCRATCH" in run_script
    assert "Ga_6h2o.rasscf.h5" in run_script
    assert meta["schema"] == "atomi.pegamoid_bridge_project.v1"


def test_molcas_xanes_spectrum_from_output(tmp_path: Path) -> None:
    from atomi.xafs import molcas_xanes_spectrum

    output = tmp_path / "ga.out"
    output.write_text(
        """
 Weights of the five most important spin-orbit-free states for each spin-orbit state.

 SO State  Total energy (au)           Spin-free states, spin, and weights
 -------------------------------------------------------------------------------------------------------
    1         0.000000       1 0.0  1.0000
    2       382.000000       2 0.0  1.0000
    3       382.100000       3 0.0  1.0000
 -------------------------------------------------------------------------------------------------------

++ Dipole transition strengths (SO states):
   ----------------------------------------
      From   To        Osc. strength     Einstein coefficients Ax, Ay, Az (sec-1)    Total A (sec-1)
     -----------------------------------------------------------------------------------------------
         1    2       2.00000000E-04  1.0  1.0  1.0  1.0
         1    3       1.00000000E-04  1.0  1.0  1.0  1.0
     -----------------------------------------------------------------------------------------------
""",
        encoding="utf-8",
    )
    outdir = tmp_path / "spectrum"
    summary = molcas_xanes_spectrum.run(
        argparse.Namespace(
            molcas_out=output,
            transitions_csv=None,
            element="Ga",
            edge="K",
            gauge="length",
            from_state=1,
            section="last",
            energy_shift_ev=0.0,
            gaussian_fwhm=1.0,
            lorentzian_fwhm=1.0,
            broadening="pseudo-voigt",
            pseudo_voigt_eta=0.5,
            normalize="max",
            emin=10390.0,
            emax=10410.0,
            step=0.2,
            outdir=outdir,
            spectrum_name="spectrum.csv",
            transitions_name="transitions.csv",
            summary_name="summary.json",
            plot_name="spectrum.png",
            title="Ga test",
            no_xraydb=True,
            no_plot=True,
        )
    )
    assert summary["schema"] == "atomi.molcas_xanes_spectrum.v1"
    assert summary["n_transitions_used"] == 2
    assert (outdir / "spectrum.csv").exists()
    assert "energy_ev" in (outdir / "transitions.csv").read_text(encoding="utf-8")
