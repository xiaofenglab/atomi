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
