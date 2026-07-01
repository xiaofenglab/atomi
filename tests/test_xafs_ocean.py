import argparse
import json
from pathlib import Path

from atomi.xafs import ocean


def test_ocean_install_plan_explains_external_runtime(capsys) -> None:
    payload = ocean.install_plan_main(argparse.Namespace(json=False))
    captured = capsys.readouterr()

    assert payload["schema"] == "atomi.ocean_xanes_install_plan.v1"
    assert "separate" in payload["recommendation"]
    assert "OCEAN" in payload["bridge_roles"]
    assert "OCEAN / Atomi HPC install plan" in captured.out


def test_ocean_main_returns_shell_success(capsys) -> None:
    assert ocean.main(["install-plan"]) == 0
    assert "OCEAN / Atomi HPC install plan" in capsys.readouterr().out


def test_ocean_prepare_writes_workspace(tmp_path: Path) -> None:
    structure = tmp_path / "POSCAR"
    structure.write_text("test structure\n1\n1 0 0\n0 1 0\n0 0 1\nU O\n1 2\nDirect\n0 0 0\n0.25 0.25 0.25\n0.75 0.75 0.75\n", encoding="utf-8")
    vasp_dir = tmp_path / "vasp_scf"
    vasp_dir.mkdir()
    outdir = tmp_path / "ocean_u_m4"
    args = argparse.Namespace(
        structure=structure,
        vasp_dir=vasp_dir,
        absorber="U",
        edge="M4",
        outdir=outdir,
        dft_engine="vasp",
        dft_plus_u="VASP LDAU U=4.5 eV on U 5f",
        executable="/private/ocean/bin/ocean.pl",
        root="/private/ocean",
        bin="/private/ocean/bin",
        module="chem/ocean/test",
        pseudo_dir="/private/ocean/pseudos",
        energy_window="-10 60 eV",
        edge_atom_index=0,
        nkpt="4 4 4",
        screen_nkpt="2 2 2",
        xmesh="",
        nbands=200,
        screen_nbands=100,
        ecut="80",
        diemac="5.0",
        broaden="0.3",
        pp_list=["U.test.UPF", "O.test.UPF"],
        extra=["# custom extra line"],
        job_name="ocean-u-m4",
        ntasks=16,
        cpus_per_task=1,
        mem="24G",
        time="12:00:00",
    )

    metadata = ocean.prepare_main(args)
    ocean_input = (outdir / "ocean.in").read_text(encoding="utf-8")
    run_script = (outdir / "run_ocean_xanes.sh").read_text(encoding="utf-8")
    project = json.loads((outdir / "ocean_xanes_project.json").read_text(encoding="utf-8"))

    assert "acell {" in ocean_input
    assert "znucl { 92 8 }" in ocean_input
    assert "typat {" in ocean_input
    assert "1 2 2" in ocean_input
    assert "edges{ 1 3 2 }" in ocean_input
    assert "pp_list{ U.test.UPF O.test.UPF }" in ocean_input
    assert "nbands 200" in ocean_input
    assert "module load \"${ATOMI_OCEAN_MODULE}\"" in run_script
    assert "ATOMI_OCEAN_MODULE=chem/ocean/test" in run_script
    assert "ATOMI_OCEAN_EXE=/private/ocean/bin/ocean.pl" in run_script
    assert metadata["module"] == "chem/ocean/test"
    assert metadata["absorber"] == project["absorber"] == "U"
    assert metadata["native_ocean"]["edge_quantum"] == {"n": 3, "l": 2}
    assert metadata["dft_plus_u"].startswith("VASP LDAU")


def test_ocean_collect_summarizes_two_column_spectrum(tmp_path: Path) -> None:
    spectrum = tmp_path / "absspct.dat"
    spectrum.write_text("# e intensity\n0.0 0.1\n1.0 2.5\n2.0 1.0\n", encoding="utf-8")
    out = tmp_path / "summary.json"
    args = argparse.Namespace(ocean_dir=tmp_path, spectrum=None, write=out)

    summary = ocean.collect_main(args)

    assert summary["n_points"] == 3
    assert summary["peak_energy"] == 1.0
    assert summary["intensity_max"] == 2.5
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == "atomi.ocean_xanes_summary.v1"
