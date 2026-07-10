import csv
import json
from pathlib import Path

import pytest

from atomi.vasp.hubbard_u import main, parse_total_charge


POSCAR = """UO2 test
1.0
5 0 0
0 5 0
0 0 5
U O
2 2
Direct
0 0 0
0.5 0.5 0.5
0.25 0.25 0.25
0.75 0.75 0.75
"""


def write_seed(path: Path) -> None:
    path.mkdir()
    (path / "CONTCAR").write_text(POSCAR, encoding="utf-8")
    (path / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (path / "INCAR").write_text(
        "ENCUT = 650\nISPIN = 2\nMAGMOM = 2.0 -2.0 2*0.0\n"
        "LDAU = .TRUE.\nLDAUTYPE = 1\nLDAUL = 3 -1\nLDAUU = 4 0\nLDAUJ = 0 0\n",
        encoding="utf-8",
    )
    (path / "POTCAR").write_text(
        "TITEL = PAW_PBE U\nEnd of Dataset\nTITEL = PAW_PBE O\nEnd of Dataset\n",
        encoding="utf-8",
    )


def charge_outcar(occupation: float) -> str:
    return f"""random output
 total charge

 # of ion       s       p       d       f       tot
 --------------------------------------------------
    1        0.100   0.200   0.300   {occupation:.6f}   2.000
    2        0.100   0.200   0.300   2.000000   2.600
 --------------------------------------------------
 tot         0.2 0.4 0.6 4.0 5.2
"""


def test_vasp_lr_prepare_splits_probe_species_and_reorders_magmom(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    write_seed(seed)
    outdir = tmp_path / "lr"
    main(
        [
            "vasp-lr-prepare",
            "--seed",
            str(seed),
            "--outdir",
            str(outdir),
            "--probe-atom",
            "2",
            "--alpha",
            "-0.1",
            "0.1",
        ]
    )

    poscar = (outdir / "reference_u0" / "POSCAR").read_text(encoding="utf-8")
    incar = (outdir / "reference_u0" / "INCAR").read_text(encoding="utf-8")
    perturb = (outdir / "alpha_p0p100" / "nscf" / "INCAR").read_text(encoding="utf-8")
    potcar = (outdir / "reference_u0" / "POTCAR").read_text(encoding="utf-8")
    assert "U_probe  U_bulk  O" in poscar
    assert "1  1  2" in poscar
    assert "MAGMOM = -2.0 2.0 0.0 0.0" in incar
    assert "LDAU = .FALSE." in incar
    assert "LDAUTYPE = 3" in perturb
    assert "LDAUL = 3 -1 -1" in perturb
    assert "LDAUU = 0.10000000 0.0 0.0" in perturb
    assert "ICHARG = 11" in perturb
    assert potcar.count("End of Dataset") == 3
    metadata = json.loads((outdir / "workflow.json").read_text(encoding="utf-8"))
    assert metadata["atom_order_new_to_old_1based"] == [2, 1, 3, 4]
    assert metadata["projector"] == "VASP PAW on-site l channel"


def test_parse_and_fit_vasp_response_u(tmp_path: Path) -> None:
    root = tmp_path / "lr"
    rows = []
    for alpha in (-0.2, -0.1, 0.1, 0.2):
        for stage, slope in (("nscf", 0.5), ("scf", 0.125)):
            directory = root / f"a{alpha:+.1f}" / stage
            directory.mkdir(parents=True)
            (directory / "OUTCAR").write_text(charge_outcar(2.0 + slope * alpha), encoding="utf-8")
            rows.append(
                {
                    "alpha_eV": alpha,
                    "stage": stage,
                    "path": str(directory.relative_to(root)),
                    "probe_atom": 1,
                    "channel": "f",
                }
            )
    with (root / "response_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = main(["vasp-lr-analyze", "--root", str(root)])
    assert result["chi0_eV_inv"] == pytest.approx(0.5)
    assert result["chi_eV_inv"] == pytest.approx(0.125)
    assert result["U_response_eV"] == pytest.approx(6.0)
    assert result["health"] == "accepted"
    assert parse_total_charge(root / "a+0.1" / "scf" / "OUTCAR", 1, "f") == pytest.approx(2.0125)


def test_vasp_scrpa_is_version_gated(tmp_path: Path) -> None:
    result = main(
        [
            "vasp-crpa-prepare",
            "--outdir",
            str(tmp_path / "crpa"),
            "--vasp-version",
            "6.2.1",
            "--num-wann",
            "14",
            "--nbands",
            "512",
            "--target-states",
            "1 2 3 4 5 6 7",
            "--crpa-bands",
            "97 98 99 100 101 102 103",
        ]
    )
    assert result["status"] == "blocked-version"
    assert "LSCRPA = .TRUE." in (tmp_path / "crpa" / "03_crpa" / "INCAR.add").read_text()


def test_qe_scaffold_separates_hp_and_wannier_routes(tmp_path: Path) -> None:
    main(["qe-prepare", "--outdir", str(tmp_path / "qe"), "--prefix", "uo2"])
    assert "HUBBARD (ortho-atomic)" in (tmp_path / "qe" / "HUBBARD.atomic.template").read_text()
    assert "HUBBARD (wf)" in (tmp_path / "qe" / "HUBBARD.wf.template").read_text()
    workflow = (tmp_path / "qe" / "WORKFLOW.md").read_text()
    assert "does not" in workflow
    assert "projector-consistent" in workflow
