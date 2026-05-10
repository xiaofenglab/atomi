import csv
import py_compile
from pathlib import Path

import pytest

from atomi.vasp.qha_run import main


def write_outcar(
    path: Path,
    energy: float = -10.0,
    volume: float = 100.0,
    natoms: int = 12,
) -> None:
    path.write_text(
        f"NIONS = {natoms}\n"
        f" free  energy   TOTEN  = {energy: .6f} eV\n"
        f" volume of cell : {volume: .6f}\n",
        encoding="utf-8",
    )


def test_qha_run_writes_ev_manifest_and_script(tmp_path: Path) -> None:
    root = tmp_path / "2x2x2"
    outdir = tmp_path / "qha_run"
    for scale, energy, volume in (("V1.000", -10.0, 100.0), ("V0.980", -9.8, 98.0)):
        parent = root / scale / "parent_static"
        parent.mkdir(parents=True)
        write_outcar(parent / "OUTCAR", energy=energy, volume=volume, natoms=12)
        (root / scale / "thermal_properties.yaml").write_text("thermal\n", encoding="utf-8")

    main(
        [
            "--root",
            str(root),
            "--outdir",
            str(outdir),
            "--phonopy-module",
            "phys/phonopy/2.38.1",
        ]
    )

    assert (outdir / "e-v.dat").read_text(encoding="utf-8").splitlines() == [
        "98.0000000000  -9.8000000000",
        "100.0000000000  -10.0000000000",
    ]
    rows = list(csv.DictReader((outdir / "qha_inputs.csv").open(encoding="utf-8")))
    assert [row["volume_folder"] for row in rows] == ["V0.980", "V1.000"]
    script = (outdir / "run_phonopy_qha.sh").read_text(encoding="utf-8")
    assert "module load phys/phonopy/2.38.1" in script
    assert "phonopy-qha e-v.dat" in script
    assert "../2x2x2/V0.980/thermal_properties.yaml" in script
    assert "../2x2x2/V1.000/thermal_properties.yaml" in script
    assert (outdir / "plot_qha_results.py").exists()
    py_compile.compile(str(outdir / "plot_qha_results.py"), doraise=True)


def test_qha_run_accepts_explicit_volume_folders(tmp_path: Path) -> None:
    root = tmp_path / "2x2x2"
    outside = tmp_path / "phonon_eq" / "V1.002_run"
    outdir = tmp_path / "qha_run"
    for folder, energy, volume in (
        (root / "V1.000", -10.0, 100.0),
        (outside, -9.95, 100.2),
    ):
        parent = folder / "parent_static"
        parent.mkdir(parents=True)
        write_outcar(parent / "OUTCAR", energy=energy, volume=volume, natoms=12)
        (folder / "thermal_properties.yaml").write_text("thermal\n", encoding="utf-8")

    main(
        [
            "--volume-folder",
            str(outside),
            "--volume-folder",
            str(root / "V1.000"),
            "--outdir",
            str(outdir),
        ]
    )

    assert (outdir / "e-v.dat").read_text(encoding="utf-8").splitlines() == [
        "100.0000000000  -10.0000000000",
        "100.2000000000  -9.9500000000",
    ]
    script = (outdir / "run_phonopy_qha.sh").read_text(encoding="utf-8")
    assert "../2x2x2/V1.000/thermal_properties.yaml" in script
    assert "../phonon_eq/V1.002_run/thermal_properties.yaml" in script


def test_qha_run_uses_explicit_thermal_yaml_order(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "volume_folder,scale_factor,root,volume_A3,energy_eV\n"
        "V1.000,1.0,/missing/V1.000,100.0,-10.0\n"
        "V1.005,1.005,/missing/V1.005,101.0,-9.9\n",
        encoding="utf-8",
    )
    t1 = tmp_path / "manual-a.yaml"
    t2 = tmp_path / "manual-b.yaml"
    t1.write_text("a\n", encoding="utf-8")
    t2.write_text("b\n", encoding="utf-8")

    main(
        [
            "--summary-csv",
            str(summary),
            "--outdir",
            str(tmp_path / "qha"),
            "--thermal-yaml",
            str(t2),
            "--thermal-yaml",
            str(t1),
        ]
    )

    manifest = (tmp_path / "qha" / "qha_inputs.csv").read_text(encoding="utf-8")
    assert "../manual-b.yaml" in manifest
    assert "../manual-a.yaml" in manifest


def test_qha_run_can_append_plot_command(tmp_path: Path) -> None:
    root = tmp_path / "2x2x2"
    outdir = tmp_path / "qha_run"
    parent = root / "V1.000" / "parent_static"
    parent.mkdir(parents=True)
    write_outcar(parent / "OUTCAR", energy=-10.0, volume=100.0, natoms=12)
    (root / "V1.000" / "thermal_properties.yaml").write_text("thermal\n", encoding="utf-8")

    main(
        [
            "--root",
            str(root),
            "--outdir",
            str(outdir),
            "--plot-after-qha",
            "--plot-t-min",
            "300",
            "--plot-t-max",
            "1500",
        ]
    )

    script = (outdir / "run_phonopy_qha.sh").read_text(encoding="utf-8")
    assert "python plot_qha_results.py --outdir qha_plots --t-min 300.0 --t-max 1500.0" in script


def test_qha_run_rejects_thermal_yaml_count_mismatch(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "volume_folder,scale_factor,root,volume_A3,energy_eV\n"
        "V1.000,1.0,/missing/V1.000,100.0,-10.0\n"
        "V1.005,1.005,/missing/V1.005,101.0,-9.9\n",
        encoding="utf-8",
    )
    t1 = tmp_path / "manual-a.yaml"
    t1.write_text("a\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(
            [
                "--summary-csv",
                str(summary),
                "--outdir",
                str(tmp_path / "qha"),
                "--thermal-yaml",
                str(t1),
            ]
        )
