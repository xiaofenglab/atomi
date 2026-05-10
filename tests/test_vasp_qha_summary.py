import csv
import gzip
from pathlib import Path

from atomi.vasp.qha_summary import (
    count_disp_runs,
    infer_scale_from_name,
    main,
    parse_outcar,
)


def write_outcar(
    path: Path,
    energy: float = -10.0,
    volume: float = 100.0,
    natoms: int = 12,
) -> None:
    path.write_text(
        f"NIONS = {natoms}\n"
        f" free  energy   TOTEN  = {energy: .6f} eV\n"
        f" volume of cell : {volume: .6f}\n"
        " TOTAL-FORCE (eV/Angst)\n"
        " ---------------------------------------------------\n"
        " 0 0 0  0.1 0.0 0.0\n"
        " 0 0 0  0.0 0.2 0.0\n"
        "\n",
        encoding="utf-8",
    )


def test_infer_scale_from_volume_folder_name() -> None:
    assert infer_scale_from_name("V1.025") == 1.025
    assert infer_scale_from_name("vol_0.975_relaxed") == 0.975


def test_parse_outcar_and_gzip_outcar(tmp_path: Path) -> None:
    outcar = tmp_path / "OUTCAR"
    write_outcar(outcar, energy=-11.0, volume=120.0, natoms=24)
    gz = tmp_path / "OUTCAR.gz"
    with outcar.open("rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())

    rec = parse_outcar(gz)

    assert rec["energy_eV"] == -11.0
    assert rec["volume_A3"] == 120.0
    assert rec["natoms"] == 24
    assert rec["force_max_eVA"] == 0.2


def test_count_disp_runs_counts_result_files(tmp_path: Path) -> None:
    for name in ("disp-001", "disp-002", "disp-003"):
        (tmp_path / name).mkdir()
    (tmp_path / "disp-001" / "vasprun.xml").write_text("<modeling />", encoding="utf-8")
    (tmp_path / "disp-002" / "OUTCAR.gz").write_text("fake", encoding="utf-8")

    counts = count_disp_runs(tmp_path, "disp-*")

    assert counts["n_disp_dirs"] == 3
    assert counts["n_disp_with_result"] == 2
    assert counts["n_vasprun_xml"] == 1
    assert counts["n_OUTCAR_gz"] == 1


def test_qha_summary_scans_volume_folders(tmp_path: Path) -> None:
    root = tmp_path / "2x2x2"
    outdir = tmp_path / "summary"
    for scale, energy, volume in (("V0.980", -9.8, 98.0), ("V1.000", -10.0, 100.0)):
        parent = root / scale / "parent_static"
        parent.mkdir(parents=True)
        write_outcar(parent / "OUTCAR", energy=energy, volume=volume, natoms=12)
        for i in range(2):
            disp = root / scale / f"disp-{i + 1:03d}"
            disp.mkdir()
            (disp / "vasprun.xml.gz").write_text("fake", encoding="utf-8")
        (root / scale / "FORCE_SETS").write_text("forces\n", encoding="utf-8")

    main(
        [
            "--root",
            str(root),
            "--outdir",
            str(outdir),
            "--atoms-per-fu",
            "3",
            "--phonopy-module",
            "phys/phonopy/2.38.1",
            "--no-plot",
        ]
    )

    csv_path = outdir / "qha_volume_summary.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert [row["volume_folder"] for row in rows] == ["V0.980", "V1.000"]
    assert rows[0]["n_disp_dirs"] == "2"
    assert rows[0]["has_FORCE_SETS"] == "True"
    assert "phys/phonopy/2.38.1" in (outdir / "qha_summary_report.txt").read_text(
        encoding="utf-8"
    )
