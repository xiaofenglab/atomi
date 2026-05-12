import subprocess
import sys
from pathlib import Path


def write_xyz(path: Path) -> None:
    path.write_text(
        "9\n"
        "frame 1\n"
        "Ga 0.0 0.0 0.0\n"
        "Cl 2.40 0.0 0.0\n"
        "Cl 0.0 2.50 0.0\n"
        "Cl 0.0 0.0 2.60\n"
        "Cl -2.70 0.0 0.0\n"
        "O 2.80 0.0 0.0\n"
        "H 3.50 0.0 0.0\n"
        "O 6.00 0.0 0.0\n"
        "H 6.80 0.0 0.0\n"
        "9\n"
        "frame 2\n"
        "Ga 0.0 0.0 0.0\n"
        "Cl 2.45 0.0 0.0\n"
        "Cl 0.0 2.55 0.0\n"
        "Cl 0.0 0.0 2.65\n"
        "Cl -2.75 0.0 0.0\n"
        "O 2.85 0.0 0.0\n"
        "H 3.55 0.0 0.0\n"
        "O 6.10 0.0 0.0\n"
        "H 6.90 0.0 0.0\n",
        encoding="utf-8",
    )


def test_bondtrack_writes_element_labels_and_five_distances(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "atomi"
        / "viz"
        / "cp2k"
        / "cp2k_md_bondtrack.py"
    )
    log = tmp_path / "cp2k.log"
    xyz = tmp_path / "cp2k-pos.xyz"
    out = tmp_path / "cp2k_md_bonds.dat"
    log.write_text("", encoding="utf-8")
    write_xyz(xyz)

    subprocess.run([sys.executable, str(script), str(log), str(xyz), str(out)], check=True)

    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == "# frame min_d max_d mean_d Cl1 Cl2 Cl3 Cl4 O1"
    rows = [line.split() for line in out.read_text(encoding="utf-8").splitlines()[1:]]
    assert len(rows) == 2
    assert len(rows[0]) == 9

    meta = out.with_suffix(".meta").read_text(encoding="utf-8")
    assert "display_count=5" in meta
    assert "display_ligand_types=Cl,O" in meta
    assert "distance_labels=Cl1,Cl2,Cl3,Cl4,O1" in meta
