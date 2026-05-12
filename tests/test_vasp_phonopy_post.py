from pathlib import Path

import pytest

from atomi.vasp.phonopy_band import load_band_yaml, main as band_plot_main, segment_indices
from atomi.vasp.phonopy_post import band_path_for_dup, main, read_poscar


POSCAR_TEXT = """UO2 conventional cell
1.0
5.47 0.0 0.0
0.0 5.47 0.0
0.0 0.0 5.47
U O
4 8
Direct
0.0 0.0 0.0
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5
0.25 0.25 0.25
0.75 0.75 0.25
0.75 0.25 0.75
0.25 0.75 0.75
0.75 0.75 0.75
0.25 0.25 0.75
0.25 0.75 0.25
0.75 0.25 0.25
"""

BAND_YAML = """segment_nqpoint:
- 2
- 2
labels:
- [G, X]
- [X, L]
phonon:
- distance: 0.0
  band:
  - frequency: 0.0
  - frequency: 1.0
- distance: 0.5
  band:
  - frequency: 0.4
  - frequency: 1.4
- distance: 0.5
  band:
  - frequency: 0.5
  - frequency: 1.5
- distance: 1.0
  band:
  - frequency: 0.8
  - frequency: 1.8
"""


def test_read_poscar_supercell_info(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    poscar.write_text(POSCAR_TEXT, encoding="utf-8")

    info = read_poscar(poscar, (2, 2, 2))

    assert info.natoms == 12
    assert info.species == ["U", "O"]
    assert info.counts == [4, 8]
    assert info.reference_lengths == (2.735, 2.735, 2.735)


def test_band_path_presets_and_override() -> None:
    assert band_path_for_dup((2, 2, 2)).startswith("0 0 0  0.5 0 0")
    assert "0 0.5 0.5" in band_path_for_dup((2, 1, 1))
    assert band_path_for_dup((1, 1, 1), "0 0 0   0.5 0.5 0") == "0 0 0 0.5 0.5 0"


def test_vasp_phonopy_post_writes_env_run_and_sbatch(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    outdir = tmp_path / "post"
    poscar.write_text(POSCAR_TEXT, encoding="utf-8")

    main(
        [
            "--poscar",
            str(poscar),
            "--dup",
            "2",
            "2",
            "2",
            "--mesh",
            "12",
            "12",
            "12",
            "--outdir",
            str(outdir),
            "--phonopy-module",
            "phys/phonopy/2.38.1",
        ]
    )

    env_text = (outdir / "poscar_info.env").read_text(encoding="utf-8")
    run_text = (outdir / "run_phonopy_post.sh").read_text(encoding="utf-8")
    sbatch_text = (outdir / "submit_phonopy_post.sbatch").read_text(encoding="utf-8")
    summary_text = (outdir / "phonopy_post_summary.txt").read_text(encoding="utf-8")

    assert 'export MESH="12 12 12"' in env_text
    assert 'export BAND_PATH="' in env_text
    assert "module load phys/phonopy/2.38.1" in run_text
    assert "phonopy -f disp-*/vasprun.xml" in run_text
    assert "phonopy-load --mesh ${MESH} -t" in run_text
    assert "phonopy-load --mesh ${MESH} --dos" in run_text
    assert 'phonopy-load --band "${BAND_PATH}"' in run_text
    assert "plot_phonopy_band.py" in run_text
    assert "phonon_band.png" in run_text
    assert "thermal_properties.yaml" in run_text
    assert "total_dos.dat" in run_text
    assert "band.yaml" in run_text
    assert (outdir / "plot_phonopy_band.py").exists()
    assert "#SBATCH --mem=96G" in sbatch_text
    assert "bash run_phonopy_post.sh" in sbatch_text
    assert "Inferred reference cell" in summary_text


def test_phonopy_band_plotter_reads_segments_and_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    pytest.importorskip("matplotlib")
    band_yaml = tmp_path / "band.yaml"
    outpng = tmp_path / "phonon_band.png"
    band_yaml.write_text(BAND_YAML, encoding="utf-8")

    distances, frequencies, segment_nqpoint, labels = load_band_yaml(band_yaml)

    assert distances == [0.0, 0.5, 0.5, 1.0]
    assert frequencies[0] == [0.0, 1.0]
    assert segment_indices(segment_nqpoint, len(distances)) == [(0, 2), (2, 4)]
    assert labels == [["G", "X"], ["X", "L"]]

    band_plot_main(["--band-yaml", str(band_yaml), "--outpng", str(outpng)])

    assert outpng.exists()
    assert outpng.stat().st_size > 0
