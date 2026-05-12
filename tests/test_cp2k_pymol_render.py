import tarfile
from pathlib import Path

from atomi.cp2k.pymol_render import main


def write_xyz(path: Path) -> None:
    path.write_text(
        "4\n"
        "MD step = 1\n"
        "Ga 0.0 0.0 0.0\n"
        "Cl 2.5 0.0 0.0\n"
        "O 2.7 0.0 0.0\n"
        "H 3.5 0.0 0.0\n",
        encoding="utf-8",
    )


def test_pymol_render_writes_workspace_and_archive(tmp_path: Path) -> None:
    xyz = tmp_path / "traj.xyz"
    outdir = tmp_path / "render"
    archive = tmp_path / "render_download.tar.gz"
    write_xyz(xyz)

    main(
        [
            str(xyz),
            "--outdir",
            str(outdir),
            "--reference-state",
            "1",
            "--start",
            "1",
            "--stop",
            "10",
            "--step",
            "2",
            "--snapshot",
            "1",
            "--snapshot",
            "5",
            "--no-ray",
            "--ga-o-cutoff",
            "2.70",
            "--archive",
            "--archive-path",
            str(archive),
        ]
    )

    helper = outdir / "aimd_render_dynamic.py"
    driver = outdir / "render_movie.pml"
    run_script = outdir / "run_pymol_render.sh"
    movie_script = outdir / "make_movie.sh"
    pack_script = outdir / "pack_for_download.sh"

    assert helper.is_file()
    assert driver.is_file()
    assert run_script.is_file()
    assert movie_script.is_file()
    assert pack_script.is_file()
    assert (outdir / "frames").is_dir()
    assert (outdir / "snapshots").is_dir()

    helper_text = helper.read_text(encoding="utf-8")
    assert 'TRAJ_OBJECT = "traj"' in helper_text
    assert "GA_O_CUTOFF = 2.7" in helper_text
    assert "cmd.extend(\"render_movie\", render_movie)" in helper_text

    driver_text = driver.read_text(encoding="utf-8")
    assert f"load {xyz.resolve()}, traj" in driver_text
    assert "run aimd_render_dynamic.py" in driver_text
    assert "set_reference_view 1" in driver_text
    assert "snapshot 1, snapshots/snapshot_0001.png" in driver_text
    assert "snapshot 5, snapshots/snapshot_0005.png" in driver_text
    assert "render_movie 1, 10, frames/frame, 2, 0" in driver_text

    assert archive.is_file()
    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
    assert "render/render_movie.pml" in names
    assert "render/aimd_render_dynamic.py" in names
