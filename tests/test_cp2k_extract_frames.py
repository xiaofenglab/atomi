from pathlib import Path

from atomi.cp2k.extract_frames import main, parse_md_step, read_xyz_trajectory


def write_tiny_traj(path: Path) -> None:
    frame = (
        "8\n"
        "MD step = {step}\n"
        "Ga 0.0 0.0 0.0\n"
        "Cl 2.2 0.0 0.0\n"
        "Cl -2.2 0.0 0.0\n"
        "Cl 0.0 2.2 0.0\n"
        "Cl 0.0 -2.2 0.0\n"
        "O 4.5 0.0 0.0\n"
        "H 5.3 0.0 0.0\n"
        "H 4.5 0.8 0.0\n"
    )
    path.write_text(frame.format(step=0) + frame.format(step=100), encoding="utf-8")


def test_parse_md_step_uses_comment_before_fallback() -> None:
    assert parse_md_step("MD step = 1200", fallback_frame=7, traj_every=10) == (1200, True)
    assert parse_md_step("no step here", fallback_frame=7, traj_every=10) == (70, False)


def test_read_xyz_trajectory_reads_multiple_frames(tmp_path: Path) -> None:
    traj = tmp_path / "traj.xyz"
    write_tiny_traj(traj)

    frames = read_xyz_trajectory(traj)

    assert len(frames) == 2
    assert frames[0][1][0] == "Ga"
    assert frames[1][0] == "MD step = 100"


def test_extract_single_frame_writes_cluster_files(tmp_path: Path) -> None:
    traj = tmp_path / "traj.xyz"
    outdir = tmp_path / "extracted"
    write_tiny_traj(traj)

    main(
        [
            str(traj),
            "--frame",
            "1",
            "--system",
            "chloro",
            "--prefix",
            "picked",
            "--outdir",
            str(outdir),
        ]
    )

    frame_dir = outdir / "picked" / "f1"
    assert (frame_dir / "frame_1.xyz").is_file()
    assert (frame_dir / "qm.xyz").is_file()
    assert (frame_dir / "embed.xyz").is_file()
    assert (frame_dir / "pointcharges.dat").is_file()
    assert (frame_dir / "report.txt").is_file()
    assert (outdir / "picked" / "summary.csv").is_file()
