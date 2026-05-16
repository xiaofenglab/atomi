from __future__ import annotations

import json
from pathlib import Path

from atomi.xafs.cp2k import build_parser, parse_cp2k_input_metadata, read_cp2k_xyz_trajectory, run_prepare


def write_cp2k_traj(path: Path) -> None:
    frame = (
        "6\n"
        "i = {step}, time = {time}\n"
        "Ga 0.0 0.0 0.0\n"
        "Cl 2.2 0.0 0.0\n"
        "Cl -2.2 0.0 0.0\n"
        "O 3.0 0.0 0.0\n"
        "H 3.8 0.0 0.0\n"
        "H 3.0 0.8 0.0\n"
    )
    path.write_text(
        frame.format(step=0, time=0.0)
        + frame.format(step=500, time=0.5)
        + frame.format(step=1000, time=1.0),
        encoding="utf-8",
    )


def write_cp2k_input(path: Path) -> None:
    path.write_text(
        "&GLOBAL\n"
        "  PROJECT ga_xafs\n"
        "&END GLOBAL\n"
        "&FORCE_EVAL\n"
        "  &SUBSYS\n"
        "    &CELL\n"
        "      ABC 18.0 18.0 18.0\n"
        "    &END CELL\n"
        "    &TOPOLOGY\n"
        "      COORD_FILE_NAME ga.xyz\n"
        "    &END TOPOLOGY\n"
        "  &END SUBSYS\n"
        "&END FORCE_EVAL\n"
        "&MOTION\n"
        "  &MD\n"
        "    TIMESTEP 1.0\n"
        "    STEPS 1000\n"
        "    TEMPERATURE 300\n"
        "  &END MD\n"
        "&END MOTION\n",
        encoding="utf-8",
    )


def test_cp2k_xyz_reader_and_input_metadata(tmp_path: Path) -> None:
    traj = tmp_path / "run-pos.xyz"
    inp = tmp_path / "run.inp"
    write_cp2k_traj(traj)
    write_cp2k_input(inp)

    frames = read_cp2k_xyz_trajectory(traj)
    info = parse_cp2k_input_metadata(inp)

    assert len(frames) == 3
    assert frames[-1].step == 1000
    assert info["project"] == "ga_xafs"
    assert info["cell_abc_A"] == [18.0, 18.0, 18.0]
    assert info["timestep_fs"] == 1.0


def test_xafs_cp2k_prepare_writes_feff_clusters_from_last_ps(tmp_path: Path) -> None:
    traj = tmp_path / "run-pos.xyz"
    inp = tmp_path / "run.inp"
    outdir = tmp_path / "xafs_cp2k"
    write_cp2k_traj(traj)
    write_cp2k_input(inp)

    args = build_parser().parse_args(
        [
            "--xyz",
            str(traj),
            "--inp",
            str(inp),
            "--outdir",
            str(outdir),
            "--last-ps",
            "0.6",
            "--max-absorber-sites",
            "1",
            "--cluster-radius",
            "4.0",
        ]
    )
    summary = run_prepare(args)

    assert summary["absorber"] == "Ga"
    assert summary["n_frames"] == 2
    assert summary["n_clusters"] == 2
    assert summary["cell_source"] == "CP2K input &CELL ABC"
    feff_inputs = sorted(outdir.glob("clusters/frame_*/site_*/feff.inp"))
    assert len(feff_inputs) == 2
    text = feff_inputs[0].read_text(encoding="utf-8")
    assert "EDGE L3" in text
    assert "Ga_absorber" in text
    metadata = json.loads((outdir / "xafs_cp2k_prepare_metadata.json").read_text(encoding="utf-8"))
    assert metadata["frame_summary"]["pbc_used"] is True
    assert metadata["source"]["input_metadata"]["temperature"] == 300.0


def test_xafs_cp2k_prepare_accepts_user_defined_metal(tmp_path: Path) -> None:
    traj = tmp_path / "run-pos.xyz"
    outdir = tmp_path / "xafs_cp2k"
    write_cp2k_traj(traj)

    args = build_parser().parse_args(
        [
            "--xyz",
            str(traj),
            "--outdir",
            str(outdir),
            "--metal",
            "Ga",
            "--last-frames",
            "1",
            "--max-absorber-sites",
            "1",
        ]
    )
    summary = run_prepare(args)

    assert summary["absorber_request"] == "Ga"
    assert summary["frame_summary"]["pbc_used"] is False
    assert (outdir / "cluster_dirs.txt").exists()
