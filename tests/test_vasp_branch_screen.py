from __future__ import annotations

import csv
import json
from pathlib import Path

from atomi.cli.main import main as atomi_main
from atomi.vasp import branch_screen


def write_branch(root: Path, energy: float, moments: list[float], magmom: str = "7 2 0") -> None:
    root.mkdir(parents=True)
    (root / "POSCAR").write_text(
        "\n".join(
            [
                "branch test",
                "1.0",
                "5 0 0",
                "0 5 0",
                "0 0 5",
                "Gd U O",
                "1 1 1",
                "Direct",
                "0 0 0",
                "0.5 0.5 0.5",
                "0.25 0.25 0.25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "INCAR").write_text(f"ISPIN = 2\nMAGMOM = {magmom}\n", encoding="utf-8")
    (root / "OSZICAR").write_text(
        "\n".join(
            [
                "DAV:   1   -1.000000E+01    1.000000E-04",
                "DAV:   2   -1.000100E+01    1.000000E-05",
                f"  1 F= {energy:.8f} E0= {energy:.8f} d E =-.1E-04",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = "\n".join(
        f"{index} 0.000 0.000 0.000 {moment:.6f}"
        for index, moment in enumerate(moments, start=1)
    )
    (root / "OUTCAR").write_text(
        "\n".join(
            [
                "vasp.6.4.3 test",
                "NIONS = 3",
                f"free  energy   TOTEN  = {energy:.8f} eV",
                "magnetization (x)",
                " ion      s      p      d      tot",
                "------------------------------------",
                rows,
                "tot      0      0      0      0",
                "General timing and accounting informations for this job:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_branch_screen_ranks_branches_per_frame(tmp_path: Path) -> None:
    root = tmp_path / "screen"
    write_branch(root / "frame_001" / "spin_low", energy=-10.0, moments=[7.1, 2.0, 0.0])
    write_branch(root / "frame_001" / "spin_high", energy=-7.5, moments=[7.0, 2.0, 0.0])
    outdir = tmp_path / "out"

    branch_screen.main(
        [
            str(root),
            "--outdir",
            str(outdir),
            "--moment-guard",
            "Gd=7@0.5",
            "--moment-guard",
            "U=2@0.5",
            "--energy-window-stop",
            "1.0",
            "--keep-per-frame",
            "1",
        ]
    )

    rows = read_csv(outdir / "stage1_branch_summary.csv")
    by_branch = {row["branch_id"]: row for row in rows}
    assert by_branch["spin_low"]["action"] == "continue"
    assert by_branch["spin_low"]["survivor"] == "True"
    assert by_branch["spin_low"]["rank_in_frame"] == "1"
    assert by_branch["spin_high"]["action"] == "stop"
    assert "higher than frame best" in by_branch["spin_high"]["reasons"]
    assert (outdir / "stage2_survivors_runlist.txt").read_text(encoding="utf-8").strip().endswith("spin_low")

    payload = json.loads((outdir / "stage1_branch_summary.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "atomi.vasp.stage1_branch_screen.v1"
    assert len(payload["reports"]) == 2


def test_branch_screen_uses_index_and_spin_guard_stop(tmp_path: Path) -> None:
    run = tmp_path / "branches" / "frame_002" / "u_lost"
    write_branch(run, energy=-5.0, moments=[7.0, 0.1, 0.0])
    index = tmp_path / "branches.csv"
    index.write_text(f"frame_id,branch_id,run_dir\nframe_002,u_lost,{run}\n", encoding="utf-8")
    outdir = tmp_path / "out"

    atomi_main(
        [
            "vasp-branch-screen",
            "--index",
            str(index),
            "--outdir",
            str(outdir),
            "--moment-guard",
            "U=2@0.5",
            "--spin-fail-action",
            "stop",
            "--track-atom",
            "2",
            "--track-fail-action",
            "stop",
        ]
    )

    rows = read_csv(outdir / "stage1_branch_summary.csv")
    assert rows[0]["action"] == "stop"
    assert rows[0]["physics_guard_status"] == "FAIL"
    assert rows[0]["tracked_site_status"] == "LOST"
    assert "moment guard failed" in rows[0]["reasons"]
    assert (outdir / "stage2_survivors_runlist.txt").read_text(encoding="utf-8") == ""


def test_branch_screen_accepts_runlist_and_formats_live_table(tmp_path: Path) -> None:
    run_a = tmp_path / "frame_003" / "u_site_a"
    run_b = tmp_path / "frame_003" / "u_site_b"
    write_branch(run_a, energy=-8.0, moments=[7.0, 2.0, 0.0])
    write_branch(run_b, energy=-7.0, moments=[7.0, 0.1, 0.0])
    runlist = tmp_path / "runlist.txt"
    runlist.write_text(f"{run_a}\n{run_b}\n", encoding="utf-8")
    outdir = tmp_path / "out"

    branch_screen.main(
        [
            "--runlist",
            str(runlist),
            "--outdir",
            str(outdir),
            "--moment-guard",
            "U=2@0.5",
            "--live",
            "--live-count",
            "1",
            "--refresh",
            "0.1",
        ]
    )

    rows = read_csv(outdir / "stage1_branch_summary.csv")
    assert [row["frame_id"] for row in rows] == ["frame_003", "frame_003"]
    assert rows[0]["current_step"] == "1"
    assert (outdir / "stage2_survivors_runlist.txt").read_text(encoding="utf-8").strip().endswith("u_site_a")

    args = branch_screen.build_parser().parse_args(
        [
            "--runlist",
            str(runlist),
            "--outdir",
            str(outdir),
            "--moment-guard",
            "U=2@0.5",
        ]
    )
    args._moment_guards = branch_screen.parse_moment_guards(args.moment_guard, args.moment_guard_tol)
    args._track_atoms = branch_screen.parse_track_atoms(args.track_atom)
    reports = branch_screen.screen_once(args)
    args.refresh = 10
    table = branch_screen.format_live_table(reports, args, iteration=1)
    assert "Atomi VASP Branch Live Monitor" in table
    assert "u_site_a" in table
    assert "GOOD" in table
    assert "BAD" in table or "WARN" in table
