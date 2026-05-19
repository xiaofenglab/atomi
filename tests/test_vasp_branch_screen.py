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


def test_compact_order_shift_lists_magnetic_elements_first() -> None:
    report = branch_screen.BranchReport(frame_id="frame", branch_id="branch", run_dir=Path("branch"))
    report.initial_element_order = {"O": "nonmagnetic", "Gd": "FM", "U": "AFM-like"}
    report.element_order = {"O": "nonmagnetic", "Gd": "FM", "U": "AFM-like"}

    assert branch_screen._compact_order_shift(report) == "Gd:FM,U:AFM-like,O:nonmagnetic"


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


def test_branch_screen_accepts_runlist_and_streams_live_scan(tmp_path: Path, capsys) -> None:
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
    output = capsys.readouterr().out
    assert "Atomi VASP Branch Scan Monitor" in output
    assert "path" in output
    assert "guard" in output
    assert "chg" in output
    assert "order" in output
    assert "1/2" in output
    assert "2/2" in output
    assert "u_site_a" in output
    assert "U:1" in output
    assert "Gd:" in output
    assert "U:" in output
    assert "Pass 1 complete" in output

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
    assert "path" in table
    assert "guard" in table
    assert "chg" in table
    assert "order" in table
    assert "u_site_a" in table
    assert "U:1" in table
    assert "GOOD" in table
    assert "BAD" in table or "WARN" in table


def test_live_monitor_uses_array_artifact_by_runlist_index(tmp_path: Path, capsys) -> None:
    run_a = tmp_path / "frame_005" / "u_site_a"
    run_b = tmp_path / "frame_005" / "u_site_b"
    write_branch(run_a, energy=-8.0, moments=[7.0, 2.0, 0.0])
    write_branch(run_b, energy=-1.0, moments=[7.0, 2.0, 0.0])
    (run_b / "OUTCAR").unlink()
    (run_b / "OSZICAR").unlink()
    artifact_run = tmp_path / "bwforcluster-vasp_array.sbatch.99999.2.260518_030213" / "scratch" / "run"
    write_branch(artifact_run, energy=-20.0, moments=[7.0, 0.1, 0.0])
    runlist = tmp_path / "runlist.txt"
    runlist.write_text("frame_005/u_site_a\nframe_005/u_site_b\n", encoding="utf-8")
    outdir = tmp_path / "out"

    branch_screen.main(
        [
            "--runlist",
            str(runlist),
            "--log-dir",
            str(tmp_path),
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

    output = capsys.readouterr().out
    assert "u_site_b" in output
    rows = read_csv(outdir / "stage1_branch_summary.csv")
    row_b = [row for row in rows if row["branch_id"] == "u_site_b"][0]
    assert row_b["energy_eV"] == "-20.0"
    assert "bwforcluster-vasp_array" in row_b["energy_source"]
    assert "bwforcluster-vasp_array" in row_b["mag_source"]
    assert row_b["output_run_dir"].endswith("scratch/run")
    assert row_b["mag_status"] == "OK"
    assert row_b["physics_guard_status"] == "FAIL"
    assert row_b["current_step"] == "1"


def test_live_monitor_defaults_to_runlist_not_directory_discovery(tmp_path: Path, monkeypatch, capsys) -> None:
    run_a = tmp_path / "frame_004" / "u_site_a"
    extra = tmp_path / "unlisted_but_vasp_like"
    write_branch(run_a, energy=-8.0, moments=[7.0, 2.0, 0.0])
    write_branch(extra, energy=-99.0, moments=[7.0, 2.0, 0.0])
    (tmp_path / "runlist.txt").write_text("frame_004/u_site_a\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    branch_screen.monitor_main(["--outdir", "out", "--live-count", "1", "--refresh", "0.1"])

    output = capsys.readouterr().out
    assert "1/1" in output
    assert "u_site_a" in output
    assert "unlisted" not in output
    rows = read_csv(tmp_path / "out" / "stage1_branch_summary.csv")
    assert len(rows) == 1
    assert rows[0]["run_dir"].endswith("frame_004/u_site_a")


def test_live_monitor_requires_runlist_unless_discover_is_explicit(tmp_path: Path, monkeypatch) -> None:
    write_branch(tmp_path / "unlisted_but_vasp_like", energy=-99.0, moments=[7.0, 2.0, 0.0])
    monkeypatch.chdir(tmp_path)

    try:
        branch_screen.monitor_main(["--outdir", "out", "--live-count", "1"])
    except FileNotFoundError as exc:
        assert "runlist.txt" in str(exc)
    else:
        raise AssertionError("vasp-branch-live should require runlist.txt unless --discover is passed")

    branch_screen.monitor_main(["--discover", "--outdir", "out", "--live-count", "1", "--refresh", "0.1"])
    rows = read_csv(tmp_path / "out" / "stage1_branch_summary.csv")
    assert len(rows) == 1
    assert rows[0]["run_dir"].endswith("unlisted_but_vasp_like")


def test_run_pointer_uses_parent_only_when_needed(tmp_path: Path) -> None:
    shared = [
        branch_screen.BranchInput("f1", "a", tmp_path / "frame_001" / "spin_a"),
        branch_screen.BranchInput("f1", "b", tmp_path / "frame_001" / "spin_b"),
    ]
    labels = branch_screen.branch_pointer_labels(shared)
    assert labels[shared[0].run_dir] == "spin_a"
    assert labels[shared[1].run_dir] == "spin_b"

    mixed = [
        branch_screen.BranchInput("f1", "a", tmp_path / "frame_001" / "spin_a"),
        branch_screen.BranchInput("f2", "a", tmp_path / "frame_002" / "spin_a"),
    ]
    labels = branch_screen.branch_pointer_labels(mixed)
    assert labels[mixed[0].run_dir] == "frame_001/spin_a"
    assert labels[mixed[1].run_dir] == "frame_002/spin_a"
