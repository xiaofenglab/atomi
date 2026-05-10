import argparse
import gzip
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_KEEP_PATTERNS = [
    "*.inp",
    "*.restart",
    "*-RESTART.wfn",
    "*-pos.xyz",
    "*-pos-1.xyz",
    "*.log.gz",
    "*.out.gz",
    "README*",
]

DEFAULT_LOG_PATTERNS = ["*.log", "*.out"]
DEFAULT_DROP_PATTERNS = [
    "*-vel.xyz",
    "*.bak*",
    "*.tmp",
    "*.cube",
    "*.wfn.bak*",
    "*-RESTART.wfn.bak*",
    "*RESTART_HISTORY*",
    "*.restart_hist*",
]


@dataclass(frozen=True)
class Action:
    kind: str
    source: Path
    target: Path | None = None
    note: str = ""


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file())


def matches_any(path: Path, patterns: list[str]) -> bool:
    return any(path.match(pattern) for pattern in patterns)


def gzip_file(path: Path, level: int = 9, keep_original: bool = False) -> Path:
    target = path.with_name(path.name + ".gz")
    with path.open("rb") as src, gzip.open(target, "wb", compresslevel=level) as dst:
        shutil.copyfileobj(src, dst)
    if not keep_original:
        path.unlink()
    return target


def read_xyz_frame(handle) -> tuple[str, list[str]] | None:
    line = handle.readline()
    if not line:
        return None
    if not line.strip():
        return None
    natoms = int(line.strip())
    comment = handle.readline()
    atoms = [handle.readline() for _ in range(natoms)]
    if len(atoms) != natoms or any(not atom for atom in atoms):
        raise ValueError("truncated XYZ trajectory")
    return line, [comment] + atoms


def thin_xyz_trajectory(
    input_path: Path,
    output_path: Path,
    stride: int,
    keep_last: bool = True,
) -> int:
    if stride <= 0:
        raise ValueError("trajectory stride must be positive")
    kept = 0
    last_frame: tuple[str, list[str]] | None = None
    last_written_index = -1
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        iframe = 0
        while True:
            frame = read_xyz_frame(src)
            if frame is None:
                break
            last_frame = frame
            if iframe % stride == 0:
                dst.write(frame[0])
                dst.writelines(frame[1])
                kept += 1
                last_written_index = iframe
            iframe += 1
        if keep_last and last_frame is not None and last_written_index != iframe - 1:
            dst.write(last_frame[0])
            dst.writelines(last_frame[1])
            kept += 1
    return kept


def classify_actions(args: argparse.Namespace) -> tuple[list[Action], list[Path]]:
    root = args.run_dir
    files = iter_files(root)
    keep_patterns = DEFAULT_KEEP_PATTERNS + args.keep_pattern
    drop_patterns = DEFAULT_DROP_PATTERNS + args.drop_pattern
    actions: list[Action] = []
    kept: list[Path] = []

    reduced_names = set()
    for traj in files:
        if args.reduce_trajectory_stride and traj.match("*-pos*.xyz"):
            reduced = traj.with_name(f"{traj.stem}_stride{args.reduce_trajectory_stride}.xyz")
            reduced_names.add(reduced.name)
            actions.append(
                Action(
                    "thin-trajectory",
                    traj,
                    reduced,
                    f"keep every {args.reduce_trajectory_stride} frames plus final frame",
                )
            )
            if args.replace_trajectory:
                actions.append(Action("move-to-trash", traj, root / args.trash_dir / traj.name))
            else:
                kept.append(traj)

    for path in files:
        if path.name in reduced_names:
            continue
        if path.suffix == ".gz":
            kept.append(path)
            continue
        if matches_any(path, DEFAULT_LOG_PATTERNS):
            actions.append(
                Action("gzip", path, path.with_name(path.name + ".gz"), "compressed log")
            )
            continue
        if matches_any(path, keep_patterns):
            kept.append(path)
            continue
        if matches_any(path, drop_patterns):
            actions.append(Action("move-to-trash", path, root / args.trash_dir / path.name))
            continue
        if args.move_unknown:
            actions.append(
                Action("move-to-trash", path, root / args.trash_dir / path.name, "unknown file")
            )
        else:
            kept.append(path)
    return actions, sorted(set(kept))


def execute_actions(actions: list[Action], args: argparse.Namespace) -> None:
    trash = args.run_dir / args.trash_dir
    for action in actions:
        if action.kind == "gzip":
            gzip_file(action.source, level=args.gzip_level)
        elif action.kind == "thin-trajectory":
            assert action.target is not None
            thin_xyz_trajectory(
                action.source,
                action.target,
                stride=args.reduce_trajectory_stride,
                keep_last=True,
            )
        elif action.kind == "move-to-trash":
            assert action.target is not None
            trash.mkdir(exist_ok=True)
            if action.target.exists():
                raise FileExistsError(f"refusing to overwrite existing trash file: {action.target}")
            shutil.move(str(action.source), str(action.target))
        elif action.kind == "delete":
            action.source.unlink()
        else:
            raise ValueError(f"unknown action kind: {action.kind}")


def write_manifest(path: Path, actions: list[Action], kept: list[Path], executed: bool) -> None:
    payload = {
        "executed": executed,
        "kept": [str(item) for item in kept],
        "actions": [
            {
                "kind": action.kind,
                "source": str(action.source),
                "target": str(action.target) if action.target else None,
                "note": action.note,
            }
            for action in actions
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_plan(actions: list[Action], kept: list[Path], manifest: Path, execute: bool) -> None:
    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"CP2K clean-run plan ({mode})")
    print(f"Manifest: {manifest}")
    print("\nKeep:")
    for path in kept:
        print(f"  keep           {path.name}")
    print("\nActions:")
    if not actions:
        print("  none")
    for action in actions:
        target = f" -> {action.target.name}" if action.target else ""
        note = f"  # {action.note}" if action.note else ""
        print(f"  {action.kind:15s} {action.source.name}{target}{note}")
    if not execute:
        print("\nNo files changed. Rerun with --execute to apply this plan.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cp2k-clean-run",
        description="Conservatively clean CP2K AIMD run folders while preserving rerun records.",
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--execute", action="store_true", help="Apply the cleanup plan.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--trash-dir", default="_atomi_removed")
    parser.add_argument("--gzip-level", type=int, default=9)
    parser.add_argument("--keep-pattern", action="append", default=[])
    parser.add_argument("--drop-pattern", action="append", default=[])
    parser.add_argument("--move-unknown", action="store_true")
    parser.add_argument(
        "--reduce-trajectory-stride",
        type=int,
        default=None,
        help="Write reduced *-pos*.xyz trajectories with every Nth frame plus final frame.",
    )
    parser.add_argument(
        "--replace-trajectory",
        action="store_true",
        help="After writing reduced trajectory, move original trajectory to trash.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if not args.run_dir.is_dir():
        parser.error(f"run directory not found: {args.run_dir}")
    if args.gzip_level < 1 or args.gzip_level > 9:
        parser.error("--gzip-level must be between 1 and 9")
    if args.replace_trajectory and not args.reduce_trajectory_stride:
        parser.error("--replace-trajectory requires --reduce-trajectory-stride")
    actions, kept = classify_actions(args)
    manifest = args.manifest or args.run_dir / "atomi_clean_manifest.json"
    print_plan(actions, kept, manifest, args.execute)
    if args.execute:
        execute_actions(actions, args)
    write_manifest(manifest, actions, kept, executed=args.execute)
    if args.execute:
        print("\nCleanup complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
