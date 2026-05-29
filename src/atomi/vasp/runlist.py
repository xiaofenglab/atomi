from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
}


def is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts if part not in {".", ".."})


def discover_poscar_dirs(
    root: Path,
    *,
    poscar_name: str = "POSCAR",
    include_hidden: bool = False,
    max_depth: int | None = None,
) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Search root is not a directory: {root}")

    found: list[Path] = []
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        rel = current_path.relative_to(root)
        depth = 0 if rel == Path(".") else len(rel.parts)
        if not include_hidden:
            dirs[:] = [name for name in dirs if name not in DEFAULT_SKIP_DIRS and not name.startswith(".")]
        else:
            dirs[:] = [name for name in dirs if name not in DEFAULT_SKIP_DIRS]
        if max_depth is not None and depth >= max_depth:
            dirs[:] = []
        if poscar_name in files:
            if include_hidden or not is_hidden_path(rel):
                found.append(current_path)
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


def format_run_path(path: Path, root: Path, *, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    rel = path.resolve().relative_to(root.resolve())
    return "." if rel == Path(".") else rel.as_posix()


def write_runlist(
    root: Path,
    output: Path,
    *,
    poscar_name: str = "POSCAR",
    absolute: bool = False,
    include_hidden: bool = False,
    max_depth: int | None = None,
) -> list[str]:
    root = root.expanduser().resolve()
    output = output.expanduser()
    if not output.is_absolute():
        output = root / output
    dirs = discover_poscar_dirs(
        root,
        poscar_name=poscar_name,
        include_hidden=include_hidden,
        max_depth=max_depth,
    )
    lines = [format_run_path(path, root, absolute=absolute) for path in dirs]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="listvasp",
        description="Find folders containing POSCAR and write their paths to a VASP runlist.txt.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Root directory to scan. Default: current directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("runlist.txt"),
        help="Runlist path. Relative paths are written under ROOT. Default: runlist.txt.",
    )
    parser.add_argument("--absolute", action="store_true", help="Write absolute run directory paths.")
    parser.add_argument(
        "--poscar-name",
        default="POSCAR",
        help="Filename to search for. Default: POSCAR.",
    )
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden directories.")
    parser.add_argument("--max-depth", type=int, help="Maximum directory depth below ROOT to scan.")
    parser.add_argument("--quiet", action="store_true", help="Only write the runlist; do not print the found paths.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_depth is not None and args.max_depth < 0:
        raise ValueError("--max-depth must be nonnegative.")
    lines = write_runlist(
        args.root,
        args.output,
        poscar_name=args.poscar_name,
        absolute=args.absolute,
        include_hidden=args.include_hidden,
        max_depth=args.max_depth,
    )
    root = args.root.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = root / output
    if not args.quiet:
        for line in lines:
            print(line)
        print(f"Wrote {len(lines)} POSCAR folder(s) to {output}")


if __name__ == "__main__":
    main()
