import argparse
from pathlib import Path

from atomi.codes.vasp import format_outcar_summary, summarize_outcar


def extv(argv: list[str] | None = None) -> None:
    """Compatibility command for quick VASP OUTCAR summaries."""
    parser = argparse.ArgumentParser(prog="extv")
    parser.add_argument("outcar", type=Path, nargs="?", default=Path("OUTCAR"))
    args = parser.parse_args(argv)

    if not args.outcar.is_file():
        raise FileNotFoundError(f"file not found: {args.outcar}")
    summary = summarize_outcar(args.outcar)
    print(format_outcar_summary(args.outcar, summary))
