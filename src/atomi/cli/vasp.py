import argparse
from pathlib import Path

from atomi.codes.vasp import format_outcar_summary, summarize_outcar


def _default_outcar() -> Path:
    for name in ("OUTCAR", "OUTCAR.gz"):
        path = Path(name)
        if path.is_file():
            return path
    return Path("OUTCAR")


def extv(argv: list[str] | None = None) -> None:
    """Compatibility command for quick VASP OUTCAR summaries."""
    parser = argparse.ArgumentParser(prog="extv")
    parser.add_argument("outcar", type=Path, nargs="?", default=_default_outcar())
    parser.add_argument(
        "--mag-lines",
        type=int,
        default=50,
        help="Number of final magnetization lines to print. Default: 50.",
    )
    args = parser.parse_args(argv)

    if not args.outcar.is_file():
        raise FileNotFoundError(f"file not found: {args.outcar}")
    summary = summarize_outcar(args.outcar, magnetization_lines=args.mag_lines)
    print(format_outcar_summary(args.outcar, summary))
