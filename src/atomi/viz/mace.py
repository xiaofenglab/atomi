import hashlib
import subprocess
from importlib.resources import files
from pathlib import Path

from atomi.viz.vasp_live import ensure_gnuplot


def plot_mace_live(logfile: Path, window: int = 100, refresh: int = 5) -> None:
    """Launch the terminal MACE training monitor."""
    if window < 1:
        raise ValueError("window must be a positive integer.")
    if refresh < 1:
        raise ValueError("refresh must be a positive integer.")
    if not logfile.is_file():
        raise FileNotFoundError(f"file not found: {logfile}")
    ensure_gnuplot()

    script = Path(str(files("atomi").joinpath("viz", "mace", "plot_mace_live.gp")))
    stem = _safe_stem(logfile)
    datafile = Path("/tmp") / f"atomi_mace_{stem}_epochs.dat"
    metafile = Path("/tmp") / f"atomi_mace_{stem}_meta.txt"
    expression = "; ".join(
        [
            f"file='{_quote(logfile)}'",
            f"win={window}",
            f"refresh={refresh}",
            f"datafile='{_quote(datafile)}'",
            f"metafile='{_quote(metafile)}'",
        ]
    )
    subprocess.run(["gnuplot", "-e", expression, str(script)], check=True)


def _safe_stem(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    name = "".join(char if char.isalnum() else "_" for char in path.stem)
    return f"{name}_{digest}"


def _quote(path: Path | str) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "\\'")
