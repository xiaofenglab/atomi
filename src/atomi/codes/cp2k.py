from pathlib import Path


def find_input_files(path: Path) -> list[Path]:
    """Find likely CP2K input files."""
    return sorted(path.glob("*.inp"))

