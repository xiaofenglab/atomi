from pathlib import Path


def find_input_files(path: Path) -> list[Path]:
    """Find likely LAMMPS input files."""
    return sorted(path.glob("in.*"))

