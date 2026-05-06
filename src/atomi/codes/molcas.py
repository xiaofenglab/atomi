from pathlib import Path


def find_input_files(path: Path) -> list[Path]:
    """Find likely OpenMolcas input files."""
    return sorted([*path.glob("*.input"), *path.glob("*.inp")])

