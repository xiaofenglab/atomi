from pathlib import Path


def has_control_file(path: Path) -> bool:
    """Return whether a Turbomole control file exists."""
    return (path / "control").exists()

