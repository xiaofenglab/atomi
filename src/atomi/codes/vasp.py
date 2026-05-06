from pathlib import Path


REQUIRED_INPUTS = ("INCAR", "POSCAR", "POTCAR", "KPOINTS")


def missing_inputs(path: Path) -> list[str]:
    """Return required VASP input files missing from a directory."""
    return [name for name in REQUIRED_INPUTS if not (path / name).exists()]

