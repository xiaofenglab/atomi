from pathlib import Path
from typing import NamedTuple


REQUIRED_INPUTS = ("INCAR", "POSCAR", "POTCAR", "KPOINTS")


def missing_inputs(path: Path) -> list[str]:
    """Return required VASP input files missing from a directory."""
    return [name for name in REQUIRED_INPUTS if not (path / name).exists()]


class OutcarSummary(NamedTuple):
    final_total_energy_line: str | None
    total_energy_changes: list[str]
    fermi_energy_line: str | None
    final_force_block: list[str]
    final_magnetization_table: list[str]
    final_lattice_vectors: list[str]
    max_force: float | None


def summarize_outcar(path: Path) -> OutcarSummary:
    """Extract a compact summary from a VASP OUTCAR file."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return OutcarSummary(
        final_total_energy_line=_last_matching(lines, "free energy    TOTEN"),
        total_energy_changes=_last_n_matching(lines, "total energy-change", 5),
        fermi_energy_line=_last_matching(lines, "E-fermi"),
        final_force_block=_last_block_after(lines, "TOTAL-FORCE (eV/Angst)", after=3, tail=20),
        final_magnetization_table=_last_block_after(lines, "magnetization", after=50, tail=50),
        final_lattice_vectors=_last_block_after(lines, "direct lattice vectors", after=3, tail=3),
        max_force=_max_force(lines),
    )


def format_outcar_summary(path: Path, summary: OutcarSummary) -> str:
    """Format an OUTCAR summary in the style of the original extv.sh helper."""
    parts = [
        "============================================",
        f"Extracting from: {path}",
        "============================================",
        "",
        "Final total energy (eV):",
        summary.final_total_energy_line or "NA",
        "",
        "total energy change",
        *_or_na(summary.total_energy_changes),
        "",
        "Fermi energy (eV):",
        summary.fermi_energy_line or "NA",
        "",
        "Final force",
        *_or_na(summary.final_force_block),
        "",
        "Final magnetization table:",
        *_or_na(summary.final_magnetization_table),
        "",
        "Final lattice vectors",
        *_or_na(summary.final_lattice_vectors),
        "",
        "Max force magnitude",
        f"Max |F| = {_fmt(summary.max_force)} eV/A",
        "============================================",
        "Done.",
    ]
    return "\n".join(parts)


def _last_matching(lines: list[str], needle: str) -> str | None:
    for line in reversed(lines):
        if needle in line:
            return line
    return None


def _last_n_matching(lines: list[str], needle: str, n: int) -> list[str]:
    return [line for line in lines if needle in line][-n:]


def _last_block_after(lines: list[str], needle: str, after: int, tail: int) -> list[str]:
    last_index = None
    for index, line in enumerate(lines):
        if needle in line:
            last_index = index
    if last_index is None:
        return []
    return lines[last_index : last_index + after + 1][-tail:]


def _max_force(lines: list[str]) -> float | None:
    in_force_block = False
    max_force = None
    for line in lines:
        if "TOTAL-FORCE (eV/Angst)" in line:
            in_force_block = True
            max_force = None
            continue
        if in_force_block and "total drift" in line:
            in_force_block = False
            continue
        if not in_force_block:
            continue

        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
        except ValueError:
            continue
        magnitude = (fx * fx + fy * fy + fz * fz) ** 0.5
        if max_force is None or magnitude > max_force:
            max_force = magnitude
    return max_force


def _or_na(lines: list[str]) -> list[str]:
    return lines if lines else ["NA"]


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.8g}"
