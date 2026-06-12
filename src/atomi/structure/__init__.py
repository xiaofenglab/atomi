"""Shared structure and trajectory adapters."""

from .adapters import (
    StructureFrame,
    cell_from_cp2k_input,
    cell_from_xyz_comment,
    read_cp2k_xyz_frames,
    read_vasp_poscar_basis,
    read_vasp_xdatcar_frames,
    vasp_xdatcar_structure_frames,
)
from .elements import (
    ElementInfo,
    ValenceMagmomInfo,
    annotate_symbols,
    atomic_mass_amu,
    atomic_number,
    element_info,
    element_table,
    normalize_element_symbol,
    valence_magmom_info,
    valence_magmom_table,
)

__all__ = [
    "ElementInfo",
    "StructureFrame",
    "ValenceMagmomInfo",
    "annotate_symbols",
    "atomic_mass_amu",
    "atomic_number",
    "cell_from_cp2k_input",
    "cell_from_xyz_comment",
    "element_info",
    "element_table",
    "normalize_element_symbol",
    "read_cp2k_xyz_frames",
    "read_vasp_poscar_basis",
    "read_vasp_xdatcar_frames",
    "valence_magmom_info",
    "valence_magmom_table",
    "vasp_xdatcar_structure_frames",
]
