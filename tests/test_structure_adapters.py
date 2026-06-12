from __future__ import annotations

from pathlib import Path

from atomi.sluschi.bridge import read_vasp_poscar_basis as sluschi_read_vasp_poscar_basis
from atomi.structure.adapters import (
    cell_from_cp2k_input,
    cell_from_xyz_comment,
    read_cp2k_xyz_frames,
    read_vasp_poscar_basis,
    read_vasp_xdatcar_frames,
    vasp_xdatcar_structure_frames,
)
from atomi.structure import StructureFrame
from atomi.structure.elements import (
    annotate_symbols,
    atomic_number,
    element_info,
    element_table,
    normalize_element_symbol,
)


def test_vasp_poscar_and_xdatcar_adapters(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "test\n"
        "1.0\n"
        "2 0 0\n"
        "0 2 0\n"
        "0 0 2\n"
        "Gd U O\n"
        "1 1 2\n"
        "Direct\n"
        "0 0 0\n"
        "0.5 0.5 0.5\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n",
        encoding="utf-8",
    )
    xdatcar = tmp_path / "XDATCAR"
    xdatcar.write_text(
        "test\n"
        "1.0\n"
        "2 0 0\n"
        "0 2 0\n"
        "0 0 2\n"
        "Gd U O\n"
        "1 1 2\n"
        "Direct configuration= 1\n"
        "0 0 0\n"
        "0.5 0.5 0.5\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n"
        "Direct configuration= 2\n"
        "0.1 0 0\n"
        "0.5 0.5 0.5\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n",
        encoding="utf-8",
    )

    basis = read_vasp_poscar_basis(poscar)
    frames = read_vasp_xdatcar_frames(xdatcar, basis["natoms"])
    generic = vasp_xdatcar_structure_frames(poscar, xdatcar)

    assert basis["symbols"] == ["Gd", "U", "O", "O"]
    assert len(frames) == 2
    assert generic[0].natoms == 4
    assert generic[1].frac[0] == [0.1, 0.0, 0.0]
    assert sluschi_read_vasp_poscar_basis(poscar)["natoms"] == 4


def test_cp2k_xyz_and_cell_adapters(tmp_path: Path) -> None:
    inp = tmp_path / "run.inp"
    inp.write_text("&CELL\n  ABC 10.0 11.0 12.0\n&END CELL\n", encoding="utf-8")
    xyz = tmp_path / "traj.xyz"
    xyz.write_text(
        '2\nLattice="10 0 0 0 11 0 0 0 12"\n'
        "Ga 0.0 0.0 0.0\n"
        "Cl 2.0 0.0 0.0\n",
        encoding="utf-8",
    )

    frames = read_cp2k_xyz_frames(xyz)

    assert cell_from_cp2k_input(inp) == [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]]
    assert cell_from_xyz_comment(frames[0]["comment"]) == [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]]
    assert frames[0]["symbols"] == ["Ga", "Cl"]
    assert frames[0]["coords"][1] == [2.0, 0.0, 0.0]


def test_element_metadata_normalizes_charged_labels_and_vacancies() -> None:
    assert normalize_element_symbol("u5+") == "U"
    assert normalize_element_symbol("Gd3+") == "Gd"
    assert normalize_element_symbol("Va") is None
    assert atomic_number("U5+") == 92

    uranium = element_info("U5+", include_xray_edges=True, edges=("L3",))

    assert uranium is not None
    assert uranium.symbol == "U"
    assert uranium.atomic_mass_amu and uranium.atomic_mass_amu > 230.0
    assert uranium.xray_edges_eV and uranium.xray_edges_eV["L3"] > 17000.0


def test_element_table_and_frame_metadata_cover_all_symbols() -> None:
    rows = annotate_symbols(["Gd3+", "U4+", "O2-", "Va"])
    table = element_table(["Gd3+", "U4+", "O2-", "Va"])
    frame = StructureFrame(symbols=["Gd3+", "U4+", "O2-", "Va"])

    assert rows[-1]["is_vacancy"] is True
    assert set(table) == {"Gd", "U", "O"}
    assert table["Gd"]["atomic_number"] == 64
    assert frame.element_table()["U"]["atomic_number"] == 92
    assert frame.symbol_metadata()[-1]["is_vacancy"] is True
