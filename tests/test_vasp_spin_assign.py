from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomi.cli.main import main as atomi_main
from atomi.vasp.magmom import existing_magmom_values, read_poscar_structure
from atomi.vasp.spin_assign import main as spin_assign_main


def write_o_u_poscar(path: Path) -> None:
    positions = [
        ("O", (0.11, 0.0, 0.0)),
        ("O", (0.22, 0.0, 0.0)),
        ("O", (0.33, 0.0, 0.0)),
        ("O", (0.44, 0.0, 0.0)),
        ("O", (0.55, 0.0, 0.0)),
        ("O", (0.66, 0.0, 0.0)),
        ("O", (0.77, 0.0, 0.0)),
        ("O", (0.88, 0.0, 0.0)),
        ("O", (0.99, 0.0, 0.0)),
        ("U", (0.00, 0.0, 0.0)),
        ("U", (0.10, 0.0, 0.0)),
        ("U", (0.20, 0.0, 0.0)),
        ("U", (0.30, 0.0, 0.0)),
        ("U", (0.40, 0.0, 0.0)),
        ("U", (0.50, 0.0, 0.0)),
    ]
    path.write_text(
        "O U source\n"
        "1.0\n"
        "6 0 0\n"
        "0 6 0\n"
        "0 0 6\n"
        "O U\n"
        "9 6\n"
        "Direct\n"
        + "".join(f"{x:.8f} {y:.8f} {z:.8f}\n" for _symbol, (x, y, z) in positions),
        encoding="utf-8",
    )


def write_u_o_poscar(path: Path) -> None:
    positions = [
        ("U", (0.00, 0.0, 0.0)),
        ("U", (0.10, 0.0, 0.0)),
        ("O", (0.11, 0.0, 0.0)),
        ("O", (0.22, 0.0, 0.0)),
    ]
    path.write_text(
        "U O target\n"
        "1.0\n"
        "6 0 0\n"
        "0 6 0\n"
        "0 0 6\n"
        "U O\n"
        "2 2\n"
        "Direct\n"
        + "".join(f"{x:.8f} {y:.8f} {z:.8f}\n" for _symbol, (x, y, z) in positions),
        encoding="utf-8",
    )


def write_u8_o2_poscar(path: Path) -> None:
    path.write_text(
        "U8 O2 target\n"
        "1.0\n"
        "8 0 0\n"
        "0 8 0\n"
        "0 0 8\n"
        "U O\n"
        "8 2\n"
        "Direct\n"
        + "".join(f"{0.1 * i:.8f} 0.00000000 0.00000000\n" for i in range(8))
        + "0.10000000 0.50000000 0.50000000\n"
        + "0.20000000 0.50000000 0.50000000\n",
        encoding="utf-8",
    )


def write_o_u_cif(path: Path) -> None:
    path.write_text(
        "data_ou\n"
        "_cell_length_a 6\n"
        "_cell_length_b 6\n"
        "_cell_length_c 6\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "O1 O 0.11 0.0 0.0 1\n"
        "O2 O 0.22 0.0 0.0 1\n"
        "O3 O 0.33 0.0 0.0 1\n"
        "O4 O 0.44 0.0 0.0 1\n"
        "U1 U 0.00 0.0 0.0 1\n"
        "U2 U 0.10 0.0 0.0 1\n",
        encoding="utf-8",
    )


def test_assign_spins_reorders_poscar_and_ldau_species_tags(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    incar = tmp_path / "INCAR"
    out = tmp_path / "spin_assigned"
    write_o_u_poscar(poscar)
    incar.write_text(
        "ENCUT = 520\n"
        "NUPDOWN = 0\n"
        "LDAU = .TRUE.\n"
        "LDAUL = -1 3\n"
        "LDAUU = 0.0 4.0\n"
        "LDAUJ = 0 0\n"
        "MAGMOM = 15*0\n",
        encoding="utf-8",
    )

    spin_assign_main(
        [
            "--poscar",
            str(poscar),
            "--incar",
            str(incar),
            "--outdir",
            str(out),
            "--cation-elements",
            "U",
            "--anion-elements",
            "O",
            "--moment",
            "U=2,O=0",
            "--special-moment",
            "U:2-3=1",
            "--magnetic-order",
            "afm",
        ]
    )

    structure = read_poscar_structure(out / "POSCAR")
    assert structure.species.symbols == ["U", "O"]
    assert structure.species.counts == [6, 9]
    moments = existing_magmom_values(out / "INCAR", 15)
    assert moments == pytest.approx([2, -1, 1, -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    incar_text = (out / "INCAR").read_text(encoding="utf-8")
    assert "#NUPDOWN = 0" in incar_text
    assert "LDAUL = 3 -1" in incar_text
    assert "LDAUU = 4.0 0.0" in incar_text
    plan = json.loads((out / "spin_assignment_plan.json").read_text(encoding="utf-8"))
    assert plan["output_species_order"] == ["U", "O"]
    assert plan["moment_summary"]["U"]["unique_abs_moments"] == [1.0, 2.0]
    assert len(plan["special_rule_atoms"]) == 2


def test_assign_spins_accepts_cif_and_reorders_ldau_tags(tmp_path: Path) -> None:
    cif = tmp_path / "structure.cif"
    incar = tmp_path / "INCAR"
    out = tmp_path / "spin_assigned_cif"
    write_o_u_cif(cif)
    incar.write_text(
        "ENCUT = 520\n"
        "LDAU = .TRUE.\n"
        "LDAUL = -1 3\n"
        "LDAUU = 0.0 4.0\n"
        "LDAUJ = 0 0\n"
        "MAGMOM = 6*0\n",
        encoding="utf-8",
    )

    spin_assign_main(
        [
            "--cif",
            str(cif),
            "--incar",
            str(incar),
            "--outdir",
            str(out),
            "--cation-elements",
            "U",
            "--anion-elements",
            "O",
            "--moment",
            "U=2,O=0",
            "--special-moment",
            "U:2=1",
            "--magnetic-order",
            "afm",
        ]
    )

    structure = read_poscar_structure(out / "POSCAR")
    assert structure.species.symbols == ["U", "O"]
    assert structure.species.counts == [2, 4]
    assert existing_magmom_values(out / "INCAR", 6) == pytest.approx([2, -1, 0, 0, 0, 0])
    incar_text = (out / "INCAR").read_text(encoding="utf-8")
    assert "LDAUL = 3 -1" in incar_text
    assert "LDAUU = 4.0 0.0" in incar_text
    plan = json.loads((out / "spin_assignment_plan.json").read_text(encoding="utf-8"))
    assert plan["source_format"].startswith("cif/")
    assert plan["source_species_order"] == ["O", "U"]
    assert plan["output_species_order"] == ["U", "O"]


def test_assign_spins_uses_template_poscar_for_incar_ldau_order(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    template = tmp_path / "template"
    template.mkdir()
    out = tmp_path / "spin_assigned"
    write_u_o_poscar(poscar)
    write_o_u_poscar(template / "POSCAR")
    (template / "INCAR").write_text(
        "ENCUT = 520\n"
        "LDAU = .TRUE.\n"
        "LDAUL = -1 3\n"
        "LDAUU = 0.0 4.0\n"
        "LDAUJ = 0 0\n"
        "MAGMOM = 4*0\n",
        encoding="utf-8",
    )

    spin_assign_main(
        [
            "--poscar",
            str(poscar),
            "--incar",
            str(template / "INCAR"),
            "--outdir",
            str(out),
            "--cation-elements",
            "U",
            "--anion-elements",
            "O",
            "--moment",
            "U=2,O=0",
        ]
    )

    incar_text = (out / "INCAR").read_text(encoding="utf-8")
    assert "LDAUL = 3 -1" in incar_text
    assert "LDAUU = 4.0 0.0" in incar_text
    assert "LDAUJ = 0 0" in incar_text
    plan = json.loads((out / "spin_assignment_plan.json").read_text(encoding="utf-8"))
    assert plan["source_species_order"] == ["U", "O"]
    assert plan["source_incar_species_order"] == ["O", "U"]
    assert plan["output_species_order"] == ["U", "O"]


def test_assign_spins_can_seed_balance_existing_magmom_counts(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    incar = tmp_path / "INCAR"
    out = tmp_path / "balanced"
    write_u8_o2_poscar(poscar)
    incar.write_text(
        "ENCUT = 520\n"
        "MAGMOM = 2 -2 1 -1 1 -1 2 -2 2*0\n",
        encoding="utf-8",
    )

    spin_assign_main(
        [
            "--poscar",
            str(poscar),
            "--incar",
            str(incar),
            "--outdir",
            str(out),
            "--species-order",
            "U,O",
            "--base-magmom-from-incar",
            "--special-moment",
            "U:3-4=2",
            "--balance-moment",
            "U:1=4,2=4",
            "--balance-seed",
            "20260605",
            "--magnetic-order",
            "afm",
        ]
    )

    moments = existing_magmom_values(out / "INCAR", 10)
    assert moments is not None
    u_moments = moments[:8]
    assert u_moments[2:4] == pytest.approx([2, -2])
    assert sum(1 for value in u_moments if abs(value) == 1) == 4
    assert sum(1 for value in u_moments if abs(value) == 2) == 4
    assert moments[8:] == pytest.approx([0, 0])
    plan = json.loads((out / "spin_assignment_plan.json").read_text(encoding="utf-8"))
    assert plan["base_magmom_from_incar"] is True
    assert plan["balance_rules"] == {"U": {"1.0": 4, "2.0": 4}}
    assert len(plan["balance_changes"]) == 2
    assert {change["element_index_1based"] for change in plan["balance_changes"]}.isdisjoint({3, 4})


def test_assign_spins_atomi_alias(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    write_o_u_poscar(poscar)
    out = tmp_path / "alias"

    atomi_main(
        [
            "vasp-assign-spins",
            "--poscar",
            str(poscar),
            "--outdir",
            str(out),
            "--cation-elements",
            "U",
            "--anion-elements",
            "O",
            "--moment",
            "U=2,O=0",
            "--special-moment",
            "U:2=1",
        ]
    )

    assert (out / "POSCAR").is_file()
    assert (out / "INCAR").is_file()


def test_assign_spins_rejects_out_of_range_element_indices(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    write_o_u_poscar(poscar)

    with pytest.raises(ValueError, match="exceeds available U atom count"):
        spin_assign_main(
            [
                "--poscar",
                str(poscar),
                "--outdir",
                str(tmp_path / "bad"),
                "--cation-elements",
                "U",
                "--moment",
                "U=2,O=0",
                "--special-moment",
                "U:7=1",
            ]
        )
