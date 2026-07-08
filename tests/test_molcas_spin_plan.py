from __future__ import annotations

from atomi.cli.registry import command_registry, registered_aliases
from atomi.qchem.molcas_spin_plan import (
    ce4_l3_d2h_plan,
    ce4_l23_d2h_plan,
    core_hole_f2_to_f3_root_counts,
    d2h_product_counts,
    f2_root_counts,
    f3_root_counts,
    render_ce4_l23_d2h_probe,
    render_u4_m45_c1_5f_probe,
    u4_m45_c1_5f_plan,
)


def test_f2_root_counts_for_u4_5f2() -> None:
    assert f2_root_counts(7) == {"triplet": 21, "singlet": 28}


def test_f3_and_core_hole_roots_for_u4_5f_only_m45() -> None:
    assert f3_root_counts(7) == {"doublet": 112, "quartet": 35}
    assert core_hole_f2_to_f3_root_counts(n_core_orbitals=5, n_acceptor_orbitals=7) == {
        "singlet": 560,
        "triplet": 735,
        "quintet": 175,
    }


def test_d2h_core_acceptor_products_for_ce_l3() -> None:
    products = d2h_product_counts(
        {"B3u": 1, "B2u": 1, "B1u": 1},
        {"Ag": 2, "B1g": 1, "B2g": 1, "B3g": 1},
    )
    assert products == {"B3u": 4, "B2u": 4, "B1u": 4, "Au": 3}


def test_ce4_l23_d2h_spin_plan_counts() -> None:
    plan = ce4_l23_d2h_plan()
    assert plan["core_shell"]["ras1"] == "0 1 1 0 1 0 0 0"
    assert plan["acceptor_shell"]["ras3"] == "2 0 0 1 0 1 1 0"
    assert plan["excited_irrep_root_counts"] == {"B3u": 4, "B2u": 4, "B1u": 4, "Au": 3}
    assert plan["element_metadata"]["symbol"] == "Ce"

    blocks = plan["blocks"]
    assert blocks[0]["label"] == "ground_singlet_Ce4_4f0"
    assert blocks[0]["spin_multiplicity"] == 1
    assert blocks[0]["roots"] == 1
    excited = [b for b in blocks if b["manifold"] == "core_excited"]
    assert len(excited) == 8
    assert sum(b["roots"] for b in excited if b["spin_multiplicity"] == 1) == 15
    assert sum(b["roots"] for b in excited if b["spin_multiplicity"] == 3) == 15


def test_ce4_l3_alias_keeps_scalar_2p_warning() -> None:
    plan = ce4_l3_d2h_plan()
    assert plan["edge"] == "L3"
    assert any("Scalar D2h RASSCF spans Ce 2p" in item for item in plan["warnings"])


def test_u4_m45_c1_5f_plan_ground_and_excited_counts() -> None:
    plan = u4_m45_c1_5f_plan()
    assert plan["element_metadata"]["symbol"] == "U"
    assert plan["nactel"] == "12 1 3"
    assert plan["ground_root_counts"] == {"triplet": 21, "singlet": 28}
    assert plan["core_excited_root_counts"] == {"singlet": 560, "triplet": 735, "quintet": 175}

    blocks = {block["label"]: block for block in plan["blocks"]}
    assert blocks["ground_triplet_U4_5f2"]["spin_multiplicity"] == 3
    assert blocks["ground_triplet_U4_5f2"]["roots"] == 21
    assert blocks["ground_singlet_U4_5f2"]["spin_multiplicity"] == 1
    assert blocks["ground_singlet_U4_5f2"]["roots"] == 28
    assert blocks["excited_spin1_U_M45_3d9_5f3"]["roots"] == 560
    assert blocks["excited_spin3_U_M45_3d9_5f3"]["roots"] == 735
    assert blocks["excited_spin5_U_M45_3d9_5f3"]["roots"] == 175


def test_u4_m45_root_cap_truncates_only_large_block() -> None:
    plan = u4_m45_c1_5f_plan(root_cap=600)
    blocks = {block["label"]: block for block in plan["blocks"]}
    assert blocks["excited_spin1_U_M45_3d9_5f3"]["roots"] == 560
    assert blocks["excited_spin3_U_M45_3d9_5f3"]["roots"] == 600
    assert blocks["excited_spin3_U_M45_3d9_5f3"]["requested_roots"] == 735
    assert blocks["excited_spin5_U_M45_3d9_5f3"]["roots"] == 175
    assert any("root-truncated" in item for item in plan["warnings"])


def test_render_ce4_l23_probe_is_rasscf_only() -> None:
    text = render_ce4_l23_d2h_probe(coord="CeO8.xyz", inactive="64 0 0 0 0 0 0 0")
    assert "Coord = CeO8.xyz" in text
    assert "Ras1 = 0 1 1 0 1 0 0 0" in text
    assert "Ras3 = 2 0 0 1 0 1 1 0" in text
    assert "CIROOTS = 4 4 1" in text
    assert "CIROOTS = 3 3 1" in text
    assert "&CASPT2" not in text
    assert "&RASSI" not in text
    assert "TBD_FROM_SCF_SCOUT" not in text


def test_render_u4_m45_probe_is_rasscf_only_and_can_cap_roots() -> None:
    text = render_u4_m45_c1_5f_probe(coord="UO8.xyz", inactive="100", root_cap=600)
    assert "Coord = UO8.xyz" in text
    assert "Group = NoSym" in text
    assert "nActEl = 12 1 3" in text
    assert "CIROOTS = 21 21 1" in text
    assert "CIROOTS = 28 28 1" in text
    assert "CIROOTS = 560 560 1" in text
    assert "CIROOTS = 600 600 1" in text
    assert "CIROOTS = 175 175 1" in text
    assert "&CASPT2" not in text
    assert "&RASSI" not in text
    assert "TBD_FROM_SCF_SCOUT" not in text


def test_molcas_spin_plan_cli_registered() -> None:
    assert "molcas-spin-plan" in registered_aliases()
    assert command_registry()["molcas-spin-plan"].target == "atomi.qchem.molcas_spin_plan:main"
    assert "molcas-symmetry-plan" in registered_aliases()
    assert command_registry()["molcas-symmetry-plan"].target == "atomi.qchem.molcas_symmetry:main"
