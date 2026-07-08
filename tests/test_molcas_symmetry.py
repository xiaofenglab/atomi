from __future__ import annotations

from atomi.qchem.molcas_symmetry import (
    classify_operations,
    closure,
    detect_molcas_sign_flip_symmetry,
    generator_set_for_operations,
    molcas_group_keyword,
    operation_product,
)


def test_molcas_operation_products_follow_sign_flip_algebra() -> None:
    assert operation_product("X", "Y") == "XY"
    assert operation_product("XY", "Z") == "XYZ"
    assert operation_product("XYZ", "XYZ") == "E"


def test_generator_closure_and_group_classification() -> None:
    assert closure(["X", "Y", "Z"]) == ("E", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ")
    assert classify_operations(closure(["X", "Y", "Z"])) == "D2h"
    assert classify_operations(closure(["X", "Y"])) == "C2v"
    assert classify_operations(closure(["XY", "XZ"])) == "D2"
    assert classify_operations(closure(["XYZ"])) == "Ci"
    assert classify_operations(closure(["X", "XYZ"])) == "C2h"


def test_generator_set_for_operations_prefers_simple_molcas_input() -> None:
    operations = closure(["X", "Y", "Z"])
    generators = generator_set_for_operations(operations)
    assert generators == ("X", "Y", "Z")
    assert molcas_group_keyword(generators) == "X Y Z"
    assert molcas_group_keyword(()) == "NOSYM"


def test_detect_ceo8_cube_as_d2h_subgroup() -> None:
    a = 1.35
    symbols = ["Ce"] + ["O"] * 8
    coords = [[0.0, 0.0, 0.0]]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                coords.append([sx * a, sy * a, sz * a])

    result = detect_molcas_sign_flip_symmetry(symbols, coords, center="origin", tolerance=1.0e-6)

    assert result["group"] == "D2h"
    assert result["generators"] == ["X", "Y", "Z"]
    assert result["molcas_group_keyword"] == "X Y Z"
    assert result["standard_irrep_labels"] == ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]


def test_detect_distorted_cluster_falls_back_to_c1() -> None:
    symbols = ["U", "O", "O"]
    coords = [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.0, 1.0, 0.3]]

    result = detect_molcas_sign_flip_symmetry(symbols, coords, center="origin", tolerance=1.0e-6)

    assert result["group"] == "C1"
    assert result["generators"] == []
    assert result["molcas_group_keyword"] == "NOSYM"
