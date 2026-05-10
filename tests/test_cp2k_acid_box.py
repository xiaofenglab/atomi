import math

from atomi.cp2k.acid_box import (
    Restraint,
    density_to_water_count,
    parse_density,
    render_colvars,
    render_constraints,
    water_count_to_box_length,
)


def test_density_round_trip_is_close_to_regular_water() -> None:
    waters = density_to_water_count(26.0, 1.0)

    assert waters == 588
    assert math.isclose(water_count_to_box_length(waters, 1.0), 26.0, rel_tol=0.01)


def test_parse_density_accepts_presets_and_numeric_values() -> None:
    assert parse_density(None, "regular") == 1.0
    assert parse_density("loose", "regular") == 0.75
    assert parse_density("0.85", "regular") == 0.85


def test_restraint_templates_match_cp2k_colvar_shape() -> None:
    restraints = [Restraint(index=1, metal_index=1, ligand_index=3, target=2.213)]

    colvars = render_colvars(restraints)
    constraints = render_constraints(restraints, restraint_k=5.0)

    assert "ATOMS 1 3" in colvars
    assert "COLVAR 1" in constraints
    assert "TARGET [angstrom] 2.213" in constraints
    assert "K [kcalmol] 5.000" in constraints

