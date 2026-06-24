import math
from pathlib import Path

from atomi.cp2k.alkaline_box import main


def _write_nb_oh6_seed(path: Path) -> None:
    path.write_text(
        """13
Nb(OH)6 seed
Nb 0.000000 0.000000 0.000000
O  1.950000 0.000000 0.000000
H  2.910000 0.000000 0.000000
O -1.950000 0.000000 0.000000
H -2.910000 0.000000 0.000000
O 0.000000 1.950000 0.000000
H 0.000000 2.910000 0.000000
O 0.000000 -1.950000 0.000000
H 0.000000 -2.910000 0.000000
O 0.000000 0.000000 1.950000
H 0.000000 0.000000 2.910000
O 0.000000 0.000000 -1.950000
H 0.000000 0.000000 -2.910000
""",
        encoding="utf-8",
    )


def _read_xyz(path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[2:]:
        parts = line.split()
        rows.append((parts[0], tuple(float(x) for x in parts[1:4])))
    return rows


def test_alkaline_box_places_ca_outer_sphere_and_writes_nb_basis(tmp_path: Path) -> None:
    seed = tmp_path / "nb_oh6.xyz"
    out = tmp_path / "nb_oh6_box.xyz"
    _write_nb_oh6_seed(seed)

    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        main(
            [
                str(seed),
                "--project",
                "nb_oh6_test",
                "--output",
                str(out),
                "--waters",
                "8",
                "--box",
                "14",
                "--ca",
                "1",
                "--oh",
                "1",
                "--basis-file",
                "BASIS_MOLOPT",
                "--basis-file-extra",
                "BASIS_MOLOPT_UZH",
                "--metal-index",
                "1",
            ]
        )
    finally:
        os.chdir(old_cwd)

    atoms = _read_xyz(out)
    symbols = [symbol for symbol, _ in atoms]
    assert symbols[:13] == ["Nb", "O", "H", "O", "H", "O", "H", "O", "H", "O", "H", "O", "H"]
    assert symbols[13] == "Ca"
    nb = atoms[0][1]
    ca = atoms[13][1]
    assert math.dist(nb, ca) > 5.0

    geoopt = (tmp_path / "nb_oh6_test_geoopt.inp").read_text(encoding="utf-8")
    assert "BASIS_SET_FILE_NAME BASIS_MOLOPT\n    BASIS_SET_FILE_NAME BASIS_MOLOPT_UZH" in geoopt
    assert "&KIND Nb" in geoopt
    assert "BASIS_SET DZVP-MOLOPT-PBE-GTH-q13" in geoopt
