import pytest

from atomi.thermo_db import jaea_anchor, parse_jaea_table


JAEA_UO2_SAMPLE = """
<html><body>
298.15000 63.73998-1.08461E6 77.41756-1.10769E6
300.00000 64.01767-1.08449E6 77.81270-1.10784E6
400.00000 73.44526-1.07754E6 97.73332-1.11664E6
</body></html>
"""


def test_parse_jaea_table_handles_concatenated_negative_values() -> None:
    rows = parse_jaea_table(JAEA_UO2_SAMPLE)

    assert len(rows) == 3
    assert rows[1]["T_K"] == 300.0
    assert rows[1]["H_J_mol"] == pytest.approx(-1.08449e6)
    assert rows[1]["G_J_mol"] == pytest.approx(-1.10784e6)


def test_jaea_anchor_interpolates_and_records_source() -> None:
    anchor = jaea_anchor("UO2", 350.0, fetcher=lambda _url: JAEA_UO2_SAMPLE)

    assert anchor["database"] == "jaea"
    assert anchor["formula"] == "UO2"
    assert anchor["temperature_value_K"] == 350.0
    assert anchor["H_J_mol_formula"] == pytest.approx((-1.08449e6 - 1.07754e6) / 2.0)
    assert anchor["url"].endswith("/UO2.html")
