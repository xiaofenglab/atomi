from __future__ import annotations

import json
from pathlib import Path

from atomi.qchem import molcas_diagnostics


INPUT_TEXT = """
&RASSCF
Title
 closed-shell seed
Symmetry
 1
Spin
 1
nActEl
 0 0 0
End of input

&RASSCF
Title
 C2v valence doublet A1
Symmetry
 1
Spin
 2
nActEl
 1 0 0
Inactive
 10 8 6 8
Ras2
 2 2 1 2
ORBL
 ALL
ORBA
 FULL
End of input

&RASSCF
Title
 C2v valence triplet A1
Symmetry
 1
Spin
 2
nActEl
 7 1 2
Inactive
 8 7 5 7
Ras1
 2 1 1 1
Ras3
 2 2 1 2
SUPSYM
 2
  2 6 7
  2 11 12
Alter
 1
 1 6 9
End of input
"""


def _module(active: int, spin: float, symmetry: int, matrix: str = "") -> str:
    return f"""
--- Start Module: rasscf at Tue Aug 4 00:00:00 2026 ---
      Number of electrons in active shells       {active}
      Spin quantum number                      {spin}
      State symmetry                             {symmetry}
{matrix}
--- Module rasscf spent 1 seconds ---
"""


MATRIX = """
      Pseudonatural active orbitals and approximate occupation numbers

      Molecular orbitals for symmetry species 1: a1

      Orbital                 1         2         3         4         5         6         7         8         9        10
      Energy             -2.0000   -1.8000   -1.6000   -1.4000   -1.2000   -1.0000   -0.8000   -0.6000   -0.4000   -0.2000
      Occ. No.            2.0000    2.0000    2.0000    2.0000    2.0000    2.0000    2.0000    2.0000    1.0000    0.5000
     1 U1     5f0        0.1000    0.2000    0.3000    0.4000    0.5000    0.6000    0.7000    0.8000    0.9000    1.0000
     2 O2     2px        0.9000    0.8000    0.7000    0.6000    0.5000    0.4000    0.3000    0.2000    0.1000    0.0500

      Orbital                11        12        13        14        15        16
      Energy              0.1000    0.3000    0.5000    0.7000    0.9000    1.1000
      Occ. No.            0.0000    0.0000    0.0000    0.0000    0.0000    0.0000
     1 U1     7s         1.0000    0.8000    0.6000    0.4000    0.2000    0.1000
     2 O2     2px        0.1000    0.2000    0.4000    0.6000    0.8000    1.0000

      Molecular orbitals for symmetry species 2: b1
"""


COMPACT_MATRIX = """
      Pseudonatural active orbitals and approximate occupation numbers

      MOLECULAR ORBITALS FOR SYMMETRY SPECIES 1: a1

      INDEX  ENERGY  OCCUPATION COEFFICIENTS ...
          1-4276.6386    2.0000
                 1 U1     1s     ( 1.0000)
          2   -2.0000    2.0000
                 2 U1     5f0    (-0.8000)   3 O2     2px    ( 0.2000)
          3   -1.8000    2.0000
                 2 U1     5f0    (-0.7000)
          4   -1.6000    2.0000
                 2 U1     5f0    (-0.6000)
          5   -1.4000    2.0000
                 2 U1     5f0    (-0.5000)
          6   -1.2000    2.0000
                 2 U1     5f0    (-0.4000)
          7   -1.0000    2.0000
                 2 U1     5f0    (-0.3000)
          8   -0.8000    2.0000
                 2 U1     5f0    (-0.2000)
          9   -0.6000    1.0000
                 2 U1     5f0    (-0.9000)
         10   -0.4000    0.5000
                 2 U1     5f0    (-0.8500)
         11    0.1000    0.0000
                 4 U1     7s     ( 0.9000)
         12    0.3000    0.0000
                 4 U1     7s     ( 0.8000)
         13    0.5000    0.0000
                 4 U1     7s     ( 0.7000)
         14    0.7000    0.0000
                 4 U1     7s     ( 0.6000)
         15    0.9000    0.0000
                 4 U1     7s     ( 0.5000)
         16    1.1000    0.0000
                 4 U1     7s     ( 0.4000)

      MOLECULAR ORBITALS FOR SYMMETRY SPECIES 2: b1
"""


LOG_TEXT = "".join(
    [
        _module(0, 0.0, 1),
        _module(1, 0.5, 1, MATRIX),
        _module(7, 0.5, 1),
    ]
)


def test_selects_last_symmetry_one_block_before_setup() -> None:
    blocks = molcas_diagnostics.parse_rasscf_input_blocks(INPUT_TEXT)
    reference, setup = molcas_diagnostics.select_reference_block(blocks)

    assert len(blocks) == 3
    assert reference.index == 2
    assert reference.title == "C2v valence doublet A1"
    assert setup is not None
    assert setup.index == 3
    assert setup.has_alter is True
    assert setup.has_supsym is True


def test_build_diagnostic_reports_six_frontier_orbitals_each(tmp_path: Path) -> None:
    inp = tmp_path / "run.inp"
    log = tmp_path / "run.log"
    inp.write_text(INPUT_TEXT, encoding="utf-8")
    log.write_text(LOG_TEXT, encoding="utf-8")

    result = molcas_diagnostics.build_diagnostic(inp, log)

    assert result["reference_input_block"]["index"] == 2
    assert result["matched_output_module"]["index"] == 2
    assert result["frontier"]["homo"] == 10
    assert result["frontier"]["lumo"] == 11
    assert len(result["frontier"]["occupied"]) == 6
    assert len(result["frontier"]["virtual"]) == 6
    assert result["frontier"]["occupied"][-1]["frontier_label"] == "HOMO"
    assert result["frontier"]["virtual"][0]["frontier_label"] == "LUMO"
    assert result["frontier"]["virtual"][0]["dominant_aos"][0]["ao"] == "7s"
    assert any("title says triplet" in warning for warning in result["warnings"])


def test_build_diagnostic_accepts_compact_orbital_listing(tmp_path: Path) -> None:
    inp = tmp_path / "run.inp"
    log = tmp_path / "run.log"
    inp.write_text(INPUT_TEXT, encoding="utf-8")
    log.write_text(
        "".join([_module(0, 0.0, 1), _module(1, 0.5, 1, COMPACT_MATRIX)]),
        encoding="utf-8",
    )

    result = molcas_diagnostics.build_diagnostic(inp, log)

    assert result["ao_listing"] == {
        "format": "compact",
        "complete_coefficient_matrix": False,
    }
    assert result["frontier"]["homo"] == 10
    assert result["frontier"]["lumo"] == 11
    assert result["frontier"]["occupied"][-1]["dominant_aos"][0]["ao"] == "5f0"
    assert result["frontier"]["virtual"][0]["dominant_aos"][0]["ao"] == "7s"
    assert any("compact AO printing" in warning for warning in result["warnings"])


def test_moccheck_cli_prints_and_writes_json(tmp_path: Path, capsys) -> None:
    inp = tmp_path / "run.inp"
    log = tmp_path / "run.out"
    json_out = tmp_path / "moccheck.json"
    inp.write_text(INPUT_TEXT, encoding="utf-8")
    log.write_text(LOG_TEXT, encoding="utf-8")

    rc = molcas_diagnostics.main(
        [str(inp), str(log), "--top-aos", "2", "--json-out", str(json_out)]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "MOCHECK - OpenMolcas pre-ALTER/SUPSYM frontier audit" in output
    assert "HOMO=10, LUMO=11" in output
    assert "rel|c|^2" in output
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema"] == "atomi.molcas_moccheck.v1"
