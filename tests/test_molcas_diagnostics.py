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

      Orbital                11        12        13        14        15        16        17        18        19        20
      Energy              0.1000    0.3000    0.5000    0.7000    0.9000    1.1000    1.3000    1.5000    1.7000    1.9000
      Occ. No.            0.0000    0.0000    0.0000    0.0000    0.0000    0.0000    0.0000    0.0000    0.0000    0.0000
     1 U1     7s         1.0000    0.8000    0.6000    0.4000    0.2000    0.1000    0.0800    0.0600    0.0400    0.0200
     2 O2     2px        0.1000    0.2000    0.4000    0.6000    0.8000    1.0000    1.1000    1.2000    1.3000    1.4000

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


def _reference_input(symmetry: int, label: str) -> str:
    return f"""
&RASSCF
Title
 C2v valence doublet {label}
Symmetry
 {symmetry}
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
"""


def _matrix_for_symmetry(symmetry: int, label: str) -> str:
    return MATRIX.replace(
        "Molecular orbitals for symmetry species 2: b1",
        "Molecular orbitals for symmetry species 99: end",
    ).replace(
        "Molecular orbitals for symmetry species 1: a1",
        f"Molecular orbitals for symmetry species {symmetry}: {label}",
    )


def _matrix_for_all_symmetries() -> str:
    section = MATRIX.split(
        "Molecular orbitals for symmetry species 1: a1", 1
    )[1].split("Molecular orbitals for symmetry species 2: b1", 1)[0]
    tables = "".join(
        f"Molecular orbitals for symmetry species {symmetry}: {label.lower()}\n{section}"
        for symmetry, label in [(1, "A1"), (2, "B1"), (3, "A2"), (4, "B2")]
    )
    return (
        "      Pseudonatural active orbitals and approximate occupation numbers\n\n"
        + tables
        + "Molecular orbitals for symmetry species 99: end\n"
    )


MULTI_INPUT_TEXT = (
    """
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
"""
    + "".join(
        _reference_input(symmetry, label)
        for symmetry, label in [(1, "A1"), (2, "B1"), (3, "A2"), (4, "B2")]
    )
    + """
&RASSCF
Title
 C2v setup
Symmetry
 1
Spin
 2
nActEl
 7 1 2
SUPSYM
 1
  2 6 7
End of input
"""
)


MULTI_LOG_TEXT = "".join(
    [_module(0, 0.0, 1)]
    + [
        _module(
            1,
            0.5,
            symmetry,
            _matrix_for_all_symmetries()
            if symmetry == 4
            else _matrix_for_symmetry(symmetry, label.lower()),
        )
        for symmetry, label in [(1, "A1"), (2, "B1"), (3, "A2"), (4, "B2")]
    ]
)


def test_selects_single_last_block_before_setup() -> None:
    blocks = molcas_diagnostics.parse_rasscf_input_blocks(INPUT_TEXT)
    reference, setup = molcas_diagnostics.select_reference_block(blocks)

    assert len(blocks) == 3
    assert reference.index == 2
    assert reference.title == "C2v valence doublet A1"
    assert setup is not None
    assert setup.index == 3
    assert setup.has_alter is True
    assert setup.has_supsym is True
    assert setup.alter_swaps == ((1, 6, 9),)
    assert setup.supsym_groups[0] == ((6, 7), (11, 12))


def test_build_diagnostic_reports_six_occupied_and_ten_virtual_orbitals(tmp_path: Path) -> None:
    inp = tmp_path / "run.inp"
    log = tmp_path / "run.log"
    inp.write_text(INPUT_TEXT, encoding="utf-8")
    log.write_text(LOG_TEXT, encoding="utf-8")

    result = molcas_diagnostics.build_diagnostic(inp, log)

    assert result["reference_input_block"]["index"] == 2
    assert result["matched_output_module"]["index"] == 2
    assert result["frontier"]["homo"] == 9
    assert result["frontier"]["lumo"] == 10
    assert len(result["frontier"]["occupied"]) == 6
    assert len(result["frontier"]["virtual"]) == 10
    assert result["frontier"]["occupied"][-1]["frontier_label"] == "HOMO"
    assert result["frontier"]["virtual"][0]["frontier_label"] == "LUMO"
    assert result["frontier"]["virtual"][1]["dominant_aos"][0]["ao"] == "7s"
    assert any("title says triplet" in warning for warning in result["warnings"])


def test_legacy_orbitals_override_remains_symmetric(tmp_path: Path) -> None:
    inp = tmp_path / "run.inp"
    log = tmp_path / "run.log"
    inp.write_text(INPUT_TEXT, encoding="utf-8")
    log.write_text(LOG_TEXT, encoding="utf-8")

    result = molcas_diagnostics.build_diagnostic(inp, log, orbitals_each=4)

    assert len(result["frontier"]["occupied"]) == 4
    assert len(result["frontier"]["virtual"]) == 4


def test_ras3_audit_proposes_minimal_virtual_homing_and_supsym() -> None:
    text = """
* ATOMI RAS3_TARGET U1:5f
&RASSCF
Title
 reference
Symmetry
 1
Spin
 2
nActEl
 1 0 0
End of input
&RASSCF
Title
 setup
Symmetry
 1
Spin
 2
nActEl
 1 0 2
Inactive
 10
Ras3
 2
SUPSYM
 0
Alter
 0
End of input
"""
    blocks = molcas_diagnostics.parse_rasscf_input_blocks(text)
    rows = [
        {
            "mo": mo,
            "energy": float(mo) / 10.0,
            "occupation": 0.0,
            "symmetry": 1,
            "symmetry_label": "a1",
            "terms": [
                {
                    "ao_index": 1,
                    "atom": "U1" if mo in {15, 16} else "O2",
                    "ao": (
                        "5f0" if mo == 15 else "5f2+" if mo == 16 else "2px"
                    ),
                    "coefficient": 0.9,
                    "coeff2": 0.81,
                }
            ],
        }
        for mo in range(1, 21)
    ]

    result = molcas_diagnostics.build_ras3_recommendation(
        text,
        Path("trial_5fonly.inp"),
        setup=blocks[1],
        rows_by_symmetry={1: rows},
        ao_listing_format="full",
        occupancy_threshold=0.999,
        mixing_window_ha=0.15,
    )

    assert result["status"] == "proposal_available"
    assert result["target_atom"] == "U1"
    assert result["safe_virtual_alter_additions"] == [[1, 15, 11], [1, 16, 12]]
    assert result["conditional_occupied_alter_additions"] == []
    assert result["supsym"]["pre_alter_source_identity_groups"] == [[15, 16]]
    assert result["supsym"]["production_final_ras3_slot_groups"] == [[11, 12]]
    assert result["supsym"]["alter_only_probe_block_preserving_existing_groups"] == [
        "SUPSYM",
        " 0",
    ]
    assert result["supsym"]["suggested_full_block_preserving_existing_groups"] == [
        "SUPSYM",
        " 1",
        "  2 15 16",
    ]
    assert "pre-ALTER source orbital identities" in result["supsym"][
        "input_index_semantics"
    ]
    assert "not required for homing" in result["supsym"]["note"]


def test_ras3_audit_does_not_call_occupied_source_a_safe_homing_swap() -> None:
    text = """
* ATOMI RAS3_TARGET U1:5f
&RASSCF
Title
 reference
Symmetry
 1
Spin
 2
nActEl
 1 0 0
End of input
&RASSCF
Title
 setup
Symmetry
 1
Spin
 2
nActEl
 1 0 1
Inactive
 13
Ras1
 1
Ras3
 1
Alter
 0
End of input
"""
    blocks = molcas_diagnostics.parse_rasscf_input_blocks(text)
    rows = [
        {
            "mo": mo,
            "energy": float(mo) / 10.0,
            "occupation": 2.0 if mo <= 14 else 0.0,
            "symmetry": 1,
            "symmetry_label": "a2",
            "terms": [
                {
                    "ao_index": 1,
                    "atom": "U1" if mo == 12 else "O2",
                    "ao": "5f2-" if mo == 12 else "2px",
                    "coefficient": 0.9,
                    "coeff2": 0.81,
                }
            ],
        }
        for mo in range(1, 21)
    ]

    result = molcas_diagnostics.build_ras3_recommendation(
        text,
        Path("trial_5fonly.inp"),
        setup=blocks[1],
        rows_by_symmetry={1: rows},
        ao_listing_format="full",
        occupancy_threshold=0.999,
        mixing_window_ha=0.15,
    )

    assert result["status"] == "review_required"
    assert result["safe_virtual_alter_additions"] == []
    assert result["conditional_occupied_alter_additions"] == [[1, 12, 15]]
    source = result["per_symmetry"][0]["selected_sources"][0]
    assert source["occupied_partition_change"] is True


def test_ras3_audit_identifies_existing_identity_lock() -> None:
    text = """
* ATOMI RAS3_TARGET U1:5f U1:7s
&RASSCF
Title
 reference
Symmetry
 1
Spin
 2
nActEl
 1 0 0
End of input
&RASSCF
Title
 setup
Symmetry
 1
Spin
 2
nActEl
 1 0 2
Inactive
 10
Ras3
 2
SUPSYM
 2
  1 6
  2 15 16
Alter
 0
End of input
"""
    blocks = molcas_diagnostics.parse_rasscf_input_blocks(text)
    rows = [
        {
            "mo": mo,
            "energy": float(mo) / 10.0,
            "occupation": 0.0,
            "symmetry": 1,
            "symmetry_label": "a1",
            "terms": [
                {
                    "ao_index": 1,
                    "atom": "U1" if mo in {15, 16} else "O2",
                    "ao": "5f0" if mo == 15 else "7s" if mo == 16 else "2px",
                    "coefficient": 0.9,
                    "coeff2": 0.81,
                }
            ],
        }
        for mo in range(1, 21)
    ]

    result = molcas_diagnostics.build_ras3_recommendation(
        text,
        Path("trial_5f7s.inp"),
        setup=blocks[1],
        rows_by_symmetry={1: rows},
        ao_listing_format="full",
        occupancy_threshold=0.999,
        mixing_window_ha=0.15,
    )

    assert result["supsym"]["ras3_constraint_levels"] == ["fully_constrained"]
    assert result["supsym"]["existing_ras3_identity_groups"] == [[[15, 16]]]
    assert result["supsym"]["mixing_preserving_candidate_block"] == [
        "SUPSYM",
        " 1",
        "  1 6",
    ]


def test_ras3_audit_rejects_more_than_openmolcas_alter_limit() -> None:
    swaps = "\n".join(f" 1 {index} {index + 1}" for index in range(1, 18))
    text = f"""
* ATOMI RAS3_TARGET U1:5f
&RASSCF
Title
 reference
Symmetry
 1
Spin
 2
nActEl
 1 0 0
End of input
&RASSCF
Title
 setup
Symmetry
 1
Spin
 2
nActEl
 1 0 1
Inactive
 10
Ras3
 1
Alter
 17
{swaps}
End of input
"""
    blocks = molcas_diagnostics.parse_rasscf_input_blocks(text)
    rows = [
        {
            "mo": mo,
            "energy": float(mo),
            "occupation": 0.0,
            "symmetry": 1,
            "symmetry_label": "a1",
            "terms": [
                {
                    "ao_index": 1,
                    "atom": "U1" if mo == 12 else "O2",
                    "ao": "5f0" if mo == 12 else "2px",
                    "coefficient": 0.9,
                    "coeff2": 0.81,
                }
            ],
        }
        for mo in range(1, 20)
    ]

    result = molcas_diagnostics.build_ras3_recommendation(
        text,
        Path("too_many_alter.inp"),
        setup=blocks[1],
        rows_by_symmetry={1: rows},
        ao_listing_format="full",
        occupancy_threshold=0.999,
        mixing_window_ha=0.15,
    )

    assert result["status"] == "invalid_alter_input"
    assert result["alter_pair_count"] == 17
    assert result["alter_pair_limit"] == 16
    assert result["alter_limit_exceeded"] is True


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
    assert result["frontier"]["homo"] == 9
    assert result["frontier"]["lumo"] == 10
    assert result["frontier"]["occupied"][-1]["dominant_aos"][0]["ao"] == "5f0"
    assert result["frontier"]["virtual"][1]["dominant_aos"][0]["ao"] == "7s"
    assert any("compact AO printing" in warning for warning in result["warnings"])


def test_build_diagnostics_defaults_to_all_pre_setup_symmetries(tmp_path: Path) -> None:
    inp = tmp_path / "run.inp"
    log = tmp_path / "run.log"
    inp.write_text(MULTI_INPUT_TEXT, encoding="utf-8")
    log.write_text(MULTI_LOG_TEXT, encoding="utf-8")

    results = molcas_diagnostics.build_diagnostics(inp, log)

    assert [result["frontier"]["symmetry"] for result in results] == [1, 2, 3, 4]
    assert [result["reference_input_block"]["index"] for result in results] == [5, 5, 5, 5]
    assert [result["matched_output_module"]["index"] for result in results] == [5, 5, 5, 5]
    assert [result["frontier"]["symmetry_label"] for result in results] == [
        "a1",
        "b1",
        "a2",
        "b2",
    ]


def test_build_diagnostics_accepts_selected_symmetries(tmp_path: Path) -> None:
    inp = tmp_path / "run.inp"
    log = tmp_path / "run.log"
    inp.write_text(MULTI_INPUT_TEXT, encoding="utf-8")
    log.write_text(MULTI_LOG_TEXT, encoding="utf-8")

    results = molcas_diagnostics.build_diagnostics(inp, log, symmetries=[4, 2])

    assert [result["frontier"]["symmetry"] for result in results] == [4, 2]


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
    assert "HOMO=9, LUMO=10" in output
    assert "LUMO+9" in output
    assert "rel|c|^2" in output
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema"] == "atomi.molcas_moccheck.v1"


def test_moccheck_cli_prints_all_symmetries_and_writes_collection_json(
    tmp_path: Path, capsys
) -> None:
    inp = tmp_path / "run.inp"
    log = tmp_path / "run.out"
    json_out = tmp_path / "moccheck_all.json"
    inp.write_text(MULTI_INPUT_TEXT, encoding="utf-8")
    log.write_text(MULTI_LOG_TEXT, encoding="utf-8")

    rc = molcas_diagnostics.main([str(inp), str(log), "--json-out", str(json_out)])

    assert rc == 0
    output = capsys.readouterr().out
    assert "MOCHECK - OpenMolcas all-symmetry" in output
    assert "Symmetries: 1 (a1), 2 (b1), 3 (a2), 4 (b2)" in output
    assert output.count("SYMMETRY ") == 4
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema"] == "atomi.molcas_moccheck_collection.v1"
    assert payload["symmetries"] == [1, 2, 3, 4]
    assert len(payload["diagnostics"]) == 4
