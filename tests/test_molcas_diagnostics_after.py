from __future__ import annotations

import json
from pathlib import Path

from atomi.qchem import molcas_diagnostics
from atomi.qchem import molcas_diagnostics_after


def _row(
    mo: int,
    *,
    uranium: float,
    oxygen: float,
    occupation: float = 0.5,
) -> dict[str, object]:
    return {
        "mo": mo,
        "energy": float(mo),
        "occupation": occupation,
        "symmetry": 3,
        "symmetry_label": "a2",
        "terms": [
            {
                "ao_index": 1,
                "atom": "U1",
                "ao": "5f2-",
                "coefficient": uranium,
                "coeff2": uranium * uranium,
            },
            {
                "ao_index": 2,
                "atom": "O2",
                "ao": "2p",
                "coefficient": oxygen,
                "coeff2": oxygen * oxygen,
            },
        ],
    }


TARGET = {
    "symmetry": 3,
    "symmetry_label": "a2",
    "final_ras3_slots": [15],
    "components": [
        {
            "ao": "5f2-",
            "source_mo": 12,
            "baseline_score": 0.90,
            "baseline_occupation": 2.0,
        }
    ],
    "baseline_ras3_residency_status": "already_resident",
    "baseline_ras3_residency_note": "Target was resident before ALTER.",
}


def test_after_audit_preserves_physical_metal_ligand_mixing() -> None:
    result = molcas_diagnostics_after.analyze_ras3_target(
        [_row(15, uranium=0.70, oxygen=0.55)],
        TARGET,
        atom="U1",
        shells=["5f"],
        listing_format="full",
    )

    assert result["status"] == "mixed_retained"
    assert result["component_assignment"][0]["final_slot"] == 15
    assert result["ligand_share"] > 0.30
    assert result["outside_target_candidates"] == []
    assert result["baseline_ras3_residency_status"] == "already_resident"
    assert "do not add SUPSYM" in result["recommendation"]


def test_after_audit_does_not_call_locked_mixing_unconstrained() -> None:
    result = molcas_diagnostics_after.analyze_ras3_target(
        [_row(15, uranium=0.70, oxygen=0.55)],
        TARGET,
        atom="U1",
        shells=["5f"],
        listing_format="full",
        baseline_listing_format="compact",
        supsym_context={
            "level": "fully_constrained",
            "identity_basis": "final RAS3 slots",
            "identities": [15],
            "groups": [[15]],
        },
    )

    assert result["status"] == "mixed_retained"
    assert result["baseline_capture_comparable"] is False
    assert "already SUPSYM-constrained" in result["recommendation"]
    assert "cannot establish" in result["recommendation"]


def test_after_audit_flags_component_outside_ras3() -> None:
    result = molcas_diagnostics_after.analyze_ras3_target(
        [
            _row(15, uranium=0.05, oxygen=0.95),
            _row(18, uranium=0.95, oxygen=0.05, occupation=0.0),
        ],
        TARGET,
        atom="U1",
        shells=["5f"],
        listing_format="full",
    )

    assert result["status"] == "drift_risk"
    assert result["outside_target_candidates"][0]["candidate_mo"] == 18
    assert result["outside_target_candidates"][0]["conditional_alter"] == [3, 18, 15]


def _compact_matrix(uranium: float, oxygen: float) -> str:
    return f"""
 Pseudonatural active orbitals and approximate occupation numbers
 Molecular orbitals for symmetry species 1: a1
 1 -1.0000 2.0000
   1 O2 2s ( 1.0000)
 2 -0.5000 2.0000
   2 O2 2p ( 1.0000)
 3  0.1000 0.5000
   3 U1 5f0 ({uranium: .4f}) 4 O2 2p ({oxygen: .4f})
"""


def _module(matrix: str = "") -> str:
    return f"""
--- Start Module: rasscf at Tue Aug 4 00:00:00 2026 ---
 Number of electrons in active shells 1
 Spin quantum number 0.5
 State symmetry 1
{matrix}
--- Module rasscf spent 1 seconds ---
"""


def test_after_command_uses_verified_handoff_and_post_setup_modules(tmp_path: Path) -> None:
    inp = tmp_path / "run.inp"
    log = tmp_path / "run.log"
    handoff_path = tmp_path / "run.moccheck-handoff.json"
    inp.write_text(
        """
&RASSCF
Title
 reference
Symmetry
 1
Spin
 2
nActEl
 1 0 1
Inactive
 2
Ras3
 1
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
 2
Ras3
 1
SUPSYM
 0
Alter
 0
End of input
&RASSCF
Title
 production
Symmetry
 1
Spin
 2
nActEl
 1 0 1
Inactive
 2
Ras3
 1
End of input
""",
        encoding="utf-8",
    )
    log.write_text(
        _module(_compact_matrix(0.95, 0.05))
        + _module(_compact_matrix(0.85, 0.25))
        + _module(_compact_matrix(0.65, 0.55)),
        encoding="utf-8",
    )
    handoff = {
        "schema": molcas_diagnostics.HANDOFF_SCHEMA,
        "input": {"path": str(inp), "sha256": molcas_diagnostics._sha256(inp)},
        "output_snapshot": {"path": str(log)},
        "reference_input_block": {"index": 1},
        "setup_input_block": {"index": 2},
        "target_atom": "U1",
        "target_shells": ["5f"],
        "targets": [
            {
                "symmetry": 1,
                "symmetry_label": "a1",
                "final_ras3_slots": [3],
                "components": [
                    {
                        "ao": "5f0",
                        "source_mo": 3,
                        "baseline_score": 0.95,
                        "baseline_occupation": 0.5,
                    }
                ],
            }
        ],
        "interpretation_guard": "test guard",
    }
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")

    result = molcas_diagnostics_after.build_after_diagnostic(
        inp, log, handoff_path=handoff_path, blocks_after=1
    )

    assert [item["input_block"]["index"] for item in result["blocks"]] == [2, 3]
    assert result["blocks"][0]["status"] == "stable_identity"
    assert result["blocks"][1]["status"] == "mixed_retained"
    assert result["blocks"][1]["output_module"]["index"] == 3
    assert "converged calculation alone" in result["transition_space_guard"]
    assert "high starting orbital energy" in result["transition_space_guard"]
    assert "transition densities together" in result["transition_space_guard"]


def test_after_command_marks_active_module_running(tmp_path: Path) -> None:
    inp = tmp_path / "run.inp"
    log = tmp_path / "run.log"
    handoff_path = tmp_path / "run.moccheck-handoff.json"
    inp.write_text(
        """
&RASSCF
Title
 reference
Symmetry
 1
Spin
 2
nActEl
 1 0 1
Inactive
 2
Ras3
 1
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
 2
Ras3
 1
Alter
 0
End of input
""",
        encoding="utf-8",
    )
    log.write_text(
        _module(_compact_matrix(0.95, 0.05))
        + """
--- Start Module: rasscf at Tue Aug 4 00:00:00 2026 ---
 Number of electrons in active shells 1
 Spin quantum number 0.5
 State symmetry 1
""",
        encoding="utf-8",
    )
    handoff = {
        "schema": molcas_diagnostics.HANDOFF_SCHEMA,
        "input": {"path": str(inp), "sha256": molcas_diagnostics._sha256(inp)},
        "output_snapshot": {"path": str(log)},
        "reference_input_block": {"index": 1},
        "setup_input_block": {"index": 2},
        "target_atom": "U1",
        "target_shells": ["5f"],
        "ao_listing_format": "compact",
        "targets": [TARGET | {"symmetry": 1, "final_ras3_slots": [3]}],
        "interpretation_guard": "test guard",
    }
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")

    result = molcas_diagnostics_after.build_after_diagnostic(
        inp, log, handoff_path=handoff_path, blocks_after=0
    )

    assert result["blocks"][0]["status"] == "running_incomplete"


def test_after_command_marks_module_failure_before_signature(tmp_path: Path) -> None:
    inp = tmp_path / "run.inp"
    log = tmp_path / "run.log"
    handoff_path = tmp_path / "run.moccheck-handoff.json"
    inp.write_text(
        """
&RASSCF
Title
 reference
Symmetry
 1
Spin
 2
nActEl
 1 0 1
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
Ras3
 1
Alter
 0
End of input
""",
        encoding="utf-8",
    )
    log.write_text(
        _module(_compact_matrix(0.95, 0.05))
        + """
--- Start Module: rasscf at Tue Aug 4 00:00:00 2026 ---
--- Stop Module: rasscf at Tue Aug 4 00:00:01 2026 /rc=2 ---
""",
        encoding="utf-8",
    )
    handoff = {
        "schema": molcas_diagnostics.HANDOFF_SCHEMA,
        "input": {"path": str(inp), "sha256": molcas_diagnostics._sha256(inp)},
        "output_snapshot": {"path": str(log)},
        "reference_input_block": {"index": 1},
        "setup_input_block": {"index": 2},
        "target_atom": "U1",
        "target_shells": ["5f"],
        "ao_listing_format": "compact",
        "targets": [TARGET | {"symmetry": 1, "final_ras3_slots": [1]}],
        "interpretation_guard": "test guard",
    }
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")

    result = molcas_diagnostics_after.build_after_diagnostic(
        inp, log, handoff_path=handoff_path, blocks_after=0
    )

    assert result["blocks"][0]["status"] == "failed_module"
    assert "not an orbital-drift" in result["blocks"][0]["reason"]
