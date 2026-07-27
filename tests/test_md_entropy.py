from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from atomi.md import entropy


@pytest.mark.parametrize("engine", ["cp2k", "lammps"])
def test_prepare_writes_engine_specific_plan(tmp_path: Path, engine: str) -> None:
    outdir = tmp_path / engine
    result = entropy.main(
        [
            "prepare",
            "--engine",
            engine,
            "--phase",
            "liquid",
            "--temperature-k",
            "1200",
            "--system",
            "KCl",
            "--formula",
            "KCl",
            "--outdir",
            str(outdir),
        ]
    )

    assert result is not None
    assert result["schema"] == entropy.SCHEMA_MD_ENTROPY_PLAN
    assert result["guard_policy"]["primary_phase_guards"] == [
        "coordination_distribution",
        "msd_or_diffusion",
    ]
    assert "npt_density_anchor" in [stage["stage"] for stage in result["stages"]]
    assert "fixed_cell_nvt_production" in [stage["stage"] for stage in result["stages"]]
    assert engine in result["engine_bridge"]["trajectory"].lower()
    assert (outdir / "md_entropy_stage_runlist.csv").is_file()


def _gate_args(outdir: Path, *, phase: str = "liquid") -> list[str]:
    return [
        "gate",
        "--engine",
        "lammps",
        "--phase",
        phase,
        "--temperature-k",
        "1400",
        "--quality",
        "production",
        "--outdir",
        str(outdir),
        "--numerical-completion",
        "pass",
        "--trajectory-continuity",
        "pass",
        "--temperature-stability",
        "pass",
        "--energy-stability",
        "pass",
        "--density-or-volume-anchor",
        "pass",
        "--tail-stationarity",
        "pass",
        "--coordination-distribution",
        "pass",
        "--msd-or-diffusion",
        "pass",
        "--xrd-or-bragg",
        "warning",
        "--rdf-or-pdf",
        "pass",
    ]


def test_gate_uses_cn_msd_as_primary_and_caps_single_tail(tmp_path: Path) -> None:
    result = entropy.main(_gate_args(tmp_path / "gate"))

    assert result is not None
    assert result["decision"] == "accepted-with-warning"
    assert result["promotable_to_entropy"] is True
    assert result["effective_quality"] == "screening-prior"
    assert any("independent" in warning for warning in result["warnings"])


def test_gate_rejects_failed_primary_guard(tmp_path: Path) -> None:
    args = _gate_args(tmp_path / "gate")
    index = args.index("--msd-or-diffusion")
    args[index + 1] = "fail"

    result = entropy.main(args)

    assert result is not None
    assert result["decision"] == "rejected"
    assert result["promotable_to_entropy"] is False


def _coordination_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["phase", "T_K", "central", "neighbor", "cn", "probability"],
        )
        writer.writeheader()
        for element in ("K", "Cl"):
            for cn, probability in ((8, 0.1), (9, 0.2), (10, 0.4), (11, 0.2), (12, 0.1)):
                writer.writerow(
                    {
                        "phase": "liquid",
                        "T_K": 1400,
                        "central": element,
                        "neighbor": element,
                        "cn": cn,
                        "probability": probability,
                    }
                )


def test_entropy_requires_gate_and_writes_decomposition(tmp_path: Path) -> None:
    gate_dir = tmp_path / "gate"
    entropy.main(_gate_args(gate_dir))
    coord = tmp_path / "coordination.csv"
    _coordination_csv(coord)
    outdir = tmp_path / "result"

    result = entropy.main(
        [
            "entropy",
            "--gate-json",
            str(gate_dir / "md_entropy_gate.json"),
            "--phase",
            "liquid",
            "--formula",
            "KCl",
            "--coordination-csv",
            str(coord),
            "--svib-j-mol-atom-k",
            "90",
            "--outdir",
            str(outdir),
        ]
    )

    assert result is not None
    assert result["decomposition"]["S_vib_J_mol_atom_K"] == pytest.approx(90.0)
    assert result["decomposition"]["S_conf_J_mol_atom_K"] > 5.0
    assert result["decomposition"]["S_total_J_mol_atom_K"] > 95.0
    written = json.loads((outdir / "md_entropy_result.json").read_text(encoding="utf-8"))
    assert written["quality"] == "screening-prior"


def test_solid_entropy_records_numeric_zero_sconf(tmp_path: Path) -> None:
    gate_dir = tmp_path / "solid_gate"
    entropy.main(_gate_args(gate_dir, phase="solid"))
    outdir = tmp_path / "solid_result"

    result = entropy.main(
        [
            "entropy",
            "--gate-json",
            str(gate_dir / "md_entropy_gate.json"),
            "--phase",
            "solid",
            "--formula",
            "KCl",
            "--svib-j-mol-atom-k",
            "75",
            "--outdir",
            str(outdir),
        ]
    )

    assert result is not None
    assert result["decomposition"]["S_conf_J_mol_atom_K"] == 0.0
    assert result["solid_sconf_policy"] == "zero"


def test_entropy_blocks_rejected_gate(tmp_path: Path) -> None:
    gate_dir = tmp_path / "gate"
    args = _gate_args(gate_dir)
    index = args.index("--trajectory-continuity")
    args[index + 1] = "fail"
    entropy.main(args)

    with pytest.raises(ValueError, match="rejected"):
        entropy.main(
            [
                "entropy",
                "--gate-json",
                str(gate_dir / "md_entropy_gate.json"),
                "--phase",
                "liquid",
                "--svib-j-mol-atom-k",
                "90",
            ]
        )
