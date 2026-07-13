from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomi.md import pdfxrd_manual, pdfxrd_run


def test_pdfxrd_manual_static_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        pdfxrd_manual.main(["static", "--help"])
    assert exc.value.code == 0
    assert "--structure" in capsys.readouterr().out


def test_pdfxrd_manual_md_frame_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        pdfxrd_manual.main(["md-frame", "--help"])
    assert exc.value.code == 0
    assert "--engine" in capsys.readouterr().out


def test_pdfxrd_run_forwards_to_phase_order_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from atomi.md import phase_order_guard

    captured: dict[str, list[str]] = {}

    def fake_main(argv: list[str] | None = None) -> dict[str, list[str]]:
        captured["argv"] = list(argv or [])
        return captured

    monkeypatch.setattr(phase_order_guard, "main", fake_main)
    result = pdfxrd_run.main(["bragg-frame", "--help"])
    assert result == {"argv": ["bragg-frame", "--help"]}


def test_phase_order_guard_cp2k_xyz_smoke(tmp_path: Path) -> None:
    from atomi.md import phase_order_guard

    xyz = tmp_path / "kcl-pos.xyz"
    xyz.write_text(
        "\n".join(
            [
                "2",
                "i = 0, time = 0.0",
                "K 0.0 0.0 0.0",
                "Cl 2.8 0.0 0.0",
                "2",
                "i = 1, time = 0.5",
                "K 0.1 0.0 0.0",
                "Cl 2.9 0.0 0.0",
                "2",
                "i = 2, time = 1.0",
                "K 0.2 0.0 0.0",
                "Cl 3.0 0.0 0.0",
                "2",
                "i = 3, time = 1.5",
                "K 0.3 0.0 0.0",
                "Cl 3.1 0.0 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outdir = tmp_path / "guard_cp2k"
    summary = phase_order_guard.main(
        [
            "cp2k-xyz",
            "--xyz",
            str(xyz),
            "--cell",
            "8,0,0;0,8,0;0,0,8",
            "--species-order",
            "K,Cl",
            "--early-start-frame",
            "0",
            "--early-stop-frame",
            "2",
            "--tail-start-frame",
            "2",
            "--tail-stop-frame",
            "4",
            "--rmax",
            "4.0",
            "--long-r-min",
            "0.1",
            "--xrd-two-theta-step",
            "1.0",
            "--xrd-scattering",
            "atomic-number",
            "--outdir",
            str(outdir),
        ]
    )
    assert summary is not None
    assert summary["engine"] == "cp2k-xyz"
    written = json.loads((outdir / "phase_order_guard_summary.json").read_text(encoding="utf-8"))
    assert written["pdf_outputs"]["early_total_pdf_csv"].endswith("early_total_pdf.csv")
    assert (outdir / "early_xrd" / "early_powder_xrd.csv").exists()


def test_phase_order_guard_lammps_dump_smoke(tmp_path: Path) -> None:
    from atomi.md import phase_order_guard

    dump = tmp_path / "kcl.dump"
    frames = []
    for step, offset in enumerate([0.0, 0.1, 0.2, 0.3]):
        frames.extend(
            [
                "ITEM: TIMESTEP",
                str(step),
                "ITEM: NUMBER OF ATOMS",
                "2",
                "ITEM: BOX BOUNDS pp pp pp",
                "0 8",
                "0 8",
                "0 8",
                "ITEM: ATOMS id type x y z",
                f"1 1 {offset:.3f} 0.0 0.0",
                f"2 2 {2.8 + offset:.3f} 0.0 0.0",
            ]
        )
    dump.write_text("\n".join(frames) + "\n", encoding="utf-8")
    outdir = tmp_path / "guard_lammps"
    summary = phase_order_guard.main(
        [
            "lammps-dump",
            "--dump",
            str(dump),
            "--type-elements",
            "1=K",
            "--type-elements",
            "2=Cl",
            "--species-order",
            "K,Cl",
            "--early-start-frame",
            "0",
            "--early-stop-frame",
            "2",
            "--tail-start-frame",
            "2",
            "--tail-stop-frame",
            "4",
            "--rmax",
            "4.0",
            "--long-r-min",
            "0.1",
            "--xrd-two-theta-step",
            "1.0",
            "--xrd-scattering",
            "atomic-number",
            "--outdir",
            str(outdir),
        ]
    )
    assert summary is not None
    assert summary["engine"] == "lammps-dump"
    assert (outdir / "tail_pdf" / "tail_total_pdf.csv").exists()
    assert (outdir / "simulated_powder_xrd_early_tail_overlay.png").exists() or (
        outdir / "simulated_powder_xrd_early_tail_overlay.svg"
    ).exists()


def test_phase_order_guard_keeps_high_t_shifted_bragg_as_warning() -> None:
    from atomi.md.phase_order_guard import bragg_reference_workflow_label

    label, notes = bragg_reference_workflow_label(
        "bragg-order-lost",
        "long-range-order-retained-or-uncertain",
    )

    assert label == "long-range-order-retained-or-uncertain"
    assert any("high-temperature solid" in note for note in notes)
