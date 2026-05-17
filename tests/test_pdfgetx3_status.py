from pathlib import Path

from atomi.lammps.pdfgetx3_status import inspect_pdfgetx3_environment


def test_pdfgetx3_status_accepts_explicit_executable(tmp_path: Path) -> None:
    exe = tmp_path / "pdfgetx3"
    exe.write_text("#!/bin/sh\necho PDFGetX3 2.4.0\n", encoding="utf-8")
    exe.chmod(0o755)

    report = inspect_pdfgetx3_environment(executable=str(exe))

    assert report["module"] == "pdfgetx3"
    assert report["ready_for_pdfgetx3"] is True
    assert report["pdfgetx3_mode"] == "executable"
    assert report["executable"]["resolved"] == str(exe)
    assert "PDFGetX3 2.4.0" in report["version_probe"]["output"]


def test_pdfgetx3_status_reads_configured_environment(monkeypatch, tmp_path: Path) -> None:
    env = tmp_path / "pdfgetx3_env"
    bin_dir = env / "bin"
    bin_dir.mkdir(parents=True)
    exe = bin_dir / "pdfgetx3"
    exe.write_text("#!/bin/sh\necho PDFGetX3 configured\n", encoding="utf-8")
    exe.chmod(0o755)

    monkeypatch.setenv("ATOMI_PDFGETX3_ENV", str(env))

    report = inspect_pdfgetx3_environment()

    assert report["ready_for_pdfgetx3"] is True
    assert report["executable"]["resolved"] == str(exe)
    assert report["atomi_environment"]["ATOMI_PDFGETX3_ENV"] == str(env)
