from pathlib import Path

from atomi.core import doctor


def test_build_report_can_include_hpc_probe(monkeypatch, tmp_path: Path) -> None:
    def fake_which(name: str) -> str | None:
        if name in {"python3", "git"}:
            return f"/usr/bin/{name}"
        return None

    def fake_shell_probe(command: str, timeout: int = 20) -> dict[str, object]:
        return {"command": command, "returncode": 0, "output": f"ran: {command}"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(doctor.shutil, "which", fake_which)
    monkeypatch.setattr(doctor, "_run_shell_probe", fake_shell_probe)

    report = doctor.build_report(include_hpc_probe=True)

    probe = report["hpc_probe"]
    assert probe["pwd"] == str(tmp_path)
    assert probe["which"]["python3"] == "/usr/bin/python3"
    assert probe["which"]["git"] == "/usr/bin/git"
    assert probe["which"]["sbatch"] is None
    assert probe["commands"]["module_avail_gcc_head60"]["output"].startswith("ran: module avail gcc")
    assert probe["commands"]["home_scratch_df"]["command"] == 'df -h "$HOME" "${SCRATCH:-$HOME}" 2>/dev/null'
