from pathlib import Path

from atomi.codes import qe_wannier


def test_probe_runtime_separates_stock_and_piotr_routes(monkeypatch, tmp_path: Path) -> None:
    qe_bin = tmp_path / "qe" / "bin"
    w90_bin = tmp_path / "w90" / "bin"
    qe_bin.mkdir(parents=True)
    w90_bin.mkdir(parents=True)
    for name in ("pw.x", "hp.x", "pmw.x", "pw2wannier90.x", "wannier2pw.x"):
        (qe_bin / name).write_text("", encoding="utf-8")
    (w90_bin / "wannier90.x").write_text("", encoding="utf-8")
    monkeypatch.setenv("ATOMI_QE_BIN", str(qe_bin))
    monkeypatch.setenv("ATOMI_WANNIER90_BIN", str(w90_bin))
    monkeypatch.setenv("ESPRESSO_VERSION", "7.5")
    monkeypatch.setenv("WANNIER90_VERSION", "3.1.0")

    result = qe_wannier.probe_runtime()

    assert result["capabilities"]["modern_hubbard_card"] is True
    assert result["capabilities"]["stock_hp_atomic_response"] is True
    assert result["capabilities"]["piotr_2022_pmw_application_layer"] is True
    assert result["capabilities"]["mlwf_hubbard_projectors"] is True
    assert result["capabilities"]["piotr_matched_response"] is False
    assert result["capabilities"]["uo2_piotr_production_ready"] is False


def test_install_plan_records_ocean_qe_limit() -> None:
    plan = qe_wannier.install_plan()
    assert plan["target"]["quantum_espresso"] == "7.5"
    assert any("QE 7.0" in item for item in plan["why_not_ocean_qe"])
    assert any("Piotr" in item for item in plan["route_gates"])


def test_write_install_script_uses_compute_node_and_checks_tools(tmp_path: Path) -> None:
    script = qe_wannier.write_install_script(
        tmp_path,
        root="$HOME/atomi_hpc/qe-wannier",
        qe_version="7.5",
        wannier_version="3.1.0",
        cpus=12,
        time_limit="04:00:00",
        module_loads=("compiler/gnu/test", "mpi/test"),
    )
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=12" in text
    assert "module load compiler/gnu/test" in text
    assert "module load mpi/test" in text
    assert "make -j \"$JOBS\" pw hp pp" in text
    assert "pmw.x" in text
    assert "wannier2pw.x" in text
    assert "activate_qe_wannier.sh" in text


def test_main_returns_success(capsys) -> None:
    assert qe_wannier.main(["install-plan", "--json"]) == 0
    assert "atomi.qe_wannier_install_plan.v1" in capsys.readouterr().out
