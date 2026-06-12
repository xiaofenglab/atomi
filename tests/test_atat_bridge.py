from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from atomi.cli.main import main as atomi_main
from atomi.atat import bridge


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_partial_gd2o3_cif(path: Path) -> None:
    path.write_text(
        "data_demo\n"
        "_symmetry_space_group_name_H-M   P1\n"
        "_cell_length_a 5\n"
        "_cell_length_b 5\n"
        "_cell_length_c 5\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "_symmetry_Int_Tables_number 1\n"
        "loop_\n"
        "_space_group_symop_operation_xyz\n"
        "x,y,z\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "Gd1 Gd 0 0 0 1\n"
        "Gd2 Gd 0.5 0.5 0.5 1\n"
        "O1 O 0.25 0.25 0.25 1\n"
        "O2 O 0.75 0.75 0.75 0.2(1)\n",
        encoding="utf-8",
    )


def write_uo2_parent_poscar(path: Path) -> None:
    path.write_text(
        "UO2 parent\n"
        "1.0\n"
        "5 0 0\n"
        "0 5 0\n"
        "0 0 5\n"
        "U O\n"
        "2 4\n"
        "Direct\n"
        "0 0 0\n"
        "0.5 0.5 0.5\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n"
        "0.25 0.75 0.25\n"
        "0.75 0.25 0.75\n",
        encoding="utf-8",
    )


def test_atat_status_detects_fake_tools(tmp_path: Path, monkeypatch) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for tool in ("mcsqs", "corrdump", "str2poscar"):
        path = fake_bin / tool
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    report = bridge.inspect_atat_environment()

    assert report["executables"]["sqs"]["mcsqs"]["available"] is True
    assert report["ready"]["can_generate_sqs"] is True
    assert report["ready"]["can_convert_structures"] is True
    assert report["ready"]["can_fit_cluster_expansion"] is False


def test_atat_bridge_init_writes_stage_and_species_maps(tmp_path: Path) -> None:
    out = tmp_path / "atat"

    atomi_main(["atat-bridge", "init", "--outdir", str(out), "--system", "(Gd,U)O2-x"])

    species = rows(out / "pseudo_species_map.csv")
    labels = {row["pseudo_species"] for row in species}
    assert {"U4", "U5", "Gd3", "O", "V_O"}.issubset(labels)
    gd = [row for row in species if row["pseudo_species"] == "Gd3"][0]
    assert gd["moment_guard"] == "Gd=7,-7@0.6"
    stages = rows(out / "atat_atomi_stage_map.csv")
    assert stages[0]["stage_id"] == "01_motif_search"
    plan = json.loads((out / "atat_bridge_plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == bridge.SCHEMA
    assert "sqs" in plan["workflow_stages"][2]["stage_id"]
    assert (out / "ATAT_ATOMI_BRIDGE_NOTES.md").exists()


def test_atat_bridge_init_uses_requested_anion_element(tmp_path: Path) -> None:
    out = tmp_path / "atat_uc2"

    atomi_main(
        [
            "atat-bridge",
            "init",
            "--outdir",
            str(out),
            "--system",
            "UC2",
            "--host",
            "U",
            "--anion-element",
            "C",
        ]
    )

    species = rows(out / "pseudo_species_map.csv")
    by_label = {row["pseudo_species"]: row for row in species}
    assert "C" in by_label
    assert "V_C" in by_label
    assert "O" not in by_label
    assert by_label["C"]["element"] == "C"
    assert by_label["C"]["sublattice"] == "anion"
    assert by_label["C"]["moment_guard"] == "C=0@0.25"
    assert by_label["V_C"]["element"] == "C"
    assert by_label["V_C"]["notes"] == "C vacancy"
    plan = json.loads((out / "atat_bridge_plan.json").read_text(encoding="utf-8"))
    assert plan["anion_element"] == "C"
    assert plan["vacancy_label"] == "V_C"


def test_atat_bridge_index_collects_candidate_files(tmp_path: Path) -> None:
    root = tmp_path / "atat_runs"
    (root / "sqs").mkdir(parents=True)
    (root / "sqs" / "bestsqs.out").write_text("structure\n", encoding="utf-8")
    (root / "enum").mkdir()
    (root / "enum" / "str0001.out").write_text("structure\n", encoding="utf-8")
    out = tmp_path / "atat_candidate_index.csv"

    bridge.index_main(["--root", str(root), "--out", str(out)])

    data = rows(out)
    assert len(data) == 2
    assert {row["source_kind"] for row in data} == {"sqs", "enumerated_structure"}
    assert all(row["atomi_next"] == "convert_to_vasp_then_vasp-branch-live" for row in data)


def test_atat_doctor_json_cli(capsys, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", os.fspath(tmp_path))

    atomi_main(["atat-doctor", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == bridge.STATUS_SCHEMA
    assert "can_generate_sqs" in payload["ready"]


def test_atat_doctor_updates_private_hpc_config(tmp_path: Path, monkeypatch, capsys) -> None:
    fake_root = tmp_path / "atat"
    fake_bin = fake_root / "src"
    fake_bin.mkdir(parents=True)
    for tool in ("mcsqs", "corrdump", "maps", "mmaps", "genstr", "emc2", "mapsrep", "checkcell", "cellcvrt"):
        path = fake_bin / tool
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    config = tmp_path / "atomi_hpc_config.kit.local.json"
    config.write_text(json.dumps({"site": "KIT", "profiles": {}, "environment_exports": {}}), encoding="utf-8")

    atomi_main(["atat-doctor", "--update-hpc-config", str(config)])

    assert "HPC config updated" in capsys.readouterr().out
    data = json.loads(config.read_text(encoding="utf-8"))
    profile = data["profiles"]["atat"]
    assert profile["root"] == str(fake_root)
    assert profile["bin"] == str(fake_bin)
    assert profile["executables"]["mcsqs"] == str(fake_bin / "mcsqs")
    assert profile["ready"]["can_generate_sqs"] is True
    assert "str2poscar" in profile["missing_executables"]
    assert data["environment_exports"]["ATOMI_ATAT_ROOT"] == str(fake_root)
    assert data["environment_exports"]["ATOMI_ATAT_BIN"] == str(fake_bin)


def test_atat_ce_handoff_filters_physics_accepted_rows(tmp_path: Path) -> None:
    summary = tmp_path / "stage1_branch_summary.csv"
    fields = [
        "branch_id",
        "run_dir",
        "output_run_dir",
        "energy_eV",
        "physics_guard_status",
        "mag_status",
        "action",
        "element_order",
        "changed_by_element",
        "frame_id",
    ]
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "branch_id": "spin_good",
                "run_dir": str(tmp_path / "runs" / "spin_good"),
                "output_run_dir": str(tmp_path / "runs" / "spin_good"),
                "energy_eV": "-10.0",
                "physics_guard_status": "OK",
                "mag_status": "OK",
                "action": "continue",
                "element_order": '{"Gd": "FM"}',
                "changed_by_element": '{"U": 1}',
                "frame_id": "frame0",
            }
        )
        writer.writerow(
            {
                "branch_id": "spin_bad",
                "run_dir": str(tmp_path / "runs" / "spin_bad"),
                "output_run_dir": str(tmp_path / "runs" / "spin_bad"),
                "energy_eV": "-11.0",
                "physics_guard_status": "FAIL",
                "mag_status": "OK",
                "action": "continue",
                "frame_id": "frame0",
            }
        )
        writer.writerow(
            {
                "branch_id": "spin_stop",
                "run_dir": str(tmp_path / "runs" / "spin_stop"),
                "output_run_dir": str(tmp_path / "runs" / "spin_stop"),
                "energy_eV": "-12.0",
                "physics_guard_status": "OK",
                "mag_status": "OK",
                "action": "stop",
                "frame_id": "frame0",
            }
        )
    candidate_index = tmp_path / "atat_candidate_index.csv"
    candidate_index.write_text(
        "candidate_id,path,source_kind,target_stage,atomi_next,notes\n"
        f"atat_0001,{(tmp_path / 'runs' / 'spin_good').resolve()},enumerated_structure,fail_fast,next,\n",
        encoding="utf-8",
    )
    out = tmp_path / "ce"

    bridge.ce_handoff_main(
        [
            "--summary-csv",
            str(summary),
            "--candidate-index",
            str(candidate_index),
            "--outdir",
            str(out),
            "--formula-units",
            "2",
        ]
    )

    training = rows(out / "ce_training_set.csv")
    assert len(training) == 1
    assert training[0]["training_id"] == "ce_00001"
    assert training[0]["energy_eV_per_fu"] == "-5"
    assert training[0]["relative_energy_eV_per_fu"] == "0"
    assert training[0]["physics_status"] == "OK"
    assert training[0]["atat_candidate_id"] == "atat_0001"
    manifest = json.loads((out / "atat_ce_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "atomi.atat.ce_handoff.v1"
    assert manifest["n_training_rows"] == 1
    assert (out / "atat_mc_population_handoff_template.csv").exists()
    assert (out / "sd_dd_interaction_template.csv").exists()


def test_atat_ce_handoff_cli_accepts_unchecked_rows(tmp_path: Path, capsys) -> None:
    summary = tmp_path / "spin_energy_run_summary.csv"
    summary.write_text(
        "index,run,status,energy_eV,physics_guard_status,mag_status,element_order,changed_by_element\n"
        "1,spin_001,OK,-3.0,NOT_APPLIED,OK,\"{}\",\"{}\"\n",
        encoding="utf-8",
    )
    out = tmp_path / "ce"

    atomi_main(
        [
            "atat-bridge",
            "ce-handoff",
            "--summary-csv",
            str(summary),
            "--outdir",
            str(out),
            "--include-unchecked",
        ]
    )

    assert "CE training rows" in capsys.readouterr().out
    training = rows(out / "ce_training_set.csv")
    assert len(training) == 1
    assert training[0]["mag_status"] == "OK"


def test_quick_materials_opt_writes_uc2_command_scaffold(tmp_path: Path, capsys) -> None:
    template = tmp_path / "VASP_TEMPLATE"
    template.mkdir()
    poscar = tmp_path / "POSCAR_uc2_2x1x1"
    poscar.write_text(
        "UC2 demo\n"
        "1.0\n"
        "1 0 0\n"
        "0 1 0\n"
        "0 0 1\n"
        "U C\n"
        "2 4\n"
        "Direct\n"
        "0 0 0\n"
        "0.5 0.5 0.5\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n"
        "0.25 0.75 0.25\n"
        "0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    for name, text in {
        "INCAR": "MAGMOM = 2 -2 4*0\n",
        "KPOINTS": "Gamma\n",
        "POTCAR": "fake\n",
    }.items():
        (template / name).write_text(text, encoding="utf-8")
    out = tmp_path / "uc2_quick"

    atomi_main(
        [
            "materials-opt",
            "--system",
            "UC2",
            "--formula",
            "UC2",
            "--supercell",
            "2x1x1",
            "--poscar",
            str(poscar),
            "--template",
            str(template),
            "--outdir",
            str(out),
            "--magnetic-element",
            "U",
            "--nonmagnetic-element",
            "C",
            "--moment",
            "U=2",
            "--max-configs",
            "8",
        ]
    )

    assert "Quick optimization workspace" in capsys.readouterr().out
    assert (out / "00_vasp_template" / "POSCAR").read_text(encoding="utf-8").startswith("UC2 demo")
    assert (out / "pseudo_species_map.csv").exists()
    species = rows(out / "pseudo_species_map.csv")
    assert {"U", "C"}.issubset({row["pseudo_species"] for row in species})
    spin_guard = rows(out / "spin_guard_map.csv")
    assert spin_guard[0]["element"] == "U"
    assert spin_guard[0]["allowed_moments"] == "2,-2"
    plan = json.loads((out / "quick_opt_plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == bridge.QUICK_OPT_SCHEMA
    assert plan["moment_guards"] == ["U=2,-2@0.7", "C=0@0.25"]
    assert plan["spin_owner"].startswith("Atomi magit")
    command_text = (out / "QUICK_OPT_COMMANDS.md").read_text(encoding="utf-8")
    assert "magit enum" in command_text
    assert "vasp-branch-live" in command_text
    assert "vasp-spin-report" in command_text
    assert "atat-bridge ce-handoff" in command_text


def test_materials_opt_vacancy_cif_writes_explicit_poscars_and_atat_input(tmp_path: Path, capsys) -> None:
    cif = tmp_path / "partial_gd2o3.cif"
    cif.write_text(
        "data_demo\n"
        "_symmetry_space_group_name_H-M   P1\n"
        "_cell_length_a 5\n"
        "_cell_length_b 5\n"
        "_cell_length_c 5\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "_symmetry_Int_Tables_number 1\n"
        "loop_\n"
        "_space_group_symop_operation_xyz\n"
        "x,y,z\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "Gd1 Gd 0 0 0 1\n"
        "Gd2 Gd 0.5 0.5 0.5 1\n"
        "O1 O 0.25 0.25 0.25 1\n"
        "O2 O 0.75 0.75 0.75 0.2(1)\n",
        encoding="utf-8",
    )
    template = tmp_path / "VASP_TEMPLATE"
    template.mkdir()
    (template / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (template / "KPOINTS").write_text("Gamma\n", encoding="utf-8")
    (template / "POTCAR").write_text("fake\n", encoding="utf-8")
    hpc_config = tmp_path / "atomi_hpc_config.kit.local.json"
    hpc_config.write_text(
        json.dumps(
            {
                "profiles": {
                    "atat": {
                        "root": "/opt/atat",
                        "bin": "/opt/atat/src",
                        "environment": {"ATOMI_ATAT_ROOT": "/opt/atat", "ATOMI_ATAT_BIN": "/opt/atat/src"},
                    },
                    "atat_sqs": {
                        "scheduler": "slurm",
                        "job_name": "atat_test",
                        "time": "03:00:00",
                        "mem": "8G",
                        "cpus_per_task": 2,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "vacancy"

    atomi_main(
        [
            "materials-opt",
            "vacancy-cif",
            "--cif",
            str(cif),
            "--outdir",
            str(out),
            "--supercell",
            "1x1x5",
            "--vasp-template",
            str(template),
            "--hpc-config",
            str(hpc_config),
            "--mcsqs-time",
            "1800",
        ]
    )

    assert "Vacancy CIF workspace" in capsys.readouterr().out
    index = rows(out / "vacancy_candidate_index.csv")
    assert len(index) == 3
    assert {row["kind"] for row in index} == {
        "vacancy_clustered",
        "vacancy_separated",
        "sqs_random_like",
    }
    assert all(row["n_vacancy"] == "4" for row in index)
    assert all(row["vacancy_label"] == "Va" for row in index)
    assert all(json.loads(row["vacancy_site_species_counts_json"]) == {"O": 1} for row in index)
    assert all(row["site_label"] == "O2" for row in index)
    assert all(row["reasonable_stoichiometry"] == "true" for row in index)
    supercells = rows(out / "supercell_candidate_analysis.csv")
    assert any(row["repeat"] == "1x1x5" and row["recommended"] == "true" for row in supercells)
    poscar = out / "candidates" / "01_vacancy_separated" / "POSCAR"
    assert "Va" not in poscar.read_text(encoding="utf-8")
    assert "ISYM = 0" in (out / "candidates" / "01_vacancy_separated" / "INCAR").read_text(encoding="utf-8")
    rndstr = (out / "atat" / "rndstr.in").read_text(encoding="utf-8")
    assert "O=0.2,Va=0.8" in rndstr
    assert rndstr.count("O=0.2,Va=0.8") == 5
    assert (out / "atat" / "run_mcsqs.sh").exists()
    run_mcsqs = (out / "atat" / "run_mcsqs.sh").read_text(encoding="utf-8")
    assert "tail_log mcsqs.err" in run_mcsqs
    assert "ATAT mcsqs search failed" in run_mcsqs
    sbatch = (out / "atat" / "submit_mcsqs.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --job-name=atat_test" in sbatch
    assert "#SBATCH --time=00:30:00" in sbatch
    assert "#SBATCH --cpus-per-task=2" in sbatch
    assert f"WORKDIR={out / 'atat'}" in sbatch
    assert "bash run_mcsqs.sh" in sbatch
    assert "ATOMI_ATAT_BIN=/opt/atat/src" in sbatch
    assert len((out / "runlist.txt").read_text(encoding="utf-8").splitlines()) == 3


def test_materials_opt_vacancy_cif_ranks_random_pool(tmp_path: Path) -> None:
    cif = tmp_path / "partial_gd2o3.cif"
    write_partial_gd2o3_cif(cif)
    out = tmp_path / "vacancy_ranked"

    atomi_main(
        [
            "materials-opt",
            "vacancy-cif",
            "--cif",
            str(cif),
            "--outdir",
            str(out),
            "--engine",
            "direct",
            "--supercell",
            "1x1x5",
            "--random-pool-size",
            "7",
            "--random-candidates",
            "2",
        ]
    )

    index = rows(out / "vacancy_candidate_index.csv")
    assert len(index) == 4
    random_rows = [row for row in index if row["kind"] == "random"]
    assert [row["rank"] for row in random_rows] == ["1", "2"]
    assert all(row["pool_index"] for row in random_rows)
    assert all(row["stability_score"] for row in index)
    assert all((out / "candidates" / row["candidate_id"] / "POSCAR").exists() for row in index)
    pool = rows(out / "random_pool_rankings.csv")
    assert len(pool) == 7
    assert [row["rank"] for row in pool[:3]] == ["1", "2", "3"]
    pool_scores = [float(row["stability_score"]) for row in pool]
    assert pool_scores == sorted(pool_scores, reverse=True)
    plan = json.loads((out / "vacancy_cif_plan.json").read_text(encoding="utf-8"))
    assert plan["random_pool"]["pool_size"] == 7
    assert plan["outputs"]["random_pool_rankings"].endswith("random_pool_rankings.csv")


def test_materials_opt_vacancy_cif_run_mcsqs_converts_bestsqs_to_poscar(tmp_path: Path, monkeypatch) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    mcsqs = fake_bin / "mcsqs"
    mcsqs.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *-2=*)\n"
        "    echo \"$@\" > mcsqs_cluster_args.txt\n"
        "    echo clusters > clusters.out\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n"
        "echo \"$@\" > mcsqs_search_args.txt\n"
        "cat > bestsqs.out <<'EOF'\n"
        "5 0 0\n"
        "0 5 0\n"
        "0 0 5\n"
        "0 0 0 Gd\n"
        "0.5 0.5 0.5 Gd\n"
        "0.25 0.25 0.25 O\n"
        "0.75 0.75 0.75 Va\n"
        "EOF\n",
        encoding="utf-8",
    )
    mcsqs.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    cif = tmp_path / "partial_gd2o3.cif"
    cif.write_text(
        "data_demo\n"
        "_symmetry_space_group_name_H-M   P1\n"
        "_cell_length_a 5\n"
        "_cell_length_b 5\n"
        "_cell_length_c 5\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "_symmetry_Int_Tables_number 1\n"
        "loop_\n"
        "_space_group_symop_operation_xyz\n"
        "x,y,z\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "Gd1 Gd 0 0 0 1\n"
        "Gd2 Gd 0.5 0.5 0.5 1\n"
        "O1 O 0.25 0.25 0.25 1\n"
        "O2 O 0.75 0.75 0.75 0.2\n",
        encoding="utf-8",
    )
    template = tmp_path / "VASP_TEMPLATE"
    template.mkdir()
    (template / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (template / "KPOINTS").write_text("Gamma\n", encoding="utf-8")
    (template / "POTCAR").write_text("fake\n", encoding="utf-8")
    out = tmp_path / "vacancy"

    bridge.vacancy_candidate_main(
        [
            "--cif",
            str(cif),
            "--outdir",
            str(out),
            "--supercell",
            "1x1x5",
            "--run-mcsqs",
            "--vasp-template",
            str(template),
        ]
    )

    assert "-2=6" in (out / "atat" / "mcsqs_cluster_args.txt").read_text(encoding="utf-8")
    search_args = (out / "atat" / "mcsqs_search_args.txt").read_text(encoding="utf-8")
    assert "-n=" in search_args
    assert "-T=" not in search_args
    poscar_text = (out / "atat_vasp" / "candidates" / "01_bestsqs" / "POSCAR").read_text(encoding="utf-8")
    assert "Va" not in poscar_text
    assert "Gd" in poscar_text
    assert "ISYM = 0" in (out / "atat_vasp" / "candidates" / "01_bestsqs" / "INCAR").read_text(encoding="utf-8")
    plan = json.loads((out / "vacancy_cif_plan.json").read_text(encoding="utf-8"))
    assert plan["mcsqs"]["status"] == "ok"
    assert "-2=6" in plan["mcsqs"]["cluster_command"]
    assert plan["outputs"]["atat_vasp"].endswith("atat_vasp")


def test_materials_opt_vacancy_cif_auto_uses_multiple_vacancy_sites(tmp_path: Path) -> None:
    cif = tmp_path / "partial_multi.cif"
    cif.write_text(
        "data_demo\n"
        "_symmetry_space_group_name_H-M   P1\n"
        "_cell_length_a 5\n"
        "_cell_length_b 5\n"
        "_cell_length_c 5\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "_symmetry_Int_Tables_number 1\n"
        "loop_\n"
        "_space_group_symop_operation_xyz\n"
        "x,y,z\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "Gd1 Gd 0 0 0 1\n"
        "Gd2 Gd 0.5 0.5 0.5 0.5\n"
        "O1 O 0.25 0.25 0.25 1\n"
        "O2 O 0.75 0.75 0.75 0.5\n",
        encoding="utf-8",
    )
    out = tmp_path / "vacancy"

    bridge.vacancy_candidate_main(
        [
            "--cif",
            str(cif),
            "--outdir",
            str(out),
            "--supercell",
            "1x1x2",
        ]
    )

    plan = json.loads((out / "vacancy_cif_plan.json").read_text(encoding="utf-8"))
    assert plan["selected_site_labels"] == ["Gd2", "O2"]
    groups = {group["label"]: group for group in plan["vacancy_groups"]}
    assert groups["Gd2"]["n_vacancy"] == 1
    assert groups["O2"]["n_vacancy"] == 1
    rndstr = (out / "atat" / "rndstr.in").read_text(encoding="utf-8")
    assert "Gd=0.5,Va=0.5" in rndstr
    assert "O=0.5,Va=0.5" in rndstr
    index = rows(out / "vacancy_candidate_index.csv")
    assert all(row["n_vacancy"] == "2" for row in index)
    assert all(row["site_label"] == "Gd2,O2" for row in index)

    filtered = tmp_path / "vacancy_o_only"
    bridge.vacancy_candidate_main(
        [
            "--cif",
            str(cif),
            "--outdir",
            str(filtered),
            "--partial-element",
            "O",
            "--supercell",
            "1x1x2",
        ]
    )
    filtered_plan = json.loads((filtered / "vacancy_cif_plan.json").read_text(encoding="utf-8"))
    assert filtered_plan["selected_site_labels"] == ["O2"]
    assert "O" in filtered_plan["vacancy_groups"][0]["species"]


def test_materials_opt_vacancy_cif_materializes_mixed_species_site(tmp_path: Path) -> None:
    cif = tmp_path / "partial_mixed.cif"
    cif.write_text(
        "data_demo\n"
        "_symmetry_space_group_name_H-M   P1\n"
        "_cell_length_a 5\n"
        "_cell_length_b 5\n"
        "_cell_length_c 5\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "_symmetry_Int_Tables_number 1\n"
        "loop_\n"
        "_space_group_symop_operation_xyz\n"
        "x,y,z\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "M1 Gd 0 0 0 0.5\n"
        "M1 U 0 0 0 0.5\n"
        "O1 O 0.25 0.25 0.25 1\n",
        encoding="utf-8",
    )
    out = tmp_path / "mixed"

    bridge.vacancy_candidate_main(
        [
            "--cif",
            str(cif),
            "--outdir",
            str(out),
            "--supercell",
            "1x1x2",
        ]
    )

    plan = json.loads((out / "vacancy_cif_plan.json").read_text(encoding="utf-8"))
    assert plan["selected_site_labels"] == ["M1"]
    assert plan["n_vacancy"] == 0
    assert plan["occupational_groups"][0]["counts"] == {"Gd": 1, "U": 1}
    assert plan["vacancy_groups"] == []
    rndstr = (out / "atat" / "rndstr.in").read_text(encoding="utf-8")
    assert "Gd=0.5,U=0.5" in rndstr
    index = rows(out / "vacancy_candidate_index.csv")
    assert all(row["n_vacancy"] == "0" for row in index)
    assert any('"U": 1' in row["assigned_site_species_json"] for row in index)
    poscar_text = (out / "candidates" / "03_sqs_random_like" / "POSCAR").read_text(encoding="utf-8")
    assert "Va" not in poscar_text
    assert "U" in poscar_text


def test_materials_opt_vacancy_cif_default_auto_prefers_compact_repeat(tmp_path: Path) -> None:
    cif = tmp_path / "partial_single.cif"
    cif.write_text(
        "data_demo\n"
        "_symmetry_space_group_name_H-M   P1\n"
        "_cell_length_a 5\n"
        "_cell_length_b 5\n"
        "_cell_length_c 5\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "_symmetry_Int_Tables_number 1\n"
        "loop_\n"
        "_space_group_symop_operation_xyz\n"
        "x,y,z\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "O1 O 0 0 0 0.5\n",
        encoding="utf-8",
    )

    bridge.vacancy_candidate_main(["--cif", str(cif), "--outdir", str(tmp_path / "out")])

    plan = json.loads((tmp_path / "out" / "vacancy_cif_plan.json").read_text(encoding="utf-8"))
    assert plan["repeat"] == [1, 1, 2]


def test_choose_integer_repeat_avoids_slab_when_compact_repeat_exists() -> None:
    repeat = bridge.choose_integer_repeat(
        [(1, 0.2)],
        max_repeat=5,
        cell_lengths=(1.0, 1.0, 1.0),
        max_aspect=2.5,
    )

    assert repeat == (2, 2, 5)


def test_vacancy_cif_auto_balances_atom_budget_and_compactness() -> None:
    group = {
        "label": "O2",
        "species": {"O": 0.2, "Va": 0.8},
        "indices": list(range(16)),
    }
    rows_ = bridge.repeat_analysis_rows(
        [(16, 0.2), (16, 0.8)],
        max_repeat=5,
        groups=[group],
        n_base_atoms=96,
        vacancy_label="Va",
        cell_lengths=(1.0, 1.0, 1.0),
        max_aspect=2.5,
    )

    small = bridge.choose_repeat_from_analysis(rows_, max_atoms=800, max_aspect=2.5, objective="balanced")
    compact = bridge.choose_repeat_from_analysis(rows_, max_atoms=2000, max_aspect=2.5, objective="balanced")

    assert small == (1, 1, 5)
    assert compact == (2, 2, 5)


def test_vacancy_guard_limits_random_missing_neighbors() -> None:
    from ase import Atoms

    positions = [(5, 5, 5), (15, 15, 15)]
    for base in ((5, 5, 5), (15, 15, 15)):
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    positions.append((base[0] + dx, base[1] + dy, base[2] + dz))
    atoms = Atoms("U2O16", positions=positions, cell=[30, 30, 30], pbc=True)
    candidates = list(range(2, 18))
    guard = bridge.VacancyGuard(
        center_elements={"U"},
        ligand_elements={"O"},
        coordination_number=8,
        max_missing=2,
        attempts=500,
    )

    chosen = bridge.guarded_random_vacancy_set(atoms, candidates, 4, 7, guard, candidates)
    report = bridge.vacancy_guard_report(atoms, chosen, candidates, guard)

    assert len(chosen) == 4
    assert report.status == "OK"
    assert report.max_missing <= 2


def test_materials_opt_atat_poscar_removes_vacancy_pseudo_atoms(tmp_path: Path) -> None:
    bestsqs = tmp_path / "bestsqs.out"
    bestsqs.write_text(
        "5 0 0\n"
        "0 5 0\n"
        "0 0 5\n"
        "0.25 0.25 0.25 O\n"
        "0 0 0 Gd\n"
        "0.5 0.5 0.5 U\n"
        "0.75 0.75 0.75 Vac\n",
        encoding="utf-8",
    )
    template = tmp_path / "VASP_TEMPLATE"
    template.mkdir()
    (template / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (template / "KPOINTS").write_text("Gamma\n", encoding="utf-8")
    (template / "POTCAR").write_text("fake\n", encoding="utf-8")
    out = tmp_path / "atat_vasp"

    atomi_main(
        [
            "materials-opt",
            "atat-poscar",
            "--input",
            str(bestsqs),
            "--outdir",
            str(out),
            "--vasp-template",
            str(template),
        ]
    )

    index = rows(out / "atat_poscar_candidate_index.csv")
    assert len(index) == 1
    assert index[0]["removed_vacancies"] == "1"
    poscar = (out / "candidates" / "01_bestsqs" / "POSCAR").read_text(encoding="utf-8")
    assert "Va" not in poscar
    assert "Vac" not in poscar
    assert poscar.splitlines()[5].split() == ["Gd", "U", "O"]
    assert "Gd" in poscar
    assert "ISYM = 0" in (out / "candidates" / "01_bestsqs" / "INCAR").read_text(encoding="utf-8")
    assert (out / "runlist.txt").read_text(encoding="utf-8").strip().endswith("01_bestsqs")


def test_materials_opt_parent_defect_charge_compensates_and_scales(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "UO2 parent\n"
        "1.0\n"
        "5 0 0\n"
        "0 5 0\n"
        "0 0 5\n"
        "U O\n"
        "2 4\n"
        "Direct\n"
        "0 0 0\n"
        "0.5 0.5 0.5\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n"
        "0.25 0.75 0.25\n"
        "0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    out = tmp_path / "parent_defect"

    atomi_main(
        [
            "materials-opt",
            "parent-defect",
            "--poscar",
            str(poscar),
            "--outdir",
            str(out),
            "--substitute",
            "U=Gd",
            "--charge",
            "U=4",
            "--charge",
            "Gd=3",
            "--charge",
            "O=-2",
            "--vacancy-element",
            "O",
            "--scale-mode",
            "ionic-radius",
            "--radius",
            "U=1.0",
            "--radius",
            "Gd=0.9",
        ]
    )

    plan = json.loads((out / "parent_defect_plan.json").read_text(encoding="utf-8"))
    assert plan["repeat"] == [1, 1, 1]
    assert plan["n_vacancy"] == 1
    assert plan["charge_after_vacancy"] == 0
    assert plan["linear_scale"] == 0.9
    index = rows(out / "parent_defect_candidate_index.csv")
    assert len(index) == 3
    assert all(row["stoichiometry"] == "Gd2 O3" for row in index)
    rndstr = (out / "atat" / "rndstr.in").read_text(encoding="utf-8")
    numeric_lines = [line for line in rndstr.splitlines() if line and len(line.split()) == 3]
    assert len(numeric_lines) == 6
    assert not rndstr.startswith("#")
    assert "O=0.75,Vac=0.25" in rndstr
    assert "Va=0.25" not in rndstr
    assert "U=Gd" not in rndstr
    assert "Gd" in (out / "candidates" / "01_ordered" / "POSCAR").read_text(encoding="utf-8")


def test_materials_opt_parent_defect_ranks_random_pool(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    write_uo2_parent_poscar(poscar)
    out = tmp_path / "parent_defect_ranked"

    atomi_main(
        [
            "materials-opt",
            "parent-defect",
            "--poscar",
            str(poscar),
            "--outdir",
            str(out),
            "--engine",
            "direct",
            "--substitute",
            "U=Gd",
            "--charge",
            "U=4",
            "--charge",
            "Gd=3",
            "--charge",
            "O=-2",
            "--vacancy-element",
            "O",
            "--random-pool-size",
            "6",
            "--random-candidates",
            "2",
        ]
    )

    index = rows(out / "parent_defect_candidate_index.csv")
    assert len(index) == 4
    assert {row["kind"] for row in index[:2]} == {"ordered", "clustered"}
    random_rows = [row for row in index if row["kind"] == "random"]
    assert [row["rank"] for row in random_rows] == ["1", "2"]
    assert all(row["charge_after_vacancy"] == "0" for row in index)
    assert all((out / "candidates" / row["candidate_id"] / "POSCAR").exists() for row in index)
    pool = rows(out / "random_pool_rankings.csv")
    assert len(pool) == 6
    assert [row["rank"] for row in pool[:3]] == ["1", "2", "3"]
    plan = json.loads((out / "parent_defect_plan.json").read_text(encoding="utf-8"))
    assert plan["random_pool"]["pool_size"] == 6
    assert plan["outputs"]["random_pool_rankings"].endswith("random_pool_rankings.csv")


def test_materials_opt_parent_defect_auto_repeats_for_integer_vacancy(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "UO2 primitive-like\n"
        "1.0\n"
        "5 0 0\n"
        "0 5 0\n"
        "0 0 5\n"
        "U O\n"
        "1 2\n"
        "Direct\n"
        "0 0 0\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n",
        encoding="utf-8",
    )
    out = tmp_path / "parent_defect_auto"

    bridge.parent_defect_main(
        [
            "--poscar",
            str(poscar),
            "--outdir",
            str(out),
            "--substitute",
            "U=Gd",
            "--charge",
            "U=4",
            "--charge",
            "Gd=3",
            "--charge",
            "O=-2",
            "--vacancy-element",
            "O",
        ]
    )

    plan = json.loads((out / "parent_defect_plan.json").read_text(encoding="utf-8"))
    assert plan["repeat"] == [1, 1, 2]
    assert plan["n_vacancy"] == 1


def test_parent_defect_run_mcsqs_converts_bestsqs_to_poscar(tmp_path: Path, monkeypatch) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    mcsqs = fake_bin / "mcsqs"
    mcsqs.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *-2=*)\n"
        "    echo \"$@\" > mcsqs_cluster_args.txt\n"
        "    echo clusters > clusters.out\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n"
        "echo \"$@\" > mcsqs_search_args.txt\n"
        "cat > bestsqs.out <<'EOF'\n"
        "5 0 0\n"
        "0 5 0\n"
        "0 0 5\n"
        "0 0 0 Gd\n"
        "0.5 0.5 0.5 U\n"
        "0.25 0.25 0.25 O\n"
        "0.75 0.75 0.75 Va\n"
        "EOF\n",
        encoding="utf-8",
    )
    mcsqs.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "UO2 parent\n"
        "1.0\n"
        "5 0 0\n"
        "0 5 0\n"
        "0 0 5\n"
        "U O\n"
        "2 4\n"
        "Direct\n"
        "0 0 0\n"
        "0.5 0.5 0.5\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n"
        "0.25 0.75 0.25\n"
        "0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    template = tmp_path / "VASP_TEMPLATE"
    template.mkdir()
    (template / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (template / "KPOINTS").write_text("Gamma\n", encoding="utf-8")
    (template / "POTCAR").write_text("fake\n", encoding="utf-8")
    out = tmp_path / "parent_defect"

    bridge.parent_defect_main(
        [
            "--poscar",
            str(poscar),
            "--outdir",
            str(out),
            "--substitute",
            "U=Gd:0.5",
            "--charge",
            "U=4",
            "--charge",
            "Gd=3",
            "--charge",
            "O=-2",
            "--vacancy-element",
            "O",
            "--engine",
            "both",
            "--run-mcsqs",
            "--vasp-template",
            str(template),
        ]
    )

    assert (out / "atat" / "bestsqs.out").exists()
    assert "-2=6" in (out / "atat" / "mcsqs_cluster_args.txt").read_text(encoding="utf-8")
    search_args = (out / "atat" / "mcsqs_search_args.txt").read_text(encoding="utf-8")
    assert "-n=" in search_args
    assert "-T=" not in search_args
    poscar_text = (out / "atat_vasp" / "candidates" / "01_bestsqs" / "POSCAR").read_text(encoding="utf-8")
    assert "Va" not in poscar_text
    assert "Gd" in poscar_text
    assert "ISYM = 0" in (out / "atat_vasp" / "candidates" / "01_bestsqs" / "INCAR").read_text(encoding="utf-8")
    plan = json.loads((out / "parent_defect_plan.json").read_text(encoding="utf-8"))
    assert plan["outputs"]["atat_vasp"].endswith("atat_vasp")
    assert "-2=6" in plan["mcsqs"]["cluster_command"]
    assert "-n=" in " ".join(plan["mcsqs"]["search_command"])


def test_parent_defect_run_mcsqs_failure_keeps_direct_outputs(tmp_path: Path, monkeypatch) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    mcsqs = fake_bin / "mcsqs"
    mcsqs.write_text("#!/bin/sh\necho bad input\necho detail >&2\nexit 1\n", encoding="utf-8")
    mcsqs.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "UO2 parent\n"
        "1.0\n"
        "5 0 0\n"
        "0 5 0\n"
        "0 0 5\n"
        "U O\n"
        "2 4\n"
        "Direct\n"
        "0 0 0\n"
        "0.5 0.5 0.5\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n"
        "0.25 0.75 0.25\n"
        "0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    out = tmp_path / "parent_defect"

    bridge.parent_defect_main(
        [
            "--poscar",
            str(poscar),
            "--outdir",
            str(out),
            "--substitute",
            "U=Gd",
            "--charge",
            "U=4",
            "--charge",
            "Gd=3",
            "--charge",
            "O=-2",
            "--vacancy-element",
            "O",
            "--engine",
            "both",
            "--run-mcsqs",
        ]
    )

    plan = json.loads((out / "parent_defect_plan.json").read_text(encoding="utf-8"))
    assert plan["mcsqs"]["status"] == "failed"
    failure = (out / "atat" / "mcsqs_failed.txt").read_text(encoding="utf-8")
    assert "bad input" in failure
    assert "detail" in failure
    assert (out / "atat" / "mcsqs_clusters.out").read_text(encoding="utf-8").strip() == "bad input"
    assert (out / "atat" / "mcsqs_clusters.err").read_text(encoding="utf-8").strip() == "detail"
    assert (out / "candidates" / "01_ordered" / "POSCAR").exists()
    assert not (out / "atat_vasp").exists()


def test_materials_opt_relax_seeds_prepares_volume_scan(tmp_path: Path, capsys) -> None:
    template = tmp_path / "VASP_TEMPLATE"
    template.mkdir()
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "UC2 demo\n"
        "1.0\n"
        "2 0 0\n"
        "0 1 0\n"
        "0 0 1\n"
        "U C\n"
        "2 4\n"
        "Direct\n"
        "0 0 0\n"
        "0.5 0.5 0.5\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n"
        "0.25 0.75 0.25\n"
        "0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    for name, text in {
        "INCAR": "ENCUT = 520\n",
        "KPOINTS": "Gamma\n",
        "POTCAR": "fake\n",
    }.items():
        (template / name).write_text(text, encoding="utf-8")
    out = tmp_path / "uc2_relax"

    atomi_main(
        [
            "materials-opt",
            "relax-seeds",
            "--system",
            "UC2",
            "--formula",
            "UC2",
            "--poscar",
            str(poscar),
            "--template",
            str(template),
            "--outdir",
            str(out),
            "--magnetic-element",
            "U",
            "--nonmagnetic-element",
            "C",
            "--moment",
            "U=2",
            "--seed-spins",
            "fm,afm",
            "--volume-scale",
            "0.98",
            "1.02",
        ]
    )

    assert "Relax-seeds workspace" in capsys.readouterr().out
    runlist = (out / "runlist.txt").read_text(encoding="utf-8").splitlines()
    assert len(runlist) == 4
    assert (out / "01_seed_spins" / "fm" / "INCAR").exists()
    assert (out / "02_volume_isif2" / "SUMMARY.csv").exists()
    index_rows = rows(out / "relax_index.csv")
    assert {row["seed"] for row in index_rows} == {"fm", "afm"}
    assert {row["volume_scale"] for row in index_rows} == {"0.98", "1.02"}
    incar = (out / runlist[0] / "INCAR").read_text(encoding="utf-8")
    assert "ISIF = 2" in incar
    assert "MAGMOM =" in incar

    atomi_main(["materials-opt", "relax-summary", "--workspace", str(out), "--no-plot"])

    assert "Rows summarized" in capsys.readouterr().out
    summary = rows(out / "SUMMARY_volume_isif2.csv")
    assert len(summary) == 4
    assert (out / "04_summary" / "volume_isif2" / "spin_energy_run_summary.csv").exists()


def test_materials_opt_relax_seeds_handles_repeated_duplicate_species_blocks(tmp_path: Path) -> None:
    template = tmp_path / "VASP_TEMPLATE"
    template.mkdir()
    poscar = tmp_path / "POSCAR_2x1x1_duplicate_blocks"
    poscar.write_text(
        "UC2 duplicate species blocks from ASE repeat\n"
        "1.0\n"
        "4 0 0\n"
        "0 1 0\n"
        "0 0 1\n"
        "U C U C\n"
        "2 4 2 4\n"
        "Direct\n"
        "0 0 0\n"
        "0.25 0 0\n"
        "0.1 0 0\n"
        "0.2 0 0\n"
        "0.3 0 0\n"
        "0.4 0 0\n"
        "0.5 0 0\n"
        "0.75 0 0\n"
        "0.6 0 0\n"
        "0.7 0 0\n"
        "0.8 0 0\n"
        "0.9 0 0\n",
        encoding="utf-8",
    )
    for name, text in {
        "POSCAR": (
            "UC2 template in C U order\n"
            "1.0\n"
            "4 0 0\n"
            "0 1 0\n"
            "0 0 1\n"
            "C U\n"
            "8 4\n"
            "Direct\n"
            "0.1 0 0\n"
            "0.2 0 0\n"
            "0.3 0 0\n"
            "0.4 0 0\n"
            "0.6 0 0\n"
            "0.7 0 0\n"
            "0.8 0 0\n"
            "0.9 0 0\n"
            "0 0 0\n"
            "0.25 0 0\n"
            "0.5 0 0\n"
            "0.75 0 0\n"
        ),
        "INCAR": "ENCUT = 520\nLDAUL = -1 3\nLDAUU = 0.0 4.0\nLDAUJ = 0 0\nNUPDOWN = 0\nMAGMOM = 2 -2 10*0\n",
        "KPOINTS": "Gamma\n",
        "POTCAR": "fake\n",
    }.items():
        (template / name).write_text(text, encoding="utf-8")
    out = tmp_path / "uc2_relax"

    atomi_main(
        [
            "materials-opt",
            "relax-seeds",
            "--system",
            "UC2",
            "--formula",
            "UC2",
            "--poscar",
            str(poscar),
            "--template",
            str(template),
            "--outdir",
            str(out),
            "--magnetic-element",
            "U",
            "--nonmagnetic-element",
            "C",
            "--moment",
            "U=2",
            "--seed-spins",
            "afm",
            "--volume-scale",
            "1.00",
        ]
    )

    incar = (out / "02_volume_isif2" / "run_0001_afm_v1" / "INCAR").read_text(encoding="utf-8")
    poscar_text = (out / "02_volume_isif2" / "run_0001_afm_v1" / "POSCAR").read_text(encoding="utf-8")
    assert "#NUPDOWN = 0" in incar
    assert "MAGMOM = 2 -2 2 -2 8*0" in incar
    assert "LDAUL = 3 -1" in incar
    assert "LDAUU = 4.0 0.0" in incar
    assert "LDAUJ = 0 0" in incar
    assert "U  C" in poscar_text
    assert "4  8" in poscar_text


def test_materials_opt_relax_seeds_accepts_seed_root_from_generators(tmp_path: Path, capsys) -> None:
    template = tmp_path / "VASP_TEMPLATE"
    template.mkdir()
    seed_root = tmp_path / "generated_candidates"
    candidate = seed_root / "candidates" / "01_vacancy_separated"
    candidate.mkdir(parents=True)
    (candidate / "POSCAR").write_text(
        "Na3U5Cl18 candidate with duplicate blocks\n"
        "1.0\n"
        "7.6 0 0\n"
        "-3.8 6.6 0\n"
        "0 0 13.0\n"
        "Na U Cl U Cl Na U Cl\n"
        "2 1 6 2 6 1 2 6\n"
        "Direct\n"
        "0 0 0\n"
        "0.1 0 0\n"
        "0.2 0 0\n"
        "0.3 0 0\n"
        "0.4 0 0\n"
        "0.5 0 0\n"
        "0.6 0 0\n"
        "0.7 0 0\n"
        "0.8 0 0\n"
        "0.9 0 0\n"
        "0.11 0 0\n"
        "0.12 0 0\n"
        "0.13 0 0\n"
        "0.14 0 0\n"
        "0.15 0 0\n"
        "0.16 0 0\n"
        "0.17 0 0\n"
        "0.18 0 0\n"
        "0.19 0 0\n"
        "0.21 0 0\n"
        "0.22 0 0\n"
        "0.23 0 0\n"
        "0.24 0 0\n"
        "0.25 0 0\n"
        "0.26 0 0\n"
        "0.27 0 0\n",
        encoding="utf-8",
    )
    for name, text in {
        "POSCAR": (
            "Na U Cl template\n"
            "1.0\n"
            "7.6 0 0\n"
            "-3.8 6.6 0\n"
            "0 0 13.0\n"
            "Na U Cl\n"
            "3 5 18\n"
            "Direct\n"
            + "\n".join(["0 0 0"] * 26)
            + "\n"
        ),
        "INCAR": "ENCUT = 520\nLDAUL = -1 3 -1\nLDAUU = 0 4 0\nLDAUJ = 0 0 0\n",
        "KPOINTS": "Gamma\n",
        "POTCAR": "fake\n",
    }.items():
        (template / name).write_text(text, encoding="utf-8")
    out = tmp_path / "na3u5cl18_relax"

    atomi_main(
        [
            "materials-opt",
            "relax-seeds",
            "--system",
            "Na3U5Cl18",
            "--formula",
            "Na3U5Cl18",
            "--seed-root",
            str(seed_root),
            "--source-kind",
            "vacancy-cif",
            "--template",
            str(template),
            "--outdir",
            str(out),
            "--magnetic-element",
            "U",
            "--nonmagnetic-element",
            "Na,Cl",
            "--moment",
            "U=3",
            "--seed-spins",
            "afm",
            "--volume-scale",
            "1.00",
        ]
    )

    assert "Input POSCAR seeds    : 1" in capsys.readouterr().out
    index_rows = rows(out / "relax_index.csv")
    assert len(index_rows) == 1
    assert index_rows[0]["source_kind"] == "vacancy-cif"
    assert index_rows[0]["input_seed"] == "01_vacancy_separated"
    assert index_rows[0]["source_poscar"].endswith("01_vacancy_separated/POSCAR")
    run_dir = out / index_rows[0]["run_dir"]
    assert (run_dir / "POSCAR").exists()
    assert "Na  U  Cl" in (run_dir / "POSCAR").read_text(encoding="utf-8")
    assert "3  5  18" in (run_dir / "POSCAR").read_text(encoding="utf-8")
    plan = json.loads((out / "relax_plan.json").read_text(encoding="utf-8"))
    assert plan["source_kind"] == "vacancy-cif"
    assert plan["seed_sources"][0]["input_seed"] == "01_vacancy_separated"
