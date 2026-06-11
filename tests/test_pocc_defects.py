from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path

from atomi.zentropy.pocc_defects import (
    DefectConfiguration,
    build_degeneracy_table,
    build_magnetic_initialization_rows,
    effective_charge,
    find_vasp_run_dirs,
    fluorite_fm3m_orbit_degeneracy,
    gduo2_charge_neutral_motif_rows,
    gduo2_observables,
    ingest_vasp_runs,
    raw_supercell_degeneracy,
    solve_static_zentropy,
)


def test_gduo2_charge_and_composition_observables() -> None:
    gd_u5 = {"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0}
    gd2_vo = {"U4": 30, "U5": 0, "Gd3": 2, "O": 63, "VaO": 1}

    assert effective_charge(gd_u5) == 0
    assert effective_charge(gd2_vo) == 0
    assert gduo2_observables(gd_u5)["x_Gd"] == gduo2_observables(gd2_vo)["x_Gd"]
    assert gduo2_observables(gd_u5)["h_U5"] == 2 / 32
    assert gduo2_observables(gd2_vo)["delta"] == 1 / 32


def test_static_zentropy_population_keeps_degeneracy_separate_from_energy() -> None:
    configs = [
        DefectConfiguration(
            config_id="gd_u5_nn",
            phase="fluorite",
            species_counts={"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0},
            sublattice_counts={"cation": 32, "anion": 64},
            degeneracy=2,
            degeneracy_type="motif_embedding",
            E_static_eV=-100.00,
            motif_labels=["Gd_U5_NN"],
            metadata={"oxidation_assignment": "manual_review"},
        ),
        DefectConfiguration(
            config_id="gd2_vo_nn",
            phase="fluorite",
            species_counts={"U4": 30, "U5": 0, "Gd3": 2, "O": 63, "VaO": 1},
            sublattice_counts={"cation": 32, "anion": 64},
            degeneracy=10,
            degeneracy_type="motif_embedding",
            E_static_eV=-99.95,
            motif_labels=["Gd2_VO_NN"],
        ),
    ]

    population, surface, motifs = solve_static_zentropy(
        configs,
        temperatures=[1200.0],
        mu_o_values=[None],
        group_by_x_gd=False,
    )

    assert len(population) == 2
    assert len(surface) == 1
    assert len(motifs) == 2
    probs = {row["config_id"]: row["probability"] for row in population}
    assert probs["gd2_vo_nn"] > probs["gd_u5_nn"]
    assert surface[0]["surface_builder_mode"] == "pocc_static_logsum"
    assert surface[0]["surface_kind"] == "bulk_F_from_static_logsum"
    assert surface[0]["configuration_energy_kind"] == "E_static_DFT"
    assert surface[0]["S_population_J_molK"] > 0
    assert "S_site_ideal_J_molK" in surface[0]
    assert "S_excess_conf_J_molK" in surface[0]


def test_cli_solve_static_outputs_tables(tmp_path: Path) -> None:
    from atomi.zentropy.pocc_defects import main

    ensemble = tmp_path / "ensemble.jsonl"
    records = [
        {
            "config_id": "gd_u5",
            "species_counts": {"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0},
            "degeneracy": 1,
            "E_static_eV": -10.0,
            "motif_labels": ["Gd_U5"],
            "oxidation_assignment": "manual_review",
        },
        {
            "config_id": "gd2_vo",
            "species_counts": {"U4": 30, "U5": 0, "Gd3": 2, "O": 63, "VaO": 1},
            "degeneracy": 4,
            "E_static_eV": -9.9,
            "motif_labels": ["Gd2_VO"],
        },
    ]
    ensemble.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    outdir = tmp_path / "out"

    main(
        [
            "static-logsum",
            "--ensemble",
            str(ensemble),
            "--outdir",
            str(outdir),
            "--temperature",
            "1000",
            "--mu-o",
            "-5",
            "--no-group-by-x-gd",
        ]
    )

    assert (outdir / "configuration_audit.csv").exists()
    assert (outdir / "population_vector.csv").exists()
    assert (outdir / "static_logsum_surface.csv").exists()
    with (outdir / "zentropy_surface.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["dominant_config_id"] in {"gd_u5", "gd2_vo"}


def test_degeneracy_table_uses_raw_supercell_counts(tmp_path: Path) -> None:
    rows, metadata, symmetry_rows = build_degeneracy_table(
        [
            {
                "motif_id": "2Gd_2U5_sc32",
                "N_cation": 32,
                "N_anion_sites": 64,
                "Gd3": 2,
                "U5": 2,
                "VaO": 0,
            },
            {
                "motif_id": "2Gd_VaO_sc32",
                "N_cation": 32,
                "N_anion_sites": 64,
                "Gd3": 2,
                "U5": 0,
                "VaO": 1,
            },
        ]
    )

    assert metadata["n_motifs"] == 2
    assert symmetry_rows == []
    by_id = {row["motif_id"]: row for row in rows}
    assert by_id["2Gd_2U5_sc32"]["g_raw_supercell"] == raw_supercell_degeneracy(
        {"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0}
    )
    assert by_id["2Gd_2U5_sc32"]["g_raw_supercell"] == 215760
    assert by_id["2Gd_VaO_sc32"]["g_raw_supercell"] == 31744

    from atomi.zentropy.pocc_defects import main

    motif_csv = tmp_path / "motifs.csv"
    motif_csv.write_text(
        "motif_id,N_cation,N_anion_sites,Gd3,U5,VaO\n"
        "2Gd_2U5_sc32,32,64,2,2,0\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "deg"
    main(["degeneracy-table", "--motif-csv", str(motif_csv), "--outdir", str(outdir)])
    assert (outdir / "motif_degeneracy_gk.csv").exists()


def test_gduo2_generated_charge_neutral_degeneracy_grid(tmp_path: Path) -> None:
    rows = gduo2_charge_neutral_motif_rows(n_cation=32, gd_counts=[1, 2, 4])
    assert [row["motif_id"] for row in rows] == [
        "1Gd_1U5_sc32",
        "2Gd_2U5_sc32",
        "2Gd_1VaO_sc32",
        "4Gd_4U5_sc32",
        "4Gd_2U5_1VaO_sc32",
        "4Gd_2VaO_sc32",
    ]
    table, metadata, _ = build_degeneracy_table(rows)
    assert metadata["n_motifs"] == 6
    assert all(row["charge_neutral"] for row in table)
    by_id = {row["motif_id"]: row for row in table}
    assert by_id["1Gd_1U5_sc32"]["g_raw_supercell"] == 992
    assert by_id["4Gd_2U5_1VaO_sc32"]["g_raw_supercell"] == raw_supercell_degeneracy(
        {"U4": 26, "U5": 2, "Gd3": 4, "O": 63, "VaO": 1}
    )

    from atomi.zentropy.pocc_defects import main

    outdir = tmp_path / "generated_grid"
    main(
        [
            "degeneracy-table",
            "--gduo2-all-charge-neutral",
            "--n-cation",
            "32",
            "--gd-count",
            "1",
            "--gd-count",
            "2",
            "--gd-count",
            "4",
            "--outdir",
            str(outdir),
        ]
    )
    assert (outdir / "motif_degeneracy_gk.csv").exists()


def test_fluorite_fm3m_symmetry_reduction_keeps_orbit_degeneracy_auditable() -> None:
    summary, orbit_rows = fluorite_fm3m_orbit_degeneracy(
        {"U4": 30, "U5": 1, "Gd3": 1, "O": 64, "VaO": 0},
        motif_id="1Gd_1U5_sc32",
    )

    assert summary["symmetry_reduction_status"] == "exact"
    assert summary["g_sigma_sum"] == 992
    assert summary["g_raw_supercell"] == 992
    assert summary["n_symmetry_distinct_configs"] > 1
    assert sum(row["g_sigma"] for row in orbit_rows) == 992


def test_degeneracy_table_can_write_fluorite_symmetry_orbits(tmp_path: Path) -> None:
    rows = gduo2_charge_neutral_motif_rows(n_cation=32, gd_counts=[1])
    table, _, symmetry_rows = build_degeneracy_table(
        rows,
        symmetry_reduce_fluorite_fm3m=True,
    )

    assert table[0]["degeneracy_kind"] == "motif_sum_of_symmetry_orbits"
    assert table[0]["degeneracy_status"] == "symmetry_reduced_audited_sum"
    assert table[0]["g_sigma_sum"] == table[0]["g_raw_supercell"]
    assert symmetry_rows

    from atomi.zentropy.pocc_defects import main

    outdir = tmp_path / "sym"
    main(
        [
            "degeneracy-table",
            "--gduo2-all-charge-neutral",
            "--n-cation",
            "32",
            "--gd-count",
            "1",
            "--fluorite-fm3m-symmetry-reduce",
            "--outdir",
            str(outdir),
        ]
    )
    assert (outdir / "motif_degeneracy_gk.csv").exists()
    assert (outdir / "symmetry_reduced_configurations.csv").exists()


def test_fluorite_layered_symmetry_reduction_handles_larger_motif() -> None:
    summary, orbit_rows = fluorite_fm3m_orbit_degeneracy(
        {"U4": 28, "U5": 0, "Gd3": 4, "O": 62, "VaO": 2},
        motif_id="4Gd_2VaO_sc32",
        max_raw_enumerate=100,
    )

    assert summary["symmetry_reduction_status"] == "exact_layered_stabilizer"
    assert summary["g_raw_supercell"] == 72495360
    assert summary["g_sigma_sum"] == 72495360
    assert summary["n_symmetry_distinct_configs"] == len(orbit_rows)
    assert orbit_rows


def test_magnetic_initialization_rows_follow_gd_u_o_order() -> None:
    rows, metadata = build_magnetic_initialization_rows(
        [
            {
                "motif_id": "2Gd_2U5_sc32",
                "config_id": "2Gd_2U5_sc32_orbit_0001",
                "g_sigma": 384,
                "representative_Gd3_sites": "0 1",
                "representative_U5_sites": "2 3",
                "representative_VaO_sites": "",
            }
        ],
        include_time_reversal=True,
    )

    assert metadata["poscar_element_order"] == "Gd U O"
    assert len(rows) == 2
    first = rows[0]
    second = rows[1]
    assert first["poscar_element_order"] == "Gd U O"
    assert first["recommended_ldau"].startswith("LDAUTYPE=1")
    moments = first["magmom_poscar_order"].split()
    assert moments[:2] == ["+7", "+7"]
    assert moments[2:6] == ["+1", "-1", "+2", "-2"]
    assert moments[-1] == "0"
    assert first["n_Gd3_up"] == 2
    assert first["n_U5_up"] == 1
    assert first["n_U5_down"] == 1
    assert second["time_reversal_of"] == "2Gd_2U5_sc32_orbit_0001_gd_fm_up"
    assert second["magmom_poscar_order"].split()[:2] == ["-7", "-7"]


def test_cli_magnetic_init_table_outputs_magmom_rows(tmp_path: Path) -> None:
    from atomi.zentropy.pocc_defects import main

    config_csv = tmp_path / "symmetry_configs.csv"
    with config_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "motif_id",
                "config_id",
                "g_sigma",
                "representative_Gd3_sites",
                "representative_U5_sites",
                "representative_VaO_sites",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "motif_id": "1Gd_1U5_sc32",
                "config_id": "1Gd_1U5_sc32_orbit_0001",
                "g_sigma": "32",
                "representative_Gd3_sites": "0",
                "representative_U5_sites": "1",
                "representative_VaO_sites": "",
            }
        )
    outdir = tmp_path / "mag"
    main(["magnetic-init-table", "--config-csv", str(config_csv), "--outdir", str(outdir)])

    table = outdir / "magnetic_initialization_table.csv"
    assert table.exists()
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["magmom_poscar_order"].split()[0] == "+7"
    assert rows[1]["magmom_poscar_order"].split()[0] == "-7"


def test_vasp_ingest_builds_auditable_pocc_ensemble(tmp_path: Path) -> None:
    from atomi.zentropy.pocc_defects import main

    run = tmp_path / "gd_u5_run"
    run.mkdir()
    (run / "CONTCAR").write_text(
        """Gd-UO2 defect motif
1.0
4 0 0
0 4 0
0 0 4
U Gd O
30 2 64
Direct
""",
        encoding="utf-8",
    )
    (run / "OUTCAR").write_text(
        """NIONS =     96 ions
 free  energy   TOTEN  =      -123.456789 eV
 volume of cell :       64.0000
""",
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "run,config_id,U5,degeneracy,motif_labels,oxidation_assignment\n"
        f"{run},gd_u5_nn,2,3,Gd_U5_NN,manual_review\n",
        encoding="utf-8",
    )

    configs, audit = ingest_vasp_runs([run], metadata={str(run): {"U5": "2", "oxidation_assignment": "manual"}})
    assert configs[0].species_counts == {"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0}
    assert audit[0]["effective_charge"] == 0

    outdir = tmp_path / "ingest"
    main(["ingest-vasp", "--run-dir", str(run), "--metadata-csv", str(metadata), "--outdir", str(outdir)])

    assert (outdir / "ensemble.jsonl").exists()
    record = json.loads((outdir / "ensemble.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["config_id"] == "gd_u5_nn"
    assert record["E_static_eV"] == -123.456789
    assert record["species_counts"]["U4"] == 28
    with (outdir / "vasp_ingest_audit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["warnings"] == ""


def test_vasp_ingest_reads_canonical_tgz_archive(tmp_path: Path) -> None:
    run = tmp_path / "archived_case"
    run.mkdir()
    archive_root = tmp_path / "scratch_result"
    archive_root.mkdir()
    (archive_root / "CONTCAR").write_text(
        """Gd-UO2 archived motif
1.0
4 0 0
0 4 0
0 0 4
U Gd O
30 2 64
Direct
""",
        encoding="utf-8",
    )
    (archive_root / "OSZICAR").write_text(
        "DAV: 1 -0.123E+03\n   1 F= -.12345679E+03 E0= -.12340000E+03  d E =-.1E-02  mag= 14.0\n",
        encoding="utf-8",
    )
    archive = run / "bwforcluster-bulk_48.sbatch.12345.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(archive_root / "CONTCAR", arcname="scratch_result/CONTCAR")
        handle.add(archive_root / "OSZICAR", arcname="scratch_result/OSZICAR")

    found = find_vasp_run_dirs([tmp_path])
    assert run.resolve() in found

    configs, audit = ingest_vasp_runs(
        [run],
        metadata={str(run): {"U5": "2", "oxidation_assignment": "manual_review", "degeneracy": "7"}},
    )
    assert configs[0].E_static_eV == -123.45679
    assert configs[0].species_counts == {"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0}
    assert configs[0].degeneracy == 7
    assert "calc_from_archive" in audit[0]["warnings"]
    assert "structure_from_archive" in audit[0]["warnings"]
