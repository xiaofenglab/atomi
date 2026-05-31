from __future__ import annotations

import gzip
import json
from pathlib import Path

from atomi.cli.main import main as atomi_main
from atomi.reporting import paper_draft


def write_vasp_run(root: Path) -> None:
    root.mkdir()
    (root / "POSCAR").write_text(
        "\n".join(
            [
                "UO2 test",
                "1.0",
                "5 0 0",
                "0 5 0",
                "0 0 5",
                "U O",
                "1 2",
                "Direct",
                "0 0 0",
                "0.25 0.25 0.25",
                "0.75 0.75 0.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "INCAR").write_text(
        "ENCUT = 520\nEDIFF = 1E-6\nISPIN = 2\nMAGMOM = 2 0 0\n",
        encoding="utf-8",
    )
    (root / "KPOINTS").write_text("mesh\n0\nGamma\n3 3 3\n0 0 0\n", encoding="utf-8")
    (root / "POTCAR").write_text(
        "TITEL  = PAW_PBE U 06Sep2000\nTITEL  = PAW_PBE O 08Apr2002\n",
        encoding="utf-8",
    )
    (root / "OUTCAR").write_text(
        "\n".join(
            [
                " vasp.6.4.3 01Jan24 (build test) standard",
                " NIONS =      3 ions",
                " volume of cell :      41.234",
                " free  energy   TOTEN  =      -25.125 eV",
                " Elapsed time (sec):  12.5",
                " General timing and accounting informations for this job:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_lammps_run(root: Path) -> None:
    root.mkdir()
    (root / "log.lammps").write_text(
        "\n".join(
            [
                "LAMMPS test",
                "Step Temp Press PotEng Volume",
                "0 300 1 -100 1000",
                "100 305 2 -99 1001",
                "Loop time of 1.0 on 1 procs for 100 steps",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_cp2k_reaction_run(root: Path) -> None:
    root.mkdir()
    (root / "ga_cl4.inp").write_text(
        "\n".join(
            [
                "&GLOBAL",
                "  PROJECT ga_cl4_test",
                "  RUN_TYPE MD",
                "&END GLOBAL",
                "&FORCE_EVAL",
                "  &DFT",
                "    CHARGE 0",
                "  &END DFT",
                "  &SUBSYS",
                "    &CELL",
                "      ABC 22 22 22",
                "    &END CELL",
                "  &END SUBSYS",
                "&END FORCE_EVAL",
                "&MOTION",
                "  &MD",
                "    ENSEMBLE NVT",
                "    STEPS 12000",
                "    TIMESTEP 0.5",
                "    TEMPERATURE 300",
                "  &END MD",
                "&END MOTION",
                "&COLVAR",
                "  &DISTANCE",
                "    ATOMS 1 7",
                "  &END DISTANCE",
                "&END COLVAR",
                "&COLVAR",
                "  &DISTANCE",
                "    ATOMS 1 698",
                "  &END DISTANCE",
                "&END COLVAR",
                "&COLLECTIVE",
                "  COLVAR 1",
                "  &RESTRAINT",
                "    TARGET [angstrom] 3.20",
                "    K [kcalmol] 50",
                "  &END RESTRAINT",
                "&END COLLECTIVE",
                "&COLLECTIVE",
                "  COLVAR 2",
                "  &RESTRAINT",
                "    TARGET [angstrom] 2.30",
                "    K [kcalmol] 40",
                "  &END RESTRAINT",
                "&END COLLECTIVE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "ga_cl4.log").write_text(
        "MD| Step number 12000\nENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): -550.0\nPROGRAM ENDED\n",
        encoding="utf-8",
    )
    (root / "ga_cl4-pos.xyz").write_text(
        "3\ni = 12000\nGa 0 0 0\nCl 3.20 0 0\nO 2.20 0 0\n",
        encoding="utf-8",
    )
    (root / "ga_cl4_ow_bonds.csv").write_text(
        "\n".join(
            [
                "file,metal,metal_index,tracked_index,tracked_symbol,tracked_mean_all,tracked_mean_tail,tracked_min_tail,tracked_max_tail,shell_mean_all,shell_mean_tail,shell_min_tail,shell_max_tail,nframes_total,tail_nframes",
                "t6-pos.xyz,Ga,1,7,Cl,2.80,3.20,3.10,3.30,2.75,2.70,2.10,3.30,100,20",
                "t6-pos.xyz,Ga,1,698,O,2.60,2.20,2.10,2.30,2.75,2.70,2.10,3.30,100,20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_transport_run(root: Path) -> None:
    root.mkdir()
    (root / "config_gk_test.json").write_text(
        json.dumps(
            {
                "runtime_profile": "lammps_gk_mliap",
                "pair_style_backend": "mliap",
                "model_file": "models/uo2.model-mliap_lammps.pt",
                "model_elements": ["O", "U"],
                "temperatures_K": [300, 900, 1500],
                "n_seeds": 3,
                "timestep_ps": 0.0005,
                "heat_flux_suffix": "kk",
                "nvt_preequilibration_ps": 10,
                "nve_time_ps": 50,
                "sample_interval_ps": 0.01,
                "correlation_time_ps": 5,
                "plateau_window_ps": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "gk_plan.json").write_text(
        json.dumps(
            {
                "n_temperatures": 3,
                "n_seeds_per_temperature": 3,
                "n_stages": 9,
                "temperatures_K": [300, 900, 1500],
                "runtime_estimate": {
                    "timestep_ps": 0.0005,
                    "timestep_fs": 0.5,
                    "nvt_time_ps_per_stage": 10,
                    "nve_time_ps_per_stage": 50,
                    "estimated_steps_per_hour": 7752,
                    "estimated_walltime_hours_per_stage": 23.2,
                    "estimated_elapsed_hours_at_array_limit": 46.4,
                    "array_limit": 6,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "gk_seed_summary.csv").write_text(
        "\n".join(
            [
                "temperature_K,seed,k_W_mK,status",
                "300,1,5.8,ok",
                "300,2,6.0,ok",
                "900,1,2.7,ok",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "thermal_conductivity_T.csv").write_text(
        "\n".join(
            [
                "temperature_K,k_W_mK,ok_seed_count,seed_count,axis_spread_fraction,seed_cv_fraction,late_drift_fraction,status",
                "300,5.9,3,3,0.07,0.05,0.12,ok",
                "900,2.8,3,3,0.18,0.09,0.15,ok",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "gk_validation_summary.json").write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "temperature_K": 300,
                        "status": "ok",
                        "k_W_mK": 5.9,
                        "ok_seed_count": 3,
                        "seed_count": 3,
                        "axis_spread_fraction": 0.07,
                        "seed_cv_fraction": 0.05,
                        "late_drift_fraction": 0.12,
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "config_rnemd_test.json").write_text(
        json.dumps(
            {
                "pair_style_backend": "mace",
                "temperatures_K": [300, 900, 1500],
                "n_seeds": 3,
                "timestep_ps": 0.0005,
                "run_time_ps": 50,
                "replicate": "1x1x3",
                "direction": "z",
                "nbin": 20,
                "swap_every": 100,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "rnemd_plan.json").write_text(
        json.dumps(
            {
                "n_temperatures": 3,
                "n_seeds_per_temperature": 3,
                "n_stages": 9,
                "runtime_estimate": {
                    "replicate": "1x1x3",
                    "run_steps_per_stage": 100000,
                    "estimated_steps_per_hour": 7752,
                    "estimated_walltime_hours_per_stage": 19.4,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "thermal_conductivity_rnemd_T.csv").write_text(
        "temperature_K,k_W_mK,ok_seed_count,seed_count,slope_disagreement_fraction,status\n1200,2.6,1,1,0.316,warn\n",
        encoding="utf-8",
    )
    (root / "rnemd_validation_summary.json").write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "temperature_K": 1200,
                        "status": "warn",
                        "k_W_mK": 2.605,
                        "ok_seed_count": 1,
                        "seed_count": 1,
                        "slope_disagreement_fraction": 0.316,
                        "warnings": ["inspect profile linearity"],
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_calphad_run(root: Path) -> None:
    root.mkdir()
    (root / "test.TDB").write_text(
        "ELEMENT U BLANK 0 0 0 !\nELEMENT O BLANK 0 0 0 !\nPHASE FLUORITE % 2 1 1 !\n",
        encoding="utf-8",
    )


def write_defect_cloud_run(root: Path) -> None:
    root.mkdir()
    (root / "defect_cloud_summary.json").write_text(
        json.dumps(
            {
                "schema": "atomi.vasp.defect_cloud.summary.v1",
                "n_seed_motifs": 2,
                "n_candidate_runs": 16,
                "per_motif_requested": 8,
                "seed": 20260518,
                "families_by_motif": {
                    "GdUO2_seed_01": {
                        "base": 1,
                        "random_displacement": 3,
                        "isotropic_strain": 2,
                        "species_biased_displacement": 1,
                        "mixed_displacement": 1,
                    },
                    "GdUO2_seed_02": {
                        "base": 1,
                        "random_displacement": 3,
                        "isotropic_strain": 2,
                        "species_biased_displacement": 1,
                        "mixed_displacement": 1,
                    },
                },
                "defaults": {
                    "random_amp_A": 0.02,
                    "structured_amp_A": 0.01,
                    "bias_species": "O",
                    "bias_amp_A": 0.05,
                    "mixed_amp_A": 0.04,
                    "iso_strains": [-0.01, 0.01],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "defect_cloud_index.csv").write_text(
        "\n".join(
            [
                "motif_id,family,run_dir",
                "GdUO2_seed_01,base,GdUO2_seed_01/base",
                "GdUO2_seed_01,random_displacement,GdUO2_seed_01/random_001",
                "GdUO2_seed_02,species_biased_displacement,GdUO2_seed_02/bias_O_001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "runlist.txt").write_text("GdUO2_seed_01/base\nGdUO2_seed_01/random_001\n", encoding="utf-8")


def write_spin_report_run(root: Path) -> None:
    root.mkdir()
    (root / "spin_energy_run_summary.csv").write_text(
        "\n".join(
            [
                "index,run,status,energy_eV,energy_kind,energy_source,mag_source,mag_status,total_moment,max_abs_moment,changed_count,changed_by_element,abs_gt5,abs_0p5_1p5,abs_1p5_2p5,initial_element_order,element_order,element_sum,physics_guard_status,physics_guard_bad_count,physics_guard_bad_by_element,spin_index_name,dopant_mode,host_mode,warning",
                '1,spin_001,OK,-100.0,TOTEN,vasp.out.1,OUTCAR,OK,0.2,7.1,1,"{""U"": 1}",2,0,10,"{""Gd"": ""FM""}","{""Gd"": ""FM"", ""U"": ""AFM-like""}","{""Gd"": 14.0, ""U"": -0.2}",OK,0,"{}",spin_001,FM,AFM-like,',
                '2,spin_002,OK,-99.5,TOTEN,vasp.out.2,OUTCAR,OK,8.1,6.0,3,"{""Gd"": 1, ""U"": 2}",1,2,8,"{""Gd"": ""AFM""}","{""Gd"": ""AFM"", ""U"": ""FM""}","{""Gd"": 0.0, ""U"": 8.1}",FAIL,2,"{""U"": 2}",spin_002,AFM,FM,',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "spin_energy_atom_moments.csv").write_text(
        "\n".join(
            [
                "run_index,run,atom,element,initial_moment,final_moment,delta,changed,mag_class,physics_ok,physics_target,physics_delta,energy_eV,mag_status",
                "1,spin_001,1,Gd,7.0,7.1,0.1,false,Gd-like,true,7,0.1,-100.0,OK",
                "1,spin_001,2,U,2.0,-2.1,-4.1,true,U4-like,true,-2,0.1,-100.0,OK",
                "2,spin_002,2,U,2.0,0.2,-1.8,true,other,false,2,1.8,-99.5,OK",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "spin_energy_report.md").write_text("# VASP Spin-Energy Report\n", encoding="utf-8")
    (root / "spin_index.csv").write_text(
        "\n".join(
            [
                "run_dir,name,dopant_mode,host_mode,moments_by_atom",
                'spin_001,spin_001,FM,AFM-like,"[{""atom"": 1, ""element"": ""Gd"", ""magmom"": 7.0}, {""atom"": 2, ""element"": ""U"", ""magmom"": -2.0}]"',
                'spin_002,spin_002,AFM,FM,"[{""atom"": 1, ""element"": ""Gd"", ""magmom"": -7.0}, {""atom"": 2, ""element"": ""U"", ""magmom"": 2.0}]"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_paper_draft_scans_and_appends(tmp_path: Path) -> None:
    vasp = tmp_path / "vasp"
    lammps = tmp_path / "md"
    calphad = tmp_path / "calphad"
    write_vasp_run(vasp)
    write_lammps_run(lammps)
    write_calphad_run(calphad)
    document = tmp_path / "paper" / "working.md"
    evidence = tmp_path / "paper" / "evidence.json"

    paper_draft.main(
        [
            "--used",
            "DFT",
            "MD",
            "CALPHAD",
            "--run",
            str(vasp),
            "--run",
            str(lammps),
            "--run",
            str(calphad),
            "--document",
            str(document),
            "--evidence-json",
            str(evidence),
            "--title",
            "Methods seed",
            "--material",
            "UO2",
        ]
    )

    text = document.read_text(encoding="utf-8")
    assert "Methods seed" in text
    assert "Electronic-structure calculations" in text
    assert "ENCUT=520" in text
    assert "VASP executable reported vasp.6.4.3" in text
    assert "PAW_PBE U" in text
    assert "final DFT energy -25.125" in text
    assert "MD thermo summary" in text
    assert "phase_count=1" in text

    parsed = json.loads(evidence.read_text(encoding="utf-8"))
    assert parsed[0]["detected_modules"] == ["DFT"]
    assert "dft_outcar" in parsed[0]["facts"]


def test_paper_draft_style_note_includes_format_rules(tmp_path: Path) -> None:
    vasp = tmp_path / "vasp"
    md = tmp_path / "md"
    write_vasp_run(vasp)
    write_lammps_run(md)
    document = tmp_path / "draft.md"

    paper_draft.main(
        [
            "--used",
            "DFT",
            "MD",
            "MLIP",
            "--run",
            str(vasp),
            "--run",
            str(md),
            "--document",
            str(document),
            "--mode",
            "overwrite",
            "--style-note",
        ]
    )

    text = document.read_text(encoding="utf-8")
    assert "### Manuscript Format Rules" in text
    assert "Use a layered Methods flow" in text
    assert "DFT: State the physical target" in text
    assert "MD: Report potential/model source" in text
    assert "MLIP: Report training-data provenance" in text


def test_paper_draft_describes_transport_and_writes_llm_metadata(tmp_path: Path) -> None:
    transport = tmp_path / "transport"
    write_transport_run(transport)
    document = tmp_path / "draft.md"
    evidence = tmp_path / "evidence.json"
    metadata = tmp_path / "llm_metadata.json"

    paper_draft.main(
        [
            "--used",
            "transport",
            "--run",
            str(transport),
            "--document",
            str(document),
            "--evidence-json",
            str(evidence),
            "--llm-metadata-json",
            str(metadata),
            "--mode",
            "overwrite",
            "--style-note",
            "--title",
            "Thermal transport draft",
            "--material",
            "UO2",
        ]
    )

    text = document.read_text(encoding="utf-8")
    assert "Requested modules: TRANSPORT" in text
    assert "Thermal-conductivity calculations" in text
    assert "Green-Kubo" in text
    assert "reverse NEMD" in text
    assert "timestep_ps=0.0005" in text
    assert "correlation_time_ps=5" in text
    assert "replicate=1x1x3" in text
    assert "Green-Kubo validation reported T=300 K k=5.9 W/m/K status=ok" in text
    assert "reverse NEMD validation reported T=1200 K k=2.605 W/m/K status=warn" in text

    parsed = json.loads(evidence.read_text(encoding="utf-8"))
    assert parsed[0]["detected_modules"] == ["MD", "TRANSPORT"]
    assert parsed[0]["facts"]["gk_config"]["settings"]["heat_flux_suffix"] == "kk"
    assert parsed[0]["facts"]["rnemd_config"]["settings"]["direction"] == "z"

    packet = json.loads(metadata.read_text(encoding="utf-8"))
    assert packet["schema"] == "atomi.paper_draft.llm_metadata.v1"
    assert "UN_thermal_transport_style" in packet["paper_lessons"]
    assert "TRANSPORT" in packet["extraction_targets"]
    assert packet["runs"][0]["facts"]["gk_plan"]["runtime_estimate"]["array_limit"] == 6


def test_paper_draft_reads_gzipped_outcar(tmp_path: Path) -> None:
    vasp = tmp_path / "vasp_gz"
    write_vasp_run(vasp)
    outcar_text = (vasp / "OUTCAR").read_text(encoding="utf-8")
    (vasp / "OUTCAR").unlink()
    with gzip.open(vasp / "OUTCAR.gz", "wt", encoding="utf-8") as handle:
        handle.write(outcar_text)

    evidence = paper_draft.scan_run(vasp, ["DFT"])

    assert evidence.files["outcar"] == ["OUTCAR.gz"]
    assert evidence.facts["dft_outcar"]["final_energy_eV"] == -25.125


def test_paper_draft_describes_vasp_defect_candidate_generation(tmp_path: Path) -> None:
    prep = tmp_path / "defect_prep"
    write_defect_cloud_run(prep)
    document = tmp_path / "draft.md"
    evidence = tmp_path / "evidence.json"

    paper_draft.main(
        [
            "--used",
            "defect-cloud",
            "DFT",
            "--run",
            str(prep),
            "--document",
            str(document),
            "--evidence-json",
            str(evidence),
            "--mode",
            "overwrite",
            "--no-style-note",
            "--title",
            "Defect candidate preparation",
            "--material",
            "(Gd,U)O2-x",
        ]
    )

    text = document.read_text(encoding="utf-8")
    assert "Requested modules: VASP_PREP, DFT" in text
    assert "Defect-seed and candidate electronic-structure folders" in text
    assert "2 seed motifs" in text
    assert "16 candidate VASP folders" in text
    assert "bias_species=O" in text
    assert "array-run index runlist.txt with 2 entries" in text

    parsed = json.loads(evidence.read_text(encoding="utf-8"))
    assert parsed[0]["detected_modules"] == ["VASP_PREP"]
    assert parsed[0]["facts"]["defect_cloud_summary"]["family_totals"]["base"] == 2


def test_paper_draft_uses_private_hpc_config_for_methods(tmp_path: Path) -> None:
    vasp = tmp_path / "vasp"
    write_vasp_run(vasp)
    config = tmp_path / "atomi_hpc_config.kit.local.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "site": "KIT / bwHPC",
                "profiles": {
                    "vasp_cpu": {
                        "modules": ["devel/python/3.11.4", "chem/vasp/6.2.1"],
                        "executable_candidates": ["vasp_std"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    document = tmp_path / "draft.md"

    paper_draft.main(
        [
            "--used",
            "DFT",
            "--run",
            str(vasp),
            "--hpc-config",
            str(config),
            "--document",
            str(document),
            "--mode",
            "overwrite",
        ]
    )

    text = document.read_text(encoding="utf-8")
    assert "Site-specific runtime information" in text
    assert "KIT / bwHPC" in text
    assert "chem/vasp/6.2.1" in text
    assert "vasp_std" in text


def test_paper_draft_describes_vasp_spin_report(tmp_path: Path) -> None:
    spin = tmp_path / "spin"
    write_spin_report_run(spin)
    document = tmp_path / "spin_draft.md"
    evidence = tmp_path / "spin_evidence.json"

    paper_draft.main(
        [
            "--used",
            "spin-report",
            "--run",
            str(spin),
            "--document",
            str(document),
            "--evidence-json",
            str(evidence),
            "--mode",
            "overwrite",
            "--no-style-note",
            "--title",
            "Spin screening",
            "--material",
            "(Gd,U)O2-x",
        ]
    )

    text = document.read_text(encoding="utf-8")
    assert "Requested modules: VASP_SPIN" in text
    assert "Spin-configuration screening was performed" in text
    assert "The spin-generation index records" in text
    assert "2 indexed spin configurations" in text
    assert "initial element moment values Gd=-7, 7; U=-2, 2" in text
    assert "physics-guard counts OK=1; FAIL=1" in text
    assert "spin-screening table" in text
    assert "spin-generation index contained 2 generated spin inputs" in text
    assert "lowest parsed run `spin_001`" in text
    assert "atom-resolved moment table showed 2 atom-level moment changes" in text

    parsed = json.loads(evidence.read_text(encoding="utf-8"))
    assert parsed[0]["detected_modules"] == ["DFT", "VASP_SPIN"]
    assert parsed[0]["facts"]["vasp_spin_summary"]["best"]["run"] == "spin_001"
    assert parsed[0]["facts"]["vasp_spin_index"]["element_moment_values"]["Gd"] == [-7.0, 7.0]
    assert parsed[0]["facts"]["vasp_spin_atoms"]["physics_bad_by_element"]["U"] == 1


def test_paper_draft_detects_cp2k_reactive_aimd_context(tmp_path: Path) -> None:
    run = tmp_path / "ga_cl4"
    write_cp2k_reaction_run(run)
    document = tmp_path / "aimd_reaction.md"
    evidence = tmp_path / "aimd_reaction.json"

    paper_draft.main(
        [
            "--used",
            "GaCl4 water-assisted-dissociation stability-constant",
            "--run",
            str(run),
            "--document",
            str(document),
            "--evidence-json",
            str(evidence),
            "--mode",
            "overwrite",
            "--no-style-note",
            "--title",
            "Ga complex ligand exchange",
        ]
    )

    text = document.read_text(encoding="utf-8")
    assert "Requested modules: AIMD_REACTION" in text
    assert "Reactive AIMD windows were treated as a constrained ligand-exchange workflow" in text
    assert "collective variables" in text
    assert "The CP2K ligand-exchange summary reported" in text
    assert "water-assisted exchange evidence detected" in text
    assert "Check before manuscript use: Requested module" not in text

    parsed = json.loads(evidence.read_text(encoding="utf-8"))
    assert parsed[0]["detected_modules"] == ["AIMD", "AIMD_REACTION"]
    assert parsed[0]["facts"]["cp2k_input"]["colvars"][0]["atoms"] == [1, 7]
    assert parsed[0]["facts"]["cp2k_bond_summary"]["tracked_tail_mean_A"]["Cl"] == 3.2
    assert parsed[0]["facts"]["cp2k_bond_summary"]["water_assisted_exchange_evidence"] is True


def test_paper_draft_top_level_cli(tmp_path: Path) -> None:
    vasp = tmp_path / "vasp"
    write_vasp_run(vasp)
    document = tmp_path / "draft.md"

    atomi_main(
        [
            "paper-draft",
            "--used",
            "DFT",
            "--run",
            str(vasp),
            "--document",
            str(document),
            "--mode",
            "overwrite",
            "--no-style-note",
        ]
    )

    text = document.read_text(encoding="utf-8")
    assert "Requested modules: DFT" in text
    assert "Style Notes" not in text
    assert "UO2 test" in text


def test_normalize_modules_keeps_unknown_keyword() -> None:
    assert paper_draft.normalize_modules(["dft, mlip", "custom"]) == ["DFT", "MLIP", "CUSTOM"]


def test_normalize_modules_accepts_defect_cloud_alias() -> None:
    assert paper_draft.normalize_modules(["defect-cloud"]) == ["VASP_PREP"]


def test_normalize_modules_accepts_spin_report_alias() -> None:
    assert paper_draft.normalize_modules(["spin-report"]) == ["VASP_SPIN"]


def test_normalize_modules_accepts_reactive_aimd_aliases() -> None:
    assert paper_draft.normalize_modules(["GaCl4", "stability-constant"]) == ["AIMD_REACTION"]
