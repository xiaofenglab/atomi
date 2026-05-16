import argparse
import sys
from pathlib import Path

from atomi import __version__
from atomi.cli.vasp import extv
from atomi.core.doctor import main as doctor_main
from atomi.core.project import create_project
from atomi.core.scheduler import render_submit_script
from atomi.viz.cp2k import plot_cp2k, plot_cp2k_all
from atomi.viz.lammps import format_summary, plot_lammps_live, read_thermo_rows, summarize_thermo
from atomi.viz.mace import plot_mace_live
from atomi.viz.vasp_live import plot_vasp_live, plot_vasp_live4
from atomi.ml.mace.datasets import main as mace_build_dataset_main


SUPPORTED_CODES = ("vasp", "cp2k", "lammps", "turbomole", "molcas", "moose", "calphad")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atomi",
        description="HPC automation helpers for atomistic modeling.",
    )
    parser.add_argument("--version", action="version", version=f"atomi {__version__}")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    init_project = subparsers.add_parser(
        "init-project",
        help="Create a standardized calculation folder.",
    )
    init_project.add_argument("path", type=Path)
    init_project.add_argument("--code", choices=SUPPORTED_CODES, required=True)

    write_submit = subparsers.add_parser(
        "write-submit",
        help="Write a scheduler submission script from a reusable template.",
    )
    write_submit.add_argument("--scheduler", choices=("slurm", "pbs"), default="slurm")
    write_submit.add_argument("--profile", default="generic_cpu")
    write_submit.add_argument("--output", type=Path, default=Path("submit.sh"))
    write_submit.add_argument("--job-name", default=None)
    write_submit.add_argument(
        "--command",
        dest="run_command",
        default=None,
        help="Run command, for example: vasp_std > vasp.out",
    )

    inspect = subparsers.add_parser(
        "inspect",
        help="Print a quick summary of a calculation folder.",
    )
    inspect.add_argument("path", type=Path, nargs="?", default=Path("."))

    doctor = subparsers.add_parser(
        "doctor",
        help="Inspect this HPC environment and write an optional JSON config.",
    )
    doctor.add_argument("doctor_args", nargs=argparse.REMAINDER)

    vasp_live = subparsers.add_parser(
        "vasp-live",
        help="Live terminal plot for one VASP output file.",
    )
    vasp_live.add_argument("output_file", type=Path)
    vasp_live.add_argument("--window", type=int, default=100)
    vasp_live.add_argument("--interval", type=float, default=2.0)
    vasp_live.add_argument("--once", action="store_true", help="Draw once and exit.")

    vasp_live4 = subparsers.add_parser(
        "vasp-live4",
        help="Live terminal plot for one to four VASP output files.",
    )
    vasp_live4.add_argument("output_files", type=Path, nargs="+")
    vasp_live4.add_argument("--window", type=int, default=100)
    vasp_live4.add_argument("--interval", type=float, default=5.0)
    vasp_live4.add_argument("--once", action="store_true", help="Draw once and exit.")

    lammps_live = subparsers.add_parser(
        "lammps-live",
        help="Live terminal plot for a LAMMPS log file.",
    )
    lammps_live.add_argument("logfile", type=Path)
    lammps_live.add_argument("--window", type=int, default=40)
    lammps_live.add_argument("--interval", type=float, default=10.0)
    lammps_live.add_argument("--once", action="store_true", help="Draw once and exit.")
    lammps_live.add_argument(
        "--keep-going",
        action="store_true",
        help="Keep monitoring after a Loop time line appears.",
    )

    lammps_summary = subparsers.add_parser(
        "lammps-summary",
        help="Summarize LAMMPS thermo data from a log file.",
    )
    lammps_summary.add_argument("logfile", type=Path)
    lammps_summary.add_argument("--last-fraction", type=float, default=0.5)

    md_engine_init = subparsers.add_parser(
        "md-engine-init",
        help="Copy LAMMPS MD engine templates into a project directory.",
    )
    md_engine_init.add_argument("workflow_args", nargs=argparse.REMAINDER)

    md_engine = subparsers.add_parser(
        "md-engine",
        help="Run or resume the config-driven LAMMPS MD engine.",
    )
    md_engine.add_argument("workflow_args", nargs=argparse.REMAINDER)

    md_engine_array = subparsers.add_parser(
        "md-engine-array",
        help="Generate or submit a Slurm array for independent production MD stages.",
    )
    md_engine_array.add_argument("workflow_args", nargs=argparse.REMAINDER)

    lammps_md_init = subparsers.add_parser(
        "lammps-md-init",
        help="Compatibility alias for md-engine-init.",
    )
    lammps_md_init.add_argument("workflow_args", nargs=argparse.REMAINDER)

    lammps_md_workflow = subparsers.add_parser(
        "lammps-md-workflow",
        help="Compatibility alias for md-engine.",
    )
    lammps_md_workflow.add_argument("workflow_args", nargs=argparse.REMAINDER)

    lammps_md_array = subparsers.add_parser(
        "lammps-md-array",
        help="Compatibility alias for md-engine-array.",
    )
    lammps_md_array.add_argument("workflow_args", nargs=argparse.REMAINDER)

    lammps_postprocess = subparsers.add_parser(
        "lammps-postprocess",
        help="Postprocess one LAMMPS NPT thermo log with window diagnostics.",
    )
    lammps_postprocess.add_argument("postprocess_args", nargs=argparse.REMAINDER)

    for command_name in ("pdf_lammps", "pdf_lammps_series", "lammps-rdf-pdf", "lammps-total-scattering"):
        lammps_rdf_pdf = subparsers.add_parser(
            command_name,
            help="Compute RDF/PDF/S(Q)/F(Q) from LAMMPS trajectories.",
        )
        lammps_rdf_pdf.add_argument("rdf_pdf_args", nargs=argparse.REMAINDER)

    for command_name in ("pdf_md_compare", "pdf_md_reweight"):
        lammps_pdf_match = subparsers.add_parser(
            command_name,
            help="Compare or reweight MD-derived PDF/S(Q)/F(Q) against experiment.",
        )
        lammps_pdf_match.add_argument("pdf_match_args", nargs=argparse.REMAINDER)

    for command_name in ("thermo_lammps", "lammps-thermo-series"):
        lammps_thermo_series = subparsers.add_parser(
            command_name,
            help="Postprocess production LAMMPS NPT thermo series with UQ.",
        )
        lammps_thermo_series.add_argument("analysis_args", nargs=argparse.REMAINDER)

    cp2k_live = subparsers.add_parser(
        "cp2k-live",
        help="Auto-detect CP2K MD/GEO logs and launch the terminal monitor.",
    )
    cp2k_live.add_argument("logfile", type=Path)
    cp2k_live.add_argument("xyzfile", type=Path, nargs="?")
    cp2k_live.add_argument("--mode", choices=("auto", "md", "geo"), default="auto")
    cp2k_live.add_argument("--window", type=int, default=300)
    cp2k_live.add_argument("--refresh", type=int, default=15)
    cp2k_live.add_argument(
        "--track-atom",
        type=int,
        help="Track one 1-based trajectory atom as a metal-distance trace.",
    )

    cp2k_all = subparsers.add_parser(
        "cp2k-all",
        help="Full CP2K GEO convergence dashboard.",
    )
    cp2k_all.add_argument("logfile", type=Path)

    cp2k_build_acid_box = subparsers.add_parser(
        "cp2k-build-acid-box",
        help="Build acidified explicit-water CP2K AIMD boxes and starter inputs.",
    )
    cp2k_build_acid_box.add_argument("builder_args", nargs=argparse.REMAINDER)

    cp2k_geoopt_input = subparsers.add_parser(
        "cp2k-geoopt-input",
        help="Write staged CP2K GEO_OPT inputs with restart-aware MAX_ITER.",
    )
    cp2k_geoopt_input.add_argument("geoopt_args", nargs=argparse.REMAINDER)

    cp2k_extract_frames = subparsers.add_parser(
        "cp2k-extract-frames",
        help="Extract CP2K AIMD frames into QM, embedding, and point-charge files.",
    )
    cp2k_extract_frames.add_argument("extract_args", nargs=argparse.REMAINDER)

    cp2k_rotate_seed = subparsers.add_parser(
        "cp2k-rotate-seed",
        help="Rotate a metal-ligand XYZ seed before CP2K box building.",
    )
    cp2k_rotate_seed.add_argument("rotate_args", nargs=argparse.REMAINDER)

    cp2k_bond_analysis = subparsers.add_parser(
        "cp2k-bond-analysis",
        help="Analyze post-MD CP2K metal-ligand bond distances.",
    )
    cp2k_bond_analysis.add_argument("analysis_args", nargs=argparse.REMAINDER)

    cp2k_clean_run = subparsers.add_parser(
        "cp2k-clean-run",
        help="Clean CP2K AIMD run folders while preserving rerun records.",
    )
    cp2k_clean_run.add_argument("clean_args", nargs=argparse.REMAINDER)

    cp2k_water_entry = subparsers.add_parser(
        "cp2k-water-entry",
        help="Find water-entry trajectory seeds and write two-CV CP2K inputs.",
    )
    cp2k_water_entry.add_argument("water_entry_args", nargs=argparse.REMAINDER)

    cp2k_pymol_render = subparsers.add_parser(
        "cp2k-pymol-render",
        help="Generate and optionally run PyMOL AIMD render/movie scripts.",
    )
    cp2k_pymol_render.add_argument("render_args", nargs=argparse.REMAINDER)

    moose_doctor = subparsers.add_parser(
        "moose-doctor",
        help="Inspect MOOSE app executables and common MOOSE environment variables.",
    )
    moose_doctor.add_argument("moose_args", nargs=argparse.REMAINDER)

    moose_info = subparsers.add_parser(
        "moose-info",
        help="Print MOOSE profile information from the local HPC config.",
    )
    moose_info.add_argument("moose_args", nargs=argparse.REMAINDER)

    moose_smoke = subparsers.add_parser(
        "moose-smoke",
        help="Run a configured MOOSE executable --help smoke check.",
    )
    moose_smoke.add_argument("moose_args", nargs=argparse.REMAINDER)

    moose_write_submit = subparsers.add_parser(
        "moose-write-submit",
        help="Write a Slurm MOOSE submission script from local config.",
    )
    moose_write_submit.add_argument("moose_args", nargs=argparse.REMAINDER)

    moose_qha_md_material = subparsers.add_parser(
        "moose-qha-md-material",
        help="Export thermo_qha_md data as MOOSE material-property inputs.",
    )
    moose_qha_md_material.add_argument("moose_material_args", nargs=argparse.REMAINDER)

    moose_material_screen = subparsers.add_parser(
        "moose-material-screen",
        help="Screen which material inputs are present/missing for a MOOSE prediction.",
    )
    moose_material_screen.add_argument("moose_material_screen_args", nargs=argparse.REMAINDER)

    moose_material_source = subparsers.add_parser(
        "moose-material-source",
        help="Fetch or normalize external MOOSE material-property source data.",
    )
    moose_material_source.add_argument("moose_material_source_args", nargs=argparse.REMAINDER)

    moose_material_compare = subparsers.add_parser(
        "moose-material-compare",
        help="Compare MOOSE material-property CSVs and write plots/tables.",
    )
    moose_material_compare.add_argument("moose_material_compare_args", nargs=argparse.REMAINDER)

    moose_thermal_stress = subparsers.add_parser(
        "moose-thermal-stress",
        help="Write a material-driven cylindrical thermal-stress MOOSE input.",
    )
    moose_thermal_stress.add_argument("moose_workflow_args", nargs=argparse.REMAINDER)

    moose_uo2_thermal_stress = subparsers.add_parser(
        "moose-uo2-thermal-stress",
        help="Write a starter UO2 pellet thermal-stress MOOSE input.",
    )
    moose_uo2_thermal_stress.add_argument("moose_workflow_args", nargs=argparse.REMAINDER)

    calphad_doctor = subparsers.add_parser(
        "calphad-doctor",
        help="Inspect pycalphad availability and optional TDB database metadata.",
    )
    calphad_doctor.add_argument("calphad_args", nargs=argparse.REMAINDER)

    mace_live = subparsers.add_parser(
        "mace-live",
        help="Live terminal plot for MACE training logs.",
    )
    mace_live.add_argument("logfile", type=Path)
    mace_live.add_argument("--window", type=int, default=100)
    mace_live.add_argument("--refresh", type=int, default=5)

    vasp_outcar = subparsers.add_parser(
        "vasp-outcar",
        help="Quick VASP OUTCAR summary.",
    )
    vasp_outcar.add_argument("outcar_args", nargs=argparse.REMAINDER)

    vasp_check = subparsers.add_parser(
        "vasp-check",
        help="Check completion state for VASP array runs in runlist.txt.",
    )
    vasp_check.add_argument("check_args", nargs=argparse.REMAINDER)

    vasp_check_scf = subparsers.add_parser(
        "vasp-check-scf",
        help="Check final DAV SCF convergence for VASP array logs.",
    )
    vasp_check_scf.add_argument("check_args", nargs=argparse.REMAINDER)

    for command_name in ("checkeng", "vasp-energies", "vasp-energy-table"):
        vasp_energies = subparsers.add_parser(
            command_name,
            help="Print latest VASP energy table for array runs in runlist.txt.",
        )
        vasp_energies.add_argument("energy_args", nargs=argparse.REMAINDER)

    vasp_update_magmom = subparsers.add_parser(
        "vasp-update-magmom",
        help="Update INCAR MAGMOM from final OUTCAR moments for selected elements.",
    )
    vasp_update_magmom.add_argument("magmom_args", nargs=argparse.REMAINDER)

    vasp_phonopy_neareq = subparsers.add_parser(
        "vasp-phonopy-neareq",
        help="Prepare phonopy and MLIP near-equilibrium VASP datasets.",
    )
    vasp_phonopy_neareq.add_argument("phonopy_args", nargs=argparse.REMAINDER)

    vasp_phonopy_post = subparsers.add_parser(
        "vasp-phonopy-post",
        help="Generate phonopy thermal/DOS/band post-analysis scripts.",
    )
    vasp_phonopy_post.add_argument("phonopy_args", nargs=argparse.REMAINDER)

    vasp_phonopy_band_plot = subparsers.add_parser(
        "vasp-phonopy-band-plot",
        help="Plot phonopy band.yaml to a PNG.",
    )
    vasp_phonopy_band_plot.add_argument("phonopy_args", nargs=argparse.REMAINDER)

    vasp_prefail_candidates = subparsers.add_parser(
        "vasp-prefail-candidates",
        help="Extract prefail MD frames and prepare distorted VASP candidate runs.",
    )
    vasp_prefail_candidates.add_argument("prefail_args", nargs=argparse.REMAINDER)

    vasp_stress_force_candidates = subparsers.add_parser(
        "vasp-stress-force-candidates",
        help="Prepare stress/force VASP candidate runs from an equilibrium POSCAR.",
    )
    vasp_stress_force_candidates.add_argument("stress_force_args", nargs=argparse.REMAINDER)

    vasp_defect_candidates = subparsers.add_parser(
        "vasp-defect-candidates",
        help="Prepare vacancy/interstitial/Frenkel/Schottky VASP defect runs.",
    )
    vasp_defect_candidates.add_argument("defect_args", nargs=argparse.REMAINDER)

    vasp_md_snapshot_candidates = subparsers.add_parser(
        "vasp-md-snapshot-candidates",
        help="Harvest successful md-engine LAMMPS frames into VASP-ready folders.",
    )
    vasp_md_snapshot_candidates.add_argument("snapshot_args", nargs=argparse.REMAINDER)

    vasp_qha_summary = subparsers.add_parser(
        "vasp-qha-summary",
        help="Summarize VASP/phonopy QHA volume folders.",
    )
    vasp_qha_summary.add_argument("qha_args", nargs=argparse.REMAINDER)

    vasp_qha_run = subparsers.add_parser(
        "vasp-qha-run",
        help="Prepare e-v.dat and a phonopy-qha run script.",
    )
    vasp_qha_run.add_argument("qha_args", nargs=argparse.REMAINDER)

    for command_name in ("thermo_qha_md", "thermo_qha-md", "vasp-qha-md-compare"):
        vasp_qha_md_compare = subparsers.add_parser(
            command_name,
            help="Overlay VASP QHA and LAMMPS MD thermodynamic functions.",
        )
        vasp_qha_md_compare.add_argument("compare_args", nargs=argparse.REMAINDER)

    mace_build_dataset = subparsers.add_parser(
        "mace-build-dataset",
        help="Build adaptive MACE train/validation extxyz datasets.",
    )
    mace_build_dataset.add_argument("dataset_args", nargs=argparse.REMAINDER)

    mace_energy_outliers = subparsers.add_parser(
        "mace-energy-outliers",
        help="Find high energy-error outliers for a MACE model.",
    )
    mace_energy_outliers.add_argument("outlier_args", nargs=argparse.REMAINDER)

    mace_update_outliers = subparsers.add_parser(
        "mace-update-outliers",
        help="Remove outlier frames and optionally append rerun extxyz frames.",
    )
    mace_update_outliers.add_argument("update_args", nargs=argparse.REMAINDER)

    mace_check_extxyz = subparsers.add_parser(
        "mace-check-extxyz",
        help="Check extxyz labels, composition, histograms, and optional REF-key rewriting.",
    )
    mace_check_extxyz.add_argument("check_args", nargs=argparse.REMAINDER)

    mace_vasp2extxyz = subparsers.add_parser(
        "mace-vasp2extxyz",
        help="Collect VASP run directories into an extxyz dataset.",
    )
    mace_vasp2extxyz.add_argument("convert_args", nargs=argparse.REMAINDER)

    mace_convert_lammps = subparsers.add_parser(
        "mace-convert-lammps",
        help="Convert a MACE .model file to a LAMMPS .pt model.",
    )
    mace_convert_lammps.add_argument("convert_args", nargs=argparse.REMAINDER)

    return parser


def main(argv: list[str] | None = None) -> None:
    raw_args = sys.argv[1:] if argv is None else argv
    if raw_args and raw_args[0] == "mace-build-dataset":
        mace_build_dataset_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "mace-energy-outliers":
        from atomi.ml.mace.outliers import main as mace_energy_outliers_main

        mace_energy_outliers_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "mace-update-outliers":
        from atomi.ml.mace.update_outliers import main as mace_update_outliers_main

        mace_update_outliers_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "mace-check-extxyz":
        from atomi.ml.mace.check_extxyz import main as mace_check_extxyz_main

        mace_check_extxyz_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "mace-vasp2extxyz":
        from atomi.ml.mace.vasp2extxyz import main as mace_vasp2extxyz_main

        mace_vasp2extxyz_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "mace-convert-lammps":
        from atomi.ml.mace.convert import main as mace_convert_main

        mace_convert_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("md-engine-init", "lammps-md-init"):
        from atomi.lammps.workflow_cli import init_workflow

        init_workflow(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("md-engine", "lammps-md-workflow"):
        from atomi.lammps.workflow_cli import run_workflow

        run_workflow(raw_args[1:])
        return

    if raw_args and raw_args[0] in ("md-engine-array", "lammps-md-array"):
        from atomi.lammps.workflow_cli import production_array

        production_array(raw_args[1:])
        return
    if raw_args and raw_args[0] == "lammps-postprocess":
        from atomi.lammps.postprocess import main as lammps_postprocess_main

        lammps_postprocess_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("pdf_lammps", "pdf_lammps_series", "lammps-rdf-pdf", "lammps-total-scattering"):
        from atomi.lammps.rdf_pdf import main as lammps_rdf_pdf_main

        lammps_rdf_pdf_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "pdf_md_compare":
        from atomi.lammps.pdf_match import compare_main

        compare_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "pdf_md_reweight":
        from atomi.lammps.pdf_match import reweight_main

        reweight_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("thermo_lammps", "lammps-thermo-series"):
        from atomi.lammps.thermo_series import main as thermo_series_main

        thermo_series_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "doctor":
        doctor_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "cp2k-build-acid-box":
        from atomi.cp2k.acid_box import main as cp2k_build_acid_box_main

        cp2k_build_acid_box_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "cp2k-geoopt-input":
        from atomi.cp2k.geoopt_input import main as cp2k_geoopt_input_main

        cp2k_geoopt_input_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "cp2k-extract-frames":
        from atomi.cp2k.extract_frames import main as cp2k_extract_frames_main

        cp2k_extract_frames_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "cp2k-rotate-seed":
        from atomi.cp2k.rotate_seed import main as cp2k_rotate_seed_main

        cp2k_rotate_seed_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "cp2k-bond-analysis":
        from atomi.cp2k.bond_analysis import main as cp2k_bond_analysis_main

        cp2k_bond_analysis_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "cp2k-clean-run":
        from atomi.cp2k.clean_run import main as cp2k_clean_run_main

        cp2k_clean_run_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "cp2k-water-entry":
        from atomi.cp2k.water_entry import main as cp2k_water_entry_main

        cp2k_water_entry_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "cp2k-pymol-render":
        from atomi.cp2k.pymol_render import main as cp2k_pymol_render_main

        cp2k_pymol_render_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "moose-doctor":
        from atomi.moose.env import main as moose_doctor_main

        moose_doctor_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "moose-info":
        from atomi.moose.workflow import info_main as moose_info_main

        moose_info_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "moose-smoke":
        from atomi.moose.workflow import smoke_main as moose_smoke_main

        moose_smoke_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "moose-write-submit":
        from atomi.moose.workflow import write_submit_main as moose_write_submit_main

        moose_write_submit_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "moose-qha-md-material":
        from atomi.moose.material_export import main as moose_qha_md_material_main

        moose_qha_md_material_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "moose-material-screen":
        from atomi.moose.material_sources import screen_main as moose_material_screen_main

        moose_material_screen_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "moose-material-source":
        from atomi.moose.material_sources import source_main as moose_material_source_main

        moose_material_source_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "moose-material-compare":
        from atomi.moose.material_sources import compare_main as moose_material_compare_main

        moose_material_compare_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "moose-thermal-stress":
        from atomi.moose.workflow import thermal_stress_main

        thermal_stress_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "moose-uo2-thermal-stress":
        from atomi.moose.workflow import uo2_thermal_stress_main

        uo2_thermal_stress_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "calphad-doctor":
        from atomi.calphad.env import main as calphad_doctor_main

        calphad_doctor_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-outcar":
        extv(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-check":
        from atomi.vasp.checks import checkvasp

        checkvasp(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-check-scf":
        from atomi.vasp.checks import checkscf

        checkscf(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("checkeng", "vasp-energies", "vasp-energy-table"):
        from atomi.vasp.checks import vasp_energies

        vasp_energies(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-update-magmom":
        from atomi.vasp.magmom import main as vasp_update_magmom_main

        vasp_update_magmom_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-phonopy-neareq":
        from atomi.vasp.phonopy_neareq import main as vasp_phonopy_neareq_main

        vasp_phonopy_neareq_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-phonopy-post":
        from atomi.vasp.phonopy_post import main as vasp_phonopy_post_main

        vasp_phonopy_post_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-phonopy-band-plot":
        from atomi.vasp.phonopy_band import main as vasp_phonopy_band_plot_main

        vasp_phonopy_band_plot_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-prefail-candidates":
        from atomi.vasp.prefail import main as vasp_prefail_candidates_main

        vasp_prefail_candidates_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-stress-force-candidates":
        from atomi.vasp.stress_force import main as vasp_stress_force_candidates_main

        vasp_stress_force_candidates_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-defect-candidates":
        from atomi.vasp.defects import main as vasp_defect_candidates_main

        vasp_defect_candidates_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-md-snapshot-candidates":
        from atomi.vasp.md_snapshots import main as vasp_md_snapshot_candidates_main

        vasp_md_snapshot_candidates_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-qha-summary":
        from atomi.vasp.qha_summary import main as vasp_qha_summary_main

        vasp_qha_summary_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "vasp-qha-run":
        from atomi.vasp.qha_run import main as vasp_qha_run_main

        vasp_qha_run_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("thermo_qha_md", "thermo_qha-md", "vasp-qha-md-compare"):
        from atomi.vasp.qha_md_compare import main as vasp_qha_md_compare_main

        vasp_qha_md_compare_main(raw_args[1:])
        return

    parser = build_parser()
    args = parser.parse_args(raw_args)

    if args.subcommand == "init-project":
        created = create_project(path=args.path, code=args.code)
        print(f"Created {created}")
        return

    if args.subcommand == "write-submit":
        script = render_submit_script(
            scheduler=args.scheduler,
            profile_name=args.profile,
            job_name=args.job_name or args.output.parent.resolve().name,
            command=args.run_command or "echo 'Replace this command with your executable'",
        )
        args.output.write_text(script, encoding="utf-8")
        args.output.chmod(0o755)
        print(f"Wrote {args.output}")
        return

    if args.subcommand == "inspect":
        files = sorted(item.name for item in args.path.iterdir())
        print(f"Path: {args.path.resolve()}")
        print("Files:")
        for name in files:
            print(f"  {name}")
        return

    if args.subcommand == "doctor":
        doctor_main(args.doctor_args)
        return

    if args.subcommand == "vasp-live":
        plot_vasp_live(
            output_file=args.output_file,
            window=args.window,
            interval=args.interval,
            once=args.once,
        )
        return

    if args.subcommand == "vasp-live4":
        plot_vasp_live4(
            output_files=args.output_files,
            window=args.window,
            interval=args.interval,
            once=args.once,
        )
        return

    if args.subcommand == "lammps-live":
        plot_lammps_live(
            logfile=args.logfile,
            window=args.window,
            interval=args.interval,
            once=args.once,
            stop_on_finish=not args.keep_going,
        )
        return

    if args.subcommand == "lammps-summary":
        rows = read_thermo_rows(args.logfile)
        summary = summarize_thermo(rows, last_fraction=args.last_fraction)
        print(format_summary(summary))
        return

    if args.subcommand in ("md-engine-init", "lammps-md-init"):
        from atomi.lammps.workflow_cli import init_workflow

        init_workflow(args.workflow_args)
        return

    if args.subcommand in ("md-engine", "lammps-md-workflow"):
        from atomi.lammps.workflow_cli import run_workflow

        run_workflow(args.workflow_args)
        return

    if args.subcommand in ("md-engine-array", "lammps-md-array"):
        from atomi.lammps.workflow_cli import production_array

        production_array(args.workflow_args)
        return

    if args.subcommand == "lammps-postprocess":
        from atomi.lammps.postprocess import main as lammps_postprocess_main

        lammps_postprocess_main(args.postprocess_args)
        return

    if args.subcommand in ("pdf_lammps", "pdf_lammps_series", "lammps-rdf-pdf", "lammps-total-scattering"):
        from atomi.lammps.rdf_pdf import main as lammps_rdf_pdf_main

        lammps_rdf_pdf_main(args.rdf_pdf_args)
        return

    if args.subcommand == "pdf_md_compare":
        from atomi.lammps.pdf_match import compare_main

        compare_main(args.pdf_match_args)
        return

    if args.subcommand == "pdf_md_reweight":
        from atomi.lammps.pdf_match import reweight_main

        reweight_main(args.pdf_match_args)
        return

    if args.subcommand in ("thermo_lammps", "lammps-thermo-series"):
        from atomi.lammps.thermo_series import main as thermo_series_main

        thermo_series_main(args.analysis_args)
        return

    if args.subcommand == "cp2k-live":
        plot_cp2k(
            logfile=args.logfile,
            xyzfile=args.xyzfile,
            mode=args.mode,
            window=args.window,
            refresh=args.refresh,
            track_atom=args.track_atom,
        )
        return

    if args.subcommand == "cp2k-all":
        plot_cp2k_all(args.logfile)
        return

    if args.subcommand == "cp2k-build-acid-box":
        from atomi.cp2k.acid_box import main as cp2k_build_acid_box_main

        cp2k_build_acid_box_main(args.builder_args)
        return

    if args.subcommand == "cp2k-geoopt-input":
        from atomi.cp2k.geoopt_input import main as cp2k_geoopt_input_main

        cp2k_geoopt_input_main(args.geoopt_args)
        return

    if args.subcommand == "cp2k-extract-frames":
        from atomi.cp2k.extract_frames import main as cp2k_extract_frames_main

        cp2k_extract_frames_main(args.extract_args)
        return

    if args.subcommand == "cp2k-rotate-seed":
        from atomi.cp2k.rotate_seed import main as cp2k_rotate_seed_main

        cp2k_rotate_seed_main(args.rotate_args)
        return

    if args.subcommand == "cp2k-bond-analysis":
        from atomi.cp2k.bond_analysis import main as cp2k_bond_analysis_main

        cp2k_bond_analysis_main(args.analysis_args)
        return

    if args.subcommand == "cp2k-clean-run":
        from atomi.cp2k.clean_run import main as cp2k_clean_run_main

        cp2k_clean_run_main(args.clean_args)
        return

    if args.subcommand == "cp2k-water-entry":
        from atomi.cp2k.water_entry import main as cp2k_water_entry_main

        cp2k_water_entry_main(args.water_entry_args)
        return

    if args.subcommand == "cp2k-pymol-render":
        from atomi.cp2k.pymol_render import main as cp2k_pymol_render_main

        cp2k_pymol_render_main(args.render_args)
        return

    if args.subcommand == "moose-doctor":
        from atomi.moose.env import main as moose_doctor_main

        moose_doctor_main(args.moose_args)
        return

    if args.subcommand == "moose-info":
        from atomi.moose.workflow import info_main as moose_info_main

        moose_info_main(args.moose_args)
        return

    if args.subcommand == "moose-smoke":
        from atomi.moose.workflow import smoke_main as moose_smoke_main

        moose_smoke_main(args.moose_args)
        return

    if args.subcommand == "moose-write-submit":
        from atomi.moose.workflow import write_submit_main as moose_write_submit_main

        moose_write_submit_main(args.moose_args)
        return

    if args.subcommand == "moose-qha-md-material":
        from atomi.moose.material_export import main as moose_qha_md_material_main

        moose_qha_md_material_main(args.moose_material_args)
        return

    if args.subcommand == "moose-material-screen":
        from atomi.moose.material_sources import screen_main as moose_material_screen_main

        moose_material_screen_main(args.moose_material_screen_args)
        return

    if args.subcommand == "moose-material-source":
        from atomi.moose.material_sources import source_main as moose_material_source_main

        moose_material_source_main(args.moose_material_source_args)
        return

    if args.subcommand == "moose-material-compare":
        from atomi.moose.material_sources import compare_main as moose_material_compare_main

        moose_material_compare_main(args.moose_material_compare_args)
        return

    if args.subcommand == "moose-thermal-stress":
        from atomi.moose.workflow import thermal_stress_main

        thermal_stress_main(args.moose_workflow_args)
        return

    if args.subcommand == "moose-uo2-thermal-stress":
        from atomi.moose.workflow import uo2_thermal_stress_main

        uo2_thermal_stress_main(args.moose_workflow_args)
        return

    if args.subcommand == "calphad-doctor":
        from atomi.calphad.env import main as calphad_doctor_main

        calphad_doctor_main(args.calphad_args)
        return

    if args.subcommand == "mace-live":
        plot_mace_live(logfile=args.logfile, window=args.window, refresh=args.refresh)
        return

    if args.subcommand == "vasp-outcar":
        extv(args.outcar_args)
        return

    if args.subcommand == "vasp-check":
        from atomi.vasp.checks import checkvasp

        checkvasp(args.check_args)
        return

    if args.subcommand == "vasp-check-scf":
        from atomi.vasp.checks import checkscf

        checkscf(args.check_args)
        return

    if args.subcommand in ("checkeng", "vasp-energies", "vasp-energy-table"):
        from atomi.vasp.checks import vasp_energies

        vasp_energies(args.energy_args)
        return

    if args.subcommand == "vasp-update-magmom":
        from atomi.vasp.magmom import main as vasp_update_magmom_main

        vasp_update_magmom_main(args.magmom_args)
        return

    if args.subcommand == "vasp-phonopy-neareq":
        from atomi.vasp.phonopy_neareq import main as vasp_phonopy_neareq_main

        vasp_phonopy_neareq_main(args.phonopy_args)
        return

    if args.subcommand == "vasp-phonopy-post":
        from atomi.vasp.phonopy_post import main as vasp_phonopy_post_main

        vasp_phonopy_post_main(args.phonopy_args)
        return

    if args.subcommand == "vasp-phonopy-band-plot":
        from atomi.vasp.phonopy_band import main as vasp_phonopy_band_plot_main

        vasp_phonopy_band_plot_main(args.phonopy_args)
        return

    if args.subcommand == "vasp-prefail-candidates":
        from atomi.vasp.prefail import main as vasp_prefail_candidates_main

        vasp_prefail_candidates_main(args.prefail_args)
        return

    if args.subcommand == "vasp-stress-force-candidates":
        from atomi.vasp.stress_force import main as vasp_stress_force_candidates_main

        vasp_stress_force_candidates_main(args.stress_force_args)
        return

    if args.subcommand == "mace-build-dataset":
        mace_build_dataset_main(args.dataset_args)
        return

    if args.subcommand == "vasp-qha-summary":
        from atomi.vasp.qha_summary import main as vasp_qha_summary_main

        vasp_qha_summary_main(args.qha_args)
        return

    if args.subcommand == "vasp-qha-run":
        from atomi.vasp.qha_run import main as vasp_qha_run_main

        vasp_qha_run_main(args.qha_args)
        return

    if args.subcommand in ("thermo_qha_md", "thermo_qha-md", "vasp-qha-md-compare"):
        from atomi.vasp.qha_md_compare import main as vasp_qha_md_compare_main

        vasp_qha_md_compare_main(args.compare_args)
        return

    if args.subcommand == "mace-energy-outliers":
        from atomi.ml.mace.outliers import main as mace_energy_outliers_main

        mace_energy_outliers_main(args.outlier_args)
        return

    if args.subcommand == "mace-update-outliers":
        from atomi.ml.mace.update_outliers import main as mace_update_outliers_main

        mace_update_outliers_main(args.update_args)
        return

    if args.subcommand == "mace-check-extxyz":
        from atomi.ml.mace.check_extxyz import main as mace_check_extxyz_main

        mace_check_extxyz_main(args.check_args)
        return

    if args.subcommand == "mace-vasp2extxyz":
        from atomi.ml.mace.vasp2extxyz import main as mace_vasp2extxyz_main

        mace_vasp2extxyz_main(args.convert_args)
        return

    if args.subcommand == "mace-convert-lammps":
        from atomi.ml.mace.convert import main as mace_convert_main

        mace_convert_main(args.convert_args)
        return


if __name__ == "__main__":
    main()
