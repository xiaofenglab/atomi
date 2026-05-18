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

    for command_name in ("pdfgetx3_status", "pdfgetx3-status"):
        pdfgetx3_status = subparsers.add_parser(
            command_name,
            help="Inspect PDFGetX3 executable and configured external environment.",
        )
        pdfgetx3_status.add_argument("pdfgetx3_status_args", nargs=argparse.REMAINDER)

    for command_name in ("pdf_md_compare", "pdf_md_reweight"):
        lammps_pdf_match = subparsers.add_parser(
            command_name,
            help="Compare or reweight MD-derived PDF/S(Q)/F(Q) against experiment.",
        )
        lammps_pdf_match.add_argument("pdf_match_args", nargs=argparse.REMAINDER)

    for command_name in ("xafs_lammps_prepare", "xafs_cp2k_prepare", "xafs_larch_run", "xafs_md_compare", "xafs_status"):
        xafs_command = subparsers.add_parser(
            command_name,
            help="Prepare, run, or compare MD-ensemble XAFS through Larch/FEFF.",
        )
        xafs_command.add_argument("xafs_args", nargs=argparse.REMAINDER)

    for command_name in ("thermo_lammps", "lammps-thermo-series"):
        lammps_thermo_series = subparsers.add_parser(
            command_name,
            help="Postprocess production LAMMPS NPT thermo series with UQ.",
        )
        lammps_thermo_series.add_argument("analysis_args", nargs=argparse.REMAINDER)

    for command_name in ("thermal_k_lammps", "thermal-k-lammps"):
        thermal_k_lammps = subparsers.add_parser(
            command_name,
            help="Collect thermal conductivity from elastic estimates or MD Green-Kubo tables.",
        )
        thermal_k_lammps.add_argument("thermal_k_args", nargs=argparse.REMAINDER)

    elastic_lammps = subparsers.add_parser(
        "elastic_lammps",
        help="Prepare and fit finite-temperature LAMMPS elastic tensors.",
    )
    elastic_lammps.add_argument("elastic_args", nargs=argparse.REMAINDER)

    for command_name in ("elastic_viz", "elastic-viz"):
        elastic_viz = subparsers.add_parser(
            command_name,
            help="Visualize VASP/MD elastic tensors and derive thermophysical properties.",
        )
        elastic_viz.add_argument("elastic_viz_args", nargs=argparse.REMAINDER)

    for command_name in ("elate_status", "elastic_viz_status"):
        elastic_status = subparsers.add_parser(
            command_name,
            help="Check optional ELATE/native elastic visualization availability.",
        )
        elastic_status.add_argument("elastic_status_args", nargs=argparse.REMAINDER)

    for command_name in ("vasp_elastic", "vasp-elastic"):
        vasp_elastic = subparsers.add_parser(
            command_name,
            help="Prepare and analyze VASP static elastic tensors.",
        )
        vasp_elastic.add_argument("vasp_elastic_args", nargs=argparse.REMAINDER)

    for command_name in (
        "elastic_vasp_md_compare",
        "elastic-vasp-md-compare",
        "elastic_qha_md_compare",
    ):
        elastic_compare = subparsers.add_parser(
            command_name,
            help="Compare VASP/static or QHA elastic data with LAMMPS MD elasticity.",
        )
        elastic_compare.add_argument("elastic_compare_args", nargs=argparse.REMAINDER)

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

    for command_name in ("moose-doctor", "moose_status", "moose-status"):
        moose_doctor = subparsers.add_parser(
            command_name,
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

    for command_name in ("moose-elastic-export", "moose_elastic_export"):
        moose_elastic_export = subparsers.add_parser(
            command_name,
            help="Export VASP/MD elastic tensors as MOOSE-friendly tensor tables.",
        )
        moose_elastic_export.add_argument("moose_elastic_args", nargs=argparse.REMAINDER)

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

    for command_name in ("calphad-doctor", "calphad_status", "calphad-status"):
        calphad_doctor = subparsers.add_parser(
            command_name,
            help="Inspect pycalphad availability and optional TDB database metadata.",
        )
        calphad_doctor.add_argument("calphad_args", nargs=argparse.REMAINDER)

    for command_name in ("calphad_export", "calphad-export"):
        calphad_export = subparsers.add_parser(
            command_name,
            help="Export CALPHAD property tables and MOOSE phase-field templates.",
        )
        calphad_export.add_argument("calphad_export_args", nargs=argparse.REMAINDER)

    for command_name in ("paper-draft", "report-draft", "atomi-paper-draft"):
        paper_draft = subparsers.add_parser(
            command_name,
            help="Append Methods and brief Results draft text from completed run folders.",
        )
        paper_draft.add_argument("paper_draft_args", nargs=argparse.REMAINDER)

    zentropy_commands = (
        "zentropy_motif_db",
        "zentropy-motif-db",
        "defect_motif_db",
        "defect-motif-db",
    )
    for command_name in zentropy_commands:
        zentropy_motif_db = subparsers.add_parser(
            command_name,
            help="Index and export defect motifs for zentropy-guided thermodynamics.",
        )
        zentropy_motif_db.add_argument("zentropy_args", nargs=argparse.REMAINDER)

    motif_path_commands = ("midx", "motifpath", "motifpaths", "motif-paths", "zentropy-motif-paths")
    for command_name in motif_path_commands:
        motif_paths = subparsers.add_parser(
            command_name,
            help="Build a path-index CSV for zentropy auto-metadata scans.",
        )
        motif_paths.add_argument("motif_path_args", nargs=argparse.REMAINDER)

    zentropy_status_commands = ("zentropy_status", "zentropy-status")
    for command_name in zentropy_status_commands:
        zentropy_status = subparsers.add_parser(
            command_name,
            help="Report active/external optional pyzentropy runtime availability.",
        )
        zentropy_status.add_argument("zentropy_status_args", nargs=argparse.REMAINDER)

    zentropy_workflow_commands = ("zentropy_workflow", "zentropy-workflow")
    for command_name in zentropy_workflow_commands:
        zentropy_workflow = subparsers.add_parser(
            command_name,
            help="Create or inspect staged zentropy-guided defect thermodynamics workflows.",
        )
        zentropy_workflow.add_argument("zentropy_workflow_args", nargs=argparse.REMAINDER)

    zentropy_free_energy_commands = ("zentropy_free_energy", "zentropy-free-energy")
    for command_name in zentropy_free_energy_commands:
        zentropy_free_energy = subparsers.add_parser(
            command_name,
            help="Assemble motif-resolved free-energy tables for zentropy solves.",
        )
        zentropy_free_energy.add_argument("zentropy_free_energy_args", nargs=argparse.REMAINDER)

    zentropy_solve_commands = ("zentropy_solve", "zentropy-solve")
    for command_name in zentropy_solve_commands:
        zentropy_solve = subparsers.add_parser(
            command_name,
            help="Solve discrete motif probabilities from G_i(T) and degeneracy.",
        )
        zentropy_solve.add_argument("zentropy_solve_args", nargs=argparse.REMAINDER)

    zentropy_export_commands = ("zentropy_export", "zentropy-export")
    for command_name in zentropy_export_commands:
        zentropy_export = subparsers.add_parser(
            command_name,
            help="Export zentropy thermo and population tables for downstream models.",
        )
        zentropy_export.add_argument("zentropy_export_args", nargs=argparse.REMAINDER)

    zentropy_active_learning_commands = ("zentropy_active_learning", "zentropy-active-learning")
    for command_name in zentropy_active_learning_commands:
        zentropy_active_learning = subparsers.add_parser(
            command_name,
            help="Rank zentropy motifs for follow-up DFT or MLIP sampling.",
        )
        zentropy_active_learning.add_argument("zentropy_active_learning_args", nargs=argparse.REMAINDER)

    solid_solution_scan_commands = ("mlip_solid_solution_scan", "mlip-solid-solution-scan")
    for command_name in solid_solution_scan_commands:
        solid_solution_scan = subparsers.add_parser(
            command_name,
            help="Plan compact MLIP solid-solution scans over composition and defect families.",
        )
        solid_solution_scan.add_argument("solid_solution_scan_args", nargs=argparse.REMAINDER)

    motif_cluster_commands = ("motif_cluster", "motif-cluster", "zentropy_motif_cluster", "zentropy-motif-cluster")
    for command_name in motif_cluster_commands:
        motif_cluster = subparsers.add_parser(
            command_name,
            help="Cluster defect motifs and select low-energy representatives.",
        )
        motif_cluster.add_argument("motif_cluster_args", nargs=argparse.REMAINDER)

    defect_thermo_commands = ("defect_thermo_export", "defect-thermo-export")
    for command_name in defect_thermo_commands:
        defect_thermo = subparsers.add_parser(
            command_name,
            help="Export defect motif energetics for zentropy/CALPHAD/MOOSE coupling.",
        )
        defect_thermo.add_argument("defect_thermo_args", nargs=argparse.REMAINDER)

    sd_dd_commands = ("sd-dd-thermo", "sd_dd_thermo", "defect-sd-dd", "defect-chem")
    for command_name in sd_dd_commands:
        sd_dd = subparsers.add_parser(
            command_name,
            help="Run dilute single-defect/double-defect thermodynamics cross-checks.",
        )
        sd_dd.add_argument("sd_dd_args", nargs=argparse.REMAINDER)

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

    for command_name in ("vasp-spin-report", "spin-report"):
        spin_report = subparsers.add_parser(
            command_name,
            help="Correlate VASP final magnetic moments with run energies.",
        )
        spin_report.add_argument("spin_report_args", nargs=argparse.REMAINDER)

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

    vasp_defect_cloud = subparsers.add_parser(
        "vasp-defect-cloud",
        help="Prepare compact local perturbation clouds from relaxed defect motifs.",
    )
    vasp_defect_cloud.add_argument("defect_cloud_args", nargs=argparse.REMAINDER)

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
    if raw_args and raw_args[0] in ("pdfgetx3_status", "pdfgetx3-status"):
        from atomi.lammps.pdfgetx3_status import main as pdfgetx3_status_main

        pdfgetx3_status_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "pdf_md_compare":
        from atomi.lammps.pdf_match import compare_main

        compare_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "pdf_md_reweight":
        from atomi.lammps.pdf_match import reweight_main

        reweight_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "xafs_lammps_prepare":
        from atomi.xafs.larch_md import prepare_main

        prepare_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "xafs_cp2k_prepare":
        from atomi.xafs.cp2k import main as xafs_cp2k_prepare_main

        xafs_cp2k_prepare_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "xafs_larch_run":
        from atomi.xafs.larch_md import larch_run_main

        larch_run_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "xafs_md_compare":
        from atomi.xafs.larch_md import compare_main

        compare_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "xafs_status":
        from atomi.xafs.status import main as xafs_status_main

        xafs_status_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("thermo_lammps", "lammps-thermo-series"):
        from atomi.lammps.thermo_series import main as thermo_series_main

        thermo_series_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("thermal_k_lammps", "thermal-k-lammps"):
        from atomi.lammps.thermal_conductivity import main as thermal_k_lammps_main

        thermal_k_lammps_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "elastic_lammps":
        from atomi.lammps.elastic import main as elastic_lammps_main

        elastic_lammps_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("elastic_viz", "elastic-viz"):
        from atomi.elastic.viz import main as elastic_viz_main

        elastic_viz_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("elate_status", "elastic_viz_status"):
        from atomi.elastic.status import main as elastic_status_main

        elastic_status_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("vasp_elastic", "vasp-elastic"):
        from atomi.vasp.elastic import main as vasp_elastic_main

        vasp_elastic_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in (
        "elastic_vasp_md_compare",
        "elastic-vasp-md-compare",
        "elastic_qha_md_compare",
    ):
        from atomi.lammps.elastic_qha_md_compare import main as elastic_qha_md_compare_main

        elastic_qha_md_compare_main(raw_args[1:])
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
    if raw_args and raw_args[0] in ("moose-doctor", "moose_status", "moose-status"):
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
    if raw_args and raw_args[0] in ("moose-elastic-export", "moose_elastic_export"):
        from atomi.moose.elastic_export import main as moose_elastic_export_main

        moose_elastic_export_main(raw_args[1:])
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
    if raw_args and raw_args[0] in ("calphad-doctor", "calphad_status", "calphad-status"):
        from atomi.calphad.env import main as calphad_doctor_main

        calphad_doctor_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("calphad_export", "calphad-export"):
        from atomi.calphad.export import main as calphad_export_main

        calphad_export_main(raw_args[1:])
        return
    if raw_args and raw_args[0] in ("paper-draft", "report-draft", "atomi-paper-draft"):
        from atomi.reporting.paper_draft import main as paper_draft_main

        paper_draft_main(raw_args[1:])
        return
    zentropy_commands = (
        "zentropy_motif_db",
        "zentropy-motif-db",
        "defect_motif_db",
        "defect-motif-db",
    )
    if raw_args and raw_args[0] in zentropy_commands:
        from atomi.zentropy.motif_db import main as zentropy_motif_db_main

        zentropy_motif_db_main(raw_args[1:])
        return
    motif_path_commands = ("midx", "motifpath", "motifpaths", "motif-paths", "zentropy-motif-paths")
    if raw_args and raw_args[0] in motif_path_commands:
        from atomi.zentropy.motif_paths import main as motif_paths_main

        motif_paths_main(raw_args[1:])
        return
    zentropy_status_commands = ("zentropy_status", "zentropy-status")
    if raw_args and raw_args[0] in zentropy_status_commands:
        from atomi.zentropy.status import main as zentropy_status_main

        zentropy_status_main(raw_args[1:])
        return
    zentropy_workflow_commands = ("zentropy_workflow", "zentropy-workflow")
    if raw_args and raw_args[0] in zentropy_workflow_commands:
        from atomi.zentropy.workflow import main as zentropy_workflow_main

        zentropy_workflow_main(raw_args[1:])
        return
    zentropy_free_energy_commands = ("zentropy_free_energy", "zentropy-free-energy")
    if raw_args and raw_args[0] in zentropy_free_energy_commands:
        from atomi.zentropy.free_energy import main as zentropy_free_energy_main

        zentropy_free_energy_main(raw_args[1:])
        return
    zentropy_solve_commands = ("zentropy_solve", "zentropy-solve")
    if raw_args and raw_args[0] in zentropy_solve_commands:
        from atomi.zentropy.solve import main as zentropy_solve_main

        zentropy_solve_main(raw_args[1:])
        return
    zentropy_export_commands = ("zentropy_export", "zentropy-export")
    if raw_args and raw_args[0] in zentropy_export_commands:
        from atomi.zentropy.export import main as zentropy_export_main

        zentropy_export_main(raw_args[1:])
        return
    zentropy_active_learning_commands = ("zentropy_active_learning", "zentropy-active-learning")
    if raw_args and raw_args[0] in zentropy_active_learning_commands:
        from atomi.zentropy.active_learning import main as zentropy_active_learning_main

        zentropy_active_learning_main(raw_args[1:])
        return
    solid_solution_scan_commands = ("mlip_solid_solution_scan", "mlip-solid-solution-scan")
    if raw_args and raw_args[0] in solid_solution_scan_commands:
        from atomi.zentropy.solid_solution_scan import main as solid_solution_scan_main

        solid_solution_scan_main(raw_args[1:])
        return
    motif_cluster_commands = ("motif_cluster", "motif-cluster", "zentropy_motif_cluster", "zentropy-motif-cluster")
    if raw_args and raw_args[0] in motif_cluster_commands:
        from atomi.zentropy.motif_cluster import main as motif_cluster_main

        motif_cluster_main(raw_args[1:])
        return
    defect_thermo_commands = ("defect_thermo_export", "defect-thermo-export")
    if raw_args and raw_args[0] in defect_thermo_commands:
        from atomi.zentropy.defect_thermo import main as defect_thermo_main

        defect_thermo_main(raw_args[1:])
        return
    sd_dd_commands = ("sd-dd-thermo", "sd_dd_thermo", "defect-sd-dd", "defect-chem")
    if raw_args and raw_args[0] in sd_dd_commands:
        from atomi.zentropy.sd_dd_thermo import main as sd_dd_main

        sd_dd_main(raw_args[1:])
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
    if raw_args and raw_args[0] in ("vasp-spin-report", "spin-report"):
        from atomi.vasp.spin_report import main as vasp_spin_report_main

        vasp_spin_report_main(raw_args[1:])
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
    if raw_args and raw_args[0] == "vasp-defect-cloud":
        from atomi.vasp.defect_cloud import main as vasp_defect_cloud_main

        vasp_defect_cloud_main(raw_args[1:])
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

    if args.subcommand in ("pdfgetx3_status", "pdfgetx3-status"):
        from atomi.lammps.pdfgetx3_status import main as pdfgetx3_status_main

        pdfgetx3_status_main(args.pdfgetx3_status_args)
        return

    if args.subcommand == "pdf_md_compare":
        from atomi.lammps.pdf_match import compare_main

        compare_main(args.pdf_match_args)
        return

    if args.subcommand == "pdf_md_reweight":
        from atomi.lammps.pdf_match import reweight_main

        reweight_main(args.pdf_match_args)
        return

    if args.subcommand == "xafs_lammps_prepare":
        from atomi.xafs.larch_md import prepare_main

        prepare_main(args.xafs_args)
        return

    if args.subcommand == "xafs_cp2k_prepare":
        from atomi.xafs.cp2k import main as xafs_cp2k_prepare_main

        xafs_cp2k_prepare_main(args.xafs_args)
        return

    if args.subcommand == "xafs_larch_run":
        from atomi.xafs.larch_md import larch_run_main

        larch_run_main(args.xafs_args)
        return

    if args.subcommand == "xafs_md_compare":
        from atomi.xafs.larch_md import compare_main

        compare_main(args.xafs_args)
        return

    if args.subcommand == "xafs_status":
        from atomi.xafs.status import main as xafs_status_main

        xafs_status_main(args.xafs_args)
        return

    if args.subcommand in ("thermo_lammps", "lammps-thermo-series"):
        from atomi.lammps.thermo_series import main as thermo_series_main

        thermo_series_main(args.analysis_args)
        return

    if args.subcommand in ("thermal_k_lammps", "thermal-k-lammps"):
        from atomi.lammps.thermal_conductivity import main as thermal_k_lammps_main

        thermal_k_lammps_main(args.thermal_k_args)
        return

    if args.subcommand == "elastic_lammps":
        from atomi.lammps.elastic import main as elastic_lammps_main

        elastic_lammps_main(args.elastic_args)
        return

    if args.subcommand in ("elastic_viz", "elastic-viz"):
        from atomi.elastic.viz import main as elastic_viz_main

        elastic_viz_main(args.elastic_viz_args)
        return

    if args.subcommand in ("elate_status", "elastic_viz_status"):
        from atomi.elastic.status import main as elastic_status_main

        elastic_status_main(args.elastic_status_args)
        return

    if args.subcommand in ("vasp_elastic", "vasp-elastic"):
        from atomi.vasp.elastic import main as vasp_elastic_main

        vasp_elastic_main(args.vasp_elastic_args)
        return

    if args.subcommand in (
        "elastic_vasp_md_compare",
        "elastic-vasp-md-compare",
        "elastic_qha_md_compare",
    ):
        from atomi.lammps.elastic_qha_md_compare import main as elastic_qha_md_compare_main

        elastic_qha_md_compare_main(args.elastic_compare_args)
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

    if args.subcommand in ("moose-doctor", "moose_status", "moose-status"):
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

    if args.subcommand in ("moose-elastic-export", "moose_elastic_export"):
        from atomi.moose.elastic_export import main as moose_elastic_export_main

        moose_elastic_export_main(args.moose_elastic_args)
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

    if args.subcommand in ("calphad-doctor", "calphad_status", "calphad-status"):
        from atomi.calphad.env import main as calphad_doctor_main

        calphad_doctor_main(args.calphad_args)
        return

    if args.subcommand in ("calphad_export", "calphad-export"):
        from atomi.calphad.export import main as calphad_export_main

        calphad_export_main(args.calphad_export_args)
        return

    if args.subcommand in ("paper-draft", "report-draft", "atomi-paper-draft"):
        from atomi.reporting.paper_draft import main as paper_draft_main

        paper_draft_main(args.paper_draft_args)
        return

    if args.subcommand in zentropy_commands:
        from atomi.zentropy.motif_db import main as zentropy_motif_db_main

        zentropy_motif_db_main(args.zentropy_args)
        return

    if args.subcommand in motif_path_commands:
        from atomi.zentropy.motif_paths import main as motif_paths_main

        motif_paths_main(args.motif_path_args)
        return

    if args.subcommand in zentropy_status_commands:
        from atomi.zentropy.status import main as zentropy_status_main

        zentropy_status_main(args.zentropy_status_args)
        return

    if args.subcommand in zentropy_workflow_commands:
        from atomi.zentropy.workflow import main as zentropy_workflow_main

        zentropy_workflow_main(args.zentropy_workflow_args)
        return

    if args.subcommand in zentropy_free_energy_commands:
        from atomi.zentropy.free_energy import main as zentropy_free_energy_main

        zentropy_free_energy_main(args.zentropy_free_energy_args)
        return

    if args.subcommand in zentropy_solve_commands:
        from atomi.zentropy.solve import main as zentropy_solve_main

        zentropy_solve_main(args.zentropy_solve_args)
        return

    if args.subcommand in zentropy_export_commands:
        from atomi.zentropy.export import main as zentropy_export_main

        zentropy_export_main(args.zentropy_export_args)
        return

    if args.subcommand in zentropy_active_learning_commands:
        from atomi.zentropy.active_learning import main as zentropy_active_learning_main

        zentropy_active_learning_main(args.zentropy_active_learning_args)
        return

    if args.subcommand in solid_solution_scan_commands:
        from atomi.zentropy.solid_solution_scan import main as solid_solution_scan_main

        solid_solution_scan_main(args.solid_solution_scan_args)
        return

    if args.subcommand in motif_cluster_commands:
        from atomi.zentropy.motif_cluster import main as motif_cluster_main

        motif_cluster_main(args.motif_cluster_args)
        return

    defect_thermo_commands = ("defect_thermo_export", "defect-thermo-export")
    if args.subcommand in defect_thermo_commands:
        from atomi.zentropy.defect_thermo import main as defect_thermo_main

        defect_thermo_main(args.defect_thermo_args)
        return

    if args.subcommand in sd_dd_commands:
        from atomi.zentropy.sd_dd_thermo import main as sd_dd_main

        sd_dd_main(args.sd_dd_args)
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

    if args.subcommand in ("vasp-spin-report", "spin-report"):
        from atomi.vasp.spin_report import main as vasp_spin_report_main

        vasp_spin_report_main(args.spin_report_args)
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

    if args.subcommand == "vasp-defect-cloud":
        from atomi.vasp.defect_cloud import main as vasp_defect_cloud_main

        vasp_defect_cloud_main(args.defect_cloud_args)
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
