"""Gradual command registry for Atomi CLI aliases.

The legacy CLI still contains hand-written routing for broad compatibility.
This registry is the migration path: new command families can register one
alias group here and avoid touching several distant branches in ``main.py``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class CommandSpec:
    """A pass-through CLI command group."""

    aliases: tuple[str, ...]
    target: str
    category: str
    help: str
    prepend_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def primary_alias(self) -> str:
        return self.aliases[0]

    def matches(self, name: str) -> bool:
        return name in self.aliases

    def load_callable(self) -> Callable[[list[str]], object]:
        module_name, _, attr_name = self.target.partition(":")
        if not module_name or not attr_name:
            raise ValueError(f"Invalid command target {self.target!r}; expected module:callable")
        module = importlib.import_module(module_name)
        func = getattr(module, attr_name)
        return func

    def invoke(self, argv: list[str]) -> object:
        func = self.load_callable()
        return func([*self.prepend_args, *argv])


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        aliases=("turbomole-define", "turbomole_define", "tm-define"),
        target="atomi.qchem.turbomole:main",
        category="qchem",
        help="Prepare blank-line-safe Turbomole define input and run scripts.",
    ),
    CommandSpec(
        aliases=("molcas-cluster", "molcas_cluster", "openmolcas-cluster"),
        target="atomi.qchem.molcas:main",
        category="qchem",
        help="Prepare OpenMolcas embedded-cluster inputs from POSCAR/CONTCAR.",
    ),
    CommandSpec(
        aliases=("molcas-bridge", "molcas_bridge", "openmolcas-bridge", "openmolcas_bridge"),
        target="atomi.qchem.openmolcas_bridge:main",
        category="qchem",
        help="Prepare, run, and summarize OpenMolcas CASSCF/CASPT2/RASSI bridge workspaces.",
    ),
    CommandSpec(
        aliases=("molcas-status", "molcas_status", "openmolcas-status"),
        target="atomi.qchem.openmolcas_bridge:status_cli",
        category="qchem",
        help="Check configured OpenMolcas runtime for Atomi bridge use.",
    ),
    CommandSpec(
        aliases=("molcas-install-plan", "molcas_install_plan"),
        target="atomi.qchem.openmolcas_bridge:install_plan_cli",
        category="qchem",
        help="Print the recommended OpenMolcas HPC/KIT setup pattern.",
    ),
    CommandSpec(
        aliases=("pegamoid-bridge", "pegamoid_bridge"),
        target="atomi.qchem.pegamoid_bridge:main",
        category="qchem",
        help="Prepare Pegamoid launch wrappers for OpenMolcas orbital/density files.",
    ),
    CommandSpec(
        aliases=("pegamoid-status", "pegamoid_status"),
        target="atomi.qchem.pegamoid_bridge:status_cli",
        category="qchem",
        help="Check configured Pegamoid runtime for OpenMolcas orbital viewing.",
    ),
    CommandSpec(
        aliases=("pegamoid-install-plan", "pegamoid_install_plan"),
        target="atomi.qchem.pegamoid_bridge:install_plan_cli",
        category="qchem",
        help="Print the recommended external Pegamoid viewer setup pattern.",
    ),
    CommandSpec(
        aliases=("xafs-routes", "xafs_routes"),
        target="atomi.xafs.routes:main",
        category="xafs",
        help="Print Atomi XAFS Route A/B/C policy: FEFF/Larch, OCEAN, and FDMNES.",
    ),
    CommandSpec(
        aliases=("fdmnes-xanes-bridge", "fdmnes_xanes_bridge", "xanes-fdmnes-bridge"),
        target="atomi.xafs.fdmnes:main",
        category="xafs",
        help="Prepare, run, and collect quick route-C FDMNES XANES workspaces from VASP structures.",
    ),
    CommandSpec(
        aliases=("fdmnes-xanes-status", "fdmnes_xanes_status"),
        target="atomi.xafs.fdmnes:main",
        category="xafs",
        help="Check configured FDMNES runtime for Atomi route-C XANES use.",
        prepend_args=("status",),
    ),
    CommandSpec(
        aliases=("fdmnes-xanes-install-plan",),
        target="atomi.xafs.fdmnes:main",
        category="xafs",
        help="Print the recommended external FDMNES HPC/KIT setup pattern.",
        prepend_args=("install-plan",),
    ),
    CommandSpec(
        aliases=("molcas-xanes-spectrum", "molcas_xanes_spectrum", "xanes-molcas-spectrum"),
        target="atomi.xafs.molcas_xanes_spectrum:main",
        category="xafs",
        help="Broaden OpenMolcas/RASSI dipole transitions into XANES spectra.",
    ),
    CommandSpec(
        aliases=("molcas-postanalysis", "molcas_postanalysis", "openmolcas-postanalysis"),
        target="atomi.qchem.molcas_postanalysis:main",
        category="qchem",
        help="Print Molcas postanalysis workflow and build report-style M4/M5 XANES plots.",
    ),
    CommandSpec(
        aliases=("molcas-exatomic-bridge", "molcas_exatomic_bridge"),
        target="atomi.qchem.exatomic_bridge:main",
        category="qchem",
        help="Optional eXatomic/NBO bridge for OpenMolcas postanalysis exports.",
    ),
    CommandSpec(
        aliases=("molcas-exatomic-status",),
        target="atomi.qchem.exatomic_bridge:status_cli",
        category="qchem",
        help="Check whether eXatomic is importable for Molcas/NBO postanalysis.",
    ),
    CommandSpec(
        aliases=("molcas-exatomic-install-plan",),
        target="atomi.qchem.exatomic_bridge:install_plan_cli",
        category="qchem",
        help="Print the recommended optional eXatomic setup for Atomi MOLCAS postanalysis.",
    ),
    CommandSpec(
        aliases=("molcas-spin-plan", "molcas_spin_plan", "openmolcas-spin-plan"),
        target="atomi.qchem.molcas_spin_plan:main",
        category="qchem",
        help="Plan OpenMolcas spin/root blocks before RASSCF/CASPT2/RASSI production.",
    ),
    CommandSpec(
        aliases=("molcas-symmetry-plan", "molcas_symmetry_plan", "openmolcas-symmetry-plan"),
        target="atomi.qchem.molcas_symmetry:main",
        category="qchem",
        help="Detect and document OpenMolcas D2h-subgroup symmetry choices.",
    ),
    CommandSpec(
        aliases=("aq-thermo-bridge", "aq_thermo_bridge", "thermofun-bridge", "thermohub-bridge"),
        target="atomi.aqueous.thermohub_bridge:main",
        category="aqueous",
        help="Bridge AIMD aqueous logK tables to ThermoHub/ThermoFun/GEMS workflows.",
    ),
    CommandSpec(
        aliases=("zentropy_mode4_surface", "zentropy-mode4-surface", "mode4-surface"),
        target="atomi.zentropy.mode4_surface:main",
        category="zentropy",
        help="Fit and sample dense defect Gibbs surfaces from motif/cluster features.",
    ),
    CommandSpec(
        aliases=("zentropy_gnn_active_learning", "zentropy-gnn-active-learning", "gnn-active-learning"),
        target="atomi.zentropy.gnn_active_learning:main",
        category="zentropy",
        help="Generate and score GNN/MLIP active-learning candidates for defect thermodynamics.",
    ),
    CommandSpec(
        aliases=("crystal-graph-dataset", "crystal_graph_dataset", "atomi-graph-dataset"),
        target="atomi.ml.crystal_graph_dataset:main",
        category="ml",
        help="Export ASE structures or CETrainingSet records as graph JSONL for GNN/MLIP labels.",
    ),
    CommandSpec(
        aliases=("local-structure", "local_structure", "structure-cluster", "local-cluster", "atomi-local-structure"),
        target="atomi.local_structure:main",
        category="structure",
        help="Extract and compare ASE/Pymatgen local clusters from VASP, CP2K, and LAMMPS structures.",
    ),
    CommandSpec(
        aliases=("thermo-prior", "thermo_prior"),
        target="atomi.thermo_prior.cli:main",
        category="thermo_prior",
        help="Create and inspect provenance-rich thermodynamic prior JSON files.",
    ),
    CommandSpec(
        aliases=("thermo-prior-mp", "materials-project-prior", "mp-thermo-cache"),
        target="atomi.thermo_prior.materials_project:main",
        category="thermo_prior",
        help="Normalize Materials Project entries into offline thermo-prior caches.",
    ),
    CommandSpec(
        aliases=("mindat-bridge", "mindat-api", "mindat"),
        target="atomi.thermo_prior.mindat:main",
        category="thermo_prior",
        help="Query token-gated OpenMindat/Mindat API endpoints and write offline JSON caches.",
    ),
    CommandSpec(
        aliases=("vasp-bader-charge", "bader-charge", "vasp-bader", "bader-vasp"),
        target="atomi.vasp.bader_charge:main",
        category="vasp",
        help="Prepare, run, and parse Bader charge analysis from VASP charge-density files.",
    ),
    CommandSpec(
        aliases=(
            "hubbard-u-workflow",
            "hubbard_u_workflow",
            "vasp-hubbard-u",
            "wannier-u-workflow",
            "qe-wannier-u",
        ),
        target="atomi.vasp.hubbard_u:main",
        category="vasp",
        help="Prepare and analyze projector-labeled VASP/QE first-principles Hubbard-U routes.",
    ),
    CommandSpec(
        aliases=("qe-wannier-bridge", "qe_wannier_bridge", "wannier90-bridge"),
        target="atomi.codes.qe_wannier:main",
        category="codes",
        help="Probe and install version-aware QE/Wannier90 sidecar runtimes.",
    ),
    CommandSpec(
        aliases=("pocc-zentropy-defects", "pocc_zentropy_defects", "atomi-defects"),
        target="atomi.zentropy.pocc_defects:main",
        category="zentropy",
        help="Run POCC/zentropy defect population thermodynamics.",
    ),
    CommandSpec(
        aliases=("defect_thermo_export", "defect-thermo-export"),
        target="atomi.zentropy.defect_thermo:main",
        category="zentropy",
        help="Export defect motif energetics for zentropy/CALPHAD/MOOSE coupling.",
    ),
    CommandSpec(
        aliases=("zentropy_motif_db", "zentropy-motif-db", "defect_motif_db", "defect-motif-db"),
        target="atomi.zentropy.motif_db:main",
        category="zentropy",
        help="Index and export defect motifs for zentropy-guided thermodynamics.",
    ),
    CommandSpec(
        aliases=("zentropy_solve", "zentropy-solve"),
        target="atomi.zentropy.solve:main",
        category="zentropy",
        help="Solve discrete motif probabilities from G_i(T) and degeneracy.",
    ),
    CommandSpec(
        aliases=("zentropy_export", "zentropy-export"),
        target="atomi.zentropy.export:main",
        category="zentropy",
        help="Export zentropy thermo and population tables for downstream models.",
    ),
    CommandSpec(
        aliases=("sluschi_bridge", "sluschi-bridge"),
        target="atomi.sluschi.bridge:main",
        category="sluschi",
        help="Prepare and inspect Atomi handoffs to external SLUSCHI workflows.",
    ),
    CommandSpec(
        aliases=("lammps_sconfig", "lammps-sconfig"),
        target="atomi.sluschi.bridge:main",
        category="sluschi",
        help="Parse SLUSCHI configurational-entropy outputs from LAMMPS NVT trajectories.",
        prepend_args=("sconfig",),
    ),
    CommandSpec(
        aliases=("calphad_workflow", "calphad-workflow"),
        target="atomi.calphad.workflow:main",
        category="calphad",
        help="Run config-driven pycalphad inspection, scans, maps, and reaction summaries.",
    ),
    CommandSpec(
        aliases=("calphad_export", "calphad-export"),
        target="atomi.calphad.export:main",
        category="calphad",
        help="Export CALPHAD property tables and MOOSE phase-field templates.",
    ),
    CommandSpec(
        aliases=("moose-qha-md-material",),
        target="atomi.moose.material_export:main",
        category="moose",
        help="Export thermo_qha_md data as MOOSE material-property inputs.",
    ),
    CommandSpec(
        aliases=("mace-build-dataset", "mace-dataset", "build-mace-dataset"),
        target="atomi.ml.mace.datasets:main",
        category="ml",
        help="Build adaptive MACE train/validation extxyz datasets.",
    ),
    CommandSpec(
        aliases=("mace-train",),
        target="atomi.ml.mace.train:main",
        category="ml",
        help="Write or submit a Slurm MACE training/retraining job.",
    ),
)


def command_specs() -> tuple[CommandSpec, ...]:
    return COMMAND_SPECS


def command_registry() -> dict[str, CommandSpec]:
    registry: dict[str, CommandSpec] = {}
    for spec in COMMAND_SPECS:
        for alias in spec.aliases:
            if alias in registry:
                raise ValueError(f"Duplicate CLI alias in registry: {alias}")
            registry[alias] = spec
    return registry


def registered_aliases() -> list[str]:
    return sorted(command_registry())


def specs_by_category() -> dict[str, list[CommandSpec]]:
    grouped: dict[str, list[CommandSpec]] = {}
    for spec in COMMAND_SPECS:
        grouped.setdefault(spec.category, []).append(spec)
    return grouped


def dispatch_registered_command(raw_args: list[str]) -> bool:
    """Dispatch a registered pass-through command.

    Returns ``True`` if a command was recognized and invoked, otherwise
    ``False`` so the legacy router can continue handling it.
    """

    if not raw_args:
        return False
    spec = command_registry().get(raw_args[0])
    if spec is None:
        return False
    spec.invoke(raw_args[1:])
    return True
