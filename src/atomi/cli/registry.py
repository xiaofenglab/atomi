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
