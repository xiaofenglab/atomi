"""Spin-state/root planning helpers for OpenMolcas spectroscopy inputs.

The helpers here are intentionally conservative.  They do not replace a
chemist's active-space decision; they make the spin, root, and block intent
explicit before a RASSCF probe or CASPT2/RASSI production input is written.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from atomi.qchem.molcas_symmetry import D2H_ORDER, d2h_product_counts
from atomi.structure.elements import element_info, valence_magmom_info


SCHEMA = "atomi.openmolcas_spin_plan.v2"

MULTIPLICITY_LABELS = {1: "singlet", 2: "doublet", 3: "triplet", 4: "quartet", 5: "quintet"}


@dataclass(frozen=True)
class ShellSpec:
    """Symmetry-resolved active-shell count."""

    label: str
    counts: dict[str, int]
    ras: str
    note: str = ""

    def vector(self, order: tuple[str, ...] = D2H_ORDER) -> str:
        return _vector_from_counts(self.counts, order=order)

    def total_orbitals(self) -> int:
        return sum(int(v) for v in self.counts.values())


@dataclass(frozen=True)
class SpinBlock:
    label: str
    manifold: str
    symmetry: int
    symmetry_label: str
    spin_multiplicity: int
    roots: int
    nactel: str
    ras1: str = ""
    ras2: str = ""
    ras3: str = ""
    hexs: bool = False
    root_source: str = ""
    requested_roots: int | None = None
    note: str = ""


@dataclass(frozen=True)
class GasSpace:
    """One symmetry-resolved GASSCF space with cumulative occupancy limits."""

    label: str
    vector: str
    min_electrons: int
    max_electrons: int
    note: str = ""


def _comb(n: int, k: int) -> int:
    if n < 0 or k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _validate_counts(counts: dict[str, int], *, order: tuple[str, ...] = D2H_ORDER) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in counts.items():
        if key not in order:
            raise ValueError(f"Unknown irrep {key!r}; expected one of {', '.join(order)}")
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError(f"Irrep counts must be non-negative, got {key}={value}")
        if ivalue:
            normalized[key] = ivalue
    return normalized


def _vector_from_counts(counts: dict[str, int], *, order: tuple[str, ...] = D2H_ORDER) -> str:
    normalized = _validate_counts(counts, order=order)
    return " ".join(str(normalized.get(irrep, 0)) for irrep in order)


def f2_root_counts(n_orbitals: int) -> dict[str, int]:
    """Return singlet/triplet spatial-root counts for two electrons in n orbitals."""

    if n_orbitals < 2:
        raise ValueError("f2 root counts require at least two spatial orbitals")
    triplet = _comb(n_orbitals, 2)
    singlet = triplet + n_orbitals
    return {"triplet": triplet, "singlet": singlet}


def f3_root_counts(n_orbitals: int) -> dict[str, int]:
    """Return doublet/quartet spatial-root counts for three electrons in n orbitals."""

    if n_orbitals < 3:
        raise ValueError("f3 root counts require at least three spatial orbitals")
    quartet = _comb(n_orbitals, 3)
    doublet = n_orbitals * (n_orbitals - 1) * (n_orbitals + 1) // 3
    return {"doublet": doublet, "quartet": quartet}


def core_hole_f2_to_f3_root_counts(*, n_core_orbitals: int, n_acceptor_orbitals: int) -> dict[str, int]:
    """Estimate C1 core-excited roots for d10 f2 -> d9 f3 blocks.

    This is the spin-coupling planner used for U4+ M4,5 3d -> 5f scouts.  It
    assumes one core hole coupled to a 5f3 acceptor manifold:

    - f3 doublet + core-hole doublet -> singlet and triplet
    - f3 quartet + core-hole doublet -> triplet and quintet

    The OpenMolcas output remains the authority; use this to avoid obviously
    wrong CIROOTS before the diagnostic run.
    """

    if n_core_orbitals < 1:
        raise ValueError("core-hole root counts require at least one core orbital")
    f3 = f3_root_counts(n_acceptor_orbitals)
    doublet = n_core_orbitals * f3["doublet"]
    quartet = n_core_orbitals * f3["quartet"]
    return {
        "singlet": doublet,
        "triplet": doublet + quartet,
        "quintet": quartet,
    }


def _element_metadata(element: str, oxidation_state: int, *, edges: tuple[str, ...]) -> dict[str, Any]:
    info = element_info(element, include_xray_edges=True, edges=edges, allow_vacancy=False)
    payload = info.to_dict() if info is not None else {"symbol": element}
    sign = "+" if oxidation_state >= 0 else "-"
    valence = valence_magmom_info(f"{element}{abs(oxidation_state)}{sign}")
    if valence:
        payload["formal_valence_prior"] = valence.to_dict()
    return payload


def _maybe_cap_roots(roots: int, root_cap: int | None) -> tuple[int, str]:
    if root_cap is None or roots <= root_cap:
        return roots, ""
    return root_cap, f"Root request truncated by cap from {roots} to {root_cap}; diagnostic/full-root audit required."


def ce4_l23_d2h_plan(*, edge_label: str = "L2,3") -> dict[str, Any]:
    """Build a Ce4+ CeO8 D2h spin/root preflight plan.

    Model:
    - Ce4+ ground is nominal 4f0, closed-shell singlet.
    - Active core shell is Ce 2p: B3u, B2u, B1u in Molcas D2h order.
    - Acceptor shell is Ce 5d: 2 Ag + B1g + B2g + B3g.
    - Core-excited 2p5 5d1 channels are singlet and triplet before RASSI/SOC.

    In scalar D2h, the RASSCF probe spans the full 2p shell.  L3-only
    interpretation is made after spin-orbit/RASSI/postanalysis.
    """

    core = ShellSpec("Ce 2p", {"B3u": 1, "B2u": 1, "B1u": 1}, "RAS1")
    acceptor = ShellSpec("Ce 5d", {"Ag": 2, "B1g": 1, "B2g": 1, "B3g": 1}, "RAS3")
    excited_counts = d2h_product_counts(core.counts, acceptor.counts)
    ras1 = core.vector()
    ras3 = acceptor.vector()
    nactel = "6 1 1"
    blocks: list[SpinBlock] = [
        SpinBlock(
            label="ground_singlet_Ce4_4f0",
            manifold="ground",
            symmetry=1,
            symmetry_label="Ag",
            spin_multiplicity=1,
            roots=1,
            nactel=nactel,
            ras1=ras1,
            ras3=ras3,
            root_source="closed-shell singlet reference",
            note="Closed-shell Ce4+ 4f0 reference with Ce 2p/5d active window.",
        )
    ]
    for spin in (1, 3):
        for irrep in ("B3u", "B2u", "B1u", "Au"):
            roots = excited_counts[irrep]
            blocks.append(
                SpinBlock(
                    label=f"excited_{irrep}_spin{spin}_Ce_L23_2p5_5d1",
                    manifold="core_excited",
                    symmetry=D2H_ORDER.index(irrep) + 1,
                    symmetry_label=irrep,
                    spin_multiplicity=spin,
                    roots=roots,
                    nactel=nactel,
                    ras1=ras1,
                    ras3=ras3,
                    hexs=True,
                    root_source="D2h products Ce 2p x Ce 5d",
                    note="Ce L2,3 2p -> 5d single core excitation; validate roots from RASSCF scout.",
                )
            )
    return {
        "schema": SCHEMA,
        "plan_kind": "closed_shell_core_to_acceptor",
        "system": "CeO8",
        "element": "Ce",
        "element_metadata": _element_metadata("Ce", 4, edges=("L3", "L2", "L1")),
        "edge": edge_label,
        "formal_oxidation": 4,
        "group": "D2h Abelian subgroup of fluorite/Oh local symmetry",
        "molcas_d2h_order": list(D2H_ORDER),
        "core_shell": {"label": core.label, "ras1": ras1, "counts": core.counts},
        "acceptor_shell": {"label": acceptor.label, "ras3": ras3, "counts": acceptor.counts},
        "nactel": nactel,
        "excited_irrep_root_counts": excited_counts,
        "blocks": [asdict(block) for block in blocks],
        "warnings": [
            "Inactive vectors and Alter rotations require SCF/RASSCF scout output before running this as production.",
            "Ce 4f acceptor/bridge orbitals are not included in the first 5d-only pilot; add a separate branch if AO/MO analysis requires them.",
            "Scalar D2h RASSCF spans Ce 2p; L3 assignment should be made after RASSI/SOC/postanalysis.",
            "Run RASSCF-only probe before CASPT2/RASSI.",
        ],
    }


def ce4_l3_d2h_plan() -> dict[str, Any]:
    """Alias plan for the Ce L3 pilot; see Ce 2p/SOC warning in the payload."""

    return ce4_l23_d2h_plan(edge_label="L3")


def ce4_l23_4f_ligand_gas_plan(*, selector_dimension: int = 10) -> dict[str, Any]:
    """Plan the Ce4+ 4f/ligand exact-sector GASSCF audit.

    This is deliberately a *diagnostic* plan.  The four independent GAS
    spaces preserve the distinction between a Ce 2p core hole and an O 2p
    ligand hole, but the OpenMolcas build used for this workflow cannot use a
    true-GAS JobIph safely for the subsequent CASPT2/RASSI handoff.  A reduced
    common-orbital RAS bridge must be designed only after the selector-root
    audit has classified the accepted screened and poorly-screened sectors.
    """

    if not 1 <= selector_dimension <= 10:
        raise ValueError("The validated Ce4+ true-GAS selector window is 1 through 10 roots.")

    spaces = (
        GasSpace("Ce 2p core", "0 1 1 0 1 0 0 0", 5, 5, "Core-excited constraint; use 6,6 for ground."),
        GasSpace("O 2p ligand SALCs", "0 2 2 0 2 0 0 1", 18, 19, "Allows L14/4f0 and L13/4f1 sectors."),
        GasSpace("Ce 4f", "0 2 2 0 2 0 0 1", 19, 19, "Balances the optional ligand hole."),
        GasSpace("Ce 6s/5d", "3 0 0 1 0 1 1 0", 20, 20, "One acceptor electron in the core-excited sector."),
    )
    ground_spaces = (
        GasSpace("Ce 2p core", spaces[0].vector, 6, 6, "Full 2p6 core."),
        GasSpace("O 2p ligand SALCs", spaces[1].vector, 19, 20, "Allows 4f0 and 4f1 ligand-hole ground sectors."),
        GasSpace("Ce 4f", spaces[2].vector, 20, 20, "Balances the optional ligand hole."),
        GasSpace("Ce 6s/5d", spaces[3].vector, 20, 20, "Excluded from the ground sector."),
    )
    full_root_counts = {
        1: {"B3u": 447, "B2u": 447, "B1u": 447, "Au": 441},
        3: {"B3u": 668, "B2u": 668, "B1u": 668, "Au": 660},
        5: {"B3u": 221, "B2u": 221, "B1u": 221, "Au": 219},
    }
    blocks = []
    for multiplicity, counts in full_root_counts.items():
        for irrep in ("B3u", "B2u", "B1u", "Au"):
            blocks.append(
                {
                    "label": f"L23_{irrep}_{MULTIPLICITY_LABELS[multiplicity]}_selector",
                    "manifold": "core_excited",
                    "symmetry": D2H_ORDER.index(irrep) + 1,
                    "symmetry_label": irrep,
                    "spin_multiplicity": multiplicity,
                    "full_root_count": counts[irrep],
                    "selector_dimension": selector_dimension,
                    "root_source": "Validated Ce4+ CN8 four-GAS root/CSF inventory",
                }
            )
    return {
        "schema": SCHEMA,
        "plan_kind": "true_gas_selector_audit",
        "system": "CeO8",
        "element": "Ce",
        "formal_oxidation": 4,
        "edge": "L2,3",
        "group": "D2h Abelian subgroup of fluorite/Oh local symmetry",
        "molcas_d2h_order": list(D2H_ORDER),
        "inactive": "14 6 6 7 6 7 7 4",
        "nactel": "20 0 0",
        "ground_spaces": [asdict(space) for space in ground_spaces],
        "core_excited_spaces": [asdict(space) for space in spaces],
        "ground_full_root_count": 14,
        "selector_dimension": selector_dimension,
        "blocks": blocks,
        "production_guard": {
            "jobiph": False,
            "caspt2": False,
            "rassi": False,
            "reason": "True GAS is not JobIph/RASSI compatible in the validated OpenMolcas build.",
        },
        "warnings": [
            "Do not use the historical 14-root GAS state average; it crashes in the per-root output path.",
            "Use one fixed-orbital selector root at a time with CIROOTS 1 <dimension>, then the selected root index.",
            "Classify screened versus poorly-screened CI character before designing a reduced common-orbital RAS bridge.",
            "Only the later, independently validated RAS bridge may create JobIph and proceed to CASPT2/RASSI.",
        ],
    }


def _render_gasscf_selector_block(
    *,
    title: str,
    symmetry: int,
    spin: int,
    inactive: str,
    spaces: tuple[GasSpace, ...],
    selector_dimension: int,
    selector_root: int,
    alter_lines: tuple[str, ...] = (),
) -> str:
    """Render one fixed-orbital true-GAS selector block without JobIph output."""

    if not 1 <= selector_root <= selector_dimension:
        raise ValueError("selector_root must be within the selector Davidson dimension")
    lines = [
        "&RASSCF",
        "Title",
        f" {title}",
        "Symmetry",
        f" {symmetry}",
        "Spin",
        f" {spin}",
        "nActEl",
        " 20 0 0",
        "Inactive",
        f" {inactive}",
        "GASSCF",
        f" {len(spaces)}",
    ]
    for space in spaces:
        lines.extend([f" {space.vector}", f" {space.min_electrons} {space.max_electrons}"])
    lines.extend(
        [
            "CIROOTS",
            f" 1 {selector_dimension}",
            f" {selector_root}",
            "CIONLY",
            *alter_lines,
            "OUTOrbitals",
            " AVERage",
            "ORBL",
            " NOTHING",
            "ORBA",
            " COMP",
            "PRWF",
            " 1.0E-03",
            "End of input",
        ]
    )
    return "\n".join(lines) + "\n"


def render_ce4_l23_4f_ligand_gas_selector_probe(
    *,
    coord: str = "CeO8_fluorite_ideal.xyz",
    basis: str = "ANO-RCC-VDZP",
    charge: int = -12,
    spin: int = 1,
    selector_dimension: int = 10,
    selector_root: int = 1,
) -> str:
    """Render one Ce4+ CN8 true-GAS selector audit deck.

    The known Ce4+ CN8 orbital rotation table is included only for this
    validated scaffold.  New elements/clusters must derive and review their
    own orbital table before reusing a GAS selector workflow.
    """

    plan = ce4_l23_4f_ligand_gas_plan(selector_dimension=selector_dimension)
    if spin not in (1, 3, 5):
        raise ValueError("Ce4+ 4f/ligand L2,3 GAS audit requires spin 1, 3, or 5")
    spaces = tuple(GasSpace(**space) for space in plan["core_excited_spaces"])
    ground_spaces = tuple(GasSpace(**space) for space in plan["ground_spaces"])
    alter = (
        "Alter",
        " 9",
        " 2 1 4",
        " 2 4 7",
        " 2 10 12",
        " 3 1 4",
        " 3 4 7",
        " 3 10 12",
        " 5 1 4",
        " 5 4 7",
        " 5 10 12",
    )
    header = [
        "* Atomi Ce4+ CN8 four-GAS exact-sector selector audit.",
        "* RASSCF-only: true GAS must not emit JobIph or enter CASPT2/RASSI.",
        "* The selector root is an audited state, not a state average.",
        "&GATEWAY",
        "Title = CeO2 CeO8 Ce4+ L2,3 4f ligand-hole GAS selector audit",
        f"Coord = {coord}",
        f"Basis = {basis}",
        "Group = X Y Z",
        "RX2C",
        "AMFI",
        "ANGM",
        "0.0 0.0 0.0",
        "",
        "&SEWARD",
        "CHOL",
        "End of input",
        "",
        "&SCF",
        "Charge",
        f" {charge}",
        "PROR",
        " 2 5.0 2",
        "THRE",
        " 1.0d-11 1.0d-6 1.5d-6 0.2d-6",
        "End of input",
        "",
        "* Stable ionic 4f0 closed-shell reference; it is not a zero-electron GASSCF optimizer.",
        "&RASSCF",
        "Title",
        " closed_shell_ionic_Ce4_4f0_reference",
        "Symmetry",
        " 1",
        "Spin",
        " 1",
        "nActEl",
        " 0 0 0",
        "Inactive",
        " 14 9 9 7 9 7 7 5",
        "Ras2",
        " 0 0 0 0 0 0 0 0",
        "CIROOTS",
        " 1 1 1",
        "Iterations",
        " 200 100",
        "levs",
        " 2.5",
        "ORBL",
        " ALL",
        "ORBA",
        " COMP",
        "End of input",
        "",
        "* Ground selector provides the common 4f0/4f1L reference before core-hole blocks.",
        _render_gasscf_selector_block(
            title="ground_Ag_singlet_4f0_4f1L_selector",
            symmetry=1,
            spin=1,
            inactive=str(plan["inactive"]),
            spaces=ground_spaces,
            selector_dimension=1,
            selector_root=1,
            alter_lines=alter,
        ).rstrip(),
        "",
        "* Every core-excited block retains 2p5 and the L14/4f0 plus L13/4f1 sectors.",
    ]
    for block in plan["blocks"]:
        if block["spin_multiplicity"] != spin:
            continue
        header.append(
            _render_gasscf_selector_block(
                title=f"{block['label']}_root{selector_root}",
                symmetry=int(block["symmetry"]),
                spin=spin,
                inactive=str(plan["inactive"]),
                spaces=spaces,
                selector_dimension=selector_dimension,
                selector_root=selector_root,
            ).rstrip()
        )
        header.append("")
    header.extend(
        [
            "* End of true-GAS selector audit.",
            "* No JobIph copy, CASPT2, JobMix, or RASSI is permitted in this deck.",
            "* Extract CI sector weights first, then build a separate reduced common-orbital RAS continuation.",
        ]
    )
    return "\n".join(header) + "\n"


def u4_m45_c1_5f_plan(*, n_f_orbitals: int = 7, n_core_orbitals: int = 5, root_cap: int | None = None) -> dict[str, Any]:
    """Build a U4+ M4,5 C1 5f-only spin/root preflight plan.

    The C1 version mirrors the low-symmetry U4O9 CN8 family-01 workflow.  It
    keeps U 3d in RAS1 and U 5f in RAS3, excludes 7s, and explicitly includes
    ground singlet/triplet plus core-excited singlet/triplet/quintet manifolds.
    """

    ground = f2_root_counts(n_f_orbitals)
    excited = core_hole_f2_to_f3_root_counts(n_core_orbitals=n_core_orbitals, n_acceptor_orbitals=n_f_orbitals)
    ras1 = str(n_core_orbitals)
    ras3 = str(n_f_orbitals)
    nactel = f"{2 * n_core_orbitals + 2} 1 3"
    blocks: list[SpinBlock] = [
        SpinBlock(
            label="ground_triplet_U4_5f2",
            manifold="ground",
            symmetry=1,
            symmetry_label="A",
            spin_multiplicity=3,
            roots=ground["triplet"],
            nactel=nactel,
            ras1=ras1,
            ras3=ras3,
            root_source="C(n,2) different-orbital 5f2 triplets",
            note="Hund-like U4+ 5f2 triplet ground-state manifold.",
        ),
        SpinBlock(
            label="ground_singlet_U4_5f2",
            manifold="ground",
            symmetry=1,
            symmetry_label="A",
            spin_multiplicity=1,
            roots=ground["singlet"],
            nactel=nactel,
            ras1=ras1,
            ras3=ras3,
            root_source="C(n,2) different-orbital plus n paired same-orbital 5f2 singlets",
            note="U4+ 5f2 singlet manifold; include for spin-orbit/RASSI mixing.",
        ),
    ]
    for spin, key in ((1, "singlet"), (3, "triplet"), (5, "quintet")):
        roots = excited[key]
        requested, cap_note = _maybe_cap_roots(roots, root_cap)
        blocks.append(
            SpinBlock(
                label=f"excited_spin{spin}_U_M45_3d9_5f3",
                manifold="core_excited",
                symmetry=1,
                symmetry_label="A",
                spin_multiplicity=spin,
                roots=requested,
                requested_roots=roots,
                nactel=nactel,
                ras1=ras1,
                ras3=ras3,
                hexs=True,
                root_source="core-hole doublet coupled to 5f3 doublet/quartet roots",
                note=cap_note
                or "U M4,5 3d -> 5f single core excitation; validate roots from RASSCF scout output.",
            )
        )
    warnings = [
        "Inactive vectors and Alter rotations require SCF/RASSCF scout output before running this as production.",
        "This is the 5f-only branch; add 7s only if AO/MO postanalysis shows dipole-relevant 5f-7s mixing.",
        "OpenMolcas output remains the authority for Number of highly excited CSFs and Number of root(s) required.",
    ]
    if root_cap is not None and any((block.requested_roots or block.roots) > block.roots for block in blocks):
        warnings.append("One or more excited blocks are root-truncated; label the run accordingly.")
    return {
        "schema": SCHEMA,
        "plan_kind": "open_shell_f2_core_to_f3",
        "system": "UO8",
        "element": "U",
        "element_metadata": _element_metadata("U", 4, edges=("M5", "M4", "M3")),
        "edge": "M4,5",
        "formal_oxidation": 4,
        "group": "C1 low-symmetry cluster",
        "core_shell": {"label": "U 3d", "ras1": ras1, "n_orbitals": n_core_orbitals},
        "acceptor_shell": {"label": "U 5f", "ras3": ras3, "n_orbitals": n_f_orbitals},
        "nactel": nactel,
        "ground_root_counts": ground,
        "f3_acceptor_root_counts": f3_root_counts(n_f_orbitals),
        "core_excited_root_counts": excited,
        "blocks": [asdict(block) for block in blocks],
        "warnings": warnings,
    }


def _render_rasscf_blocks(
    *,
    header: list[str],
    blocks: list[dict[str, Any]],
    inactive: str,
    block_note: str,
) -> str:
    lines = list(header)
    for block in blocks:
        lines.extend(
            [
                f"* Block: {block['label']}",
                f"* {block_note}",
                "&RASSCF",
                f"Title = {block['label']}",
                f"Symmetry = {block['symmetry']}  * {block['symmetry_label']}",
                f"Spin = {block['spin_multiplicity']}",
                f"nActEl = {block['nactel']}",
                f"Inactive = {inactive}",
                f"Ras1 = {block['ras1']}",
            ]
        )
        if block.get("ras2"):
            lines.append(f"Ras2 = {block['ras2']}")
        if block.get("ras3"):
            lines.append(f"Ras3 = {block['ras3']}")
        if block.get("hexs"):
            lines.extend(["HEXS", " 1", " 1"])
        if block.get("requested_roots") and block["requested_roots"] != block["roots"]:
            lines.append(f"* {block['note']}")
        lines.extend(
            [
                f"CIROOTS = {block['roots']} {block['roots']} 1",
                "Iterations = 300 100",
                "levs = 2.5",
                "ORBL = ALL",
                "ORBA = COMP",
                "TDM",
                "End of input",
                f">>COPY $Project.JobIph $Project.JobIph_{block['label']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def render_ce4_l23_d2h_probe(
    *,
    coord: str = "CeO8_fluorite_ideal.xyz",
    basis: str = "ANO-RCC-VDZP",
    charge: int = -12,
    inactive: str = "TBD_FROM_SCF_SCOUT",
    project_note: str = "CeO2 CeO8 Ce4+ L2,3 RASSCF-only draft",
    edge_label: str = "L2,3",
) -> str:
    """Render an annotated Ce4+ L2,3/L3 RASSCF-only D2h draft input."""

    plan = ce4_l23_d2h_plan(edge_label=edge_label)
    header = [
        f"* Atomi OpenMolcas Ce4+ CeO8 {edge_label}-edge RASSCF-only draft",
        "* Review before running: fill Inactive vector and Alter rotations from SCF scout.",
        "&GATEWAY",
        f"Title = {project_note}",
        f"Coord = {coord}",
        f"Basis = {basis}",
        "Group = Full",
        "RX2C",
        "AMFI",
        "ANGM",
        "0.0 0.0 0.0",
        "",
        "&SEWARD",
        "Cholesky",
        "",
        "&SCF",
        f"Charge = {charge}",
        "Iterations = 200",
        "",
    ]
    text = _render_rasscf_blocks(
        header=header,
        blocks=plan["blocks"],
        inactive=inactive,
        block_note="Inactive and Alter are intentionally placeholders until scout orbital inspection.",
    )
    return (
        text
        + "* No CASPT2/RASSI here. Build production only after RASSCF probe/root audit.\n"
        + "* Future RASSI should include ground singlet plus singlet/triplet core-excited JobMix groups.\n"
    )


def render_u4_m45_c1_5f_probe(
    *,
    coord: str = "U4O9_01_cube_U4_CN8_average.xyz",
    basis: str = "ANO-RCC-VDZP",
    charge: int = -10,
    inactive: str = "TBD_FROM_SCF_SCOUT",
    root_cap: int | None = None,
) -> str:
    """Render an annotated U4+ M4,5 5f-only C1 RASSCF-only draft input."""

    plan = u4_m45_c1_5f_plan(root_cap=root_cap)
    header = [
        "* Atomi OpenMolcas U4+ UO8 M4,5-edge 5f-only RASSCF-only draft",
        "* Review before running: fill Inactive and Alter rotations from SCF/RASSCF scouts.",
        "&GATEWAY",
        "Title = U4O9 UO8 CN8 family-01 U4+ 5f-only M4,5 RASSCF probe",
        f"Coord = {coord}",
        f"Basis = {basis}",
        "Group = NoSym",
        "RX2C",
        "AMFI",
        "ANGM",
        "0.0 0.0 0.0",
        "",
        "&SEWARD",
        "Cholesky",
        "",
        "&SCF",
        f"Charge = {charge}",
        "Iterations = 200",
        "",
    ]
    text = _render_rasscf_blocks(
        header=header,
        blocks=plan["blocks"],
        inactive=inactive,
        block_note="This probing input excludes CASPT2/RASSI and uses scout-validated rotations later.",
    )
    return (
        text
        + "* No CASPT2/RASSI here. Add production stages only after RASSCF probe/root audit.\n"
        + "* The Spin=3 excited block is above 600 roots for 5f-only C1 unless root-truncated.\n"
    )


def _write_json(payload: dict[str, Any], path: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _ce_l23(args: argparse.Namespace) -> int:
    plan = ce4_l23_d2h_plan(edge_label=args.edge_label)
    if args.write_json:
        _write_json(plan, args.write_json)
    if args.write_inp:
        args.write_inp.parent.mkdir(parents=True, exist_ok=True)
        args.write_inp.write_text(
            render_ce4_l23_d2h_probe(
                coord=args.coord,
                basis=args.basis,
                charge=args.charge,
                inactive=args.inactive,
                edge_label=args.edge_label,
            ),
            encoding="utf-8",
        )
    if not args.write_json and not args.write_inp:
        _write_json(plan, None)
    return 0


def _u4_m45(args: argparse.Namespace) -> int:
    plan = u4_m45_c1_5f_plan(n_f_orbitals=args.n_f_orbitals, n_core_orbitals=args.n_core_orbitals, root_cap=args.root_cap)
    if args.write_json:
        _write_json(plan, args.write_json)
    if args.write_inp:
        args.write_inp.parent.mkdir(parents=True, exist_ok=True)
        args.write_inp.write_text(
            render_u4_m45_c1_5f_probe(
                coord=args.coord,
                basis=args.basis,
                charge=args.charge,
                inactive=args.inactive,
                root_cap=args.root_cap,
            ),
            encoding="utf-8",
        )
    if not args.write_json and not args.write_inp:
        _write_json(plan, None)
    return 0


def _f2_roots(args: argparse.Namespace) -> int:
    payload = {
        "schema": SCHEMA,
        "model": "two electrons in n spatial orbitals",
        "n_orbitals": args.n_orbitals,
        "roots": f2_root_counts(args.n_orbitals),
        "notes": {
            "triplet": "C(n,2), different-orbital pairs only.",
            "singlet": "C(n,2) plus n same-orbital paired singlets.",
        },
    }
    _write_json(payload, args.write_json)
    return 0


def _core_f2_roots(args: argparse.Namespace) -> int:
    payload = {
        "schema": SCHEMA,
        "model": "one core hole coupled to f3 generated from f2 -> f3 excitation",
        "n_core_orbitals": args.n_core_orbitals,
        "n_acceptor_orbitals": args.n_acceptor_orbitals,
        "f3_roots": f3_root_counts(args.n_acceptor_orbitals),
        "core_excited_roots": core_hole_f2_to_f3_root_counts(
            n_core_orbitals=args.n_core_orbitals,
            n_acceptor_orbitals=args.n_acceptor_orbitals,
        ),
    }
    _write_json(payload, args.write_json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan OpenMolcas spin/root blocks before RASSCF/CASPT2/RASSI.")
    sub = parser.add_subparsers(dest="command", required=True)

    ce = sub.add_parser("ce-l23", help="Write/print a Ce4+ CeO8 L2,3/L3 D2h spin/root plan.")
    ce.add_argument("--coord", default="CeO8_fluorite_ideal.xyz")
    ce.add_argument("--basis", default="ANO-RCC-VDZP")
    ce.add_argument("--charge", type=int, default=-12)
    ce.add_argument("--inactive", default="TBD_FROM_SCF_SCOUT")
    ce.add_argument("--edge-label", default="L2,3", choices=("L2,3", "L3"))
    ce.add_argument("--write-json", type=Path)
    ce.add_argument("--write-inp", type=Path)
    ce.set_defaults(func=_ce_l23)

    ce_l3 = sub.add_parser("ce-l3", help="Alias for a Ce4+ CeO8 L3-labelled D2h plan.")
    ce_l3.add_argument("--coord", default="CeO8_fluorite_ideal.xyz")
    ce_l3.add_argument("--basis", default="ANO-RCC-VDZP")
    ce_l3.add_argument("--charge", type=int, default=-12)
    ce_l3.add_argument("--inactive", default="TBD_FROM_SCF_SCOUT")
    ce_l3.add_argument("--edge-label", default="L3")
    ce_l3.add_argument("--write-json", type=Path)
    ce_l3.add_argument("--write-inp", type=Path)
    ce_l3.set_defaults(func=_ce_l23)

    u4 = sub.add_parser("u4-m45-5f", help="Write/print a U4+ M4,5 C1 5f-only spin/root plan.")
    u4.add_argument("--n-f-orbitals", type=int, default=7)
    u4.add_argument("--n-core-orbitals", type=int, default=5)
    u4.add_argument("--root-cap", type=int, default=None, help="Optionally truncate requested roots, e.g. 600.")
    u4.add_argument("--coord", default="U4O9_01_cube_U4_CN8_average.xyz")
    u4.add_argument("--basis", default="ANO-RCC-VDZP")
    u4.add_argument("--charge", type=int, default=-10)
    u4.add_argument("--inactive", default="TBD_FROM_SCF_SCOUT")
    u4.add_argument("--write-json", type=Path)
    u4.add_argument("--write-inp", type=Path)
    u4.set_defaults(func=_u4_m45)

    f2 = sub.add_parser("f2-roots", help="Compute singlet/triplet root counts for two electrons in n orbitals.")
    f2.add_argument("--n-orbitals", type=int, default=7)
    f2.add_argument("--write-json", type=Path)
    f2.set_defaults(func=_f2_roots)

    core_f2 = sub.add_parser(
        "core-f2-roots",
        help="Compute U4+-style core-hole f2 -> f3 singlet/triplet/quintet root counts.",
    )
    core_f2.add_argument("--n-core-orbitals", type=int, default=5)
    core_f2.add_argument("--n-acceptor-orbitals", type=int, default=7)
    core_f2.add_argument("--write-json", type=Path)
    core_f2.set_defaults(func=_core_f2_roots)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
