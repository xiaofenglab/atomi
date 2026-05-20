"""Bridge ATAT configuration tools into Atomi defect workflows."""

from __future__ import annotations

import argparse
import itertools
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


SCHEMA = "atomi.atat.bridge.v1"
STATUS_SCHEMA = "atomi.atat.status.v1"
QUICK_OPT_SCHEMA = "atomi.atat.quick_opt.v1"

ATAT_TOOLS: dict[str, list[str]] = {
    "cluster_expansion": ["maps", "mmaps", "corrdump", "genstr", "emc2"],
    "sqs": ["mcsqs", "corrdump"],
    "structure_conversion": ["str2poscar", "poscar2str"],
    "diagnostics": ["mapsrep", "checkcell", "cellcvrt"],
}

WORKFLOW_STAGES: list[dict[str, str]] = [
    {
        "stage_id": "01_motif_search",
        "atat_role": "Enumerate low-energy occupational, ionic, and defect configurations.",
        "atomi_role": "Prepare VASP folders, run fail-fast branch screening, then keep physics-accepted low-energy motifs.",
        "typical_inputs": "lat.in, clusters.out, pseudo_species_map.csv, VASP_TEMPLATE",
        "typical_outputs": "atat_candidate_index.csv, atomi_runlist.txt, stage1_branch_summary.csv",
    },
    {
        "stage_id": "02_sd_dd_cluster_expansion",
        "atat_role": "Fit effective interactions from accepted DFT energies and accelerate motif/defect arrangement search.",
        "atomi_role": "Convert accepted motif energies into SD/DD defect tables and CALPHAD-ready interaction summaries.",
        "typical_inputs": "energy-labeled structures, defects.csv, reference_energies.csv",
        "typical_outputs": "ce_fit_manifest.json, pair_interactions.csv, sd_dd_defects_from_ce.csv",
    },
    {
        "stage_id": "03_sqs_mid_high_concentration",
        "atat_role": "Generate SQS or representative disordered cells for mid/high composition defect chemistry.",
        "atomi_role": "Use SQS structures as seed motifs for spin branching, MLIP/DFT screening, and zentropy microstates.",
        "typical_inputs": "rndstr.in, target composition, pseudo_species_map.csv",
        "typical_outputs": "bestsqs.out, sqs_candidate_index.csv, VASP-ready runlist",
    },
    {
        "stage_id": "04_feedback_to_atomi",
        "atat_role": "Provide ranked structures, correlations, CE uncertainties, and MC population hints.",
        "atomi_role": "Feed candidates to vasp-branch-live, vasp-spin-report, zentropy motif DB, SD/DD, and MLIP active learning.",
        "typical_inputs": "accepted DFT results, spin_energy_run_summary.csv, stage1_branch_summary.csv",
        "typical_outputs": "defect_motif_db.json, active_learning_candidates.csv, calphad_pseudodata.csv",
    },
]

SPECIES_FIELDS = [
    "pseudo_species",
    "element",
    "role",
    "spin_value",
    "charge_state",
    "sublattice",
    "moment_guard",
    "vasp_element",
    "notes",
]

CE_TRAINING_FIELDS = [
    "training_id",
    "source_csv",
    "source_row",
    "run",
    "resolved_run",
    "output_run_dir",
    "structure_path",
    "energy_eV",
    "energy_eV_per_fu",
    "relative_energy_eV_per_fu",
    "physics_status",
    "mag_status",
    "decision",
    "element_order",
    "changed_by_element",
    "atat_candidate_id",
    "motif_family",
    "use_for",
    "notes",
]

QUICK_COMMAND_FIELDS = ["step", "purpose", "command"]
SPIN_GUARD_FIELDS = ["element", "allowed_moments", "tolerance", "role", "notes"]
RELAX_INDEX_FIELDS = [
    "run_index",
    "stage",
    "seed",
    "spin_pattern",
    "volume_scale",
    "linear_scale",
    "volume_A3",
    "volume_per_atom_A3",
    "run_dir",
]
RELAX_SUMMARY_FIELDS = RELAX_INDEX_FIELDS + [
    "energy_eV",
    "relative_energy_eV",
    "energy_kind",
    "status",
    "physics_guard_status",
    "physics_guard_bad_count",
    "mag_status",
    "total_moment",
    "max_abs_moment",
    "element_order",
    "changed_by_element",
    "energy_source",
    "mag_source",
    "warning",
]
VACANCY_CANDIDATE_FIELDS = [
    "candidate_id",
    "kind",
    "poscar",
    "n_Gd",
    "n_O",
    "n_Va",
    "n_partial_element",
    "species_counts_json",
    "site_label",
    "vacancy_fraction",
    "min_vacancy_distance_A",
    "stoichiometry",
    "reasonable_stoichiometry",
    "removed_partial_site_indices",
    "kept_partial_site_indices",
    "notes",
]


@dataclass
class PseudoSpecies:
    pseudo_species: str
    element: str
    role: str
    spin_value: str = ""
    charge_state: str = ""
    sublattice: str = ""
    moment_guard: str = ""
    vasp_element: str = ""
    notes: str = ""

    def row(self) -> dict[str, str]:
        return {
            "pseudo_species": self.pseudo_species,
            "element": self.element,
            "role": self.role,
            "spin_value": self.spin_value,
            "charge_state": self.charge_state,
            "sublattice": self.sublattice,
            "moment_guard": self.moment_guard,
            "vasp_element": self.vasp_element or self.element,
            "notes": self.notes,
        }


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_+-]+", "_", value.strip()).strip("_") or "item"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def split_items(items: list[str] | None) -> list[str]:
    values: list[str] = []
    for raw in items or []:
        for part in str(raw).split(","):
            text = part.strip()
            if text:
                values.append(text)
    return values


def shell_join(parts: list[str]) -> str:
    return " \\\n  ".join(parts)


def parse_moment_specs(items: list[str] | None) -> dict[str, list[float]]:
    moments: dict[str, list[float]] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --moment {raw!r}; use Element=value[,value].")
        element, values = raw.split("=", 1)
        element = element.strip()
        if not element:
            raise ValueError(f"Invalid --moment {raw!r}; missing element.")
        parsed: list[float] = []
        for value in values.split(","):
            text = value.strip()
            if not text:
                continue
            try:
                parsed.append(abs(float(text)))
            except ValueError as exc:
                raise ValueError(f"Invalid magnetic moment value {text!r} in {raw!r}.") from exc
        if not parsed:
            raise ValueError(f"Invalid --moment {raw!r}; no values were provided.")
        existing = moments.setdefault(element, [])
        for value in parsed:
            if value not in existing:
                existing.append(value)
    return moments


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1.0e-10:
        return str(int(round(value)))
    return f"{value:g}"


def build_guard_specs(
    magnetic_elements: list[str],
    nonmagnetic_elements: list[str],
    moment_specs: dict[str, list[float]],
    guard_tol: float,
    nonmagnetic_tol: float,
    explicit_guards: list[str] | None,
) -> list[str]:
    if explicit_guards:
        return split_items(explicit_guards)
    guards: list[str] = []
    for element in magnetic_elements:
        magnitudes = moment_specs.get(element) or [1.0]
        targets: list[str] = []
        for magnitude in magnitudes:
            targets.append(format_number(magnitude))
            targets.append(format_number(-magnitude))
        guards.append(f"{element}={','.join(targets)}@{format_number(guard_tol)}")
    for element in nonmagnetic_elements:
        guards.append(f"{element}=0@{format_number(nonmagnetic_tol)}")
    return guards


def quick_ionic_species_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    magnetic_elements = split_items(args.magnetic_element)
    nonmagnetic_elements = split_items(args.nonmagnetic_element)
    moment_specs = parse_moment_specs(args.moment)
    rows: list[dict[str, str]] = []
    for element in magnetic_elements:
        magnitudes = moment_specs.get(element) or []
        guard = ""
        if magnitudes:
            targets: list[str] = []
            for magnitude in magnitudes:
                targets.extend([format_number(magnitude), format_number(-magnitude)])
            guard = f"{element}={','.join(targets)}@{format_number(args.moment_guard_tol)}"
        rows.append(
            PseudoSpecies(
                safe_name(element),
                element,
                "magnetic_ion",
                "",
                "",
                "magnetic_sublattice",
                guard,
                element,
                "Ionic species for ATAT; spin branches are generated only by magit.",
            ).row()
        )
    for element in nonmagnetic_elements:
        rows.append(
            PseudoSpecies(
                safe_name(element),
                element,
                "nonmagnetic_species",
                "0",
                "",
                "nonmagnetic_sublattice",
                f"{element}=0@{format_number(args.nonmagnetic_tolerance)}",
                element,
                "Nonmagnetic guard used during fail-fast screening.",
            ).row()
        )
    return rows


def quick_spin_guard_rows(args: argparse.Namespace, guards: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    guard_by_element: dict[str, str] = {}
    for guard in guards:
        if "=" in guard:
            element, rest = guard.split("=", 1)
            guard_by_element[element.strip()] = rest.strip()
    for element in split_items(args.magnetic_element):
        guard = guard_by_element.get(element, "")
        moments, _, tol = guard.partition("@")
        rows.append(
            {
                "element": element,
                "allowed_moments": moments,
                "tolerance": tol or format_number(args.moment_guard_tol),
                "role": "magit_spin_branch",
                "notes": "Spin is not generated by ATAT; use magit enum and validate with moment guards.",
            }
        )
    for element in split_items(args.nonmagnetic_element):
        guard = guard_by_element.get(element, "0")
        moments, _, tol = guard.partition("@")
        rows.append(
            {
                "element": element,
                "allowed_moments": moments or "0",
                "tolerance": tol or format_number(args.nonmagnetic_tolerance),
                "role": "nonmagnetic_guard",
                "notes": "Used by vasp-branch-live and vasp-spin-report.",
            }
        )
    return rows


def cell_volume(cell: list[list[float]]) -> float:
    a, b, c = cell
    return abs(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def scaled_cell(cell: list[list[float]], volume_scale: float, scale_kind: str) -> tuple[list[list[float]], float]:
    if volume_scale <= 0:
        raise ValueError("--volume-scale values must be positive.")
    linear = volume_scale if scale_kind == "linear" else volume_scale ** (1.0 / 3.0)
    return [[value * linear for value in vector] for vector in cell], linear


def write_poscar_text(
    comment: str,
    symbols: list[str],
    counts: list[int],
    cell: list[list[float]],
    scaled_positions: list[list[float]],
) -> str:
    lines = [comment, "1.0"]
    lines.extend("  " + "  ".join(f"{value: .16f}" for value in vector) for vector in cell)
    lines.append("  " + "  ".join(symbols))
    lines.append("  " + "  ".join(str(count) for count in counts))
    lines.append("Direct")
    lines.extend("  " + "  ".join(f"{value: .16f}" for value in position) for position in scaled_positions)
    return "\n".join(lines) + "\n"


def replace_or_append_incar_tag(text: str, tag: str, value: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(tag)}\s*=", re.IGNORECASE)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = f"{tag} = {value}"
            return "\n".join(lines) + "\n"
    lines.append(f"{tag} = {value}")
    return "\n".join(lines) + "\n"


def template_incar_with_tags(template: Path, magmom_line: str, isif: int) -> str:
    incar = template / "INCAR"
    if not incar.is_file():
        raise FileNotFoundError(f"Missing template file: {incar}")
    text = incar.read_text(encoding="utf-8", errors="replace")
    text = replace_or_append_incar_tag(text, "MAGMOM", magmom_line.split("=", 1)[1].strip())
    text = replace_or_append_incar_tag(text, "ISPIN", "2")
    text = replace_or_append_incar_tag(text, "ISIF", str(isif))
    return text


def copy_relax_vasp_files(template: Path, run_dir: Path, poscar_text: str, incar_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "POSCAR").write_text(poscar_text, encoding="utf-8")
    (run_dir / "INCAR").write_text(incar_text, encoding="utf-8")
    for name in ("KPOINTS", "POTCAR"):
        src = template / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing template file: {src}")
        shutil.copy2(src, run_dir / name)


def parse_seed_spins(raw: str) -> list[str]:
    seeds = []
    for item in raw.replace(";", ",").split(","):
        seed = item.strip().lower()
        if not seed:
            continue
        if seed not in {"fm", "afm"}:
            raise ValueError("--seed-spins accepts fm, afm, or comma-separated fm,afm.")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("At least one seed spin mode is required.")
    return seeds


def seed_moments(
    species: Any,
    magnetic_elements: list[str],
    moment_specs: dict[str, list[float]],
    seed: str,
) -> tuple[list[float], str]:
    from atomi.vasp.magmom import element_atom_indices

    moments = [0.0] * species.total_atoms
    pattern_tokens: list[str] = []
    for element in magnetic_elements:
        indices = element_atom_indices(species, element)
        if not indices:
            raise ValueError(f"Element {element} not found in POSCAR.")
        magnitude = (moment_specs.get(element) or [1.0])[0]
        signs = [1] * len(indices)
        if seed == "afm":
            signs = [1 if index % 2 == 0 else -1 for index in range(len(indices))]
        for atom_index, sign in zip(indices, signs):
            moments[atom_index] = magnitude * sign
        pattern_tokens.append(f"{element}:{' '.join(format_number(magnitude * sign) for sign in signs)}")
    return moments, "; ".join(pattern_tokens)


def safe_float_label(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def relative_run_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_runlist(path: Path, run_dirs: list[Path], root: Path) -> None:
    path.write_text(
        "\n".join(relative_run_path(run_dir, root) for run_dir in run_dirs) + ("\n" if run_dirs else ""),
        encoding="utf-8",
    )


def parse_relax_index(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Relaxation index not found: {path}")
    return read_csv(path)


def tool_status(tool: str) -> dict[str, str | bool]:
    path = shutil.which(tool)
    return {"available": bool(path), "path": path or ""}


def inspect_atat_environment() -> dict[str, Any]:
    grouped = {
        group: {tool: tool_status(tool) for tool in tools}
        for group, tools in ATAT_TOOLS.items()
    }
    flat = {tool: tool_status(tool) for tools in ATAT_TOOLS.values() for tool in tools}
    ready = {
        "can_generate_sqs": bool(flat.get("mcsqs", {}).get("available")),
        "can_fit_cluster_expansion": bool(flat.get("maps", {}).get("available") or flat.get("mmaps", {}).get("available")),
        "can_compute_correlations": bool(flat.get("corrdump", {}).get("available")),
        "can_convert_structures": bool(
            flat.get("str2poscar", {}).get("available") or flat.get("poscar2str", {}).get("available")
        ),
    }
    suggestions = []
    if not ready["can_generate_sqs"]:
        suggestions.append("mcsqs not found on PATH; load or install ATAT before SQS generation.")
    if not ready["can_fit_cluster_expansion"]:
        suggestions.append("maps/mmaps not found on PATH; cluster-expansion fitting is not ready.")
    if not ready["can_convert_structures"]:
        suggestions.append("str2poscar/poscar2str not found; keep conversion manual or add ATAT tools to PATH.")
    if all(ready.values()):
        suggestions.append("ATAT core tools are visible; Atomi can use this environment as an external workflow bridge.")
    return {
        "schema": STATUS_SCHEMA,
        "path": os.environ.get("PATH", ""),
        "executables": grouped,
        "ready": ready,
        "suggestions": suggestions,
    }


def flatten_atat_tools(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    flat: dict[str, dict[str, Any]] = {}
    for tools in report.get("executables", {}).values():
        if isinstance(tools, dict):
            flat.update(tools)
    return flat


def common_executable_parent(paths: list[str]) -> str:
    parents = sorted({str(Path(path).parent) for path in paths if path})
    if not parents:
        return ""
    if len(parents) == 1:
        return parents[0]
    try:
        return os.path.commonpath(parents)
    except ValueError:
        return parents[0]


def atat_profile_from_report(report: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_atat_tools(report)
    available = {
        tool: str(info.get("path"))
        for tool, info in sorted(flat.items())
        if isinstance(info, dict) and info.get("available") and info.get("path")
    }
    missing = [
        tool
        for tool, info in sorted(flat.items())
        if isinstance(info, dict) and not info.get("available")
    ]
    bin_dir = common_executable_parent(list(available.values()))
    root = str(Path(bin_dir).parent) if bin_dir.endswith("/src") else bin_dir
    return {
        "status": "configured from atat-doctor",
        "root": root,
        "bin": bin_dir,
        "executables": available,
        "missing_executables": missing,
        "ready": report.get("ready", {}),
        "environment": {
            "ATOMI_ATAT_ROOT": root,
            "ATOMI_ATAT_BIN": bin_dir,
        },
        "notes": [
            "Generated by atat-doctor; keep this private/local.",
            "confighpc exports ATOMI_ATAT_ROOT and ATOMI_ATAT_BIN and prepends ATOMI_ATAT_BIN to PATH.",
            "If str2poscar/poscar2str are missing, Atomi can still write direct vacancy-cif POSCAR candidates; arbitrary ATAT structure conversion may need another converter.",
        ],
    }


def default_hpc_config_for_atat() -> Path | None:
    try:
        from atomi.core.doctor import CONFIG_ENV_VAR, DEFAULT_HPC_DIR, LOCAL_CONFIG_PATTERNS
    except Exception:
        CONFIG_ENV_VAR = "ATOMI_HPC_CONFIG"
        DEFAULT_HPC_DIR = Path("~/atomi_hpc").expanduser()
        LOCAL_CONFIG_PATTERNS = ("atomi_hpc_config*.local.json", "*.local.json")
    env = os.environ.get(CONFIG_ENV_VAR)
    if env and Path(env).expanduser().is_file():
        return Path(env).expanduser().resolve()
    root = DEFAULT_HPC_DIR.expanduser()
    if not root.is_dir():
        return None
    matches: list[Path] = []
    for pattern in LOCAL_CONFIG_PATTERNS:
        matches.extend(sorted(root.glob(pattern)))
    matches = [path.resolve() for path in matches if path.is_file()]
    preferred = [path for path in matches if path.name.startswith("atomi_hpc_config")]
    return (preferred or matches)[0] if matches else None


def update_hpc_config_with_atat(config_path: Path | None, report: dict[str, Any]) -> Path:
    target = config_path.expanduser().resolve() if config_path is not None else default_hpc_config_for_atat()
    if target is None:
        raise FileNotFoundError(
            "No HPC config was provided or found. Pass --update-hpc-config ~/atomi_hpc/atomi_hpc_config.<site>.local.json."
        )
    data = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    profiles = data.setdefault("profiles", {})
    profiles["atat"] = atat_profile_from_report(report)
    env = data.setdefault("environment_exports", {})
    profile_env = profiles["atat"].get("environment", {})
    for key, value in profile_env.items():
        if value:
            env[key] = value
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def print_status(report: dict[str, Any]) -> None:
    print("Atomi ATAT bridge status")
    for group, tools in report["executables"].items():
        print(f"{group}:")
        for tool, info in tools.items():
            label = "OK" if info["available"] else "missing"
            print(f"  {tool:<12} {label:<8} {info['path']}")
    print("Ready:")
    for key, value in report["ready"].items():
        print(f"  {key:<28} {value}")
    if report["suggestions"]:
        print("Notes:")
        for item in report["suggestions"]:
            print(f"  - {item}")


def parse_pseudo_species(items: list[str] | None) -> list[PseudoSpecies]:
    species = []
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(
                "Use --pseudo-species LABEL=element,role,state,charge,sublattice,guard,notes"
            )
        label, spec = raw.split("=", 1)
        parts = [part.strip() for part in spec.split(",")]
        while len(parts) < 7:
            parts.append("")
        species.append(
            PseudoSpecies(
                pseudo_species=label.strip(),
                element=parts[0],
                role=parts[1] or "occupational_state",
                spin_value=parts[2],
                charge_state=parts[3],
                sublattice=parts[4],
                moment_guard=parts[5],
                vasp_element=parts[0],
                notes=parts[6],
            )
        )
    return species


def default_pseudo_species(args: argparse.Namespace) -> list[PseudoSpecies]:
    rows: list[PseudoSpecies] = []
    host = args.host
    dopants = args.dopant or []
    oxygen = args.oxygen
    if host:
        if host == "U":
            rows.extend(
                [
                    PseudoSpecies(
                        "U4",
                        "U",
                        "host_valence_state",
                        "4+",
                        "4+",
                        "cation",
                        "U=2,-2@0.7",
                    ),
                    PseudoSpecies(
                        "U5",
                        "U",
                        "host_valence_state",
                        "5+",
                        "5+",
                        "cation",
                        "U=1,-1@0.7",
                    ),
                ]
            )
        else:
            rows.append(PseudoSpecies(safe_name(host), host, "host_ionic_state", "", "", "cation"))
    for dopant in dopants:
        if dopant == "Gd":
            rows.append(
                PseudoSpecies(
                    "Gd3",
                    "Gd",
                    "dopant_ionic_state",
                    "3+",
                    "3+",
                    "cation",
                    "Gd=7,-7@0.6",
                )
            )
        else:
            label = safe_name(dopant)
            rows.append(PseudoSpecies(label, dopant, "dopant_ionic_state", "", "", "cation"))
    if oxygen:
        rows.append(PseudoSpecies(oxygen, oxygen, "anion", "0", "2-", "anion", f"{oxygen}=0@0.25"))
        rows.append(PseudoSpecies(args.vacancy_label, oxygen, "vacancy", "0", "", "anion", "", "oxygen vacancy"))
    return rows


def build_species_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    species = default_pseudo_species(args)
    species.extend(parse_pseudo_species(args.pseudo_species))
    seen: set[str] = set()
    rows = []
    for item in species:
        if item.pseudo_species in seen:
            continue
        seen.add(item.pseudo_species)
        rows.append(item.row())
    return rows


def write_stage_map(path: Path) -> None:
    write_csv(
        path,
        WORKFLOW_STAGES,
        ["stage_id", "atat_role", "atomi_role", "typical_inputs", "typical_outputs"],
    )


def write_bridge_readme(path: Path, args: argparse.Namespace) -> None:
    text = f"""# Atomi-ATAT Bridge

System: {args.system}

This workspace treats ATAT as the lattice-configuration engine and Atomi as the
DFT preparation, spin guard, fail-fast screening, SD/DD, zentropy, and MLIP
bookkeeping layer.

Core logic:

1. Encode cation/anion/defect/ionic-valence states as pseudo-species in
   `pseudo_species_map.csv`.
2. Use ATAT to enumerate, fit cluster expansions, or generate SQS structures.
3. Convert selected ATAT structures to VASP-ready seed folders.
4. Use Atomi `magit` spin branching and fail-fast VASP screening to reject unphysical
   or high-energy configurations.
5. Feed accepted structures and energies to SD/DD, zentropy motif databases,
   MLIP training, or CALPHAD pseudo-data export.

Important: ATAT is spin-blind. Do not rely on ATAT to enumerate spin-up/spin-down
states. Pseudo-species such as U5 or Gd3 are ionic/defect bookkeeping labels.
Generate spin/localization branches with `magit`, then use `vasp-spin-report`
or `vasp-branch-live` moment guards to verify that DFT preserved the intended
local moment/valence character.
"""
    path.write_text(text, encoding="utf-8")


def init_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="atat-bridge init",
        description="Create an ATAT-to-Atomi bridge workspace for defect thermodynamics.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("atat_bridge"))
    parser.add_argument("--system", default="(Gd,U)O2-x")
    parser.add_argument("--host", default="U")
    parser.add_argument("--dopant", action="append", default=["Gd"])
    parser.add_argument("--oxygen", default="O")
    parser.add_argument("--vacancy-label", default="V_O")
    parser.add_argument(
        "--pseudo-species",
        action="append",
        default=[],
        help="Add/override pseudo species: LABEL=element,role,state,charge,sublattice,guard,notes.",
    )
    args = parser.parse_args(argv)
    root = args.outdir.resolve()
    for stage in WORKFLOW_STAGES:
        (root / stage["stage_id"]).mkdir(parents=True, exist_ok=True)
    species_rows = build_species_rows(args)
    write_csv(root / "pseudo_species_map.csv", species_rows, SPECIES_FIELDS)
    write_stage_map(root / "atat_atomi_stage_map.csv")
    write_json(
        root / "atat_bridge_plan.json",
        {
            "schema": SCHEMA,
            "system": args.system,
            "host": args.host,
            "dopants": args.dopant,
            "oxygen": args.oxygen,
            "vacancy_label": args.vacancy_label,
            "workflow_stages": WORKFLOW_STAGES,
            "pseudo_species": species_rows,
            "atat_status": inspect_atat_environment(),
            "handoff_commands": {
                "index_atat_candidates": "atat-bridge index --root 01_motif_search --out atat_candidate_index.csv",
                "build_ce_training_set": "atat-bridge ce-handoff --summary-csv stage1_screen/stage1_branch_summary.csv --outdir 02_sd_dd_cluster_expansion/ce_handoff",
                "fail_fast_screen": "vasp-branch-live --runlist runlist.txt --log-dir . --outdir stage1_screen --moment-guard ...",
                "spin_report": "vasp-spin-report --runlist runlist.txt --log-dir . --output-prefix spin_energy --moment-guard ...",
                "sd_dd": "defect-chem build-defects ... ; defect-chem run ...",
                "zentropy": "zentropy_motif_db index ... ; zentropy-free-energy ... ; zentropy-solve ...",
            },
        },
    )
    write_bridge_readme(root / "ATAT_ATOMI_BRIDGE_NOTES.md", args)
    (root / "01_motif_search" / "README.md").write_text(
        "Place ATAT lat.in, clusters.out, enum/fit outputs, or converted structures here.\n",
        encoding="utf-8",
    )
    (root / "03_sqs_mid_high_concentration" / "README.md").write_text(
        "Place rndstr.in, bestsqs.out, and SQS composition notes here.\n",
        encoding="utf-8",
    )
    print(f"Wrote ATAT bridge workspace : {root}")
    print(f"Pseudo-species map          : {root / 'pseudo_species_map.csv'}")
    print(f"Stage map                   : {root / 'atat_atomi_stage_map.csv'}")
    print(f"Plan                        : {root / 'atat_bridge_plan.json'}")
    print("Next                        : run atat-doctor, then fill lat.in/rndstr.in or index ATAT candidates.")


def classify_candidate(path: Path) -> str:
    name = path.name.lower()
    if "bestsqs" in name or "rndstr" in name:
        return "sqs"
    if name.startswith("str") or "enum" in name:
        return "enumerated_structure"
    if "lat" in name:
        return "lattice_definition"
    return "atat_structure"


def index_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="atat-bridge index",
        description="Index ATAT-generated structure/candidate files for Atomi handoff.",
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("atat_candidate_index.csv"))
    parser.add_argument("--pattern", action="append", default=["bestsqs*.out", "str*.out", "rndstr*.out", "lat.in"])
    parser.add_argument("--target-stage", default="fail_fast")
    args = parser.parse_args(argv)
    candidates: list[Path] = []
    for pattern in args.pattern:
        candidates.extend(path for path in args.root.rglob(pattern) if path.is_file())
    unique = sorted({path.resolve() for path in candidates})
    rows = []
    for index, path in enumerate(unique, start=1):
        kind = classify_candidate(path)
        rows.append(
            {
                "candidate_id": f"atat_{index:04d}",
                "path": str(path),
                "source_kind": kind,
                "target_stage": args.target_stage,
                "atomi_next": "convert_to_vasp_then_vasp-branch-live",
                "notes": "Use pseudo_species_map.csv to restore ionic/defect semantics; generate spin with magit.",
            }
        )
    write_csv(
        args.out,
        rows,
        ["candidate_id", "path", "source_kind", "target_stage", "atomi_next", "notes"],
    )
    print(f"ATAT candidates indexed : {len(rows)}")
    print(f"Candidate index         : {args.out.resolve()}")


def first_present(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return ""


def row_physics_status(row: dict[str, str]) -> str:
    return first_present(row, ["physics_guard_status", "guard", "spin_status"]) or "NOT_APPLIED"


def row_decision(row: dict[str, str]) -> str:
    return first_present(row, ["action", "decision", "status"]) or "unknown"


def row_is_accepted(row: dict[str, str], include_warning: bool, include_unchecked: bool) -> bool:
    physics = row_physics_status(row)
    if physics in {"FAIL", "NO_MATCHED_ELEMENTS"}:
        return False
    if physics in {"NOT_APPLIED", ""} and not include_unchecked:
        return False
    decision = row_decision(row).lower()
    if decision in {"stop", "bad", "error", "missing", "nodir"}:
        return False
    if decision in {"warning", "warn"}:
        return include_warning
    return True


def infer_structure_path(row: dict[str, str]) -> str:
    for key in ("structure_path", "poscar", "POSCAR"):
        if row.get(key):
            return row[key]
    for key in ("output_run_dir", "resolved_run", "run_dir", "run"):
        value = row.get(key)
        if value:
            return str(Path(value) / "POSCAR")
    return ""


def candidate_lookup(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    lookup: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        candidate_id = row.get("candidate_id") or ""
        for key in ("path", "run", "run_dir", "structure_path"):
            value = row.get(key)
            if value:
                lookup[str(Path(value).resolve())] = row
        if candidate_id:
            lookup[candidate_id] = row
    return lookup


def match_candidate(row: dict[str, str], lookup: dict[str, dict[str, str]]) -> dict[str, str] | None:
    for key in ("atat_candidate_id", "candidate_id"):
        value = row.get(key)
        if value and value in lookup:
            return lookup[value]
    for key in ("structure_path", "output_run_dir", "resolved_run", "run_dir", "run"):
        value = row.get(key)
        if not value:
            continue
        path = Path(value)
        keys = [str(path.resolve())]
        if path.is_dir():
            keys.append(str((path / "POSCAR").resolve()))
        for item in keys:
            if item in lookup:
                return lookup[item]
    return None


def ce_training_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    lookup = candidate_lookup(args.candidate_index)
    rows: list[dict[str, Any]] = []
    for source in args.summary_csv:
        for source_row, row in enumerate(read_csv(source), start=1):
            if not row_is_accepted(row, args.include_warning, args.include_unchecked):
                continue
            energy = finite_float(first_present(row, ["energy_eV", "energy", "E_eV"]))
            if energy is None:
                continue
            energy_per_fu = energy / args.formula_units
            candidate = match_candidate(row, lookup)
            structure_path = infer_structure_path(row)
            rows.append(
                {
                    "training_id": f"ce_{len(rows) + 1:05d}",
                    "source_csv": str(source),
                    "source_row": source_row,
                    "run": first_present(row, ["run", "run_dir", "branch_id", "resolved_run"]),
                    "resolved_run": first_present(row, ["resolved_run", "run_dir", "run"]),
                    "output_run_dir": row.get("output_run_dir", ""),
                    "structure_path": structure_path,
                    "energy_eV": f"{energy:.12g}",
                    "energy_eV_per_fu": f"{energy_per_fu:.12g}",
                    "relative_energy_eV_per_fu": "",
                    "physics_status": row_physics_status(row),
                    "mag_status": row.get("mag_status", ""),
                    "decision": row_decision(row),
                    "element_order": row.get("element_order", ""),
                    "changed_by_element": row.get("changed_by_element", ""),
                    "atat_candidate_id": "" if candidate is None else candidate.get("candidate_id", ""),
                    "motif_family": first_present(row, ["motif_family", "frame_id", "family"]),
                    "use_for": args.use_for,
                    "notes": "accepted by Atomi physics/decision filters; convert structure_path to ATAT str.out for CE fitting",
                }
            )
    if rows:
        minimum = min(float(row["energy_eV_per_fu"]) for row in rows)
        for row in rows:
            row["relative_energy_eV_per_fu"] = f"{float(row['energy_eV_per_fu']) - minimum:.12g}"
    return rows


def write_ce_command_notes(path: Path, args: argparse.Namespace) -> None:
    text = f"""# ATAT CE / MC handoff generated by Atomi

# Inputs prepared by Atomi:
#   {args.outdir / 'ce_training_set.csv'}
#   {args.outdir / 'atat_ce_manifest.json'}
#
# Recommended flow:
# 1. Convert accepted structure_path entries to ATAT str.out-like structures.
# 2. Use ATAT maps/mmaps/corrdump to fit a cluster expansion from the accepted
#    DFT energies.
# 3. Use ATAT emc2 or your preferred Monte Carlo runner to estimate
#    composition/T-dependent motif populations.
# 4. Normalize MC/population outputs into:
#      {args.outdir / 'atat_mc_population_handoff_template.csv'}
#    and pass them to zentropy-free-energy / zentropy-solve as motif-family
#    priors or pseudo-data.
#
# These command sketches intentionally avoid fixed flags because ATAT lat.in,
# composition variables, and pseudo-species encoding are system-specific.
# Keep pseudo-species labels synchronized with ionic/defect states in
# pseudo_species_map.csv. Spin is handled separately by magit and validated with
# vasp-branch-live / vasp-spin-report moment guards.
"""
    path.write_text(text, encoding="utf-8")


def ce_handoff_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="atat-bridge ce-handoff",
        description="Build an ATAT cluster-expansion training handoff from accepted Atomi DFT rows.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        action="append",
        required=True,
        help="Atomi stage1_branch_summary.csv or spin_energy_run_summary.csv; repeatable.",
    )
    parser.add_argument("--candidate-index", type=Path, help="Optional atat_candidate_index.csv to link structures.")
    parser.add_argument("--outdir", type=Path, default=Path("atat_ce_handoff"))
    parser.add_argument("--formula-units", type=float, default=1.0, help="Formula units represented by each energy.")
    parser.add_argument("--include-warning", action="store_true", help="Keep Atomi warning rows in the CE candidate set.")
    parser.add_argument(
        "--include-unchecked",
        action="store_true",
        help="Keep rows where no physics moment guard was applied.",
    )
    parser.add_argument("--use-for", default="ce_fit,sd_dd,zentropy", help="Free-form downstream role label.")
    args = parser.parse_args(argv)
    if args.formula_units <= 0:
        raise ValueError("--formula-units must be positive.")
    rows = ce_training_rows(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    training_csv = args.outdir / "ce_training_set.csv"
    write_csv(training_csv, rows, CE_TRAINING_FIELDS)
    interaction_template = args.outdir / "sd_dd_interaction_template.csv"
    write_csv(
        interaction_template,
        [
            {
                "interaction_id": "atat_ce_pair_or_cluster_001",
                "source": "ATAT_CE",
                "species_or_cluster": "",
                "effective_interaction_eV": "",
                "temperature_K": "",
                "composition": "",
                "notes": "Fill from fitted CE/ECI or coarse-grained pair binding values.",
            }
        ],
        ["interaction_id", "source", "species_or_cluster", "effective_interaction_eV", "temperature_K", "composition", "notes"],
    )
    population_template = args.outdir / "atat_mc_population_handoff_template.csv"
    write_csv(
        population_template,
        [
            {
                "T_K": "",
                "composition": "",
                "oxygen_delta": "",
                "motif_family": "",
                "motif_id": "",
                "probability": "",
                "G_eV_per_fu": "",
                "source": "ATAT_MC",
                "notes": "Use as zentropy motif prior or pseudo-data after MC normalization.",
            }
        ],
        ["T_K", "composition", "oxygen_delta", "motif_family", "motif_id", "probability", "G_eV_per_fu", "source", "notes"],
    )
    command_notes = args.outdir / "atat_ce_commands.md"
    write_ce_command_notes(command_notes, args)
    manifest = {
        "schema": "atomi.atat.ce_handoff.v1",
        "summary_csv": [str(path) for path in args.summary_csv],
        "candidate_index": "" if args.candidate_index is None else str(args.candidate_index),
        "formula_units": args.formula_units,
        "n_training_rows": len(rows),
        "filters": {
            "include_warning": args.include_warning,
            "include_unchecked": args.include_unchecked,
            "rejected_physics_status": ["FAIL", "NO_MATCHED_ELEMENTS"],
            "rejected_decisions": ["stop", "bad", "error", "missing", "nodir"],
        },
        "outputs": {
            "ce_training_set": str(training_csv),
            "sd_dd_interaction_template": str(interaction_template),
            "atat_mc_population_handoff_template": str(population_template),
            "atat_ce_commands": str(command_notes),
        },
        "downstream": {
            "sd_dd": "Use interaction template or coarse-grained CE interactions in defect-chem build-defects / solution-model fitting.",
            "zentropy": "Use MC population handoff as motif-family priors or pseudo-data beside microstate G_i(T).",
            "fail_fast": "Return new ATAT-selected ionic/defect structures to magit and vasp-branch-live before expensive DFT.",
        },
    }
    write_json(args.outdir / "atat_ce_manifest.json", manifest)
    print(f"CE training rows       : {len(rows)}")
    print(f"CE training set        : {training_csv.resolve()}")
    print(f"CE manifest            : {(args.outdir / 'atat_ce_manifest.json').resolve()}")
    print(f"MC population template : {population_template.resolve()}")


def prepare_quick_template(args: argparse.Namespace, root: Path) -> tuple[Path, list[str]]:
    template = args.template.expanduser().resolve()
    if not template.is_dir():
        raise FileNotFoundError(f"VASP template directory not found: {template}")
    source_poscar = args.poscar.expanduser().resolve() if args.poscar else template / "POSCAR"
    if not source_poscar.is_file():
        raise FileNotFoundError(
            "POSCAR not found. Pass --poscar, or provide POSCAR inside --template."
        )

    target = root / "00_vasp_template"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_poscar, target / "POSCAR")
    missing: list[str] = []
    for name in ("INCAR", "KPOINTS", "POTCAR"):
        src = template / name
        if src.is_file():
            shutil.copy2(src, target / name)
        else:
            missing.append(name)
    return target, missing


def quick_command_rows(args: argparse.Namespace, guards: list[str]) -> list[dict[str, str]]:
    dopants = split_items(args.dopant)
    hosts = split_items(args.host)
    magnetic_elements = split_items(args.magnetic_element)
    spin_parts = [
        "magit enum",
        "--template 00_vasp_template",
        "--output-root 02_spin_candidates",
        "--runlist runlist.txt",
        "--index spin_index.csv",
    ]
    if dopants or hosts:
        for element in dopants:
            spin_parts.append(f"--dopant {element}")
        for element in hosts:
            spin_parts.append(f"--host {element}")
    else:
        for element in magnetic_elements:
            spin_parts.append(f"--element {element}")
    for raw in args.moment or []:
        spin_parts.append(f"--moment {raw}")
    spin_parts.extend(
        [
            f"--dopant-mode {args.spin_mode}",
            f"--max-configs {args.max_configs}",
        ]
    )
    if args.truncate:
        spin_parts.append("--truncate")

    guard_parts = [f"--moment-guard {guard}" for guard in guards]
    live_parts = [
        "vasp-branch-live",
        "--runlist runlist.txt",
        "--log-dir .",
        "--outdir 03_fail_fast",
        f"--single-frame-id {safe_name(args.system)}_{safe_name(args.supercell)}",
        f"--keep-per-frame {args.keep_per_frame}",
        f"--stopped-after-min {format_number(args.stopped_after_min)}",
        f"--refresh {format_number(args.refresh)}",
    ] + guard_parts
    screen_parts = [
        "vasp-branch-screen",
        "--runlist runlist.txt",
        "--log-dir .",
        "--outdir 03_fail_fast",
        f"--single-frame-id {safe_name(args.system)}_{safe_name(args.supercell)}",
        f"--keep-per-frame {args.keep_per_frame}",
        f"--stopped-after-min {format_number(args.stopped_after_min)}",
    ] + guard_parts
    report_parts = [
        "vasp-spin-report",
        "--runlist runlist.txt",
        "--spin-index spin_index.csv",
        "--log-dir .",
        "--output-prefix 04_final_report/spin_energy",
        f"--stopped-after-min {format_number(args.stopped_after_min)}",
    ] + guard_parts
    ce_parts = [
        "atat-bridge ce-handoff",
        "--summary-csv 03_fail_fast/stage1_branch_summary.csv",
        "--outdir 05_atat_ce_handoff",
        "--include-warning",
    ]
    return [
        {
            "step": "00_atat_status",
            "purpose": "Check whether external ATAT tools are visible on this machine.",
            "command": "atat-doctor",
        },
        {
            "step": "01_optional_atat_index",
            "purpose": "After external ATAT enumeration/SQS files are placed in 01_atat_candidates, index them for Atomi.",
            "command": "atat-bridge index --root 01_atat_candidates --out atat_candidate_index.csv",
        },
        {
            "step": "02_spin_candidates",
            "purpose": "Generate compact spin/local-moment branches from the VASP template.",
            "command": shell_join(spin_parts),
        },
        {
            "step": "03_array_dft",
            "purpose": "Submit your VASP array workflow against runlist.txt; Atomi leaves scheduler submission to your local script.",
            "command": "# sbatch your_vasp_array_script.sbatch runlist.txt",
        },
        {
            "step": "04_live_fail_fast",
            "purpose": "Scan running/stopped branches, show energy and spin health, and write survivor tables.",
            "command": shell_join(live_parts),
        },
        {
            "step": "05_one_shot_screen",
            "purpose": "Run the same branch ranking once after jobs stop or finish.",
            "command": shell_join(screen_parts),
        },
        {
            "step": "06_spin_energy_report",
            "purpose": "Generate final energy vs magnetic-moment tables and plots.",
            "command": shell_join(report_parts),
        },
        {
            "step": "07_atat_ce_handoff",
            "purpose": "Prepare accepted low-energy rows for ATAT cluster expansion or MC population work.",
            "command": shell_join(ce_parts),
        },
    ]


def write_quick_commands(path: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    lines = [
        f"# Quick Materials Optimization: {args.system}",
        "",
        f"Formula: {args.formula}",
        f"Starting cell: {args.supercell}",
        "",
        "Run these commands from this workspace directory.",
        "Spin configurations are generated by Atomi magit, not ATAT.",
        "The ATAT path becomes active after you add ionic/defect/SQS structure outputs to 01_atat_candidates.",
        "",
    ]
    for row in rows:
        lines.extend([f"## {row['step']}", row["purpose"], "", "```bash", row["command"], "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quick_shell(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated command sketch. Review before running on an HPC login node.",
        "",
    ]
    for row in rows:
        lines.extend([f"# {row['step']}: {row['purpose']}", row["command"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def import_ase_atoms() -> tuple[Any, Any]:
    try:
        from ase.io import read, write
    except ImportError as exc:
        raise RuntimeError(
            "ASE is required for materials-opt vacancy-cif. Install atomi with ASE support."
        ) from exc
    return read, write


def parse_cif_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip("'\"")
    if not text or text in {".", "?"}:
        return None
    text = re.sub(r"\([^)]*\)$", "", text)
    try:
        return float(text)
    except ValueError:
        return None


def occupancy_from_ase_info(atoms: Any, default: float = 1.0) -> list[float]:
    occupancies = [default for _ in atoms]
    raw = atoms.info.get("occupancy") or atoms.info.get("occupancies")
    if not isinstance(raw, dict):
        return occupancies
    if not raw or len(raw) != len(atoms) or max((int(key) for key in raw if str(key).isdigit()), default=-1) >= len(atoms):
        return occupancies
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(occupancies):
            continue
        if isinstance(value, dict):
            symbol = atoms[index].symbol
            number = finite_float(value.get(symbol))
            if number is None and value:
                number = max((finite_float(item) or 0.0) for item in value.values())
        else:
            number = finite_float(value)
        if number is not None and 0.0 < number <= 1.0:
            occupancies[index] = float(number)
    return occupancies


def cif_atom_site_occupancies(path: Path) -> list[dict[str, Any]]:
    """Small CIF loop parser for atom-site occupancy fallback.

    ASE handles the structure and symmetry. This parser only recovers explicit
    occupancy values from the input loop when ASE does not expose them.
    """
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows: list[dict[str, Any]] = []
    i = 0
    while i < len(text):
        if text[i].strip().lower() != "loop_":
            i += 1
            continue
        i += 1
        tags: list[str] = []
        while i < len(text) and text[i].strip().startswith("_"):
            tags.append(text[i].strip())
            i += 1
        if not any(tag.startswith("_atom_site_") for tag in tags):
            continue
        while i < len(text):
            line = text[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            if line.lower() == "loop_" or line.startswith("_") or line.startswith("data_"):
                break
            parts = line.replace("'", "").replace('"', "").split()
            if len(parts) >= len(tags):
                item = dict(zip(tags, parts))
                label = item.get("_atom_site_label") or ""
                symbol = item.get("_atom_site_type_symbol") or item.get("_atom_site_label") or ""
                symbol = re.sub(r"[^A-Za-z]+", "", symbol)
                occ = parse_cif_number(item.get("_atom_site_occupancy"))
                fx = parse_cif_number(item.get("_atom_site_fract_x"))
                fy = parse_cif_number(item.get("_atom_site_fract_y"))
                fz = parse_cif_number(item.get("_atom_site_fract_z"))
                if symbol and occ is not None:
                    rows.append({"label": label, "symbol": symbol, "occupancy": occ, "frac": (fx, fy, fz)})
            i += 1
    return rows


def group_cif_sites(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        frac = row.get("frac") or (None, None, None)
        if any(value is None for value in frac):
            continue
        key = tuple(round(float(value) % 1.0, 7) for value in frac)
        group = groups.setdefault(
            key,
            {
                "label": row.get("label") or safe_name(row.get("symbol") or "site"),
                "frac": tuple(float(value) % 1.0 for value in frac),
                "occupants": {},
                "raw_labels": [],
            },
        )
        symbol = row.get("symbol")
        if symbol:
            group["occupants"][symbol] = group["occupants"].get(symbol, 0.0) + float(row["occupancy"])
        if row.get("label"):
            group["raw_labels"].append(row["label"])
    return list(groups.values())


def fractional_close(a: Any, b: Any, tol: float = 1.0e-4) -> bool:
    return all(abs(((float(x) - float(y) + 0.5) % 1.0) - 0.5) <= tol for x, y in zip(a, b))


def equivalent_site_positions(atoms: Any, frac: tuple[float, float, float]) -> list[Any]:
    spacegroup = atoms.info.get("spacegroup")
    if spacegroup is None:
        return [frac]
    try:
        sites, _ = spacegroup.equivalent_sites([frac], symprec=1.0e-3, onduplicates="keep")
    except Exception:
        return [frac]
    return list(sites)


def match_cif_group_indices(atoms: Any, group: dict[str, Any]) -> list[int]:
    positions = equivalent_site_positions(atoms, group["frac"])
    allowed = set(group["occupants"])
    scaled = atoms.get_scaled_positions(wrap=True)
    matched: list[int] = []
    for index, atom in enumerate(atoms):
        if allowed and atom.symbol not in allowed:
            continue
        if any(fractional_close(scaled[index], site) for site in positions):
            matched.append(index)
    return matched


def site_spec_from_occupants(
    occupants: dict[str, float],
    vacancy_label: str,
    target_element: str | None = None,
    target_occupancy: float | None = None,
) -> tuple[dict[str, float], str]:
    clean = {symbol: float(value) for symbol, value in occupants.items() if float(value) > 1.0e-10}
    if target_element and target_occupancy is not None and target_element in clean:
        scale_rest = max(0.0, 1.0 - float(target_occupancy))
        other_total = sum(value for symbol, value in clean.items() if symbol != target_element)
        adjusted = {target_element: float(target_occupancy)}
        if other_total > 1.0e-12:
            for symbol, value in clean.items():
                if symbol != target_element:
                    adjusted[symbol] = scale_rest * value / other_total
        clean = adjusted
    total = sum(clean.values())
    if total < 1.0 - 1.0e-10:
        clean[vacancy_label] = 1.0 - total
    if vacancy_label not in clean and len(clean) == 1:
        symbol, value = next(iter(clean.items()))
        if math.isclose(value, 1.0, abs_tol=1.0e-10):
            return clean, symbol
    spec = ",".join(f"{symbol}={value:.12g}" for symbol, value in clean.items())
    return clean, spec


def infer_site_occupancy_specs_from_cif(
    atoms: Any,
    cif_path: Path,
    partial_elements: set[str],
    target_occupancy: float | None,
    vacancy_label: str,
    site_label: str | None = None,
) -> tuple[list[float], list[str], list[str], list[dict[str, Any]]]:
    occupancies = [1.0 for _ in atoms]
    site_labels = ["" for _ in atoms]
    site_specs = [atom.symbol for atom in atoms]
    raw_rows = cif_atom_site_occupancies(cif_path)
    groups = group_cif_sites(raw_rows)
    annotated_groups: list[dict[str, Any]] = []
    for group in groups:
        labels = {str(label) for label in group.get("raw_labels", [])}
        site_elements = set(group["occupants"])
        element_allowed = not partial_elements or bool(site_elements & partial_elements)
        selected_by_label = site_label is not None and site_label in labels
        selected_by_partial = site_label is None and element_allowed and sum(group["occupants"].values()) < 0.999
        selected = (selected_by_label and element_allowed) or selected_by_partial
        target_element = None
        if selected and target_occupancy is not None:
            allowed_occupants = [symbol for symbol in group["occupants"] if not partial_elements or symbol in partial_elements]
            if len(allowed_occupants) != 1:
                raise ValueError(
                    "--target-occupancy needs exactly one matching species on each selected site. "
                    "Use --partial-element or --site-label to make the override unambiguous."
                )
            target_element = allowed_occupants[0]
        occupants, spec = site_spec_from_occupants(
            group["occupants"],
            vacancy_label,
            target_element=target_element,
            target_occupancy=target_occupancy if selected else None,
        )
        indices = match_cif_group_indices(atoms, group)
        for index in indices:
            site_labels[index] = group["label"]
            if selected:
                site_specs[index] = spec
                occupancies[index] = occupants.get(atoms[index].symbol, 1.0)
        annotated_groups.append(
            {
                **group,
                "occupants": occupants,
                "site_spec": spec,
                "indices": indices,
                "selected": selected,
            }
        )
    if groups:
        return occupancies, site_labels, site_specs, annotated_groups
    ase_occupancies = occupancy_from_ase_info(atoms)
    return ase_occupancies, site_labels, site_specs, annotated_groups


def parse_repeat(value: str | None) -> tuple[int, int, int] | None:
    if value is None or value.lower() == "auto":
        return None
    parts = [part for part in re.split(r"[x, ]+", value.strip()) if part]
    if len(parts) != 3:
        raise ValueError("--supercell must be auto or three integers such as 1 1 5 / 1x1x5.")
    repeat = tuple(int(part) for part in parts)
    if any(item <= 0 for item in repeat):
        raise ValueError("--supercell repeat values must be positive.")
    return repeat  # type: ignore[return-value]


def choose_integer_repeat(requirements: list[tuple[int, float]], max_repeat: int) -> tuple[int, int, int]:
    if not requirements:
        return (1, 1, 1)
    best: tuple[int, int, int] | None = None
    best_volume: int | None = None
    fractions = [(n_sites, Fraction(occupancy).limit_denominator(128)) for n_sites, occupancy in requirements]
    for a in range(1, max_repeat + 1):
        for b in range(1, max_repeat + 1):
            for c in range(1, max_repeat + 1):
                volume = a * b * c
                if any((n_sites * volume * frac.numerator) % frac.denominator != 0 for n_sites, frac in fractions):
                    continue
                if best is None or volume < (best_volume or 10**9) or (
                    volume == best_volume and max(a, b, c) < max(best)
                ):
                    best = (a, b, c)
                    best_volume = volume
    if best is None:
        raise ValueError(
            "Could not find a compact integer supercell for the partial occupancy. "
            "Pass --supercell explicitly or increase --max-repeat."
        )
    return best


def repeated_occupancies(occupancies: list[float], repeat: tuple[int, int, int]) -> list[float]:
    return occupancies * (repeat[0] * repeat[1] * repeat[2])


def repeat_metadata(values: list[Any], repeat: tuple[int, int, int]) -> list[Any]:
    return list(values) * (repeat[0] * repeat[1] * repeat[2])


def site_spec_contains_vacancy(spec: str, vacancy_label: str) -> bool:
    return f"{vacancy_label}=" in spec


def distance_between_indices(atoms: Any, i: int, j: int) -> float:
    return float(atoms.get_distance(i, j, mic=True))


def min_pair_distance(atoms: Any, indices: list[int]) -> float | None:
    if len(indices) < 2:
        return None
    return min(distance_between_indices(atoms, i, j) for i, j in itertools.combinations(indices, 2))


def greedy_vacancy_set(atoms: Any, candidates: list[int], n_vacancy: int, mode: str) -> list[int]:
    if n_vacancy >= len(candidates):
        return list(candidates)
    if n_vacancy <= 0:
        return []
    n_keep = len(candidates) - n_vacancy
    if n_keep < n_vacancy:
        keep_mode = "clustered" if mode == "separated" else "separated"
        kept = set(greedy_vacancy_set(atoms, candidates, n_keep, keep_mode))
        return sorted(set(candidates) - kept)
    selected = [candidates[0]]
    remaining = candidates[1:]
    while len(selected) < n_vacancy and remaining:
        if mode == "clustered":
            chosen = min(
                remaining,
                key=lambda idx: min(distance_between_indices(atoms, idx, item) for item in selected),
            )
        else:
            chosen = max(
                remaining,
                key=lambda idx: min(distance_between_indices(atoms, idx, item) for item in selected),
            )
        selected.append(chosen)
        remaining.remove(chosen)
    return sorted(selected)


def random_vacancy_set(candidates: list[int], n_vacancy: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if n_vacancy >= len(candidates):
        return list(candidates)
    return sorted(rng.sample(candidates, n_vacancy))


def vacancy_groups_from_metadata(
    atoms: Any,
    occupancies: list[float],
    site_labels: list[str],
    site_specs: list[str],
    vacancy_label: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, (atom, occupancy, label, spec) in enumerate(zip(atoms, occupancies, site_labels, site_specs)):
        if not site_spec_contains_vacancy(spec, vacancy_label):
            continue
        if not (0.0 <= occupancy < 0.999):
            continue
        key = (label or f"{atom.symbol}_partial", atom.symbol, spec)
        group = groups.setdefault(
            key,
            {
                "label": key[0],
                "element": atom.symbol,
                "site_spec": spec,
                "occupancy": float(occupancy),
                "indices": [],
            },
        )
        group["indices"].append(index)
    return list(groups.values())


def vacancy_requirements(groups: list[dict[str, Any]]) -> list[tuple[int, float]]:
    return [
        (len(group.get("indices", [])), float(group.get("occupancy", 0.0)))
        for group in groups
        if group.get("indices") and float(group.get("occupancy", 0.0)) < 0.999
    ]


def vacancy_count_for_group(group: dict[str, Any]) -> int:
    indices = group.get("indices", [])
    occupancy = float(group.get("occupancy", 0.0))
    keep = int(round(len(indices) * occupancy))
    if not math.isclose(keep, len(indices) * occupancy, abs_tol=1.0e-6):
        raise ValueError(
            f"Selected supercell does not make occupancy integer for site {group.get('label')} "
            f"({len(indices)} sites at occupancy {occupancy:g})."
        )
    return len(indices) - keep


def choose_vacancies_by_group(atoms: Any, groups: list[dict[str, Any]], mode: str, seed: int) -> list[int]:
    vacancies: list[int] = []
    for offset, group in enumerate(groups):
        indices = sorted(int(index) for index in group.get("indices", []))
        n_vacancy = vacancy_count_for_group(group)
        if mode == "sqs_random_like":
            chosen = random_vacancy_set(indices, n_vacancy, seed + offset)
        else:
            chosen = greedy_vacancy_set(atoms, indices, n_vacancy, "clustered" if mode == "vacancy_clustered" else "separated")
        vacancies.extend(chosen)
    return sorted(vacancies)


def composition_string(atoms: Any) -> str:
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return " ".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))


def atoms_without_indices(atoms: Any, remove: list[int]) -> Any:
    clean = atoms.copy()
    for index in sorted(remove, reverse=True):
        del clean[index]
    return clean


def ensure_isym_zero(incar_text: str) -> str:
    lines = incar_text.splitlines()
    replaced = False
    out: list[str] = []
    for line in lines:
        if re.match(r"^\s*ISYM\s*=", line, flags=re.IGNORECASE):
            out.append("ISYM = 0")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append("ISYM = 0")
    return "\n".join(out).rstrip() + "\n"


def copy_vasp_template_for_vacancy(template: Path | None, run_dir: Path) -> None:
    if template is None:
        return
    if not template.is_dir():
        raise FileNotFoundError(f"VASP template directory not found: {template}")
    for item in template.iterdir():
        if item.name == "POSCAR":
            continue
        target = run_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    incar = run_dir / "INCAR"
    if incar.is_file():
        incar.write_text(ensure_isym_zero(incar.read_text(encoding="utf-8")), encoding="utf-8")


def write_atat_rndstr(
    path: Path,
    atoms: Any,
    site_specs: list[str],
) -> None:
    cell = atoms.cell.array
    scaled = atoms.get_scaled_positions(wrap=True)
    lines = [
        "# Generated by atomi materials-opt vacancy-cif",
        "# Fixed sites are written as single species; disordered sites use ATAT species fractions.",
    ]
    for vector in cell:
        lines.append(" ".join(f"{float(value):.12f}" for value in vector))
    for index, atom in enumerate(atoms):
        species = site_specs[index] if index < len(site_specs) else atom.symbol
        lines.append(
            " ".join(f"{float(value):.12f}" for value in scaled[index]) + f" {species}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_atat_vacancy_scripts(atat_dir: Path, args: argparse.Namespace) -> None:
    mcsqs_parts = ["mcsqs", f"-n={args.atat_atoms}"]
    if args.mcsqs_time is not None:
        mcsqs_parts.append(f"-T={args.mcsqs_time:g}")
    script = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run this inside the ATAT handoff directory.",
        "# bestsqs.out still contains the vacancy pseudo-species; convert/remove Va before VASP.",
        " ".join(mcsqs_parts),
        "if command -v str2poscar >/dev/null 2>&1 && [ -f bestsqs.out ]; then",
        "  str2poscar < bestsqs.out > POSCAR_with_Va",
        "fi",
    ]
    path = atat_dir / "run_mcsqs.sh"
    path.write_text("\n".join(script) + "\n", encoding="utf-8")
    path.chmod(0o755)
    (atat_dir / "README.md").write_text(
        "This folder contains ATAT inputs for the partially occupied vacancy sublattice.\n"
        "Use `./run_mcsqs.sh` to generate `bestsqs.out`. Atomi's direct candidates already "
        "remove vacancy pseudo-atoms from POSCAR; ATAT-converted structures should be cleaned "
        "the same way before VASP.\n",
        encoding="utf-8",
    )


def vacancy_candidate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="materials-opt vacancy-cif",
        description=(
            "Convert CIF partial-occupancy vacancy sites into explicit atom/vacancy "
            "VASP POSCAR candidates and ATAT rndstr.in handoff files."
        ),
    )
    parser.add_argument("--cif", type=Path, required=True, help="Input CIF with partial occupancy.")
    parser.add_argument("--outdir", type=Path, default=Path("VACANCY_CIF_CANDIDATES"))
    parser.add_argument(
        "--partial-element",
        action="append",
        default=[],
        help=(
            "Optional element filter for partial vacancy sites. Repeatable. "
            "If omitted, every CIF site with occupancy sum < 1 is used."
        ),
    )
    parser.add_argument("--site-label", help="Optional CIF atom-site label to select, e.g. O2.")
    parser.add_argument("--target-occupancy", type=float, help="Override CIF occupancy for the partial sublattice.")
    parser.add_argument("--vacancy-label", default="Va", help="ATAT vacancy pseudo-species label.")
    parser.add_argument("--supercell", default="auto", help="auto or repeat such as 1x1x5.")
    parser.add_argument("--max-repeat", type=int, default=6, help="Largest repeat searched for --supercell auto.")
    parser.add_argument("--vasp-template", type=Path, help="Optional VASP template copied beside each POSCAR.")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--atat-atoms", type=int, default=0, help="Optional mcsqs -n target. Default: full supercell atom count.")
    parser.add_argument("--mcsqs-time", type=float, help="Optional mcsqs -T wall-clock limit.")
    parser.add_argument("--run-mcsqs", action="store_true", help="Run mcsqs immediately if available.")
    args = parser.parse_args(argv)

    if args.target_occupancy is not None and not (0.0 < args.target_occupancy <= 1.0):
        raise ValueError("--target-occupancy must be in (0, 1].")
    read, write = import_ase_atoms()
    cif = args.cif.expanduser().resolve()
    partial_elements = set(split_items(args.partial_element))
    atoms0 = read(cif, fractional_occupancies=True, store_tags=True)
    occupancies0, labels0, specs0, site_groups = infer_site_occupancy_specs_from_cif(
        atoms0,
        cif,
        partial_elements,
        args.target_occupancy,
        args.vacancy_label,
        args.site_label,
    )
    initial_vacancy_groups = vacancy_groups_from_metadata(
        atoms0,
        occupancies0,
        labels0,
        specs0,
        args.vacancy_label,
    )
    if not initial_vacancy_groups:
        raise ValueError(
            "No vacancy-bearing partial site was detected. "
            "Pass --site-label and/or --target-occupancy if CIF parsing does not expose the intended site."
        )
    selected_labels = sorted({str(group["label"]) for group in initial_vacancy_groups if group.get("label")})
    repeat = parse_repeat(args.supercell) or choose_integer_repeat(
        vacancy_requirements(initial_vacancy_groups),
        args.max_repeat,
    )
    atoms = atoms0.repeat(repeat)
    occupancies = repeated_occupancies(occupancies0, repeat)
    site_labels = repeat_metadata(labels0, repeat)
    site_specs = repeat_metadata(specs0, repeat)
    vacancy_groups = vacancy_groups_from_metadata(
        atoms,
        occupancies,
        site_labels,
        site_specs,
        args.vacancy_label,
    )
    vacancy_total = sum(vacancy_count_for_group(group) for group in vacancy_groups)
    keep_total = sum(len(group["indices"]) - vacancy_count_for_group(group) for group in vacancy_groups)

    root = args.outdir.expanduser().resolve()
    candidates_dir = root / "candidates"
    atat_dir = root / "atat"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    write_atat_rndstr(
        atat_dir / "rndstr.in",
        atoms,
        site_specs,
    )
    if args.atat_atoms <= 0:
        args.atat_atoms = len(atoms)
    write_atat_vacancy_scripts(atat_dir, args)

    vacancy_sets = {
        "vacancy_separated": choose_vacancies_by_group(atoms, vacancy_groups, "vacancy_separated", args.seed),
        "vacancy_clustered": choose_vacancies_by_group(atoms, vacancy_groups, "vacancy_clustered", args.seed),
        "sqs_random_like": choose_vacancies_by_group(atoms, vacancy_groups, "sqs_random_like", args.seed),
    }
    rows: list[dict[str, Any]] = []
    runlist: list[str] = []
    for index, (kind, vacancies) in enumerate(vacancy_sets.items(), start=1):
        case_id = f"{index:02d}_{kind}"
        run_dir = candidates_dir / case_id
        run_dir.mkdir(parents=True, exist_ok=True)
        final_atoms = atoms_without_indices(atoms, vacancies)
        write(run_dir / "POSCAR", final_atoms, format="vasp", direct=True, vasp5=True, sort=False)
        copy_vasp_template_for_vacancy(args.vasp_template.expanduser().resolve() if args.vasp_template else None, run_dir)
        all_partial_indices = sorted({index for group in vacancy_groups for index in group["indices"]})
        kept = sorted(set(all_partial_indices) - set(vacancies))
        min_vv = min_pair_distance(atoms, vacancies)
        vacancy_fraction = len(vacancies) / len(all_partial_indices) if all_partial_indices else 0.0
        reasonable = len(vacancies) == vacancy_total
        count_symbols = final_atoms.get_chemical_symbols()
        counts = {symbol: count_symbols.count(symbol) for symbol in sorted(set(count_symbols))}
        rows.append(
            {
                "candidate_id": case_id,
                "kind": kind,
                "poscar": str((run_dir / "POSCAR").resolve()),
                "n_Gd": count_symbols.count("Gd"),
                "n_O": count_symbols.count("O"),
                "n_Va": len(vacancies),
                "n_partial_element": sum(count_symbols.count(element) for element in (partial_elements or {group["element"] for group in vacancy_groups})),
                "species_counts_json": json.dumps(counts, sort_keys=True),
                "site_label": ",".join(selected_labels),
                "vacancy_fraction": f"{vacancy_fraction:.12g}",
                "min_vacancy_distance_A": "" if min_vv is None else f"{min_vv:.8f}",
                "stoichiometry": composition_string(final_atoms),
                "reasonable_stoichiometry": "true" if reasonable else "false",
                "removed_partial_site_indices": " ".join(str(item + 1) for item in vacancies),
                "kept_partial_site_indices": " ".join(str(item + 1) for item in kept),
                "notes": "Vacancy pseudo-atoms removed; use ISYM=0 for VASP relaxation.",
            }
        )
        runlist.append(str(run_dir.resolve()))

    write_csv(root / "vacancy_candidate_index.csv", rows, VACANCY_CANDIDATE_FIELDS)
    (root / "runlist.txt").write_text("\n".join(runlist) + "\n", encoding="utf-8")
    write_json(
        root / "vacancy_cif_plan.json",
        {
            "schema": "atomi.materials.vacancy_cif.v1",
            "source_cif": str(cif),
            "partial_elements": sorted(partial_elements),
            "site_label": args.site_label,
            "selected_site_labels": selected_labels,
            "vacancy_label": args.vacancy_label,
            "repeat": repeat,
            "n_partial_sites": sum(len(group["indices"]) for group in vacancy_groups),
            "n_keep_partial_element": keep_total,
            "n_vacancy": vacancy_total,
            "vacancy_groups": [
                {
                    "label": group["label"],
                    "element": group["element"],
                    "site_spec": group["site_spec"],
                    "occupancy": group["occupancy"],
                    "multiplicity": len(group["indices"]),
                    "n_keep": len(group["indices"]) - vacancy_count_for_group(group),
                    "n_vacancy": vacancy_count_for_group(group),
                }
                for group in vacancy_groups
            ],
            "site_groups": [
                {
                    "label": group.get("label"),
                    "raw_labels": group.get("raw_labels"),
                    "occupants": group.get("occupants"),
                    "site_spec": group.get("site_spec"),
                    "multiplicity": len(group.get("indices", [])),
                    "selected": group.get("selected"),
                }
                for group in site_groups
            ],
            "outputs": {
                "rndstr": str(atat_dir / "rndstr.in"),
                "run_mcsqs": str(atat_dir / "run_mcsqs.sh"),
                "candidate_index": str(root / "vacancy_candidate_index.csv"),
                "runlist": str(root / "runlist.txt"),
            },
            "notes": [
                "Final POSCAR files never contain fractional occupancy or vacancy pseudo-atoms.",
                "ATAT handles atom/Va occupational sublattices in rndstr.in; VASP receives only explicit atoms.",
                "Use ISYM=0 for subsequent VASP relaxations.",
            ],
        },
    )

    if args.run_mcsqs:
        if shutil.which("mcsqs") is None:
            raise RuntimeError("mcsqs was requested with --run-mcsqs, but it is not on PATH.")
        subprocess.run(["mcsqs", f"-n={args.atat_atoms}"], cwd=atat_dir, check=True)

    print(f"Vacancy CIF workspace : {root}")
    print(f"Vacancy-bearing sites : {sum(len(group['indices']) for group in vacancy_groups)}")
    print(f"Site labels           : {','.join(selected_labels) or 'unknown'}")
    for group in vacancy_groups:
        print(
            f"  {group['label']} {group['element']}  "
            f"occ={group['occupancy']:.12g} keep={len(group['indices']) - vacancy_count_for_group(group)} "
            f"vacancy={vacancy_count_for_group(group)}"
        )
    print(f"Supercell repeat      : {repeat[0]} {repeat[1]} {repeat[2]}")
    print(f"ATAT rndstr.in        : {atat_dir / 'rndstr.in'}")
    print(f"Candidates            : {len(rows)}")
    for row in rows:
        print(
            f"  {row['candidate_id']:>18s}  {row['stoichiometry']}  "
            f"Va={row['n_Va']}  min Va-Va={row['min_vacancy_distance_A'] or 'n/a'} A"
        )
    print(f"Candidate index       : {root / 'vacancy_candidate_index.csv'}")
    print(f"Runlist               : {root / 'runlist.txt'}")


def quick_opt_main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in {"vacancy-cif", "vacany-cif", "cif-vacancy", "partial-occupancy"}:
        vacancy_candidate_main(argv[1:])
        return
    if argv and argv[0] in {"relax-seeds", "relax_seed", "relax-seed"}:
        relax_seeds_main(argv[1:])
        return
    if argv and argv[0] in {"relax-summary", "summary", "summarize-relax"}:
        relax_summary_main(argv[1:])
        return

    parser = argparse.ArgumentParser(
        prog="materials-opt",
        description="Create a compact Atomi spin and ATAT ionic/lattice optimization scaffold.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("materials_opt_quick"))
    parser.add_argument("--system", default="UC2")
    parser.add_argument("--formula", default="UC2")
    parser.add_argument("--supercell", default="2x1x1")
    parser.add_argument("--poscar", type=Path, help="Starting POSCAR. If omitted, use POSCAR from --template.")
    parser.add_argument("--template", type=Path, default=Path("VASP_TEMPLATE"))
    parser.add_argument("--magnetic-element", action="append", default=[], help="Magnetic element to branch, repeatable.")
    parser.add_argument("--nonmagnetic-element", action="append", default=[], help="Nonmagnetic element guard, repeatable.")
    parser.add_argument("--dopant", action="append", default=[], help="Optional dopant-style magnetic element for magit enum.")
    parser.add_argument("--host", action="append", default=[], help="Optional host magnetic element for magit enum.")
    parser.add_argument("--moment", action="append", default=[], help="Moment magnitude, e.g. U=2 or U=2,1.")
    parser.add_argument("--moment-guard", action="append", default=[], help="Explicit moment guard, e.g. U=2,-2@0.7.")
    parser.add_argument("--moment-guard-tol", type=float, default=0.7)
    parser.add_argument("--nonmagnetic-tolerance", type=float, default=0.25)
    parser.add_argument("--spin-mode", choices=("all", "fm", "afm", "both"), default="all")
    parser.add_argument("--max-configs", type=int, default=16)
    parser.add_argument("--no-truncate", dest="truncate", action="store_false", default=True)
    parser.add_argument("--keep-per-frame", type=int, default=2)
    parser.add_argument("--stopped-after-min", type=float, default=15.0)
    parser.add_argument("--refresh", type=float, default=10.0)
    args = parser.parse_args(argv)

    moment_specs = parse_moment_specs(args.moment)
    magnetic_elements = split_items(args.magnetic_element)
    for element in split_items(args.dopant) + split_items(args.host):
        if element not in magnetic_elements:
            magnetic_elements.append(element)
    if not magnetic_elements:
        magnetic_elements = list(moment_specs)
    if not magnetic_elements:
        raise ValueError("Provide --magnetic-element or --moment Element=value for spin branching.")
    args.magnetic_element = magnetic_elements
    nonmagnetic_elements = split_items(args.nonmagnetic_element)

    root = args.outdir.expanduser().resolve()
    for name in (
        "01_atat_candidates",
        "02_spin_candidates",
        "03_fail_fast",
        "04_final_report",
        "05_atat_ce_handoff",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    template_path, missing_template_files = prepare_quick_template(args, root)
    guards = build_guard_specs(
        magnetic_elements,
        nonmagnetic_elements,
        moment_specs,
        args.moment_guard_tol,
        args.nonmagnetic_tolerance,
        args.moment_guard,
    )
    species_rows = quick_ionic_species_rows(args)
    spin_guard_rows = quick_spin_guard_rows(args, guards)
    command_rows = quick_command_rows(args, guards)

    write_csv(root / "pseudo_species_map.csv", species_rows, SPECIES_FIELDS)
    write_csv(root / "spin_guard_map.csv", spin_guard_rows, SPIN_GUARD_FIELDS)
    write_csv(root / "quick_opt_commands.csv", command_rows, QUICK_COMMAND_FIELDS)
    write_quick_commands(root / "QUICK_OPT_COMMANDS.md", command_rows, args)
    write_quick_shell(root / "quick_opt_commands.sh", command_rows)
    write_json(
        root / "quick_opt_plan.json",
        {
            "schema": QUICK_OPT_SCHEMA,
            "system": args.system,
            "formula": args.formula,
            "supercell": args.supercell,
            "magnetic_elements": magnetic_elements,
            "nonmagnetic_elements": nonmagnetic_elements,
            "moment_specs": {
                key: [format_number(value) for value in values]
                for key, values in moment_specs.items()
            },
            "moment_guards": guards,
            "spin_owner": "Atomi magit enum and Atomi VASP moment guards",
            "atat_owner": (
                "Ionic, occupational, vacancy, SQS, cluster-expansion, and Monte Carlo "
                "configuration workflows"
            ),
            "max_configs": args.max_configs,
            "spin_mode": args.spin_mode,
            "template": str(template_path),
            "missing_template_files": missing_template_files,
            "outputs": {
                "commands": str(root / "QUICK_OPT_COMMANDS.md"),
                "shell_commands": str(root / "quick_opt_commands.sh"),
                "spin_guard_map": str(root / "spin_guard_map.csv"),
                "runlist_after_spin_enum": str(root / "runlist.txt"),
                "spin_index_after_spin_enum": str(root / "spin_index.csv"),
                "fail_fast_summary": str(root / "03_fail_fast" / "stage1_branch_summary.csv"),
                "stage2_survivors": str(root / "03_fail_fast" / "stage2_survivors_runlist.txt"),
                "spin_report": str(root / "04_final_report" / "spin_energy_run_summary.csv"),
            },
            "atat_status": inspect_atat_environment(),
            "notes": [
                "For a stoichiometric UC2 2x1x1 demo, spin ordering starts with magit enum, not ATAT.",
                "Use ATAT in 01_atat_candidates when occupational, vacancy, or SQS lattice candidates are needed.",
                "Lowest-energy selection should use physics-accepted rows from 03_fail_fast or 04_final_report.",
            ],
        },
    )
    (root / "01_atat_candidates" / "README.md").write_text(
        "Optional ATAT area. Put lat.in/rndstr.in and ATAT structure outputs here, then run:\n"
        "  atat-bridge index --root 01_atat_candidates --out atat_candidate_index.csv\n",
        encoding="utf-8",
    )

    print(f"Quick optimization workspace : {root}")
    print(f"VASP template copy           : {template_path}")
    print(f"Ionic pseudo-species map     : {root / 'pseudo_species_map.csv'}")
    print(f"Spin guard map               : {root / 'spin_guard_map.csv'}")
    print(f"Command guide                : {root / 'QUICK_OPT_COMMANDS.md'}")
    print(f"Plan                         : {root / 'quick_opt_plan.json'}")
    if missing_template_files:
        print(f"Template warning             : missing {', '.join(missing_template_files)}")
    print("Next                         : cd into the workspace and run the magit enum command from QUICK_OPT_COMMANDS.md.")


def relax_seeds_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="materials-opt relax-seeds",
        description="Prepare FM/AFM seed and ISIF=2 volume-scan VASP folders.",
    )
    parser.add_argument("--poscar", type=Path, default=Path("POSCAR"))
    parser.add_argument("--template", type=Path, default=Path("VASP_TEMPLATE"))
    parser.add_argument("--outdir", type=Path, default=Path("RELAX_SEEDS_OPT"))
    parser.add_argument("--system", default="material")
    parser.add_argument("--formula", default="")
    parser.add_argument("--magnetic-element", action="append", required=True)
    parser.add_argument("--nonmagnetic-element", action="append", default=[])
    parser.add_argument("--moment", action="append", required=True, help="Moment magnitude, e.g. U=2.")
    parser.add_argument("--seed-spins", default="fm,afm", help="Comma-separated seed modes: fm,afm.")
    parser.add_argument(
        "--volume-scale",
        type=float,
        nargs="+",
        default=[0.94, 0.97, 1.0, 1.03, 1.06],
        help="Relative volume scale factors by default.",
    )
    parser.add_argument("--scale-kind", choices=("volume", "linear"), default="volume")
    parser.add_argument("--isif-volume", type=int, default=2)
    parser.add_argument("--isif-shape", type=int, default=3)
    parser.add_argument("--moment-guard", action="append", default=[])
    parser.add_argument("--moment-guard-tol", type=float, default=0.7)
    parser.add_argument("--nonmagnetic-tolerance", type=float, default=0.25)
    parser.add_argument("--decimals", type=int, default=3)
    args = parser.parse_args(argv)

    from atomi.vasp.magmom import format_magmom_line, read_poscar_structure

    template = args.template.expanduser().resolve()
    if not template.is_dir():
        raise FileNotFoundError(f"VASP template directory not found: {template}")
    structure = read_poscar_structure(args.poscar.expanduser().resolve())
    species = structure.species
    moment_specs = parse_moment_specs(args.moment)
    magnetic_elements = split_items(args.magnetic_element)
    nonmagnetic_elements = split_items(args.nonmagnetic_element)
    seeds = parse_seed_spins(args.seed_spins)
    guards = build_guard_specs(
        magnetic_elements,
        nonmagnetic_elements,
        moment_specs,
        args.moment_guard_tol,
        args.nonmagnetic_tolerance,
        args.moment_guard,
    )

    root = args.outdir.expanduser().resolve()
    seed_root = root / "01_seed_spins"
    volume_root = root / "02_volume_isif2"
    shape_root = root / "03_shape_isif3"
    summary_root = root / "04_summary"
    for path in (seed_root, volume_root, shape_root, summary_root):
        path.mkdir(parents=True, exist_ok=True)

    original_volume = cell_volume(structure.cell)
    rows: list[dict[str, Any]] = []
    run_dirs: list[Path] = []
    run_index = 0
    for seed in seeds:
        moments, pattern_text = seed_moments(species, magnetic_elements, moment_specs, seed)
        magmom_line = format_magmom_line(
            species,
            moments,
            selected_elements=magnetic_elements + nonmagnetic_elements,
            decimals=args.decimals,
            compact_zero=True,
        )
        seed_poscar = write_poscar_text(
            f"{args.system} {seed} seed",
            species.symbols,
            species.counts,
            structure.cell,
            structure.scaled_positions,
        )
        seed_incar = template_incar_with_tags(template, magmom_line, args.isif_volume)
        copy_relax_vasp_files(template, seed_root / seed, seed_poscar, seed_incar)

        for volume_scale in args.volume_scale:
            run_index += 1
            new_cell, linear_scale = scaled_cell(structure.cell, volume_scale, args.scale_kind)
            volume = cell_volume(new_cell)
            name = f"run_{run_index:04d}_{seed}_v{safe_float_label(volume_scale)}"
            run_dir = volume_root / name
            poscar_text = write_poscar_text(
                f"{args.system} {seed} volume_scale={volume_scale:g}",
                species.symbols,
                species.counts,
                new_cell,
                structure.scaled_positions,
            )
            incar_text = template_incar_with_tags(template, magmom_line, args.isif_volume)
            copy_relax_vasp_files(template, run_dir, poscar_text, incar_text)
            run_dirs.append(run_dir)
            rows.append(
                {
                    "run_index": run_index,
                    "stage": "volume_isif2",
                    "seed": seed,
                    "spin_pattern": pattern_text,
                    "volume_scale": f"{volume_scale:.10g}",
                    "linear_scale": f"{linear_scale:.10g}",
                    "volume_A3": f"{volume:.10f}",
                    "volume_per_atom_A3": f"{volume / species.total_atoms:.10f}",
                    "run_dir": relative_run_path(run_dir, root),
                }
            )

    write_csv(root / "relax_index.csv", rows, RELAX_INDEX_FIELDS)
    write_csv(root / "SUMMARY.csv", rows, RELAX_INDEX_FIELDS)
    write_csv(volume_root / "SUMMARY.csv", rows, RELAX_INDEX_FIELDS)
    write_runlist(root / "runlist.txt", run_dirs, root)
    write_runlist(root / "runlist_volume_isif2.txt", run_dirs, root)
    write_json(
        root / "relax_plan.json",
        {
            "schema": "atomi.materials.relax_seeds.v1",
            "system": args.system,
            "formula": args.formula,
            "source_poscar": str(args.poscar.expanduser().resolve()),
            "template": str(template),
            "magnetic_elements": magnetic_elements,
            "nonmagnetic_elements": nonmagnetic_elements,
            "moment_specs": {
                key: [format_number(value) for value in values]
                for key, values in moment_specs.items()
            },
            "moment_guards": guards,
            "seed_spins": seeds,
            "scale_kind": args.scale_kind,
            "volume_scales": args.volume_scale,
            "original_volume_A3": original_volume,
            "isif_volume": args.isif_volume,
            "isif_shape": args.isif_shape,
            "runlist": str(root / "runlist.txt"),
            "summary_command": (
                "materials-opt relax-summary --workspace . --stage volume_isif2 "
                "--moment-guard " + " --moment-guard ".join(guards)
            ),
            "notes": [
                "Run VASP array jobs with runlist.txt first.",
                "After jobs finish or stop, run materials-opt relax-summary from the workspace.",
                "Use the best physical rows to start ISIF=3 shape relaxation.",
            ],
        },
    )
    (shape_root / "README.md").write_text(
        "After the ISIF=2 volume scan, copy/promote the best physical volume candidates here "
        "with ISIF=3 for full shape relaxation.\n",
        encoding="utf-8",
    )

    print(f"Relax-seeds workspace : {root}")
    print(f"Seed folders          : {seed_root}")
    print(f"Volume scan folders   : {len(run_dirs)}")
    print(f"Runlist               : {root / 'runlist.txt'}")
    print(f"Index                 : {root / 'relax_index.csv'}")
    print(f"Initial SUMMARY       : {root / 'SUMMARY.csv'}")
    print("Next                  : run your VASP array on runlist.txt, then run materials-opt relax-summary --workspace .")


def relax_summary_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="materials-opt relax-summary",
        description="Summarize relax-seeds VASP outputs with energy, spin, and E-V plots.",
    )
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--stage", default="volume_isif2")
    parser.add_argument("--runlist", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--energy", choices=("toten", "without_entropy", "e0", "f", "dav"), default="toten")
    parser.add_argument("--stopped-after-min", type=float, default=15.0)
    parser.add_argument("--dav-average-window", type=int, default=10)
    parser.add_argument("--moment-guard", action="append", default=[])
    parser.add_argument("--moment-guard-tol", type=float, default=0.7)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    from atomi.vasp.spin_report import (
        build_run_reports,
        parse_moment_guards,
        write_atom_table,
        write_markdown_report,
        write_magmom_lines,
        write_physics_filtered_tables,
        write_run_summary,
        output_paths,
    )

    root = args.workspace.expanduser().resolve()
    plan_path = root / "relax_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else {}
    index_path = args.index or root / "relax_index.csv"
    rows = [row for row in parse_relax_index(index_path) if row.get("stage") == args.stage]
    if not rows:
        raise ValueError(f"No rows for stage {args.stage!r} in {index_path}")
    runlist = args.runlist or root / ("runlist_volume_isif2.txt" if args.stage == "volume_isif2" else f"runlist_{args.stage}.txt")
    if not runlist.is_file():
        raise FileNotFoundError(f"Runlist not found: {runlist}")
    guards = args.moment_guard or plan.get("moment_guards", [])
    moment_guards = parse_moment_guards(guards, args.moment_guard_tol)
    prefix = args.output_prefix or root / "04_summary" / args.stage / "spin_energy"
    reports = build_run_reports(
        runlist=runlist,
        spin_index=None,
        log_dir=args.log_dir or root,
        energy_kind=args.energy,
        stopped_after_minutes=args.stopped_after_min,
        dav_average_window=args.dav_average_window,
        species_override=None,
        natoms=None,
        change_threshold=0.25,
        order_threshold=0.2,
        moment_guards=moment_guards,
    )
    paths = output_paths(prefix)
    write_run_summary(reports, paths["summary"])
    write_atom_table(reports, paths["atoms"])
    if moment_guards:
        write_physics_filtered_tables(reports, paths)
    write_markdown_report(reports, paths["report"])
    write_magmom_lines(reports, prefix.parent, decimals=3, compress_tol=0.05)

    report_by_index = {report.index: report for report in reports}
    enriched: list[dict[str, Any]] = []
    energies = [
        report.energy_eV
        for report in reports
        if report.energy_eV is not None
    ]
    minimum = min(energies) if energies else None
    for row in rows:
        report = report_by_index.get(int(row["run_index"]))
        energy = None if report is None else report.energy_eV
        enriched.append(
            {
                **row,
                "energy_eV": "" if energy is None else f"{energy:.10f}",
                "relative_energy_eV": "" if energy is None or minimum is None else f"{energy - minimum:.10f}",
                "energy_kind": "" if report is None else report.energy_kind,
                "status": "" if report is None else report.status,
                "physics_guard_status": "" if report is None else report.physics_guard_status,
                "physics_guard_bad_count": "" if report is None else report.physics_guard_bad_count,
                "mag_status": "" if report is None else report.mag_status,
                "total_moment": "" if report is None or report.total_moment is None else f"{report.total_moment:.8f}",
                "max_abs_moment": "" if report is None or report.max_abs_moment is None else f"{report.max_abs_moment:.8f}",
                "element_order": "" if report is None else json.dumps(report.element_order, sort_keys=True),
                "changed_by_element": "" if report is None else json.dumps(report.changed_by_element, sort_keys=True),
                "energy_source": "" if report is None or report.energy_source is None else str(report.energy_source),
                "mag_source": "" if report is None or report.mag_source is None else str(report.mag_source),
                "warning": "" if report is None else report.warning,
            }
        )

    summary_path = root / f"SUMMARY_{args.stage}.csv"
    write_csv(summary_path, enriched, RELAX_SUMMARY_FIELDS)
    write_csv(root / "SUMMARY.csv", enriched, RELAX_SUMMARY_FIELDS)
    stage_dir = root / ("02_volume_isif2" if args.stage == "volume_isif2" else args.stage)
    if stage_dir.is_dir():
        write_csv(stage_dir / "SUMMARY.csv", enriched, RELAX_SUMMARY_FIELDS)
    plot_path = None if args.no_plot else write_energy_volume_plot(enriched, root / "04_summary" / args.stage / "energy_volume.png")

    usable = sum(1 for row in enriched if row["energy_eV"])
    accepted = sum(1 for row in enriched if row["physics_guard_status"] == "OK")
    print(f"Rows summarized      : {len(enriched)}")
    print(f"Rows with energy     : {usable}")
    if moment_guards:
        print(f"Physics accepted     : {accepted}")
    print(f"SUMMARY              : {summary_path}")
    print(f"Spin-report summary  : {paths['summary']}")
    if plot_path:
        print(f"Energy-volume plot   : {plot_path}")
    elif not args.no_plot:
        print("Energy-volume plot   : not written (matplotlib unavailable or no energy rows)")


def write_energy_volume_plot(rows: list[dict[str, Any]], path: Path) -> Path | None:
    usable = []
    for row in rows:
        energy = finite_float(row.get("relative_energy_eV"))
        volume = finite_float(row.get("volume_A3"))
        if energy is None or volume is None:
            continue
        usable.append((volume, energy, row.get("seed", ""), row.get("physics_guard_status", "")))
    if not usable:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    seed_colors = {"fm": "tab:blue", "afm": "tab:orange"}
    seen_labels: set[str] = set()
    for volume, energy, seed, physics in usable:
        physical = physics in {"OK", "NOT_APPLIED", ""}
        color = seed_colors.get(seed, "tab:green") if physical else "0.65"
        alpha = 0.95 if physical else 0.35
        label = seed if physical else f"{seed} unphysical"
        if label in seen_labels:
            label = None
        else:
            seen_labels.add(label)
        ax.scatter(volume, energy, s=42, color=color, alpha=alpha, edgecolors="black" if physical else "none", label=label)
    ax.set_xlabel("Cell volume (A^3)")
    ax.set_ylabel("Relative energy (eV)")
    ax.set_title("Energy-volume scan")
    ax.grid(True, alpha=0.3)
    if seen_labels:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def status_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="atat-doctor", description="Report ATAT tool availability for Atomi bridges.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--update-hpc-config",
        nargs="?",
        const="",
        type=Path,
        help=(
            "Write/update profiles.atat in a private Atomi HPC config. "
            "Pass a path, or omit the value to use ATOMI_HPC_CONFIG / ~/atomi_hpc."
        ),
    )
    parser.add_argument(
        "--write-hpc-config",
        nargs="?",
        const="",
        type=Path,
        help="Alias for --update-hpc-config.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    report = inspect_atat_environment()
    update_target = args.update_hpc_config
    if update_target is None:
        update_target = args.write_hpc_config
    written: Path | None = None
    if update_target is not None:
        written = update_hpc_config_with_atat(update_target if str(update_target) else None, report)
        report["hpc_config_updated"] = str(written)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_status(report)
        if written is not None:
            print(f"HPC config updated: {written}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atat-bridge", description="Bridge ATAT lattice workflows to Atomi.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("status", help="Report ATAT executable availability.")
    subparsers.add_parser("doctor", help="Alias for status.")
    init_parser = subparsers.add_parser("init", help="Create an ATAT/Atomi bridge workspace.")
    init_parser.add_argument("args", nargs=argparse.REMAINDER)
    index_parser = subparsers.add_parser("index", help="Index ATAT candidate structure files.")
    index_parser.add_argument("args", nargs=argparse.REMAINDER)
    ce_parser = subparsers.add_parser("ce-handoff", help="Build a CE/MC handoff from accepted Atomi DFT rows.")
    ce_parser.add_argument("args", nargs=argparse.REMAINDER)
    vacancy_parser = subparsers.add_parser("vacancy-cif", help="Build explicit vacancy POSCARs from partial-occupancy CIF.")
    vacancy_parser.add_argument("args", nargs=argparse.REMAINDER)
    quick_parser = subparsers.add_parser("quick-opt", help="Create a quick materials optimization scaffold.")
    quick_parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        status_main([])
        return
    command, rest = argv[0], argv[1:]
    if command in {"status", "doctor"}:
        status_main(rest)
    elif command == "init":
        init_main(rest)
    elif command == "index":
        index_main(rest)
    elif command in {"ce-handoff", "ce", "ce-plan"}:
        ce_handoff_main(rest)
    elif command in {"vacancy-cif", "vacany-cif", "cif-vacancy", "partial-occupancy"}:
        vacancy_candidate_main(rest)
    elif command in {"quick-opt", "quick", "materials-opt"}:
        quick_opt_main(rest)
    else:
        parser = build_parser()
        parser.parse_args(argv)


def doctor_main(argv: list[str] | None = None) -> None:
    status_main(argv)


if __name__ == "__main__":
    main()
