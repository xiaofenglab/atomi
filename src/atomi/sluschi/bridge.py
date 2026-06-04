"""Bridge Atomi CALPHAD/MD/MLIP workflows to external SLUSCHI installs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atomi.thermo_prior import PRIOR_SCHEMA


SCHEMA_PLAN = "atomi.sluschi.bridge.plan.v1"
SCHEMA_STATUS = "atomi.sluschi.bridge.status.v1"
SCHEMA_RESULTS = "atomi.sluschi.bridge.results.v1"
SCHEMA_SCONFIG = "atomi.sluschi.lammps_sconfig.v1"

DEFAULT_REPO = "https://github.com/qjhong/SLUSCHI.git"
DEFAULT_SUPERSALT_REF = "SuperSalt chloride MLIP; install/provide externally, not vendored by Atomi."
SUPERSALT_DOI = "10.5281/zenodo.15734798"
SUPERSALT_DOWNLOAD_URL = "https://zenodo.org/records/15734798/files/SuperSalt.zip?download=1"
SUPERSALT_CITATION = "Shen et al., Nat. Commun. 16, 7280 (2025), doi:10.1038/s41467-025-62450-1"
SUPERSALT_ELEMENTS = ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Zr", "Cl"]

OBSERVABLE_UNITS = {
    "melting_temperature_K": "K",
    "solidus_temperature_K": "K",
    "liquidus_temperature_K": "K",
    "heat_of_fusion_J_mol": "J/mol",
    "enthalpy_J_mol": "J/mol",
    "entropy_J_mol_K": "J/mol/K",
    "heat_capacity_J_mol_K": "J/mol/K",
    "density_g_cm3": "g/cm^3",
    "volume_cm3_mol": "cm^3/mol",
    "diffusion_m2_s": "m^2/s",
    "viscosity_Pa_s": "Pa*s",
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def parse_float_list(value: str | None) -> list[float]:
    return [float(item) for item in parse_csv_list(value)]


def parse_composition_states(value: str | None) -> list[str]:
    if not value:
        return []
    if ";" in value:
        return [item.strip() for item in value.split(";") if item.strip()]
    if "|" in value:
        return [item.strip() for item in value.split("|") if item.strip()]
    return [value.strip()] if value.strip() else []


def which_many(names: list[str]) -> dict[str, str | None]:
    return {name: shutil.which(name) for name in names}


def file_info(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"path": "", "exists": False}
    candidate = Path(path).expanduser()
    info: dict[str, Any] = {"path": str(candidate), "exists": candidate.exists()}
    if not candidate.exists() or not candidate.is_file():
        return info
    info["size_bytes"] = candidate.stat().st_size
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    info["sha256"] = digest.hexdigest()
    return info


def load_hpc_profile(config_path: Path | None, profile_name: str) -> dict[str, Any]:
    if config_path is None:
        env_config = os.environ.get("ATOMI_HPC_CONFIG")
        config_path = Path(env_config).expanduser() if env_config else None
    if config_path is None or not config_path.exists():
        return {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    profiles = data.get("profiles", {})
    profile = profiles.get(profile_name, {})
    return profile if isinstance(profile, dict) else {}


def inspect_environment(config_path: Path | None = None, profile_name: str = "sluschi") -> dict[str, Any]:
    profile = load_hpc_profile(config_path, profile_name)
    env_root = os.environ.get("ATOMI_SLUSCHI_ROOT")
    env_bin = os.environ.get("ATOMI_SLUSCHI_BIN")
    env_mlip = os.environ.get("ATOMI_SUPERSALT_MODEL") or os.environ.get("ATOMI_MLIP_MODEL")
    env_lmp = os.environ.get("ATOMI_LMP_EXE") or os.environ.get("ATOMI_LMP_RNEMD_EXE") or os.environ.get("ATOMI_LMP_GK_EXE")
    root = str(profile.get("root") or env_root or "")
    bin_dir = str(profile.get("bin") or env_bin or "")
    env_path = str(profile.get("env_path") or os.environ.get("ATOMI_SLUSCHI_ENV") or "")
    lammps_prefix = str(profile.get("lammps_prefix") or os.environ.get("ATOMI_SLUSCHI_LAMMPS_PREFIX") or "")
    model = str(profile.get("mlip_model") or env_mlip or "")
    lammps_executable = str(profile.get("lammps_executable") or env_lmp or "")
    provider = str(profile.get("mlip_provider") or os.environ.get("ATOMI_MLIP_PROVIDER") or "SuperSalt")
    executables = which_many(["sluschi", "SLUSCHI", "mds_lmp", "lmp", "lammps", "vasp_std", "sbatch", "python3"])
    if lammps_executable and Path(lammps_executable).expanduser().exists():
        executables["lmp"] = str(Path(lammps_executable).expanduser())
    if bin_dir:
        for exe in ("sluschi", "SLUSCHI", "mds_lmp"):
            candidate = Path(bin_dir).expanduser() / exe
            if candidate.exists():
                executables[exe] = str(candidate)
    if not executables.get("sluschi") and executables.get("SLUSCHI"):
        executables["sluschi"] = executables["SLUSCHI"]
    status = {
        "schema": SCHEMA_STATUS,
        "profile": profile_name,
        "config_path": str(config_path) if config_path else "",
        "root": root,
        "bin": bin_dir,
        "env_path": env_path,
        "lammps_prefix": lammps_prefix,
        "mlip_model": model,
        "mlip_provider": provider,
        "mlip_model_info": file_info(model),
        "lammps_executable": lammps_executable,
        "executables": executables,
        "ready_for_bridge": bool(root or bin_dir or executables.get("sluschi") or executables.get("mds_lmp")),
        "ready_for_lammps_mlip": bool(executables.get("lmp") or executables.get("lammps")) and bool(model),
        "recommendation": (
            "Use SLUSCHI as an external dependency. Keep Atomi responsible for input staging, "
            "MLIP manifest/provenance, parsing, and CALPHAD/thermo-prior handoff. Treat melting point "
            "and heat of fusion as native SLUSCHI strengths; treat Cp as a phase-MD observable that "
            "requires a validated MLIP and explicit solid/liquid enthalpy-fluctuation or H(T)-slope data."
        ),
        "install_hint": f"git clone {DEFAULT_REPO} ~/SLUSCHI && export ATOMI_SLUSCHI_ROOT=$HOME/SLUSCHI",
        "supersalt": {
            "doi": SUPERSALT_DOI,
            "download_url": SUPERSALT_DOWNLOAD_URL,
            "citation": SUPERSALT_CITATION,
            "covered_elements": SUPERSALT_ELEMENTS,
            "kcl_licl_in_domain": provider.lower() == "supersalt",
        },
    }
    return status


def write_readme(path: Path, plan: dict[str, Any]) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            # Atomi-SLUSCHI bridge workspace

            System: {plan["system"]}
            Components: {", ".join(plan["components"])}

            Atomi treats SLUSCHI as an external workflow engine. Keep SLUSCHI and any
            MLIP package/model in their own environments, then point this workspace at
            those paths through the HPC JSON or environment variables.

            Suggested order:

            1. Check external tools:
               `atomi sluschi-bridge status --hpc-config "$HOME/atomi_hpc/atomi_hpc_config.kit.local.json"`

            2. Install or locate SLUSCHI externally:
               `git clone {DEFAULT_REPO} ~/SLUSCHI`

            3. Provide an MLIP model for MD acceleration. For KCl-LiCl, start with a
               validated chloride-melt potential such as SuperSalt if you have access,
               or a LiCl-KCl-specific GAP/DP/MACE model. Record the model path in
               `mlip/sluschi_mlip_manifest.json`.

            4. Fill SLUSCHI-specific `job.in`, VASP/LAMMPS templates, and scheduler
               settings in `sluschi_inputs/`.

            5. Run SLUSCHI externally, then parse outputs:
               `atomi sluschi-bridge parse --root . --outdir results`

            6. Use the resulting CSV/JSON/thermo-prior handoff as CALPHAD fitting/check
               data: melting point, heat of fusion, enthalpy, density/volume,
               entropy, or phase-specific heat capacity.

            Scientific guard:

            - SLUSCHI is a robust melting/coexistence engine.
            - Cp is only a robust prior when it is parsed from explicit solid/liquid
              MD observables, e.g. NPT enthalpy fluctuations or H(T) slopes with a
              validated MLIP. For UO2/UC2 solids, keep QHA/phonopy as the baseline
              at low temperature and use SLUSCHI/MLIP-MD to probe high-temperature
              anharmonic and liquid behavior.
            """
        ),
        encoding="utf-8",
    )


def render_job_in(plan: dict[str, Any]) -> str:
    temps = plan["temperature_grid_K"]
    compositions = plan["composition_grid"]
    return textwrap.dedent(
        f"""\
        # Atomi-generated SLUSCHI starter job.in
        # This is a handoff template, not a guarantee of SLUSCHI-version syntax.
        # Review against your installed SLUSCHI manual before submission.
        system = {plan["system"]}
        engine = {plan["engine"]}
        components = {",".join(plan["components"])}
        compositions = {",".join(compositions)}
        temperatures = {",".join(str(t) for t in temps)}
        kmesh = -1
        mlip_manifest = ../mlip/sluschi_mlip_manifest.json
        """
    )


def render_lammps_supersalt_probe(plan: dict[str, Any]) -> str:
    model = plan["mlip"]["model_path"] or "REPLACE_WITH_SUPERSALT_MODEL"
    elements = plan["mlip"].get("elements") or ["Li", "K", "Cl"]
    return textwrap.dedent(
        f"""\
        # Atomi SuperSalt/MACE LAMMPS smoke-test template for {plan["system"]}
        # This is a backend probe, not a production SLUSCHI input.
        # Provide a small equilibrated KCl-LiCl LAMMPS data file before running.
        units           metal
        atom_style      atomic
        boundary        p p p

        read_data       kcl_licl_probe.data

        pair_style      mace no_domain_decomposition
        pair_coeff      * * {model} {" ".join(elements)}

        timestep        0.001
        thermo          10
        thermo_style    custom step temp pe etotal press density

        run             0
        """
    )


def render_supersalt_probe_sbatch(plan: dict[str, Any]) -> str:
    lmp = plan.get("runtime", {}).get("lammps_executable") or "lmp"
    env_path = plan.get("runtime", {}).get("env_path") or ""
    lammps_prefix = plan.get("runtime", {}).get("lammps_prefix") or ""
    activate = ""
    if env_path:
        activate += f'if [ -f "{env_path}/bin/activate" ]; then source "{env_path}/bin/activate"; fi\n'
    if lammps_prefix:
        activate += f'export LD_LIBRARY_PATH="{lammps_prefix}/lib:{lammps_prefix}/lib64:${{LD_LIBRARY_PATH:-}}"\n'
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name=ss-{plan["system"].lower().replace("-", "")}-probe
        #SBATCH --output=../logs/supersalt_probe.%j.out
        #SBATCH --error=../logs/supersalt_probe.%j.err
        #SBATCH --time=00:10:00
        #SBATCH --ntasks=1

        set -euo pipefail
        cd "$(dirname "$0")"

        {activate.rstrip()}
        : "${{ATOMI_LMP_EXE:={lmp}}}"
        echo "LAMMPS executable: ${{ATOMI_LMP_EXE}}"
        echo "Input            : in.supersalt_probe"
        "${{ATOMI_LMP_EXE}}" -in in.supersalt_probe
        """
    )


def default_kcl_licl_compositions() -> list[str]:
    return ["LiCl=0.00,KCl=1.00", "LiCl=0.25,KCl=0.75", "LiCl=0.50,KCl=0.50", "LiCl=0.75,KCl=0.25", "LiCl=1.00,KCl=0.00"]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    components = parse_csv_list(args.components) or ["LiCl", "KCl"]
    temperatures = parse_float_list(args.temperatures) or [900.0, 1000.0, 1100.0, 1200.0]
    compositions = parse_composition_states(args.compositions) or default_kcl_licl_compositions()
    model_path = args.mlip_model or os.environ.get("ATOMI_SUPERSALT_MODEL") or ""
    lammps_executable = getattr(args, "lammps_executable", "") or os.environ.get("ATOMI_LMP_EXE") or ""
    env_path = getattr(args, "env_path", "") or os.environ.get("ATOMI_SLUSCHI_ENV") or ""
    lammps_prefix = getattr(args, "lammps_prefix", "") or os.environ.get("ATOMI_SLUSCHI_LAMMPS_PREFIX") or ""
    provider = args.mlip_provider
    provider_meta: dict[str, Any] = {}
    elements = ["Li", "K", "Cl"] if args.system.lower().replace("-", "") in {"kcllicl", "liclkcl"} else []
    if provider.lower() == "supersalt":
        provider_meta = {
            "doi": SUPERSALT_DOI,
            "download_url": SUPERSALT_DOWNLOAD_URL,
            "citation": SUPERSALT_CITATION,
            "covered_elements": SUPERSALT_ELEMENTS,
            "domain": "chloride melts over Li, Na, K, Rb, Cs, Mg, Ca, Sr, Ba, Zn, Zr cations",
        }
        elements = elements or SUPERSALT_ELEMENTS
    return {
        "schema": SCHEMA_PLAN,
        "system": args.system,
        "components": components,
        "engine": args.engine,
        "phase_target": args.phase_target,
        "temperature_grid_K": temperatures,
        "composition_grid": compositions,
        "sluschi": {
            "dependency_mode": "external",
            "repo": DEFAULT_REPO,
            "root_env": "ATOMI_SLUSCHI_ROOT",
            "bin_env": "ATOMI_SLUSCHI_BIN",
        },
        "mlip": {
            "mode": args.mlip_mode,
            "provider": provider,
            "model_path": model_path,
            "model_info": file_info(model_path),
            "provider_metadata": provider_meta,
            "elements": elements,
            "note": DEFAULT_SUPERSALT_REF if args.mlip_provider.lower() == "supersalt" else "",
        },
        "runtime": {
            "lammps_executable": lammps_executable,
            "env_path": env_path,
            "lammps_prefix": lammps_prefix,
        },
        "handoff": {
            "calphad_observables": [
                "melting_temperature_K",
                "solidus_temperature_K",
                "liquidus_temperature_K",
                "heat_of_fusion_J_mol",
                "enthalpy_J_mol",
                "entropy_J_mol_K",
                "density_g_cm3",
                "volume_cm3_mol",
            ],
            "md_observables": [
                "diffusion_m2_s",
                "density_g_cm3",
                "heat_capacity_J_mol_K",
                "viscosity_Pa_s",
                "thermal_conductivity_if_requested",
            ],
            "thermo_prior_schema": PRIOR_SCHEMA,
        },
    }


def init_workspace(args: argparse.Namespace) -> dict[str, Any]:
    root = args.outdir.resolve()
    for folder in ("sluschi_inputs", "mlip", "results", "calphad_handoff", "logs"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    write_json(root / "sluschi_bridge_plan.json", plan)
    (root / "sluschi_inputs" / "job.in").write_text(render_job_in(plan), encoding="utf-8")
    if plan["mlip"]["provider"].lower() == "supersalt":
        (root / "sluschi_inputs" / "in.supersalt_probe").write_text(render_lammps_supersalt_probe(plan), encoding="utf-8")
        probe = root / "sluschi_inputs" / "run_supersalt_probe.sbatch"
        probe.write_text(render_supersalt_probe_sbatch(plan), encoding="utf-8")
        probe.chmod(0o755)
    mlip_manifest = {
        "schema": "atomi.sluschi.bridge.mlip_manifest.v1",
        "system": plan["system"],
        "provider": plan["mlip"]["provider"],
        "mode": plan["mlip"]["mode"],
        "model_path": plan["mlip"]["model_path"],
        "model_info": plan["mlip"]["model_info"],
        "provider_metadata": plan["mlip"].get("provider_metadata", {}),
        "elements": plan["mlip"].get("elements", []),
        "validation_required": [
            "composition coverage includes target LiCl-KCl range",
            "temperature coverage includes melt/coexistence range",
            "density/RDF/enthalpy checked against DFT or experiment",
            "LAMMPS/ASE backend tested on a short NVT melt before SLUSCHI production",
            "solid/coexistence use validated separately because SuperSalt was primarily trained for liquids",
        ],
    }
    write_json(root / "mlip" / "sluschi_mlip_manifest.json", mlip_manifest)
    write_readme(root / "README_SLUSCHI_ATOMI_BRIDGE.md", plan)
    print(f"Wrote SLUSCHI bridge workspace: {root}")
    print(f"Plan                         : {root / 'sluschi_bridge_plan.json'}")
    print(f"MLIP manifest                : {root / 'mlip' / 'sluschi_mlip_manifest.json'}")
    print(f"Starter job.in               : {root / 'sluschi_inputs' / 'job.in'}")
    if plan["mlip"]["provider"].lower() == "supersalt":
        print(f"SuperSalt LAMMPS probe       : {root / 'sluschi_inputs' / 'in.supersalt_probe'}")
    return {"root": str(root), "plan": str(root / "sluschi_bridge_plan.json")}


RESULT_PATTERNS = {
    "melting_temperature_K": re.compile(
        r"(?:melting\s+temperature|melting\s+point|m\.p\.|tm)\b[^\n:=]*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*K?",
        re.IGNORECASE,
    ),
    "solidus_temperature_K": re.compile(r"\bsolidus\b[^\n:=]*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*K?", re.IGNORECASE),
    "liquidus_temperature_K": re.compile(r"\bliquidus\b[^\n:=]*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*K?", re.IGNORECASE),
    "heat_of_fusion_J_mol": re.compile(
        r"(?:heat\s+of\s+fusion|enthalpy\s+of\s+fusion|delta\s*h(?:fus)?|dh(?:fus)?)\b[^\n:=]*[:=]?\s*([-+]?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    "enthalpy_J_mol": re.compile(r"\b(?:enthalpy|h)\b(?!\s+of\s+fusion)[^\n:=]*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    "entropy_J_mol_K": re.compile(r"\b(?:entropy|s)\b[^\n:=]*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    "heat_capacity_J_mol_K": re.compile(
        r"\b(?:heat\s+capacity|cp|c_p)\b[^\n:=]*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    "density_g_cm3": re.compile(r"\b(?:density|rho)\b[^\n:=]*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    "volume_cm3_mol": re.compile(r"\b(?:molar\s+volume|volume)\b[^\n:=]*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    "diffusion_m2_s": re.compile(r"\b(?:diffusion|diffusivity|D)\b[^\n:=]*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    "viscosity_Pa_s": re.compile(r"\b(?:viscosity|eta)\b[^\n:=]*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.IGNORECASE),
}


@dataclass
class ParsedResult:
    file: str
    observable: str
    value: float
    unit: str
    phase: str
    temperature_K: float | None
    composition: str
    line: str


@dataclass
class SconfigPair:
    file: str
    pair: str
    state: str
    recommended_statistic: str
    sconfig_J_mol_atom_K: float
    line: str


def parse_numeric_file(path: Path) -> list[float]:
    if not path.exists() or not path.is_file():
        return []
    values: list[float] = []
    for token in re.split(r"[\s,]+", path.read_text(encoding="utf-8", errors="replace").strip()):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def parse_sconfig_pairs(root: Path) -> list[SconfigPair]:
    rows: list[SconfigPair] = []
    pattern = re.compile(
        r"The\s+pair\s+between\s+element\s+([^\s]+)\s+appears\s+to\s+be\s+([^.]+)\.\s+"
        r"I\s+suggest\s+that\s+you\s+take\s+the\s+([^:]+):\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name not in {"collect.stdout", "collect.out", "sluschi_collect.out"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end < 0:
                line_end = len(text)
            rows.append(
                SconfigPair(
                    file=str(path),
                    pair=match.group(1).strip(),
                    state=match.group(2).strip().lower(),
                    recommended_statistic=match.group(3).strip().lower(),
                    sconfig_J_mol_atom_K=float(match.group(4)),
                    line=text[line_start:line_end].strip(),
                )
            )
    return rows


def summarize_sconfig_case(args: argparse.Namespace, pairs: list[SconfigPair]) -> dict[str, Any]:
    root = args.root.resolve()
    sconf = parse_numeric_file(root / "Sconf.txt")
    sconf_min = parse_numeric_file(root / "Sconf_min.txt")
    pair_values = [row.sconfig_J_mol_atom_K for row in pairs]
    liquid_like = sum(1 for row in pairs if "liquid" in row.state)
    solid_like = sum(1 for row in pairs if "solid" in row.state)
    mean_pair = sum(pair_values) / len(pair_values) if pair_values else None
    return {
        "schema": SCHEMA_SCONFIG,
        "root": str(root),
        "system": args.system,
        "formula": args.formula,
        "components": parse_csv_list(args.components),
        "phase": args.phase,
        "temperature_K": args.temperature_k,
        "composition": args.composition,
        "source_engine": "lammps",
        "method": "sluschi_sconfig_from_lammps_nvt",
        "quality": args.quality,
        "n_pair_recommendations": len(pairs),
        "n_liquid_like_pairs": liquid_like,
        "n_solid_like_pairs": solid_like,
        "mean_pair_sconfig_J_mol_atom_K": mean_pair,
        "sconf_values_J_mol_atom_K": sconf,
        "sconf_min_values_J_mol_atom_K": sconf_min,
        "dump_stride_note": args.dump_stride_note,
        "warnings": [
            warning
            for warning in [
                "No SLUSCHI pair recommendations found." if not pairs else "",
                "Sconf.txt not found or empty." if not sconf else "",
                "Sconf_min.txt not found or empty." if not sconf_min else "",
            ]
            if warning
        ],
    }


def build_sconfig_prior_payload(args: argparse.Namespace, summary: dict[str, Any], pairs: list[SconfigPair]) -> dict[str, Any]:
    observable = {
        "observable": "configurational_entropy_J_mol_atom_K",
        "value": summary["mean_pair_sconfig_J_mol_atom_K"],
        "unit": "J/mol-atom/K",
        "phase": args.phase,
        "temperature_K": args.temperature_k,
        "composition": args.composition,
        "quality": args.quality,
        "source_engine": "lammps",
        "method": "sluschi_sconfig_pair_recommendation",
        "n_pair_recommendations": len(pairs),
    }
    return {
        "schema": PRIOR_SCHEMA,
        "kind": "sluschi_lammps_sconfig",
        "system": args.system or args.root.resolve().name,
        "formula": args.formula or "",
        "components": parse_csv_list(args.components),
        "thermo": {"observables": [observable] if observable["value"] is not None else []},
        "source": {
            "method": "lammps_sconfig",
            "root": str(args.root.resolve()),
            "bridge_schema": SCHEMA_SCONFIG,
        },
        "notes": [
            "This is a SLUSCHI configurational-entropy descriptor from LAMMPS NVT trajectory post-processing.",
            "Use as a screening prior unless the trajectory length, volume, dump stride, and type mapping were validated for production.",
            "For Svib, generate dense uniformly spaced NVT frames after equilibration at the target volume/phase state.",
        ],
    }


def sconfig_main(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    outdir = args.outdir.resolve()
    pairs = parse_sconfig_pairs(root)
    summary = summarize_sconfig_case(args, pairs)
    pair_rows = [row.__dict__ for row in pairs]
    pair_csv = outdir / "lammps_sconfig_pairs.csv"
    summary_csv = outdir / "lammps_sconfig_summary.csv"
    summary_json = outdir / "lammps_sconfig_summary.json"
    prior_json = args.prior_out or outdir / "lammps_sconfig_thermo_prior.json"
    write_csv(
        pair_csv,
        pair_rows,
        ["file", "pair", "state", "recommended_statistic", "sconfig_J_mol_atom_K", "line"],
    )
    write_csv(
        summary_csv,
        [summary],
        [
            "root",
            "system",
            "formula",
            "phase",
            "temperature_K",
            "composition",
            "quality",
            "n_pair_recommendations",
            "n_liquid_like_pairs",
            "n_solid_like_pairs",
            "mean_pair_sconfig_J_mol_atom_K",
        ],
    )
    write_json(summary_json, {**summary, "pairs": pair_rows})
    write_json(prior_json, build_sconfig_prior_payload(args, summary, pairs))
    print(f"Parsed SLUSCHI/LAMMPS Sconfig pair recommendations: {len(pairs)}")
    print(f"Wrote pair CSV             : {pair_csv}")
    print(f"Wrote summary CSV          : {summary_csv}")
    print(f"Wrote summary JSON         : {summary_json}")
    print(f"Wrote thermo-prior JSON    : {prior_json}")
    return {
        "schema": SCHEMA_SCONFIG,
        "n_pair_recommendations": len(pairs),
        "outputs": {
            "pairs_csv": str(pair_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "thermo_prior_json": str(prior_json),
        },
        "summary": summary,
    }


def infer_phase(path: Path, line: str, default: str = "") -> str:
    haystack = f"{path} {line}".lower()
    if "coexist" in haystack or "solid-liquid" in haystack or "solid_liquid" in haystack:
        return "solid-liquid"
    if "liquid" in haystack or "melt" in haystack or "_liq" in haystack:
        return "liquid"
    if "solid" in haystack or "_sol" in haystack:
        return "solid"
    return default


def infer_temperature(line: str, default: float | None = None) -> float | None:
    match = re.search(r"(?:temperature|temp|T)\b[^\n:=]*[:=]\s*([-+]?\d+(?:\.\d+)?)\s*K?", line, re.IGNORECASE)
    return float(match.group(1)) if match else default


def parse_results(root: Path) -> list[ParsedResult]:
    rows: list[ParsedResult] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {"", ".out", ".log", ".txt", ".dat"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for observable, pattern in RESULT_PATTERNS.items():
            for match in pattern.finditer(text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                if line_end < 0:
                    line_end = len(text)
                rows.append(
                    ParsedResult(
                        file=str(path),
                        observable=observable,
                        value=float(match.group(1)),
                        unit=OBSERVABLE_UNITS.get(observable, ""),
                        phase=infer_phase(path, text[line_start:line_end].strip()),
                        temperature_K=infer_temperature(text[line_start:line_end].strip()),
                        composition="",
                        line=text[line_start:line_end].strip(),
                    )
                )
    return rows


def load_bridge_plan(root: Path) -> dict[str, Any]:
    plan = root / "sluschi_bridge_plan.json"
    if not plan.exists():
        return {}
    try:
        data = json.loads(plan.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def build_thermo_prior_payload(args: argparse.Namespace, rows: list[ParsedResult]) -> dict[str, Any]:
    plan = load_bridge_plan(args.root.resolve())
    components = parse_csv_list(args.components) or plan.get("components", [])
    observables = []
    for row in rows:
        record = row.__dict__.copy()
        if args.phase and not record["phase"]:
            record["phase"] = args.phase
        if args.temperature_k is not None and record["temperature_K"] is None:
            record["temperature_K"] = args.temperature_k
        if args.composition and not record["composition"]:
            record["composition"] = args.composition
        record["file"] = str(Path(record["file"]).resolve())
        observables.append(record)
    return {
        "schema": PRIOR_SCHEMA,
        "kind": "sluschi_phase_observable_set",
        "system": args.system or plan.get("system") or args.root.resolve().name,
        "formula": args.formula or "",
        "components": components,
        "thermo": {"observables": observables},
        "source": {
            "method": "sluschi_bridge_parse",
            "root": str(args.root.resolve()),
            "sluschi_repo": DEFAULT_REPO,
            "bridge_schema": SCHEMA_RESULTS,
        },
        "notes": [
            "SLUSCHI-native priors are strongest for melting/coexistence and heat of fusion.",
            "Cp entries should be used only when generated from explicit phase MD fluctuation or H(T)-slope analyses with a validated MLIP.",
        ],
    }


def parse_main(args: argparse.Namespace) -> dict[str, Any]:
    rows = parse_results(args.root.resolve())
    outdir = args.outdir.resolve()
    csv_rows = [row.__dict__ for row in rows]
    table = outdir / "sluschi_parsed_results.csv"
    write_csv(table, csv_rows, ["file", "observable", "value", "unit", "phase", "temperature_K", "composition", "line"])
    prior_path = args.prior_out or outdir / "sluschi_thermo_prior.json"
    prior_payload = build_thermo_prior_payload(args, rows)
    write_json(prior_path, prior_payload)
    payload = {
        "schema": SCHEMA_RESULTS,
        "root": str(args.root.resolve()),
        "n_results": len(rows),
        "outputs": {"csv": str(table), "thermo_prior_json": str(prior_path)},
        "results": csv_rows,
    }
    write_json(outdir / "sluschi_parsed_results.json", payload)
    print(f"Parsed SLUSCHI-like outputs: {len(rows)} result(s)")
    print(f"Wrote CSV                 : {table}")
    print(f"Wrote thermo-prior JSON   : {prior_path}")
    return payload


def status_main(args: argparse.Namespace) -> dict[str, Any]:
    status = inspect_environment(args.hpc_config, args.profile)
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print("Atomi SLUSCHI bridge status")
        print("---------------------------")
        print(f"profile          : {status['profile']}")
        print(f"SLUSCHI root     : {status['root'] or 'not set'}")
        print(f"SLUSCHI bin      : {status['bin'] or 'not set'}")
        print(f"runtime env      : {status['env_path'] or 'not set'}")
        print(f"LAMMPS prefix    : {status['lammps_prefix'] or 'not set'}")
        print(f"MLIP provider    : {status['mlip_provider'] or 'not set'}")
        print(f"MLIP model       : {status['mlip_model'] or 'not set'}")
        print(f"LAMMPS executable: {status['lammps_executable'] or status['executables'].get('lmp') or 'not set'}")
        if status["mlip_model"]:
            model_info = status.get("mlip_model_info", {})
            print(f"MLIP model exists: {model_info.get('exists')}")
            if model_info.get("sha256"):
                print(f"MLIP sha256      : {model_info['sha256']}")
        for name, path in status["executables"].items():
            print(f"{name:16}: {path or 'not found'}")
        print(f"bridge ready     : {status['ready_for_bridge']}")
        print(f"LAMMPS+MLIP ready: {status['ready_for_lammps_mlip']}")
    return status


def supersalt_example_main(args: argparse.Namespace) -> dict[str, Any]:
    status = inspect_environment(args.hpc_config, args.profile)
    model = args.mlip_model or status.get("mlip_model") or os.environ.get("ATOMI_SUPERSALT_MODEL") or ""
    lammps_executable = args.lammps_executable or status.get("lammps_executable") or status.get("executables", {}).get("lmp") or ""
    env_path = args.env_path or status.get("env_path") or ""
    lammps_prefix = args.lammps_prefix or status.get("lammps_prefix") or ""
    ns = argparse.Namespace(
        outdir=args.outdir,
        system="KCl-LiCl",
        components="LiCl,KCl",
        engine="lammps",
        phase_target="solid-liquid-coexistence",
        temperatures=args.temperatures,
        compositions=args.compositions,
        mlip_mode="external-model",
        mlip_provider="SuperSalt",
        mlip_model=model,
        lammps_executable=lammps_executable,
        env_path=env_path,
        lammps_prefix=lammps_prefix,
    )
    result = init_workspace(ns)
    root = Path(result["root"])
    readme = root / "README_KCL_LICL_SUPERSALT_DEMO.md"
    readme.write_text(
        textwrap.dedent(
            f"""\
            # KCl-LiCl SuperSalt + SLUSCHI Demonstration

            This workspace demonstrates how Atomi connects the public SuperSalt
            chloride MLIP to a SLUSCHI phase-equilibria workflow.

            Model path:
            `{model or "not configured; set --mlip-model or profiles.sluschi.mlip_model"}`

            LAMMPS executable:
            `{lammps_executable or "not configured; set --lammps-executable or profiles.sluschi.lammps_executable"}`

            Runtime environment:
            `{env_path or "not configured; set --env-path or profiles.sluschi.env_path"}`

            LAMMPS prefix:
            `{lammps_prefix or "not configured; set --lammps-prefix or profiles.sluschi.lammps_prefix"}`

            SuperSalt provenance:

            - DOI: {SUPERSALT_DOI}
            - Citation: {SUPERSALT_CITATION}
            - Download: {SUPERSALT_DOWNLOAD_URL}
            - Covered elements: {", ".join(SUPERSALT_ELEMENTS)}

            Recommended sequence:

            1. Confirm the model manifest:
               `cat mlip/sluschi_mlip_manifest.json`
            2. Put a tiny KCl-LiCl LAMMPS data file at
               `sluschi_inputs/kcl_licl_probe.data`.
            3. Run the backend probe only when a MACE-capable LAMMPS executable is loaded:
               `sbatch sluschi_inputs/run_supersalt_probe.sbatch`
            4. Review the generated `sluschi_inputs/job.in` against the installed
               SLUSCHI/SuperSalt workflow before production.
            5. After SLUSCHI/MLIP-MD outputs are produced, parse them:
               `atomi sluschi-bridge parse --root . --outdir results --system KCl-LiCl --components LiCl,KCl`

            Scientific guard:

            SuperSalt was designed for chloride melts and is in-domain for the
            LiCl-KCl liquid. Solid and solid-liquid coexistence predictions should
            be validated by short probes against known KCl-LiCl melting/eutectic
            anchors before treating them as CALPHAD constraints.
            """
        ),
        encoding="utf-8",
    )
    print(f"KCl-LiCl SuperSalt README    : {readme}")
    return {
        **result,
        "readme": str(readme),
        "mlip_model": model,
        "lammps_executable": lammps_executable,
        "env_path": env_path,
        "lammps_prefix": lammps_prefix,
        "status": status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sluschi-bridge",
        description="Prepare and inspect Atomi handoffs to external SLUSCHI/MLIP melting workflows.",
    )
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Inspect SLUSCHI, LAMMPS, and MLIP paths.")
    status.add_argument("--hpc-config", type=Path)
    status.add_argument("--profile", default="sluschi")
    status.add_argument("--json", action="store_true")

    init = sub.add_parser("init", help="Create a SLUSCHI bridge workspace.")
    init.add_argument("--outdir", type=Path, default=Path("sluschi_kcl_licl_bridge"))
    init.add_argument("--system", default="KCl-LiCl")
    init.add_argument("--components", default="LiCl,KCl")
    init.add_argument("--engine", choices=("lammps", "vasp", "external"), default="lammps")
    init.add_argument("--phase-target", default="solid-liquid-coexistence")
    init.add_argument("--temperatures", help="Comma-separated temperatures in K.")
    init.add_argument(
        "--compositions",
        help="Composition states separated by semicolons, e.g. 'LiCl=0.25,KCl=0.75;LiCl=0.5,KCl=0.5'.",
    )
    init.add_argument("--mlip-mode", choices=("external-model", "train-local", "none"), default="external-model")
    init.add_argument("--mlip-provider", default="SuperSalt")
    init.add_argument("--mlip-model", default="")
    init.add_argument("--lammps-executable", default="", help="Optional LAMMPS executable for generated MLIP probe scripts.")
    init.add_argument("--env-path", default="", help="Optional Python/MLIP environment to activate in generated probe scripts.")
    init.add_argument("--lammps-prefix", default="", help="Optional LAMMPS install prefix for LD_LIBRARY_PATH in generated probe scripts.")

    demo = sub.add_parser("supersalt-example", help="Create a KCl-LiCl SuperSalt + SLUSCHI demonstration workspace.")
    demo.add_argument("--outdir", type=Path, default=Path("sluschi_kcl_licl_supersalt_demo"))
    demo.add_argument("--hpc-config", type=Path)
    demo.add_argument("--profile", default="sluschi")
    demo.add_argument("--mlip-model", default="", help="SuperSalt model path; default from profiles.sluschi.mlip_model.")
    demo.add_argument("--lammps-executable", default="", help="MACE-capable LAMMPS executable; default from profile/status.")
    demo.add_argument("--env-path", default="", help="Python/MLIP environment; default from profiles.sluschi.env_path.")
    demo.add_argument("--lammps-prefix", default="", help="LAMMPS install prefix; default from profiles.sluschi.lammps_prefix.")
    demo.add_argument("--temperatures", default="900,1000,1100,1200", help="Comma-separated temperatures in K.")
    demo.add_argument(
        "--compositions",
        default="LiCl=0.00,KCl=1.00;LiCl=0.25,KCl=0.75;LiCl=0.50,KCl=0.50;LiCl=0.75,KCl=0.25;LiCl=1.00,KCl=0.00",
        help="KCl-LiCl composition states separated by semicolons.",
    )

    parse = sub.add_parser("parse", help="Parse SLUSCHI-like text outputs for CALPHAD handoff observables.")
    parse.add_argument("--root", type=Path, default=Path("."))
    parse.add_argument("--outdir", type=Path, default=Path("results"))
    parse.add_argument("--prior-out", type=Path, help="Optional thermo-prior JSON output path.")
    parse.add_argument("--system", default="", help="System label for the thermo-prior JSON.")
    parse.add_argument("--formula", default="", help="Formula label for unary phase priors, if applicable.")
    parse.add_argument("--components", default="", help="Comma-separated component labels for mixture priors.")
    parse.add_argument("--phase", default="", help="Default phase label when not inferable from output text.")
    parse.add_argument("--temperature-k", type=float, help="Default temperature for parsed observables.")
    parse.add_argument("--composition", default="", help="Default composition label for parsed observables.")

    sconfig = sub.add_parser(
        "sconfig",
        help="Parse SLUSCHI configurational-entropy outputs from a LAMMPS NVT case folder.",
    )
    sconfig.add_argument("--root", type=Path, default=Path("."))
    sconfig.add_argument("--outdir", type=Path, default=Path("sconfig_results"))
    sconfig.add_argument("--prior-out", type=Path, help="Optional thermo-prior JSON output path.")
    sconfig.add_argument("--system", default="", help="System label, e.g. KCl-LiCl or UO2.")
    sconfig.add_argument("--formula", default="", help="Formula label for unary or defect-compound priors.")
    sconfig.add_argument("--components", default="", help="Comma-separated component labels for mixture priors.")
    sconfig.add_argument("--phase", default="", help="Phase label, e.g. solid, liquid, fluorite, defective-fluorite.")
    sconfig.add_argument("--temperature-k", type=float, help="Trajectory temperature in K.")
    sconfig.add_argument("--composition", default="", help="Composition label for mixture or defect priors.")
    sconfig.add_argument(
        "--quality",
        choices=("descriptor", "screening-prior", "production"),
        default="descriptor",
        help="Confidence tier for downstream thermo-prior use.",
    )
    sconfig.add_argument(
        "--dump-stride-note",
        default="Dense uniformly spaced NVT frames are required for Svib; Sconfig parsing records the SLUSCHI pair recommendation only.",
        help="Trajectory/dump-stride note stored in the summary JSON.",
    )

    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        return status_main(args)
    if args.command == "init":
        return init_workspace(args)
    if args.command == "supersalt-example":
        return supersalt_example_main(args)
    if args.command == "parse":
        return parse_main(args)
    if args.command == "sconfig":
        return sconfig_main(args)
    build_parser().print_help()
    return None


def sconfig_cli_main(argv: list[str] | None = None) -> dict[str, Any] | None:
    return main(["sconfig", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
