"""Bridge Atomi CALPHAD/MD/MLIP workflows to external SLUSCHI installs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atomi.structure.adapters import (
    cell_from_cp2k_input,
    cell_from_xyz_comment,
    read_cp2k_xyz_frames,
    read_vasp_poscar_basis,
    read_vasp_xdatcar_frames,
)
from atomi.thermo_prior import PRIOR_SCHEMA


SCHEMA_PLAN = "atomi.sluschi.bridge.plan.v1"
SCHEMA_STATUS = "atomi.sluschi.bridge.status.v1"
SCHEMA_RESULTS = "atomi.sluschi.bridge.results.v1"
SCHEMA_SCONFIG = "atomi.sluschi.lammps_sconfig.v1"
SCHEMA_ENTROPY_SUMMARY = "atomi.sluschi.lammps_entropy_summary.v1"
SCHEMA_PHASE_HEALTH = "atomi.sluschi.phase_health.v1"
SCHEMA_PHASE_WINDOW_SAMPLE = "atomi.sluschi.phase_window_sample.v1"
SCHEMA_WORKFLOW_GUIDE = "atomi.sluschi.workflow_guide.v1"
SCHEMA_MELTING_ANCHOR = "atomi.sluschi.melting_anchor.v1"
SCHEMA_LAMMPS_PREP = "atomi.sluschi.lammps_prep.v1"
SCHEMA_CP2K_PREP = "atomi.sluschi.cp2k_prep.v1"
SCHEMA_VASP_PREP = "atomi.sluschi.vasp_prep.v1"
SCHEMA_MDS_ENTROPY_RUN = "atomi.sluschi.mds_entropy_run.v1"

DEFAULT_REPO = "https://github.com/qjhong/SLUSCHI.git"
DEFAULT_SUPERSALT_REF = "SuperSalt chloride MLIP; install/provide externally, not vendored by Atomi."
SUPERSALT_DOI = "10.5281/zenodo.15734798"
SUPERSALT_DOWNLOAD_URL = "https://zenodo.org/records/15734798/files/SuperSalt.zip?download=1"
SUPERSALT_CITATION = "Shen et al., Nat. Commun. 16, 7280 (2025), doi:10.1038/s41467-025-62450-1"
SUPERSALT_ELEMENTS = ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Zr", "Cl"]
DEFAULT_ELEMENT_MASSES = {
    # Light/common nonmetals and gases used in carbide, nitride, oxide, and salt workflows.
    "H": 1.008,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.453,
    "Ar": 39.948,
    # Alkali/alkaline-earth/chloride-melt cations.
    "Li": 6.941,
    "Na": 22.98976928,
    "K": 39.0983,
    "Rb": 85.4678,
    "Cs": 132.90545196,
    "Mg": 24.305,
    "Ca": 40.078,
    "Sr": 87.62,
    "Ba": 137.327,
    # Transition metals commonly appearing in current Atomi tests/projects.
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Zr": 91.224,
    "Mo": 95.95,
    "Hf": 178.49,
    "W": 183.84,
    # Rare-earth/actinide elements used in oxide defect and molten-salt studies.
    "Ce": 140.116,
    "Gd": 157.25,
    "Th": 232.0377,
    "U": 238.02891,
    "Np": 237.0,
    "Pu": 244.0,
}

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


def parse_key_float_map(value: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in parse_csv_list(value):
        if "=" not in item:
            raise ValueError(f"Expected key=value item, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = float(raw)
    return out


def parse_element_list(value: str | None) -> list[str]:
    return [item.strip() for item in parse_csv_list(value)]


def parse_cell_abc(value: str | None) -> list[list[float]] | None:
    parts = parse_csv_list(value)
    if not parts:
        return None
    if len(parts) != 3:
        raise ValueError("--cell must have exactly three lengths, e.g. '12.58,12.58,18.87'.")
    a, b, c = (float(item) for item in parts)
    return [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]


def parse_cell_vector(value: str) -> list[float]:
    parts = [item for item in re.split(r"[\s,]+", value.strip()) if item]
    if len(parts) != 3:
        raise ValueError(f"Cell vector must have three components, got {value!r}.")
    return [float(item) for item in parts]


def write_sluschi_native_frames(
    *,
    outdir: Path,
    elements: list[str],
    counts: dict[str, int],
    masses: dict[str, float],
    lattice_frames: list[list[list[float]]],
    symbol_frames: list[list[str]],
    frac_frames: list[list[list[float]]],
    step_interval_fs: float,
    phase_temp_label: str = "",
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    if not frac_frames:
        raise ValueError("No frames selected for SLUSCHI prep.")
    natoms = len(frac_frames[0])
    order = {element: idx for idx, element in enumerate(elements)}
    with (outdir / "param").open("w", encoding="utf-8") as handle:
        handle.write(f"{len(elements)}\n")
        handle.write(" ".join(str(counts[element]) for element in elements) + "\n")
        for element in elements:
            handle.write(f"{masses[element]}\n")
        handle.write(f"{step_interval_fs / 1000.0:.12g}\n")
        handle.write(f"{natoms}\n")
        for element in elements:
            handle.write(f"{element}\n")
        for _ in range(8):
            handle.write("0.0\n")
    with (outdir / "latt").open("w", encoding="utf-8") as latt, (outdir / "pos").open("w", encoding="utf-8") as pos, (
        outdir / "step"
    ).open("w", encoding="utf-8") as step:
        for cell, symbols, frame in zip(lattice_frames, symbol_frames, frac_frames):
            if len(symbols) != natoms or len(frame) != natoms:
                raise ValueError("All selected frames must have the same atom count.")
            for vector in cell:
                latt.write(f"{vector[0]:.12g} {vector[1]:.12g} {vector[2]:.12g} 0 0 0\n")
            atoms = sorted(zip(symbols, frame), key=lambda item: order[item[0]])
            for symbol, frac in atoms:
                if symbol not in order:
                    raise ValueError(f"Frame contains element {symbol!r}, not listed in --elements={','.join(elements)!r}.")
                cart = _cart_from_fractional(frac, cell)
                pos.write(f"{cart[0]:.12g} {cart[1]:.12g} {cart[2]:.12g} 0 0 0\n")
            step.write(f"{step_interval_fs / 1000.0:.12g}\n")
    outputs = {
        "param": str(outdir / "param"),
        "latt": str(outdir / "latt"),
        "pos": str(outdir / "pos"),
        "step": str(outdir / "step"),
        "phase_temp": "",
    }
    if phase_temp_label:
        (outdir / "phase_temp").write_text(f"{phase_temp_label}\n", encoding="utf-8")
        outputs["phase_temp"] = str(outdir / "phase_temp")
    return outputs


def mat_det(matrix: list[list[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def cart_to_frac(coord: list[float], cell: list[list[float]]) -> list[float]:
    # Cell vectors are rows. Fractional coordinates f solve f @ cell = coord.
    det = mat_det(cell)
    if abs(det) < 1.0e-15:
        raise ValueError("Cell matrix is singular.")
    a, b, c = cell
    inv = [
        [(b[1] * c[2] - b[2] * c[1]) / det, (a[2] * c[1] - a[1] * c[2]) / det, (a[1] * b[2] - a[2] * b[1]) / det],
        [(b[2] * c[0] - b[0] * c[2]) / det, (a[0] * c[2] - a[2] * c[0]) / det, (a[2] * b[0] - a[0] * b[2]) / det],
        [(b[0] * c[1] - b[1] * c[0]) / det, (a[1] * c[0] - a[0] * c[1]) / det, (a[0] * b[1] - a[1] * b[0]) / det],
    ]
    return [
        coord[0] * inv[0][i] + coord[1] * inv[1][i] + coord[2] * inv[2][i]
        for i in range(3)
    ]


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
        if env_config:
            config_path = Path(env_config).expanduser()
        else:
            default_config = Path.home() / "atomi_hpc" / "atomi_hpc_config.kit.local.json"
            config_path = default_config if default_config.exists() else None
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


def parse_type_element_map(value: str) -> dict[int, str]:
    if not value.strip():
        raise ValueError("--type-elements is required, e.g. '1=K,2=Cl'")
    out: dict[int, str] = {}
    for item in parse_csv_list(value):
        if "=" not in item:
            raise ValueError(f"Expected type=Element item, got {item!r}")
        raw_idx, raw_element = item.split("=", 1)
        try:
            idx = int(raw_idx.strip())
        except ValueError as exc:
            raise ValueError(f"LAMMPS type indices must be integers, got {raw_idx!r}") from exc
        if idx <= 0:
            raise ValueError(f"LAMMPS type indices must be positive, got {idx}")
        element = raw_element.strip()
        if not element:
            raise ValueError(f"Missing element symbol for type {idx}")
        out[idx] = element
    expected = list(range(1, len(out) + 1))
    observed = sorted(out)
    if observed != expected:
        raise ValueError(f"--type-elements must define contiguous types {expected}; got {observed}")
    return out


def default_element_masses() -> dict[str, float]:
    """Return built-in masses, enriched from ASE when available.

    SLUSCHI prep should work for ordinary chemical symbols without forcing users
    to pass --element-masses for every new chemistry. ASE is optional in Atomi,
    so keep a local fallback table for the projects we commonly run.
    """

    masses = dict(DEFAULT_ELEMENT_MASSES)
    try:
        from ase.data import atomic_masses, atomic_numbers
    except Exception:
        return masses
    for symbol, number in atomic_numbers.items():
        try:
            mass = float(atomic_masses[number])
        except Exception:
            continue
        if mass > 0:
            masses.setdefault(symbol, mass)
    return masses


def parse_element_mass_map(value: str | None) -> dict[str, float]:
    masses = default_element_masses()
    for item in parse_csv_list(value):
        if "=" not in item:
            raise ValueError(f"Expected Element=mass item, got {item!r}")
        element, raw_mass = item.split("=", 1)
        masses[element.strip()] = float(raw_mass)
    return masses


def render_lammps_dump_pos_py(type_elements: dict[int, str]) -> str:
    order = {element: i for i, element in enumerate(type_elements.values())}
    return textwrap.dedent(
        f"""\
        from pathlib import Path

        from ase import Atoms
        from ase.io import write


        symbols_by_type = {dict(type_elements)!r}
        order = {order!r}

        lines = Path("lmp.dump").read_text().splitlines()
        frames = []
        i = 0
        while i < len(lines):
            if not lines[i].startswith("ITEM: TIMESTEP"):
                i += 1
                continue
            i += 2
            if not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("unexpected dump format near atom count")
            nat = int(lines[i + 1].strip())
            i += 2
            if not lines[i].startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("unexpected dump format near box")
            bounds = []
            for j in range(3):
                lo, hi, *_ = map(float, lines[i + 1 + j].split())
                bounds.append((lo, hi))
            cell = [
                bounds[0][1] - bounds[0][0],
                bounds[1][1] - bounds[1][0],
                bounds[2][1] - bounds[2][0],
            ]
            i += 4
            header = lines[i].split()[2:]
            idx = {{name: n for n, name in enumerate(header)}}
            i += 1
            rows = [lines[i + j].split() for j in range(nat)]
            i += nat
            atoms = []
            for row in rows:
                typ = int(row[idx["type"]])
                sym = symbols_by_type[typ]
                xkey = "x" if "x" in idx else "xu"
                ykey = "y" if "y" in idx else "yu"
                zkey = "z" if "z" in idx else "zu"
                atoms.append(
                    (
                        sym,
                        (
                            float(row[idx[xkey]]),
                            float(row[idx[ykey]]),
                            float(row[idx[zkey]]),
                        ),
                    )
                )
            atoms.sort(key=lambda item: order[item[0]])
            frames.append(
                Atoms(
                    [atom[0] for atom in atoms],
                    positions=[atom[1] for atom in atoms],
                    cell=cell,
                    pbc=True,
                )
            )

        for n, atoms in enumerate(frames):
            write(f"POSCAR_{{n}}", atoms, format="vasp", direct=True, vasp5=True, sort=False)
        print(f"wrote {{len(frames)}} POSCAR frames")
        """
    )


def render_lammps_dump_prep_csh(type_elements: dict[int, str], masses: dict[str, float]) -> str:
    missing = [element for element in type_elements.values() if element not in masses]
    if missing:
        raise ValueError(f"No masses available for element(s): {', '.join(missing)}")
    lines = [
        "#!/bin/csh",
        "",
        "if ( ! -e POSCAR_0 ) then",
        "  python lmp_pos.py",
        "endif",
        "",
        "set datafile = `grep '^read_data' ../in* | awk '{print $2}'`",
        "@ natoms = `head -3 ../$datafile | tail -1 | awk '{print $1}'`",
        "set step = `grep '^timestep[[:space:]]' ../in* | head -1 | awk '{print $2}'`",
        f"@ nelms = {len(type_elements)}",
        "rm -f pos latt step param",
        "",
        "echo $nelms > param",
        "head -7 POSCAR_0 | tail -1 >> param",
    ]
    lines.extend(f"echo {masses[element]} >> param" for element in type_elements.values())
    lines.extend(["echo $step >> param", "echo $natoms >> param"])
    lines.extend(f"echo {element} >> param" for element in type_elements.values())
    lines.extend("echo 0.0 >> param" for _ in range(8))
    lines.extend(
        [
            "",
            "@ iter = $1",
            "while ( -e POSCAR_$iter )",
            "  head -5 POSCAR_$iter | tail -3 | awk '{print $1,$2,$3,0,0,0}' >> latt",
            "  tail -$natoms POSCAR_$iter | awk '{print $1,$2,$3,0,0,0}' >> pos",
            "  echo $step >> step",
            "  @ iter = $iter + 1",
            "end",
            "",
        ]
    )
    return "\n".join(lines)


def prep_scripts_main(args: argparse.Namespace) -> dict[str, Any]:
    type_elements = parse_type_element_map(args.type_elements)
    masses = parse_element_mass_map(args.element_masses)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    pos_path = outdir / args.pos_script
    prep_path = outdir / args.prep_script
    manifest_path = outdir / "sluschi_lammps_prep_manifest.json"
    pos_path.write_text(render_lammps_dump_pos_py(type_elements), encoding="utf-8")
    prep_path.write_text(render_lammps_dump_prep_csh(type_elements, masses), encoding="utf-8")
    try:
        prep_path.chmod(0o755)
    except OSError:
        pass
    manifest = {
        "schema": "atomi.sluschi.lammps_prep_scripts.v1",
        "type_elements": {str(k): v for k, v in type_elements.items()},
        "elements": list(type_elements.values()),
        "masses": {element: masses[element] for element in type_elements.values()},
        "pos_script": str(pos_path),
        "prep_script": str(prep_path),
        "usage": [
            "copy lmp.dump, lmp_pos.py, and lmp_prep.csh into a SLUSCHI run01 folder",
            "copy the matching LAMMPS input/read_data files one level above run01",
            "run `csh lmp_prep.csh <frame_start_index>` before MATLAB entropy/main.m",
        ],
        "guard": (
            "The type_elements basis must match the LAMMPS data/dump type map. "
            "Do not reuse Li,K,Cl scripts for pure KCl or other element subsets."
        ),
    }
    write_json(manifest_path, manifest)
    print(f"Wrote SLUSCHI LAMMPS prep scripts: {outdir}")
    print(f"Type basis: {', '.join(f'{idx}={element}' for idx, element in type_elements.items())}")
    return manifest


def _cell_lengths(cell: list[list[float]]) -> list[float]:
    return [math.sqrt(sum(component * component for component in vector)) for vector in cell]


def _cart_from_fractional(frac: list[float], cell: list[list[float]]) -> list[float]:
    return [sum(frac[j] * cell[j][i] for j in range(3)) for i in range(3)]


def _minimum_image(delta: list[float], lengths: list[float]) -> list[float]:
    out = []
    for value, length in zip(delta, lengths):
        if length > 0.0:
            value -= round(value / length) * length
        out.append(value)
    return out


def _distance(a: list[float], b: list[float], lengths: list[float]) -> float:
    delta = _minimum_image([a[i] - b[i] for i in range(3)], lengths)
    return math.sqrt(sum(value * value for value in delta))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _lammps_cell_from_bounds(
    header_line: str, bound_rows: list[list[float]]
) -> tuple[list[list[float]], list[float]]:
    has_tilt = any(token in header_line.split()[3:] for token in ("xy", "xz", "yz"))
    if not has_tilt:
        origin = [bound_rows[0][0], bound_rows[1][0], bound_rows[2][0]]
        cell = [
            [bound_rows[0][1] - bound_rows[0][0], 0.0, 0.0],
            [0.0, bound_rows[1][1] - bound_rows[1][0], 0.0],
            [0.0, 0.0, bound_rows[2][1] - bound_rows[2][0]],
        ]
        return cell, origin
    xy = bound_rows[0][2] if len(bound_rows[0]) >= 3 else 0.0
    xz = bound_rows[1][2] if len(bound_rows[1]) >= 3 else 0.0
    yz = bound_rows[2][2] if len(bound_rows[2]) >= 3 else 0.0
    xlo_bound, xhi_bound = bound_rows[0][:2]
    ylo_bound, yhi_bound = bound_rows[1][:2]
    zlo, zhi = bound_rows[2][:2]
    xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
    xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
    ylo = ylo_bound - min(0.0, yz)
    yhi = yhi_bound - max(0.0, yz)
    return [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]], [xlo, ylo, zlo]


def _coords_from_lammps_atom_row(
    parts: list[str],
    col: dict[str, int],
    *,
    cell: list[list[float]],
    origin: list[float],
    coordinate_preference: str,
) -> list[float]:
    def shifted(keys: tuple[str, str, str]) -> list[float]:
        return [float(parts[col[key]]) - origin[idx] for idx, key in enumerate(keys)]

    def scaled(keys: tuple[str, str, str]) -> list[float]:
        frac = [float(parts[col[key]]) for key in keys]
        return _cart_from_fractional(frac, cell)

    ordered_modes = (
        (("xu", "yu", "zu"), "cart"),
        (("xsu", "ysu", "zsu"), "scaled"),
        (("x", "y", "z"), "cart"),
        (("xs", "ys", "zs"), "scaled"),
    )
    if coordinate_preference == "wrapped":
        ordered_modes = (
            (("x", "y", "z"), "cart"),
            (("xs", "ys", "zs"), "scaled"),
            (("xu", "yu", "zu"), "cart"),
            (("xsu", "ysu", "zsu"), "scaled"),
        )
    for keys, mode in ordered_modes:
        if set(keys) <= set(col):
            return shifted(keys) if mode == "cart" else scaled(keys)
    raise ValueError("LAMMPS dump must contain x/y/z, xu/yu/zu, xs/ys/zs, or xsu/ysu/zsu columns")


def parse_lammps_dump_frames(
    path: Path,
    type_elements: dict[int, str],
    *,
    coordinate_preference: str = "wrapped",
) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    frames: list[dict[str, Any]] = []
    idx_line = 0
    while idx_line < len(lines):
        if not lines[idx_line].startswith("ITEM: TIMESTEP"):
            idx_line += 1
            continue
        timestep = int(float(lines[idx_line + 1].strip()))
        idx_line += 2
        if not lines[idx_line].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Malformed LAMMPS dump near timestep {timestep}: missing atom count")
        natoms = int(lines[idx_line + 1].strip())
        idx_line += 2
        if not lines[idx_line].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"Malformed LAMMPS dump near timestep {timestep}: missing box bounds")
        box_header = lines[idx_line]
        bound_rows = []
        for axis in range(3):
            bound_rows.append([float(value) for value in lines[idx_line + 1 + axis].split()[:3]])
        cell, origin = _lammps_cell_from_bounds(box_header, bound_rows)
        idx_line += 4
        if not lines[idx_line].startswith("ITEM: ATOMS"):
            raise ValueError(f"Malformed LAMMPS dump near timestep {timestep}: missing atom rows")
        header = lines[idx_line].split()[2:]
        col = {name: number for number, name in enumerate(header)}
        idx_line += 1
        atoms = []
        for row_line in lines[idx_line : idx_line + natoms]:
            parts = row_line.split()
            atom_id = int(parts[col["id"]]) if "id" in col else len(atoms)
            typ = int(parts[col["type"]])
            symbol = type_elements.get(typ)
            if symbol is None:
                raise ValueError(f"LAMMPS dump uses type {typ}, missing from --type-elements")
            coords = _coords_from_lammps_atom_row(
                parts,
                col,
                cell=cell,
                origin=origin,
                coordinate_preference=coordinate_preference,
            )
            atoms.append((atom_id, symbol, coords))
        idx_line += natoms
        atoms.sort(key=lambda item: item[0])
        frames.append(
            {
                "timestep": timestep,
                "symbols": [item[1] for item in atoms],
                "coords": [item[2] for item in atoms],
                "cell": cell,
            }
        )
    return frames


def _infer_lammps_frame_stride_md_steps(frames: list[dict[str, Any]], fallback: float | None) -> float:
    if fallback is not None:
        return float(fallback)
    if len(frames) >= 2:
        delta = float(frames[1]["timestep"] - frames[0]["timestep"])
        if delta > 0:
            return delta
    return 1.0


def lammps_prep_main(args: argparse.Namespace) -> dict[str, Any]:
    type_elements = parse_type_element_map(args.type_elements)
    frames = parse_lammps_dump_frames(args.trajectory, type_elements, coordinate_preference=args.coordinate_preference)
    if not frames:
        raise ValueError(f"No frames found in LAMMPS dump trajectory: {args.trajectory}")
    elements = parse_element_list(args.elements)
    if not elements:
        elements = []
        for idx in sorted(type_elements):
            element = type_elements[idx]
            if element not in elements:
                elements.append(element)
    masses = parse_element_mass_map(args.element_masses)
    missing = [element for element in elements if element not in masses]
    if missing:
        raise ValueError(f"No masses available for element(s): {', '.join(missing)}")
    if args.timestep_ps is None and args.timestep_fs is None:
        raise ValueError("LAMMPS prep requires --timestep-ps for units metal runs or --timestep-fs.")
    if args.timestep_ps is not None and args.timestep_fs is not None:
        raise ValueError("Pass only one of --timestep-ps or --timestep-fs.")
    start = max(args.start_frame, 0)
    stop = args.stop_frame if args.stop_frame is not None else len(frames)
    stride = max(args.stride, 1)
    selected = frames[start:stop:stride]
    if not selected:
        raise ValueError("Frame selection is empty.")
    frame_stride_md_steps = _infer_lammps_frame_stride_md_steps(selected, args.frame_stride_md_steps)
    timestep_fs = float(args.timestep_fs) if args.timestep_fs is not None else float(args.timestep_ps) * 1000.0
    step_interval_fs = timestep_fs * frame_stride_md_steps
    counts = {element: 0 for element in elements}
    for symbol in selected[0]["symbols"]:
        if symbol not in counts:
            raise ValueError(f"Frame contains element {symbol!r}, not listed in --elements={','.join(elements)!r}.")
        counts[symbol] += 1
    phase_temp_label = args.phase_temp_label
    if not phase_temp_label and args.phase:
        phase_temp_label = f"{args.phase}_{int(round(args.temperature_k))}" if args.temperature_k is not None else args.phase
    outdir = args.outdir.resolve()
    lattice_frames = [frame["cell"] for frame in selected]
    symbol_frames = [frame["symbols"] for frame in selected]
    frac_frames = [
        [cart_to_frac(coord, frame["cell"]) for coord in frame["coords"]]
        for frame in selected
    ]
    outputs = write_sluschi_native_frames(
        outdir=outdir,
        elements=elements,
        counts=counts,
        masses=masses,
        lattice_frames=lattice_frames,
        symbol_frames=symbol_frames,
        frac_frames=frac_frames,
        step_interval_fs=step_interval_fs,
        phase_temp_label=phase_temp_label,
    )
    manifest = {
        "schema": SCHEMA_LAMMPS_PREP,
        "source_engine": "lammps",
        "trajectory": str(args.trajectory.resolve()),
        "outdir": str(outdir),
        "type_elements": {str(key): value for key, value in type_elements.items()},
        "elements": elements,
        "counts": counts,
        "natoms": len(selected[0]["symbols"]),
        "n_source_frames": len(frames),
        "n_selected_frames": len(selected),
        "start_frame": start,
        "stop_frame": stop,
        "stride": stride,
        "timestep_ps": float(args.timestep_ps) if args.timestep_ps is not None else timestep_fs / 1000.0,
        "timestep_fs": timestep_fs,
        "frame_stride_md_steps": frame_stride_md_steps,
        "sluschi_step_ps": step_interval_fs / 1000.0,
        "coordinate_preference": args.coordinate_preference,
        "temperature_K": args.temperature_k,
        "phase": args.phase,
        "phase_temp_label": phase_temp_label,
        "outputs": outputs,
        "notes": [
            "This command prepares existing LAMMPS dump frames for the SLUSCHI entropy-prior lane.",
            "The SLUSCHI pos file is written as Cartesian Angstrom coordinates regardless of whether the dump stores x/y/z, xu/yu/zu, xs/ys/zs, or xsu/ysu/zsu.",
            "For Svib, prefer unwrapped LAMMPS dump columns (xu/yu/zu or xsu/ysu/zsu) to avoid periodic-boundary velocity spikes.",
            "Use separate validated NVT solid/liquid windows; this is not a true SLUSCHI coexistence melting calculation.",
        ],
    }
    write_json(outdir / "sluschi_lammps_prep_manifest.json", manifest)
    print(f"Wrote LAMMPS dump-to-SLUSCHI prep files: {outdir}")
    print(f"Selected frames: {len(selected)} / {len(frames)}")
    return manifest


def phase_window_frames_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.engine == "lammps-dump":
        type_elements = parse_type_element_map(args.type_elements)
        frames = parse_lammps_dump_frames(args.trajectory, type_elements)
    elif args.engine == "cp2k-xyz":
        raw_frames = read_cp2k_xyz_frames(args.trajectory)
        cell = None
        if args.cell_vector:
            vectors = [parse_cell_vector(item) for item in args.cell_vector]
            if len(vectors) != 3:
                raise ValueError("--cell-vector must be supplied exactly three times for A, B, C.")
            cell = vectors
        if cell is None:
            cell = parse_cell_abc(args.cell)
        if cell is None:
            cell = cell_from_cp2k_input(args.inp)
        if cell is None and raw_frames:
            cell = cell_from_xyz_comment(raw_frames[0].get("comment", ""))
        if cell is None:
            raise ValueError("CP2K XYZ sampling needs --cell, three --cell-vector values, --inp, or extxyz Lattice metadata.")
        frames = [
            {"timestep": index, "symbols": frame["symbols"], "coords": frame["coords"], "cell": cell}
            for index, frame in enumerate(raw_frames)
        ]
    elif args.engine == "vasp-xdatcar":
        if args.poscar is None:
            raise ValueError("VASP XDATCAR sampling requires --poscar")
        basis = read_vasp_poscar_basis(args.poscar)
        frac_frames = read_vasp_xdatcar_frames(args.trajectory, basis["natoms"])
        frames = [
            {
                "timestep": index,
                "symbols": basis["symbols"],
                "coords": [_cart_from_fractional(frac, basis["lattice"]) for frac in coords],
                "cell": basis["lattice"],
            }
            for index, coords in enumerate(frac_frames)
        ]
    else:
        raise ValueError(f"Unsupported phase-window engine: {args.engine}")
    start = max(args.start_frame, 0)
    stop = args.stop_frame if args.stop_frame is not None else len(frames)
    stride = max(args.stride, 1)
    selected = frames[start:stop:stride]
    if not selected:
        raise ValueError(f"No trajectory frames selected from {args.trajectory}")
    return selected


def unwrap_window_coords(frames: list[dict[str, Any]]) -> list[list[list[float]]]:
    if not frames:
        return []
    lengths = _cell_lengths(frames[0]["cell"])
    unwrapped = [[coord[:] for coord in frames[0]["coords"]]]
    previous_wrapped = frames[0]["coords"]
    previous_unwrapped = unwrapped[0]
    for frame in frames[1:]:
        current: list[list[float]] = []
        for atom_idx, coord in enumerate(frame["coords"]):
            delta = _minimum_image([coord[i] - previous_wrapped[atom_idx][i] for i in range(3)], lengths)
            current.append([previous_unwrapped[atom_idx][i] + delta[i] for i in range(3)])
        unwrapped.append(current)
        previous_wrapped = frame["coords"]
        previous_unwrapped = current
    return unwrapped


def phase_window_metrics(
    frames: list[dict[str, Any]],
    *,
    species_a: str,
    species_b: str,
    neighbor_cutoff_a: float,
) -> dict[str, Any]:
    symbols = frames[0]["symbols"]
    idx_a = [idx for idx, symbol in enumerate(symbols) if symbol == species_a]
    idx_b = [idx for idx, symbol in enumerate(symbols) if symbol == species_b]
    if not idx_a or not idx_b:
        raise ValueError(f"Selected frames do not contain requested species pair {species_a}-{species_b}")
    lengths = _cell_lengths(frames[0]["cell"])
    unwrapped = unwrap_window_coords(frames)
    displacements = []
    for atom_idx in range(len(symbols)):
        delta = [unwrapped[-1][atom_idx][axis] - unwrapped[0][atom_idx][axis] for axis in range(3)]
        displacements.append(math.sqrt(sum(value * value for value in delta)))
    nearest_ab: list[float] = []
    coord_ab: list[float] = []
    for frame in frames:
        coords = frame["coords"]
        for atom_a in idx_a:
            distances = [_distance(coords[atom_a], coords[atom_b], lengths) for atom_b in idx_b]
            if distances:
                nearest_ab.append(min(distances))
                coord_ab.append(float(sum(1 for dist in distances if dist <= neighbor_cutoff_a)))
    rms = math.sqrt(sum(value * value for value in displacements) / len(displacements)) if displacements else None
    return {
        "rms_displacement_A": rms,
        "mean_displacement_A": _mean(displacements),
        "max_displacement_A": max(displacements) if displacements else None,
        "nearest_ab_mean_A": _mean(nearest_ab),
        "nearest_ab_sd_A": _std(nearest_ab),
        "coord_ab_mean": _mean(coord_ab),
        "coord_ab_sd": _std(coord_ab),
        "n_species_a": len(idx_a),
        "n_species_b": len(idx_b),
        "n_atoms": len(symbols),
    }


def classify_phase_window(metrics: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    rms = metrics.get("rms_displacement_A")
    nearest_sd = metrics.get("nearest_ab_sd_A")
    coord = metrics.get("coord_ab_mean")
    mode = getattr(args, "liquid_check_mode", "generic")
    solid_votes = 0
    liquid_votes = 0
    notes: list[str] = []
    if mode == "network":
        if coord is not None and coord >= args.solid_coord_min:
            notes.append(
                "Network-liquid mode: high A-B coordination is reported as a network descriptor, not a solid-like vote."
            )
        if nearest_sd is not None and nearest_sd <= args.solid_nearest_sd_max_a:
            notes.append(
                "Network-liquid mode: narrow A-B nearest-neighbor spread can persist in molten actinide/lanthanide chloride networks."
            )
        if rms is None:
            notes.append("Network-liquid mode requires a mobility metric; label remains mixed.")
            return "mixed", notes
        if rms <= args.solid_rms_max_a:
            notes.append("Network-liquid mode: bounded window RMS is still treated as solid-like or unequilibrated.")
            return "solid-like", notes
        if rms >= args.liquid_rms_min_a:
            notes.append("Network-liquid mode: mobility threshold crossed; confirm with longer MSD/tail-window checks.")
            return "liquid-like", notes
        notes.append(
            "Network-liquid mode: mobility is intermediate; keep as mixed until longer MSD and thermodynamic stability support a liquid tail."
        )
        return "mixed", notes
    if rms is not None:
        if rms <= args.solid_rms_max_a:
            solid_votes += 1
        if rms >= args.liquid_rms_min_a:
            liquid_votes += 1
    if nearest_sd is not None:
        if nearest_sd <= args.solid_nearest_sd_max_a:
            solid_votes += 1
        if nearest_sd >= args.liquid_nearest_sd_min_a:
            liquid_votes += 1
    if coord is not None:
        if coord >= args.solid_coord_min:
            solid_votes += 1
        if args.liquid_coord_max is not None and coord <= args.liquid_coord_max:
            liquid_votes += 1
    if solid_votes >= 2 and liquid_votes == 0:
        return "solid-like", notes
    if liquid_votes >= 2 and solid_votes == 0:
        return "liquid-like", notes
    if solid_votes and liquid_votes:
        notes.append("Solid-like and liquid-like metrics disagree; treat as mixed/window-boundary region.")
    else:
        notes.append("Too few metrics crossed conservative phase thresholds.")
    return "mixed", notes


def phase_window_sample_main(args: argparse.Namespace) -> dict[str, Any]:
    frames = phase_window_frames_from_args(args)
    window_frames = max(2, int(round(args.window_ps / args.frame_step_ps)) + 1)
    stride_frames = max(1, int(round(args.stride_ps / args.frame_step_ps)))
    if len(frames) < window_frames:
        raise ValueError(
            f"Need at least {window_frames} selected frames for a {args.window_ps:g} ps window; got {len(frames)}."
        )
    rows: list[dict[str, Any]] = []
    for start in range(0, len(frames) - window_frames + 1, stride_frames):
        stop = start + window_frames
        metrics = phase_window_metrics(
            frames[start:stop],
            species_a=args.species_a,
            species_b=args.species_b,
            neighbor_cutoff_a=args.neighbor_cutoff_a,
        )
        label, notes = classify_phase_window(metrics, args)
        row = {
            "window_index": len(rows),
            "frame_start": start,
            "frame_stop_exclusive": stop,
            "time_start_ps": start * args.frame_step_ps,
            "time_stop_ps": (stop - 1) * args.frame_step_ps,
            "phase_window_label": label,
            "species_a": args.species_a,
            "species_b": args.species_b,
            "neighbor_cutoff_A": args.neighbor_cutoff_a,
            "notes": "; ".join(notes),
            **metrics,
        }
        rows.append(row)
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["phase_window_label"])
        counts[label] = counts.get(label, 0) + 1
    outdir = args.outdir.resolve()
    fields = [
        "window_index",
        "frame_start",
        "frame_stop_exclusive",
        "time_start_ps",
        "time_stop_ps",
        "phase_window_label",
        "species_a",
        "species_b",
        "neighbor_cutoff_A",
        "rms_displacement_A",
        "mean_displacement_A",
        "max_displacement_A",
        "nearest_ab_mean_A",
        "nearest_ab_sd_A",
        "coord_ab_mean",
        "coord_ab_sd",
        "n_species_a",
        "n_species_b",
        "n_atoms",
        "notes",
    ]
    csv_path = outdir / "sluschi_phase_windows.csv"
    json_path = outdir / "sluschi_phase_windows.json"
    write_csv(csv_path, rows, fields)
    payload = {
        "schema": SCHEMA_PHASE_WINDOW_SAMPLE,
        "engine": args.engine,
        "trajectory": str(args.trajectory.resolve()),
        "n_selected_frames": len(frames),
        "window_ps": args.window_ps,
        "stride_ps": args.stride_ps,
        "frame_step_ps": args.frame_step_ps,
        "window_frames": window_frames,
        "stride_frames": stride_frames,
        "species_pair": [args.species_a, args.species_b],
        "thresholds": {
            "neighbor_cutoff_A": args.neighbor_cutoff_a,
            "liquid_check_mode": args.liquid_check_mode,
            "solid_rms_max_A": args.solid_rms_max_a,
            "liquid_rms_min_A": args.liquid_rms_min_a,
            "solid_nearest_sd_max_A": args.solid_nearest_sd_max_a,
            "liquid_nearest_sd_min_A": args.liquid_nearest_sd_min_a,
            "solid_coord_min": args.solid_coord_min,
            "liquid_coord_max": args.liquid_coord_max,
        },
        "counts": counts,
        "outputs": {"csv": str(csv_path), "json": str(json_path)},
        "windows": rows,
        "notes": [
            "This is a conservative generic phase-window screen, not a replacement for SLUSCHI coexistence analysis.",
            "Use species/thresholds appropriate to the chemistry; alkali-halide defaults are only screening priors.",
            "For actinide/lanthanide chloride network liquids, use --liquid-check-mode network so cation-Cl coordination is not misused as a solidness veto.",
            "Accept phase-specific entropy only from windows whose label agrees with the intended phase and whose thermodynamics are stable.",
        ],
    }
    write_json(json_path, payload)
    print(f"Phase windows: {counts}")
    print(f"Wrote phase-window CSV: {csv_path}")
    return payload


def cp2k_prep_main(args: argparse.Namespace) -> dict[str, Any]:
    frames = read_cp2k_xyz_frames(args.xyz)
    if not frames:
        raise ValueError(f"No frames found in CP2K XYZ trajectory: {args.xyz}")
    elements = parse_element_list(args.elements)
    if not elements:
        seen: list[str] = []
        for symbol in frames[0]["symbols"]:
            if symbol not in seen:
                seen.append(symbol)
        elements = seen
    masses = parse_element_mass_map(args.element_masses)
    missing = [element for element in elements if element not in masses]
    if missing:
        raise ValueError(f"No masses available for element(s): {', '.join(missing)}")
    cell = None
    if args.cell_vector:
        vectors = [parse_cell_vector(item) for item in args.cell_vector]
        if len(vectors) != 3:
            raise ValueError("--cell-vector must be supplied exactly three times for A, B, C.")
        cell = vectors
    if cell is None:
        cell = parse_cell_abc(args.cell)
    if cell is None:
        cell = cell_from_cp2k_input(args.inp)
    if cell is None:
        cell = cell_from_xyz_comment(frames[0]["comment"])
    if cell is None:
        raise ValueError("No cell found. Pass --cell, three --cell-vector values, --inp with &CELL, or extxyz Lattice metadata.")

    start = max(args.start_frame, 0)
    stop = args.stop_frame if args.stop_frame is not None else len(frames)
    stride = max(args.stride, 1)
    selected = frames[start:stop:stride]
    if not selected:
        raise ValueError("Frame selection is empty.")
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    counts = {element: 0 for element in elements}
    for symbol in selected[0]["symbols"]:
        if symbol not in counts:
            raise ValueError(f"Frame contains element {symbol!r}, not listed in --elements={','.join(elements)!r}.")
        counts[symbol] += 1
    natoms = len(selected[0]["symbols"])
    timestep_fs = args.timestep_fs
    if timestep_fs is None:
        timestep_fs = 1.0
    step_interval_fs = timestep_fs * args.frame_stride_md_steps

    lattice_frames = [cell_from_xyz_comment(frame["comment"]) or cell for frame in selected]
    symbol_frames = [frame["symbols"] for frame in selected]
    frac_frames = [
        [cart_to_frac(coord, frame_cell) for coord in frame["coords"]]
        for frame, frame_cell in zip(selected, lattice_frames)
    ]
    outputs = write_sluschi_native_frames(
        outdir=outdir,
        elements=elements,
        counts=counts,
        masses=masses,
        lattice_frames=lattice_frames,
        symbol_frames=symbol_frames,
        frac_frames=frac_frames,
        step_interval_fs=step_interval_fs,
        phase_temp_label=args.phase,
    )
    manifest = {
        "schema": SCHEMA_CP2K_PREP,
        "source_engine": "cp2k",
        "xyz": str(args.xyz.resolve()),
        "inp": str(args.inp.resolve()) if args.inp else "",
        "outdir": str(outdir),
        "elements": elements,
        "counts": counts,
        "natoms": natoms,
        "n_source_frames": len(frames),
        "n_selected_frames": len(selected),
        "start_frame": start,
        "stop_frame": stop,
        "stride": stride,
        "timestep_fs": timestep_fs,
        "frame_stride_md_steps": args.frame_stride_md_steps,
        "sluschi_step_ps": step_interval_fs / 1000.0,
        "phase": args.phase,
        "cell_angstrom": cell,
        "outputs": outputs,
        "notes": [
            "This command prepares existing CP2K AIMD XYZ frames for the SLUSCHI entropy-prior lane.",
            "It does not run CP2K and it does not run true SLUSCHI coexistence/MPFit melting.",
            "Use separate solid and liquid AIMD trajectories when benchmarking against Hong/Shang entropy values.",
        ],
    }
    write_json(outdir / "sluschi_cp2k_prep_manifest.json", manifest)
    print(f"Wrote CP2K-to-SLUSCHI prep files: {outdir}")
    print(f"Selected frames: {len(selected)} / {len(frames)}")
    return manifest


def vasp_prep_main(args: argparse.Namespace) -> dict[str, Any]:
    basis = read_vasp_poscar_basis(args.poscar)
    elements = parse_element_list(args.elements) or list(basis["elements"])
    masses = parse_element_mass_map(args.element_masses)
    missing = [element for element in elements if element not in masses]
    if missing:
        raise ValueError(f"No masses available for element(s): {', '.join(missing)}")
    natoms = int(basis["natoms"])
    frames = read_vasp_xdatcar_frames(args.xdatcar, natoms)
    if not frames:
        raise ValueError(f"No Direct configuration frames found in XDATCAR: {args.xdatcar}")
    start = max(args.start_frame, 0)
    stop = args.stop_frame if args.stop_frame is not None else len(frames)
    stride = max(args.stride, 1)
    selected = frames[start:stop:stride]
    if not selected:
        raise ValueError("Frame selection is empty.")
    counts = {element: 0 for element in elements}
    for element, count in zip(basis["elements"], basis["counts"]):
        if element not in counts:
            raise ValueError(f"POSCAR contains element {element!r}, not listed in --elements={','.join(elements)!r}.")
        counts[element] += int(count)
    step_interval_fs = args.timestep_fs * args.frame_stride_md_steps
    outdir = args.outdir.resolve()
    phase_temp_label = args.phase_temp_label
    if not phase_temp_label and args.phase:
        phase_temp_label = f"{args.phase}_{int(round(args.temperature_k))}" if args.temperature_k is not None else args.phase
    outputs = write_sluschi_native_frames(
        outdir=outdir,
        elements=elements,
        counts=counts,
        masses=masses,
        lattice_frames=[basis["lattice"] for _ in selected],
        symbol_frames=[basis["symbols"] for _ in selected],
        frac_frames=selected,
        step_interval_fs=step_interval_fs,
        phase_temp_label=phase_temp_label,
    )
    manifest = {
        "schema": SCHEMA_VASP_PREP,
        "source_engine": "vasp",
        "poscar": str(args.poscar.resolve()),
        "xdatcar": str(args.xdatcar.resolve()),
        "outdir": str(outdir),
        "elements": elements,
        "counts": counts,
        "natoms": natoms,
        "n_source_frames": len(frames),
        "n_selected_frames": len(selected),
        "start_frame": start,
        "stop_frame": stop,
        "stride": stride,
        "timestep_fs": args.timestep_fs,
        "frame_stride_md_steps": args.frame_stride_md_steps,
        "sluschi_step_ps": step_interval_fs / 1000.0,
        "temperature_K": args.temperature_k,
        "phase": args.phase,
        "phase_temp_label": phase_temp_label,
        "cell_angstrom": basis["lattice"],
        "outputs": outputs,
        "notes": [
            "This command prepares existing VASP XDATCAR AIMD frames for the SLUSCHI entropy-prior lane.",
            "It does not run VASP and it does not run true SLUSCHI coexistence/MPFit melting.",
            "For production entropy, validate thermostat equilibration, XDATCAR stride, phase health, and element order.",
        ],
    }
    write_json(outdir / "sluschi_vasp_prep_manifest.json", manifest)
    print(f"Wrote VASP XDATCAR-to-SLUSCHI prep files: {outdir}")
    print(f"Selected frames: {len(selected)} / {len(frames)}")
    return manifest


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


@dataclass
class MeltingAnchor:
    source_file: str
    melting_temperature_K: float
    temperature_std_error_K: float | None
    method: str
    quality: str
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
    if len(pair_values) > 1 and mean_pair is not None:
        variance = sum((value - mean_pair) ** 2 for value in pair_values) / (len(pair_values) - 1)
        std_pair = variance**0.5
        sem_pair = std_pair / (len(pair_values) ** 0.5)
    else:
        std_pair = None
        sem_pair = None
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
        "std_pair_sconfig_J_mol_atom_K": std_pair,
        "sem_pair_sconfig_J_mol_atom_K": sem_pair,
        "min_pair_sconfig_J_mol_atom_K": min(pair_values) if pair_values else None,
        "max_pair_sconfig_J_mol_atom_K": max(pair_values) if pair_values else None,
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
            "std_pair_sconfig_J_mol_atom_K",
            "sem_pair_sconfig_J_mol_atom_K",
            "min_pair_sconfig_J_mol_atom_K",
            "max_pair_sconfig_J_mol_atom_K",
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


def parse_sluschi_svib_from_text(text: str) -> tuple[list[float], str]:
    svib_lines = []
    for line in text.splitlines():
        match = re.search(r"Svib:\s+(.+)", line)
        if not match:
            continue
        values = []
        for token in match.group(1).split():
            try:
                values.append(float(token))
            except ValueError:
                break
        if values:
            svib_lines.append((values, line.strip()))
    for values, line in svib_lines:
        if "constrained" in line.lower():
            return values, line
    for values, line in svib_lines:
        lowered = line.lower()
        if "use this value" in lowered and "do not use" not in lowered:
            return values, line
    if svib_lines:
        return svib_lines[-1]
    return [], ""


def validate_sluschi_svib_outputs(root: Path, values: list[float], line: str) -> tuple[list[float], bool, str]:
    """Guard against SLUSCHI's zero-valued fallback being parsed as real Svib."""
    if not values:
        return values, False, "missing_svib_line"
    if not all(math.isfinite(value) for value in values):
        return [], False, "nonfinite_svib_line"
    all_zero = all(abs(value) < 1.0e-12 for value in values)
    vib_candidates = [root / "vib.out", root / "entropy" / "vib.out"]
    entropy_candidates = [root / "entropy.out", root / "entropy" / "entropy.out"]
    existing = [path for path in vib_candidates + entropy_candidates if path.is_file()]
    empty_support = bool(existing) and all(path.stat().st_size == 0 for path in existing)
    nonfinite_support = False
    for path in existing:
        if path.stat().st_size == 0:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if re.search(r"(^|[^a-z])nan([^a-z]|$)|(^|[^a-z])inf([^a-z]|$)", text):
            nonfinite_support = True
            break
    lowered = line.lower()
    constrained_zero = all_zero and ("constrained" in lowered or "use this value" in lowered)
    if constrained_zero and (empty_support or nonfinite_support):
        reason = "zero_constrained_svib_with_empty_support" if empty_support else "zero_constrained_svib_with_nonfinite_support"
        return [], False, reason
    if constrained_zero:
        return [], False, "zero_constrained_svib"
    return values, True, "ok"


def load_collect_text(root: Path, collect_path: Path | None = None) -> tuple[str, Path | None]:
    candidates = [collect_path] if collect_path else [
        root / "collect.stdout",
        root / "collect.out",
        root / "sluschi_collect.out",
        root / "entropy" / "entropy.out",
    ]
    chunks = []
    used: list[Path] = []
    for path in candidates:
        if path and path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            used.append(path)
    return "\n".join(chunks), used[0] if used else None


def type_stoich_weights(args: argparse.Namespace, n_types: int) -> dict[int, float]:
    raw = parse_key_float_map(args.type_stoich)
    weights: dict[int, float] = {}
    for key, value in raw.items():
        key_clean = key.strip().lower().replace("type", "")
        try:
            idx = int(key_clean)
        except ValueError as exc:
            raise ValueError(f"--type-stoich keys must be type indices such as 1=2,2=1; got {key!r}") from exc
        weights[idx] = value
    if not weights and args.atoms_per_formula:
        if n_types == 1:
            weights[1] = args.atoms_per_formula
        else:
            weights = {idx: args.atoms_per_formula / n_types for idx in range(1, n_types + 1)}
    if not weights:
        weights = {idx: 1.0 for idx in range(1, n_types + 1)}
    missing = [idx for idx in range(1, n_types + 1) if idx not in weights]
    if missing:
        raise ValueError(f"--type-stoich is missing type(s): {missing}")
    return weights


def infer_n_types_from_type_stoich(raw_type_stoich: str) -> int:
    max_type = 0
    for key in parse_key_float_map(raw_type_stoich):
        key_clean = key.strip().lower().replace("type", "")
        try:
            max_type = max(max_type, int(key_clean))
        except ValueError:
            continue
    return max_type


def _pair_type_indices(pair: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", pair)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def select_sconfig_pair_values(pairs: list[SconfigPair], mode: str) -> tuple[list[SconfigPair], str]:
    if mode == "mean-all-pairs":
        return pairs, "mean of all SLUSCHI pair recommendations"
    if mode == "same-species-liquid":
        same_liquid = [
            pair
            for pair in pairs
            if (indices := _pair_type_indices(pair.pair)) is not None
            and indices[0] == indices[1]
            and "liquid" in pair.state
        ]
        if same_liquid:
            return same_liquid, "Hong/Shang liquid rule: same-species liquid pair recommendations"
        same_any = [
            pair
            for pair in pairs
            if (indices := _pair_type_indices(pair.pair)) is not None and indices[0] == indices[1]
        ]
        if same_any:
            return same_any, "fallback same-species pair recommendations"
        return pairs, "fallback mean of all SLUSCHI pair recommendations"
    raise ValueError(f"Unsupported Sconf reduction mode: {mode!r}")


def mean_sem(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean_value = sum(values) / len(values)
    if len(values) == 1:
        return mean_value, None, None
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    std_value = variance**0.5
    sem_value = std_value / (len(values) ** 0.5)
    return mean_value, std_value, sem_value


def entropy_summary_main(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    text, collect_used = load_collect_text(root, args.collect)
    if not text.strip():
        raise FileNotFoundError(f"No collect/entropy text found under {root}")
    svib_values, svib_line = parse_sluschi_svib_from_text(text)
    svib_values, svib_valid, svib_status = validate_sluschi_svib_outputs(root, svib_values, svib_line)
    pairs = parse_sconfig_pairs(root)
    summary_ns = argparse.Namespace(
        root=root,
        system=args.system,
        formula=args.formula,
        components=args.components,
        phase=args.phase,
        temperature_k=args.temperature_k,
        composition=args.composition,
        quality=args.quality,
        dump_stride_note=args.dump_stride_note,
    )
    sconfig_summary = summarize_sconfig_case(summary_ns, pairs)
    selected_pairs, sconf_reduction_note = select_sconfig_pair_values(pairs, args.sconf_reduction)
    selected_values = [pair.sconfig_J_mol_atom_K for pair in selected_pairs]
    selected_sconf_atom, selected_sconf_std_atom, selected_sconf_sem_atom = mean_sem(selected_values)
    n_weight_types = len(svib_values) or infer_n_types_from_type_stoich(args.type_stoich)
    weights = type_stoich_weights(args, n_weight_types) if n_weight_types else {}
    atoms_per_formula = sum(weights.values()) if weights else (args.atoms_per_formula or None)
    svib_formula = sum(weights[idx] * svib_values[idx - 1] for idx in weights) if svib_values else None
    sconf_atom = selected_sconf_atom
    sconf_sem_atom = selected_sconf_sem_atom
    sconf_formula = sconf_atom * atoms_per_formula if sconf_atom is not None and atoms_per_formula else None
    sconf_sem_formula = sconf_sem_atom * atoms_per_formula if sconf_sem_atom is not None and atoms_per_formula else None
    total = (svib_formula + (sconf_formula or 0.0)) if svib_formula is not None else None
    row: dict[str, Any] = {
        "temperature_K": args.temperature_k,
        "system": args.system,
        "formula": args.formula,
        "phase": args.phase,
        "composition": args.composition,
        "atoms_per_formula": atoms_per_formula,
        "Svib_J_mol_formula_K": svib_formula,
        "Sconf_J_mol_formula_K": sconf_formula,
        "Sconf_stderr_J_mol_formula_K": sconf_sem_formula,
        "Stotal_J_mol_formula_K": total,
        "total_entropy_stderr_J_mol_formula_K": sconf_sem_formula,
        "Sconf_reduction": args.sconf_reduction,
        "Sconf_reduction_note": sconf_reduction_note,
        "Sconf_selected_n_pairs": len(selected_pairs),
        "Sconf_selected_pairs": ",".join(pair.pair for pair in selected_pairs),
        "Sconf_selected_mean_J_mol_atom_K": selected_sconf_atom,
        "Sconf_selected_std_J_mol_atom_K": selected_sconf_std_atom,
        "Sconf_selected_sem_J_mol_atom_K": selected_sconf_sem_atom,
        "Sconf_all_pair_mean_J_mol_atom_K": sconfig_summary.get("mean_pair_sconfig_J_mol_atom_K"),
        "Sconf_all_pair_sem_J_mol_atom_K": sconfig_summary.get("sem_pair_sconfig_J_mol_atom_K"),
        "quality": args.quality,
        "source": str(root),
        "collect_file": str(collect_used) if collect_used else "",
        "svib_line": svib_line,
        "Svib_valid": svib_valid,
        "Svib_status": svib_status,
    }
    for idx, value in enumerate(svib_values, start=1):
        row[f"Svib_type{idx}_J_mol_atom_K"] = value
        row[f"type{idx}_stoich"] = weights.get(idx, "")
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "sluschi_entropy_summary.csv"
    fields = list(row)
    write_csv(summary_csv, [row], fields)
    write_csv(
        outdir / "lammps_sconfig_pairs.csv",
        [pair.__dict__ for pair in pairs],
        ["file", "pair", "state", "recommended_statistic", "sconfig_J_mol_atom_K", "line"],
    )
    payload = {
        "schema": SCHEMA_ENTROPY_SUMMARY,
        "workflow_lane": "entropy_prior",
        "method_contract": {
            "phase_gate": "Use only accepted single-phase, stable NVT tail windows; RDF/PDF alone is insufficient.",
            "svib": "Constrained/use-this-value Svib line from collect.stdout; reject zero fallback, NaN, empty support files, or invalid MDS frame windows.",
            "sconf": "SLUSCHI pair-channel recommendations reduced with explicit Sconf_reduction; same-species-liquid is for pure binary liquid KCl/Hong-Shang style benchmarks, not a universal mixed-salt rule.",
            "units": "Record coordinate/lattice/time units and report entropy basis explicitly before plotting or MIVM/pycalphad handoff.",
            "quality": "Use block/tail uncertainty and quality tiers; screening-prior rows are not final thermodynamic anchors.",
        },
        "root": str(root),
        "collect_file": str(collect_used) if collect_used else None,
        "n_svib_types": len(svib_values),
        "svib_valid": svib_valid,
        "svib_status": svib_status,
        "n_pair_recommendations": len(pairs),
        "type_stoich": {str(key): value for key, value in weights.items()},
        "outputs": {"summary_csv": str(summary_csv)},
        "summary": row,
        "sconfig_summary": sconfig_summary,
        "notes": [
            "This command parses an existing SLUSCHI/LAMMPS postprocess folder as an entropy prior.",
            "It is not, by itself, a small-cell solid-liquid coexistence melting calculation.",
            "Svib values are taken from the constrained/use-this-value SLUSCHI line when present and rejected if SLUSCHI only emitted a zero fallback with empty/non-finite vibrational support files.",
            "Sconf is reduced from SLUSCHI pair recommendations according to Sconf_reduction, then multiplied by atoms_per_formula.",
            "For pure binary liquid KCl benchmarking against Hong/Shang Fig. 3, use --sconf-reduction same-species-liquid; keep mean-all-pairs as a diagnostic.",
            "For mixed salts, choose a chemically documented pair selector; do not blindly apply KCl same-species-liquid to networked multicomponent systems.",
            "For production entropy, validate dense NVT sampling, type ordering, SLUSCHI frame spacing, and phase health.",
        ],
    }
    write_json(outdir / "sluschi_entropy_summary.json", payload)
    print(f"Wrote SLUSCHI entropy summary: {summary_csv}")
    return payload


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


@dataclass(frozen=True)
class SluschiParamLayout:
    n_elements: int
    counts: list[int]
    masses: list[float]
    timestep: float
    natoms: int
    timestep_line_index: int
    natoms_line_index: int


def _format_sluschi_float(value: float) -> str:
    return f"{value:.12g}"


def _sluschi_param_layout(param_path: Path) -> SluschiParamLayout:
    lines = _read_lines(param_path)
    if len(lines) < 6:
        raise ValueError(f"SLUSCHI param file is too short: {param_path}")
    try:
        n_elements = int(float(lines[0].split()[0]))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse SLUSCHI element count from {param_path}") from exc
    counts_line_index = 1
    masses_start = 2
    timestep_line_index = masses_start + n_elements
    natoms_line_index = timestep_line_index + 1
    if len(lines) <= natoms_line_index:
        raise ValueError(
            f"SLUSCHI param file {param_path} is too short for {n_elements} element(s); "
            f"expected natoms on line {natoms_line_index + 1}"
        )
    try:
        counts = [int(float(item)) for item in lines[counts_line_index].split()]
        masses = [float(lines[masses_start + idx].split()[0]) for idx in range(n_elements)]
        timestep = float(lines[timestep_line_index].split()[0])
        natoms = int(float(lines[natoms_line_index].split()[0]))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse SLUSCHI param layout from {param_path}") from exc
    if len(counts) != n_elements:
        raise ValueError(f"SLUSCHI param counts length {len(counts)} does not match n_elements={n_elements}")
    if sum(counts) != natoms:
        raise ValueError(f"SLUSCHI param natoms={natoms} does not match sum(counts)={sum(counts)}")
    return SluschiParamLayout(
        n_elements=n_elements,
        counts=counts,
        masses=masses,
        timestep=timestep,
        natoms=natoms,
        timestep_line_index=timestep_line_index,
        natoms_line_index=natoms_line_index,
    )


def _sluschi_param_natoms(param_path: Path) -> int:
    return _sluschi_param_layout(param_path).natoms


def _sluschi_step_multiplier_to_fs(unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized == "fs":
        return 1.0
    if normalized == "ps":
        return 1000.0
    raise ValueError(f"Unsupported prepared step unit {unit!r}; expected 'ps' or 'fs'")


def _convert_sluschi_step_lines_to_fs(lines: list[str], *, prepared_step_unit: str) -> list[str]:
    multiplier = _sluschi_step_multiplier_to_fs(prepared_step_unit)
    converted: list[str] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            value = float(parts[0]) * multiplier
        except ValueError as exc:
            raise ValueError(f"Could not parse SLUSCHI step value {line!r}") from exc
        converted.append(_format_sluschi_float(value))
    return converted


def _write_mds_param_with_step_fs(param: Path, target: Path, *, prepared_step_unit: str) -> tuple[float, float]:
    layout = _sluschi_param_layout(param)
    multiplier = _sluschi_step_multiplier_to_fs(prepared_step_unit)
    converted_timestep = layout.timestep * multiplier
    lines = _read_lines(param)
    lines[layout.timestep_line_index] = _format_sluschi_float(converted_timestep)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return layout.timestep, converted_timestep


def _write_legacy_mds_latt_step(
    *,
    prepared_root: Path,
    workdir: Path,
    block_size: int,
    prepared_step_unit: str,
) -> dict[str, Any]:
    if block_size <= 0:
        raise ValueError("--legacy-mds-block-size must be positive")
    param = prepared_root / "param"
    pos = prepared_root / "pos"
    latt = prepared_root / "latt"
    step = prepared_root / "step"
    for path in (param, pos, latt, step):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared SLUSCHI file: {path}")
    if block_size != 80:
        raise ValueError("Legacy SLUSCHI MDS read_files.m hard-codes one latt/step record per 80 frames; use --legacy-mds-block-size 80.")
    param_layout = _sluschi_param_layout(param)
    natoms = param_layout.natoms
    pos_lines = _read_lines(pos)
    latt_lines = _read_lines(latt)
    step_lines = [line for line in _read_lines(step) if line.strip()]
    if len(pos_lines) % natoms != 0:
        raise ValueError(f"pos line count {len(pos_lines)} is not divisible by natoms={natoms}")
    n_frames = len(pos_lines) // natoms
    if n_frames % block_size != 0:
        raise ValueError(f"selected frame count {n_frames} is not divisible by legacy MDS block size {block_size}")
    n_blocks = n_frames // block_size
    if len(latt_lines) % 3 != 0:
        raise ValueError(f"latt line count {len(latt_lines)} is not divisible by 3")
    n_latt_records = len(latt_lines) // 3
    if n_latt_records == n_frames:
        block_latt = []
        for block in range(n_blocks):
            offset = block * block_size * 3
            block_latt.extend(latt_lines[offset : offset + 3])
    elif n_latt_records == n_blocks:
        block_latt = latt_lines
    else:
        raise ValueError(
            f"latt has {n_latt_records} records; expected either frame-level {n_frames} "
            f"or legacy block-level {n_blocks}"
        )
    if len(step_lines) == n_frames:
        block_step = [step_lines[block * block_size] for block in range(n_blocks)]
    elif len(step_lines) == n_blocks:
        block_step = step_lines
    else:
        raise ValueError(
            f"step has {len(step_lines)} records; expected either frame-level {n_frames} "
            f"or legacy block-level {n_blocks}"
        )
    workdir.mkdir(parents=True, exist_ok=True)
    input_param_timestep, legacy_param_timestep_fs = _write_mds_param_with_step_fs(
        param,
        workdir / "param",
        prepared_step_unit=prepared_step_unit,
    )
    shutil.copy2(pos, workdir / "pos")
    (workdir / "latt").write_text("\n".join(block_latt) + "\n", encoding="utf-8")
    legacy_block_step_fs = _convert_sluschi_step_lines_to_fs(block_step, prepared_step_unit=prepared_step_unit)
    (workdir / "step").write_text("\n".join(legacy_block_step_fs) + "\n", encoding="utf-8")
    step_multiplier_to_fs = _sluschi_step_multiplier_to_fs(prepared_step_unit)
    return {
        "natoms": natoms,
        "n_elements": param_layout.n_elements,
        "counts": param_layout.counts,
        "n_frames": n_frames,
        "legacy_mds_block_size": block_size,
        "n_legacy_blocks": n_blocks,
        "input_latt_records": n_latt_records,
        "input_step_records": len(step_lines),
        "prepared_step_unit": prepared_step_unit,
        "legacy_mds_step_unit": "fs",
        "step_unit_multiplier_to_fs": step_multiplier_to_fs,
        "input_param_timestep": input_param_timestep,
        "legacy_param_timestep_fs": legacy_param_timestep_fs,
        "legacy_step_values_fs": legacy_block_step_fs,
    }


def _patch_sluschi_mds_entropy_template(entropy_dir: Path) -> list[str]:
    """Patch copied legacy SLUSCHI MDS MATLAB helpers for robust batch runs.

    The upstream MATLAB files are copied into each work directory first; these
    edits are intentionally local to the generated run folder and do not modify
    the user's SLUSCHI installation.
    """
    patches: list[str] = []
    onephase = entropy_dir / "onephase_v6.m"
    if onephase.is_file():
        text = onephase.read_text(encoding="utf-8", errors="replace")
        if "E_1_c = 0;" not in text and "flag_correction=1;" in text:
            text = text.replace(
                "flag_correction=1;\n",
                "flag_correction=1;\narea_c = 0;\nE_1_c = 0;\nF_1_c = 0;\nF_2_c = 0;\n",
                1,
            )
            onephase.write_text(text, encoding="utf-8")
            patches.append("onephase_v6_init_correction_terms")

    pdf = entropy_dir / "pdf_v6.m"
    if pdf.is_file():
        text = pdf.read_text(encoding="utf-8", errors="replace")
        needle = "end\nn_NN;\nR_cut0 = R_cut;\n"
        fallback = """end
if ~exist('R_cut','var')
    half_n_R = floor(n_R/2);
    [~, peak_idx] = max(R_anal(1:half_n_R));
    search_start = min(peak_idx + 1, half_n_R);
    search_end = half_n_R;
    if search_start <= search_end
        [~, local_min_idx] = min(R_anal(search_start:search_end));
        cut_idx = search_start + local_min_idx - 1;
    else
        cut_idx = min(n_R, peak_idx + 1);
    end
    R_cut = R_x(cut_idx);
    n_NN = 0;
    for j = 1:cut_idx
        n_NN = n_NN + 4*pi*R_x(j)^2*dR * (n_atoms_total-1)/V * R_anal(j);
    end
end
n_NN;
R_cut0 = R_cut;
"""
        if "if ~exist('R_cut','var')" not in text and needle in text:
            text = text.replace(needle, fallback, 1)
            pdf.write_text(text, encoding="utf-8")
            patches.append("pdf_v6_r_cut_first_minimum_fallback")
    return patches


def _copy_sluschi_entropy_template(sluschi_src: Path, workdir: Path, label: str) -> list[str]:
    entropy_src = sluschi_src / "mds_src" / "entropy"
    if not entropy_src.is_dir():
        raise FileNotFoundError(f"SLUSCHI entropy template directory not found: {entropy_src}")
    entropy_dir = workdir / "entropy"
    entropy_dir.mkdir(parents=True, exist_ok=True)
    for source in entropy_src.iterdir():
        if source.is_file():
            shutil.copy2(source, entropy_dir / source.name)
    for helper_name in ("onephase_v6.m", "pdf_v6.m"):
        helper = sluschi_src / "mds_src" / helper_name
        if helper.is_file() and not (entropy_dir / helper_name).is_file():
            shutil.copy2(helper, entropy_dir / helper_name)
    for name in ("pos", "param", "latt", "step"):
        shutil.copy2(workdir / name, entropy_dir / f"{name}_{label}")
    main_m = entropy_dir / "main.m"
    jobsub = entropy_dir / "jobsub_master"
    if main_m.is_file():
        text = main_m.read_text(encoding="utf-8", errors="replace")
        text = text.replace("replace_here", label).replace("replace_folder_here", str(sluschi_src / "mds_src"))
        main_m.write_text(text, encoding="utf-8")
    if jobsub.is_file():
        jobsub.write_text(jobsub.read_text(encoding="utf-8", errors="replace").replace("replace_here", label), encoding="utf-8")
    return _patch_sluschi_mds_entropy_template(entropy_dir)


def sluschi_onephase_svib_preflight(n_frames: int) -> dict[str, Any]:
    """Mirror the upstream onephase_v6 vibrational window sizing."""
    nsteps = math.floor(n_frames / 2) - 100
    tau = nsteps / 2 if nsteps > 0 else 0
    return {
        "n_frames": n_frames,
        "onephase_v6_nsteps": nsteps,
        "onephase_v6_tau": tau,
        "svib_window_valid": nsteps > 0,
        "minimum_frames_for_svib_window": 202,
        "recommended_minimum_frames": 320,
        "rule": "SLUSCHI onephase_v6 uses nsteps=floor(niter/2)-100; nsteps must be positive for a vibrational DOS window.",
    }


def _render_mds_run_script(args: argparse.Namespace, sluschi_src: Path, label: str) -> str:
    atomi_bin = args.atomi_bin or "$HOME/m_lammps_env/bin/atomi"
    matlab_command = args.matlab_command or "matlab -batch main"
    module_line = f"module load {args.matlab_module}\n" if args.matlab_module else ""
    env_exports = textwrap.dedent(
        f"""\
        export ATOMI_SLUSCHI_ROOT="{args.sluschi_root or '$HOME/SLUSCHI'}"
        export ATOMI_SLUSCHI_BIN="{args.sluschi_bin or '$HOME/SLUSCHI/src'}"
        export ATOMI_SLUSCHI_ENV="{args.atomi_env or '$HOME/m_lammps_env'}"
        export PATH="$ATOMI_SLUSCHI_ENV/bin:$ATOMI_SLUSCHI_BIN:$PATH"
        printf "sluschipath=%s\\n" "{sluschi_src}" > "$HOME/.sluschi.rc"
        """
    )
    summary_cmd = textwrap.dedent(
        f"""\
        "{atomi_bin}" sluschi-bridge entropy-summary \\
          --root "$WORKDIR" \\
          --collect "$WORKDIR/collect.stdout" \\
          --outdir "$WORKDIR/atomi_entropy_summary" \\
          --system {args.system!r} \\
          --formula {args.formula!r} \\
          --components {args.components!r} \\
          --phase {args.phase!r} \\
          --temperature-k {args.temperature_k} \\
          --quality {args.quality!r} \\
          --dump-stride-note {args.dump_stride_note!r}"""
    )
    if args.type_stoich:
        summary_cmd += f" \\\n  --type-stoich {args.type_stoich!r}"
    elif args.atoms_per_formula is not None:
        summary_cmd += f" \\\n  --atoms-per-formula {args.atoms_per_formula}"
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        WORKDIR="{args.workdir.resolve()}"
        {env_exports}
        cd "$WORKDIR/entropy"
        {module_line}{matlab_command} > entropy.out
        cd "$WORKDIR"
        cp pos param latt step entropy/
        cd entropy
        "{sluschi_src}/mds_src/summary.csh" > ../collect.stdout || true
        cd "$WORKDIR"
        {summary_cmd}
        echo "WORKDIR=$WORKDIR"
        """
    )


def _render_mds_sbatch(args: argparse.Namespace) -> str:
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={args.job_name}
        #SBATCH --output={args.job_name}_%j.out
        #SBATCH --error={args.job_name}_%j.err
        #SBATCH --partition={args.partition}
        #SBATCH --nodes={args.nodes}
        #SBATCH --ntasks={args.ntasks}
        #SBATCH --cpus-per-task={args.cpus_per_task}
        #SBATCH --time={args.walltime}
        #SBATCH --mem={args.mem}

        bash run_mds_entropy.sh
        """
    )


def mds_entropy_run_main(args: argparse.Namespace) -> dict[str, Any]:
    prepared_root = args.prepared_root.resolve()
    workdir = args.workdir.resolve()
    sluschi_raw = args.sluschi_bin or args.sluschi_root or os.environ.get("ATOMI_SLUSCHI_BIN", "") or "$HOME/SLUSCHI/src"
    sluschi_src = Path(os.path.expandvars(sluschi_raw)).expanduser()
    if sluschi_src.name != "src" and (sluschi_src / "src" / "mds_src").is_dir():
        sluschi_src = sluschi_src / "src"
    label = args.label or f"s_{int(round(args.temperature_k))}"
    layout = _write_legacy_mds_latt_step(
        prepared_root=prepared_root,
        workdir=workdir,
        block_size=args.legacy_mds_block_size,
        prepared_step_unit=args.prepared_step_unit,
    )
    svib_preflight = sluschi_onephase_svib_preflight(int(layout["n_frames"]))
    if not svib_preflight["svib_window_valid"] and not args.allow_invalid_svib_window:
        raise ValueError(
            "SLUSCHI MDS Svib preflight failed: "
            f"n_frames={svib_preflight['n_frames']} gives "
            f"onephase_v6_nsteps={svib_preflight['onephase_v6_nsteps']}. "
            "onephase_v6 requires at least "
            f"{svib_preflight['minimum_frames_for_svib_window']} frames and "
            f"{svib_preflight['recommended_minimum_frames']}+ frames are recommended. "
            "Increase the prepared trajectory window, or pass "
            "--allow-invalid-svib-window only for explicit Sconf/descriptor-only debugging."
        )
    compatibility_patches = _copy_sluschi_entropy_template(sluschi_src, workdir, label)
    run_script = workdir / "run_mds_entropy.sh"
    sbatch_script = workdir / "submit_mds_entropy.sbatch"
    run_script.write_text(_render_mds_run_script(args, sluschi_src, label), encoding="utf-8")
    sbatch_script.write_text(_render_mds_sbatch(args), encoding="utf-8")
    try:
        run_script.chmod(0o755)
    except OSError:
        pass
    manifest = {
        "schema": SCHEMA_MDS_ENTROPY_RUN,
        "workflow_lane": "entropy_prior",
        "prepared_root": str(prepared_root),
        "workdir": str(workdir),
        "label": label,
        "sluschi_src": str(sluschi_src),
        "environment_rule": {
            "primary_atomi_env": args.atomi_env or "$HOME/m_lammps_env",
            "mliap_lammps_env": "$HOME/m_lammps_gk_v2 for MLIAP/LAMMPS-specific runs only",
        },
        "layout": layout,
        "svib_preflight": svib_preflight,
        "outputs": {
            "run_script": str(run_script),
            "sbatch_script": str(sbatch_script),
            "collect_stdout": str(workdir / "collect.stdout"),
            "entropy_summary_dir": str(workdir / "atomi_entropy_summary"),
        },
        "sluschi_template_compatibility_patches": compatibility_patches,
        "notes": [
            "This command prepares a legacy SLUSCHI MDS entropy run from Atomi pos/param/latt/step files.",
            "Legacy MDS expects position frames per MD frame, but latt/step one record per 80-frame block by default.",
            "Atomi VASP/CP2K prep writes native step/param timestep in ps; legacy SLUSCHI MDS MATLAB expects fs, so mds-entropy-run converts the prepared units before writing the workdir.",
            "Run through sbatch for MATLAB/SLUSCHI workloads that may take more than a quick interactive parse.",
            "After completion, use the constrained/use-this-value Svib line and phase-health/phase-window checks before accepting entropy.",
            "Svib requires a positive onephase_v6 vibrational window; use at least 202 frames and preferably 320+ frames for screening.",
            "The copied legacy MATLAB template is locally patched when needed for undefined MDS correction variables and missing RDF cutoff fallbacks; the upstream SLUSCHI installation is not modified.",
        ],
    }
    write_json(workdir / "sluschi_mds_entropy_run_manifest.json", manifest)
    if args.submit:
        completed = subprocess.run(["sbatch", str(sbatch_script)], cwd=workdir, text=True, capture_output=True, check=True)
        manifest["submission_stdout"] = completed.stdout.strip()
        write_json(workdir / "sluschi_mds_entropy_run_manifest.json", manifest)
        print(completed.stdout.strip())
    else:
        print(f"Wrote SLUSCHI MDS entropy run: {workdir}")
        print(f"Submit with: cd {workdir} && sbatch {sbatch_script.name}")
    return manifest


def _pair_counts_from_summary(summary: dict[str, Any]) -> tuple[int, int, int]:
    n_pairs = int(summary.get("n_pair_recommendations") or 0)
    n_liquid = int(summary.get("n_liquid_like_pairs") or 0)
    n_solid = int(summary.get("n_solid_like_pairs") or 0)
    if not n_pairs:
        n_pairs = n_liquid + n_solid
    return n_pairs, n_liquid, n_solid


def assess_phase_health(
    *,
    expected_phase: str,
    n_pair_recommendations: int,
    n_liquid_like_pairs: int,
    n_solid_like_pairs: int,
    max_mixed_fraction: float = 0.25,
    min_classified_fraction: float = 0.75,
) -> dict[str, Any]:
    """Classify SLUSCHI phase health from solid/liquid pair recommendations."""
    expected = expected_phase.strip().lower()
    classified = n_liquid_like_pairs + n_solid_like_pairs
    denominator = n_pair_recommendations or classified
    liquid_fraction = n_liquid_like_pairs / classified if classified else None
    solid_fraction = n_solid_like_pairs / classified if classified else None
    classified_fraction = classified / denominator if denominator else None
    warnings: list[str] = []
    if not classified:
        label = "unknown"
        warnings.append("No solid/liquid pair classifications were found.")
    elif classified_fraction is not None and classified_fraction < min_classified_fraction:
        label = "under-classified"
        warnings.append("Too few pair recommendations were classified as solid-like or liquid-like.")
    elif expected in {"solid", "crystal", "crystalline"}:
        mixed_fraction = liquid_fraction or 0.0
        label = "solid-like" if mixed_fraction <= max_mixed_fraction else "mixed"
        if label == "mixed":
            warnings.append("Solid-labeled trajectory has too many liquid-like pair recommendations.")
    elif expected in {"liquid", "melt", "molten"}:
        mixed_fraction = solid_fraction or 0.0
        label = "liquid-like" if mixed_fraction <= max_mixed_fraction else "mixed"
        if label == "mixed":
            warnings.append("Liquid-labeled trajectory has too many solid-like pair recommendations.")
    elif expected in {"solid-liquid", "solid-liquid-coexistence", "coexist", "coexistence"}:
        has_both = n_liquid_like_pairs > 0 and n_solid_like_pairs > 0
        label = "coexistence-like" if has_both else "single-phase-like"
        if not has_both:
            warnings.append("Coexistence-labeled trajectory did not show both solid-like and liquid-like pairs.")
    else:
        if liquid_fraction is not None and liquid_fraction > 1.0 - max_mixed_fraction:
            label = "liquid-like"
        elif solid_fraction is not None and solid_fraction > 1.0 - max_mixed_fraction:
            label = "solid-like"
        else:
            label = "mixed"
    accepted = label in {"solid-like", "liquid-like", "coexistence-like"} and not warnings
    if label == "mixed":
        warnings.append("Treat this row as screening-prior until RDF/MSD/order checks validate the phase state.")
    return {
        "schema": SCHEMA_PHASE_HEALTH,
        "expected_phase": expected_phase,
        "phase_health_label": label,
        "accepted_for_phase_label": accepted,
        "n_pair_recommendations": n_pair_recommendations,
        "n_liquid_like_pairs": n_liquid_like_pairs,
        "n_solid_like_pairs": n_solid_like_pairs,
        "liquid_like_fraction": liquid_fraction,
        "solid_like_fraction": solid_fraction,
        "classified_fraction": classified_fraction,
        "max_mixed_fraction": max_mixed_fraction,
        "min_classified_fraction": min_classified_fraction,
        "warnings": warnings,
        "recommended_use": "production" if accepted else "screening-prior",
    }


def phase_health_from_payload(
    payload: dict[str, Any],
    expected_phase: str | None = None,
    *,
    max_mixed_fraction: float = 0.25,
    min_classified_fraction: float = 0.75,
) -> dict[str, Any]:
    sconfig_summary = payload.get("sconfig_summary") if isinstance(payload.get("sconfig_summary"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    source = sconfig_summary or summary or payload
    n_pairs, n_liquid, n_solid = _pair_counts_from_summary(source)
    phase = expected_phase or str(source.get("phase") or summary.get("phase") or "")
    health = assess_phase_health(
        expected_phase=phase,
        n_pair_recommendations=n_pairs,
        n_liquid_like_pairs=n_liquid,
        n_solid_like_pairs=n_solid,
        max_mixed_fraction=max_mixed_fraction,
        min_classified_fraction=min_classified_fraction,
    )
    health.update(
        {
            "system": source.get("system") or summary.get("system") or "",
            "formula": source.get("formula") or summary.get("formula") or "",
            "temperature_K": source.get("temperature_K") or summary.get("temperature_K"),
            "composition": source.get("composition") or summary.get("composition") or "",
            "source_summary_schema": payload.get("schema", ""),
        }
    )
    return health


def phase_health_main(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any]
    if args.summary_json:
        payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
        health = phase_health_from_payload(
            payload,
            args.expected_phase or None,
            max_mixed_fraction=args.max_mixed_fraction,
            min_classified_fraction=args.min_classified_fraction,
        )
    else:
        root = args.root.resolve()
        pairs = parse_sconfig_pairs(root)
        n_liquid = sum(1 for pair in pairs if "liquid" in pair.state)
        n_solid = sum(1 for pair in pairs if "solid" in pair.state)
        health = assess_phase_health(
            expected_phase=args.expected_phase,
            n_pair_recommendations=len(pairs),
            n_liquid_like_pairs=n_liquid,
            n_solid_like_pairs=n_solid,
            max_mixed_fraction=args.max_mixed_fraction,
            min_classified_fraction=args.min_classified_fraction,
        )
        health.update(
            {
                "system": args.system,
                "formula": args.formula,
                "temperature_K": args.temperature_k,
                "composition": args.composition,
                "source": str(root),
                "source_summary_schema": "",
            }
        )
    if args.summary_json:
        health["source"] = str(args.summary_json.resolve())
    health["max_mixed_fraction"] = args.max_mixed_fraction
    health["min_classified_fraction"] = args.min_classified_fraction
    fields = [
        "system",
        "formula",
        "temperature_K",
        "composition",
        "expected_phase",
        "phase_health_label",
        "accepted_for_phase_label",
        "recommended_use",
        "n_pair_recommendations",
        "n_liquid_like_pairs",
        "n_solid_like_pairs",
        "liquid_like_fraction",
        "solid_like_fraction",
        "classified_fraction",
        "warnings",
        "source",
    ]
    csv_row = {**health, "warnings": "; ".join(health.get("warnings", []))}
    write_csv(outdir / "sluschi_phase_health.csv", [csv_row], fields)
    write_json(outdir / "sluschi_phase_health.json", health)
    print(f"Phase health: {health['phase_health_label']} ({health['recommended_use']})")
    print(f"Wrote phase-health JSON: {outdir / 'sluschi_phase_health.json'}")
    return health


def workflow_guide_main(args: argparse.Namespace) -> dict[str, Any]:
    guide = {
        "schema": SCHEMA_WORKFLOW_GUIDE,
        "system": args.system,
        "lanes": {
            "coexistence": {
                "purpose": "melting point / solid-liquid phase boundary",
                "method": "small-cell solid-liquid coexistence with hovering interfaces",
                "steps": [
                    "optimize and validate the solid unit cell",
                    "estimate target-temperature volume, commonly with NPT or thermal expansion",
                    "construct a half-solid/half-liquid coexistence cell",
                    "run coexistence trajectories near candidate melting temperatures",
                    "classify whether the small cell remains coexistence-like, fully melts, or fully solidifies",
                    "use coexistence statistics to bracket or estimate the melting temperature",
                ],
                "required_outputs": ["coexistence trajectory", "phase-label history", "melting-temperature estimate"],
            },
            "entropy_prior": {
                "purpose": "Svib/Sconf/Stotal rows for zentropy, CALPHAD, QHA/MD overlay, or screening",
                "method": "parse existing phase-specific MD postprocess output; not a melting calculation by itself",
                "steps": [
                    "stage type-safe LAMMPS dump conversion with explicit type-elements",
                    "run SLUSCHI postprocess on dense NVT frames",
                    "parse Svib and Sconf with entropy-summary",
                    "run phase-health before using the row as production thermodynamics",
                ],
                "required_outputs": ["entropy summary CSV/JSON", "phase-health CSV/JSON", "type-basis manifest"],
            },
        },
        "decision_rules": [
            "Use coexistence lane for melting point or solid-liquid boundary claims.",
            "Use entropy-prior lane for independent MD entropy data only after phase-health passes.",
            "A mixed solid/liquid pair classification in a solid or liquid row is a warning, not automatic proof of failure.",
            "For production, complement pair classification with RDF, MSD/diffusion, and order-parameter checks.",
        ],
    }
    if args.outdir:
        outdir = args.outdir.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        write_json(outdir / "sluschi_workflow_guide.json", guide)
        (outdir / "SLUSCHI_WORKFLOW_GUIDE.md").write_text(
            textwrap.dedent(
                f"""\
                # SLUSCHI Workflow Guide

                System: {args.system}

                SLUSCHI should be interpreted as a small-cell solid-liquid coexistence
                method first. Atomi separates this from the entropy-prior workflow that
                parses existing MD postprocess output.

                ## Coexistence Lane

                Purpose: melting point and solid-liquid phase boundaries.

                Steps:
                - optimize and validate the solid unit cell
                - estimate target-temperature volume
                - construct a half-solid/half-liquid coexistence cell
                - run coexistence trajectories near candidate melting temperatures
                - classify whether trajectories remain coexistence-like, melt, or solidify
                - estimate or bracket the melting point

                ## Entropy-Prior Lane

                Purpose: Svib/Sconf/Stotal rows for zentropy, CALPHAD, or QHA/MD overlays.
                This is not a melting calculation by itself.

                Required guard: run `sluschi-bridge phase-health` on every parsed entropy row.
                """
            ),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(guide, indent=2, sort_keys=True))
    else:
        print("SLUSCHI has two Atomi lanes: coexistence for melting, entropy-prior for parsed MD entropy rows.")
        print("Run phase-health before accepting entropy-prior rows as production thermodynamics.")
    return guide


def parse_melting_anchors_from_text(path: Path, text: str, quality: str) -> list[MeltingAnchor]:
    anchors: list[MeltingAnchor] = []
    line_pattern = re.compile(
        r"Melting\s+temperature\s+and\s+std\s+error:\s*"
        r"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for match in line_pattern.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end < 0:
            line_end = len(text)
        anchors.append(
            MeltingAnchor(
                source_file=str(path),
                melting_temperature_K=float(match.group(1)),
                temperature_std_error_K=float(match.group(2)),
                method="sluschi_mpfit",
                quality=quality,
                line=text[line_start:line_end].strip(),
            )
        )
    if path.name == "MPFit.out":
        values: list[float] = []
        for token in re.split(r"[\s,]+", text.strip()):
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                values = []
                break
        if len(values) >= 2:
            anchors.append(
                MeltingAnchor(
                    source_file=str(path),
                    melting_temperature_K=values[0],
                    temperature_std_error_K=values[1],
                    method="sluschi_mpfit",
                    quality=quality,
                    line=text.strip().splitlines()[0] if text.strip() else "",
                )
            )
    return anchors


def find_melting_anchors(root: Path, quality: str) -> list[MeltingAnchor]:
    anchors: list[MeltingAnchor] = []
    target_names = {"SLUSCHI.out", "MPFit.out"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name not in target_names:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        anchors.extend(parse_melting_anchors_from_text(path, text, quality))
    return anchors


def melting_anchor_from_phase_health(paths: list[Path], quality: str) -> MeltingAnchor | None:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            rows.append(data)
    coexistence = [
        row
        for row in rows
        if str(row.get("phase_health_label", "")).lower() == "coexistence-like"
        and row.get("temperature_K") is not None
    ]
    if not coexistence:
        return None
    temperatures = [float(row["temperature_K"]) for row in coexistence]
    tm = sum(temperatures) / len(temperatures)
    all_temperatures = sorted(float(row["temperature_K"]) for row in rows if row.get("temperature_K") is not None)
    lower = max((temp for temp in all_temperatures if temp < tm), default=None)
    upper = min((temp for temp in all_temperatures if temp > tm), default=None)
    if lower is not None and upper is not None:
        stderr = (upper - lower) / 2.0
        bracket = f"bracket=[{lower:g},{upper:g}] K"
    elif len(temperatures) > 1:
        mean = tm
        variance = sum((temp - mean) ** 2 for temp in temperatures) / (len(temperatures) - 1)
        stderr = variance**0.5
        bracket = "coexistence_temperature_std"
    else:
        stderr = None
        bracket = "coexistence_temperature_only"
    return MeltingAnchor(
        source_file=",".join(str(path) for path in paths),
        melting_temperature_K=tm,
        temperature_std_error_K=stderr,
        method="phase_health_bracket",
        quality=quality,
        line=f"coexistence-like phase-health at {','.join(str(t) for t in temperatures)} K; {bracket}",
    )


def build_melting_prior_payload(args: argparse.Namespace, anchors: list[MeltingAnchor]) -> dict[str, Any]:
    observables = []
    for anchor in anchors:
        observables.append(
            {
                "observable": "melting_temperature_K",
                "value": anchor.melting_temperature_K,
                "unit": "K",
                "phase": "solid-liquid",
                "temperature_std_error_K": anchor.temperature_std_error_K,
                "composition": args.composition,
                "quality": anchor.quality,
                "source_engine": "sluschi",
                "method": anchor.method,
                "file": str(Path(anchor.source_file).resolve()) if "," not in anchor.source_file else anchor.source_file,
                "line": anchor.line,
            }
        )
    return {
        "schema": PRIOR_SCHEMA,
        "kind": "sluschi_melting_anchor",
        "system": args.system or args.root.resolve().name,
        "formula": args.formula or "",
        "components": parse_csv_list(args.components),
        "thermo": {"observables": observables},
        "source": {
            "method": "sluschi_melting_anchor",
            "root": str(args.root.resolve()),
            "sluschi_repo": DEFAULT_REPO,
            "bridge_schema": SCHEMA_MELTING_ANCHOR,
        },
        "notes": [
            "SLUSCHI determines melting temperature from small-cell solid-liquid coexistence outcomes.",
            "Preferred anchors come from SLUSCHI MPFit output: 'Melting temperature and std error: Tm sigma'.",
            "Phase-health bracket anchors are useful screening constraints but should be lower weight than MPFit anchors.",
        ],
    }


def melting_anchor_main(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    anchors = [] if args.skip_root_scan else find_melting_anchors(root, args.quality)
    if args.phase_health_json:
        bracket_anchor = melting_anchor_from_phase_health(args.phase_health_json, args.quality)
        if bracket_anchor is not None:
            anchors.append(bracket_anchor)
    if not anchors and not args.allow_empty:
        raise FileNotFoundError(
            f"No SLUSCHI melting anchors found under {root}; expected SLUSCHI.out/MPFit.out or --phase-health-json."
        )
    outdir = args.outdir.resolve()
    rows = [
        {
            "system": args.system,
            "formula": args.formula,
            "components": args.components,
            "composition": args.composition,
            "melting_temperature_K": anchor.melting_temperature_K,
            "temperature_std_error_K": anchor.temperature_std_error_K,
            "temperature_low_K": (
                anchor.melting_temperature_K - anchor.temperature_std_error_K
                if anchor.temperature_std_error_K is not None
                else ""
            ),
            "temperature_high_K": (
                anchor.melting_temperature_K + anchor.temperature_std_error_K
                if anchor.temperature_std_error_K is not None
                else ""
            ),
            "method": anchor.method,
            "quality": anchor.quality,
            "source_file": anchor.source_file,
            "line": anchor.line,
        }
        for anchor in anchors
    ]
    fields = [
        "system",
        "formula",
        "components",
        "composition",
        "melting_temperature_K",
        "temperature_std_error_K",
        "temperature_low_K",
        "temperature_high_K",
        "method",
        "quality",
        "source_file",
        "line",
    ]
    anchor_csv = outdir / "sluschi_melting_anchor.csv"
    anchor_json = outdir / "sluschi_melting_anchor.json"
    prior_json = args.prior_out or outdir / "sluschi_melting_anchor_thermo_prior.json"
    write_csv(anchor_csv, rows, fields)
    payload = {
        "schema": SCHEMA_MELTING_ANCHOR,
        "root": str(root),
        "n_anchors": len(anchors),
        "outputs": {"anchor_csv": str(anchor_csv), "thermo_prior_json": str(prior_json)},
        "anchors": rows,
    }
    write_json(anchor_json, payload)
    write_json(prior_json, build_melting_prior_payload(args, anchors))
    print(f"Wrote SLUSCHI melting anchor CSV: {anchor_csv}")
    return payload


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

    guide = sub.add_parser(
        "workflow-guide",
        help="Write or print the Atomi interpretation of SLUSCHI coexistence vs entropy-prior workflows.",
    )
    guide.add_argument("--system", default="", help="Optional system label for the guide.")
    guide.add_argument("--outdir", type=Path, help="Optional directory for JSON/Markdown guide files.")
    guide.add_argument("--json", action="store_true", help="Print the guide JSON to stdout.")

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

    prep = sub.add_parser(
        "lammps-prep-scripts",
        help="Write type-safe LAMMPS dump to SLUSCHI prep scripts for one element basis.",
    )
    prep.add_argument("--outdir", type=Path, default=Path("sluschi_lammps_prep"))
    prep.add_argument(
        "--type-elements",
        required=True,
        help="LAMMPS type map, e.g. '1=K,2=Cl' for pure KCl or '1=Li,2=K,3=Cl' for LiCl-KCl.",
    )
    prep.add_argument(
        "--element-masses",
        default="",
        help="Optional Element=mass overrides, e.g. 'K=39.0983,Cl=35.453'.",
    )
    prep.add_argument("--pos-script", default="lmp_pos.py")
    prep.add_argument("--prep-script", default="lmp_prep.csh")

    lammps_prep = sub.add_parser(
        "lammps-prep",
        help="Prepare LAMMPS dump frames as SLUSCHI pos/latt/step/param files.",
    )
    lammps_prep.add_argument("--trajectory", type=Path, required=True, help="LAMMPS dump trajectory.")
    lammps_prep.add_argument("--outdir", type=Path, default=Path("sluschi_lammps_prep"))
    lammps_prep.add_argument(
        "--type-elements",
        required=True,
        help="LAMMPS type map, e.g. '1=K,2=Cl' for pure KCl or '1=Li,2=K,3=Cl' for LiCl-KCl.",
    )
    lammps_prep.add_argument("--elements", default="", help="Optional element order override. Default follows type-elements.")
    lammps_prep.add_argument(
        "--element-masses",
        default="",
        help="Optional Element=mass overrides, e.g. K=39.0983,Cl=35.453.",
    )
    lammps_prep.add_argument("--temperature-k", type=float, help="Trajectory temperature in K, used for manifest and phase_temp label.")
    lammps_prep.add_argument("--phase", default="", help="Optional phase label, e.g. solid or liquid.")
    lammps_prep.add_argument(
        "--phase-temp-label",
        default="",
        help="Exact phase_temp content. Default is '<phase>_<rounded temperature>' when both are provided, else phase.",
    )
    lammps_prep.add_argument(
        "--timestep-ps",
        type=float,
        help="LAMMPS timestep in ps for units metal runs, e.g. 0.00025. Mutually exclusive with --timestep-fs.",
    )
    lammps_prep.add_argument("--timestep-fs", type=float, help="LAMMPS timestep in fs. Mutually exclusive with --timestep-ps.")
    lammps_prep.add_argument(
        "--frame-stride-md-steps",
        type=float,
        default=None,
        help="MD steps between consecutive selected dump frames. Default infers from dump timestep differences.",
    )
    lammps_prep.add_argument(
        "--coordinate-preference",
        choices=("unwrapped", "wrapped"),
        default="unwrapped",
        help="Prefer unwrapped or wrapped dump columns when both are present.",
    )
    lammps_prep.add_argument("--start-frame", type=int, default=0)
    lammps_prep.add_argument("--stop-frame", type=int)
    lammps_prep.add_argument("--stride", type=int, default=1)

    cp2k_prep = sub.add_parser(
        "cp2k-prep",
        help="Prepare CP2K AIMD XYZ frames as SLUSCHI pos/latt/step/param files.",
    )
    cp2k_prep.add_argument("--xyz", type=Path, required=True, help="CP2K multi-frame *-pos.xyz trajectory.")
    cp2k_prep.add_argument("--inp", type=Path, help="Optional CP2K input used to read &CELL ABC/A/B/C.")
    cp2k_prep.add_argument("--outdir", type=Path, default=Path("sluschi_cp2k_prep"))
    cp2k_prep.add_argument("--elements", default="", help="Element order for SLUSCHI param/pos grouping, e.g. K,Cl.")
    cp2k_prep.add_argument(
        "--element-masses",
        default="",
        help="Optional Element=mass overrides, e.g. K=39.0983,Cl=35.453.",
    )
    cp2k_prep.add_argument("--cell", default="", help="Orthorhombic cell lengths in A: a,b,c.")
    cp2k_prep.add_argument(
        "--cell-vector",
        action="append",
        default=[],
        help="Cell vector in A; repeat exactly three times for A, B, C.",
    )
    cp2k_prep.add_argument("--timestep-fs", type=float, help="CP2K MD timestep in fs.")
    cp2k_prep.add_argument(
        "--frame-stride-md-steps",
        type=float,
        default=1.0,
        help="MD steps between consecutive written XYZ frames; CP2K &TRAJECTORY &EACH MD N usually sets this.",
    )
    cp2k_prep.add_argument("--start-frame", type=int, default=0)
    cp2k_prep.add_argument("--stop-frame", type=int)
    cp2k_prep.add_argument("--stride", type=int, default=1)
    cp2k_prep.add_argument("--phase", default="", help="Optional SLUSCHI phase label written to phase_temp, e.g. solid or liquid.")

    vasp_prep = sub.add_parser(
        "vasp-prep",
        help="Prepare VASP XDATCAR frames as SLUSCHI pos/latt/step/param files.",
    )
    vasp_prep.add_argument("--poscar", type=Path, default=Path("POSCAR"), help="VASP POSCAR/CONTCAR with the XDATCAR atom basis.")
    vasp_prep.add_argument("--xdatcar", type=Path, default=Path("XDATCAR"), help="VASP XDATCAR trajectory.")
    vasp_prep.add_argument("--outdir", type=Path, default=Path("sluschi_vasp_prep"))
    vasp_prep.add_argument("--elements", default="", help="Optional element order override, e.g. Na,U,Cl. Default: POSCAR order.")
    vasp_prep.add_argument(
        "--element-masses",
        default="",
        help="Optional Element=mass overrides, e.g. U=238.02891,Cl=35.453.",
    )
    vasp_prep.add_argument("--temperature-k", type=float, help="Trajectory temperature in K, used for manifest and default phase_temp label.")
    vasp_prep.add_argument("--phase", default="", help="Optional phase label, e.g. solid or liquid.")
    vasp_prep.add_argument(
        "--phase-temp-label",
        default="",
        help="Exact phase_temp content. Default is '<phase>_<rounded temperature>' when both are provided, else phase.",
    )
    vasp_prep.add_argument("--timestep-fs", type=float, default=1.0, help="VASP POTIM timestep in fs.")
    vasp_prep.add_argument(
        "--frame-stride-md-steps",
        type=float,
        default=1.0,
        help="MD steps between consecutive XDATCAR frames; usually NBLOCK or output stride.",
    )
    vasp_prep.add_argument("--start-frame", type=int, default=0)
    vasp_prep.add_argument("--stop-frame", type=int)
    vasp_prep.add_argument("--stride", type=int, default=1)

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

    entropy = sub.add_parser(
        "entropy-summary",
        help="Parse constrained SLUSCHI Svib plus Sconf from one prepared LAMMPS NVT postprocess folder.",
    )
    entropy.add_argument("--root", type=Path, default=Path("."))
    entropy.add_argument("--collect", type=Path, default=None, help="Optional collect.stdout/entropy.out path.")
    entropy.add_argument("--outdir", type=Path, default=Path("sluschi_entropy_results"))
    entropy.add_argument("--system", default="", help="System label, e.g. UO2.")
    entropy.add_argument("--formula", default="", help="Formula label, e.g. UO2.")
    entropy.add_argument("--components", default="", help="Comma-separated component labels for mixture priors.")
    entropy.add_argument("--phase", default="", help="Phase label, e.g. fluorite, liquid.")
    entropy.add_argument("--temperature-k", type=float, required=True, help="NVT trajectory temperature in K.")
    entropy.add_argument("--composition", default="", help="Composition label for mixture or defect priors.")
    entropy.add_argument(
        "--type-stoich",
        default="",
        help="LAMMPS/SLUSCHI type stoichiometry per formula, e.g. UO2 with type1=O,type2=U uses '1=2,2=1'.",
    )
    entropy.add_argument(
        "--atoms-per-formula",
        type=float,
        default=None,
        help="Fallback atoms per formula if --type-stoich is omitted. For multiple types this evenly weights types.",
    )
    entropy.add_argument(
        "--sconf-reduction",
        choices=("mean-all-pairs", "same-species-liquid"),
        default="mean-all-pairs",
        help=(
            "How to reduce SLUSCHI pair-channel Sconf recommendations. "
            "mean-all-pairs is a conservative diagnostic default; same-species-liquid matches the "
            "Hong/Shang liquid KCl guideline by using liquid same-species pair channels."
        ),
    )
    entropy.add_argument(
        "--quality",
        choices=("descriptor", "screening-prior", "production"),
        default="screening-prior",
        help="Confidence tier for downstream thermo-prior use.",
    )
    entropy.add_argument(
        "--dump-stride-note",
        default="Svib requires dense uniformly spaced NVT frames; this command parses existing SLUSCHI postprocess output only.",
        help="Trajectory/dump-stride note stored in the summary JSON.",
    )

    mds = sub.add_parser(
        "mds-entropy-run",
        help="Prepare and optionally submit a legacy SLUSCHI MDS Svib/Sconf entropy run from Atomi-prepared pos/latt/step files.",
    )
    mds.add_argument("--prepared-root", type=Path, required=True, help="Folder containing Atomi-prepared pos/param/latt/step files.")
    mds.add_argument("--workdir", type=Path, required=True, help="Output run folder for legacy SLUSCHI MDS entropy calculation.")
    mds.add_argument("--temperature-k", type=float, required=True, help="NVT trajectory temperature in K.")
    mds.add_argument("--system", default="", help="System label, e.g. UC2 or KCl.")
    mds.add_argument("--formula", default="", help="Formula label, e.g. UC2.")
    mds.add_argument("--components", default="", help="Comma-separated components/species labels.")
    mds.add_argument("--phase", default="", help="Phase label, e.g. solid or liquid.")
    mds.add_argument("--label", default="", help="SLUSCHI filename label. Default: s_<rounded temperature>.")
    mds.add_argument("--sluschi-root", default="", help="SLUSCHI root, e.g. $HOME/SLUSCHI.")
    mds.add_argument("--sluschi-bin", default="", help="SLUSCHI src/bin folder containing mds_src, e.g. $HOME/SLUSCHI/src.")
    mds.add_argument("--atomi-env", default="$HOME/m_lammps_env", help="Primary Atomi environment path.")
    mds.add_argument("--atomi-bin", default="$HOME/m_lammps_env/bin/atomi", help="Atomi executable used by the generated run script.")
    mds.add_argument("--matlab-module", default="math/matlab/R2022a", help="Module loaded before MATLAB. Empty string disables module load.")
    mds.add_argument("--matlab-command", default="matlab -batch main", help="MATLAB command run inside the entropy folder.")
    mds.add_argument("--legacy-mds-block-size", type=int, default=80, help="SLUSCHI MDS block size for latt/step records.")
    mds.add_argument(
        "--prepared-step-unit",
        choices=["ps", "fs"],
        default="ps",
        help=(
            "Unit used by the prepared-root param/step files. Atomi vasp-prep/cp2k-prep write ps; "
            "legacy SLUSCHI MDS MATLAB expects fs in the generated work folder."
        ),
    )
    mds.add_argument("--type-stoich", default="", help="Type stoichiometry per formula for entropy-summary, e.g. '1=1,2=2'.")
    mds.add_argument("--atoms-per-formula", type=float, default=None, help="Fallback atoms per formula for entropy-summary.")
    mds.add_argument(
        "--quality",
        choices=("descriptor", "screening-prior", "production"),
        default="screening-prior",
        help="Confidence tier for parsed entropy rows.",
    )
    mds.add_argument(
        "--dump-stride-note",
        default="Dense uniformly spaced NVT frames; legacy MDS latt/step one record per block.",
        help="Trajectory/dump-stride note stored in parsed entropy summary.",
    )
    mds.add_argument(
        "--allow-invalid-svib-window",
        action="store_true",
        help=(
            "Allow preparing/submitting a run whose frame count cannot produce a positive SLUSCHI onephase_v6 "
            "Svib window. Use only for Sconf/descriptor debugging; production entropy should not set this."
        ),
    )
    mds.add_argument("--submit", action="store_true", help="Submit the generated sbatch immediately.")
    mds.add_argument("--job-name", default="sluschi_mds_entropy")
    mds.add_argument("--partition", default="single")
    mds.add_argument("--nodes", type=int, default=1)
    mds.add_argument("--ntasks", type=int, default=1)
    mds.add_argument("--cpus-per-task", type=int, default=4)
    mds.add_argument("--walltime", default="06:00:00")
    mds.add_argument("--mem", default="16G")

    health = sub.add_parser(
        "phase-health",
        help="Classify whether a SLUSCHI entropy/Sconfig row is solid-like, liquid-like, coexistence-like, or mixed.",
    )
    health.add_argument("--summary-json", type=Path, help="Existing sluschi_entropy_summary.json or Sconfig summary JSON.")
    health.add_argument("--root", type=Path, default=Path("."), help="Fallback run folder to parse collect.stdout files.")
    health.add_argument("--outdir", type=Path, default=Path("sluschi_phase_health"))
    health.add_argument("--expected-phase", default="", help="Expected phase label: solid, liquid, or solid-liquid-coexistence.")
    health.add_argument("--system", default="")
    health.add_argument("--formula", default="")
    health.add_argument("--temperature-k", type=float)
    health.add_argument("--composition", default="")
    health.add_argument("--max-mixed-fraction", type=float, default=0.25)
    health.add_argument("--min-classified-fraction", type=float, default=0.75)

    windows = sub.add_parser(
        "phase-window-sample",
        help="Sample trajectory windows and conservatively label solid-like, liquid-like, or mixed regions.",
    )
    windows.add_argument(
        "--engine",
        choices=("lammps-dump", "cp2k-xyz", "vasp-xdatcar"),
        required=True,
        help="Trajectory format to read.",
    )
    windows.add_argument("--trajectory", type=Path, required=True, help="LAMMPS dump, CP2K XYZ, or VASP XDATCAR path.")
    windows.add_argument("--outdir", type=Path, default=Path("sluschi_phase_windows"))
    windows.add_argument("--species-a", required=True, help="First species used for mobility/order screening, e.g. K or U.")
    windows.add_argument("--species-b", required=True, help="Neighbor species used for local-order screening, e.g. Cl or O.")
    windows.add_argument(
        "--type-elements",
        default="",
        help="LAMMPS type map for --engine lammps-dump, e.g. '1=K,2=Cl'.",
    )
    windows.add_argument("--poscar", type=Path, help="VASP POSCAR/CONTCAR for --engine vasp-xdatcar.")
    windows.add_argument("--inp", type=Path, help="CP2K input used to read &CELL for --engine cp2k-xyz.")
    windows.add_argument("--cell", default="", help="Orthorhombic cell lengths in A: a,b,c for CP2K XYZ.")
    windows.add_argument(
        "--cell-vector",
        action="append",
        default=[],
        help="Cell vector in A for CP2K XYZ; repeat exactly three times for A, B, C.",
    )
    windows.add_argument("--start-frame", type=int, default=0)
    windows.add_argument("--stop-frame", type=int)
    windows.add_argument("--stride", type=int, default=1)
    windows.add_argument(
        "--frame-step-ps",
        type=float,
        required=True,
        help="Time in ps between selected neighboring frames after any trajectory output stride.",
    )
    windows.add_argument("--window-ps", type=float, default=0.5, help="Window duration in ps for each phase sample.")
    windows.add_argument("--stride-ps", type=float, default=0.25, help="Time stride in ps between adjacent windows.")
    windows.add_argument("--neighbor-cutoff-a", type=float, default=3.8, help="A-B coordination cutoff in Angstrom.")
    windows.add_argument(
        "--liquid-check-mode",
        choices=("generic", "network"),
        default="generic",
        help=(
            "Phase-window classifier. generic uses RMS, nearest-neighbor spread, and coordination votes. "
            "network uses mobility as the classifier and reports persistent A-B network coordination without "
            "treating it as a solidness veto, useful for UCl3/CeCl3-rich chloride liquids."
        ),
    )
    windows.add_argument("--solid-rms-max-a", type=float, default=0.8, help="RMS displacement upper bound for solid-like windows.")
    windows.add_argument("--liquid-rms-min-a", type=float, default=1.5, help="RMS displacement lower bound for liquid-like windows.")
    windows.add_argument(
        "--solid-nearest-sd-max-a",
        type=float,
        default=0.12,
        help="Nearest A-B distance standard-deviation upper bound for solid-like windows.",
    )
    windows.add_argument(
        "--liquid-nearest-sd-min-a",
        type=float,
        default=0.25,
        help="Nearest A-B distance standard-deviation lower bound for liquid-like windows.",
    )
    windows.add_argument("--solid-coord-min", type=float, default=5.5, help="Mean A-B coordination lower bound for solid-like windows.")
    windows.add_argument(
        "--liquid-coord-max",
        type=float,
        default=None,
        help="Optional mean A-B coordination upper bound contributing a liquid-like vote.",
    )

    melting = sub.add_parser(
        "melting-anchor",
        help="Parse SLUSCHI MPFit/coexistence outputs into a CALPHAD-ready melting-temperature anchor.",
    )
    melting.add_argument("--root", type=Path, default=Path("."))
    melting.add_argument("--outdir", type=Path, default=Path("sluschi_melting_anchor"))
    melting.add_argument("--prior-out", type=Path, help="Optional thermo-prior JSON output path.")
    melting.add_argument("--system", default="")
    melting.add_argument("--formula", default="")
    melting.add_argument("--components", default="")
    melting.add_argument("--composition", default="")
    melting.add_argument(
        "--phase-health-json",
        type=Path,
        action="append",
        default=[],
        help="Optional phase-health JSON files used to build a lower-confidence coexistence bracket anchor.",
    )
    melting.add_argument(
        "--quality",
        choices=("descriptor", "screening-prior", "production"),
        default="production",
        help="Confidence tier for downstream CALPHAD/thermo-prior use.",
    )
    melting.add_argument("--allow-empty", action="store_true", help="Write empty outputs instead of failing.")
    melting.add_argument(
        "--skip-root-scan",
        action="store_true",
        help="Do not walk --root for SLUSCHI.out/MPFit.out; useful when building an anchor only from phase-health JSONs.",
    )

    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        return status_main(args)
    if args.command == "workflow-guide":
        return workflow_guide_main(args)
    if args.command == "init":
        return init_workspace(args)
    if args.command == "supersalt-example":
        return supersalt_example_main(args)
    if args.command == "lammps-prep-scripts":
        return prep_scripts_main(args)
    if args.command == "lammps-prep":
        return lammps_prep_main(args)
    if args.command == "cp2k-prep":
        return cp2k_prep_main(args)
    if args.command == "vasp-prep":
        return vasp_prep_main(args)
    if args.command == "parse":
        return parse_main(args)
    if args.command == "sconfig":
        return sconfig_main(args)
    if args.command == "entropy-summary":
        return entropy_summary_main(args)
    if args.command == "mds-entropy-run":
        return mds_entropy_run_main(args)
    if args.command == "phase-health":
        return phase_health_main(args)
    if args.command == "phase-window-sample":
        return phase_window_sample_main(args)
    if args.command == "melting-anchor":
        return melting_anchor_main(args)
    build_parser().print_help()
    return None


def sconfig_cli_main(argv: list[str] | None = None) -> None:
    main(["sconfig", *(sys.argv[1:] if argv is None else argv)])


def entropy_summary_cli_main(argv: list[str] | None = None) -> None:
    main(["entropy-summary", *(sys.argv[1:] if argv is None else argv)])


def phase_health_cli_main(argv: list[str] | None = None) -> None:
    main(["phase-health", *(sys.argv[1:] if argv is None else argv)])


def workflow_guide_cli_main(argv: list[str] | None = None) -> None:
    main(["workflow-guide", *(sys.argv[1:] if argv is None else argv)])


def melting_anchor_cli_main(argv: list[str] | None = None) -> None:
    main(["melting-anchor", *(sys.argv[1:] if argv is None else argv)])


def cp2k_prep_cli_main(argv: list[str] | None = None) -> None:
    main(["cp2k-prep", *(sys.argv[1:] if argv is None else argv)])


def vasp_prep_cli_main(argv: list[str] | None = None) -> None:
    main(["vasp-prep", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
