"""Config-driven helpers for MOOSE application workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any

from atomi.core.doctor import load_hpc_config


DEFAULT_PROFILE = "moose_gpu_kokkos"
UO2_REQUIRED_MATERIAL_FIELDS = (
    "T_K",
    "k_W_mK",
    "Cp_J_kgK",
    "rho_kg_m3",
    "E_Pa",
    "nu",
)
DEFAULT_FUNCTIONS = {
    "k_W_mK": "uo2_k",
    "Cp_J_kgK": "uo2_Cp",
    "rho_kg_m3": "uo2_rho",
    "alpha_1_K": "uo2_alpha",
    "dilatation": "uo2_dilatation",
    "E_Pa": "uo2_E",
    "nu": "uo2_nu",
}


def load_moose_profile(
    hpc_config: Path | None = None,
    profile_name: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Load one MOOSE profile from the local atomi HPC config."""
    config = load_hpc_config(hpc_config)
    profile = config.get("profiles", {}).get(profile_name)
    if not isinstance(profile, dict):
        available = sorted(config.get("profiles", {}).keys())
        available_text = ", ".join(available) if available else "none"
        raise SystemExit(
            f"MOOSE profile {profile_name!r} was not found. Available profiles: {available_text}"
        )
    return profile


def _shell_join(command: list[str] | str) -> str:
    if isinstance(command, str):
        return command
    return " ".join(shlex.quote(part) for part in command)


def _profile_command(profile: dict[str, Any], app: str | None = None) -> str:
    if app:
        return shlex.quote(app) + " --help"
    executable = profile.get("test_executable") or profile.get("executable")
    if executable:
        return shlex.quote(str(executable)) + " --help"
    return "moose-opt --help"


def activation_lines(profile: dict[str, Any]) -> list[str]:
    """Return shell lines that activate the configured MOOSE runtime."""
    lines = ["set -euo pipefail"]
    activation = profile.get("activation_script")
    if activation:
        lines.append(f"source {shlex.quote(str(activation))}")
        return lines

    module_commands = profile.get("module_commands") or []
    if module_commands:
        lines.extend(str(command) for command in module_commands)
    elif profile.get("modules"):
        lines.append("module purge")
        lines.extend(f"module load {shlex.quote(str(module))}" for module in profile["modules"])

    python_env = profile.get("python_env") or profile.get("env_path")
    if python_env:
        lines.append(f"source {shlex.quote(str(python_env))}/bin/activate")

    exports = profile.get("build_environment_exports") or {}
    for key, value in sorted(exports.items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    if profile.get("mpi", {}).get("psm2_cuda"):
        lines.append(f"export PSM2_CUDA={shlex.quote(str(profile['mpi']['psm2_cuda']))}")
    return lines


def render_slurm_submit(
    profile: dict[str, Any],
    command: str | list[str] | None = None,
    *,
    job_name: str = "moose",
    output: str = "moose_%j.out",
    error: str = "moose_%j.err",
) -> str:
    """Render a portable Slurm submission script for one configured MOOSE profile."""
    run_command = _shell_join(command) if command else _profile_command(profile)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output}",
        f"#SBATCH --error={error}",
    ]
    if profile.get("partition"):
        lines.append(f"#SBATCH --partition={profile['partition']}")
    if profile.get("gres"):
        lines.append(f"#SBATCH --gres={profile['gres']}")
    if profile.get("nodes"):
        lines.append(f"#SBATCH --nodes={profile['nodes']}")
    if profile.get("ntasks"):
        lines.append(f"#SBATCH --ntasks={profile['ntasks']}")
    if profile.get("cpus_per_task"):
        lines.append(f"#SBATCH --cpus-per-task={profile['cpus_per_task']}")
    if profile.get("time"):
        lines.append(f"#SBATCH --time={profile['time']}")
    if profile.get("mem"):
        lines.append(f"#SBATCH --mem={profile['mem']}")
    if profile.get("mem_per_cpu"):
        lines.append(f"#SBATCH --mem-per-cpu={profile['mem_per_cpu']}")

    lines.append("")
    lines.extend(activation_lines(profile))
    lines.append("")
    lines.append(run_command)
    return "\n".join(lines) + "\n"


def validate_moose_material_csv(path: Path) -> list[str]:
    """Validate that a MOOSE material table has the columns needed by the starter input."""
    if not path.exists():
        raise SystemExit(f"Material CSV was not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    missing = [field for field in UO2_REQUIRED_MATERIAL_FIELDS if field not in fields]
    if "alpha_1_K" not in fields and "dilatation" not in fields:
        missing.append("alpha_1_K or dilatation")
    if missing:
        raise SystemExit(
            "Material CSV is missing fields needed for the UO2 MOOSE starter input: "
            + ", ".join(missing)
        )
    if not rows:
        raise SystemExit(f"Material CSV has no data rows: {path}")
    return sorted(fields)


def _material_prefix(material: str) -> str:
    prefix = "".join(char.lower() if char.isalnum() else "_" for char in material.strip())
    prefix = "_".join(part for part in prefix.split("_") if part)
    return prefix or "material"


def _default_function_map(material: str) -> dict[str, str]:
    prefix = _material_prefix(material)
    return {
        "k_W_mK": f"{prefix}_k",
        "Cp_J_kgK": f"{prefix}_Cp",
        "rho_kg_m3": f"{prefix}_rho",
        "alpha_1_K": f"{prefix}_alpha",
        "dilatation": f"{prefix}_dilatation",
        "E_Pa": f"{prefix}_E",
        "nu": f"{prefix}_nu",
    }


def load_material_metadata(path: Path | None) -> dict[str, Any]:
    """Load optional metadata from moose-qha-md-material."""
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Material metadata is not a JSON object: {path}")
    return data


def render_thermal_stress_input(
    *,
    material_include: Path = Path("uo2_material_functions.i"),
    material: str = "UO2",
    function_names: dict[str, str] | None = None,
    radius_m: float = 5.27e-3,
    height_m: float = 11.0e-3,
    linear_heat_rate_w_m: float = 2.0e4,
    surface_temperature_k: float = 900.0,
    initial_temperature_k: float = 300.0,
    stress_free_temperature_k: float = 300.0,
    radial_elements: int = 24,
    axial_elements: int = 48,
    transient: bool = False,
    end_time_s: float = 1.0,
    dt_s: float = 0.05,
) -> str:
    """Render a first-pass 2D RZ cylindrical pellet thermal-stress MOOSE input."""
    if radius_m <= 0 or height_m <= 0:
        raise ValueError("radius_m and height_m must be positive")
    if radial_elements < 1 or axial_elements < 1:
        raise ValueError("radial_elements and axial_elements must be positive")
    functions = _default_function_map(material)
    if function_names:
        functions.update({key: str(value) for key, value in function_names.items() if value})
    alpha_function = functions.get("alpha_1_K") or functions.get("dilatation")
    if not alpha_function:
        alpha_function = f"{_material_prefix(material)}_alpha"
    volumetric_heat_w_m3 = linear_heat_rate_w_m / (math.pi * radius_m * radius_m)
    executioner = (
        "\n".join(
            [
                "[Executioner]",
                "  type = Transient",
                f"  dt = {dt_s:.12g}",
                f"  end_time = {end_time_s:.12g}",
                "  solve_type = NEWTON",
                "[]",
            ]
        )
        if transient
        else "\n".join(
            [
                "[Executioner]",
                "  type = Steady",
                "  solve_type = NEWTON",
                "[]",
            ]
        )
    )
    include_line = f"!include {shlex.quote(str(material_include))}"
    return f"""# Generated by atomi moose-thermal-stress
# Use moose-qha-md-material first to create {material_include}.
# Geometry is a 2D axisymmetric R-Z standalone {material} pellet/cylinder.

{include_line}

[Mesh]
  [pellet]
    type = GeneratedMeshGenerator
    dim = 2
    nx = {radial_elements}
    ny = {axial_elements}
    xmin = 0
    xmax = {radius_m:.12g}
    ymin = 0
    ymax = {height_m:.12g}
  []
[]

[Problem]
  coord_type = RZ
[]

[Variables]
  [temperature]
    initial_condition = {initial_temperature_k:.12g}
  []
  [disp_r]
  []
  [disp_z]
  []
[]

[Kernels]
  [heat_conduction]
    type = HeatConduction
    variable = temperature
  []
  [fuel_heat_source]
    type = BodyForce
    variable = temperature
    value = {volumetric_heat_w_m3:.12g}
  []
[]

[Physics]
  [SolidMechanics]
    [QuasiStatic]
      [all]
        strain = SMALL
        add_variables = false
        displacements = 'disp_r disp_z'
        generate_output = 'stress_xx stress_yy stress_zz vonmises_stress'
      []
    []
  []
[]

[Materials]
  [thermal_conductivity]
    type = GenericFunctionMaterial
    prop_names = 'thermal_conductivity'
    prop_values = '{functions["k_W_mK"]}'
  []
  [density]
    type = GenericFunctionMaterial
    prop_names = 'density'
    prop_values = '{functions["rho_kg_m3"]}'
  []
  [specific_heat]
    type = GenericFunctionMaterial
    prop_names = 'specific_heat'
    prop_values = '{functions["Cp_J_kgK"]}'
  []
  [youngs_modulus]
    type = GenericFunctionMaterial
    prop_names = 'youngs_modulus'
    prop_values = '{functions["E_Pa"]}'
  []
  [poissons_ratio]
    type = GenericFunctionMaterial
    prop_names = 'poissons_ratio'
    prop_values = '{functions["nu"]}'
  []
  [elasticity_tensor]
    type = ComputeVariableIsotropicElasticityTensor
    args = temperature
    youngs_modulus = youngs_modulus
    poissons_ratio = poissons_ratio
  []
  [thermal_expansion]
    type = ComputeInstantaneousThermalExpansionFunctionEigenstrain
    temperature = temperature
    thermal_expansion_function = {alpha_function}
    stress_free_temperature = {stress_free_temperature_k:.12g}
    eigenstrain_name = thermal_expansion
  []
  [strain]
    type = ComputeSmallStrain
    displacements = 'disp_r disp_z'
    eigenstrain_names = thermal_expansion
  []
  [stress]
    type = ComputeLinearElasticStress
  []
[]

[BCs]
  [axis_symmetry]
    type = DirichletBC
    variable = disp_r
    boundary = left
    value = 0
  []
  [axial_anchor]
    type = DirichletBC
    variable = disp_z
    boundary = bottom
    value = 0
  []
  [outer_surface_temperature]
    type = DirichletBC
    variable = temperature
    boundary = right
    value = {surface_temperature_k:.12g}
  []
  [bottom_insulated]
    type = NeumannBC
    variable = temperature
    boundary = bottom
    value = 0
  []
  [top_insulated]
    type = NeumannBC
    variable = temperature
    boundary = top
    value = 0
  []
[]

{executioner}

[Outputs]
  exodus = true
  csv = true
[]
"""


def render_uo2_thermal_stress_input(**kwargs: Any) -> str:
    """Render a first-pass 2D RZ UO2 pellet thermal-stress MOOSE input."""
    kwargs.setdefault("material", "UO2")
    kwargs.setdefault("function_names", DEFAULT_FUNCTIONS)
    return render_thermal_stress_input(**kwargs)


def build_info(profile: dict[str, Any], profile_name: str) -> dict[str, Any]:
    """Return a compact profile summary suitable for JSON or text output."""
    return {
        "profile": profile_name,
        "status": profile.get("status"),
        "scheduler": profile.get("scheduler"),
        "partition": profile.get("partition"),
        "gres": profile.get("gres"),
        "modules": profile.get("modules", []),
        "activation_script": profile.get("activation_script"),
        "moose_root": profile.get("moose_root"),
        "test_executable": profile.get("test_executable"),
        "python_env": profile.get("python_env") or profile.get("env_path"),
        "calphad_work": profile.get("calphad_work"),
        "cuda": profile.get("cuda", {}),
        "next_tests": profile.get("next_tests", []),
    }


def run_smoke(profile: dict[str, Any], app: str | None = None, timeout: int = 30) -> dict[str, Any]:
    """Run a local MOOSE --help smoke command in the current environment."""
    command = shlex.split(_profile_command(profile, app=app))
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except OSError as exc:
        return {"command": command, "returncode": None, "output": str(exc)}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {"command": command, "returncode": None, "timed_out": True, "output": output}
    return {"command": command, "returncode": result.returncode, "output": result.stdout}


def info_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-info",
        description="Print MOOSE profile information from the local atomi HPC config.",
    )
    parser.add_argument("--hpc-config", type=Path, help="Local atomi HPC JSON config.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="MOOSE profile name.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    profile = load_moose_profile(args.hpc_config, args.profile)
    info = build_info(profile, args.profile)
    if args.json:
        print(json.dumps(info, indent=2))
        return
    print("Atomi MOOSE profile")
    for key in ("profile", "status", "scheduler", "partition", "gres"):
        if info.get(key):
            print(f"{key}: {info[key]}")
    if info.get("activation_script"):
        print(f"activation_script: {info['activation_script']}")
    if info.get("test_executable"):
        print(f"test_executable: {info['test_executable']}")
    if info.get("python_env"):
        print(f"python_env: {info['python_env']}")


def smoke_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-smoke",
        description="Run a local MOOSE executable --help smoke check.",
    )
    parser.add_argument("--hpc-config", type=Path, help="Local atomi HPC JSON config.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="MOOSE profile name.")
    parser.add_argument("--app", help="Override the configured executable.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = run_smoke(load_moose_profile(args.hpc_config, args.profile), args.app, args.timeout)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print("$ " + " ".join(report["command"]))
    print(report.get("output", "").rstrip())
    if report.get("returncode") not in (0, None):
        raise SystemExit(report["returncode"])


def write_submit_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-write-submit",
        description="Write a Slurm MOOSE submission script from a local profile.",
    )
    parser.add_argument("--hpc-config", type=Path, help="Local atomi HPC JSON config.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="MOOSE profile name.")
    parser.add_argument("--output", type=Path, default=Path("submit_moose.sh"))
    parser.add_argument("--job-name", default="moose")
    parser.add_argument("--command", help="Command to run after environment activation.")
    args = parser.parse_args(argv)

    profile = load_moose_profile(args.hpc_config, args.profile)
    script = render_slurm_submit(profile, command=args.command, job_name=args.job_name)
    args.output.write_text(script, encoding="utf-8")
    args.output.chmod(0o755)
    print(f"Wrote {args.output}")


def thermal_stress_main(
    argv: list[str] | None = None,
    *,
    default_material: str = "UO2",
    default_material_csv: Path = Path("uo2_moose_material_properties.csv"),
    default_material_include: Path = Path("uo2_material_functions.i"),
    default_output: Path = Path("uo2_pellet_thermal_stress.i"),
    prog: str = "moose-thermal-stress",
) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Write a starter cylindrical pellet thermal-stress MOOSE input that consumes "
            "moose-qha-md-material CSV/[Functions] output."
        ),
    )
    parser.add_argument("--material", default=default_material)
    parser.add_argument("--material-csv", type=Path, default=default_material_csv)
    parser.add_argument(
        "--material-meta",
        type=Path,
        help="Metadata JSON from moose-qha-md-material. Defaults to CSV basename + .meta.json.",
    )
    parser.add_argument("--material-include", type=Path, default=default_material_include)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--radius-m", type=float, default=5.27e-3)
    parser.add_argument("--height-m", type=float, default=11.0e-3)
    parser.add_argument("--linear-heat-rate-w-m", type=float, default=2.0e4)
    parser.add_argument("--surface-temperature-K", type=float, default=900.0)
    parser.add_argument("--initial-temperature-K", type=float, default=300.0)
    parser.add_argument("--stress-free-T", type=float, default=300.0)
    parser.add_argument("--radial-elements", type=int, default=24)
    parser.add_argument("--axial-elements", type=int, default=48)
    parser.add_argument("--transient", action="store_true")
    parser.add_argument("--end-time-s", type=float, default=1.0)
    parser.add_argument("--dt-s", type=float, default=0.05)
    parser.add_argument(
        "--no-validate-material",
        action="store_true",
        help="Write the input even if the material CSV is not present yet.",
    )
    parser.add_argument("--write-submit", type=Path, help="Optional Slurm script to write.")
    parser.add_argument("--hpc-config", type=Path, help="Local atomi HPC JSON config.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="MOOSE profile name for submit file.")
    parser.add_argument(
        "--app-command",
        help="Command for the submit script. Defaults to the configured test executable and input.",
    )
    args = parser.parse_args(argv)

    meta_path = args.material_meta
    if meta_path is None:
        meta_path = args.material_csv.with_suffix(".meta.json")
    metadata = load_material_metadata(meta_path)
    material = str(metadata.get("material") or args.material)
    stress_free_t = float(metadata.get("stress_free_T_K", args.stress_free_T))
    function_names = metadata.get("moose_functions")
    if not isinstance(function_names, dict):
        function_names = _default_function_map(material)

    if not args.no_validate_material:
        validate_moose_material_csv(args.material_csv)
    input_text = render_thermal_stress_input(
        material_include=args.material_include,
        material=material,
        function_names=function_names,
        radius_m=args.radius_m,
        height_m=args.height_m,
        linear_heat_rate_w_m=args.linear_heat_rate_w_m,
        surface_temperature_k=args.surface_temperature_K,
        initial_temperature_k=args.initial_temperature_K,
        stress_free_temperature_k=stress_free_t,
        radial_elements=args.radial_elements,
        axial_elements=args.axial_elements,
        transient=args.transient,
        end_time_s=args.end_time_s,
        dt_s=args.dt_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(input_text, encoding="utf-8")
    print(f"Wrote {args.output}")

    if args.write_submit:
        profile = load_moose_profile(args.hpc_config, args.profile)
        executable = profile.get("test_executable") or profile.get("executable") or "moose-opt"
        command = args.app_command or f"{shlex.quote(str(executable))} -i {shlex.quote(str(args.output))}"
        script = render_slurm_submit(profile, command=command, job_name="uo2_pellet")
        args.write_submit.parent.mkdir(parents=True, exist_ok=True)
        args.write_submit.write_text(script, encoding="utf-8")
        args.write_submit.chmod(0o755)
        print(f"Wrote {args.write_submit}")


def uo2_thermal_stress_main(argv: list[str] | None = None) -> None:
    thermal_stress_main(
        argv,
        default_material="UO2",
        default_material_csv=Path("uo2_moose_material_properties.csv"),
        default_material_include=Path("uo2_material_functions.i"),
        default_output=Path("uo2_pellet_thermal_stress.i"),
        prog="moose-uo2-thermal-stress",
    )


if __name__ == "__main__":
    info_main()
