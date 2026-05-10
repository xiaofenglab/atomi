#!/usr/bin/env python3

"""

md-engine --config config_600_1200K.json

md-engine --resume --config config_600_1200K.json

md-engine --resume --start-from nvt_ramp_700K --config config_600_1200K.json


"""

import json
import math
import subprocess
import time
import re
from difflib import get_close_matches
from pathlib import Path
from statistics import mean
import argparse
import shutil


ROOT = Path.cwd()
STAGES = ROOT / "stages"
ANALYSIS = ROOT / "analysis"


def set_project_root(root):
    global ROOT, STAGES, ANALYSIS
    ROOT = Path(root).resolve()
    STAGES = ROOT / "stages"
    ANALYSIS = ROOT / "analysis"


# ---------------------------------------------------
# CONFIG / PATHS
# ---------------------------------------------------

def load_config(config_path=None):
    if config_path is None:
        config_path = ROOT / "config.json"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = (ROOT / config_path).resolve()

    with open(config_path) as f:
        cfg = json.load(f)

    cfg["wrapper_script"] = str(_resolve_project_path(cfg["wrapper_script"]))
    cfg["model_file"] = str(_resolve_project_path(cfg["model_file"]))
    if "initial_structure" in cfg:
        cfg["initial_structure"] = str(_resolve_project_path(cfg["initial_structure"]))

    cfg["_config_path"] = str(config_path)
    return cfg


def _resolve_project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def path_for_lammps(p):
    return str(Path(p).resolve())


def make_read_command(structure_path):
    p = Path(structure_path).resolve()
    name = p.name.lower()

    is_restart = (p.suffix.lower() == ".restart") or name.startswith("restart.")
    if is_restart:
        print(f"[make_read_command] Using read_restart for: {p}", flush=True)
        return f"read_restart    {path_for_lammps(p)}"

    print(f"[make_read_command] Using read_data for: {p}", flush=True)
    return f"read_data       {path_for_lammps(p)}"


# ---------------------------------------------------
# WALLTIME ESTIMATION
# ---------------------------------------------------

def estimate_walltime(cfg, stage, steps):
    if "walltime_hours" in stage:
        return max(float(stage["walltime_hours"]), 0.0)
    if "walltime_minutes" in stage:
        return max(float(stage["walltime_minutes"]) / 60.0, 0.0)

    perf = cfg.get("performance", {})
    if "walltime_hours" in perf:
        return max(float(perf["walltime_hours"]), 0.0)
    if "walltime_minutes" in perf:
        return max(float(perf["walltime_minutes"]) / 60.0, 0.0)

    ref_atoms = perf.get("reference_atoms", 96)
    ref_steps = perf.get("reference_steps", 20000)
    if "reference_walltime_hours" in perf:
        ref_hours = perf["reference_walltime_hours"]
        safety_factor = 1.0
    else:
        ref_hours = perf.get("reference_hours", 0.25)
        safety_factor = perf.get("safety_factor", 1.5)

    atoms = perf.get("atoms_small", ref_atoms)
    if "atoms" in perf:
        atoms = perf["atoms"]
    if stage.get("large_cell", False):
        atoms = perf.get("atoms_large", ref_atoms)

    hours = ref_hours * (atoms / ref_atoms) * (steps / ref_steps)
    hours *= safety_factor

    return max(hours, 0.25)


def hours_to_slurm(hours):
    total = math.ceil(hours * 3600)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def timestep_ps(cfg):
    return float(cfg.get("timestep_ps", cfg.get("timestep", 0.0001)))


def steps_from_time_ps(cfg, time_ps):
    return int(round(float(time_ps) / timestep_ps(cfg)))


def steps_for_time_ps(cfg, time_ps):
    return max(1, int(round(float(time_ps) / timestep_ps(cfg))))


def resolve_run_steps(cfg, stage, default_steps=None):
    for key in ("fixed_steps", "steps", "run_steps", "nsteps"):
        if key in stage:
            return int(stage[key])

    for key in ("time_ps", "run_time_ps", "duration_ps", "production_time_ps"):
        if key in stage:
            return steps_from_time_ps(cfg, stage[key])

    if default_steps is not None:
        return int(default_steps)

    raise ValueError(
        f"Stage {stage['name']} needs fixed_steps/steps or time_ps/run_time_ps."
    )


def stage_uses_fixed_steps(stage):
    return any(
        key in stage
        for key in (
            "fixed_steps",
            "steps",
            "run_steps",
            "nsteps",
            "time_ps",
            "run_time_ps",
            "duration_ps",
            "production_time_ps",
        )
    )


def stage_uses_constant_chunk_steps(stage):
    if stage.get("constant_chunk_steps", False):
        return True
    if stage.get("fixed_chunk_steps", False):
        return True
    if stage.get("adaptive_growth", False):
        return False
    name = stage.get("name", "").lower()
    return stage.get("type") == "npt" and "_eqm" in name


# ---------------------------------------------------
# RAMP CONTINUATION
# ---------------------------------------------------

def is_nvt_ramp_stage(stage):
    if stage.get("type") not in ("nvt", "nvt_replicate"):
        return False
    if "temperature_start" not in stage or "temperature_end" not in stage:
        return False
    return float(stage["temperature_start"]) != float(stage["temperature_end"])


def ramp_rules(cfg, stage):
    rules = {
        "tail_average_ps": 2.0,
        "continue_from_tail_temperature": True,
        "accept_temperature_tol_min": None,
        "accept_temperature_tol_fraction": None,
    }
    rules.update(cfg.get("ramp_rules", {}))
    rules.update(stage.get("ramp_override", {}))

    eq = cfg.get("equilibrium_rules", {})
    if rules["accept_temperature_tol_min"] is None:
        rules["accept_temperature_tol_min"] = eq.get("temperature_tol_min", 20.0)
    if rules["accept_temperature_tol_fraction"] is None:
        rules["accept_temperature_tol_fraction"] = eq.get("temperature_tol_fraction", 0.03)
    return rules


def tail_average_temperature(cfg, steps, temperatures, tail_ps):
    if not steps or not temperatures:
        return None
    tail_steps = steps_for_time_ps(cfg, tail_ps)
    final_step = steps[-1]
    selected = [
        temp
        for step, temp in zip(steps, temperatures)
        if step >= final_step - tail_steps
    ]
    if not selected:
        selected = temperatures[-max(1, min(len(temperatures), tail_steps)):]
    return mean(selected)


def ramp_target_reached(cfg, stage, steps, temperatures):
    if not is_nvt_ramp_stage(stage):
        return False, "not_ramp"
    rules = ramp_rules(cfg, stage)
    target = float(stage["temperature_end"])
    tail_mean = tail_average_temperature(cfg, steps, temperatures, rules["tail_average_ps"])
    if tail_mean is None:
        return False, "no_temperature_tail"
    tol = max(
        float(rules["accept_temperature_tol_min"]),
        abs(target) * float(rules["accept_temperature_tol_fraction"]),
    )
    if abs(tail_mean - target) <= tol:
        return True, (
            f"ramp target reached: tail_{rules['tail_average_ps']}ps_T="
            f"{tail_mean:.3f} K target={target:.3f} K tol={tol:.3f} K"
        )
    return False, (
        f"ramp target not reached: tail_{rules['tail_average_ps']}ps_T="
        f"{tail_mean:.3f} K target={target:.3f} K tol={tol:.3f} K"
    )


# ---------------------------------------------------
# CREATE PER-CHUNK WRAPPER
# ---------------------------------------------------

def create_stage_wrapper(cfg, chunk_dir, walltime):
    template = Path(cfg["wrapper_script"]).read_text()

    new_script, nsubs = re.subn(
        r'(?m)^#SBATCH\s+--time=\S+\s*$',
        f"#SBATCH --time={walltime}",
        template
    )

    if nsubs == 0:
        raise RuntimeError(
            "Could not find a '#SBATCH --time=...' line in run_lammps_gpu.sh. "
            "Please include one SBATCH time line in the wrapper."
        )

    wrapper = chunk_dir / "run_stage.sh"
    wrapper.write_text(new_script)
    wrapper.chmod(0o755)

    return wrapper


# ---------------------------------------------------
# SLURM
# ---------------------------------------------------

def submit_job(wrapper, input_file, workdir):
    cmd = ["sbatch", str(wrapper), input_file]
    out = subprocess.check_output(cmd, cwd=workdir).decode()

    m = re.search(r"Submitted batch job\s+(\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse job ID from sbatch output:\n{out}")

    return int(m.group(1))


def wait_job(job_id, poll):
    while True:
        out = subprocess.check_output(
            ["squeue", "-j", str(job_id), "-h"]
        ).decode()

        if out.strip() == "":
            break

        time.sleep(poll)


def check_slurm_outputs(chunk_dir):
    out_files = sorted(chunk_dir.glob("lammps_gpu.*.out"))
    err_files = sorted(chunk_dir.glob("lammps_gpu.*.err"))

    out_text = ""
    err_text = ""

    if out_files:
        out_text = out_files[-1].read_text(errors="ignore")
    if err_files:
        err_text = err_files[-1].read_text(errors="ignore")

    combined = out_text + "\n" + err_text

    m = re.search(r"EXIT_STATUS\s*=\s*(\d+)", combined)
    if m:
        code = int(m.group(1))
        if code != 0:
            raise RuntimeError(f"LAMMPS wrapper reported nonzero EXIT_STATUS={code}")

    if "ERROR:" in combined and "EXIT_STATUS       = 0" not in combined:
        raise RuntimeError("Detected ERROR in SLURM output before normal finish")


# ---------------------------------------------------
# PARSE LAMMPS THERMO
# ---------------------------------------------------

def parse_thermo(logfile):
    steps = []
    T = []
    P = []
    V = []
    PE = []

    pattern = re.compile(
        r"^\s*(\d+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)"
    )

    with open(logfile) as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                steps.append(int(m.group(1)))
                T.append(float(m.group(2)))
                PE.append(float(m.group(3)))
                P.append(float(m.group(5)))
                V.append(float(m.group(6)))

    return steps, T, P, V, PE


# ---------------------------------------------------
# EQUILIBRIUM RULES
# ---------------------------------------------------

def effective_rules(cfg, stage):
    rules = dict(cfg["equilibrium_rules"])
    override = stage.get("equilibrium_override", {})
    rules.update(override)
    return rules


def slope(x, y):
    xm = mean(x)
    ym = mean(y)

    num = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    den = sum((a - xm) ** 2 for a in x)

    if den == 0:
        return 0.0

    return num / den


def stage_target_temperature(stage):
    """
    For fixed-T stages, returns 'temperature'.
    For ramp stages, use the end temperature as the equilibrium target.
    """
    if "temperature_end" in stage:
        return stage["temperature_end"]
    if "temperature" in stage:
        return stage["temperature"]
    return None


def summarize_thermo(stage, steps, T, P, V, PE):
    if len(steps) < 2:
        return None

    tail = max(1, int(len(steps) * 0.7))

    return {
        "npoints": len(steps),
        "tail_start_index": tail,
        "T_mean": mean(T[tail:]),
        "T_min": min(T),
        "T_max": max(T),
        "P_mean": mean(P[tail:]),
        "P_mean_bar": mean(P[tail:]),
        "P_mean_GPa": mean(P[tail:]) * 1.0e-4,
        "V_mean": mean(V[tail:]),
        "V_slope": slope(steps[tail:], V[tail:]),
        "V_slope_per_step": slope(steps[tail:], V[tail:]),
        "PE_mean": mean(PE[tail:]),
        "PE_slope": slope(steps[tail:], PE[tail:]),
        "PE_slope_per_step": slope(steps[tail:], PE[tail:]),
    }


def check_equilibrium(cfg, stage, steps, T, V, PE):
    if len(steps) < 10:
        return False, "too_few_points"

    rules = effective_rules(cfg, stage)
    summary = summarize_thermo(stage, steps, T, [0.0] * len(T), V, PE)

    if summary is None:
        return False, "no_summary"

    target = stage_target_temperature(stage)
    if target is not None:
        tol = max(
            rules["temperature_tol_min"],
            target * rules["temperature_tol_fraction"]
        )

        if abs(summary["T_mean"] - target) > tol:
            return False, f"T_mean_outside_tol ({summary['T_mean']:.3f} vs {target}; tol={tol:.3f})"

        if summary["T_max"] > target * rules["runaway_temperature_factor"]:
            raise RuntimeError("Thermal runaway detected")

    if stage["type"] == "npt":
        if abs(summary["V_slope"]) > rules["volume_slope_tol"]:
            return False, f"V_slope_too_large ({summary['V_slope']:.6e}; tol={rules['volume_slope_tol']:.6e})"

    if abs(summary["PE_slope"]) > rules["energy_slope_tol"]:
        return False, f"PE_slope_too_large ({summary['PE_slope']:.6e}; tol={rules['energy_slope_tol']:.6e})"

    return True, "strict_equilibrium"


def check_stable_enough(cfg, stage, steps, T, V, PE):
    if len(steps) < 10:
        return False, "too_few_points"

    rules = effective_rules(cfg, stage)
    summary = summarize_thermo(stage, steps, T, [0.0] * len(T), V, PE)

    if summary is None:
        return False, "no_summary"

    loose_temp_fraction = rules.get("stable_temperature_tol_fraction", rules["temperature_tol_fraction"] * 2.0)
    loose_temp_min = rules.get("stable_temperature_tol_min", rules["temperature_tol_min"] * 2.0)
    loose_v_slope = rules.get("stable_volume_slope_tol", rules["volume_slope_tol"] * 5.0)
    loose_pe_slope = rules.get("stable_energy_slope_tol", rules["energy_slope_tol"] * 5.0)
    stable_runaway_factor = rules.get("stable_runaway_temperature_factor", rules["runaway_temperature_factor"])

    target = stage_target_temperature(stage)
    if target is not None:
        tol = max(loose_temp_min, target * loose_temp_fraction)

        if abs(summary["T_mean"] - target) > tol:
            return False, f"stable_T_mean_outside_tol ({summary['T_mean']:.3f} vs {target}; tol={tol:.3f})"

        if summary["T_max"] > target * stable_runaway_factor:
            raise RuntimeError("Thermal runaway detected in stable check")

    if stage["type"] == "npt":
        if abs(summary["V_slope"]) > loose_v_slope:
            return False, f"stable_V_slope_too_large ({summary['V_slope']:.6e}; tol={loose_v_slope:.6e})"

    if abs(summary["PE_slope"]) > loose_pe_slope:
        return False, f"stable_PE_slope_too_large ({summary['PE_slope']:.6e}; tol={loose_pe_slope:.6e})"

    return True, "stable_enough"


# ---------------------------------------------------
# PLOT / SUMMARY
# ---------------------------------------------------

def plot_thermo(outdir, steps, T, P, V, PE):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warning] matplotlib not available; skipping thermo.png", flush=True)
        return

    fig, ax = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

    ax[0].plot(steps, T)
    ax[0].set_ylabel("T (K)")

    ax[1].plot(steps, P)
    ax[1].set_ylabel("P (bar)")

    ax[2].plot(steps, V)
    ax[2].set_ylabel("Volume")

    ax[3].plot(steps, PE)
    ax[3].set_ylabel("PE (eV)")
    ax[3].set_xlabel("Step")

    plt.tight_layout()
    fig.savefig(outdir / "thermo.png", dpi=150)
    plt.close()


def write_chunk_summary(chunk_dir, stage, steps, T, P, V, PE, note=""):
    summary = summarize_thermo(stage, steps, T, P, V, PE)
    lines = [f"stage: {stage['name']}", f"note: {note}"]

    if summary is not None:
        for k, v in summary.items():
            lines.append(f"{k}: {v}")

    (chunk_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def write_decision(chunk_dir, lines):
    (chunk_dir / "decision.txt").write_text("\n".join(lines) + "\n")


def write_production_chunk_summary(chunk_dir, stage, steps, T, P, V, PE, note=""):
    summary = summarize_thermo(stage, steps, T, P, V, PE)
    lines = [f"stage: {stage['name']}", f"note: {note}"]

    if summary is not None:
        for k, v in summary.items():
            lines.append(f"{k}: {v}")

    (chunk_dir / "chunk_summary.txt").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------
# INPUT GENERATION
# ---------------------------------------------------

def get_tdamp(cfg, stage):
    if "tdamp" in stage:
        return stage["tdamp"]

    thermo_cfg = cfg.get("thermostat", {})
    default_tdamp = thermo_cfg.get("tdamp", 1.0)

    tdamp_by_temp = thermo_cfg.get("tdamp_by_temp", [])
    Tref = stage_target_temperature(stage)

    if Tref is not None and tdamp_by_temp:
        for entry in tdamp_by_temp:
            if Tref <= entry["tmax"]:
                return entry["tdamp"]

    return default_tdamp


def get_pdamp(cfg, stage):
    if "pdamp" in stage:
        return stage["pdamp"]

    baro_cfg = cfg.get("barostat", {})
    if stage.get("large_cell", False):
        return baro_cfg.get("pdamp_large", 5.0)
    return baro_cfg.get("pdamp_small", 5.0)


def generate_input(
    cfg,
    stage,
    run_steps,
    structure_path,
    stage_name,
    chunk_idx,
    resume_mode=False,
    temperature_start_override=None,
):
    model = path_for_lammps(cfg["model_file"])
    read_cmd = make_read_command(structure_path)

    chunk_tag = f"{stage_name}_c{chunk_idx:02d}"
    dump_name = f"dump.{chunk_tag}.lammpstrj"
    final_restart = f"{chunk_tag}.restart"
    final_data = f"{chunk_tag}.data"

    txt = f"""
units           metal
dimension       3
boundary        p p p
atom_style      atomic
atom_modify     map yes
newton          on

{read_cmd}

mass            1 {cfg["mass_O"]}
mass            2 {cfg["mass_U"]}

pair_style      mace no_domain_decomposition
pair_coeff      * * {model} O U

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

timestep        {cfg["timestep"]}
"""

    if stage["type"] == "relax":
        r = cfg["relax"]
        txt += f"""
fix             1 all box/relax iso 0.0 vmax {r["vmax"]}
min_style       cg
minimize        {r["etol"]} {r["ftol"]} {r["maxiter"]} {r["maxeval"]}

write_data      {final_data}
"""
        return txt, final_data, None

    if stage["type"] == "replicate":
        rep = stage["replicate"]
        txt += f"""
replicate       {rep[0]} {rep[1]} {rep[2]}

write_data      {final_data}
"""
        return txt, final_data, None

    if stage["type"] == "nvt_replicate":
        rep = stage.get("replicate", [1, 1, 1])
        if rep != [1, 1, 1]:
            txt += f"""
replicate       {rep[0]} {rep[1]} {rep[2]}
"""

    # Support both fixed-T and ramp-T NVT stages
    Tstart = stage.get("temperature_start", stage.get("temperature", 300.0))
    Tend = stage.get("temperature_end", stage.get("temperature", 300.0))
    if temperature_start_override is not None and is_nvt_ramp_stage(stage):
        Tstart = float(temperature_start_override)
    tdamp = get_tdamp(cfg, stage)

    p = Path(structure_path).resolve()
    name = p.name.lower()
    is_restart = (p.suffix.lower() == ".restart") or name.startswith("restart.")

    # Rule:
    # - if input is data file: create velocity
    # - if input is restart: keep the carried velocities unless explicitly requested
    if (not is_restart) or stage.get("recreate_velocity", False):
        txt += f"""
velocity        all create {Tstart} {cfg["velocity_seed"]} mom yes rot yes dist gaussian
"""

    txt += """
fix             2 all momentum 1000 linear 1 1 1
"""

    if stage["type"] in ("nvt", "nvt_replicate"):
        txt += f"""
fix             1 all nvt temp {Tstart} {Tend} {tdamp}
"""
    elif stage["type"] == "npt":
        pdamp = get_pdamp(cfg, stage)
        txt += f"""
fix             1 all npt temp {Tend} {Tend} {tdamp} iso 0.0 0.0 {pdamp}
"""
    else:
        raise ValueError(f"Unknown stage type: {stage['type']}")

    txt += f"""
thermo          100
thermo_style    custom step temp pe etotal press vol lx ly lz
thermo_modify   flush yes

dump            1 all custom 500 {dump_name} id type x y z
dump_modify     1 sort id

restart         5000 {chunk_tag}.restart1 {chunk_tag}.restart2
run             {run_steps}

write_restart   {final_restart}
write_data      {final_data}
"""
    return txt, final_data, final_restart


def stage_temperature(stage):
    if "temperature" in stage:
        return stage["temperature"]
    if "temperature_end" in stage:
        return stage["temperature_end"]
    raise ValueError(f"Production stage {stage['name']} needs temperature or temperature_end")


def production_stage_selected(stage):
    if stage.get("production_run", False):
        return True
    return stage.get("name", "").startswith("npt_prod")


def generate_production_input(cfg, stage, structure_path, chunk_tag):
    model = path_for_lammps(cfg["model_file"])
    read_cmd = make_read_command(structure_path)

    temperature = stage_temperature(stage)
    tdamp = get_tdamp(cfg, stage)
    pdamp = get_pdamp(cfg, stage)
    pressure = stage.get("pressure_bar", cfg.get("pressure_bar", 0.0))
    timestep = cfg.get("timestep", timestep_ps(cfg))
    dump_every = stage.get("dump_every", cfg.get("dump_every", 500))
    run_steps = resolve_run_steps(cfg, stage)

    dump_name = f"dump.{chunk_tag}.lammpstrj"
    final_restart = f"{chunk_tag}.restart"
    final_data = f"{chunk_tag}.data"

    p = Path(structure_path).resolve()
    name = p.name.lower()
    is_restart = (p.suffix.lower() == ".restart") or name.startswith("restart.")
    velocity_text = ""
    if not is_restart or stage.get("recreate_velocity", False):
        seed = cfg.get("velocity_seed", 12345)
        velocity_text = (
            f"velocity        all create {temperature} {seed} mom yes rot yes dist gaussian\n"
        )

    txt = f"""units           metal
dimension       3
boundary        p p p
atom_style      atomic
atom_modify     map yes
newton          on

{read_cmd}

mass            1 {cfg["mass_O"]}
mass            2 {cfg["mass_U"]}

pair_style      mace no_domain_decomposition
pair_coeff      * * {model} O U

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

timestep        {timestep}

{velocity_text}fix             2 all momentum 1000 linear 1 1 1
fix             1 all npt temp {temperature} {temperature} {tdamp} iso {pressure} {pressure} {pdamp}

thermo          100
thermo_style    custom step temp pe etotal press vol lx ly lz
thermo_modify   flush yes

dump            1 all custom {dump_every} {dump_name} id type x y z
dump_modify     1 sort id

restart         50000 {chunk_tag}.restart1 {chunk_tag}.restart2
run             {run_steps}

write_restart   {final_restart}
write_data      {final_data}
"""
    return txt, final_data, final_restart, run_steps


# ---------------------------------------------------
# FORCE PASS / STAGE OUTPUTS
# ---------------------------------------------------

def stage_force_pass_requested(stage_dir):
    return (stage_dir / "FORCE_PASS").exists()


def write_stage_outputs(stage_dir, stage_name, final_data_path, final_restart_path=None, pass_note=""):
    stage_final_data = stage_dir / f"{stage_name}.data"
    shutil.copy2(final_data_path, stage_final_data)

    if final_restart_path is not None and Path(final_restart_path).exists():
        stage_final_restart = stage_dir / f"{stage_name}.restart"
        shutil.copy2(final_restart_path, stage_final_restart)

    (stage_dir / "PASS").write_text(pass_note + "\n")


# ---------------------------------------------------
# STAGE RESUME HELPERS
# ---------------------------------------------------

def find_latest_chunk_restart(stage_dir, stage_name, chunk_name=None):
    """
    Find the latest completed chunk restart for a stage.
    Returns:
      (next_chunk_idx, restart_path) or (1, None)
    """
    if chunk_name:
        chunk_dirs = [stage_dir / chunk_name]
    else:
        chunk_dirs = sorted(stage_dir.glob("chunk_*"))
    best_chunk = None
    best_restart = None

    for chunk_dir in chunk_dirs:
        if chunk_name:
            idx = 1
        else:
            m = re.match(r"chunk_(\d+)$", chunk_dir.name)
            if not m:
                continue
            idx = int(m.group(1))
        if not chunk_dir.exists():
            continue

        restart_path = chunk_dir / f"{stage_name}_c{idx:02d}.restart"
        log_path = chunk_dir / f"log.in.{stage_name}_c{idx:02d}"

        if restart_path.exists() and log_path.exists():
            if best_chunk is None or idx > best_chunk:
                best_chunk = idx
                best_restart = restart_path.resolve()

    if best_chunk is None:
        return 1, None

    return best_chunk + 1, best_restart


def chunk_dir_for_index(stage_dir, stage, chunk):
    return stage_dir / stage.get("chunk_name", f"chunk_{chunk:02d}")


def previous_chunk_log(stage_dir, stage, stage_name, chunk):
    if chunk <= 1:
        return None
    prev_chunk = chunk - 1
    chunk_dir = chunk_dir_for_index(stage_dir, stage, prev_chunk)
    log = chunk_dir / f"log.in.{stage_name}_c{prev_chunk:02d}"
    if log.exists():
        return log
    matches = sorted(chunk_dir.glob("log.in.*"))
    if matches:
        return matches[-1]
    return None


def ramp_start_temperature_for_chunk(cfg, stage, stage_dir, stage_name, chunk):
    if chunk <= 1 or not is_nvt_ramp_stage(stage):
        return None

    rules = ramp_rules(cfg, stage)
    if not rules.get("continue_from_tail_temperature", True):
        return None

    log = previous_chunk_log(stage_dir, stage, stage_name, chunk)
    if log is None:
        print(
            f"[warning] Could not find previous chunk log for ramp continuation in {stage_name}; "
            "using configured temperature_start.",
            flush=True,
        )
        return None

    steps, T, _P, _V, _PE = parse_thermo(log)
    tail_mean = tail_average_temperature(cfg, steps, T, rules["tail_average_ps"])
    if tail_mean is None:
        print(
            f"[warning] Could not read previous tail temperature from {log}; "
            "using configured temperature_start.",
            flush=True,
        )
        return None

    target = float(stage["temperature_end"])
    original_start = float(stage["temperature_start"])
    if target >= original_start:
        tail_mean = min(max(tail_mean, original_start), target)
    else:
        tail_mean = max(min(tail_mean, original_start), target)

    print(
        f"  ramp continuation: tail {rules['tail_average_ps']} ps average from "
        f"{log.parent.name} is {tail_mean:.3f} K; ramping toward {target:.3f} K",
        flush=True,
    )
    return tail_mean


def stage_artifact(stage_dir, stage_name):
    restart_candidate = stage_dir / f"{stage_name}.restart"
    data_candidate = stage_dir / f"{stage_name}.data"

    if restart_candidate.exists():
        return restart_candidate.resolve()
    if data_candidate.exists():
        return data_candidate.resolve()
    return None


def stage_prefers_restart(stage):
    name = stage["name"].lower()
    return (stage["type"] == "npt") or ("nvt_relax" in name) or ("nvt_eqm" in name) or ("nvt_ramp" in name)


def _resolve_stage_input(stage):
    candidates = [stage["input_structure"]]
    if stage.get("input_data_fallback"):
        candidates.append(stage["input_data_fallback"])

    for candidate in candidates:
        path = _resolve_project_path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find stage input_structure or input_data_fallback for "
        f"{stage['name']}: {', '.join(candidates)}"
    )


def run_production_stage(cfg, stage, resume_mode=False):
    stage_name = stage["name"]
    stage_dir = STAGES / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    chunk_name = stage.get("chunk_name", "chunk_production")
    chunk_dir = stage_dir / chunk_name
    chunk_dir.mkdir(parents=True, exist_ok=True)

    pass_file = stage_dir / "PASS"
    final_stage_data = stage_dir / f"{stage_name}.data"
    final_stage_restart = stage_dir / f"{stage_name}.restart"

    if resume_mode and pass_file.exists() and final_stage_restart.exists():
        print(f"Skipping completed production stage: {stage_name}", flush=True)
        return final_stage_restart.resolve()

    if "input_structure" not in stage:
        raise ValueError(
            f"Production stage {stage_name} needs input_structure. "
            "Production runs do not use previous stage output implicitly."
        )
    structure_path = _resolve_stage_input(stage)

    chunk_tag = f"{stage_name}_production"
    input_name = f"in.{chunk_tag}"
    inputfile = chunk_dir / input_name

    input_text, final_data_name, final_restart_name, run_steps = generate_production_input(
        cfg,
        stage,
        structure_path,
        chunk_tag,
    )
    inputfile.write_text(input_text)

    wall_hours = estimate_walltime(cfg, stage, run_steps)
    walltime = hours_to_slurm(wall_hours)
    wrapper = create_stage_wrapper(cfg, chunk_dir, walltime)

    print(f"Running production stage {stage_name}", flush=True)
    print(f"  input_structure: {structure_path}", flush=True)
    print(f"  steps: {run_steps}", flush=True)
    print(f"  walltime: {walltime}", flush=True)

    job = submit_job(wrapper, inputfile.name, chunk_dir)
    wait_job(job, cfg.get("poll_seconds", 10))
    check_slurm_outputs(chunk_dir)

    log = chunk_dir / f"log.{inputfile.name}"
    if not log.exists():
        raise RuntimeError(f"LAMMPS log file not found: {log}")

    steps, T, P, V, PE = parse_thermo(log)
    if len(T) == 0:
        raise RuntimeError(f"LAMMPS produced no thermo output in {log}")

    plot_thermo(chunk_dir, steps, T, P, V, PE)
    write_production_chunk_summary(
        chunk_dir,
        stage,
        steps,
        T,
        P,
        V,
        PE,
        note="production fixed-length NPT",
    )

    final_data_path = (chunk_dir / final_data_name).resolve()
    final_restart_path = (chunk_dir / final_restart_name).resolve()

    if not final_data_path.exists():
        raise RuntimeError(f"Expected output data file not found: {final_data_path}")
    if not final_restart_path.exists():
        raise RuntimeError(f"Expected output restart file not found: {final_restart_path}")

    shutil.copy2(final_data_path, final_stage_data)
    shutil.copy2(final_restart_path, final_stage_restart)

    summary = summarize_thermo(stage, steps, T, P, V, PE)
    pass_lines = [
        f"stage: {stage_name}",
        "status: completed production run",
        f"input_structure: {structure_path}",
        f"steps: {run_steps}",
        f"final_data: {final_stage_data}",
        f"final_restart: {final_stage_restart}",
    ]
    if summary:
        pass_lines.append("summary:")
        for k, v in summary.items():
            pass_lines.append(f"  {k}: {v}")
    pass_file.write_text("\n".join(pass_lines) + "\n")

    print(f"Completed production stage {stage_name}", flush=True)
    print(f"  final restart: {final_stage_restart}", flush=True)
    return final_stage_restart.resolve()


def production_stage_from_equilibrium(cfg, stage, fixed_steps, production_time_ps):
    temperature = stage_target_temperature(stage)
    if temperature is None:
        return None

    stage_name = stage["name"]
    stage_dir = STAGES / stage_name
    if not (stage_dir / "PASS").exists():
        return None

    restart = stage_dir / f"{stage_name}.restart"
    data = stage_dir / f"{stage_name}.data"
    if not restart.exists() and not data.exists():
        return None

    temp_label = _temperature_label(temperature)
    production_stage = {
        "name": f"npt_prod_{temp_label}K",
        "type": "npt",
        "large_cell": bool(stage.get("large_cell", False)),
        "temperature": temperature,
        "chunk_name": "chunk_production",
        "fixed_steps": int(fixed_steps),
        "max_chunks": 1,
        "production_run": True,
        "min_chunks_before_accept": 1,
        "accept_if_stable": False,
        "source_equilibration_stage": stage_name,
    }

    if restart.exists():
        production_stage["input_structure"] = _relative_to_root(restart)
        if data.exists():
            production_stage["input_data_fallback"] = _relative_to_root(data)
    else:
        production_stage["input_structure"] = _relative_to_root(data)

    if "equilibrium_override" in stage:
        production_stage["equilibrium_override"] = stage["equilibrium_override"]
    if production_time_ps is not None:
        production_stage["production_time_ps"] = production_time_ps

    return production_stage


def write_production_config_from_equilibration(
    cfg,
    output_path=Path("config_production.json"),
    fixed_steps=None,
    production_time_ps=None,
):
    settings = cfg.get("production_settings", {})
    timestep_ps = settings.get("timestep_ps", cfg.get("timestep", 0.0001))
    if fixed_steps is None:
        fixed_steps = settings.get("fixed_steps")
    if production_time_ps is None:
        production_time_ps = settings.get("production_time_ps", 100.0)
    if fixed_steps is None:
        fixed_steps = int(round(float(production_time_ps) / float(timestep_ps)))

    stages = []
    for stage in cfg.get("stages", []):
        if production_stage_selected(stage):
            continue
        if stage.get("type") != "npt":
            continue
        production_stage = production_stage_from_equilibrium(
            cfg,
            stage,
            fixed_steps=fixed_steps,
            production_time_ps=production_time_ps,
        )
        if production_stage is not None:
            stages.append(production_stage)

    if not stages:
        print(
            "[production-config] No completed NPT equilibrium stages found; "
            "config_production.json was not written.",
            flush=True,
        )
        return None

    production_cfg = {
        "generated_by": "atomi md-engine",
        "source_config": _relative_to_root(Path(cfg["_config_path"])),
        "description": (
            "Generated from completed NPT equilibrium stages. Review HPC paths, "
            "model file, and production settings before submitting production MD."
        ),
        "wrapper_script": _relative_to_root(Path(cfg["wrapper_script"])),
        "model_file": _relative_to_root(Path(cfg["model_file"])),
        "timestep": cfg.get("timestep", timestep_ps),
        "mass_O": cfg["mass_O"],
        "mass_U": cfg["mass_U"],
        "velocity_seed": cfg.get("velocity_seed", 12345),
        "poll_seconds": cfg.get("poll_seconds", 10),
        "thermostat": cfg.get("thermostat", {}),
        "barostat": cfg.get("barostat", {}),
        "relax": cfg.get("relax", {}),
        "adaptive_steps": {
            "initial_small": int(fixed_steps),
            "initial_large": int(fixed_steps),
            "growth_factor": 1.0,
            "max_chunk_steps": int(fixed_steps),
        },
        "max_chunks_small": 1,
        "max_chunks_large": 1,
        "performance": cfg.get("performance", {}),
        "equilibrium_rules": cfg.get("equilibrium_rules", {}),
        "stages": stages,
        "production_settings": {
            "timestep_ps": timestep_ps,
            "production_time_ps": production_time_ps,
            "fixed_steps": int(fixed_steps),
            "source": "generated_from_completed_npt_equilibration",
            "assumption": "Each production stage starts from the matching completed NPT equilibrium restart/data.",
        },
        "instability_rules": cfg.get("instability_rules", {}),
    }

    output = _resolve_project_path(output_path)
    output.write_text(json.dumps(production_cfg, indent=2) + "\n")
    print(
        f"[production-config] Wrote {output} from {len(stages)} completed NPT equilibrium stages.",
        flush=True,
    )
    return output


def _temperature_label(value):
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(value).replace(".", "p")


def _relative_to_root(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------
# RUN STAGE
# ---------------------------------------------------

def run_stage(cfg, stage, structure, resume_mode=False):
    stage_name = stage["name"]
    stage_dir = STAGES / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    steps_cfg = cfg["adaptive_steps"]

    initial_steps = steps_cfg["initial_small"]
    max_chunks = cfg["max_chunks_small"]

    if stage.get("large_cell", False):
        initial_steps = steps_cfg["initial_large"]
        max_chunks = cfg["max_chunks_large"]

    fixed_step_stage = stage_uses_fixed_steps(stage)
    constant_chunk_steps = fixed_step_stage or stage_uses_constant_chunk_steps(stage)
    if fixed_step_stage:
        initial_steps = resolve_run_steps(cfg, stage, default_steps=initial_steps)
    if "max_chunks" in stage:
        max_chunks = int(stage["max_chunks"])
    elif fixed_step_stage:
        max_chunks = 1

    start_chunk, resumed_restart = find_latest_chunk_restart(
        stage_dir,
        stage_name,
        chunk_name=stage.get("chunk_name"),
    )

    if resumed_restart is not None:
        current_structure = resumed_restart
        print(f"Resuming stage {stage_name} from {resumed_restart} (starting chunk {start_chunk})", flush=True)
    elif "input_structure" in stage:
        current_structure = _resolve_stage_input(stage)
    else:
        current_structure = Path(structure).resolve()

    run_steps = initial_steps
    if not constant_chunk_steps:
        for _ in range(1, start_chunk):
            run_steps = int(run_steps * steps_cfg["growth_factor"])
            run_steps = min(run_steps, steps_cfg["max_chunk_steps"])

    min_chunks_before_accept = stage.get("min_chunks_before_accept", 1)
    accept_if_stable = stage.get("accept_if_stable", False)

    for chunk in range(start_chunk, max_chunks + 1):
        chunk_dir = chunk_dir_for_index(stage_dir, stage, chunk)
        chunk_dir.mkdir(exist_ok=True)

        input_name = f"in.{stage_name}_c{chunk:02d}"
        inputfile = chunk_dir / input_name
        ramp_Tstart = ramp_start_temperature_for_chunk(cfg, stage, stage_dir, stage_name, chunk)

        input_text, final_data_name, final_restart_name = generate_input(
            cfg,
            stage,
            run_steps,
            current_structure,
            stage_name,
            chunk,
            resume_mode=resume_mode,
            temperature_start_override=ramp_Tstart,
        )
        inputfile.write_text(input_text)

        wall_hours = estimate_walltime(cfg, stage, run_steps)
        walltime = hours_to_slurm(wall_hours)
        wrapper = create_stage_wrapper(cfg, chunk_dir, walltime)

        print(f"  chunk {chunk}: steps={run_steps} walltime={walltime}", flush=True)

        job = submit_job(wrapper, inputfile.name, chunk_dir)
        wait_job(job, cfg["poll_seconds"])

        check_slurm_outputs(chunk_dir)

        log = chunk_dir / f"log.{inputfile.name}"
        if not log.exists():
            raise RuntimeError(f"LAMMPS log file not found: {log}")

        if stage["type"] in ("relax", "replicate"):
            final_data_path = (chunk_dir / final_data_name).resolve()
            if not final_data_path.exists():
                raise RuntimeError(f"Expected output data file not found: {final_data_path}")

            write_stage_outputs(
                stage_dir,
                stage_name,
                final_data_path,
                final_restart_path=None,
                pass_note="stage complete"
            )
            return (stage_dir / f"{stage_name}.data").resolve()

        steps, T, P, V, PE = parse_thermo(log)

        if len(T) == 0:
            raise RuntimeError(f"LAMMPS produced no thermo output in {log}")

        plot_thermo(chunk_dir, steps, T, P, V, PE)
        write_chunk_summary(chunk_dir, stage, steps, T, P, V, PE, note=f"chunk {chunk}")

        final_data_path = (chunk_dir / final_data_name).resolve()
        final_restart_path = (chunk_dir / final_restart_name).resolve()

        if not final_data_path.exists():
            raise RuntimeError(f"Expected output data file not found: {final_data_path}")
        if not final_restart_path.exists():
            raise RuntimeError(f"Expected output restart file not found: {final_restart_path}")

        if stage_force_pass_requested(stage_dir):
            write_decision(
                chunk_dir,
                [
                    f"stage: {stage_name}",
                    f"chunk: {chunk}",
                    "decision: PASS",
                    "reason: force_pass requested",
                ],
            )
            write_stage_outputs(
                stage_dir,
                stage_name,
                final_data_path,
                final_restart_path=final_restart_path,
                pass_note=f"force_pass at chunk {chunk}"
            )
            return (stage_dir / f"{stage_name}.restart").resolve()

        decision_lines = [
            f"stage: {stage_name}",
            f"chunk: {chunk}",
            f"accept_if_stable: {accept_if_stable}",
            f"min_chunks_before_accept: {min_chunks_before_accept}",
        ]

        ramp_ok, ramp_msg = ramp_target_reached(cfg, stage, steps, T)
        decision_lines.append(f"ramp_check: {ramp_msg}")
        if ramp_ok:
            write_decision(chunk_dir, decision_lines + ["decision: PASS", f"reason: {ramp_msg}"])
            write_stage_outputs(
                stage_dir,
                stage_name,
                final_data_path,
                final_restart_path=final_restart_path,
                pass_note=f"ramp accepted at chunk {chunk}: {ramp_msg}"
            )
            return (stage_dir / f"{stage_name}.restart").resolve()

        strict_ok, strict_msg = check_equilibrium(cfg, stage, steps, T, V, PE)
        decision_lines.append(f"strict_check: {strict_msg}")
        if strict_ok:
            write_decision(chunk_dir, decision_lines + ["decision: PASS", f"reason: {strict_msg}"])
            write_stage_outputs(
                stage_dir,
                stage_name,
                final_data_path,
                final_restart_path=final_restart_path,
                pass_note=f"strict equilibrium at chunk {chunk}: {strict_msg}"
            )
            return (stage_dir / f"{stage_name}.restart").resolve()

        if accept_if_stable and chunk >= min_chunks_before_accept:
            stable_ok, stable_msg = check_stable_enough(cfg, stage, steps, T, V, PE)
            decision_lines.append(f"stable_check: {stable_msg}")
            if stable_ok:
                write_decision(chunk_dir, decision_lines + ["decision: PASS", f"reason: {stable_msg}"])
                write_stage_outputs(
                    stage_dir,
                    stage_name,
                    final_data_path,
                    final_restart_path=final_restart_path,
                    pass_note=f"accepted stable enough at chunk {chunk}: {stable_msg}"
                )
                return (stage_dir / f"{stage_name}.restart").resolve()
        elif accept_if_stable:
            decision_lines.append(f"stable_check: skipped until chunk >= {min_chunks_before_accept}")
        else:
            decision_lines.append("stable_check: disabled")

        decision_lines.append("decision: CONTINUE")
        write_decision(chunk_dir, decision_lines)
        print("  decision: " + "; ".join(decision_lines[4:]), flush=True)
        current_structure = final_restart_path
        if not constant_chunk_steps:
            run_steps = int(run_steps * steps_cfg["growth_factor"])
            run_steps = min(run_steps, steps_cfg["max_chunk_steps"])

    raise RuntimeError(f"stage {stage_name} did not converge")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog="md-engine")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--only", default=None, help="Run only one stage by name.")
    parser.add_argument(
        "--production-config-out",
        default="config_production.json",
        help="Where to write generated production config after a regular equilibration workflow.",
    )
    parser.add_argument(
        "--no-write-production-config",
        action="store_true",
        help="Do not auto-generate config_production.json after equilibration.",
    )
    parser.add_argument(
        "--production-steps",
        type=int,
        default=None,
        help="Override generated production fixed_steps.",
    )
    parser.add_argument(
        "--production-time-ps",
        type=float,
        default=None,
        help="Production time used when deriving fixed_steps from timestep.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workflow project directory containing config, stages, models, and structures.",
    )
    
    args = parser.parse_args(argv)
    set_project_root(args.root)

    cfg = load_config(args.config)
    print(f"Using config: {cfg['_config_path']}", flush=True)

    STAGES.mkdir(exist_ok=True)
    ANALYSIS.mkdir(exist_ok=True)

    if args.only:
        selected = [stage for stage in cfg["stages"] if stage["name"] == args.only]
        if not selected:
            close = get_close_matches(args.only, [stage["name"] for stage in cfg["stages"]], n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            parser.error(f"No stage named {args.only}.{hint}")
        cfg["stages"] = selected

    stage_names = [stage["name"] for stage in cfg["stages"]]
    if args.start_from is not None and args.start_from not in stage_names:
        close = get_close_matches(args.start_from, stage_names, n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        parser.error(f"No stage named {args.start_from}.{hint}")

    production_only = cfg["stages"] and all(production_stage_selected(stage) for stage in cfg["stages"])

    structure = Path(cfg["initial_structure"]).resolve() if "initial_structure" in cfg else None
    started = args.start_from is None

    for stage in cfg["stages"]:
        stage_name = stage["name"]
        stage_dir = STAGES / stage_name

        # Before start-from is reached, keep advancing "structure"
        # through passed stages so a later nvt_eqm/nvt_relax/npt can inherit
        # the correct previous artifact.
        if not started:
            if not production_stage_selected(stage):
                artifact = stage_artifact(stage_dir, stage_name)
                if artifact is not None and (stage_dir / "PASS").exists():
                    structure = artifact

            if stage_name == args.start_from:
                started = True
            else:
                continue

        if production_stage_selected(stage):
            run_production_stage(cfg, stage, resume_mode=args.resume)
            continue

        if args.resume and (stage_dir / "PASS").exists():
            artifact = stage_artifact(stage_dir, stage_name)
            if artifact is not None:
                structure = artifact

            print(f"Skipping {stage_name}", flush=True)
            continue

        if stage_prefers_restart(stage):
            p = Path(structure).resolve()
            is_restart = (p.suffix.lower() == ".restart") or p.name.lower().startswith("restart.")
            if not is_restart:
                print(
                    f"[warning] {stage_name} prefers restart input, but current structure is {p.name}.",
                    flush=True
                )

        print(f"Running {stage_name}", flush=True)
        if structure is None:
            raise RuntimeError(
                f"Stage {stage_name} needs an initial_structure or prior stage artifact."
            )
        structure = run_stage(cfg, stage, structure, resume_mode=args.resume)

    if production_only:
        print("All requested production stages completed.", flush=True)
    else:
        print("Workflow finished", flush=True)
        if not args.no_write_production_config:
            print(
                "[production-config] Regular equilibration workflow finished; "
                "generating production config from completed NPT equilibrium stages.",
                flush=True,
            )
            write_production_config_from_equilibration(
                cfg,
                output_path=Path(args.production_config_out),
                fixed_steps=args.production_steps,
                production_time_ps=args.production_time_ps,
            )


if __name__ == "__main__":
    main()
