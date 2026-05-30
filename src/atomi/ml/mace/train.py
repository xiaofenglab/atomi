"""Prepare and submit portable MACE training jobs on HPC systems."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from atomi.core.doctor import load_hpc_config


DEFAULT_PROFILE = "mace_training_gpu"
DEFAULT_TRAINING = {
    "epochs": 200,
    "batch_size": 16,
    "lr": 0.001,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "stress_weight": 10.0,
    "dtype": "float32",
    "device": "cuda",
    "seed": 7,
    "num_workers": 2,
    "model_name": "MACE",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "loss": "weighted",
    "ema_decay": 0.99,
}
DEFAULT_HIDDEN_IRREPS = {
    "yes": "128x0e + 128x1o + 128x2e",
    "no": "128x0e + 128x1o",
}


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _nonempty(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def _profile(config: dict[str, Any], name: str) -> dict[str, Any]:
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        return {}
    profile = profiles.get(name, {})
    return profile if isinstance(profile, dict) else {}


def _training_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(DEFAULT_TRAINING)
    raw = profile.get("default_training_parameters", {})
    if isinstance(raw, dict):
        for key, value in raw.items():
            if _nonempty(value):
                defaults[key] = value
    for old_key, new_key in (
        ("default_dtype", "dtype"),
        ("device", "device"),
        ("command", "command"),
    ):
        if _nonempty(profile.get(old_key)):
            defaults[new_key] = profile[old_key]
    return defaults


def _hidden_irreps(profile: dict[str, Any], use_2e: str) -> str:
    defaults = dict(DEFAULT_HIDDEN_IRREPS)
    raw = profile.get("hidden_irreps_defaults", {})
    if isinstance(raw, dict):
        if _nonempty(raw.get("use_2e_yes")):
            defaults["yes"] = str(raw["use_2e_yes"])
        if _nonempty(raw.get("use_2e_no")):
            defaults["no"] = str(raw["use_2e_no"])
    return defaults[use_2e]


def _arg_or_default(args: argparse.Namespace, name: str, defaults: dict[str, Any]) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return defaults[name]


def _sbatch_line(key: str, value: object) -> str | None:
    if not _nonempty(value):
        return None
    return f"#SBATCH --{key}={value}"


def _split_extra_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    for item in args.extra_arg or []:
        if item:
            extra.append(item)
    if args.extra_args:
        extra.extend(shlex.split(args.extra_args))
    return extra


def _render_modules(profile: dict[str, Any], cli_modules: list[str] | None) -> list[str]:
    lines: list[str] = []
    module_commands = profile.get("module_commands", [])
    if isinstance(module_commands, list):
        for command in module_commands:
            if _nonempty(command):
                lines.append(str(command))
    modules = list(cli_modules or [])
    profile_modules = profile.get("modules", [])
    if isinstance(profile_modules, list):
        modules.extend(str(module) for module in profile_modules if _nonempty(module))
    for module in modules:
        lines.append(f"module load {_q(module)}")
    return lines


def render_training_sbatch(args: argparse.Namespace, profile: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    defaults = _training_defaults(profile)
    use_2e = args.use_2e or "no"
    hidden_irreps = args.hidden_irreps or _hidden_irreps(profile, use_2e)
    mode = args.mode
    e0_mode = args.e0_mode or ("average" if mode == "new" else "estimated")
    command = args.command or str(profile.get("command") or "mace_run_train")
    env_path = args.env_path or str(profile.get("env_path") or os.environ.get("ATOMI_MACE_TRAIN_ENV", ""))
    activation_script = args.activation_script or ""
    if not activation_script and not env_path:
        activation_script = str(profile.get("env_activate_from_script") or "")

    epochs = _arg_or_default(args, "epochs", defaults)
    batch_size = _arg_or_default(args, "batch_size", defaults)
    lr = _arg_or_default(args, "lr", defaults)
    energy_weight = _arg_or_default(args, "energy_weight", defaults)
    forces_weight = _arg_or_default(args, "forces_weight", defaults)
    stress_weight = _arg_or_default(args, "stress_weight", defaults)
    dtype = _arg_or_default(args, "dtype", defaults)
    device = _arg_or_default(args, "device", defaults)
    seed = _arg_or_default(args, "seed", defaults)
    num_workers = _arg_or_default(args, "num_workers", defaults)
    model_name = _arg_or_default(args, "model_name", defaults)
    energy_key = _arg_or_default(args, "energy_key", defaults)
    forces_key = _arg_or_default(args, "forces_key", defaults)
    stress_key = _arg_or_default(args, "stress_key", defaults)
    loss = _arg_or_default(args, "loss", defaults)
    ema_decay = _arg_or_default(args, "ema_decay", defaults)
    extra_args = _split_extra_args(args)

    train_file = str(args.train_file)
    valid_file = str(args.valid_file)
    foundation_model = str(args.foundation_model or "")
    test_file = str(args.test_file or "")
    e0s = str(args.e0s or "")

    sbatch_values = {
        "job-name": args.job_name or profile.get("job_name") or f"mace-{args.run_name}",
        "partition": args.partition or profile.get("partition"),
        "gres": args.gres or profile.get("gres"),
        "nodes": args.nodes if args.nodes is not None else profile.get("nodes", 1),
        "ntasks": args.ntasks if args.ntasks is not None else profile.get("ntasks", 1),
        "cpus-per-task": args.cpus_per_task
        if args.cpus_per_task is not None
        else profile.get("cpus_per_task", 8),
        "mem": args.mem or profile.get("mem", "32G"),
        "time": args.time or profile.get("time", "16:00:00"),
        "output": args.output or profile.get("output", "logs/mace_%j.out"),
        "error": args.error or profile.get("error", "logs/mace_%j.err"),
        "account": args.account or profile.get("account"),
        "qos": args.qos or profile.get("qos"),
        "constraint": args.constraint or profile.get("constraint"),
    }

    header = ["#!/bin/bash"]
    for key, value in sbatch_values.items():
        line = _sbatch_line(key, value)
        if line is not None:
            header.append(line)

    module_lines = _render_modules(profile, args.module)
    environment = profile.get("environment", {})
    env_lines: list[str] = []
    if isinstance(environment, dict):
        for key, value in sorted(environment.items()):
            if _nonempty(value) and str(key) != "ATOMI_MACE_TRAIN_ENV":
                env_lines.append(f"export {key}={_q(value)}")

    cmd_items = [
        command,
        f"--name={args.run_name}",
        f"--train_file={train_file}",
        f"--valid_file={valid_file}",
        f"--energy_key={energy_key}",
        f"--forces_key={forces_key}",
        f"--stress_key={stress_key}",
        f"--model={model_name}",
        f"--batch_size={batch_size}",
        f"--max_num_epochs={epochs}",
        f"--num_workers={num_workers}",
        f"--loss={loss}",
        f"--energy_weight={energy_weight}",
        f"--forces_weight={forces_weight}",
        f"--stress_weight={stress_weight}",
        f"--default_dtype={dtype}",
        f"--device={device}",
        f"--lr={lr}",
        "--ema",
        f"--ema_decay={ema_decay}",
        "--amsgrad",
        f"--seed={seed}",
    ]
    if e0_mode in {"average", "estimated"}:
        cmd_items.append(f"--E0s={e0_mode}")
    elif e0_mode == "explicit":
        cmd_items.append(f"--E0s={e0s}")
    else:
        raise ValueError(f"Unsupported E0 mode: {e0_mode}")

    if mode == "new":
        cmd_items.append(f"--hidden_irreps={hidden_irreps}")
    elif mode == "retrain":
        cmd_items.append(f"--foundation_model={foundation_model}")
        cmd_items.append("--multiheads_finetuning=False")
    else:
        raise ValueError(f"Unsupported MACE mode: {mode}")
    if test_file:
        cmd_items.append(f"--test_file={test_file}")
    cmd_items.extend(extra_args)

    body: list[str] = [
        "",
        "set -euo pipefail",
        "",
        'cd "$SLURM_SUBMIT_DIR"',
        "mkdir -p logs checkpoints results",
    ]
    if module_lines:
        body.append("")
        body.extend(module_lines)
    if env_lines:
        body.append("")
        body.extend(env_lines)
    body.append("")
    if activation_script:
        if activation_script.startswith("~/"):
            body.append(f'source "${{HOME}}/{activation_script[2:]}"')
        else:
            body.append(f"source {_q(activation_script)}")
    elif env_path:
        body.append(f"source {_q(env_path)}/bin/activate")
    else:
        body.append('echo "WARNING: no MACE training environment was configured; using current PATH."')
    body.extend(
        [
            "",
            f"RUNNAME={_q(args.run_name)}",
            f"MODE={_q(mode)}",
            f"USE_2E={_q(use_2e)}",
            f"TRAIN_FILE={_q(train_file)}",
            f"VALID_FILE={_q(valid_file)}",
            f"FOUNDATION_MODEL={_q(foundation_model)}",
            f"TEST_FILE={_q(test_file)}",
            f"E0_MODE={_q(e0_mode)}",
            f"E0S={_q(e0s)}",
            f"HIDDEN_IRREPS={_q(hidden_irreps)}",
            "",
            'for required_file in "$TRAIN_FILE" "$VALID_FILE"; do',
            '  if [ ! -f "$required_file" ]; then',
            '    echo "ERROR: required file not found: $required_file"',
            "    exit 2",
            "  fi",
            "done",
            'if [ "$MODE" = "retrain" ]; then',
            '  if [ -z "$FOUNDATION_MODEL" ] || [ ! -f "$FOUNDATION_MODEL" ]; then',
            '    echo "ERROR: retrain mode requires an existing --foundation-model."',
            "    exit 2",
            "  fi",
            "fi",
            'if [ -n "$TEST_FILE" ] && [ ! -f "$TEST_FILE" ]; then',
            '  echo "ERROR: test file not found: $TEST_FILE"',
            "  exit 2",
            "fi",
            'if [ "$E0_MODE" = "explicit" ] && [ -z "$E0S" ]; then',
            '  echo "ERROR: --e0-mode explicit requires --e0s."',
            "  exit 2",
            "fi",
            "",
            'echo "=============================="',
            'echo "Atomi MACE training session"',
            'echo "Run name          : $RUNNAME"',
            'echo "Mode              : $MODE"',
            'echo "Use 2e            : $USE_2E"',
            'echo "Train file        : $TRAIN_FILE"',
            'echo "Validation file   : $VALID_FILE"',
            'echo "Foundation model  : ${FOUNDATION_MODEL:-none}"',
            'echo "Test file         : ${TEST_FILE:-none}"',
            'echo "E0 mode           : $E0_MODE"',
            'echo "Hidden irreps     : $HIDDEN_IRREPS"',
            f'echo "Epochs            : {epochs}"',
            f'echo "Batch size        : {batch_size}"',
            f'echo "LR                : {lr}"',
            f'echo "Weights E/F/S     : {energy_weight} / {forces_weight} / {stress_weight}"',
            f'echo "Dtype/device      : {dtype} / {device}"',
            'echo "=============================="',
            "nvidia-smi || true",
            "python - <<'PY'",
            "import torch",
            'print("torch", torch.__version__)',
            'print("cuda available", torch.cuda.is_available())',
            "if torch.cuda.is_available():",
            '    print("gpu", torch.cuda.get_device_name(0))',
            "PY",
            "",
            "CMD=(",
        ]
    )
    body.extend(f"  {_q(item)}" for item in cmd_items)
    body.extend(
        [
            ")",
            "",
            'echo "Command:"',
            'printf " %q" "${CMD[@]}"',
            "echo",
            '"${CMD[@]}"',
            "",
            'echo "MACE training finished"',
            "python - <<'PY' || true",
            "from pathlib import Path",
            "import json, math, os",
            "run = os.environ.get('RUNNAME_FOR_SUMMARY') or " + repr(args.run_name),
            "candidates = sorted(Path('results').glob(f'{run}*train*.txt')) + sorted(Path('logs').glob(f'{run}*train*.txt'))",
            "if not candidates:",
            "    print('WARNING: no MACE train log found for automatic diagnostics.')",
            "    raise SystemExit",
            "rows = []",
            "for line in candidates[0].read_text(errors='replace').splitlines():",
            "    try:",
            "        row = json.loads(line)",
            "    except Exception:",
            "        continue",
            "    if row.get('mode') == 'eval':",
            "        rows.append(row)",
            "if not rows:",
            "    print(f'WARNING: no eval rows found in {candidates[0]}')",
            "    raise SystemExit",
            "last = rows[-1]",
            "summary = [f'log: {candidates[0]}', f'epoch: {last.get(\"epoch\")}', "
            "f'rmse_f: {last.get(\"rmse_f\")}', f'q95_f: {last.get(\"q95_f\")}', "
            "f'rmse_e_per_atom: {last.get(\"rmse_e_per_atom\")}']",
            "out = Path('logs') / f'{run}_diagnostics_summary.txt'",
            "out.write_text('\\n'.join(summary) + '\\n')",
            "print('\\n'.join(summary))",
            "print(f'wrote {out}')",
            "PY",
            "",
        ]
    )
    script = "\n".join(header + body)

    plan = {
        "run_name": args.run_name,
        "mode": mode,
        "use_2e": use_2e,
        "train_file": train_file,
        "valid_file": valid_file,
        "foundation_model": foundation_model,
        "test_file": test_file,
        "e0_mode": e0_mode,
        "e0s": e0s if e0_mode == "explicit" else "",
        "command": command,
        "command_preview": " ".join(shlex.quote(item) for item in cmd_items),
        "env_path": env_path,
        "activation_script": activation_script,
        "sbatch": sbatch_values,
        "training_parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "stress_weight": stress_weight,
            "dtype": dtype,
            "device": device,
            "seed": seed,
            "num_workers": num_workers,
            "model_name": model_name,
            "energy_key": energy_key,
            "forces_key": forces_key,
            "stress_key": stress_key,
            "loss": loss,
            "ema_decay": ema_decay,
            "hidden_irreps": hidden_irreps,
            "extra_args": extra_args,
        },
    }
    return script, plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mace-train",
        description="Write or submit a Slurm MACE training/retraining job from an Atomi HPC profile.",
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "submit"),
        default="prepare",
        help="Write the sbatch script only, or write and submit it.",
    )
    parser.add_argument("--hpc-config", type=Path, default=None, help="Atomi local HPC JSON.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="HPC profile name.")
    parser.add_argument("--outdir", type=Path, default=Path("."), help="Directory for script and plan.")
    parser.add_argument("--script-out", type=Path, default=None, help="Output sbatch script path.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("new", "retrain"), default="new")
    use_2e = parser.add_mutually_exclusive_group()
    use_2e.add_argument("--use-2e", choices=("yes", "no"), default=None)
    use_2e.add_argument("--with-2e", dest="use_2e", action="store_const", const="yes")
    use_2e.add_argument("--no-2e", dest="use_2e", action="store_const", const="no")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file", required=True)
    parser.add_argument("--foundation-model", default="")
    parser.add_argument("--test-file", default="")
    parser.add_argument("--e0-mode", choices=("average", "estimated", "explicit"), default=None)
    parser.add_argument("--e0s", default="", help="Explicit E0 dictionary string, e.g. '{8: -1.0, 92: -2.0}'.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--forces-weight", type=float, default=None)
    parser.add_argument("--stress-weight", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--energy-key", default=None)
    parser.add_argument("--forces-key", default=None)
    parser.add_argument("--stress-key", default=None)
    parser.add_argument("--loss", default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--hidden-irreps", default=None)
    parser.add_argument("--extra-arg", action="append", default=[], help="Append one raw MACE CLI argument.")
    parser.add_argument("--extra-args", default="", help="Raw extra MACE CLI arguments parsed with shlex.")
    parser.add_argument("--command", default=None, help="Training executable, default mace_run_train.")
    parser.add_argument("--env-path", default=None, help="Python env prefix to activate.")
    parser.add_argument("--activation-script", default=None, help="Explicit activation script to source.")
    parser.add_argument("--module", action="append", default=[], help="Extra module to load before training.")
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--gres", default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--ntasks", type=int, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem", default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--error", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--constraint", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_hpc_config(args.hpc_config)
    profile = _profile(config, args.profile)

    if args.mode == "retrain" and not args.foundation_model:
        parser.error("--mode retrain requires --foundation-model")
    if (args.e0_mode == "explicit") and not args.e0s:
        parser.error("--e0-mode explicit requires --e0s")

    script, plan = render_training_sbatch(args, profile)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    script_path = args.script_out or outdir / f"train_{args.run_name}.sbatch"
    if not script_path.is_absolute():
        script_path = Path.cwd() / script_path
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | 0o111)

    plan_path = script_path.with_suffix(".plan.json")
    plan["script"] = str(script_path)
    plan["profile"] = args.profile
    plan["hpc_config"] = str(args.hpc_config) if args.hpc_config else ""
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote MACE training script: {script_path}")
    print(f"Wrote MACE training plan: {plan_path}")
    print("Submit with:")
    print(f"  sbatch {script_path}")

    if args.action == "submit":
        result = subprocess.run(
            ["sbatch", str(script_path)],
            cwd=script_path.parent,
            check=True,
            text=True,
            capture_output=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())


def submit_main(argv: list[str] | None = None) -> None:
    args = list(argv or [])
    if not args or args[0] != "submit":
        args.insert(0, "submit")
    main(args)


if __name__ == "__main__":
    main()
