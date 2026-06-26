"""Bader charge workflow helpers for VASP outputs.

The module intentionally separates three concerns:

* checking whether a VASP folder has charge-density files suitable for Bader,
* writing/running a portable Bader script when the external binaries exist,
* parsing ``ACF.dat`` into CSV/JSON summaries with species and ZVAL context.

Bader itself is not bundled with Atomi.  This bridge records missing binaries
clearly so HPC workflows can be prepared before the site-local executable is
installed or loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "atomi.vasp.bader_charge.v1"


@dataclass
class BaderFileStatus:
    run_dir: str
    chgcar: str
    aeccar0: str
    aeccar2: str
    acf: str
    poscar: str
    potcar: str
    has_chgcar: bool
    has_aeccar0: bool
    has_aeccar2: bool
    has_acf: bool
    has_poscar: bool
    has_potcar: bool
    bader_executable: str | None
    chgsum_executable: str | None
    ready_for_bader: bool
    ready_for_reference_bader: bool
    ready_for_parse: bool
    warnings: list[str]


@dataclass
class BaderAtomCharge:
    index: int
    species: str
    x: float
    y: float
    z: float
    bader_electrons: float
    zval: float | None = None
    net_charge: float | None = None
    min_dist: float | None = None
    atomic_volume: float | None = None


def _is_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _resolve_run_file(run_dir: Path, path: Path | None, default_name: str) -> Path:
    if path is None:
        return run_dir / default_name
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return run_dir / expanded


def _which(command: str | None) -> str | None:
    if not command:
        return None
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") or os.sep in command:
        return str(candidate) if candidate.exists() else None
    return shutil.which(command)


def read_vasp_species(poscar: Path) -> list[str]:
    """Return per-site species labels from a VASP 5 style POSCAR/CONTCAR."""

    lines = poscar.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(f"{poscar} is too short to be a POSCAR/CONTCAR")
    maybe_species = lines[5].split()
    maybe_counts = lines[6].split()
    if maybe_species and all(not _looks_like_number(token) for token in maybe_species):
        species = maybe_species
        counts_tokens = maybe_counts
    else:
        raise ValueError(
            f"{poscar} does not look like a VASP 5 POSCAR with species on line 6; "
            "pass --species or use a VASP 5 CONTCAR/POSCAR."
        )
    counts = [int(float(token)) for token in counts_tokens]
    if len(species) != len(counts):
        raise ValueError(f"Species/count mismatch in {poscar}: {species!r} vs {counts!r}")
    labels: list[str] = []
    for symbol, count in zip(species, counts):
        labels.extend([symbol] * count)
    return labels


def _looks_like_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def parse_species_override(values: list[str]) -> list[str]:
    labels: list[str] = []
    for value in values:
        for token in re.split(r"[\s,]+", value.strip()):
            if not token:
                continue
            if ":" in token:
                symbol, count_text = token.split(":", 1)
                labels.extend([symbol] * int(float(count_text)))
            else:
                labels.append(token)
    return labels


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def parse_potcar_zvals(potcar: Path, species_order: list[str]) -> dict[str, float]:
    text = potcar.read_text(encoding="utf-8", errors="replace")
    zvals = [float(match.group(1)) for match in re.finditer(r"ZVAL\s*=\s*([-+0-9.]+)", text)]
    ordered_species = unique_in_order(species_order)
    if not zvals:
        return {}
    if len(zvals) < len(ordered_species):
        raise ValueError(
            f"{potcar} has {len(zvals)} ZVAL entries but POSCAR has {len(ordered_species)} species."
        )
    return {symbol: zvals[i] for i, symbol in enumerate(ordered_species)}


def parse_zval_overrides(values: list[str]) -> dict[str, float]:
    zvals: dict[str, float] = {}
    for value in values:
        for token in re.split(r"[\s,]+", value.strip()):
            if not token:
                continue
            if "=" not in token:
                raise ValueError(f"ZVAL override must look like U=14, got {token!r}")
            symbol, number = token.split("=", 1)
            zvals[symbol] = float(number)
    return zvals


def inspect_run(
    run_dir: Path,
    *,
    chgcar: Path | None = None,
    aeccar0: Path | None = None,
    aeccar2: Path | None = None,
    acf: Path | None = None,
    poscar: Path | None = None,
    potcar: Path | None = None,
    bader_cmd: str = "bader",
    chgsum_cmd: str = "chgsum.pl",
) -> BaderFileStatus:
    run_dir = run_dir.expanduser().resolve()
    chgcar_path = _resolve_run_file(run_dir, chgcar, "CHGCAR")
    aeccar0_path = _resolve_run_file(run_dir, aeccar0, "AECCAR0")
    aeccar2_path = _resolve_run_file(run_dir, aeccar2, "AECCAR2")
    acf_path = _resolve_run_file(run_dir, acf, "ACF.dat")
    poscar_path = _resolve_run_file(run_dir, poscar, "CONTCAR")
    if not poscar_path.exists() and poscar is None:
        poscar_path = run_dir / "POSCAR"
    potcar_path = _resolve_run_file(run_dir, potcar, "POTCAR")

    warnings: list[str] = []
    has_chgcar = _is_nonempty(chgcar_path)
    has_aeccar0 = _is_nonempty(aeccar0_path)
    has_aeccar2 = _is_nonempty(aeccar2_path)
    has_acf = _is_nonempty(acf_path)
    has_poscar = _is_nonempty(poscar_path)
    has_potcar = _is_nonempty(potcar_path)
    bader_exe = _which(bader_cmd)
    chgsum_exe = _which(chgsum_cmd)

    if not has_chgcar:
        warnings.append("CHGCAR is missing or empty; Bader cannot run.")
    if has_chgcar and not (has_aeccar0 and has_aeccar2):
        warnings.append("AECCAR0/AECCAR2 are incomplete; run will fall back to CHGCAR-only Bader.")
    if (has_aeccar0 or has_aeccar2) and not chgsum_exe:
        warnings.append("chgsum.pl is not on PATH; cannot build AECCAR0+2 reference automatically.")
    if has_chgcar and not bader_exe:
        warnings.append("bader executable is not on PATH; use setup script after loading/installing Bader.")
    if has_acf and not has_poscar:
        warnings.append("ACF.dat exists but POSCAR/CONTCAR is missing; pass --species to parse.")

    return BaderFileStatus(
        run_dir=str(run_dir),
        chgcar=str(chgcar_path),
        aeccar0=str(aeccar0_path),
        aeccar2=str(aeccar2_path),
        acf=str(acf_path),
        poscar=str(poscar_path),
        potcar=str(potcar_path),
        has_chgcar=has_chgcar,
        has_aeccar0=has_aeccar0,
        has_aeccar2=has_aeccar2,
        has_acf=has_acf,
        has_poscar=has_poscar,
        has_potcar=has_potcar,
        bader_executable=bader_exe,
        chgsum_executable=chgsum_exe,
        ready_for_bader=has_chgcar and bader_exe is not None,
        ready_for_reference_bader=has_chgcar and has_aeccar0 and has_aeccar2 and bader_exe is not None and chgsum_exe is not None,
        ready_for_parse=has_acf,
        warnings=warnings,
    )


def write_bader_script(
    run_dir: Path,
    *,
    script_name: str = "run_bader.sh",
    bader_cmd: str = "bader",
    chgsum_cmd: str = "chgsum.pl",
    chgcar_name: str = "CHGCAR",
    aeccar0_name: str = "AECCAR0",
    aeccar2_name: str = "AECCAR2",
    use_reference: bool = True,
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    script = run_dir / script_name
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        "",
        f'BADER_CMD="${{BADER_CMD:-{bader_cmd}}}"',
        f'CHGSUM_CMD="${{CHGSUM_CMD:-{chgsum_cmd}}}"',
        f'CHGCAR_FILE="${{CHGCAR_FILE:-{chgcar_name}}}"',
        f'AECCAR0_FILE="${{AECCAR0_FILE:-{aeccar0_name}}}"',
        f'AECCAR2_FILE="${{AECCAR2_FILE:-{aeccar2_name}}}"',
        "",
        'if [ ! -s "$CHGCAR_FILE" ]; then',
        '  echo "ERROR: $CHGCAR_FILE is missing or empty." >&2',
        "  exit 2",
        "fi",
        'if ! command -v "$BADER_CMD" >/dev/null 2>&1; then',
        '  echo "ERROR: bader executable not found: $BADER_CMD" >&2',
        "  exit 3",
        "fi",
    ]
    if use_reference:
        lines.extend(
            [
                'if [ -s "$AECCAR0_FILE" ] && [ -s "$AECCAR2_FILE" ] && command -v "$CHGSUM_CMD" >/dev/null 2>&1; then',
                '  "$CHGSUM_CMD" "$AECCAR0_FILE" "$AECCAR2_FILE" | tee chgsum.log',
                '  "$BADER_CMD" "$CHGCAR_FILE" -ref AECCAR0+2 | tee bader.log',
                "else",
                '  echo "WARNING: AECCAR reference or chgsum.pl unavailable; running CHGCAR-only Bader." >&2',
                '  "$BADER_CMD" "$CHGCAR_FILE" | tee bader.log',
                "fi",
            ]
        )
    else:
        lines.append('"$BADER_CMD" "$CHGCAR_FILE" | tee bader.log')
    lines.append("")
    script.write_text("\n".join(lines), encoding="utf-8")
    script.chmod(0o755)
    return script


def write_incar_fragment(run_dir: Path, filename: str = "INCAR.bader_fragment") -> Path:
    path = run_dir.expanduser().resolve() / filename
    path.write_text(
        "\n".join(
            [
                "# Add these tags to the static VASP run used for Bader charge analysis.",
                "LCHARG = .TRUE.",
                "LAECHG = .TRUE.",
                "LVTOT = .FALSE.",
                "LWAVE = .FALSE.",
                "LORBIT = 11",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def run_bader_command(
    run_dir: Path,
    *,
    bader_cmd: str = "bader",
    chgsum_cmd: str = "chgsum.pl",
    skip_chgsum: bool = False,
    allow_missing: bool = False,
) -> dict[str, Any]:
    status = inspect_run(run_dir, bader_cmd=bader_cmd, chgsum_cmd=chgsum_cmd)
    run_dir = Path(status.run_dir)
    if not status.has_chgcar:
        if allow_missing:
            return {
                "schema": SCHEMA,
                "stage": "run",
                "status": "MISSING_CHGCAR",
                "message": f"Missing CHGCAR in {run_dir}.",
            }
        raise FileNotFoundError(f"Missing CHGCAR in {run_dir}")
    if status.bader_executable is None:
        message = f"Missing bader executable {bader_cmd!r}."
        if allow_missing:
            return {"schema": SCHEMA, "stage": "run", "status": "MISSING_BADER", "message": message}
        raise FileNotFoundError(message)

    commands: list[list[str]] = []
    reference_ready = status.has_aeccar0 and status.has_aeccar2 and status.chgsum_executable is not None and not skip_chgsum
    if reference_ready:
        commands.append([status.chgsum_executable or chgsum_cmd, "AECCAR0", "AECCAR2"])
        commands.append([status.bader_executable or bader_cmd, "CHGCAR", "-ref", "AECCAR0+2"])
    else:
        commands.append([status.bader_executable or bader_cmd, "CHGCAR"])

    logs: list[dict[str, Any]] = []
    for command in commands:
        proc = subprocess.run(command, cwd=run_dir, text=True, capture_output=True, check=False)
        log_name = "chgsum.log" if "chgsum" in Path(command[0]).name else "bader.log"
        (run_dir / log_name).write_text(proc.stdout + proc.stderr, encoding="utf-8")
        logs.append({"command": command, "returncode": proc.returncode, "log": str(run_dir / log_name)})
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(command)}")
    return {"schema": SCHEMA, "stage": "run", "status": "OK", "reference_ready": reference_ready, "logs": logs}


def parse_acf(
    acf: Path,
    *,
    species: list[str],
    zvals: dict[str, float] | None = None,
) -> list[BaderAtomCharge]:
    zvals = zvals or {}
    rows: list[BaderAtomCharge] = []
    for raw in acf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) < 5:
            continue
        index = int(parts[0])
        if index < 1 or index > len(species):
            symbol = "X"
        else:
            symbol = species[index - 1]
        bader_electrons = float(parts[4])
        zval = zvals.get(symbol)
        net_charge = zval - bader_electrons if zval is not None else None
        rows.append(
            BaderAtomCharge(
                index=index,
                species=symbol,
                x=float(parts[1]),
                y=float(parts[2]),
                z=float(parts[3]),
                bader_electrons=bader_electrons,
                zval=zval,
                net_charge=net_charge,
                min_dist=float(parts[5]) if len(parts) > 5 and _looks_like_number(parts[5]) else None,
                atomic_volume=float(parts[6]) if len(parts) > 6 and _looks_like_number(parts[6]) else None,
            )
        )
    if not rows:
        raise ValueError(f"No atom rows parsed from {acf}")
    return rows


def summarize_rows(rows: list[BaderAtomCharge]) -> list[dict[str, Any]]:
    grouped: dict[str, list[BaderAtomCharge]] = {}
    for row in rows:
        grouped.setdefault(row.species, []).append(row)
    summary: list[dict[str, Any]] = []
    for species, group in sorted(grouped.items()):
        electrons = [row.bader_electrons for row in group]
        charges = [row.net_charge for row in group if row.net_charge is not None]
        entry: dict[str, Any] = {
            "species": species,
            "count": len(group),
            "bader_electrons_mean": mean(electrons),
            "bader_electrons_min": min(electrons),
            "bader_electrons_max": max(electrons),
            "bader_electrons_std": std(electrons),
        }
        if charges:
            entry.update(
                {
                    "net_charge_mean": mean(charges),
                    "net_charge_min": min(charges),
                    "net_charge_max": max(charges),
                    "net_charge_std": std(charges),
                    "net_charge_sum": sum(charges),
                }
            )
        summary.append(entry)
    return summary


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def write_atom_csv(path: Path, rows: list[BaderAtomCharge]) -> None:
    fields = [
        "index",
        "species",
        "x",
        "y",
        "z",
        "bader_electrons",
        "zval",
        "net_charge",
        "min_dist",
        "atomic_volume",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "species",
        "count",
        "bader_electrons_mean",
        "bader_electrons_min",
        "bader_electrons_max",
        "bader_electrons_std",
        "net_charge_mean",
        "net_charge_min",
        "net_charge_max",
        "net_charge_std",
        "net_charge_sum",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def status_command(args: argparse.Namespace) -> dict[str, Any]:
    status = inspect_run(
        args.run_dir,
        chgcar=args.chgcar,
        aeccar0=args.aeccar0,
        aeccar2=args.aeccar2,
        acf=args.acf,
        poscar=args.poscar,
        potcar=args.potcar,
        bader_cmd=args.bader_cmd,
        chgsum_cmd=args.chgsum_cmd,
    )
    payload = {"schema": SCHEMA, "stage": "status", "status": asdict(status)}
    _print_json(payload)
    return payload


def setup_command(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    script = write_bader_script(
        run_dir,
        script_name=args.script_name,
        bader_cmd=args.bader_cmd,
        chgsum_cmd=args.chgsum_cmd,
        use_reference=not args.no_reference,
    )
    fragment = write_incar_fragment(run_dir, args.incar_fragment)
    status = inspect_run(run_dir, bader_cmd=args.bader_cmd, chgsum_cmd=args.chgsum_cmd)
    payload = {
        "schema": SCHEMA,
        "stage": "setup",
        "script": str(script),
        "incar_fragment": str(fragment),
        "status": asdict(status),
    }
    _print_json(payload)
    return payload


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    payload = run_bader_command(
        args.run_dir,
        bader_cmd=args.bader_cmd,
        chgsum_cmd=args.chgsum_cmd,
        skip_chgsum=args.skip_chgsum,
        allow_missing=args.allow_missing,
    )
    _print_json(payload)
    return payload


def parse_command(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    acf = _resolve_run_file(run_dir, args.acf, "ACF.dat")
    poscar = _resolve_run_file(run_dir, args.poscar, "CONTCAR")
    if not poscar.exists() and args.poscar is None:
        poscar = run_dir / "POSCAR"
    potcar = _resolve_run_file(run_dir, args.potcar, "POTCAR")

    species = parse_species_override(args.species)
    if not species:
        species = read_vasp_species(poscar)

    zvals: dict[str, float] = {}
    if potcar.exists():
        zvals.update(parse_potcar_zvals(potcar, species))
    zvals.update(parse_zval_overrides(args.zval))

    rows = parse_acf(acf, species=species, zvals=zvals)
    summary = summarize_rows(rows)
    out_prefix = args.out_prefix
    atom_csv = run_dir / f"{out_prefix}_atoms.csv"
    summary_csv = run_dir / f"{out_prefix}_summary.csv"
    json_path = run_dir / f"{out_prefix}.json"
    write_atom_csv(atom_csv, rows)
    write_summary_csv(summary_csv, summary)
    payload = {
        "schema": SCHEMA,
        "stage": "parse",
        "run_dir": str(run_dir),
        "acf": str(acf),
        "poscar": str(poscar),
        "potcar": str(potcar) if potcar.exists() else "",
        "zvals": zvals,
        "atom_csv": str(atom_csv),
        "summary_csv": str(summary_csv),
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["json"] = str(json_path)
    _print_json(payload)
    return payload


def all_command(args: argparse.Namespace) -> dict[str, Any]:
    setup = setup_command(args)
    run = run_bader_command(
        args.run_dir,
        bader_cmd=args.bader_cmd,
        chgsum_cmd=args.chgsum_cmd,
        skip_chgsum=args.skip_chgsum,
        allow_missing=args.allow_missing,
    )
    parse_payload: dict[str, Any] | None = None
    acf_path = args.run_dir.expanduser().resolve() / "ACF.dat"
    if acf_path.exists():
        parse_payload = parse_command(args)
    payload = {"schema": SCHEMA, "stage": "all", "setup": setup, "run": run, "parse": parse_payload}
    _print_json(payload)
    return payload


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, default=Path("."), help="VASP run folder.")
    parser.add_argument("--chgcar", type=Path, help="CHGCAR path relative to run-dir or absolute.")
    parser.add_argument("--aeccar0", type=Path, help="AECCAR0 path relative to run-dir or absolute.")
    parser.add_argument("--aeccar2", type=Path, help="AECCAR2 path relative to run-dir or absolute.")
    parser.add_argument("--acf", type=Path, help="ACF.dat path relative to run-dir or absolute.")
    parser.add_argument("--poscar", type=Path, help="POSCAR/CONTCAR path relative to run-dir or absolute.")
    parser.add_argument("--potcar", type=Path, help="POTCAR path relative to run-dir or absolute.")


def add_binary_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bader-cmd", default="bader", help="Bader executable name or path.")
    parser.add_argument("--chgsum-cmd", default="chgsum.pl", help="chgsum.pl executable name or path.")


def add_parse_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--species", action="append", default=[], help="Override species labels, e.g. U:8,C:16 or U U C C.")
    parser.add_argument("--zval", action="append", default=[], help="Override ZVAL map, e.g. U=14,C=4.")
    parser.add_argument("--out-prefix", default="bader_charge", help="Output prefix for CSV/JSON files.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-bader-charge",
        description="Prepare, run, and parse Bader charge analysis for VASP folders.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Inspect Bader inputs and available executables.")
    add_common_paths(status)
    add_binary_options(status)
    status.set_defaults(func=status_command)

    setup = sub.add_parser("setup", help="Write run_bader.sh and an INCAR fragment.")
    add_common_paths(setup)
    add_binary_options(setup)
    setup.add_argument("--script-name", default="run_bader.sh", help="Name of generated run script.")
    setup.add_argument("--incar-fragment", default="INCAR.bader_fragment", help="Name of generated INCAR fragment.")
    setup.add_argument("--no-reference", action="store_true", help="Do not use AECCAR0+2 reference density in script.")
    setup.set_defaults(func=setup_command)

    run = sub.add_parser("run", help="Run bader/chgsum.pl in the VASP folder.")
    add_common_paths(run)
    add_binary_options(run)
    run.add_argument("--skip-chgsum", action="store_true", help="Run Bader on CHGCAR only.")
    run.add_argument("--allow-missing", action="store_true", help="Return JSON status instead of failing when bader is missing.")
    run.set_defaults(func=run_command)

    parse = sub.add_parser("parse", help="Parse ACF.dat into atom and element summary tables.")
    add_common_paths(parse)
    add_parse_options(parse)
    parse.set_defaults(func=parse_command)

    allp = sub.add_parser("all", help="Setup, run Bader, and parse ACF.dat if produced.")
    add_common_paths(allp)
    add_binary_options(allp)
    add_parse_options(allp)
    allp.add_argument("--script-name", default="run_bader.sh", help="Name of generated run script.")
    allp.add_argument("--incar-fragment", default="INCAR.bader_fragment", help="Name of generated INCAR fragment.")
    allp.add_argument("--no-reference", action="store_true", help="Do not use AECCAR0+2 reference density in script.")
    allp.add_argument("--skip-chgsum", action="store_true", help="Run Bader on CHGCAR only.")
    allp.add_argument("--allow-missing", action="store_true", help="Return JSON status instead of failing when bader is missing.")
    allp.set_defaults(func=all_command)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
