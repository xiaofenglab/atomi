"""Prepare and analyze first-principles Hubbard-U workflows.

The module keeps three routes distinct:

* VASP PAW-projector linear response (``LDAUTYPE=3``),
* VASP Wannier-basis cRPA, and
* Quantum ESPRESSO HP / research Wannier-projector workflows.

The resulting U values are tagged by projector and response definition.  They
must not be silently transferred between those routes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


LR_TAGS = {
    "IBRION",
    "ICHARG",
    "ISYM",
    "LCHARG",
    "LDAU",
    "LDAUJ",
    "LDAUL",
    "LDAUPRINT",
    "LDAUTYPE",
    "LDAUU",
    "LMAXMIX",
    "LORBIT",
    "LWAVE",
    "NSW",
    "SYSTEM",
}


@dataclass(frozen=True)
class PoscarData:
    header: tuple[str, ...]
    symbols: tuple[str, ...]
    counts: tuple[int, ...]
    selective_line: str | None
    coordinate_line: str
    atoms: tuple[str, ...]
    tail: tuple[str, ...]

    @property
    def natoms(self) -> int:
        return sum(self.counts)


def _is_int_list(parts: Sequence[str]) -> bool:
    try:
        [int(part) for part in parts]
    except ValueError:
        return False
    return bool(parts)


def read_poscar(path: Path) -> PoscarData:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 8:
        raise ValueError(f"POSCAR is too short: {path}")
    if _is_int_list(lines[5].split()):
        raise ValueError("VASP4 POSCAR without an element-symbol line is not supported")
    symbols = tuple(lines[5].split())
    counts = tuple(int(value) for value in lines[6].split())
    if len(symbols) != len(counts):
        raise ValueError("POSCAR symbol/count lengths differ")
    cursor = 7
    selective = None
    if lines[cursor].strip().lower().startswith("s"):
        selective = lines[cursor]
        cursor += 1
    coordinate_line = lines[cursor]
    cursor += 1
    natoms = sum(counts)
    atoms = tuple(lines[cursor : cursor + natoms])
    if len(atoms) != natoms:
        raise ValueError(f"POSCAR expected {natoms} atom lines, found {len(atoms)}")
    return PoscarData(
        header=tuple(lines[:5]),
        symbols=symbols,
        counts=counts,
        selective_line=selective,
        coordinate_line=coordinate_line,
        atoms=atoms,
        tail=tuple(lines[cursor + natoms :]),
    )


def atom_species_indices(data: PoscarData) -> list[int]:
    indices: list[int] = []
    for species_index, count in enumerate(data.counts):
        indices.extend([species_index] * count)
    return indices


def split_probe_species(
    data: PoscarData,
    probe_atom: int,
    probe_label: str | None = None,
) -> tuple[PoscarData, list[int], int]:
    """Split one atom into a duplicate VASP species and move it to atom 1.

    Returns the new POSCAR, the new-to-old atom order (zero based), and the
    original species index that must be duplicated in POTCAR.
    """

    old_index = probe_atom - 1
    if old_index < 0 or old_index >= data.natoms:
        raise ValueError(f"--probe-atom must be between 1 and {data.natoms}")
    species_by_atom = atom_species_indices(data)
    probe_species = species_by_atom[old_index]
    symbol = data.symbols[probe_species]
    label = probe_label or f"{symbol}_probe"
    bulk_label = f"{symbol}_bulk"

    order = [old_index]
    new_symbols = [label]
    new_counts = [1]
    for species_index, (species, count) in enumerate(zip(data.symbols, data.counts)):
        members = [i for i, value in enumerate(species_by_atom) if value == species_index]
        if species_index == probe_species:
            members = [i for i in members if i != old_index]
            if members:
                new_symbols.append(bulk_label)
                new_counts.append(len(members))
                order.extend(members)
        else:
            new_symbols.append(species)
            new_counts.append(count)
            order.extend(members)
    if len(order) != data.natoms or len(set(order)) != data.natoms:
        raise RuntimeError("internal atom reordering error")
    split = PoscarData(
        header=data.header,
        symbols=tuple(new_symbols),
        counts=tuple(new_counts),
        selective_line=data.selective_line,
        coordinate_line=data.coordinate_line,
        atoms=tuple(data.atoms[index] for index in order),
        tail=data.tail,
    )
    return split, order, probe_species


def write_poscar(path: Path, data: PoscarData) -> None:
    lines = [*data.header, "  " + "  ".join(data.symbols), "  " + "  ".join(map(str, data.counts))]
    if data.selective_line is not None:
        lines.append(data.selective_line)
    lines.extend([data.coordinate_line, *data.atoms, *data.tail])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def potcar_datasets(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "End of Dataset"
    datasets: list[str] = []
    cursor = 0
    while True:
        end = text.find(marker, cursor)
        if end < 0:
            break
        end += len(marker)
        while end < len(text) and text[end] in "\r\n":
            end += 1
        datasets.append(text[cursor:end])
        cursor = end
    if cursor < len(text) and text[cursor:].strip():
        datasets.append(text[cursor:])
    if not datasets:
        raise ValueError(f"could not split POTCAR datasets: {path}")
    return datasets


def write_split_potcar(path: Path, source: Path, probe_species: int) -> None:
    datasets = potcar_datasets(source)
    if probe_species >= len(datasets):
        raise ValueError(
            f"POTCAR has {len(datasets)} datasets; POSCAR probe species index is {probe_species}"
        )
    output: list[str] = []
    for index, dataset in enumerate(datasets):
        if index == probe_species:
            output.extend([dataset, dataset])
        else:
            output.append(dataset)
    path.write_text("".join(output), encoding="utf-8")


def _expand_numeric_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for token in value.split():
        match = re.fullmatch(r"(\d+)\*([^*]+)", token)
        if match:
            tokens.extend([match.group(2)] * int(match.group(1)))
        else:
            tokens.append(token)
    return tokens


def reordered_magmom(value: str, order: Sequence[int], natoms: int) -> str:
    tokens = _expand_numeric_tokens(value)
    if len(tokens) == natoms:
        return " ".join(tokens[index] for index in order)
    if len(tokens) == 3 * natoms:
        vectors = [tokens[3 * i : 3 * i + 3] for i in range(natoms)]
        return " ".join(component for index in order for component in vectors[index])
    raise ValueError(
        "MAGMOM must expand to NIONS or 3*NIONS values before atom reordering; "
        f"found {len(tokens)} values for {natoms} atoms"
    )


def incar_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def incar_value(lines: Iterable[str], tag: str) -> str | None:
    target = tag.upper()
    found = None
    for line in lines:
        body = line.split("!", 1)[0].split("#", 1)[0]
        if "=" not in body:
            continue
        key, value = body.split("=", 1)
        if key.strip().upper() == target:
            found = value.strip()
    return found


def render_incar(
    source_lines: Sequence[str],
    overrides: dict[str, str | None],
    *,
    magmom: str | None = None,
) -> str:
    replaced = {tag.upper() for tag in overrides}
    if magmom is not None:
        replaced.add("MAGMOM")
    kept: list[str] = []
    for line in source_lines:
        body = line.split("!", 1)[0].split("#", 1)[0]
        key = body.split("=", 1)[0].strip().upper() if "=" in body else ""
        if key in replaced:
            continue
        kept.append(line)
    kept.extend(["", "# Atomi Hubbard-U workflow overrides"])
    if magmom is not None:
        kept.append(f"MAGMOM = {magmom}")
    for key, value in overrides.items():
        if value is not None:
            kept.append(f"{key} = {value}")
    return "\n".join(kept).rstrip() + "\n"


def alpha_label(alpha: float) -> str:
    sign = "p" if alpha >= 0 else "m"
    return f"alpha_{sign}{abs(alpha):.3f}".replace(".", "p")


def _copy_small_seed_files(seed: Path, destination: Path, names: Sequence[str]) -> None:
    for name in names:
        source = seed / name
        if not source.is_file():
            raise FileNotFoundError(f"missing seed file: {source}")
        shutil.copy2(source, destination / name)


def write_lr_run_script(path: Path, reference: Path, alpha_dirs: Sequence[Path]) -> None:
    relative_reference = reference.name
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'VASP_COMMAND="${VASP_COMMAND:-srun vasp_std}"',
        "",
        f"cd {relative_reference}",
        '$VASP_COMMAND > vasp.out 2>&1',
        "cd ..",
        "",
    ]
    for alpha_dir in alpha_dirs:
        rel = alpha_dir.name
        lines.extend(
            [
                f"for stage in {rel}/nscf {rel}/scf; do",
                f"  cp -f {relative_reference}/CHGCAR \"$stage/CHGCAR\"",
                f"  cp -f {relative_reference}/WAVECAR \"$stage/WAVECAR\"",
                "done",
                f"cd {rel}/nscf",
                '$VASP_COMMAND > vasp.out 2>&1',
                "cd ../..",
                f"cp -f {rel}/nscf/WAVECAR {rel}/scf/WAVECAR",
                f"cd {rel}/scf",
                '$VASP_COMMAND > vasp.out 2>&1',
                "cd ../..",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def write_lr_slurm_scripts(
    outdir: Path,
    alpha_dirs: Sequence[Path],
    *,
    job_name: str,
    time_limit: str,
    nodes: int,
    ntasks_per_node: int,
    cpus_per_task: int,
    mem_per_cpu: str | None,
    array_limit: int,
    vasp_command: str,
    module_loads: Sequence[str],
) -> None:
    (outdir / "alpha_dirs.txt").write_text(
        "\n".join(directory.name for directory in alpha_dirs) + "\n", encoding="utf-8"
    )
    command = vasp_command.replace("\\", "\\\\").replace('"', '\\"')
    module_lines = ["module purge"] if module_loads else []
    module_lines.extend(f"module load {module}" for module in module_loads)
    module_block = "\n".join(module_lines)
    memory_directive = f"#SBATCH --mem-per-cpu={mem_per_cpu}" if mem_per_cpu else ""
    reference = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}-ref
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
{memory_directive}
#SBATCH --time={time_limit}

set -euo pipefail
ulimit -s 200000
{module_block}
ROOT=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT/logs"
cd "$ROOT/reference_u0"
read -r -a VASP_CMD <<< "${{VASP_COMMAND:-{command}}}"
"${{VASP_CMD[@]}}" > vasp.out 2>&1
"""
    response = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}-resp
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
{memory_directive}
#SBATCH --time={time_limit}
#SBATCH --array=0-{len(alpha_dirs) - 1}%{array_limit}

set -euo pipefail
ulimit -s 200000
{module_block}
ROOT=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT/logs"
alpha_dir=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/alpha_dirs.txt")
if [[ -z "$alpha_dir" ]]; then
  echo "No alpha directory for task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
for stage in nscf scf; do
  cp -f "$ROOT/reference_u0/CHGCAR" "$ROOT/$alpha_dir/$stage/CHGCAR"
  cp -f "$ROOT/reference_u0/WAVECAR" "$ROOT/$alpha_dir/$stage/WAVECAR"
done
read -r -a VASP_CMD <<< "${{VASP_COMMAND:-{command}}}"
cd "$ROOT/$alpha_dir/nscf"
"${{VASP_CMD[@]}}" > vasp.out 2>&1
cp -f WAVECAR ../scf/WAVECAR
cd ../scf
"${{VASP_CMD[@]}}" > vasp.out 2>&1
"""
    submit = """#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
reference_job=$(sbatch --parsable submit_reference.sbatch)
echo "reference_job=$reference_job"
response_job=$(sbatch --parsable --dependency="afterok:$reference_job" submit_response_array.sbatch)
echo "response_job=$response_job"
"""
    for name, content in (
        ("submit_reference.sbatch", reference),
        ("submit_response_array.sbatch", response),
        ("submit_all.sh", submit),
    ):
        path = outdir / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)


def prepare_vasp_lr(args: argparse.Namespace) -> dict[str, object]:
    seed = args.seed.resolve()
    outdir = args.outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {outdir}; use --overwrite")
    outdir.mkdir(parents=True, exist_ok=True)
    poscar_source = seed / ("CONTCAR" if (seed / "CONTCAR").is_file() else "POSCAR")
    source_data = read_poscar(poscar_source)
    split_data, order, probe_species = split_probe_species(
        source_data, args.probe_atom, args.probe_label
    )
    source_incar = incar_lines(seed / "INCAR")
    magmom_source = incar_value(source_incar, "MAGMOM")
    magmom = (
        reordered_magmom(magmom_source, order, source_data.natoms)
        if magmom_source is not None
        else None
    )
    ntypes = len(split_data.symbols)
    ldau_l = ["-1"] * ntypes
    ldau_l[0] = str(args.l)
    zeros = ["0.0"] * ntypes

    reference = outdir / "reference_u0"
    reference.mkdir(exist_ok=True)
    _copy_small_seed_files(seed, reference, ("KPOINTS",))
    write_poscar(reference / "POSCAR", split_data)
    write_split_potcar(reference / "POTCAR", seed / "POTCAR", probe_species)
    reference_overrides = {
        "SYSTEM": f"Atomi Hubbard-U reference {split_data.symbols[0]}",
        "LDAU": ".FALSE.",
        "LMAXMIX": str(args.lmaxmix),
        "LORBIT": "11",
        "ISYM": "0",
        "NSW": "0",
        "IBRION": "-1",
        "LWAVE": ".TRUE.",
        "LCHARG": ".TRUE.",
        "ICHARG": "2",
    }
    (reference / "INCAR").write_text(
        render_incar(source_incar, reference_overrides, magmom=magmom), encoding="utf-8"
    )

    alpha_dirs: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    for alpha in args.alpha:
        alpha_dir = outdir / alpha_label(alpha)
        alpha_dirs.append(alpha_dir)
        for stage in ("nscf", "scf"):
            run_dir = alpha_dir / stage
            run_dir.mkdir(parents=True, exist_ok=True)
            _copy_small_seed_files(seed, run_dir, ("KPOINTS",))
            shutil.copy2(reference / "POSCAR", run_dir / "POSCAR")
            shutil.copy2(reference / "POTCAR", run_dir / "POTCAR")
            u_values = list(zeros)
            j_values = list(zeros)
            u_values[0] = f"{alpha:.8f}"
            j_values[0] = f"{alpha:.8f}"
            overrides = {
                "SYSTEM": f"Atomi Hubbard-U {stage} alpha={alpha:+.4f} eV",
                "LDAU": ".TRUE.",
                "LDAUTYPE": "3",
                "LDAUL": " ".join(ldau_l),
                "LDAUU": " ".join(u_values),
                "LDAUJ": " ".join(j_values),
                "LDAUPRINT": "2",
                "LMAXMIX": str(args.lmaxmix),
                "LORBIT": "11",
                "ISYM": "0",
                "NSW": "0",
                "IBRION": "-1",
                "LWAVE": ".TRUE.",
                "LCHARG": ".TRUE.",
                "ICHARG": "11" if stage == "nscf" else None,
            }
            (run_dir / "INCAR").write_text(
                render_incar(source_incar, overrides, magmom=magmom), encoding="utf-8"
            )
            manifest_rows.append(
                {
                    "alpha_eV": alpha,
                    "stage": stage,
                    "path": str(run_dir.relative_to(outdir)),
                    "probe_atom": 1,
                    "channel": args.channel,
                }
            )
    with (outdir / "response_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    metadata = {
        "schema": "atomi.vasp_hubbard_lr.v1",
        "route": "vasp-paw-linear-response",
        "seed": str(seed),
        "source_structure": str(poscar_source),
        "probe_atom_old_1based": args.probe_atom,
        "probe_atom_new_1based": 1,
        "probe_species_old": source_data.symbols[probe_species],
        "species_new": list(split_data.symbols),
        "atom_order_new_to_old_1based": [value + 1 for value in order],
        "alphas_eV": list(args.alpha),
        "projector": "VASP PAW on-site l channel",
        "l": args.l,
        "scheduler": {
            "kind": "slurm",
            "reference_script": "submit_reference.sbatch",
            "response_array_script": "submit_response_array.sbatch",
            "array_limit": args.array_limit,
            "nodes": args.nodes,
            "ntasks_per_node": args.ntasks_per_node,
            "cpus_per_task": args.cpus_per_task,
            "mem_per_cpu": args.mem_per_cpu,
            "module_loads": list(args.module_load),
            "vasp_command": args.vasp_command,
        },
        "warnings": [
            "The reference is deliberately U=0; verify that UO2 remains on the intended AFM/occupation branch.",
            "This U is projector-specific and is not a Wannier-projector U.",
            "Do not compare total energies across different perturbing alpha values.",
        ],
    }
    (outdir / "workflow.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_lr_run_script(outdir / "run_response.sh", reference, alpha_dirs)
    write_lr_slurm_scripts(
        outdir,
        alpha_dirs,
        job_name=args.job_name,
        time_limit=args.time,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks_per_node,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu=args.mem_per_cpu,
        array_limit=args.array_limit,
        vasp_command=args.vasp_command,
        module_loads=args.module_load,
    )
    return metadata


def parse_total_charge(path: Path, atom: int, channel: str) -> float:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tables: list[dict[int, dict[str, float]]] = []
    index = 0
    while index < len(lines):
        if lines[index].strip().lower() != "total charge":
            index += 1
            continue
        cursor = index + 1
        while cursor < len(lines) and "# of ion" not in lines[cursor]:
            cursor += 1
        if cursor >= len(lines):
            break
        headers = lines[cursor].split()[3:]
        cursor += 1
        while cursor < len(lines) and set(lines[cursor].strip()) <= {"-"}:
            cursor += 1
        table: dict[int, dict[str, float]] = {}
        while cursor < len(lines):
            parts = lines[cursor].split()
            if not parts or not parts[0].isdigit():
                break
            values = [float(value) for value in parts[1:]]
            table[int(parts[0])] = dict(zip(headers, values))
            cursor += 1
        if table:
            tables.append(table)
        index = cursor
    if not tables:
        raise ValueError(f"no total charge table found in {path}")
    row = tables[-1].get(atom)
    if row is None:
        raise ValueError(f"atom {atom} not found in final total charge table: {path}")
    if channel not in row:
        raise ValueError(f"channel {channel!r} not available; found {sorted(row)}")
    return row[channel]


def parse_magnetization(path: Path, atom: int, component: str = "x") -> float | None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    target = f"magnetization ({component})"
    tables: list[dict[int, float]] = []
    index = 0
    while index < len(lines):
        if lines[index].strip().lower() != target:
            index += 1
            continue
        cursor = index + 1
        while cursor < len(lines) and "# of ion" not in lines[cursor]:
            cursor += 1
        if cursor >= len(lines):
            break
        headers = lines[cursor].split()[3:]
        cursor += 1
        while cursor < len(lines) and set(lines[cursor].strip()) <= {"-"}:
            cursor += 1
        table: dict[int, float] = {}
        while cursor < len(lines):
            parts = lines[cursor].split()
            if not parts or not parts[0].isdigit():
                break
            row = dict(zip(headers, [float(value) for value in parts[1:]]))
            if "tot" in row:
                table[int(parts[0])] = row["tot"]
            cursor += 1
        if table:
            tables.append(table)
        index = cursor
    return tables[-1].get(atom) if tables else None


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    if len(xs) < 2 or len(xs) != len(ys):
        raise ValueError("linear fit requires at least two matched points")
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("alpha values have zero variance")
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator
    intercept = ybar - slope * xbar
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - ybar) ** 2 for y in ys)
    r2 = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total if total else math.nan
    return slope, intercept, r2


def analyze_vasp_lr(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve()
    rows = list(csv.DictReader((root / "response_manifest.csv").open(encoding="utf-8")))
    parsed: list[dict[str, object]] = []
    for row in rows:
        outcar = root / row["path"] / "OUTCAR"
        occupation = parse_total_charge(outcar, int(row["probe_atom"]), row["channel"])
        moment = parse_magnetization(outcar, int(row["probe_atom"]))
        parsed.append({**row, "occupation": occupation, "moment": moment})
    by_stage: dict[str, list[dict[str, object]]] = {"nscf": [], "scf": []}
    for row in parsed:
        by_stage[str(row["stage"])].append(row)
    fits: dict[str, dict[str, float]] = {}
    for stage, stage_rows in by_stage.items():
        stage_rows.sort(key=lambda row: float(row["alpha_eV"]))
        xs = [float(row["alpha_eV"]) for row in stage_rows]
        ys = [float(row["occupation"]) for row in stage_rows]
        slope, intercept, r2 = _linear_fit(xs, ys)
        fits[stage] = {"slope_eV_inv": slope, "intercept": intercept, "r2": r2}
    chi0 = fits["nscf"]["slope_eV_inv"]
    chi = fits["scf"]["slope_eV_inv"]
    if abs(chi0) < args.min_slope or abs(chi) < args.min_slope:
        raise ValueError(f"response slope too small to invert: chi0={chi0}, chi={chi}")
    u_value = 1.0 / chi - 1.0 / chi0
    warnings: list[str] = []
    if min(fits["nscf"]["r2"], fits["scf"]["r2"]) < args.min_r2:
        warnings.append("occupation response is not sufficiently linear")
    reference_outcar = root / "reference_u0" / "OUTCAR"
    reference_moment = parse_magnetization(reference_outcar, 1) if reference_outcar.is_file() else None
    moments = [float(row["moment"]) for row in parsed if row["moment"] is not None]
    if args.guard_moment_sign and reference_moment is not None and abs(reference_moment) > 1.0e-8:
        if any(moment * reference_moment < 0 for moment in moments):
            warnings.append("probe magnetic moment changed sign across the response series")
    if args.min_abs_moment > 0 and any(abs(moment) < args.min_abs_moment for moment in moments):
        warnings.append("probe magnetic moment fell below the configured branch threshold")
    health = "accepted" if not warnings else "warning"
    result = {
        "schema": "atomi.vasp_hubbard_lr_result.v1",
        "projector": "VASP PAW on-site channel",
        "chi0_eV_inv": chi0,
        "chi_eV_inv": chi,
        "U_response_eV": u_value,
        "fits": fits,
        "health": health,
        "warnings": warnings,
        "reference_probe_moment": reference_moment,
        "minimum_r2": args.min_r2,
    }
    (root / "hubbard_response.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (root / "hubbard_response.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["alpha_eV", "stage", "path", "occupation", "moment"]
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames} for row in parsed
        )
    return result


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def prepare_vasp_crpa(args: argparse.Namespace) -> dict[str, object]:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    version = tuple(int(value) for value in re.findall(r"\d+", args.vasp_version)[:2])
    spectral_supported = version >= (6, 6)
    metadata = {
        "schema": "atomi.vasp_wannier_crpa.v1",
        "route": "vasp-wannier-scrpa" if args.spectral else "vasp-wannier-crpa",
        "vasp_version": args.vasp_version,
        "spectral_requested": args.spectral,
        "spectral_supported": spectral_supported,
        "status": "ready" if (not args.spectral or spectral_supported) else "blocked-version",
        "projector": "VASP MLWF cRPA target space",
        "not_equivalent_to": "VASP PAW DFT+U projector",
    }
    if args.spectral and not spectral_supported:
        metadata["warning"] = "LSCRPA requires VASP 6.6.0 or newer; do not submit this scaffold."
    _write_text(
        outdir / "00_ground" / "INCAR.add",
        """
LWAVE = .TRUE.
LCHARG = .TRUE.
LORBIT = 11
ISYM = 0
""",
    )
    _write_text(
        outdir / "01_wannier" / "INCAR.add",
        f"""
LWANNIER90 = .TRUE.
NUM_WANN = {args.num_wann}
LWRITE_WANPROJ = .TRUE.
""",
    )
    _write_text(
        outdir / "02_virtual" / "INCAR.add",
        f"""
ALGO = Exact
NELM = 1
NBANDS = {args.nbands}
LOPTICS = .TRUE.
""",
    )
    crpa = f"""
ALGO = CRPAR
NBANDS = {args.nbands}
LOCALIZED_BASIS = MLWF
NTARGET_STATES = {args.target_states}
NCRPA_BANDS = {args.crpa_bands}
NOMEGA = {args.nomega}
NOMEGA_DUMP = 0
PRECFOCK = Fast
"""
    if args.spectral:
        crpa += "LSCRPA = .TRUE.\n"
    _write_text(outdir / "03_crpa" / "INCAR.add", crpa)
    _write_text(
        outdir / "WORKFLOW.md",
        f"""
# VASP Wannier-cRPA workflow

Status: `{metadata['status']}` for declared VASP `{args.vasp_version}`.

1. Run a guarded DFT ground state.
2. Construct and inspect MLWFs; compare Wannier-interpolated and VASP bands.
3. Generate sufficiently many virtual orbitals and optical matrix elements.
4. Run cRPA and parse `UIJKL`; convergence-test NBANDS, k mesh, target bands,
   Wannier windows, and frequency grid.

The resulting U/J refer to the MLWF target space. They are not automatically
projector-consistent with standard VASP `LDAUL/LDAUU/LDAUJ` DFT+U.
""",
    )
    (outdir / "workflow.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def prepare_qe(args: argparse.Namespace) -> dict[str, object]:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    _write_text(
        outdir / "hp.in",
        f"""
&INPUTHP
 prefix = '{args.prefix}'
 outdir = './tmp'
 nq1 = {args.nq[0]}
 nq2 = {args.nq[1]}
 nq3 = {args.nq[2]}
 iverbosity = 2
 find_atpert = 1
 conv_thr_chi = 1.0d-5
/
""",
    )
    _write_text(
        outdir / "HUBBARD.atomic.template",
        f"""
HUBBARD (ortho-atomic)
U {args.element}-{args.manifold} 0.0
""",
    )
    _write_text(
        outdir / "HUBBARD.wf.template",
        f"""
HUBBARD (wf)
U {args.element}-{args.manifold} U_FROM_MATCHED_WANNIER_RESPONSE
""",
    )
    _write_text(
        outdir / "WORKFLOW.md",
        f"""
# QE Hubbard-U routes for {args.element} {args.manifold}

## Baseline supported route

Run `pw.x` with the `ortho-atomic` HUBBARD card, then `hp.x -in hp.in` to
obtain projector-consistent U/V for stock QE. Record pseudopotential, magnetic
state, structure, q mesh, and projector definition.

## Wannier-projector research route

Use `pw2wannier90.x`/Wannier90 and `wannier2pw.x` to build the `.hub` projector
file, then use `HUBBARD (wf)`. Stock QE documents `wf` projectors but does not
provide forces/stress for them. Do not assume stock `hp.x` determines U for the
same Wannier projector. A matched Piotr/research linear-response branch or an
explicitly validated alternative is required; record its commit and equations.

Never transfer an atomic-projector HP value to a Wannier projector without a
separate labeled comparison.
""",
    )
    metadata = {
        "schema": "atomi.qe_hubbard_u.v1",
        "prefix": args.prefix,
        "baseline": "QE hp.x with ortho-atomic projectors",
        "experimental": "QE wf projector via wannier2pw.x plus matched research response",
        "forces_stress_wf": False,
        "projector_consistency_required": True,
    }
    (outdir / "workflow.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hubbard-u-workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    lr = sub.add_parser("vasp-lr-prepare", help="Prepare VASP PAW linear-response U runs")
    lr.add_argument("--seed", type=Path, required=True)
    lr.add_argument("--outdir", type=Path, required=True)
    lr.add_argument("--probe-atom", type=int, required=True, help="1-based atom index in seed POSCAR")
    lr.add_argument("--probe-label")
    lr.add_argument("--l", type=int, default=3)
    lr.add_argument("--channel", default="f")
    lr.add_argument("--lmaxmix", type=int, default=6)
    lr.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=(-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20),
    )
    lr.add_argument("--overwrite", action="store_true")
    lr.add_argument("--job-name", default="hubbard-u-lr")
    lr.add_argument("--time", default="12:00:00")
    lr.add_argument("--nodes", type=int, default=1)
    lr.add_argument("--ntasks-per-node", type=int, default=1)
    lr.add_argument("--cpus-per-task", type=int, default=1)
    lr.add_argument("--mem-per-cpu")
    lr.add_argument("--array-limit", type=int, default=4)
    lr.add_argument(
        "--module-load",
        action="append",
        default=[],
        help="Environment module to load in generated Slurm jobs; repeat as needed",
    )
    lr.add_argument("--vasp-command", default="srun vasp_std")
    lr.set_defaults(func=prepare_vasp_lr)

    analyze = sub.add_parser("vasp-lr-analyze", help="Fit chi0, chi and VASP response U")
    analyze.add_argument("--root", type=Path, required=True)
    analyze.add_argument("--min-r2", type=float, default=0.98)
    analyze.add_argument("--min-slope", type=float, default=1.0e-8)
    analyze.add_argument("--guard-moment-sign", action=argparse.BooleanOptionalAction, default=True)
    analyze.add_argument("--min-abs-moment", type=float, default=0.0)
    analyze.set_defaults(func=analyze_vasp_lr)

    crpa = sub.add_parser("vasp-crpa-prepare", help="Prepare a version-gated VASP Wannier-cRPA scaffold")
    crpa.add_argument("--outdir", type=Path, required=True)
    crpa.add_argument("--vasp-version", default="6.6.0")
    crpa.add_argument("--spectral", action=argparse.BooleanOptionalAction, default=True)
    crpa.add_argument("--num-wann", type=int, required=True)
    crpa.add_argument("--nbands", type=int, required=True)
    crpa.add_argument("--target-states", required=True)
    crpa.add_argument("--crpa-bands", required=True)
    crpa.add_argument("--nomega", type=int, default=8)
    crpa.set_defaults(func=prepare_vasp_crpa)

    qe = sub.add_parser("qe-prepare", help="Prepare QE HP and research Wannier-projector scaffolds")
    qe.add_argument("--outdir", type=Path, required=True)
    qe.add_argument("--prefix", default="uo2")
    qe.add_argument("--element", default="U")
    qe.add_argument("--manifold", default="5f")
    qe.add_argument("--nq", type=int, nargs=3, default=(2, 2, 2))
    qe.set_defaults(func=prepare_qe)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object] | None:
    console_entry = argv is None
    args = build_parser().parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, indent=2))
    return None if console_entry else result


if __name__ == "__main__":
    main()
