"""Draft Methods and brief Results text from completed calculation folders."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from atomi.core.doctor import find_config_path, load_hpc_config


MODULE_ALIASES = {
    "DFT": {"DFT", "VASP", "ELECTRONIC", "ELECTRONIC_STRUCTURE", "ELECTRONIC-STRUCTURE"},
    "VASP_PREP": {
        "VASP_PREP",
        "DFT_PREP",
        "STRUCTURE_PREP",
        "DEFECT_CLOUD",
        "DEFECT-CLOUD",
        "VASP_DEFECT_CLOUD",
        "VASP-DEFECT-CLOUD",
        "DEFECT_CANDIDATES",
        "DEFECT-CANDIDATES",
        "VASP_CANDIDATES",
        "VASP-CANDIDATES",
        "SEED_DFT",
        "SEED-DFT",
    },
    "VASP_SPIN": {
        "VASP_SPIN",
        "VASP-SPIN",
        "SPIN",
        "SPIN_REPORT",
        "SPIN-REPORT",
        "VASP_SPIN_REPORT",
        "VASP-SPIN-REPORT",
        "SPIN_SCREENING",
        "SPIN-SCREENING",
        "MAGMOM_SCREENING",
        "MAGMOM-SCREENING",
    },
    "AIMD": {"AIMD", "CP2K", "AB_INITIO_MD", "AB-INITIO-MD"},
    "MD": {"MD", "LAMMPS", "MOLECULAR_DYNAMICS", "MOLECULAR-DYNAMICS"},
    "TRANSPORT": {
        "TRANSPORT",
        "THERMAL",
        "THERMAL_TRANSPORT",
        "THERMAL-TRANSPORT",
        "THERMAL_CONDUCTIVITY",
        "THERMAL-CONDUCTIVITY",
        "GK",
        "GREEN_KUBO",
        "GREEN-KUBO",
        "GREENKUBO",
        "RNEMD",
        "REVERSE_NEMD",
        "REVERSE-NEMD",
        "NEMD",
    },
    "MLIP": {"MLIP", "MACE", "ML", "MACHINE_LEARNING_POTENTIAL", "MACHINE-LEARNING-POTENTIAL"},
    "CALPHAD": {"CALPHAD", "TDB", "THERMODYNAMIC_DATABASE"},
    "MOOSE": {"MOOSE", "MULTIPHYSICS"},
    "QHA": {"QHA", "PHONOPY", "THERMO_QHA_MD", "THERMO-QHA-MD"},
    "SCATTERING": {"SCATTERING", "PDF", "RDF", "XAFS", "NEUTRON", "X-RAY", "XRAY"},
}

STYLE_REFERENCES = [
    {
        "label": "npj Computational Materials reporting standards",
        "url": "https://www.nature.com/npjcompumats/for-authors-and-referees/about/editorial-policies/reporting-standards",
        "note": "Emphasizes reproducibility, data availability, and code availability.",
    },
    {
        "label": "ACS computational reporting guidance",
        "url": "https://researcher-resources.acs.org/publish/author_guidelines?coden=inocaj",
        "note": "Requests software/version, model details, convergence criteria, and spin treatment.",
    },
    {
        "label": "Materials Project calculation methodology",
        "url": "https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga+u-calculations/parameters-and-convergence",
        "note": "Shows compact reporting of PAW, cutoff, k-point density, and convergence choices.",
    },
    {
        "label": "Computational Materials Science guide for authors",
        "url": "https://www.sciencedirect.com/journal/computational-materials-science/publish/guide-for-authors",
        "note": "Encourages data availability and clearly structured calculation sections.",
    },
]

METHOD_FORMAT_RULES = {
    "DFT": [
        "State the physical target, reference structure, charge/spin state, supercell size, and whether the calculation is a relaxation, static energy, or response calculation.",
        "Report code/version, exchange-correlation functional, Hubbard U or other corrections, PAW/pseudopotential set, cutoff, k-point sampling, smearing, convergence thresholds, relaxation criteria, and magnetic initialization.",
        "For results, begin with validated ground-state quantities such as energy differences, volume, local moments, and convergence or stability checks before interpreting trends.",
    ],
    "VASP_PREP": [
        "Describe seed-motif provenance, template files, atom ordering, perturbation families, random seed, runlist construction, and screening criteria for failed or duplicate candidates.",
        "Separate chemically distinct motif classes, defect-compensation models, concentration/cell-size choices, and spin or valence labels so later thermodynamics can reuse the metadata.",
    ],
    "VASP_SPIN": [
        "Report how initial MAGMOM patterns were enumerated, which atoms were allowed to flip, and which nominal moment windows define physically retained states.",
        "Compare initial and final/last moments by element and use energy rankings only together with the final spin/valence classification.",
    ],
    "AIMD": [
        "Report ensemble, temperature/pressure schedule, timestep, thermostat/barostat, restraint or collective-variable definitions, equilibration window, production length, and frame-selection rule.",
        "For results, summarize trajectory stability, reaction-coordinate or bond statistics, and representative-frame selection before mechanistic interpretation.",
    ],
    "MD": [
        "Report potential/model source, supercell size, timestep, ensemble sequence, temperature grid, equilibration criteria, production length, uncertainty/blocking method, and normalization basis.",
        "When extracting elastic or thermal properties, state the deformation or fluctuation formula, averaging window, finite-size checks, and comparison reference.",
    ],
    "TRANSPORT": [
        "For Green-Kubo, report the equilibrated structure source, ML/LAMMPS binary/profile, pair style, suffix, timestep, NVT/NPT pre-equilibration length, NVE production length, heat-current sampling interval, HCACF correlation time, plateau window, seed count, and unit conversion.",
        "For rNEMD/NEMD, report the equilibrated structure source, replicate/slab geometry, heat-flow direction, bin count, swap interval or imposed flux, timestep, production length, gradient-fit windows, mirrored-slope agreement, seed count, and finite-size/gradient checks.",
        "For results, tabulate k(T) with per-axis/per-seed diagnostics, seed coefficient of variation, anisotropy or slope mismatch, late-plateau drift, validation status, and a comparison against literature/benchmark values before interpreting mechanisms.",
    ],
    "MLIP": [
        "Report training-data provenance, train/validation/test split, motif or temperature coverage, model architecture/descriptor, loss weights, active-learning or outlier rules, and validation metrics.",
        "Show that the ML model is used within the domain spanned by the reference data, with explicit checks on energies, forces, stresses, stability, and relevant properties.",
    ],
    "QHA": [
        "Report phonon/displacement settings, volume grid, free-energy fit, splice or anchor choices, temperature grid, uncertainty treatment, and normalization to atom/formula-unit/cell.",
    ],
    "CALPHAD": [
        "Report reference states, endmember or pseudo-endmember definitions, source priority between database/literature/DFT/MD values, fitted interaction form, and composition/temperature validity range.",
    ],
    "MOOSE": [
        "Report the exact material properties transferred, interpolation basis, units, temperature range, missing-property source, and how the exported tables map into the multiphysics input.",
    ],
    "SCATTERING": [
        "Report trajectory/source frames, absorber or pair definitions, instrument corrections, Q/r/k ranges, comparison metric, and uncertainty or frame averaging.",
    ],
}

COMMON_FORMAT_RULES = [
    "Use a layered Methods flow: electronic-structure reference, finite-temperature sampling, ML/data model if used, property extraction equations, uncertainty/convergence checks, and reproducibility artifacts.",
    "Use a layered Results flow: first validate the reference state, then show temperature/composition/property trends, then compare with experiment or prior calculation, and finally state limitations.",
    "Do not let the draft invent missing settings. Leave missing convergence, citation, unit, and reference-state information as verification items.",
]


@dataclass
class RunEvidence:
    path: Path
    requested_modules: list[str]
    detected_modules: list[str] = field(default_factory=list)
    files: dict[str, list[str]] = field(default_factory=dict)
    facts: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add_file(self, key: str, path: Path) -> None:
        values = self.files.setdefault(key, [])
        rel = _safe_relative(path, self.path)
        if rel not in values:
            values.append(rel)

    def add_module(self, module: str) -> None:
        if module not in self.detected_modules:
            self.detected_modules.append(module)


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_text(path: Path, limit: int = 2_000_000) -> str:
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            data = handle.read()
    else:
        data = path.read_bytes()
    if len(data) > limit:
        data = data[-limit:]
    return data.decode("utf-8", errors="replace")


def _strip_comment(value: str) -> str:
    for marker in ("#", "!"):
        if marker in value:
            value = value.split(marker, 1)[0]
    return value.strip()


def _find_files(root: Path, max_files: int = 5000) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if len(files) >= max_files:
            break
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def _by_name(files: Iterable[Path], *names: str) -> list[Path]:
    wanted = {name.lower() for name in names}
    return [path for path in files if path.name.lower() in wanted]


def _by_suffix(files: Iterable[Path], *suffixes: str) -> list[Path]:
    wanted = tuple(suffix.lower() for suffix in suffixes)
    return [path for path in files if path.name.lower().endswith(wanted)]


def _by_glob_name(files: Iterable[Path], pattern: str) -> list[Path]:
    regex = re.compile(pattern, re.IGNORECASE)
    return [path for path in files if regex.search(path.name)]


def normalize_modules(values: Iterable[str]) -> list[str]:
    modules: list[str] = []
    for value in values:
        for token in re.split(r"[\s,;]+", value.strip()):
            if not token:
                continue
            normalized = token.upper().replace(" ", "_")
            matched = None
            for module, aliases in MODULE_ALIASES.items():
                if normalized in aliases:
                    matched = module
                    break
            matched = matched or normalized
            if matched not in modules:
                modules.append(matched)
    return modules


def parse_poscar(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        return {"path": str(path), "warning": "POSCAR too short to parse composition"}
    species_tokens = lines[5].split()
    count_line_index = 6
    if all(_is_number(token) for token in species_tokens):
        species = [f"X{i + 1}" for i in range(len(species_tokens))]
        count_tokens = species_tokens
        count_line_index = 5
    else:
        species = species_tokens
        count_tokens = lines[count_line_index].split()
    counts: list[int] = []
    for token in count_tokens:
        try:
            counts.append(int(float(token)))
        except ValueError:
            break
    formula = " ".join(f"{element}{count}" for element, count in zip(species, counts))
    return {
        "comment": lines[0].strip(),
        "species": species,
        "counts": counts,
        "natoms": sum(counts),
        "formula": formula,
        "count_line": count_line_index + 1,
    }


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def parse_incar(path: Path) -> dict[str, str]:
    tags: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean = _strip_comment(line)
        if "=" not in clean:
            continue
        key, value = clean.split("=", 1)
        key = key.strip().upper()
        if key:
            tags[key] = value.strip()
    return tags


def parse_kpoints(path: Path) -> dict[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    info: dict[str, str] = {}
    if lines:
        info["comment"] = lines[0]
    if len(lines) >= 4:
        info["mode"] = lines[2]
        info["mesh"] = lines[3]
    return info


def parse_potcar(path: Path) -> dict[str, object]:
    text = _read_text(path)
    titles = re.findall(r"^\s*TITEL\s*=\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not titles:
        titles = re.findall(r"^\s*PAW\S*\s+(.+?)\s*$", text, flags=re.MULTILINE)
    return {"titles": titles[:20], "title_count": len(titles)}


def parse_outcar(path: Path) -> dict[str, object]:
    text = _read_text(path)
    info: dict[str, object] = {}
    version_match = re.search(r"^\s*(vasp\.\S+.*)$", text, flags=re.MULTILINE | re.IGNORECASE)
    if version_match:
        version_line = version_match.group(1).strip()
        info["vasp_version_line"] = version_line
        info["vasp_version"] = version_line.split()[0]
    energies = re.findall(r"free\s+energy\s+TOTEN\s+=\s*([-+0-9.Ee]+)", text)
    if energies:
        info["final_energy_eV"] = float(energies[-1])
    volumes = re.findall(r"volume of cell\s*:\s*([-+0-9.Ee]+)", text)
    if volumes:
        info["final_volume_A3"] = float(volumes[-1])
    nions = re.findall(r"NIONS\s*=\s*(\d+)", text)
    if nions:
        info["nions"] = int(nions[-1])
    elapsed = re.findall(r"Elapsed time \(sec\):\s*([-+0-9.Ee]+)", text)
    if elapsed:
        info["elapsed_s"] = float(elapsed[-1])
    info["finished"] = "General timing and accounting" in text or "Voluntary context switches" in text
    return info


def parse_cp2k_input(path: Path) -> dict[str, object]:
    text = _read_text(path)
    info: dict[str, object] = {}
    patterns = {
        "project": r"^\s*PROJECT\s+(\S+)",
        "run_type": r"^\s*RUN_TYPE\s+(\S+)",
        "ensemble": r"^\s*ENSEMBLE\s+(\S+)",
        "steps": r"^\s*STEPS\s+(\d+)",
        "timestep_fs": r"^\s*TIMESTEP\s+([-+0-9.Ee]+)",
        "temperature_K": r"^\s*TEMPERATURE\s+([-+0-9.Ee]+)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if not matches:
            continue
        value = matches[-1]
        if key in {"steps"}:
            info[key] = int(value)
        elif key in {"timestep_fs", "temperature_K"}:
            info[key] = float(value)
        else:
            info[key] = value
    return info


def parse_cp2k_log(path: Path) -> dict[str, object]:
    text = _read_text(path)
    info: dict[str, object] = {}
    energies = re.findall(r"ENERGY\|.*?energy.*?:\s*([-+0-9.Ee]+)", text)
    if energies:
        info["final_energy_au"] = float(energies[-1])
        info["energy_records"] = len(energies)
    steps = re.findall(r"\bSTEP\s+NUMBER\s*[:=]?\s*(\d+)", text, flags=re.IGNORECASE)
    if steps:
        info["last_step"] = int(steps[-1])
    info["finished"] = "PROGRAM ENDED" in text or "CP2K| run finished" in text
    return info


def count_xyz_frames(path: Path, max_frames: int | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            first = first.strip()
            if not first:
                continue
            try:
                natoms = int(first)
            except ValueError:
                break
            handle.readline()
            for _ in range(natoms):
                if not handle.readline():
                    break
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    return count


def parse_lammps_log(path: Path) -> dict[str, object]:
    text = _read_text(path)
    rows: list[dict[str, float]] = []
    header: list[str] | None = None
    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        if "Step" in parts and ("Temp" in parts or "Temperature" in parts):
            header = parts
            continue
        if header and len(parts) >= len(header) and all(_is_number(item) for item in parts[: len(header)]):
            row = {key: float(value) for key, value in zip(header, parts)}
            rows.append(row)
            continue
        if header and (raw.startswith("Loop time") or raw.startswith("ERROR")):
            header = None
    info: dict[str, object] = {"thermo_rows": len(rows)}
    if rows:
        last = rows[-1]
        for key in ("Step", "Temp", "Press", "PotEng", "PE", "TotEng", "Volume"):
            if key in last:
                info[f"last_{key}"] = last[key]
    info["finished"] = "Loop time of" in text
    return info


def parse_csv_summary(path: Path) -> dict[str, object]:
    rows, fieldnames = _read_csv_dicts(path)
    info: dict[str, object] = {"rows": len(rows), "columns": fieldnames}
    temp_key = next((key for key in ("T_K", "temperature_K", "temperature", "T") if key in fieldnames), None)
    if temp_key and rows:
        values = [float(row[temp_key]) for row in rows if row.get(temp_key) and _is_number(row[temp_key])]
        if values:
            info["T_min_K"] = min(values)
            info["T_max_K"] = max(values)
    return info


def _read_csv_dicts(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if path.name.lower().endswith(".gz"):
        handle = gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    else:
        handle = path.open("r", encoding="utf-8", errors="replace", newline="")
    with handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, reader.fieldnames or []


def _load_json(path: Path) -> object:
    return json.loads(_read_text(path))


def _find_first_key(payload: object, keys: Iterable[str]) -> object | None:
    wanted = {key.lower() for key in keys}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in wanted:
                return value
        for value in payload.values():
            found = _find_first_key(value, wanted)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_first_key(item, wanted)
            if found is not None:
                return found
    return None


def _number_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if _is_number(str(value)):
        return float(str(value))
    return None


def _int_or_none(value: object) -> int | None:
    number = _number_or_none(value)
    if number is None:
        return None
    return int(number)


def _list_numbers(value: object) -> list[float]:
    if isinstance(value, list | tuple):
        numbers = [_number_or_none(item) for item in value]
        return [number for number in numbers if number is not None]
    number = _number_or_none(value)
    return [] if number is None else [number]


def _temperature_values(payload: object) -> list[float]:
    for key in ("temperatures_K", "temperature_grid_K", "temperature_grid", "temperatures", "T_values", "T_grid"):
        values = _list_numbers(_find_first_key(payload, [key]))
        if values:
            return sorted(set(values))
    values: list[float] = []

    def collect(obj: object) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = str(key).lower()
                if key_lower in {"temperature_k", "t_k"} or key_lower.endswith("_temperature_k"):
                    number = _number_or_none(value)
                    if number is not None:
                        values.append(number)
                collect(value)
        elif isinstance(obj, list):
            for item in obj:
                collect(item)

    collect(payload)
    return sorted(set(values))


def _selected_runtime_estimate(payload: object) -> dict[str, object]:
    runtime = _find_first_key(payload, ["runtime_estimate"])
    if not isinstance(runtime, dict):
        return {}
    keep = (
        "timestep_ps",
        "timestep_fs",
        "run_steps_per_stage",
        "nvt_steps_per_stage",
        "nve_steps_per_stage",
        "run_time_ps_per_stage",
        "nvt_time_ps_per_stage",
        "nve_time_ps_per_stage",
        "estimated_steps_per_hour",
        "walltime_safety_factor",
        "estimated_walltime_hours_per_stage",
        "estimated_gpu_hours_all_stages",
        "estimated_elapsed_hours_at_array_limit",
        "array_limit",
        "replicate",
        "replicate_factor",
        "throughput_source",
    )
    return {key: runtime[key] for key in keep if key in runtime}


def parse_transport_config(path: Path, route: str) -> dict[str, object]:
    payload = _load_json(path)
    common = {
        "runtime_profile": _find_first_key(payload, ["runtime_profile"]),
        "pair_style_backend": _find_first_key(payload, ["pair_style_backend"]),
        "model_file": _find_first_key(payload, ["model_file"]),
        "model_elements": _find_first_key(payload, ["model_elements"]),
        "timestep_ps": _number_or_none(_find_first_key(payload, ["timestep_ps"])),
        "n_seeds": _int_or_none(_find_first_key(payload, ["n_seeds", "n_seeds_per_temperature"])),
        "array_limit": _int_or_none(_find_first_key(payload, ["array_limit"])),
    }
    route_keys: dict[str, object]
    if route == "gk":
        route_keys = {
            "heat_flux_suffix": _find_first_key(payload, ["heat_flux_suffix"]),
            "nvt_preequilibration_ps": _number_or_none(_find_first_key(payload, ["nvt_preequilibration_ps"])),
            "nve_time_ps": _number_or_none(_find_first_key(payload, ["nve_time_ps"])),
            "sample_interval_ps": _number_or_none(_find_first_key(payload, ["sample_interval_ps"])),
            "correlation_time_ps": _number_or_none(_find_first_key(payload, ["correlation_time_ps"])),
            "plateau_window_ps": _number_or_none(_find_first_key(payload, ["plateau_window_ps"])),
        }
    else:
        route_keys = {
            "run_time_ps": _number_or_none(_find_first_key(payload, ["run_time_ps"])),
            "replicate": _find_first_key(payload, ["replicate"]),
            "direction": _find_first_key(payload, ["direction"]),
            "nbin": _int_or_none(_find_first_key(payload, ["nbin"])),
            "swap_every": _int_or_none(_find_first_key(payload, ["swap_every"])),
        }
    settings = {key: value for key, value in {**common, **route_keys}.items() if value not in (None, "", [], {})}
    temperatures = _temperature_values(payload)
    return {
        "route": route,
        "file": path.name,
        "settings": settings,
        "temperatures_K": temperatures,
        "n_temperatures": len(temperatures) or None,
    }


def parse_transport_plan(path: Path, route: str) -> dict[str, object]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {"route": route, "file": path.name}
    temperatures = _temperature_values(payload)
    return {
        "route": route,
        "file": path.name,
        "root": payload.get("root", ""),
        "n_temperatures": payload.get("n_temperatures") or len(temperatures) or None,
        "n_seeds_per_temperature": payload.get("n_seeds_per_temperature"),
        "n_stages": payload.get("n_stages"),
        "temperatures_K": temperatures,
        "runtime_estimate": _selected_runtime_estimate(payload),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _sample_stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _csv_temperature_key(fieldnames: list[str]) -> str | None:
    return next((key for key in ("T_K", "temperature_K", "temperature", "T") if key in fieldnames), None)


def _csv_k_key(fieldnames: list[str]) -> str | None:
    candidates = (
        "k_W_mK",
        "k_mean_W_mK",
        "k_iso_W_mK",
        "thermal_conductivity_W_mK",
        "kappa_W_mK",
        "k",
    )
    return next((key for key in candidates if key in fieldnames), None)


def parse_transport_temperature_table(path: Path, route: str) -> dict[str, object]:
    rows, fieldnames = _read_csv_dicts(path)
    temp_key = _csv_temperature_key(fieldnames)
    k_key = _csv_k_key(fieldnames)
    compact_rows: list[dict[str, object]] = []
    by_temperature: dict[float, list[float]] = {}
    for row in rows:
        temperature = _number_or_none(row.get(temp_key, "") if temp_key else None)
        conductivity = _number_or_none(row.get(k_key, "") if k_key else None)
        compact: dict[str, object] = {}
        if temperature is not None:
            compact["temperature_K"] = temperature
        if conductivity is not None:
            compact["k_W_mK"] = conductivity
        for key in (
            "status",
            "ok_seed_count",
            "seed_count",
            "axis_spread_fraction",
            "seed_cv_fraction",
            "late_drift_fraction",
            "slope_disagreement_fraction",
        ):
            if key in row and row[key] not in ("", None):
                compact[key] = row[key]
        if compact:
            compact_rows.append(compact)
        if temperature is not None and conductivity is not None:
            by_temperature.setdefault(temperature, []).append(conductivity)
    return {
        "route": route,
        "file": path.name,
        "rows": len(rows),
        "columns": fieldnames,
        "temperature_key": temp_key,
        "conductivity_key": k_key,
        "temperatures_K": sorted(by_temperature),
        "records": compact_rows[:50],
    }


def parse_transport_seed_summary(path: Path, route: str) -> dict[str, object]:
    rows, fieldnames = _read_csv_dicts(path)
    temp_key = _csv_temperature_key(fieldnames)
    k_key = _csv_k_key(fieldnames)
    by_temperature: dict[float, list[float]] = {}
    statuses: dict[str, int] = {}
    for row in rows:
        status = row.get("status") or row.get("fit_status") or ""
        if status:
            statuses[status] = statuses.get(status, 0) + 1
        temperature = _number_or_none(row.get(temp_key, "") if temp_key else None)
        conductivity = _number_or_none(row.get(k_key, "") if k_key else None)
        if temperature is not None and conductivity is not None:
            by_temperature.setdefault(temperature, []).append(conductivity)
    temperature_summary = []
    for temperature, values in sorted(by_temperature.items()):
        mean = _mean(values)
        stdev = _sample_stdev(values)
        temperature_summary.append(
            {
                "temperature_K": temperature,
                "seed_count": len(values),
                "k_mean_W_mK": mean,
                "k_stdev_W_mK": stdev,
                "k_cv_fraction": None if mean in (None, 0) or stdev is None else abs(stdev / mean),
            }
        )
    return {
        "route": route,
        "file": path.name,
        "rows": len(rows),
        "columns": fieldnames,
        "status_counts": statuses,
        "temperature_summary": temperature_summary,
    }


def _percent_like(report: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = _number_or_none(report.get(key))
        if value is not None:
            if key.endswith("_percent") or abs(value) > 2:
                return value
            return 100.0 * value
    return None


def parse_transport_validation(path: Path, route: str) -> dict[str, object]:
    payload = _load_json(path)
    reports = payload.get("reports", []) if isinstance(payload, dict) else []
    if not isinstance(reports, list):
        reports = []
    compact_reports: list[dict[str, object]] = []
    status_counts: dict[str, int] = {}
    for raw in reports:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status", ""))
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        compact = {
            "temperature_K": _number_or_none(raw.get("temperature_K")),
            "status": status,
            "k_W_mK": _number_or_none(raw.get("k_W_mK")),
            "ok_seed_count": _int_or_none(raw.get("ok_seed_count")),
            "seed_count": _int_or_none(raw.get("seed_count")),
            "seed_cv_percent": _percent_like(raw, "seed_cv_fraction", "seed_cv_percent"),
            "axis_spread_percent": _percent_like(raw, "axis_spread_fraction", "axis_spread_percent"),
            "late_drift_percent": _percent_like(raw, "late_drift_fraction", "late_drift_percent"),
            "slope_mismatch_percent": _percent_like(
                raw,
                "slope_disagreement_fraction",
                "slope_mismatch_fraction",
                "slope_mismatch_percent",
            ),
            "warnings": raw.get("warnings", []),
        }
        compact_reports.append({key: value for key, value in compact.items() if value not in (None, "", [], {})})
    return {
        "route": route,
        "file": path.name,
        "reports": compact_reports,
        "status_counts": status_counts,
    }


def parse_runlist(path: Path) -> dict[str, object]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return {"rows": len(rows), "first_entries": rows[:5]}


def parse_defect_cloud_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    families_by_motif = payload.get("families_by_motif", {})
    family_totals: dict[str, int] = {}
    if isinstance(families_by_motif, dict):
        for motif_counts in families_by_motif.values():
            if isinstance(motif_counts, dict):
                for family, count in motif_counts.items():
                    try:
                        family_totals[str(family)] = family_totals.get(str(family), 0) + int(count)
                    except (TypeError, ValueError):
                        continue
    defaults = payload.get("defaults", {})
    return {
        "schema": payload.get("schema", ""),
        "n_seed_motifs": payload.get("n_seed_motifs"),
        "n_candidate_runs": payload.get("n_candidate_runs"),
        "per_motif_requested": payload.get("per_motif_requested"),
        "seed": payload.get("seed"),
        "family_totals": family_totals,
        "defaults": defaults if isinstance(defaults, dict) else {},
    }


def parse_defect_cloud_index(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    motif_ids = sorted({row.get("motif_id", "") for row in rows if row.get("motif_id")})
    family_counts: dict[str, int] = {}
    for row in rows:
        family = row.get("family", "")
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
    return {
        "rows": len(rows),
        "columns": reader.fieldnames or [],
        "motif_count": len(motif_ids),
        "motif_ids": motif_ids[:20],
        "family_counts": family_counts,
    }


def parse_json_counts(value: str) -> dict[str, int]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in parsed.items():
        try:
            out[str(key)] = int(raw)
        except (TypeError, ValueError):
            continue
    return out


def _count_values(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(column, "") or "blank"
        counts[value] = counts.get(value, 0) + 1
    return counts


def _float_or_none(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_spin_summary(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    energies: list[tuple[float, dict[str, str]]] = []
    changed_by_element: dict[str, int] = {}
    for row in rows:
        energy = _float_or_none(row.get("energy_eV"))
        if energy is not None:
            energies.append((energy, row))
        for element, count in parse_json_counts(row.get("changed_by_element", "")).items():
            changed_by_element[element] = changed_by_element.get(element, 0) + count
    best: dict[str, object] = {}
    if energies:
        energy, row = min(energies, key=lambda item: item[0])
        best = {
            "index": row.get("index", ""),
            "run": row.get("run", ""),
            "energy_eV": energy,
            "relative_energy_eV": 0.0,
            "total_moment": _float_or_none(row.get("total_moment")),
            "max_abs_moment": _float_or_none(row.get("max_abs_moment")),
            "host_mode": row.get("host_mode", ""),
            "dopant_mode": row.get("dopant_mode", ""),
            "physics_guard_status": row.get("physics_guard_status", ""),
        }
    return {
        "rows": len(rows),
        "columns": reader.fieldnames or [],
        "status_counts": _count_values(rows, "status"),
        "mag_status_counts": _count_values(rows, "mag_status"),
        "physics_guard_counts": _count_values(rows, "physics_guard_status"),
        "host_mode_counts": _count_values(rows, "host_mode"),
        "dopant_mode_counts": _count_values(rows, "dopant_mode"),
        "energy_rows": len(energies),
        "energy_min_eV": min((item[0] for item in energies), default=None),
        "energy_max_eV": max((item[0] for item in energies), default=None),
        "best": best,
        "changed_by_element": changed_by_element,
    }


def parse_spin_atoms(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    changed = [row for row in rows if str(row.get("changed", "")).lower() in {"1", "true", "yes"}]
    physics_bad = [row for row in rows if str(row.get("physics_ok", "")).lower() in {"0", "false", "no"}]
    changed_by_element: dict[str, int] = {}
    bad_by_element: dict[str, int] = {}
    for row in changed:
        element = row.get("element", "") or "X"
        changed_by_element[element] = changed_by_element.get(element, 0) + 1
    for row in physics_bad:
        element = row.get("element", "") or "X"
        bad_by_element[element] = bad_by_element.get(element, 0) + 1
    return {
        "rows": len(rows),
        "changed_rows": len(changed),
        "physics_bad_rows": len(physics_bad),
        "changed_by_element": changed_by_element,
        "physics_bad_by_element": bad_by_element,
    }


def parse_spin_index(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    element_values: dict[str, set[float]] = {}
    for row in rows:
        raw = row.get("moments_by_atom", "")
        if not raw:
            continue
        try:
            moments = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(moments, list):
            continue
        for item in moments:
            if not isinstance(item, dict):
                continue
            element = str(item.get("element") or "X")
            value = _float_or_none(str(item.get("magmom", "")))
            if value is not None:
                element_values.setdefault(element, set()).add(round(value, 6))
    return {
        "rows": len(rows),
        "columns": reader.fieldnames or [],
        "dopant_mode_counts": _count_values(rows, "dopant_mode"),
        "host_mode_counts": _count_values(rows, "host_mode"),
        "element_moment_values": {
            element: sorted(values)
            for element, values in sorted(element_values.items())
        },
    }


def parse_tdb(path: Path) -> dict[str, object]:
    text = _read_text(path)
    elements = re.findall(r"^\s*ELEMENT\s+(\S+)", text, flags=re.MULTILINE | re.IGNORECASE)
    phases = re.findall(r"^\s*PHASE\s+(\S+)", text, flags=re.MULTILINE | re.IGNORECASE)
    return {
        "elements": elements[:20],
        "phase_count": len(phases),
        "phases": phases[:20],
    }


def scan_run(path: Path, requested_modules: list[str], max_files: int = 5000) -> RunEvidence:
    root = path.resolve()
    evidence = RunEvidence(path=root, requested_modules=requested_modules)
    if not root.exists():
        evidence.warnings.append(f"Run path does not exist: {root}")
        return evidence
    files = _find_files(root, max_files=max_files)
    if len(files) >= max_files:
        evidence.warnings.append(f"File scan stopped at --max-files={max_files}.")

    defect_cloud_summaries = _by_name(files, "defect_cloud_summary.json")
    defect_cloud_indexes = _by_name(files, "defect_cloud_index.csv")
    if defect_cloud_summaries or defect_cloud_indexes:
        evidence.add_module("VASP_PREP")
        runlists = _by_name(files, "runlist.txt")
        for item in defect_cloud_summaries[:5]:
            evidence.add_file("defect_cloud_summary", item)
        for item in defect_cloud_indexes[:5]:
            evidence.add_file("defect_cloud_index", item)
        for item in runlists[:5]:
            evidence.add_file("runlist", item)
        if defect_cloud_summaries:
            try:
                evidence.facts["defect_cloud_summary"] = parse_defect_cloud_summary(defect_cloud_summaries[0])
            except (json.JSONDecodeError, OSError) as exc:
                evidence.warnings.append(f"Could not parse defect-cloud summary: {exc}")
        if defect_cloud_indexes:
            evidence.facts["defect_cloud_index"] = parse_defect_cloud_index(defect_cloud_indexes[0])
        if runlists:
            evidence.facts["vasp_runlist"] = parse_runlist(runlists[0])

    spin_summaries = [
        item
        for item in _by_suffix(files, ".csv")
        if item.name.endswith("_run_summary.csv")
        and "spin" in item.name.lower()
        and "physics_filtered" not in item.name.lower()
    ]
    spin_atom_tables = [
        item
        for item in _by_suffix(files, ".csv")
        if item.name.endswith("_atom_moments.csv")
        and "spin" in item.name.lower()
        and "physics_filtered" not in item.name.lower()
    ]
    spin_filtered = [
        item
        for item in _by_suffix(files, ".csv")
        if "spin" in item.name.lower() and "physics_filtered" in item.name.lower()
    ]
    spin_markdown = [
        item
        for item in _by_suffix(files, ".md")
        if "spin" in item.name.lower() and ("report" in item.name.lower() or "energy" in item.name.lower())
    ]
    spin_indexes = [
        item
        for item in _by_suffix(files, ".csv")
        if item.name.lower() == "spin_index.csv"
        or (item.name.lower().endswith("_spin_index.csv") and "spin" in item.name.lower())
    ]
    magmom_dirs = [item for item in files if item.name == "MAGMOM_expanded.txt" or item.name == "MAGMOM_vasp.txt"]
    if spin_summaries or spin_atom_tables or spin_filtered or spin_markdown or spin_indexes:
        evidence.add_module("DFT")
        evidence.add_module("VASP_SPIN")
        for item in spin_summaries[:5]:
            evidence.add_file("spin_run_summary", item)
        for item in spin_atom_tables[:5]:
            evidence.add_file("spin_atom_moments", item)
        for item in spin_filtered[:5]:
            evidence.add_file("spin_physics_filtered", item)
        for item in spin_markdown[:5]:
            evidence.add_file("spin_markdown_report", item)
        for item in spin_indexes[:5]:
            evidence.add_file("spin_index", item)
        for item in magmom_dirs[:5]:
            evidence.add_file("magmom_restart_line", item)
        if spin_summaries:
            evidence.facts["vasp_spin_summary"] = parse_spin_summary(spin_summaries[0])
        if spin_atom_tables:
            evidence.facts["vasp_spin_atoms"] = parse_spin_atoms(spin_atom_tables[0])
        if spin_indexes:
            evidence.facts["vasp_spin_index"] = parse_spin_index(spin_indexes[0])

    poscars = _by_name(files, "POSCAR", "CONTCAR")
    incars = _by_name(files, "INCAR")
    kpoints = _by_name(files, "KPOINTS")
    potcars = _by_name(files, "POTCAR")
    outcars = _by_name(files, "OUTCAR", "OUTCAR.gz")
    if poscars or incars or outcars:
        evidence.add_module("DFT")
        for item in poscars[:5]:
            evidence.add_file("structure", item)
        for item in incars[:5]:
            evidence.add_file("incar", item)
        for item in kpoints[:5]:
            evidence.add_file("kpoints", item)
        for item in potcars[:5]:
            evidence.add_file("potcar", item)
        for item in outcars[:5]:
            evidence.add_file("outcar", item)
        if poscars:
            evidence.facts["dft_structure"] = parse_poscar(poscars[0])
        if incars:
            tags = parse_incar(incars[0])
            keep = ("ENCUT", "EDIFF", "EDIFFG", "ISPIN", "MAGMOM", "LDAU", "LDAUU", "GGA", "ISMEAR", "SIGMA")
            evidence.facts["dft_incar_tags"] = {key: tags[key] for key in keep if key in tags}
        if kpoints:
            evidence.facts["dft_kpoints"] = parse_kpoints(kpoints[0])
        if potcars:
            evidence.facts["dft_potcar"] = parse_potcar(potcars[0])
        if outcars:
            evidence.facts["dft_outcar"] = parse_outcar(outcars[0])

    cp2k_inputs = _by_suffix(files, ".inp")
    cp2k_logs = [item for item in _by_suffix(files, ".log", ".out") if "cp2k" in item.name.lower() or cp2k_inputs]
    xyz_files = _by_suffix(files, ".xyz")
    pos_xyz = [item for item in xyz_files if item.name.lower().endswith("-pos.xyz")]
    if cp2k_inputs or pos_xyz or cp2k_logs:
        evidence.add_module("AIMD")
        for item in cp2k_inputs[:5]:
            evidence.add_file("cp2k_input", item)
        for item in cp2k_logs[:5]:
            evidence.add_file("cp2k_log", item)
        for item in pos_xyz[:5] or xyz_files[:3]:
            evidence.add_file("trajectory_xyz", item)
        if cp2k_inputs:
            evidence.facts["cp2k_input"] = parse_cp2k_input(cp2k_inputs[0])
        if cp2k_logs:
            evidence.facts["cp2k_log"] = parse_cp2k_log(cp2k_logs[0])
        traj = (pos_xyz or xyz_files)[:1]
        if traj:
            try:
                evidence.facts["xyz_frames"] = count_xyz_frames(traj[0])
            except OSError as exc:
                evidence.warnings.append(f"Could not count XYZ frames in {traj[0]}: {exc}")

    lammps_logs = [item for item in _by_glob_name(files, r"(^log\.|\.lammps$|lammps.*\.log$)")]
    lammps_configs = [item for item in _by_suffix(files, ".json") if "config" in item.name.lower()]
    if lammps_logs or lammps_configs:
        evidence.add_module("MD")
        for item in lammps_logs[:5]:
            evidence.add_file("lammps_log", item)
        for item in lammps_configs[:5]:
            evidence.add_file("md_config", item)
        if lammps_logs:
            evidence.facts["lammps_log"] = parse_lammps_log(lammps_logs[0])
        if lammps_configs:
            try:
                evidence.facts["md_config"] = json.loads(lammps_configs[0].read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                evidence.warnings.append(f"Could not parse JSON config: {lammps_configs[0]}")

    json_files = _by_suffix(files, ".json")
    gk_configs = [
        item
        for item in json_files
        if "config" in item.name.lower() and ("gk" in item.name.lower() or "green" in item.name.lower())
    ]
    gk_plans = _by_name(files, "gk_plan.json")
    gk_validations = _by_name(files, "gk_validation_summary.json")
    gk_seed_summaries = _by_name(files, "gk_seed_summary.csv")
    gk_temperature_tables = _by_name(files, "thermal_conductivity_T.csv")
    gk_manifests = _by_name(files, "gk_manifest.csv")
    gk_hcacf = _by_name(files, "heatflux_hcacf.dat")
    rnemd_configs = [
        item
        for item in json_files
        if "config" in item.name.lower() and ("rnemd" in item.name.lower() or "nemd" in item.name.lower())
    ]
    rnemd_plans = _by_name(files, "rnemd_plan.json")
    rnemd_validations = _by_name(files, "rnemd_validation_summary.json")
    rnemd_seed_summaries = _by_name(files, "rnemd_seed_summary.csv")
    rnemd_temperature_tables = _by_name(files, "thermal_conductivity_rnemd_T.csv")
    rnemd_profiles = _by_name(files, "rnemd_temperature_profiles.csv")
    rnemd_manifests = _by_name(files, "rnemd_manifest.tsv")
    if (
        gk_configs
        or gk_plans
        or gk_validations
        or gk_seed_summaries
        or gk_temperature_tables
        or gk_hcacf
        or rnemd_configs
        or rnemd_plans
        or rnemd_validations
        or rnemd_seed_summaries
        or rnemd_temperature_tables
        or rnemd_profiles
    ):
        evidence.add_module("MD")
        evidence.add_module("TRANSPORT")
        for key, items in (
            ("gk_config", gk_configs),
            ("gk_plan", gk_plans),
            ("gk_validation", gk_validations),
            ("gk_seed_summary", gk_seed_summaries),
            ("gk_temperature_table", gk_temperature_tables),
            ("gk_manifest", gk_manifests),
            ("gk_hcacf", gk_hcacf),
            ("rnemd_config", rnemd_configs),
            ("rnemd_plan", rnemd_plans),
            ("rnemd_validation", rnemd_validations),
            ("rnemd_seed_summary", rnemd_seed_summaries),
            ("rnemd_temperature_table", rnemd_temperature_tables),
            ("rnemd_profile_summary", rnemd_profiles),
            ("rnemd_manifest", rnemd_manifests),
        ):
            for item in items[:5]:
                evidence.add_file(key, item)
        if gk_configs:
            try:
                evidence.facts["gk_config"] = parse_transport_config(gk_configs[0], "gk")
            except (json.JSONDecodeError, OSError) as exc:
                evidence.warnings.append(f"Could not parse GK config: {exc}")
        if gk_plans:
            try:
                evidence.facts["gk_plan"] = parse_transport_plan(gk_plans[0], "gk")
            except (json.JSONDecodeError, OSError) as exc:
                evidence.warnings.append(f"Could not parse GK plan: {exc}")
        if gk_seed_summaries:
            evidence.facts["gk_seed_summary"] = parse_transport_seed_summary(gk_seed_summaries[0], "gk")
        if gk_temperature_tables:
            evidence.facts["gk_temperature_table"] = parse_transport_temperature_table(gk_temperature_tables[0], "gk")
        if gk_validations:
            try:
                evidence.facts["gk_validation"] = parse_transport_validation(gk_validations[0], "gk")
            except (json.JSONDecodeError, OSError) as exc:
                evidence.warnings.append(f"Could not parse GK validation: {exc}")
        if rnemd_configs:
            try:
                evidence.facts["rnemd_config"] = parse_transport_config(rnemd_configs[0], "rnemd")
            except (json.JSONDecodeError, OSError) as exc:
                evidence.warnings.append(f"Could not parse rNEMD config: {exc}")
        if rnemd_plans:
            try:
                evidence.facts["rnemd_plan"] = parse_transport_plan(rnemd_plans[0], "rnemd")
            except (json.JSONDecodeError, OSError) as exc:
                evidence.warnings.append(f"Could not parse rNEMD plan: {exc}")
        if rnemd_seed_summaries:
            evidence.facts["rnemd_seed_summary"] = parse_transport_seed_summary(rnemd_seed_summaries[0], "rnemd")
        if rnemd_temperature_tables:
            evidence.facts["rnemd_temperature_table"] = parse_transport_temperature_table(
                rnemd_temperature_tables[0],
                "rnemd",
            )
        if rnemd_profiles:
            evidence.facts["rnemd_profile_summary"] = parse_csv_summary(rnemd_profiles[0])
        if rnemd_validations:
            try:
                evidence.facts["rnemd_validation"] = parse_transport_validation(rnemd_validations[0], "rnemd")
            except (json.JSONDecodeError, OSError) as exc:
                evidence.warnings.append(f"Could not parse rNEMD validation: {exc}")

    extxyz = _by_suffix(files, ".extxyz")
    model_files = _by_suffix(files, ".model", ".pt")
    manifests = [item for item in _by_suffix(files, ".csv") if "manifest" in item.name.lower()]
    if extxyz or model_files or manifests:
        evidence.add_module("MLIP")
        for item in extxyz[:5]:
            evidence.add_file("extxyz", item)
        for item in model_files[:5]:
            evidence.add_file("model", item)
        for item in manifests[:5]:
            evidence.add_file("dataset_manifest", item)
        if extxyz:
            try:
                evidence.facts["extxyz_frames"] = count_xyz_frames(extxyz[0])
            except OSError as exc:
                evidence.warnings.append(f"Could not count EXTXYZ frames in {extxyz[0]}: {exc}")
        if manifests:
            evidence.facts["dataset_manifest"] = parse_csv_summary(manifests[0])

    tdb_files = _by_suffix(files, ".tdb")
    calphad_csv = [item for item in _by_suffix(files, ".csv") if "calphad" in item.name.lower()]
    if tdb_files or calphad_csv:
        evidence.add_module("CALPHAD")
        for item in tdb_files[:5]:
            evidence.add_file("tdb", item)
        for item in calphad_csv[:5]:
            evidence.add_file("calphad_csv", item)
        if tdb_files:
            evidence.facts["tdb"] = parse_tdb(tdb_files[0])

    moose_inputs = _by_suffix(files, ".i")
    moose_csv = [item for item in _by_suffix(files, ".csv") if "moose" in item.name.lower() or "material" in item.name.lower()]
    if moose_inputs or moose_csv:
        evidence.add_module("MOOSE")
        for item in moose_inputs[:5]:
            evidence.add_file("moose_input", item)
        for item in moose_csv[:5]:
            evidence.add_file("moose_material_csv", item)
        if moose_csv:
            evidence.facts["moose_material_csv"] = parse_csv_summary(moose_csv[0])

    thermo_csv = [item for item in _by_suffix(files, ".csv") if item.name in {"thermo_functions_grid.csv", "all_T_summary.csv"}]
    qha_dat = [item for item in _by_suffix(files, ".dat") if "thermal" in item.name.lower() or "qha" in item.name.lower()]
    if thermo_csv or qha_dat:
        evidence.add_module("QHA")
        for item in thermo_csv[:5]:
            evidence.add_file("thermo_csv", item)
        for item in qha_dat[:5]:
            evidence.add_file("qha_dat", item)
        if thermo_csv:
            evidence.facts["thermo_csv"] = parse_csv_summary(thermo_csv[0])

    if requested_modules:
        for module in requested_modules:
            if module not in evidence.detected_modules:
                evidence.warnings.append(f"Requested module {module} was not detected from files.")

    return evidence


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={_format_value(val)}" for key, val in value.items())
    return str(value)


def _sentence_join(parts: Iterable[str]) -> str:
    return "; ".join(part for part in parts if part)


def format_rule_lines(modules: list[str]) -> list[str]:
    modules_to_write = modules or ["DFT", "MD", "MLIP"]
    lines = [f"- {rule}" for rule in COMMON_FORMAT_RULES]
    seen: set[str] = set()
    for module in modules_to_write:
        if module in seen:
            continue
        seen.add(module)
        rules = METHOD_FORMAT_RULES.get(module, [])
        if not rules:
            continue
        lines.append(f"- {module}: " + " ".join(rules))
    return lines


def parse_magmom_summary(value: str) -> dict[str, object]:
    tokens = [token for token in re.split(r"\s+", value.strip()) if token]
    expanded: list[float] = []
    for token in tokens:
        if "*" in token:
            left, right = token.split("*", 1)
            try:
                count = int(float(left))
                moment = float(right)
            except ValueError:
                continue
            expanded.extend([moment] * max(count, 0))
            continue
        try:
            expanded.append(float(token))
        except ValueError:
            continue
    counts: dict[str, int] = {}
    for moment in expanded:
        label = f"{moment:g}"
        counts[label] = counts.get(label, 0) + 1
    return {
        "n_moments": len(expanded),
        "values": counts,
    }


def compact_incar_methods(tags: dict[str, str]) -> str:
    if not tags:
        return ""
    sentences: list[str] = []
    if tags.get("ENCUT"):
        sentences.append(f"The plane-wave cutoff was {tags['ENCUT']} eV (ENCUT={tags['ENCUT']})")
    if tags.get("EDIFF") or tags.get("EDIFFG"):
        convergence = []
        if tags.get("EDIFF"):
            convergence.append(f"electronic threshold {tags['EDIFF']} eV")
        if tags.get("EDIFFG"):
            convergence.append(f"ionic threshold {tags['EDIFFG']} eV A^-1")
        sentences.append("Convergence criteria used " + " and ".join(convergence))
    if tags.get("GGA"):
        sentences.append(f"The exchange-correlation tag was GGA={tags['GGA']}")
    if tags.get("LDAU"):
        u_bits = [f"LDAU={tags['LDAU']}"]
        for key in ("LDAUU", "LDAUJ", "LDAUL"):
            if tags.get(key):
                u_bits.append(f"{key}={tags[key]}")
        sentences.append("On-site Hubbard corrections were controlled by " + ", ".join(u_bits))
    if tags.get("ISMEAR") or tags.get("SIGMA"):
        smear = []
        if tags.get("ISMEAR"):
            smear.append(f"ISMEAR={tags['ISMEAR']}")
        if tags.get("SIGMA"):
            smear.append(f"SIGMA={tags['SIGMA']} eV")
        sentences.append("Electronic occupations used " + " and ".join(smear))
    if tags.get("ISPIN"):
        sentences.append(f"Spin polarization was enabled with ISPIN={tags['ISPIN']}")
    if tags.get("MAGMOM"):
        magmom = parse_magmom_summary(tags["MAGMOM"])
        if magmom["n_moments"]:
            sentences.append(
                f"Initial site moments were specified for {magmom['n_moments']} sites "
                f"with values {_format_value(magmom['values'])}"
            )
    return ". ".join(sentences) + ("." if sentences else "")


def compact_hpc_value(value: Any, *, show_private: bool) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(compact_hpc_value(item, show_private=show_private) for item in value if item not in (None, ""))
    if isinstance(value, dict):
        return "; ".join(
            f"{key}={compact_hpc_value(val, show_private=show_private)}"
            for key, val in value.items()
            if val not in (None, "", [], {})
        )
    text = str(value)
    if show_private or "/" not in text:
        return text
    return Path(text).name or text


def hpc_profile_summary(
    config: dict[str, Any],
    profile_name: str,
    *,
    label: str,
    show_private: bool,
) -> str:
    profile = (config.get("profiles") or {}).get(profile_name)
    if not isinstance(profile, dict):
        return ""
    bits: list[str] = []
    modules = profile.get("modules") or profile.get("module_commands")
    if modules:
        if isinstance(modules, list):
            bits.append("modules " + ", ".join(str(item) for item in modules if item))
        else:
            bits.append("modules " + str(modules))
    versions = profile.get("resolved_versions")
    if versions:
        bits.append("versions " + compact_hpc_value(versions, show_private=show_private))
    for key in ("executable", "lammps_executable", "python", "module", "env_path"):
        if profile.get(key):
            bits.append(f"{key} {compact_hpc_value(profile[key], show_private=show_private)}")
            break
    candidates = profile.get("executable_candidates")
    if candidates:
        bits.append("executables " + compact_hpc_value(candidates, show_private=show_private))
    if not bits:
        return ""
    return f"{label}: " + "; ".join(bits)


def _transport_settings_sentence(label: str, config: object, plan: object) -> str:
    if not isinstance(config, dict) and not isinstance(plan, dict):
        return ""
    settings = config.get("settings", {}) if isinstance(config, dict) else {}
    runtime = plan.get("runtime_estimate", {}) if isinstance(plan, dict) else {}
    bits: list[str] = []
    if isinstance(config, dict):
        temperatures = config.get("temperatures_K") or []
        if isinstance(temperatures, list) and temperatures:
            bits.append("temperatures " + ", ".join(f"{float(temp):g} K" for temp in temperatures[:12]))
    if isinstance(settings, dict):
        for key in (
            "pair_style_backend",
            "runtime_profile",
            "heat_flux_suffix",
            "timestep_ps",
            "nvt_preequilibration_ps",
            "nve_time_ps",
            "sample_interval_ps",
            "correlation_time_ps",
            "plateau_window_ps",
            "run_time_ps",
            "replicate",
            "direction",
            "nbin",
            "swap_every",
            "n_seeds",
        ):
            if key in settings:
                bits.append(f"{key}={_format_value(settings[key])}")
    if isinstance(plan, dict):
        if plan.get("n_stages") is not None:
            bits.append(f"stages={plan['n_stages']}")
        if plan.get("n_seeds_per_temperature") is not None:
            bits.append(f"seeds_per_T={plan['n_seeds_per_temperature']}")
    if isinstance(runtime, dict):
        for key in ("estimated_steps_per_hour", "estimated_walltime_hours_per_stage", "estimated_elapsed_hours_at_array_limit"):
            if key in runtime:
                bits.append(f"{key}={_format_value(runtime[key])}")
    return f"{label}: " + "; ".join(bits) if bits else ""


def _transport_result_sentences(label: str, table: object, validation: object) -> list[str]:
    lines: list[str] = []
    if isinstance(table, dict) and table.get("records"):
        records = table["records"]
        if isinstance(records, list):
            pieces = []
            for record in records[:6]:
                if not isinstance(record, dict):
                    continue
                temperature = record.get("temperature_K")
                conductivity = record.get("k_W_mK")
                if temperature is not None and conductivity is not None:
                    pieces.append(f"T={float(temperature):g} K k={float(conductivity):.4g} W/m/K")
            if pieces:
                lines.append(f"{label} k(T) table contained " + "; ".join(pieces) + ".")
    if isinstance(validation, dict) and validation.get("reports"):
        reports = validation["reports"]
        if isinstance(reports, list):
            pieces = []
            for report in reports[:6]:
                if not isinstance(report, dict):
                    continue
                temperature = report.get("temperature_K")
                status = report.get("status")
                conductivity = report.get("k_W_mK")
                metrics = []
                for key, name in (
                    ("axis_spread_percent", "axis spread"),
                    ("seed_cv_percent", "seed CV"),
                    ("late_drift_percent", "late drift"),
                    ("slope_mismatch_percent", "slope mismatch"),
                ):
                    if report.get(key) is not None:
                        metrics.append(f"{name}={float(report[key]):.1f}%")
                head = []
                if temperature is not None:
                    head.append(f"T={float(temperature):g} K")
                if conductivity is not None:
                    head.append(f"k={float(conductivity):.4g} W/m/K")
                if status:
                    head.append(f"status={status}")
                pieces.append(" ".join(head + metrics))
            if pieces:
                lines.append(f"{label} validation reported " + "; ".join(pieces) + ".")
    return lines


def build_hpc_context(path: Path | None, *, show_private: bool) -> dict[str, Any]:
    config_path = find_config_path(path)
    if config_path is None:
        return {"found": False}
    config = load_hpc_config(config_path)
    return {
        "found": True,
        "path": str(config_path.resolve()) if show_private else config_path.name,
        "site": config.get("site") or "",
        "profiles": {
            "vasp": hpc_profile_summary(config, "vasp_cpu", label="VASP", show_private=show_private),
            "lammps": hpc_profile_summary(config, "lammps_md_engine", label="MD engine", show_private=show_private),
            "lammps_gk": hpc_profile_summary(config, "lammps_gk_mliap", label="GK ML-IAP", show_private=show_private),
            "cp2k": hpc_profile_summary(config, "cp2k", label="AIMD", show_private=show_private),
            "phonopy": hpc_profile_summary(config, "phonopy", label="QHA", show_private=show_private),
        },
    }


def hpc_methods_paragraph(modules: list[str], hpc_context: dict[str, Any] | None) -> str:
    if not hpc_context or not hpc_context.get("found"):
        return ""
    modules_to_write = set(modules)
    if not modules_to_write:
        modules_to_write = {"DFT", "VASP_SPIN", "AIMD", "MD", "QHA"}
    profile_keys = []
    if {"DFT", "VASP_SPIN", "VASP_PREP"} & modules_to_write:
        profile_keys.append("vasp")
    if {"MD", "TRANSPORT"} & modules_to_write:
        profile_keys.append("lammps")
    if "TRANSPORT" in modules_to_write:
        profile_keys.append("lammps_gk")
    if "AIMD" in modules_to_write:
        profile_keys.append("cp2k")
    if "QHA" in modules_to_write:
        profile_keys.append("phonopy")
    details = [hpc_context["profiles"].get(key, "") for key in profile_keys]
    details = [detail for detail in details if detail]
    if not details:
        return ""
    site = hpc_context.get("site") or "the configured HPC environment"
    return (
        f"Site-specific runtime information was taken from the local Atomi HPC configuration for {site}. "
        + " ".join(details)
        + "."
    )


def methods_paragraphs(
    evidences: list[RunEvidence],
    modules: list[str],
    hpc_context: dict[str, Any] | None = None,
) -> list[str]:
    paragraphs: list[str] = []
    modules_to_write = modules or sorted({module for ev in evidences for module in ev.detected_modules})

    hpc_paragraph = hpc_methods_paragraph(modules_to_write, hpc_context)
    if hpc_paragraph:
        paragraphs.append(hpc_paragraph)

    if "VASP_PREP" in modules_to_write:
        summary = _first_fact(evidences, "defect_cloud_summary", {})
        index = _first_fact(evidences, "defect_cloud_index", {})
        runlist = _first_fact(evidences, "vasp_runlist", {})
        details = []
        if isinstance(summary, dict) and summary:
            if summary.get("n_seed_motifs") is not None:
                details.append(f"{summary['n_seed_motifs']} seed motifs")
            if summary.get("n_candidate_runs") is not None:
                details.append(f"{summary['n_candidate_runs']} candidate VASP folders")
            if summary.get("per_motif_requested") is not None:
                details.append(f"{summary['per_motif_requested']} requested variants per motif")
            if summary.get("seed") is not None:
                details.append(f"random seed {summary['seed']}")
            defaults = summary.get("defaults", {})
            if isinstance(defaults, dict) and defaults:
                prep_settings = []
                for key in ("random_amp_A", "structured_amp_A", "bias_species", "bias_amp_A", "mixed_amp_A", "iso_strains"):
                    if key in defaults:
                        prep_settings.append(f"{key}={_format_value(defaults[key])}")
                if prep_settings:
                    details.append("generation settings " + "; ".join(prep_settings))
            family_totals = summary.get("family_totals", {})
            if isinstance(family_totals, dict) and family_totals:
                details.append("variant families " + _format_value(family_totals))
        elif isinstance(index, dict) and index:
            details.append(f"{index.get('rows', 'unknown')} candidate rows")
            if index.get("motif_count") is not None:
                details.append(f"{index['motif_count']} motifs")
            family_counts = index.get("family_counts", {})
            if isinstance(family_counts, dict) and family_counts:
                details.append("variant families " + _format_value(family_counts))
        if isinstance(runlist, dict) and runlist:
            details.append(f"array-run index runlist.txt with {runlist.get('rows', 'unknown')} entries")
        paragraphs.append(
            "Defect-seed and candidate electronic-structure folders were generated by "
            "starting from relaxed motif structures, preserving atom ordering, copying "
            "the calculation template files, and writing one calculation directory per "
            "candidate structure. "
            + (_sentence_join(details) + ". " if details else "")
            + "For manuscript use, describe the motif source, charge or spin labels, "
            "which perturbation families were retained, and how the indexed folders "
            "were submitted and screened."
        )

    if "DFT" in modules_to_write:
        tags = _first_fact(evidences, "dft_incar_tags", {})
        structure = _first_fact(evidences, "dft_structure", {})
        kpoints = _first_fact(evidences, "dft_kpoints", {})
        potcar = _first_fact(evidences, "dft_potcar", {})
        outcar = _first_fact(evidences, "dft_outcar", {})
        details = []
        if isinstance(outcar, dict) and outcar.get("vasp_version"):
            details.append(f"the VASP executable reported {outcar['vasp_version']}")
        if structure:
            if structure.get("comment"):
                details.append(f"the representative POSCAR label was {structure['comment']}")
            composition = structure.get("formula", "not parsed")
            natoms = structure.get("natoms", "unknown")
            details.append(f"the representative supercell contained {composition} ({natoms} atoms)")
        if kpoints:
            k_bits = []
            if kpoints.get("mode"):
                k_bits.append(str(kpoints["mode"]))
            if kpoints.get("mesh"):
                k_bits.append(str(kpoints["mesh"]))
            if k_bits:
                details.append("Brillouin-zone sampling used " + " ".join(k_bits))
        if isinstance(potcar, dict) and potcar.get("titles"):
            details.append("PAW datasets were " + _format_value(potcar["titles"]))
        incar_text = compact_incar_methods(tags) if isinstance(tags, dict) else ""
        paragraphs.append(
            "Electronic-structure calculations were carried out with VASP using PAW "
            "projector datasets and spin-polarized input settings. "
            + (_sentence_join(details) + ". " if details else "")
            + (incar_text + " " if incar_text else "")
            + "These fields are reported explicitly because reproducibility of "
            "correlated, magnetic oxides depends on the potential set, cutoff, "
            "k-point mesh, convergence criteria, smearing, and magnetic initialization."
        )

    if "VASP_SPIN" in modules_to_write:
        spin = _first_fact(evidences, "vasp_spin_summary", {})
        atoms = _first_fact(evidences, "vasp_spin_atoms", {})
        spin_index = _first_fact(evidences, "vasp_spin_index", {})
        details = []
        generation_details = []
        if isinstance(spin_index, dict) and spin_index:
            if spin_index.get("rows") is not None:
                generation_details.append(f"{spin_index['rows']} generated spin inputs")
            dopant_counts = spin_index.get("dopant_mode_counts", {})
            if isinstance(dopant_counts, dict) and dopant_counts:
                generation_details.append("dopant enumeration " + _format_value(dopant_counts))
            host_counts = spin_index.get("host_mode_counts", {})
            if isinstance(host_counts, dict) and host_counts:
                generation_details.append("host enumeration " + _format_value(host_counts))
            moment_values = spin_index.get("element_moment_values", {})
            if isinstance(moment_values, dict) and moment_values:
                generation_details.append("initial element moment values " + _format_value(moment_values))
        if isinstance(spin, dict) and spin:
            if spin.get("rows") is not None:
                details.append(f"{spin['rows']} indexed spin configurations")
            if spin.get("energy_rows") is not None:
                details.append(f"{spin['energy_rows']} configurations with parsed energies")
            physics_counts = spin.get("physics_guard_counts", {})
            if isinstance(physics_counts, dict) and physics_counts:
                details.append("physics-guard counts " + _format_value(physics_counts))
            host_counts = spin.get("host_mode_counts", {})
            if isinstance(host_counts, dict) and host_counts:
                details.append("host spin labels " + _format_value(host_counts))
            dopant_counts = spin.get("dopant_mode_counts", {})
            if isinstance(dopant_counts, dict) and dopant_counts:
                details.append("dopant spin labels " + _format_value(dopant_counts))
        if isinstance(atoms, dict) and atoms:
            if atoms.get("changed_rows") is not None:
                details.append(f"{atoms['changed_rows']} atom-level initial/final moment changes")
            bad = atoms.get("physics_bad_by_element", {})
            if isinstance(bad, dict) and bad:
                details.append("physics-guard atom flags " + _format_value(bad))
        paragraphs.append(
            "Spin-configuration screening was performed by enumerating initial "
            "MAGMOM patterns for the magnetic sublattice and running each pattern as a "
            "separate VASP calculation. Final, or for interrupted jobs the last complete, "
            "site-resolved magnetic moments were extracted from OUTCAR-like outputs and "
            "compared with the initial MAGMOM pattern. Element-resolved moment labels and "
            "physics guards were used to separate configurations that retained the intended "
            "nominal moment states from configurations that relaxed to unintended valence "
            "or spin states. "
            + (
                "The spin-generation index records "
                + _sentence_join(generation_details)
                + ". "
                if generation_details
                else ""
            )
            + (_sentence_join(details) + ". " if details else "")
            + "Energy comparisons should therefore be interpreted using both the parsed "
            "total energies and the final moment-state classification."
        )

    if "AIMD" in modules_to_write:
        cp2k_info = _first_fact(evidences, "cp2k_input", {})
        frames = _first_fact(evidences, "xyz_frames", None)
        details = []
        if cp2k_info:
            details.append(_format_value(cp2k_info))
        if frames is not None:
            details.append(f"{frames} trajectory frames")
        paragraphs.append(
            "Ab initio molecular dynamics and related trajectory calculations were "
            "summarized from input, log, and trajectory files. "
            + (_sentence_join(details) + ". " if details else "")
            + "The final Methods text should state ensemble, timestep, thermostat or "
            "barostat choices, total simulated time, equilibration protocol, and any "
            "collective variables or restraints."
        )

    if "MD" in modules_to_write:
        log_info = _first_fact(evidences, "lammps_log", {})
        details = _format_value(log_info) if log_info else ""
        paragraphs.append(
            "Classical or machine-learning-potential molecular dynamics calculations "
            "were summarized from configuration files and thermo logs. "
            + (details + ". " if details else "")
            + "Report the potential/model, ensemble sequence, timestep, temperature "
            "schedule, equilibration criteria, production length, and uncertainty "
            "estimation procedure."
        )

    if "TRANSPORT" in modules_to_write:
        gk_config = _first_fact(evidences, "gk_config", {})
        gk_plan = _first_fact(evidences, "gk_plan", {})
        rnemd_config = _first_fact(evidences, "rnemd_config", {})
        rnemd_plan = _first_fact(evidences, "rnemd_plan", {})
        details = [
            detail
            for detail in (
                _transport_settings_sentence("Green-Kubo", gk_config, gk_plan),
                _transport_settings_sentence("reverse NEMD", rnemd_config, rnemd_plan),
            )
            if detail
        ]
        paragraphs.append(
            "Thermal-conductivity calculations were inventoried as finite-temperature "
            "LAMMPS workflows rather than generic MD logs. Green-Kubo runs should be "
            "reported through the equilibrated-cell source, ensemble sequence, heat-current "
            "definition, HCACF sampling/correlation window, plateau-selection rule, seed "
            "averaging, and validation metrics. Reverse-NEMD runs should be reported through "
            "the replicated slab geometry, heat-flow direction, binning, swap interval, "
            "gradient-fit windows, imposed-flux convention, and mirrored-slope consistency. "
            + (_sentence_join(details) + ". " if details else "")
            + "Following the UN thermal-transport workflow, the manuscript should compare "
            "Green-Kubo and non-equilibrium or mode-analysis estimates before assigning a "
            "mechanism to phonon-like, defect-limited, or strongly anharmonic transport."
        )

    if "MLIP" in modules_to_write:
        frame_count = _first_fact(evidences, "extxyz_frames", None)
        manifest = _first_fact(evidences, "dataset_manifest", {})
        details = []
        if frame_count is not None:
            details.append(f"{frame_count} frames in representative extxyz file")
        if manifest:
            details.append("manifest " + _format_value(manifest))
        paragraphs.append(
            "Machine-learning interatomic-potential data handling was drafted from "
            "extxyz datasets, manifests, model files, and outlier reports. "
            + (_sentence_join(details) + ". " if details else "")
            + "The paper draft should distinguish training, validation, and test data; "
            "state weighting/oversampling choices; and report validation metrics and "
            "outlier-cleaning rules."
        )

    if "CALPHAD" in modules_to_write:
        tdb = _first_fact(evidences, "tdb", {})
        details = _format_value(tdb) if tdb else ""
        paragraphs.append(
            "Thermodynamic database or tabulated thermodynamic inputs were inventoried "
            "from TDB and CSV files. "
            + (details + ". " if details else "")
            + "State whether database values, DFT/MD-derived values, or user-supplied "
            "values have priority when quantities overlap."
        )

    if "MOOSE" in modules_to_write:
        material = _first_fact(evidences, "moose_material_csv", {})
        details = _format_value(material) if material else ""
        paragraphs.append(
            "Multiphysics handoff files were summarized from input decks and material "
            "property tables. "
            + (details + ". " if details else "")
            + "The draft should identify which properties were transferred, their "
            "temperature ranges, interpolation method, unit conversions, and any "
            "literature values used to fill missing columns."
        )

    if "QHA" in modules_to_write:
        thermo = _first_fact(evidences, "thermo_csv", {})
        details = _format_value(thermo) if thermo else ""
        paragraphs.append(
            "Quasi-harmonic and MD thermodynamic post-processing was summarized from "
            "temperature-grid tables and QHA outputs. "
            + (details + ". " if details else "")
            + "Report normalization basis, low-temperature splice or anchor choices, "
            "bootstrap settings, and uncertainty bands."
        )

    if "SCATTERING" in modules_to_write:
        paragraphs.append(
            "Scattering and spectroscopy digital-twin calculations should be reported "
            "with the trajectory source, frame-selection window, absorber or pair "
            "definitions, Q/r/k ranges, instrument corrections, and comparison metric."
        )

    return paragraphs


def _first_fact(evidences: list[RunEvidence], key: str, default: object) -> object:
    for evidence in evidences:
        if key in evidence.facts:
            return evidence.facts[key]
    return default


def result_lines(evidences: list[RunEvidence]) -> list[str]:
    lines: list[str] = []
    for evidence in evidences:
        modules = ", ".join(evidence.detected_modules) or "no module detected"
        run_label = evidence.path.name or str(evidence.path)
        lines.append(f"For `{run_label}`, Atomi detected {modules}.")
        prep = evidence.facts.get("defect_cloud_summary") or evidence.facts.get("defect_cloud_index")
        if isinstance(prep, dict) and prep:
            bits = []
            if "n_seed_motifs" in prep:
                bits.append(f"{prep['n_seed_motifs']} seed motifs")
            elif "motif_count" in prep:
                bits.append(f"{prep['motif_count']} seed motifs")
            if "n_candidate_runs" in prep:
                bits.append(f"{prep['n_candidate_runs']} candidate folders")
            elif "rows" in prep:
                bits.append(f"{prep['rows']} candidate rows")
            if prep.get("family_totals"):
                bits.append("families " + _format_value(prep["family_totals"]))
            elif prep.get("family_counts"):
                bits.append("families " + _format_value(prep["family_counts"]))
            if bits:
                lines.append("The VASP preparation stage generated " + "; ".join(bits) + ".")
        dft = evidence.facts.get("dft_outcar")
        if isinstance(dft, dict):
            bits = []
            if "vasp_version" in dft:
                bits.append(f"VASP version {dft['vasp_version']}")
            if "final_energy_eV" in dft:
                bits.append(f"final DFT energy {dft['final_energy_eV']:.8g} eV")
            if "final_volume_A3" in dft:
                bits.append(f"volume {dft['final_volume_A3']:.8g} A^3")
            if "nions" in dft:
                bits.append(f"{dft['nions']} ions")
            if bits:
                lines.append("The representative electronic-structure output reported " + "; ".join(bits) + ".")
        spin_index = evidence.facts.get("vasp_spin_index")
        if isinstance(spin_index, dict) and spin_index:
            bits = []
            if spin_index.get("rows") is not None:
                bits.append(f"{spin_index['rows']} generated spin inputs")
            moment_values = spin_index.get("element_moment_values", {})
            if isinstance(moment_values, dict) and moment_values:
                bits.append("initial moments " + _format_value(moment_values))
            dopant_counts = spin_index.get("dopant_mode_counts", {})
            if isinstance(dopant_counts, dict) and dopant_counts:
                bits.append("dopant modes " + _format_value(dopant_counts))
            host_counts = spin_index.get("host_mode_counts", {})
            if isinstance(host_counts, dict) and host_counts:
                bits.append("host modes " + _format_value(host_counts))
            if bits:
                lines.append("The spin-generation index contained " + "; ".join(bits) + ".")
        spin = evidence.facts.get("vasp_spin_summary")
        if isinstance(spin, dict) and spin:
            bits = [
                f"{spin.get('rows', 'unknown')} spin configurations",
                f"{spin.get('energy_rows', 'unknown')} with parsed energies",
            ]
            physics_counts = spin.get("physics_guard_counts", {})
            if isinstance(physics_counts, dict) and physics_counts:
                bits.append("physics guard " + _format_value(physics_counts))
            best = spin.get("best", {})
            if isinstance(best, dict) and best:
                best_bits = []
                if best.get("run"):
                    best_bits.append(f"lowest parsed run `{best['run']}`")
                if best.get("energy_eV") is not None:
                    best_bits.append(f"E={float(best['energy_eV']):.8g} eV")
                if best.get("total_moment") is not None:
                    best_bits.append(f"total moment={float(best['total_moment']):.6g}")
                if best_bits:
                    bits.append("; ".join(best_bits))
            lines.append("The spin-screening table contained " + "; ".join(bits) + ".")
        spin_atoms = evidence.facts.get("vasp_spin_atoms")
        if isinstance(spin_atoms, dict) and spin_atoms:
            atom_bits = []
            if spin_atoms.get("changed_rows") is not None:
                atom_bits.append(f"{spin_atoms['changed_rows']} atom-level moment changes")
            bad = spin_atoms.get("physics_bad_by_element", {})
            if isinstance(bad, dict) and bad:
                atom_bits.append("physics-guard atom flags " + _format_value(bad))
            if atom_bits:
                lines.append("The atom-resolved moment table showed " + "; ".join(atom_bits) + ".")
        cp2k = evidence.facts.get("cp2k_log")
        if isinstance(cp2k, dict) and cp2k:
            lines.append("The AIMD log summary was " + _format_value(cp2k) + ".")
        lammps = evidence.facts.get("lammps_log")
        if isinstance(lammps, dict) and lammps:
            lines.append("The MD thermo summary was " + _format_value(lammps) + ".")
        lines.extend(
            _transport_result_sentences(
                "Green-Kubo",
                evidence.facts.get("gk_temperature_table"),
                evidence.facts.get("gk_validation"),
            )
        )
        lines.extend(
            _transport_result_sentences(
                "reverse NEMD",
                evidence.facts.get("rnemd_temperature_table"),
                evidence.facts.get("rnemd_validation"),
            )
        )
        thermo = evidence.facts.get("thermo_csv")
        if isinstance(thermo, dict) and thermo:
            lines.append("The thermodynamic table summary was " + _format_value(thermo) + ".")
        if evidence.warnings:
            for warning in evidence.warnings:
                lines.append(f"Check before manuscript use: {warning}")
    return lines


def evidence_table(evidences: list[RunEvidence]) -> list[str]:
    lines = ["| Run | Requested | Detected | Key files |", "| --- | --- | --- | --- |"]
    for evidence in evidences:
        key_files = []
        for key, values in sorted(evidence.files.items()):
            if values:
                key_files.append(f"{key}: {', '.join(values[:3])}")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(evidence.path),
                    ", ".join(evidence.requested_modules) or "auto",
                    ", ".join(evidence.detected_modules) or "none",
                    "<br>".join(key_files) if key_files else "none",
                ]
            )
            + " |"
        )
    return lines


def compose_markdown(
    evidences: list[RunEvidence],
    modules: list[str],
    title: str,
    material: str,
    study_label: str,
    notes: list[str],
    include_style_note: bool,
    hpc_context: dict[str, Any] | None = None,
    include_inventory: bool = True,
) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    heading = title or "Atomi Draft Entry"
    parts: list[str] = [
        f"## {heading}",
        "",
        f"- Generated: {timestamp}",
        f"- Study label: {study_label or 'not specified'}",
        f"- Material/system: {material or 'not specified'}",
        f"- Requested modules: {', '.join(modules) or 'auto-detect'}",
        "",
    ]
    if notes:
        parts.extend(["### User Notes", ""])
        parts.extend(f"- {note}" for note in notes)
        parts.append("")
    if include_style_note:
        parts.extend(["### Editorial Notes", ""])
        parts.append(
            "This draft follows common computational-materials writing practice: "
            "Methods prioritize reproducibility, software/input settings, convergence, "
            "sampling protocols, and data/code availability; Results remain brief and "
            "number-first, with unverified interpretation left as placeholders."
        )
        parts.append("")
        for ref in STYLE_REFERENCES:
            parts.append(f"- {ref['label']}: {ref['url']} ({ref['note']})")
        parts.append("")
        parts.extend(["### Manuscript Format Rules", ""])
        parts.extend(format_rule_lines(modules))
        parts.append("")
    parts.extend(["### Methods", ""])
    for paragraph in methods_paragraphs(evidences, modules, hpc_context=hpc_context):
        parts.append(paragraph)
        parts.append("")
    parts.extend(["### Results", ""])
    result = result_lines(evidences)
    parts.extend(result if result else ["No parseable run evidence was found."])
    parts.append("")
    if include_inventory:
        parts.extend(["### Evidence Inventory", ""])
        parts.extend(evidence_table(evidences))
        parts.append("")
    parts.extend(["### Verification Checklist", ""])
    parts.extend(
        [
            "- Software names, versions, compilation options, and citation requirements.",
            "- Functional, potential/basis, cutoff, k-point, convergence, and spin settings.",
            "- Ensemble definitions, timestep, thermostat/barostat, production length, and sampling windows.",
            "- Dataset split, weighting, model version, validation metrics, and outlier rules.",
            "- Unit conversions, normalization basis, thermodynamic reference states, and uncertainty method.",
            "- Ground-state validation, finite-temperature trends, model/domain validity, literature comparison, and residual limitations.",
            "- Data availability, code availability, and whether any private/local paths must be removed.",
        ]
    )
    parts.append("")
    return "\n".join(parts)


def evidences_to_json(evidences: list[RunEvidence]) -> list[dict[str, object]]:
    return [
        {
            "path": str(evidence.path),
            "requested_modules": evidence.requested_modules,
            "detected_modules": evidence.detected_modules,
            "files": evidence.files,
            "facts": evidence.facts,
            "warnings": evidence.warnings,
        }
        for evidence in evidences
    ]


def llm_metadata_json(
    evidences: list[RunEvidence],
    modules: list[str],
    *,
    title: str,
    material: str,
    study_label: str,
    notes: list[str],
    hpc_context: dict[str, Any] | None = None,
) -> dict[str, object]:
    detected_modules = sorted({module for evidence in evidences for module in evidence.detected_modules})
    active_modules = modules or detected_modules
    return {
        "schema": "atomi.paper_draft.llm_metadata.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "manuscript_context": {
            "title": title,
            "material": material,
            "study_label": study_label,
            "user_notes": notes,
            "requested_modules": modules,
            "detected_modules": detected_modules,
        },
        "paper_lessons": {
            "USi_elastic_RUS_style": [
                "Tie every property result to the exact reference structure, supercell, finite-temperature sampling protocol, and property-extraction equation.",
                "Report convergence and validation before interpreting trends: ground-state structure, elastic constants, phonon/MD stability, and comparison with experiment or prior calculations.",
                "Keep Methods layered: DFT reference, ML-assisted sampling if used, finite-temperature MD, post-processing formulae, uncertainty, and data availability.",
            ],
            "UN_thermal_transport_style": [
                "Stable MLIP-MD is a prerequisite, not the conclusion; thermal transport needs route-specific validation.",
                "For Green-Kubo, extract timestep, ensemble history, NVE length, heat-current sampling interval, HCACF correlation time, plateau window, seed count, per-axis k, seed CV, and late-drift diagnostics.",
                "For non-equilibrium or normal-mode routes, extract slab/replicate geometry, heat-flow direction, binning/gradient windows, imposed flux or modal lifetime protocol, finite-size checks, and agreement/disagreement with GK.",
                "Compare GK, rNEMD/NEMD, NMA, and benchmark values cautiously; mismatches can indicate finite-size effects, poor gradient linearity, defect disorder, anharmonicity, coherent/off-diagonal transport, or MLIP-domain issues.",
            ],
        },
        "extraction_targets": {
            "DFT": [
                "structure/composition",
                "VASP version",
                "INCAR convergence and spin tags",
                "KPOINTS mesh",
                "POTCAR titles",
                "final energy/volume/NIONS",
            ],
            "VASP_PREP": [
                "motif provenance",
                "defect/cloud/candidate counts",
                "random seeds",
                "charge/spin labels",
                "runlist entries",
            ],
            "TRANSPORT": [
                "GK config/plan/validation/seed-summary/k(T) tables",
                "rNEMD config/plan/validation/profile/k(T) tables",
                "timestep, ensemble lengths, sampling/correlation/plateau windows",
                "replicate geometry, heat-flow direction, bins, swap interval",
                "seed count, per-axis spread, seed CV, late drift, slope mismatch, validation status",
                "runtime and array estimates from plans/HPC config when available",
            ],
            "MLIP": ["model file", "training manifest", "dataset frame count", "validation/outlier notes"],
            "MOOSE": ["material-property table columns", "temperature range", "unit/interpolation details"],
        },
        "writing_checklist": format_rule_lines(active_modules),
        "verification_items": [
            "Fill in any missing software versions, citations, units, and private-path redactions before manuscript submission.",
            "Verify that all thermal-conductivity unit conversions and normalization conventions match the equations used in the manuscript.",
            "For GK, inspect HCACF decay and running-integral plateau for every temperature/seed before trusting k(T).",
            "For rNEMD/NEMD, inspect temperature-profile linearity, excluded hot/cold bins, and mirrored-slope agreement.",
            "For U/actinide systems, verify magnetic/valence state labels and whether the selected potential/model was validated for that chemistry and temperature range.",
        ],
        "hpc_context": hpc_context or {},
        "runs": evidences_to_json(evidences),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-draft",
        description="Append Methods and brief Results draft text from completed run folders.",
    )
    parser.add_argument(
        "--used",
        "--modules",
        "-m",
        nargs="+",
        action="append",
        default=[],
        help="Workflow keywords used in this entry, e.g. DFT VASP_PREP VASP_SPIN MLIP AIMD MD TRANSPORT CALPHAD MOOSE.",
    )
    parser.add_argument(
        "--run",
        "-r",
        type=Path,
        action="append",
        default=[],
        help="Completed run folder to scan. Repeatable. Defaults to current directory.",
    )
    parser.add_argument(
        "--document",
        "--append-to",
        "-d",
        type=Path,
        default=Path("atomi_working_report.md"),
        help="Working Markdown document to append or overwrite.",
    )
    parser.add_argument("--fragment-out", type=Path, help="Also write this entry to a standalone file.")
    parser.add_argument("--evidence-json", type=Path, help="Write parsed evidence as JSON.")
    parser.add_argument(
        "--llm-metadata-json",
        "--metadata-json",
        type=Path,
        help="Write a richer LLM-readable metadata packet for manuscript drafting.",
    )
    parser.add_argument("--title", default="Atomi Draft Entry", help="Section heading for this entry.")
    parser.add_argument("--study-label", default="", help="Short label for this calculation set.")
    parser.add_argument("--material", default="", help="Material or chemical system label.")
    parser.add_argument("--note", action="append", default=[], help="User note to include in the entry.")
    parser.add_argument(
        "--mode",
        choices=("append", "overwrite", "fragment"),
        default="append",
        help="How to write --document. Fragment writes only --fragment-out and JSON metadata outputs.",
    )
    parser.add_argument("--max-files", type=int, default=5000, help="Maximum files to scan per run folder.")
    parser.add_argument("--hpc-config", type=Path, help="Optional private Atomi HPC config JSON. Defaults to ATOMI_HPC_CONFIG when set.")
    parser.add_argument(
        "--show-private-hpc",
        action="store_true",
        help="Include private executable/env paths from the HPC config in the local draft.",
    )
    parser.add_argument("--style-note", action="store_true", help="Include editorial style notes and reference links.")
    parser.add_argument("--no-style-note", action="store_true", help="Deprecated alias retained for older scripts.")
    parser.add_argument("--no-inventory", action="store_true", help="Omit the evidence inventory table from the generated draft.")
    parser.add_argument("--dry-run", action="store_true", help="Print the draft without writing files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    module_tokens = [token for group in args.used for token in group]
    modules = normalize_modules(module_tokens)
    runs = args.run or [Path(".")]
    evidences = [scan_run(path, modules, max_files=args.max_files) for path in runs]
    hpc_context = build_hpc_context(args.hpc_config, show_private=args.show_private_hpc)
    draft = compose_markdown(
        evidences=evidences,
        modules=modules,
        title=args.title,
        material=args.material,
        study_label=args.study_label,
        notes=args.note,
        include_style_note=args.style_note and not args.no_style_note,
        hpc_context=hpc_context,
        include_inventory=not args.no_inventory,
    )
    if args.dry_run:
        print(draft)
        return

    if args.fragment_out:
        args.fragment_out.parent.mkdir(parents=True, exist_ok=True)
        args.fragment_out.write_text(draft + "\n", encoding="utf-8")
        print(f"Wrote draft fragment: {args.fragment_out}")

    if args.evidence_json:
        args.evidence_json.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_json.write_text(
            json.dumps(evidences_to_json(evidences), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote evidence JSON: {args.evidence_json}")

    if args.llm_metadata_json:
        args.llm_metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.llm_metadata_json.write_text(
            json.dumps(
                llm_metadata_json(
                    evidences,
                    modules,
                    title=args.title,
                    material=args.material,
                    study_label=args.study_label,
                    notes=args.note,
                    hpc_context=hpc_context,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote LLM metadata JSON: {args.llm_metadata_json}")

    if args.mode != "fragment":
        args.document.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "overwrite":
            args.document.write_text(draft + "\n", encoding="utf-8")
        else:
            with args.document.open("a", encoding="utf-8") as handle:
                if args.document.exists() and args.document.stat().st_size > 0:
                    handle.write("\n")
                handle.write(draft + "\n")
        print(f"{'Updated' if args.mode == 'append' else 'Wrote'} working document: {args.document}")


if __name__ == "__main__":
    main()
