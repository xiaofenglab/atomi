"""POCC/zentropy defect thermodynamics helpers.

This module keeps a clear boundary between configurational counting and
thermodynamic weighting.  POCC, enumlib, motif embedding, or VASP ingestion
provide configurations and degeneracies.  The zentropy layer then attaches
finite-temperature free energies and computes population vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from atomi.zentropy.stage_utils import K_B_EV_PER_K, finite_float, format_value


SCHEMA = "atomi.zentropy.pocc_defects.v1"
R_J_MOLK = 8.31446261815324
EV_TO_KJ_MOL = 96.48533212331002


@dataclass(frozen=True)
class DefectSpecies:
    name: str
    element: str | None
    sublattice: str
    oxidation: float | None
    effective_charge: int


@dataclass(frozen=True)
class Sublattice:
    name: str
    sites_per_formula_unit: float
    allowed_species: tuple[str, ...]


@dataclass
class DefectConfiguration:
    config_id: str
    phase: str
    species_counts: dict[str, int]
    sublattice_counts: dict[str, int]
    degeneracy: float = 1.0
    degeneracy_type: str = "unknown"
    degeneracy_basis: str = "finite_supercell"
    E_static_eV: float | None = None
    G_eV_T: dict[float, float] = field(default_factory=dict)
    motif_labels: list[str] = field(default_factory=list)
    motif_features: dict[str, float] = field(default_factory=dict)
    structure_path: str | None = None
    source: str = "unknown"
    energy_status: str = "static"
    uncertainty_eV: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PopulationRow:
    macrostate_id: str
    T_K: float
    mu_O_eV: float | None
    config_id: str
    motif_labels: str
    degeneracy: float
    G_eV: float
    omega_eV: float
    delta_omega_eV: float
    probability: float
    N_cation: int
    N_O: int
    N_VaO: int
    N_Gd3: int
    N_U5: int
    x_Gd: float
    delta: float
    h_U5: float


GDUO2_SPECIES = {
    "U4": DefectSpecies("U4", "U", "cation", 4.0, 0),
    "U5": DefectSpecies("U5", "U", "cation", 5.0, 1),
    "Gd3": DefectSpecies("Gd3", "Gd", "cation", 3.0, -1),
    "O": DefectSpecies("O", "O", "anion", -2.0, 0),
    "VaO": DefectSpecies("VaO", None, "anion", None, 2),
}

GDUO2_SUBLATTICES = {
    "cation": Sublattice("cation", 1.0, ("U4", "U5", "Gd3")),
    "anion": Sublattice("anion", 2.0, ("O", "VaO")),
}


def _as_int(value: Any, default: int = 0) -> int:
    number = finite_float(value)
    if number is None:
        return default
    return int(round(number))


def _as_float(value: Any, default: float = 0.0) -> float:
    number = finite_float(value)
    return default if number is None else float(number)


def gduo2_default_config() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "system": {
            "phase": "fluorite",
            "reference_formula": "UO2",
            "basis": "per_cation_formula_unit",
        },
        "species": {key: asdict(value) for key, value in GDUO2_SPECIES.items()},
        "sublattices": {key: asdict(value) for key, value in GDUO2_SUBLATTICES.items()},
        "constraints": {
            "charge_neutrality": "N_U5 + 2*N_VaO - N_Gd3 == 0",
            "composition": {
                "x_Gd": "N_Gd3 / N_cation",
                "h_U5": "N_U5 / N_cation",
                "delta": "N_VaO / N_cation",
            },
        },
        "zentropy": {
            "ensemble_mode": "oxygen_semi_grand",
            "warnings": [
                "Degeneracy is a counting/symmetry quantity, not a probability.",
                "Population weights require both degeneracy and G_sigma(T) or Omega_sigma(T, mu_O).",
                "Do not call VASP electronic-smearing TOTEN the thermodynamic F_sigma(T).",
                "Do not silently infer U5; require metadata, Bader/charge, magnetic-polaron evidence, or manual review.",
            ],
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def _csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_configurations(path: Path) -> list[DefectConfiguration]:
    if path.suffix.lower() == ".jsonl":
        records = read_jsonl(path)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", payload if isinstance(payload, list) else [])
    else:
        records = _csv_dict(path)
    configs: list[DefectConfiguration] = []
    for idx, row in enumerate(records, start=1):
        counts = row.get("species_counts")
        if isinstance(counts, str):
            counts = json.loads(counts)
        if not isinstance(counts, dict):
            counts = {
                key: _as_int(row.get(key))
                for key in ("U4", "U5", "Gd3", "O", "VaO")
                if row.get(key) not in (None, "")
            }
        counts = {str(key): _as_int(value) for key, value in counts.items()}
        n_cat = counts.get("U4", 0) + counts.get("U5", 0) + counts.get("Gd3", 0)
        n_an = counts.get("O", 0) + counts.get("VaO", 0)
        sub_counts = row.get("sublattice_counts")
        if isinstance(sub_counts, str):
            sub_counts = json.loads(sub_counts)
        if not isinstance(sub_counts, dict):
            sub_counts = {"cation": n_cat, "anion": n_an}
        g_t = row.get("G_eV_T") or row.get("free_energy")
        if isinstance(g_t, str):
            try:
                g_t = json.loads(g_t)
            except json.JSONDecodeError:
                g_t = {}
        parsed_g: dict[float, float] = {}
        if isinstance(g_t, dict):
            for key, value in g_t.items():
                t = finite_float(key)
                g = finite_float(value)
                if t is not None and g is not None:
                    parsed_g[float(t)] = float(g)
        e_static = finite_float(row.get("E_static_eV") or row.get("energy_eV") or row.get("E_eV"))
        motif_labels = row.get("motif_labels") or row.get("motif_label") or []
        if isinstance(motif_labels, str):
            motif_labels = [part for part in motif_labels.replace(";", ",").split(",") if part.strip()]
        motif_features = row.get("motif_features") or row.get("motif_feature_vector") or {}
        if isinstance(motif_features, str):
            try:
                motif_features = json.loads(motif_features)
            except json.JSONDecodeError:
                motif_features = {}
        configs.append(
            DefectConfiguration(
                config_id=str(row.get("config_id") or row.get("motif_id") or row.get("id") or f"config_{idx:04d}"),
                phase=str(row.get("phase") or "fluorite"),
                species_counts=counts,
                sublattice_counts={str(key): _as_int(value) for key, value in sub_counts.items()},
                degeneracy=max(_as_float(row.get("degeneracy"), 1.0), 0.0),
                degeneracy_type=str(row.get("degeneracy_type") or "input"),
                degeneracy_basis=str(row.get("degeneracy_basis") or "finite_supercell"),
                E_static_eV=e_static,
                G_eV_T=parsed_g,
                motif_labels=[str(item).strip() for item in motif_labels if str(item).strip()],
                motif_features={str(key): _as_float(value) for key, value in dict(motif_features).items()},
                structure_path=str(row.get("structure_path") or "") or None,
                source=str(row.get("source") or ""),
                energy_status=str(row.get("energy_status") or ("static" if e_static is not None else "missing")),
                uncertainty_eV=finite_float(row.get("uncertainty_eV")),
                metadata={key: value for key, value in row.items() if key not in {"species_counts", "motif_features"}},
            )
        )
    return configs


def effective_charge(counts: dict[str, int], species: dict[str, DefectSpecies] | None = None) -> int:
    model = species or GDUO2_SPECIES
    return int(sum(counts.get(name, 0) * spec.effective_charge for name, spec in model.items()))


def gduo2_observables(counts: dict[str, int]) -> dict[str, float | int]:
    n_cat = counts.get("U4", 0) + counts.get("U5", 0) + counts.get("Gd3", 0)
    n_an = counts.get("O", 0) + counts.get("VaO", 0)
    n_o = counts.get("O", 0)
    n_vo = counts.get("VaO", 0)
    n_gd = counts.get("Gd3", 0)
    n_u5 = counts.get("U5", 0)
    return {
        "N_cation": n_cat,
        "N_anion": n_an,
        "N_O": n_o,
        "N_VaO": n_vo,
        "N_Gd3": n_gd,
        "N_U5": n_u5,
        "effective_charge": effective_charge(counts),
        "x_Gd": n_gd / n_cat if n_cat else math.nan,
        "delta": n_vo / n_cat if n_cat else math.nan,
        "h_U5": n_u5 / n_cat if n_cat else math.nan,
    }


def validate_configurations(configs: list[DefectConfiguration]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad_charge = 0
    missing_energy = 0
    for config in configs:
        obs = gduo2_observables(config.species_counts)
        charge_ok = obs["effective_charge"] == 0
        bad_charge += 0 if charge_ok else 1
        missing_energy += 0 if (config.E_static_eV is not None or config.G_eV_T) else 1
        rows.append(
            {
                "config_id": config.config_id,
                "phase": config.phase,
                "charge_neutral": charge_ok,
                "effective_charge": obs["effective_charge"],
                "x_Gd": obs["x_Gd"],
                "delta": obs["delta"],
                "h_U5": obs["h_U5"],
                "N_cation": obs["N_cation"],
                "N_O": obs["N_O"],
                "N_VaO": obs["N_VaO"],
                "N_Gd3": obs["N_Gd3"],
                "N_U5": obs["N_U5"],
                "degeneracy": config.degeneracy,
                "has_energy": config.E_static_eV is not None or bool(config.G_eV_T),
                "motif_labels": ";".join(config.motif_labels),
                "warnings": ";".join(config_warnings(config)),
            }
        )
    return rows, {
        "schema": f"{SCHEMA}.validation",
        "n_configurations": len(configs),
        "n_non_neutral": bad_charge,
        "n_missing_energy": missing_energy,
        "notes": [
            "Charge neutrality for Gd-UO2 uses N_U5 + 2*N_VaO - N_Gd3 == 0.",
            "Rows with missing energy have known counting metadata but cannot be Boltzmann weighted yet.",
        ],
    }


def config_warnings(config: DefectConfiguration) -> list[str]:
    warnings: list[str] = []
    obs = gduo2_observables(config.species_counts)
    if obs["effective_charge"] != 0:
        warnings.append("non_neutral")
    if config.degeneracy <= 0:
        warnings.append("non_positive_degeneracy")
    if config.E_static_eV is None and not config.G_eV_T:
        warnings.append("missing_energy")
    if config.species_counts.get("U5", 0) and "oxidation_assignment" not in config.metadata:
        warnings.append("u5_assignment_not_declared")
    if config.degeneracy_basis == "finite_supercell":
        warnings.append("finite_supercell_degeneracy")
    return warnings


def _energy_at_temperature(config: DefectConfiguration, temperature: float) -> float | None:
    if config.G_eV_T:
        exact = config.G_eV_T.get(float(temperature))
        if exact is not None:
            return exact
        nearest = min(config.G_eV_T, key=lambda item: abs(float(item) - temperature))
        if abs(float(nearest) - temperature) <= 1.0e-8:
            return config.G_eV_T[nearest]
    return config.E_static_eV


def _site_ideal_entropy_j_molK(configs: list[DefectConfiguration], probs: list[float]) -> float:
    totals = {"U4": 0.0, "U5": 0.0, "Gd3": 0.0, "O": 0.0, "VaO": 0.0}
    n_cat = 0.0
    n_an = 0.0
    for config, prob in zip(configs, probs):
        counts = config.species_counts
        for key in totals:
            totals[key] += prob * counts.get(key, 0)
        n_cat += prob * (counts.get("U4", 0) + counts.get("U5", 0) + counts.get("Gd3", 0))
        n_an += prob * (counts.get("O", 0) + counts.get("VaO", 0))
    if n_cat <= 0:
        return 0.0
    cat_entropy = 0.0
    for key in ("U4", "U5", "Gd3"):
        y = totals[key] / n_cat
        if y > 0:
            cat_entropy -= y * math.log(y)
    an_entropy = 0.0
    if n_an > 0:
        for key in ("O", "VaO"):
            y = totals[key] / n_an
            if y > 0:
                an_entropy -= y * math.log(y)
    return R_J_MOLK * (cat_entropy + 2.0 * an_entropy)


def solve_static_zentropy(
    configs: list[DefectConfiguration],
    *,
    temperatures: list[float],
    mu_o_values: list[float | None],
    group_by_x_gd: bool = True,
    require_neutral: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [cfg for cfg in configs if (not require_neutral or effective_charge(cfg.species_counts) == 0)]
    groups: dict[str, list[DefectConfiguration]] = {}
    for cfg in candidates:
        obs = gduo2_observables(cfg.species_counts)
        key = f"x_Gd={obs['x_Gd']:.8g}" if group_by_x_gd else "all"
        groups.setdefault(key, []).append(cfg)

    population_rows: list[dict[str, Any]] = []
    thermo_rows: list[dict[str, Any]] = []
    motif_rows: list[dict[str, Any]] = []
    for group_key, group_configs in sorted(groups.items()):
        for temperature in temperatures:
            for mu_o in mu_o_values:
                active: list[tuple[DefectConfiguration, float, float]] = []
                for cfg in group_configs:
                    g_value = _energy_at_temperature(cfg, temperature)
                    if g_value is None:
                        continue
                    obs = gduo2_observables(cfg.species_counts)
                    omega = g_value - (mu_o or 0.0) * int(obs["N_O"])
                    active.append((cfg, g_value, omega))
                if not active:
                    continue
                omega_min = min(item[2] for item in active)
                beta = 1.0 / (K_B_EV_PER_K * temperature)
                weights = [max(cfg.degeneracy, 0.0) * math.exp(-beta * (omega - omega_min)) for cfg, _, omega in active]
                z_value = sum(weights)
                if z_value <= 0:
                    continue
                probs = [weight / z_value for weight in weights]
                macrostate = f"{group_key}|T={temperature:g}|muO={mu_o if mu_o is not None else 'closed'}"
                pop_entropy = 0.0
                motif_pop: dict[str, float] = {}
                avg: dict[str, float] = {
                    "x_Gd": 0.0,
                    "delta": 0.0,
                    "h_U5": 0.0,
                    "N_O": 0.0,
                    "N_VaO": 0.0,
                    "N_U5": 0.0,
                }
                for (cfg, g_value, omega), prob in zip(active, probs):
                    obs = gduo2_observables(cfg.species_counts)
                    degeneracy = max(cfg.degeneracy, 1.0e-300)
                    pop_entropy -= R_J_MOLK * prob * math.log(max(prob / degeneracy, 1.0e-300))
                    for key in avg:
                        avg[key] += prob * float(obs[key])
                    labels = cfg.motif_labels or ["unlabeled"]
                    for label in labels:
                        motif_pop[label] = motif_pop.get(label, 0.0) + prob / len(labels)
                    population_rows.append(
                        asdict(
                            PopulationRow(
                                macrostate_id=macrostate,
                                T_K=temperature,
                                mu_O_eV=mu_o,
                                config_id=cfg.config_id,
                                motif_labels=";".join(labels),
                                degeneracy=cfg.degeneracy,
                                G_eV=g_value,
                                omega_eV=omega,
                                delta_omega_eV=omega - omega_min,
                                probability=prob,
                                N_cation=int(obs["N_cation"]),
                                N_O=int(obs["N_O"]),
                                N_VaO=int(obs["N_VaO"]),
                                N_Gd3=int(obs["N_Gd3"]),
                                N_U5=int(obs["N_U5"]),
                                x_Gd=float(obs["x_Gd"]),
                                delta=float(obs["delta"]),
                                h_U5=float(obs["h_U5"]),
                            )
                        )
                    )
                omega_ensemble = omega_min - (1.0 / beta) * math.log(z_value)
                site_entropy = _site_ideal_entropy_j_molK([item[0] for item in active], probs)
                dominant_index = max(range(len(probs)), key=lambda idx: probs[idx])
                dominant = active[dominant_index][0]
                thermo_rows.append(
                    {
                        "macrostate_id": macrostate,
                        "group_key": group_key,
                        "T_K": temperature,
                        "mu_O_eV": mu_o,
                        "n_states": len(active),
                        "Omega_eV": omega_ensemble,
                        "Omega_kJ_mol": omega_ensemble * EV_TO_KJ_MOL,
                        "S_population_J_molK": pop_entropy,
                        "S_site_ideal_J_molK": site_entropy,
                        "S_excess_conf_J_molK": pop_entropy - site_entropy,
                        "avg_x_Gd": avg["x_Gd"],
                        "avg_delta": avg["delta"],
                        "avg_h_U5": avg["h_U5"],
                        "avg_N_O": avg["N_O"],
                        "avg_N_VaO": avg["N_VaO"],
                        "avg_N_U5": avg["N_U5"],
                        "dominant_config_id": dominant.config_id,
                        "dominant_motif_labels": ";".join(dominant.motif_labels or ["unlabeled"]),
                        "dominant_probability": probs[dominant_index],
                    }
                )
                for motif, probability in sorted(motif_pop.items()):
                    motif_rows.append(
                        {
                            "macrostate_id": macrostate,
                            "group_key": group_key,
                            "T_K": temperature,
                            "mu_O_eV": mu_o,
                            "motif_label": motif,
                            "probability": probability,
                        }
                    )
    return population_rows, thermo_rows, motif_rows


POP_FIELDS = [field.name for field in PopulationRow.__dataclass_fields__.values()]
THERMO_FIELDS = [
    "macrostate_id",
    "group_key",
    "T_K",
    "mu_O_eV",
    "n_states",
    "Omega_eV",
    "Omega_kJ_mol",
    "S_population_J_molK",
    "S_site_ideal_J_molK",
    "S_excess_conf_J_molK",
    "avg_x_Gd",
    "avg_delta",
    "avg_h_U5",
    "avg_N_O",
    "avg_N_VaO",
    "avg_N_U5",
    "dominant_config_id",
    "dominant_motif_labels",
    "dominant_probability",
]
MOTIF_FIELDS = ["macrostate_id", "group_key", "T_K", "mu_O_eV", "motif_label", "probability"]
VALIDATION_FIELDS = [
    "config_id",
    "phase",
    "charge_neutral",
    "effective_charge",
    "x_Gd",
    "delta",
    "h_U5",
    "N_cation",
    "N_O",
    "N_VaO",
    "N_Gd3",
    "N_U5",
    "degeneracy",
    "has_energy",
    "motif_labels",
    "warnings",
]


def _parse_grid(values: list[str] | None, *, default: list[float | None]) -> list[float | None]:
    if not values:
        return list(default)
    out: list[float | None] = []
    for raw in values:
        text = str(raw).strip()
        if text.lower() in {"none", "closed"}:
            out.append(None)
        elif ":" in text:
            start, stop, step = [float(part) for part in text.split(":")]
            current = start
            if step == 0:
                raise ValueError("Grid step cannot be zero.")
            if step > 0:
                while current <= stop + abs(step) * 1.0e-9:
                    out.append(round(current, 12))
                    current += step
            else:
                while current >= stop - abs(step) * 1.0e-9:
                    out.append(round(current, 12))
                    current += step
        else:
            out.append(float(text))
    return out


def _add_common_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ensemble", type=Path, required=True, help="DefectConfiguration JSONL/JSON/CSV.")
    parser.add_argument("--outdir", type=Path, default=Path("pocc_zentropy_defects"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pocc-zentropy-defects",
        description="POCC/zentropy defect thermodynamic engine for Gd-doped UO2 and related fluorite oxides.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template", help="Write the default Gd-UO2 defect-engine YAML/JSON template.")
    template.add_argument("--output", type=Path, default=Path("gd_uo2.defect_engine.json"))

    validate = sub.add_parser("validate", help="Validate charge, degeneracy, and energy metadata.")
    _add_common_input(validate)

    solve = sub.add_parser("solve-static", help="Compute static oxygen semi-grand zentropy populations.")
    _add_common_input(solve)
    solve.add_argument("--temperature", action="append", default=[], help="T in K or start:stop:step.")
    solve.add_argument("--mu-o", action="append", default=[], help="mu_O in eV/O, grid start:stop:step, or closed.")
    solve.add_argument("--no-group-by-x-gd", action="store_true")
    solve.add_argument("--allow-non-neutral", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.command == "template":
        write_json(args.output.resolve(), gduo2_default_config())
        print(f"Wrote Gd-UO2 defect-engine template: {args.output.resolve()}")
        return {"output": str(args.output.resolve())}

    configs = load_configurations(args.ensemble.resolve())
    validation_rows, validation_metadata = validate_configurations(configs)
    outdir = args.outdir.resolve()
    write_csv(outdir / "configuration_audit.csv", validation_rows, VALIDATION_FIELDS)
    write_json(outdir / "configuration_audit.json", {"metadata": validation_metadata, "rows": validation_rows})

    if args.command == "validate":
        print(f"Configurations : {len(configs)}")
        print(f"Non-neutral    : {validation_metadata['n_non_neutral']}")
        print(f"Missing energy : {validation_metadata['n_missing_energy']}")
        print(f"Wrote audit    : {outdir / 'configuration_audit.csv'}")
        return validation_metadata

    temperatures = [float(value) for value in _parse_grid(args.temperature, default=[1000.0]) if value is not None]
    mu_values = _parse_grid(args.mu_o, default=[None])
    population_rows, thermo_rows, motif_rows = solve_static_zentropy(
        configs,
        temperatures=temperatures,
        mu_o_values=mu_values,
        group_by_x_gd=not args.no_group_by_x_gd,
        require_neutral=not args.allow_non_neutral,
    )
    write_csv(outdir / "population_vector.csv", population_rows, POP_FIELDS)
    write_csv(outdir / "zentropy_surface.csv", thermo_rows, THERMO_FIELDS)
    write_csv(outdir / "motif_populations.csv", motif_rows, MOTIF_FIELDS)
    metadata = {
        "schema": SCHEMA,
        "inputs": {"ensemble": str(args.ensemble.resolve())},
        "outputs": {
            "configuration_audit": str(outdir / "configuration_audit.csv"),
            "population_vector": str(outdir / "population_vector.csv"),
            "zentropy_surface": str(outdir / "zentropy_surface.csv"),
            "motif_populations": str(outdir / "motif_populations.csv"),
        },
        "temperatures_K": temperatures,
        "mu_O_eV": mu_values,
        "n_population_rows": len(population_rows),
        "n_surface_rows": len(thermo_rows),
        "notes": [
            "Degeneracy is retained as counting metadata and only becomes a Boltzmann weight inside the zentropy solve.",
            "S_population, S_site_ideal, and S_excess_conf are reported separately as a guard against ideal-mixing overuse.",
        ],
    }
    write_json(outdir / "pocc_zentropy_metadata.json", metadata)
    print(f"Population rows : {len(population_rows)}")
    print(f"Surface rows    : {len(thermo_rows)}")
    print(f"Wrote surface   : {outdir / 'zentropy_surface.csv'}")
    return metadata


if __name__ == "__main__":
    main()
