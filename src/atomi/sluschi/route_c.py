"""SLUSCHI Route C / zentropy-MD entropy workflow helpers.

Route C is the single-phase MD entropy route: use guarded solid and liquid
trajectories, estimate S_vib from SLUSCHI MDS/VACF data, estimate S_conf from
nearest-neighbor coordination probabilities, then assemble H, S, G and optional
Cp tables for zentropy/CALPHAD handoff.

The implementation deliberately separates three layers:

* prepare/kcl-demo writes reviewable workflow manifests and starter folders.
* analyze parses SLUSCHI MDS summaries and coordination histograms into a
  guarded entropy table.
* plot makes Fig. 3/4-style solid-vs-liquid overlays from the summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


R_GAS = 8.31446261815324  # J mol^-1 K^-1
SCHEMA_ROUTE_C_PLAN = "atomi.sluschi.route_c.plan.v1"
SCHEMA_ROUTE_C_SUMMARY = "atomi.sluschi.route_c.summary.v1"
SCHEMA_ROUTE_C_HEALTH = "atomi.sluschi.route_c.phase_health.v1"

SUMMARY_FIELDS = [
    "phase",
    "T_K",
    "H_kJ_mol_atom",
    "Svib_J_mol_atom_K",
    "Sconf_J_mol_atom_K",
    "Selec_J_mol_atom_K",
    "Stotal_J_mol_atom_K",
    "G_kJ_mol_atom",
    "Cp_J_mol_atom_K",
]


@dataclass(frozen=True)
class CoordinationDistribution:
    central: str
    neighbor: str
    probabilities: dict[int, float]
    cutoff_A: float | None = None
    phase: str = ""
    temperature_K: float | None = None

    @property
    def pair(self) -> str:
        return f"{self.central}-{self.neighbor}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_formula_atoms(formula: str) -> float | None:
    if not formula:
        return None
    total = 0.0
    for _element, count in re.findall(r"([A-Z][a-z]*)([0-9.]*)", formula):
        total += float(count) if count else 1.0
    return total or None


def normalize_probabilities(values: dict[int, float]) -> dict[int, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        return {}
    return {int(key): max(0.0, value) / total for key, value in sorted(values.items())}


def row_value(row: dict[str, str], *keys: str) -> str | None:
    """Read CSV fields while tolerating guard-output capitalization variants."""
    lower = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        value = lower.get(key.lower())
        if value not in (None, ""):
            return value
    return None


def parse_coordination_pair(pair: str) -> tuple[str, str]:
    cleaned = pair.strip().replace("→", "_to_").replace("->", "_to_")
    for separator in ("_to_", "-to-", " to ", "-"):
        if separator in cleaned:
            left, right = cleaned.split(separator, 1)
            return left.strip(), right.strip()
    return "", ""


def zentropy_sconf_from_probabilities(
    probabilities: dict[int, float],
    *,
    same_species: bool = False,
    pair_weight: float | None = None,
) -> float:
    """Return coordination configurational entropy in J mol-atom^-1 K^-1.

    The route-C coordination entropy is a Shannon entropy over nearest-neighbor
    coordination probabilities p_n. For same-species/same-sublattice pair
    channels, the default factor is 1/2 to avoid double-counting symmetric
    channels in a per-atom pair average.
    """
    probs = normalize_probabilities(probabilities)
    entropy = -R_GAS * sum(p * math.log(p) for p in probs.values() if p > 0.0)
    if pair_weight is None:
        pair_weight = 0.5 if same_species else 1.0
    return pair_weight * entropy


def distribution_moments(probabilities: dict[int, float]) -> dict[str, float | None]:
    probs = normalize_probabilities(probabilities)
    if not probs:
        return {"mean_cn": None, "std_cn": None, "max_probability": None, "n_states": 0}
    mean = sum(n * p for n, p in probs.items())
    variance = sum(((n - mean) ** 2) * p for n, p in probs.items())
    return {
        "mean_cn": mean,
        "std_cn": math.sqrt(max(0.0, variance)),
        "max_probability": max(probs.values()),
        "n_states": float(len(probs)),
    }


def rdf_first_minimum_cutoff(r_values: Iterable[float], g_values: Iterable[float]) -> float | None:
    """Find the first RDF minimum after the first non-trivial maximum."""
    r = list(r_values)
    g = list(g_values)
    if len(r) < 5 or len(r) != len(g):
        return None
    peak_idx = None
    for idx in range(1, len(g) - 1):
        if r[idx] <= 1.0e-8:
            continue
        if g[idx] >= g[idx - 1] and g[idx] > g[idx + 1]:
            peak_idx = idx
            break
    if peak_idx is None:
        return None
    for idx in range(peak_idx + 1, len(g) - 1):
        if g[idx] <= g[idx - 1] and g[idx] < g[idx + 1]:
            return float(r[idx])
    return None


def load_coordination_distributions(path: Path) -> list[CoordinationDistribution]:
    rows = read_csv_rows(path)
    grouped: dict[tuple[str, str, str, float | None, float | None], dict[int, float]] = defaultdict(dict)
    for row in rows:
        central = (row_value(row, "central", "center", "central_species") or "").strip()
        neighbor = (row_value(row, "neighbor", "neighbour", "neighbor_species") or "").strip()
        if not central or not neighbor:
            central, neighbor = parse_coordination_pair(row_value(row, "pair") or "")
        if not central or not neighbor:
            continue
        cn_raw = row_value(row, "cn", "CN", "coordination", "n")
        try:
            cn = int(float(cn_raw))
        except (TypeError, ValueError):
            continue
        value = finite_float(row_value(row, "probability"))
        if value is None:
            value = finite_float(row_value(row, "p_n"))
        if value is None:
            value = finite_float(row_value(row, "fraction"))
        if value is None:
            value = finite_float(row_value(row, "count"))
        if value is None:
            continue
        phase = (row_value(row, "phase") or "").strip().lower()
        temp = finite_float(row_value(row, "T_K", "temperature_K"))
        cutoff = finite_float(row_value(row, "cutoff_A", "cutoff"))
        grouped[(central, neighbor, phase, temp, cutoff)][cn] = grouped[(central, neighbor, phase, temp, cutoff)].get(cn, 0.0) + value
    return [
        CoordinationDistribution(central, neighbor, normalize_probabilities(values), cutoff, phase, temp)
        for (central, neighbor, phase, temp, cutoff), values in grouped.items()
    ]


def pair_rank(dist: CoordinationDistribution, phase: str = "") -> tuple[int, str]:
    central = dist.central
    neighbor = dist.neighbor
    if central == neighbor:
        return 0, "same-sublattice/same-element"
    # Conservative heavy-heavy heuristic without requiring a periodic table.
    light = {"H", "B", "C", "N", "O", "F", "Cl"}
    central_heavy = central not in light
    neighbor_heavy = neighbor not in light
    if central_heavy and neighbor_heavy:
        return 1, "heavy-heavy"
    if central_heavy != neighbor_heavy:
        return 2, "light-heavy"
    return 3, "fallback"


def select_route_c_pairs(
    distributions: list[CoordinationDistribution],
    *,
    phase: str,
    policy: str = "auto",
    solid_sconf_warning_threshold: float = 0.5,
) -> tuple[list[CoordinationDistribution], str, list[str]]:
    if not distributions:
        return [], "none", ["No coordination distributions were provided."]
    if policy == "all":
        return distributions, "all pairs averaged", []
    if policy not in {"auto", "same-species", "heavy-heavy", "light-heavy"}:
        raise ValueError(f"Unsupported Route C pair policy: {policy}")
    ranked = sorted((pair_rank(dist, phase)[0], dist.pair, dist) for dist in distributions)
    selected_rank = ranked[0][0]
    if policy == "same-species":
        selected_rank = 0
    elif policy == "heavy-heavy":
        selected_rank = 1
    elif policy == "light-heavy":
        selected_rank = 2
    selected = [dist for rank, _pair, dist in ranked if rank == selected_rank]
    if not selected:
        selected = [dist for rank, _pair, dist in ranked if rank == ranked[0][0]]
    note = pair_rank(selected[0], phase)[1] if selected else "none"
    warnings: list[str] = []
    if phase.lower() == "solid":
        svals = [
            zentropy_sconf_from_probabilities(dist.probabilities, same_species=dist.central == dist.neighbor)
            for dist in selected
        ]
        if svals and sum(svals) / len(svals) > solid_sconf_warning_threshold:
            warnings.append("Declared solid has non-negligible Route C S_conf; inspect phase/order guards and pair averaging.")
    return selected, note, warnings


def coordination_health_rows(
    distributions: list[CoordinationDistribution],
    *,
    declared_phase: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    phase = declared_phase.lower()
    for dist in distributions:
        moments = distribution_moments(dist.probabilities)
        sconf = zentropy_sconf_from_probabilities(dist.probabilities, same_species=dist.central == dist.neighbor)
        max_p = moments["max_probability"] or 0.0
        std_cn = moments["std_cn"] or 0.0
        status = "ok"
        if phase == "solid" and (max_p < 0.85 or sconf > 1.0 or std_cn > 0.4):
            status = "warning"
            warnings.append(f"Solid-like guard warning for {dist.pair}: broad p_n or S_conf={sconf:.3g}.")
        if phase == "liquid" and (max_p > 0.92 or std_cn < 0.25):
            status = "warning"
            warnings.append(f"Liquid-like guard warning for {dist.pair}: singular p_n / weak coordination disorder.")
        rows.append(
            {
                "phase": declared_phase,
                "T_K": dist.temperature_K,
                "pair": dist.pair,
                "central": dist.central,
                "neighbor": dist.neighbor,
                "cutoff_A": dist.cutoff_A,
                "mean_cn": moments["mean_cn"],
                "std_cn": moments["std_cn"],
                "max_probability": moments["max_probability"],
                "n_states": moments["n_states"],
                "Sconf_pair_J_mol_atom_K": sconf,
                "status": status,
            }
        )
    return rows, warnings


def coordination_distribution_raw_rows(
    distributions: list[CoordinationDistribution],
    *,
    selected: list[CoordinationDistribution],
    declared_phase: str,
    temperature_K: float,
) -> list[dict[str, Any]]:
    selected_keys = {(dist.central, dist.neighbor, dist.temperature_K, dist.cutoff_A) for dist in selected}
    rows: list[dict[str, Any]] = []
    for dist in distributions:
        key = (dist.central, dist.neighbor, dist.temperature_K, dist.cutoff_A)
        for cn, probability in normalize_probabilities(dist.probabilities).items():
            rows.append(
                {
                    "phase": dist.phase or declared_phase,
                    "T_K": dist.temperature_K if dist.temperature_K is not None else temperature_K,
                    "pair": dist.pair,
                    "central": dist.central,
                    "neighbor": dist.neighbor,
                    "cn": cn,
                    "probability": probability,
                    "cutoff_A": dist.cutoff_A,
                    "selected_for_Sconf": key in selected_keys,
                }
            )
    return rows


def plot_coordination_distribution(rows: list[dict[str, Any]], path: Path) -> str | None:
    if not rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair"])].append(row)
    n = len(grouped)
    fig, axes = plt.subplots(n, 1, figsize=(7, max(3, 2.2 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (pair, pair_rows) in zip(axes, sorted(grouped.items())):
        pair_rows = sorted(pair_rows, key=lambda row: int(row["cn"]))
        ax.bar([int(row["cn"]) for row in pair_rows], [float(row["probability"]) for row in pair_rows], width=0.8)
        selected = any(str(row.get("selected_for_Sconf", "")).lower() == "true" for row in pair_rows)
        ax.set_ylabel("p(n)")
        ax.set_title(f"{pair} coordination distribution" + (" (selected)" if selected else ""))
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xlabel("coordination number n")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def parse_sluschi_mds_outputs(path: Path, *, formula: str = "") -> dict[str, Any]:
    """Parse Atomi/SLUSCHI MDS summaries or collect text into route-C quantities."""
    atoms_per_formula = parse_formula_atoms(formula)
    result: dict[str, Any] = {}
    candidates: list[Path]
    if path.is_dir():
        candidates = [
            path / "sluschi_entropy_summary.csv",
            path / "atomi_entropy_summary" / "sluschi_entropy_summary.csv",
            path / "route_c_summary.csv",
            path / "collect.stdout",
            path / "collect.out",
        ]
    else:
        candidates = [path]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".csv":
            rows = read_csv_rows(candidate)
            if not rows:
                continue
            row = rows[0]
            apf = finite_float(row.get("atoms_per_formula"), atoms_per_formula) or atoms_per_formula or 1.0
            svib = finite_float(row.get("Svib_J_mol_atom_K"))
            if svib is None:
                svib_formula = finite_float(row.get("Svib_J_mol_formula_K"))
                svib = svib_formula / apf if svib_formula is not None and apf else None
            sconf = finite_float(row.get("Sconf_J_mol_atom_K"))
            if sconf is None:
                sconf_formula = finite_float(row.get("Sconf_J_mol_formula_K"))
                sconf = sconf_formula / apf if sconf_formula is not None and apf else None
            result.update(
                {
                    "Svib_J_mol_atom_K": svib,
                    "Sconf_J_mol_atom_K": sconf,
                    "source": str(candidate),
                    "atoms_per_formula": apf,
                }
            )
            return result
        text = candidate.read_text(encoding="utf-8", errors="replace")
        svib_match = re.search(r"Svib[^0-9+\-.]*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
        sconf_match = re.search(r"Sconf[^0-9+\-.]*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
        if svib_match:
            result["Svib_J_mol_atom_K"] = float(svib_match.group(1))
        if sconf_match:
            result["Sconf_J_mol_atom_K"] = float(sconf_match.group(1))
        if result:
            result["source"] = str(candidate)
            return result
    return result


def parse_enthalpy(path: Path | None, *, phase: str, temperature_K: float) -> float | None:
    if path is None or not path.is_file():
        return None
    rows = read_csv_rows(path)
    if not rows:
        return None
    best = None
    for row in rows:
        row_phase = (row.get("phase") or phase).lower()
        row_t = finite_float(row.get("T_K") or row.get("temperature_K"), temperature_K)
        if row_phase == phase.lower() and row_t is not None:
            score = abs(row_t - temperature_K)
            if best is None or score < best[0]:
                best = (score, row)
    row = best[1] if best else rows[0]
    for key in ("H_kJ_mol_atom", "enthalpy_kJ_mol_atom", "H"):
        value = finite_float(row.get(key))
        if value is not None:
            return value
    value = finite_float(row.get("TotEng_eV_atom") or row.get("E_eV_atom"))
    if value is not None:
        return value * 96.48533212
    return None


def add_cp(rows: list[dict[str, Any]]) -> None:
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_phase[str(row["phase"])].append(row)
    for phase_rows in by_phase.values():
        phase_rows.sort(key=lambda row: float(row["T_K"]))
        for idx, row in enumerate(phase_rows):
            cp = None
            if len(phase_rows) >= 2:
                left = phase_rows[max(0, idx - 1)]
                right = phase_rows[min(len(phase_rows) - 1, idx + 1)]
                if left is not right and left.get("H_kJ_mol_atom") is not None and right.get("H_kJ_mol_atom") is not None:
                    dt = float(right["T_K"]) - float(left["T_K"])
                    if abs(dt) > 1.0e-12:
                        cp = (float(right["H_kJ_mol_atom"]) - float(left["H_kJ_mol_atom"])) * 1000.0 / dt
            row["Cp_J_mol_atom_K"] = cp


def analyze_main(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    phase = args.phase.lower()
    temp = float(args.temperature_k)
    selec = float(args.selec_j_mol_atom_k)
    distributions: list[CoordinationDistribution] = []
    if args.coordination_csv:
        distributions = [
            dist
            for dist in load_coordination_distributions(args.coordination_csv)
            if (not dist.phase or dist.phase == phase) and (dist.temperature_K is None or abs(dist.temperature_K - temp) < 1.0e-6)
        ]
    selected, pair_note, pair_warnings = select_route_c_pairs(distributions, phase=phase, policy=args.pair_policy)
    pair_sconf = [
        zentropy_sconf_from_probabilities(dist.probabilities, same_species=dist.central == dist.neighbor)
        for dist in selected
    ]
    sconf_coord = sum(pair_sconf) / len(pair_sconf) if pair_sconf else None
    mds = parse_sluschi_mds_outputs(args.mds_summary, formula=args.formula) if args.mds_summary else {}
    svib = finite_float(args.svib_j_mol_atom_k, mds.get("Svib_J_mol_atom_K"))
    sconf = finite_float(args.sconf_j_mol_atom_k)
    if sconf is None:
        sconf = sconf_coord if sconf_coord is not None else finite_float(mds.get("Sconf_J_mol_atom_K"))
    h = finite_float(args.h_kj_mol_atom)
    if h is None:
        h = parse_enthalpy(args.thermo_csv, phase=phase, temperature_K=temp)
    stotal = (svib or 0.0) + (sconf or 0.0) + selec if (svib is not None or sconf is not None or selec) else None
    g = h - temp * stotal / 1000.0 if h is not None and stotal is not None else None
    row = {
        "phase": phase,
        "T_K": temp,
        "H_kJ_mol_atom": h,
        "Svib_J_mol_atom_K": svib,
        "Sconf_J_mol_atom_K": sconf,
        "Selec_J_mol_atom_K": selec,
        "Stotal_J_mol_atom_K": stotal,
        "G_kJ_mol_atom": g,
        "Cp_J_mol_atom_K": None,
        "formula": args.formula,
        "pair_policy": args.pair_policy,
        "pair_selection_note": pair_note,
        "selected_pairs": ",".join(dist.pair for dist in selected),
        "mds_source": mds.get("source", ""),
        "quality": args.quality,
    }
    health_rows, health_warnings = coordination_health_rows(distributions, declared_phase=phase)
    warnings = pair_warnings + health_warnings
    write_csv(outdir / "route_c_summary.csv", [row], list(row))
    if health_rows:
        write_csv(outdir / "coordination_distribution_health.csv", health_rows, list(health_rows[0]))
    distribution_rows = coordination_distribution_raw_rows(
        distributions, selected=selected, declared_phase=phase, temperature_K=temp
    )
    coordination_png = None
    if distribution_rows:
        write_csv(
            outdir / "coordination_distribution.csv",
            distribution_rows,
            [
                "phase",
                "T_K",
                "pair",
                "central",
                "neighbor",
                "cn",
                "probability",
                "cutoff_A",
                "selected_for_Sconf",
            ],
        )
        coordination_png = plot_coordination_distribution(distribution_rows, outdir / "coordination_distribution.png")
    health = {
        "schema": SCHEMA_ROUTE_C_HEALTH,
        "phase": phase,
        "temperature_K": temp,
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "n_coordination_pairs": len(distributions),
        "selected_pairs": [dist.pair for dist in selected],
        "coordination_distribution_png": coordination_png,
    }
    write_json(outdir / "phase_health_route_c.json", health)
    write_json(
        outdir / "route_c_summary.json",
        {"schema": SCHEMA_ROUTE_C_SUMMARY, "summary": row, "phase_health": health, "coordination_rows": health_rows},
    )
    print(f"Wrote Route C summary: {outdir / 'route_c_summary.csv'}")
    print(f"Wrote Route C phase health: {outdir / 'phase_health_route_c.json'}")
    return {"summary": row, "phase_health": health}


def prepare_main(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    phases = [token.strip() for token in args.phases.split(",") if token.strip()]
    temps = [float(token) for token in args.temperatures.split(",") if token.strip()]
    plan = {
        "schema": SCHEMA_ROUTE_C_PLAN,
        "method": "sluschi_route_c_zentropy_md_entropy",
        "system": args.system,
        "formula": args.formula,
        "engine": args.engine,
        "phases": phases,
        "temperatures_K": temps,
        "workflow": [
            "Generate pure single-phase solid/liquid boxes, not solid-liquid coexistence boxes.",
            "Liquid seed: high-T melt -> hold -> cool to target -> equilibrate -> NVT/NPT tail as documented.",
            "Solid seed: correct crystal phase -> guarded equilibration below/near target without melting.",
            "Run SLUSCHI mds/mds_lmp or Atomi Route C analyze on accepted single-phase tails.",
            "Guard with coordination p_n, Bragg/XRD/PDF, MSD/diffusion, temperature, and enthalpy stability.",
        ],
        "outputs_expected": SUMMARY_FIELDS,
        "route_c_commands": {
            "analyze": "atomi sluschi-route-c analyze --phase solid --temperature-k 1000 --coordination-csv coordination.csv --mds-summary atomi_entropy_summary/sluschi_entropy_summary.csv",
            "plot": "atomi sluschi-route-c plot --summary route_c_summary_all.csv --outdir route_c_plots",
        },
        "notes": [
            "Route C is zentropy-MD entropy from single-phase trajectories.",
            "S_elec defaults to zero for insulating fluoride/chloride salts unless supplied.",
            "Cp is finite-difference only when multiple T points exist.",
        ],
    }
    write_json(outdir / "route_c_plan.json", plan)
    (outdir / "README_ROUTE_C.md").write_text(
        textwrap.dedent(
            f"""\
            # SLUSCHI Route C / zentropy-MD entropy plan

            System: `{args.system}` / formula `{args.formula}`.

            Route C uses single-phase solid and liquid trajectories. It does not
            use coexistence boxes. Use existing Atomi LAMMPS/CP2K preparation
            tools to generate guarded MD tails, then run `sluschi-route-c analyze`
            on each accepted phase/T point.
            """
        ),
        encoding="utf-8",
    )
    print(f"Wrote Route C plan: {outdir / 'route_c_plan.json'}")
    return plan


def kcl_demo_main(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    coord_rows = []
    # Sharp rocksalt-like solid and broad liquid-like demonstration distributions.
    for temp in [800.0, 1000.0]:
        for pair in [("K", "K"), ("Cl", "Cl")]:
            for cn, prob in [(12, 0.98), (11, 0.01), (13, 0.01)]:
                coord_rows.append({"phase": "solid", "T_K": temp, "central": pair[0], "neighbor": pair[1], "cn": cn, "probability": prob, "cutoff_A": 5.2})
    for temp in [1200.0, 1400.0]:
        for pair in [("K", "K"), ("Cl", "Cl")]:
            for cn, prob in [(8, 0.08), (9, 0.16), (10, 0.24), (11, 0.24), (12, 0.16), (13, 0.08), (14, 0.04)]:
                coord_rows.append({"phase": "liquid", "T_K": temp, "central": pair[0], "neighbor": pair[1], "cn": cn, "probability": prob, "cutoff_A": 5.6})
    write_csv(outdir / "kcl_route_c_demo_coordination.csv", coord_rows, list(coord_rows[0]))
    prepare_main(
        argparse.Namespace(
            outdir=outdir,
            system="KCl",
            formula="KCl",
            engine=args.engine,
            phases="solid,liquid",
            temperatures="800,1000,1200,1400",
        )
    )
    demo = {
        "coordination_csv": str(outdir / "kcl_route_c_demo_coordination.csv"),
        "recommended_real_runs": [
            "KCl solid: use rocksalt seed, NPT anchor at 800/1000 K, then NVT production tail if phase guard passes.",
            "KCl liquid: use high-T melt or accepted scout liquid tail, cool/equilibrate to 1200/1400 K, then NVT production tail.",
            "For speed, prefer the historically faster KCl LAMMPS/SuperSalt route unless CP2K exact-method symmetry is required.",
        ],
    }
    write_json(outdir / "kcl_route_c_demo_manifest.json", demo)
    print(f"Wrote KCl Route C demo scaffold: {outdir}")
    return demo


def plot_main(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_csv_rows(args.summary)
    if not rows:
        raise ValueError(f"No rows found in {args.summary}")
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    numeric_rows: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {"phase": row.get("phase", "")}
        for field in SUMMARY_FIELDS[1:]:
            converted[field] = finite_float(row.get(field))
        numeric_rows.append(converted)
    add_cp(numeric_rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for sluschi-route-c plot") from exc

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    fields = [
        ("Svib_J_mol_atom_K", "S_vib (J mol-atom$^{-1}$ K$^{-1}$)"),
        ("Sconf_J_mol_atom_K", "S_conf (J mol-atom$^{-1}$ K$^{-1}$)"),
        ("Stotal_J_mol_atom_K", "S_total (J mol-atom$^{-1}$ K$^{-1}$)"),
        ("Cp_J_mol_atom_K", "C_p (J mol-atom$^{-1}$ K$^{-1}$)"),
    ]
    phases = sorted({str(row["phase"]) for row in numeric_rows})
    for ax, (field, ylabel) in zip(axes.ravel(), fields):
        for phase in phases:
            phase_rows = sorted([row for row in numeric_rows if row["phase"] == phase], key=lambda row: row["T_K"] or 0)
            xs = [row["T_K"] for row in phase_rows if row.get(field) is not None and row.get("T_K") is not None]
            ys = [row[field] for row in phase_rows if row.get(field) is not None and row.get("T_K") is not None]
            if xs:
                ax.plot(xs, ys, marker="o", label=phase)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Temperature (K)")
    axes[0, 0].legend()
    fig.suptitle(args.title or "SLUSCHI Route C zentropy-MD entropy")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = outdir / "route_c_entropy_overlay.png"
    fig.savefig(fig_path, dpi=220)
    write_csv(outdir / "route_c_summary_with_cp.csv", numeric_rows, ["phase", *SUMMARY_FIELDS[1:]])
    print(f"Wrote Route C plot: {fig_path}")
    return {"figure": str(fig_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sluschi-route-c", description="SLUSCHI Route C / zentropy-MD entropy workflow.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Write a general Route C workflow plan.")
    prep.add_argument("--outdir", type=Path, default=Path("route_c_plan"))
    prep.add_argument("--system", default="")
    prep.add_argument("--formula", default="")
    prep.add_argument("--engine", choices=("lammps", "cp2k", "vasp", "mixed"), default="lammps")
    prep.add_argument("--phases", default="solid,liquid")
    prep.add_argument("--temperatures", default="1000,1200,1400")

    analyze = sub.add_parser("analyze", help="Analyze one accepted single-phase trajectory/SLUSCHI summary.")
    analyze.add_argument("--outdir", type=Path, default=Path("route_c_analysis"))
    analyze.add_argument("--phase", choices=("solid", "liquid"), required=True)
    analyze.add_argument("--temperature-k", type=float, required=True)
    analyze.add_argument("--formula", default="")
    analyze.add_argument("--coordination-csv", type=Path)
    analyze.add_argument("--mds-summary", type=Path)
    analyze.add_argument("--thermo-csv", type=Path)
    analyze.add_argument("--h-kj-mol-atom", type=float)
    analyze.add_argument("--svib-j-mol-atom-k", type=float)
    analyze.add_argument("--sconf-j-mol-atom-k", type=float)
    analyze.add_argument("--selec-j-mol-atom-k", type=float, default=0.0)
    analyze.add_argument("--pair-policy", choices=("auto", "all", "same-species", "heavy-heavy", "light-heavy"), default="auto")
    analyze.add_argument("--quality", choices=("descriptor", "screening-prior", "production"), default="screening-prior")

    demo = sub.add_parser("kcl-demo", help="Write a KCl Route C demo scaffold and synthetic guard table.")
    demo.add_argument("--outdir", type=Path, default=Path("kcl_route_c_demo"))
    demo.add_argument("--engine", choices=("lammps", "cp2k", "mixed"), default="lammps")

    plot = sub.add_parser("plot", help="Plot Route C summary rows.")
    plot.add_argument("--summary", type=Path, required=True)
    plot.add_argument("--outdir", type=Path, default=Path("route_c_plots"))
    plot.add_argument("--title", default="")

    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return prepare_main(args)
    if args.command == "analyze":
        return analyze_main(args)
    if args.command == "kcl-demo":
        return kcl_demo_main(args)
    if args.command == "plot":
        return plot_main(args)
    return None


if __name__ == "__main__":
    main()
