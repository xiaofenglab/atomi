"""Canonical postanalysis helpers for OpenMolcas spectroscopy runs.

This module deliberately stays at the postprocessing layer: it does not choose
active spaces or chemistry-specific root numbers.  It connects the durable Atomi
tools used after a Molcas run finishes: root audits, RASSI transition parsing,
XANES broadening, M4/M5 two-panel plotting, and orbital-viewer handoff.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from atomi.xafs.molcas_xanes_spectrum import (
    broaden,
    parse_so_states,
    parse_transition_sections,
    read_text,
    transitions_from_csv,
    write_spectrum_csv,
    write_transitions_csv,
    xraydb_metadata,
)


SCHEMA_WORKFLOW = "atomi.molcas_postanalysis_workflow.v1"
SCHEMA_M45 = "atomi.molcas_m45_two_panel.v1"
SCHEMA_TRANSITIONS = "atomi.molcas_important_dipole_transitions.v1"
SCHEMA_ORBITAL_HANDOFF = "atomi.molcas_orbital_handoff.v1"
SCHEMA_MO_DIAGRAM = "atomi.molcas_mo_diagram.v1"
SCHEMA_U5F_SPLITTING = "atomi.u5f_so_lf_splitting_diagram.v1"
SCHEMA_M45_EXTRACT = "atomi.molcas_m45_transition_extract.v1"


def workflow_record() -> dict[str, Any]:
    """Return the canonical Molcas postanalysis workflow Sarah/Anna should use."""

    return {
        "schema": SCHEMA_WORKFLOW,
        "scope": "OpenMolcas CASSCF/RASSCF, CASPT2/RASPT2, RASSI spectroscopy postanalysis",
        "roles": {
            "anna": "Owns Atomi commands, private cookbook rules, KIT/HPC runtime consistency.",
            "sarah": "Owns portfolio-level decision gates, figure expectations, and project handoff.",
            "student": "Owns project-specific chemistry, run provenance, physical interpretation, and report figures.",
        },
        "commands": [
            {
                "stage": "runtime/provenance",
                "command": "molcas-status --json",
                "purpose": "Confirm OpenMolcas/Pegamoid/xraydb-capable environment before starting or interpreting runs.",
            },
            {
                "stage": "finished-run summary",
                "command": "molcas-bridge collect --output RUN.out --write openmolcas_summary.json",
                "purpose": "Collect module return codes, CASPT2 roots, RASSI presence, and error markers.",
            },
            {
                "stage": "root audit",
                "command": "molcas-root-helper audit --output RUN.out --write molcas_root_audit.json",
                "purpose": "Read actual Molcas root sections: CSFs, highly excited CSFs, roots required, and highest root.",
            },
            {
                "stage": "single-edge XANES",
                "command": "molcas-xanes-spectrum from-output --molcas-out RUN.out --element Ga --edge K --from-state 1 --section last",
                "purpose": "Direct Ga K-edge or simple single-edge check from SO-RASSI transition tables.",
            },
            {
                "stage": "actinide M4/M5 edge broadening",
                "command": "molcas-xanes-spectrum from-csv --transitions-csv M5_transitions.csv --element U --edge M5 --broadening voigt",
                "purpose": "Broaden curated/averaged RASSI dipole transition CSVs with Voigt broadening and xraydb core-hole widths.",
            },
            {
                "stage": "actinide M4/M5 transition extraction",
                "command": "molcas-postanalysis extract-m45-transitions --molcas-out RUN.out --initial-states 1,2 --prefix u5_cn8_ground_doublet_avg",
                "purpose": "Extract and initial-state-average SO RASSI sticks from a completed Molcas output, then split lower/upper manifolds into M5/M4 CSVs.",
            },
            {
                "stage": "actinide M4/M5 two-panel figure",
                "command": "molcas-postanalysis m45-two-panel --m5-transitions-csv M5.csv --m4-transitions-csv M4.csv --element U --outdir xanes_m45",
                "purpose": "Make the report-style two-panel M5/M4 envelope plus tall stick transition figure.",
            },
            {
                "stage": "important dipole transitions",
                "command": "molcas-postanalysis rank-transitions --transitions-csv M5_transitions.csv --relative-threshold 0.95 --plot",
                "purpose": "Write the near-maximum transition table/JSON/plot so the student discusses only transitions with intensity >= 95% of the maximum.",
            },
            {
                "stage": "orbital/mixing",
                "command": "molcas-postanalysis orbital-handoff --molcas-dir RUN_DIR --archive openmolcas.tgz --outdir orbital_handoff",
                "purpose": "Find orbital/NTO artifacts and write a Pegamoid/NTO handoff; use targeted extraction from large Molcas archives.",
            },
            {
                "stage": "orbital viewing",
                "command": "pegamoid-bridge prepare --molcas-dir RUN_DIR --glob '*.rasscf.h5' --outdir pegamoid_orbitals",
                "purpose": "Prepare orbital/NTO visualization wrappers after relevant files have been identified or extracted.",
            },
            {
                "stage": "schematic MO diagram",
                "command": "molcas-postanalysis mo-diagram --orbitals-csv mo_orbitals.csv --transitions-csv important_dipole_transitions.csv --outdir mo_diagram",
                "purpose": "Create a project-specific schematic MO diagram linking occupations to strong dipole transitions.",
            },
            {
                "stage": "U M-edge 5f splitting schematic",
                "command": "molcas-postanalysis u5f-splitting --structure UO8_average.xyz --outdir u5f_splitting",
                "purpose": "Make an SO versus local-cluster ligand-field U 5f correlation diagram for U M4,5 interpretation.",
            },
        ],
        "decision_rules": [
            "Never promote a spectrum if the Molcas output has nonzero return codes or incomplete RASSI blocks.",
            "For high-root actinide runs, audit output root counts; do not infer computed roots from input alone.",
            "For Kramers systems, average the ground doublet before report-level isotropic spectra unless a polarized/site-specific comparison is intended.",
            "Treat absolute XANES energies as uncalibrated cluster transition energies until aligned to standards or an internal oxidation-state series.",
            "Use Voigt broadening, not pure Lorentzian broadening, for report-level XANES envelopes.",
            "Keep stick transitions visible in report figures; by default show the near-maximum stick band with intensity >= 95% of the maximum for that edge.",
            "Always write a ranked important-dipole-transition table for the peaks used in scientific discussion; the default important set is the >=95% max-intensity band.",
            "Orbital plots should show the active orbital manifold and important transition/mixing orbitals, not just pretty frontier orbitals.",
            "When RASSI NTOCalc, BINAtorb, SONOrb, NTORB, or MD_NTO outputs exist, prefer transition-character/NTO plots for peak assignments.",
            "For reports, add a schematic MO diagram when orbital/transition assignments are central to the scientific argument.",
            "For actinide U M-edge clusters, use `u5f-splitting --structure CLUSTER.xyz` so the LF side is local-cluster based; use `--mode uranyl-reference` only as a paper-style reference.",
        ],
        "toolset": [
            "OpenMolcas output, JobIph/JobMix, .rasscf.h5/.rassi.h5, .RasOrb, .molden, NTORB, MD_NTO, SIORB/BIORB",
            "molcas-bridge, molcas-root-helper, molcas-xanes-spectrum, molcas-postanalysis",
            "xraydb for edge references and core-hole widths",
            "Pegamoid for OpenMolcas orbital/density viewing",
            "project report + Sarah portfolio memory for accepted figures and decisions",
        ],
    }


def print_workflow(args: argparse.Namespace) -> int:
    record = workflow_record()
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    print("# Molcas Postanalysis Workflow\n")
    print(record["scope"])
    print("\n## Commands")
    for item in record["commands"]:
        print(f"- {item['stage']}: `{item['command']}`")
        print(f"  {item['purpose']}")
    print("\n## Decision Rules")
    for rule in record["decision_rules"]:
        print(f"- {rule}")
    print("\n## Sarah/Anna Toolset")
    for tool in record["toolset"]:
        print(f"- {tool}")
    return 0


def _read_spectrum_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    energy: list[float] = []
    intensity: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            e = row.get("energy_ev") or row.get("energy") or row.get("E")
            y = row.get("intensity") or row.get("mu") or row.get("norm_intensity")
            if e is None or y is None:
                continue
            energy.append(float(e))
            intensity.append(float(y))
    if not energy:
        raise ValueError(f"No spectrum rows found in {path}")
    return np.asarray(energy, dtype=float), np.asarray(intensity, dtype=float)


def _edge_width(element: str, edge: str, disabled: bool, override: float | None) -> tuple[float, dict[str, Any]]:
    if override is not None:
        meta = {"element": element, "edge": edge, "source": "manual", "core_hole_width_ev": override}
        return override, meta
    if disabled:
        meta = {"element": element, "edge": edge, "source": "disabled", "core_hole_width_ev": 1.0}
        return 1.0, meta
    meta = xraydb_metadata(element, edge)
    width = float(meta.get("core_hole_width_ev", 1.0))
    return width, meta


def _broaden_edge(
    transitions_csv: Path,
    *,
    element: str,
    edge: str,
    emin: float | None,
    emax: float | None,
    step: float,
    gaussian_fwhm: float,
    lorentzian_fwhm: float,
    broadening: str,
    eta: float,
    normalize: str,
    energy_shift_ev: float,
    outdir: Path,
    prefix: str,
) -> dict[str, Any]:
    transitions = transitions_from_csv(transitions_csv)
    energy, spectrum, rows = broaden(
        transitions,
        emin=emin,
        emax=emax,
        step=step,
        energy_shift=energy_shift_ev,
        gaussian_fwhm=gaussian_fwhm,
        lorentzian_fwhm=lorentzian_fwhm,
        mode=broadening,
        eta=eta,
        normalize=normalize,
    )
    spectrum_csv = outdir / f"{prefix}_xanes.csv"
    rows_csv = outdir / f"{prefix}_transitions.csv"
    write_spectrum_csv(spectrum_csv, energy, spectrum)
    write_transitions_csv(rows_csv, rows)
    return {
        "edge": edge,
        "transitions_source": str(transitions_csv),
        "spectrum_csv": str(spectrum_csv),
        "transitions_csv": str(rows_csv),
        "broadening": broadening,
        "energy": energy,
        "spectrum": spectrum,
        "rows": rows,
        "peak_energy_ev": float(energy[int(np.argmax(spectrum))]),
        "n_transitions_used": len(rows),
    }


def _parse_state_list(text: str) -> list[int]:
    states: list[int] = []
    for token in text.replace(",", " ").split():
        if not token.strip():
            continue
        states.append(int(token))
    if not states:
        raise ValueError("At least one initial SO state is required")
    return states


def _write_transition_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "state_from",
        "state_to",
        "energy_ev",
        "oscillator_strength",
        "gauge",
        "state_basis",
        "section_index",
        "initial_state_label",
        "components_from_initial_states",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _split_m45_rows(
    rows: list[dict[str, Any]],
    *,
    m5_min: float | None,
    m5_max: float | None,
    m4_min: float | None,
    m4_max: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], [], {"mode": "empty"}
    if any(value is not None for value in [m5_min, m5_max, m4_min, m4_max]):
        m5_rows = [
            row
            for row in rows
            if (m5_min is None or float(row["energy_ev"]) >= m5_min)
            and (m5_max is None or float(row["energy_ev"]) <= m5_max)
        ]
        m4_rows = [
            row
            for row in rows
            if (m4_min is None or float(row["energy_ev"]) >= m4_min)
            and (m4_max is None or float(row["energy_ev"]) <= m4_max)
        ]
        return m5_rows, m4_rows, {
            "mode": "explicit_windows",
            "m5_window_ev": [m5_min, m5_max],
            "m4_window_ev": [m4_min, m4_max],
        }

    ordered = sorted(rows, key=lambda row: float(row["energy_ev"]))
    if len(ordered) < 2:
        return ordered, [], {"mode": "single_cluster"}
    gaps = [
        (float(ordered[i + 1]["energy_ev"]) - float(ordered[i]["energy_ev"]), i)
        for i in range(len(ordered) - 1)
    ]
    gap, idx = max(gaps, key=lambda item: item[0])
    split_energy = 0.5 * (float(ordered[idx]["energy_ev"]) + float(ordered[idx + 1]["energy_ev"]))
    m5_rows = [row for row in ordered if float(row["energy_ev"]) <= split_energy]
    m4_rows = [row for row in ordered if float(row["energy_ev"]) > split_energy]
    return m5_rows, m4_rows, {
        "mode": "largest_gap_auto",
        "split_energy_ev": split_energy,
        "largest_gap_ev": gap,
        "lower_edge_assigned": "M5",
        "upper_edge_assigned": "M4",
    }


def extract_m45_transitions(args: argparse.Namespace) -> int:
    text = read_text(args.molcas_out)
    section_choice = "first" if args.section == "first" else "last"
    states = {state.state: state.energy_ev for state in parse_so_states(text, which=section_choice)}
    if not states:
        raise ValueError(f"No SO-state energy table found in {args.molcas_out}")
    initial_states = _parse_state_list(args.initial_states)
    transitions = []
    for tr in parse_transition_sections(text):
        if tr.state_basis != "so":
            continue
        if args.gauge != "any" and tr.gauge != args.gauge:
            continue
        if tr.state_from not in initial_states:
            continue
        if tr.state_from not in states or tr.state_to not in states:
            continue
        transitions.append(tr)
    if args.section in {"first", "last"} and transitions:
        target_section = min(tr.section_index for tr in transitions) if args.section == "first" else max(
            tr.section_index for tr in transitions
        )
        transitions = [tr for tr in transitions if tr.section_index == target_section]
    grouped: dict[int, list[Any]] = {}
    for tr in transitions:
        grouped.setdefault(tr.state_to, []).append(tr)

    rows: list[dict[str, Any]] = []
    for state_to, group in sorted(grouped.items(), key=lambda item: states.get(item[0], 0.0)):
        energy_values = [states[tr.state_to] - states[tr.state_from] for tr in group]
        osc_values = [float(tr.oscillator_strength) for tr in group]
        if not osc_values:
            continue
        components = ";".join(f"{tr.state_from}:{tr.oscillator_strength:.8g}" for tr in sorted(group, key=lambda x: x.state_from))
        rows.append(
            {
                "state_from": initial_states[0],
                "state_to": state_to,
                "energy_ev": float(np.mean(energy_values)) + float(args.energy_shift_ev),
                "oscillator_strength": float(np.mean(osc_values)),
                "gauge": args.gauge,
                "state_basis": "so_initial_state_average" if len(initial_states) > 1 else "so_single_initial_state",
                "section_index": group[0].section_index,
                "initial_state_label": "avg_" + "_".join(str(x) for x in initial_states),
                "components_from_initial_states": components,
            }
        )

    if args.min_oscillator_strength is not None:
        rows = [row for row in rows if float(row["oscillator_strength"]) >= args.min_oscillator_strength]
    rows = [row for row in rows if float(row["oscillator_strength"]) > 0.0]
    if not rows:
        raise ValueError("No positive SO dipole transitions survived filtering")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    all_csv = outdir / f"{args.prefix}_m45_all_transitions_for_atomi.csv"
    m5_csv = outdir / f"{args.prefix}_m5_transitions_for_atomi.csv"
    m4_csv = outdir / f"{args.prefix}_m4_transitions_for_atomi.csv"
    m5_rows, m4_rows, split_info = _split_m45_rows(
        rows,
        m5_min=args.m5_min,
        m5_max=args.m5_max,
        m4_min=args.m4_min,
        m4_max=args.m4_max,
    )
    _write_transition_csv(all_csv, rows)
    _write_transition_csv(m5_csv, m5_rows)
    _write_transition_csv(m4_csv, m4_rows)

    summary = {
        "schema": SCHEMA_M45_EXTRACT,
        "molcas_out": str(args.molcas_out),
        "initial_states": initial_states,
        "gauge": args.gauge,
        "section": args.section,
        "energy_shift_ev": args.energy_shift_ev,
        "split": split_info,
        "n_all": len(rows),
        "n_m5": len(m5_rows),
        "n_m4": len(m4_rows),
        "all_transitions_csv": str(all_csv),
        "m5_transitions_csv": str(m5_csv),
        "m4_transitions_csv": str(m4_csv),
        "energy_range_all_ev": [float(min(row["energy_ev"] for row in rows)), float(max(row["energy_ev"] for row in rows))],
    }
    summary_path = outdir / f"{args.prefix}_m45_extract_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote all transitions: {all_csv}")
    print(f"Wrote M5 transitions: {m5_csv} ({len(m5_rows)} rows)")
    print(f"Wrote M4 transitions: {m4_csv} ({len(m4_rows)} rows)")
    print(f"Wrote summary: {summary_path}")
    return 0


def _plot_m45(
    *,
    m5: dict[str, Any],
    m4: dict[str, Any],
    m5_meta: dict[str, Any],
    m4_meta: dict[str, Any],
    outpath: Path,
    title: str,
    stick_height: float,
    stick_relative_threshold: float,
    broadening: str,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.7), sharex=False)
    configs = [
        (axes[0], m5, m5_meta, "#1f4e79", "U M5"),
        (axes[1], m4, m4_meta, "#9b2d20", "U M4"),
    ]
    stick_relative_threshold = max(0.0, min(1.0, stick_relative_threshold))
    stick_percent = int(round(100.0 * stick_relative_threshold))
    for ax, data, meta, color, label in configs:
        energy = data["energy"]
        spectrum = data["spectrum"]
        rows = data["rows"]
        ax.plot(energy, spectrum, color=color, lw=2.4, label="broadened envelope")
        edge_energy = meta.get("edge_energy_ev")
        if edge_energy is not None:
            ax.axvline(float(edge_energy), color="#888888", ls="--", lw=1.2, label=f"xraydb {label.split()[-1]} {float(edge_energy):.0f} eV")
        if rows:
            osc = [float(row["oscillator_strength"]) for row in rows]
            max_osc = max(osc) if osc else 1.0
            stick_rows = [row for row in rows if float(row["oscillator_strength"]) >= stick_relative_threshold * max_osc]
            for row in rows:
                if row not in stick_rows:
                    continue
                height = stick_height * float(row["oscillator_strength"]) / max_osc
                ax.vlines(float(row["energy_ev"]), 0.0, -height, color=color, alpha=0.62, lw=1.0)
        width = meta.get("core_hole_width_ev")
        width_label = f", xraydb width {float(width):.2f} eV" if width is not None else ""
        broadening_label = "Voigt" if broadening == "voigt" else broadening.replace("-", " ")
        ax.set_title(f"{label}: >={stick_percent}% strongest sticks + {broadening_label} envelope{width_label}")
        ax.set_ylabel("norm. intensity")
        ax.set_ylim(-max(stick_height * 1.12, 0.1), 1.08)
        ax.axhline(0.0, color="#888888", lw=0.8)
        ax.grid(alpha=0.20)
        ax.legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("As-computed transition energy (eV; no empirical alignment)")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=240)
    plt.close(fig)
    return True



def _read_transition_dicts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            energy_text = raw.get("energy_ev") or raw.get("transition_energy_ev") or raw.get("energy")
            osc_text = raw.get("oscillator_strength") or raw.get("osc_strength") or raw.get("intensity") or raw.get("fosc")
            if not energy_text or not osc_text:
                continue
            row = dict(raw)
            row["energy_ev"] = float(energy_text)
            row["oscillator_strength"] = float(osc_text)
            row["state_from"] = int(float(raw.get("state_from") or raw.get("from") or 0))
            row["state_to"] = int(float(raw.get("state_to") or raw.get("to") or raw.get("final_state") or 0))
            rows.append(row)
    return rows


def _write_ranked_transition_plot(path: Path, rows: list[dict[str, Any]], title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if not rows:
        return False
    energies = [float(row["energy_ev"]) for row in rows]
    osc = [float(row["oscillator_strength"]) for row in rows]
    labels = [str(row.get("rank", idx + 1)) for idx, row in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    max_osc = max(osc) if osc else 1.0
    ax.vlines(energies, 0, [val / max_osc for val in osc], color="#1f4e79", lw=1.8, alpha=0.78)
    ax.scatter(energies, [val / max_osc for val in osc], color="#9b2d20", s=18, zorder=3)
    for idx, (energy, height, label) in enumerate(zip(energies, [val / max_osc for val in osc], labels)):
        ax.text(energy, height + 0.035 + 0.028 * (idx % 3), label, ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Transition energy (eV)")
    ax.set_ylabel("relative oscillator strength")
    ax.set_title(textwrap.fill(title, width=78), fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def rank_transitions(args: argparse.Namespace) -> int:
    rows = _read_transition_dicts(args.transitions_csv)
    if args.emin is not None:
        rows = [row for row in rows if float(row["energy_ev"]) >= args.emin]
    if args.emax is not None:
        rows = [row for row in rows if float(row["energy_ev"]) <= args.emax]
    rows = sorted(rows, key=lambda row: float(row["oscillator_strength"]), reverse=True)
    max_osc = float(rows[0]["oscillator_strength"]) if rows else 1.0
    relative_threshold = max(0.0, min(1.0, float(args.relative_threshold)))
    selected = [row for row in rows if float(row["oscillator_strength"]) >= relative_threshold * max_osc]
    top = selected[: args.top] if args.top and args.top > 0 else selected
    ranked: list[dict[str, Any]] = []
    for idx, row in enumerate(top, start=1):
        item = dict(row)
        item["rank"] = idx
        item["relative_oscillator_strength"] = float(row["oscillator_strength"]) / max_osc if max_osc else 0.0
        ranked.append(item)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / args.csv_name
    json_path = outdir / args.summary_name
    keys = [
        "rank",
        "state_from",
        "state_to",
        "energy_ev",
        "oscillator_strength",
        "relative_oscillator_strength",
        "gauge",
        "state_basis",
        "section_index",
        "initial_state_label",
        "top_spinfree_weights",
        "to_state_top_spinfree_weights",
        "components_from_1_2",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)
    plot_path = outdir / args.plot_name
    plotted = False
    if args.plot:
        plotted = _write_ranked_transition_plot(plot_path, ranked, args.title or "Important Molcas/RASSI dipole transitions")
    summary = {
        "schema": SCHEMA_TRANSITIONS,
        "source": str(args.transitions_csv),
        "n_input_rows_in_window": len(rows),
        "relative_threshold": relative_threshold,
        "selection_rule": "oscillator_strength >= relative_threshold * max_oscillator_strength",
        "max_oscillator_strength": max_osc,
        "n_selected_by_threshold": len(selected),
        "n_ranked": len(ranked),
        "emin_ev": args.emin,
        "emax_ev": args.emax,
        "csv": str(csv_path),
        "plot": str(plot_path) if plotted else "",
        "top": ranked,
        "note": "Use these near-maximum rows for peak assignment and orbital/NTO follow-up; broad spectra alone are not enough for interpretation.",
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote ranked transitions: {csv_path}")
    if plotted:
        print(f"Wrote transition plot: {plot_path}")
    print(f"Wrote summary: {json_path}")
    return 0


def _classify_orbital_file(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".rasscf.h5"):
        return "rasscf_h5_active_orbitals"
    if lower.endswith(".rassi.h5"):
        return "rassi_h5_state_interaction"
    if lower.endswith(".rasorb"):
        return "rasorb_active_orbitals"
    if lower.endswith(".molden"):
        return "molden_orbitals"
    if "ntorb" in lower or "md_nto" in lower:
        return "natural_transition_orbitals"
    if "siorb" in lower:
        return "rassi_natural_orbitals"
    if "biorb" in lower:
        return "binatural_transition_orbitals"
    if "jobiph" in lower or "jobmix" in lower:
        return "molcas_wavefunction_restart"
    return "other"


def orbital_handoff(args: argparse.Namespace) -> int:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    patterns = args.glob or ["*.rasscf.h5", "*.rassi.h5", "*.RasOrb", "*.molden", "*NTORB*", "*MD_NTO*", "*SIORB*", "*BIORB*", "*.JobIph*", "*.JobMix*"]
    local_files: list[dict[str, Any]] = []
    if args.molcas_dir:
        root = args.molcas_dir.resolve()
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    local_files.append({"path": str(path), "name": path.name, "kind": _classify_orbital_file(path.name), "size_bytes": path.stat().st_size})
    archive_members: list[dict[str, Any]] = []
    if args.archive:
        with tarfile.open(args.archive, mode="r:*") as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                kind = _classify_orbital_file(name)
                if kind != "other":
                    archive_members.append({"archive": str(args.archive), "member": member.name, "name": name, "kind": kind, "size_bytes": member.size})
    record = {
        "schema": SCHEMA_ORBITAL_HANDOFF,
        "molcas_dir": str(args.molcas_dir.resolve()) if args.molcas_dir else "",
        "archive": str(args.archive.resolve()) if args.archive else "",
        "patterns": patterns,
        "local_files": local_files,
        "archive_members": archive_members,
        "recommended_next_steps": [
            "Use .rasscf.h5/.RasOrb/.molden to inspect state-averaged active orbitals and metal-ligand mixing.",
            "If NTORB/MD_NTO/BIORB/SIORB files exist, prioritize them for assigning strong dipole transitions.",
            "For large archives, extract only listed orbital files rather than unpacking the whole OpenMolcas scratch archive.",
            "Use pegamoid-bridge prepare after extracting or locating the needed files.",
        ],
    }
    json_path = outdir / args.summary_name
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = outdir / args.readme_name
    lines = [
        "# Molcas Orbital/NTO Handoff",
        "",
        "## Local Files",
    ]
    for item in local_files:
        lines.append(f"- `{item['name']}`: {item['kind']} ({item['size_bytes']} bytes)")
    lines.extend(["", "## Archive Members"])
    for item in archive_members:
        lines.append(f"- `{item['member']}`: {item['kind']} ({item['size_bytes']} bytes)")
    lines.extend(["", "## Next Steps"])
    for step in record["recommended_next_steps"]:
        lines.append(f"- {step}")
    if archive_members:
        lines.extend(["", "Example targeted extraction:", "```bash"])
        for item in archive_members[:8]:
            lines.append(f"tar -xzf {args.archive} {item['member']!r}")
        lines.append("```")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote orbital handoff: {json_path}")
    print(f"Wrote readme: {md_path}")
    return 0




def _read_mo_orbitals(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, raw in enumerate(reader, start=1):
            label = raw.get("label") or raw.get("orbital") or f"orbital_{idx}"
            block = raw.get("block") or raw.get("group") or "active"
            energy_text = raw.get("energy_ev") or raw.get("energy") or raw.get("relative_energy_ev") or idx
            occ_text = raw.get("occupation") or raw.get("occ") or raw.get("occupancy") or ""
            rows.append(
                {
                    "label": label,
                    "block": block,
                    "energy_ev": float(energy_text),
                    "occupation": float(occ_text) if str(occ_text).strip() else None,
                    "character": raw.get("character") or raw.get("assignment") or "",
                    "state": raw.get("state") or "",
                    "color": raw.get("color") or "",
                }
            )
    return rows


def _transition_arrows_for_mo(path: Path | None, max_arrows: int) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = _read_transition_dicts(path)
    rows = sorted(rows, key=lambda row: float(row["oscillator_strength"]), reverse=True)[:max_arrows]
    max_osc = max([float(row["oscillator_strength"]) for row in rows], default=1.0)
    arrows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        arrows.append(
            {
                "rank": idx,
                "state_from": row.get("state_from", 0),
                "state_to": row.get("state_to", 0),
                "energy_ev": float(row["energy_ev"]),
                "oscillator_strength": float(row["oscillator_strength"]),
                "relative_oscillator_strength": float(row["oscillator_strength"]) / max_osc if max_osc else 0.0,
                "label": row.get("label") or f"#{idx}",
            }
        )
    return arrows


def _write_mo_diagram_plot(path: Path, orbitals: list[dict[str, Any]], arrows: list[dict[str, Any]], title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if not orbitals:
        return False
    blocks = list(dict.fromkeys(str(row["block"]) for row in orbitals))
    block_x = {block: idx for idx, block in enumerate(blocks)}
    colors = {
        "ground": "#2f5597",
        "core": "#c0504d",
        "core-excited": "#c0504d",
        "ras1": "#8064a2",
        "ras2": "#666666",
        "ras3": "#9bbb59",
        "active": "#4bacc6",
    }
    fig, ax = plt.subplots(figsize=(max(9.5, 3.1 * len(blocks) + 2.5), 5.8))
    emin = min(float(row["energy_ev"]) for row in orbitals)
    emax = max(float(row["energy_ev"]) for row in orbitals)
    span = max(emax - emin, 1.0)
    for row in orbitals:
        x = block_x[str(row["block"])]
        y = float(row["energy_ev"])
        color = row["color"] or colors.get(str(row["block"]).lower(), "#444444")
        ax.hlines(y, x - 0.28, x + 0.28, color=color, lw=2.4)
        occ = row.get("occupation")
        if occ is not None:
            if occ >= 1.5:
                ax.text(x - 0.08, y + 0.02 * span, "up down", ha="center", va="bottom", fontsize=7, color=color)
            elif occ >= 0.5:
                ax.text(x, y + 0.02 * span, "up", ha="center", va="bottom", fontsize=8, color=color)
        label = str(row["label"])
        char = str(row.get("character") or "")
        ax.text(x + 0.34, y, f"{label}" + (f" ({char})" if char else ""), va="center", fontsize=8)
    if arrows:
        x0 = -0.45
        x1 = len(blocks) - 0.55
        y0 = emin + 0.18 * span
        for idx, arrow in enumerate(arrows):
            y = y0 + idx * 0.09 * span
            lw = 0.8 + 2.2 * float(arrow["relative_oscillator_strength"])
            ax.annotate(
                "",
                xy=(x1, y),
                xytext=(x0, y),
                arrowprops={"arrowstyle": "->", "lw": lw, "color": "#d97a00", "alpha": 0.72},
            )
            ax.text((x0 + x1) / 2, y + 0.025 * span, f"{arrow['label']} {arrow['energy_ev']:.2f} eV", ha="center", fontsize=7, color="#7f3f00")
    ax.set_xticks([block_x[block] for block in blocks], blocks)
    ax.set_ylabel("relative orbital / state energy (eV)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.75, len(blocks) + 1.35)
    ax.set_ylim(emin - 0.18 * span, emax + 0.32 * span)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def mo_diagram(args: argparse.Namespace) -> int:
    orbitals = _read_mo_orbitals(args.orbitals_csv)
    arrows = _transition_arrows_for_mo(args.transitions_csv, args.max_arrows)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plot_path = outdir / args.plot_name
    plotted = _write_mo_diagram_plot(plot_path, orbitals, arrows, args.title)
    summary = {
        "schema": SCHEMA_MO_DIAGRAM,
        "orbitals_csv": str(args.orbitals_csv),
        "transitions_csv": str(args.transitions_csv) if args.transitions_csv else "",
        "n_orbitals": len(orbitals),
        "n_transition_arrows": len(arrows),
        "plot": str(plot_path) if plotted else "",
        "orbitals": orbitals,
        "transition_arrows": arrows,
        "note": "Schematic diagram for interpretation only; orbital shapes still come from Pegamoid/Molden/NTO viewers.",
    }
    summary_path = outdir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote MO diagram summary: {summary_path}")
    if plotted:
        print(f"Wrote MO diagram: {plot_path}")
    return 0



def _draw_level(ax: Any, x: float, y: float, label: str, *, side: str = "right", width: float = 0.46, lw: float = 2.4) -> None:
    ax.hlines(y, x - width / 2, x + width / 2, color="black", lw=lw)
    dx = 0.13 if side == "right" else -0.13
    ha = "left" if side == "right" else "right"
    ax.text(x + (width / 2 + dx if side == "right" else -width / 2 + dx), y, label, va="center", ha=ha, fontsize=9)


def _draw_correlation(ax: Any, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.plot([x0, x1], [y0, y1], color="#555555", lw=0.8, ls=":", alpha=0.95)


def _read_xyz_cluster(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ file is too short: {path}")
    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First XYZ line must be atom count: {path}") from exc
    comment = lines[1].strip()
    atoms: list[dict[str, Any]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        atoms.append(
            {
                "element": parts[0],
                "xyz": np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float),
            }
        )
    if len(atoms) != n_atoms:
        raise ValueError(f"XYZ atom count mismatch in {path}: expected {n_atoms}, parsed {len(atoms)}")
    return {"path": str(path), "comment": comment, "atoms": atoms}


def _cluster_ligand_field_descriptor(path: Path, central_element: str, ligand_element: str) -> dict[str, Any]:
    cluster = _read_xyz_cluster(path)
    central = [atom for atom in cluster["atoms"] if atom["element"].lower() == central_element.lower()]
    if not central:
        raise ValueError(f"No central element {central_element!r} found in {path}")
    center = central[0]["xyz"]
    ligands = [atom for atom in cluster["atoms"] if atom["element"].lower() == ligand_element.lower()]
    if not ligands:
        raise ValueError(f"No ligand element {ligand_element!r} found in {path}")
    rel = np.vstack([atom["xyz"] - center for atom in ligands])
    distances = np.linalg.norm(rel, axis=1)
    cov = np.cov(rel.T) if len(ligands) > 1 else np.eye(3)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    shape_ratio = float(eigvals[0] / eigvals[-1]) if eigvals[-1] > 0 else 0.0
    return {
        "structure": str(path),
        "comment": cluster["comment"],
        "central_element": central_element,
        "ligand_element": ligand_element,
        "coordination_number": len(ligands),
        "bond_distance_min_A": float(np.min(distances)),
        "bond_distance_max_A": float(np.max(distances)),
        "bond_distance_mean_A": float(np.mean(distances)),
        "bond_distance_std_A": float(np.std(distances)),
        "ligand_cloud_covariance_eigenvalues_A2": [float(x) for x in eigvals],
        "ligand_cloud_min_max_eigenvalue_ratio": shape_ratio,
    }


def _draw_local_cluster_u5f_splitting(
    ax: Any,
    *,
    descriptor: dict[str, Any] | None,
    central_element: str,
    ligand_element: str,
) -> dict[str, Any]:
    x_so, x_mix, x_lf = 0.0, 1.45, 2.92
    so = {
        "5f$_{5/2}$": 0.86,
        "5f$_{7/2}$": 1.62,
    }
    # Low-symmetry actinide clusters should be treated as seven local 5f
    # ligand-field orbitals coupled by spin-orbit interaction into Kramers
    # doublets.  No linear-molecule sigma/pi/delta/phi labels are assigned here.
    mixed = [
        ("KD1", 0.66),
        ("KD2", 0.86),
        ("KD3", 1.07),
        ("KD4", 1.46),
        ("KD5", 1.72),
        ("KD6", 2.02),
        ("KD7", 2.40),
    ]
    lf = [
        ("local 5f LF1", 0.62),
        ("local 5f LF2", 0.82),
        ("local 5f LF3", 1.04),
        ("local 5f LF4", 1.34),
        ("local 5f LF5", 1.62),
        ("local 5f LF6", 1.96),
        ("local 5f LF7", 2.32),
    ]

    for label, y in so.items():
        _draw_level(ax, x_so, y, label, side="left", width=0.52, lw=2.7)
    for label, y in mixed:
        _draw_level(ax, x_mix, y, label, side="right", width=0.42, lw=2.15)
    for label, y in lf:
        _draw_level(ax, x_lf, y, label, side="right", width=0.45, lw=2.5)

    for _, y in mixed[:3]:
        _draw_correlation(ax, x_so + 0.26, so["5f$_{5/2}$"], x_mix - 0.21, y)
    for _, y in mixed[3:]:
        _draw_correlation(ax, x_so + 0.26, so["5f$_{7/2}$"], x_mix - 0.21, y)
    for (_, ym), (_, yl) in zip(mixed, lf):
        _draw_correlation(ax, x_mix + 0.21, ym, x_lf - 0.23, yl)

    ax.text(x_so, 2.85, "SO", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(x_mix, 2.85, "SO + local LF", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(x_lf, 2.85, "local LF", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(x_so, 0.18, f"{central_element}$^{{5+}}$ 5f$^1$\natomic-like", ha="center", va="top", fontsize=9)
    if descriptor:
        cn = descriptor["coordination_number"]
        mean = descriptor["bond_distance_mean_A"]
        std = descriptor["bond_distance_std_A"]
        min_d = descriptor["bond_distance_min_A"]
        max_d = descriptor["bond_distance_max_A"]
        middle = f"{central_element}{ligand_element}{cn} cluster\nCN={cn}, {central_element}-{ligand_element}={mean:.2f}+-{std:.2f} A"
        note = f"range {min_d:.2f}-{max_d:.2f} A; no point symmetry assumed"
    else:
        middle = f"{central_element}{ligand_element}$_n$ local cluster"
        note = "no point symmetry assumed"
    ax.text(x_mix, 0.18, middle, ha="center", va="top", fontsize=9)
    ax.text(x_lf, 0.18, note, ha="center", va="top", fontsize=8.5)
    ax.text(
        0.03,
        2.62,
        "Schematic only: assign LF characters from Molcas orbitals/NTOs,\nnot from uranyl symmetry labels.",
        ha="left",
        va="top",
        fontsize=8.2,
        color="#555555",
    )
    return {"so_levels": so, "mixed_levels": dict(mixed), "lf_levels": dict(lf)}


def _draw_uranyl_reference_u5f_splitting(ax: Any) -> dict[str, Any]:
    x_so, x_mid, x_lf = 0.0, 1.38, 2.75
    so = {
        "5f$_{5/2}$": 0.82,
        "5f$_{7/2}$": 1.42,
    }
    mid = [
        ("$\\omega$=5/2", 0.72),
        ("$\\omega$=3/2", 0.94),
        ("$\\omega$=7/2", 1.08),
        ("$\\omega$=5/2", 1.22),
        ("$\\omega$=1/2", 2.55),
        ("$\\omega$=3/2", 2.72),
        ("$\\omega$=1/2", 4.56),
    ]
    lf = [
        ("5f(phi$_u$)", 0.95),
        ("5f(delta$_u$)", 1.05),
        ("5f(pi$_u^*$)", 2.63),
        ("5f(sigma$_u^*$)", 4.56),
    ]
    for label, y in so.items():
        _draw_level(ax, x_so, y, label, side="left", width=0.54, lw=2.7)
    for label, y in mid:
        _draw_level(ax, x_mid, y, label, side="right", width=0.50, lw=2.4)
    for label, y in lf:
        _draw_level(ax, x_lf, y, label, side="right", width=0.52, lw=2.7)

    for y in [0.72, 0.94, 1.08, 1.22]:
        _draw_correlation(ax, x_so + 0.27, so["5f$_{5/2}$"], x_mid - 0.25, y)
    for y in [2.55, 2.72, 4.56]:
        _draw_correlation(ax, x_so + 0.27, so["5f$_{7/2}$"], x_mid - 0.25, y)
    for y in [0.72, 0.94, 1.08, 1.22]:
        target = 0.95 if y < 1.0 else 1.05
        _draw_correlation(ax, x_mid + 0.25, y, x_lf - 0.26, target)
    _draw_correlation(ax, x_mid + 0.25, 2.55, x_lf - 0.26, 2.63)
    _draw_correlation(ax, x_mid + 0.25, 2.72, x_lf - 0.26, 2.63)
    _draw_correlation(ax, x_mid + 0.25, 4.56, x_lf - 0.26, 4.56)

    ax.text(x_so, 5.05, "SO", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(x_lf, 5.05, "LF", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(x_so, 0.18, "U$^{6+}$", ha="center", va="top", fontsize=10)
    ax.text(x_mid, 0.18, "UO$_2^{2+}$ (D$_{\\infty h}^{*}$)", ha="center", va="top", fontsize=10)
    ax.text(x_lf, 0.18, "UO$_2^{2+}$ (D$_{\\infty h}$)", ha="center", va="top", fontsize=10)
    return {"so_levels": so, "mixed_levels": dict(mid), "lf_levels": dict(lf)}


def u5f_splitting(args: argparse.Namespace) -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for u5f-splitting") from exc

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plot_path = outdir / args.plot_name
    descriptor = None
    if args.structure is not None:
        descriptor = _cluster_ligand_field_descriptor(args.structure, args.central_element, args.ligand_element)

    if args.mode == "uranyl-reference":
        fig, ax = plt.subplots(figsize=(6.6, 6.2))
        levels = _draw_uranyl_reference_u5f_splitting(ax)
        ax.set_xlim(-0.82, 3.35)
        ax.set_ylim(0.0, 5.35)
        source_style = "Polly and Bagus, Inorg. Chem. 2026, Figure 1 uranyl reference"
        note = "Uranyl reference schematic; use only when the local chemistry justifies linear-molecule labels."
    else:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        levels = _draw_local_cluster_u5f_splitting(
            ax,
            descriptor=descriptor,
            central_element=args.central_element,
            ligand_element=args.ligand_element,
        )
        ax.set_xlim(-0.80, 3.55)
        ax.set_ylim(0.0, 3.10)
        source_style = "Polly and Bagus, Inorg. Chem. 2026, Figure 1 conceptual SO/LF template"
        note = (
            "Local-cluster schematic; no point symmetry, uranyl sigma/pi/delta/phi, or quantitative energy "
            "splitting is assumed.  Assign level characters from Molcas orbitals, NTOs, and transition analysis."
        )

    if args.title:
        ax.set_title(textwrap.fill(args.title, width=68), fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema": SCHEMA_U5F_SPLITTING,
        "plot": str(plot_path),
        "mode": args.mode,
        "source_style": source_style,
        "note": note,
        "structure_descriptor": descriptor,
        **levels,
    }
    summary_path = outdir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote U 5f SO/LF splitting diagram: {plot_path}")
    print(f"Wrote summary: {summary_path}")
    return 0



def m45_two_panel(args: argparse.Namespace) -> int:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    m5_width, m5_meta = _edge_width(args.element, args.m5_edge, args.no_xraydb, args.m5_lorentzian_fwhm)
    m4_width, m4_meta = _edge_width(args.element, args.m4_edge, args.no_xraydb, args.m4_lorentzian_fwhm)
    m5 = _broaden_edge(
        args.m5_transitions_csv,
        element=args.element,
        edge=args.m5_edge,
        emin=args.m5_emin,
        emax=args.m5_emax,
        step=args.step,
        gaussian_fwhm=args.gaussian_fwhm,
        lorentzian_fwhm=m5_width,
        broadening=args.broadening,
        eta=args.pseudo_voigt_eta,
        normalize=args.normalize,
        energy_shift_ev=args.energy_shift_ev,
        outdir=outdir,
        prefix=args.prefix + "_m5",
    )
    m4 = _broaden_edge(
        args.m4_transitions_csv,
        element=args.element,
        edge=args.m4_edge,
        emin=args.m4_emin,
        emax=args.m4_emax,
        step=args.step,
        gaussian_fwhm=args.gaussian_fwhm,
        lorentzian_fwhm=m4_width,
        broadening=args.broadening,
        eta=args.pseudo_voigt_eta,
        normalize=args.normalize,
        energy_shift_ev=args.energy_shift_ev,
        outdir=outdir,
        prefix=args.prefix + "_m4",
    )
    plot_path = outdir / args.plot_name
    plotted = _plot_m45(
        m5=m5,
        m4=m4,
        m5_meta=m5_meta,
        m4_meta=m4_meta,
        outpath=plot_path,
        title=args.title,
        stick_height=args.stick_height,
        stick_relative_threshold=args.stick_relative_threshold,
        broadening=args.broadening,
    )
    summary = {
        "schema": SCHEMA_M45,
        "element": args.element,
        "m5": {k: v for k, v in m5.items() if k not in {"energy", "spectrum", "rows"}},
        "m4": {k: v for k, v in m4.items() if k not in {"energy", "spectrum", "rows"}},
        "m5_xraydb": m5_meta,
        "m4_xraydb": m4_meta,
        "gaussian_fwhm_ev": args.gaussian_fwhm,
        "broadening": args.broadening,
        "normalize": args.normalize,
        "energy_shift_ev": args.energy_shift_ev,
        "stick_height": args.stick_height,
        "stick_relative_threshold": args.stick_relative_threshold,
        "stick_selection_rule": "sticks shown only when oscillator_strength >= stick_relative_threshold * edge_max_oscillator_strength",
        "plot": str(plot_path) if plotted else "",
        "note": "As-computed transition energies; apply standard alignment before direct experimental comparison. The envelope uses all positive transitions; the visible/report sticks use the configured near-maximum intensity threshold.",
    }
    summary_path = outdir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote M5 spectrum: {m5['spectrum_csv']}")
    print(f"Wrote M4 spectrum: {m4['spectrum_csv']}")
    if plotted:
        print(f"Wrote two-panel plot: {plot_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Postanalysis workflow helpers for OpenMolcas spectroscopy runs.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("workflow", help="Print Sarah/Anna Molcas postanalysis workflow and toolset.")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=print_workflow)

    p = sub.add_parser("rank-transitions", help="Rank and plot important dipole transitions from a transition CSV.")
    p.add_argument("--transitions-csv", type=Path, required=True)
    p.add_argument("--top", type=int, default=0, help="Optional cap after relative-threshold filtering; 0 keeps all selected rows.")
    p.add_argument("--relative-threshold", type=float, default=0.95, help="Keep rows with oscillator strength >= this fraction of the maximum.")
    p.add_argument("--emin", type=float)
    p.add_argument("--emax", type=float)
    p.add_argument("--outdir", type=Path, default=Path("molcas_important_transitions"))
    p.add_argument("--csv-name", default="important_dipole_transitions.csv")
    p.add_argument("--summary-name", default="important_dipole_transitions_summary.json")
    p.add_argument("--plot-name", default="important_dipole_transitions.png")
    p.add_argument("--title", default="")
    p.add_argument("--plot", action="store_true")
    p.set_defaults(func=rank_transitions)

    p = sub.add_parser("orbital-handoff", help="Find Molcas orbital/NTO files and write a Pegamoid/NTO handoff.")
    p.add_argument("--molcas-dir", type=Path)
    p.add_argument("--archive", type=Path)
    p.add_argument("--glob", action="append", default=[])
    p.add_argument("--outdir", type=Path, default=Path("molcas_orbital_handoff"))
    p.add_argument("--summary-name", default="molcas_orbital_handoff.json")
    p.add_argument("--readme-name", default="MOLCAS_ORBITAL_HANDOFF.md")
    p.set_defaults(func=orbital_handoff)

    p = sub.add_parser("mo-diagram", help="Build a schematic MO diagram with optional important-transition arrows.")
    p.add_argument("--orbitals-csv", type=Path, required=True)
    p.add_argument("--transitions-csv", type=Path)
    p.add_argument("--max-arrows", type=int, default=6)
    p.add_argument("--outdir", type=Path, default=Path("molcas_mo_diagram"))
    p.add_argument("--plot-name", default="molcas_schematic_mo_diagram.png")
    p.add_argument("--summary-name", default="molcas_schematic_mo_diagram_summary.json")
    p.add_argument("--title", default="Schematic MO diagram from Molcas postanalysis")
    p.set_defaults(func=mo_diagram)

    p = sub.add_parser("u5f-splitting", help="Build a U 5f spin-orbit versus local ligand-field splitting diagram.")
    p.add_argument("--outdir", type=Path, default=Path("u5f_so_lf_splitting"))
    p.add_argument("--plot-name", default="u5f_so_lf_splitting.png")
    p.add_argument("--summary-name", default="u5f_so_lf_splitting_summary.json")
    p.add_argument("--title", default="U 5f spin-orbit versus ligand-field splitting")
    p.add_argument("--structure", type=Path, help="Optional XYZ cluster; if given, annotate CN and local geometry descriptors.")
    p.add_argument("--central-element", default="U")
    p.add_argument("--ligand-element", default="O")
    p.add_argument(
        "--mode",
        choices=["local-cluster", "uranyl-reference"],
        default="local-cluster",
        help="Use local low-symmetry cluster labels by default; uranyl-reference keeps the Polly/Bagus linear-molecule labels.",
    )
    p.set_defaults(func=u5f_splitting)

    p = sub.add_parser(
        "extract-m45-transitions",
        help="Extract/average SO RASSI transitions from a Molcas output and split them into M5/M4 CSVs.",
    )
    p.add_argument("--molcas-out", type=Path, required=True)
    p.add_argument("--initial-states", default="1,2", help="Comma/space separated SO initial states to average, e.g. 1,2.")
    p.add_argument("--gauge", choices=["length", "velocity", "any"], default="length")
    p.add_argument("--section", choices=["last", "first", "all"], default="last")
    p.add_argument("--energy-shift-ev", type=float, default=0.0)
    p.add_argument("--min-oscillator-strength", type=float)
    p.add_argument("--m5-min", type=float)
    p.add_argument("--m5-max", type=float)
    p.add_argument("--m4-min", type=float)
    p.add_argument("--m4-max", type=float)
    p.add_argument("--outdir", type=Path, default=Path("molcas_m45_transitions"))
    p.add_argument("--prefix", default="molcas_ground_initial_avg")
    p.set_defaults(func=extract_m45_transitions)

    p = sub.add_parser("m45-two-panel", help="Build a two-panel U M5/M4 XANES envelope plus tall-stick figure.")
    p.add_argument("--m5-transitions-csv", type=Path, required=True)
    p.add_argument("--m4-transitions-csv", type=Path, required=True)
    p.add_argument("--element", default="U")
    p.add_argument("--m5-edge", default="M5")
    p.add_argument("--m4-edge", default="M4")
    p.add_argument("--energy-shift-ev", type=float, default=0.0)
    p.add_argument("--gaussian-fwhm", type=float, default=1.0)
    p.add_argument("--m5-lorentzian-fwhm", type=float)
    p.add_argument("--m4-lorentzian-fwhm", type=float)
    p.add_argument("--broadening", choices=("voigt", "pseudo-voigt", "gaussian", "lorentzian"), default="voigt")
    p.add_argument("--pseudo-voigt-eta", type=float, default=0.5)
    p.add_argument("--normalize", choices=("max", "area", "none"), default="max")
    p.add_argument("--m5-emin", type=float)
    p.add_argument("--m5-emax", type=float)
    p.add_argument("--m4-emin", type=float)
    p.add_argument("--m4-emax", type=float)
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--stick-height", type=float, default=0.45)
    p.add_argument("--stick-relative-threshold", type=float, default=0.95, help="Show stick transitions with intensity >= this fraction of the edge maximum.")
    p.add_argument("--outdir", type=Path, default=Path("molcas_m45_postanalysis"))
    p.add_argument("--prefix", default="molcas_m45")
    p.add_argument("--plot-name", default="molcas_u_m45_xanes_2panel_tall_sticks.png")
    p.add_argument("--summary-name", default="molcas_u_m45_xanes_2panel_summary.json")
    p.add_argument("--title", default="OpenMolcas/RASSI U M4,5 XANES")
    p.add_argument("--no-xraydb", action="store_true")
    p.set_defaults(func=m45_two_panel)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
