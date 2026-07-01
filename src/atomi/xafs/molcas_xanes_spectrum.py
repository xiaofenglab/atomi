"""Convert OpenMolcas/RASSI transition strengths into broadened XANES spectra."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HARTREE_TO_EV = 27.211386245988
SCHEMA_SUMMARY = "atomi.molcas_xanes_spectrum.v1"


@dataclass(frozen=True)
class SOState:
    state: int
    energy_raw_au: float
    energy_ev: float


@dataclass(frozen=True)
class Transition:
    state_from: int
    state_to: int
    oscillator_strength: float
    energy_ev: float | None
    gauge: str
    state_basis: str
    source: str


def _float(text: str) -> float:
    return float(text.replace("D", "E"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_so_states(text: str) -> list[SOState]:
    """Parse the SO-state mixing table and return transition energies."""

    marker = re.search(r"SO State\s+Total energy\s*\(au\)\s+Spin-free states, spin, and weights", text)
    if not marker:
        return []
    section = text[marker.end() :]
    stop = re.search(r"\n\s*-{20,}\s*\n\s*\n", section)
    if stop:
        section = section[: stop.start()]
    rows: list[tuple[int, float]] = []
    for line in section.splitlines():
        m = re.match(r"^\s*(\d+)\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+", line)
        if m:
            rows.append((int(m.group(1)), _float(m.group(2))))
    if not rows:
        return []
    e0 = rows[0][1]
    use_relative = abs(e0) > 1.0e-6
    states = []
    for state, energy in rows:
        rel_au = energy - e0 if use_relative else energy
        states.append(SOState(state=state, energy_raw_au=energy, energy_ev=rel_au * HARTREE_TO_EV))
    return states


def parse_transition_sections(text: str) -> list[Transition]:
    """Parse OpenMolcas transition-strength tables.

    The most useful RASSI tables look like:
    ``++ Dipole transition strengths (SO states):`` followed by
    ``From To Osc. strength`` rows.  Velocity gauge tables are retained too.
    """

    transitions: list[Transition] = []
    lines = text.splitlines()
    active_title = ""
    active_gauge = ""
    active_basis = ""
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if stripped.startswith("++") and "transition strengths" in lower:
            active_title = stripped
            active_basis = "so" if "(so states)" in lower else "spin-free" if "(spin-free states)" in lower else "unknown"
            if "velocity" in lower:
                active_gauge = "velocity"
            elif "dipole" in lower:
                active_gauge = "length"
            elif "second-order" in lower:
                active_gauge = "second-order"
            else:
                active_gauge = "unknown"
            continue
        if not active_title:
            continue
        if stripped.startswith("++") or stripped.startswith("--"):
            continue
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\b", line)
        if m:
            transitions.append(
                Transition(
                    state_from=int(m.group(1)),
                    state_to=int(m.group(2)),
                    oscillator_strength=_float(m.group(3)),
                    energy_ev=None,
                    gauge=active_gauge,
                    state_basis=active_basis,
                    source=active_title,
                )
            )
    return transitions


def transitions_from_output(path: Path, *, gauge: str = "length", state_from: int = 1) -> list[Transition]:
    text = read_text(path)
    states = {s.state: s.energy_ev for s in parse_so_states(text)}
    rows = []
    for tr in parse_transition_sections(text):
        if tr.state_basis != "so":
            continue
        if gauge != "any" and tr.gauge != gauge:
            continue
        if tr.state_from != state_from:
            continue
        if tr.state_to not in states or tr.state_from not in states:
            continue
        rows.append(
            Transition(
                state_from=tr.state_from,
                state_to=tr.state_to,
                oscillator_strength=tr.oscillator_strength,
                energy_ev=states[tr.state_to] - states[tr.state_from],
                gauge=tr.gauge,
                state_basis=tr.state_basis,
                source=str(path),
            )
        )
    return rows


def transitions_from_csv(path: Path) -> list[Transition]:
    rows: list[Transition] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            energy_text = raw.get("energy_ev") or raw.get("energy") or raw.get("transition_energy_ev")
            intensity_text = raw.get("oscillator_strength") or raw.get("osc_strength") or raw.get("intensity") or raw.get("fosc")
            if not energy_text or not intensity_text:
                continue
            rows.append(
                Transition(
                    state_from=int(float(raw.get("state_from") or raw.get("from") or 0)),
                    state_to=int(float(raw.get("state_to") or raw.get("to") or 0)),
                    oscillator_strength=float(intensity_text),
                    energy_ev=float(energy_text),
                    gauge=raw.get("gauge") or "csv",
                    state_basis=raw.get("state_basis") or "csv",
                    source=str(path),
                )
            )
    return rows


def xraydb_metadata(element: str, edge: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"element": element, "edge": edge, "source": "xraydb"}
    try:
        import xraydb  # type: ignore

        edge_obj = xraydb.xray_edge(element, edge)
        meta["edge_energy_ev"] = float(edge_obj.energy)
        meta["fluorescence_yield"] = float(edge_obj.fyield)
        meta["jump_ratio"] = float(edge_obj.jump_ratio)
        width = xraydb.core_width(element, edge)
        if width is not None:
            meta["core_hole_width_ev"] = float(width)
    except Exception as exc:
        meta["source"] = "unavailable"
        meta["warning"] = str(exc)
    return meta


def gaussian_kernel(grid: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    if sigma <= 0:
        raise ValueError("Gaussian FWHM must be positive")
    return np.exp(-0.5 * ((grid - center) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def lorentzian_kernel(grid: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    gamma = fwhm / 2.0
    if gamma <= 0:
        raise ValueError("Lorentzian FWHM must be positive")
    return (gamma / math.pi) / ((grid - center) ** 2 + gamma**2)


def broaden(
    transitions: list[Transition],
    *,
    emin: float | None,
    emax: float | None,
    step: float,
    energy_shift: float,
    gaussian_fwhm: float,
    lorentzian_fwhm: float,
    mode: str,
    eta: float,
    normalize: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    usable = [tr for tr in transitions if tr.energy_ev is not None and tr.oscillator_strength > 0]
    if not usable:
        raise ValueError("No positive transitions with energies were found.")
    energies = np.array([float(tr.energy_ev) + energy_shift for tr in usable], dtype=float)
    intensities = np.array([float(tr.oscillator_strength) for tr in usable], dtype=float)
    if emin is None:
        emin = float(np.min(energies) - 20.0)
    if emax is None:
        emax = float(np.max(energies) + 20.0)
    grid = np.arange(emin, emax + step * 0.5, step, dtype=float)
    spectrum = np.zeros_like(grid)
    eta = max(0.0, min(1.0, eta))
    for energy, intensity in zip(energies, intensities):
        if mode == "gaussian":
            line = gaussian_kernel(grid, energy, gaussian_fwhm)
        elif mode == "lorentzian":
            line = lorentzian_kernel(grid, energy, lorentzian_fwhm)
        else:
            line = eta * lorentzian_kernel(grid, energy, lorentzian_fwhm) + (1.0 - eta) * gaussian_kernel(
                grid, energy, gaussian_fwhm
            )
        spectrum += intensity * line
    if normalize == "max" and float(np.max(spectrum)) > 0:
        spectrum = spectrum / float(np.max(spectrum))
    elif normalize == "area":
        area = float(np.trapz(spectrum, grid))
        if area > 0:
            spectrum = spectrum / area
    rows = [
        {
            "state_from": tr.state_from,
            "state_to": tr.state_to,
            "energy_ev": float(en),
            "oscillator_strength": tr.oscillator_strength,
            "gauge": tr.gauge,
            "state_basis": tr.state_basis,
        }
        for tr, en in zip(usable, energies)
    ]
    return grid, spectrum, rows


def write_transitions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = ["state_from", "state_to", "energy_ev", "oscillator_strength", "gauge", "state_basis"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_spectrum_csv(path: Path, energy: np.ndarray, intensity: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy_ev", "intensity"])
        for e, y in zip(energy, intensity):
            writer.writerow([f"{float(e):.8f}", f"{float(y):.12g}"])


def maybe_plot(path: Path, energy: np.ndarray, intensity: np.ndarray, rows: list[dict[str, Any]], title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(energy, intensity, color="#1f4e79", lw=1.8)
    if rows:
        max_y = float(np.max(intensity)) if len(intensity) else 1.0
        stem_y = [row["oscillator_strength"] for row in rows]
        max_stem = max(stem_y) if stem_y else 1.0
        for row in rows:
            ax.vlines(row["energy_ev"], 0, max_y * 0.18 * row["oscillator_strength"] / max_stem, color="#9b2d20", alpha=0.5)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "molcas_out", None):
        transitions = transitions_from_output(args.molcas_out, gauge=args.gauge, state_from=args.from_state)
        source = str(args.molcas_out)
    else:
        transitions = transitions_from_csv(args.transitions_csv)
        source = str(args.transitions_csv)

    edge_meta = {"element": args.element, "edge": args.edge, "source": "disabled"} if args.no_xraydb else xraydb_metadata(args.element, args.edge)
    lorentzian = args.lorentzian_fwhm
    if lorentzian is None:
        lorentzian = float(edge_meta.get("core_hole_width_ev", 1.0))
    energy, spectrum, rows = broaden(
        transitions,
        emin=args.emin,
        emax=args.emax,
        step=args.step,
        energy_shift=args.energy_shift_ev,
        gaussian_fwhm=args.gaussian_fwhm,
        lorentzian_fwhm=lorentzian,
        mode=args.broadening,
        eta=args.pseudo_voigt_eta,
        normalize=args.normalize,
    )
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    spectrum_csv = outdir / args.spectrum_name
    transitions_csv = outdir / args.transitions_name
    summary_json = outdir / args.summary_name
    write_spectrum_csv(spectrum_csv, energy, spectrum)
    write_transitions_csv(transitions_csv, rows)
    plot_path = outdir / args.plot_name
    plotted = False if args.no_plot else maybe_plot(plot_path, energy, spectrum, rows, args.title or f"{args.element} {args.edge} Molcas XANES")
    summary = {
        "schema": SCHEMA_SUMMARY,
        "source": source,
        "element": args.element,
        "edge": args.edge,
        "xraydb": edge_meta,
        "n_transitions_total": len(transitions),
        "n_transitions_used": len(rows),
        "energy_shift_ev": args.energy_shift_ev,
        "gaussian_fwhm_ev": args.gaussian_fwhm,
        "lorentzian_fwhm_ev": lorentzian,
        "broadening": args.broadening,
        "normalize": args.normalize,
        "spectrum_csv": str(spectrum_csv),
        "transitions_csv": str(transitions_csv),
        "plot": str(plot_path) if plotted else "",
        "peak_energy_ev": float(energy[int(np.argmax(spectrum))]),
        "intensity_max": float(np.max(spectrum)),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote spectrum: {spectrum_csv}")
    print(f"Wrote transitions: {transitions_csv}")
    if plotted:
        print(f"Wrote plot: {plot_path}")
    print(f"Wrote summary: {summary_json}")
    return summary


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--element", default="Ga")
    parser.add_argument("--edge", default="K")
    parser.add_argument("--gauge", choices=("length", "velocity", "any"), default="length")
    parser.add_argument("--from-state", type=int, default=1)
    parser.add_argument("--energy-shift-ev", type=float, default=0.0)
    parser.add_argument("--gaussian-fwhm", type=float, default=1.0)
    parser.add_argument("--lorentzian-fwhm", type=float)
    parser.add_argument("--broadening", choices=("pseudo-voigt", "gaussian", "lorentzian"), default="pseudo-voigt")
    parser.add_argument("--pseudo-voigt-eta", type=float, default=0.5)
    parser.add_argument("--normalize", choices=("max", "area", "none"), default="max")
    parser.add_argument("--emin", type=float)
    parser.add_argument("--emax", type=float)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--outdir", type=Path, default=Path("molcas_xanes_spectrum"))
    parser.add_argument("--spectrum-name", default="molcas_xanes_spectrum.csv")
    parser.add_argument("--transitions-name", default="molcas_xanes_transitions.csv")
    parser.add_argument("--summary-name", default="molcas_xanes_summary.json")
    parser.add_argument("--plot-name", default="molcas_xanes_spectrum.png")
    parser.add_argument("--title", default="")
    parser.add_argument("--no-xraydb", action="store_true", help="Skip xraydb edge/core-width lookup; useful for minimal tests.")
    parser.add_argument("--no-plot", action="store_true", help="Skip optional matplotlib PNG generation.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Broaden OpenMolcas/RASSI transitions into XANES-like spectra.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("from-output", help="Parse SO-state dipole strengths from a Molcas output.")
    p.add_argument("--molcas-out", type=Path, required=True)
    add_common(p)
    p.set_defaults(func=run)

    p = sub.add_parser("from-csv", help="Broaden a CSV with energy_ev and intensity/oscillator_strength columns.")
    p.add_argument("--transitions-csv", type=Path, required=True)
    add_common(p)
    p.set_defaults(func=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
