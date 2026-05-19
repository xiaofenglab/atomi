"""Fail-fast stage-1 screening for VASP spin/localization branches."""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atomi.vasp.checks import (
    _latest_vasp_energy,
    _run_is_done,
    array_indexed_output_candidates,
)
from atomi.vasp.magmom import existing_magmom_values
from atomi.vasp.spin_report import (
    AtomReport,
    apply_moment_guards,
    build_atom_reports,
    changed_counts_by_element,
    element_order_from_atoms,
    extract_first_available_magnetization,
    first_incar_file,
    first_species_file,
    initial_order_from_atoms,
    magnetization_candidate_files,
    parse_moment_guards,
    read_species_labels,
    summarize_counts,
)


DONE_TEXT = "General timing and accounting informations for this job"
CONVERGED_TEXT = "reached required accuracy"
ERROR_PATTERNS = (
    "zbrent: fatal error",
    "edddav:",
    "error in subspace rotation",
    "internal error",
)
VASP_EVIDENCE_NAMES = (
    "INCAR",
    "OSZICAR",
    "OUTCAR",
    "OUTCAR.gz",
    "vasprun.xml",
    "vasprun.xml.gz",
)
VASP_GLOB_EVIDENCE = ("vasp.out*",)


@dataclass
class BranchInput:
    frame_id: str
    branch_id: str
    run_dir: Path
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class BranchReport:
    frame_id: str
    branch_id: str
    run_dir: Path
    action: str = "warning"
    survivor: bool = False
    rank_in_frame: int | None = None
    energy_eV: float | None = None
    relative_energy_eV: float | None = None
    energy_kind: str = ""
    energy_source: Path | None = None
    output_run_dir: Path | None = None
    convergence_status: str = "NO_OUTPUT"
    newest_output: Path | None = None
    newest_output_age_min: float | None = None
    current_step: int | None = None
    last_de: float | None = None
    scf_status: str = "UNKNOWN"
    scf_steps: int = 0
    scf_oscillation_ratio: float | None = None
    scf_recent_median_abs_de: float | None = None
    mag_source: Path | None = None
    mag_status: str = "NO_OUTCAR"
    total_moment: float | None = None
    max_abs_moment: float | None = None
    changed_count: int = 0
    changed_by_element: dict[str, int] = field(default_factory=dict)
    initial_element_order: dict[str, str] = field(default_factory=dict)
    element_order: dict[str, str] = field(default_factory=dict)
    element_sum: dict[str, float] = field(default_factory=dict)
    summary_counts: dict[str, int] = field(default_factory=dict)
    physics_guard_status: str = "NOT_APPLIED"
    physics_guard_bad_count: int = 0
    physics_guard_bad_by_element: dict[str, int] = field(default_factory=dict)
    tracked_site_status: str = "NOT_APPLIED"
    tracked_site_notes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _read_text(path: Path, limit: int | None = None) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        if limit is None:
            return handle.read()
        data = handle.read()
    return data[-limit:]


def _has_vasp_evidence(path: Path) -> bool:
    if any((path / name).is_file() for name in VASP_EVIDENCE_NAMES):
        return True
    return any(any(path.glob(pattern)) for pattern in VASP_GLOB_EVIDENCE)


def _resolve_run_dir(raw: str, base: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_branch_index(path: Path) -> list[BranchInput]:
    rows: list[BranchInput] = []
    base = path.parent.resolve()
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            raw_path = (
                row.get("run_dir")
                or row.get("path")
                or row.get("branch_dir")
                or row.get("dir")
                or row.get("folder")
                or ""
            ).strip()
            if not raw_path:
                raise ValueError(f"{path}:{line_number}: missing run_dir/path column")
            run_dir = _resolve_run_dir(raw_path, base)
            frame_id = (
                row.get("frame_id")
                or row.get("frame")
                or row.get("md_frame")
                or row.get("snapshot")
                or run_dir.parent.name
            )
            branch_id = row.get("branch_id") or row.get("branch") or row.get("name") or run_dir.name
            rows.append(
                BranchInput(
                    frame_id=str(frame_id),
                    branch_id=str(branch_id),
                    run_dir=run_dir,
                    metadata={key: value for key, value in row.items() if value is not None},
                )
            )
    return rows


def load_branch_runlist(path: Path, single_frame_id: str | None = None) -> list[BranchInput]:
    rows: list[BranchInput] = []
    base = path.parent.resolve()
    run_index = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        run_index += 1
        run_dir = _resolve_run_dir(line, base)
        rows.append(
            BranchInput(
                frame_id=single_frame_id or run_dir.parent.name,
                branch_id=run_dir.name or f"branch_{line_number:04d}",
                run_dir=run_dir,
                metadata={"runlist": str(path), "runlist_index": str(run_index), "runlist_line": str(line_number)},
            )
        )
    return rows


def discover_branches(root: Path, max_depth: int, single_frame_id: str | None = None) -> list[BranchInput]:
    root = root.resolve()
    branches: list[BranchInput] = []
    candidates = [root]
    candidates.extend(path for path in root.rglob("*") if path.is_dir())
    for path in sorted(candidates):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        depth = 0 if rel == Path(".") else len(rel.parts)
        if depth > max_depth:
            continue
        if not _has_vasp_evidence(path):
            continue
        if path == root:
            frame_id = single_frame_id or root.name
            branch_id = root.name
        elif depth == 1:
            frame_id = single_frame_id or root.name
            branch_id = path.name
        else:
            frame_id = path.parent.name
            branch_id = path.name
        branches.append(BranchInput(frame_id=frame_id, branch_id=branch_id, run_dir=path))
    return branches


def vasp_output_candidates(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    if not run_dir.is_dir():
        return candidates
    for name in ("OUTCAR", "OUTCAR.gz", "OSZICAR", "vasprun.xml", "vasprun.xml.gz"):
        path = run_dir / name
        if path.is_file():
            candidates.append(path)
    for pattern in ("vasp.out*",):
        candidates.extend(path for path in run_dir.glob(pattern) if path.is_file())
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique, key=lambda item: item.stat().st_mtime if item.exists() else 0.0, reverse=True)


def branch_array_index(branch: BranchInput) -> int | None:
    raw = branch.metadata.get("runlist_index") or branch.metadata.get("index") or branch.metadata.get("array_index")
    if raw is None:
        return None
    try:
        return int(str(raw))
    except ValueError:
        return None


def branch_log_dir(branch: BranchInput, args: argparse.Namespace) -> Path:
    if args.log_dir is not None:
        return args.log_dir
    runlist = branch.metadata.get("runlist")
    if runlist:
        return Path(runlist).expanduser().resolve().parent
    return branch.run_dir.parent


def branch_output_candidates(branch: BranchInput, args: argparse.Namespace, deep_artifacts: bool = True) -> list[Path]:
    candidates = list(vasp_output_candidates(branch.run_dir))
    index = branch_array_index(branch)
    if index is not None:
        candidates.extend(array_indexed_output_candidates(index, branch_log_dir(branch, args), deep=deep_artifacts))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique, key=lambda item: item.stat().st_mtime if item.exists() else 0.0, reverse=True)


def _energy_rank(path: Path) -> int:
    name = path.name
    if name.startswith("vasp.out"):
        return 0
    if name.startswith("vasprun.xml"):
        return 1
    if name == "OSZICAR":
        return 2
    if name.startswith("OUTCAR"):
        return 3
    return 4


def latest_branch_energy(
    candidates: list[Path],
    preferred_kind: str = "toten",
    dav_average_window: int = 10,
) -> tuple[float | None, str, Path | None]:
    candidates = sorted(candidates, key=lambda path: (_energy_rank(path), -path.stat().st_mtime))
    for path in candidates:
        if path.name.startswith("vasprun.xml"):
            energy = latest_vasprun_energy(path)
            if energy is not None:
                return energy, "vasprun_e_fr_energy", path
            continue
        energy, kind = _latest_vasp_energy(
            path,
            preferred_kind=preferred_kind,
            dav_average_window=dav_average_window,
        )
        if energy is not None:
            return energy, kind, path
    return None, "", candidates[0] if candidates else None


def latest_vasprun_energy(path: Path) -> float | None:
    try:
        text = _read_text(path, limit=2_000_000)
    except OSError:
        return None
    matches = re.findall(
        r'<i\s+name="(?:e_fr_energy|e_wo_entrp|e_0_energy)">\s*([-+0-9.Ee]+)\s*</i>',
        text,
    )
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def classify_convergence(
    run_dir: Path,
    stopped_after_minutes: float,
    candidates: list[Path],
) -> tuple[str, Path | None, float | None, list[str]]:
    if not candidates:
        return "NO_OUTPUT", None, None, ["missing outputs"]
    newest = candidates[0]
    age_min = (time.time() - newest.stat().st_mtime) / 60.0
    reasons: list[str] = []
    outcar_text = ""
    for path in candidates:
        if path.name not in {"OUTCAR", "OUTCAR.gz"}:
            continue
        try:
            outcar_text = _read_text(path, limit=2_000_000).lower()
        except OSError:
            outcar_text = ""
        break
    if any(pattern in outcar_text for pattern in ERROR_PATTERNS):
        return "ERROR", newest, age_min, ["VASP error marker found"]
    if _run_is_done(run_dir) or DONE_TEXT.lower() in outcar_text:
        if CONVERGED_TEXT in outcar_text or _run_is_done(run_dir):
            return "CONVERGED", newest, age_min, []
        return "DONE", newest, age_min, []
    if CONVERGED_TEXT in outcar_text:
        return "CONVERGED", newest, age_min, []
    if age_min > stopped_after_minutes:
        reasons.append(f"newest output older than {stopped_after_minutes:g} min")
        return "STOPPED", newest, age_min, reasons
    return "RUNNING", newest, age_min, []


def scf_candidate(candidates: list[Path]) -> Path | None:
    ranks = {"OSZICAR": 0, "vasp.out": 1, "OUTCAR": 2, "OUTCAR.gz": 3}
    ranked: list[tuple[int, float, Path]] = []
    for path in candidates:
        if path.name == "OSZICAR":
            rank = ranks["OSZICAR"]
        elif path.name.startswith("vasp.out"):
            rank = ranks["vasp.out"]
        elif path.name == "OUTCAR":
            rank = ranks["OUTCAR"]
        else:
            continue
        ranked.append((rank, -(path.stat().st_mtime if path.exists() else 0.0), path))
    if not ranked:
        return None
    return sorted(ranked)[0][2]


def parse_oszicar_scf(path: Path) -> tuple[list[float], list[float]]:
    dEs: list[float] = []
    energies: list[float] = []
    if not path.is_file():
        return dEs, energies
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if not parts or parts[0] not in {"DAV:", "RMM:"}:
                    continue
                if len(parts) >= 3:
                    try:
                        energies.append(float(parts[2]))
                    except ValueError:
                        pass
                if len(parts) >= 4:
                    try:
                        dEs.append(float(parts[3]))
                    except ValueError:
                        pass
    except OSError:
        return [], []
    return dEs, energies


def parse_oszicar_current_step(path: Path) -> int | None:
    if not path.is_file():
        return None
    current_step = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if not parts:
                    continue
                if not parts[0].isdigit():
                    continue
                if "F=" in parts or "E0=" in parts or "T=" in parts:
                    current_step = int(parts[0])
    except OSError:
        return None
    return current_step


def sign_change_ratio(values: list[float]) -> float | None:
    signed = [1 if value > 0 else -1 if value < 0 else 0 for value in values if value != 0.0]
    if len(signed) < 2:
        return None
    changes = sum(1 for left, right in zip(signed, signed[1:]) if left != right)
    return changes / (len(signed) - 1)


def classify_scf(
    dEs: list[float],
    warning_de: float,
    stop_de: float,
    stall_window: int,
    oscillation_ratio_threshold: float,
) -> tuple[str, float | None, float | None, float | None, list[str]]:
    if not dEs:
        return "UNKNOWN", None, None, None, ["no DAV/RMM dE values"]
    last_de = dEs[-1]
    recent = dEs[-max(1, stall_window) :]
    recent_median = statistics.median(abs(value) for value in recent)
    ratio = sign_change_ratio(recent)
    reasons: list[str] = []
    if abs(last_de) >= stop_de:
        reasons.append(f"last |dE|={abs(last_de):.3g} >= stop threshold {stop_de:g}")
        return "BAD_DE", last_de, recent_median, ratio, reasons
    if ratio is not None and ratio >= oscillation_ratio_threshold and recent_median > warning_de:
        reasons.append(f"SCF dE sign oscillation ratio {ratio:.2f}")
        return "OSCILLATING", last_de, recent_median, ratio, reasons
    if len(dEs) >= 2 * stall_window:
        previous = dEs[-2 * stall_window : -stall_window]
        previous_median = statistics.median(abs(value) for value in previous)
        if recent_median > warning_de and recent_median >= 0.8 * previous_median:
            reasons.append("recent |dE| is not improving")
            return "STALLED", last_de, recent_median, ratio, reasons
    if abs(last_de) > warning_de:
        reasons.append(f"last |dE|={abs(last_de):.3g} > warning threshold {warning_de:g}")
        return "WARNING_DE", last_de, recent_median, ratio, reasons
    return "OK", last_de, recent_median, ratio, []


def default_species_file(run_dir: Path) -> Path | None:
    for name in ("CONTCAR", "POSCAR"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def spin_candidate(run_dir: Path) -> Path | None:
    for name in ("OUTCAR", "OUTCAR.gz"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def parse_track_atoms(values: list[str]) -> list[int]:
    atoms: list[int] = []
    for raw in values:
        for token in re.split(r"[,;\s]+", raw):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                token = token.rsplit(":", 1)[1]
            atoms.append(int(token))
    return sorted(set(atoms))


def tracked_site_status(
    atoms: list[AtomReport],
    track_atoms: list[int],
    localized_min_abs: float,
    change_threshold: float,
) -> tuple[str, list[str]]:
    if not track_atoms:
        return "NOT_APPLIED", []
    notes: list[str] = []
    status = "OK"
    by_index = {atom.index: atom for atom in atoms}
    for index in track_atoms:
        atom = by_index.get(index)
        if atom is None:
            status = "WARN"
            notes.append(f"atom {index} missing")
            continue
        initial = atom.initial
        if initial is None:
            if abs(atom.final) < localized_min_abs:
                status = "WARN"
                notes.append(f"{atom.element}{index} final moment {atom.final:.3f} below localized threshold")
            continue
        sign_flipped = abs(initial) >= change_threshold and abs(atom.final) >= change_threshold and (initial > 0) != (atom.final > 0)
        localized_lost = abs(initial) >= localized_min_abs and abs(atom.final) < localized_min_abs
        if sign_flipped or localized_lost or abs(atom.final - initial) > change_threshold:
            status = "LOST"
            notes.append(f"{atom.element}{index}: initial {initial:.3f} -> final {atom.final:.3f}")
    return status, notes


def analyze_spin(
    run_dir: Path,
    branch_index: int | None,
    log_dir: Path | None,
    moment_guards: dict[str, tuple[list[float], float]],
    track_atoms: list[int],
    change_threshold: float,
    order_threshold: float,
    localized_min_abs: float,
    natoms: int | None,
) -> dict[str, Any]:
    if branch_index is None:
        outcar = spin_candidate(run_dir)
        candidates = [outcar] if outcar is not None else []
    else:
        candidates = magnetization_candidate_files(branch_index, run_dir, log_dir)
    if not candidates:
        return {"mag_status": "NO_OUTCAR"}
    mag_source, block, mag_warning = extract_first_available_magnetization(candidates, expected=natoms)
    if block is None or mag_source is None:
        return {"mag_status": "NO_MAGNETIZATION", "mag_source": candidates[0], "warning": mag_warning}
    species_file = first_species_file(run_dir, mag_source)
    expected = natoms
    labels: list[str] = []
    species = None
    species_warning = ""
    try:
        labels, species, species_warning = read_species_labels(species_file, expected)
        if expected is None and species is not None:
            expected = species.total_atoms
        if expected is not None and len(block.rows) != expected:
            mag_source, block, mag_warning = extract_first_available_magnetization(candidates, expected=expected)
            if block is None or mag_source is None:
                return {"mag_status": "NO_MAGNETIZATION", "mag_source": candidates[0], "warning": mag_warning}
    except Exception as exc:
        return {"mag_status": "NO_MAGNETIZATION", "mag_source": mag_source, "warning": str(exc)}
    if not labels:
        labels = ["X"] * len(block.rows)
    if len(labels) < len(block.rows):
        labels = labels + ["X"] * (len(block.rows) - len(labels))
    initial = None
    incar = first_incar_file(run_dir, mag_source)
    if incar is not None and species is not None:
        initial = existing_magmom_values(incar, species.total_atoms)
    atom_reports = build_atom_reports(block.moments, labels, initial, change_threshold=change_threshold)
    physics_status, physics_bad_count, physics_bad_by_element = apply_moment_guards(atom_reports, moment_guards)
    track_status, track_notes = tracked_site_status(
        atom_reports,
        track_atoms=track_atoms,
        localized_min_abs=localized_min_abs,
        change_threshold=change_threshold,
    )
    element_order, element_sum = element_order_from_atoms(atom_reports, threshold=order_threshold)
    return {
        "mag_source": mag_source,
        "output_run_dir": mag_source.parent,
        "mag_status": "ONSITE_MATRIX" if block.source_kind == "onsite_density_matrix" else ("WARN" if block.warning else "OK"),
        "total_moment": sum(block.moments),
        "max_abs_moment": max((abs(value) for value in block.moments), default=0.0),
        "changed_count": sum(1 for atom in atom_reports if atom.changed),
        "changed_by_element": changed_counts_by_element(atom_reports),
        "initial_element_order": initial_order_from_atoms(atom_reports, threshold=order_threshold),
        "element_order": element_order,
        "element_sum": element_sum,
        "summary_counts": summarize_counts(block.moments),
        "physics_guard_status": physics_status,
        "physics_guard_bad_count": physics_bad_count,
        "physics_guard_bad_by_element": physics_bad_by_element,
        "tracked_site_status": track_status,
        "tracked_site_notes": track_notes,
        "warning": " ".join(item for item in (species_warning, block.warning or "") if item),
    }


def evaluate_branch(branch: BranchInput, args: argparse.Namespace) -> BranchReport:
    report = BranchReport(frame_id=branch.frame_id, branch_id=branch.branch_id, run_dir=branch.run_dir)
    candidates = branch_output_candidates(branch, args, deep_artifacts=True)
    energy, kind, source = latest_branch_energy(
        candidates,
        preferred_kind=args.energy,
        dav_average_window=args.dav_average_window,
    )
    report.energy_eV = energy
    report.energy_kind = kind
    report.energy_source = source
    if source is not None:
        report.output_run_dir = source.parent
    convergence, newest, age_min, convergence_reasons = classify_convergence(branch.run_dir, args.stopped_after_min, candidates)
    report.convergence_status = convergence
    report.newest_output = newest
    report.newest_output_age_min = age_min
    report.reasons.extend(convergence_reasons)

    scf_path = scf_candidate(candidates)
    dEs, _energies = parse_oszicar_scf(scf_path) if scf_path is not None else ([], [])
    report.current_step = parse_oszicar_current_step(scf_path) if scf_path is not None else None
    scf_status, last_de, recent_median, ratio, scf_reasons = classify_scf(
        dEs,
        warning_de=args.scf_warning_de,
        stop_de=args.scf_stop_de,
        stall_window=args.scf_window,
        oscillation_ratio_threshold=args.scf_oscillation_ratio,
    )
    report.scf_status = scf_status
    report.scf_steps = len(dEs)
    report.last_de = last_de
    report.scf_recent_median_abs_de = recent_median
    report.scf_oscillation_ratio = ratio
    report.reasons.extend(scf_reasons)

    spin = analyze_spin(
        branch.run_dir,
        branch_index=branch_array_index(branch),
        log_dir=branch_log_dir(branch, args),
        moment_guards=args._moment_guards,
        track_atoms=args._track_atoms,
        change_threshold=args.spin_change_threshold,
        order_threshold=args.order_threshold,
        localized_min_abs=args.localized_min_abs,
        natoms=args.natoms,
    )
    for key, value in spin.items():
        if hasattr(report, key):
            setattr(report, key, value)
    if spin.get("warning"):
        report.warnings.append(str(spin["warning"]))

    if report.energy_eV is None:
        report.reasons.append("no parseable energy")
    if report.convergence_status in {"NO_OUTPUT", "ERROR"}:
        report.action = "stop" if report.convergence_status == "ERROR" else "warning"
    elif report.convergence_status == "STOPPED":
        report.action = _max_action(report.action, "warning")
    elif report.energy_eV is not None:
        report.action = "continue"

    if report.scf_status in {"BAD_DE", "OSCILLATING"}:
        report.action = _max_action(report.action, args.scf_bad_action)
    elif report.scf_status in {"STALLED", "WARNING_DE"}:
        report.action = _max_action(report.action, "warning")

    if report.physics_guard_status == "FAIL":
        report.reasons.append("moment guard failed")
        report.action = _max_action(report.action, args.spin_fail_action)
    if report.tracked_site_status == "LOST":
        report.reasons.append("tracked localized/spin site changed")
        report.action = _max_action(report.action, args.track_fail_action)
    elif report.tracked_site_status == "WARN":
        report.action = _max_action(report.action, "warning")
    if report.mag_status in {"NO_OUTCAR", "NO_MAGNETIZATION"}:
        report.action = _max_action(report.action, "warning")
    return report


def _max_action(current: str, proposed: str) -> str:
    order = {"continue": 0, "warning": 1, "stop": 2}
    return proposed if order.get(proposed, 0) > order.get(current, 0) else current


def _append_unique(items: list[str], text: str) -> None:
    if text not in items:
        items.append(text)


def rank_and_classify(reports: list[BranchReport], keep_per_frame: int, warn_window: float, stop_window: float) -> None:
    ranking_prefixes = (
        "higher than frame best by ",
        "above warning window by ",
        "rank ",
    )
    for report in reports:
        report.reasons = [
            reason
            for reason in report.reasons
            if not (
                reason.startswith(ranking_prefixes[0])
                or reason.startswith(ranking_prefixes[1])
                or (reason.startswith(ranking_prefixes[2]) and "exceeds keep-per-frame=" in reason)
            )
        ]
    by_frame: dict[str, list[BranchReport]] = {}
    for report in reports:
        by_frame.setdefault(report.frame_id, []).append(report)
    for frame_reports in by_frame.values():
        with_energy = [report for report in frame_reports if report.energy_eV is not None]
        if not with_energy:
            for report in frame_reports:
                report.action = _max_action(report.action, "warning")
            continue
        ranked = sorted(with_energy, key=lambda report: report.energy_eV if report.energy_eV is not None else float("inf"))
        best_energy = ranked[0].energy_eV
        assert best_energy is not None
        for rank, report in enumerate(ranked, start=1):
            report.rank_in_frame = rank
            report.relative_energy_eV = (report.energy_eV or best_energy) - best_energy
            if report.relative_energy_eV > stop_window:
                report.action = "stop"
                _append_unique(report.reasons, f"higher than frame best by {report.relative_energy_eV:.3g} eV")
            elif report.relative_energy_eV > warn_window:
                report.action = _max_action(report.action, "warning")
                _append_unique(report.reasons, f"above warning window by {report.relative_energy_eV:.3g} eV")
            if rank > keep_per_frame:
                report.action = "stop"
                _append_unique(report.reasons, f"rank {rank} exceeds keep-per-frame={keep_per_frame}")
        for report in frame_reports:
            report.survivor = (
                report.energy_eV is not None
                and report.rank_in_frame is not None
                and report.rank_in_frame <= keep_per_frame
                and report.action in {"continue", "warning"}
            )


def report_to_dict(report: BranchReport) -> dict[str, Any]:
    return {
        "frame_id": report.frame_id,
        "branch_id": report.branch_id,
        "run_dir": str(report.run_dir),
        "action": report.action,
        "survivor": report.survivor,
        "rank_in_frame": report.rank_in_frame,
        "energy_eV": report.energy_eV,
        "relative_energy_eV": report.relative_energy_eV,
        "energy_kind": report.energy_kind,
        "energy_source": "" if report.energy_source is None else str(report.energy_source),
        "output_run_dir": "" if report.output_run_dir is None else str(report.output_run_dir),
        "convergence_status": report.convergence_status,
        "newest_output": "" if report.newest_output is None else str(report.newest_output),
        "newest_output_age_min": report.newest_output_age_min,
        "current_step": report.current_step,
        "last_de": report.last_de,
        "scf_status": report.scf_status,
        "scf_steps": report.scf_steps,
        "scf_oscillation_ratio": report.scf_oscillation_ratio,
        "scf_recent_median_abs_de": report.scf_recent_median_abs_de,
        "mag_source": "" if report.mag_source is None else str(report.mag_source),
        "mag_status": report.mag_status,
        "total_moment": report.total_moment,
        "max_abs_moment": report.max_abs_moment,
        "changed_count": report.changed_count,
        "changed_by_element": report.changed_by_element,
        "initial_element_order": report.initial_element_order,
        "element_order": report.element_order,
        "element_sum": report.element_sum,
        "physics_guard_status": report.physics_guard_status,
        "physics_guard_bad_count": report.physics_guard_bad_count,
        "physics_guard_bad_by_element": report.physics_guard_bad_by_element,
        "tracked_site_status": report.tracked_site_status,
        "tracked_site_notes": report.tracked_site_notes,
        "reasons": report.reasons,
        "warnings": report.warnings,
    }


def write_csv(reports: list[BranchReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_id",
        "branch_id",
        "run_dir",
        "action",
        "survivor",
        "rank_in_frame",
        "energy_eV",
        "relative_energy_eV",
        "energy_kind",
        "energy_source",
        "output_run_dir",
        "convergence_status",
        "current_step",
        "last_de",
        "scf_status",
        "mag_source",
        "mag_status",
        "total_moment",
        "max_abs_moment",
        "changed_count",
        "changed_by_element",
        "initial_element_order",
        "element_order",
        "physics_guard_status",
        "physics_guard_bad_count",
        "physics_guard_bad_by_element",
        "tracked_site_status",
        "tracked_site_notes",
        "reasons",
        "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            row = report_to_dict(report)
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True) if isinstance(row[key], (dict, list)) else row[key]
                    for key in fields
                }
            )


def write_json(reports: list[BranchReport], path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "atomi.vasp.stage1_branch_screen.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "settings": {
            "keep_per_frame": args.keep_per_frame,
            "energy_window_warning_eV": args.energy_window_warning,
            "energy_window_stop_eV": args.energy_window_stop,
            "scf_warning_de": args.scf_warning_de,
            "scf_stop_de": args.scf_stop_de,
            "spin_fail_action": args.spin_fail_action,
            "track_fail_action": args.track_fail_action,
        },
        "reports": [report_to_dict(report) for report in reports],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_survivors(reports: list[BranchReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(report.run_dir) for report in reports if report.survivor]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def print_summary(reports: list[BranchReport], outdir: Path) -> None:
    counts: dict[str, int] = {}
    survivor_count = 0
    frames = {report.frame_id for report in reports}
    for report in reports:
        counts[report.action] = counts.get(report.action, 0) + 1
        survivor_count += 1 if report.survivor else 0
    print(f"Frames scanned      : {len(frames)}")
    print(f"Branches scanned    : {len(reports)}")
    print(f"Actions             : {counts}")
    print(f"Stage-2 survivors   : {survivor_count}")
    print(f"Summary CSV         : {outdir / 'stage1_branch_summary.csv'}")
    print(f"Summary JSON        : {outdir / 'stage1_branch_summary.json'}")
    print(f"Stage-2 runlist     : {outdir / 'stage2_survivors_runlist.txt'}")


def _short(text: str, width: int) -> str:
    if len(text) <= width:
        return text.ljust(width)
    if width <= 1:
        return text[:width]
    return (text[: width - 1] + "~").ljust(width)


def _fmt_float(value: float | None, width: int, precision: int = 3) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:.{precision}f}".rjust(width)


def _fmt_scientific(value: float | None, width: int) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:.1e}".rjust(width)


def _action_label(action: str) -> str:
    return {"continue": "GOOD", "warning": "WARN", "stop": "BAD"}.get(action, action.upper())


def compact_reason(report: BranchReport) -> str:
    parts: list[str] = []
    if report.relative_energy_eV is not None and report.relative_energy_eV > 0:
        parts.append(f"dE={report.relative_energy_eV:.2f}")
    if report.scf_status not in {"OK", "UNKNOWN"}:
        parts.append(report.scf_status)
    if report.physics_guard_status == "FAIL":
        parts.append("spin-guard")
    if report.tracked_site_status == "LOST":
        parts.append("track-lost")
    if report.mag_status in {"NO_OUTCAR", "NO_MAGNETIZATION"}:
        parts.append(report.mag_status)
    if report.convergence_status in {"STOPPED", "ERROR", "NO_OUTPUT"}:
        parts.append(report.convergence_status)
    if report.reasons and not parts:
        parts.append(report.reasons[0])
    return ";".join(parts) if parts else "ok"


def _spin_status(report: BranchReport) -> str:
    spin_status = report.physics_guard_status
    if spin_status == "NOT_APPLIED":
        spin_status = report.mag_status
    return spin_status


def _compact_count_map(values: dict[str, int], empty: str = "-") -> str:
    parts = [f"{key}:{values[key]}" for key in sorted(values) if values[key]]
    return ",".join(parts) if parts else empty


def _compact_order_shift(report: BranchReport) -> str:
    elements = sorted(set(report.initial_element_order) | set(report.element_order))
    parts: list[str] = []
    for element in elements:
        initial = report.initial_element_order.get(element)
        final = report.element_order.get(element)
        if final is None:
            continue
        if initial is not None and initial != final:
            parts.append(f"{element}:{initial}>{final}")
        else:
            parts.append(f"{element}:{final}")
    return ",".join(parts) if parts else "-"


def _run_pointer(path: Path, parent_is_shared: bool) -> str:
    name = path.name or str(path)
    parent = path.parent.name
    if parent_is_shared or not parent:
        return name
    return f"{parent}/{name}"


def _shared_parent(paths: list[Path]) -> bool:
    parents = {path.parent.name for path in paths if path.parent.name}
    return len(parents) <= 1


def run_pointer_labels(reports: list[BranchReport]) -> dict[Path, str]:
    parent_is_shared = _shared_parent([report.run_dir for report in reports])
    return {report.run_dir: _run_pointer(report.run_dir, parent_is_shared) for report in reports}


def branch_pointer_labels(branches: list[BranchInput]) -> dict[Path, str]:
    parent_is_shared = _shared_parent([branch.run_dir for branch in branches])
    return {branch.run_dir: _run_pointer(branch.run_dir, parent_is_shared) for branch in branches}


def format_live_table(reports: list[BranchReport], args: argparse.Namespace, iteration: int) -> str:
    counts: dict[str, int] = {}
    for report in reports:
        counts[report.action] = counts.get(report.action, 0) + 1
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "Atomi VASP Branch Live Monitor",
        f"Updated: {now}   refresh={args.refresh:g}s   iteration={iteration}",
        f"Branches={len(reports)}   GOOD={counts.get('continue', 0)}   WARN={counts.get('warning', 0)}   BAD={counts.get('stop', 0)}",
        f"Outputs: {args.outdir / 'stage1_branch_summary.csv'} ; {args.outdir / 'stage2_survivors_runlist.txt'}",
        "",
        "run  path                    frame        branch        step      energy     relE      dE      conv      scf       guard     chg       order             trk   rec   note",
        "---  ----------------------  -----------  ------------  -----  ----------  -------  --------  --------  --------  --------  --------  ----------------  ----  ----  ----------------",
    ]
    sorted_reports = sorted(
        reports,
        key=lambda item: (item.frame_id, item.rank_in_frame or 999999, item.branch_id),
    )
    path_labels = run_pointer_labels(sorted_reports)
    for index, report in enumerate(sorted_reports, start=1):
        step = "-" if report.current_step is None else str(report.current_step)
        spin_status = _spin_status(report)
        lines.append(
            f"{str(index).rjust(3)}  "
            f"{_short(path_labels.get(report.run_dir, report.run_dir.name), 22)}  "
            f"{_short(report.frame_id, 11)}  "
            f"{_short(report.branch_id, 12)}  "
            f"{step.rjust(5)}  "
            f"{_fmt_float(report.energy_eV, 10, 4)}  "
            f"{_fmt_float(report.relative_energy_eV, 7, 3)}  "
            f"{_fmt_scientific(report.last_de, 8)}  "
            f"{_short(report.convergence_status, 8)}  "
            f"{_short(report.scf_status, 8)}  "
            f"{_short(spin_status, 8)}  "
            f"{_short(_compact_count_map(report.changed_by_element), 8)}  "
            f"{_short(_compact_order_shift(report), 16)}  "
            f"{_short(report.tracked_site_status, 4)}  "
            f"{_short(_action_label(report.action), 4)}  "
            f"{_short(compact_reason(report), 16)}"
        )
    lines.extend(
        [
            "",
            "Legend: rec GOOD/WARN/BAD is the same recommendation used by the CSV/JSON branch screen.",
            "guard shows physics moment-guard status when guards are set; otherwise it shows magnetization extraction status.",
            "chg counts atoms whose final local moment changed from initial MAGMOM by element; order compares initial and final element-level FM/AFM labels.",
            "Ctrl-C exits cleanly.",
        ]
    )
    return "\n".join(lines)


def format_scan_header(args: argparse.Namespace, iteration: int, total: int) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 118,
        f"Atomi VASP Branch Scan Monitor | pass {iteration} | {now} | branches={total}",
        f"Outputs after pass: {args.outdir / 'stage1_branch_summary.csv'} ; {args.outdir / 'stage2_survivors_runlist.txt'}",
        "Rows are streamed as soon as each branch is parsed; per-frame ranks are final after the pass completes.",
        "run        path                    frame        branch        step      energy     relE      dE      conv      scf       guard     chg       order             trk   rec   note",
        "---------  ----------------------  -----------  ------------  -----  ----------  -------  --------  --------  --------  --------  --------  ----------------  ----  ----  ----------------",
    ]
    return "\n".join(lines)


def format_scan_row(report: BranchReport, index: int, total: int, run_label: str | None = None) -> str:
    step = "-" if report.current_step is None else str(report.current_step)
    return (
        f"{str(index).rjust(3)}/{str(total).ljust(3)}  "
        f"{_short(run_label or report.run_dir.name, 22)}  "
        f"{_short(report.frame_id, 11)}  "
        f"{_short(report.branch_id, 12)}  "
        f"{step.rjust(5)}  "
        f"{_fmt_float(report.energy_eV, 10, 4)}  "
        f"{_fmt_float(report.relative_energy_eV, 7, 3)}  "
        f"{_fmt_scientific(report.last_de, 8)}  "
        f"{_short(report.convergence_status, 8)}  "
        f"{_short(report.scf_status, 8)}  "
        f"{_short(_spin_status(report), 8)}  "
        f"{_short(_compact_count_map(report.changed_by_element), 8)}  "
        f"{_short(_compact_order_shift(report), 16)}  "
        f"{_short(report.tracked_site_status, 4)}  "
        f"{_short(_action_label(report.action), 4)}  "
        f"{_short(compact_reason(report), 16)}"
    )


def format_scan_footer(reports: list[BranchReport], args: argparse.Namespace, iteration: int) -> str:
    counts: dict[str, int] = {}
    for report in reports:
        counts[report.action] = counts.get(report.action, 0) + 1
    lines = [
        "-" * 118,
        (
            f"Pass {iteration} complete: branches={len(reports)} "
            f"GOOD={counts.get('continue', 0)} WARN={counts.get('warning', 0)} BAD={counts.get('stop', 0)}"
        ),
        f"Wrote: {args.outdir / 'stage1_branch_summary.csv'}",
        f"Wrote: {args.outdir / 'stage2_survivors_runlist.txt'}",
    ]
    if args.live_count == 0 or iteration < args.live_count:
        lines.append(f"Next scan starts in {args.refresh:g} s. Ctrl-C exits cleanly.")
    return "\n".join(lines)


class ScanSpinner:
    def __init__(self, label: str, enabled: bool | None = None) -> None:
        self.label = label
        self.enabled = sys.stdout.isatty() if enabled is None else enabled
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ScanSpinner":
        if not self.enabled:
            return self

        def run() -> None:
            for mark in itertools.cycle("|/-\\"):
                if self._done.is_set():
                    break
                sys.stdout.write(f"\r{mark} {self.label}")
                sys.stdout.flush()
                self._done.wait(0.15)
            sys.stdout.write("\r" + " " * (len(self.label) + 4) + "\r")
            sys.stdout.flush()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self.enabled:
            return
        self._done.set()
        if self._thread is not None:
            self._thread.join()


def write_outputs(reports: list[BranchReport], args: argparse.Namespace) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(reports, args.outdir / "stage1_branch_summary.csv")
    write_json(reports, args.outdir / "stage1_branch_summary.json", args)
    write_survivors(reports, args.outdir / "stage2_survivors_runlist.txt")


def load_branches_from_args(args: argparse.Namespace) -> list[BranchInput]:
    if args.index:
        branches = load_branch_index(args.index)
    elif args.runlist:
        branches = load_branch_runlist(args.runlist, single_frame_id=args.single_frame_id)
    else:
        branches = discover_branches(args.root, args.max_depth, args.single_frame_id)
    if not branches:
        raise FileNotFoundError("No VASP branch directories found. Use --index, --runlist, or adjust --max-depth.")
    return branches


def screen_once(args: argparse.Namespace) -> list[BranchReport]:
    branches = load_branches_from_args(args)
    reports = [evaluate_branch(branch, args) for branch in branches]
    rank_and_classify(
        reports,
        keep_per_frame=args.keep_per_frame,
        warn_window=args.energy_window_warning,
        stop_window=args.energy_window_stop,
    )
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-branch-screen",
        description="Stage-1 fail-fast screening/ranking of VASP spin/localization branches per MD frame.",
    )
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="Root containing frame/branch VASP folders.")
    parser.add_argument("--index", type=Path, help="CSV with run_dir/path plus optional frame_id and branch_id columns.")
    parser.add_argument("--runlist", type=Path, help="Plain runlist.txt containing one branch directory per line.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory containing array logs/artifacts such as vasp.out*.<index> and bwforcluster*.<index>.*. Default: runlist parent.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("."), help="Output directory for stage-1 summaries.")
    parser.add_argument("--max-depth", type=int, default=2, help="Discovery depth below root when --index is not given.")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Allow live mode to discover VASP branches under root when no --runlist/--index is supplied.",
    )
    parser.add_argument("--single-frame-id", help="Frame id for one-level branch folders under root.")
    parser.add_argument("--keep-per-frame", type=int, default=1, help="Number of best branches to pass to stage 2 per frame.")
    parser.add_argument("--energy-window-warning", type=float, default=0.5, help="Warn when branch energy is this many eV above frame best.")
    parser.add_argument("--energy-window-stop", type=float, default=2.0, help="Stop when branch energy is this many eV above frame best.")
    parser.add_argument("--energy", default="toten", choices=("toten", "without_entropy", "e0", "f", "dav"), help="Preferred energy kind.")
    parser.add_argument("--dav-average-window", type=int, default=10, help="Average latest DAV energies when only DAV fallback exists.")
    parser.add_argument("--stopped-after-min", type=float, default=15.0, help="Mark active-looking branches as stopped after this idle time.")
    parser.add_argument("--scf-warning-de", type=float, default=1e-3, help="Warn when latest electronic |dE| remains above this.")
    parser.add_argument("--scf-stop-de", type=float, default=1e-1, help="Stop when latest electronic |dE| is above this.")
    parser.add_argument("--scf-window", type=int, default=8, help="Recent DAV/RMM window for stall/oscillation checks.")
    parser.add_argument("--scf-oscillation-ratio", type=float, default=0.55, help="Sign-change ratio treated as SCF oscillation.")
    parser.add_argument("--scf-bad-action", choices=("warning", "stop"), default="stop", help="Action for BAD_DE/OSCILLATING SCF.")
    parser.add_argument("--moment-guard", action="append", default=[], help="Element moment targets, e.g. U=-2,-1,1,2@0.8. Repeatable.")
    parser.add_argument("--moment-guard-tol", type=float, default=0.8, help="Default tolerance for --moment-guard.")
    parser.add_argument("--spin-fail-action", choices=("warning", "stop"), default="warning", help="Action when element moment guard fails.")
    parser.add_argument("--track-atom", action="append", default=[], help="Track intended localized atom indices, e.g. 35 or U:35. Repeatable or comma-separated.")
    parser.add_argument("--track-fail-action", choices=("warning", "stop"), default="warning", help="Action when tracked atom loses/flips moment.")
    parser.add_argument("--spin-change-threshold", type=float, default=1.0, help="Moment delta/sign threshold for changed-site flags.")
    parser.add_argument("--localized-min-abs", type=float, default=0.5, help="Tracked site is lost below this |moment|.")
    parser.add_argument("--order-threshold", type=float, default=0.2, help="Moment threshold for FM/AFM/AFM-like labels.")
    parser.add_argument("--natoms", type=int, help="Expected atom count when POSCAR/CONTCAR is missing.")
    parser.add_argument("--live", action="store_true", help="Run a repeating terminal monitor instead of printing one summary.")
    parser.add_argument(
        "--live-mode",
        choices=("scan", "dashboard"),
        default="scan",
        help="Live display style. scan streams rows as they are evaluated; dashboard repaints one full table. Default: scan.",
    )
    parser.add_argument("--refresh", type=float, default=10.0, help="Seconds between live scan passes. Default: 10.")
    parser.add_argument("--live-count", type=int, default=0, help="Number of live scan passes; 0 means until Ctrl-C.")
    parser.add_argument("--watch-interval", type=float, default=0.0, help="Repeat screening every N seconds. Default one-shot.")
    parser.add_argument("--watch-count", type=int, default=1, help="Number of watch iterations; use 0 for until Ctrl-C.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args._moment_guards = parse_moment_guards(args.moment_guard, args.moment_guard_tol)
    args._track_atoms = parse_track_atoms(args.track_atom)
    if args.live:
        if args.index is None and args.runlist is None and not args.discover:
            default_runlist = args.root / "runlist.txt"
            if not default_runlist.is_file():
                raise FileNotFoundError(
                    "vasp-branch-live reads runlist.txt by default and did not find one. "
                    "Pass --runlist PATH, run from the runlist directory, or pass --discover "
                    "to scan nearby VASP-like folders."
                )
            args.runlist = default_runlist
        run_live(args)
        return
    iteration = 0
    while True:
        iteration += 1
        reports = screen_once(args)
        write_outputs(reports, args)
        print_summary(reports, args.outdir)
        if args.watch_interval <= 0:
            break
        if args.watch_count > 0 and iteration >= args.watch_count:
            break
        time.sleep(args.watch_interval)


def run_live(args: argparse.Namespace) -> None:
    if args.live_mode == "dashboard":
        run_live_dashboard(args)
    else:
        run_live_scan(args)


def run_live_dashboard(args: argparse.Namespace) -> None:
    iteration = 0
    try:
        while True:
            iteration += 1
            reports = screen_once(args)
            write_outputs(reports, args)
            print("\033[2J\033[H", end="")
            print(format_live_table(reports, args, iteration))
            if args.live_count > 0 and iteration >= args.live_count:
                break
            time.sleep(max(args.refresh, 0.1))
    except KeyboardInterrupt:
        print("\nStopped live VASP branch monitor.")


def run_live_scan(args: argparse.Namespace) -> None:
    iteration = 0
    try:
        while True:
            iteration += 1
            branches = load_branches_from_args(args)
            reports: list[BranchReport] = []
            total = len(branches)
            path_labels = branch_pointer_labels(branches)
            print(format_scan_header(args, iteration, total), flush=True)
            for index, branch in enumerate(branches, start=1):
                run_label = path_labels.get(branch.run_dir, branch.run_dir.name)
                label = f"scanning run {index}/{total} {run_label}"
                with ScanSpinner(label):
                    report = evaluate_branch(branch, args)
                reports.append(report)
                rank_and_classify(
                    reports,
                    keep_per_frame=args.keep_per_frame,
                    warn_window=args.energy_window_warning,
                    stop_window=args.energy_window_stop,
                )
                print(format_scan_row(report, index, total, run_label=run_label), flush=True)
            rank_and_classify(
                reports,
                keep_per_frame=args.keep_per_frame,
                warn_window=args.energy_window_warning,
                stop_window=args.energy_window_stop,
            )
            write_outputs(reports, args)
            print(format_scan_footer(reports, args, iteration), flush=True)
            if args.live_count > 0 and iteration >= args.live_count:
                break
            time.sleep(max(args.refresh, 0.1))
    except KeyboardInterrupt:
        print("\nStopped live VASP branch scan monitor.")


def monitor_main(argv: list[str] | None = None) -> None:
    args = list(argv or [])
    if "--live" not in args:
        args.insert(0, "--live")
    main(args)


if __name__ == "__main__":  # pragma: no cover
    main()
