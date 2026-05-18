from __future__ import annotations

import argparse
import fnmatch
import gzip
import re
import time
from dataclasses import dataclass
from pathlib import Path


DONE_MARKER = "General timing and accounting informations for this job"
CLEAN_PATTERNS = ("OUTCAR*", "CONTCAR", "vasprun.xml", "OSZICAR")
OUTPUT_PATTERNS = ("vasprun.xml", "OUTCAR*", "OSZICAR", "CONTCAR")
ARRAY_ARTIFACT_PATTERNS = (
    "vasp.out*",
    "OUTCAR",
    "OUTCAR.gz",
    "OSZICAR",
    "vasprun.xml",
    "vasprun.xml.gz",
)
DEFAULT_CHECKENG_CLEAN_PATTERNS = (
    "vasp.out*",
    "OUTCAR*",
    "OSZICAR",
    "vasprun.xml*",
    "WAVECAR",
    "CHG",
    "CHGCAR",
    "EIGENVAL",
    "PROCAR",
    "DOSCAR",
    "XDATCAR",
    "REPORT",
    "PCDAT",
)
DEFAULT_CHECKVASP_STOPPED_AFTER_MINUTES = 5.0
DEFAULT_CHECKENG_STOPPED_AFTER_MINUTES = 15.0
DEFAULT_CHECKENG_DAV_AVERAGE_WINDOW = 10
ENERGY_PATTERNS = {
    "toten": re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)"),
    "without_entropy": re.compile(r"energy\s+without\s+entropy\s*=\s*([-+0-9.Ee]+)"),
    "e0": re.compile(r"\bE0=\s*([-+0-9.Ee]+)"),
    "f": re.compile(r"\bF=\s*([-+0-9.Ee]+)"),
    "dav": re.compile(r"^\s*DAV:\s+\d+\s+([-+0-9.Ee]+)"),
}


@dataclass
class RunStatusCounts:
    total: int = 0
    done: int = 0
    running: int = 0
    stopped: int = 0
    not_started: int = 0
    missing: int = 0


@dataclass
class ScfCounts:
    total: int = 0
    checked: int = 0
    above: int = 0
    missing_log: int = 0
    no_dav: int = 0
    missing_dir: int = 0
    cleaned_dirs: int = 0
    stale_nolog: int = 0


@dataclass
class EnergyRecord:
    index: int
    run: Path
    energy_eV: float | None
    energy_kind: str = ""
    source: Path | None = None
    status: str = "NOLOG"


@dataclass
class StoppedCleanCounts:
    stopped_runs: int = 0
    matched_files: int = 0
    removed_files: int = 0
    dry_run: bool = True


def iter_runlist(runlist: Path) -> list[Path]:
    runs: list[Path] = []
    for raw_line in runlist.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        runs.append(Path(line))
    return runs


def check_runs(
    runlist: Path,
    stopped_after_minutes: float = DEFAULT_CHECKVASP_STOPPED_AFTER_MINUTES,
    log_dir: Path | None = None,
) -> RunStatusCounts:
    if not runlist.is_file():
        raise FileNotFoundError(f"cannot find {runlist}")

    counts = RunStatusCounts()
    now = time.time()
    stale_seconds = max(0.0, stopped_after_minutes) * 60.0
    search_log_dir = runlist.parent if log_dir is None else log_dir
    for index, runpath in enumerate(iter_runlist(runlist), start=1):
        counts.total += 1
        resolved_runpath = runpath if runpath.is_absolute() else (runlist.parent / runpath)
        if not resolved_runpath.is_dir():
            print(f"MISSING   {runpath}")
            counts.missing += 1
            continue

        if _run_is_done(resolved_runpath):
            print(f"DONE      {runpath}")
            counts.done += 1
            continue

        latest = _latest_vasp_output(resolved_runpath, index=index, log_dir=search_log_dir)
        if latest is None:
            print(f"NOTSTART  {runpath}")
            counts.not_started += 1
            continue

        if now - latest.stat().st_mtime > stale_seconds:
            age_min = (now - latest.stat().st_mtime) / 60.0
            print(f"STOPPED   {runpath}   last_write={age_min:.1f} min   file={latest.name}")
            counts.stopped += 1
            continue

        print(f"RUNNING   {runpath}   last_write=active   file={latest.name}")
        counts.running += 1

    return counts


def print_run_status_summary(counts: RunStatusCounts) -> None:
    print()
    print("Summary:")
    print(f"DONE      : {counts.done} / {counts.total} ({_pct(counts.done, counts.total)}%)")
    print(f"RUNNING   : {counts.running} / {counts.total} ({_pct(counts.running, counts.total)}%)")
    print(f"STOPPED   : {counts.stopped} / {counts.total} ({_pct(counts.stopped, counts.total)}%)")
    print(
        f"NOTSTART  : {counts.not_started} / {counts.total} "
        f"({_pct(counts.not_started, counts.total)}%)"
    )
    print(f"MISSING   : {counts.missing} / {counts.total} ({_pct(counts.missing, counts.total)}%)")


def check_scf(
    runlist: Path,
    threshold: float,
    outfile: Path | None = None,
    clean: bool = False,
    dry_run: bool = False,
    log_dir: Path = Path("."),
) -> ScfCounts:
    if not runlist.is_file():
        raise FileNotFoundError(f"cannot find {runlist}")

    counts = ScfCounts()
    if outfile is not None:
        outfile.write_text("", encoding="utf-8")

    for index, runpath in enumerate(iter_runlist(runlist), start=1):
        counts.total += 1
        vfile = _find_vasp_log_for_index(index, log_dir=log_dir)
        if vfile is None:
            print(f"NOLOG        run {index}   {runpath}")
            counts.missing_log += 1

            if runpath.is_dir() and _has_vasp_outputs(runpath):
                print(f"STALEFILES   run {index}   found VASP outputs in NOLOG directory")
                counts.stale_nolog += 1
                _append_bad_run(outfile, runpath)

                if clean:
                    print(f"CLEANING     run {index}   {runpath}")
                    _clean_one_dir(runpath, dry_run=dry_run)
                    counts.cleaned_dirs += 1

            continue

        if not runpath.is_dir():
            print(f"NODIR        run {index}   {runpath}")
            counts.missing_dir += 1

        last_de = _last_dav_energy_change(vfile)
        if last_de is None:
            print(f"NODAV        run {index}   {runpath}   file={vfile}")
            counts.no_dav += 1
            continue

        counts.checked += 1
        if abs(last_de) > threshold:
            print(f"ABOVE        run {index}   dE={last_de:g}   file={vfile}   path={runpath}")
            counts.above += 1
            _append_bad_run(outfile, runpath)

            if clean:
                print(f"CLEANING     run {index}   {runpath}")
                _clean_one_dir(runpath, dry_run=dry_run)
                counts.cleaned_dirs += 1

    return counts


def collect_run_energies(
    runlist: Path,
    log_dir: Path | None = None,
    energy_kind: str = "toten",
    stopped_after_minutes: float = DEFAULT_CHECKENG_STOPPED_AFTER_MINUTES,
    dav_average_window: int = DEFAULT_CHECKENG_DAV_AVERAGE_WINDOW,
) -> list[EnergyRecord]:
    if not runlist.is_file():
        raise FileNotFoundError(f"cannot find {runlist}")
    if energy_kind not in ENERGY_PATTERNS:
        raise ValueError(f"unknown energy kind: {energy_kind}")

    records: list[EnergyRecord] = []
    now = time.time()
    stale_seconds = max(0.0, stopped_after_minutes) * 60.0
    search_log_dir = runlist.parent if log_dir is None else log_dir
    for index, runpath in enumerate(iter_runlist(runlist), start=1):
        resolved_runpath = runpath if runpath.is_absolute() else (runlist.parent / runpath)
        candidates = _energy_candidate_files(index, resolved_runpath, search_log_dir)
        record = EnergyRecord(index=index, run=runpath, energy_eV=None)
        if not resolved_runpath.is_dir():
            record.status = "NODIR"
        if not candidates:
            records.append(record)
            continue

        record.source = candidates[0]
        energy, found_kind = _latest_vasp_energy(
            candidates[0],
            preferred_kind=energy_kind,
            dav_average_window=dav_average_window,
        )
        if energy is None:
            record.status = "NOENERGY"
        else:
            record.status = "OK"
            record.energy_eV = energy
            record.energy_kind = found_kind
        if not _run_is_done(resolved_runpath) and now - record.source.stat().st_mtime > stale_seconds:
            record.status = "STOPPED"
        records.append(record)
    return records


def print_energy_table(records: list[EnergyRecord], delimiter: str = "  ") -> None:
    if delimiter == "tab":
        delimiter = "\t"
    header = ["index", "run", "energy_eV", "kind", "status", "source"]
    rows = [
        [
            str(record.index),
            str(record.run),
            "" if record.energy_eV is None else f"{record.energy_eV:.10f}",
            record.energy_kind,
            record.status,
            "" if record.source is None else str(record.source),
        ]
        for record in records
    ]
    if delimiter == "\t":
        print(delimiter.join(header))
        for row in rows:
            print(delimiter.join(row))
        return

    widths = [
        max(len(header[col]), *(len(row[col]) for row in rows)) if rows else len(header[col])
        for col in range(len(header))
    ]
    print(delimiter.join(header[col].ljust(widths[col]) for col in range(len(header))))
    for row in rows:
        print(delimiter.join(row[col].ljust(widths[col]) for col in range(len(row))))


def clean_stopped_energy_outputs(
    runlist: Path,
    records: list[EnergyRecord],
    log_dir: Path | None = None,
    patterns: list[str] | None = None,
    scope: str = "artifacts",
    execute: bool = False,
) -> StoppedCleanCounts:
    search_log_dir = runlist.parent if log_dir is None else log_dir
    clean_patterns = patterns or list(DEFAULT_CHECKENG_CLEAN_PATTERNS)
    counts = StoppedCleanCounts(dry_run=not execute)
    for record in records:
        if record.status != "STOPPED":
            continue
        counts.stopped_runs += 1
        runpath = record.run if record.run.is_absolute() else runlist.parent / record.run
        roots: list[Path] = []
        if scope in {"artifacts", "both"}:
            roots.extend(array_indexed_artifact_roots(record.index, search_log_dir, exclude=runpath))
        if scope in {"run", "both"}:
            roots.append(runpath)
        files = _clean_candidate_files(roots, clean_patterns)
        counts.matched_files += len(files)
        print(f"STOPPED-CLEAN run {record.index}   {record.run}   files={len(files)}")
        for path in files:
            action = "remove" if execute else "would remove"
            print(f"  {action}: {path}")
            if execute:
                try:
                    path.unlink(missing_ok=True)
                    counts.removed_files += 1
                except OSError as exc:
                    print(f"  failed: {path} ({exc})")
    return counts


def print_stopped_clean_summary(counts: StoppedCleanCounts) -> None:
    print()
    print("Stopped cleanup summary")
    print(f"mode        : {'dry-run' if counts.dry_run else 'delete'}")
    print(f"stopped runs: {counts.stopped_runs}")
    print(f"matched     : {counts.matched_files}")
    if not counts.dry_run:
        print(f"removed     : {counts.removed_files}")


def print_scf_summary(
    counts: ScfCounts,
    threshold: float,
    runlist: Path,
    outfile: Path | None,
    clean: bool,
    dry_run: bool,
) -> None:
    print()
    print("SCF summary")
    print(f"threshold   : {threshold:g} eV")
    print(f"runlist     : {runlist}")
    print(f"total runs  : {counts.total}")
    print(f"checked     : {counts.checked}")
    print(f"above thr   : {counts.above} / {counts.checked} ({_pct(counts.above, counts.checked)}%)")
    print(f"no DAV      : {counts.no_dav}")
    print(f"missing log : {counts.missing_log}")
    print(f"missing dir : {counts.missing_dir}")
    print(f"stale NOLOG : {counts.stale_nolog}")
    if outfile is not None:
        print(f"bad list    : {outfile}")
    if clean:
        print(f"clean mode  : {'dry-run' if dry_run else 'delete'}")
        print(f"cleaned dirs: {counts.cleaned_dirs}")


def checkvasp(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="checkvasp",
        description="Check completion state for VASP array runs listed in runlist.txt.",
    )
    parser.add_argument("runlist", nargs="?", type=Path, default=Path("runlist.txt"))
    parser.add_argument(
        "--stopped-after-min",
        type=float,
        default=DEFAULT_CHECKVASP_STOPPED_AFTER_MINUTES,
        help="Report RUNNING-looking runs as STOPPED if the newest VASP output is older than this many minutes.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory containing array logs such as vasp.out_std.<job>.<index>; default is runlist parent.",
    )
    args = parser.parse_args(argv)

    counts = check_runs(
        args.runlist,
        stopped_after_minutes=args.stopped_after_min,
        log_dir=args.log_dir,
    )
    print_run_status_summary(counts)


def checkscf(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="checkscf",
        description="Check final DAV energy convergence for VASP array logs.",
    )
    parser.add_argument(
        "args",
        nargs="+",
        help="Either THRESHOLD or RUNLIST THRESHOLD, matching the original script.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write bad run paths to this file.")
    parser.add_argument("--clean", action="store_true", help="Remove selected VASP output files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --clean, print files that would be removed without deleting them.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("."),
        help="Directory containing vasp.out*.<index> logs. Default: current directory.",
    )
    parsed = parser.parse_args(argv)

    runlist, threshold = _parse_checkscf_positionals(parsed.args, parser)
    counts = check_scf(
        runlist=runlist,
        threshold=threshold,
        outfile=parsed.out,
        clean=parsed.clean,
        dry_run=parsed.dry_run,
        log_dir=parsed.log_dir,
    )
    print_scf_summary(
        counts=counts,
        threshold=threshold,
        runlist=runlist,
        outfile=parsed.out,
        clean=parsed.clean,
        dry_run=parsed.dry_run,
    )


def vasp_energies(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="checkeng",
        description="Print latest VASP energy for each run in a runlist.txt.",
    )
    parser.add_argument("runlist", nargs="?", type=Path, default=Path("runlist.txt"))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory containing array logs such as vasp.out*.N. Default: runlist parent.",
    )
    parser.add_argument(
        "--energy",
        choices=tuple(ENERGY_PATTERNS),
        default="toten",
        help="Preferred energy to report. Falls back to other known VASP energy lines if missing.",
    )
    parser.add_argument(
        "--stopped-after-min",
        type=float,
        default=DEFAULT_CHECKENG_STOPPED_AFTER_MINUTES,
        help="Mark non-DONE rows as STOPPED if the source log has not been written for this many minutes.",
    )
    parser.add_argument(
        "--dav-average-window",
        type=int,
        default=DEFAULT_CHECKENG_DAV_AVERAGE_WINDOW,
        help="When reporting DAV fallback energy, average this many latest DAV electronic energies.",
    )
    parser.add_argument(
        "--delimiter",
        default="  ",
        help="Column delimiter. Use 'tab' for TSV output.",
    )
    parser.add_argument(
        "--clean-stopped",
        action="store_true",
        help="After reporting energies, clean files belonging to STOPPED rows. Dry-run unless --clean-execute is set.",
    )
    parser.add_argument(
        "--clean-execute",
        action="store_true",
        help="Actually delete files selected by --clean-stopped. Without this flag, cleanup is a dry-run.",
    )
    parser.add_argument(
        "--clean-scope",
        choices=("artifacts", "run", "both"),
        default="artifacts",
        help="Where to clean stopped outputs: scheduler artifacts, run folders, or both. Default: artifacts.",
    )
    parser.add_argument(
        "--clean-pattern",
        action="append",
        default=None,
        help=(
            "File glob to clean for stopped rows. Repeat for several patterns. "
            "Default removes VASP output artifacts but keeps input files."
        ),
    )
    args = parser.parse_args(argv)

    records = collect_run_energies(
        args.runlist,
        log_dir=args.log_dir,
        energy_kind=args.energy,
        stopped_after_minutes=args.stopped_after_min,
        dav_average_window=args.dav_average_window,
    )
    print_energy_table(records, delimiter=args.delimiter)
    if args.clean_stopped:
        counts = clean_stopped_energy_outputs(
            runlist=args.runlist,
            records=records,
            log_dir=args.log_dir,
            patterns=args.clean_pattern,
            scope=args.clean_scope,
            execute=args.clean_execute,
        )
        print_stopped_clean_summary(counts)


def _parse_checkscf_positionals(
    values: list[str], parser: argparse.ArgumentParser
) -> tuple[Path, float]:
    if len(values) == 1:
        runlist = Path("runlist.txt")
        threshold_text = values[0]
    elif len(values) == 2:
        runlist = Path(values[0])
        threshold_text = values[1]
    else:
        parser.error("usage is checkscf THRESHOLD or checkscf RUNLIST THRESHOLD")

    try:
        threshold = float(threshold_text)
    except ValueError:
        parser.error(f"invalid threshold: {threshold_text}")

    return runlist, threshold


def _has_done_marker(path: Path) -> bool:
    if not path.is_file():
        return False

    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            return any(DONE_MARKER in line for line in handle)
    except OSError:
        return False


def _run_is_done(runpath: Path) -> bool:
    return _has_done_marker(runpath / "OUTCAR") or _has_done_marker(runpath / "OUTCAR.gz")


def _latest_vasp_output(
    runpath: Path,
    index: int | None = None,
    log_dir: Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("vasp.out*", "OUTCAR", "OUTCAR.gz", "OSZICAR", "vasprun.xml"):
        candidates.extend(path for path in runpath.glob(pattern) if path.is_file())
    if index is not None and log_dir is not None:
        candidates.extend(array_indexed_output_candidates(index, log_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _find_vasp_log_for_index(index: int, log_dir: Path) -> Path | None:
    matches = sorted(array_indexed_output_candidates(index, log_dir), key=_energy_candidate_sort_key)
    return matches[0] if matches else None


def _energy_candidate_files(index: int, runpath: Path, log_dir: Path) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    candidates.extend((_array_energy_rank(path), path) for path in array_indexed_output_candidates(index, log_dir))
    if runpath.is_dir():
        for pattern in ("vasp.out*", "OUTCAR", "OUTCAR.gz", "OSZICAR"):
            candidates.extend(
                (_run_folder_energy_rank(path), path)
                for path in sorted(runpath.glob(pattern))
                if path.is_file()
            )
    seen: dict[Path, tuple[int, Path]] = {}
    for rank, path in candidates:
        resolved = path.resolve()
        existing = seen.get(resolved)
        if existing is not None and existing[0] <= rank:
            continue
        seen[resolved] = (rank, path)
    return [
        path
        for rank, path in sorted(
            seen.values(),
            key=lambda item: (item[0], -(item[1].stat().st_mtime if item[1].exists() else 0.0)),
        )
    ]


def _energy_candidate_sort_key(path: Path) -> tuple[int, float]:
    return (_array_energy_rank(path), -(path.stat().st_mtime if path.exists() else 0.0))


def _array_energy_rank(path: Path) -> int:
    if path.name.startswith("vasp.out"):
        return 0
    if path.name == "OSZICAR":
        return 1
    if path.name.startswith("OUTCAR"):
        return 2
    if path.name.startswith("vasprun.xml"):
        return 3
    return 4


def _run_folder_energy_rank(path: Path) -> int:
    return 5 + _array_energy_rank(path)


def array_indexed_output_candidates(index: int, log_dir: Path) -> list[Path]:
    """Return root-level or scheduler-folder VASP artifacts for one array index."""
    if not log_dir.is_dir():
        return []
    candidates: list[Path] = []
    for path in log_dir.iterdir():
        if not _name_has_array_index(path.name, index):
            continue
        if path.is_file():
            candidates.append(path)
            continue
        if not path.is_dir():
            continue
        for pattern in ARRAY_ARTIFACT_PATTERNS:
            candidates.extend(child for child in path.rglob(pattern) if child.is_file())
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique, key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)


def array_indexed_artifact_roots(index: int, log_dir: Path, exclude: Path | None = None) -> list[Path]:
    if not log_dir.is_dir():
        return []
    excluded = exclude.resolve() if exclude is not None else None
    roots = []
    for path in log_dir.iterdir():
        if not _name_has_array_index(path.name, index):
            continue
        if excluded is not None and path.resolve() == excluded:
            continue
        roots.append(path)
    return sorted(roots, key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)


def _clean_candidate_files(roots: list[Path], patterns: list[str]) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        if root.is_file():
            if _matches_any(root, patterns):
                candidates.append(root)
            continue
        if not root.is_dir():
            continue
        for pattern in patterns:
            candidates.extend(path for path in root.rglob(pattern) if path.is_file() or path.is_symlink())
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique)


def _matches_any(path: Path, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def _name_has_array_index(name: str, index: int) -> bool:
    return re.search(rf"(?:^|[._-])0*{index}(?:[._-]|$)", name) is not None


def _last_dav_energy_change(path: Path) -> float | None:
    last_de = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if parts and parts[0] == "DAV:" and len(parts) >= 4:
                try:
                    last_de = float(parts[3])
                except ValueError:
                    continue
    return last_de


def _latest_vasp_energy(
    path: Path,
    preferred_kind: str = "toten",
    dav_average_window: int = DEFAULT_CHECKENG_DAV_AVERAGE_WINDOW,
) -> tuple[float | None, str]:
    latest: dict[str, float] = {}
    dav_energies: list[float] = []
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for kind, pattern in ENERGY_PATTERNS.items():
                    match = pattern.search(line)
                    if not match:
                        continue
                    try:
                        value = float(match.group(1))
                    except ValueError:
                        continue
                    latest[kind] = value
                    if kind == "dav":
                        dav_energies.append(value)
    except OSError:
        return None, ""

    for kind in (preferred_kind, "toten", "without_entropy", "e0", "f", "dav"):
        if kind == "dav" and dav_energies:
            window = max(1, int(dav_average_window))
            selected = dav_energies[-window:]
            return sum(selected) / len(selected), f"dav_avg{min(window, len(selected))}"
        if kind in latest:
            return latest[kind], kind
    return None, ""


def _has_vasp_outputs(runpath: Path) -> bool:
    for pattern in OUTPUT_PATTERNS:
        if any(runpath.glob(pattern)):
            return True
    return False


def _clean_one_dir(runpath: Path, dry_run: bool) -> None:
    if not runpath.is_dir():
        print(f"  skip cleaning; directory not found: {runpath}")
        return

    files: list[Path] = []
    for pattern in CLEAN_PATTERNS:
        files.extend(path for path in runpath.glob(pattern) if path.is_file() or path.is_symlink())

    unique_files = sorted(set(files))
    if not unique_files:
        print(f"  no matching files found in {runpath}")
        return

    for path in unique_files:
        if dry_run:
            print(f"  would remove: {path}")
        else:
            path.unlink(missing_ok=True)
            print(f"  removed: {path}")


def _append_bad_run(outfile: Path | None, runpath: Path) -> None:
    if outfile is None:
        return
    with outfile.open("a", encoding="utf-8") as handle:
        handle.write(f"{runpath}\n")


def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.0"
    return f"{100 * part / total:.1f}"
