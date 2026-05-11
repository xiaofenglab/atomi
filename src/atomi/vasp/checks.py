from __future__ import annotations

import argparse
import gzip
import re
import time
from dataclasses import dataclass
from pathlib import Path


DONE_MARKER = "General timing and accounting informations for this job"
CLEAN_PATTERNS = ("OUTCAR*", "CONTCAR", "vasprun.xml", "OSZICAR")
OUTPUT_PATTERNS = ("vasprun.xml", "OUTCAR*", "OSZICAR", "CONTCAR")
DEFAULT_STOPPED_AFTER_MINUTES = 5.0
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
    stopped_after_minutes: float = DEFAULT_STOPPED_AFTER_MINUTES,
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
    stopped_after_minutes: float = DEFAULT_STOPPED_AFTER_MINUTES,
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
        energy, found_kind = _latest_vasp_energy(candidates[0], preferred_kind=energy_kind)
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
        default=DEFAULT_STOPPED_AFTER_MINUTES,
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
        default=DEFAULT_STOPPED_AFTER_MINUTES,
        help="Mark non-DONE rows as STOPPED if the source log has not been written for this many minutes.",
    )
    parser.add_argument(
        "--delimiter",
        default="  ",
        help="Column delimiter. Use 'tab' for TSV output.",
    )
    args = parser.parse_args(argv)

    records = collect_run_energies(
        args.runlist,
        log_dir=args.log_dir,
        energy_kind=args.energy,
        stopped_after_minutes=args.stopped_after_min,
    )
    print_energy_table(records, delimiter=args.delimiter)


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
        candidates.extend(
            sorted(path for path in log_dir.glob(f"vasp.out*.{index}") if path.is_file())
        )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _find_vasp_log_for_index(index: int, log_dir: Path) -> Path | None:
    matches = sorted(log_dir.glob(f"vasp.out*.{index}"))
    return matches[0] if matches else None


def _energy_candidate_files(index: int, runpath: Path, log_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(sorted(log_dir.glob(f"vasp.out*.{index}")))
    if runpath.is_dir():
        for pattern in ("vasp.out*", "OUTCAR", "OUTCAR.gz", "OSZICAR"):
            candidates.extend(sorted(path for path in runpath.glob(pattern) if path.is_file()))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique, key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)


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


def _latest_vasp_energy(path: Path, preferred_kind: str = "toten") -> tuple[float | None, str]:
    latest: dict[str, float] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for kind, pattern in ENERGY_PATTERNS.items():
                    match = pattern.search(line)
                    if not match:
                        continue
                    try:
                        latest[kind] = float(match.group(1))
                    except ValueError:
                        continue
    except OSError:
        return None, ""

    for kind in (preferred_kind, "toten", "without_entropy", "e0", "f", "dav"):
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
