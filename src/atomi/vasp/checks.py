from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path


DONE_MARKER = "General timing and accounting informations for this job"
CLEAN_PATTERNS = ("OUTCAR*", "CONTCAR", "vasprun.xml", "OSZICAR")
OUTPUT_PATTERNS = ("vasprun.xml", "OUTCAR*", "OSZICAR", "CONTCAR")


@dataclass
class RunStatusCounts:
    total: int = 0
    done: int = 0
    running: int = 0
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


def iter_runlist(runlist: Path) -> list[Path]:
    runs: list[Path] = []
    for raw_line in runlist.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        runs.append(Path(line))
    return runs


def check_runs(runlist: Path) -> RunStatusCounts:
    if not runlist.is_file():
        raise FileNotFoundError(f"cannot find {runlist}")

    counts = RunStatusCounts()
    for runpath in iter_runlist(runlist):
        counts.total += 1
        if not runpath.is_dir():
            print(f"MISSING   {runpath}")
            counts.missing += 1
            continue

        if _has_done_marker(runpath / "OUTCAR") or _has_done_marker(runpath / "OUTCAR.gz"):
            print(f"DONE      {runpath}")
            counts.done += 1
            continue

        if _has_running_output(runpath):
            print(f"RUNNING   {runpath}")
            counts.running += 1
        else:
            print(f"NOTSTART  {runpath}")
            counts.not_started += 1

    return counts


def print_run_status_summary(counts: RunStatusCounts) -> None:
    print()
    print("Summary:")
    print(f"DONE      : {counts.done} / {counts.total} ({_pct(counts.done, counts.total)}%)")
    print(f"RUNNING   : {counts.running} / {counts.total} ({_pct(counts.running, counts.total)}%)")
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
    args = parser.parse_args(argv)

    counts = check_runs(args.runlist)
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


def _has_running_output(runpath: Path) -> bool:
    return (
        (runpath / "OUTCAR").is_file()
        or (runpath / "OUTCAR.gz").is_file()
        or any(runpath.glob("vasp.out*"))
    )


def _find_vasp_log_for_index(index: int, log_dir: Path) -> Path | None:
    matches = sorted(log_dir.glob(f"vasp.out*.{index}"))
    return matches[0] if matches else None


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
