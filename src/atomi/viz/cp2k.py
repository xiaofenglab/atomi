import subprocess
from importlib.resources import files
from pathlib import Path

from atomi.viz.vasp_live import ensure_gnuplot


def plot_cp2k(
    logfile: Path,
    xyzfile: Path | None = None,
    mode: str = "auto",
    window: int = 300,
    refresh: int = 15,
) -> None:
    """Auto-detect CP2K MD/GEO logs and launch the terminal monitor."""
    if not logfile.is_file():
        raise FileNotFoundError(f"file not found: {logfile}")
    if mode == "auto":
        mode = detect_cp2k_mode(logfile)
    if mode not in {"md", "geo"}:
        raise ValueError("mode must be auto, md, or geo")

    ensure_gnuplot()
    xyz = xyzfile or auto_find_xyz(logfile, required=False)

    if mode == "md":
        _plot_cp2k_md(logfile=logfile, xyzfile=xyz, window=window, refresh=refresh)
        return

    geodat = Path("cp2k_geo_steps.dat")
    scfdat = Path("cp2k_geo_scf.dat")
    write_geo_tables(logfile=logfile, geodat=geodat, scfdat=scfdat)
    _plot_cp2k_geo(logfile=logfile, geodat=geodat, scfdat=scfdat, xyzfile=xyz, refresh=refresh)


def plot_cp2k_all(logfile: Path) -> None:
    """Launch the full CP2K GEO convergence dashboard."""
    if not logfile.is_file():
        raise FileNotFoundError(f"file not found: {logfile}")
    ensure_gnuplot()
    script = _resource("plot_cp2k_all.gp")
    _run_gnuplot([f"file='{_quote(logfile)}'"], script)


def detect_cp2k_mode(logfile: Path) -> str:
    """Return md or geo from common CP2K log markers."""
    saw_geo = False
    with logfile.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            upper = line.upper()
            if "MD|" in line or "RUN_TYPE" in upper and "MD" in upper:
                return "md"
            if "STEP NUMBER" in upper or "TEMPERATURE [K]" in upper:
                return "md"
            if (
                "INFORMATIONS AT STEP" in upper
                or "OPTIMIZATION STEP" in upper
                or "STARTING GEOMETRY OPTIMIZATION" in upper
            ):
                saw_geo = True
    return "geo" if saw_geo else "md"


def auto_find_xyz(logfile: Path, required: bool = False) -> Path | None:
    """Find the likely CP2K position trajectory next to a log file."""
    stem = logfile.with_suffix("")
    parent = logfile.parent
    preferred = [
        Path(str(stem) + "-pos.xyz"),
        Path(str(stem) + "-pos-1.xyz"),
        Path(str(stem) + ".pos.xyz"),
        Path(str(stem) + ".pos-1.xyz"),
        Path(str(stem) + "_pos.xyz"),
        Path(str(stem) + "_pos-1.xyz"),
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate

    pos_candidates = sorted(parent.glob("*pos*.xyz"))
    if pos_candidates:
        return sorted(
            pos_candidates,
            key=lambda path: (path.stat().st_size, path.stat().st_mtime),
            reverse=True,
        )[0]

    plain = Path(str(stem) + ".xyz")
    if plain.is_file():
        return plain

    xyzs = sorted(parent.glob("*.xyz"))
    if len(xyzs) == 1:
        return xyzs[0]
    if required:
        raise FileNotFoundError("Could not auto-find a CP2K trajectory xyz file.")
    return None


def write_geo_tables(logfile: Path, geodat: Path, scfdat: Path) -> None:
    """Write compact GEO and outer-SCF tables for the CP2K GEO gnuplot monitor."""
    geo_rows: list[str] = []
    scf_rows: list[str] = []
    state: dict[str, str] = {}
    gstep = ""

    with logfile.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "outer SCF iter" in line:
                parsed = _parse_outer_scf(line)
                if gstep and parsed:
                    scf_rows.append(f"{gstep} {parsed[0]} {parsed[1]} {parsed[2]} {parsed[1]}")
            elif "Informations at step" in line:
                gstep = _after_step_equals(line)
                state = {"step": gstep, "trust": "0.0"}
            elif "Total Energy" in line and "=" in line:
                state["energy"] = _after_equals(line)
            elif "Decrease in energy" in line and "=" in line:
                value = _after_equals(line)
                state["decrease"] = "1" if value == "YES" else "0"
            elif "Used time" in line and "=" in line:
                state["used"] = _after_equals(line)
            elif "Max. step size" in line and "=" in line:
                state["smax"] = _after_equals(line)
            elif "RMS step size" in line and "=" in line:
                state["srms"] = _after_equals(line)
            elif "Max. gradient" in line and "=" in line:
                state["gmax"] = _after_equals(line)
            elif "RMS gradient" in line and "=" in line:
                state["grms"] = _after_equals(line)
                if state.get("step") and state.get("energy"):
                    geo_rows.append(
                        " ".join(
                            [
                                state.get("step", "0"),
                                state.get("energy", "nan"),
                                state.get("decrease", "0"),
                                state.get("smax", "nan"),
                                state.get("srms", "nan"),
                                state.get("used", "nan"),
                                state.get("gmax", "nan"),
                                state.get("grms", "nan"),
                                state.get("trust", "0.0"),
                            ]
                        )
                    )
            elif "Trust radius" in line and "=" in line:
                state["trust"] = _after_equals(line)

    geodat.write_text("\n".join(geo_rows) + ("\n" if geo_rows else ""), encoding="utf-8")
    scfdat.write_text("\n".join(scf_rows) + ("\n" if scf_rows else ""), encoding="utf-8")


def _plot_cp2k_md(logfile: Path, xyzfile: Path | None, window: int, refresh: int) -> None:
    if xyzfile is None:
        print("WARNING: no XYZ trajectory found; bond panels will be skipped.")

    assignments = [
        f"file='{_quote(logfile)}'",
        f"helper_py='{_quote(_resource('cp2k_md_bondtrack.py'))}'",
        f"eta_py='{_quote(_resource('cp2k_md_eta.py'))}'",
        f"win={window}",
        f"refresh={refresh}",
    ]
    if xyzfile is not None:
        assignments.append(f"xyzfile='{_quote(xyzfile)}'")
    _run_gnuplot(assignments, _resource("plot_cp2k_md_live.gp"))


def _plot_cp2k_geo(
    logfile: Path,
    geodat: Path,
    scfdat: Path,
    xyzfile: Path | None,
    refresh: int,
) -> None:
    assignments = [
        f"file='{_quote(logfile)}'",
        f"geodat='{_quote(geodat)}'",
        f"scfdat='{_quote(scfdat)}'",
        f"helper_py='{_quote(_resource('cp2k_md_bondtrack.py'))}'",
        f"refresh={refresh}",
    ]
    if xyzfile is not None:
        assignments.append(f"xyzfile='{_quote(xyzfile)}'")
    _run_gnuplot(assignments, _resource("plot_cp2k_geo_live.gp"))


def _run_gnuplot(assignments: list[str], script: Path) -> None:
    command = ["gnuplot", "-e", "; ".join(assignments), str(script)]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        message = [
            f"gnuplot failed with exit code {result.returncode}.",
            "Command:",
            " ".join(command),
        ]
        if stderr:
            message.extend(["gnuplot stderr:", stderr])
        raise RuntimeError("\n".join(message))


def _resource(name: str) -> Path:
    return Path(str(files("atomi").joinpath("viz", "cp2k", name)))


def _quote(path: Path | str) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "\\'")


def _after_equals(line: str) -> str:
    return line.rsplit("=", 1)[-1].strip().split()[0]


def _after_step_equals(line: str) -> str:
    value = line.rsplit("=", 1)[-1].strip().split()
    return value[0] if value else ""


def _parse_outer_scf(line: str) -> tuple[str, str, str] | None:
    parts = line.replace(",", " ").split()
    iter_value = _value_after_tokens(parts, "iter")
    energy = _value_after_tokens(parts, "energy")
    rms_gradient = _value_after_tokens(parts, "gradient")
    if iter_value and energy and rms_gradient:
        return iter_value, energy, rms_gradient
    return None


def _value_after_tokens(parts: list[str], token: str) -> str | None:
    for index, part in enumerate(parts[:-1]):
        if part == token and parts[index + 1] == "=" and index + 2 < len(parts):
            return parts[index + 2]
    return None
