"""Bridge AIMD aqueous free-energy results to ThermoHub/ThermoFun/GEMS workflows.

The command intentionally has no hard dependency on ThermoFun. It can run in the
main Atomi environment to normalize AIMD logK tables and write database request
files. When a ThermoFun-capable Python and local ThermoFun JSON database are
provided, it also queries reaction standard-state properties.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

R_GAS_CONSTANT = 8.31446261815324


@dataclass(frozen=True)
class ReactionSpec:
    """A database reaction or AIMD conditional step to track."""

    name: str
    equation: str
    role: str = "database"
    degeneracy: int = 1
    note: str = ""


DEFAULT_GA_CL_REACTIONS: tuple[ReactionSpec, ...] = (
    ReactionSpec("GaCl+2", "Ga+3 + Cl- = GaCl+2", note="First chloride association."),
    ReactionSpec("GaCl2+", "GaCl+2 + Cl- = GaCl2+", note="Second chloride association."),
    ReactionSpec("GaCl3", "GaCl2+ + Cl- = GaCl3", note="Third chloride association."),
    ReactionSpec(
        "GaCl4-",
        "GaCl3 + Cl- = GaCl4-",
        role="aimd_anchor",
        degeneracy=1,
        note="AIMD TI/PMF conditional step used in the Ga-Cl-Ow case study.",
    ),
    ReactionSpec("GaOH+2", "Ga+3 + H2O = GaOH+2 + H+", note="Hydrolysis guard species."),
    ReactionSpec("Ga(OH)2+", "GaOH+2 + H2O = Ga(OH)2+ + H+", note="Hydrolysis guard species."),
    ReactionSpec("Ga(OH)3", "Ga(OH)2+ + H2O = Ga(OH)3 + H+", note="Hydrolysis guard species."),
    ReactionSpec("Ga(OH)4-", "Ga(OH)3 + H2O = Ga(OH)4- + H+", note="Hydrolysis guard species."),
)

DEFAULT_SPECIES = (
    "Ga+3",
    "Cl-",
    "H+",
    "H2O",
    "GaCl+2",
    "GaCl2+",
    "GaCl3",
    "GaCl4-",
    "GaOH+2",
    "Ga(OH)2+",
    "Ga(OH)3",
    "Ga(OH)4-",
)


@dataclass
class AimdLogKRow:
    temperature_c: float
    log10_k: float
    sigma: float | None = None
    source: str = "aimd"
    note: str = ""


@dataclass
class ThermoFunResult:
    reaction: str
    equation: str
    temperature_c: float
    pressure_bar: float
    log10_k: float | None
    ln_k: float | None
    delta_g_j_mol: float | None
    delta_h_j_mol: float | None
    delta_s_j_mol_k: float | None
    status: str
    message: str = ""


@dataclass
class GemsCapability:
    python: str
    modules: dict[str, dict[str, Any]]
    gems3k_python_module: str | None
    equilibrium_layer: str | None
    status: str
    message: str = ""


def _module_status(import_name: str, distribution_name: str | None = None, include_symbols: bool = False) -> dict[str, Any]:
    spec = importlib.util.find_spec(import_name)
    row: dict[str, Any] = {"available": spec is not None, "origin": getattr(spec, "origin", None) if spec else None}
    if spec is None:
        return row
    try:
        row["version"] = importlib.metadata.version(distribution_name or import_name)
    except importlib.metadata.PackageNotFoundError:
        row["version"] = "unknown"
    if include_symbols:
        try:
            module = __import__(import_name)
            row["public_symbols"] = [name for name in dir(module) if not name.startswith("_")][:80]
        except Exception as exc:  # optional package import can fail from missing shared libs.
            row["import_error"] = str(exc)
    return row


def _probe_gems_inprocess(include_symbols: bool = False) -> GemsCapability:
    modules = {
        "thermofun": _module_status("thermofun", include_symbols=include_symbols),
        "thermohubclient": _module_status("thermohubclient", include_symbols=include_symbols),
        "chemicalfun": _module_status("chemicalfun", include_symbols=include_symbols),
        "solmod4rkt": _module_status("solmod4rkt", include_symbols=include_symbols),
        "easygems": _module_status("easygems", include_symbols=include_symbols),
        "xgems": _module_status("xgems", include_symbols=include_symbols),
        "gems3k": _module_status("gems3k", include_symbols=include_symbols),
        "gems": _module_status("gems", include_symbols=include_symbols),
    }
    gems3k_module = None
    if modules["solmod4rkt"]["available"]:
        gems3k_module = "solmod4rkt"
    elif modules["gems3k"]["available"]:
        gems3k_module = "gems3k"
    elif modules["gems"]["available"]:
        gems3k_module = "gems"
    equilibrium_layer = None
    if modules["easygems"]["available"]:
        equilibrium_layer = "easygems"
    elif modules["xgems"]["available"]:
        equilibrium_layer = "xgems"
    elif gems3k_module:
        equilibrium_layer = gems3k_module
    status = "ready" if gems3k_module and equilibrium_layer else "partial" if gems3k_module else "missing"
    if status == "ready":
        message = "GEMS3K Python bindings are available; a Ga-Cl-O-H GEMS/ThermoFun database project is still required for a true equilibrium logK/speciation curve."
    elif status == "partial":
        message = "GEMS3K low-level bindings are available, but no higher-level equilibrium helper was detected."
    else:
        message = "No GEMS3K Python bindings detected in this Python environment."
    return GemsCapability(sys.executable, modules, gems3k_module, equilibrium_layer, status, message)


def probe_gems(gems_python: Path | None = None, include_symbols: bool = False) -> GemsCapability:
    if gems_python is None:
        return _probe_gems_inprocess(include_symbols=include_symbols)
    helper = """
import json
from atomi.aqueous.thermohub_bridge import _probe_gems_inprocess
capability = _probe_gems_inprocess(include_symbols=__import__('sys').argv[1] == '1')
print(json.dumps(capability.__dict__))
"""
    with tempfile.TemporaryDirectory(prefix="atomi_gems_probe_") as tmp:
        tmp_path = Path(tmp)
        helper_path = tmp_path / "probe_gems.py"
        helper_path.write_text(helper, encoding="utf-8")
        env = os.environ.copy()
        src_root = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [str(gems_python), str(helper_path), "1" if include_symbols else "0"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            return GemsCapability(
                str(gems_python),
                {},
                None,
                None,
                "probe_failed",
                proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}",
            )
        json_lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
        if not json_lines:
            return GemsCapability(str(gems_python), {}, None, None, "probe_failed", proc.stdout.strip())
        payload = json.loads(json_lines[-1])
        return GemsCapability(**payload)


def _split_float_values(values: Sequence[str] | None, default: Sequence[float]) -> list[float]:
    if not values:
        return list(default)
    out: list[float] = []
    for value in values:
        for chunk in str(value).split(","):
            chunk = chunk.strip()
            if chunk:
                out.append(float(chunk))
    return out or list(default)


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    for attr in ("val", "value", "scalar", "number"):
        if hasattr(value, attr):
            try:
                candidate = getattr(value, attr)
                if callable(candidate):
                    candidate = candidate()
                return _coerce_float(candidate)
            except Exception:
                pass
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _property_value(obj: object, *names: str) -> float | None:
    for name in names:
        if not hasattr(obj, name):
            continue
        try:
            value = getattr(obj, name)
            if callable(value):
                value = value()
        except Exception:
            continue
        number = _coerce_float(value)
        if number is not None:
            return number
    return None


def load_reactions(path: Path | None) -> list[ReactionSpec]:
    if path is None:
        return list(DEFAULT_GA_CL_REACTIONS)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        records = payload.get("reactions", payload) if isinstance(payload, dict) else payload
        return [ReactionSpec(**record) for record in records]
    reactions: list[ReactionSpec] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            equation = row.get("equation") or row.get("reaction") or row.get("formula")
            if not equation:
                continue
            reactions.append(
                ReactionSpec(
                    name=row.get("name") or f"reaction_{idx}",
                    equation=equation,
                    role=row.get("role") or "database",
                    degeneracy=int(float(row.get("degeneracy") or 1)),
                    note=row.get("note") or "",
                )
            )
    return reactions


def read_aimd_logk(path: Path | None) -> list[AimdLogKRow]:
    if path is None:
        return []
    rows: list[AimdLogKRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            t_raw = (
                row.get("temperature_c")
                or row.get("temperature_C")
                or row.get("T_C")
                or row.get("T(C)")
                or row.get("T")
                or row.get("temperature")
            )
            k_raw = (
                row.get("log10_K")
                or row.get("log10K")
                or row.get("logK")
                or row.get("log_k")
                or row.get("log10_k")
                or row.get("log10K_fixed_pmf_extrapolated")
                or row.get("corrected_conditional_log10_K4")
            )
            if t_raw is None or k_raw is None:
                continue
            sigma_raw = row.get("sigma") or row.get("stderr") or row.get("ci_half_width")
            rows.append(
                AimdLogKRow(
                    temperature_c=float(t_raw),
                    log10_k=float(k_raw),
                    sigma=float(sigma_raw) if sigma_raw not in (None, "") else None,
                    source=row.get("source") or path.name,
                    note=row.get("note") or "",
                )
            )
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _query_thermofun_inprocess(
    database: Path,
    reactions: Sequence[ReactionSpec],
    temperatures_c: Sequence[float],
    pressure_bar: float,
) -> list[ThermoFunResult]:
    import thermofun  # type: ignore[import-not-found]

    db = thermofun.Database(str(database))
    engine = thermofun.ThermoEngine(db)
    results: list[ThermoFunResult] = []
    for reaction in reactions:
        for temp_c in temperatures_c:
            temp_k = temp_c + 273.15
            try:
                props = engine.thermoPropertiesReaction(temp_k, pressure_bar, reaction.equation)
                log10_k = _property_value(props, "logKr", "logK", "log10K")
                ln_k = _property_value(props, "lnKr", "lnK")
                delta_g = _property_value(props, "reaction_gibbs_energy", "gibbs_energy", "deltaG")
                delta_h = _property_value(props, "reaction_enthalpy", "enthalpy", "deltaH")
                delta_s = _property_value(props, "reaction_entropy", "entropy", "deltaS")
                if log10_k is None and ln_k is not None:
                    log10_k = ln_k / math.log(10.0)
                if ln_k is None and log10_k is not None:
                    ln_k = log10_k * math.log(10.0)
                if delta_g is None and ln_k is not None:
                    delta_g = -R_GAS_CONSTANT * temp_k * ln_k
                results.append(
                    ThermoFunResult(
                        reaction=reaction.name,
                        equation=reaction.equation,
                        temperature_c=temp_c,
                        pressure_bar=pressure_bar,
                        log10_k=log10_k,
                        ln_k=ln_k,
                        delta_g_j_mol=delta_g,
                        delta_h_j_mol=delta_h,
                        delta_s_j_mol_k=delta_s,
                        status="ok" if log10_k is not None or delta_g is not None else "no_numeric_property",
                    )
                )
            except Exception as exc:  # ThermoFun reports missing species/reactions here.
                results.append(
                    ThermoFunResult(
                        reaction=reaction.name,
                        equation=reaction.equation,
                        temperature_c=temp_c,
                        pressure_bar=pressure_bar,
                        log10_k=None,
                        ln_k=None,
                        delta_g_j_mol=None,
                        delta_h_j_mol=None,
                        delta_s_j_mol_k=None,
                        status="query_failed",
                        message=str(exc),
                    )
                )
    return results


def query_thermofun(
    database: Path,
    reactions: Sequence[ReactionSpec],
    temperatures_c: Sequence[float],
    pressure_bar: float,
    thermofun_python: Path | None = None,
) -> list[ThermoFunResult]:
    """Query ThermoFun either in-process or through a sidecar Python."""

    if thermofun_python is None:
        return _query_thermofun_inprocess(database, reactions, temperatures_c, pressure_bar)

    payload = {
        "database": str(database),
        "reactions": [asdict(reaction) for reaction in reactions],
        "temperatures_c": list(temperatures_c),
        "pressure_bar": pressure_bar,
    }
    helper = """
import json
from pathlib import Path
from atomi.aqueous.thermohub_bridge import ReactionSpec, _query_thermofun_inprocess
payload = json.loads(Path(__import__('sys').argv[1]).read_text())
reactions = [ReactionSpec(**record) for record in payload['reactions']]
results = _query_thermofun_inprocess(Path(payload['database']), reactions, payload['temperatures_c'], payload['pressure_bar'])
print(json.dumps([result.__dict__ for result in results]))
"""
    with tempfile.TemporaryDirectory(prefix="atomi_thermofun_bridge_") as tmp:
        tmp_path = Path(tmp)
        payload_path = tmp_path / "payload.json"
        helper_path = tmp_path / "query_thermofun.py"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        helper_path.write_text(helper, encoding="utf-8")
        env = os.environ.copy()
        src_root = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [str(thermofun_python), str(helper_path), str(payload_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            message = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
            return [
                ThermoFunResult(
                    reaction=reaction.name,
                    equation=reaction.equation,
                    temperature_c=temp_c,
                    pressure_bar=pressure_bar,
                    log10_k=None,
                    ln_k=None,
                    delta_g_j_mol=None,
                    delta_h_j_mol=None,
                    delta_s_j_mol_k=None,
                    status="sidecar_failed",
                    message=message,
                )
                for reaction in reactions
                for temp_c in temperatures_c
            ]
        return [ThermoFunResult(**record) for record in json.loads(proc.stdout)]


def write_species_request(path: Path, reactions: Sequence[ReactionSpec], temperatures_c: Sequence[float]) -> None:
    species = sorted(set(DEFAULT_SPECIES))
    lines = [
        "# ThermoHub / GEMS Species Request",
        "",
        "This file was written by `atomi aq-thermo-bridge`.",
        "",
        "## Species Needed",
        "",
    ]
    lines.extend(f"- `{species_name}`" for species_name in species)
    lines.extend([
        "",
        "## Reactions Needed",
        "",
        "| name | role | degeneracy | equation | note |",
        "| --- | --- | ---: | --- | --- |",
    ])
    for reaction in reactions:
        lines.append(
            f"| {reaction.name} | {reaction.role} | {reaction.degeneracy} | `{reaction.equation}` | {reaction.note} |"
        )
    lines.extend([
        "",
        "## Temperature Grid",
        "",
        ", ".join(f"{temp:g} C" for temp in temperatures_c),
        "",
        "## Database Notes",
        "",
        "Export a ThermoFun JSON database containing the species above from ThermoHub, then rerun:",
        "",
        "```bash",
        "atomi aq-thermo-bridge --database thermohub_ga_cl_ow.json --thermofun-python ~/m_lammps_env_thermofun/bin/python --out aq_bridge_results",
        "```",
        "",
        "For GEMS/GEMS3K, use the same species list as the basis for equilibrium-speciation input generation.",
        "The conda-forge `gems3k` package exposes its Python-facing bindings as `solmod4rkt`; Atomi probes that module plus `easygems` and `xgems` before attempting solver-backed workflows.",
        "A true GEMS-refined logK/speciation curve requires a matching GEMS/ThermoFun database or project containing this Ga-Cl-O-H species basis.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    aimd_rows: Sequence[AimdLogKRow],
    reactions: Sequence[ReactionSpec],
    thermofun_rows: Sequence[ThermoFunResult],
    database: Path | None,
    thermofun_python: Path | None,
    pressure_bar: float,
    gems_capability: GemsCapability | None = None,
) -> None:
    ok_rows = [row for row in thermofun_rows if row.status == "ok"]
    failed_rows = [row for row in thermofun_rows if row.status != "ok"]
    lines = [
        "# Aqueous Thermodynamic Bridge Report",
        "",
        "## Purpose",
        "",
        "Connect AIMD/TI/PMF conditional stability constants to ThermoHub/ThermoFun database reactions and later GEMS equilibrium speciation.",
        "",
        "## Inputs",
        "",
        f"- ThermoFun database: `{database}`" if database else "- ThermoFun database: not supplied; request/template mode only.",
        f"- ThermoFun Python: `{thermofun_python}`" if thermofun_python else "- ThermoFun Python: current Atomi Python if ThermoFun is importable.",
        f"- Pressure: {pressure_bar:g} bar",
        f"- AIMD logK rows: {len(aimd_rows)}",
        f"- Reaction definitions: {len(reactions)}",
        "",
        "## GEMS / GEMS3K Capability",
        "",
    ]
    if gems_capability:
        lines.extend([
            f"- Python: `{gems_capability.python}`",
            f"- Status: {gems_capability.status}",
            f"- GEMS3K Python module: {gems_capability.gems3k_python_module or 'not detected'}",
            f"- Equilibrium layer: {gems_capability.equilibrium_layer or 'not detected'}",
            f"- Message: {gems_capability.message}",
            "",
            "| module | available | version | origin |",
            "| --- | --- | --- | --- |",
        ])
        for module, row in gems_capability.modules.items():
            lines.append(
                f"| {module} | {row.get('available')} | {row.get('version', '')} | `{row.get('origin', '')}` |"
            )
        lines.append("")
    else:
        lines.extend(["- Not probed.", ""])
    lines.extend([
        "## AIMD Conditional Constants",
        "",
        "| T (C) | log10 K | sigma | source | note |",
        "| ---: | ---: | ---: | --- | --- |",
    ])
    if aimd_rows:
        for row in aimd_rows:
            sigma = "" if row.sigma is None else f"{row.sigma:.6g}"
            lines.append(f"| {row.temperature_c:g} | {row.log10_k:.6g} | {sigma} | {row.source} | {row.note} |")
    else:
        lines.append("| | | | | No AIMD table supplied. |")
    lines.extend([
        "",
        "## ThermoFun Query Status",
        "",
        f"- Successful rows: {len(ok_rows)}",
        f"- Non-success rows: {len(failed_rows)}",
        "",
        "| reaction | T (C) | log10 K | dG (J/mol) | status | message |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ])
    if thermofun_rows:
        for row in thermofun_rows:
            logk = "" if row.log10_k is None else f"{row.log10_k:.6g}"
            dg = "" if row.delta_g_j_mol is None else f"{row.delta_g_j_mol:.6g}"
            msg = row.message.replace("|", "/")[:180]
            lines.append(f"| {row.reaction} | {row.temperature_c:g} | {logk} | {dg} | {row.status} | {msg} |")
    else:
        lines.append("| | | | | request_only | No database was supplied. |")
    lines.extend([
        "",
        "## Interpretation Rules",
        "",
        "- Treat AIMD logK values as conditional constants for the simulated stoichiometric step and standard-state convention.",
        "- Use ThermoFun/ThermoHub rows to anchor database standard-state reactions across T/P, not to replace AIMD PMF corrections.",
        "- Map degeneracy, radial/Jacobian, and concentration standard-state corrections before comparing AIMD and database constants.",
        "- Use GEMS/GEMS3K after the species basis is complete to solve bulk aqueous speciation from total Ga, total Cl, pH, T, P, and ionic strength.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aimd-k", type=Path, help="CSV table with temperature_c and log10_K/logK columns.")
    parser.add_argument("--database", type=Path, help="Local ThermoFun JSON database exported/downloaded from ThermoHub.")
    parser.add_argument("--thermofun-python", type=Path, help="Sidecar Python with thermofun installed, e.g. ~/m_lammps_env_thermofun/bin/python.")
    parser.add_argument("--gems-python", type=Path, help="Sidecar Python with GEMS3K/easygems/xgems installed. Defaults to --thermofun-python when supplied.")
    parser.add_argument("--gems-symbols", action="store_true", help="Include public symbol lists in gems_capability_probe.json for API debugging.")
    parser.add_argument("--reactions", type=Path, help="Optional reaction CSV/JSON. Defaults to Ga-Cl-Ow request set.")
    parser.add_argument("--temperatures-c", nargs="*", help="Temperature grid in C; comma-separated values are accepted.")
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("aqueous_thermo_bridge"))
    parser.add_argument("--write-templates", action="store_true", help="Write request/templates even if no database is supplied.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)
    aimd_rows = read_aimd_logk(args.aimd_k)
    reactions = load_reactions(args.reactions)
    temperatures_c = _split_float_values(args.temperatures_c, [25.0, 50.0, 100.0, 150.0, 200.0, 250.0])

    _write_csv(outdir / "reaction_request.csv", ["name", "equation", "role", "degeneracy", "note"], (asdict(r) for r in reactions))
    _write_csv(outdir / "aimd_logk_overlay.csv", ["temperature_c", "log10_k", "sigma", "source", "note"], (asdict(r) for r in aimd_rows))
    write_species_request(outdir / "thermohub_gems_species_request.md", reactions, temperatures_c)
    gems_python = args.gems_python or args.thermofun_python
    gems_capability = probe_gems(gems_python, include_symbols=args.gems_symbols)
    (outdir / "gems_capability_probe.json").write_text(json.dumps(asdict(gems_capability), indent=2) + "\n", encoding="utf-8")

    thermofun_rows: list[ThermoFunResult] = []
    if args.database:
        thermofun_rows = query_thermofun(args.database, reactions, temperatures_c, args.pressure_bar, args.thermofun_python)
        _write_csv(
            outdir / "thermofun_reaction_properties.csv",
            [
                "reaction",
                "equation",
                "temperature_c",
                "pressure_bar",
                "log10_k",
                "ln_k",
                "delta_g_j_mol",
                "delta_h_j_mol",
                "delta_s_j_mol_k",
                "status",
                "message",
            ],
            (asdict(r) for r in thermofun_rows),
        )
    else:
        (outdir / "thermofun_reaction_properties.csv").write_text(
            "reaction,equation,temperature_c,pressure_bar,log10_k,ln_k,delta_g_j_mol,delta_h_j_mol,delta_s_j_mol_k,status,message\n",
            encoding="utf-8",
        )

    status = {
        "outdir": str(outdir),
        "database": str(args.database) if args.database else None,
        "thermofun_python": str(args.thermofun_python) if args.thermofun_python else None,
        "gems_python": str(gems_python) if gems_python else None,
        "gems_status": gems_capability.status,
        "gems3k_python_module": gems_capability.gems3k_python_module,
        "gems_equilibrium_layer": gems_capability.equilibrium_layer,
        "pressure_bar": args.pressure_bar,
        "temperatures_c": temperatures_c,
        "aimd_rows": len(aimd_rows),
        "reactions": len(reactions),
        "thermofun_rows": len(thermofun_rows),
        "thermofun_success_rows": sum(1 for row in thermofun_rows if row.status == "ok"),
    }
    (outdir / "aqueous_bridge_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    write_report(
        outdir / "aqueous_bridge_report.md",
        aimd_rows=aimd_rows,
        reactions=reactions,
        thermofun_rows=thermofun_rows,
        database=args.database,
        thermofun_python=args.thermofun_python,
        pressure_bar=args.pressure_bar,
        gems_capability=gems_capability,
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
