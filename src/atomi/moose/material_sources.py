"""Fetch and compare external material-property sources for MOOSE workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from atomi.moose.material_export import MOOSE_FIELDS, format_value


SOURCE_FIELDS = MOOSE_FIELDS + ["provider", "material_id", "citation", "notes"]
COMPARE_FIELDS = [
    "source",
    "T_K",
    "field",
    "value",
    "reference_source",
    "reference_value",
    "delta",
    "relative_delta",
]
PLOT_FIELDS = [
    "k_W_mK",
    "Cp_J_kgK",
    "rho_kg_m3",
    "alpha_1_K",
    "dilatation",
    "E_Pa",
    "nu",
    "K_Pa",
    "G_Pa",
]


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_source_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Source label is empty in {value!r}")
    return label, Path(path)


def formula_to_aflow_compound(formula: str) -> str:
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    if not tokens:
        raise SystemExit(f"Cannot parse formula for AFLOW query: {formula!r}")
    parts = []
    for element, count in sorted(tokens):
        parts.append(f"{element}{count or '1'}")
    return "".join(parts)


def elastic_row_from_kg(
    *,
    provider: str,
    material: str,
    material_id: str,
    k_gpa: float | None,
    g_gpa: float | None,
    nu: float | None,
    citation: str,
    notes: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {field: None for field in SOURCE_FIELDS}
    row["provider"] = provider
    row["material_id"] = material_id
    row["citation"] = citation
    row["notes"] = notes
    row["source_tag"] = f"{provider}:{material_id or material}"
    if k_gpa is not None:
        row["K_Pa"] = k_gpa * 1e9
    if g_gpa is not None:
        row["G_Pa"] = g_gpa * 1e9
    if k_gpa is not None and g_gpa is not None and (3.0 * k_gpa + g_gpa) != 0:
        row["E_Pa"] = 9.0 * k_gpa * g_gpa / (3.0 * k_gpa + g_gpa) * 1e9
    if nu is not None:
        row["nu"] = nu
    elif k_gpa is not None and g_gpa is not None and (2.0 * (3.0 * k_gpa + g_gpa)) != 0:
        row["nu"] = (3.0 * k_gpa - 2.0 * g_gpa) / (2.0 * (3.0 * k_gpa + g_gpa))
    return row


def fetch_materials_project(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from mp_api.client import MPRester  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Materials Project fetching requires the optional mp-api package. "
            "Install mp-api and set MP_API_KEY, or provide --api-key-env."
        ) from exc
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    material_ids = [args.material_id] if args.material_id else None
    fields = [
        "material_id",
        "formula_pretty",
        "k_vrh",
        "g_vrh",
        "homogeneous_poisson",
    ]
    with MPRester(api_key) as mpr:
        if material_ids:
            docs = mpr.materials.summary.search(material_ids=material_ids, fields=fields)
        else:
            docs = mpr.materials.summary.search(formula=args.material, fields=fields)
    if not docs:
        raise SystemExit(f"No Materials Project summary results for {args.material!r}")
    doc = docs[0]
    get = doc.get if isinstance(doc, dict) else lambda key, default=None: getattr(doc, key, default)
    row = elastic_row_from_kg(
        provider="materials-project",
        material=args.material,
        material_id=str(get("material_id", "")),
        k_gpa=finite_float(get("k_vrh")),
        g_gpa=finite_float(get("g_vrh")),
        nu=finite_float(get("homogeneous_poisson")),
        citation="Materials Project API; cite Materials Project and mp-api for retrieved data.",
        notes="0 K DFT-derived elastic summary; use as comparison/filler, not silent truth.",
    )
    metadata = {
        "provider": "materials-project",
        "material": args.material,
        "material_id": row["material_id"],
        "fields": ["k_vrh", "g_vrh", "homogeneous_poisson"],
        "source_url": "https://docs.materialsproject.org/",
    }
    return [row], metadata


def fetch_aflow(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compound = formula_to_aflow_compound(args.material)
    keywords = [
        "compound",
        "auid",
        "ael_bulk_modulus_vrh",
        "ael_shear_modulus_vrh",
        "ael_poisson_ratio",
    ]
    query = (
        f"compound({compound}),"
        + ",".join(keywords[1:])
        + ",$paging(1),$format(json)"
    )
    url = "https://aflow.org/API/aflux/v1.0/?" + urllib.parse.quote(query, safe="(),$:")
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise SystemExit(f"AFLOW AFLUX query failed: {url}\n{exc}") from exc
    if not payload:
        raise SystemExit(f"No AFLOW results for compound query {compound!r}")
    datum = payload[0]
    row = elastic_row_from_kg(
        provider="aflow",
        material=args.material,
        material_id=str(datum.get("auid", "")),
        k_gpa=finite_float(datum.get("ael_bulk_modulus_vrh")),
        g_gpa=finite_float(datum.get("ael_shear_modulus_vrh")),
        nu=finite_float(datum.get("ael_poisson_ratio")),
        citation="AFLOW AFLUX API; cite AFLOW/AFLOWLIB and the retrieved entry.",
        notes="AFLOW AEL elastic summary; use as comparison/filler with provenance.",
    )
    metadata = {
        "provider": "aflow",
        "material": args.material,
        "compound_query": compound,
        "query_url": url,
        "keywords": keywords,
    }
    return [row], metadata


def normalize_user_csv(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not args.input:
        raise SystemExit("--input is required for provider=user-csv")
    rows = []
    for source in read_rows(args.input):
        row = {field: source.get(field) for field in SOURCE_FIELDS}
        row["provider"] = source.get("provider") or args.source_label or "user-csv"
        row["citation"] = source.get("citation") or args.citation
        row["notes"] = source.get("notes") or args.notes
        row["source_tag"] = source.get("source_tag") or row["provider"]
        rows.append(row)
    metadata = {
        "provider": "user-csv",
        "input": str(args.input),
        "citation": args.citation,
        "notes": args.notes,
    }
    return rows, metadata


def source_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-material-source",
        description="Fetch or normalize external material properties for moose-qha-md-material.",
    )
    parser.add_argument(
        "--provider",
        choices=("materials-project", "aflow", "user-csv"),
        required=True,
    )
    parser.add_argument("--material", default="UO2")
    parser.add_argument("--material-id", help="Provider-specific material id, e.g. mp-1234.")
    parser.add_argument("--input", type=Path, help="Curated CSV for provider=user-csv.")
    parser.add_argument("--out-csv", type=Path, default=Path("material_source_properties.csv"))
    parser.add_argument("--out-meta", type=Path, default=Path("material_source_properties.meta.json"))
    parser.add_argument("--api-key-env", default="MP_API_KEY")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--source-label")
    parser.add_argument("--citation", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    if args.provider == "materials-project":
        rows, metadata = fetch_materials_project(args)
    elif args.provider == "aflow":
        rows, metadata = fetch_aflow(args)
    else:
        rows, metadata = normalize_user_csv(args)
    write_rows(args.out_csv, rows, SOURCE_FIELDS)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_meta}")


def collect_series(label: str, rows: list[dict[str, str]]) -> dict[str, dict[float, float]]:
    series: dict[str, dict[float, float]] = {field: {} for field in PLOT_FIELDS}
    for row in rows:
        temp = finite_float(row.get("T_K"))
        if temp is None:
            continue
        for field in PLOT_FIELDS:
            value = finite_float(row.get(field))
            if value is not None:
                series[field][temp] = value
    return series


def compare_sources(
    sources: list[tuple[str, Path]],
    *,
    reference_label: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[float, float]]]]:
    all_series: dict[str, dict[str, dict[float, float]]] = {}
    for label, path in sources:
        all_series[label] = collect_series(label, read_rows(path))
    if reference_label is None:
        reference_label = sources[0][0]
    reference = all_series.get(reference_label)
    if reference is None:
        raise SystemExit(f"Reference source {reference_label!r} was not found.")
    comparison_rows: list[dict[str, Any]] = []
    for label, series_by_field in all_series.items():
        for field, values_by_temp in series_by_field.items():
            ref_by_temp = reference.get(field, {})
            for temp, value in sorted(values_by_temp.items()):
                ref = ref_by_temp.get(temp)
                delta = None if ref is None else value - ref
                rel = None if ref in (None, 0.0) else delta / ref
                comparison_rows.append(
                    {
                        "source": label,
                        "T_K": temp,
                        "field": field,
                        "value": value,
                        "reference_source": reference_label,
                        "reference_value": ref,
                        "delta": delta,
                        "relative_delta": rel,
                    }
                )
    return comparison_rows, all_series


def plot_comparisons(outdir: Path, series: dict[str, dict[str, dict[float, float]]]) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written = []
    for field in PLOT_FIELDS:
        plotted = False
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for label, by_field in series.items():
            values = by_field.get(field, {})
            if not values:
                continue
            temps = sorted(values)
            ax.plot(temps, [values[temp] for temp in temps], marker="o", linewidth=1.8, label=label)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("T (K)")
        ax.set_ylabel(field)
        ax.set_title(f"{field} comparison")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = outdir / f"{field}_comparison.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def compare_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-material-compare",
        description="Compare MOOSE material-property CSVs from Atomi and external sources.",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Material CSV to compare; first source is default reference.",
    )
    parser.add_argument("--reference", help="Reference source label. Defaults to first --source.")
    parser.add_argument("--outdir", type=Path, default=Path("moose_material_comparison"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    sources = [parse_source_arg(item) for item in args.source]
    rows, series = compare_sources(sources, reference_label=args.reference)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table = args.outdir / "material_property_comparison.csv"
    write_rows(table, rows, COMPARE_FIELDS)
    summary = {
        "sources": [{"label": label, "path": str(path)} for label, path in sources],
        "reference": args.reference or sources[0][0],
        "comparison_table": str(table),
    }
    if not args.no_plots:
        summary["plots"] = [str(path) for path in plot_comparisons(args.outdir, series)]
    meta = args.outdir / "material_property_comparison.meta.json"
    meta.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {table}")
    print(f"Wrote {meta}")
    for plot in summary.get("plots", []):
        print(f"Wrote {plot}")


if __name__ == "__main__":
    compare_main()
