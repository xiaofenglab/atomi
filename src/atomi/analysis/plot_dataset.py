"""Freeze scientific analysis outputs into reusable plotting-data bundles.

The bundle is the boundary between domain-sensitive analysis and rendering.
Project students or domain leads create it after parsing and scientific guards;
plotting tools may then change style and layout without reparsing raw HPC data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atomi.core.run_record import RunArtifact


SCHEMA = "atomi.analysis.plot_dataset.v1"
MANIFEST_NAME = "plot_dataset.json"
VALID_STATUSES = {
    "accepted",
    "provisional",
    "screening",
    "rejected",
    "conceptual",
    "mixed",
}
VALID_QUALITIES = {
    "descriptor",
    "screening-prior",
    "production",
    "publication",
    "diagnostic",
}
VALID_COLUMN_ROLES = {
    "x",
    "y",
    "error",
    "lower",
    "upper",
    "category",
    "label",
    "status",
    "auxiliary",
}
NUMERIC_DTYPES = {"float", "int"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ColumnSpec:
    """Scientific meaning of one rectangular-table column."""

    name: str
    dtype: str
    role: str
    quantity: str = ""
    unit: str = ""
    basis: str = ""
    description: str = ""
    uncertainty_of: str = ""
    uncertainty_meaning: str = ""


@dataclass(frozen=True)
class TableSpec:
    """One frozen rectangular table in a plotting-data bundle."""

    id: str
    path: str
    format: str = "csv"
    columns: tuple[ColumnSpec, ...] = ()
    row_key: tuple[str, ...] = ()
    row_scope: str = ""
    rows: int | None = None
    sha256: str = ""
    bytes: int | None = None


@dataclass(frozen=True)
class SeriesSpec:
    """Backend-neutral mapping from table columns to one plotted series."""

    id: str
    table: str
    x: str
    y: str
    yerr: str = ""
    lower: str = ""
    upper: str = ""
    label: str = ""


@dataclass(frozen=True)
class ViewSpec:
    """Approved data selection for a renderer."""

    id: str
    series: tuple[SeriesSpec, ...]
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    permitted_render_transforms: tuple[str, ...] = (
        "axis_limits",
        "log_scale",
        "panel_layout",
        "legend_position",
    )
    forbidden_render_transforms: tuple[str, ...] = (
        "filter_rows",
        "renormalize",
        "smooth",
        "fit",
    )


@dataclass
class PlotDataset:
    """Scientific analysis packet consumed by plotting/reporting backends."""

    dataset_id: str
    revision: int
    project: str
    chart_family: str
    scientific_question: str
    science_owner: str
    canonical_report: str
    scientific_status: str
    quality: str
    tables: list[TableSpec]
    views: list[ViewSpec]
    sources: list[RunArtifact] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    normalization: str = "none"
    uncertainty: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    supersedes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Plot-dataset metadata cannot contain NaN or infinity.")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    try:
        converted = value.item()
    except AttributeError as exc:
        raise TypeError(f"Plot-dataset value is not JSON serializable: {value!r}") from exc
    return _normalize_json(converted)


def _delimiter(table: TableSpec) -> str:
    if table.format == "csv":
        return ","
    if table.format == "tsv":
        return "\t"
    raise ValueError(f"Unsupported table format {table.format!r}; use csv or tsv.")


def _validate_id(kind: str, value: str) -> None:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{kind} {value!r} must start with an alphanumeric character and "
            "contain only letters, numbers, '.', '_', or '-'."
        )


def _bundle_table_path(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Frozen table path must be relative: {relative_path!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(
            f"Frozen table path escapes the bundle: {relative_path!r}"
        )
    return resolved


def _resolve_reference(reference: str, base: Path) -> str:
    if "://" in reference:
        return reference
    path_text, separator, fragment = reference.partition("#")
    if not path_text:
        return reference
    path = Path(path_text)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    return f"{resolved}{separator}{fragment}" if separator else str(resolved)


def _inspect_table(table: TableSpec, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=_delimiter(table))
        header = list(reader.fieldnames or [])
        if not header:
            raise ValueError(f"Table has no header: {path}")
        declared = [column.name for column in table.columns]
        missing = [name for name in declared if name not in header]
        if missing:
            raise ValueError(f"Table {table.id!r} is missing declared columns: {missing}")
        numeric = {
            column.name
            for column in table.columns
            if column.dtype in NUMERIC_DTYPES
        }
        rows = 0
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            for name in numeric:
                text = str(row.get(name, "")).strip()
                if not text:
                    continue
                try:
                    value = float(text)
                except ValueError as exc:
                    raise ValueError(
                        f"Table {table.id!r} row {row_number} column {name!r} "
                        f"is not numeric: {text!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"Table {table.id!r} row {row_number} column {name!r} "
                        "contains NaN or infinity."
                    )
    if rows == 0:
        raise ValueError(f"Table contains no data rows: {path}")
    stat = path.stat()
    return {
        "columns": header,
        "rows": rows,
        "sha256": _sha256(path),
        "bytes": stat.st_size,
    }


def _validate_model(dataset: PlotDataset, *, root: Path | None = None) -> None:
    if not dataset.dataset_id.strip():
        raise ValueError("dataset_id is required.")
    _validate_id("dataset_id", dataset.dataset_id)
    if dataset.revision < 1:
        raise ValueError("revision must be at least 1.")
    for name, value in (
        ("project", dataset.project),
        ("chart_family", dataset.chart_family),
        ("scientific_question", dataset.scientific_question),
        ("science_owner", dataset.science_owner),
        ("canonical_report", dataset.canonical_report),
    ):
        if not value.strip():
            raise ValueError(f"{name} is required.")
    if dataset.scientific_status not in VALID_STATUSES:
        raise ValueError(
            f"scientific_status must be one of {sorted(VALID_STATUSES)}."
        )
    if dataset.quality not in VALID_QUALITIES:
        raise ValueError(f"quality must be one of {sorted(VALID_QUALITIES)}.")
    if not dataset.tables:
        raise ValueError("At least one table is required.")
    if not dataset.views:
        raise ValueError("At least one approved view is required.")

    table_ids: set[str] = set()
    columns_by_table: dict[str, set[str]] = {}
    for table in dataset.tables:
        if not table.id or table.id in table_ids:
            raise ValueError(f"Table IDs must be non-empty and unique: {table.id!r}")
        _validate_id("Table ID", table.id)
        table_ids.add(table.id)
        if table.format not in {"csv", "tsv"}:
            raise ValueError(f"Unsupported table format: {table.format!r}")
        names: set[str] = set()
        for column in table.columns:
            if not column.name or column.name in names:
                raise ValueError(
                    f"Column names in table {table.id!r} must be non-empty and unique."
                )
            names.add(column.name)
            if column.role not in VALID_COLUMN_ROLES:
                raise ValueError(
                    f"Column {column.name!r} has unsupported role {column.role!r}."
                )
            if (
                column.dtype in NUMERIC_DTYPES
                and column.quantity
                and column.quantity != "dimensionless"
                and not column.unit
            ):
                raise ValueError(
                    f"Numeric dimensional column {column.name!r} requires a unit."
                )
        columns_by_table[table.id] = names
        if root is not None:
            _inspect_table(table, _bundle_table_path(root, table.path))

    view_ids: set[str] = set()
    for view in dataset.views:
        if not view.id or view.id in view_ids:
            raise ValueError(f"View IDs must be non-empty and unique: {view.id!r}")
        _validate_id("View ID", view.id)
        view_ids.add(view.id)
        if not view.series:
            raise ValueError(f"View {view.id!r} contains no series.")
        for series in view.series:
            _validate_id("Series ID", series.id)
            if series.table not in table_ids:
                raise ValueError(
                    f"Series {series.id!r} refers to unknown table {series.table!r}."
                )
            available = columns_by_table[series.table]
            required = [series.x, series.y]
            optional = [series.yerr, series.lower, series.upper]
            missing = [name for name in required + optional if name and name not in available]
            if missing:
                raise ValueError(
                    f"Series {series.id!r} refers to missing columns: {missing}"
                )
            if series.yerr and (series.lower or series.upper):
                raise ValueError(
                    f"Series {series.id!r} cannot mix yerr with lower/upper bounds."
                )

    _normalize_json(asdict(dataset))


def _column_from_dict(payload: dict[str, Any]) -> ColumnSpec:
    return ColumnSpec(
        name=str(payload["name"]),
        dtype=str(payload.get("dtype") or "float"),
        role=str(payload.get("role") or "auxiliary"),
        quantity=str(payload.get("quantity") or ""),
        unit=str(payload.get("unit") or ""),
        basis=str(payload.get("basis") or ""),
        description=str(payload.get("description") or ""),
        uncertainty_of=str(payload.get("uncertainty_of") or ""),
        uncertainty_meaning=str(payload.get("uncertainty_meaning") or ""),
    )


def _table_from_dict(payload: dict[str, Any]) -> TableSpec:
    return TableSpec(
        id=str(payload["id"]),
        path=str(payload["path"]),
        format=str(payload.get("format") or "csv"),
        columns=tuple(
            _column_from_dict(dict(item)) for item in payload.get("columns") or []
        ),
        row_key=tuple(str(item) for item in payload.get("row_key") or []),
        row_scope=str(payload.get("row_scope") or ""),
        rows=int(payload["rows"]) if payload.get("rows") is not None else None,
        sha256=str(payload.get("sha256") or ""),
        bytes=int(payload["bytes"]) if payload.get("bytes") is not None else None,
    )


def _series_from_dict(payload: dict[str, Any]) -> SeriesSpec:
    return SeriesSpec(
        id=str(payload["id"]),
        table=str(payload["table"]),
        x=str(payload["x"]),
        y=str(payload["y"]),
        yerr=str(payload.get("yerr") or ""),
        lower=str(payload.get("lower") or ""),
        upper=str(payload.get("upper") or ""),
        label=str(payload.get("label") or ""),
    )


def _view_from_dict(payload: dict[str, Any]) -> ViewSpec:
    permitted = payload.get("permitted_render_transforms")
    forbidden = payload.get("forbidden_render_transforms")
    return ViewSpec(
        id=str(payload["id"]),
        series=tuple(
            _series_from_dict(dict(item)) for item in payload.get("series") or []
        ),
        xlabel=str(payload.get("xlabel") or ""),
        ylabel=str(payload.get("ylabel") or ""),
        title=str(payload.get("title") or ""),
        permitted_render_transforms=tuple(
            str(item)
            for item in (
                permitted
                if permitted is not None
                else ViewSpec.__dataclass_fields__["permitted_render_transforms"].default
            )
        ),
        forbidden_render_transforms=tuple(
            str(item)
            for item in (
                forbidden
                if forbidden is not None
                else ViewSpec.__dataclass_fields__["forbidden_render_transforms"].default
            )
        ),
    )


def dataset_from_dict(payload: dict[str, Any]) -> PlotDataset:
    """Create a validated model from a spec or manifest payload."""

    body = dict(payload.get("dataset") or payload)
    science = dict(payload.get("science") or body.get("science") or {})
    sources = [
        RunArtifact(**dict(item))
        for item in (payload.get("sources") or body.get("sources") or [])
    ]
    dataset = PlotDataset(
        dataset_id=str(body["dataset_id"]),
        revision=int(body.get("revision") or 1),
        project=str(body["project"]),
        chart_family=str(body["chart_family"]),
        scientific_question=str(body["scientific_question"]),
        science_owner=str(body["science_owner"]),
        canonical_report=str(body["canonical_report"]),
        scientific_status=str(body["scientific_status"]),
        quality=str(body["quality"]),
        tables=[
            _table_from_dict(dict(item))
            for item in (payload.get("tables") or body.get("tables") or [])
        ],
        views=[
            _view_from_dict(dict(item))
            for item in (payload.get("views") or body.get("views") or [])
        ],
        sources=sources,
        transformations=[
            str(item)
            for item in (
                science.get("transformations")
                or body.get("transformations")
                or []
            )
        ],
        normalization=str(
            science.get("normalization") or body.get("normalization") or "none"
        ),
        uncertainty=dict(
            science.get("uncertainty") or body.get("uncertainty") or {}
        ),
        caveats=[
            str(item)
            for item in (science.get("caveats") or body.get("caveats") or [])
        ],
        supersedes=str(body.get("supersedes") or ""),
        metadata=dict(payload.get("metadata") or body.get("metadata") or {}),
    )
    _validate_model(dataset)
    return dataset


def _manifest_payload(
    dataset: PlotDataset,
    *,
    tables: list[TableSpec],
    sources: list[RunArtifact],
) -> dict[str, Any]:
    body = {
        "dataset_id": dataset.dataset_id,
        "revision": dataset.revision,
        "project": dataset.project,
        "chart_family": dataset.chart_family,
        "scientific_question": dataset.scientific_question,
        "science_owner": dataset.science_owner,
        "canonical_report": dataset.canonical_report,
        "scientific_status": dataset.scientific_status,
        "quality": dataset.quality,
        "supersedes": dataset.supersedes,
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": _utc_now(),
        "dataset": body,
        "sources": [asdict(item) for item in sources],
        "tables": [asdict(item) for item in tables],
        "views": [asdict(item) for item in dataset.views],
        "science": {
            "transformations": dataset.transformations,
            "normalization": dataset.normalization,
            "uncertainty": dataset.uncertainty,
            "caveats": dataset.caveats,
        },
        "metadata": dataset.metadata,
    }
    normalized = _normalize_json(payload)
    normalized["integrity"] = {
        "manifest_content_sha256": _canonical_sha256(normalized)
    }
    return normalized


def freeze_dataset(dataset: PlotDataset, outdir: Path) -> Path:
    """Copy curated tables into a new immutable bundle and write its manifest."""

    _validate_model(dataset)
    outdir = outdir.resolve()
    if outdir.exists():
        raise FileExistsError(
            f"Plot-data bundle already exists: {outdir}. "
            "Use a new revision directory instead of overwriting it."
        )
    outdir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{outdir.name}.", dir=str(outdir.parent))
    )
    try:
        table_dir = stage / "tables"
        table_dir.mkdir()
        frozen_tables: list[TableSpec] = []
        for table in dataset.tables:
            source = Path(table.path).resolve()
            details = _inspect_table(table, source)
            suffix = ".csv" if table.format == "csv" else ".tsv"
            target = table_dir / f"{table.id}{suffix}"
            shutil.copy2(source, target)
            frozen_tables.append(
                TableSpec(
                    id=table.id,
                    path=str(target.relative_to(stage)),
                    format=table.format,
                    columns=table.columns,
                    row_key=table.row_key,
                    row_scope=table.row_scope,
                    rows=int(details["rows"]),
                    sha256=str(details["sha256"]),
                    bytes=int(details["bytes"]),
                )
            )
        frozen_sources: list[RunArtifact] = []
        for artifact in dataset.sources:
            path = Path(artifact.path).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            frozen_sources.append(
                RunArtifact(
                    path=str(path),
                    role=artifact.role,
                    exists=True,
                    sha256=_sha256(path) if path.is_file() else None,
                    metadata=_normalize_json(artifact.metadata),
                )
            )
        payload = _manifest_payload(
            dataset, tables=frozen_tables, sources=frozen_sources
        )
        manifest = stage / MANIFEST_NAME
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.rename(outdir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return outdir / MANIFEST_NAME


def _resolve_spec_paths(dataset: PlotDataset, base: Path) -> PlotDataset:
    tables = [
        TableSpec(
            id=table.id,
            path=str(
                (base / table.path).resolve()
                if not Path(table.path).is_absolute()
                else Path(table.path).resolve()
            ),
            format=table.format,
            columns=table.columns,
            row_key=table.row_key,
            row_scope=table.row_scope,
        )
        for table in dataset.tables
    ]
    sources = [
        RunArtifact(
            path=str(
                (base / artifact.path).resolve()
                if not Path(artifact.path).is_absolute()
                else Path(artifact.path).resolve()
            ),
            role=artifact.role,
            exists=artifact.exists,
            sha256=artifact.sha256,
            metadata=artifact.metadata,
        )
        for artifact in dataset.sources
    ]
    return PlotDataset(
        **{
            **asdict(dataset),
            "canonical_report": _resolve_reference(
                dataset.canonical_report, base
            ),
            "tables": tables,
            "views": dataset.views,
            "sources": sources,
        }
    )


def freeze_from_spec(spec_path: Path, outdir: Path) -> Path:
    """Freeze a JSON specification into a self-contained plotting bundle."""

    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    dataset = _resolve_spec_paths(dataset_from_dict(payload), spec_path.parent)
    return freeze_dataset(dataset, outdir)


def load_manifest(manifest_path: Path, *, verify: bool = True) -> dict[str, Any]:
    """Load a bundle manifest and optionally verify its tables and digest."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(
            f"Unsupported plot-dataset schema {payload.get('schema')!r}; "
            f"expected {SCHEMA!r}."
        )
    dataset = dataset_from_dict(payload)
    if verify:
        report = validate_bundle(manifest_path)
        if not report["valid"]:
            raise ValueError("; ".join(report["errors"]))
    _validate_model(dataset)
    return payload


def validate_bundle(manifest_path: Path) -> dict[str, Any]:
    """Return an integrity and schema validation report."""

    errors: list[str] = []
    manifest_path = manifest_path.resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"schema": SCHEMA, "valid": False, "errors": [str(exc)]}
    if payload.get("schema") != SCHEMA:
        errors.append(f"Unexpected schema: {payload.get('schema')!r}")
    expected_manifest = str(
        dict(payload.get("integrity") or {}).get("manifest_content_sha256") or ""
    )
    body = dict(payload)
    body.pop("integrity", None)
    if expected_manifest and _canonical_sha256(body) != expected_manifest:
        errors.append("Manifest content fingerprint does not match.")
    try:
        dataset = dataset_from_dict(payload)
        root = manifest_path.parent
        for table in dataset.tables:
            path = _bundle_table_path(root, table.path)
            details = _inspect_table(table, path)
            if table.sha256 and details["sha256"] != table.sha256:
                errors.append(f"Table fingerprint mismatch: {table.id}")
            if table.rows is not None and details["rows"] != table.rows:
                errors.append(f"Table row-count mismatch: {table.id}")
        _validate_model(dataset)
    except Exception as exc:
        errors.append(str(exc))
    return {
        "schema": SCHEMA,
        "manifest": str(manifest_path),
        "valid": not errors,
        "errors": errors,
    }


def resolve_view(manifest_path: Path, view_id: str = "") -> dict[str, Any]:
    """Resolve one approved view to absolute table paths and column mappings."""

    payload = load_manifest(manifest_path, verify=True)
    dataset = dataset_from_dict(payload)
    view = next(
        (item for item in dataset.views if item.id == view_id),
        dataset.views[0] if not view_id else None,
    )
    if view is None:
        available = ", ".join(item.id for item in dataset.views)
        raise KeyError(f"Unknown view {view_id!r}; available: {available}")
    tables = {table.id: table for table in dataset.tables}
    root = manifest_path.resolve().parent
    return {
        "schema": SCHEMA,
        "dataset_id": dataset.dataset_id,
        "revision": dataset.revision,
        "scientific_status": dataset.scientific_status,
        "quality": dataset.quality,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path.resolve()),
        "view": asdict(view),
        "series": [
            {
                **asdict(series),
                "table_path": str(
                    _bundle_table_path(root, tables[series.table].path)
                ),
            }
            for series in view.series
        ],
        "science": {
            "transformations": dataset.transformations,
            "normalization": dataset.normalization,
            "uncertainty": dataset.uncertainty,
            "caveats": dataset.caveats,
        },
    }


def _template_payload() -> dict[str, Any]:
    return {
        "dataset": {
            "dataset_id": "project-analysis-name",
            "revision": 1,
            "project": "Project name",
            "chart_family": "thermochemistry",
            "scientific_question": "Question answered by the curated table",
            "science_owner": "Project student or domain lead",
            "canonical_report": "/absolute/project/report.md#section",
            "scientific_status": "provisional",
            "quality": "production",
        },
        "sources": [
            {
                "path": "/absolute/read-only/source.out",
                "role": "raw-computation",
            }
        ],
        "tables": [
            {
                "id": "main",
                "path": "curated_analysis.csv",
                "format": "csv",
                "row_key": ["temperature_K"],
                "row_scope": "Rows passing the recorded physical guards.",
                "columns": [
                    {
                        "name": "temperature_K",
                        "dtype": "float",
                        "role": "x",
                        "quantity": "temperature",
                        "unit": "K",
                    },
                    {
                        "name": "entropy",
                        "dtype": "float",
                        "role": "y",
                        "quantity": "entropy",
                        "unit": "J mol-1 K-1",
                        "basis": "formula_unit",
                    },
                ],
            }
        ],
        "views": [
            {
                "id": "main",
                "series": [
                    {
                        "id": "entropy",
                        "table": "main",
                        "x": "temperature_K",
                        "y": "entropy",
                        "label": "Entropy",
                    }
                ],
                "xlabel": "Temperature (K)",
                "ylabel": "Entropy (J mol-1 K-1)",
            }
        ],
        "science": {
            "transformations": ["Describe ordered scientific transformations."],
            "normalization": "none",
            "uncertainty": {"meaning": "none"},
            "caveats": ["Record unresolved scientific limitations."],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and validate reusable scientific plotting datasets."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template", help="Write an editable JSON spec.")
    template.add_argument("--write", type=Path, required=True)
    freeze = sub.add_parser("freeze", help="Freeze a spec and its curated tables.")
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--outdir", type=Path, required=True)
    validate = sub.add_parser("validate", help="Verify a frozen bundle.")
    validate.add_argument("manifest", type=Path)
    describe = sub.add_parser("describe", help="Print bundle metadata and views.")
    describe.add_argument("manifest", type=Path)
    resolve = sub.add_parser("resolve", help="Resolve an approved plotting view.")
    resolve.add_argument("manifest", type=Path)
    resolve.add_argument("--view", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "template":
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(_template_payload(), indent=2) + "\n", encoding="utf-8"
        )
        print(args.write)
        return 0
    if args.command == "freeze":
        print(freeze_from_spec(args.spec, args.outdir))
        return 0
    if args.command == "validate":
        report = validate_bundle(args.manifest)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 1
    if args.command == "describe":
        payload = load_manifest(args.manifest)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "resolve":
        print(json.dumps(resolve_view(args.manifest, args.view), indent=2))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
