"""Materials Project cache bridge for thermo-prior workflows.

The thermo-prior layer should not depend on live API calls at fit time.  This
module normalizes Materials Project summary/thermo documents into a compact
JSON cache that can be versioned in project folders and reused by downstream
line-compound, GNN, and CALPHAD-prior steps.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "atomi.thermo_prior.materials_project_cache.v1"

DEFAULT_FIELDS = [
    "material_id",
    "formula_pretty",
    "formula_anonymous",
    "chemsys",
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "volume",
    "nsites",
    "nelements",
    "is_stable",
    "theoretical",
]


@dataclass
class MPCacheSummary:
    """Small summary for normalized Materials Project cache files."""

    schema: str = SCHEMA
    path: str = ""
    n_records: int = 0
    n_stable: int = 0
    formulas: list[str] = field(default_factory=list)
    chemsys: list[str] = field(default_factory=list)
    min_energy_above_hull_eV: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get(doc: Any, key: str, default: Any = None) -> Any:
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


def _as_dict(doc: Any) -> dict[str, Any]:
    if isinstance(doc, dict):
        return dict(doc)
    if hasattr(doc, "model_dump"):
        return dict(doc.model_dump())
    if hasattr(doc, "dict"):
        return dict(doc.dict())
    return {key: getattr(doc, key) for key in dir(doc) if not key.startswith("_")}


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    number = _finite_float(value)
    if number is not None:
        return number
    return str(value)


def normalize_mp_doc(doc: Any) -> dict[str, Any]:
    """Normalize one MP summary/thermo document into Atomi cache fields."""

    material_id = _get(doc, "material_id") or _get(doc, "task_id") or _get(doc, "mpid")
    if material_id is not None:
        material_id = str(material_id)
    energy_above_hull = _finite_float(
        _get(doc, "energy_above_hull", _get(doc, "e_above_hull"))
    )
    formation = _finite_float(
        _get(doc, "formation_energy_per_atom", _get(doc, "formation_energy_per_atom_eV"))
    )
    energy = _finite_float(_get(doc, "energy_per_atom", _get(doc, "energy_per_atom_eV")))
    volume = _finite_float(_get(doc, "volume", _get(doc, "volume_A3")))
    nsites = _get(doc, "nsites")
    try:
        nsites_value = int(nsites) if nsites is not None else None
    except (TypeError, ValueError):
        nsites_value = None

    normalized = {
        "schema": SCHEMA,
        "source": "materials_project",
        "database_id": material_id,
        "material_id": material_id,
        "formula_pretty": _get(doc, "formula_pretty") or _get(doc, "pretty_formula"),
        "formula_anonymous": _get(doc, "formula_anonymous") or _get(doc, "anonymous_formula"),
        "chemsys": _get(doc, "chemsys"),
        "energy_per_atom_eV": energy,
        "formation_energy_per_atom_eV": formation,
        "energy_above_hull_eV": energy_above_hull,
        "band_gap_eV": _finite_float(_get(doc, "band_gap", _get(doc, "band_gap_eV"))),
        "volume_A3": volume,
        "nsites": nsites_value,
        "volume_per_atom_A3": (volume / nsites_value) if volume is not None and nsites_value else None,
        "nelements": _get(doc, "nelements"),
        "is_stable": bool(_get(doc, "is_stable")) if _get(doc, "is_stable") is not None else None,
        "theoretical": _get(doc, "theoretical"),
        "thermo_type": _get(doc, "thermo_type"),
        "raw_keys": sorted(str(key) for key in _as_dict(doc).keys()),
    }
    return _jsonable(normalized)


def normalize_mp_docs(docs: list[Any]) -> list[dict[str, Any]]:
    return [normalize_mp_doc(doc) for doc in docs]


def _load_raw_docs(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "data", "docs", "materials"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise ValueError(f"Unsupported MP raw JSON payload in {path}")


def _load_api_key_json(path: Path, provider: str, env_name: str) -> tuple[str | None, str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = [
        payload.get(env_name),
        payload.get("materials_project_api_key") if provider in {"materials_project", "materials-project"} else None,
    ]
    for provider_key in {provider, provider.replace("-", "_")}:
        nested = payload.get(provider_key)
        if isinstance(nested, dict):
            candidates.extend(
                [
                    nested.get("api_key"),
                    nested.get(env_name),
                    nested.get("materials_project_api_key"),
                ]
            )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip(), str(path)
    return None, str(path)


def resolve_mp_api_key(*, api_key_env: str = "MP_API_KEY", api_key_json: Path | None = None) -> tuple[str | None, str]:
    """Resolve MP API key with the same local KIT convention used elsewhere."""

    if api_key_env:
        value = os.environ.get(api_key_env)
        if value:
            return value, f"env:{api_key_env}"
    candidates: list[Path] = []
    if api_key_json is not None:
        candidates.append(api_key_json)
    if os.environ.get("ATOMI_API_KEYS_JSON"):
        candidates.append(Path(os.environ["ATOMI_API_KEYS_JSON"]))
    candidates.extend(
        [
            Path.home() / "atomi_hpc/atomi_hpc_config.kit.local.json",
            Path.home() / "hpc_atomi/atomi_hpc_config.kit.local.json",
        ]
    )
    seen: set[Path] = set()
    for path in candidates:
        path = path.expanduser()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        key, source = _load_api_key_json(path, "materials_project", api_key_env)
        if key:
            return key, f"json:{source}"
    return None, "none"


def write_mp_cache(
    output: Path,
    records: list[dict[str, Any]],
    *,
    query: dict[str, Any] | None = None,
) -> MPCacheSummary:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "source": "materials_project",
        "query": query or {},
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summarize_mp_cache(output)


def summarize_mp_cache(path: Path) -> MPCacheSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.get("records") or [])
    hull_values = [
        value
        for value in (_finite_float(record.get("energy_above_hull_eV")) for record in records)
        if value is not None
    ]
    stable = [
        record
        for record in records
        if record.get("is_stable") is True
        or (_finite_float(record.get("energy_above_hull_eV")) is not None and abs(float(record["energy_above_hull_eV"])) < 1.0e-8)
    ]
    formulas = sorted({str(record.get("formula_pretty")) for record in records if record.get("formula_pretty")})
    chemsys = sorted({str(record.get("chemsys")) for record in records if record.get("chemsys")})
    return MPCacheSummary(
        path=str(path),
        n_records=len(records),
        n_stable=len(stable),
        formulas=formulas,
        chemsys=chemsys,
        min_energy_above_hull_eV=min(hull_values) if hull_values else None,
    )


def fetch_mp_summary_docs(
    *,
    formulas: list[str] | None = None,
    material_ids: list[str] | None = None,
    api_key: str | None = None,
    fields: list[str] | None = None,
) -> list[Any]:
    """Fetch MP summary docs when mp-api and credentials are available."""

    try:
        from mp_api.client import MPRester
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("mp-api is required for live Materials Project fetches.") from exc

    docs: list[Any] = []
    with MPRester(api_key) as mpr:
        if material_ids:
            docs.extend(mpr.materials.summary.search(material_ids=material_ids, fields=fields or DEFAULT_FIELDS))
        for formula in formulas or []:
            docs.extend(mpr.materials.summary.search(formula=formula, fields=fields or DEFAULT_FIELDS))
    return docs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermo-prior-mp",
        description="Build and inspect offline Materials Project caches for thermo-prior workflows.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    from_json = sub.add_parser("from-json", help="Normalize a saved MP API JSON response.")
    from_json.add_argument("--input", type=Path, required=True)
    from_json.add_argument("--out", type=Path, required=True)

    fetch = sub.add_parser("fetch", help="Fetch MP summary docs and write a normalized cache.")
    fetch.add_argument("--formula", action="append", default=[])
    fetch.add_argument("--material-id", action="append", default=[])
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--api-key-env", default="MP_API_KEY")
    fetch.add_argument("--api-key-json", type=Path)
    fetch.add_argument("--field", action="append", default=[])

    summarize = sub.add_parser("summarize", help="Summarize a normalized MP cache.")
    summarize.add_argument("cache", type=Path)

    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.command == "from-json":
        raw_docs = _load_raw_docs(args.input)
        summary = write_mp_cache(
            args.out,
            normalize_mp_docs(raw_docs),
            query={"offline_input": str(args.input)},
        )
    elif args.command == "fetch":
        api_key, api_key_source = resolve_mp_api_key(
            api_key_env=args.api_key_env,
            api_key_json=args.api_key_json,
        )
        docs = fetch_mp_summary_docs(
            formulas=list(args.formula or []),
            material_ids=list(args.material_id or []),
            api_key=api_key,
            fields=list(args.field or []) or DEFAULT_FIELDS,
        )
        summary = write_mp_cache(
            args.out,
            normalize_mp_docs(docs),
            query={
                "formulas": args.formula,
                "material_ids": args.material_id,
                "api_key_source": api_key_source,
            },
        )
    elif args.command == "summarize":
        summary = summarize_mp_cache(args.cache)
    else:  # pragma: no cover - argparse enforces this.
        raise ValueError(f"Unsupported command: {args.command}")

    payload = summary.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
