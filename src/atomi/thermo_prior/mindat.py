"""Small OpenMindat / Mindat API bridge for Atomi.

The Mindat API is token-gated. This module deliberately keeps endpoint paths
configurable because Mindat endpoint names have changed across public examples.
It provides safe status, raw get, and search helpers that write provenance-rich
offline JSON caches for later ThermoFun/GEMS/thermo-prior use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


DEFAULT_BASE_URL = "https://api.mindat.org/v1"
DEFAULT_TIMEOUT_S = 30.0


def _candidate_hpc_config_paths() -> list[Path]:
    paths: list[Path] = []
    for env_name in ("ATOMI_HPC_CONFIG", "ATOMI_API_KEYS_JSON"):
        if value := os.environ.get(env_name):
            paths.append(Path(value).expanduser())
    paths.extend(
        [
            Path.home() / "atomi_hpc" / "atomi_hpc_config.kit.local.json",
            Path.home() / "hpc_atomi" / "atomi_hpc_config.kit.local.json",
        ]
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _kit_value(*key_paths: tuple[str, ...]) -> str | None:
    for path in _candidate_hpc_config_paths():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for keys in key_paths:
            value: Any = data
            for key in keys:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            if value:
                return str(value)
    return None


def _clean_token(value: str | None) -> str | None:
    if not value:
        return None
    token = value.strip()
    if not token:
        return None
    placeholders = ("SET_", "REPLACE_", "YOUR_", "PASTE_", "<", "${")
    if token.upper().startswith(placeholders):
        return None
    return token


def _token_default() -> str | None:
    return _clean_token(
        os.environ.get("MINDAT_API_TOKEN")
        or os.environ.get("OPENMINDAT_API_TOKEN")
        or _kit_value(
            ("mindat", "api_token"),
            ("mindat", "token"),
            ("api_keys", "mindat"),
            ("environment_exports", "MINDAT_API_TOKEN"),
        )
    )


def _base_url_default() -> str:
    return (
        os.environ.get("MINDAT_BASE_URL")
        or _kit_value(("mindat", "base_url"), ("environment_exports", "MINDAT_BASE_URL"))
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _redact_token(token: str | None) -> str | None:
    if not token:
        return None
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


@dataclass(frozen=True)
class MindatConfig:
    base_url: str = DEFAULT_BASE_URL
    token: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def safe_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["token"] = _redact_token(self.token)
        return data


class MindatClient:
    """Minimal requests-based Mindat API client."""

    def __init__(self, config: MindatConfig):
        self.config = config

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "atomi-mindat-bridge/0.9",
        }
        if self.config.token:
            headers["Authorization"] = f"Token {self.config.token}"
        return headers

    def url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.config.base_url}/{path.lstrip('/')}"

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            response = requests.get(
                self.url(path),
                params=clean_params,
                headers=self._headers(),
                timeout=self.config.timeout_s,
            )
        except requests.RequestException as exc:
            url = self.url(path)
            if clean_params:
                url = f"{url}?{urlencode(clean_params)}"
            return {
                "schema": "atomi.mindat.response.v1",
                "request": {
                    "url": url,
                    "path": path,
                    "params": params or {},
                    "base_url": self.config.base_url,
                    "token_present": bool(self.config.token),
                },
                "response": {
                    "status_code": None,
                    "ok": False,
                    "reason": None,
                    "content_type": None,
                    "error": repr(exc),
                },
                "data": None,
            }
        payload: Any
        try:
            payload = response.json()
        except Exception:
            payload = {"text": response.text[:4000]}
        return {
            "schema": "atomi.mindat.response.v1",
            "request": {
                "url": response.url,
                "path": path,
                "params": params or {},
                "base_url": self.config.base_url,
                "token_present": bool(self.config.token),
            },
            "response": {
                "status_code": response.status_code,
                "ok": response.ok,
                "reason": response.reason,
                "content_type": response.headers.get("content-type"),
            },
            "data": payload,
        }

    def status(self) -> dict[str, Any]:
        root_url = self.config.base_url.rsplit("/", 1)[0] if self.config.base_url.endswith("/v1") else self.config.base_url
        probes = []
        for path in ("", "localities", "geomaterials", "minerals"):
            target = root_url if path == "" else path
            probes.append(self.get(target))
        return {
            "schema": "atomi.mindat.status.v1",
            "config": self.config.safe_dict,
            "token_present": bool(self.config.token),
            "probes": probes,
            "notes": [
                "OpenMindat/Mindat API access is token-gated; 401 means configure MINDAT_API_TOKEN or mindat.api_token in the private KIT JSON.",
                "Use `mindat-get --path ...` for exact endpoint paths from your Mindat API account/docs.",
            ],
        }


def _write_or_print(payload: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote Mindat JSON: {out}")
    else:
        print(text)


def _client_from_args(args: argparse.Namespace) -> MindatClient:
    token = args.token if getattr(args, "token", None) else _token_default()
    base_url = getattr(args, "base_url", None) or _base_url_default()
    timeout = getattr(args, "timeout_s", DEFAULT_TIMEOUT_S)
    return MindatClient(MindatConfig(base_url=base_url.rstrip("/"), token=token, timeout_s=timeout))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mindat-bridge",
        description="Token-aware OpenMindat/Mindat API bridge for Atomi.",
    )
    parser.add_argument("--base-url", default=_base_url_default(), help=f"Mindat API base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--token", default=None, help="Mindat API token. Prefer MINDAT_API_TOKEN or private KIT JSON.")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Probe Mindat API reachability and authentication state.")
    status.add_argument("--out", type=Path)

    get = sub.add_parser("get", help="Fetch a raw Mindat API path and write/print JSON.")
    get.add_argument("--path", required=True, help="Endpoint path, for example /localities or /geomaterials.")
    get.add_argument("--param", action="append", default=[], help="Query parameter as key=value; repeatable.")
    get.add_argument("--out", type=Path)

    search = sub.add_parser("search", help="Convenience query wrapper for a configurable search endpoint.")
    search.add_argument("query")
    search.add_argument("--endpoint", default="geomaterials", help="Search endpoint path. Default: geomaterials.")
    search.add_argument("--query-param", default="q", help="Name of the text query parameter. Default: q.")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--out", type=Path)
    return parser


def _parse_params(values: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--param must be key=value, got {value!r}")
        key, raw = value.split("=", 1)
        params[key] = raw
    return params


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    client = _client_from_args(args)
    if args.command == "status":
        payload = client.status()
        _write_or_print(payload, args.out)
        return payload
    if args.command == "get":
        payload = client.get(args.path, params=_parse_params(args.param))
        _write_or_print(payload, args.out)
        if payload["response"].get("status_code") == 401:
            print("Mindat returned 401 Unauthorized; configure MINDAT_API_TOKEN or private KIT JSON.", file=sys.stderr)
        return payload
    if args.command == "search":
        params = {args.query_param: args.query, "limit": args.limit}
        payload = client.get(args.endpoint, params=params)
        payload["search"] = {
            "query": args.query,
            "endpoint": args.endpoint,
            "query_string": urlencode(params),
        }
        _write_or_print(payload, args.out)
        return payload
    return None


def console_main(argv: list[str] | None = None) -> None:
    main(argv)


if __name__ == "__main__":
    console_main()
