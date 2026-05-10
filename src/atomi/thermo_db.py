"""Small thermodynamic database helpers."""

from __future__ import annotations

import html
import math
import re
from datetime import datetime, timezone
from urllib.request import urlopen


JAEA_BASE_URL = "https://thermodb.jaea.go.jp/data/en/td"
_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")
_TAG_RE = re.compile(r"<[^>]+>")


def _numbers_from_text(text: str) -> list[float]:
    values = []
    for match in _FLOAT_RE.findall(text):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def _strip_tags(text: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", text))


def fetch_text(url: str, timeout: float = 20.0) -> str:
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_jaea_table(text: str) -> list[dict[str, float]]:
    rows = []
    plain = _strip_tags(text)
    for line in plain.splitlines():
        values = _numbers_from_text(line)
        if len(values) < 5:
            continue
        temp, cp, enthalpy, entropy, gibbs = values[:5]
        if temp <= 0.0 or not all(math.isfinite(value) for value in values[:5]):
            continue
        rows.append(
            {
                "T_K": temp,
                "Cp_J_mol_K": cp,
                "H_J_mol": enthalpy,
                "S_J_mol_K": entropy,
                "G_J_mol": gibbs,
            }
        )
    rows.sort(key=lambda row: row["T_K"])
    return rows


def interpolate_rows(rows: list[dict[str, float]], temperature: float) -> dict[str, float]:
    if not rows:
        raise ValueError("No thermodynamic rows were parsed")
    temperature = float(temperature)
    if temperature < rows[0]["T_K"] or temperature > rows[-1]["T_K"]:
        raise ValueError(
            f"Requested temperature {temperature:g} K is outside database range "
            f"{rows[0]['T_K']:g}--{rows[-1]['T_K']:g} K"
        )
    for row in rows:
        if abs(row["T_K"] - temperature) <= 1.0e-8:
            return dict(row)
    for lower, upper in zip(rows, rows[1:]):
        if lower["T_K"] <= temperature <= upper["T_K"]:
            span = upper["T_K"] - lower["T_K"]
            if span == 0.0:
                return dict(lower)
            frac = (temperature - lower["T_K"]) / span
            out = {"T_K": temperature}
            for key in ("Cp_J_mol_K", "H_J_mol", "S_J_mol_K", "G_J_mol"):
                out[key] = lower[key] + frac * (upper[key] - lower[key])
            return out
    raise ValueError(f"Could not interpolate database value at {temperature:g} K")


def jaea_formula_url(formula: str) -> str:
    return f"{JAEA_BASE_URL}/{formula}.html"


def jaea_anchor(
    formula: str,
    temperature: float,
    *,
    phase: str = "solid",
    fetcher=fetch_text,
) -> dict:
    url = jaea_formula_url(formula)
    text = fetcher(url)
    rows = parse_jaea_table(text)
    value = interpolate_rows(rows, temperature)
    return {
        "database": "jaea",
        "formula": formula,
        "phase": phase,
        "url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "temperature_requested_K": float(temperature),
        "temperature_value_K": value["T_K"],
        "Cp_J_mol_formula_K": value["Cp_J_mol_K"],
        "H_J_mol_formula": value["H_J_mol"],
        "S_J_mol_formula_K": value["S_J_mol_K"],
        "G_J_mol_formula": value["G_J_mol"],
        "available_temperature_min_K": rows[0]["T_K"],
        "available_temperature_max_K": rows[-1]["T_K"],
        "note": "JAEA values are interpreted as per mole of formula unit.",
    }
