"""Shared helpers for lightweight zentropy bridge stages."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


K_B_EV_PER_K = 8.617333262145e-5
EV_PER_KJ_MOL = 1.0 / 96.48533212331002


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.12g}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if isinstance(payload, dict):
            records = payload.get("records") or payload.get("rows") or payload.get("data")
            if isinstance(records, list):
                return [dict(row) for row in records if isinstance(row, dict)]
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, dict)]
        raise ValueError(f"JSON table {path} does not contain a list or records/rows/data list.")
    return [dict(row) for row in read_csv(path)]


def load_motif_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if isinstance(payload, dict):
            records = payload.get("records")
            if not isinstance(records, list):
                raise ValueError(f"Motif DB {path} does not contain a records list.")
            return payload, [dict(row) for row in records if isinstance(row, dict)]
        if isinstance(payload, list):
            return {"schema": "atomi.zentropy.motif_list.v1"}, [dict(row) for row in payload if isinstance(row, dict)]
        raise ValueError(f"Unsupported motif JSON payload in {path}.")
    rows = [dict(row) for row in read_csv(path)]
    return {"schema": "atomi.zentropy.motif_csv.v1"}, rows


def nested_get(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        current: Any = row
        ok = True
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok and current not in (None, ""):
            return current
    return None


def motif_id(row: dict[str, Any], fallback: str = "motif") -> str:
    return str(nested_get(row, "motif_id", "id", "name", "case_name") or fallback)


def motif_family(row: dict[str, Any]) -> str:
    return str(nested_get(row, "motif_family", "family", "motif_type", "defect_label") or "")


def formula_units(row: dict[str, Any]) -> float | None:
    return finite_float(
        nested_get(
            row,
            "formula_units",
            "size_normalization.formula_units",
            "normalization.formula_units",
        )
    )


def energy_per_fu(row: dict[str, Any]) -> float | None:
    direct = finite_float(
        nested_get(
            row,
            "energy_per_formula_unit_eV",
            "E_eV_per_formula_unit",
            "energy_eV_per_fu",
            "G_eV_per_fu",
        )
    )
    if direct is not None:
        return direct
    energy = finite_float(nested_get(row, "energy_eV", "E_eV"))
    fu = formula_units(row)
    if energy is not None and fu not in (None, 0.0):
        return energy / fu
    return None


def volume_per_fu(row: dict[str, Any]) -> float | None:
    direct = finite_float(
        nested_get(
            row,
            "volume_per_formula_unit_A3",
            "volume_A3_per_fu",
            "V_A3_per_fu",
        )
    )
    if direct is not None:
        return direct
    volume = finite_float(nested_get(row, "volume_A3", "V_A3"))
    fu = formula_units(row)
    if volume is not None and fu not in (None, 0.0):
        return volume / fu
    return None


def composition_label(row: dict[str, Any]) -> str:
    norm = row.get("size_normalization") if isinstance(row.get("size_normalization"), dict) else {}
    guest = finite_float(nested_get(row, "guest_cation_fraction", "x_guest"))
    if guest is None and isinstance(norm, dict):
        guest = finite_float(norm.get("guest_cation_fraction"))
    delta = finite_float(nested_get(row, "oxygen_delta_per_formula_unit", "oxygen_delta", "delta_O"))
    if delta is None and isinstance(norm, dict):
        delta = finite_float(norm.get("oxygen_delta_per_formula_unit"))
    if guest is not None or delta is not None:
        return f"x={guest if guest is not None else 0:g};delta={delta if delta is not None else 0:g}"
    return str(nested_get(row, "composition", "formula") or "")


def parse_temperature_values(values: list[str] | None) -> list[float]:
    if not values:
        return [300.0]
    temps: list[float] = []
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if ":" in text:
            pieces = [float(part) for part in text.split(":")]
            if len(pieces) != 3:
                raise ValueError("Temperature ranges must use start:stop:step.")
            start, stop, step = pieces
            if step == 0:
                raise ValueError("Temperature step cannot be zero.")
            current = start
            if step > 0:
                while current <= stop + abs(step) * 1.0e-9:
                    temps.append(round(current, 10))
                    current += step
            else:
                while current >= stop - abs(step) * 1.0e-9:
                    temps.append(round(current, 10))
                    current += step
        else:
            temps.append(float(text))
    unique = sorted({float(temp) for temp in temps})
    if not unique:
        raise ValueError("No temperatures were parsed.")
    return unique


def parse_float_values(values: list[str] | None, *, default: list[float]) -> list[float]:
    if not values:
        return list(default)
    out: list[float] = []
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if ":" in text:
            out.extend(parse_temperature_values([text]))
        else:
            out.append(float(text))
    return sorted({float(value) for value in out})


def value_eV(row: dict[str, Any], *eV_keys: str, kj_mol_keys: tuple[str, ...] = ()) -> float | None:
    for key in eV_keys:
        value = finite_float(nested_get(row, key))
        if value is not None:
            return value
    for key in kj_mol_keys:
        value = finite_float(nested_get(row, key))
        if value is not None:
            return value * EV_PER_KJ_MOL
    return None
