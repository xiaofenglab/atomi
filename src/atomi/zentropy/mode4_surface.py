"""Mode4 skeleton for sparse defect motifs -> dense Gibbs surfaces.

This module is intentionally lightweight: it defines the data contracts and a
small regularized motif/cluster Hamiltonian that can be replaced by smol/CASM
or a richer Atomi mode4 backend later.  The important guarantee is that dense
surfaces are produced from fitted interaction parameters and carry uncertainty
metadata, instead of reusing composition-specific POCC degeneracies outside
their finite composition cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from atomi.zentropy.backends.base import CETrainingRecord, CETrainingSet, read_ce_training_jsonl, write_ce_training_jsonl

SCHEMA = "atomi.zentropy.mode4_surface.v1"
MODEL_SCHEMA = "atomi.zentropy.mode4_surface.linear_motif_hamiltonian.v1"
K_B_EV_PER_K = 8.617333262145e-5
EV_TO_KJ_MOL = 96.4853321233

DEFAULT_FEATURES = [
    "x_Gd",
    "delta_VO",
    "h_U5",
    "x_Gd:delta_VO",
    "x_Gd:h_U5",
    "delta_VO:h_U5",
    "x_Gd^2",
    "delta_VO^2",
    "h_U5^2",
]

TRAINING_FIELDS = [
    "record_id",
    "source",
    "structure_path",
    "energy_eV",
    "uncertainty_eV",
    "x_Gd",
    "delta_VO",
    "h_U5",
    "motif_features_json",
    "species_counts_json",
    "metadata_json",
]

SURFACE_FIELDS = [
    "phase",
    "backend",
    "T_K",
    "x_Gd",
    "delta_VO",
    "h_U5",
    "G_eV_per_fu",
    "G_kJ_mol",
    "fit_sigma_eV",
    "data_distance",
    "extrapolation_score",
    "confidence_label",
    "dominant_model_terms",
    "missing_feature_count",
]

PYQ_FIELDS = [
    "phase",
    "T_K",
    "x_Gd",
    "delta_VO",
    "h_U5",
    "G_kJ_mol",
    "sigma_kJ_mol",
    "confidence_label",
    "source",
]

MOOSE_FIELDS = [
    "material",
    "T_K",
    "x_Gd",
    "oxygen_vacancy_fraction",
    "h_U5",
    "free_energy_kJ_mol",
    "uncertainty_kJ_mol",
    "extrapolation_score",
    "confidence_label",
]


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _finite(row.get(key))
        if value is not None:
            return value
    return None


def parse_grid(values: list[str], *, default: list[float]) -> list[float]:
    if not values:
        return default
    out: list[float] = []
    for item in values:
        for token in str(item).split(","):
            text = token.strip()
            if not text:
                continue
            if ":" in text:
                parts = [float(part) for part in text.split(":")]
                if len(parts) != 3:
                    raise ValueError(f"Grid range must be start:stop:step, got {text!r}")
                start, stop, step = parts
                if step == 0:
                    raise ValueError("Grid step cannot be zero.")
                current = start
                if step > 0:
                    while current <= stop + abs(step) * 1.0e-10:
                        out.append(round(current, 12))
                        current += step
                else:
                    while current >= stop - abs(step) * 1.0e-10:
                        out.append(round(current, 12))
                        current += step
            else:
                out.append(float(text))
    return out


def composition_from_row(row: dict[str, Any]) -> dict[str, float]:
    x_gd = _first_float(row, "x_Gd", "x_gd", "guest_cation_fraction", "Gd_fraction")
    delta = _first_float(row, "delta_VO", "delta", "oxygen_delta", "oxygen_delta_per_formula_unit")
    h_u5 = _first_float(row, "h_U5", "u5_fraction", "U5_fraction", "redox_U5_fraction")
    comp: dict[str, float] = {}
    if x_gd is not None:
        comp["x_Gd"] = x_gd
    if delta is not None:
        comp["delta_VO"] = delta
    if h_u5 is not None:
        comp["h_U5"] = h_u5
    return comp


def species_counts_from_row(row: dict[str, Any]) -> dict[str, int]:
    counts = _json_dict(row.get("species_counts") or row.get("species_counts_json"))
    if counts:
        return {str(key): int(float(value)) for key, value in counts.items() if _finite(value) is not None}
    out: dict[str, int] = {}
    for key, value in row.items():
        if key.startswith("species_"):
            number = _finite(value)
            if number is not None:
                out[key.removeprefix("species_")] = int(number)
    return out


def motif_features_from_row(row: dict[str, Any]) -> dict[str, float]:
    features = _json_dict(row.get("motif_features") or row.get("motif_features_json"))
    out = {str(key): float(value) for key, value in features.items() if _finite(value) is not None}
    for key, value in row.items():
        if key.startswith("feature_"):
            number = _finite(value)
            if number is not None:
                out[key.removeprefix("feature_")] = number
        elif key.startswith("motif_count_"):
            number = _finite(value)
            if number is not None:
                out[key] = number
    return out


def build_training_set_from_csv(path: Path, *, system_name: str, parent_structure: str) -> CETrainingSet:
    records: list[CETrainingRecord] = []
    for index, row in enumerate(_read_csv(path), start=1):
        record_id = str(row.get("record_id") or row.get("config_id") or row.get("motif_id") or f"record_{index:04d}")
        energy = _first_float(row, "energy_eV", "E_static_eV", "G_eV_per_fu", "G_ensemble_eV_per_fu")
        uncertainty = _first_float(row, "uncertainty_eV", "uncertainty_eV_per_fu", "sigma_G_eV_per_fu")
        records.append(
            CETrainingRecord(
                record_id=record_id,
                structure_path=str(row.get("structure_path") or row.get("run_dir") or ""),
                species_counts=species_counts_from_row(row),
                composition=composition_from_row(row),
                motif_features=motif_features_from_row(row),
                energy_eV=energy,
                uncertainty_eV=uncertainty,
                source=str(row.get("source") or "csv"),
                metadata=_json_dict(row.get("metadata") or row.get("metadata_json")),
            )
        )
    return CETrainingSet(
        system_name=system_name,
        parent_structure_path=parent_structure,
        sublattice_model={"cation": ["U4", "U5", "Gd3"], "anion": ["O", "VaO"]},
        species={
            "U4": {"role": "host_cation"},
            "U5": {"role": "redox_cation"},
            "Gd3": {"role": "dopant_cation"},
            "O": {"role": "anion"},
            "VaO": {"role": "oxygen_vacancy"},
        },
        charge_constraints=["N_U5 + 2*N_VaO - N_Gd3 == 0"],
        composition_axes={"x_Gd": "N_Gd3/N_cation", "delta_VO": "N_VaO/N_cation", "h_U5": "N_U5/N_cation"},
        records=records,
        metadata={"schema": SCHEMA, "source_csv": str(path.resolve())},
    )


def composition_feature_values(composition: dict[str, float]) -> dict[str, float]:
    x = float(composition.get("x_Gd", 0.0))
    d = float(composition.get("delta_VO", composition.get("delta", 0.0)))
    h = float(composition.get("h_U5", 0.0))
    return {
        "x_Gd": x,
        "delta_VO": d,
        "h_U5": h,
        "x_Gd:delta_VO": x * d,
        "x_Gd:h_U5": x * h,
        "delta_VO:h_U5": d * h,
        "x_Gd^2": x * x,
        "delta_VO^2": d * d,
        "h_U5^2": h * h,
    }


def feature_vector(record: CETrainingRecord, feature_names: list[str]) -> tuple[list[float], int]:
    values = {**composition_feature_values(record.composition), **record.motif_features}
    missing = 0
    vector = [1.0]
    for name in feature_names:
        if name in values:
            vector.append(float(values[name]))
        else:
            vector.append(0.0)
            missing += 1
    return vector, missing


def infer_feature_names(training_set: CETrainingSet, requested: list[str] | None = None) -> list[str]:
    if requested:
        return list(dict.fromkeys(requested))
    names = list(DEFAULT_FEATURES)
    for record in training_set.records:
        for name in sorted(record.motif_features):
            if name not in names:
                names.append(name)
    return names


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    aug = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1.0e-14:
            aug[col][col] += 1.0e-12
            pivot = col
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        if abs(pivot_value) < 1.0e-14:
            continue
        for item in range(col, n + 1):
            aug[col][item] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for item in range(col, n + 1):
                aug[row][item] -= factor * aug[col][item]
    return [aug[row][n] for row in range(n)]


@dataclass
class LinearMode4Model:
    phase: str
    target: str
    feature_names: list[str]
    coefficients: list[float]
    ridge_lambda: float
    training_rmse_eV: float
    training_count: int
    composition_ranges: dict[str, dict[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"schema": MODEL_SCHEMA, **asdict(self)}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "LinearMode4Model":
        if payload.get("schema") != MODEL_SCHEMA:
            raise ValueError(f"Unexpected model schema: {payload.get('schema')}")
        data = dict(payload)
        data.pop("schema", None)
        return cls(**data)

    def predict_from_values(self, composition: dict[str, float], motif_features: dict[str, float] | None = None) -> tuple[float, int, list[tuple[str, float]]]:
        record = CETrainingRecord(
            record_id="prediction",
            structure_path="",
            composition=composition,
            motif_features=motif_features or {},
        )
        vector, missing = feature_vector(record, self.feature_names)
        value = sum(coef * item for coef, item in zip(self.coefficients, vector))
        terms = [("intercept", self.coefficients[0])]
        for name, coef, item in zip(self.feature_names, self.coefficients[1:], vector[1:]):
            terms.append((name, coef * item))
        terms.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return value, missing, terms[:5]


def fit_linear_model(
    training_set: CETrainingSet,
    *,
    phase: str,
    target: str = "energy_eV",
    feature_names: list[str] | None = None,
    ridge_lambda: float = 1.0e-8,
) -> LinearMode4Model:
    names = infer_feature_names(training_set, feature_names)
    rows: list[list[float]] = []
    y: list[float] = []
    for record in training_set.records:
        target_value = _finite(getattr(record, target, None))
        if target_value is None and record.free_energy_terms:
            target_value = _finite(record.free_energy_terms.get(target))
        if target_value is None:
            continue
        vector, _missing = feature_vector(record, names)
        rows.append(vector)
        y.append(target_value)
    if not rows:
        raise ValueError(f"No training records have target {target!r}.")
    nfeat = len(rows[0])
    xtx = [[0.0 for _ in range(nfeat)] for _ in range(nfeat)]
    xty = [0.0 for _ in range(nfeat)]
    for vector, target_value in zip(rows, y):
        for i in range(nfeat):
            xty[i] += vector[i] * target_value
            for j in range(nfeat):
                xtx[i][j] += vector[i] * vector[j]
    for i in range(1, nfeat):
        xtx[i][i] += ridge_lambda
    coeffs = _solve_linear_system(xtx, xty)
    residuals = []
    for vector, target_value in zip(rows, y):
        pred = sum(coef * item for coef, item in zip(coeffs, vector))
        residuals.append(pred - target_value)
    rmse = math.sqrt(sum(value * value for value in residuals) / max(len(residuals), 1))
    ranges: dict[str, dict[str, float]] = {}
    for axis in ("x_Gd", "delta_VO", "h_U5"):
        values = [record.composition[axis] for record in training_set.records if axis in record.composition]
        if values:
            ranges[axis] = {"min": min(values), "max": max(values)}
    return LinearMode4Model(
        phase=phase,
        target=target,
        feature_names=names,
        coefficients=coeffs,
        ridge_lambda=ridge_lambda,
        training_rmse_eV=rmse,
        training_count=len(rows),
        composition_ranges=ranges,
        metadata={
            "schema": SCHEMA,
            "system_name": training_set.system_name,
            "parent_structure_path": training_set.parent_structure_path,
            "notes": [
                "First-pass regularized motif/cluster Hamiltonian skeleton.",
                "Replace or validate with mode4/smol/CASM CE-MC before production thermodynamics.",
            ],
        },
    )


def _range_distance(value: float, axis_range: dict[str, float] | None) -> float:
    if not axis_range:
        return 1.0
    lo = axis_range.get("min", 0.0)
    hi = axis_range.get("max", 0.0)
    width = max(hi - lo, 1.0e-12)
    if lo <= value <= hi:
        return 0.0
    return min(abs(value - lo), abs(value - hi)) / width


def extrapolation_score(model: LinearMode4Model, composition: dict[str, float]) -> float:
    total = 0.0
    for axis in ("x_Gd", "delta_VO", "h_U5"):
        total += _range_distance(float(composition.get(axis, 0.0)), model.composition_ranges.get(axis))
    return total


def confidence_label(score: float, missing_features: int) -> str:
    if missing_features > 0 or score > 0.5:
        return "extrapolated"
    if score > 0.0:
        return "edge"
    return "interpolated"


def sample_surface(
    model: LinearMode4Model,
    *,
    temperatures: list[float],
    x_grid: list[float],
    delta_grid: list[float],
    h_grid: list[float],
    backend: str = "mode4_linear_skeleton",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for temp in temperatures:
        for x_gd in x_grid:
            for delta in delta_grid:
                for h_u5 in h_grid:
                    composition = {"x_Gd": x_gd, "delta_VO": delta, "h_U5": h_u5}
                    g_value, missing, terms = model.predict_from_values(composition)
                    score = extrapolation_score(model, composition)
                    sigma = model.training_rmse_eV * (1.0 + score + 0.25 * missing)
                    rows.append(
                        {
                            "phase": model.phase,
                            "backend": backend,
                            "T_K": temp,
                            "x_Gd": x_gd,
                            "delta_VO": delta,
                            "h_U5": h_u5,
                            "G_eV_per_fu": g_value,
                            "G_kJ_mol": g_value * EV_TO_KJ_MOL,
                            "fit_sigma_eV": sigma,
                            "data_distance": score,
                            "extrapolation_score": score,
                            "confidence_label": confidence_label(score, missing),
                            "dominant_model_terms": json.dumps(dict(terms), sort_keys=True),
                            "missing_feature_count": missing,
                        }
                    )
    return rows


def build_training_command(args: argparse.Namespace) -> dict[str, Any]:
    training = build_training_set_from_csv(args.input_csv.resolve(), system_name=args.system, parent_structure=args.parent_structure)
    write_ce_training_jsonl(args.output.resolve(), training)
    metadata = {
        "schema": SCHEMA,
        "stage": "build_training",
        "input_csv": str(args.input_csv.resolve()),
        "output": str(args.output.resolve()),
        "n_records": len(training.records),
        "composition_axes": training.composition_axes,
    }
    _write_json(args.output.resolve().with_suffix(".metadata.json"), metadata)
    print(f"Training records: {len(training.records)}")
    print(f"Wrote training  : {args.output.resolve()}")
    return metadata


def fit_command(args: argparse.Namespace) -> dict[str, Any]:
    training = read_ce_training_jsonl(args.training_jsonl.resolve())
    feature_names = args.feature or None
    model = fit_linear_model(
        training,
        phase=args.phase,
        target=args.target,
        feature_names=feature_names,
        ridge_lambda=args.ridge_lambda,
    )
    outdir = args.outdir.resolve()
    model_path = outdir / "mode4_linear_model.json"
    _write_json(model_path, model.to_json())
    coeff_rows = [
        {"term": "intercept", "coefficient_eV": model.coefficients[0]},
        *[
            {"term": name, "coefficient_eV": coef}
            for name, coef in zip(model.feature_names, model.coefficients[1:])
        ],
    ]
    _write_csv(outdir / "mode4_interaction_parameters.csv", coeff_rows, ["term", "coefficient_eV"])
    print(f"Training records : {model.training_count}")
    print(f"Training RMSE eV : {model.training_rmse_eV:.6g}")
    print(f"Wrote model      : {model_path}")
    return model.to_json()


def sample_command(args: argparse.Namespace) -> dict[str, Any]:
    model = LinearMode4Model.from_json(json.loads(args.model_json.resolve().read_text(encoding="utf-8")))
    rows = sample_surface(
        model,
        temperatures=parse_grid(args.temperature, default=[300.0]),
        x_grid=parse_grid(args.x_grid, default=[0.0]),
        delta_grid=parse_grid(args.delta_grid, default=[0.0]),
        h_grid=parse_grid(args.h_u5_grid, default=[0.0]),
    )
    outdir = args.outdir.resolve()
    surface_csv = outdir / "mode4_dense_gibbs_surface.csv"
    _write_csv(surface_csv, rows, SURFACE_FIELDS)
    metadata = {
        "schema": SCHEMA,
        "stage": "sample_surface",
        "model": str(args.model_json.resolve()),
        "surface_csv": str(surface_csv),
        "n_rows": len(rows),
        "notes": ["Rows labeled extrapolated should not be used as production thermodynamics without validation."],
    }
    _write_json(outdir / "mode4_dense_gibbs_surface.metadata.json", metadata)
    print(f"Surface rows : {len(rows)}")
    print(f"Wrote surface: {surface_csv}")
    return metadata


def export_command(args: argparse.Namespace) -> dict[str, Any]:
    surface_rows = _read_csv(args.surface_csv.resolve())
    outdir = args.outdir.resolve()
    py_rows = []
    moose_rows = []
    for row in surface_rows:
        sigma_eV = _finite(row.get("fit_sigma_eV")) or 0.0
        py_rows.append(
            {
                "phase": row.get("phase") or args.phase,
                "T_K": row.get("T_K"),
                "x_Gd": row.get("x_Gd"),
                "delta_VO": row.get("delta_VO"),
                "h_U5": row.get("h_U5"),
                "G_kJ_mol": row.get("G_kJ_mol"),
                "sigma_kJ_mol": sigma_eV * EV_TO_KJ_MOL,
                "confidence_label": row.get("confidence_label"),
                "source": "mode4_dense_surface",
            }
        )
        moose_rows.append(
            {
                "material": args.material,
                "T_K": row.get("T_K"),
                "x_Gd": row.get("x_Gd"),
                "oxygen_vacancy_fraction": row.get("delta_VO"),
                "h_U5": row.get("h_U5"),
                "free_energy_kJ_mol": row.get("G_kJ_mol"),
                "uncertainty_kJ_mol": sigma_eV * EV_TO_KJ_MOL,
                "extrapolation_score": row.get("extrapolation_score"),
                "confidence_label": row.get("confidence_label"),
            }
        )
    py_path = outdir / "pycalphad_parameterized_g_surface.csv"
    moose_path = outdir / "moose_parameterized_g_surface.csv"
    _write_csv(py_path, py_rows, PYQ_FIELDS)
    _write_csv(moose_path, moose_rows, MOOSE_FIELDS)
    payload = {
        "schema": SCHEMA,
        "stage": "export_parameterized_surface",
        "surface_csv": str(args.surface_csv.resolve()),
        "pycalphad_table": str(py_path),
        "moose_table": str(moose_path),
        "parameterization_status": "tabulated_skeleton",
        "state_variables": ["T_K", "x_Gd", "delta_VO", "h_U5", "mu_O_or_pO2", "eta"],
        "notes": [
            "This is a dense tabulated bridge, not a final TDB assessment.",
            "Fit analytic CALPHAD parameters after validation and uncertainty review.",
        ],
    }
    _write_json(outdir / "parameterized_gibbs_surface_handoff.json", payload)
    print(f"Wrote pycalphad table: {py_path}")
    print(f"Wrote MOOSE table    : {moose_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy-mode4-surface",
        description="Skeleton bridge from guarded defect motifs to dense G(T,x,delta,h_U5) surfaces.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-training", help="Convert motif/free-energy CSV rows to CETrainingSet JSONL.")
    build.add_argument("--input-csv", type=Path, required=True)
    build.add_argument("--output", type=Path, default=Path("mode4_training_set.jsonl"))
    build.add_argument("--system", default="Gd-UO2")
    build.add_argument("--parent-structure", default="fluorite_Fm-3m")

    fit = sub.add_parser("fit", help="Fit a regularized linear motif/cluster Hamiltonian skeleton.")
    fit.add_argument("--training-jsonl", type=Path, required=True)
    fit.add_argument("--outdir", type=Path, default=Path("mode4_fit"))
    fit.add_argument("--phase", default="FLUORITE_GD_U_O_DEFECT")
    fit.add_argument("--target", default="energy_eV")
    fit.add_argument("--feature", action="append", default=[])
    fit.add_argument("--ridge-lambda", type=float, default=1.0e-8)

    sample = sub.add_parser("sample-surface", help="Sample a dense G surface from a fitted skeleton model.")
    sample.add_argument("--model-json", type=Path, required=True)
    sample.add_argument("--outdir", type=Path, default=Path("mode4_surface"))
    sample.add_argument("--temperature", action="append", default=[])
    sample.add_argument("--x-grid", action="append", default=[])
    sample.add_argument("--delta-grid", action="append", default=[])
    sample.add_argument("--h-u5-grid", action="append", default=[])

    export = sub.add_parser("export-parameterized", help="Export dense surface tables for pycalphad/MOOSE handoff.")
    export.add_argument("--surface-csv", type=Path, required=True)
    export.add_argument("--outdir", type=Path, default=Path("mode4_parameterized_export"))
    export.add_argument("--phase", default="FLUORITE_GD_U_O_DEFECT")
    export.add_argument("--material", default="GdUO2_defect_fluorite")

    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.command == "build-training":
        return build_training_command(args)
    if args.command == "fit":
        return fit_command(args)
    if args.command == "sample-surface":
        return sample_command(args)
    if args.command == "export-parameterized":
        return export_command(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
