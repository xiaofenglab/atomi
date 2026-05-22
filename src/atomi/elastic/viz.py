"""Visualize and enrich VASP/MD elastic tensors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from atomi.core.cell import cell_metadata, infer_formula_units
from atomi.elastic.derived import (
    complete_elastic_derived,
    complete_thermophysical_derived,
    debye_thermal_table,
    formula_atom_count,
    fracture_toughness_from_fracture_energy,
)
from atomi.lammps.elastic import tensor_components


VOIGT_MAT = ((0, 5, 4), (5, 1, 3), (4, 3, 2))


@dataclass
class ElasticRecord:
    label: str
    temperature_K: float | None
    tensor_GPa: np.ndarray
    row: dict[str, Any]
    source: str


def write_json(path: Path, data: Any) -> None:
    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def table_lookup(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[float, dict[str, str]]]:
    by_label: dict[str, dict[str, str]] = {}
    by_temperature: dict[float, dict[str, str]] = {}
    for row in rows:
        label = row.get("label")
        if label:
            by_label[label] = row
        temperature = float_or_none(row.get("temperature_K"))
        if temperature is not None:
            by_temperature[round(temperature, 8)] = row
    return by_label, by_temperature


def merged_row(
    *,
    label: str,
    temperature_K: float | None,
    source: str,
    tensor: np.ndarray,
    moduli: dict[str, Any] | None,
    table_rows: list[dict[str, str]],
) -> dict[str, Any]:
    by_label, by_temperature = table_lookup(table_rows)
    row: dict[str, Any] = {}
    if label in by_label:
        row.update(by_label[label])
    elif temperature_K is not None and round(temperature_K, 8) in by_temperature:
        row.update(by_temperature[round(temperature_K, 8)])
    row.update({"label": label, "source": source})
    if temperature_K is not None:
        row["temperature_K"] = temperature_K
    if moduli:
        row.update(moduli)
    for key, value in tensor_components(tensor).items():
        row.setdefault(key, value)
    return row


def load_elastic_records(tensors_path: Path, table_path: Path | None = None) -> list[ElasticRecord]:
    payload = json.loads(tensors_path.read_text(encoding="utf-8"))
    table_rows = read_csv_rows(table_path)
    records: list[ElasticRecord] = []
    if isinstance(payload, dict) and isinstance(payload.get("tensors"), list):
        for item in payload["tensors"]:
            tensor_raw = item.get("symmetry_reduced_tensor_GPa") or item.get("C_symmetry_reduced_GPa")
            if tensor_raw is None:
                continue
            tensor = np.asarray(tensor_raw, dtype=float)
            label = str(item.get("label") or item.get("run_dir") or f"record_{len(records) + 1}")
            temperature = float_or_none(item.get("temperature_K"))
            row = merged_row(
                label=label,
                temperature_K=temperature,
                source=str(item.get("source") or "VASP/static elastic"),
                tensor=tensor,
                moduli=item.get("moduli", {}),
                table_rows=table_rows,
            )
            row.setdefault("symmetry", item.get("symmetry", ""))
            records.append(ElasticRecord(label, temperature, tensor, row, row["source"]))
        return records
    if isinstance(payload, dict):
        for key, item in payload.items():
            if not isinstance(item, dict):
                continue
            tensor_raw = item.get("C_symmetry_reduced_GPa") or item.get("symmetry_reduced_tensor_GPa")
            if tensor_raw is None:
                continue
            tensor = np.asarray(tensor_raw, dtype=float)
            temperature = float_or_none(item.get("temperature_K"))
            label = str(item.get("label") or (f"T{temperature:g}K" if temperature is not None else key))
            row = merged_row(
                label=label,
                temperature_K=temperature,
                source=str(item.get("source") or "LAMMPS/MD elastic"),
                tensor=tensor,
                moduli=item.get("moduli", {}),
                table_rows=table_rows,
            )
            row.setdefault("symmetry", item.get("symmetry", ""))
            row.setdefault("inferred_symmetry", item.get("inferred_symmetry", ""))
            if item.get("md_box"):
                row.update(
                    {
                        "V_mean_A3": item["md_box"].get("volume_A3_mean", row.get("V_mean_A3", "")),
                        "a_mean_A": item["md_box"].get("a_A_mean", row.get("a_mean_A", "")),
                        "b_mean_A": item["md_box"].get("b_A_mean", row.get("b_mean_A", "")),
                        "c_mean_A": item["md_box"].get("c_A_mean", row.get("c_mean_A", "")),
                    }
                )
            records.append(ElasticRecord(label, temperature, tensor, row, row["source"]))
    if not records:
        raise ValueError(f"No elastic tensors found in {tensors_path}")
    return sorted(records, key=lambda r: (r.temperature_K is None, r.temperature_K or 0.0, r.label))


class DirectionalElastic:
    def __init__(self, tensor_GPa: np.ndarray):
        self.C = np.asarray(tensor_GPa, dtype=float)
        self.S = np.linalg.inv(self.C)
        self.S4 = self._compliance_tensor()

    def _compliance_tensor(self) -> np.ndarray:
        out = np.zeros((3, 3, 3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                p = VOIGT_MAT[i][j]
                for k in range(3):
                    for ell in range(3):
                        q = VOIGT_MAT[k][ell]
                        out[i, j, k, ell] = self.S[p, q] / ((1 + p // 3) * (1 + q // 3))
        return out

    @staticmethod
    def direction(theta: float, phi: float) -> np.ndarray:
        return np.asarray(
            [
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ],
            dtype=float,
        )

    def young(self, theta: float, phi: float) -> float:
        n = self.direction(theta, phi)
        denom = float(np.einsum("i,j,k,l,ijkl", n, n, n, n, self.S4))
        return 1.0 / denom if abs(denom) > 1.0e-14 else math.nan

    def linear_compressibility(self, theta: float, phi: float) -> float:
        n = self.direction(theta, phi)
        value = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    value += n[i] * n[j] * self.S4[i, j, k, k]
        return 1000.0 * value


class ElateDirectional:
    def __init__(self, tensor_GPa: np.ndarray):
        from ELATE.elastic import Elastic

        self.obj = Elastic(np.asarray(tensor_GPa, dtype=float).tolist())

    def young(self, theta: float, phi: float) -> float:
        return float(self.obj.Young_2(theta, phi))

    def linear_compressibility(self, theta: float, phi: float) -> float:
        return float(self.obj.LC_2(theta, phi))


def directional_backend(tensor: np.ndarray, backend: str) -> tuple[Any, str, str | None]:
    if backend == "none":
        raise ValueError("3D plotting requested with --backend none")
    if backend in {"auto", "elate"}:
        try:
            return ElateDirectional(tensor), "elate", None
        except Exception as exc:
            if backend == "elate":
                raise
            return DirectionalElastic(tensor), "native", str(exc)
    return DirectionalElastic(tensor), "native", None


def safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in label).strip("_") or "elastic"


def surface_arrays(
    func: Callable[[float, float], float],
    *,
    npoints: int,
    signed_radius: bool = False,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    theta = np.linspace(0.0, math.pi, npoints)
    phi = np.linspace(0.0, 2.0 * math.pi, 2 * npoints)
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    z_rows: list[list[float]] = []
    values: list[list[float]] = []
    for th in theta:
        x_row: list[float] = []
        y_row: list[float] = []
        z_row: list[float] = []
        v_row: list[float] = []
        for ph in phi:
            value = float(func(float(th), float(ph)))
            radius = abs(value) if signed_radius else value
            x_row.append(radius * math.sin(th) * math.cos(ph))
            y_row.append(radius * math.sin(th) * math.sin(ph))
            z_row.append(radius * math.cos(th))
            v_row.append(value)
        x_rows.append(x_row)
        y_rows.append(y_row)
        z_rows.append(z_row)
        values.append(v_row)
    return x_rows, y_rows, z_rows, values


def write_surface_html(
    path: Path,
    *,
    title: str,
    property_label: str,
    unit: str,
    arrays: tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]],
) -> None:
    x_rows, y_rows, z_rows, values = arrays
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
</head>
<body>
  <div id="plot" style="width: 900px; height: 760px;"></div>
  <script>
    const trace = {{
      type: "surface",
      x: {json.dumps(x_rows)},
      y: {json.dumps(y_rows)},
      z: {json.dumps(z_rows)},
      surfacecolor: {json.dumps(values)},
      colorscale: "Viridis",
      colorbar: {{title: "{property_label} ({unit})"}},
      hovertemplate: "{property_label}: %{{surfacecolor:.4g}} {unit}<extra></extra>"
    }};
    const layout = {{
      title: {json.dumps(title)},
      scene: {{aspectmode: "data"}},
      margin: {{l: 0, r: 0, b: 0, t: 55}}
    }};
    Plotly.newPlot("plot", [trace], layout);
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_elate_input(path: Path, tensor: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in np.asarray(tensor, dtype=float):
            handle.write(" ".join(f"{value:16.8f}" for value in row) + "\n")


def fmt_float(value: Any, digits: int = 3) -> str:
    number = float_or_none(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def format_tensor_matrix(tensor: np.ndarray) -> str:
    c = np.asarray(tensor, dtype=float)
    labels = ["xx", "yy", "zz", "yz", "xz", "xy"]
    lines = ["             " + " ".join(f"{label:>10}" for label in labels)]
    for label, row in zip(labels, c):
        lines.append(f"    {label:>3}      " + " ".join(f"{value:10.3f}" for value in row))
    return "\n".join(lines)


def print_terminal_record_report(record: ElasticRecord, row: dict[str, Any]) -> None:
    temperature = fmt_float(record.temperature_K, digits=1) if record.temperature_K is not None else "NA"
    symmetry = row.get("symmetry") or row.get("inferred_symmetry") or "unknown"
    print("")
    print("Elastic tensor summary")
    print("----------------------")
    print(f"Label       : {record.label}")
    print(f"Source      : {record.source}")
    print(f"Temperature : {temperature} K")
    print(f"Symmetry    : {symmetry}")
    print("C_ij tensor : GPa, Voigt order xx yy zz yz xz xy")
    print(format_tensor_matrix(record.tensor_GPa))
    print(
        "Bulk modulus K (GPa): "
        f"Voigt upper={fmt_float(row.get('K_V_GPa'))}, "
        f"Reuss lower={fmt_float(row.get('K_R_GPa'))}, "
        f"Hill={fmt_float(row.get('K_H_GPa'))}"
    )
    print(
        "Shear modulus G (GPa): "
        f"Voigt upper={fmt_float(row.get('G_V_GPa'))}, "
        f"Reuss lower={fmt_float(row.get('G_R_GPa'))}, "
        f"Hill={fmt_float(row.get('G_H_GPa'))}"
    )
    print(
        f"Young/Poisson: E_H={fmt_float(row.get('E_H_GPa'))} GPa, "
        f"nu_H={fmt_float(row.get('nu_H'), digits=4)}"
    )
    print(f"Stability    : positive definite={row.get('mechanically_stable_positive_definite', 'NA')}")


def print_terminal_moduli_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print("")
    print("Elastic moduli vs temperature")
    print("-----------------------------")
    print(
        " T(K)  symmetry    K_V    K_R    K_H    G_V    G_R    G_H    "
        "E_H     nu_H    stable"
    )
    for row in rows:
        print(
            f"{fmt_float(row.get('temperature_K'), 1):>6} "
            f"{str(row.get('symmetry') or row.get('inferred_symmetry') or 'unknown')[:10]:>10} "
            f"{fmt_float(row.get('K_V_GPa')):>6} "
            f"{fmt_float(row.get('K_R_GPa')):>6} "
            f"{fmt_float(row.get('K_H_GPa')):>6} "
            f"{fmt_float(row.get('G_V_GPa')):>6} "
            f"{fmt_float(row.get('G_R_GPa')):>6} "
            f"{fmt_float(row.get('G_H_GPa')):>6} "
            f"{fmt_float(row.get('E_H_GPa')):>7} "
            f"{fmt_float(row.get('nu_H'), 4):>7} "
            f"{row.get('mechanically_stable_positive_definite', 'NA')}"
        )


def maybe_plot_lines(outdir: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    if not rows:
        return []
    t = np.asarray([float_or_none(row.get("temperature_K")) or i for i, row in enumerate(rows)], dtype=float)
    plot_specs = [
        ("elastic_moduli_vs_T.png", ["K_H_GPa", "G_H_GPa", "E_H_GPa"], "Modulus (GPa)"),
        ("elastic_anisotropy_vs_T.png", ["universal_anisotropy_AU", "zener_anisotropy", "pugh_K_over_G"], "Value"),
        ("elastic_sound_velocity_vs_T.png", ["v_s_km_s", "v_p_km_s", "v_m_km_s"], "Velocity (km/s)"),
        ("elastic_debye_density_vs_T.png", ["theta_D_K", "density_g_cm3"], "Value"),
        ("elastic_min_thermal_conductivity_vs_T.png", ["k_min_cahill_W_mK", "k_min_clarke_W_mK"], "k_min (W/m/K)"),
        ("elastic_hardness_vs_T.png", ["hardness_teter_GPa", "hardness_chen_GPa", "hardness_tian_GPa"], "Hardness estimate (GPa)"),
        ("elastic_strain_energy_density_vs_T.png", ["strain_energy_density_1pct_x_MJ_m3", "strain_energy_density_1pct_shear_xy_MJ_m3"], "Energy density (MJ/m^3)"),
    ]
    written: list[str] = []
    for filename, columns, ylabel in plot_specs:
        available = [col for col in columns if any(float_or_none(row.get(col)) is not None for row in rows)]
        if not available:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
        for col in available:
            y = np.asarray([float_or_none(row.get(col)) or np.nan for row in rows], dtype=float)
            ax.plot(t, y, marker="o", linewidth=1.6, label=col)
        has_temperature = any(float_or_none(row.get("temperature_K")) is not None for row in rows)
        ax.set_xlabel("Temperature (K)" if has_temperature else "Record")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        target = outdir / filename
        fig.savefig(target, dpi=220)
        plt.close(fig)
        written.append(str(target))
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elastic_viz",
        description=(
            "Postprocess and visualize VASP/MD elastic tensors with optional "
            "ELATE-style 3D surfaces."
        ),
    )
    parser.add_argument(
        "--elastic-dir",
        type=Path,
        default=Path("."),
        help="Directory containing elastic_tensors.json and elastic_moduli_T.csv.",
    )
    parser.add_argument("--elastic-tensors", type=Path, help="Explicit elastic_tensors.json path.")
    parser.add_argument("--elastic-table", type=Path, help="Explicit elastic_moduli_T.csv path.")
    parser.add_argument("--outdir", type=Path, default=Path("analysis/elastic_viz"))
    parser.add_argument("--source-label", default="", help="Optional label stored in metadata.")
    parser.add_argument("--formula", help="Formula for density/Debye calculations, e.g. Si.")
    parser.add_argument("--natoms", type=float, help="Atoms in the elastic simulation cell.")
    parser.add_argument(
        "--atoms-per-formula-unit",
        type=float,
        help="Atoms per formula unit. Defaults to the parsed formula atom count when --formula is given.",
    )
    parser.add_argument(
        "--formula-units",
        type=float,
        help="Formula units represented by each elastic tensor cell. If omitted, infer from --natoms/--atoms-per-formula-unit.",
    )
    parser.add_argument(
        "--target-z",
        type=float,
        default=4.0,
        help="Formula units in the crystallographic target cell used for normalized reporting.",
    )
    parser.add_argument("--density-kg-m3", type=float, help="Override density in kg/m^3.")
    parser.add_argument("--density-g-cm3", type=float, help="Override density in g/cm^3.")
    parser.add_argument(
        "--fracture-energy-J-m2",
        type=float,
        help="Optional fracture/cleavage energy for Griffith K_IC estimate.",
    )
    parser.add_argument(
        "--surface-energy-J-m2",
        type=float,
        help="Optional surface energy; K_IC uses fracture energy 2*surface_energy when fracture energy is absent.",
    )
    parser.add_argument(
        "--fracture-plane-stress",
        action="store_true",
        help="Use plane-stress E instead of plane-strain E/(1-nu^2) for K_IC estimate.",
    )
    parser.add_argument(
        "--molar-mass-g-mol",
        type=float,
        help="Molar mass used with density-only Debye calculations.",
    )
    parser.add_argument("--backend", choices=("auto", "elate", "native", "none"), default="auto")
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="Write ELATE-style 3D HTML surfaces for selected tensors.",
    )
    parser.add_argument("--max-3d-records", type=int, default=6)
    parser.add_argument("--surface-npoints", type=int, default=50)
    parser.add_argument(
        "--no-elate-inputs",
        action="store_true",
        help="Do not write ELATE-compatible tensor text files.",
    )
    parser.add_argument(
        "--no-terminal-report",
        action="store_true",
        help="Do not print tensor matrices and Voigt/Reuss/Hill moduli to the terminal.",
    )
    parser.add_argument(
        "--write-debye-thermal",
        action="store_true",
        help="Write Debye Cv/H/S/F tables using derived theta_D.",
    )
    parser.add_argument("--debye-T-min", type=float, default=0.0)
    parser.add_argument("--debye-T-max", type=float, default=1500.0)
    parser.add_argument("--debye-T-step", type=float, default=10.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    elastic_dir = args.elastic_dir.resolve()
    tensors_path = (args.elastic_tensors or (elastic_dir / "elastic_tensors.json")).resolve()
    table_path = (args.elastic_table or (elastic_dir / "elastic_moduli_T.csv")).resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    records = load_elastic_records(tensors_path, table_path if table_path.exists() else None)
    density = args.density_kg_m3
    if density is None and args.density_g_cm3 is not None:
        density = args.density_g_cm3 * 1000.0
    atoms_per_formula_unit = args.atoms_per_formula_unit
    if atoms_per_formula_unit is None and args.formula:
        atoms_per_formula_unit = formula_atom_count(args.formula)
    formula_units = infer_formula_units(
        formula_units=args.formula_units,
        natoms=args.natoms,
        atoms_per_formula_unit=atoms_per_formula_unit,
        formula=args.formula,
    )
    cell_meta = cell_metadata(
        formula=args.formula,
        natoms=args.natoms,
        atoms_per_formula_unit=atoms_per_formula_unit,
        formula_units=formula_units,
        target_z=args.target_z,
        cell_role="elastic-simulation-cell",
        normalization_basis="per-formula",
    )
    summary_rows: list[dict[str, Any]] = []
    elate_inputs: list[str] = []
    surface_outputs: list[str] = []
    backend_notes: list[dict[str, Any]] = []
    debye_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = dict(record.row)
        row.update(complete_elastic_derived(row, record.tensor_GPa))
        row.update(
            complete_thermophysical_derived(
                row,
                formula=args.formula,
                formula_units=formula_units,
                density_kg_m3=density,
                molar_mass_g_mol=args.molar_mass_g_mol,
                atoms_per_formula_unit=atoms_per_formula_unit,
            )
        )
        row.update(
            {
                "formula": cell_meta["formula"],
                "natoms": cell_meta["natoms"],
                "atoms_per_formula_unit": cell_meta["atoms_per_formula_unit"],
                "n_formula_units": cell_meta["n_formula_units"],
                "target_z_formula_units": cell_meta["target_z_formula_units"],
                "normalization_basis": cell_meta["normalization_basis"],
                "cell_role": cell_meta["cell_role"],
            }
        )
        fracture_energy = args.fracture_energy_J_m2
        if fracture_energy is None and args.surface_energy_J_m2 is not None:
            fracture_energy = 2.0 * args.surface_energy_J_m2
        if fracture_energy is not None:
            e_h = float_or_none(row.get("E_H_GPa"))
            nu_h = float_or_none(row.get("nu_H"))
            if e_h is not None:
                row["fracture_energy_J_m2"] = fracture_energy
                row["fracture_toughness_griffith_MPa_sqrt_m"] = fracture_toughness_from_fracture_energy(
                    young_GPa=e_h,
                    poisson=nu_h,
                    fracture_energy_J_m2=fracture_energy,
                    plane_strain=not args.fracture_plane_stress,
                )
        summary_rows.append(row)
        if not args.no_terminal_report:
            print_terminal_record_report(record, row)
        name = safe_label(record.label)
        if not args.no_elate_inputs:
            target = outdir / "elate_inputs" / f"{name}_tensor_GPa.txt"
            write_elate_input(target, record.tensor_GPa)
            elate_inputs.append(str(target))
        if args.write_debye_thermal and row.get("theta_D_K") and atoms_per_formula_unit:
            for thermal in debye_thermal_table(
                float(row["theta_D_K"]),
                atoms_per_formula_unit=float(atoms_per_formula_unit),
                T_min=args.debye_T_min,
                T_max=args.debye_T_max,
                T_step=args.debye_T_step,
            ):
                debye_rows.append({"label": record.label, "elastic_temperature_K": record.temperature_K, **thermal})
        if args.plot_3d and index < args.max_3d_records and args.backend != "none":
            adapter, used_backend, fallback_reason = directional_backend(record.tensor_GPa, args.backend)
            backend_notes.append(
                {
                    "label": record.label,
                    "requested_backend": args.backend,
                    "used_backend": used_backend,
                    "fallback_reason": fallback_reason or "",
                }
            )
            young_html = outdir / "elate_surfaces" / f"{name}_young_3d.html"
            write_surface_html(
                young_html,
                title=f"{record.label} directional Young modulus",
                property_label="Young modulus",
                unit="GPa",
                arrays=surface_arrays(adapter.young, npoints=args.surface_npoints),
            )
            surface_outputs.append(str(young_html))
            lc_html = outdir / "elate_surfaces" / f"{name}_linear_compressibility_3d.html"
            write_surface_html(
                lc_html,
                title=f"{record.label} directional linear compressibility",
                property_label="Linear compressibility",
                unit="TPa^-1",
                arrays=surface_arrays(
                    adapter.linear_compressibility,
                    npoints=args.surface_npoints,
                    signed_radius=True,
                ),
            )
            surface_outputs.append(str(lc_html))
    if not args.no_terminal_report:
        print_terminal_moduli_table(summary_rows)
    summary_csv = outdir / "elastic_thermophysical_summary.csv"
    write_csv(summary_csv, summary_rows)
    debye_csv = ""
    if debye_rows:
        debye_path = outdir / "debye_thermal_functions.csv"
        write_csv(debye_path, debye_rows)
        debye_csv = str(debye_path)
    line_plots = maybe_plot_lines(outdir, summary_rows)
    metadata = {
        "inputs": {
            "elastic_tensors": str(tensors_path),
            "elastic_table": str(table_path) if table_path.exists() else "",
        },
        "source_label": args.source_label,
        "n_records": len(records),
        "formula": args.formula or "",
        "cell_metadata": cell_meta,
        "formula_units": formula_units,
        "density_kg_m3_override": density,
        "elate": {
            "note": (
                "3D surfaces use ELATE when importable; native directional tensor "
                "formulas are used as fallback in auto mode."
            ),
            "backend_notes": backend_notes,
            "input_files": elate_inputs,
            "surface_outputs": surface_outputs,
        },
        "elastool_postanalysis_mapping": {
            "implemented_independently": [
                "Voigt-Reuss-Hill moduli",
                "Pugh ratio",
                "Cauchy pressure",
                "Zener and universal anisotropy",
                "sound velocities",
                "Debye temperature",
                "optional Debye Cv/H/S/F relative table",
                "minimum thermal conductivity estimates (Cahill/Clarke)",
                "empirical hardness estimates",
                "strain-energy-density screening values",
                "Griffith K_IC when fracture/surface energy is provided",
            ],
            "external_only": [
                "Christoffel ray surfaces and power-flow angles",
                "calibrated fracture/damage laws",
            ],
        },
        "outputs": {
            "summary_csv": str(summary_csv),
            "debye_thermal_csv": debye_csv,
            "line_plots": line_plots,
        },
    }
    write_json(outdir / "elastic_viz_metadata.json", metadata)
    print(f"Wrote elastic thermophysical summary: {summary_csv}")
    if line_plots:
        print(f"Wrote {len(line_plots)} summary plots.")
    if surface_outputs:
        print(f"Wrote {len(surface_outputs)} ELATE-style 3D HTML surfaces.")
    return metadata


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
