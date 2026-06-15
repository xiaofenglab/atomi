"""Command-line interface for thermodynamic prior JSON records."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from .aeris import AerisAdapter, AerisConfig
from .core import (
    read_prior,
    salt_reference_gform_kj_mol,
    write_cp_prior,
    write_line_compound_prior,
)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _candidate_hpc_config_paths() -> list[Path]:
    paths: list[Path] = []
    for env_name in ("ATOMI_HPC_CONFIG", "ATOMI_API_KEYS_JSON"):
        value = os.environ.get(env_name)
        if value:
            paths.append(Path(value).expanduser())
    paths.extend(
        [
            Path.home() / "atomi_hpc" / "atomi_hpc_config.kit.local.json",
            Path.home() / "hpc_atomi" / "atomi_hpc_config.kit.local.json",
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
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


def _aeris_path_default(field: str, env_name: str) -> Path | None:
    return _env_path(env_name) or (
        Path(value).expanduser()
        if (value := _kit_value(("environment_exports", env_name), ("aeris", field)))
        else None
    )


def _aeris_device_default() -> str:
    return (
        os.environ.get("ATOMI_AERIS_DEVICE")
        or _kit_value(("environment_exports", "ATOMI_AERIS_DEVICE"), ("aeris", "device"))
        or "cpu"
    )



def _optional_module_status(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    return {
        "module": module_name,
        "installed": spec is not None,
        "origin": getattr(spec, "origin", None) if spec is not None else None,
    }


def _gnn_status(args: argparse.Namespace) -> dict[str, Any]:
    if args.aeris_root and args.aeris_model:
        aeris = AerisAdapter(AerisConfig(root=args.aeris_root, model=args.aeris_model, device=args.aeris_device)).status()
    else:
        aeris = {
            "root": str(args.aeris_root) if args.aeris_root else None,
            "model": str(args.aeris_model) if args.aeris_model else None,
            "root_exists": bool(args.aeris_root and args.aeris_root.exists()),
            "aeris_py_exists": False,
            "model_exists": bool(args.aeris_model and args.aeris_model.exists()),
            "ready": False,
            "missing_configuration": "Set ATOMI_AERIS_ROOT and ATOMI_AERIS_MODEL or configure aeris in the private KIT JSON.",
        }
    backends = {
        "aeris": {**aeris, "role": "local project checkpoint for formation-energy prior"},
        "chgnet": {
            **_optional_module_status("chgnet"),
            "role": "public pretrained universal graph/MLIP prior; screen only unless fine-tuned/validated for target chemistry",
        },
        "matgl": {
            **_optional_module_status("matgl"),
            "role": "public pretrained M3GNet/MatGL prior; screen only unless fine-tuned/validated for target chemistry",
        },
    }
    ready = [name for name, status in backends.items() if status.get("ready") or status.get("installed")]
    return {
        "schema": "atomi.thermo_prior.gnn_status.v1",
        "ready_backends": ready,
        "backends": backends,
        "recommendation": (
            "Use AERIS when model_exists=true. Use CHGNet/MatGL only as screening priors; "
            "for uranium chlorides, build graph JSONL from relaxed POSCAR/CONTCAR and fine-tune/validate against local DFT before CALPHAD use."
        ),
    }

def _require_path(value: Path | None, *, flag: str, env_name: str) -> Path:
    if value is None:
        raise ValueError(
            f"Provide {flag}, set {env_name}, or configure aeris/environment_exports in "
            "ATOMI_HPC_CONFIG or ~/atomi_hpc/atomi_hpc_config.kit.local.json."
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atomi thermo-prior",
        description="Create and inspect provenance-rich thermodynamic prior JSON files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    line = sub.add_parser("line-compound", help="Write a line-compound prior for CALPHAD/MIVM diagnostics.")
    line.add_argument("--formula", required=True)
    line.add_argument("--component-a", required=True)
    line.add_argument("--component-b", required=True)
    line.add_argument("--label")
    line.add_argument("--gform-kj-mol", type=float, help="Formation Gibbs term relative to terminal salts.")
    line.add_argument("--formation-energy-ev-atom", type=float, help="Elemental-basis compound formation energy.")
    line.add_argument("--component-a-formation-energy-ev-atom", type=float)
    line.add_argument("--component-b-formation-energy-ev-atom", type=float)
    line.add_argument("--aeris-root", type=Path, default=_aeris_path_default("root", "ATOMI_AERIS_ROOT"))
    line.add_argument("--aeris-model", type=Path, default=_aeris_path_default("model", "ATOMI_AERIS_MODEL"))
    line.add_argument("--aeris-device", default=_aeris_device_default())
    line.add_argument("--dcp-form", type=float, default=0.0, help="Formation Cp correction in J/mol/K.")
    line.add_argument("--tref-k", type=float, default=298.15)
    line.add_argument("--temperature-min-k", type=float, help="Optional lower stability bound for this compound.")
    line.add_argument("--temperature-max-k", type=float, help="Optional upper stability bound for this compound.")
    line.add_argument("--uncertainty-kj-mol", type=float)
    line.add_argument("--source-label", default="manual")
    line.add_argument("--out", type=Path, required=True)

    cp = sub.add_parser("cp-solid", help="Write a placeholder solid Cp prior.")
    cp.add_argument("--formula", required=True)
    cp.add_argument("--cp-j-mol-k", type=float, required=True)
    cp.add_argument("--temperature-min-k", type=float, default=298.15)
    cp.add_argument("--temperature-max-k", type=float, default=1200.0)
    cp.add_argument("--uncertainty-j-mol-k", type=float)
    cp.add_argument("--source-label", default="manual_placeholder")
    cp.add_argument("--out", type=Path, required=True)

    validate = sub.add_parser("validate", help="Validate and print a prior JSON summary.")
    validate.add_argument("prior", type=Path)

    spec = sub.add_parser("line-spec", help="Print a benchmark-uq-phase --line-compound spec from a prior.")
    spec.add_argument("prior", type=Path)
    spec.add_argument("--default-tref-k", type=float, default=298.15)

    aeris = sub.add_parser("aeris-status", help="Check a configured local AERIS checkout/checkpoint.")
    aeris.add_argument("--aeris-root", type=Path, default=_aeris_path_default("root", "ATOMI_AERIS_ROOT"))
    aeris.add_argument("--aeris-model", type=Path, default=_aeris_path_default("model", "ATOMI_AERIS_MODEL"))
    aeris.add_argument("--aeris-device", default=_aeris_device_default())

    gnn = sub.add_parser("gnn-status", help="Check optional GNN/graph-prior prediction backends.")
    gnn.add_argument("--aeris-root", type=Path, default=_aeris_path_default("root", "ATOMI_AERIS_ROOT"))
    gnn.add_argument("--aeris-model", type=Path, default=_aeris_path_default("model", "ATOMI_AERIS_MODEL"))
    gnn.add_argument("--aeris-device", default=_aeris_device_default())

    return parser


def _line_compound_from_args(args: argparse.Namespace) -> dict[str, Any]:
    gform = args.gform_kj_mol
    source: dict[str, Any] = {"method": args.source_label}
    if gform is None:
        formation_energy = args.formation_energy_ev_atom
        if formation_energy is None and args.aeris_root and args.aeris_model:
            adapter = AerisAdapter(AerisConfig(root=args.aeris_root, model=args.aeris_model, device=args.aeris_device))
            prediction = adapter.predict_formation_energy_ev_atom(args.formula)
            formation_energy = float(prediction["formation_energy_ev_atom"])
            source = {"method": "aeris", **prediction}
        if formation_energy is None:
            raise ValueError("Provide --gform-kj-mol, --formation-energy-ev-atom, or configured --aeris-root/--aeris-model.")
        if args.component_a_formation_energy_ev_atom is None or args.component_b_formation_energy_ev_atom is None:
            raise ValueError(
                "Elemental-basis formation energies require --component-a-formation-energy-ev-atom "
                "and --component-b-formation-energy-ev-atom for salt-reference conversion."
            )
        gform = salt_reference_gform_kj_mol(
            formula=args.formula,
            component_a=args.component_a,
            component_b=args.component_b,
            formation_energy_ev_atom=formation_energy,
            component_a_formation_energy_ev_atom=args.component_a_formation_energy_ev_atom,
            component_b_formation_energy_ev_atom=args.component_b_formation_energy_ev_atom,
        )
        source = {
            **source,
            "formation_energy_ev_atom": formation_energy,
            "component_a_formation_energy_ev_atom": args.component_a_formation_energy_ev_atom,
            "component_b_formation_energy_ev_atom": args.component_b_formation_energy_ev_atom,
            "basis_conversion": "elemental formation energy to pseudo-binary salt-reference gform",
        }
    prior = write_line_compound_prior(
        out=args.out,
        formula=args.formula,
        component_a=args.component_a,
        component_b=args.component_b,
        label=args.label,
        gform_ref_kj_mol=float(gform),
        dcp_form_j_mol_k=args.dcp_form,
        tref_k=args.tref_k,
        temperature_min_k=args.temperature_min_k,
        temperature_max_k=args.temperature_max_k,
        uncertainty_kj_mol=args.uncertainty_kj_mol,
        source=source,
        notes=["Thermo-ML prior for screening; refine with DFT/phonopy/CALPHAD before final assessment."],
    )
    return prior


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "line-compound":
        prior = _line_compound_from_args(args)
        print(f"Wrote line-compound prior: {args.out}")
        print(prior["calphad_mivm"]["line_compound_spec"])
        return prior
    if args.command == "cp-solid":
        prior = write_cp_prior(
            out=args.out,
            formula=args.formula,
            cp_j_mol_k=args.cp_j_mol_k,
            temperature_min_k=args.temperature_min_k,
            temperature_max_k=args.temperature_max_k,
            uncertainty_j_mol_k=args.uncertainty_j_mol_k,
            source={"method": args.source_label},
        )
        print(f"Wrote Cp prior: {args.out}")
        return prior
    if args.command == "validate":
        prior = read_prior(args.prior)
        print(json.dumps(prior, indent=2, sort_keys=True))
        return prior
    if args.command == "line-spec":
        from .core import line_compound_spec_from_prior

        prior = read_prior(args.prior)
        compound = line_compound_spec_from_prior(prior, default_tref_k=args.default_tref_k)
        fields = [
            compound["label"],
            f"{compound['x_B']:.12g}",
            f"{compound['gform_ref_kJ_mol']:.12g}",
            f"{compound['dCp_form_J_mol_K']:.12g}",
            f"{compound['tref_K']:.12g}",
        ]
        if compound.get("tmin_K") is not None or compound.get("tmax_K") is not None:
            fields.append("" if compound.get("tmin_K") is None else f"{compound['tmin_K']:.12g}")
            fields.append("" if compound.get("tmax_K") is None else f"{compound['tmax_K']:.12g}")
        spec = ":".join(fields).rstrip(":")
        print(spec)
        return {"line_compound_spec": spec}
    if args.command == "aeris-status":
        root = _require_path(args.aeris_root, flag="--aeris-root", env_name="ATOMI_AERIS_ROOT")
        model = _require_path(args.aeris_model, flag="--aeris-model", env_name="ATOMI_AERIS_MODEL")
        adapter = AerisAdapter(AerisConfig(root=root, model=model, device=args.aeris_device))
        status = adapter.status()
        print(json.dumps(status, indent=2, sort_keys=True))
        return status
    if args.command == "gnn-status":
        status = _gnn_status(args)
        print(json.dumps(status, indent=2, sort_keys=True))
        return status
    return None


def console_main(argv: list[str] | None = None) -> None:
    main(argv)
    return None
