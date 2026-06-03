"""Optional AERIS adapter for formation-energy priors."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AerisConfig:
    root: Path
    model: Path
    device: str = "cpu"


class AerisAdapter:
    """Thin optional adapter around a local AERIS checkout and checkpoint."""

    def __init__(self, config: AerisConfig):
        self.config = config

    def status(self) -> dict[str, Any]:
        root = self.config.root
        model = self.config.model
        return {
            "root": str(root),
            "model": str(model),
            "root_exists": root.exists(),
            "aeris_py_exists": (root / "aeris.py").exists(),
            "model_exists": model.exists(),
            "ready": root.exists() and (root / "aeris.py").exists() and model.exists(),
        }

    def _load_module(self):
        status = self.status()
        if not status["ready"]:
            raise FileNotFoundError(f"AERIS adapter is not ready: {status}")
        root_str = str(self.config.root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return importlib.import_module("aeris")

    def predict_formation_energy_ev_atom(self, formula: str) -> dict[str, Any]:
        import pandas as pd

        aeris = self._load_module()
        model, feature_names, scaler = aeris.load_structure_model(str(self.config.model), device=self.config.device)
        df = pd.DataFrame([{"composition": formula, "structure": ""}])
        features, _ = aeris.build_features_in_ckpt_order(df, feature_names, target_col=None)
        value = aeris.predict_energy(features[0].tolist(), model, scaler=scaler, device=self.config.device)
        if isinstance(value, dict):
            energy = float(value.get("per_atom_eV_per_atom", value.get("energy_per_atom", value.get("value"))))
        else:
            energy = float(value)
        return {
            "formation_energy_ev_atom": energy,
            "model": str(self.config.model),
            "root": str(self.config.root),
            "device": self.config.device,
        }
