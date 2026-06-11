"""POCC/GQCA population-vector backend skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PoccGQCAPopulationBackend:
    name = "pocc_gqca_population_vector"

    def available(self) -> bool:
        return True

    def capability_report(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "available": True,
            "maturity": "schema_ready",
            "capabilities": {
                "finite_population_vector": True,
                "kl_entropy_loss_to_reference": True,
                "oxygen_grand_potential": True,
                "spatial_mc": False,
            },
            "notes": [
                "Uses POCC/enum motif classes, degeneracies, composition vectors, and reference populations.",
                "Appropriate before explicit real-space MC when motif compatibility can be mean-field.",
            ],
        }

    def prepare(self, config: dict[str, Any], ensemble: Any) -> dict[str, Any]:
        return {"backend": self.name, "status": "prepared", "config": config, "n_records": len(ensemble or [])}

    def run(self, config: dict[str, Any], ensemble: Any, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "backend": self.name,
            "status": "stub",
            "output_dir": str(output_dir),
            "next_step": "connect constrained population-vector minimizer to the common ThermoSurface schema",
        }

    def collect(self, output_dir: Path) -> dict[str, Any]:
        return {"backend": self.name, "status": "stub", "output_dir": str(output_dir)}
