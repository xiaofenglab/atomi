"""POCC-informed motif-Hamiltonian Monte Carlo backend skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PoccMotifMCBackend:
    name = "pocc_motif_mc"

    def available(self) -> bool:
        return True

    def capability_report(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "available": True,
            "maturity": "schema_ready",
            "capabilities": {
                "motif_hamiltonian": True,
                "charge_neutral_moves": "planned",
                "large_lattice_sampling": "planned",
                "thermodynamic_integration": "planned",
                "uses_orbit_degeneracy_as_mc_weight": False,
            },
            "double_counting_guard": (
                "Use POCC degeneracies for finite logsum/grouping/reference populations; "
                "do not multiply explicit-lattice MC samples by motif embedding degeneracy."
            ),
        }

    def prepare(self, config: dict[str, Any], ensemble: Any) -> dict[str, Any]:
        return {"backend": self.name, "status": "prepared", "config": config, "n_records": len(ensemble or [])}

    def run(self, config: dict[str, Any], ensemble: Any, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "backend": self.name,
            "status": "stub",
            "output_dir": str(output_dir),
            "next_step": "fit a motif-count Hamiltonian before enabling MC sampling",
        }

    def collect(self, output_dir: Path) -> dict[str, Any]:
        return {"backend": self.name, "status": "stub", "output_dir": str(output_dir)}
