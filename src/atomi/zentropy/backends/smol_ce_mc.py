"""Lazy smol CE-MC backend adapter skeleton."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

from .base import unavailable_run_result


class SmolCEMCBackend:
    name = "smol_ce_mc"
    install_hint = (
        "Install in a separate environment, e.g. `python -m pip install smol` "
        "then `python -m pip install -e '.[defects,smol]'`."
    )

    def _import_status(self) -> tuple[bool, str, str]:
        try:
            import smol  # noqa: F401
        except Exception as exc:
            return False, "", str(exc)
        try:
            version = importlib.metadata.version("smol")
        except importlib.metadata.PackageNotFoundError:
            version = getattr(smol, "__version__", "unknown")  # type: ignore[name-defined]
        return True, version, ""

    def available(self) -> bool:
        ok, _version, _error = self._import_status()
        return ok

    def capability_report(self) -> dict[str, Any]:
        ok, version, error = self._import_status()
        report = {
            "backend": self.name,
            "available": ok,
            "version": version or "not_found",
            "capabilities": {
                "cluster_expansion": ok,
                "canonical_mc": ok,
                "semigrand_mc": ok,
                "charge_neutral_semigrand_mc": "backend_dependent" if ok else False,
                "thermodynamic_integration": "atomi_wrapper_needed" if ok else False,
            },
            "install_hint": "" if ok else self.install_hint,
        }
        if error:
            report["error"] = error
        return report

    def prepare(self, config: dict[str, Any], ensemble: Any) -> dict[str, Any]:
        if not self.available():
            return unavailable_run_result(self.name, "smol is not importable", self.install_hint)
        return {"backend": self.name, "status": "prepared", "config": config, "n_records": len(ensemble or [])}

    def run(self, config: dict[str, Any], ensemble: Any, output_dir: Path) -> dict[str, Any]:
        if not self.available():
            return unavailable_run_result(self.name, "smol is not importable", self.install_hint)
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "backend": self.name,
            "status": "stub",
            "output_dir": str(output_dir),
            "next_step": "map CETrainingSet to smol ClusterSubspace/ClusterExpansion and MC samplers",
        }

    def collect(self, output_dir: Path) -> dict[str, Any]:
        return {"backend": self.name, "status": "stub", "output_dir": str(output_dir)}
