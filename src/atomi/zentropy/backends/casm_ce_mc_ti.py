"""Lazy CASM CE-MC/TI backend adapter skeleton."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .base import unavailable_run_result


class CASMCEMCTIBackend:
    name = "casm_ce_mc_ti"
    install_hint = (
        "Install CASM in an isolated conda/module/container environment; keep casm-cpp out of "
        "base Atomi dependencies. Then run `atomi-defects backend doctor --backend casm_ce_mc_ti`."
    )

    def _casm_python_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {}
        try:
            import casm  # noqa: F401

            status["casm_python"] = True
        except Exception as exc:
            status["casm_python"] = False
            status["casm_python_error"] = str(exc)
        for module_name, key in (
            ("libcasm.xtal", "libcasm_xtal"),
            ("libcasm.monte", "libcasm_monte"),
            ("libcasm.clexmonte", "libcasm_clexmonte"),
        ):
            try:
                __import__(module_name)
                status[key] = True
            except Exception as exc:
                status[key] = False
                status[f"{key}_error"] = str(exc)
        return status

    def available(self) -> bool:
        status = self._casm_python_status()
        return bool(
            status.get("casm_python")
            or status.get("libcasm_xtal")
            or status.get("libcasm_monte")
            or shutil.which("casm")
            or shutil.which("ccasm")
        )

    def capability_report(self) -> dict[str, Any]:
        status = self._casm_python_status()
        casm_cli = shutil.which("casm")
        ccasm_cli = shutil.which("ccasm")
        available = self.available()
        return {
            "backend": self.name,
            "available": available,
            "availability": "available" if available else "missing",
            "casm_cli": casm_cli or "",
            "ccasm_cli": ccasm_cli or "",
            **status,
            "capabilities": {
                "project_export": available,
                "occupational_ce": available,
                "monte_carlo": bool(status.get("libcasm_monte") or ccasm_cli or casm_cli),
                "thermodynamic_integration": "external_casm_workflow" if available else False,
            },
            "install_hint": "" if available else self.install_hint,
        }

    def prepare(self, config: dict[str, Any], ensemble: Any) -> dict[str, Any]:
        if not self.available():
            return unavailable_run_result(self.name, "CASM Python modules/CLI are not visible", self.install_hint)
        return {"backend": self.name, "status": "prepared", "config": config, "n_records": len(ensemble or [])}

    def run(self, config: dict[str, Any], ensemble: Any, output_dir: Path) -> dict[str, Any]:
        if not self.available():
            return unavailable_run_result(self.name, "CASM Python modules/CLI are not visible", self.install_hint)
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "backend": self.name,
            "status": "stub",
            "output_dir": str(output_dir),
            "next_step": "write a CASM project spec and launch configured CASM command/container",
        }

    def collect(self, output_dir: Path) -> dict[str, Any]:
        return {"backend": self.name, "status": "stub", "output_dir": str(output_dir)}
