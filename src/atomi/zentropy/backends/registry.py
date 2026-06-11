"""Registry for defect thermodynamic large-state backends."""

from __future__ import annotations

from typing import Any

from .casm_ce_mc_ti import CASMCEMCTIBackend
from .pocc_gqca import PoccGQCAPopulationBackend
from .pocc_motif_mc import PoccMotifMCBackend
from .smol_ce_mc import SmolCEMCBackend


def backend_registry() -> dict[str, Any]:
    backends = [
        PoccGQCAPopulationBackend(),
        PoccMotifMCBackend(),
        SmolCEMCBackend(),
        CASMCEMCTIBackend(),
    ]
    return {backend.name: backend for backend in backends}


def get_backend(name: str) -> Any:
    registry = backend_registry()
    if name not in registry:
        choices = ", ".join(sorted(registry))
        raise KeyError(f"Unknown defect thermodynamics backend {name!r}. Choices: {choices}")
    return registry[name]


def backend_names() -> list[str]:
    return sorted(backend_registry())
