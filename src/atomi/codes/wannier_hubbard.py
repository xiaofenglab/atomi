"""Plan and audit projector-consistent QE Wannier+U workflows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ACCEPTED = "accepted"
TERMINAL_OPTIONAL = {"not_required", "rejected"}


@dataclass(frozen=True)
class TargetManifold:
    element: str
    manifold: str
    required: bool = True

    @property
    def target_id(self) -> str:
        return f"{self.element}:{self.manifold}"

    @property
    def w90_projection(self) -> str:
        shell = self.manifold[-1].lower()
        return f"{self.element}:{shell}"


def parse_target(value: str, *, required: bool = True) -> TargetManifold:
    match = re.fullmatch(r"\s*([A-Z][a-z]?)\s*:\s*([1-9]?[spdfgSPDFG])\s*", value)
    if not match:
        raise ValueError(
            f"invalid target {value!r}; use ELEMENT:MANIFOLD, for example U:5f or O:2p"
        )
    element, manifold = match.groups()
    return TargetManifold(element, manifold.lower(), required)


def _deduplicate_targets(targets: Iterable[TargetManifold]) -> list[TargetManifold]:
    result: list[TargetManifold] = []
    seen: set[str] = set()
    for target in targets:
        if target.target_id in seen:
            raise ValueError(f"duplicate target manifold: {target.target_id}")
        seen.add(target.target_id)
        result.append(target)
    if not result:
        raise ValueError("at least one --target or --optional-target is required")
    return result


def workflow_stages() -> list[dict[str, Any]]:
    return [
        {
            "stage": 0,
            "name": "parent",
            "gate": "Converged structure/electronic branch with recorded code, functional, pseudopotentials, cutoffs, k mesh, spin order, occupations, and gap/metal character.",
        },
        {
            "stage": 1,
            "name": "manifold-definition",
            "gate": "Declare each correlated element/manifold, orbital count per site and spin, and whether ligand states are separate targets.",
        },
        {
            "stage": 2,
            "name": "bloch-space",
            "gate": "NSCF basis contains the complete target space; Fermi reference, projectability, and per-k singular-value rank are accepted.",
        },
        {
            "stage": 3,
            "name": "wannier-projectors",
            "gate": "WF counts, site centres, angular character, spreads, interpolation, and occupation eigenvalues pass for every target.",
        },
        {
            "stage": 4,
            "name": "export-smoke",
            "gate": "wannier2pw writes projector files and a zero-U HUBBARD(wf) occupation test reads them. Export alone is not scientific acceptance.",
        },
        {
            "stage": 5,
            "name": "response",
            "gate": "U/J response records code/commit, perturbation protocol, fit diagnostics, and the exact same projector identifier used in application.",
        },
        {
            "stage": 6,
            "name": "application",
            "gate": "No-U, per-manifold U-only, and combined-U tests preserve the intended branch and report energies, gap, moments, occupations, and convergence.",
        },
        {
            "stage": 7,
            "name": "force-relaxation",
            "gate": "Force/stress support for the chosen projector implementation is proven before relaxation; otherwise restrict the result to fixed-geometry single points.",
        },
        {
            "stage": 8,
            "name": "archive",
            "gate": "Manifest, inputs, hashes, projector QA, response data, application comparison, and decision are stored with compute/report locations separated.",
        },
    ]


def _target_manifest(target: TargetManifold) -> dict[str, Any]:
    projector_id = f"{target.target_id}:wannier90-mlwf:pending"
    return {
        "target": target.target_id,
        "required": target.required,
        "decision": "required" if target.required else "pending",
        "rationale": "",
        "projector": {
            "id": projector_id,
            "family": "wannier90-mlwf/wannier2pw",
            "trial_projection": target.w90_projection,
            "status": "pending",
            "expected_per_site_spin": None,
            "accepted_count": None,
            "window_definition": "",
            "projectability_guard": "pending",
            "rank_guard": "pending",
            "centre_spread_guard": "pending",
            "real_space_guard": "pending",
            "interpolation_guard": "pending",
            "source_paths": [],
        },
        "response": {
            "id": f"{target.target_id}:response:pending",
            "status": "pending",
            "method": "matched-projector linear response or cRPA",
            "projector_id": projector_id,
            "projector_matched": False,
            "value_eV": None,
            "fit_guard": "pending",
            "code_version_or_commit": "",
            "source_paths": [],
        },
        "application": {
            "status": "pending",
            "projector_id": projector_id,
            "response_id": f"{target.target_id}:response:pending",
            "zero_u_read_guard": "pending",
            "u_only_guard": "pending",
            "combined_u_guard": "pending",
            "force_stress_scope": "unverified",
            "source_paths": [],
        },
    }


def prepare_plan(
    outdir: Path,
    *,
    system: str,
    required_targets: Sequence[str],
    optional_targets: Sequence[str] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    targets = _deduplicate_targets(
        [parse_target(value) for value in required_targets]
        + [parse_target(value, required=False) for value in optional_targets]
    )
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {outdir}; use --overwrite")
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "atomi.qe_wannier_hubbard_manifest.v1",
        "system": system,
        "route": "QE Wannier90 MLWF projectors plus explicitly labeled response",
        "overall_decision": "pending",
        "parent": {
            "status": "pending",
            "code_version": "",
            "functional": "",
            "pseudopotentials": [],
            "structure": "",
            "k_mesh": "",
            "cutoffs": "",
            "spin_order": "",
            "occupation_branch": "",
            "gap_or_metal_guard": "pending",
            "source_paths": [],
        },
        "targets": [_target_manifest(target) for target in targets],
        "notes": [
            "A .hub file is a projector artifact, not a numerical Hubbard U.",
            "Stock hp.x atomic/ortho-atomic U is not a matched MLWF U.",
            "A U borrowed across projector definitions must be labeled as an application comparison, not a projector-consistent demonstration.",
            "Molecular-orbital or spectroscopy comparisons can validate character but cannot determine solid-state screening U.",
        ],
    }
    plan = {
        "schema": "atomi.qe_wannier_hubbard_plan.v1",
        "system": system,
        "targets": [
            {
                "target": target.target_id,
                "required": target.required,
                "w90_projection": target.w90_projection,
            }
            for target in targets
        ],
        "stages": workflow_stages(),
        "completion_rule": "Every required target must have accepted projector, matched response, and application gates; optional targets require an explicit accepted/rejected/not_required decision with rationale.",
    }
    (outdir / "wannier_hubbard_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    (outdir / "wannier_hubbard_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    stage_lines = "\n".join(
        f"{item['stage']}. **{item['name']}**: {item['gate']}" for item in workflow_stages()
    )
    target_lines = "\n".join(
        f"- `{target.target_id}` ({'required' if target.required else 'optional'}), W90 trial `{target.w90_projection}`"
        for target in targets
    )
    (outdir / "WORKFLOW.md").write_text(
        f"""# {system} projector-consistent Wannier+U workflow

## Targets

{target_lines}

## Stages and gates

{stage_lines}

Update `wannier_hubbard_manifest.json` with immutable projector, response, and
application identifiers. Run `hubbard-u-workflow qe-wannier-audit --root .`
before promotion. Do not enter a numerical U unless its response operator and
the DFT+U application use the recorded projector identifier.
""",
        encoding="utf-8",
    )
    return plan


def _projection_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(
        r"begin\s+projections(.*?)end\s+projections", text, flags=re.IGNORECASE | re.DOTALL
    )
    values: list[str] = []
    for block in blocks:
        for line in block.splitlines():
            body = line.split("!", 1)[0].split("#", 1)[0].strip()
            if body:
                values.append(re.sub(r"\s+", "", body))
    return values


def _w90_completed(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return "Final State" in text and (
        "All done" in text or "Wannier90 exiting" in text or "Final Spread" in text
    )


def discover_artifacts(root: Path) -> dict[str, Any]:
    win_files = sorted(root.rglob("*.win"))
    wout_files = sorted(root.rglob("*.wout"))
    hub_files = sorted(path for path in root.rglob("*.hub*") if path.is_file())
    response_files = sorted(
        {
            *root.rglob("parameters.out"),
            *root.rglob("*hp*.out"),
            *root.rglob("*response*.json"),
        }
    )
    projections = sorted({value for path in win_files for value in _projection_blocks(path)})
    return {
        "root": str(root),
        "win_files": [str(path.relative_to(root)) for path in win_files],
        "projections": projections,
        "wout_files": [str(path.relative_to(root)) for path in wout_files],
        "w90_completed": sum(_w90_completed(path) for path in wout_files),
        "hub_files": [str(path.relative_to(root)) for path in hub_files],
        "hub_count": len(hub_files),
        "response_candidates": [str(path.relative_to(root)) for path in response_files],
    }


def _status_ok(value: Any) -> bool:
    return str(value).strip().lower() == ACCEPTED


def _target_audit(
    row: dict[str, Any], artifacts: dict[str, Any], *, combined_required: bool
) -> dict[str, Any]:
    target = parse_target(str(row.get("target", "")), required=bool(row.get("required", True)))
    projector = row.get("projector") or {}
    response = row.get("response") or {}
    application = row.get("application") or {}
    projector_id = str(projector.get("id", ""))
    response_id = str(response.get("id", ""))
    projection_present = target.w90_projection in artifacts["projections"]
    projector_guard_keys = (
        "projectability_guard",
        "rank_guard",
        "centre_spread_guard",
        "real_space_guard",
        "interpolation_guard",
    )
    projector_guards_ok = all(_status_ok(projector.get(key)) for key in projector_guard_keys)
    projector_ok = all(
        (
            _status_ok(projector.get("status")),
            projector_guards_ok,
            projection_present,
            artifacts["w90_completed"] > 0,
            artifacts["hub_count"] > 0,
            bool(projector_id),
        )
    )
    value = response.get("value_eV")
    value_ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    response_ok = all(
        (
            _status_ok(response.get("status")),
            value_ok,
            bool(response_id),
            bool(projector_id),
            response.get("projector_id") == projector_id,
            response.get("projector_matched") is True,
            bool(str(response.get("code_version_or_commit", "")).strip()),
        )
    )
    application_ok = all(
        (
            _status_ok(application.get("status")),
            application.get("projector_id") == projector_id,
            application.get("response_id") == response_id,
            _status_ok(application.get("zero_u_read_guard")),
            _status_ok(application.get("u_only_guard")),
            not combined_required or _status_ok(application.get("combined_u_guard")),
        )
    )
    required = bool(row.get("required", True))
    decision = str(row.get("decision", "pending")).strip().lower()
    rationale = str(row.get("rationale", "")).strip()
    optional_terminal = not required and decision in TERMINAL_OPTIONAL and bool(rationale)
    complete = projector_ok and response_ok and application_ok
    warnings: list[str] = []
    if not projection_present:
        warnings.append(f"no `{target.w90_projection}` trial projection found in .win files")
    if artifacts["hub_count"] and not response_ok:
        warnings.append("projector export exists, but no accepted projector-matched numerical U is recorded")
    if value_ok and response.get("projector_id") != projector_id:
        warnings.append("numerical U projector_id does not match the accepted application projector")
    if not required and decision in TERMINAL_OPTIONAL and not rationale:
        warnings.append("optional target decision needs a scientific rationale")
    return {
        "target": target.target_id,
        "required": required,
        "decision": decision,
        "projector": {"accepted": projector_ok, "projection_present": projection_present},
        "response": {"accepted": response_ok, "value_eV": value},
        "application": {"accepted": application_ok},
        "complete": complete,
        "resolved": complete or optional_terminal,
        "warnings": warnings,
    }


def audit_workflow(root: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = (manifest_path or root / "wannier_hubbard_manifest.json").resolve()
    artifacts = discover_artifacts(root)
    if not manifest_path.is_file():
        return {
            "schema": "atomi.qe_wannier_hubbard_audit.v1",
            "status": "incomplete",
            "manifest": str(manifest_path),
            "manifest_found": False,
            "artifacts": artifacts,
            "warnings": [
                "No provenance manifest found; artifacts alone cannot establish a projector-consistent U.",
                ".hub files are projector coefficients, not numerical Hubbard values.",
            ],
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "atomi.qe_wannier_hubbard_manifest.v1":
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')!r}")
    target_rows = manifest.get("targets", [])
    required_count = sum(bool(row.get("required", True)) for row in target_rows)
    targets = [
        _target_audit(row, artifacts, combined_required=required_count > 1)
        for row in target_rows
    ]
    parent_ok = _status_ok((manifest.get("parent") or {}).get("status"))
    status = "complete" if parent_ok and targets and all(row["resolved"] for row in targets) else "incomplete"
    warnings = [warning for row in targets for warning in row["warnings"]]
    if not parent_ok:
        warnings.insert(0, "parent electronic/structural branch is not accepted")
    return {
        "schema": "atomi.qe_wannier_hubbard_audit.v1",
        "status": status,
        "manifest": str(manifest_path),
        "manifest_found": True,
        "system": manifest.get("system", ""),
        "parent_accepted": parent_ok,
        "targets": targets,
        "artifacts": artifacts,
        "warnings": warnings,
    }
