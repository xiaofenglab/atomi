"""Post-setup OpenMolcas RAS3 identity and mixing diagnostics."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from atomi.qchem.molcas_diagnostics import (
    HANDOFF_SCHEMA,
    RasscfInputBlock,
    RasscfLogModule,
    _component_score,
    _read_text,
    _sha256,
    _shell_matches,
    _target_score,
    build_diagnostics,
    build_moccheck_handoff,
    match_log_module,
    parse_compact_mo_listing,
    parse_full_mo_matrix,
    parse_rasscf_input_blocks,
    parse_rasscf_log_modules,
)


SCHEMA = "atomi.molcas_moccheckafter.v1"
STATUS_ORDER = {
    "unavailable": 5,
    "drift_risk": 4,
    "review": 3,
    "mixed_retained": 2,
    "stable_identity": 1,
}


def _rows_for_symmetry(
    module: RasscfLogModule, symmetry: int
) -> tuple[list[dict[str, Any]], str]:
    rows = parse_full_mo_matrix(module.text, symmetry=symmetry)
    if rows:
        return rows, "full"
    rows = parse_compact_mo_listing(module.text, symmetry=symmetry)
    return rows, "compact" if rows else "unavailable"


def _best_component_assignment(
    rows: list[dict[str, Any]],
    components: list[dict[str, Any]],
    *,
    atom: str,
) -> list[dict[str, Any]]:
    if not rows or not components:
        return []
    count = min(len(rows), len(components))
    best_rows: tuple[dict[str, Any], ...] | None = None
    best_total = -1.0
    for candidate in itertools.permutations(rows, count):
        total = sum(
            _component_score(row, atom=atom, ao=str(component["ao"]))
            for component, row in zip(components[:count], candidate)
        )
        if total > best_total:
            best_total = total
            best_rows = candidate
    if best_rows is None:
        return []
    return [
        {
            "component": str(component["ao"]),
            "baseline_source_mo": int(component["source_mo"]),
            "baseline_score": float(component["baseline_score"]),
            "final_slot": int(row["mo"]),
            "score": _component_score(row, atom=atom, ao=str(component["ao"])),
            "occupation": float(row["occupation"]),
            "energy_hartree": float(row["energy"]),
        }
        for component, row in zip(components[:count], best_rows)
    ]


def _dominant_non_target_terms(
    rows: list[dict[str, Any]],
    *,
    atom: str,
    shells: list[str],
    limit: int = 5,
) -> tuple[list[dict[str, Any]], float, float]:
    totals: dict[tuple[str, str], float] = {}
    all_coeff2 = 0.0
    ligand_coeff2 = 0.0
    central_other_coeff2 = 0.0
    for row in rows:
        for term in row["terms"]:
            value = float(term["coeff2"])
            all_coeff2 += value
            term_atom = str(term["atom"])
            term_ao = str(term["ao"])
            is_target = term_atom.lower() == atom.lower() and any(
                _shell_matches(term_ao, shell) for shell in shells
            )
            if is_target:
                continue
            totals[(term_atom, term_ao)] = totals.get((term_atom, term_ao), 0.0) + value
            if term_atom.lower() == atom.lower():
                central_other_coeff2 += value
            else:
                ligand_coeff2 += value
    denominator = all_coeff2 or 1.0
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:limit]
    return (
        [
            {
                "atom": key[0],
                "ao": key[1],
                "display_share": value / denominator,
            }
            for key, value in ranked
        ],
        ligand_coeff2 / denominator,
        central_other_coeff2 / denominator,
    )


def analyze_ras3_target(
    rows: list[dict[str, Any]],
    target: dict[str, Any],
    *,
    atom: str,
    shells: list[str],
    listing_format: str,
) -> dict[str, Any]:
    """Assess one symmetry's final RAS3 subspace without fixing MO ordering."""

    slots = [int(value) for value in target["final_ras3_slots"]]
    row_by_mo = {int(row["mo"]): row for row in rows}
    active_rows = [row_by_mo[slot] for slot in slots if slot in row_by_mo]
    if len(active_rows) != len(slots):
        return {
            "symmetry": int(target["symmetry"]),
            "symmetry_label": str(target.get("symmetry_label", "")),
            "status": "unavailable",
            "reason": "One or more final RAS3 slots are absent from the printed MO table",
            "final_ras3_slots": slots,
            "available_slots": [int(row["mo"]) for row in active_rows],
            "ao_listing_format": listing_format,
        }

    components = list(target.get("components", []))
    assignment = _best_component_assignment(active_rows, components, atom=atom)
    baseline_total = sum(float(item["baseline_score"]) for item in components) or 1.0
    captured_total = sum(float(item["score"]) for item in assignment)
    capture_ratio = captured_total / baseline_total
    shell_scores = [_target_score(row, atom=atom, shells=shells) for row in active_rows]
    mean_shell_score = sum(shell_scores) / len(shell_scores) if shell_scores else 0.0
    non_target, ligand_share, central_other_share = _dominant_non_target_terms(
        active_rows, atom=atom, shells=shells
    )

    if mean_shell_score < 0.15 or capture_ratio < 0.25:
        status = "drift_risk"
    elif mean_shell_score < 0.30 or capture_ratio < 0.50:
        status = "review"
    elif ligand_share >= 0.10 or central_other_share >= 0.10 or mean_shell_score < 0.85:
        status = "mixed_retained"
    else:
        status = "stable_identity"

    outside_candidates: list[dict[str, Any]] = []
    outside_rows = [row for row in rows if int(row["mo"]) not in slots]
    for item in assignment:
        ranked = sorted(
            (
                (
                    _component_score(row, atom=atom, ao=str(item["component"])),
                    row,
                )
                for row in outside_rows
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not ranked:
            continue
        score, row = ranked[0]
        if score >= 0.20 and score >= float(item["score"]) + 0.15:
            outside_candidates.append(
                {
                    "component": item["component"],
                    "candidate_mo": int(row["mo"]),
                    "candidate_score": score,
                    "current_ras3_slot": int(item["final_slot"]),
                    "current_score": float(item["score"]),
                    "conditional_alter": [
                        int(target["symmetry"]),
                        int(row["mo"]),
                        int(item["final_slot"]),
                    ],
                }
            )

    if status == "drift_risk":
        recommendation = (
            "The intended AO subspace is no longer retained. Verify the setup ALTER map and "
            "the JobIph actually inherited by this block. If setup is healthy but this later "
            "optimization drifts reproducibly, test a minimal SUPSYM label on the reported "
            "final RAS3 slots. Apply a proposed ALTER only when restarting from the same "
            "orbital file whose MO numbering was analyzed."
        )
    elif status == "review":
        recommendation = (
            "Repeat or inspect the short RASSCF probe and root characters before constraining "
            "orbitals; this result is between clear retention and clear identity loss."
        )
    elif status == "mixed_retained":
        recommendation = (
            "The intended central-atom subspace remains present with non-target character. "
            "Treat this as potential physical metal-ligand mixing and do not add SUPSYM unless "
            "root instability or destructive active-space exchange is independently observed."
        )
    else:
        recommendation = "The intended RAS3 identity is retained; no new ALTER or SUPSYM is indicated."

    return {
        "symmetry": int(target["symmetry"]),
        "symmetry_label": str(target.get("symmetry_label", "")),
        "status": status,
        "final_ras3_slots": slots,
        "component_assignment": assignment,
        "baseline_normalized_capture": capture_ratio,
        "mean_target_shell_share": mean_shell_score,
        "ligand_share": ligand_share,
        "central_atom_other_shell_share": central_other_share,
        "ras3_occupation": sum(float(row["occupation"]) for row in active_rows),
        "dominant_non_target_aos": non_target,
        "outside_target_candidates": outside_candidates,
        "optional_final_slot_supsym_group": slots,
        "ao_listing_format": listing_format,
        "recommendation": recommendation,
        "weight_guard": (
            "Shares are coefficient-squared diagnostics over printed AO terms, not Mulliken or "
            "Loewdin populations; compact listings omit small coefficients."
        ),
    }


def _load_or_build_handoff(
    inp_path: Path,
    output_path: Path,
    handoff_path: Path | None,
) -> tuple[dict[str, Any], str]:
    candidate = handoff_path
    if candidate is None:
        automatic = inp_path.with_name(f"{inp_path.stem}.moccheck-handoff.json")
        candidate = automatic if automatic.exists() else None
    if candidate is not None:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if payload.get("schema") != HANDOFF_SCHEMA:
            raise ValueError(f"Unsupported moccheck handoff schema in {candidate}")
        expected_hash = payload.get("input", {}).get("sha256")
        if expected_hash != _sha256(inp_path):
            raise ValueError(
                "The moccheck handoff does not match the current input file; regenerate it"
            )
        return payload, f"loaded {candidate}"

    results = build_diagnostics(inp_path, output_path)
    return build_moccheck_handoff(results, inp_path, output_path), "built in memory"


def build_after_diagnostic(
    inp_path: Path,
    output_path: Path,
    *,
    handoff_path: Path | None = None,
    blocks_after: int = 4,
) -> dict[str, Any]:
    handoff, handoff_source = _load_or_build_handoff(inp_path, output_path, handoff_path)
    input_blocks = parse_rasscf_input_blocks(_read_text(inp_path))
    modules = parse_rasscf_log_modules(_read_text(output_path))
    setup_index = int(handoff["setup_input_block"]["index"])
    selected_blocks = [
        block
        for block in input_blocks
        if setup_index <= block.index <= setup_index + blocks_after
    ]
    target_atom = handoff.get("target_atom")
    shells = [str(value) for value in handoff.get("target_shells", [])]
    if not target_atom or not shells:
        raise ValueError("The moccheck handoff has no unambiguous target atom/shell intent")

    block_results: list[dict[str, Any]] = []
    for block in selected_blocks:
        try:
            module, mapping = match_log_module(block, modules)
        except ValueError as exc:
            block_results.append(
                {
                    "input_block": _block_summary(block),
                    "status": "pending_or_unavailable",
                    "reason": str(exc),
                }
            )
            continue
        symmetry_results = []
        formats = set()
        for target in handoff.get("targets", []):
            rows, listing_format = _rows_for_symmetry(module, int(target["symmetry"]))
            formats.add(listing_format)
            symmetry_results.append(
                analyze_ras3_target(
                    rows,
                    target,
                    atom=str(target_atom),
                    shells=shells,
                    listing_format=listing_format,
                )
            )
        available_statuses = [
            item["status"] for item in symmetry_results if item["status"] in STATUS_ORDER
        ]
        status = (
            max(available_statuses, key=lambda value: STATUS_ORDER[value])
            if available_statuses
            else "unavailable"
        )
        total_occupation = sum(
            float(item.get("ras3_occupation", 0.0)) for item in symmetry_results
        )
        max_ras3 = block.nactel[2] if len(block.nactel) >= 3 else None
        occupation_warning = None
        if max_ras3 is not None and total_occupation > max_ras3 + 0.10:
            occupation_warning = (
                f"Printed total RAS3 occupation {total_occupation:.3f} exceeds the input "
                f"maximum {max_ras3}; verify block matching and orbital partition."
            )
        block_results.append(
            {
                "input_block": _block_summary(block),
                "output_module": {
                    "index": module.index,
                    "start_line": module.start_line,
                    "end_line": module.end_line,
                    "mapping": mapping,
                },
                "status": status,
                "total_ras3_occupation": total_occupation,
                "maximum_ras3_electrons": max_ras3,
                "occupation_warning": occupation_warning,
                "ao_listing_formats": sorted(formats),
                "symmetries": symmetry_results,
            }
        )

    return {
        "schema": SCHEMA,
        "input_file": str(inp_path.resolve()),
        "output_file": str(output_path.resolve()),
        "handoff_source": handoff_source,
        "setup_input_block_index": setup_index,
        "blocks_after_setup_requested": blocks_after,
        "target_atom": target_atom,
        "target_shells": shells,
        "blocks": block_results,
        "interpretation_guard": handoff["interpretation_guard"],
    }


def _block_summary(block: RasscfInputBlock) -> dict[str, Any]:
    return {
        "index": block.index,
        "title": block.title,
        "start_line": block.start_line,
        "end_line": block.end_line,
        "symmetry": block.symmetry,
        "spin_multiplicity": block.spin_multiplicity,
        "nactel": list(block.nactel),
        "is_setup": block.has_alter or block.has_supsym,
    }


def print_after_diagnostic(result: dict[str, Any]) -> None:
    print("MOCCHECKAFTER - post-ALTER RAS3 identity and mixing audit")
    print(f"Input : {result['input_file']}")
    print(f"Output: {result['output_file']}")
    print(f"Baseline: {result['handoff_source']}")
    print(
        f"Target: {result['target_atom']} shells [{', '.join(result['target_shells'])}] | "
        f"setup block {result['setup_input_block_index']}"
    )
    for block in result["blocks"]:
        input_block = block["input_block"]
        print("\n" + "=" * 78)
        print(
            f"RASSCF #{input_block['index']} | {input_block['title'] or '(untitled)'} | "
            f"{block['status']}"
        )
        if "output_module" not in block:
            print(f"  {block['reason']}")
            continue
        print(
            f"  Output module {block['output_module']['index']} lines "
            f"{block['output_module']['start_line']}-{block['output_module']['end_line']}"
        )
        print(
            f"  Total RAS3 occupation: {block['total_ras3_occupation']:.3f}"
            + (
                f" / maximum {block['maximum_ras3_electrons']}"
                if block["maximum_ras3_electrons"] is not None
                else ""
            )
        )
        if block.get("occupation_warning"):
            print(f"  WARNING: {block['occupation_warning']}")
        for item in block["symmetries"]:
            label = item.get("symmetry_label") or "unlabeled"
            print(
                f"  Sym {item['symmetry']} ({label}) slots {item.get('final_ras3_slots', [])}: "
                f"{item['status']}"
            )
            if item["status"] == "unavailable":
                print(f"    {item['reason']}")
                continue
            print(
                f"    target-shell share={item['mean_target_shell_share']:.3f}, "
                f"baseline capture={item['baseline_normalized_capture']:.3f}, "
                f"ligand share={item['ligand_share']:.3f}"
            )
            for assignment in item["component_assignment"]:
                print(
                    f"    {assignment['component']}: source MO {assignment['baseline_source_mo']} "
                    f"-> slot {assignment['final_slot']}, score "
                    f"{assignment['baseline_score']:.3f}->{assignment['score']:.3f}, "
                    f"occ={assignment['occupation']:.3f}"
                )
            for candidate in item["outside_target_candidates"]:
                print(
                    f"    OUTSIDE CANDIDATE {candidate['component']}: MO "
                    f"{candidate['candidate_mo']} score={candidate['candidate_score']:.3f}; "
                    f"conditional ALTER {' '.join(str(v) for v in candidate['conditional_alter'])}"
                )
            if item["dominant_non_target_aos"]:
                labels = ", ".join(
                    f"{term['atom']}:{term['ao']} {100.0 * term['display_share']:.1f}%"
                    for term in item["dominant_non_target_aos"][:3]
                )
                print(f"    non-target AO character: {labels}")
            print(f"    Decision: {item['recommendation']}")
    print("\nGuard: " + result["interpretation_guard"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moccheckafter",
        description=(
            "Audit whether intended RAS3 AO character is retained after the ALTER/SUPSYM "
            "setup and in the next state-specific RASSCF blocks."
        ),
    )
    parser.add_argument("inp", type=Path, help="OpenMolcas .inp file")
    parser.add_argument("output", type=Path, help="Matching .log/.out file, optionally .gz")
    parser.add_argument(
        "--handoff",
        type=Path,
        default=None,
        help=(
            "Compact JSON written by moccheck --handoff-out. If omitted, use a matching "
            "<input-stem>.moccheck-handoff.json or build the baseline in memory."
        ),
    )
    parser.add_argument(
        "--blocks-after",
        type=int,
        default=4,
        help="State-specific RASSCF blocks after setup to inspect (default: 4)",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.blocks_after < 0:
        raise ValueError("--blocks-after must be non-negative")
    result = build_after_diagnostic(
        args.inp,
        args.output,
        handoff_path=args.handoff,
        blocks_after=args.blocks_after,
    )
    print_after_diagnostic(result)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nWrote post-setup diagnostic JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
