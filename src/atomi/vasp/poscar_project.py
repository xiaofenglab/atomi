from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from atomi.vasp.magmom import (
    PoscarSpecies,
    PoscarStructure,
    expand_magmom_tokens,
    find_magmom_line,
    format_magmom_line,
    parse_element_list,
    read_poscar_structure,
    strip_incar_comment,
)


@dataclass(frozen=True)
class SiteMatch:
    target_index: int
    source_index: int
    distance_A: float


@dataclass(frozen=True)
class ProjectionResult:
    source_poscar: Path
    target_poscar: Path
    output_poscar: Path
    prepared_source_poscar: Path | None
    output_incar: Path | None
    match_csv: Path
    plan_json: Path
    cation_matches: list[SiteMatch]
    max_cation_distance_A: float
    output_magmom_count: int | None
    output_species: PoscarSpecies
    source_cation_magmom_summary: dict[str, dict[str, object]]
    cation_magmom_summary: dict[str, dict[str, object]]
    cation_magmom_comparison: dict[str, dict[str, object]]
    warnings: list[str]


@dataclass(frozen=True)
class PreparedStructure:
    structure: PoscarStructure
    origin_indices: list[int]
    operations: list[dict[str, object]]


@dataclass(frozen=True)
class RepresentativeCandidate:
    source_index: int
    origin_index: int
    symbol: str
    folded_position: tuple[float, float, float]
    sign: str
    abs_moment: str


def project_poscar_elements(
    source_poscar: Path,
    target_poscar: Path,
    *,
    output_poscar: Path,
    prepared_source_poscar: Path | None = None,
    source_incar: Path | None = None,
    output_incar: Path | None = None,
    match_csv: Path | None = None,
    plan_json: Path | None = None,
    cation_elements: list[str] | None = None,
    anion_elements: list[str] | None = None,
    species_order: list[str] | None = None,
    max_cation_distance_A: float = 1.5,
    strict: bool = True,
    magmom_decimals: int = 3,
    source_repeat: tuple[int, int, int] | None = None,
    target_repeat: tuple[int, int, int] | None = None,
    source_supercell: tuple[int, int, int] | None = None,
    source_keep_cells: tuple[int, int, int] | None = None,
    source_reduce_cells: tuple[int, int, int] | None = None,
    source_reduce_beam_width: int = 256,
    source_reduce_site_tolerance: float = 0.05,
    source_crop_fraction: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    source_crop_policy: str = "defect_preserving",
    scale_target_volume_to_source: bool = False,
    align_cation_origin: bool = True,
    strict_magmom_preservation: bool = True,
) -> ProjectionResult:
    """Project cation identities and MAGMOMs from source POSCAR A onto structure POSCAR B."""
    source_poscar = source_poscar.expanduser().resolve()
    target_poscar = target_poscar.expanduser().resolve()
    output_poscar = output_poscar.expanduser().resolve()
    source_incar = source_incar.expanduser().resolve() if source_incar is not None else None
    output_incar = output_incar.expanduser().resolve() if output_incar is not None else None
    match_csv = match_csv.expanduser().resolve() if match_csv is not None else output_poscar.with_name("poscar_projection_map.csv")
    plan_json = plan_json.expanduser().resolve() if plan_json is not None else output_poscar.with_name("poscar_projection_plan.json")

    source_raw = read_poscar_structure(source_poscar)
    target_raw = read_poscar_structure(target_poscar)
    raw_source_symbols = atom_symbols(source_raw)
    raw_source_moments = strict_existing_magmom_values(source_incar, len(raw_source_symbols)) if source_incar is not None else None
    anion_set = set(anion_elements or ["O"])
    cation_order = list(cation_elements or [])
    cation_set = set(cation_order)
    target_prepared = prepare_structure(target_raw, repeat=target_repeat)
    target = target_prepared.structure
    target_symbols = atom_symbols(target)
    target_cations = selected_cation_indices(target_symbols, cation_set, anion_set)
    source_prepared = prepare_structure(
        source_raw,
        repeat=source_repeat,
        crop_supercell=source_supercell,
        crop_keep_cells=source_keep_cells,
        reduce_cells=source_reduce_cells,
        reduce_beam_width=source_reduce_beam_width,
        reduce_site_tolerance=source_reduce_site_tolerance,
        crop_fraction=source_crop_fraction,
        crop_policy=source_crop_policy,
        source_moments=raw_source_moments,
        cation_elements=cation_set,
        anion_elements=anion_set,
        expected_cation_count=len(target_cations),
    )
    source = source_prepared.structure
    if scale_target_volume_to_source:
        target, scale_operation = scale_structure_volume_to_reference(target, source)
        target_prepared = PreparedStructure(
            structure=target,
            origin_indices=target_prepared.origin_indices,
            operations=[*target_prepared.operations, scale_operation],
        )
    prepared_source_poscar = (
        prepared_source_poscar.expanduser().resolve()
        if prepared_source_poscar is not None
        else output_poscar.with_name("POSCAR_A_prepared")
        if source_prepared.operations
        else None
    )
    if prepared_source_poscar is not None:
        source_text = write_poscar_text(
            f"Prepared source {source_poscar.name} used for projection",
            source.species,
            source.cell,
            source.scaled_positions,
        )
        prepared_source_poscar.parent.mkdir(parents=True, exist_ok=True)
        prepared_source_poscar.write_text(source_text, encoding="utf-8")

    source_symbols = atom_symbols(source)
    source_cations = selected_cation_indices(source_symbols, cation_set, anion_set)
    if len(source_cations) != len(target_cations):
        raise ValueError(
            "Source/target cation counts differ: "
            f"{len(source_cations)} in {source_poscar}, {len(target_cations)} in {target_poscar}. "
            "Use --cation-elements/--anion-elements to define the projected sublattice, or use "
            "--source-repeat/--target-repeat/--source-supercell/--source-keep-cells to make the cells commensurate."
        )
    if not source_cations:
        raise ValueError("No cation sites found to project.")

    source_for_matching = source
    cation_origin_alignment = {
        "enabled": align_cation_origin,
        "shift_fractional": [0.0, 0.0, 0.0],
        "max_cation_distance_before_A": None,
        "max_cation_distance_after_A": None,
        "mean_cation_distance_after_A": None,
    }
    if align_cation_origin:
        source_for_matching, cation_origin_alignment = align_source_cation_origin(
            source,
            source_cations,
            target,
            target_cations,
        )
    matches = nearest_site_assignment(
        source_for_matching.scaled_positions,
        source_cations,
        target.scaled_positions,
        target_cations,
        target.cell,
    )
    worst = max((match.distance_A for match in matches), default=0.0)
    warnings: list[str] = []
    if worst > max_cation_distance_A:
        message = (
            f"Worst cation match distance is {worst:.4g} A, above "
            f"--max-cation-distance {max_cation_distance_A:.4g} A."
        )
        if strict:
            write_match_csv(match_csv, matches, source_symbols, target_symbols, source_for_matching, target)
            write_plan_json(
                plan_json,
                {
                    "schema": "atomi.vasp.poscar_projection.v1",
                    "status": "failed_distance_check",
                    "source_poscar": str(source_poscar),
                    "target_poscar": str(target_poscar),
                    "prepared_source_poscar": str(prepared_source_poscar) if prepared_source_poscar else "",
                    "match_csv": str(match_csv),
                    "source_operations": source_prepared.operations,
                    "target_operations": target_prepared.operations,
                    "cation_origin_alignment": cation_origin_alignment,
                    "source_cation_count": len(source_cations),
                    "target_cation_count": len(target_cations),
                    "max_cation_distance_A": worst,
                    "distance_limit_A": max_cation_distance_A,
                    "warnings": [message],
                    "notes": [
                        "Distance check failed before final POSCAR writing.",
                        "Inspect the diagnostic map CSV and cation_origin_alignment before retrying with --allow-large-cation-distance.",
                    ],
                },
            )
            raise ValueError(message + " Use --allow-large-cation-distance only after inspecting the map CSV.")
        warnings.append(message)

    projected_symbols = list(target_symbols)
    source_by_target: dict[int, int] = {}
    for match in matches:
        projected_symbols[match.target_index] = source_symbols[match.source_index]
        source_by_target[match.target_index] = match.source_index

    default_order = default_projection_species_order(
        source.species.symbols,
        target.species.symbols,
        projected_symbols,
        cation_order,
        cation_set,
        anion_set,
    )
    output_order = complete_species_order(species_order or default_order, projected_symbols)
    output_species, output_indices = grouped_species_and_indices(projected_symbols, output_order)
    poscar_text = write_poscar_text(
        f"Projected {source_poscar.name} elements onto {target_poscar.name} structure",
        output_species,
        target.cell,
        [target.scaled_positions[index] for index in output_indices],
    )
    output_poscar.parent.mkdir(parents=True, exist_ok=True)
    output_poscar.write_text(poscar_text, encoding="utf-8")

    source_moments: list[float] | None = None
    output_moments: list[float] | None = None
    output_moments_grouped: list[float] | None = None
    output_magmom_count: int | None = None
    if source_incar is not None:
        if output_incar is None:
            output_incar = output_poscar.with_name("INCAR")
        source_moments = (
            [raw_source_moments[index] for index in source_prepared.origin_indices]
            if raw_source_moments is not None
            else None
        )
        output_moments = projected_magmom_values(
            source,
            target,
            source_symbols,
            target_symbols,
            projected_symbols,
            source_by_target,
            source_moments,
            cation_set,
            anion_set,
        )
        incar_text = source_incar.read_text(encoding="utf-8", errors="replace")
        if output_moments is not None:
            output_moments_grouped = [output_moments[index] for index in output_indices]
            selected = species_with_nonzero_moments(output_species, output_moments_grouped)
            magmom_line = format_magmom_line(
                output_species,
                output_moments_grouped,
                selected_elements=selected,
                decimals=magmom_decimals,
                compact_zero=True,
            )
            output_magmom_count = validate_magmom_line_count(
                magmom_line,
                output_species.total_atoms,
                context=f"projected INCAR for {output_poscar}",
            )
            incar_text = replace_or_append_magmom_text(incar_text, magmom_line)
        output_incar.parent.mkdir(parents=True, exist_ok=True)
        output_incar.write_text(incar_text, encoding="utf-8")

    source_cation_magmom_summary = final_cation_magmom_summary(
        source.species,
        source_moments,
        cation_set,
        anion_set,
        magmom_decimals,
    )
    cation_magmom_summary = final_cation_magmom_summary(
        output_species,
        output_moments_grouped,
        cation_set,
        anion_set,
        magmom_decimals,
    )
    cation_magmom_comparison = compare_cation_magmom_summaries(
        source_cation_magmom_summary,
        cation_magmom_summary,
        output_species.symbols,
        magmom_decimals,
    )
    magmom_warnings = cation_magmom_preservation_warnings(cation_magmom_comparison)
    warnings.extend(magmom_warnings)
    write_match_csv(match_csv, matches, source_symbols, target_symbols, source_for_matching, target)
    write_plan_json(
        plan_json,
        {
            "schema": "atomi.vasp.poscar_projection.v1",
            "status": "failed_magmom_preservation_check" if strict_magmom_preservation and magmom_warnings else "ok",
            "source_poscar": str(source_poscar),
            "target_poscar": str(target_poscar),
            "source_incar": str(source_incar) if source_incar else "",
            "output_poscar": str(output_poscar),
            "prepared_source_poscar": str(prepared_source_poscar) if prepared_source_poscar else "",
            "output_incar": str(output_incar) if output_incar else "",
            "source_operations": source_prepared.operations,
            "target_operations": target_prepared.operations,
            "cation_origin_alignment": cation_origin_alignment,
            "cation_elements": cation_order or sorted(cation_set),
            "anion_elements": sorted(anion_set),
            "source_cation_count": len(source_cations),
            "target_cation_count": len(target_cations),
            "source_magmom_count": len(raw_source_moments) if raw_source_moments is not None else None,
            "output_magmom_count": output_magmom_count,
            "max_cation_distance_A": worst,
            "distance_limit_A": max_cation_distance_A,
            "species_order": output_species.symbols,
            "species_counts": dict(zip(output_species.symbols, output_species.counts)),
            "source_cation_magmom_summary": source_cation_magmom_summary,
            "cation_magmom_summary": cation_magmom_summary,
            "cation_magmom_comparison": cation_magmom_comparison,
            "strict_magmom_preservation": strict_magmom_preservation,
            "magmom_preservation_ok": not magmom_warnings,
            "magmom_preservation_warnings": magmom_warnings,
            "warnings": warnings,
            "notes": [
                "Cation species and cation MAGMOM values were projected from source POSCAR A.",
                "Target POSCAR B supplied cell and coordinates.",
                "Non-cation sites are kept from target POSCAR B; their MAGMOMs are matched by element/position when possible, otherwise set to 0.",
            ],
        },
    )
    if strict_magmom_preservation and magmom_warnings:
        raise ValueError(
            "Projected cation MAGMOM is not preserved from prepared source A: "
            + "; ".join(magmom_warnings)
            + ". Inspect POSCAR_A_prepared, INCAR, and the plan JSON, or use --allow-magmom-mismatch only after review."
        )
    return ProjectionResult(
        source_poscar=source_poscar,
        target_poscar=target_poscar,
        output_poscar=output_poscar,
        prepared_source_poscar=prepared_source_poscar,
        output_incar=output_incar,
        match_csv=match_csv,
        plan_json=plan_json,
        cation_matches=matches,
        max_cation_distance_A=worst,
        output_magmom_count=output_magmom_count,
        output_species=output_species,
        source_cation_magmom_summary=source_cation_magmom_summary,
        cation_magmom_summary=cation_magmom_summary,
        cation_magmom_comparison=cation_magmom_comparison,
        warnings=warnings,
    )


def strict_existing_magmom_values(incar: Path, total_atoms: int) -> list[float] | None:
    lines = incar.read_text(encoding="utf-8", errors="replace").splitlines()
    line_index, line = find_magmom_line(lines)
    if line_index is None or line is None:
        return None
    body = strip_incar_comment(line).split("=", 1)[-1]
    try:
        values = expand_magmom_tokens(body.split())
    except ValueError as exc:
        raise ValueError(f"Could not parse MAGMOM in {incar}.") from exc
    if len(values) != total_atoms:
        raise ValueError(
            f"MAGMOM count in {incar} is {len(values)}, but source POSCAR has {total_atoms} atoms. "
            "Use a matching INCAR_A for POSCAR_A; Atomi will not truncate or pad MAGMOM during projection."
        )
    return values


def validate_magmom_line_count(magmom_line: str, total_atoms: int, *, context: str) -> int:
    body = strip_incar_comment(magmom_line).split("=", 1)[-1]
    values = expand_magmom_tokens(body.split())
    if len(values) != total_atoms:
        raise ValueError(
            f"MAGMOM count for {context} is {len(values)}, but output POSCAR has {total_atoms} atoms. "
            "This indicates an internal projection/order bug; do not run VASP with this INCAR."
        )
    return len(values)


def cation_magmom_preservation_warnings(comparison: dict[str, dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    for symbol, entry in comparison.items():
        if int(entry.get("count_delta", 0)) != 0:
            warnings.append(f"{symbol} cation count changed by {entry['count_delta']:+d}")
        sign_changed = (
            int(entry.get("positive_delta", 0)) != 0
            or int(entry.get("negative_delta", 0)) != 0
            or int(entry.get("zero_delta", 0)) != 0
        )
        if sign_changed:
            warnings.append(
                f"{symbol} sign counts changed "
                f"(+ {entry['source_positive']}->{entry['output_positive']}, "
                f"- {entry['source_negative']}->{entry['output_negative']}, "
                f"0 {entry['source_zero']}->{entry['output_zero']})"
            )
        if abs(float(entry.get("sum_delta", 0.0))) > 1.0e-8:
            warnings.append(f"{symbol} moment sum changed by {float(entry['sum_delta']):+g}")
        if entry.get("unique_abs_moments_match") is not True:
            warnings.append(f"{symbol} absolute moment set changed")
    return warnings


def atom_symbols(structure: PoscarStructure) -> list[str]:
    symbols: list[str] = []
    for symbol, count in zip(structure.species.symbols, structure.species.counts):
        symbols.extend([symbol] * count)
    return symbols


def cell_volume(cell: list[list[float]]) -> float:
    a, b, c = cell
    return abs(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def scale_structure_volume_to_reference(
    structure: PoscarStructure,
    reference: PoscarStructure,
) -> tuple[PoscarStructure, dict[str, object]]:
    source_volume = cell_volume(reference.cell)
    target_volume = cell_volume(structure.cell)
    if source_volume <= 0.0 or target_volume <= 0.0:
        raise ValueError(
            "Cannot scale target volume because source or target cell volume is zero "
            f"(source={source_volume:.8g}, target={target_volume:.8g})."
        )
    linear_scale = (source_volume / target_volume) ** (1.0 / 3.0)
    scaled = PoscarStructure(
        species=structure.species,
        cell=[[value * linear_scale for value in vector] for vector in structure.cell],
        scaled_positions=structure.scaled_positions,
    )
    return (
        scaled,
        {
            "kind": "scale_volume_to_source",
            "source_volume_A3": source_volume,
            "target_volume_before_A3": target_volume,
            "target_volume_after_A3": cell_volume(scaled.cell),
            "linear_scale": linear_scale,
            "note": "Target fractional coordinates are preserved; target cell is uniformly scaled to prepared source volume.",
        },
    )


def prepare_structure(
    structure: PoscarStructure,
    *,
    repeat: tuple[int, int, int] | None = None,
    crop_supercell: tuple[int, int, int] | None = None,
    crop_keep_cells: tuple[int, int, int] | None = None,
    reduce_cells: tuple[int, int, int] | None = None,
    reduce_beam_width: int = 256,
    reduce_site_tolerance: float = 0.05,
    crop_fraction: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    crop_policy: str = "defect_preserving",
    source_moments: list[float] | None = None,
    cation_elements: set[str] | None = None,
    anion_elements: set[str] | None = None,
    expected_cation_count: int | None = None,
) -> PreparedStructure:
    origin_indices = list(range(structure.species.total_atoms))
    operations: list[dict[str, object]] = []
    if repeat is not None and repeat != (1, 1, 1):
        structure, origin_indices = repeat_structure(structure, origin_indices, repeat)
        operations.append({"kind": "repeat", "repeat": list(repeat)})
    if reduce_cells is not None:
        if crop_supercell is None:
            raise ValueError("--source-reduce-cells requires --source-supercell.")
        if crop_keep_cells is not None:
            raise ValueError("--source-reduce-cells cannot be combined with --source-keep-cells.")
        structure, origin_indices, reduce_meta = reduce_structure_to_representative_cell(
            structure,
            origin_indices,
            crop_supercell,
            reduce_cells,
            source_moments=source_moments,
            cation_elements=cation_elements or set(),
            anion_elements=anion_elements or {"O"},
            expected_cation_count=expected_cation_count,
            beam_width=reduce_beam_width,
            site_tolerance=reduce_site_tolerance,
        )
        operations.append(reduce_meta)
    if crop_keep_cells is not None or (crop_supercell is not None and reduce_cells is None):
        if crop_supercell is None or crop_keep_cells is None:
            raise ValueError("--source-supercell and --source-keep-cells must be used together.")
        if crop_policy == "origin":
            ranges = crop_ranges_from_origin(crop_supercell, crop_keep_cells, (0, 0, 0))
            crop_meta = {
                "selection_policy": "origin",
                "crop_origin_cells": [0, 0, 0],
            }
        elif crop_policy == "defect_preserving":
            ranges, crop_meta = choose_defect_preserving_crop_ranges(
                structure,
                origin_indices,
                crop_supercell,
                crop_keep_cells,
                source_moments=source_moments,
                cation_elements=cation_elements or set(),
                anion_elements=anion_elements or {"O"},
            )
        else:
            raise ValueError(f"Unknown source crop policy {crop_policy!r}.")
        structure, origin_indices, repair_meta = crop_structure_fraction(
            structure,
            origin_indices,
            ranges,
            expected_cation_count=expected_cation_count,
            cation_elements=cation_elements or set(),
            anion_elements=anion_elements or {"O"},
        )
        operations.append(
            {
                "kind": "source_keep_cells",
                "source_supercell": list(crop_supercell),
                "keep_cells": list(crop_keep_cells),
                "fraction_ranges": [list(item) for item in ranges],
                **crop_meta,
                **repair_meta,
            }
        )
    if crop_fraction is not None:
        structure, origin_indices, repair_meta = crop_structure_fraction(
            structure,
            origin_indices,
            crop_fraction,
            expected_cation_count=expected_cation_count,
            cation_elements=cation_elements or set(),
            anion_elements=anion_elements or {"O"},
        )
        operations.append({"kind": "fraction_crop", "fraction_ranges": [list(item) for item in crop_fraction], **repair_meta})
    return PreparedStructure(structure=structure, origin_indices=origin_indices, operations=operations)


def reduce_structure_to_representative_cell(
    structure: PoscarStructure,
    origin_indices: list[int],
    supercell: tuple[int, int, int],
    reduce_cells: tuple[int, int, int],
    *,
    source_moments: list[float] | None,
    cation_elements: set[str],
    anion_elements: set[str],
    expected_cation_count: int | None,
    beam_width: int,
    site_tolerance: float,
) -> tuple[PoscarStructure, list[int], dict[str, object]]:
    for axis in range(3):
        if supercell[axis] <= 0 or reduce_cells[axis] <= 0:
            raise ValueError(f"Source supercell and reduce-cells values must be positive, got {supercell} and {reduce_cells}.")
        if reduce_cells[axis] > supercell[axis]:
            raise ValueError(f"Cannot reduce source supercell {supercell} to larger cell {reduce_cells}.")
    if beam_width <= 0:
        raise ValueError(f"--source-reduce-beam-width must be positive, got {beam_width}.")
    if site_tolerance < 0.0:
        raise ValueError(f"--source-reduce-site-tolerance must be non-negative, got {site_tolerance}.")

    symbols = atom_symbols(structure)
    cation_indices = selected_cation_indices(symbols, cation_elements, anion_elements)
    if not cation_indices:
        raise ValueError("No cation sites found for representative source reduction.")
    cation_index_set = set(cation_indices)
    non_cation_indices = [index for index in range(len(symbols)) if index not in cation_index_set]

    slot_candidates = folded_atom_candidate_slots(
        structure,
        origin_indices,
        cation_indices,
        supercell,
        reduce_cells,
        source_moments,
        site_tolerance,
    )
    target_count = expected_cation_count if expected_cation_count is not None else len(slot_candidates)
    if len(slot_candidates) != target_count:
        raise ValueError(
            "Representative source reduction produced "
            f"{len(slot_candidates)} folded cation sites, but target has {target_count} cation sites. "
            "Use a commensurate --source-supercell/--source-reduce-cells pair or adjust --target-repeat."
        )

    source_species_counts = Counter(symbols[index] for index in cation_indices)
    desired_species_counts = proportional_count_targets(source_species_counts, target_count, preserve_present=True)
    source_sign_counts = Counter(
        (symbols[index], cation_moment_sign(index, origin_indices, source_moments))
        for index in cation_indices
    )
    desired_sign_counts = proportional_group_targets(source_sign_counts, desired_species_counts)
    source_abs_counts = Counter(
        (symbols[index], cation_moment_sign(index, origin_indices, source_moments), cation_abs_moment_label(index, origin_indices, source_moments))
        for index in cation_indices
    )
    desired_abs_counts = proportional_group_targets(source_abs_counts, desired_sign_counts)

    selected, score = choose_representative_candidates(
        slot_candidates,
        desired_species_counts,
        desired_sign_counts,
        desired_abs_counts,
        beam_width=beam_width,
    )
    non_cation_slot_candidates = folded_atom_candidate_slots(
        structure,
        origin_indices,
        non_cation_indices,
        supercell,
        reduce_cells,
        source_moments,
        site_tolerance,
    )
    selected_non_cations: list[RepresentativeCandidate] = []
    non_cation_score = 0.0
    desired_non_cation_species_counts: dict[str, int] = {}
    desired_non_cation_sign_counts: dict[tuple[str, ...], int] = {}
    desired_non_cation_abs_counts: dict[tuple[str, ...], int] = {}
    if non_cation_slot_candidates:
        desired_non_cation_species_counts = proportional_count_targets(
            Counter(symbols[index] for index in non_cation_indices),
            len(non_cation_slot_candidates),
            preserve_present=True,
        )
        source_non_cation_sign_counts = Counter(
            (symbols[index], cation_moment_sign(index, origin_indices, source_moments))
            for index in non_cation_indices
        )
        desired_non_cation_sign_counts = proportional_group_targets(
            source_non_cation_sign_counts,
            desired_non_cation_species_counts,
        )
        source_non_cation_abs_counts = Counter(
            (
                symbols[index],
                cation_moment_sign(index, origin_indices, source_moments),
                cation_abs_moment_label(index, origin_indices, source_moments),
            )
            for index in non_cation_indices
        )
        desired_non_cation_abs_counts = proportional_group_targets(
            source_non_cation_abs_counts,
            desired_non_cation_sign_counts,
        )
        selected_non_cations, non_cation_score = choose_representative_candidates(
            non_cation_slot_candidates,
            desired_non_cation_species_counts,
            desired_non_cation_sign_counts,
            desired_non_cation_abs_counts,
            beam_width=beam_width,
        )
    selected_all = [*selected, *selected_non_cations]
    selected_symbols = [candidate.symbol for candidate in selected_all]
    selected_positions = [list(candidate.folded_position) for candidate in selected_all]
    selected_origins = [candidate.origin_index for candidate in selected_all]
    selected_source_indices = [candidate.source_index for candidate in selected]
    cell = [
        [value * reduce_cells[axis] / supercell[axis] for value in vector]
        for axis, vector in enumerate(structure.cell)
    ]
    order = [symbol for symbol in structure.species.symbols if symbol in set(selected_symbols)]
    reduced, reduced_origins = grouped_structure(selected_symbols, selected_positions, selected_origins, cell, order)
    selected_cation_counts = Counter(candidate.symbol for candidate in selected)
    selected_sign_counts = Counter((candidate.symbol, candidate.sign) for candidate in selected)
    selected_abs_counts = Counter((candidate.symbol, candidate.sign, candidate.abs_moment) for candidate in selected)
    selected_non_cation_counts = Counter(candidate.symbol for candidate in selected_non_cations)
    selected_non_cation_sign_counts = Counter((candidate.symbol, candidate.sign) for candidate in selected_non_cations)
    selected_non_cation_abs_counts = Counter(
        (candidate.symbol, candidate.sign, candidate.abs_moment)
        for candidate in selected_non_cations
    )
    meta = {
        "kind": "source_representative_reduce",
        "source_supercell": list(supercell),
        "reduce_cells": list(reduce_cells),
        "source_cation_count": len(cation_indices),
        "folded_cation_site_count": len(slot_candidates),
        "selected_cation_count": len(selected),
        "source_non_cation_count": len(non_cation_indices),
        "folded_non_cation_site_count": len(non_cation_slot_candidates),
        "selected_non_cation_count": len(selected_non_cations),
        "beam_width": beam_width,
        "folded_site_tolerance_fractional": site_tolerance,
        "selection_score": score,
        "non_cation_selection_score": non_cation_score,
        "cation_species_target_counts": dict(sorted(desired_species_counts.items())),
        "cation_species_selected_counts": dict(sorted(selected_cation_counts.items())),
        "cation_sign_target_counts": stringify_counter(desired_sign_counts),
        "cation_sign_selected_counts": stringify_counter(selected_sign_counts),
        "cation_abs_moment_target_counts": stringify_counter(desired_abs_counts),
        "cation_abs_moment_selected_counts": stringify_counter(selected_abs_counts),
        "non_cation_species_target_counts": dict(sorted(desired_non_cation_species_counts.items())),
        "non_cation_species_selected_counts": dict(sorted(selected_non_cation_counts.items())),
        "non_cation_sign_target_counts": stringify_counter(desired_non_cation_sign_counts),
        "non_cation_sign_selected_counts": stringify_counter(selected_non_cation_sign_counts),
        "non_cation_abs_moment_target_counts": stringify_counter(desired_non_cation_abs_counts),
        "non_cation_abs_moment_selected_counts": stringify_counter(selected_non_cation_abs_counts),
        "source_magnetic_signature": magnetic_signature(symbols, cation_indices, origin_indices, source_moments),
        "reduced_magnetic_signature": magnetic_signature(symbols, selected_source_indices, origin_indices, source_moments),
        "note": (
            "Source sites were folded into the requested smaller source cell. One representative "
            "source cation was selected per folded cation site for projection, and one representative "
            "source non-cation was selected per folded non-cation site so POSCAR_A_prepared remains "
            "chemically inspectable. Final non-cation coordinates/species still come from POSCAR B."
        ),
    }
    return reduced, reduced_origins, meta


def folded_atom_candidate_slots(
    structure: PoscarStructure,
    origin_indices: list[int],
    atom_indices: list[int],
    supercell: tuple[int, int, int],
    reduce_cells: tuple[int, int, int],
    source_moments: list[float] | None,
    site_tolerance: float,
) -> list[list[RepresentativeCandidate]]:
    symbols = atom_symbols(structure)
    candidates: list[RepresentativeCandidate] = []
    for index in atom_indices:
        folded = folded_position_in_reduced_cell(structure.scaled_positions[index], supercell, reduce_cells)
        candidates.append(
            RepresentativeCandidate(
                source_index=index,
                origin_index=origin_indices[index],
                symbol=symbols[index],
                folded_position=tuple(folded),
                sign=cation_moment_sign(index, origin_indices, source_moments),
                abs_moment=cation_abs_moment_label(index, origin_indices, source_moments),
            )
        )
    return cluster_folded_candidates(candidates, site_tolerance)


def cluster_folded_candidates(
    candidates: list[RepresentativeCandidate],
    site_tolerance: float,
) -> list[list[RepresentativeCandidate]]:
    slots: list[list[RepresentativeCandidate]] = []
    centers: list[tuple[float, float, float]] = []
    for candidate in sorted(candidates, key=lambda item: item.source_index):
        distances = [
            (fractional_l2_distance(candidate.folded_position, center), index)
            for index, center in enumerate(centers)
        ]
        if distances:
            distance, slot_index = min(distances, key=lambda item: item[0])
            if distance <= site_tolerance:
                slots[slot_index].append(candidate)
                continue
        centers.append(candidate.folded_position)
        slots.append([candidate])
    return [
        sorted(candidates, key=lambda item: (item.symbol, item.sign, item.abs_moment, item.source_index))
        for _center, candidates in sorted(zip(centers, slots), key=lambda item: item[0])
    ]


def fractional_l2_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    diff = [left[axis] - right[axis] for axis in range(3)]
    diff = [value - round(value) for value in diff]
    return math.sqrt(sum(value * value for value in diff))


def folded_position_in_reduced_cell(
    position: list[float],
    supercell: tuple[int, int, int],
    reduce_cells: tuple[int, int, int],
) -> list[float]:
    folded: list[float] = []
    for axis, value in enumerate(position):
        scaled = (value % 1.0) * supercell[axis]
        cell_index = min(supercell[axis] - 1, max(0, int(math.floor(scaled + 1.0e-10))))
        local = scaled - cell_index
        reduced_cell = cell_index % reduce_cells[axis]
        folded.append((reduced_cell + local) / reduce_cells[axis])
    return folded


def cation_moment_sign(index: int, origin_indices: list[int], source_moments: list[float] | None) -> str:
    if source_moments is None or origin_indices[index] >= len(source_moments):
        return "unknown"
    return moment_sign(source_moments[origin_indices[index]])


def cation_abs_moment_label(index: int, origin_indices: list[int], source_moments: list[float] | None) -> str:
    if source_moments is None or origin_indices[index] >= len(source_moments):
        return "unknown"
    return f"{round(abs(source_moments[origin_indices[index]]), 3):g}"


def proportional_count_targets(counts: Counter[str], total: int, *, preserve_present: bool = False) -> dict[str, int]:
    if total <= 0:
        return {}
    symbols = sorted(counts)
    if not symbols:
        return {}
    base = {symbol: 0 for symbol in symbols}
    remaining = total
    if preserve_present and total >= len(symbols):
        base = {symbol: 1 for symbol in symbols}
        remaining -= len(symbols)
    available = sum(counts.values())
    if available <= 0:
        return base
    raw = {symbol: counts[symbol] / available * remaining for symbol in symbols}
    for symbol, value in raw.items():
        assigned = int(math.floor(value))
        base[symbol] += assigned
    short = total - sum(base.values())
    ranked = sorted(
        symbols,
        key=lambda symbol: (
            raw[symbol] - math.floor(raw[symbol]),
            -counts[symbol],
            symbol,
        ),
        reverse=True,
    )
    for symbol in ranked[:short]:
        base[symbol] += 1
    return base


def proportional_group_targets(
    counts: Counter[tuple[str, ...]],
    parent_targets: dict[str | tuple[str, ...], int],
) -> dict[tuple[str, ...], int]:
    result: dict[tuple[str, ...], int] = {}
    parent_groups: dict[str | tuple[str, ...], Counter[tuple[str, ...]]] = {}
    for key, count in counts.items():
        parent: str | tuple[str, ...] = key[0] if len(key) == 2 else key[:2]
        parent_groups.setdefault(parent, Counter())[key] = count
    for parent, target_total in parent_targets.items():
        group_counts = parent_groups.get(parent, Counter())
        if not group_counts:
            continue
        preserve = target_total >= len(group_counts)
        group_target = proportional_count_targets(group_counts, target_total, preserve_present=preserve)
        result.update(group_target)
    return result


def choose_representative_candidates(
    slot_candidates: list[list[RepresentativeCandidate]],
    desired_species_counts: dict[str, int],
    desired_sign_counts: dict[tuple[str, ...], int],
    desired_abs_counts: dict[tuple[str, ...], int],
    *,
    beam_width: int,
) -> tuple[list[RepresentativeCandidate], float]:
    slots = sorted(slot_candidates, key=lambda candidates: (len(candidates), candidates[0].folded_position if candidates else (0.0, 0.0, 0.0)))
    states: list[tuple[float, list[RepresentativeCandidate], Counter[str], Counter[tuple[str, str]], Counter[tuple[str, str, str]]]] = [
        (0.0, [], Counter(), Counter(), Counter())
    ]
    total_slots = len(slots)
    for slot_index, candidates in enumerate(slots, start=1):
        next_states: list[tuple[float, list[RepresentativeCandidate], Counter[str], Counter[tuple[str, str]], Counter[tuple[str, str, str]]]] = []
        for _score, selected, species_counts, sign_counts, abs_counts in states:
            for candidate in candidates:
                next_species = species_counts.copy()
                next_sign = sign_counts.copy()
                next_abs = abs_counts.copy()
                next_species[candidate.symbol] += 1
                next_sign[(candidate.symbol, candidate.sign)] += 1
                next_abs[(candidate.symbol, candidate.sign, candidate.abs_moment)] += 1
                next_selected = [*selected, candidate]
                score = representative_partial_score(
                    next_species,
                    next_sign,
                    next_abs,
                    desired_species_counts,
                    desired_sign_counts,
                    desired_abs_counts,
                    selected_count=slot_index,
                    total_count=total_slots,
                )
                next_states.append((score, next_selected, next_species, next_sign, next_abs))
        next_states.sort(key=lambda item: (item[0], [candidate.source_index for candidate in item[1]]))
        states = next_states[:beam_width]
    final_states = [
        (
            representative_final_score(
                species_counts,
                sign_counts,
                abs_counts,
                desired_species_counts,
                desired_sign_counts,
                desired_abs_counts,
            ),
            selected,
        )
        for _partial, selected, species_counts, sign_counts, abs_counts in states
    ]
    final_states.sort(key=lambda item: (item[0], [candidate.source_index for candidate in item[1]]))
    best_score, best_selected = final_states[0]
    return best_selected, best_score


def representative_partial_score(
    species_counts: Counter[str],
    sign_counts: Counter[tuple[str, str]],
    abs_counts: Counter[tuple[str, str, str]],
    desired_species_counts: dict[str, int],
    desired_sign_counts: dict[tuple[str, ...], int],
    desired_abs_counts: dict[tuple[str, ...], int],
    *,
    selected_count: int,
    total_count: int,
) -> float:
    progress = selected_count / total_count if total_count else 1.0
    return (
        counter_progress_penalty(species_counts, desired_species_counts, progress, over_weight=500.0, drift_weight=20.0)
        + counter_progress_penalty(sign_counts, desired_sign_counts, progress, over_weight=120.0, drift_weight=5.0)
        + counter_progress_penalty(abs_counts, desired_abs_counts, progress, over_weight=40.0, drift_weight=1.0)
    )


def representative_final_score(
    species_counts: Counter[str],
    sign_counts: Counter[tuple[str, str]],
    abs_counts: Counter[tuple[str, str, str]],
    desired_species_counts: dict[str, int],
    desired_sign_counts: dict[tuple[str, ...], int],
    desired_abs_counts: dict[tuple[str, ...], int],
) -> float:
    return (
        counter_final_penalty(species_counts, desired_species_counts, weight=1000.0)
        + counter_final_penalty(sign_counts, desired_sign_counts, weight=250.0)
        + counter_final_penalty(abs_counts, desired_abs_counts, weight=50.0)
    )


def counter_progress_penalty(
    counts: Counter,
    desired: dict,
    progress: float,
    *,
    over_weight: float,
    drift_weight: float,
) -> float:
    penalty = 0.0
    for key in set(counts) | set(desired):
        count = counts.get(key, 0)
        target = desired.get(key, 0)
        if count > target:
            penalty += (count - target) * over_weight
        penalty += abs(count - target * progress) * drift_weight
    return penalty


def counter_final_penalty(counts: Counter, desired: dict, *, weight: float) -> float:
    return sum(abs(counts.get(key, 0) - desired.get(key, 0)) for key in set(counts) | set(desired)) * weight


def stringify_counter(counter: dict | Counter) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in counter.items():
        if isinstance(key, tuple):
            label = "|".join(str(item) for item in key)
        else:
            label = str(key)
        result[label] = int(value)
    return dict(sorted(result.items()))


def crop_ranges_from_origin(
    supercell: tuple[int, int, int],
    keep_cells: tuple[int, int, int],
    origin_cells: tuple[int, int, int],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    for axis in range(3):
        if supercell[axis] <= 0 or keep_cells[axis] <= 0:
            raise ValueError(f"Source supercell and keep-cells values must be positive, got {supercell} and {keep_cells}.")
        if keep_cells[axis] > supercell[axis]:
            raise ValueError(f"Cannot keep {keep_cells} cells from source supercell {supercell}.")
        if origin_cells[axis] < 0 or origin_cells[axis] + keep_cells[axis] > supercell[axis]:
            raise ValueError(f"Crop origin {origin_cells} with keep-cells {keep_cells} is outside source supercell {supercell}.")
    return tuple((origin_cells[i] / supercell[i], (origin_cells[i] + keep_cells[i]) / supercell[i]) for i in range(3))


def choose_defect_preserving_crop_ranges(
    structure: PoscarStructure,
    origin_indices: list[int],
    supercell: tuple[int, int, int],
    keep_cells: tuple[int, int, int],
    *,
    source_moments: list[float] | None,
    cation_elements: set[str],
    anion_elements: set[str],
) -> tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], dict[str, object]]:
    candidate_origins = [
        tuple(origin)
        for origin in product(*(range(supercell[axis] - keep_cells[axis] + 1) for axis in range(3)))
    ]
    if not candidate_origins:
        raise ValueError(f"Cannot keep {keep_cells} cells from source supercell {supercell}.")

    symbols = atom_symbols(structure)
    cation_indices = selected_cation_indices(symbols, cation_elements, anion_elements)
    cation_counts = Counter(symbols[index] for index in cation_indices)
    max_count = max(cation_counts.values(), default=0)
    minority_symbols = {symbol for symbol, count in cation_counts.items() if count < max_count}
    if not minority_symbols:
        origin = (0, 0, 0)
        ranges = crop_ranges_from_origin(supercell, keep_cells, origin)
        kept = indices_in_fraction_ranges(structure.scaled_positions, ranges)
        meta = {
            "selection_policy": "defect_preserving",
            "selection_reason": "regular_origin_crop_no_minority_cation",
            "crop_origin_cells": list(origin),
            "minority_cation_elements": [],
            "minority_cations_available": 0,
            "minority_cations_kept": 0,
            "charge_variant_cations_available": 0,
            "charge_variant_cations_kept": 0,
            "source_magnetic_signature": magnetic_signature(symbols, cation_indices, origin_indices, source_moments),
            "crop_magnetic_signature": magnetic_signature(symbols, sorted(kept & set(cation_indices)), origin_indices, source_moments),
        }
        return ranges, meta

    charge_variant_indices = charge_variant_cation_indices(symbols, cation_indices, origin_indices, source_moments)
    source_magnetic_buckets = magnetic_buckets(symbols, cation_indices, origin_indices, source_moments)

    best_origin = candidate_origins[0]
    best_ranges = crop_ranges_from_origin(supercell, keep_cells, best_origin)
    best_score: tuple[int, int, int, int, int, int, int] | None = None
    best_kept: set[int] = set()
    for origin in candidate_origins:
        ranges = crop_ranges_from_origin(supercell, keep_cells, origin)
        kept = indices_in_fraction_ranges(structure.scaled_positions, ranges)
        kept_cations = sorted(kept & set(cation_indices))
        kept_magnetic_buckets = magnetic_buckets(symbols, kept_cations, origin_indices, source_moments)
        score = (
            sum(1 for index in kept if symbols[index] in minority_symbols),
            sum(1 for index in kept if index in charge_variant_indices),
            len(source_magnetic_buckets & kept_magnetic_buckets),
            sum(1 for index in kept if index in cation_indices),
            -origin[0],
            -origin[1],
            -origin[2],
        )
        if best_score is None or score > best_score:
            best_origin = origin
            best_ranges = ranges
            best_score = score
            best_kept = kept

    meta = {
        "selection_policy": "defect_preserving",
        "crop_origin_cells": list(best_origin),
        "minority_cation_elements": sorted(minority_symbols),
        "minority_cations_available": sum(1 for index in cation_indices if symbols[index] in minority_symbols),
        "minority_cations_kept": sum(1 for index in best_kept if symbols[index] in minority_symbols),
        "charge_variant_cations_available": len(charge_variant_indices),
        "charge_variant_cations_kept": sum(1 for index in best_kept if index in charge_variant_indices),
        "magnetic_signature_note": "Signs are inferred from INCAR MAGMOM for cations: positive, negative, or zero.",
        "source_magnetic_signature": magnetic_signature(symbols, cation_indices, origin_indices, source_moments),
        "crop_magnetic_signature": magnetic_signature(
            symbols,
            sorted(best_kept & set(cation_indices)),
            origin_indices,
            source_moments,
        ),
    }
    return best_ranges, meta


def indices_in_fraction_ranges(
    positions: list[list[float]],
    ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> set[int]:
    tol = 1.0e-10
    return {
        index
        for index, position in enumerate(positions)
        if all(ranges[axis][0] - tol <= position[axis] < ranges[axis][1] - tol for axis in range(3))
    }


def charge_variant_cation_indices(
    symbols: list[str],
    cation_indices: list[int],
    origin_indices: list[int],
    source_moments: list[float] | None,
) -> set[int]:
    if source_moments is None:
        return set()
    by_symbol: dict[str, list[tuple[int, float]]] = {}
    for index in cation_indices:
        if origin_indices[index] >= len(source_moments):
            continue
        by_symbol.setdefault(symbols[index], []).append((index, abs(source_moments[origin_indices[index]])))
    variants: set[int] = set()
    for group in by_symbol.values():
        if len(group) < 2:
            continue
        bins = Counter(round(moment, 3) for _index, moment in group)
        dominant_value, dominant_count = bins.most_common(1)[0]
        if dominant_count == len(group):
            continue
        for index, moment in group:
            if round(moment, 3) != dominant_value:
                variants.add(index)
    return variants


def magnetic_buckets(
    symbols: list[str],
    indices: list[int],
    origin_indices: list[int],
    source_moments: list[float] | None,
) -> set[tuple[str, str]]:
    if source_moments is None:
        return set()
    return {
        (symbols[index], moment_sign(source_moments[origin_indices[index]]))
        for index in indices
        if origin_indices[index] < len(source_moments)
    }


def magnetic_signature(
    symbols: list[str],
    indices: list[int],
    origin_indices: list[int],
    source_moments: list[float] | None,
) -> dict[str, dict[str, int]]:
    signature: dict[str, dict[str, int]] = {}
    if source_moments is None:
        return signature
    for index in indices:
        if origin_indices[index] >= len(source_moments):
            continue
        symbol = symbols[index]
        sign = moment_sign(source_moments[origin_indices[index]])
        signature.setdefault(symbol, {"positive": 0, "negative": 0, "zero": 0})
        signature[symbol][sign] += 1
    return signature


def final_cation_magmom_summary(
    species: PoscarSpecies,
    moments: list[float] | None,
    cation_elements: set[str],
    anion_elements: set[str],
    decimals: int,
) -> dict[str, dict[str, object]]:
    if moments is None:
        return {}
    symbols: list[str] = []
    for symbol, count in zip(species.symbols, species.counts):
        symbols.extend([symbol] * count)
    cation_indices = selected_cation_indices(symbols, cation_elements, anion_elements)
    summary: dict[str, dict[str, object]] = {}
    for index in cation_indices:
        symbol = symbols[index]
        value = moments[index]
        rounded_value = round(value, decimals)
        entry = summary.setdefault(
            symbol,
            {
                "count": 0,
                "positive": 0,
                "negative": 0,
                "zero": 0,
                "sum": 0.0,
                "min": rounded_value,
                "max": rounded_value,
                "moment_histogram": {},
                "unique_abs_moments": [],
            },
        )
        entry["count"] = int(entry["count"]) + 1
        sign = moment_sign(value)
        entry[sign] = int(entry[sign]) + 1
        entry["sum"] = float(entry["sum"]) + value
        entry["min"] = min(float(entry["min"]), rounded_value)
        entry["max"] = max(float(entry["max"]), rounded_value)
        histogram = entry["moment_histogram"]
        if isinstance(histogram, dict):
            key = f"{rounded_value:g}"
            histogram[key] = int(histogram.get(key, 0)) + 1
    for symbol, entry in summary.items():
        count = int(entry["count"])
        total = float(entry["sum"])
        histogram = entry["moment_histogram"]
        abs_values = []
        if isinstance(histogram, dict):
            abs_values = sorted({round(abs(float(value)), decimals) for value in histogram})
        entry["sum"] = round(total, decimals)
        entry["mean"] = round(total / count, decimals) if count else 0.0
        entry["unique_abs_moments"] = abs_values
        summary[symbol] = entry
    return {symbol: summary[symbol] for symbol in species.symbols if symbol in summary}


def format_cation_magmom_summary(summary: dict[str, dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for symbol, entry in summary.items():
        abs_values = entry.get("unique_abs_moments", [])
        if isinstance(abs_values, list):
            abs_text = ",".join(f"{float(value):g}" for value in abs_values) if abs_values else "none"
        else:
            abs_text = "none"
        lines.append(
            f"{symbol}: n={entry.get('count', 0)} "
            f"+{entry.get('positive', 0)} -{entry.get('negative', 0)} 0={entry.get('zero', 0)} "
            f"sum={float(entry.get('sum', 0.0)):g} mean={float(entry.get('mean', 0.0)):g} "
            f"abs=[{abs_text}]"
        )
    return lines


def compare_cation_magmom_summaries(
    source: dict[str, dict[str, object]],
    output: dict[str, dict[str, object]],
    output_order: list[str],
    decimals: int,
) -> dict[str, dict[str, object]]:
    symbols = [symbol for symbol in output_order if symbol in source or symbol in output]
    symbols.extend(symbol for symbol in source if symbol not in symbols)
    comparison: dict[str, dict[str, object]] = {}
    for symbol in symbols:
        source_entry = source.get(symbol, {})
        output_entry = output.get(symbol, {})
        comparison[symbol] = {
            "source_count": int(source_entry.get("count", 0)),
            "output_count": int(output_entry.get("count", 0)),
            "count_delta": int(output_entry.get("count", 0)) - int(source_entry.get("count", 0)),
            "source_positive": int(source_entry.get("positive", 0)),
            "output_positive": int(output_entry.get("positive", 0)),
            "positive_delta": int(output_entry.get("positive", 0)) - int(source_entry.get("positive", 0)),
            "source_negative": int(source_entry.get("negative", 0)),
            "output_negative": int(output_entry.get("negative", 0)),
            "negative_delta": int(output_entry.get("negative", 0)) - int(source_entry.get("negative", 0)),
            "source_zero": int(source_entry.get("zero", 0)),
            "output_zero": int(output_entry.get("zero", 0)),
            "zero_delta": int(output_entry.get("zero", 0)) - int(source_entry.get("zero", 0)),
            "source_sum": float(source_entry.get("sum", 0.0)),
            "output_sum": float(output_entry.get("sum", 0.0)),
            "sum_delta": round(float(output_entry.get("sum", 0.0)) - float(source_entry.get("sum", 0.0)), decimals),
            "source_unique_abs_moments": source_entry.get("unique_abs_moments", []),
            "output_unique_abs_moments": output_entry.get("unique_abs_moments", []),
            "unique_abs_moments_match": source_entry.get("unique_abs_moments", []) == output_entry.get("unique_abs_moments", []),
        }
    return comparison


def format_cation_magmom_comparison(comparison: dict[str, dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for symbol, entry in comparison.items():
        lines.append(
            f"{symbol}: n {entry['source_count']}->{entry['output_count']} "
            f"(delta {entry['count_delta']:+d}); "
            f"+/-/0 {entry['source_positive']}/{entry['source_negative']}/{entry['source_zero']}"
            f"->{entry['output_positive']}/{entry['output_negative']}/{entry['output_zero']}; "
            f"sum {float(entry['source_sum']):g}->{float(entry['output_sum']):g} "
            f"(delta {float(entry['sum_delta']):+g}); "
            f"abs_match={entry['unique_abs_moments_match']}"
        )
    return lines


def moment_sign(value: float) -> str:
    if value > 1.0e-8:
        return "positive"
    if value < -1.0e-8:
        return "negative"
    return "zero"


def repeat_structure(
    structure: PoscarStructure,
    origin_indices: list[int],
    repeat: tuple[int, int, int],
) -> tuple[PoscarStructure, list[int]]:
    if any(value <= 0 for value in repeat):
        raise ValueError(f"Repeat values must be positive, got {repeat}.")
    symbols = atom_symbols(structure)
    repeated_symbols: list[str] = []
    repeated_positions: list[list[float]] = []
    repeated_origins: list[int] = []
    for atom_index, (symbol, position) in enumerate(zip(symbols, structure.scaled_positions)):
        for i in range(repeat[0]):
            for j in range(repeat[1]):
                for k in range(repeat[2]):
                    repeated_symbols.append(symbol)
                    repeated_positions.append(
                        [
                            (position[0] + i) / repeat[0],
                            (position[1] + j) / repeat[1],
                            (position[2] + k) / repeat[2],
                        ]
                    )
                    repeated_origins.append(origin_indices[atom_index])
    cell = [
        [value * repeat[axis] for value in vector]
        for axis, vector in enumerate(structure.cell)
    ]
    return grouped_structure(repeated_symbols, repeated_positions, repeated_origins, cell, structure.species.symbols)


def crop_structure_fraction(
    structure: PoscarStructure,
    origin_indices: list[int],
    ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    *,
    expected_cation_count: int | None = None,
    cation_elements: set[str] | None = None,
    anion_elements: set[str] | None = None,
) -> tuple[PoscarStructure, list[int], dict[str, object]]:
    for lower, upper in ranges:
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError(f"Crop fractions must satisfy 0 <= lower < upper <= 1, got {ranges}.")
    symbols = atom_symbols(structure)
    kept_indices = indices_in_fraction_ranges(structure.scaled_positions, ranges)
    repair_meta: dict[str, object] = {}
    if expected_cation_count is not None:
        cation_indices = selected_cation_indices(symbols, cation_elements or set(), anion_elements or {"O"})
        kept_cations = sorted(kept_indices & set(cation_indices))
        if len(kept_cations) != expected_cation_count:
            cation_count_targets = balanced_cation_count_targets(symbols, cation_indices, expected_cation_count)
            repaired_cations = repaired_crop_cation_indices(
                structure.scaled_positions,
                symbols,
                cation_indices,
                ranges,
                expected_cation_count,
                cation_count_targets,
            )
            kept_indices = (kept_indices - set(cation_indices)) | set(repaired_cations)
            repair_meta = {
                "cation_boundary_repair": True,
                "cation_count_before_repair": len(kept_cations),
                "cation_count_after_repair": len(repaired_cations),
                "expected_cation_count": expected_cation_count,
                "cation_species_target_counts": cation_count_targets,
            }
        else:
            repair_meta = {
                "cation_boundary_repair": False,
                "cation_count_before_repair": len(kept_cations),
                "cation_count_after_repair": len(kept_cations),
                "expected_cation_count": expected_cation_count,
            }

    kept_symbols: list[str] = []
    kept_positions: list[list[float]] = []
    kept_origins: list[int] = []
    for atom_index, (symbol, position) in enumerate(zip(symbols, structure.scaled_positions)):
        if atom_index in kept_indices:
            kept_symbols.append(symbol)
            kept_positions.append(
                [
                    (position[axis] - ranges[axis][0]) / (ranges[axis][1] - ranges[axis][0])
                    for axis in range(3)
                ]
            )
            kept_origins.append(origin_indices[atom_index])
    if not kept_symbols:
        raise ValueError(f"Source crop {ranges} removed all atoms.")
    cell = [
        [value * (ranges[axis][1] - ranges[axis][0]) for value in vector]
        for axis, vector in enumerate(structure.cell)
    ]
    prepared, prepared_origins = grouped_structure(kept_symbols, kept_positions, kept_origins, cell, structure.species.symbols)
    return prepared, prepared_origins, repair_meta


def repaired_crop_cation_indices(
    positions: list[list[float]],
    symbols: list[str],
    cation_indices: list[int],
    ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    expected_count: int,
    species_target_counts: dict[str, int] | None = None,
) -> list[int]:
    if expected_count > len(cation_indices):
        raise ValueError(f"Cannot repair crop to {expected_count} cations; source has only {len(cation_indices)} cations.")
    if species_target_counts:
        repaired: list[int] = []
        for symbol, count in species_target_counts.items():
            species_indices = [index for index in cation_indices if symbols[index] == symbol]
            if count > len(species_indices):
                raise ValueError(f"Cannot repair crop to {count} {symbol} cations; source has only {len(species_indices)}.")
            repaired.extend(
                sorted(
                    species_indices,
                    key=lambda index: (
                        distance_to_fraction_box(positions[index], ranges),
                        index,
                    ),
                )[:count]
            )
        if len(repaired) == expected_count:
            return sorted(repaired)
    ranked = sorted(
        cation_indices,
        key=lambda index: (
            distance_to_fraction_box(positions[index], ranges),
            index,
        ),
    )
    return sorted(ranked[:expected_count])


def balanced_cation_count_targets(
    symbols: list[str],
    cation_indices: list[int],
    expected_count: int,
) -> dict[str, int] | None:
    counts = Counter(symbols[index] for index in cation_indices)
    if len(counts) < 2:
        return None
    if len(set(counts.values())) != 1:
        return None
    if expected_count % len(counts) != 0:
        return None
    per_species = expected_count // len(counts)
    return {symbol: per_species for symbol in sorted(counts)}


def distance_to_fraction_box(
    position: list[float],
    ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> float:
    distance_sq = 0.0
    for axis, value in enumerate(position):
        lower, upper = ranges[axis]
        if value < lower:
            distance_sq += (lower - value) ** 2
        elif value >= upper:
            distance_sq += (value - upper) ** 2
    return math.sqrt(distance_sq)


def grouped_structure(
    symbols: list[str],
    positions: list[list[float]],
    origins: list[int],
    cell: list[list[float]],
    order: list[str],
) -> tuple[PoscarStructure, list[int]]:
    full_order = complete_species_order(order, symbols)
    grouped_symbols: list[str] = []
    grouped_positions: list[list[float]] = []
    grouped_origins: list[int] = []
    counts: list[int] = []
    for symbol in full_order:
        group = [index for index, item in enumerate(symbols) if item == symbol]
        if not group:
            continue
        grouped_symbols.append(symbol)
        counts.append(len(group))
        grouped_positions.extend(positions[index] for index in group)
        grouped_origins.extend(origins[index] for index in group)
    return (
        PoscarStructure(
            species=PoscarSpecies(symbols=grouped_symbols, counts=counts),
            cell=cell,
            scaled_positions=grouped_positions,
        ),
        grouped_origins,
    )


def selected_cation_indices(symbols: list[str], cation_elements: set[str], anion_elements: set[str]) -> list[int]:
    if cation_elements:
        return [index for index, symbol in enumerate(symbols) if symbol in cation_elements]
    return [index for index, symbol in enumerate(symbols) if symbol not in anion_elements]


def align_source_cation_origin(
    source: PoscarStructure,
    source_indices: list[int],
    target: PoscarStructure,
    target_indices: list[int],
) -> tuple[PoscarStructure, dict[str, object]]:
    before = nearest_site_assignment(source.scaled_positions, source_indices, target.scaled_positions, target_indices, target.cell)
    before_worst = max((match.distance_A for match in before), default=0.0)
    best_shift = [0.0, 0.0, 0.0]
    best_matches = before
    best_score = assignment_score(before)
    candidates = cation_origin_shift_candidates(source.scaled_positions, source_indices, target.scaled_positions, target_indices)
    for shift in candidates:
        shifted_positions = shift_fractional_positions(source.scaled_positions, shift)
        matches = nearest_site_assignment(shifted_positions, source_indices, target.scaled_positions, target_indices, target.cell)
        score = assignment_score(matches)
        if score < best_score:
            best_shift = shift
            best_matches = matches
            best_score = score
    shifted_source = PoscarStructure(
        species=source.species,
        cell=source.cell,
        scaled_positions=shift_fractional_positions(source.scaled_positions, best_shift),
    )
    return shifted_source, {
        "enabled": True,
        "shift_fractional": [round(wrap_fractional_delta(value), 12) for value in best_shift],
        "max_cation_distance_before_A": before_worst,
        "max_cation_distance_after_A": best_score[0],
        "mean_cation_distance_after_A": best_score[1],
        "candidate_count": len(candidates),
        "improved": assignment_score(best_matches) < assignment_score(before),
    }


def assignment_score(matches: list[SiteMatch]) -> tuple[float, float]:
    if not matches:
        return (0.0, 0.0)
    worst = max(match.distance_A for match in matches)
    mean = sum(match.distance_A for match in matches) / len(matches)
    return (worst, mean)


def cation_origin_shift_candidates(
    source_positions: list[list[float]],
    source_indices: list[int],
    target_positions: list[list[float]],
    target_indices: list[int],
) -> list[list[float]]:
    if not source_indices or not target_indices:
        return [[0.0, 0.0, 0.0]]
    reference = source_positions[source_indices[0]]
    seen: set[tuple[float, float, float]] = set()
    candidates: list[list[float]] = []
    for shift in [[0.0, 0.0, 0.0], *([target_positions[index][axis] - reference[axis] for axis in range(3)] for index in target_indices)]:
        normalized = [value % 1.0 for value in shift]
        key = tuple(round(value, 10) for value in normalized)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(normalized)
    return candidates


def shift_fractional_positions(positions: list[list[float]], shift: list[float]) -> list[list[float]]:
    return [[(position[axis] + shift[axis]) % 1.0 for axis in range(3)] for position in positions]


def wrap_fractional_delta(value: float) -> float:
    wrapped = value - round(value)
    if wrapped <= -0.5:
        wrapped += 1.0
    if wrapped > 0.5:
        wrapped -= 1.0
    return wrapped


def nearest_site_assignment(
    source_positions: list[list[float]],
    source_indices: list[int],
    target_positions: list[list[float]],
    target_indices: list[int],
    cell: list[list[float]],
) -> list[SiteMatch]:
    candidates: list[tuple[float, int, int]] = []
    for target_index in target_indices:
        for source_index in source_indices:
            distance = fractional_distance_A(target_positions[target_index], source_positions[source_index], cell)
            candidates.append((distance, target_index, source_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_targets: set[int] = set()
    used_sources: set[int] = set()
    matches: list[SiteMatch] = []
    for distance, target_index, source_index in candidates:
        if target_index in used_targets or source_index in used_sources:
            continue
        used_targets.add(target_index)
        used_sources.add(source_index)
        matches.append(SiteMatch(target_index=target_index, source_index=source_index, distance_A=distance))
        if len(matches) == len(target_indices):
            break
    if len(matches) != len(target_indices):
        raise RuntimeError("Could not build a complete source-target site assignment.")
    return sorted(matches, key=lambda match: match.target_index)


def fractional_distance_A(left: list[float], right: list[float], cell: list[list[float]]) -> float:
    diff = [left[i] - right[i] for i in range(3)]
    diff = [value - round(value) for value in diff]
    cart = [
        diff[0] * cell[0][j] + diff[1] * cell[1][j] + diff[2] * cell[2][j]
        for j in range(3)
    ]
    return math.sqrt(sum(value * value for value in cart))


def default_species_order(source_order: list[str], target_order: list[str]) -> list[str]:
    order: list[str] = []
    for symbol in [*source_order, *target_order]:
        if symbol not in order:
            order.append(symbol)
    return order


def default_projection_species_order(
    source_order: list[str],
    target_order: list[str],
    projected_symbols: list[str],
    cation_order: list[str],
    cation_elements: set[str],
    anion_elements: set[str],
) -> list[str]:
    present = set(projected_symbols)
    cations = {symbol for symbol in present if symbol in cation_elements} if cation_elements else present - anion_elements
    anions = present & anion_elements

    order: list[str] = []
    for symbol in cation_order:
        if symbol in cations and symbol not in order:
            order.append(symbol)
    for symbol in target_order:
        if symbol in cations and symbol not in order:
            order.append(symbol)
    for symbol in source_order:
        if symbol in cations and symbol not in order:
            order.append(symbol)
    for symbol in target_order:
        if symbol in anions and symbol not in order:
            order.append(symbol)
    for symbol in source_order:
        if symbol in anions and symbol not in order:
            order.append(symbol)
    for symbol in default_species_order(source_order, target_order):
        if symbol in present and symbol not in order:
            order.append(symbol)
    return order


def complete_species_order(order: list[str], symbols: list[str]) -> list[str]:
    complete = [symbol for symbol in order if symbol in symbols]
    for symbol in symbols:
        if symbol not in complete:
            complete.append(symbol)
    return complete


def grouped_species_and_indices(symbols: list[str], order: list[str]) -> tuple[PoscarSpecies, list[int]]:
    counts: list[int] = []
    indices: list[int] = []
    kept_symbols: list[str] = []
    for symbol in order:
        group = [index for index, item in enumerate(symbols) if item == symbol]
        if not group:
            continue
        kept_symbols.append(symbol)
        counts.append(len(group))
        indices.extend(group)
    return PoscarSpecies(symbols=kept_symbols, counts=counts), indices


def write_poscar_text(
    comment: str,
    species: PoscarSpecies,
    cell: list[list[float]],
    scaled_positions: list[list[float]],
) -> str:
    lines = [comment, "1.0"]
    lines.extend("  " + "  ".join(f"{value: .16f}" for value in vector) for vector in cell)
    lines.append("  " + "  ".join(species.symbols))
    lines.append("  " + "  ".join(str(count) for count in species.counts))
    lines.append("Direct")
    lines.extend("  " + "  ".join(f"{value % 1.0: .16f}" for value in position) for position in scaled_positions)
    return "\n".join(lines) + "\n"


def projected_magmom_values(
    source: PoscarStructure,
    target: PoscarStructure,
    source_symbols: list[str],
    target_symbols: list[str],
    projected_symbols: list[str],
    source_by_target: dict[int, int],
    source_moments: list[float] | None,
    cation_elements: set[str],
    anion_elements: set[str],
) -> list[float] | None:
    if source_moments is None:
        return None
    moments = [0.0] * len(target_symbols)
    for target_index, source_index in source_by_target.items():
        moments[target_index] = source_moments[source_index]

    mapped_targets = set(source_by_target)
    mapped_sources = set(source_by_target.values())
    target_non_cations = [index for index in range(len(target_symbols)) if index not in mapped_targets]
    source_non_cations = [index for index in range(len(source_symbols)) if index not in mapped_sources]
    for symbol in sorted(set(projected_symbols[index] for index in target_non_cations)):
        target_group = [index for index in target_non_cations if projected_symbols[index] == symbol]
        source_group = [index for index in source_non_cations if source_symbols[index] == symbol]
        if len(target_group) != len(source_group):
            continue
        matches = nearest_site_assignment(
            source.scaled_positions,
            source_group,
            target.scaled_positions,
            target_group,
            target.cell,
        )
        for match in matches:
            moments[match.target_index] = source_moments[match.source_index]
    return moments


def species_with_nonzero_moments(species: PoscarSpecies, moments: list[float]) -> list[str]:
    selected: list[str] = []
    cursor = 0
    for symbol, count in zip(species.symbols, species.counts):
        block = moments[cursor : cursor + count]
        if any(abs(value) > 1.0e-12 for value in block):
            selected.append(symbol)
        cursor += count
    return selected


def replace_or_append_magmom_text(text: str, magmom_line: str) -> str:
    lines = text.splitlines()
    line_index, _line = find_magmom_line(lines)
    if line_index is None:
        lines.append(magmom_line)
    else:
        lines[line_index] = magmom_line
    return "\n".join(lines) + "\n"


def write_match_csv(
    path: Path,
    matches: list[SiteMatch],
    source_symbols: list[str],
    target_symbols: list[str],
    source: PoscarStructure,
    target: PoscarStructure,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_atom",
                "target_element_before",
                "source_atom",
                "source_element",
                "distance_A",
                "target_frac",
                "source_frac",
            ],
        )
        writer.writeheader()
        for match in matches:
            writer.writerow(
                {
                    "target_atom": match.target_index + 1,
                    "target_element_before": target_symbols[match.target_index],
                    "source_atom": match.source_index + 1,
                    "source_element": source_symbols[match.source_index],
                    "distance_A": f"{match.distance_A:.8f}",
                    "target_frac": " ".join(f"{value:.10f}" for value in target.scaled_positions[match.target_index]),
                    "source_frac": " ".join(f"{value:.10f}" for value in source.scaled_positions[match.source_index]),
                }
            )


def write_plan_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-project-poscar",
        description=(
            "Project element identities from POSCAR A onto relaxed coordinates from POSCAR B. "
            "Cations are matched by nearest periodic fractional site; anions are left as in B."
        ),
    )
    parser.add_argument("--element-poscar", "--poscar-a", dest="source_poscar", type=Path, required=True)
    parser.add_argument("--structure-poscar", "--poscar-b", dest="target_poscar", type=Path, required=True)
    parser.add_argument("--incar-a", "--source-incar", dest="source_incar", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("PROJECTED_POSCAR"))
    parser.add_argument("--out-poscar", type=Path)
    parser.add_argument(
        "--out-source-poscar",
        "--out-prepared-source-poscar",
        dest="prepared_source_poscar",
        type=Path,
        help="Write the repeated/cropped POSCAR A used for projection. Default: POSCAR_A_prepared when source operations are applied.",
    )
    parser.add_argument("--out-incar", type=Path)
    parser.add_argument("--match-csv", type=Path)
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--cation-elements", action="append", default=[], help="Projected sublattice elements, e.g. U,Gd. Default: all non-anion elements.")
    parser.add_argument("--anion-elements", action="append", default=["O"], help="Elements to leave on the anion/non-projected sublattice. Default: O.")
    parser.add_argument("--species-order", action="append", default=[], help="Output POSCAR species order, comma-separated or repeatable.")
    parser.add_argument("--source-repeat", help="Repeat POSCAR A before projection, e.g. 2x2x2.")
    parser.add_argument("--target-repeat", help="Repeat POSCAR B before projection/output, e.g. 1x2x2.")
    parser.add_argument("--source-supercell", help="Cell count represented by POSCAR A, e.g. 2x3x3.")
    parser.add_argument("--source-keep-cells", help="Crop POSCAR A to a matching cell block, e.g. 2x2x2.")
    parser.add_argument(
        "--source-reduce-cells",
        help=(
            "Fold POSCAR A cation sites into this smaller source-cell grid and select a representative "
            "cation per folded site, e.g. 2x2x2. Requires --source-supercell and cannot be combined "
            "with --source-keep-cells."
        ),
    )
    parser.add_argument(
        "--source-reduce-beam-width",
        type=int,
        default=256,
        help="Representative reduction search width. Larger values preserve composition/MAGMOM statistics more carefully. Default: 256.",
    )
    parser.add_argument(
        "--source-reduce-site-tolerance",
        type=float,
        default=0.05,
        help=(
            "Fractional tolerance in the reduced source cell for clustering folded cation sites. "
            "Increase slightly if POSCAR A is relaxed/noisy. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--source-crop-policy",
        choices=["defect-preserving", "origin"],
        default="defect-preserving",
        help="How to choose a --source-keep-cells crop window. Default preserves minority/charge-coupled cations.",
    )
    parser.add_argument(
        "--source-crop-fraction",
        help="Explicit POSCAR A fractional crop, e.g. 0:1,0:2/3,0:2/3. Applied after --source-repeat.",
    )
    parser.add_argument(
        "--scale-target-volume-to-source",
        action="store_true",
        help="Uniformly scale POSCAR B cell to the prepared POSCAR A volume before matching/output; B fractional coordinates are preserved.",
    )
    parser.add_argument(
        "--no-align-cation-origin",
        action="store_true",
        help="Disable automatic global fractional-origin alignment of source cation sites before nearest-site matching.",
    )
    parser.add_argument("--max-cation-distance", type=float, default=1.5, help="Fail if any projected cation match exceeds this distance in Angstrom.")
    parser.add_argument("--allow-large-cation-distance", action="store_true", help="Warn instead of failing when the cation match exceeds the distance limit.")
    parser.add_argument(
        "--allow-magmom-mismatch",
        action="store_true",
        help="Warn instead of failing if projected cation MAGMOM signs/counts/magnitudes differ from prepared source A.",
    )
    parser.add_argument("--magmom-decimals", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    outdir = args.outdir.expanduser().resolve()
    output_poscar = (args.out_poscar or (outdir / "POSCAR")).expanduser().resolve()
    output_incar = None
    if args.source_incar is not None:
        output_incar = (args.out_incar or (outdir / "INCAR")).expanduser().resolve()
    result = project_poscar_elements(
        args.source_poscar,
        args.target_poscar,
        output_poscar=output_poscar,
        prepared_source_poscar=args.prepared_source_poscar,
        source_incar=args.source_incar,
        output_incar=output_incar,
        match_csv=args.match_csv or (outdir / "poscar_projection_map.csv"),
        plan_json=args.plan_json or (outdir / "poscar_projection_plan.json"),
        cation_elements=parse_element_list(args.cation_elements),
        anion_elements=parse_element_list(args.anion_elements),
        species_order=parse_element_list(args.species_order),
        max_cation_distance_A=args.max_cation_distance,
        strict=not args.allow_large_cation_distance,
        magmom_decimals=args.magmom_decimals,
        source_repeat=parse_repeat(args.source_repeat) if args.source_repeat else None,
        target_repeat=parse_repeat(args.target_repeat) if args.target_repeat else None,
        source_supercell=parse_repeat(args.source_supercell) if args.source_supercell else None,
        source_keep_cells=parse_repeat(args.source_keep_cells) if args.source_keep_cells else None,
        source_reduce_cells=parse_repeat(args.source_reduce_cells) if args.source_reduce_cells else None,
        source_reduce_beam_width=args.source_reduce_beam_width,
        source_reduce_site_tolerance=args.source_reduce_site_tolerance,
        source_crop_fraction=parse_fraction_ranges(args.source_crop_fraction) if args.source_crop_fraction else None,
        source_crop_policy=args.source_crop_policy.replace("-", "_"),
        scale_target_volume_to_source=args.scale_target_volume_to_source,
        align_cation_origin=not args.no_align_cation_origin,
        strict_magmom_preservation=not args.allow_magmom_mismatch,
    )
    print(f"Projected POSCAR : {result.output_poscar}")
    if result.prepared_source_poscar:
        print(f"Prepared source  : {result.prepared_source_poscar}")
    if result.output_incar:
        print(f"Projected INCAR  : {result.output_incar}")
        if result.output_magmom_count is not None:
            print(f"Projected MAGMOM : {result.output_magmom_count} values for {result.output_species.total_atoms} atoms")
    print(f"Projection map   : {result.match_csv}")
    print(f"Projection plan  : {result.plan_json}")
    print(f"Cation matches   : {len(result.cation_matches)}")
    print(f"Worst distance   : {result.max_cation_distance_A:.6g} A")
    print(
        "Output species   : "
        + " ".join(f"{symbol}:{count}" for symbol, count in zip(result.output_species.symbols, result.output_species.counts))
    )
    for operation in json.loads(result.plan_json.read_text(encoding="utf-8")).get("source_operations", []):
        if operation.get("kind") == "source_representative_reduce":
            print(
                "Source reduction: "
                f"{'x'.join(str(value) for value in operation.get('source_supercell', []))}"
                f" -> {'x'.join(str(value) for value in operation.get('reduce_cells', []))}; "
                f"selected {operation.get('selected_cation_count')} cations; "
                f"score={float(operation.get('selection_score', 0.0)):g}"
            )
    if result.source_cation_magmom_summary:
        print("Prepared A MAGMOM:")
        for line in format_cation_magmom_summary(result.source_cation_magmom_summary):
            print(f"  {line}")
    if result.cation_magmom_summary:
        print("Projected C MAGMOM:")
        for line in format_cation_magmom_summary(result.cation_magmom_summary):
            print(f"  {line}")
    if result.cation_magmom_comparison:
        print("Cation MAGMOM check:")
        for line in format_cation_magmom_comparison(result.cation_magmom_comparison):
            print(f"  {line}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")


def parse_repeat(raw: str) -> tuple[int, int, int]:
    parts = raw.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise ValueError(f"Expected repeat AxBxC, got {raw!r}.")
    repeat = tuple(int(part.strip()) for part in parts)
    if any(value <= 0 for value in repeat):
        raise ValueError(f"Repeat values must be positive, got {raw!r}.")
    return repeat


def parse_fraction_ranges(
    raw: str,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if len(pieces) != 3:
        raise ValueError(f"Expected three fractional ranges, got {raw!r}.")
    ranges = []
    for piece in pieces:
        if ":" not in piece:
            raise ValueError(f"Expected range lower:upper, got {piece!r}.")
        lower, upper = piece.split(":", 1)
        ranges.append((parse_fraction_value(lower), parse_fraction_value(upper)))
    return tuple(ranges)  # type: ignore[return-value]


def parse_fraction_value(raw: str) -> float:
    raw = raw.strip()
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        return float(numerator) / float(denominator)
    return float(raw)


if __name__ == "__main__":
    main()
