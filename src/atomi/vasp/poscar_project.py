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
    existing_magmom_values,
    find_magmom_line,
    format_magmom_line,
    parse_element_list,
    read_poscar_structure,
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
    output_incar: Path | None
    match_csv: Path
    plan_json: Path
    cation_matches: list[SiteMatch]
    max_cation_distance_A: float
    output_species: PoscarSpecies
    warnings: list[str]


@dataclass(frozen=True)
class PreparedStructure:
    structure: PoscarStructure
    origin_indices: list[int]
    operations: list[dict[str, object]]


def project_poscar_elements(
    source_poscar: Path,
    target_poscar: Path,
    *,
    output_poscar: Path,
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
    source_crop_fraction: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    source_crop_policy: str = "defect_preserving",
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
    raw_source_moments = existing_magmom_values(source_incar, len(raw_source_symbols)) if source_incar is not None else None
    anion_set = set(anion_elements or ["O"])
    cation_set = set(cation_elements or [])
    source_prepared = prepare_structure(
        source_raw,
        repeat=source_repeat,
        crop_supercell=source_supercell,
        crop_keep_cells=source_keep_cells,
        crop_fraction=source_crop_fraction,
        crop_policy=source_crop_policy,
        source_moments=raw_source_moments,
        cation_elements=cation_set,
        anion_elements=anion_set,
    )
    target_prepared = prepare_structure(target_raw, repeat=target_repeat)
    source = source_prepared.structure
    target = target_prepared.structure
    source_symbols = atom_symbols(source)
    target_symbols = atom_symbols(target)
    source_cations = selected_cation_indices(source_symbols, cation_set, anion_set)
    target_cations = selected_cation_indices(target_symbols, cation_set, anion_set)
    if len(source_cations) != len(target_cations):
        raise ValueError(
            "Source/target cation counts differ: "
            f"{len(source_cations)} in {source_poscar}, {len(target_cations)} in {target_poscar}. "
            "Use --cation-elements/--anion-elements to define the projected sublattice, or use "
            "--source-repeat/--target-repeat/--source-supercell/--source-keep-cells to make the cells commensurate."
        )
    if not source_cations:
        raise ValueError("No cation sites found to project.")

    matches = nearest_site_assignment(
        source.scaled_positions,
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

    output_moments: list[float] | None = None
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
            incar_text = replace_or_append_magmom_text(incar_text, magmom_line)
        output_incar.parent.mkdir(parents=True, exist_ok=True)
        output_incar.write_text(incar_text, encoding="utf-8")

    write_match_csv(match_csv, matches, source_symbols, target_symbols, source, target)
    write_plan_json(
        plan_json,
        {
            "schema": "atomi.vasp.poscar_projection.v1",
            "source_poscar": str(source_poscar),
            "target_poscar": str(target_poscar),
            "source_incar": str(source_incar) if source_incar else "",
            "output_poscar": str(output_poscar),
            "output_incar": str(output_incar) if output_incar else "",
            "source_operations": source_prepared.operations,
            "target_operations": target_prepared.operations,
            "cation_elements": sorted(cation_set),
            "anion_elements": sorted(anion_set),
            "source_cation_count": len(source_cations),
            "target_cation_count": len(target_cations),
            "max_cation_distance_A": worst,
            "distance_limit_A": max_cation_distance_A,
            "species_order": output_species.symbols,
            "species_counts": dict(zip(output_species.symbols, output_species.counts)),
            "warnings": warnings,
            "notes": [
                "Cation species and cation MAGMOM values were projected from source POSCAR A.",
                "Target POSCAR B supplied cell and coordinates.",
                "Non-cation sites are kept from target POSCAR B; their MAGMOMs are matched by element/position when possible, otherwise set to 0.",
            ],
        },
    )
    return ProjectionResult(
        source_poscar=source_poscar,
        target_poscar=target_poscar,
        output_poscar=output_poscar,
        output_incar=output_incar,
        match_csv=match_csv,
        plan_json=plan_json,
        cation_matches=matches,
        max_cation_distance_A=worst,
        output_species=output_species,
        warnings=warnings,
    )


def atom_symbols(structure: PoscarStructure) -> list[str]:
    symbols: list[str] = []
    for symbol, count in zip(structure.species.symbols, structure.species.counts):
        symbols.extend([symbol] * count)
    return symbols


def prepare_structure(
    structure: PoscarStructure,
    *,
    repeat: tuple[int, int, int] | None = None,
    crop_supercell: tuple[int, int, int] | None = None,
    crop_keep_cells: tuple[int, int, int] | None = None,
    crop_fraction: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    crop_policy: str = "defect_preserving",
    source_moments: list[float] | None = None,
    cation_elements: set[str] | None = None,
    anion_elements: set[str] | None = None,
) -> PreparedStructure:
    origin_indices = list(range(structure.species.total_atoms))
    operations: list[dict[str, object]] = []
    if repeat is not None and repeat != (1, 1, 1):
        structure, origin_indices = repeat_structure(structure, origin_indices, repeat)
        operations.append({"kind": "repeat", "repeat": list(repeat)})
    if crop_supercell is not None or crop_keep_cells is not None:
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
        structure, origin_indices = crop_structure_fraction(structure, origin_indices, ranges)
        operations.append(
            {
                "kind": "source_keep_cells",
                "source_supercell": list(crop_supercell),
                "keep_cells": list(crop_keep_cells),
                "fraction_ranges": [list(item) for item in ranges],
                **crop_meta,
            }
        )
    if crop_fraction is not None:
        structure, origin_indices = crop_structure_fraction(structure, origin_indices, crop_fraction)
        operations.append({"kind": "fraction_crop", "fraction_ranges": [list(item) for item in crop_fraction]})
    return PreparedStructure(structure=structure, origin_indices=origin_indices, operations=operations)


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
    charge_variant_indices = charge_variant_cation_indices(symbols, cation_indices, origin_indices, source_moments)

    best_origin = candidate_origins[0]
    best_ranges = crop_ranges_from_origin(supercell, keep_cells, best_origin)
    best_score: tuple[int, int, int, int, int, int] | None = None
    best_kept: set[int] = set()
    for origin in candidate_origins:
        ranges = crop_ranges_from_origin(supercell, keep_cells, origin)
        kept = indices_in_fraction_ranges(structure.scaled_positions, ranges)
        score = (
            sum(1 for index in kept if symbols[index] in minority_symbols),
            sum(1 for index in kept if index in charge_variant_indices),
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
) -> tuple[PoscarStructure, list[int]]:
    for lower, upper in ranges:
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError(f"Crop fractions must satisfy 0 <= lower < upper <= 1, got {ranges}.")
    symbols = atom_symbols(structure)
    kept_symbols: list[str] = []
    kept_positions: list[list[float]] = []
    kept_origins: list[int] = []
    tol = 1.0e-10
    for atom_index, (symbol, position) in enumerate(zip(symbols, structure.scaled_positions)):
        if all(ranges[axis][0] - tol <= position[axis] < ranges[axis][1] - tol for axis in range(3)):
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
    return grouped_structure(kept_symbols, kept_positions, kept_origins, cell, structure.species.symbols)


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
    cation_elements: set[str],
    anion_elements: set[str],
) -> list[str]:
    present = set(projected_symbols)
    cations = {symbol for symbol in present if symbol in cation_elements} if cation_elements else present - anion_elements
    anions = present & anion_elements

    order: list[str] = []
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
        "--source-crop-policy",
        choices=["defect-preserving", "origin"],
        default="defect-preserving",
        help="How to choose a --source-keep-cells crop window. Default preserves minority/charge-coupled cations.",
    )
    parser.add_argument(
        "--source-crop-fraction",
        help="Explicit POSCAR A fractional crop, e.g. 0:1,0:2/3,0:2/3. Applied after --source-repeat.",
    )
    parser.add_argument("--max-cation-distance", type=float, default=1.5, help="Fail if any projected cation match exceeds this distance in Angstrom.")
    parser.add_argument("--allow-large-cation-distance", action="store_true", help="Warn instead of failing when the cation match exceeds the distance limit.")
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
        source_crop_fraction=parse_fraction_ranges(args.source_crop_fraction) if args.source_crop_fraction else None,
        source_crop_policy=args.source_crop_policy.replace("-", "_"),
    )
    print(f"Projected POSCAR : {result.output_poscar}")
    if result.output_incar:
        print(f"Projected INCAR  : {result.output_incar}")
    print(f"Projection map   : {result.match_csv}")
    print(f"Projection plan  : {result.plan_json}")
    print(f"Cation matches   : {len(result.cation_matches)}")
    print(f"Worst distance   : {result.max_cation_distance_A:.6g} A")
    print(
        "Output species   : "
        + " ".join(f"{symbol}:{count}" for symbol, count in zip(result.output_species.symbols, result.output_species.counts))
    )
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
