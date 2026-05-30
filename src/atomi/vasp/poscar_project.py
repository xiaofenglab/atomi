from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import shutil
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from atomi.vasp.magmom import (
    PoscarSpecies,
    PoscarStructure,
    existing_magmom_values,
    expand_magmom_tokens,
    find_magmom_line,
    format_magmom_line,
    parse_element_list,
    read_poscar_structure,
    reorder_incar_species_tags,
    strip_incar_comment,
)

MOMENT_FAMILY_TOLERANCE = 0.35


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
    anion_vacancy_summary: dict[str, object]
    charge_summary: dict[str, object]
    guest_cation_distance_summary: dict[str, object]
    direct_candidate_summary: dict[str, object]
    randomized_candidate_summary: dict[str, object]
    initial_spin_candidate_summary: dict[str, object]
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
    preserve_anion_vacancies: bool = True,
    oxidation_states: dict[str, float] | None = None,
    magmom_oxidation_states: dict[str, list[tuple[float, float]]] | None = None,
    randomize_candidates: int = 0,
    randomize_pool_size: int | None = None,
    randomize_seed: int = 12345,
    randomize_sublattices: set[str] | None = None,
    randomize_vacancy_label: str = "Va",
    randomize_min_guest_distance_A: float | None = None,
    randomize_max_guest_vacancy_distance_A: float | None = None,
    randomize_atat_atoms: int | None = None,
    randomize_atat_job_name: str = "atat-random",
    randomize_mcsqs_walltime: str = "04:00:00",
    randomize_mcsqs_pair_diameter: float = 6.0,
    randomize_mcsqs_triplet_diameter: float | None = None,
    randomize_mcsqs_quadruplet_diameter: float | None = None,
    randomize_mcsqs_temperature: float | None = None,
    randomize_mcsqs_max_steps: int | None = None,
) -> ProjectionResult:
    """Project cation identities and MAGMOMs from source POSCAR A onto structure POSCAR B."""
    source_poscar = source_poscar.expanduser().resolve()
    target_poscar = target_poscar.expanduser().resolve()
    output_poscar = output_poscar.expanduser().resolve()
    source_incar = source_incar.expanduser().resolve() if source_incar is not None else None
    output_incar = output_incar.expanduser().resolve() if output_incar is not None else None
    match_csv = match_csv.expanduser().resolve() if match_csv is not None else output_poscar.with_name("poscar_projection_map.csv")
    plan_json = plan_json.expanduser().resolve() if plan_json is not None else output_poscar.with_name("poscar_projection_plan.json")
    static_vasp_input_dirs = static_vasp_source_dirs(source_poscar, source_incar)

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

    source_moments: list[float] | None = None
    if source_incar is not None:
        source_moments = (
            [raw_source_moments[index] for index in source_prepared.origin_indices]
            if raw_source_moments is not None
            else None
        )

    removed_anion_targets: set[int] = set()
    anion_vacancy_summary: dict[str, object] = {}
    charge_summary: dict[str, object] = {}
    if preserve_anion_vacancies:
        removed_anion_targets, anion_vacancy_summary, charge_summary = select_target_anion_vacancies(
            source_for_matching,
            target,
            source_symbols,
            target_symbols,
            projected_symbols,
            source_by_target,
            source_moments,
            cation_set,
            anion_set,
            oxidation_states or {},
            magmom_oxidation_states or {},
        )
        if removed_anion_targets:
            warnings.append(
                "Preserved source anion vacancy count by removing "
                + ", ".join(
                    f"{symbol}:{count}"
                    for symbol, count in sorted(Counter(target_symbols[index] for index in removed_anion_targets).items())
                )
                + " target anion site(s)."
            )

    default_order = default_projection_species_order(
        source.species.symbols,
        target.species.symbols,
        [symbol for index, symbol in enumerate(projected_symbols) if index not in removed_anion_targets],
        cation_order,
        cation_set,
        anion_set,
    )
    output_order = complete_species_order(
        species_order or default_order,
        [symbol for index, symbol in enumerate(projected_symbols) if index not in removed_anion_targets],
    )
    output_species, output_indices = grouped_species_and_indices(
        projected_symbols,
        output_order,
        excluded_indices=removed_anion_targets,
    )
    candidate_folder_mode = randomize_candidates > 0 or (randomize_pool_size or 0) > 0
    poscar_text = write_poscar_text(
        f"Projected {source_poscar.name} elements onto {target_poscar.name} structure",
        output_species,
        target.cell,
        [target.scaled_positions[index] for index in output_indices],
    )
    output_poscar.parent.mkdir(parents=True, exist_ok=True)
    output_poscar.write_text(poscar_text, encoding="utf-8")
    copied_static_inputs = [] if candidate_folder_mode else copy_static_vasp_inputs(output_poscar.parent, static_vasp_input_dirs)

    output_moments: list[float] | None = None
    output_moments_grouped: list[float] | None = None
    output_magmom_count: int | None = None
    if source_incar is not None:
        if output_incar is None:
            output_incar = output_poscar.with_name("INCAR")
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
        incar_text = reorder_incar_species_tags(incar_text, source_raw.species, output_species)
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

    direct_candidate_summary = write_direct_projected_candidate_folder(
        output_poscar.parent,
        output_poscar,
        output_incar,
        static_vasp_input_dirs,
        enabled=candidate_folder_mode,
    )
    reported_output_poscar = Path(str(direct_candidate_summary["poscar"])) if direct_candidate_summary.get("enabled") else output_poscar
    reported_output_incar = (
        Path(str(direct_candidate_summary["incar"]))
        if direct_candidate_summary.get("enabled") and direct_candidate_summary.get("incar")
        else output_incar
    )
    cleanup_root_vasp_run_files(
        output_poscar.parent,
        output_poscar,
        output_incar,
        enabled=candidate_folder_mode,
    )
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
    guest_cation_distance_summary = summarize_guest_cation_distances(
        source_for_matching,
        target,
        source_symbols,
        projected_symbols,
        matches,
        cation_set,
        anion_set,
    )
    warnings.extend(guest_cation_distance_warnings(guest_cation_distance_summary))
    randomized_candidate_summary = write_randomized_projection_candidates(
        output_poscar.parent,
        target,
        target_symbols,
        projected_symbols,
        removed_anion_targets,
        output_order,
        output_moments,
        source_incar,
        source_raw.species,
        cation_set,
        anion_set,
        oxidation_states or {},
        magmom_oxidation_states or {},
        n_candidates=randomize_candidates,
        pool_size=randomize_pool_size,
        seed=randomize_seed,
        sublattices=randomize_sublattices,
        vacancy_label=randomize_vacancy_label,
        magmom_decimals=magmom_decimals,
        min_guest_distance_A=randomize_min_guest_distance_A,
        max_guest_vacancy_distance_A=randomize_max_guest_vacancy_distance_A,
        atat_atoms=randomize_atat_atoms,
        atat_job_name=randomize_atat_job_name,
        mcsqs_walltime=randomize_mcsqs_walltime,
        mcsqs_pair_diameter=randomize_mcsqs_pair_diameter,
        mcsqs_triplet_diameter=randomize_mcsqs_triplet_diameter,
        mcsqs_quadruplet_diameter=randomize_mcsqs_quadruplet_diameter,
        mcsqs_temperature=randomize_mcsqs_temperature,
        mcsqs_max_steps=randomize_mcsqs_max_steps,
        static_vasp_input_dirs=static_vasp_input_dirs,
    )
    initial_spin_candidate_summary = write_initial_spin_candidate_mirrors(
        output_poscar.parent,
        cation_set,
        magmom_oxidation_states or {},
        magmom_decimals=magmom_decimals,
        enabled=candidate_folder_mode,
    )
    write_match_csv(match_csv, matches, source_symbols, target_symbols, source_for_matching, target)
    write_plan_json(
        plan_json,
        {
            "schema": "atomi.vasp.poscar_projection.v1",
            "status": "failed_magmom_preservation_check" if strict_magmom_preservation and magmom_warnings else "ok",
            "source_poscar": str(source_poscar),
            "target_poscar": str(target_poscar),
            "source_incar": str(source_incar) if source_incar else "",
            "output_poscar": str(reported_output_poscar),
            "prepared_source_poscar": str(prepared_source_poscar) if prepared_source_poscar else "",
            "output_incar": str(reported_output_incar) if reported_output_incar else "",
            "copied_static_vasp_inputs": [str(path) for path in copied_static_inputs],
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
            "anion_vacancy_summary": anion_vacancy_summary,
            "charge_summary": charge_summary,
            "guest_cation_distance_summary": guest_cation_distance_summary,
            "direct_candidate_summary": direct_candidate_summary,
            "randomized_candidate_summary": randomized_candidate_summary,
            "initial_spin_candidate_summary": initial_spin_candidate_summary,
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
        output_poscar=reported_output_poscar,
        prepared_source_poscar=prepared_source_poscar,
        output_incar=reported_output_incar,
        match_csv=match_csv,
        plan_json=plan_json,
        cation_matches=matches,
        max_cation_distance_A=worst,
        output_magmom_count=output_magmom_count,
        output_species=output_species,
        source_cation_magmom_summary=source_cation_magmom_summary,
        cation_magmom_summary=cation_magmom_summary,
        cation_magmom_comparison=cation_magmom_comparison,
        anion_vacancy_summary=anion_vacancy_summary,
        charge_summary=charge_summary,
        guest_cation_distance_summary=guest_cation_distance_summary,
        direct_candidate_summary=direct_candidate_summary,
        randomized_candidate_summary=randomized_candidate_summary,
        initial_spin_candidate_summary=initial_spin_candidate_summary,
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
    return species_atom_symbols(structure.species)


def species_atom_symbols(species: PoscarSpecies) -> list[str]:
    symbols: list[str] = []
    for symbol, count in zip(species.symbols, species.counts):
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
            source_moments=source_moments,
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
            source_moments=source_moments,
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
    cation_moment_labels = moment_family_labels(symbols, cation_indices, origin_indices, source_moments)
    non_cation_moment_labels = moment_family_labels(symbols, non_cation_indices, origin_indices, source_moments)

    slot_candidates = folded_atom_candidate_slots(
        structure,
        origin_indices,
        cation_indices,
        supercell,
        reduce_cells,
        source_moments,
        moment_labels=cation_moment_labels,
        site_tolerance=site_tolerance,
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
        (
            symbols[index],
            cation_moment_sign(index, origin_indices, source_moments),
            cation_abs_moment_label(
                index,
                origin_indices,
                source_moments,
                cation_moment_labels,
            ),
        )
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
        moment_labels=non_cation_moment_labels,
        site_tolerance=site_tolerance,
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
                cation_abs_moment_label(
                    index,
                    origin_indices,
                    source_moments,
                    non_cation_moment_labels,
                ),
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
        "source_moment_family_counts": stringify_counter(
            Counter(
                (
                    symbols[index],
                    cation_moment_sign(index, origin_indices, source_moments),
                    cation_abs_moment_label(
                        index,
                        origin_indices,
                        source_moments,
                        cation_moment_labels,
                    ),
                )
                for index in cation_indices
            )
        ),
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
    *,
    moment_labels: dict[int, str] | None = None,
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
                abs_moment=cation_abs_moment_label(index, origin_indices, source_moments, moment_labels),
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


def cation_abs_moment_label(
    index: int,
    origin_indices: list[int],
    source_moments: list[float] | None,
    moment_labels: dict[int, str] | None = None,
) -> str:
    if moment_labels and index in moment_labels:
        return moment_labels[index]
    if source_moments is None or origin_indices[index] >= len(source_moments):
        return "unknown"
    return f"{round(abs(source_moments[origin_indices[index]]), 3):g}"


def moment_family_labels(
    symbols: list[str],
    atom_indices: list[int],
    origin_indices: list[int],
    source_moments: list[float] | None,
    *,
    tolerance: float = MOMENT_FAMILY_TOLERANCE,
) -> dict[int, str]:
    """Group noisy cation moment magnitudes into chemically meaningful families.

    Relaxed VASP moments commonly vary by a few hundredths around the same
    nominal charge/spin state.  We therefore cluster |MAGMOM| by element before
    selecting representatives, so U4-like ~2 moments do not appear as many
    separate groups that can crowd out rare U5-like ~1 moments.
    """
    if source_moments is None:
        return {}
    labels: dict[int, str] = {}
    by_symbol: dict[str, list[tuple[int, float]]] = {}
    for index in atom_indices:
        if origin_indices[index] >= len(source_moments):
            continue
        by_symbol.setdefault(symbols[index], []).append((index, abs(source_moments[origin_indices[index]])))

    for group in by_symbol.values():
        clusters: list[tuple[float, list[tuple[int, float]]]] = []
        for index, value in sorted(group, key=lambda item: item[1]):
            if not clusters:
                clusters.append((value, [(index, value)]))
                continue
            nearest_cluster = min(
                enumerate(clusters),
                key=lambda item: abs(value - item[1][0]),
            )
            cluster_index, (center, members) = nearest_cluster
            if abs(value - center) <= tolerance:
                updated_members = [*members, (index, value)]
                updated_center = sum(member_value for _member_index, member_value in updated_members) / len(updated_members)
                clusters[cluster_index] = (updated_center, updated_members)
            else:
                clusters.append((value, [(index, value)]))
        for center, members in clusters:
            label = f"{round(center, 1):g}"
            for index, _value in members:
                labels[index] = label
    return labels


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
    candidate_kept = {
        origin: indices_in_fraction_ranges(structure.scaled_positions, crop_ranges_from_origin(supercell, keep_cells, origin))
        for origin in candidate_origins
    }
    candidate_anion_counts = {
        origin: Counter(symbols[index] for index in kept if symbols[index] in anion_elements)
        for origin, kept in candidate_kept.items()
    }
    max_anion_counts = {
        symbol: max((counts.get(symbol, 0) for counts in candidate_anion_counts.values()), default=0)
        for symbol in anion_elements
    }
    any_anion_deficit = any(
        any(counts.get(symbol, 0) < max_count for symbol, max_count in max_anion_counts.items())
        for counts in candidate_anion_counts.values()
    )
    if not minority_symbols and not any_anion_deficit:
        origin = (0, 0, 0)
        ranges = crop_ranges_from_origin(supercell, keep_cells, origin)
        kept = candidate_kept[origin]
        meta = {
            "selection_policy": "defect_preserving",
            "selection_reason": "regular_origin_crop_no_minority_cation",
            "crop_origin_cells": list(origin),
            "minority_cation_elements": [],
            "minority_cations_available": 0,
            "minority_cations_kept": 0,
            "charge_variant_cations_available": 0,
            "charge_variant_cations_kept": 0,
            "anion_vacancy_elements": [],
            "anion_vacancies_kept": 0,
            "source_magnetic_signature": magnetic_signature(symbols, cation_indices, origin_indices, source_moments),
            "crop_magnetic_signature": magnetic_signature(symbols, sorted(kept & set(cation_indices)), origin_indices, source_moments),
        }
        return ranges, meta

    charge_variant_indices = charge_variant_cation_indices(symbols, cation_indices, origin_indices, source_moments)
    source_magnetic_buckets = magnetic_buckets(symbols, cation_indices, origin_indices, source_moments)

    best_origin = candidate_origins[0]
    best_ranges = crop_ranges_from_origin(supercell, keep_cells, best_origin)
    best_score: tuple[int, int, int, int, int, int, int, int] | None = None
    best_kept: set[int] = set()
    for origin in candidate_origins:
        ranges = crop_ranges_from_origin(supercell, keep_cells, origin)
        kept = candidate_kept[origin]
        kept_cations = sorted(kept & set(cation_indices))
        kept_magnetic_buckets = magnetic_buckets(symbols, kept_cations, origin_indices, source_moments)
        anion_deficit_score = sum(
            max_count - candidate_anion_counts[origin].get(symbol, 0)
            for symbol, max_count in max_anion_counts.items()
        )
        score = (
            sum(1 for index in kept if symbols[index] in minority_symbols),
            sum(1 for index in kept if index in charge_variant_indices),
            len(source_magnetic_buckets & kept_magnetic_buckets),
            anion_deficit_score,
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
        "anion_vacancy_elements": sorted(
            symbol
            for symbol, max_count in max_anion_counts.items()
            if candidate_anion_counts[best_origin].get(symbol, 0) < max_count
        ),
        "anion_vacancies_kept": sum(
            max_count - candidate_anion_counts[best_origin].get(symbol, 0)
            for symbol, max_count in max_anion_counts.items()
        ),
        "candidate_max_anion_counts": dict(sorted(max_anion_counts.items())),
        "crop_anion_counts": dict(sorted(candidate_anion_counts[best_origin].items())),
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
    family_labels = moment_family_labels(symbols, cation_indices, origin_indices, source_moments)
    by_symbol: dict[str, list[tuple[int, str]]] = {}
    for index in cation_indices:
        if origin_indices[index] >= len(source_moments):
            continue
        by_symbol.setdefault(symbols[index], []).append((index, family_labels.get(index, "unknown")))
    variants: set[int] = set()
    for group in by_symbol.values():
        if len(group) < 2:
            continue
        bins = Counter(label for _index, label in group)
        dominant_label, dominant_count = bins.most_common(1)[0]
        if dominant_count == len(group):
            continue
        for index, label in group:
            if label != dominant_label:
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
    source_moments: list[float] | None = None,
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
        cation_counts = Counter(symbols[index] for index in cation_indices)
        max_cation_count = max(cation_counts.values(), default=0)
        minority_symbols = {symbol for symbol, count in cation_counts.items() if count < max_cation_count}
        charge_variant_indices = charge_variant_cation_indices(symbols, cation_indices, origin_indices, source_moments)
        protected_cation_indices = {
            index
            for index in cation_indices
            if symbols[index] in minority_symbols or index in charge_variant_indices
        }
        if len(kept_cations) != expected_cation_count:
            cation_count_targets = balanced_cation_count_targets(symbols, cation_indices, expected_cation_count)
            repaired_cations = repaired_crop_cation_indices(
                structure.scaled_positions,
                symbols,
                cation_indices,
                ranges,
                expected_cation_count,
                cation_count_targets,
                protected_indices=protected_cation_indices,
            )
            kept_indices = (kept_indices - set(cation_indices)) | set(repaired_cations)
            repair_meta = {
                "cation_boundary_repair": True,
                "cation_count_before_repair": len(kept_cations),
                "cation_count_after_repair": len(repaired_cations),
                "expected_cation_count": expected_cation_count,
                "cation_species_target_counts": cation_count_targets,
                "minority_cations_after_repair": sum(
                    1 for index in repaired_cations if symbols[index] in minority_symbols
                ),
                "charge_variant_cations_after_repair": sum(
                    1 for index in repaired_cations if index in charge_variant_indices
                ),
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
                    ((position[axis] - ranges[axis][0]) / (ranges[axis][1] - ranges[axis][0])) % 1.0
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
    *,
    protected_indices: set[int] | None = None,
) -> list[int]:
    if expected_count > len(cation_indices):
        raise ValueError(f"Cannot repair crop to {expected_count} cations; source has only {len(cation_indices)} cations.")
    protected_indices = protected_indices or set()
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
                        0 if index in protected_indices else 1,
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
            0 if index in protected_indices else 1,
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
    ranked_candidates = sorted(
        candidates,
        key=lambda shift: quick_cation_shift_score(
            source.scaled_positions,
            source_indices,
            target.scaled_positions,
            target_indices,
            target.cell,
            shift,
        ),
    )
    evaluated_candidates = ranked_candidates[: min(len(ranked_candidates), 96)]
    for shift in evaluated_candidates:
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
        "evaluated_candidate_count": len(evaluated_candidates),
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
    seen: set[tuple[float, float, float]] = set()
    candidates: list[list[float]] = []
    raw_candidates = [[0.0, 0.0, 0.0]]
    raw_candidates.extend(
        [target_positions[target_index][axis] - source_positions[source_index][axis] for axis in range(3)]
        for source_index in source_indices
        for target_index in target_indices
    )
    for shift in raw_candidates:
        normalized = [value % 1.0 for value in shift]
        key = tuple(round(value, 10) for value in normalized)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(normalized)
    return candidates


def quick_cation_shift_score(
    source_positions: list[list[float]],
    source_indices: list[int],
    target_positions: list[list[float]],
    target_indices: list[int],
    cell: list[list[float]],
    shift: list[float],
) -> tuple[float, float]:
    shifted = shift_fractional_positions(source_positions, shift)
    nearest: list[float] = []
    for target_index in target_indices:
        nearest.append(
            min(
                fractional_distance_A(target_positions[target_index], shifted[source_index], cell)
                for source_index in source_indices
            )
        )
    return (max(nearest, default=0.0), sum(nearest) / len(nearest) if nearest else 0.0)


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
    if len(source_indices) < len(target_indices):
        raise RuntimeError("Not enough source sites to assign every target site.")
    distances: dict[tuple[int, int], float] = {}
    unique_distances: list[float] = []
    for target_index in target_indices:
        for source_index in source_indices:
            distance = fractional_distance_A(target_positions[target_index], source_positions[source_index], cell)
            distances[(target_index, source_index)] = distance
            unique_distances.append(distance)
    sorted_distances = sorted(set(unique_distances))
    low = 0
    high = len(sorted_distances) - 1
    best_match: dict[int, int] | None = None
    while low <= high:
        mid = (low + high) // 2
        matched = threshold_site_assignment(source_indices, target_indices, distances, sorted_distances[mid])
        if len(matched) == len(target_indices):
            best_match = matched
            high = mid - 1
        else:
            low = mid + 1
    if best_match is not None:
        return [
            SiteMatch(
                target_index=target_index,
                source_index=source_index,
                distance_A=distances[(target_index, source_index)],
            )
            for target_index, source_index in sorted(best_match.items())
        ]
    raise RuntimeError("Could not build a complete source-target site assignment.")


def threshold_site_assignment(
    source_indices: list[int],
    target_indices: list[int],
    distances: dict[tuple[int, int], float],
    threshold: float,
) -> dict[int, int]:
    source_match: dict[int, int] = {}

    def assign(target_index: int, seen: set[int]) -> bool:
        candidates = sorted(
            source_index
            for source_index in source_indices
            if distances[(target_index, source_index)] <= threshold + 1.0e-12
        )
        for source_index in candidates:
            if source_index in seen:
                continue
            seen.add(source_index)
            if source_index not in source_match or assign(source_match[source_index], seen):
                source_match[source_index] = target_index
                return True
        return False

    for target_index in sorted(target_indices):
        assign(target_index, set())
    target_match = {target_index: source_index for source_index, target_index in source_match.items()}
    return target_match


def fractional_distance_A(left: list[float], right: list[float], cell: list[list[float]]) -> float:
    diff = [left[i] - right[i] for i in range(3)]
    diff = [value - round(value) for value in diff]
    cart = [
        diff[0] * cell[0][j] + diff[1] * cell[1][j] + diff[2] * cell[2][j]
        for j in range(3)
    ]
    return math.sqrt(sum(value * value for value in cart))


def pairwise_distances_A(structure: PoscarStructure, indices: list[int]) -> list[float]:
    distances: list[float] = []
    for left_offset, left_index in enumerate(indices):
        for right_index in indices[left_offset + 1 :]:
            distances.append(
                fractional_distance_A(
                    structure.scaled_positions[left_index],
                    structure.scaled_positions[right_index],
                    structure.cell,
                )
            )
    return sorted(distances)


def summarize_guest_cation_distances(
    source: PoscarStructure,
    target: PoscarStructure,
    source_symbols: list[str],
    projected_symbols: list[str],
    matches: list[SiteMatch],
    cation_elements: set[str],
    anion_elements: set[str],
) -> dict[str, object]:
    cation_indices = selected_cation_indices(projected_symbols, cation_elements, anion_elements)
    cation_counts = Counter(projected_symbols[index] for index in cation_indices)
    max_count = max(cation_counts.values(), default=0)
    guest_symbols = sorted(
        symbol
        for symbol, count in cation_counts.items()
        if count < max_count and count >= 2
    )
    summary: dict[str, object] = {
        "enabled": True,
        "periodic_boundary_conditions": True,
        "guest_definition": (
            "Cation species with lower count than the most abundant projected cation "
            "and at least two projected atoms."
        ),
        "cation_counts": dict(sorted(cation_counts.items())),
        "guest_symbols": guest_symbols,
        "symbols": {},
    }
    if not guest_symbols:
        summary["note"] = "No minority cation species has at least two atoms, so no guest-guest pair distance was checked."
        return summary

    by_symbol: dict[str, list[tuple[int, int]]] = {symbol: [] for symbol in guest_symbols}
    for match in matches:
        source_symbol = source_symbols[match.source_index]
        target_symbol = projected_symbols[match.target_index]
        if source_symbol == target_symbol and source_symbol in by_symbol:
            by_symbol[source_symbol].append((match.source_index, match.target_index))

    symbol_summaries: dict[str, object] = {}
    for symbol in guest_symbols:
        pairs = sorted(by_symbol.get(symbol, []), key=lambda item: (item[0], item[1]))
        if len(pairs) < 2:
            symbol_summaries[symbol] = {
                "source_atom_indices_1based": [source_index + 1 for source_index, _target_index in pairs],
                "output_atom_indices_1based": [target_index + 1 for _source_index, target_index in pairs],
                "pair_count": 0,
                "note": "Fewer than two matched guest cations were available for pair-distance analysis.",
            }
            continue
        source_indices = [source_index for source_index, _target_index in pairs]
        target_indices = [target_index for _source_index, target_index in pairs]
        source_distances = pairwise_distances_A(source, source_indices)
        output_distances = pairwise_distances_A(target, target_indices)
        deltas = [
            output_distance - source_distance
            for source_distance, output_distance in zip(source_distances, output_distances)
        ]
        source_min = min(source_distances)
        output_min = min(output_distances)
        source_mean = sum(source_distances) / len(source_distances)
        output_mean = sum(output_distances) / len(output_distances)
        min_delta = output_min - source_min
        mean_delta = output_mean - source_mean
        nearest_ratio = output_min / source_min if source_min > 1.0e-12 else None
        compressed = (
            source_min > 1.0e-12
            and output_min < source_min - 0.5
            and output_min / source_min < 0.75
        )
        symbol_summaries[symbol] = {
            "source_atom_indices_1based": [index + 1 for index in source_indices],
            "output_atom_indices_1based": [index + 1 for index in target_indices],
            "pair_count": len(source_distances),
            "source_distances_A": [round(value, 6) for value in source_distances],
            "output_distances_A": [round(value, 6) for value in output_distances],
            "source_min_distance_A": round(source_min, 6),
            "output_min_distance_A": round(output_min, 6),
            "source_mean_distance_A": round(source_mean, 6),
            "output_mean_distance_A": round(output_mean, 6),
            "min_distance_delta_A": round(min_delta, 6),
            "mean_distance_delta_A": round(mean_delta, 6),
            "nearest_distance_ratio": round(nearest_ratio, 6) if nearest_ratio is not None else None,
            "max_pair_distance_delta_A": round(max((abs(delta) for delta in deltas), default=0.0), 6),
            "nearest_distance_preserved": not compressed,
            "note": (
                "Distances are minimum-image guest-guest separations under periodic boundary conditions. "
                "The output values are measured in the projected POSCAR cell."
            ),
        }
    summary["symbols"] = symbol_summaries
    return summary


def guest_cation_distance_warnings(summary: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    symbols = summary.get("symbols", {})
    if not isinstance(symbols, dict):
        return warnings
    for symbol, raw_entry in symbols.items():
        if not isinstance(raw_entry, dict):
            continue
        if raw_entry.get("nearest_distance_preserved") is False:
            warnings.append(
                f"{symbol}-{symbol} nearest periodic distance compressed from "
                f"{float(raw_entry.get('source_min_distance_A', 0.0)):g} A to "
                f"{float(raw_entry.get('output_min_distance_A', 0.0)):g} A"
            )
    return warnings


def write_randomized_projection_candidates(
    outdir: Path,
    target: PoscarStructure,
    target_symbols: list[str],
    projected_symbols: list[str],
    removed_anion_targets: set[int],
    output_order: list[str],
    output_moments: list[float] | None,
    source_incar: Path | None,
    source_species: PoscarSpecies,
    cation_elements: set[str],
    anion_elements: set[str],
    oxidation_states: dict[str, float],
    magmom_oxidation_states: dict[str, list[tuple[float, float]]],
    *,
    n_candidates: int,
    pool_size: int | None,
    seed: int,
    sublattices: set[str] | None,
    vacancy_label: str,
    magmom_decimals: int,
    min_guest_distance_A: float | None = None,
    max_guest_vacancy_distance_A: float | None = None,
    atat_atoms: int | None = None,
    atat_job_name: str = "atat-random",
    mcsqs_walltime: str = "04:00:00",
    mcsqs_pair_diameter: float = 6.0,
    mcsqs_triplet_diameter: float | None = None,
    mcsqs_quadruplet_diameter: float | None = None,
    mcsqs_temperature: float | None = None,
    mcsqs_max_steps: int | None = None,
    static_vasp_input_dirs: list[Path] | None = None,
) -> dict[str, object]:
    if n_candidates <= 0 and (pool_size or 0) > 0:
        n_candidates = 3
    if n_candidates <= 0:
        return {"enabled": False, "candidate_count": 0}
    pool_count = max(n_candidates, pool_size or n_candidates)
    active_sublattices = sublattices or {"cation", "anion"}
    invalid = active_sublattices - {"cation", "anion"}
    if invalid:
        raise ValueError(f"Unknown randomized sublattice(s): {', '.join(sorted(invalid))}. Use cation and/or anion.")

    rng = random.Random(seed)
    candidates_dir = outdir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    cation_sites = selected_cation_indices(projected_symbols, cation_elements, anion_elements)
    anion_sites = [index for index, symbol in enumerate(target_symbols) if symbol in anion_elements]
    cation_decorations = [
        (projected_symbols[index], output_moments[index] if output_moments is not None else None)
        for index in cation_sites
    ]
    anion_decorations = [
        (None, None) if index in removed_anion_targets else (projected_symbols[index], 0.0)
        for index in anion_sites
    ]
    base_incar_text = source_incar.read_text(encoding="utf-8", errors="replace") if source_incar is not None else None
    base_vacancy_guest_summary = summarize_vacancy_guest_distances(
        removed_anion_targets,
        target,
        target_symbols,
        projected_symbols,
        cation_elements,
        anion_elements,
    )
    base_vacancy_guest_min = base_vacancy_guest_summary.get("min_distance_A")
    reference_vacancy_guest_min_A = float(base_vacancy_guest_min) if base_vacancy_guest_min is not None else None
    pool_rows: list[dict[str, object]] = []
    for pool_index in range(1, pool_count + 1):
        candidate_symbols = list(projected_symbols)
        candidate_moments = list(output_moments) if output_moments is not None else None
        candidate_removed = set(removed_anion_targets)

        if "cation" in active_sublattices:
            shuffled_cations = list(cation_decorations)
            rng.shuffle(shuffled_cations)
            for target_index, (symbol, moment) in zip(cation_sites, shuffled_cations):
                candidate_symbols[target_index] = symbol or target_symbols[target_index]
                if candidate_moments is not None and moment is not None:
                    candidate_moments[target_index] = moment

        if "anion" in active_sublattices and anion_sites:
            shuffled_anions = list(anion_decorations)
            rng.shuffle(shuffled_anions)
            candidate_removed = set()
            for target_index, (symbol, moment) in zip(anion_sites, shuffled_anions):
                if symbol is None:
                    candidate_removed.add(target_index)
                    candidate_symbols[target_index] = target_symbols[target_index]
                    if candidate_moments is not None:
                        candidate_moments[target_index] = 0.0
                else:
                    candidate_symbols[target_index] = symbol
                    if candidate_moments is not None and moment is not None:
                        candidate_moments[target_index] = moment

        candidate_species, candidate_indices = grouped_species_and_indices(
            candidate_symbols,
            output_order,
            excluded_indices=candidate_removed,
        )
        charge = charge_summary_from_symbols_and_moments(
            candidate_symbols,
            candidate_moments,
            cation_elements,
            anion_elements,
            oxidation_states,
            magmom_oxidation_states,
            removed_anion_targets=candidate_removed,
        )
        guest_distances = summarize_guest_cation_distances_in_structure(
            target,
            candidate_symbols,
            cation_elements,
            anion_elements,
            removed_indices=candidate_removed,
        )
        vacancy_locality = removed_anion_locality(
            candidate_removed,
            target,
            target_symbols,
            candidate_symbols,
            cation_elements,
        )
        vacancy_guest = summarize_vacancy_guest_distances(
            candidate_removed,
            target,
            target_symbols,
            candidate_symbols,
            cation_elements,
            anion_elements,
        )
        rank_summary = rank_randomized_candidate(
            charge,
            guest_distances,
            vacancy_guest,
            min_guest_distance_A=min_guest_distance_A,
            max_guest_vacancy_distance_A=max_guest_vacancy_distance_A,
            reference_vacancy_guest_min_A=reference_vacancy_guest_min_A,
        )
        species_counts = dict(zip(candidate_species.symbols, candidate_species.counts))
        pool_rows.append(
            {
                "pool_index": pool_index,
                "candidate_id": "",
                "run_dir": "",
                "poscar": "",
                "incar": "",
                "species_counts": species_counts,
                "removed_anion_indices_1based": [index + 1 for index in sorted(candidate_removed)],
                "charge_summary": charge,
                "guest_cation_distance_summary": guest_distances,
                "vacancy_guest_distance_summary": vacancy_guest,
                "removed_anion_nearest_cations": vacancy_locality,
                "stability_rank": rank_summary,
                "_candidate_symbols": candidate_symbols,
                "_candidate_moments": candidate_moments,
                "_candidate_removed": candidate_removed,
                "_candidate_species": candidate_species,
                "_candidate_indices": candidate_indices,
            }
        )

    ranked_rows = sorted(
        pool_rows,
        key=lambda row: (
            float(cast_dict(row.get("stability_rank")).get("score", 0.0)),
            float(cast_dict(row.get("guest_cation_distance_summary")).get("global_min_distance_A", 0.0) or 0.0),
        ),
        reverse=True,
    )
    selected_rows = [
        row
        for row in ranked_rows
        if cast_dict(row.get("stability_rank")).get("status") != "fail"
    ][:n_candidates]
    rows: list[dict[str, object]] = []
    runlist: list[str] = []
    for candidate_number, row in enumerate(selected_rows, start=1):
        candidate_id = f"random_{candidate_number:03d}"
        run_dir = candidates_dir / candidate_id
        run_dir.mkdir(parents=True, exist_ok=True)
        candidate_symbols = list(row.pop("_candidate_symbols"))
        candidate_moments_raw = row.pop("_candidate_moments")
        candidate_moments = list(candidate_moments_raw) if candidate_moments_raw is not None else None
        candidate_removed = set(row.pop("_candidate_removed"))
        candidate_species = row.pop("_candidate_species")
        candidate_indices = list(row.pop("_candidate_indices"))
        poscar_path = run_dir / "POSCAR"
        poscar_path.write_text(
            write_poscar_text(
                f"Randomized projected candidate {candidate_id}",
                candidate_species,
                target.cell,
                [target.scaled_positions[index] for index in candidate_indices],
            ),
            encoding="utf-8",
        )
        copied_static_inputs = copy_static_vasp_inputs(run_dir, static_vasp_input_dirs or [])

        incar_path = ""
        if base_incar_text is not None and candidate_moments is not None:
            grouped_moments = [candidate_moments[index] for index in candidate_indices]
            selected = species_with_nonzero_moments(candidate_species, grouped_moments)
            magmom_line = format_magmom_line(
                candidate_species,
                grouped_moments,
                selected_elements=selected,
                decimals=magmom_decimals,
                compact_zero=True,
            )
            validate_magmom_line_count(
                magmom_line,
                candidate_species.total_atoms,
                context=f"randomized projected INCAR for {poscar_path}",
            )
            incar_path = str((run_dir / "INCAR").resolve())
            candidate_incar_text = reorder_incar_species_tags(base_incar_text, source_species, candidate_species)
            (run_dir / "INCAR").write_text(replace_or_append_magmom_text(candidate_incar_text, magmom_line), encoding="utf-8")

        row["candidate_id"] = candidate_id
        row["run_dir"] = str(run_dir.resolve())
        row["poscar"] = str(poscar_path.resolve())
        row["incar"] = incar_path
        row["copied_static_vasp_inputs"] = [str(path) for path in copied_static_inputs]
        row["rank"] = candidate_number
        rows.append(row)
        runlist.append(str(run_dir.resolve()))

    pool_index = strip_internal_candidate_fields(pool_rows)
    write_randomized_candidate_index(outdir / "randomized_candidate_index.csv", rows)
    write_randomized_candidate_index(outdir / "randomized_pool_rankings.csv", pool_index)
    (outdir / "randomized_runlist.txt").write_text("\n".join(runlist) + "\n", encoding="utf-8")
    atat_dir = outdir / "atat_random"
    write_projection_atat_rndstr(
        atat_dir / "rndstr.in",
        target,
        target_symbols,
        cation_sites,
        anion_sites,
        cation_decorations,
        anion_decorations,
        vacancy_label,
        active_sublattices,
        anion_elements,
    )
    script_paths = write_projection_atat_mcsqs_scripts(
        atat_dir,
        atat_atoms=atat_atoms or len(target_symbols),
        job_name=atat_job_name,
        walltime=mcsqs_walltime,
        pair_diameter=mcsqs_pair_diameter,
        triplet_diameter=mcsqs_triplet_diameter,
        quadruplet_diameter=mcsqs_quadruplet_diameter,
        temperature=mcsqs_temperature,
        max_steps=mcsqs_max_steps,
        vacancy_label=vacancy_label,
    )
    return {
        "enabled": True,
        "mode": "randomized_projected_sublattices",
        "requested_candidate_count": n_candidates,
        "candidate_count": len(rows),
        "pool_size": pool_count,
        "selected_count": len(rows),
        "seed": seed,
        "sublattices": sorted(active_sublattices),
        "candidates_dir": str(candidates_dir),
        "candidate_index": str(outdir / "randomized_candidate_index.csv"),
        "pool_index": str(outdir / "randomized_pool_rankings.csv"),
        "runlist": str(outdir / "randomized_runlist.txt"),
        "atat_rndstr": str(atat_dir / "rndstr.in"),
        "atat_pseudo_species_map": str(atat_dir / "pseudo_species_map.csv"),
        "atat_run_script": str(script_paths["run_script"]),
        "atat_submit_script": str(script_paths["submit_script"]),
        "atat_readme": str(script_paths["readme"]),
        "notes": [
            "Direct candidates preserve the projected B cell/coordinates but shuffle occupation labels across selected sublattices.",
            "Cation shuffling moves element and MAGMOM decorations together, so valence/spin counts are preserved.",
            "Anion shuffling moves anion/vacancy decorations together; vacancy pseudo-atoms are not written to VASP POSCARs.",
            "When --randomize-pool-size is larger than --randomize-candidates, Atomi scores the pool and writes only the top ranked structures.",
            "Ranking is a heuristic screen, not a substitute for VASP relaxation: neutral structures are required when oxidation data is present, guest-guest separation is rewarded, and guest-vacancy locality is kept near the projected reference when possible.",
            "rndstr.in is an ATAT/SQS handoff with pseudo-species labels for valence/spin/vacancy states.",
        ],
        "candidates": rows,
    }


def write_randomized_candidate_index(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "rank",
        "candidate_id",
        "pool_index",
        "run_dir",
        "poscar",
        "incar",
        "copied_static_vasp_inputs",
        "stability_score",
        "stability_status",
        "stability_notes",
        "species_counts_json",
        "total_charge",
        "neutrality_ok",
        "guest_min_distances_A_json",
        "guest_global_min_distance_A",
        "vacancy_guest_min_distance_A",
        "removed_anion_indices_1based",
        "nearest_removed_anion_cations_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            charge = row.get("charge_summary", {})
            if not isinstance(charge, dict):
                charge = {}
            guest = row.get("guest_cation_distance_summary", {})
            guest_min: dict[str, float] = {}
            if isinstance(guest, dict):
                symbols = guest.get("symbols", {})
                if isinstance(symbols, dict):
                    for symbol, entry in symbols.items():
                        if isinstance(entry, dict) and entry.get("min_distance_A") is not None:
                            guest_min[str(symbol)] = float(entry["min_distance_A"])
            else:
                guest = {}
            vacancy_guest = row.get("vacancy_guest_distance_summary", {})
            if not isinstance(vacancy_guest, dict):
                vacancy_guest = {}
            stability = row.get("stability_rank", {})
            if not isinstance(stability, dict):
                stability = {}
            writer.writerow(
                {
                    "rank": row.get("rank", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "pool_index": row.get("pool_index", ""),
                    "run_dir": row.get("run_dir", ""),
                    "poscar": row.get("poscar", ""),
                    "incar": row.get("incar", ""),
                    "copied_static_vasp_inputs": " ".join(str(item) for item in row.get("copied_static_vasp_inputs", [])),
                    "stability_score": stability.get("score", ""),
                    "stability_status": stability.get("status", ""),
                    "stability_notes": "; ".join(str(item) for item in stability.get("notes", [])),
                    "species_counts_json": json.dumps(row.get("species_counts", {}), sort_keys=True),
                    "total_charge": charge.get("total_charge", ""),
                    "neutrality_ok": charge.get("neutrality_ok", ""),
                    "guest_min_distances_A_json": json.dumps(guest_min, sort_keys=True),
                    "guest_global_min_distance_A": guest.get("global_min_distance_A", ""),
                    "vacancy_guest_min_distance_A": vacancy_guest.get("min_distance_A", ""),
                    "removed_anion_indices_1based": " ".join(str(item) for item in row.get("removed_anion_indices_1based", [])),
                    "nearest_removed_anion_cations_json": json.dumps(row.get("removed_anion_nearest_cations", []), sort_keys=True),
                }
            )


def charge_summary_from_symbols_and_moments(
    symbols: list[str],
    moments: list[float] | None,
    cation_elements: set[str],
    anion_elements: set[str],
    oxidation_states: dict[str, float],
    magmom_oxidation_states: dict[str, list[tuple[float, float]]],
    *,
    removed_anion_targets: set[int],
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    charge_by_symbol: Counter[str] = Counter()
    missing: set[str] = set()
    cation_charge = 0.0
    anion_charge = 0.0
    for index, symbol in enumerate(symbols):
        if index in removed_anion_targets:
            continue
        counts[symbol] += 1
        moment = moments[index] if moments is not None and index < len(moments) else None
        oxidation = oxidation_for_symbol(symbol, moment, oxidation_states, magmom_oxidation_states)
        if oxidation is None:
            if symbol in cation_elements or symbol in anion_elements:
                missing.add(symbol)
            continue
        charge_by_symbol[symbol] += oxidation
        if symbol in anion_elements:
            anion_charge += oxidation
        else:
            cation_charge += oxidation
    total = cation_charge + anion_charge
    return {
        "enabled": bool(oxidation_states or magmom_oxidation_states),
        "species_counts": dict(sorted(counts.items())),
        "charge_by_symbol": {symbol: float(charge) for symbol, charge in sorted(charge_by_symbol.items())},
        "cation_charge": float(cation_charge),
        "anion_charge": float(anion_charge),
        "total_charge": float(total),
        "neutrality_ok": abs(total) < 1.0e-6 if oxidation_states or magmom_oxidation_states else None,
        "missing_oxidation_elements": sorted(missing),
    }


def summarize_guest_cation_distances_in_structure(
    structure: PoscarStructure,
    symbols: list[str],
    cation_elements: set[str],
    anion_elements: set[str],
    *,
    removed_indices: set[int],
) -> dict[str, object]:
    cation_indices = [
        index
        for index in selected_cation_indices(symbols, cation_elements, anion_elements)
        if index not in removed_indices
    ]
    cation_counts = Counter(symbols[index] for index in cation_indices)
    max_count = max(cation_counts.values(), default=0)
    guest_symbols = sorted(symbol for symbol, count in cation_counts.items() if count < max_count and count >= 2)
    symbol_summaries: dict[str, object] = {}
    global_distances: list[float] = []
    for symbol in guest_symbols:
        indices = [index for index in cation_indices if symbols[index] == symbol]
        distances = pairwise_distances_A(structure, indices)
        if not distances:
            continue
        global_distances.extend(distances)
        symbol_summaries[symbol] = {
            "atom_indices_1based": [index + 1 for index in indices],
            "pair_count": len(distances),
            "distances_A": [round(value, 6) for value in distances],
            "min_distance_A": round(min(distances), 6),
            "mean_distance_A": round(sum(distances) / len(distances), 6),
        }
    return {
        "enabled": True,
        "periodic_boundary_conditions": True,
        "cation_counts": dict(sorted(cation_counts.items())),
        "guest_symbols": guest_symbols,
        "global_min_distance_A": round(min(global_distances), 6) if global_distances else None,
        "global_mean_distance_A": round(sum(global_distances) / len(global_distances), 6) if global_distances else None,
        "symbols": symbol_summaries,
    }


def summarize_vacancy_guest_distances(
    removed_anion_targets: set[int],
    structure: PoscarStructure,
    target_symbols: list[str],
    candidate_symbols: list[str],
    cation_elements: set[str],
    anion_elements: set[str],
) -> dict[str, object]:
    cation_indices = selected_cation_indices(candidate_symbols, cation_elements, anion_elements)
    cation_counts = Counter(candidate_symbols[index] for index in cation_indices)
    max_count = max(cation_counts.values(), default=0)
    guest_symbols = sorted(symbol for symbol, count in cation_counts.items() if count < max_count)
    guest_indices = [index for index in cation_indices if candidate_symbols[index] in guest_symbols]
    entries: list[dict[str, object]] = []
    distances: list[float] = []
    for anion_index in sorted(removed_anion_targets):
        nearest: tuple[float, int] | None = None
        for guest_index in guest_indices:
            distance = fractional_distance_A(
                structure.scaled_positions[anion_index],
                structure.scaled_positions[guest_index],
                structure.cell,
            )
            if nearest is None or distance < nearest[0]:
                nearest = (distance, guest_index)
        if nearest is None:
            entries.append(
                {
                    "removed_target_atom": anion_index + 1,
                    "removed_element": target_symbols[anion_index],
                    "nearest_guest_atom": None,
                    "nearest_guest_element": None,
                    "nearest_guest_distance_A": None,
                }
            )
            continue
        distances.append(nearest[0])
        entries.append(
            {
                "removed_target_atom": anion_index + 1,
                "removed_element": target_symbols[anion_index],
                "nearest_guest_atom": nearest[1] + 1,
                "nearest_guest_element": candidate_symbols[nearest[1]],
                "nearest_guest_distance_A": round(nearest[0], 6),
            }
        )
    return {
        "enabled": bool(removed_anion_targets and guest_indices),
        "guest_symbols": guest_symbols,
        "removed_anion_count": len(removed_anion_targets),
        "min_distance_A": round(min(distances), 6) if distances else None,
        "mean_distance_A": round(sum(distances) / len(distances), 6) if distances else None,
        "entries": entries,
    }


def rank_randomized_candidate(
    charge_summary: dict[str, object],
    guest_distances: dict[str, object],
    vacancy_guest_summary: dict[str, object],
    *,
    min_guest_distance_A: float | None,
    max_guest_vacancy_distance_A: float | None,
    reference_vacancy_guest_min_A: float | None,
) -> dict[str, object]:
    score = 100.0
    status = "ok"
    notes: list[str] = []
    neutrality = charge_summary.get("neutrality_ok")
    if neutrality is False:
        score -= 1000.0
        status = "fail"
        notes.append("non-neutral charge state")
    elif neutrality is True:
        score += 100.0
        notes.append("neutral charge")
    else:
        notes.append("charge neutrality not evaluated")

    guest_min_raw = guest_distances.get("global_min_distance_A")
    guest_min = float(guest_min_raw) if guest_min_raw is not None else None
    if guest_min is not None:
        score += 10.0 * guest_min
        notes.append(f"guest-guest min distance {guest_min:.3f} A")
        if min_guest_distance_A is not None and guest_min < min_guest_distance_A:
            score -= 500.0
            status = "fail"
            notes.append(f"guest-guest distance below {min_guest_distance_A:.3f} A")

    vacancy_guest_raw = vacancy_guest_summary.get("min_distance_A")
    vacancy_guest = float(vacancy_guest_raw) if vacancy_guest_raw is not None else None
    if vacancy_guest is not None:
        if reference_vacancy_guest_min_A is not None:
            delta = abs(vacancy_guest - reference_vacancy_guest_min_A)
            score += max(0.0, 30.0 - 10.0 * delta)
            notes.append(
                f"guest-vacancy min distance {vacancy_guest:.3f} A; "
                f"reference {reference_vacancy_guest_min_A:.3f} A"
            )
        else:
            score += max(0.0, 30.0 - vacancy_guest)
            notes.append(f"guest-vacancy min distance {vacancy_guest:.3f} A")
        if max_guest_vacancy_distance_A is not None and vacancy_guest > max_guest_vacancy_distance_A:
            score -= 500.0
            status = "fail"
            notes.append(f"guest-vacancy distance above {max_guest_vacancy_distance_A:.3f} A")

    return {
        "score": round(score, 6),
        "status": status,
        "notes": notes,
        "heuristic": (
            "Neutrality is required when oxidation data is provided; larger guest-guest separation is rewarded; "
            "guest-vacancy distance is rewarded when it stays near the projected reference locality."
        ),
    }


def cast_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def static_vasp_source_dirs(source_poscar: Path, source_incar: Path | None) -> list[Path]:
    dirs: list[Path] = []
    for path in (source_incar, source_poscar):
        if path is None:
            continue
        directory = path.expanduser().resolve().parent
        if directory not in dirs:
            dirs.append(directory)
    return dirs


def copy_static_vasp_inputs(destination_dir: Path, source_dirs: list[Path]) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in ("KPOINTS", "POTCAR"):
        for source_dir in source_dirs:
            source = source_dir / name
            if source.is_file():
                target = destination_dir / name
                shutil.copy2(source, target)
                copied.append(target.resolve())
                break
    return copied


def cleanup_root_vasp_run_files(
    outdir: Path,
    output_poscar: Path,
    output_incar: Path | None,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    candidates = [output_poscar, output_incar, outdir / "KPOINTS", outdir / "POTCAR"]
    for path in candidates:
        if path is None:
            continue
        if path.parent == outdir and path.name in {"POSCAR", "INCAR", "KPOINTS", "POTCAR"}:
            path.unlink(missing_ok=True)


def write_direct_projected_candidate_folder(
    outdir: Path,
    output_poscar: Path,
    output_incar: Path | None,
    source_dirs: list[Path],
    *,
    enabled: bool,
) -> dict[str, object]:
    if not enabled:
        return {"enabled": False}
    run_dir = outdir / "candidates" / "direct_projected"
    run_dir.mkdir(parents=True, exist_ok=True)
    poscar_target = run_dir / "POSCAR"
    shutil.copy2(output_poscar, poscar_target)
    incar_target: Path | None = None
    if output_incar is not None and output_incar.is_file():
        incar_target = run_dir / "INCAR"
        shutil.copy2(output_incar, incar_target)
    copied_static = copy_static_vasp_inputs(run_dir, source_dirs)
    return {
        "enabled": True,
        "candidate_id": "direct_projected",
        "run_dir": str(run_dir.resolve()),
        "poscar": str(poscar_target.resolve()),
        "incar": "" if incar_target is None else str(incar_target.resolve()),
        "copied_static_vasp_inputs": [str(path) for path in copied_static],
        "notes": "Direct projected POSCAR mirrored into a candidate folder next to randomized candidates.",
    }


def write_initial_spin_candidate_mirrors(
    outdir: Path,
    cation_elements: set[str],
    magmom_oxidation_states: dict[str, list[tuple[float, float]]],
    *,
    magmom_decimals: int,
    enabled: bool,
) -> dict[str, object]:
    source_root = outdir / "candidates"
    if not enabled or not source_root.is_dir():
        return {"enabled": False, "candidate_count": 0}
    multi_valence_elements = sorted(
        symbol
        for symbol in cation_elements
        if len({round(moment, 6) for moment, _charge in magmom_oxidation_states.get(symbol, [])}) > 1
    )
    if not multi_valence_elements:
        return {"enabled": False, "candidate_count": 0, "reason": "No cation element has multiple MAGMOM oxidation families."}

    mirror_root = outdir / "candidates_i"
    rows: list[dict[str, object]] = []
    runlist: list[str] = []
    for run_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        source_poscar = run_dir / "POSCAR"
        source_incar = run_dir / "INCAR"
        if not source_poscar.is_file() or not source_incar.is_file():
            continue
        structure = read_poscar_structure(source_poscar)
        moments = existing_magmom_values(source_incar, structure.species.total_atoms)
        if moments is None:
            continue
        adjusted_moments, change_summary = initial_spin_reference_moments(
            structure.species,
            moments,
            cation_elements,
            magmom_oxidation_states,
        )
        if int(change_summary.get("changed_count", 0)) <= 0:
            continue
        mirror_dir = mirror_root / run_dir.name
        mirror_dir.mkdir(parents=True, exist_ok=True)
        copied_files = copy_named_files(run_dir, mirror_dir, ("POSCAR", "KPOINTS", "POTCAR"))
        incar_text = source_incar.read_text(encoding="utf-8", errors="replace")
        selected = species_with_nonzero_moments(structure.species, adjusted_moments)
        magmom_line = format_magmom_line(
            structure.species,
            adjusted_moments,
            selected_elements=selected,
            decimals=magmom_decimals,
            compact_zero=True,
        )
        validate_magmom_line_count(
            magmom_line,
            structure.species.total_atoms,
            context=f"initial-spin INCAR mirror for {mirror_dir / 'INCAR'}",
        )
        (mirror_dir / "INCAR").write_text(replace_or_append_magmom_text(incar_text, magmom_line), encoding="utf-8")
        row = {
            "candidate_id": run_dir.name,
            "source_run_dir": str(run_dir.resolve()),
            "run_dir": str(mirror_dir.resolve()),
            "poscar": str((mirror_dir / "POSCAR").resolve()),
            "incar": str((mirror_dir / "INCAR").resolve()),
            "copied_files": [str(path) for path in copied_files],
            "changed_count": change_summary["changed_count"],
            "changed_by_element": change_summary["changed_by_element"],
            "majority_families": change_summary["majority_families"],
            "changed_atoms_1based": change_summary["changed_atoms_1based"],
        }
        rows.append(row)
        runlist.append(str(mirror_dir.resolve()))
    if not rows:
        return {
            "enabled": False,
            "candidate_count": 0,
            "reason": "No minority MAGMOM oxidation families were present in candidate INCAR files.",
            "multi_valence_elements": multi_valence_elements,
        }
    index_path = outdir / "candidates_i_index.csv"
    runlist_path = outdir / "candidates_i_runlist.txt"
    write_initial_spin_candidate_index(index_path, rows)
    runlist_path.write_text("\n".join(runlist) + "\n", encoding="utf-8")
    return {
        "enabled": True,
        "mode": "majority_host_initial_spin_mirror",
        "candidate_count": len(rows),
        "multi_valence_elements": multi_valence_elements,
        "candidates_dir": str(mirror_root.resolve()),
        "candidate_index": str(index_path.resolve()),
        "runlist": str(runlist_path.resolve()),
        "notes": [
            "POSCAR/KPOINTS/POTCAR mirror candidates/* exactly.",
            "INCAR MAGMOM values for minority cation valence families are shifted to the majority host-spin family while preserving sign and local moment offset.",
        ],
        "candidates": rows,
    }


def copy_named_files(source_dir: Path, destination_dir: Path, names: tuple[str, ...]) -> list[Path]:
    copied: list[Path] = []
    for name in names:
        source = source_dir / name
        if source.is_file():
            target = destination_dir / name
            shutil.copy2(source, target)
            copied.append(target.resolve())
    return copied


def initial_spin_reference_moments(
    species: PoscarSpecies,
    moments: list[float],
    cation_elements: set[str],
    magmom_oxidation_states: dict[str, list[tuple[float, float]]],
) -> tuple[list[float], dict[str, object]]:
    adjusted = list(moments)
    symbols = species_atom_symbols(species)
    changed_by_element: Counter[str] = Counter()
    majority_families: dict[str, dict[str, object]] = {}
    changed_atoms: list[int] = []
    for symbol in sorted(cation_elements):
        family_targets = sorted({float(moment) for moment, _charge in magmom_oxidation_states.get(symbol, [])})
        if len(family_targets) < 2:
            continue
        classified: list[tuple[int, float, float]] = []
        for index, (site_symbol, moment) in enumerate(zip(symbols, moments)):
            if site_symbol != symbol:
                continue
            family = nearest_moment_family(abs(moment), family_targets)
            if family is None:
                continue
            classified.append((index, moment, family))
        family_counts = Counter(family for _index, _moment, family in classified)
        if len(family_counts) < 2:
            continue
        most_common = family_counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            continue
        majority_family = float(most_common[0][0])
        majority_families[symbol] = {
            "majority_abs_moment": majority_family,
            "family_counts": {f"{family:g}": int(count) for family, count in sorted(family_counts.items())},
        }
        for index, moment, family in classified:
            if abs(family - majority_family) <= 1.0e-12:
                continue
            sign = -1.0 if moment < 0 else 1.0
            residual = abs(moment) - family
            adjusted_abs = max(0.0, majority_family + residual)
            adjusted[index] = sign * adjusted_abs
            changed_by_element[symbol] += 1
            changed_atoms.append(index + 1)
    return adjusted, {
        "changed_count": int(sum(changed_by_element.values())),
        "changed_by_element": dict(sorted(changed_by_element.items())),
        "majority_families": majority_families,
        "changed_atoms_1based": changed_atoms,
    }


def nearest_moment_family(moment_abs: float, family_targets: list[float]) -> float | None:
    if not family_targets:
        return None
    family = min(family_targets, key=lambda value: abs(moment_abs - value))
    return family if abs(moment_abs - family) <= MOMENT_FAMILY_TOLERANCE else None


def write_initial_spin_candidate_index(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "source_run_dir",
        "run_dir",
        "poscar",
        "incar",
        "changed_count",
        "changed_by_element_json",
        "majority_families_json",
        "changed_atoms_1based",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "source_run_dir": row.get("source_run_dir", ""),
                    "run_dir": row.get("run_dir", ""),
                    "poscar": row.get("poscar", ""),
                    "incar": row.get("incar", ""),
                    "changed_count": row.get("changed_count", ""),
                    "changed_by_element_json": json.dumps(row.get("changed_by_element", {}), sort_keys=True),
                    "majority_families_json": json.dumps(row.get("majority_families", {}), sort_keys=True),
                    "changed_atoms_1based": " ".join(str(item) for item in row.get("changed_atoms_1based", [])),
                }
            )


def strip_internal_candidate_fields(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    stripped: list[dict[str, object]] = []
    for row in rows:
        stripped.append({key: value for key, value in row.items() if not key.startswith("_")})
    return stripped


def write_projection_atat_rndstr(
    path: Path,
    target: PoscarStructure,
    target_symbols: list[str],
    cation_sites: list[int],
    anion_sites: list[int],
    cation_decorations: list[tuple[str | None, float | None]],
    anion_decorations: list[tuple[str | None, float | None]],
    vacancy_label: str,
    active_sublattices: set[str],
    anion_elements: set[str],
) -> None:
    cation_spec = atat_site_spec(cation_decorations, vacancy_label)
    anion_spec = atat_site_spec(anion_decorations, vacancy_label)
    cation_set = set(cation_sites)
    anion_set = set(anion_sites)
    lines: list[str] = []
    for vector in target.cell:
        lines.append(" ".join(f"{float(value):.12f}" for value in vector))
    for vector in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        lines.append(" ".join(f"{float(value):.12f}" for value in vector))
    for index, position in enumerate(target.scaled_positions):
        if index in cation_set and "cation" in active_sublattices:
            species = cation_spec
        elif index in anion_set and "anion" in active_sublattices:
            species = anion_spec
        else:
            species = target_symbols[index]
        lines.append(" ".join(f"{float(value):.12f}" for value in position) + f" {species}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_atat_projection_pseudo_species_map(
        path.with_name("pseudo_species_map.csv"),
        [*cation_decorations, *anion_decorations],
        vacancy_label,
        anion_elements,
    )


def atat_site_spec(decorations: list[tuple[str | None, float | None]], vacancy_label: str) -> str:
    counts = Counter(pseudo_species_label(symbol, moment, vacancy_label) for symbol, moment in decorations)
    total = sum(counts.values())
    if total <= 0:
        return vacancy_label
    return ",".join(f"{label}={count / total:.12g}" for label, count in sorted(counts.items()))


def pseudo_species_label(symbol: str | None, moment: float | None, vacancy_label: str) -> str:
    if symbol is None:
        return vacancy_label
    if moment is None or abs(moment) < 1.0e-8:
        return symbol
    sign = "p" if moment >= 0.0 else "m"
    family = int(round(abs(moment)))
    return f"{symbol}_{sign}{family:g}"


def write_atat_projection_pseudo_species_map(
    path: Path,
    decorations: list[tuple[str | None, float | None]],
    vacancy_label: str,
    anion_elements: set[str],
) -> None:
    rows = []
    seen: set[str] = set()
    for symbol, moment in decorations:
        label = pseudo_species_label(symbol, moment, vacancy_label)
        if label in seen:
            continue
        seen.add(label)
        rows.append(
            {
                "pseudo_species": label,
                "element": "" if symbol is None else symbol,
                "sublattice": "anion" if symbol is None or symbol in anion_elements else "cation",
                "magmom": "" if moment is None else f"{moment:.8g}",
                "is_vacancy": "true" if symbol is None else "false",
                "notes": "pseudo-species label for ATAT rndstr.in; map back to element/MAGMOM before VASP",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["pseudo_species", "element", "sublattice", "magmom", "is_vacancy", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_projection_atat_mcsqs_scripts(
    atat_dir: Path,
    *,
    atat_atoms: int,
    job_name: str,
    walltime: str,
    pair_diameter: float,
    triplet_diameter: float | None,
    quadruplet_diameter: float | None,
    temperature: float | None,
    max_steps: int | None,
    vacancy_label: str,
) -> dict[str, Path]:
    atat_dir = atat_dir.expanduser().resolve()
    atat_dir.mkdir(parents=True, exist_ok=True)
    cluster_parts = ["mcsqs"]
    if pair_diameter > 0:
        cluster_parts.append(f"-2={pair_diameter:g}")
    if triplet_diameter is not None and triplet_diameter > 0:
        cluster_parts.append(f"-3={triplet_diameter:g}")
    if quadruplet_diameter is not None and quadruplet_diameter > 0:
        cluster_parts.append(f"-4={quadruplet_diameter:g}")
    search_parts = ["mcsqs", f"-n={atat_atoms}"]
    if temperature is not None:
        search_parts.append(f"-T={temperature:g}")
    if max_steps is not None:
        search_parts.append(f"-ms={max_steps}")

    run_script = atat_dir / "run_mcsqs.sh"
    run_script.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                'cd "$(dirname "$0")"',
                'if [ -n "${ATOMI_ATAT_BIN:-}" ]; then',
                '  export PATH="${ATOMI_ATAT_BIN}:${PATH}"',
                "fi",
                'echo "mcsqs executable: $(command -v mcsqs || true)"',
                'if ! command -v mcsqs >/dev/null 2>&1; then',
                '  echo "ERROR: mcsqs is not on PATH. Run confighpc or load/install ATAT first." >&2',
                "  exit 127",
                "fi",
                "tail_log() {",
                '  local file="$1"',
                '  if [ -s "${file}" ]; then',
                '    echo "----- tail ${file} -----" >&2',
                '    tail -80 "${file}" >&2 || true',
                "  fi",
                "}",
                'echo "Generating ATAT clusters..."',
                "set +e",
                " ".join(cluster_parts) + " > mcsqs_clusters.out 2> mcsqs_clusters.err",
                "cluster_status=$?",
                "set -e",
                'if [ "${cluster_status}" -ne 0 ]; then',
                '  echo "ERROR: ATAT cluster generation failed with exit status ${cluster_status}." >&2',
                "  tail_log mcsqs_clusters.out",
                "  tail_log mcsqs_clusters.err",
                '  exit "${cluster_status}"',
                "fi",
                'echo "Running ATAT mcsqs search..."',
                "set +e",
                " ".join(search_parts) + " > mcsqs.out 2> mcsqs.err",
                "mcsqs_status=$?",
                "set -e",
                'if [ "${mcsqs_status}" -ne 0 ]; then',
                '  echo "ERROR: ATAT mcsqs search failed with exit status ${mcsqs_status}." >&2',
                "  tail_log mcsqs.out",
                "  tail_log mcsqs.err",
                '  exit "${mcsqs_status}"',
                "fi",
                'if [ -f bestsqs.out ]; then',
                '  echo "Wrote bestsqs.out"',
                '  echo "Next: convert with Atomi using pseudo_species_map.csv; vacancy label is ' + vacancy_label + '."',
                "else",
                '  echo "WARNING: mcsqs finished without bestsqs.out. Inspect mcsqs.out and mcsqs.err." >&2',
                "  tail_log mcsqs.out",
                "  tail_log mcsqs.err",
                "fi",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    submit_script = atat_dir / "submit_mcsqs.sbatch"
    submit_script.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                f"#SBATCH --job-name={job_name}",
                "#SBATCH --output=atat_mcsqs.%x.%j.out",
                "#SBATCH --error=atat_mcsqs.%x.%j.err",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                "#SBATCH --cpus-per-task=1",
                "#SBATCH --mem-per-cpu=3500M",
                f"#SBATCH --time={walltime}",
                "",
                "set -euo pipefail",
                f"SCRIPT_DIR={shlex.quote(str(atat_dir))}",
                'cd "${SCRIPT_DIR}"',
                'if command -v confighpc >/dev/null 2>&1; then',
                '  if [ -n "${ATOMI_HPC_CONFIG:-}" ]; then',
                '    eval "$(confighpc --config "$ATOMI_HPC_CONFIG" --shell)"',
                '  elif [ -f "$HOME/atomi_hpc/atomi_hpc_config.kit.local.json" ]; then',
                '    eval "$(confighpc --config "$HOME/atomi_hpc/atomi_hpc_config.kit.local.json" --shell)"',
                "  fi",
                "fi",
                'if [ -n "${ATOMI_ATAT_BIN:-}" ]; then',
                '  export PATH="${ATOMI_ATAT_BIN}:${PATH}"',
                "fi",
                'echo "ATOMI_ATAT_BIN=${ATOMI_ATAT_BIN:-}"',
                'echo "PATH mcsqs=$(command -v mcsqs || true)"',
                "bash run_mcsqs.sh",
                "",
            ]
        ),
        encoding="utf-8",
    )
    submit_script.chmod(0o755)

    readme = atat_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Atomi POSCAR Projection ATAT Randomization",
                "",
                "This folder was generated by `vasp-project-poscar --randomize-candidates`.",
                "",
                "Files:",
                "- `rndstr.in`: ATAT/SQS random structure input using pseudo-species labels.",
                "- `pseudo_species_map.csv`: maps pseudo-species labels back to element, sublattice, MAGMOM, and vacancy state.",
                "- `run_mcsqs.sh`: foreground/debug mcsqs run.",
                "- `submit_mcsqs.sbatch`: Slurm wrapper for the same mcsqs run.",
                "- `mcsqs_clusters.out/.err` and `mcsqs.out/.err`: ATAT logs to inspect if the job fails.",
                "",
                "Submit on HPC:",
                "",
                "```bash",
                "sbatch submit_mcsqs.sbatch",
                "```",
                "",
                "If `bestsqs.out` is written, convert it only with a converter that understands",
                "`pseudo_species_map.csv`; vacancy pseudo-species must be removed before VASP.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"run_script": run_script, "submit_script": submit_script, "readme": readme}


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


def grouped_species_and_indices(
    symbols: list[str],
    order: list[str],
    excluded_indices: set[int] | None = None,
) -> tuple[PoscarSpecies, list[int]]:
    excluded_indices = excluded_indices or set()
    counts: list[int] = []
    indices: list[int] = []
    kept_symbols: list[str] = []
    for symbol in order:
        group = [index for index, item in enumerate(symbols) if item == symbol and index not in excluded_indices]
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


def select_target_anion_vacancies(
    source: PoscarStructure,
    target: PoscarStructure,
    source_symbols: list[str],
    target_symbols: list[str],
    projected_symbols: list[str],
    source_by_target: dict[int, int],
    source_moments: list[float] | None,
    cation_elements: set[str],
    anion_elements: set[str],
    oxidation_states: dict[str, float],
    magmom_oxidation_states: dict[str, list[tuple[float, float]]],
) -> tuple[set[int], dict[str, object], dict[str, object]]:
    target_anion_indices = [index for index, symbol in enumerate(target_symbols) if symbol in anion_elements]
    source_anion_indices = [index for index, symbol in enumerate(source_symbols) if symbol in anion_elements]
    target_counts = Counter(target_symbols[index] for index in target_anion_indices)
    source_counts = Counter(source_symbols[index] for index in source_anion_indices)
    desired_counts = dict(source_counts)
    desired_reason = "prepared_source_anion_count"
    charge_summary = projected_charge_summary(
        projected_symbols,
        source_by_target,
        source_moments,
        cation_elements,
        anion_elements,
        oxidation_states,
        magmom_oxidation_states,
    )
    charge_desired = charge_neutral_anion_counts(charge_summary, target_counts, oxidation_states)
    if charge_desired:
        desired_counts = charge_desired
        desired_reason = "charge_neutrality"
    removed: set[int] = set()
    assignment_distances: list[float] = []
    for symbol, target_count in target_counts.items():
        desired_count = min(max(int(desired_counts.get(symbol, target_count)), 0), target_count)
        if desired_count >= target_count:
            continue
        target_group = [index for index in target_anion_indices if target_symbols[index] == symbol]
        source_group = [index for index in source_anion_indices if source_symbols[index] == symbol]
        keep_count = min(desired_count, len(source_group), len(target_group))
        keep, distances = nearest_partial_site_assignment(
            source.scaled_positions,
            source_group,
            target.scaled_positions,
            target_group,
            target.cell,
            keep_count,
        )
        assignment_distances.extend(distances)
        removed.update(index for index in target_group if index not in keep)
    output_counts = Counter(projected_symbols[index] for index in target_anion_indices if index not in removed)
    removed_locality = removed_anion_locality(
        removed,
        target,
        target_symbols,
        projected_symbols,
        cation_elements,
    )
    summary = {
        "enabled": True,
        "desired_count_reason": desired_reason,
        "source_anion_counts": dict(sorted(source_counts.items())),
        "target_anion_counts_before": dict(sorted(target_counts.items())),
        "desired_anion_counts": dict(sorted((symbol, int(count)) for symbol, count in desired_counts.items())),
        "output_anion_counts": dict(sorted(output_counts.items())),
        "removed_target_atom_indices_1based": [index + 1 for index in sorted(removed)],
        "removed_anion_counts": dict(sorted(Counter(target_symbols[index] for index in removed).items())),
        "removed_anion_nearest_cations": removed_locality,
        "anion_assignment_max_distance_A": max(assignment_distances, default=0.0),
        "anion_assignment_mean_distance_A": (
            sum(assignment_distances) / len(assignment_distances) if assignment_distances else 0.0
        ),
        "note": (
            "When source/prepared A has fewer anions than target B, Atomi keeps the target anion "
            "sites that best match source anions and removes unmatched target anions as vacancies."
        ),
    }
    charge_summary = projected_charge_summary(
        projected_symbols,
        source_by_target,
        source_moments,
        cation_elements,
        anion_elements,
        oxidation_states,
        magmom_oxidation_states,
        removed_anion_targets=removed,
    )
    return removed, summary, charge_summary


def removed_anion_locality(
    removed_anion_targets: set[int],
    target: PoscarStructure,
    target_symbols: list[str],
    projected_symbols: list[str],
    cation_elements: set[str],
) -> list[dict[str, object]]:
    cation_indices = [index for index, symbol in enumerate(projected_symbols) if symbol in cation_elements]
    rows: list[dict[str, object]] = []
    for anion_index in sorted(removed_anion_targets):
        if not cation_indices:
            rows.append(
                {
                    "removed_target_atom": anion_index + 1,
                    "removed_element": target_symbols[anion_index],
                    "nearest_cation_atom": None,
                    "nearest_cation_element": None,
                    "nearest_cation_distance_A": None,
                }
            )
            continue
        nearest = min(
            (
                (
                    fractional_distance_A(
                        target.scaled_positions[anion_index],
                        target.scaled_positions[cation_index],
                        target.cell,
                    ),
                    cation_index,
                )
                for cation_index in cation_indices
            ),
            key=lambda item: (item[0], item[1]),
        )
        rows.append(
            {
                "removed_target_atom": anion_index + 1,
                "removed_element": target_symbols[anion_index],
                "nearest_cation_atom": nearest[1] + 1,
                "nearest_cation_element": projected_symbols[nearest[1]],
                "nearest_cation_distance_A": nearest[0],
            }
        )
    return rows


def nearest_partial_site_assignment(
    source_positions: list[list[float]],
    source_indices: list[int],
    target_positions: list[list[float]],
    target_indices: list[int],
    cell: list[list[float]],
    keep_count: int,
) -> tuple[set[int], list[float]]:
    if keep_count <= 0:
        return set(), []
    candidates: list[tuple[float, int, int]] = []
    for target_index in target_indices:
        for source_index in source_indices:
            distance = fractional_distance_A(target_positions[target_index], source_positions[source_index], cell)
            candidates.append((distance, target_index, source_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_targets: set[int] = set()
    used_sources: set[int] = set()
    distances: list[float] = []
    for distance, target_index, source_index in candidates:
        if target_index in used_targets or source_index in used_sources:
            continue
        used_targets.add(target_index)
        used_sources.add(source_index)
        distances.append(distance)
        if len(used_targets) == keep_count:
            break
    return used_targets, distances


def projected_charge_summary(
    projected_symbols: list[str],
    source_by_target: dict[int, int],
    source_moments: list[float] | None,
    cation_elements: set[str],
    anion_elements: set[str],
    oxidation_states: dict[str, float],
    magmom_oxidation_states: dict[str, list[tuple[float, float]]],
    *,
    removed_anion_targets: set[int] | None = None,
) -> dict[str, object]:
    removed_anion_targets = removed_anion_targets or set()
    counts: Counter[str] = Counter()
    charge_by_symbol: Counter[str] = Counter()
    missing: set[str] = set()
    cation_charge = 0.0
    anion_charge = 0.0
    for index, symbol in enumerate(projected_symbols):
        if index in removed_anion_targets:
            continue
        counts[symbol] += 1
        moment = None
        if index in source_by_target and source_moments is not None:
            source_index = source_by_target[index]
            if source_index < len(source_moments):
                moment = source_moments[source_index]
        oxidation = oxidation_for_symbol(symbol, moment, oxidation_states, magmom_oxidation_states)
        if oxidation is None:
            if symbol in cation_elements or symbol in anion_elements:
                missing.add(symbol)
            continue
        charge_by_symbol[symbol] += oxidation
        if symbol in anion_elements:
            anion_charge += oxidation
        else:
            cation_charge += oxidation
    total = cation_charge + anion_charge
    return {
        "enabled": bool(oxidation_states or magmom_oxidation_states),
        "species_counts": dict(sorted(counts.items())),
        "charge_by_symbol": {symbol: float(charge) for symbol, charge in sorted(charge_by_symbol.items())},
        "cation_charge": float(cation_charge),
        "anion_charge": float(anion_charge),
        "total_charge": float(total),
        "neutrality_ok": abs(total) < 1.0e-6 if oxidation_states or magmom_oxidation_states else None,
        "missing_oxidation_elements": sorted(missing),
    }


def oxidation_for_symbol(
    symbol: str,
    moment: float | None,
    oxidation_states: dict[str, float],
    magmom_oxidation_states: dict[str, list[tuple[float, float]]],
) -> float | None:
    if moment is not None and symbol in magmom_oxidation_states:
        target = abs(moment)
        moment_value, charge = min(
            magmom_oxidation_states[symbol],
            key=lambda item: abs(target - item[0]),
        )
        if abs(target - moment_value) <= MOMENT_FAMILY_TOLERANCE:
            return charge
    return oxidation_states.get(symbol)


def charge_neutral_anion_counts(
    charge_summary: dict[str, object],
    target_counts: Counter[str],
    oxidation_states: dict[str, float],
) -> dict[str, int]:
    if len(target_counts) != 1:
        return {}
    symbol = next(iter(target_counts))
    anion_charge = oxidation_states.get(symbol)
    if anion_charge is None or anion_charge >= 0:
        return {}
    missing = charge_summary.get("missing_oxidation_elements", [])
    if missing:
        return {}
    cation_charge = float(charge_summary.get("cation_charge", 0.0))
    desired = int(round(-cation_charge / anion_charge))
    if desired < 0:
        return {}
    return {symbol: min(desired, int(target_counts[symbol]))}


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
            "Cations are matched by nearest periodic fractional site; source anion vacancies can be preserved in B."
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
    parser.add_argument("--anion-elements", action="append", default=[], help="Elements to leave on the anion/non-projected sublattice. Default: O.")
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
    parser.add_argument(
        "--no-preserve-anion-vacancies",
        action="store_true",
        help="Keep all target-B anion sites even if prepared/source A has fewer anions.",
    )
    parser.add_argument(
        "--oxidation-state",
        action="append",
        default=[],
        help="Element oxidation state, repeatable or comma-separated, e.g. Gd=3,U=4,O=-2.",
    )
    parser.add_argument(
        "--magmom-oxidation",
        action="append",
        default=[],
        help="Map |MAGMOM| to oxidation state, e.g. U:1=5,U:2=4,Gd:7=3.",
    )
    parser.add_argument(
        "--randomize-candidates",
        type=int,
        default=0,
        help=(
            "Write this many randomized occupation candidates on the projected B lattice. "
            "Cation element/MAGMOM decorations and anion/vacancy decorations are shuffled within their sublattices."
        ),
    )
    parser.add_argument(
        "--randomize-pool-size",
        type=int,
        help=(
            "Generate and score this many random decorations, then write the top --randomize-candidates. "
            "If used without --randomize-candidates, Atomi writes the top 3."
        ),
    )
    parser.add_argument("--randomize-seed", type=int, default=12345, help="Random seed for --randomize-candidates.")
    parser.add_argument(
        "--randomize-sublattice",
        action="append",
        default=[],
        help="Sublattice(s) to randomize: cation, anion, or both. Default with --randomize-candidates: cation,anion.",
    )
    parser.add_argument(
        "--randomize-vacancy-label",
        default="Va",
        help="Vacancy pseudo-species label for the ATAT rndstr.in handoff. Default: Va.",
    )
    parser.add_argument(
        "--randomize-min-guest-distance",
        type=float,
        help="Hard filter: fail ranked candidates whose minority guest-cation minimum periodic distance is below this Angstrom value.",
    )
    parser.add_argument(
        "--randomize-max-guest-vacancy-distance",
        type=float,
        help="Hard filter: fail ranked candidates whose nearest guest-to-vacancy distance is above this Angstrom value.",
    )
    parser.add_argument(
        "--randomize-atat-atoms",
        type=int,
        help="Target atom/site count passed to ATAT mcsqs -n. Default: full projected target site count.",
    )
    parser.add_argument(
        "--randomize-atat-job-name",
        default="atat-random",
        help="Slurm job name written into atat_random/submit_mcsqs.sbatch.",
    )
    parser.add_argument(
        "--randomize-mcsqs-walltime",
        default="04:00:00",
        help="Slurm walltime written into atat_random/submit_mcsqs.sbatch. Default: 04:00:00.",
    )
    parser.add_argument(
        "--randomize-mcsqs-pair-diameter",
        type=float,
        default=6.0,
        help="ATAT mcsqs pair cluster diameter passed as -2. Use <=0 to disable. Default: 6.0.",
    )
    parser.add_argument(
        "--randomize-mcsqs-triplet-diameter",
        type=float,
        help="Optional ATAT mcsqs triplet cluster diameter passed as -3.",
    )
    parser.add_argument(
        "--randomize-mcsqs-quadruplet-diameter",
        type=float,
        help="Optional ATAT mcsqs quadruplet cluster diameter passed as -4.",
    )
    parser.add_argument(
        "--randomize-mcsqs-temperature",
        type=float,
        help="Optional ATAT mcsqs Monte Carlo temperature passed as -T.",
    )
    parser.add_argument(
        "--randomize-mcsqs-max-steps",
        type=int,
        help="Optional ATAT mcsqs maximum Monte Carlo steps passed as -ms.",
    )
    parser.add_argument("--magmom-decimals", type=int, default=3)
    return parser


def format_worst_cation_match_lines(match_csv: Path, *, limit: int) -> list[str]:
    if limit <= 0 or not match_csv.exists():
        return []
    rows = list(csv.DictReader(match_csv.read_text(encoding="utf-8").splitlines()))
    if not rows:
        return []
    rows.sort(key=lambda row: float(row.get("distance_A", "0") or 0.0), reverse=True)
    lines: list[str] = []
    for row in rows[:limit]:
        lines.append(
            f"target {row.get('target_atom', ''):>4} {row.get('target_element_before', ''):>2} "
            f"<- source {row.get('source_atom', ''):>4} {row.get('source_element', ''):>2} "
            f"d={float(row.get('distance_A', '0') or 0.0):.3f} A"
        )
    return lines


def format_removed_anion_locality_lines(summary: dict[str, object]) -> list[str]:
    locality = summary.get("removed_anion_nearest_cations", [])
    if not isinstance(locality, list):
        return []
    lines: list[str] = []
    for entry in locality:
        if not isinstance(entry, dict):
            continue
        lines.append(
            f"removed target {entry.get('removed_target_atom', '')} {entry.get('removed_element', '')} "
            f"near target {entry.get('nearest_cation_atom', '')} {entry.get('nearest_cation_element', '')} "
            f"d={float(entry.get('nearest_cation_distance_A', 0.0)):.3f} A"
        )
    return lines


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
        preserve_anion_vacancies=not args.no_preserve_anion_vacancies,
        oxidation_states=parse_oxidation_states(args.oxidation_state),
        magmom_oxidation_states=parse_magmom_oxidation_states(args.magmom_oxidation),
        randomize_candidates=args.randomize_candidates or (3 if args.randomize_pool_size else 0),
        randomize_pool_size=args.randomize_pool_size,
        randomize_seed=args.randomize_seed,
        randomize_sublattices=parse_randomize_sublattices(args.randomize_sublattice),
        randomize_vacancy_label=args.randomize_vacancy_label,
        randomize_min_guest_distance_A=args.randomize_min_guest_distance,
        randomize_max_guest_vacancy_distance_A=args.randomize_max_guest_vacancy_distance,
        randomize_atat_atoms=args.randomize_atat_atoms,
        randomize_atat_job_name=args.randomize_atat_job_name,
        randomize_mcsqs_walltime=args.randomize_mcsqs_walltime,
        randomize_mcsqs_pair_diameter=args.randomize_mcsqs_pair_diameter,
        randomize_mcsqs_triplet_diameter=args.randomize_mcsqs_triplet_diameter,
        randomize_mcsqs_quadruplet_diameter=args.randomize_mcsqs_quadruplet_diameter,
        randomize_mcsqs_temperature=args.randomize_mcsqs_temperature,
        randomize_mcsqs_max_steps=args.randomize_mcsqs_max_steps,
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
    removed_counts = result.anion_vacancy_summary.get("removed_anion_counts", {})
    if removed_counts:
        print(
            "Anion vacancies : "
            + " ".join(f"{symbol}:{count}" for symbol, count in removed_counts.items())
            + " target site(s) removed"
        )
    if result.charge_summary.get("enabled"):
        print(
            "Charge check     : "
            f"cation={float(result.charge_summary.get('cation_charge', 0.0)):g} "
            f"anion={float(result.charge_summary.get('anion_charge', 0.0)):g} "
            f"total={float(result.charge_summary.get('total_charge', 0.0)):g} "
            f"neutral={result.charge_summary.get('neutrality_ok')}"
        )
    guest_distances = result.guest_cation_distance_summary.get("symbols", {})
    if isinstance(guest_distances, dict) and guest_distances:
        print("Guest-cation distances:")
        for symbol, entry in guest_distances.items():
            if not isinstance(entry, dict) or int(entry.get("pair_count", 0)) == 0:
                continue
            source_atoms = ",".join(str(index) for index in entry.get("source_atom_indices_1based", []))
            output_atoms = ",".join(str(index) for index in entry.get("output_atom_indices_1based", []))
            print(
                f"  {symbol}: nearest {float(entry.get('source_min_distance_A', 0.0)):g} A"
                f" -> {float(entry.get('output_min_distance_A', 0.0)):g} A; "
                f"mean {float(entry.get('source_mean_distance_A', 0.0)):g} A"
                f" -> {float(entry.get('output_mean_distance_A', 0.0)):g} A; "
                f"pairs={int(entry.get('pair_count', 0))}; "
                f"source atoms [{source_atoms}] -> output atoms [{output_atoms}]"
            )
    worst_match_lines = format_worst_cation_match_lines(result.match_csv, limit=10)
    if worst_match_lines:
        print("Worst cation matches:")
        for line in worst_match_lines:
            print(f"  {line}")
    vacancy_lines = format_removed_anion_locality_lines(result.anion_vacancy_summary)
    if vacancy_lines:
        print("Removed anion vacancy locality:")
        for line in vacancy_lines:
            print(f"  {line}")
    if result.direct_candidate_summary.get("enabled"):
        print(f"Direct candidate : {result.direct_candidate_summary.get('run_dir')}")
    if result.randomized_candidate_summary.get("enabled"):
        print(f"Randomized candidates: {result.randomized_candidate_summary.get('candidate_count', 0)}")
        print(f"  Pool       : {result.randomized_candidate_summary.get('pool_size')}")
        print(f"  Selected   : {result.randomized_candidate_summary.get('selected_count')}")
        print(f"  Candidates : {result.randomized_candidate_summary.get('candidates_dir')}")
        print(f"  Index      : {result.randomized_candidate_summary.get('candidate_index')}")
        print(f"  Pool index : {result.randomized_candidate_summary.get('pool_index')}")
        print(f"  Runlist    : {result.randomized_candidate_summary.get('runlist')}")
        print(f"  ATAT rndstr: {result.randomized_candidate_summary.get('atat_rndstr')}")
        print(f"  ATAT run   : {result.randomized_candidate_summary.get('atat_run_script')}")
        print(f"  ATAT sbatch: {result.randomized_candidate_summary.get('atat_submit_script')}")
    if result.initial_spin_candidate_summary.get("enabled"):
        print(f"Initial-spin mirrors: {result.initial_spin_candidate_summary.get('candidate_count', 0)}")
        print(f"  Candidates : {result.initial_spin_candidate_summary.get('candidates_dir')}")
        print(f"  Index      : {result.initial_spin_candidate_summary.get('candidate_index')}")
        print(f"  Runlist    : {result.initial_spin_candidate_summary.get('runlist')}")
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


def parse_randomize_sublattices(raw_items: list[str]) -> set[str] | None:
    if not raw_items:
        return None
    values = {item.strip().lower() for raw in raw_items for item in raw.replace(",", " ").split() if item.strip()}
    aliases = {
        "cations": "cation",
        "cat": "cation",
        "anions": "anion",
        "an": "anion",
        "both": "both",
    }
    normalized = {aliases.get(value, value) for value in values}
    if "both" in normalized:
        normalized.update({"cation", "anion"})
        normalized.discard("both")
    invalid = normalized - {"cation", "anion"}
    if invalid:
        raise ValueError(f"Unknown --randomize-sublattice value(s): {', '.join(sorted(invalid))}.")
    return normalized


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


def parse_oxidation_states(values: list[str]) -> dict[str, float]:
    states: dict[str, float] = {}
    for item in values:
        for part in str(item).replace(",", " ").split():
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Expected oxidation state like U=4, got {part!r}.")
            symbol, value = part.split("=", 1)
            states[symbol.strip()] = float(value)
    return states


def parse_magmom_oxidation_states(values: list[str]) -> dict[str, list[tuple[float, float]]]:
    states: dict[str, list[tuple[float, float]]] = {}
    for item in values:
        for part in str(item).replace(",", " ").split():
            if not part:
                continue
            if ":" not in part or "=" not in part:
                raise ValueError(f"Expected MAGMOM oxidation map like U:2=4, got {part!r}.")
            left, charge = part.split("=", 1)
            symbol, moment = left.split(":", 1)
            states.setdefault(symbol.strip(), []).append((abs(float(moment)), float(charge)))
    return states


if __name__ == "__main__":
    main()
