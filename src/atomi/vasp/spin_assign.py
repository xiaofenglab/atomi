from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from atomi.vasp.magmom import (
    PoscarSpecies,
    find_magmom_line,
    format_magmom_line,
    parse_element_list,
    read_poscar_structure,
    reorder_incar_species_tags,
)


@dataclass(frozen=True)
class SpinAssignmentResult:
    output_poscar: Path
    output_incar: Path
    plan_json: Path
    output_species: PoscarSpecies
    magmom_line: str
    summary: dict[str, object]


@dataclass(frozen=True)
class MomentRule:
    symbol: str
    ranges: list[tuple[int, int]]
    magnitude: float


def atom_symbols(species: PoscarSpecies) -> list[str]:
    symbols: list[str] = []
    for symbol, count in zip(species.symbols, species.counts):
        symbols.extend([symbol] * count)
    return symbols


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


def parse_key_values(items: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in items:
        for part in str(item).replace(",", " ").split():
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Expected element=value assignment, got {part!r}.")
            symbol, value = part.split("=", 1)
            values[symbol.strip()] = float(value)
    return values


def parse_index_ranges(raw: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for part in raw.replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        if "-" in text:
            start_text, end_text = text.split("-", 1)
            start = int(start_text)
            end = int(end_text)
        else:
            start = end = int(text)
        if start <= 0 or end <= 0 or end < start:
            raise ValueError(f"Invalid 1-based atom range: {text!r}.")
        ranges.append((start, end))
    return ranges


def parse_moment_rule(raw: str) -> MomentRule:
    if "=" not in raw or ":" not in raw:
        raise ValueError(
            f"Expected special moment rule like U:9-14,94-98=1, got {raw!r}."
        )
    left, value = raw.split("=", 1)
    symbol, ranges = left.split(":", 1)
    return MomentRule(symbol=symbol.strip(), ranges=parse_index_ranges(ranges), magnitude=abs(float(value)))


def special_rule_indices(rules: list[MomentRule]) -> dict[str, dict[int, float]]:
    by_symbol: dict[str, dict[int, float]] = {}
    for rule in rules:
        target = by_symbol.setdefault(rule.symbol, {})
        for start, end in rule.ranges:
            for index in range(start, end + 1):
                target[index] = rule.magnitude
    return by_symbol


def default_species_order(
    source_species: PoscarSpecies,
    cation_order: list[str],
    cation_elements: set[str],
    anion_elements: set[str],
) -> list[str]:
    source_order = source_species.symbols
    present = set(source_order)
    cations = {symbol for symbol in present if symbol in cation_elements} if cation_elements else present - anion_elements
    anions = present & anion_elements
    order: list[str] = []
    for symbol in cation_order:
        if symbol in cations and symbol not in order:
            order.append(symbol)
    for symbol in source_order:
        if symbol in cations and symbol not in order:
            order.append(symbol)
    for symbol in source_order:
        if symbol in anions and symbol not in order:
            order.append(symbol)
    for symbol in source_order:
        if symbol not in order:
            order.append(symbol)
    return order


def grouped_species_and_indices(symbols: list[str], order: list[str]) -> tuple[PoscarSpecies, list[int]]:
    grouped_symbols: list[str] = []
    counts: list[int] = []
    indices: list[int] = []
    for symbol in order:
        group = [index for index, item in enumerate(symbols) if item == symbol]
        if not group:
            continue
        grouped_symbols.append(symbol)
        counts.append(len(group))
        indices.extend(group)
    return PoscarSpecies(grouped_symbols, counts), indices


def afm_sign(index_within_element: int) -> float:
    return 1.0 if index_within_element % 2 == 1 else -1.0


def ordered_moments(
    source_symbols: list[str],
    output_indices: list[int],
    default_moments: dict[str, float],
    rules: list[MomentRule],
    magnetic_order: str,
) -> tuple[list[float], list[dict[str, object]]]:
    special = special_rule_indices(rules)
    source_occurrence: dict[str, int] = {}
    source_occurrences: list[int] = []
    for symbol in source_symbols:
        source_occurrence[symbol] = source_occurrence.get(symbol, 0) + 1
        source_occurrences.append(source_occurrence[symbol])
    for rule in rules:
        available = source_occurrence.get(rule.symbol, 0)
        if available == 0:
            raise ValueError(f"Special moment rule references absent element {rule.symbol!r}.")
        for _start, end in rule.ranges:
            if end > available:
                raise ValueError(
                    f"Special moment rule {rule.symbol}:{end} exceeds available "
                    f"{rule.symbol} atom count {available}."
                )

    moments_by_source: list[float] = []
    assigned_rules: list[dict[str, object]] = []
    for index, symbol in enumerate(source_symbols):
        occurrence = source_occurrences[index]
        magnitude = abs(default_moments.get(symbol, 0.0))
        rule_applied = False
        if occurrence in special.get(symbol, {}):
            magnitude = special[symbol][occurrence]
            rule_applied = True
        if magnetic_order in {"afm", "afm-like", "afmlike"}:
            sign = afm_sign(occurrence)
        elif magnetic_order in {"fm", "positive"}:
            sign = 1.0
        elif magnetic_order == "negative":
            sign = -1.0
        else:
            raise ValueError(f"Unsupported magnetic order: {magnetic_order!r}.")
        value = 0.0 if magnitude == 0.0 else sign * magnitude
        moments_by_source.append(value)
        if rule_applied:
            assigned_rules.append(
                {
                    "source_atom_index_1based": index + 1,
                    "element": symbol,
                    "element_index_1based": occurrence,
                    "moment": value,
                }
            )
    return [moments_by_source[index] for index in output_indices], assigned_rules


def replace_or_append_magmom_text(text: str, magmom_line: str) -> str:
    lines = text.splitlines()
    line_index, _line = find_magmom_line(lines)
    if line_index is None:
        lines.append(magmom_line)
    else:
        lines[line_index] = magmom_line
    return "\n".join(lines).rstrip() + "\n"


def comment_incar_tag(text: str, tag: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(tag)}\s*=", re.IGNORECASE)
    lines = []
    for line in text.splitlines():
        if pattern.match(line):
            lines.append("#" + line if not line.lstrip().startswith("#") else line)
        else:
            lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def write_incar(
    source_incar: Path | None,
    incar_species: PoscarSpecies,
    output_species: PoscarSpecies,
    magmom_line: str,
    *,
    comment_nupdown: bool,
) -> str:
    text = source_incar.read_text(encoding="utf-8", errors="replace") if source_incar is not None else ""
    if text:
        text = reorder_incar_species_tags(text, incar_species, output_species)
    if comment_nupdown and text:
        text = comment_incar_tag(text, "NUPDOWN")
    return replace_or_append_magmom_text(text, magmom_line)


def infer_incar_species(
    source_incar: Path | None,
    fallback_species: PoscarSpecies,
    *,
    incar_poscar: Path | None = None,
) -> tuple[PoscarSpecies, str]:
    """Return the POSCAR species order that the source INCAR species tags use."""
    if source_incar is None:
        return fallback_species, "no source INCAR; input POSCAR order"
    if incar_poscar is not None:
        path = incar_poscar.expanduser().resolve()
        return read_poscar_structure(path).species, str(path)
    neighbor = source_incar.parent / "POSCAR"
    if neighbor.is_file():
        return read_poscar_structure(neighbor).species, str(neighbor)
    return fallback_species, "input POSCAR order fallback; no POSCAR beside source INCAR"


def summarize_moments(species: PoscarSpecies, moments: list[float]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    cursor = 0
    for symbol, count in zip(species.symbols, species.counts):
        values = moments[cursor : cursor + count]
        cursor += count
        summary[symbol] = {
            "count": count,
            "positive": sum(1 for value in values if value > 1.0e-8),
            "negative": sum(1 for value in values if value < -1.0e-8),
            "zero": sum(1 for value in values if abs(value) <= 1.0e-8),
            "sum": round(sum(values), 6),
            "unique_abs_moments": sorted({round(abs(value), 6) for value in values}),
        }
    return summary


def copy_static_inputs(source_dir: Path, outdir: Path, names: tuple[str, ...] = ("KPOINTS", "POTCAR")) -> list[str]:
    copied: list[str] = []
    for name in names:
        source = source_dir / name
        if source.is_file():
            target = outdir / name
            shutil.copy2(source, target)
            copied.append(str(target.resolve()))
    return copied


def assign_spins(
    poscar: Path,
    *,
    outdir: Path,
    incar: Path | None = None,
    incar_poscar: Path | None = None,
    cation_elements: list[str] | None = None,
    anion_elements: list[str] | None = None,
    species_order: list[str] | None = None,
    default_moments: dict[str, float] | None = None,
    moment_rules: list[MomentRule] | None = None,
    magnetic_order: str = "afm",
    magmom_decimals: int = 3,
    comment_nupdown: bool = True,
    copy_inputs: bool = True,
) -> SpinAssignmentResult:
    poscar = poscar.expanduser().resolve()
    incar = incar.expanduser().resolve() if incar is not None else None
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    structure = read_poscar_structure(poscar)
    source_symbols = atom_symbols(structure.species)
    cation_order = list(cation_elements or [])
    anion_set = set(anion_elements or ["O"])
    cation_set = set(cation_order)
    order = species_order or default_species_order(structure.species, cation_order, cation_set, anion_set)
    output_species, output_indices = grouped_species_and_indices(source_symbols, order)
    output_positions = [structure.scaled_positions[index] for index in output_indices]
    moments, assigned_rules = ordered_moments(
        source_symbols,
        output_indices,
        default_moments or {},
        moment_rules or [],
        magnetic_order,
    )
    selected = [symbol for symbol, entry in summarize_moments(output_species, moments).items() if entry["unique_abs_moments"] != [0.0]]
    magmom_line = format_magmom_line(
        output_species,
        moments,
        selected_elements=selected,
        decimals=magmom_decimals,
        compact_zero=True,
    )

    output_poscar = outdir / "POSCAR"
    output_incar = outdir / "INCAR"
    plan_json = outdir / "spin_assignment_plan.json"
    output_poscar.write_text(
        write_poscar_text(
            f"Spin-assigned {poscar.name}",
            output_species,
            structure.cell,
            output_positions,
        ),
        encoding="utf-8",
    )
    source_incar = incar if incar is not None else (poscar.parent / "INCAR" if (poscar.parent / "INCAR").is_file() else None)
    incar_species, incar_species_source = infer_incar_species(
        source_incar,
        structure.species,
        incar_poscar=incar_poscar,
    )
    output_incar.write_text(
        write_incar(
            source_incar,
            incar_species,
            output_species,
            magmom_line,
            comment_nupdown=comment_nupdown,
        ),
        encoding="utf-8",
    )
    copied = copy_static_inputs((source_incar or poscar).parent, outdir) if copy_inputs else []
    summary = {
        "source_poscar": str(poscar),
        "source_incar": "" if source_incar is None else str(source_incar),
        "source_incar_species_order_source": incar_species_source,
        "source_incar_species_order": incar_species.symbols,
        "source_incar_species_counts": dict(zip(incar_species.symbols, incar_species.counts)),
        "output_poscar": str(output_poscar),
        "output_incar": str(output_incar),
        "source_species_order": structure.species.symbols,
        "source_species_counts": dict(zip(structure.species.symbols, structure.species.counts)),
        "output_species_order": output_species.symbols,
        "output_species_counts": dict(zip(output_species.symbols, output_species.counts)),
        "magnetic_order": magnetic_order,
        "default_moments": default_moments or {},
        "special_rule_atoms": assigned_rules,
        "magmom_count": len(moments),
        "expected_magmom_count": output_species.total_atoms,
        "magmom_count_ok": len(moments) == output_species.total_atoms,
        "moment_summary": summarize_moments(output_species, moments),
        "copied_static_inputs": copied,
    }
    plan_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SpinAssignmentResult(
        output_poscar=output_poscar,
        output_incar=output_incar,
        plan_json=plan_json,
        output_species=output_species,
        magmom_line=magmom_line,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-assign-spins",
        description="Write a cation-first POSCAR and INCAR MAGMOM from element/range spin rules.",
    )
    parser.add_argument("--poscar", type=Path, default=Path("POSCAR"), help="Input POSCAR.")
    parser.add_argument("--incar", type=Path, help="Reference INCAR. Defaults to POSCAR folder INCAR when present.")
    parser.add_argument(
        "--incar-poscar",
        type=Path,
        help=(
            "POSCAR whose species order defines source INCAR LDAUL/LDAUU/LDAUJ. "
            "Defaults to POSCAR beside --incar, then the input --poscar."
        ),
    )
    parser.add_argument("--outdir", type=Path, required=True, help="Output folder for POSCAR/INCAR.")
    parser.add_argument("--cation-elements", action="append", default=[], help="Cation order, e.g. U,Gd.")
    parser.add_argument("--anion-elements", action="append", default=[], help="Anion elements. Default: O.")
    parser.add_argument("--species-order", action="append", default=[], help="Explicit output species order, e.g. U,O.")
    parser.add_argument("--moment", action="append", default=[], help="Default moment magnitude, e.g. U=2,O=0.")
    parser.add_argument(
        "--special-moment",
        action="append",
        default=[],
        help="Element-index range override, e.g. U:9-14,94-98,117-122=1.",
    )
    parser.add_argument(
        "--magnetic-order",
        default="afm",
        choices=("afm", "afm-like", "afmlike", "fm", "positive", "negative"),
        help="Sign pattern for nonzero moments. Default: afm, alternating by element occurrence.",
    )
    parser.add_argument("--magmom-decimals", type=int, default=3)
    parser.add_argument("--keep-nupdown", action="store_true", help="Do not comment out an existing NUPDOWN tag.")
    parser.add_argument("--no-copy-inputs", action="store_true", help="Do not copy KPOINTS/POTCAR beside output INCAR.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = assign_spins(
        args.poscar,
        outdir=args.outdir,
        incar=args.incar,
        incar_poscar=args.incar_poscar,
        cation_elements=parse_element_list(args.cation_elements),
        anion_elements=parse_element_list(args.anion_elements) or ["O"],
        species_order=parse_element_list(args.species_order),
        default_moments=parse_key_values(args.moment),
        moment_rules=[parse_moment_rule(item) for item in args.special_moment],
        magnetic_order=args.magnetic_order,
        magmom_decimals=args.magmom_decimals,
        comment_nupdown=not args.keep_nupdown,
        copy_inputs=not args.no_copy_inputs,
    )
    print(f"Output POSCAR : {result.output_poscar}")
    print(f"Output INCAR  : {result.output_incar}")
    print(f"Plan JSON     : {result.plan_json}")
    print(
        "Output species: "
        + " ".join(f"{symbol}:{count}" for symbol, count in zip(result.output_species.symbols, result.output_species.counts))
    )
    print(f"MAGMOM count  : {result.summary['magmom_count']}/{result.summary['expected_magmom_count']}")
    print("Moment summary:")
    for symbol, entry in result.summary["moment_summary"].items():
        print(
            f"  {symbol}: n={entry['count']} +{entry['positive']} -{entry['negative']} "
            f"0={entry['zero']} abs={entry['unique_abs_moments']} sum={entry['sum']}"
        )
    if result.summary["special_rule_atoms"]:
        print(f"Special atoms : {len(result.summary['special_rule_atoms'])}")


if __name__ == "__main__":
    main()
