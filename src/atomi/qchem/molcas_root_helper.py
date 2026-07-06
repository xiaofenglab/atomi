"""OpenMolcas RASSCF root-count helper.

This module audits RASSCF root sections in OpenMolcas output and helps build
small-root diagnostic inputs that can later be promoted to full-root inputs.
It is intentionally text based and conservative: original inputs are never
modified in place, and automatic promotion defaults to HEXS blocks only.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_AUDIT = "atomi.openmolcas_root_audit.v1"
SCHEMA_REWRITE = "atomi.openmolcas_root_rewrite.v1"

ROOT_AUDIT_PATTERNS = (
    "CI expansion specifications",
    "Number of CSFs",
    "Number of highly excited CSFs",
    "Number of determinants",
    "Number of root(s) required",
    "CI roots used",
    "highest root included in the CI",
    "max. size of the explicit Hamiltonian",
)


@dataclass
class RootAuditBlock:
    block_index: int
    start_line: int
    end_line: int
    number_csfs: int | None = None
    highly_excited_csfs: int | None = None
    determinants: int | None = None
    roots_required: int | None = None
    ci_roots_min: int | None = None
    ci_roots_max: int | None = None
    ci_roots_count: int | None = None
    highest_root: int | None = None
    max_explicit_hamiltonian: int | None = None

    @property
    def recommended_full_roots(self) -> int | None:
        if self.highly_excited_csfs:
            return self.highly_excited_csfs
        return self.number_csfs


@dataclass
class RasscfInputBlock:
    block_index: int
    start_line: int
    end_line: int
    has_hexs: bool
    ciroots_line_index: int | None = None
    ciroots_value_line_index: int | None = None
    roots_requested: int | None = None
    roots_selected: int | None = None
    root_step: int | None = None
    spin: int | None = None
    title: str = ""


def _parse_int(line: str) -> int | None:
    match = re.search(r"([-+]?\d+)", line)
    return int(match.group(1)) if match else None


def _parse_ints(line: str) -> list[int]:
    return [int(value) for value in re.findall(r"[-+]?\d+", line)]


def _json_dump(payload: dict[str, Any], path: Path | None = None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def audit_output(path: Path) -> dict[str, Any]:
    lines = path.expanduser().read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: list[RootAuditBlock] = []
    idx = 0
    while idx < len(lines):
        if "CI expansion specifications" not in lines[idx]:
            idx += 1
            continue
        start = idx
        idx += 1
        block = RootAuditBlock(block_index=len(blocks) + 1, start_line=start + 1, end_line=start + 1)
        while idx < len(lines):
            line = lines[idx]
            if idx > start + 1 and (line.startswith("++") or line.startswith("--- Start Module:")):
                break
            if "Number of CSFs" in line and "highly" not in line:
                block.number_csfs = _parse_int(line)
            elif "Number of highly excited CSFs" in line:
                block.highly_excited_csfs = _parse_int(line)
            elif "Number of determinants" in line:
                block.determinants = _parse_int(line)
            elif "Number of root(s) required" in line:
                block.roots_required = _parse_int(line)
            elif "CI roots used" in line:
                values = _parse_ints(line)
                next_idx = idx + 1
                while next_idx < len(lines) and re.match(r"^\s+\d", lines[next_idx]):
                    values.extend(_parse_ints(lines[next_idx]))
                    next_idx += 1
                if values:
                    block.ci_roots_min = min(values)
                    block.ci_roots_max = max(values)
                    block.ci_roots_count = len(values)
            elif "highest root included in the CI" in line:
                block.highest_root = _parse_int(line)
            elif "max. size of the explicit Hamiltonian" in line:
                block.max_explicit_hamiltonian = _parse_int(line)
            block.end_line = idx + 1
            idx += 1
        blocks.append(block)
    payload = {
        "schema": SCHEMA_AUDIT,
        "output": str(path.expanduser().resolve()),
        "block_count": len(blocks),
        "blocks": [asdict(block) | {"recommended_full_roots": block.recommended_full_roots} for block in blocks],
        "keyword_rule": {
            "actual_roots": "Number of root(s) required",
            "available_hexs_space": "Number of highly excited CSFs",
            "highest_computed_root": "highest root included in the CI",
            "not_root_count": "max. size of the explicit Hamiltonian",
        },
        "grep_command": (
            r"grep -n -E "
            r'"CI expansion specifications|Number of CSFs|Number of highly excited CSFs|'
            r'Number of determinants|Number of root\(s\) required|CI roots used|'
            r'highest root included in the CI|max\. size of the explicit Hamiltonian" *.out*'
        ),
    }
    return payload


def parse_rasscf_blocks(text: str) -> list[RasscfInputBlock]:
    lines = text.splitlines()
    starts = [idx for idx, line in enumerate(lines) if re.match(r"^\s*&RASSCF\b", line, flags=re.IGNORECASE)]
    blocks: list[RasscfInputBlock] = []
    for block_i, start in enumerate(starts, start=1):
        following = [value for value in starts if value > start]
        end = following[0] if following else len(lines)
        block_lines = lines[start:end]
        block = RasscfInputBlock(
            block_index=block_i,
            start_line=start + 1,
            end_line=end,
            has_hexs=any(re.match(r"^\s*HEXS\b", line, flags=re.IGNORECASE) for line in block_lines),
        )
        for rel, line in enumerate(block_lines):
            abs_idx = start + rel
            if re.match(r"^\s*Title\b", line, flags=re.IGNORECASE) and rel + 1 < len(block_lines):
                block.title = block_lines[rel + 1].strip()
            elif re.match(r"^\s*Spin\b", line, flags=re.IGNORECASE) and rel + 1 < len(block_lines):
                block.spin = _parse_int(block_lines[rel + 1])
            elif re.match(r"^\s*CIROOTS\b", line, flags=re.IGNORECASE):
                block.ciroots_line_index = abs_idx
                value_idx = _next_value_line(lines, abs_idx + 1, end)
                block.ciroots_value_line_index = value_idx
                if value_idx is not None:
                    values = _parse_ints(lines[value_idx])
                    if values:
                        block.roots_requested = values[0]
                    if len(values) > 1:
                        block.roots_selected = values[1]
                    if len(values) > 2:
                        block.root_step = values[2]
        blocks.append(block)
    return blocks


def _next_value_line(lines: list[str], start: int, end: int) -> int | None:
    for idx in range(start, end):
        stripped = lines[idx].strip()
        if stripped and not stripped.startswith("*"):
            return idx
    return None


def _replace_ciroots_line(lines: list[str], value_idx: int, new_root: int) -> tuple[int | None, int | None]:
    old_values = _parse_ints(lines[value_idx])
    if not old_values:
        raise ValueError(f"CIROOTS value line {value_idx + 1} has no integers")
    old_root = old_values[0]
    old_selected = old_values[1] if len(old_values) > 1 else old_root
    old_step = old_values[2] if len(old_values) > 2 else 1
    new_values = list(old_values)
    new_values[0] = new_root
    if len(new_values) > 1:
        new_values[1] = new_root
    indent = re.match(r"^(\s*)", lines[value_idx]).group(1)
    lines[value_idx] = indent + " ".join(str(value) for value in new_values)
    return old_root, old_selected if old_step else old_selected


def _state_lines(count: int, per_line: int = 20) -> list[str]:
    values = [str(idx) for idx in range(1, count + 1)]
    return [" ".join(values[idx : idx + per_line]) for idx in range(0, len(values), per_line)]


def _find_rassi_count_lines(lines: list[str]) -> list[tuple[int, int, int]]:
    """Return (counts_line, first_state_line, after_state_lines) for each RASSI count block."""
    found: list[tuple[int, int, int]] = []
    idx = 0
    while idx < len(lines):
        if not re.match(r"^\s*Nr of JobIph files\s*:", lines[idx], flags=re.IGNORECASE):
            idx += 1
            continue
        count_idx = _next_value_line(lines, idx + 1, len(lines))
        if count_idx is None:
            idx += 1
            continue
        nums = _parse_ints(lines[count_idx])
        if not nums:
            idx = count_idx + 1
            continue
        job_count = nums[0]
        cursor = count_idx + 1
        consumed = 0
        while cursor < len(lines) and consumed < sum(nums[1 : 1 + job_count]):
            vals = _parse_ints(lines[cursor])
            if not vals:
                break
            consumed += len(vals)
            cursor += 1
        found.append((count_idx, count_idx + 1, cursor))
        idx = cursor
    return found


def _rewrite_rassi(lines: list[str], old_to_new: list[tuple[int, int]], explicit_counts: list[int] | None = None) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for count_idx, states_start, states_end in reversed(_find_rassi_count_lines(lines)):
        nums = _parse_ints(lines[count_idx])
        if not nums:
            continue
        job_count = nums[0]
        counts = nums[1 : 1 + job_count]
        old_counts = list(counts)
        if explicit_counts is not None:
            if len(explicit_counts) != job_count:
                raise ValueError(f"RASSI count override has {len(explicit_counts)} counts but block expects {job_count}")
            counts = explicit_counts
        else:
            for old, new in old_to_new:
                for pos, value in enumerate(counts):
                    if value == old:
                        counts[pos] = new
                        break
        if counts == old_counts:
            continue
        indent = re.match(r"^(\s*)", lines[count_idx]).group(1)
        lines[count_idx : states_end] = [
            f"{indent}{job_count}   " + "   ".join(str(value) for value in counts),
            *[f"{indent} {line}" for count in counts for line in _state_lines(count)],
        ]
        changes.append({"line": count_idx + 1, "old_counts": old_counts, "new_counts": counts})
    return list(reversed(changes))


def _parse_set(values: list[str]) -> dict[int, int]:
    updates: dict[int, int] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Expected BLOCK=NROOT, got {item!r}")
        left, right = item.split("=", 1)
        updates[int(left)] = int(right)
    return updates


def _audit_blocks_by_index(audit: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(block["block_index"]): block for block in audit.get("blocks", [])}


def rewrite_input(
    *,
    input_path: Path,
    output_path: Path,
    set_roots: dict[int, int],
    from_output: Path | None = None,
    diagnostic_root: int | None = None,
    hexs_only: bool = True,
    include_non_hexs: bool = False,
    max_root: int | None = None,
    rassi: str = "auto",
    rassi_counts: list[int] | None = None,
) -> dict[str, Any]:
    text = input_path.expanduser().read_text(encoding="utf-8")
    lines = text.splitlines()
    blocks = parse_rasscf_blocks(text)
    audit = audit_output(from_output) if from_output else None
    audit_blocks = _audit_blocks_by_index(audit) if audit else {}
    changes: list[dict[str, Any]] = []
    old_to_new: list[tuple[int, int]] = []
    for block in blocks:
        if block.ciroots_value_line_index is None:
            continue
        if hexs_only and not block.has_hexs and not include_non_hexs:
            continue
        new_root = set_roots.get(block.block_index)
        source = "manual"
        if diagnostic_root is not None:
            new_root = diagnostic_root
            source = "diagnostic_root"
        elif new_root is None and audit:
            audit_block = audit_blocks.get(block.block_index)
            if audit_block:
                new_root = audit_block.get("highly_excited_csfs") or audit_block.get("number_csfs")
                source = "audit_highly_excited_csfs" if audit_block.get("highly_excited_csfs") else "audit_number_csfs"
        if new_root is None:
            continue
        if max_root is not None and new_root > max_root:
            new_root = max_root
            source += "_clamped"
        old_root = block.roots_requested
        if old_root == new_root:
            continue
        _replace_ciroots_line(lines, block.ciroots_value_line_index, new_root)
        old_to_new.append((old_root or 0, new_root))
        changes.append(
            {
                "block_index": block.block_index,
                "title": block.title,
                "has_hexs": block.has_hexs,
                "line": block.ciroots_value_line_index + 1,
                "old_root": old_root,
                "new_root": new_root,
                "source": source,
            }
        )
    rassi_changes: list[dict[str, Any]] = []
    if rassi != "none":
        rassi_changes = _rewrite_rassi(lines, old_to_new, explicit_counts=rassi_counts if rassi == "counts" else None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "schema": SCHEMA_REWRITE,
        "input": str(input_path.expanduser().resolve()),
        "output": str(output_path.expanduser().resolve()),
        "from_output": str(from_output.expanduser().resolve()) if from_output else "",
        "rasscf_block_count": len(blocks),
        "changes": changes,
        "rassi_changes": rassi_changes,
        "policy": {
            "hexs_only": hexs_only,
            "include_non_hexs": include_non_hexs,
            "max_root": max_root,
            "rassi": rassi,
        },
    }
    return payload


def audit_main(args: argparse.Namespace) -> dict[str, Any]:
    payload = audit_output(args.output)
    if args.write:
        _json_dump(payload, args.write)
    else:
        _json_dump(payload)
    return payload


def rewrite_main(args: argparse.Namespace) -> dict[str, Any]:
    rassi_counts = [int(item) for item in args.rassi_counts.replace(",", " ").split()] if args.rassi_counts else None
    payload = rewrite_input(
        input_path=args.input,
        output_path=args.output,
        set_roots=_parse_set(args.set_root),
        from_output=args.from_output,
        diagnostic_root=args.diagnostic_root,
        hexs_only=not args.no_hexs_only,
        include_non_hexs=args.include_non_hexs,
        max_root=args.max_root,
        rassi=args.rassi,
        rassi_counts=rassi_counts,
    )
    if args.write:
        _json_dump(payload, args.write)
    else:
        _json_dump(payload)
    return payload


def make_diagnostic_main(args: argparse.Namespace) -> dict[str, Any]:
    args.diagnostic_root = args.nroot
    return rewrite_main(args)


def promote_main(args: argparse.Namespace) -> dict[str, Any]:
    args.diagnostic_root = None
    return rewrite_main(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit", help="Parse OpenMolcas RASSCF root/CSF audit sections from an output file.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--write", type=Path)
    p.set_defaults(func=audit_main)

    def add_rewrite_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--from-output", type=Path, help="Diagnostic OpenMolcas output used to infer full roots.")
        p.add_argument("--set-root", action="append", default=[], help="Manual BLOCK=NROOT override for RASSCF block index.")
        p.add_argument("--max-root", type=int, help="Clamp inferred roots to this ceiling.")
        p.add_argument("--no-hexs-only", action="store_true", help="Allow all RASSCF blocks to be considered.")
        p.add_argument("--include-non-hexs", action="store_true", help="Promote non-HEXS blocks too when using --from-output.")
        p.add_argument("--rassi", choices=("auto", "none", "counts"), default="auto")
        p.add_argument("--rassi-counts", default="", help="Explicit RASSI counts for --rassi counts, e.g. '28,1120,280'.")
        p.add_argument("--write", type=Path, help="Optional JSON rewrite summary path.")

    p = sub.add_parser("make-diagnostic", help="Write a small-root diagnostic input from a production input.")
    add_rewrite_args(p)
    p.add_argument("--nroot", type=int, required=True, help="Diagnostic root count for selected blocks.")
    p.set_defaults(func=make_diagnostic_main)

    p = sub.add_parser("promote", help="Promote a diagnostic input to inferred full roots using diagnostic output.")
    add_rewrite_args(p)
    p.set_defaults(func=promote_main)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
