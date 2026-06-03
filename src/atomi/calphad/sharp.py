"""SHARP attribution analysis for CALPHAD/MIVM parameter searches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "atomi.calphad.sharp.analysis.v1"


@dataclass(frozen=True)
class ParameterSummary:
    name: str
    feature_type: str
    sensitivity_score: float
    ablation_score: float
    posterior_constraint_score: float
    composite_importance: float
    interpretation: str


def _float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_target(text: str) -> tuple[str, float]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Target must be COLUMN=VALUE, got {text!r}")
    name, value = text.split("=", 1)
    name = name.strip()
    target = _float(value.strip())
    if not name or not math.isfinite(target):
        raise argparse.ArgumentTypeError(f"Target must be COLUMN=finite_number, got {text!r}")
    return name, target


def parse_csv_arg(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                out.append(item)
    return out


def is_numeric_column(rows: list[dict[str, Any]], column: str, *, min_fraction: float = 0.85) -> bool:
    values = [_float(row.get(column)) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return bool(values) and len(finite) / len(values) >= min_fraction


def infer_parameter_columns(
    rows: list[dict[str, Any]],
    *,
    explicit: list[str],
    metric_columns: list[str],
    score_column: str,
    target_columns: list[str],
) -> list[str]:
    if explicit:
        return explicit
    if not rows:
        return []
    excluded = set(metric_columns) | {score_column} | set(target_columns)
    excluded |= {f"abs_{name}_minus_target" for name in target_columns}
    candidates: list[str] = []
    for column in rows[0]:
        if column in excluded:
            continue
        values = [str(row.get(column, "")).strip() for row in rows]
        nonempty = [value for value in values if value]
        if not nonempty:
            continue
        unique = set(nonempty)
        if len(unique) <= max(80, len(nonempty) // 2):
            candidates.append(column)
    return candidates


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for idx in order[i:j]:
            ranks[idx] = rank
        i = j
    return ranks


def pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 3:
        return math.nan
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    den_x = sum((x - x_mean) ** 2 for x in x_values)
    den_y = sum((y - y_mean) ** 2 for y in y_values)
    if den_x <= 0 or den_y <= 0:
        return math.nan
    return num / math.sqrt(den_x * den_y)


def spearman(x_values: list[float], y_values: list[float]) -> float:
    return pearson(rankdata(x_values), rankdata(y_values))


def categorical_eta_squared(groups: dict[str, list[float]]) -> float:
    values = [value for group in groups.values() for value in group]
    if len(values) < 3:
        return math.nan
    mean_all = statistics.fmean(values)
    ss_total = sum((value - mean_all) ** 2 for value in values)
    if ss_total <= 0:
        return math.nan
    ss_between = 0.0
    for group in groups.values():
        if group:
            ss_between += len(group) * (statistics.fmean(group) - mean_all) ** 2
    return ss_between / ss_total


def feature_type(rows: list[dict[str, Any]], column: str) -> str:
    return "numeric" if is_numeric_column(rows, column) else "categorical"


def add_target_residuals(rows: list[dict[str, Any]], targets: dict[str, float]) -> list[str]:
    added: list[str] = []
    for column, target in targets.items():
        residual_column = f"abs_{column}_minus_target"
        added.append(residual_column)
        for row in rows:
            value = _float(row.get(column))
            row[residual_column] = abs(value - target) if math.isfinite(value) else math.nan
    return added


def sensitivity_rows(
    rows: list[dict[str, Any]],
    *,
    parameters: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for parameter in parameters:
        kind = feature_type(rows, parameter)
        for metric in metrics:
            if kind == "numeric":
                pairs = [(_float(row.get(parameter)), _float(row.get(metric))) for row in rows]
                pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
                coefficient = spearman([x for x, _ in pairs], [y for _, y in pairs])
                effect = abs(coefficient) if math.isfinite(coefficient) else math.nan
                direction = "positive" if math.isfinite(coefficient) and coefficient > 0 else "negative" if math.isfinite(coefficient) and coefficient < 0 else "flat"
                output.append(
                    {
                        "parameter": parameter,
                        "parameter_type": kind,
                        "metric": metric,
                        "statistic": "spearman_rho",
                        "effect": effect,
                        "signed_effect": coefficient,
                        "direction": direction,
                        "n": len(pairs),
                        "n_groups": "",
                    }
                )
            else:
                groups: dict[str, list[float]] = defaultdict(list)
                for row in rows:
                    value = str(row.get(parameter, "")).strip()
                    metric_value = _float(row.get(metric))
                    if value and math.isfinite(metric_value):
                        groups[value].append(metric_value)
                eta = categorical_eta_squared(groups)
                output.append(
                    {
                        "parameter": parameter,
                        "parameter_type": kind,
                        "metric": metric,
                        "statistic": "eta_squared",
                        "effect": eta,
                        "signed_effect": eta,
                        "direction": "categorical",
                        "n": sum(len(group) for group in groups.values()),
                        "n_groups": len(groups),
                    }
                )
    return output


def ablation_rows(rows: list[dict[str, Any]], *, parameters: list[str], score_column: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for parameter in parameters:
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = str(row.get(parameter, "")).strip()
            score = _float(row.get(score_column))
            if value and math.isfinite(score):
                groups[value].append(score)
        if not groups:
            continue
        group_stats: list[tuple[str, float, float, int]] = []
        for value, scores in groups.items():
            group_stats.append((value, min(scores), statistics.fmean(scores), len(scores)))
        best = min(group_stats, key=lambda item: item[1])
        worst = max(group_stats, key=lambda item: item[1])
        spread = worst[1] - best[1]
        output.append(
            {
                "parameter": parameter,
                "score_column": score_column,
                "best_value": best[0],
                "best_group_min_score": best[1],
                "best_group_mean_score": best[2],
                "best_group_n": best[3],
                "worst_value": worst[0],
                "worst_group_min_score": worst[1],
                "delta_best_score": spread,
                "n_groups": len(group_stats),
            }
        )
    return output


def top_subset(rows: list[dict[str, Any]], *, score_column: str, top_fraction: float, top_n: int | None) -> list[dict[str, Any]]:
    scored = [(row, _float(row.get(score_column))) for row in rows]
    scored = [(row, score) for row, score in scored if math.isfinite(score)]
    scored.sort(key=lambda item: item[1])
    if top_n is None:
        top_n = max(1, int(math.ceil(len(scored) * top_fraction)))
    top_n = min(max(1, top_n), len(scored))
    return [row for row, _ in scored[:top_n]]


def robustness_rows(
    rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    *,
    parameters: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for parameter in parameters:
        kind = feature_type(rows, parameter)
        if kind == "numeric":
            full_values = [_float(row.get(parameter)) for row in rows]
            top_values = [_float(row.get(parameter)) for row in top_rows]
            full_values = [value for value in full_values if math.isfinite(value)]
            top_values = [value for value in top_values if math.isfinite(value)]
            if not full_values or not top_values:
                continue
            full_range = max(full_values) - min(full_values)
            top_range = max(top_values) - min(top_values)
            constraint = 1.0 - (top_range / full_range) if full_range > 0 else 1.0
            output.append(
                {
                    "parameter": parameter,
                    "parameter_type": kind,
                    "top_n": len(top_values),
                    "posterior_constraint_score": max(0.0, min(1.0, constraint)),
                    "top_mode_or_range": f"{min(top_values):.8g}..{max(top_values):.8g}",
                    "top_mode_fraction": "",
                    "top_unique_values": len(set(top_values)),
                    "full_range": full_range,
                    "top_range": top_range,
                }
            )
        else:
            values = [str(row.get(parameter, "")).strip() for row in top_rows if str(row.get(parameter, "")).strip()]
            if not values:
                continue
            counts = Counter(values)
            mode, mode_count = counts.most_common(1)[0]
            mode_fraction = mode_count / len(values)
            output.append(
                {
                    "parameter": parameter,
                    "parameter_type": kind,
                    "top_n": len(values),
                    "posterior_constraint_score": mode_fraction,
                    "top_mode_or_range": mode,
                    "top_mode_fraction": mode_fraction,
                    "top_unique_values": len(counts),
                    "full_range": "",
                    "top_range": "",
                }
            )
    return output


def normalize(values: dict[str, float]) -> dict[str, float]:
    finite = [value for value in values.values() if math.isfinite(value)]
    if not finite:
        return {key: 0.0 for key in values}
    max_value = max(finite)
    if max_value <= 0:
        return {key: 0.0 for key in values}
    return {key: (value / max_value if math.isfinite(value) else 0.0) for key, value in values.items()}


def interpretation_for(parameter: str) -> str:
    lower = parameter.lower()
    if "liquid_family" in lower or "liquid_lambda" in lower:
        return "liquid excess Gibbs/Hmix shape and derivative; controls liquid activity asymmetry and liquidus curvature"
    if "sexcess" in lower or "entropy" in lower:
        return "liquid excess entropy/T-dependence; controls eutectic temperature and liquidus height"
    if "compound_label" in lower or "compound_x" in lower or "gform" in lower:
        return "intermediate branch/speciation thermodynamics; controls peritectic and high-UCl3 topology"
    if "tmax" in lower or "window" in lower or "cap" in lower:
        return "intermediate branch stability window; can create plateau/invariant features"
    if "dcp" in lower or "cp" in lower or "fusion" in lower:
        return "UCl3 pure/fusion/Cp anchor; controls UCl3-rich temperature dependence"
    if "hmix" in lower or "scale" in lower:
        return "liquid enthalpy magnitude; useful constraint but weak alone if shape is wrong"
    return "empirical search parameter; inspect ablation and posterior support before chemical interpretation"


def parameter_ranking(
    *,
    parameters: list[str],
    sensitivity: list[dict[str, Any]],
    ablation: list[dict[str, Any]],
    robustness: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[ParameterSummary]:
    sensitivity_by_param = {
        parameter: max((_float(row.get("effect")) for row in sensitivity if row["parameter"] == parameter), default=0.0) for parameter in parameters
    }
    ablation_by_param = {row["parameter"]: _float(row.get("delta_best_score")) for row in ablation}
    robustness_by_param = {row["parameter"]: _float(row.get("posterior_constraint_score")) for row in robustness}
    sens_norm = normalize(sensitivity_by_param)
    abl_norm = normalize(ablation_by_param)
    rob_norm = normalize(robustness_by_param)
    output: list[ParameterSummary] = []
    for parameter in parameters:
        composite = 0.45 * sens_norm.get(parameter, 0.0) + 0.35 * abl_norm.get(parameter, 0.0) + 0.20 * rob_norm.get(parameter, 0.0)
        output.append(
            ParameterSummary(
                name=parameter,
                feature_type=feature_type(rows, parameter),
                sensitivity_score=sens_norm.get(parameter, 0.0),
                ablation_score=abl_norm.get(parameter, 0.0),
                posterior_constraint_score=rob_norm.get(parameter, 0.0),
                composite_importance=composite,
                interpretation=interpretation_for(parameter),
            )
        )
    output.sort(key=lambda item: item.composite_importance, reverse=True)
    return output


def format_table(rows: list[dict[str, Any]], fields: list[str], *, max_rows: int = 12) -> str:
    if not rows:
        return "_No rows._"
    head = rows[:max_rows]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in head:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    search_csv: Path,
    score_column: str,
    targets: dict[str, float],
    ranking: list[ParameterSummary],
    ablation: list[dict[str, Any]],
    robustness: list[dict[str, Any]],
    n_rows: int,
    n_top: int,
) -> None:
    ranking_rows = [
        {
            "rank": idx,
            "parameter": item.name,
            "importance": item.composite_importance,
            "interpretation": item.interpretation,
        }
        for idx, item in enumerate(ranking, start=1)
    ]
    lines = [
        "# SHARP CALPHAD parameter attribution report",
        "",
        f"- Schema: `{SCHEMA}`",
        f"- Search CSV: `{search_csv}`",
        f"- Rows analyzed: `{n_rows}`",
        f"- Score column: `{score_column}` (lower is better)",
        f"- Top posterior subset size: `{n_top}`",
    ]
    if targets:
        lines.append("- Target anchors: " + ", ".join(f"`{key}={value}`" for key, value in targets.items()))
    lines.extend(
        [
            "",
            "## Ranking",
            "",
            format_table(ranking_rows, ["rank", "parameter", "importance", "interpretation"]),
            "",
            "## What SHARP Means Here",
            "",
            "- **S**ensitivity: correlation/ANOVA effect of each parameter on scores and benchmark residuals.",
            "- **H**ierarchy: composite ranking over sensitivity, ablation, and posterior constraint.",
            "- **A**blation: best-score degradation when a parameter group is changed.",
            "- **R**obustness: how tightly the top posterior subset constrains that parameter.",
            "- **P**hysical interpretation: map the statistical ranking back to liquid speciation, branch stability, and pure-component anchors.",
            "",
            "## Top Ablation Effects",
            "",
            format_table(ablation, ["parameter", "best_value", "best_group_min_score", "worst_value", "worst_group_min_score", "delta_best_score"]),
            "",
            "## Posterior Constraint",
            "",
            format_table(robustness, ["parameter", "posterior_constraint_score", "top_mode_or_range", "top_unique_values"]),
            "",
            "## Guardrail",
            "",
            "Use this as a ranking and model-form diagnostic, not as proof of a specific compound identity. If a high-ranking parameter has no defensible chemistry, the result is evidence that the model form or input database needs refinement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot_ranking(outdir: Path, ranking: list[ParameterSummary]) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not ranking:
        return None
    shown = ranking[:12]
    labels = [item.name for item in reversed(shown)]
    values = [item.composite_importance for item in reversed(shown)]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.barh(labels, values, color="#1f5a85")
    ax.set_xlabel("SHARP composite importance")
    ax.set_title("CALPHAD parameter attribution")
    ax.grid(True, axis="x", color="0.88")
    fig.tight_layout()
    path = outdir / "sharp_parameter_ranking.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def analyze_main(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = [dict(row) for row in read_csv(args.search_csv.resolve())]
    if not rows:
        raise ValueError(f"No rows found in {args.search_csv}")
    targets = dict(args.target or [])
    residual_metrics = add_target_residuals(rows, targets)
    metric_columns = parse_csv_arg(args.metric_column)
    if not metric_columns:
        metric_columns = [
            column
            for column in rows[0]
            if column == args.score_column or column.endswith("_rmse_k") or column.endswith("_rmse_kj_mol") or column.startswith("abs_")
        ]
    for residual in residual_metrics:
        if residual not in metric_columns:
            metric_columns.append(residual)
    if args.score_column not in metric_columns:
        metric_columns.append(args.score_column)
    metric_columns = [column for column in metric_columns if column in rows[0]]
    parameters = infer_parameter_columns(
        rows,
        explicit=parse_csv_arg(args.parameter_column),
        metric_columns=metric_columns,
        score_column=args.score_column,
        target_columns=list(targets),
    )
    sensitivity = sensitivity_rows(rows, parameters=parameters, metrics=metric_columns)
    ablation = ablation_rows(rows, parameters=parameters, score_column=args.score_column)
    top_rows = top_subset(rows, score_column=args.score_column, top_fraction=args.top_fraction, top_n=args.top_n)
    robustness = robustness_rows(rows, top_rows, parameters=parameters)
    ranking = parameter_ranking(parameters=parameters, sensitivity=sensitivity, ablation=ablation, robustness=robustness, rows=rows)

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(
        outdir / "sharp_sensitivity.csv",
        sensitivity,
        ["parameter", "parameter_type", "metric", "statistic", "effect", "signed_effect", "direction", "n", "n_groups"],
    )
    write_csv(
        outdir / "sharp_ablation.csv",
        ablation,
        [
            "parameter",
            "score_column",
            "best_value",
            "best_group_min_score",
            "best_group_mean_score",
            "best_group_n",
            "worst_value",
            "worst_group_min_score",
            "delta_best_score",
            "n_groups",
        ],
    )
    write_csv(
        outdir / "sharp_robustness.csv",
        robustness,
        [
            "parameter",
            "parameter_type",
            "top_n",
            "posterior_constraint_score",
            "top_mode_or_range",
            "top_mode_fraction",
            "top_unique_values",
            "full_range",
            "top_range",
        ],
    )
    ranking_rows = [
        {
            "rank": idx,
            "parameter": item.name,
            "parameter_type": item.feature_type,
            "composite_importance": item.composite_importance,
            "sensitivity_score": item.sensitivity_score,
            "ablation_score": item.ablation_score,
            "posterior_constraint_score": item.posterior_constraint_score,
            "interpretation": item.interpretation,
        }
        for idx, item in enumerate(ranking, start=1)
    ]
    write_csv(
        outdir / "sharp_parameter_ranking.csv",
        ranking_rows,
        [
            "rank",
            "parameter",
            "parameter_type",
            "composite_importance",
            "sensitivity_score",
            "ablation_score",
            "posterior_constraint_score",
            "interpretation",
        ],
    )
    report = outdir / "sharp_report.md"
    write_report(
        report,
        search_csv=args.search_csv.resolve(),
        score_column=args.score_column,
        targets=targets,
        ranking=ranking,
        ablation=ablation,
        robustness=robustness,
        n_rows=len(rows),
        n_top=len(top_rows),
    )
    plot = maybe_plot_ranking(outdir, ranking) if args.plot else None
    metadata = {
        "schema": SCHEMA,
        "search_csv": str(args.search_csv.resolve()),
        "n_rows": len(rows),
        "score_column": args.score_column,
        "metric_columns": metric_columns,
        "parameter_columns": parameters,
        "target_anchors": targets,
        "top_subset_n": len(top_rows),
        "outputs": {
            "ranking_csv": str(outdir / "sharp_parameter_ranking.csv"),
            "sensitivity_csv": str(outdir / "sharp_sensitivity.csv"),
            "ablation_csv": str(outdir / "sharp_ablation.csv"),
            "robustness_csv": str(outdir / "sharp_robustness.csv"),
            "report": str(report),
            "ranking_plot": plot,
        },
    }
    write_json(outdir / "sharp_metadata.json", metadata)
    print(f"Wrote SHARP ranking: {outdir / 'sharp_parameter_ranking.csv'}")
    print(f"Wrote SHARP report : {report}")
    if plot:
        print(f"Wrote SHARP plot   : {plot}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calphad-sharp",
        description="SHARP sensitivity/hierarchy/ablation/robustness/physics attribution for CALPHAD/MIVM searches.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Rank parameter impacts from a CALPHAD/MIVM search CSV.")
    analyze.add_argument("--search-csv", type=Path, required=True, help="Parameter-search CSV with score and metric columns.")
    analyze.add_argument("--outdir", type=Path, default=Path("sharp_analysis"))
    analyze.add_argument("--score-column", default="total_score", help="Lower-is-better score column.")
    analyze.add_argument("--parameter-column", action="append", help="Parameter column(s), repeatable or comma-separated. Defaults to inferred.")
    analyze.add_argument("--metric-column", action="append", help="Metric column(s), repeatable or comma-separated. Defaults to inferred.")
    analyze.add_argument("--target", type=parse_target, action="append", help="Benchmark target COLUMN=VALUE; adds absolute residual metrics.")
    analyze.add_argument("--top-fraction", type=float, default=0.10, help="Posterior subset fraction for robustness if --top-n is omitted.")
    analyze.add_argument("--top-n", type=int, help="Posterior subset size for robustness.")
    analyze.add_argument("--plot", action="store_true", help="Write a ranking PNG if matplotlib is available.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        return analyze_main(args)
    return None


if __name__ == "__main__":
    main()
