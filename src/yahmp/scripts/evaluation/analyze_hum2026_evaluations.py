"""Analyze paired Humanoids 2026 cross-evaluation results."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from scipy.stats import binomtest, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_ROOT = REPO_ROOT / "assets/logs/hum2026_cross_eval"

TRACKING_METRICS = (
  "error_anchor_pos",
  "error_anchor_rot",
  "error_anchor_lin_vel",
  "error_anchor_ang_vel",
  "error_body_pos",
  "error_body_rot",
  "error_body_lin_vel",
  "error_body_ang_vel",
  "error_joint_pos",
  "error_joint_vel",
)

TORQUE_METRICS = (
  "avg_abs_torque_all",
  "max_abs_torque_all",
  "avg_abs_torque_upper",
  "max_abs_torque_upper",
  "avg_abs_torque_lower",
  "max_abs_torque_lower",
)

CONTINUOUS_METRICS = ("completion_ratio", *TRACKING_METRICS, *TORQUE_METRICS)
ALL_METRICS = ("success", *CONTINUOUS_METRICS)

DISPLAY_NAMES = {
  "YAHMP": "Baseline",
  "YAHMP-NoHistory": "No history",
  "YAHMP-History20": "History 20",
  "YAHMP-QOnly": "Q only",
  "YAHMP-QOnly-NoHistory": "Q only, no history",
  "YAHMP-NoResidual": "No residual",
  "YAHMP-StiffPD": "StiffPD",
}

METRIC_LABELS = {
  "success": "Success rate",
  "completion_ratio": "Completion ratio",
  "error_anchor_pos": "Base position error",
  "error_anchor_rot": "Base orientation error",
  "error_anchor_lin_vel": "Base linear-velocity error",
  "error_anchor_ang_vel": "Base angular-velocity error",
  "error_body_pos": "Key-body position error",
  "error_body_rot": "Key-body orientation error",
  "error_body_lin_vel": "Key-body linear-velocity error",
  "error_body_ang_vel": "Key-body angular-velocity error",
  "error_joint_pos": "Joint-position error",
  "error_joint_vel": "Joint-velocity error",
  "avg_abs_torque_all": "Mean absolute torque",
  "max_abs_torque_all": "Maximum absolute torque",
  "avg_abs_torque_upper": "Mean upper-body torque",
  "max_abs_torque_upper": "Maximum upper-body torque",
  "avg_abs_torque_lower": "Mean lower-body torque",
  "max_abs_torque_lower": "Maximum lower-body torque",
}

HEADLINE_METRICS = (
  "success",
  "completion_ratio",
  "error_body_pos",
  "error_joint_pos",
  "avg_abs_torque_all",
  "max_abs_torque_all",
)


@dataclass(frozen=True)
class AnalyzeConfig:
  input_root: str = str(DEFAULT_INPUT_ROOT)
  output_dir: str | None = None
  baseline_policy: str = "YAHMP"
  alpha: float = 0.05
  practical_relative_change_pct: float = 5.0
  practical_success_change_pp: float = 2.0
  practical_completion_change: float = 0.02


@dataclass(frozen=True)
class PolicyData:
  key: str
  display_name: str
  csv_path: Path
  run_id: str
  task_id: str
  motion_folders: tuple[str, ...]
  rows: dict[str, dict[str, Any]]


def _parse_bool(value: str | bool | None) -> bool:
  if isinstance(value, bool):
    return value
  return str(value).strip().lower() in {"1", "true", "yes"}


def _parse_float(value: str | float | int | None) -> float:
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    return math.nan
  return parsed if math.isfinite(parsed) else math.nan


def _motion_key(row: dict[str, str]) -> str:
  motion_file = str(row.get("motion_file", "")).strip()
  if not motion_file:
    raise ValueError("Evaluation row is missing `motion_file`.")
  return Path(motion_file).name


def _load_policy(csv_path: Path, input_root: Path) -> PolicyData:
  with csv_path.open("r", encoding="utf-8", newline="") as file:
    raw_rows = list(csv.DictReader(file))
  if not raw_rows:
    raise ValueError(f"No rows found in {csv_path}.")

  required = {
    "motion_file",
    "success",
    "completion_ratio",
    "valid_for_success_eval",
    *TRACKING_METRICS,
    *TORQUE_METRICS,
  }
  missing = sorted(required - set(raw_rows[0]))
  if missing:
    raise ValueError(f"Missing columns in {csv_path}: {missing}")

  rows: dict[str, dict[str, Any]] = {}
  for raw in raw_rows:
    key = _motion_key(raw)
    if key in rows:
      raise ValueError(f"Duplicate motion `{key}` in {csv_path}.")
    parsed: dict[str, Any] = {
      "success": _parse_bool(raw["success"]),
      "valid": _parse_bool(raw["valid_for_success_eval"]),
    }
    for metric in CONTINUOUS_METRICS:
      parsed[metric] = _parse_float(raw[metric])
    rows[key] = parsed

  relative = csv_path.relative_to(input_root)
  policy_key = relative.parts[0]
  run_id = relative.parts[1] if len(relative.parts) > 1 else csv_path.parent.name
  metadata_path = csv_path.parent / "wandb_evaluation_metadata.json"
  metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
  policy_meta = metadata.get("policy", {})
  source_meta = metadata.get("motion_source", {})
  folders = source_meta.get("folders", ()) if isinstance(source_meta, dict) else ()
  return PolicyData(
    key=policy_key,
    display_name=DISPLAY_NAMES.get(policy_key, policy_key),
    csv_path=csv_path,
    run_id=str(metadata.get("wandb", {}).get("run_id", run_id)),
    task_id=str(policy_meta.get("task_id", policy_key)),
    motion_folders=tuple(str(folder) for folder in folders),
    rows=rows,
  )


def discover_policies(input_root: Path) -> dict[str, PolicyData]:
  csv_paths = sorted(input_root.glob("*/*/per_motion_success.csv"))
  if not csv_paths:
    raise FileNotFoundError(f"No per-motion evaluation CSVs found under {input_root}.")

  by_key: dict[str, list[Path]] = defaultdict(list)
  for csv_path in csv_paths:
    by_key[csv_path.relative_to(input_root).parts[0]].append(csv_path)
  duplicates = {key: paths for key, paths in by_key.items() if len(paths) > 1}
  if duplicates:
    formatted = "\n".join(
      f"  {key}: {[str(path) for path in paths]}" for key, paths in duplicates.items()
    )
    raise ValueError(
      "Multiple runs found for the same policy. Select a clean input root before "
      f"analysis:\n{formatted}"
    )
  return {key: _load_policy(paths[0], input_root) for key, paths in by_key.items()}


def _finite(values: list[float]) -> np.ndarray:
  array = np.asarray(values, dtype=np.float64)
  return array[np.isfinite(array)]


def _describe(values: np.ndarray) -> dict[str, float | int]:
  values = values[np.isfinite(values)]
  if values.size == 0:
    return {
      "n": 0,
      "mean": math.nan,
      "std": math.nan,
      "median": math.nan,
      "q25": math.nan,
      "q75": math.nan,
    }
  return {
    "n": int(values.size),
    "mean": float(values.mean()),
    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    "median": float(np.median(values)),
    "q25": float(np.quantile(values, 0.25)),
    "q75": float(np.quantile(values, 0.75)),
  }


def policy_summary(policy: PolicyData) -> list[dict[str, Any]]:
  valid_rows = [row for row in policy.rows.values() if row["valid"]]
  successful_rows = [row for row in valid_rows if row["success"]]
  result: list[dict[str, Any]] = []

  success_values = np.asarray(
    [float(row["success"]) for row in valid_rows], dtype=np.float64
  )
  result.append(
    {
      "policy": policy.key,
      "policy_display": policy.display_name,
      "run_id": policy.run_id,
      "metric": "success",
      "metric_label": METRIC_LABELS["success"],
      "subset": "valid motions",
      **_describe(success_values),
    }
  )
  completion = _finite([row["completion_ratio"] for row in valid_rows])
  result.append(
    {
      "policy": policy.key,
      "policy_display": policy.display_name,
      "run_id": policy.run_id,
      "metric": "completion_ratio",
      "metric_label": METRIC_LABELS["completion_ratio"],
      "subset": "valid motions",
      **_describe(completion),
    }
  )
  for metric in (*TRACKING_METRICS, *TORQUE_METRICS):
    values = _finite([row[metric] for row in successful_rows])
    result.append(
      {
        "policy": policy.key,
        "policy_display": policy.display_name,
        "run_id": policy.run_id,
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "subset": "successful motions",
        **_describe(values),
      }
    )
  return result


def _wilcoxon_pvalue(differences: np.ndarray) -> float:
  differences = differences[np.isfinite(differences)]
  if differences.size == 0 or np.all(np.abs(differences) <= 1.0e-12):
    return 1.0
  return float(
    wilcoxon(
      differences,
      alternative="two-sided",
      zero_method="wilcox",
      method="auto",
    ).pvalue
  )


def _paired_values(
  reference: PolicyData,
  variant: PolicyData,
  metric: str,
) -> tuple[np.ndarray, np.ndarray, str]:
  shared = sorted(set(reference.rows) & set(variant.rows))
  if metric in ("success", "completion_ratio"):
    shared = [
      key
      for key in shared
      if reference.rows[key]["valid"] and variant.rows[key]["valid"]
    ]
    subset = "common valid motions"
  else:
    shared = [
      key
      for key in shared
      if reference.rows[key]["valid"]
      and variant.rows[key]["valid"]
      and reference.rows[key]["success"]
      and variant.rows[key]["success"]
    ]
    subset = "common successful motions"

  reference_values = np.asarray(
    [float(reference.rows[key][metric]) for key in shared], dtype=np.float64
  )
  variant_values = np.asarray(
    [float(variant.rows[key][metric]) for key in shared], dtype=np.float64
  )
  finite = np.isfinite(reference_values) & np.isfinite(variant_values)
  return reference_values[finite], variant_values[finite], subset


def compare_pair(
  reference: PolicyData,
  variant: PolicyData,
  *,
  cfg: AnalyzeConfig,
) -> list[dict[str, Any]]:
  results: list[dict[str, Any]] = []
  for metric in ALL_METRICS:
    reference_values, variant_values, subset = _paired_values(
      reference, variant, metric
    )
    differences = variant_values - reference_values

    discordant_reference_only = 0
    discordant_variant_only = 0
    if metric == "success":
      reference_bool = reference_values.astype(bool)
      variant_bool = variant_values.astype(bool)
      discordant_reference_only = int(np.sum(reference_bool & ~variant_bool))
      discordant_variant_only = int(np.sum(~reference_bool & variant_bool))
      discordant = discordant_reference_only + discordant_variant_only
      p_value = (
        float(
          binomtest(
            min(discordant_reference_only, discordant_variant_only),
            discordant,
            0.5,
            alternative="two-sided",
          ).pvalue
        )
        if discordant > 0
        else 1.0
      )
      test_name = "exact McNemar"
    else:
      p_value = _wilcoxon_pvalue(differences)
      test_name = "paired Wilcoxon"

    reference_mean = (
      float(reference_values.mean()) if reference_values.size else math.nan
    )
    variant_mean = float(variant_values.mean()) if variant_values.size else math.nan
    delta_mean = float(differences.mean()) if differences.size else math.nan
    delta_std = (
      float(differences.std(ddof=1))
      if differences.size > 1
      else 0.0
      if differences.size == 1
      else math.nan
    )
    delta_median = float(np.median(differences)) if differences.size else math.nan
    higher_is_better = metric in {"success", "completion_ratio"}
    improvement = delta_mean if higher_is_better else -delta_mean
    relative_improvement = (
      100.0 * improvement / abs(reference_mean)
      if math.isfinite(reference_mean) and abs(reference_mean) > 1.0e-12
      else math.nan
    )
    improvement_per_motion = differences if higher_is_better else -differences
    wins = int(np.sum(improvement_per_motion > 1.0e-12))
    losses = int(np.sum(improvement_per_motion < -1.0e-12))
    ties = int(differences.size - wins - losses)
    denominator = int(differences.size)

    results.append(
      {
        "comparison": f"{reference.key} vs {variant.key}",
        "reference": reference.key,
        "reference_display": reference.display_name,
        "variant": variant.key,
        "variant_display": variant.display_name,
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "metric_group": (
          "outcome"
          if metric in {"success", "completion_ratio"}
          else "tracking"
          if metric in TRACKING_METRICS
          else "torque"
        ),
        "subset": subset,
        "n_pairs": int(differences.size),
        "reference_mean": reference_mean,
        "reference_median": (
          float(np.median(reference_values)) if reference_values.size else math.nan
        ),
        "variant_mean": variant_mean,
        "variant_median": (
          float(np.median(variant_values)) if variant_values.size else math.nan
        ),
        "delta_variant_minus_reference": delta_mean,
        "delta_std": delta_std,
        "delta_median": delta_median,
        "improvement": improvement,
        "relative_improvement_pct": relative_improvement,
        "variant_wins": wins,
        "reference_wins": losses,
        "ties": ties,
        "variant_win_rate": wins / denominator if denominator else math.nan,
        "reference_win_rate": losses / denominator if denominator else math.nan,
        "tie_rate": ties / denominator if denominator else math.nan,
        "test": test_name,
        "p_value": p_value,
        "p_holm_within_comparison": math.nan,
        "p_holm_global": math.nan,
        "reference_only_successes": discordant_reference_only,
        "variant_only_successes": discordant_variant_only,
        "result_class": "pending",
      }
    )
  return results


def _holm_adjust(p_values: list[float]) -> list[float]:
  adjusted = [math.nan] * len(p_values)
  finite_indices = [idx for idx, value in enumerate(p_values) if math.isfinite(value)]
  ordered = sorted(finite_indices, key=lambda idx: p_values[idx])
  previous = 0.0
  total = len(ordered)
  for rank, index in enumerate(ordered):
    value = min(max(previous, (total - rank) * p_values[index]), 1.0)
    adjusted[index] = value
    previous = value
  return adjusted


def _apply_multiple_testing(results: list[dict[str, Any]]) -> None:
  global_adjusted = _holm_adjust([float(row["p_value"]) for row in results])
  for row, adjusted in zip(results, global_adjusted, strict=True):
    row["p_holm_global"] = adjusted

  grouped: dict[str, list[int]] = defaultdict(list)
  for index, row in enumerate(results):
    grouped[str(row["comparison"])].append(index)
  for indices in grouped.values():
    adjusted = _holm_adjust([float(results[index]["p_value"]) for index in indices])
    for index, value in zip(indices, adjusted, strict=True):
      results[index]["p_holm_within_comparison"] = value


def _classify_result(row: dict[str, Any], cfg: AnalyzeConfig) -> str:
  significant = float(row["p_holm_within_comparison"]) < cfg.alpha
  metric = str(row["metric"])
  if metric == "success":
    practical = abs(float(row["improvement"])) >= cfg.practical_success_change_pp / 100
  elif metric == "completion_ratio":
    practical = abs(float(row["improvement"])) >= cfg.practical_completion_change
  else:
    practical = (
      abs(float(row["relative_improvement_pct"])) >= cfg.practical_relative_change_pct
    )

  if significant and practical:
    return (
      "meaningful improvement"
      if float(row["improvement"]) > 0.0
      else "meaningful degradation"
    )
  if significant:
    return (
      "small improvement" if float(row["improvement"]) > 0.0 else "small degradation"
    )
  return "inconclusive"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    return
  with path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _fmt_p_value(value: float) -> str:
  if not math.isfinite(value):
    return "n/a"
  if value < 0.001:
    return f"{value:.2e}"
  return f"{value:.3f}"


def _headline_table(
  baseline_key: str, results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
  lookup = {
    (str(row["reference"]), str(row["variant"]), str(row["metric"])): row
    for row in results
  }
  variants = sorted(
    {str(row["variant"]) for row in results if row["reference"] == baseline_key}
  )
  rows = []
  for variant in variants:
    metric_rows = {
      metric: lookup[(baseline_key, variant, metric)] for metric in HEADLINE_METRICS
    }
    report_rows = [
      row
      for row in results
      if row["reference"] == baseline_key
      and row["variant"] == variant
      and row["metric"] in ALL_METRICS
    ]
    rows.append(
      {
        "variant": variant,
        "variant_display": metric_rows["success"]["variant_display"],
        "success_delta_pp": 100.0 * metric_rows["success"]["improvement"],
        "success_p_holm": metric_rows["success"]["p_holm_within_comparison"],
        "completion_delta_pp": 100.0 * metric_rows["completion_ratio"]["improvement"],
        "body_pos_improvement_pct": metric_rows["error_body_pos"][
          "relative_improvement_pct"
        ],
        "body_pos_win_rate": metric_rows["error_body_pos"]["variant_win_rate"],
        "joint_pos_improvement_pct": metric_rows["error_joint_pos"][
          "relative_improvement_pct"
        ],
        "joint_pos_win_rate": metric_rows["error_joint_pos"]["variant_win_rate"],
        "mean_torque_improvement_pct": metric_rows["avg_abs_torque_all"][
          "relative_improvement_pct"
        ],
        "mean_torque_win_rate": metric_rows["avg_abs_torque_all"]["variant_win_rate"],
        "max_torque_improvement_pct": metric_rows["max_abs_torque_all"][
          "relative_improvement_pct"
        ],
        "max_torque_win_rate": metric_rows["max_abs_torque_all"]["variant_win_rate"],
        "meaningful_improvements": sum(
          row["result_class"] == "meaningful improvement" for row in report_rows
        ),
        "meaningful_degradations": sum(
          row["result_class"] == "meaningful degradation" for row in report_rows
        ),
      }
    )
  return rows


def _report_markdown(
  cfg: AnalyzeConfig,
  policies: dict[str, PolicyData],
  summaries: list[dict[str, Any]],
  tests: list[dict[str, Any]],
  headlines: list[dict[str, Any]],
) -> str:
  summary_lookup = {(str(row["policy"]), str(row["metric"])): row for row in summaries}
  lines = [
    "# Humanoids 2026 Evaluation Analysis",
    "",
    "## Protocol",
    "",
    f"- Baseline: `{cfg.baseline_policy}`",
    f"- Significance level: {cfg.alpha}",
    "- Success uses the exact McNemar test on common valid motions.",
    "- Continuous metrics use paired Wilcoxon tests.",
    "- Win rates are the observed fraction of paired motions improved by the variant.",
    "- Tracking and torque comparisons use motions completed by both policies.",
    "- P-values are Holm-corrected within each policy comparison and globally.",
    "- Positive changes below indicate improvement; tracking errors and torques are inverted accordingly.",
  ]
  motion_folders = sorted(
    {folder for policy in policies.values() for folder in policy.motion_folders}
  )
  if motion_folders:
    lines.append(
      f"- Motion source(s): {', '.join(f'`{path}`' for path in motion_folders)}"
    )
  lines.extend(
    [
      "",
      "## Policy Overview",
      "",
      "| Policy | Run | Valid | Success | Completion |",
      "|---|---:|---:|---:|---:|",
    ]
  )
  for key in sorted(policies):
    policy = policies[key]
    success = summary_lookup[(key, "success")]
    completion = summary_lookup[(key, "completion_ratio")]
    lines.append(
      f"| {policy.display_name} | `{policy.run_id}` | {success['n']} | "
      f"{100.0 * float(success['mean']):.1f}% | "
      f"{100.0 * float(completion['mean']):.1f}% |"
    )

  success_rates = [float(summary_lookup[(key, "success")]["mean"]) for key in policies]
  if success_rates and max(success_rates) - min(success_rates) <= 1.0e-12:
    lines.extend(
      [
        "",
        "> Success rate is saturated under the current criterion; it does not "
        "separate these policies. Tracking and torque metrics carry the ablation signal.",
      ]
    )

  lines.extend(
    [
      "",
      "## Headline Changes vs Baseline",
      "",
      "| Variant | Success (pp) | Completion (pp) | Body pos. change / win | Joint pos. change / win | Mean torque change / win | Max torque change / win | Meaningful + / - |",
      "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
  )
  for row in headlines:
    lines.append(
      f"| {row['variant_display']} | {float(row['success_delta_pp']):+.1f} | "
      f"{float(row['completion_delta_pp']):+.1f} | "
      f"{float(row['body_pos_improvement_pct']):+.1f}% / "
      f"{100.0 * float(row['body_pos_win_rate']):.1f}% | "
      f"{float(row['joint_pos_improvement_pct']):+.1f}% / "
      f"{100.0 * float(row['joint_pos_win_rate']):.1f}% | "
      f"{float(row['mean_torque_improvement_pct']):+.1f}% / "
      f"{100.0 * float(row['mean_torque_win_rate']):.1f}% | "
      f"{float(row['max_torque_improvement_pct']):+.1f}% / "
      f"{100.0 * float(row['max_torque_win_rate']):.1f}% | "
      f"{row['meaningful_improvements']} / {row['meaningful_degradations']} |"
    )

  lines.extend(["", "## All Metrics vs Baseline", ""])
  metric_order = {metric: index for index, metric in enumerate(ALL_METRICS)}
  baseline_comparisons = [
    row for row in tests if row["reference"] == cfg.baseline_policy
  ]
  for variant in sorted({str(row["variant"]) for row in baseline_comparisons}):
    variant_rows = sorted(
      (row for row in baseline_comparisons if row["variant"] == variant),
      key=lambda row: metric_order[str(row["metric"])],
    )
    lines.extend(
      [
        f"### {variant_rows[0]['variant_display']}",
        "",
        "| Metric | Baseline mean | Variant mean | Change | Variant wins | Holm p | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
      ]
    )
    for row in variant_rows:
      outcome_metric = row["metric"] in {"success", "completion_ratio"}
      change = (
        100.0 * float(row["improvement"])
        if outcome_metric
        else float(row["relative_improvement_pct"])
      )
      change_unit = "pp" if outcome_metric else "%"
      lines.append(
        f"| {row['metric_label']} | {float(row['reference_mean']):.4f} | "
        f"{float(row['variant_mean']):.4f} | {change:+.2f} {change_unit} | "
        f"{100.0 * float(row['variant_win_rate']):.1f}% | "
        f"{_fmt_p_value(float(row['p_holm_within_comparison']))} | "
        f"{row['result_class']} |"
      )
    lines.append("")

  lines.extend(["", "## Potentially Paper-Worthy Findings", ""])
  baseline_tests = [
    row
    for row in tests
    if row["reference"] == cfg.baseline_policy
    and str(row["result_class"]).startswith("meaningful")
  ]
  if baseline_tests:
    for row in baseline_tests:
      change = (
        100.0 * float(row["improvement"])
        if row["metric"] in {"success", "completion_ratio"}
        else float(row["relative_improvement_pct"])
      )
      unit = "pp" if row["metric"] in {"success", "completion_ratio"} else "%"
      lines.append(
        f"- **{row['variant_display']} / {row['metric_label']}:** "
        f"{change:+.2f} {unit} ({row['result_class']}, "
        f"Holm p={_fmt_p_value(float(row['p_holm_within_comparison']))}, "
        f"n={row['n_pairs']})."
      )
  else:
    lines.append(
      "- No headline metric passed both statistical and practical thresholds."
    )

  lines.extend(["", "## Interaction Checks", ""])
  interaction_tests = [
    row
    for row in tests
    if row["reference"] != cfg.baseline_policy
    and str(row["result_class"]).startswith("meaningful")
  ]
  if interaction_tests:
    for row in interaction_tests:
      change = (
        100.0 * float(row["improvement"])
        if row["metric"] in {"success", "completion_ratio"}
        else float(row["relative_improvement_pct"])
      )
      unit = "pp" if row["metric"] in {"success", "completion_ratio"} else "%"
      lines.append(
        f"- **{row['reference_display']} -> {row['variant_display']} / "
        f"{row['metric_label']}:** {change:+.2f} {unit} "
        f"({row['result_class']}, Holm p="
        f"{_fmt_p_value(float(row['p_holm_within_comparison']))})."
      )
  else:
    lines.append("- No interaction comparison passed the practical threshold.")

  weak_ablation_lines = []
  for headline in headlines:
    if (
      int(headline["meaningful_improvements"]) == 0
      and int(headline["meaningful_degradations"]) == 0
    ):
      weak_ablation_lines.append(
        f"- **{headline['variant_display']}** has no meaningful change among the "
        "headline metrics under the current thresholds."
      )
  if weak_ablation_lines:
    lines.extend(["", "## Weak or Inconclusive Ablations", "", *weak_ablation_lines])

  lines.extend(
    [
      "",
      "## Interpretation Notes",
      "",
      "- Statistical significance alone is not treated as paper-worthy; a practical effect threshold is also required.",
      "- A lower tracking error or torque is reported as a positive improvement percentage.",
      "- Tracking/torque results are conditional on both policies succeeding and should be read together with success rate.",
      "- Motion-level tests treat clips as paired units; clips derived from the same source sequence may not be fully independent.",
      "- These results reflect one trained checkpoint per variant. Training-seed uncertainty is not measured here.",
      "",
      "Full results are available in `pairwise_tests.csv` and `policy_summary.csv`.",
    ]
  )
  return "\n".join(lines) + "\n"


def run(cfg: AnalyzeConfig) -> dict[str, str]:
  input_root = Path(cfg.input_root).expanduser().resolve()
  output_dir = (
    Path(cfg.output_dir).expanduser().resolve()
    if cfg.output_dir is not None
    else input_root / "analysis"
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  policies = discover_policies(input_root)
  if cfg.baseline_policy not in policies:
    raise KeyError(
      f"Baseline `{cfg.baseline_policy}` not found. Available: {sorted(policies)}"
    )
  print(f"[INFO] Input root: {input_root}")
  print(f"[INFO] Policies: {sorted(policies)}")
  print(f"[INFO] Output directory: {output_dir}")

  summaries = [row for policy in policies.values() for row in policy_summary(policy)]
  baseline = policies[cfg.baseline_policy]
  comparisons: list[tuple[PolicyData, PolicyData]] = [
    (baseline, policies[key]) for key in sorted(policies) if key != baseline.key
  ]
  interaction_pairs = (
    ("YAHMP-NoHistory", "YAHMP-QOnly-NoHistory"),
    ("YAHMP-QOnly", "YAHMP-QOnly-NoHistory"),
  )
  for reference_key, variant_key in interaction_pairs:
    if reference_key in policies and variant_key in policies:
      comparisons.append((policies[reference_key], policies[variant_key]))

  tests: list[dict[str, Any]] = []
  for reference, variant in comparisons:
    tests.extend(compare_pair(reference, variant, cfg=cfg))
  _apply_multiple_testing(tests)
  for row in tests:
    row["result_class"] = _classify_result(row, cfg)

  headlines = _headline_table(cfg.baseline_policy, tests)
  summary_csv = output_dir / "policy_summary.csv"
  tests_csv = output_dir / "pairwise_tests.csv"
  headline_csv = output_dir / "headline_comparisons.csv"
  report_path = output_dir / "report.md"
  json_path = output_dir / "analysis.json"
  _write_csv(summary_csv, summaries)
  _write_csv(tests_csv, tests)
  _write_csv(headline_csv, headlines)
  report_path.write_text(
    _report_markdown(cfg, policies, summaries, tests, headlines), encoding="utf-8"
  )
  json_path.write_text(
    json.dumps(
      {
        "config": asdict(cfg),
        "policies": {
          key: {
            "display_name": policy.display_name,
            "run_id": policy.run_id,
            "task_id": policy.task_id,
            "csv_path": str(policy.csv_path),
            "motion_folders": list(policy.motion_folders),
          }
          for key, policy in policies.items()
        },
        "policy_summary": summaries,
        "pairwise_tests": tests,
        "headline_comparisons": headlines,
      },
      indent=2,
      allow_nan=True,
    ),
    encoding="utf-8",
  )
  return {
    "report": str(report_path),
    "policy_summary": str(summary_csv),
    "pairwise_tests": str(tests_csv),
    "headline_comparisons": str(headline_csv),
    "analysis_json": str(json_path),
  }


def main() -> None:
  cfg = tyro.cli(AnalyzeConfig)
  print(json.dumps(run(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
