"""Plot Humanoids 2026 tracking ablation boxplots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/yahmp_matplotlib"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_ROOT = REPO_ROOT / "assets/logs/hum2026_cross_eval"
DEFAULT_ANALYSIS_JSON = DEFAULT_INPUT_ROOT / "analysis/analysis.json"
DEFAULT_OUTPUT = (
  REPO_ROOT
  / "paper/Humanoids2026_YAHMP/Figures/hum2026-tracking-ablation-boxplots.pdf"
)

POLICIES = (
  ("YAHMP", "YAHMP", "m9b6wla7"),
  ("YAHMP-NoResidual", "No res.", "vgkib246"),
  ("YAHMP-StiffPD", "Stiff-PD", "n82dzqva"),
  ("YAHMP-NoHistory", "No hist.", "ecv01j9q"),
  ("YAHMP-History20", "Hist-20", "s3dqxo63"),
  ("YAHMP-QOnly", "Pos-ref", "6j0rl3lu"),
  ("TWIST2", "TWIST2", "retrained"),
)

METRICS = (
  ("error_anchor_pos", "Base pos.", "[m]"),
  ("error_anchor_rot", "Base rot.", "[rad]"),
  ("error_body_pos", "Key-body pos.", "[m]"),
  ("error_body_rot", "Key-body rot.", "[rad]"),
  ("error_joint_pos", "Joint pos.", "[rad]"),
  ("error_joint_vel", "Joint vel.", "[rad/s]"),
)

PALETTE = {
  "YAHMP": "#4D4D4D",
  "YAHMP-NoResidual": "#1B9E77",
  "YAHMP-StiffPD": "#D95F02",
  "YAHMP-NoHistory": "#7570B3",
  "YAHMP-History20": "#E7298A",
  "YAHMP-QOnly": "#66A61E",
  "TWIST2": "#A6761D",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-root",
    type=Path,
    default=DEFAULT_INPUT_ROOT,
    help="Root containing per-policy cross-evaluation folders.",
  )
  parser.add_argument(
    "--analysis-json",
    type=Path,
    default=DEFAULT_ANALYSIS_JSON,
    help="Analysis JSON with Holm-corrected paired-test p-values.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=DEFAULT_OUTPUT,
    help="Output figure path. PNG/PDF siblings are both written.",
  )
  return parser.parse_args()


def _parse_bool(value: str | None) -> bool:
  return str(value).strip().lower() in {"1", "true", "yes"}


def _parse_float(value: str | None) -> float:
  try:
    parsed = float(value) if value is not None else math.nan
  except ValueError:
    return math.nan
  return parsed if math.isfinite(parsed) else math.nan


def _csv_path(input_root: Path, policy_key: str, run_id: str) -> Path:
  return input_root / policy_key / run_id / "per_motion_success.csv"


def _load_metric_values(csv_path: Path, metric: str) -> np.ndarray:
  if not csv_path.is_file():
    raise FileNotFoundError(f"Missing evaluation CSV: {csv_path}")
  values: list[float] = []
  with csv_path.open("r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    required = {"valid_for_success_eval", "success", metric}
    missing = required - set(reader.fieldnames or ())
    if missing:
      raise KeyError(f"Missing columns in {csv_path}: {sorted(missing)}")
    for row in reader:
      if not _parse_bool(row["valid_for_success_eval"]):
        continue
      if not _parse_bool(row["success"]):
        continue
      value = _parse_float(row[metric])
      if math.isfinite(value):
        values.append(value)
  return np.asarray(values, dtype=np.float64)


def _load_p_values(path: Path) -> dict[tuple[str, str], float]:
  if not path.is_file():
    raise FileNotFoundError(f"Missing analysis JSON: {path}")
  payload = json.loads(path.read_text(encoding="utf-8"))
  p_values: dict[tuple[str, str], float] = {}
  for row in payload.get("pairwise_tests", ()):
    if row.get("reference") != "YAHMP":
      continue
    variant = str(row.get("variant", ""))
    metric = str(row.get("metric", ""))
    p_values[(variant, metric)] = _parse_float(row.get("p_holm_within_comparison"))
  return p_values


def _sig_label(p_value: float) -> str:
  if not math.isfinite(p_value):
    return ""
  if p_value < 1.0e-4:
    return "****"
  if p_value < 1.0e-3:
    return "***"
  if p_value < 1.0e-2:
    return "**"
  if p_value < 5.0e-2:
    return "*"
  return ""


def _style_axis(axis: Any) -> None:
  axis.spines["top"].set_visible(False)
  axis.spines["right"].set_visible(False)
  axis.spines["left"].set_color("#B0B0B0")
  axis.spines["bottom"].set_color("#B0B0B0")
  axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
  axis.set_axisbelow(True)
  axis.tick_params(axis="x", labelsize=5.5, length=0)
  axis.tick_params(axis="y", labelsize=5.5, length=0)


def _boxplot_whisker_bounds(
  values_by_policy: list[np.ndarray],
) -> tuple[float, float, list[float]]:
  lows: list[float] = []
  highs: list[float] = []
  high_by_policy: list[float] = []
  for values in values_by_policy:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
      high_by_policy.append(math.nan)
      continue
    q1, q3 = np.quantile(finite, (0.25, 0.75))
    iqr = q3 - q1
    low_limit = q1 - 1.5 * iqr
    high_limit = q3 + 1.5 * iqr
    inlier_values = finite[(finite >= low_limit) & (finite <= high_limit)]
    if inlier_values.size == 0:
      inlier_values = finite
    lows.append(float(np.min(inlier_values)))
    high = float(np.max(inlier_values))
    highs.append(high)
    high_by_policy.append(high)
  if not lows or not highs:
    return 0.0, 1.0, [math.nan] * len(values_by_policy)
  return min(lows), max(highs), high_by_policy


def _add_significance(
  *,
  axis: Any,
  x_position: float,
  y_position: float,
  p_value: float,
) -> None:
  label = _sig_label(p_value)
  if not label:
    return
  axis.text(
    x_position,
    y_position,
    label,
    ha="center",
    va="bottom",
    fontsize=5.5,
    color="#222222",
  )


def _plot_metric(
  *,
  axis: Any,
  input_root: Path,
  p_values: dict[tuple[str, str], float],
  metric: str,
  title: str,
  unit: str,
) -> None:
  labels: list[str] = []
  values_by_policy: list[np.ndarray] = []
  colors: list[str] = []
  for policy_key, display_name, run_id in POLICIES:
    labels.append(display_name)
    values = _load_metric_values(_csv_path(input_root, policy_key, run_id), metric)
    values_by_policy.append(values)
    colors.append(PALETTE[policy_key])

  positions = np.arange(1, len(POLICIES) + 1)
  rng = np.random.default_rng(7)
  yahmp_median = float(np.median(values_by_policy[0]))
  axis.axhline(
    yahmp_median,
    color="#4D4D4D",
    linewidth=0.8,
    linestyle=":",
    alpha=0.85,
    zorder=0,
  )
  for x_position, values, color in zip(positions, values_by_policy, colors, strict=True):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
      continue
    if finite.size > 350:
      finite = rng.choice(finite, size=350, replace=False)
    jitter = rng.uniform(-0.18, 0.18, size=finite.size)
    axis.scatter(
      np.full(finite.size, x_position, dtype=np.float64) + jitter,
      finite,
      s=4.0,
      color=color,
      alpha=0.12,
      linewidths=0.0,
      rasterized=True,
      zorder=1,
    )
  y_min, y_max, whisker_highs = _boxplot_whisker_bounds(values_by_policy)
  y_min = max(0.0, y_min)
  y_span = max(y_max - y_min, 1.0e-6)
  star_positions: dict[str, float] = {}
  for policy_key, whisker_high in zip(
    (policy[0] for policy in POLICIES[1:]), whisker_highs[1:], strict=True
  ):
    label = _sig_label(p_values.get((policy_key, metric), math.nan))
    if label and math.isfinite(whisker_high):
      star_positions[policy_key] = whisker_high + 0.008 * y_span
  y_top = max([y_max, *star_positions.values()]) + 0.04 * y_span
  axis.set_ylim(y_min - 0.005 * y_span, y_top)
  box = axis.boxplot(
    values_by_policy,
    positions=positions,
    widths=0.55,
    patch_artist=True,
    showfliers=False,
    medianprops={"linewidth": 1.0, "color": "#222222"},
    whiskerprops={"linewidth": 0.8, "color": "#555555"},
    capprops={"linewidth": 0.8, "color": "#555555"},
    boxprops={"linewidth": 0.8},
  )
  for patch, color in zip(box["boxes"], colors, strict=True):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
    patch.set_alpha(0.62)

  axis.set_title(f"{title} {unit}", fontsize=5.8, pad=6.0)
  axis.set_xticks(positions, labels, rotation=35, ha="right")
  axis.set_xlim(0.4, len(POLICIES) + 0.6)
  _style_axis(axis)
  for x_position, (policy_key, _, _) in zip(
    positions[1:], POLICIES[1:], strict=True
  ):
    _add_significance(
      axis=axis,
      x_position=float(x_position),
      y_position=float(star_positions.get(policy_key, y_max + 0.008 * y_span)),
      p_value=p_values.get((policy_key, metric), math.nan),
    )


def save_figure(fig: Any, output: Path) -> tuple[Path, Path]:
  output = output.expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  pdf_path = output.with_suffix(".pdf")
  png_path = output.with_suffix(".png")
  fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
  fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
  return pdf_path, png_path


def main() -> None:
  import matplotlib.pyplot as plt

  args = parse_args()
  input_root = args.input_root.expanduser().resolve()
  p_values = _load_p_values(args.analysis_json.expanduser().resolve())

  figure, axes = plt.subplots(2, 3, figsize=(7.25, 3.0), constrained_layout=True)
  figure.set_constrained_layout_pads(w_pad=0.015, h_pad=0.01, wspace=0.01, hspace=0.02)
  for axis, (metric, title, unit) in zip(axes.flat, METRICS, strict=True):
    _plot_metric(
      axis=axis,
      input_root=input_root,
      p_values=p_values,
      metric=metric,
      title=title,
      unit=unit,
    )
  pdf_path, png_path = save_figure(figure, args.output)
  plt.close(figure)
  print(f"Saved plot: {pdf_path}")
  print(f"Saved plot: {png_path}")


if __name__ == "__main__":
  main()
