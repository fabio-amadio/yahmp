"""Generate minimalist box plots comparing per-motion tracking metrics."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yahmp_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

DEFAULT_TWIST2_CSV = Path(
  "assets/logs/twist2_onnx_success_eval_w_torques/per_motion_success.csv"
)
DEFAULT_YAHMP_CSV = Path("assets/logs/yahmp_onnx_success_eval/per_motion_success.csv")
DEFAULT_OUTPUT = Path("assets/logs/plots/tracking_metrics_twist2_vs_yahmp_boxplot.png")

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

METRIC_LABELS = {
  "error_anchor_pos": "Base Pos",
  "error_anchor_rot": "Base Rot",
  "error_anchor_lin_vel": "Base Lin Vel",
  "error_anchor_ang_vel": "Base Ang Vel",
  "error_body_pos": "Body Pos",
  "error_body_rot": "Body Rot",
  "error_body_lin_vel": "Body Lin Vel",
  "error_body_ang_vel": "Body Ang Vel",
  "error_joint_pos": "Joint Pos",
  "error_joint_vel": "Joint Vel",
}

COLOR_PRESETS = {
  "blue": "#1F77B4",
  "red": "#D62728",
  "green": "#2CA02C",
  "orange": "#FF7F0E",
  "purple": "#9467BD",
  "lime": "#BCBD22",
  "cyan": "#17BECF",
  "gray": "#7F7F7F",
  "light_blue": "#AEC7E8",
  "salmon": "#FF9896",
  "light_green": "#98DF8A",
  "teal": "#4DB6AC",
}

plt: Any | None = None
wilcoxon = None


def _load_plot_deps() -> tuple[Any, Any]:
  global plt, wilcoxon
  if plt is None:
    import matplotlib.pyplot as _plt

    plt = _plt
  if wilcoxon is None:
    from scipy.stats import wilcoxon as _wilcoxon

    wilcoxon = _wilcoxon
  return plt, wilcoxon


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--twist2-csv",
    type=Path,
    default=DEFAULT_TWIST2_CSV,
    help="Path to the TWIST2 per-motion CSV.",
  )
  parser.add_argument(
    "--yahmp-csv",
    type=Path,
    default=DEFAULT_YAHMP_CSV,
    help="Path to the YAHMP per-motion CSV.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=DEFAULT_OUTPUT,
    help="Output image path. A PDF with the same stem is also written.",
  )
  parser.add_argument(
    "--no-stats",
    action="store_true",
    help="Disable paired Wilcoxon tests and significance annotations.",
  )
  parser.add_argument(
    "--twist2-color",
    default="blue",
    help=(
      "TWIST2 color preset or hex color. Presets: " + ", ".join(sorted(COLOR_PRESETS))
    ),
  )
  parser.add_argument(
    "--yahmp-color",
    default="orange",
    help=(
      "YAHMP color preset or hex color. Presets: " + ", ".join(sorted(COLOR_PRESETS))
    ),
  )
  return parser.parse_args()


def _parse_bool(value: str) -> bool:
  return value.strip().lower() in {"1", "true", "yes"}


def _parse_float(value: str | None) -> float:
  if value is None:
    return math.nan
  try:
    parsed = float(value)
  except ValueError:
    return math.nan
  return parsed if math.isfinite(parsed) else math.nan


def resolve_color(value: str) -> str:
  color = value.strip()
  if color in COLOR_PRESETS:
    return COLOR_PRESETS[color]
  if color.startswith("#") and len(color) in {4, 7}:
    return color
  raise ValueError(
    f"Unknown color `{value}`. Use a hex color or one of: {', '.join(sorted(COLOR_PRESETS))}"
  )


def load_metric_rows(csv_path: Path) -> dict[int, dict[str, float]]:
  if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path}")

  rows: dict[int, dict[str, float]] = {}
  with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    required = ("motion_index", *TRACKING_METRICS)
    missing = [column for column in required if column not in reader.fieldnames]
    if missing:
      raise KeyError(f"Missing required columns in {csv_path}: {missing}")

    for row in reader:
      if not _parse_bool(row.get("valid_for_success_eval", "True")):
        continue
      motion_index = int(row["motion_index"])
      metric_values: dict[str, float] = {}
      for metric in TRACKING_METRICS:
        value = _parse_float(row[metric])
        if math.isfinite(value):
          metric_values[metric] = value
      rows[motion_index] = metric_values

  return rows


def paired_metric_values(
  twist2_rows: dict[int, dict[str, float]],
  yahmp_rows: dict[int, dict[str, float]],
  metric_name: str,
) -> tuple[list[float], list[float]]:
  twist2_values: list[float] = []
  yahmp_values: list[float] = []
  for motion_index in sorted(set(twist2_rows) & set(yahmp_rows)):
    twist2_value = twist2_rows[motion_index].get(metric_name, math.nan)
    yahmp_value = yahmp_rows[motion_index].get(metric_name, math.nan)
    if math.isfinite(twist2_value) and math.isfinite(yahmp_value):
      twist2_values.append(twist2_value)
      yahmp_values.append(yahmp_value)
  return twist2_values, yahmp_values


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
  finite_items = sorted(
    ((key, value) for key, value in p_values.items() if math.isfinite(value)),
    key=lambda item: item[1],
  )
  adjusted = {key: math.nan for key in p_values}
  previous = 0.0
  total = len(finite_items)
  for rank, (key, p_value) in enumerate(finite_items):
    correction = total - rank
    adjusted_value = min(max(previous, correction * p_value), 1.0)
    adjusted[key] = adjusted_value
    previous = adjusted_value
  return adjusted


def significance_label(p_value: float) -> str:
  if not math.isfinite(p_value):
    return "n/a"
  if p_value < 0.0001:
    return "****"
  if p_value < 0.001:
    return "***"
  if p_value < 0.01:
    return "**"
  if p_value < 0.05:
    return "*"
  return "n.s."


def compute_significance(
  paired_values: dict[str, tuple[list[float], list[float]]],
) -> dict[str, tuple[float, float, int]]:
  _, wilcoxon_fn = _load_plot_deps()
  raw_p_values: dict[str, float] = {}
  sample_counts: dict[str, int] = {}
  for metric_name, (twist2_values, yahmp_values) in paired_values.items():
    sample_counts[metric_name] = len(twist2_values)
    if len(twist2_values) == 0:
      raw_p_values[metric_name] = math.nan
      continue
    if all(
      abs(a - b) <= 1e-12 for a, b in zip(twist2_values, yahmp_values, strict=True)
    ):
      raw_p_values[metric_name] = 1.0
      continue
    raw_p_values[metric_name] = float(
      wilcoxon_fn(twist2_values, yahmp_values, alternative="two-sided").pvalue
    )

  adjusted_p_values = holm_adjust(raw_p_values)
  return {
    metric_name: (
      raw_p_values[metric_name],
      adjusted_p_values[metric_name],
      sample_counts[metric_name],
    )
    for metric_name in paired_values
  }


def style_axis(ax: plt.Axes) -> None:
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_color("#A7A7A7")
  ax.spines["bottom"].set_color("#A7A7A7")
  ax.spines["left"].set_linewidth(0.8)
  ax.spines["bottom"].set_linewidth(0.8)
  ax.tick_params(axis="x", colors="#4F4F4F", labelsize=11, length=0)
  ax.tick_params(axis="y", colors="#4F4F4F", labelsize=16, length=0)
  ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
  ax.set_axisbelow(True)


def add_boxplot(
  ax: Any,
  twist2_values: list[float],
  yahmp_values: list[float],
  metric_name: str,
  adjusted_p_value: float | None,
  palette: dict[str, str],
) -> None:
  positions = (0.88, 1.12)
  box = ax.boxplot(
    [twist2_values, yahmp_values],
    positions=positions,
    patch_artist=True,
    widths=0.18,
    showfliers=False,
    showmeans=True,
    meanline=True,
    medianprops={"linewidth": 1.6},
    meanprops={"linestyle": "--", "linewidth": 1.2},
    whiskerprops={"linewidth": 1.2},
    capprops={"linewidth": 1.2},
    boxprops={"linewidth": 1.8},
  )
  for patch, label in zip(box["boxes"], ("TWIST2", "YAHMP"), strict=True):
    color = palette[label]
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
    patch.set_alpha(0.60)
    patch.set_linewidth(1.8)

  for box_idx, label in enumerate(("TWIST2", "YAHMP")):
    color = palette[label]
    for line in box["whiskers"][2 * box_idx : 2 * box_idx + 2]:
      line.set_color(color)
    for line in box["caps"][2 * box_idx : 2 * box_idx + 2]:
      line.set_color(color)
    box["medians"][box_idx].set_color(color)
    box["means"][box_idx].set_color(color)

  ax.set_xlim(0.72, 1.28)
  ax.set_xticks(positions, ["TWIST2", "YAHMP"])
  ax.set_title(METRIC_LABELS[metric_name], fontsize=19, color="#2F2F2F", pad=10)
  style_axis(ax)
  if adjusted_p_value is not None:
    annotate_significance(ax, adjusted_p_value)


def annotate_significance(ax: Any, adjusted_p_value: float) -> None:
  y_min, y_max = ax.get_ylim()
  span = y_max - y_min
  y_line = y_max + 0.045 * span
  y_text = y_max + 0.085 * span
  ax.plot(
    [0.88, 0.88, 1.12, 1.12],
    [y_line, y_text, y_text, y_line],
    color="#777777",
    linewidth=0.9,
  )
  ax.text(
    1.0,
    y_text,
    significance_label(adjusted_p_value),
    ha="center",
    va="bottom",
    fontsize=20,
    color="#333333",
  )
  ax.set_ylim(y_min, y_max + 0.18 * span)


def save_figure(fig: Any, output_path: Path) -> tuple[Path, Path]:
  output_path = output_path.expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  pdf_path = output_path.with_suffix(".pdf")
  fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
  fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
  return output_path, pdf_path


def main() -> None:
  plt_mod, _ = _load_plot_deps()
  args = parse_args()
  palette = {
    "TWIST2": resolve_color(args.twist2_color),
    "YAHMP": resolve_color(args.yahmp_color),
  }
  twist2_rows = load_metric_rows(args.twist2_csv.expanduser().resolve())
  yahmp_rows = load_metric_rows(args.yahmp_csv.expanduser().resolve())
  paired_values = {
    metric_name: paired_metric_values(twist2_rows, yahmp_rows, metric_name)
    for metric_name in TRACKING_METRICS
  }
  significance = None if args.no_stats else compute_significance(paired_values)

  fig, axes = plt_mod.subplots(2, 5, figsize=(9.6, 6.8), constrained_layout=True)
  fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.035, wspace=0.01, hspace=0.05)
  fig.patch.set_facecolor("white")

  for ax, metric_name in zip(axes.flat, TRACKING_METRICS, strict=True):
    twist2_values, yahmp_values = paired_values[metric_name]
    adjusted_p_value = None if significance is None else significance[metric_name][1]
    add_boxplot(ax, twist2_values, yahmp_values, metric_name, adjusted_p_value, palette)

  png_path, pdf_path = save_figure(fig, args.output)
  print(f"Saved plot: {png_path}")
  print(f"Saved plot: {pdf_path}")
  if significance is not None:
    print(
      "Significance test: paired Wilcoxon signed-rank, Holm-adjusted across metrics"
    )
    for metric_name in TRACKING_METRICS:
      raw_p_value, adjusted_p_value, sample_count = significance[metric_name]
      print(
        f"{metric_name}: n={sample_count}, "
        f"p={raw_p_value:.6g}, p_holm={adjusted_p_value:.6g}, "
        f"{significance_label(adjusted_p_value)}"
      )


if __name__ == "__main__":
  main()
