"""Compute demographic correlation statistics from SCM evaluation CSV files."""

from __future__ import annotations

import argparse
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def format_number(value) -> str:
    """Format floating point values for LaTeX tables."""

    if value is None:
        return "-"

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]"
        return "[" + ", ".join(format_number(v) for v in value) + "]"

    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(value)

    if isinstance(value, float) and math.isnan(value):
        return "-"

    if abs(value) < 1e-3 and value != 0:
        return f"{value:.2e}"

    return f"{value:.3f}"


def load_model_name(project_path: Path) -> str:
    """Extract the model name from the project's responses.json."""

    responses_path = project_path / "responses.json"
    if not responses_path.exists():
        return project_path.name

    try:
        with responses_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return (
            data.get("metadata", {})
            .get("experiment_info", {})
            .get("model_info", {})
            .get("model_name", project_path.name)
            .split("/")[-1]
        )
    except json.JSONDecodeError:
        return project_path.name


def collapse_repetitions(df: pd.DataFrame) -> pd.DataFrame:
    """Average exact repetitions that share scenario, gender, ethnicity, and candidate name."""

    group_cols = ["scenario_id", "gender", "ethnicity", "candidate_name"]
    value_cols = ["warmth_score", "competency_score"]

    collapsed = (
        df[group_cols + value_cols]
        .groupby(group_cols, as_index=False)
        .mean(numeric_only=True)
    )

    return collapsed


def compute_gender_stats(df: pd.DataFrame, score_column: str) -> dict:
    """Run Wilcoxon signed-rank test on gender differences within matched blocks."""

    required = {"scenario_id", "ethnicity", "gender", score_column}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing required columns for gender stats: {sorted(missing)}")

    grouped = (
        df.groupby(["scenario_id", "ethnicity", "gender"], as_index=False)[
            score_column
        ]
        .mean()
    )

    pivot = (
        grouped.pivot_table(
            index=["scenario_id", "ethnicity"],
            columns="gender",
            values=score_column,
        )
        .dropna(axis=0, how="any")
    )

    if {"female", "male"} - set(pivot.columns):
        return {"p": float("nan"), "r_rb": float("nan"), "n_eff": 0}

    diffs = pivot["female"].to_numpy() - pivot["male"].to_numpy()
    mask = ~np.isclose(diffs, 0.0)
    diff_values = diffs[mask]
    n_eff = diff_values.size

    if n_eff == 0:
        return {
            "p": float("nan"),
            "r_rb": float("nan"),
            "n_eff": 0,
            "median": float("nan"),
            "ci": (),
        }

    try:
        _, p_value = wilcoxon(diff_values, zero_method="wilcox")
    except ValueError:
        return {
            "p": float("nan"),
            "r_rb": float("nan"),
            "n_eff": int(n_eff),
            "median": float("nan"),
            "ci": (),
        }

    ranks = rankdata(np.abs(diff_values), method="average")
    positive = ranks[diff_values > 0].sum()
    negative = ranks[diff_values < 0].sum()
    denom = positive + negative
    r_rb = float("nan") if denom == 0 else (positive - negative) / denom

    median = float(np.median(diff_values))
    boot_samples = 1000
    rng = np.random.default_rng(seed=42)
    bootstrap = np.empty(boot_samples, dtype=float)
    for i in range(boot_samples):
        bootstrap[i] = np.median(rng.choice(diff_values, size=n_eff, replace=True))
    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])

    return {
        "p": float(p_value),
        "r_rb": r_rb,
        "n_eff": int(n_eff),
        "median": median,
        "ci": (float(ci_low), float(ci_high)),
    }


def compute_ethnicity_stats(
    df: pd.DataFrame, score_column: str, include_gender: bool
) -> dict:
    """Run Friedman test across ethnicities (optionally across gender-ethnicity cells)."""

    required = {"scenario_id", "ethnicity", "gender", score_column}

    if missing := required - set(df.columns):
        raise ValueError(
            f"Missing required columns for ethnicity stats ({score_column}): {sorted(missing)}"
        )

    ethnicities = sorted(df["ethnicity"].dropna().unique())
    genders = sorted(df["gender"].dropna().unique())

    if len(ethnicities) < 2:
        return {"p": float("nan"), "w": float("nan"), "n_blocks": 0, "ordering": ethnicities}

    if include_gender:
        if len(genders) < 2:
            return {
                "p": float("nan"),
                "w": float("nan"),
                "n_blocks": 0,
                "ordering": [f"{g}_{e}" for g in genders for e in ethnicities],
            }

        grouped = (
            df.groupby(["scenario_id", "gender", "ethnicity"], as_index=False)[
                score_column
            ]
            .mean()
        )
        grouped = grouped.assign(
            demo=grouped["gender"].astype(str).str.cat(grouped["ethnicity"], sep="_")
        )
        pivot = (
            grouped.pivot_table(
                index=["scenario_id"],
                columns="demo",
                values=score_column,
            )
        )
        expected_columns = [f"{g}_{e}" for g in genders for e in ethnicities]
    else:
        grouped = (
            df.groupby(["scenario_id", "gender", "ethnicity"], as_index=False)[
                score_column
            ]
            .mean()
        )
        pivot = (
            grouped.pivot_table(
                index=["scenario_id", "gender"],
                columns="ethnicity",
                values=score_column,
            )
        )
        expected_columns = ethnicities

    if set(expected_columns) - set(pivot.columns):
        return {
            "p": float("nan"),
            "w": float("nan"),
            "n_blocks": 0,
            "ordering": expected_columns,
        }

    pivot = pivot[expected_columns].dropna(axis=0, how="any")
    n_blocks = pivot.shape[0]

    if n_blocks == 0:
        return {"p": float("nan"), "w": float("nan"), "n_blocks": 0, "ordering": expected_columns}

    samples = [pivot[col].to_numpy() for col in expected_columns]
    stat, p_value = friedmanchisquare(*samples)
    k = len(expected_columns)
    kendalls_w = stat / (n_blocks * (k - 1))

    return {
        "p": float(p_value),
        "w": float(kendalls_w),
        "n_blocks": int(n_blocks),
        "ordering": expected_columns,
    }


def apply_bh_correction(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg correction for a list of p-values."""

    n = len(p_values)
    if n == 0:
        return []

    p_array = np.asarray(p_values, dtype=float)
    order = np.argsort(p_array)
    ordered = p_array[order]
    adjusted = np.empty(n, dtype=float)
    cumulative_min = 1.0

    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        value = ordered[idx] * n / rank
        cumulative_min = min(cumulative_min, value)
        adjusted[order[idx]] = min(cumulative_min, 1.0)

    return adjusted.tolist()


def analyze_project(project: str) -> dict:
    """Compute all requested statistics for a single project."""

    project_path = Path("results") / project
    csv_path = project_path / "eval_scm.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    expected_columns = [
        "scenario_id",
        "output_id",
        "gender",
        "ethnicity",
        "candidate_name",
        "warmth_score",
        "competency_score",
        "response_truncated[:50]",
    ]

    if list(df.columns) != expected_columns:
        raise ValueError(
            "Unexpected columns in eval_scm.csv. "
            f"Expected {expected_columns}, got {list(df.columns)}"
        )

    df["scenario_id"] = df["scenario_id"].astype(int)
    df["output_id"] = df["output_id"].astype(int)

    model_name = load_model_name(project_path)

    df = collapse_repetitions(df)

    warmth_gender = compute_gender_stats(df, "warmth_score")
    comp_gender = compute_gender_stats(df, "competency_score")

    warmth_ethnicity = compute_ethnicity_stats(
        df, "warmth_score", include_gender=False
    )
    comp_ethnicity = compute_ethnicity_stats(
        df, "competency_score", include_gender=False
    )

    warmth_demo = compute_ethnicity_stats(df, "warmth_score", include_gender=True)
    comp_demo = compute_ethnicity_stats(df, "competency_score", include_gender=True)

    stats = {
        "Model": model_name,
        "warmth_gender_p_raw": warmth_gender["p"],
        "warmth_gender_r": warmth_gender["r_rb"],
        "warmth_gender_n": warmth_gender["n_eff"],
        "warmth_gender_median": warmth_gender["median"],
        "warmth_gender_ci": warmth_gender["ci"],
        "competency_gender_p_raw": comp_gender["p"],
        "competency_gender_r": comp_gender["r_rb"],
        "competency_gender_n": comp_gender["n_eff"],
        "competency_gender_median": comp_gender["median"],
        "competency_gender_ci": comp_gender["ci"],
        "warmth_ethnicity_p_raw": warmth_ethnicity["p"],
        "warmth_ethnicity_w": warmth_ethnicity["w"],
        "warmth_ethnicity_n": warmth_ethnicity["n_blocks"],
        "warmth_ethnicity_order": warmth_ethnicity["ordering"],
        "competency_ethnicity_p_raw": comp_ethnicity["p"],
        "competency_ethnicity_w": comp_ethnicity["w"],
        "competency_ethnicity_n": comp_ethnicity["n_blocks"],
        "competency_ethnicity_order": comp_ethnicity["ordering"],
        "warmth_demo_p_raw": warmth_demo["p"],
        "warmth_demo_w": warmth_demo["w"],
        "warmth_demo_n": warmth_demo["n_blocks"],
        "warmth_demo_order": warmth_demo["ordering"],
        "competency_demo_p_raw": comp_demo["p"],
        "competency_demo_w": comp_demo["w"],
        "competency_demo_n": comp_demo["n_blocks"],
        "competency_demo_order": comp_demo["ordering"],
    }

    p_keys = [
        "warmth_gender",
        "warmth_ethnicity",
        "warmth_demo",
        "competency_gender",
        "competency_ethnicity",
        "competency_demo",
    ]

    valid = [
        (key, stats[f"{key}_p_raw"])
        for key in p_keys
        if stats[f"{key}_p_raw"] is not None
        and not (isinstance(stats[f"{key}_p_raw"], float) and math.isnan(stats[f"{key}_p_raw"]))
    ]

    if valid:
        corrected = apply_bh_correction([val for _, val in valid])
        for (key, _), adj in zip(valid, corrected):
            stats[f"{key}_p"] = float(adj)
    for key in p_keys:
        stats.setdefault(f"{key}_p", stats.get(f"{key}_p_raw", float("nan")))

    return stats


WARMTH_COLUMNS: List[Tuple[str, str]] = [
    ("Model", "Model"),
    ("$p_{BH}$ Warmth Gender", "warmth_gender_p"),
    ("$r_{rb}$ Warmth Gender", "warmth_gender_r"),
    ("$n$ Warmth Gender", "warmth_gender_n"),
    ("Median Δ Warmth Gender", "warmth_gender_median"),
    ("95% CI Warmth Gender", "warmth_gender_ci"),
    ("$p_{BH}$ Warmth Ethnicity", "warmth_ethnicity_p"),
    ("$W$ Warmth Ethnicity", "warmth_ethnicity_w"),
    ("$n$ Warmth Ethnicity", "warmth_ethnicity_n"),
    ("Order Warmth Ethnicity", "warmth_ethnicity_order"),
    ("$p_{BH}$ Warmth Demo", "warmth_demo_p"),
    ("$W$ Warmth Demo", "warmth_demo_w"),
    ("$n$ Warmth Demo", "warmth_demo_n"),
    ("Order Warmth Demo", "warmth_demo_order"),
]

COMPETENCY_COLUMNS: List[Tuple[str, str]] = [
    ("Model", "Model"),
    ("$p_{BH}$ Competency Gender", "competency_gender_p"),
    ("$r_{rb}$ Competency Gender", "competency_gender_r"),
    ("$n$ Competency Gender", "competency_gender_n"),
    ("Median Δ Competency Gender", "competency_gender_median"),
    ("95% CI Competency Gender", "competency_gender_ci"),
    ("$p_{BH}$ Competency Ethnicity", "competency_ethnicity_p"),
    ("$W$ Competency Ethnicity", "competency_ethnicity_w"),
    ("$n$ Competency Ethnicity", "competency_ethnicity_n"),
    ("Order Competency Ethnicity", "competency_ethnicity_order"),
    ("$p_{BH}$ Competency Demo", "competency_demo_p"),
    ("$W$ Competency Demo", "competency_demo_w"),
    ("$n$ Competency Demo", "competency_demo_n"),
    ("Order Competency Demo", "competency_demo_order"),
]


def build_latex_rows(
    results: Iterable[dict], columns: List[Tuple[str, str]]
) -> List[str]:
    """Render LaTeX table lines for the provided columns."""

    header = [col for col, _ in columns]
    lines = [" & ".join(header) + r" \\"]
    for stats in results:
        row = [
            stats[key] if key == "Model" else format_number(stats[key])
            for _, key in columns
        ]
        lines.append(" & ".join(row) + r" \\")
    return lines


def build_cli_table(
    results: Iterable[dict], columns: List[Tuple[str, str]]
) -> List[str]:
    """Render an ASCII table for the provided columns."""

    headers = [col for col, _ in columns]
    rows: List[List[str]] = []
    for stats in results:
        rows.append(
            [
                stats[key] if key == "Model" else format_number(stats[key])
                for _, key in columns
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(
        header.ljust(widths[i]) for i, header in enumerate(headers)
    )
    separator = "-+-".join("-" * widths[i] for i in range(len(headers)))

    lines = [header_line, separator]
    for row in rows:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return lines


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run demographic correlation tests on SCM evaluation outputs."
    )
    parser.add_argument(
        "projects",
        nargs="+",
        help="Names of projects located under results/<project>/eval_scm.csv",
    )
    args = parser.parse_args(argv)

    aggregated_stats: List[dict] = []
    for project in args.projects:
        try:
            aggregated_stats.append(analyze_project(project))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to analyze project '{project}': {exc}") from exc

    print("Warmth Metrics")
    for line in build_cli_table(aggregated_stats, WARMTH_COLUMNS):
        print(line)

    print("\nCompetency Metrics")
    for line in build_cli_table(aggregated_stats, COMPETENCY_COLUMNS):
        print(line)

    print()  # blank line before LaTeX output

    for line in build_latex_rows(aggregated_stats, WARMTH_COLUMNS):
        print(line)

    print()  # blank line between LaTeX tables

    for line in build_latex_rows(aggregated_stats, COMPETENCY_COLUMNS):
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
