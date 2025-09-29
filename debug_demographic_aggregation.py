#!/usr/bin/env python3
"""
Debug script to check if demographic aggregation is working correctly.
Examines individual scores vs aggregated demographic means.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse
import logging

logger = logging.getLogger(__name__)


def load_eval_file(project_name: str) -> Dict[str, Any]:
    """Load evaluation file for a project."""
    project_dir = Path("results") / project_name
    eval_file = project_dir / "eval_scm.json"

    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    with open(eval_file, "r") as f:
        return json.load(f)


def extract_individual_scores(eval_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract individual scores from evaluation data."""
    # Try different possible locations for detailed scores
    detailed_scores = eval_data.get("detailed_scores") or eval_data.get(
        "evaluation_results", {}
    ).get("detailed_scores", [])

    if not detailed_scores:
        raise ValueError("No detailed scores found in evaluation data")

    rows = []
    for score in detailed_scores:
        row = {
            "index": score.get("index", -1),
            "text": score.get("text", ""),
            "warmth_score": score["warmth"]["score"],
            "competency_score": score["competency"]["score"],
        }

        if "demographic" in score:
            row.update(
                {
                    "gender": score["demographic"]["gender"],
                    "ethnicity": score["demographic"]["ethnicity"],
                    "demographic_group": score["demographic"]["group"],
                }
            )
        else:
            row.update(
                {
                    "gender": "unknown",
                    "ethnicity": "unknown",
                    "demographic_group": "unknown_unknown",
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def extract_aggregated_means(eval_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract aggregated demographic means."""
    # Try different possible locations for demographic groups
    demographic_groups = (
        eval_data.get("demographic_groups")
        or eval_data.get("aggregated_analysis", {}).get("by_demographic_group", {})
        or eval_data.get("evaluation_results", {}).get("demographic_groups", {})
    )

    if not demographic_groups:
        raise ValueError("No demographic groups found in evaluation data")

    rows = []
    for group_name, group_data in demographic_groups.items():
        # Handle both single and multi-model formats
        if isinstance(group_data["warmth"]["mean"], list):
            warmth_mean = group_data["warmth"]["mean"][0]
            competency_mean = group_data["competency"]["mean"][0]
            warmth_std = group_data["warmth"]["std"][0]
            competency_std = group_data["competency"]["std"][0]
        else:
            warmth_mean = group_data["warmth"]["mean"]
            competency_mean = group_data["competency"]["mean"]
            warmth_std = group_data["warmth"]["std"]
            competency_std = group_data["competency"]["std"]

        sample_count = group_data.get("sample_count") or group_data.get(
            "demographic_info", {}
        ).get("n_samples", 1)

        rows.append(
            {
                "demographic_group": group_name,
                "aggregated_warmth_mean": warmth_mean,
                "aggregated_competency_mean": competency_mean,
                "aggregated_warmth_std": warmth_std,
                "aggregated_competency_std": competency_std,
                "sample_count": sample_count,
            }
        )

    return pd.DataFrame(rows)


def verify_aggregation(individual_df: pd.DataFrame, aggregated_df: pd.DataFrame):
    """Verify that aggregated means match manually calculated means from individual scores."""
    print("=" * 80)
    print("DEMOGRAPHIC AGGREGATION VERIFICATION")
    print("=" * 80)

    # Calculate manual means from individual scores
    manual_means = (
        individual_df.groupby("demographic_group")
        .agg(
            {
                "warmth_score": ["mean", "std", "count"],
                "competency_score": ["mean", "std", "count"],
            }
        )
        .round(6)
    )

    print(f"\nFound {len(individual_df)} individual scores")
    print(f"Found {len(aggregated_df)} aggregated demographic groups")

    # Compare each demographic group
    for _, agg_row in aggregated_df.iterrows():
        group_name = agg_row["demographic_group"]

        if group_name not in manual_means.index:
            print(
                f"\n❌ ERROR: Group '{group_name}' in aggregated data but not in individual scores!"
            )
            continue

        manual_warmth_mean = manual_means.loc[group_name, ("warmth_score", "mean")]
        manual_competency_mean = manual_means.loc[
            group_name, ("competency_score", "mean")
        ]
        manual_warmth_std = manual_means.loc[group_name, ("warmth_score", "std")]
        manual_competency_std = manual_means.loc[
            group_name, ("competency_score", "std")
        ]
        manual_count = manual_means.loc[group_name, ("warmth_score", "count")]

        agg_warmth_mean = agg_row["aggregated_warmth_mean"]
        agg_competency_mean = agg_row["aggregated_competency_mean"]
        agg_warmth_std = agg_row["aggregated_warmth_std"]
        agg_competency_std = agg_row["aggregated_competency_std"]
        agg_count = agg_row["sample_count"]

        print(f"\n📊 {group_name.upper().replace('_', ' → ')} (n={manual_count})")

        # Check warmth
        warmth_diff = abs(manual_warmth_mean - agg_warmth_mean)
        warmth_std_diff = abs(manual_warmth_std - agg_warmth_std)

        if warmth_diff < 1e-6:
            print(f"  ✅ Warmth mean: {manual_warmth_mean:.6f} (matches aggregated)")
        else:
            print(
                f"  ❌ Warmth mean: manual={manual_warmth_mean:.6f}, agg={agg_warmth_mean:.6f}, diff={warmth_diff:.6f}"
            )

        if warmth_std_diff < 1e-6:
            print(f"  ✅ Warmth std:  {manual_warmth_std:.6f} (matches aggregated)")
        else:
            print(
                f"  ❌ Warmth std:  manual={manual_warmth_std:.6f}, agg={agg_warmth_std:.6f}, diff={warmth_std_diff:.6f}"
            )

        # Check competency
        comp_diff = abs(manual_competency_mean - agg_competency_mean)
        comp_std_diff = abs(manual_competency_std - agg_competency_std)

        if comp_diff < 1e-6:
            print(
                f"  ✅ Comp mean:   {manual_competency_mean:.6f} (matches aggregated)"
            )
        else:
            print(
                f"  ❌ Comp mean:   manual={manual_competency_mean:.6f}, agg={agg_competency_mean:.6f}, diff={comp_diff:.6f}"
            )

        if comp_std_diff < 1e-6:
            print(f"  ✅ Comp std:    {manual_competency_std:.6f} (matches aggregated)")
        else:
            print(
                f"  ❌ Comp std:    manual={manual_competency_std:.6f}, agg={agg_competency_std:.6f}, diff={comp_std_diff:.6f}"
            )

        # Check sample count
        if manual_count == agg_count:
            print(f"  ✅ Sample count: {manual_count} (matches aggregated)")
        else:
            print(f"  ❌ Sample count: manual={manual_count}, agg={agg_count}")


def analyze_individual_score_distribution(individual_df: pd.DataFrame):
    """Analyze the distribution of individual scores within each demographic group."""
    print("\n" + "=" * 80)
    print("INDIVIDUAL SCORE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    for group_name in individual_df["demographic_group"].unique():
        group_data = individual_df[individual_df["demographic_group"] == group_name]

        print(f"\n📈 {group_name.upper().replace('_', ' → ')} (n={len(group_data)})")

        warmth_scores = group_data["warmth_score"].values
        comp_scores = group_data["competency_score"].values

        print(
            f"  Warmth scores:     {warmth_scores[:5]}{'...' if len(warmth_scores) > 5 else ''}"
        )
        print(
            f"  Competency scores: {comp_scores[:5]}{'...' if len(comp_scores) > 5 else ''}"
        )

        # Check for identical scores (indicating potential aggregation issues)
        warmth_unique = len(np.unique(warmth_scores))
        comp_unique = len(np.unique(comp_scores))

        if warmth_unique == 1:
            print(
                f"  🚨 WARNING: All warmth scores are identical ({warmth_scores[0]:.6f})"
            )
        else:
            print(
                f"  ✅ Warmth has {warmth_unique} unique values (range: {warmth_scores.min():.6f} to {warmth_scores.max():.6f})"
            )

        if comp_unique == 1:
            print(
                f"  🚨 WARNING: All competency scores are identical ({comp_scores[0]:.6f})"
            )
        else:
            print(
                f"  ✅ Competency has {comp_unique} unique values (range: {comp_scores.min():.6f} to {comp_scores.max():.6f})"
            )

        # Show a few example texts if all scores are identical
        if warmth_unique == 1 or comp_unique == 1:
            print(f"  📝 Sample texts:")
            for i, text in enumerate(group_data["text"].head(3)):
                print(f"    {i+1}. {text[:80]}{'...' if len(text) > 80 else ''}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Debug demographic aggregation in SCM evaluation"
    )

    parser.add_argument("project_name", help="Name of project to analyze")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        print(f"Debugging demographic aggregation for project: {args.project_name}")

        # Load evaluation data
        eval_data = load_eval_file(args.project_name)

        # Extract individual and aggregated scores
        individual_df = extract_individual_scores(eval_data)
        aggregated_df = extract_aggregated_means(eval_data)

        # Verify aggregation
        verify_aggregation(individual_df, aggregated_df)

        # Analyze individual score distribution
        analyze_individual_score_distribution(individual_df)

        print("\n" + "=" * 80)
        print("AGGREGATION DEBUG COMPLETE")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error during debug: {e}")
        raise


if __name__ == "__main__":
    main()
