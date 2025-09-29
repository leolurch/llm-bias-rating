#!/usr/bin/env python3
"""
Analysis script to examine demographic means from evaluation files.
Checks if there are suspiciously similar values or only discrete combinations.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def load_eval_data(project_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load evaluation data from multiple projects."""
    eval_data = {}

    for project_name in project_names:
        project_dir = Path("results") / project_name
        eval_file = project_dir / "eval_scm.json"

        if not eval_file.exists():
            logger.warning(f"Evaluation file not found: {eval_file}")
            continue

        with open(eval_file, "r") as f:
            data = json.load(f)
            eval_data[project_name] = data

    logger.info(f"Loaded evaluation data from {len(eval_data)} projects")
    return eval_data


def extract_demographic_means(eval_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Extract demographic group means from evaluation data."""
    rows = []

    for project_name, data in eval_data.items():
        # Try different possible locations for demographic data
        demographic_groups = (
            data.get("demographic_groups", {})
            or data.get("aggregated_analysis", {}).get("by_demographic_group", {})
            or data.get("evaluation_results", {}).get("demographic_groups", {})
        )

        if not demographic_groups:
            logger.warning(f"No demographic groups found in {project_name}")
            continue

        for group_name, group_data in demographic_groups.items():
            # Handle both single and multi-model formats
            if isinstance(group_data["warmth"]["mean"], list):
                warmth_mean = group_data["warmth"]["mean"][0]
                competency_mean = group_data["competency"]["mean"][0]
            else:
                warmth_mean = group_data["warmth"]["mean"]
                competency_mean = group_data["competency"]["mean"]

            # Get sample count
            sample_count = group_data.get("sample_count") or group_data.get(
                "demographic_info", {}
            ).get("n_samples", 1)

            gender, ethnicity = group_name.split("_", 1)

            rows.append(
                {
                    "project": project_name,
                    "demographic_group": group_name,
                    "gender": gender,
                    "ethnicity": ethnicity,
                    "warmth_mean": warmth_mean,
                    "competency_mean": competency_mean,
                    "sample_count": sample_count,
                    "warmth_rounded_3": round(warmth_mean, 3),
                    "competency_rounded_3": round(competency_mean, 3),
                    "warmth_rounded_2": round(warmth_mean, 2),
                    "competency_rounded_2": round(competency_mean, 2),
                }
            )

    return pd.DataFrame(rows)


def analyze_value_distributions(df: pd.DataFrame):
    """Analyze the distribution of warmth and competency values."""
    print("=" * 80)
    print("VALUE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Check unique values at different precision levels
    print(f"\nTotal data points: {len(df)}")
    print(f"Projects analyzed: {df['project'].nunique()}")
    print(f"Demographic groups: {df['demographic_group'].nunique()}")

    print(f"\nUnique warmth values (full precision): {df['warmth_mean'].nunique()}")
    print(
        f"Unique competency values (full precision): {df['competency_mean'].nunique()}"
    )

    print(f"\nUnique warmth values (3 decimals): {df['warmth_rounded_3'].nunique()}")
    print(
        f"Unique competency values (3 decimals): {df['competency_rounded_3'].nunique()}"
    )

    print(f"\nUnique warmth values (2 decimals): {df['warmth_rounded_2'].nunique()}")
    print(
        f"Unique competency values (2 decimals): {df['competency_rounded_2'].nunique()}"
    )

    # Check unique coordinate pairs
    unique_pairs_full = df[["warmth_mean", "competency_mean"]].drop_duplicates()
    unique_pairs_3dec = df[
        ["warmth_rounded_3", "competency_rounded_3"]
    ].drop_duplicates()
    unique_pairs_2dec = df[
        ["warmth_rounded_2", "competency_rounded_2"]
    ].drop_duplicates()

    print(f"\nUnique (warmth, competency) coordinate pairs:")
    print(f"  Full precision: {len(unique_pairs_full)}")
    print(f"  3 decimals: {len(unique_pairs_3dec)}")
    print(f"  2 decimals: {len(unique_pairs_2dec)}")

    # Show the actual unique values
    print(f"\nActual unique warmth values (sorted):")
    unique_warmth = sorted(df["warmth_mean"].unique())
    for i, val in enumerate(unique_warmth):
        if i < 20:  # Show first 20
            print(f"  {val:.6f}")
        elif i == 20:
            print(f"  ... and {len(unique_warmth) - 20} more")
            break

    print(f"\nActual unique competency values (sorted):")
    unique_comp = sorted(df["competency_mean"].unique())
    for i, val in enumerate(unique_comp):
        if i < 20:  # Show first 20
            print(f"  {val:.6f}")
        elif i == 20:
            print(f"  ... and {len(unique_comp) - 20} more")
            break

    # Show unique coordinate pairs
    print(f"\nUnique coordinate pairs (warmth, competency) - first 20:")
    for i, (_, row) in enumerate(unique_pairs_full.iterrows()):
        if i < 20:
            print(f"  ({row['warmth_mean']:.6f}, {row['competency_mean']:.6f})")
        elif i == 20:
            print(f"  ... and {len(unique_pairs_full) - 20} more")
            break


def analyze_cross_project_similarity(df: pd.DataFrame):
    """Analyze if values are suspiciously similar across projects."""
    print("\n" + "=" * 80)
    print("CROSS-PROJECT SIMILARITY ANALYSIS")
    print("=" * 80)

    # Group by demographic group and compare across projects
    for demo_group in df["demographic_group"].unique():
        group_data = df[df["demographic_group"] == demo_group]
        if len(group_data) < 2:
            continue

        print(f"\n{demo_group.upper().replace('_', ' → ')}:")
        print(f"{'Project':<20} {'Warmth':<12} {'Competency':<12} {'Samples':<8}")
        print("-" * 55)

        warmth_values = []
        comp_values = []

        for _, row in group_data.iterrows():
            print(
                f"{row['project']:<20} {row['warmth_mean']:<12.6f} {row['competency_mean']:<12.6f} {row['sample_count']:<8}"
            )
            warmth_values.append(row["warmth_mean"])
            comp_values.append(row["competency_mean"])

        # Calculate statistics
        if len(warmth_values) > 1:
            warmth_std = np.std(warmth_values)
            comp_std = np.std(comp_values)
            warmth_range = max(warmth_values) - min(warmth_values)
            comp_range = max(comp_values) - min(comp_values)

            print(f"{'Statistics:':<20} W_std={warmth_std:.6f} C_std={comp_std:.6f}")
            print(f"{'Range:':<20} W_range={warmth_range:.6f} C_range={comp_range:.6f}")

            # Flag suspiciously similar values
            if warmth_std < 0.001 and comp_std < 0.001:
                print("  🚨 SUSPICIOUS: Values are nearly identical across projects!")
            elif warmth_range < 0.01 and comp_range < 0.01:
                print("  ⚠️  WARNING: Values are very similar across projects")


def analyze_value_clustering(df: pd.DataFrame):
    """Analyze if values cluster around specific points."""
    print("\n" + "=" * 80)
    print("VALUE CLUSTERING ANALYSIS")
    print("=" * 80)

    # Check if values cluster around specific points
    warmth_values = df["warmth_mean"].values
    comp_values = df["competency_mean"].values

    # Find the most common values (rounded to different precisions)
    print("\nMost common warmth values (rounded to 4 decimals):")
    warmth_rounded = np.round(warmth_values, 4)
    warmth_counts = pd.Series(warmth_rounded).value_counts().head(10)
    for val, count in warmth_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {val:.4f}: {count} occurrences ({percentage:.1f}%)")

    print("\nMost common competency values (rounded to 4 decimals):")
    comp_rounded = np.round(comp_values, 4)
    comp_counts = pd.Series(comp_rounded).value_counts().head(10)
    for val, count in comp_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {val:.4f}: {count} occurrences ({percentage:.1f}%)")

    # Check coordinate pair clustering
    print("\nMost common coordinate pairs (rounded to 3 decimals):")
    df_rounded = df.copy()
    df_rounded["coord_pair"] = df_rounded.apply(
        lambda row: f"({row['warmth_rounded_3']:.3f}, {row['competency_rounded_3']:.3f})",
        axis=1,
    )
    pair_counts = df_rounded["coord_pair"].value_counts().head(10)
    for pair, count in pair_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {pair}: {count} occurrences ({percentage:.1f}%)")


def generate_summary_statistics(df: pd.DataFrame):
    """Generate overall summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nOverall warmth statistics:")
    print(f"  Mean: {df['warmth_mean'].mean():.6f}")
    print(f"  Std:  {df['warmth_mean'].std():.6f}")
    print(f"  Min:  {df['warmth_mean'].min():.6f}")
    print(f"  Max:  {df['warmth_mean'].max():.6f}")
    print(f"  Range: {df['warmth_mean'].max() - df['warmth_mean'].min():.6f}")

    print(f"\nOverall competency statistics:")
    print(f"  Mean: {df['competency_mean'].mean():.6f}")
    print(f"  Std:  {df['competency_mean'].std():.6f}")
    print(f"  Min:  {df['competency_mean'].min():.6f}")
    print(f"  Max:  {df['competency_mean'].max():.6f}")
    print(f"  Range: {df['competency_mean'].max() - df['competency_mean'].min():.6f}")

    # Check if the range is suspiciously small
    warmth_range = df["warmth_mean"].max() - df["warmth_mean"].min()
    comp_range = df["competency_mean"].max() - df["competency_mean"].min()

    if warmth_range < 0.1:
        print(f"  🚨 SUSPICIOUS: Warmth range ({warmth_range:.6f}) is very small!")
    if comp_range < 0.1:
        print(f"  🚨 SUSPICIOUS: Competency range ({comp_range:.6f}) is very small!")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze demographic means from evaluation files to detect suspicious patterns"
    )

    parser.add_argument("project_names", nargs="+", help="Names of projects to analyze")

    parser.add_argument("--save-csv", help="Save detailed data to CSV file")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        # Load evaluation data
        eval_data = load_eval_data(args.project_names)

        if not eval_data:
            logger.error("No evaluation data could be loaded")
            return

        # Extract demographic means
        df = extract_demographic_means(eval_data)

        if df.empty:
            logger.error("No demographic data could be extracted")
            return

        print(
            f"Analyzing {len(df)} demographic group measurements from {len(args.project_names)} projects"
        )

        # Run analyses
        analyze_value_distributions(df)
        analyze_cross_project_similarity(df)
        analyze_value_clustering(df)
        generate_summary_statistics(df)

        # Save to CSV if requested
        if args.save_csv:
            df.to_csv(args.save_csv, index=False)
            print(f"\nDetailed data saved to {args.save_csv}")

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
