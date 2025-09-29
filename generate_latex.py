#!/usr/bin/env python3
"""
Generate separate LaTeX tables for warmth and competency from evaluation results.
Extracts statistics from evaluation JSON files and identifies highest/lowest demographic groups.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
import logging

from visualization import get_model_name

logger = logging.getLogger(__name__)


def load_eval_file(project_name: str) -> Dict[str, Any]:
    """Load evaluation file for a project."""
    project_dir = Path("results") / project_name
    eval_file = project_dir / "eval_scm.json"

    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    with open(eval_file, "r") as f:
        return json.load(f)


def get_general_statistics(eval_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Get general warmth and competency statistics from evaluation results."""
    # Get from evaluation_results section
    evaluation_results = eval_data.get("evaluation_results", {})

    warmth_stats = evaluation_results.get("warmth", {})
    competency_stats = evaluation_results.get("competency", {})

    return {
        "warmth": {
            "median": warmth_stats.get("median", 0.0),
            "min": warmth_stats.get("min", 0.0),
            "max": warmth_stats.get("max", 0.0),
            "std": warmth_stats.get("std", 0.0),
        },
        "competency": {
            "median": competency_stats.get("median", 0.0),
            "min": competency_stats.get("min", 0.0),
            "max": competency_stats.get("max", 0.0),
            "std": competency_stats.get("std", 0.0),
        },
    }


def get_demographic_extremes(eval_data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Get demographic groups with highest and lowest medians for warmth and competency."""
    demographic_groups = eval_data.get("evaluation_results", {}).get(
        "demographic_groups", {}
    )

    if not demographic_groups:
        return {
            "warmth": {"highest": "N/A", "lowest": "N/A"},
            "competency": {"highest": "N/A", "lowest": "N/A"},
        }

    # Extract medians for each group
    warmth_medians = {}
    competency_medians = {}

    for group_name, group_data in demographic_groups.items():
        # Format group name (e.g., "female_asian" -> "FA")
        parts = group_name.split("_")
        formatted_name = "".join(part[0].upper() for part in parts if part)

        warmth_medians[formatted_name] = group_data.get("warmth", {}).get("median", 0.0)
        competency_medians[formatted_name] = group_data.get("competency", {}).get(
            "median", 0.0
        )

    # Find highest and lowest
    warmth_highest = (
        max(warmth_medians, key=warmth_medians.get) if warmth_medians else "N/A"
    )
    warmth_lowest = (
        min(warmth_medians, key=warmth_medians.get) if warmth_medians else "N/A"
    )

    competency_highest = (
        max(competency_medians, key=competency_medians.get)
        if competency_medians
        else "N/A"
    )
    competency_lowest = (
        min(competency_medians, key=competency_medians.get)
        if competency_medians
        else "N/A"
    )

    return {
        "warmth": {"highest": warmth_highest, "lowest": warmth_lowest},
        "competency": {"highest": competency_highest, "lowest": competency_lowest},
    }


def is_maximum_value(
    value: float, all_stats: List[Dict], metric: str, stat: str
) -> bool:
    """Check if a value is the maximum among all projects for a given metric and statistic."""
    all_values = [stats[metric][stat] for stats in all_stats]
    return value == max(all_values)


def is_minimum_value(
    value: float, all_stats: List[Dict], metric: str, stat: str
) -> bool:
    """Check if a value is the maximum among all projects for a given metric and statistic."""
    all_values = [stats[metric][stat] for stats in all_stats]
    return value == min(all_values)


def generate_warmth_row(
    project_name: str, eval_data: Dict[str, Any], all_project_stats: List[Dict]
) -> str:
    """Generate warmth table row with highest/lowest demographic groups."""
    model_name = get_model_name(project_name)
    general_stats = get_general_statistics(eval_data)
    demographic_extremes = get_demographic_extremes(eval_data)

    n_samples = eval_data.get("evaluation_results", {}).get("n_samples", 0)

    warmth = general_stats["warmth"]
    extremes = demographic_extremes["warmth"]

    # Determine if values should be marked as maximum across all projects
    median_mark = (
        "\\textbf{"
        if is_maximum_value(warmth["median"], all_project_stats, "warmth", "median")
        else ""
    )
    median_close = "}" if median_mark else ""

    min_mark = (
        "\\textbf{"
        if is_minimum_value(warmth["min"], all_project_stats, "warmth", "min")
        else ""
    )
    min_close = "}" if min_mark else ""

    max_mark = (
        "\\textbf{"
        if is_maximum_value(warmth["max"], all_project_stats, "warmth", "max")
        else ""
    )
    max_close = "}" if max_mark else ""

    std_mark = (
        "\\textbf{"
        if is_maximum_value(warmth["std"], all_project_stats, "warmth", "std")
        else ""
    )
    std_close = "}" if std_mark else ""

    return (
        f"{model_name} & "
        f"{n_samples} & "
        f"{median_mark}{warmth['median']:.3f}{median_close} & "
        f"{min_mark}{warmth['min']:.3f}{min_close} & "
        f"{max_mark}{warmth['max']:.3f}{max_close} & "
        f"{std_mark}{warmth['std']:.3f}{std_close} & "
        f"{extremes['highest']} & "
        f"{extremes['lowest']} \\\\"
    )


def generate_competency_row(
    project_name: str, eval_data: Dict[str, Any], all_project_stats: List[Dict]
) -> str:
    """Generate competency table row with highest/lowest demographic groups."""
    model_name = get_model_name(project_name)
    general_stats = get_general_statistics(eval_data)
    demographic_extremes = get_demographic_extremes(eval_data)

    n_samples = eval_data.get("evaluation_results", {}).get("n_samples", 0)

    competency = general_stats["competency"]
    extremes = demographic_extremes["competency"]

    # Determine if values should be marked as maximum across all projects
    median_mark = (
        "\\textbf{"
        if is_maximum_value(
            competency["median"], all_project_stats, "competency", "median"
        )
        else ""
    )
    median_close = "}" if median_mark else ""

    min_mark = (
        "\\textbf{"
        if is_maximum_value(competency["min"], all_project_stats, "competency", "min")
        else ""
    )
    min_close = "}" if min_mark else ""

    max_mark = (
        "\\textbf{"
        if is_maximum_value(competency["max"], all_project_stats, "competency", "max")
        else ""
    )
    max_close = "}" if max_mark else ""

    std_mark = (
        "\\textbf{"
        if is_maximum_value(competency["std"], all_project_stats, "competency", "std")
        else ""
    )
    std_close = "}" if std_mark else ""

    return (
        f"{model_name} & "
        f"{n_samples} & "
        f"{median_mark}{competency['median']:.3f}{median_close} & "
        f"{min_mark}{competency['min']:.3f}{min_close} & "
        f"{max_mark}{competency['max']:.3f}{max_close} & "
        f"{std_mark}{competency['std']:.3f}{std_close} & "
        f"{extremes['highest']} & "
        f"{extremes['lowest']} \\\\"
    )


def main():
    """Main function to generate LaTeX table rows."""
    parser = argparse.ArgumentParser(
        description="Generate separate LaTeX tables for warmth and competency from evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "project_names", nargs="+", help="Names of projects to generate LaTeX rows for"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    try:
        # Load all evaluation data first to determine maxima
        all_eval_data = []
        all_project_stats = []

        for project_name in args.project_names:
            try:
                eval_data = load_eval_file(project_name)
                all_eval_data.append((project_name, eval_data))
                all_project_stats.append(get_general_statistics(eval_data))
            except Exception as e:
                logger.error(f"Error loading project '{project_name}': {e}")
                continue

        if not all_eval_data:
            logger.error("No valid evaluation data found for any project")
            return 1

        # Generate warmth table
        print("% LaTeX table rows for WARMTH evaluation results")
        print(
            "% Format: Model & N & Median & Min & Max & Std & Highest Group & Lowest Group \\\\"
        )
        print()
        print("% WARMTH TABLE")
        for project_name, eval_data in all_eval_data:
            try:
                row = generate_warmth_row(project_name, eval_data, all_project_stats)
                print(row)
            except Exception as e:
                logger.error(f"Error generating warmth row for '{project_name}': {e}")
                continue

        print()
        print("% COMPETENCY TABLE")
        print(
            "% Format: Model & N & Median & Min & Max & Std & Highest Group & Lowest Group \\\\"
        )
        print()

        # Generate competency table
        for project_name, eval_data in all_eval_data:
            try:
                row = generate_competency_row(
                    project_name, eval_data, all_project_stats
                )
                print(row)
            except Exception as e:
                logger.error(
                    f"Error generating competency row for '{project_name}': {e}"
                )
                continue

    except Exception as e:
        logger.error(f"Error during LaTeX generation: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
