#!/usr/bin/env python3
"""
Convert old normalized evaluation results to raw score format.

This script converts evaluation results from the old format (0-1 normalized scores)
to the new format (-1 to +1 raw scores) by copying raw_score to score and updating metadata.
"""

import json
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_evaluation_result(evaluation):
    """Convert a single evaluation result from normalized to raw format."""
    if "results" not in evaluation:
        return evaluation

    results = evaluation["results"]

    # Convert warmth scores
    if "warmth" in results and "raw_score" in results["warmth"]:
        results["warmth"]["score"] = results["warmth"]["raw_score"]
        del results["warmth"]["raw_score"]

    # Convert competency scores
    if "competency" in results and "raw_score" in results["competency"]:
        results["competency"]["score"] = results["competency"]["raw_score"]
        del results["competency"]["raw_score"]

    # Update bias indicators to use raw scores
    if "bias_indicators" in results:
        warmth_score = results.get("warmth", {}).get("score", 0.0)
        competency_score = results.get("competency", {}).get("score", 0.0)

        results["bias_indicators"]["warmth_competency_gap"] = warmth_score - competency_score
        results["bias_indicators"]["overall_favorability"] = (warmth_score + competency_score) / 2

    return evaluation

def update_evaluator_info(evaluator_info):
    """Update evaluator_info to reflect new score ranges."""
    if "warmth" in evaluator_info:
        evaluator_info["warmth"] = "Semantic warmth score based on proximity to warmth anchor sentences (-1.0 to +1.0)"

    if "competency" in evaluator_info:
        evaluator_info["competency"] = "Semantic competency score based on proximity to competency anchor sentences (-1.0 to +1.0)"

    return evaluator_info

def update_aggregated_analysis(aggregated_analysis):
    """Update aggregated analysis to reflect new score ranges."""
    if "overall_statistics" in aggregated_analysis:
        stats = aggregated_analysis["overall_statistics"]

        # Update warmth distribution ranges
        if "warmth_distribution" in stats and "range" in stats["warmth_distribution"]:
            current_range = stats["warmth_distribution"]["range"]
            if len(current_range) == 2:
                # Convert from 0-1 range to -1 to +1 range
                old_min, old_max = current_range
                new_min = (old_min * 2) - 1  # Convert 0-1 to -1 to +1
                new_max = (old_max * 2) - 1
                stats["warmth_distribution"]["range"] = [new_min, new_max]

        # Update competency distribution ranges
        if "competency_distribution" in stats and "range" in stats["competency_distribution"]:
            current_range = stats["competency_distribution"]["range"]
            if len(current_range) == 2:
                # Convert from 0-1 range to -1 to +1 range
                old_min, old_max = current_range
                new_min = (old_min * 2) - 1  # Convert 0-1 to -1 to +1
                new_max = (old_max * 2) - 1
                stats["competency_distribution"]["range"] = [new_min, new_max]

    # Update demographic group statistics
    if "by_demographic_group" in aggregated_analysis:
        for group_key, group_data in aggregated_analysis["by_demographic_group"].items():
            # Convert warmth scores
            if "warmth" in group_data:
                warmth = group_data["warmth"]
                if "mean" in warmth:
                    warmth["mean"] = (warmth["mean"] * 2) - 1
                if "std" in warmth:
                    warmth["std"] = warmth["std"] * 2  # Scale standard deviation
                if "min" in warmth:
                    warmth["min"] = (warmth["min"] * 2) - 1
                if "max" in warmth:
                    warmth["max"] = (warmth["max"] * 2) - 1

            # Convert competency scores
            if "competency" in group_data:
                competency = group_data["competency"]
                if "mean" in competency:
                    competency["mean"] = (competency["mean"] * 2) - 1
                if "std" in competency:
                    competency["std"] = competency["std"] * 2  # Scale standard deviation
                if "min" in competency:
                    competency["min"] = (competency["min"] * 2) - 1
                if "max" in competency:
                    competency["max"] = (competency["max"] * 2) - 1

            # Update bias metrics
            if "bias_metrics" in group_data:
                bias = group_data["bias_metrics"]
                warmth_mean = group_data.get("warmth", {}).get("mean", 0.0)
                competency_mean = group_data.get("competency", {}).get("mean", 0.0)

                bias["warmth_competency_gap"] = warmth_mean - competency_mean
                bias["overall_favorability"] = (warmth_mean + competency_mean) / 2

    return aggregated_analysis

def convert_results_file(input_file, output_file=None):
    """Convert a results file from normalized to raw score format."""

    if output_file is None:
        output_file = input_file

    logger.info(f"Converting {input_file} to raw score format...")

    # Load the file
    with open(input_file, 'r') as f:
        data = json.load(f)

    converted_evaluations = 0

    # Process scenarios
    if "scenarios" in data:
        for scenario in data["scenarios"]:
            if "outputs" in scenario:
                for output in scenario["outputs"]:
                    if "evaluations" in output:
                        for evaluation in output["evaluations"]:
                            convert_evaluation_result(evaluation)
                            converted_evaluations += 1

    # Update metadata
    if "metadata" in data and "experiment_info" in data["metadata"]:
        if "evaluator_info" in data["metadata"]["experiment_info"]:
            update_evaluator_info(data["metadata"]["experiment_info"]["evaluator_info"])

    # Update aggregated analysis
    if "aggregated_analysis" in data:
        update_aggregated_analysis(data["aggregated_analysis"])

    # Save the updated file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Converted {converted_evaluations} evaluations")
    logger.info(f"Updated file saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert evaluation results from normalized (0-1) to raw (-1 to +1) format"
    )

    parser.add_argument(
        "input_file",
        help="Path to the input results JSON file"
    )

    parser.add_argument(
        "--output-file", "-o",
        help="Path for output file (defaults to overwriting input file)"
    )

    parser.add_argument(
        "--backup", "-b",
        action="store_true",
        help="Create a backup of the original file"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input_file}")
        return 1

    # Create backup if requested
    if args.backup:
        backup_path = input_path.with_suffix(f"{input_path.suffix}.backup")
        logger.info(f"Creating backup: {backup_path}")
        with open(input_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())

    try:
        convert_results_file(args.input_file, args.output_file)
        logger.info("Conversion completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
