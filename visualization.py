"""
Simplified visualization tool for LLM bias evaluation results.
Creates combined scatter plots from multiple project evaluation results.
Reads eval_scm.json files from results/<project-name>/ directories.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def get_model_name(project_name: str) -> str:
    """Extract model type from responses.json in project directory."""
    project_dir = Path("results") / project_name
    responses_file = project_dir / "responses.json"

    if not responses_file.exists():
        logger.warning(f"responses.json not found in {project_dir}, using fallback")
        return project_name.title()

    with open(responses_file, "r") as f:
        responses_data = json.load(f)

    # Get model type from responses metadata
    try:
        model_type = responses_data["metadata"]["experiment_info"]["model_info"][
            "model_name"
        ]
        return model_type.split("/")[-1]
    except KeyError:
        logger.warning("Could not find model_type in responses.json, using fallback")
        return project_name.title()


def load_results_from_projects(project_names: List[str]) -> List[Dict[str, Any]]:
    """Load evaluation results from multiple project directories."""
    results = []
    for project_name in project_names:
        # Construct path following the project structure: results/<project-name>/eval_scm.json
        project_dir = Path("results") / project_name
        eval_file = project_dir / "eval_scm.json"

        if not eval_file.exists():
            logger.warning(f"Evaluation file not found: {eval_file}")
            continue

        with open(eval_file, "r") as f:
            data = json.load(f)
            data["_source_file"] = project_name  # Use project name as identifier
            results.append(data)

    logger.info(
        f"Successfully loaded {len(results)} evaluation files from {len(project_names)} projects"
    )
    return results


def get_demographic_groups(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract demographic groups from results with fallback for different formats."""
    return (
        results.get("demographic_groups", {})
        or results.get("aggregated_analysis", {}).get("by_demographic_group", {})
        or results.get("evaluation_results", {}).get("demographic_groups", {})
    )


def calculate_global_mean(all_results: List[Dict[str, Any]]) -> tuple:
    """
    Calculate the global mean warmth and competency across all models and demographic groups.

    Args:
        all_results: List of evaluation results from different projects

    Returns:
        Tuple of (global_warmth_mean, global_competency_mean)
    """
    all_warmth_scores = []
    all_competency_scores = []

    for results in all_results:
        demographic_groups = get_demographic_groups(results)
        if not demographic_groups:
            continue

        for group_name, group_data in demographic_groups.items():
            # Extract scores (handle both single and multi-model formats)
            if isinstance(group_data["warmth"]["median"], list):
                warmth_median = group_data["warmth"]["median"][0]
                competency_median = group_data["competency"]["median"][0]
            else:
                warmth_median = group_data["warmth"]["median"]
                competency_median = group_data["competency"]["median"]

            all_warmth_scores.append(warmth_median)
            all_competency_scores.append(competency_median)

    global_warmth_mean = np.mean(all_warmth_scores) if all_warmth_scores else 0.0
    global_competency_mean = (
        np.mean(all_competency_scores) if all_competency_scores else 0.0
    )

    return global_warmth_mean, global_competency_mean


def create_single_model_zoomed_plot(
    results: Dict[str, Any],
    model_name: str,
    output_path: str,
):
    """
    Create a zoomed scatter plot for a single model centered on that model's own mean.

    Args:
        results: Evaluation results for the single model
        model_name: Name of the model for the title
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Ethnicity markers (same as main plot)
    ethnicity_markers = {
        "white": "o",
        "black": "s",
        "hispanic": "^",
        "asian": "D",
        "middle_eastern": "v",
    }

    # Gender visual distinction (same as main plot)
    gender_patterns = {"male": None, "female": "///"}
    gender_alpha = {"male": 0.8, "female": 0.6}

    # Use a single color for this model (can use first color from colormap)
    model_color = plt.cm.Set1(0.1)

    demographic_groups = get_demographic_groups(results)
    if not demographic_groups:
        logger.warning(f"No demographic groups found for {model_name}")
        return

    # Calculate this model's mean warmth and competency
    model_warmth_scores = []
    model_competency_scores = []

    for group_name, group_data in demographic_groups.items():
        # Extract scores (handle both single and multi-model formats)
        if isinstance(group_data["warmth"]["median"], list):
            warmth_median = group_data["warmth"]["median"][0]
            competency_median = group_data["competency"]["median"][0]
        else:
            warmth_median = group_data["warmth"]["median"]
            competency_median = group_data["competency"]["median"]

        model_warmth_scores.append(warmth_median)
        model_competency_scores.append(competency_median)

    model_warmth_mean = np.mean(model_warmth_scores) if model_warmth_scores else 0.0
    model_competency_mean = (
        np.mean(model_competency_scores) if model_competency_scores else 0.0
    )

    for group_name, group_data in demographic_groups.items():
        gender, ethnicity = group_name.split("_", 1)

        # Extract scores (handle both single and multi-model formats)
        if isinstance(group_data["warmth"]["median"], list):
            warmth_median = group_data["warmth"]["median"][0]
            competency_median = group_data["competency"]["median"][0]
        else:
            warmth_median = group_data["warmth"]["median"]
            competency_median = group_data["competency"]["median"]

        # Visual styling (same as main plot)
        marker = ethnicity_markers.get(ethnicity, "o")
        alpha = gender_alpha.get(gender, 0.7)
        hatch = gender_patterns.get(gender)
        size = 120  # Slightly larger for zoomed view

        # Plot point
        ax.scatter(
            warmth_median,
            competency_median,
            s=size,
            color=model_color,
            marker=marker,
            alpha=alpha,
            hatch=hatch,
            edgecolors="black",
            linewidth=0.5,
        )

    # Configure zoomed plot (0.075 x 0.075 area around this model's mean)
    warmth_range = (model_warmth_mean - 0.0375, model_warmth_mean + 0.0375)
    competency_range = (
        model_competency_mean - 0.0375,
        model_competency_mean + 0.0375,
    )

    ax.set_xlim(warmth_range)
    ax.set_ylim(competency_range)
    ax.set_xlabel("Warmth Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Competency Score", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} - Zoomed View", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add reference lines at this model's mean (red dashed)
    ax.axhline(
        y=model_competency_mean,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label="Model Mean",
    )
    ax.axvline(
        x=model_warmth_mean, color="red", linestyle="--", alpha=0.7, linewidth=1.5
    )

    # Add reference lines at origin
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=0.8)
    ax.axvline(x=0, color="k", linestyle="-", alpha=0.3, linewidth=0.8)

    # Create ethnicity legend (no model legend for individual plots)
    ethnicity_handles = [
        plt.scatter(
            [],
            [],
            color="gray",
            marker=marker,
            s=100,
            label=ethnicity.replace("_", " ").title(),
        )
        for ethnicity, marker in ethnicity_markers.items()
    ]
    ethnicity_legend = ax.legend(
        handles=ethnicity_handles,
        title="Ethnicity",
        loc="upper right",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()  # Close the figure to free memory
    logger.info(f"Zoomed plot for {model_name} saved to {output_path}")


def create_multi_project_scatter_plot(
    all_results: List[Dict[str, Any]], output_path: str = "combined_scatter_plot.png"
):
    """
    Create a combined scatter plot from multiple project evaluation results.

    Args:
        all_results: List of evaluation results from different projects
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Generate distinct colors for each file
    file_colors = plt.cm.Set1(np.linspace(0, 1, len(all_results)))

    # Ethnicity markers
    ethnicity_markers = {
        "white": "o",
        "black": "s",
        "hispanic": "^",
        "asian": "D",
        "middle_eastern": "v",
    }

    # Gender visual distinction: male=solid, female=hatched
    gender_patterns = {"male": None, "female": "///"}
    gender_alpha = {"male": 0.8, "female": 0.6}

    file_handles = []  # For legend

    for file_idx, results in enumerate(all_results):
        demographic_groups = get_demographic_groups(results)
        if not demographic_groups:
            logger.warning(f"No demographic groups found in file {file_idx}")
            continue

        file_color = file_colors[file_idx]
        project_name = results.get("_source_file", f"Project {file_idx}")

        for group_name, group_data in demographic_groups.items():
            gender, ethnicity = group_name.split("_", 1)

            # Extract scores (handle both single and multi-model formats)
            if isinstance(group_data["warmth"]["median"], list):
                warmth_median = group_data["warmth"]["median"][0]
                competency_median = group_data["competency"]["median"][0]
            else:
                warmth_median = group_data["warmth"]["median"]
                competency_median = group_data["competency"]["median"]

            # Get sample count
            sample_count = group_data.get("sample_count") or group_data.get(
                "demographic_info", {}
            ).get("n_samples", 1)

            # Visual styling
            marker = ethnicity_markers.get(ethnicity, "o")
            alpha = gender_alpha.get(gender, 0.7)
            hatch = gender_patterns.get(gender)
            size = 100

            # Plot point
            scatter = ax.scatter(
                warmth_median,
                competency_median,
                s=size,
                color=file_color,
                marker=marker,
                alpha=alpha,
                hatch=hatch,
                edgecolors="black",
                linewidth=0.5,
            )

            # # Add text label
            # ax.annotate(
            #     f"{gender[0].upper()}{ethnicity[0].upper()}",
            #     (warmth_median, competency_median),
            #     xytext=(3, 3),
            #     textcoords="offset points",
            #     fontsize=8,
            #     alpha=0.9,
            # )

        # Add to legend (one entry per project)
        model_name = get_model_name(project_name).split("/")[-1]
        file_handles.append(
            plt.scatter([], [], color=file_color, s=100, label=model_name, alpha=0.8)
        )

    # Add reference lines
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=0.8)
    ax.axvline(x=0, color="k", linestyle="-", alpha=0.3, linewidth=0.8)

    # Customize plot
    ax.set_xlabel("Warmth Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Competency Score", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True, alpha=0.3)

    # Add quadrant labels
    quadrant_labels = [
        (-0.42, 0.75, "High Competency\nLow Warmth", "wheat"),
        (0.42, 0.75, "High Competency\nHigh Warmth", "lightgreen"),
        (-0.42, -0.75, "Low Competency\nLow Warmth", "lightcoral"),
        (0.42, -0.75, "Low Competency\nHigh Warmth", "lightblue"),
    ]

    for x, y, text, color in quadrant_labels:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.5),
        )

    # # Title and legend
    # plt.title(
    #     f"Combined SCM Warmth-Competency Analysis\n"
    #     f"Solid=Male, Hatched=Female | Shapes=Ethnicity",
    #     fontsize=13,
    #     fontweight="bold",
    # )

    # Create legends
    project_legend = ax.legend(
        handles=file_handles,
        title="Projects",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    ethnicity_handles = [
        plt.scatter(
            [],
            [],
            color="gray",
            marker=marker,
            s=100,
            label=ethnicity.replace("_", " ").title(),
        )
        for ethnicity, marker in ethnicity_markers.items()
    ]
    ethnicity_legend = ax.legend(
        handles=ethnicity_handles,
        title="Ethnicity",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.6),
    )
    ax.add_artist(project_legend)

    plt.tight_layout()
    plt.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
        bbox_extra_artists=[project_legend, ethnicity_legend],
        pad_inches=0.5,
    )
    logger.info(f"Combined scatter plot saved to {output_path}")
    # plt.show()


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description="Create combined visualizations from multiple LLM bias evaluation projects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "project_names",
        nargs="+",
        help="Names of projects to analyze (will read eval_scm.json from results/<project-name>/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/combined_scatter_plot.svg",
        help="Output path for the combined plot",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate project directories
    missing_projects = []
    for project_name in args.project_names:
        project_dir = Path("results") / project_name
        eval_file = project_dir / "eval_scm.json"
        if not eval_file.exists():
            missing_projects.append(project_name)

    if missing_projects:
        logger.error(f"Evaluation files not found for projects: {missing_projects}")
        logger.error(
            "Make sure to run evaluation first: python evaluators.py <project-name> scm"
        )
        return

    try:
        # Load all results from projects
        all_results = load_results_from_projects(args.project_names)

        if not all_results:
            logger.error("No evaluation results could be loaded")
            return

        # Create combined plot
        create_multi_project_scatter_plot(all_results, args.output)

        # Create individual zoomed plots for each model (centered on each model's own mean)
        output_dir = output_path.parent
        base_name = output_path.stem

        for results in all_results:
            project_name = results.get("_source_file", "unknown")
            model_name = get_model_name(project_name).split("/")[-1]

            # Create filename with model name
            zoomed_filename = f"{base_name}_{model_name}_zoomed.svg"
            zoomed_path = output_dir / zoomed_filename

            # Generate individual zoomed plot (centered on this model's mean)
            create_single_model_zoomed_plot(
                results,
                model_name,
                str(zoomed_path),
            )

        logger.info("Visualization complete!")
        logger.info(f"Generated {len(all_results)} individual zoomed plots")

    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        raise


if __name__ == "__main__":
    main()
