"""
Visualization tools for LLM bias evaluation results.

Creates scatter plots and heatmaps from evaluation JSON files.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")


def load_results(json_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def create_bias_heatmap(results: Dict[str, Any], output_path: str = "bias_heatmap.png", model_idx: int = None):
    """
    Create a heatmap showing bias patterns across demographic groups.

    Args:
        results: Evaluation results dictionary
        output_path: Path to save the plot
        model_idx: Index of the embedding model to visualize (None for combined view)
    """
    # Support both old and new demographic group structures
    demographic_groups = (
        results.get("demographic_groups", {}) or
        results.get("aggregated_analysis", {}).get("by_demographic_group", {}) or
        results.get("evaluation_results", {}).get("demographic_groups", {})
    )

    if not demographic_groups:
        logger.warning("No demographic groups found in results")
        return

    # Extract data for heatmap
    genders = []
    ethnicities = []
    warmth_scores = []
    competency_scores = []
    bias_gaps = []

    for group_name, group_data in demographic_groups.items():
        gender, ethnicity = group_name.split('_', 1)
        genders.append(gender)
        ethnicities.append(ethnicity)
        # Handle multi-model results
        if isinstance(group_data["warmth"]["mean"], list) and model_idx is not None:
            warmth_scores.append(group_data["warmth"]["mean"][model_idx])
            competency_scores.append(group_data["competency"]["mean"][model_idx])
        elif isinstance(group_data["warmth"]["mean"], list):
            # Use first model if no specific index given
            warmth_scores.append(group_data["warmth"]["mean"][0])
            competency_scores.append(group_data["competency"]["mean"][0])
        else:
            # Legacy single-model format
            warmth_scores.append(group_data["warmth"]["mean"])
            competency_scores.append(group_data["competency"]["mean"])
        bias_gaps.append(group_data["bias_metrics"]["warmth_competency_gap"])

    # Create matrices for heatmap
    unique_genders = sorted(list(set(genders)))
    unique_ethnicities = sorted(list(set(ethnicities)))

    # Initialize with NaN to handle missing combinations
    warmth_matrix = np.full((len(unique_genders), len(unique_ethnicities)), np.nan)
    competency_matrix = np.full((len(unique_genders), len(unique_ethnicities)), np.nan)
    bias_matrix = np.full((len(unique_genders), len(unique_ethnicities)), np.nan)

    for i, (gender, ethnicity, warmth, competency, bias) in enumerate(zip(genders, ethnicities, warmth_scores, competency_scores, bias_gaps)):
        g_idx = unique_genders.index(gender)
        e_idx = unique_ethnicities.index(ethnicity)
        warmth_matrix[g_idx, e_idx] = warmth
        competency_matrix[g_idx, e_idx] = competency
        bias_matrix[g_idx, e_idx] = bias

    # Create subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Warmth heatmap
    sns.heatmap(warmth_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1,
                xticklabels=[e.replace('_', ' ').title() for e in unique_ethnicities],
                yticklabels=[g.title() for g in unique_genders],
                ax=axes[0], cbar_kws={'label': 'Warmth Score (-1 to +1)'})
    axes[0].set_title('Warmth Scores by Demographics\n(-1.0 = Cold, +1.0 = Warm)')
    axes[0].set_xlabel('Ethnicity')
    axes[0].set_ylabel('Gender')

    # Competency heatmap
    sns.heatmap(competency_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1,
                xticklabels=[e.replace('_', ' ').title() for e in unique_ethnicities],
                yticklabels=[g.title() for g in unique_genders],
                ax=axes[1], cbar_kws={'label': 'Competency Score (-1 to +1)'})
    axes[1].set_title('Competency Scores by Demographics\n(-1.0 = Incompetent, +1.0 = Competent)')
    axes[1].set_xlabel('Ethnicity')
    axes[1].set_ylabel('Gender')

    # Bias gap heatmap
    sns.heatmap(bias_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-2, vmax=2,  # Gap can range from -2 to +2
                xticklabels=[e.replace('_', ' ').title() for e in unique_ethnicities],
                yticklabels=[g.title() for g in unique_genders],
                ax=axes[2], cbar_kws={'label': 'Warmth - Competency Gap'})
    axes[2].set_title('Bias Gap by Demographics\n(Warmth - Competency)\nPositive = More Warm than Competent')
    axes[2].set_xlabel('Ethnicity')
    axes[2].set_ylabel('Gender')

    # Get model name for title
    if "embedding_models" in results and model_idx is not None:
        model_name = results["embedding_models"][model_idx].get("model_name", "Unknown")
        title_suffix = f" - Model {model_idx}: {model_name}"
    elif "embedding_models" in results:
        model_name = results["embedding_models"][0].get("model_name", "Unknown")
        title_suffix = f" - Primary Model: {model_name}"
    else:
        model_name = results.get("metadata", {}).get("experiment_info", {}).get("model_info", {}).get("model_name", "Unknown")
        title_suffix = f" - {model_name}"

    fig.suptitle(f'Bias Analysis Heatmaps{title_suffix}\nScore Range: -1.0 to +1.0', fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Bias heatmap saved to {output_path}")
    plt.show()


def create_scatter_plot(results: Dict[str, Any], output_path: str = "scatter_plot.png", model_idx: int = None):
    """
    Create a scatter plot of warmth vs competency scores.

    Args:
        results: Evaluation results dictionary
        output_path: Path to save the plot
        model_idx: Index of the embedding model to visualize (None for combined view)
    """
    # Support both old and new demographic group structures
    demographic_groups = (
        results.get("demographic_groups", {}) or
        results.get("aggregated_analysis", {}).get("by_demographic_group", {}) or
        results.get("evaluation_results", {}).get("demographic_groups", {})
    )

    if not demographic_groups:
        logger.warning("No demographic groups found in results")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color mapping by gender, shape by ethnicity
    gender_colors = {'male': 'blue', 'female': 'red'}
    ethnicity_markers = {'white': 'o', 'black': 's', 'hispanic': '^', 'asian': 'D', 'middle_eastern': 'v'}

    for group_name, group_data in demographic_groups.items():
        gender, ethnicity = group_name.split('_', 1)

        # Handle multi-model results
        if isinstance(group_data["warmth"]["mean"], list) and model_idx is not None and model_idx < len(group_data["warmth"]["mean"]):
            warmth_mean = group_data["warmth"]["mean"][model_idx]
            competency_mean = group_data["competency"]["mean"][model_idx]
        elif isinstance(group_data["warmth"]["mean"], list):
            # Use first model if no specific index given
            warmth_mean = group_data["warmth"]["mean"][0]
            competency_mean = group_data["competency"]["mean"][0]
        else:
            # Legacy single-model format
            warmth_mean = group_data["warmth"]["mean"]
            competency_mean = group_data["competency"]["mean"]
        # Support both old and new sample count structures
        sample_count = group_data.get("sample_count") or group_data.get("demographic_info", {}).get("n_samples", 1)

        color = gender_colors.get(gender, 'gray')
        marker = ethnicity_markers.get(ethnicity, 'o')

        # Plot with size proportional to sample count
        size = max(80, sample_count * 15)  # Slightly larger points for better visibility
        ax.scatter(warmth_mean, competency_mean,
                  s=size, c=color, marker=marker, alpha=0.7,
                  edgecolors='black', linewidth=0.5,
                  label=f"{gender.title()} {ethnicity.replace('_', ' ').title()}")

        # Add text labels
        ax.annotate(f"{gender[0].upper()}{ethnicity[0].upper()}",
                   (warmth_mean, competency_mean),
                   xytext=(3, 3), textcoords='offset points',
                   fontsize=9, alpha=0.9, fontweight='bold')

    # Add diagonal line (warmth = competency)
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, linewidth=1, label='Warmth = Competency')

    # Add axis lines at zero
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)

    # Customize plot
    ax.set_xlabel('Warmth Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Competency Score', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)

    # Add quadrant labels
    ax.text(-0.5, 0.5, 'High Competency\nLow Warmth', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.5, 0.5, 'High Competency\nHigh Warmth', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(-0.5, -0.5, 'Low Competency\nLow Warmth', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax.text(0.5, -0.5, 'Low Competency\nHigh Warmth', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Get model name for title
    if "embedding_models" in results and model_idx is not None:
        model_name = results["embedding_models"][model_idx].get("model_name", "Unknown")
        title_suffix = f" - Model {model_idx}: {model_name}"
    elif "embedding_models" in results:
        model_name = results["embedding_models"][0].get("model_name", "Unknown")
        title_suffix = f" - Primary Model: {model_name}"
    else:
        model_name = results.get("metadata", {}).get("experiment_info", {}).get("model_info", {}).get("model_name", "Unknown")
        title_suffix = f" - {model_name}"

    total_scenarios = results.get("metadata", {}).get("experiment_info", {}).get("total_scenarios", 0)
    n_samples = results.get("n_samples", 0)

    plt.title(f'Warmth vs Competency Scatter Plot{title_suffix}\n({max(total_scenarios, n_samples)} samples)\nScores range from -1.0 (negative) to +1.0 (positive)',
             fontsize=13, fontweight='bold')

    # Create custom legend
    gender_handles = [plt.scatter([], [], c=color, s=100, label=gender.title())
                     for gender, color in gender_colors.items()]
    ethnicity_handles = [plt.scatter([], [], c='gray', marker=marker, s=100, label=ethnicity.replace('_', ' ').title())
                         for ethnicity, marker in ethnicity_markers.items()]

    gender_legend = ax.legend(handles=gender_handles, title='Gender',
                             loc='upper left', bbox_to_anchor=(1.02, 1))
    ethnicity_legend = ax.legend(handles=ethnicity_handles, title='Ethnicity',
                           loc='upper left', bbox_to_anchor=(1.02, 0.6))
    ax.add_artist(gender_legend)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Scatter plot saved to {output_path}")
    plt.show()


def create_combined_scatter_plot(results: Dict[str, Any], output_path: str = "combined_scatter_plot.png"):
    """
    Create a combined scatter plot showing all embedding models together.

    Args:
        results: Evaluation results dictionary
        output_path: Path to save the plot
    """
    if "embedding_models" not in results:
        logger.warning("No embedding models found in results - using single model visualization")
        create_scatter_plot(results, output_path)
        return

    # Support both old and new demographic group structures
    demographic_groups = (
        results.get("demographic_groups", {}) or
        results.get("aggregated_analysis", {}).get("by_demographic_group", {}) or
        results.get("evaluation_results", {}).get("demographic_groups", {})
    )

    if not demographic_groups:
        logger.warning("No demographic groups found in results")
        return

    num_models = len(results["embedding_models"])
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color mapping by model, shape by demographic group
    model_colors = plt.cm.Set1(np.linspace(0, 1, num_models))
    ethnicity_markers = {'white': 'o', 'black': 's', 'hispanic': '^', 'asian': 'D', 'middle_eastern': 'v'}
    gender_alpha = {'male': 1.0, 'female': 0.6, 'non-binary': 0.8}

    for model_idx in range(num_models):
        model_name = results["embedding_models"][model_idx].get("model_name", f"Model {model_idx}")

        for group_name, group_data in demographic_groups.items():
            gender, ethnicity = group_name.split('_', 1)

            # Handle array structure for multi-model data
            if isinstance(group_data["warmth"]["mean"], list) and model_idx < len(group_data["warmth"]["mean"]):
                warmth_mean = group_data["warmth"]["mean"][model_idx]
                competency_mean = group_data["competency"]["mean"][model_idx]
            else:
                # Skip this group if model_idx is out of bounds or not array format
                continue
            # Support both old and new sample count structures
            sample_count = group_data.get("sample_count") or group_data.get("demographic_info", {}).get("n_samples", 1)

            color = model_colors[model_idx]
            marker = ethnicity_markers.get(ethnicity, 'o')
            alpha = gender_alpha.get(gender, 0.7)

            # Plot with size proportional to sample count
            size = max(80, sample_count * 10)
            ax.scatter(warmth_mean, competency_mean,
                      s=size, c=[color], marker=marker, alpha=alpha,
                      edgecolors='black', linewidth=0.5,
                      label=f"{model_name} - {gender.title()} {ethnicity.replace('_', ' ').title()}" if model_idx == 0 else "")

    # Add diagonal line and axis lines
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, linewidth=1, label='Warmth = Competency')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)

    # Customize plot
    ax.set_xlabel('Warmth Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Competency Score', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)

    # Add quadrant labels
    ax.text(-0.5, 0.5, 'High Competency\nLow Warmth', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.5, 0.5, 'High Competency\nHigh Warmth', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(-0.5, -0.5, 'Low Competency\nLow Warmth', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax.text(0.5, -0.5, 'Low Competency\nHigh Warmth', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    n_samples = results.get("n_samples", 0)
    plt.title(f'Combined Warmth vs Competency Scatter Plot\n{num_models} Embedding Models ({n_samples} samples)\nScores range from -1.0 (negative) to +1.0 (positive)',
             fontsize=13, fontweight='bold')

    # Create legends for models and demographics
    model_handles = [plt.scatter([], [], c=model_colors[i], s=100, label=results["embedding_models"][i].get("model_name", f"Model {i}"))
                    for i in range(num_models)]
    ethnicity_handles = [plt.scatter([], [], c='gray', marker=marker, s=100, label=ethnicity.replace('_', ' ').title())
                         for ethnicity, marker in ethnicity_markers.items()]

    model_legend = ax.legend(handles=model_handles, title='Embedding Models',
                            loc='upper left', bbox_to_anchor=(1.02, 1))
    ethnicity_legend = ax.legend(handles=ethnicity_handles, title='Ethnicity',
                           loc='upper left', bbox_to_anchor=(1.02, 0.6))
    ax.add_artist(model_legend)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Combined scatter plot saved to {output_path}")
    plt.show()


def create_individual_model_plots(results: Dict[str, Any], output_dir: str = "visualizations"):
    """
    Create individual plots for each embedding model.

    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save plots
    """
    if "embedding_models" not in results:
        logger.warning("No embedding models found in results - creating single model plots")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        create_bias_heatmap(results, str(output_path / "bias_heatmap.png"))
        create_scatter_plot(results, str(output_path / "scatter_plot.png"))
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    num_models = len(results["embedding_models"])

    for model_idx in range(num_models):
        model_info = results["embedding_models"][model_idx]
        model_name = model_info.get("model_name", f"model_{model_idx}")
        model_type = model_info.get("adapter_type", "unknown")

        # Sanitize model name for filename
        safe_model_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")

        logger.info(f"Creating plots for model {model_idx}: {model_name}")

        # Create heatmap for this model
        heatmap_path = output_path / f"bias_heatmap_model_{model_idx}_{safe_model_name}.png"
        create_bias_heatmap(results, str(heatmap_path), model_idx=model_idx)

        # Create scatter plot for this model
        scatter_path = output_path / f"scatter_plot_model_{model_idx}_{safe_model_name}.png"
        create_scatter_plot(results, str(scatter_path), model_idx=model_idx)


def create_all_visualizations(json_file: str, output_dir: str = "visualizations"):
    """
    Create all available visualizations from a results JSON file.

    Args:
        json_file: Path to the evaluation results JSON file
        output_dir: Directory to save visualization files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load results
    logger.info(f"Loading results from {json_file}")
    results = load_results(json_file)

    # Check if we have multi-model results
    if "embedding_models" in results and len(results["embedding_models"]) > 1:
        logger.info("Multi-model results detected - creating comprehensive visualizations")

        # Create combined visualizations
        logger.info("Creating combined scatter plot...")
        create_combined_scatter_plot(results, str(output_path / "combined_scatter_plot.png"))

        # Create individual model plots
        logger.info("Creating individual model plots...")
        create_individual_model_plots(results, output_dir)

        # Also create primary model plots for backwards compatibility
        logger.info("Creating primary model plots...")
        create_bias_heatmap(results, str(output_path / "bias_heatmap_primary.png"))
        create_scatter_plot(results, str(output_path / "scatter_plot_primary.png"))

    else:
        logger.info("Single model results - creating standard visualizations")
        # Create standard visualizations
        logger.info("Creating bias heatmap...")
        create_bias_heatmap(results, str(output_path / "bias_heatmap.png"))

        logger.info("Creating scatter plot...")
        create_scatter_plot(results, str(output_path / "scatter_plot.png"))

    logger.info(f"All visualizations saved to {output_dir}/")


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description="Create visualizations from LLM bias evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "json_file",
        help="Path to the evaluation results JSON file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization files"
    )

    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["heatmap", "scatter", "all"],
        default="all",
        help="Type of plot to create"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not Path(args.json_file).exists():
        logger.error(f"JSON file not found: {args.json_file}")
        return

    try:
        # Load results
        results = load_results(args.json_file)

        # Auto-detect project structure and set output directory
        json_file_path = Path(args.json_file)
        if args.output_dir == "visualizations" and "results" in json_file_path.parts:
            # Check if this is a project-based evaluation file
            if json_file_path.parent.name != "results" and json_file_path.name.startswith("eval_"):
                # This is a project-based evaluation file: results/<project>/eval_*.json
                project_dir = json_file_path.parent
                output_dir = project_dir / "visualizations"
                logger.info(f"Auto-detected project structure, using output directory: {output_dir}")
            else:
                output_dir = Path(args.output_dir)
        else:
            output_dir = Path(args.output_dir)

        # Create output directory
        output_dir.mkdir(exist_ok=True)

        # Create requested visualizations
        if args.plot_type == "all":
            create_all_visualizations(args.json_file, str(output_dir))
        elif args.plot_type == "heatmap":
            create_bias_heatmap(results, str(output_dir / "bias_heatmap.png"))
        elif args.plot_type == "scatter":
            create_scatter_plot(results, str(output_dir / "scatter_plot.png"))

        logger.info(f"All visualizations saved to {output_dir}/")

        logger.info("Visualization complete!")

    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        raise


if __name__ == "__main__":
    main()
