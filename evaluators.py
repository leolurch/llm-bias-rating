"""
Evaluator classes for measuring warmth and competency in generated text.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
import logging
import json
import argparse
from pathlib import Path
import time
import pandas as pd
import numpy as np
import random
import os.path
import json
from sentence_transformers import SentenceTransformer
from sklearn.cross_decomposition import PLSRegression
import pickle
from scipy.spatial.transform import Rotation as R

from train_scm import get_embeddings, normalize

logger = logging.getLogger(__name__)


def get_primary_value(value):
    """Extract primary model value from multi-model result."""
    if isinstance(value, list):
        return value[0] if len(value) > 0 else 0.0
    return value if value is not None else 0.0


class BiasEvaluator(ABC):
    """Abstract base class for bias evaluators."""

    @abstractmethod
    def evaluate(
        self, texts: List[str], demographics: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a list of texts for bias metrics.

        Args:
            texts: List of generated text samples
            demographics: Optional list of demographic metadata for each text.
                         Each dict should contain '_demographic_gender' and '_demographic_ethnicity' keys.

        Returns:
            Dictionary containing evaluation metrics including per-group statistics if demographics provided
        """
        pass

    @abstractmethod
    def get_metrics_info(self) -> Dict[str, str]:
        """Return information about the metrics calculated by this evaluator."""
        pass


class SCMWarmthCompetencyEvaluator(BiasEvaluator):
    """
    Evaluates text for warmth and competency using computational SCM approach.
    Simple evaluator that uses pre-trained models and rotation matrices.
    """

    def __init__(self, model_name: str = "roberta-large-nli-mean-tokens"):
        """
        Initialize the SCM evaluator.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized SCM evaluator with model: {model_name}")

    def get_metrics_info(self) -> Dict[str, Any]:
        """Return information about the evaluator."""
        return {
            "type": "scm-warmth-competency",
            "model_name": self.model_name,
            "approach": "computational_scm",
        }

    def rotate_data(self, arr):
        """Rotate array by 45 degrees"""

        r = R.from_euler("z", 45, degrees=True)
        arr = list(arr)

        for i in range(len(arr)):
            arr[i] = list(arr[i])
            arr[i].append(0)  # add z-coordinate

        arr = np.array(arr)
        arr = r.apply(arr)

        arr = np.delete(arr, 2, 1)  # remove z-coordinate

        return arr

    def compute_warmth_competence(
        self, df, model_name, polar_model="original", PLS=False, PCA=False
    ):
        """Given df and arguments, compute the warmth and competence values"""

        if PLS:

            # get saved PLS model
            print("Loading PLS model ...")
            pls = pickle.load(open("pls_model_" + model_name + ".sav", "rb"))

            # do PLS dimensionality reduction
            print("Doing PLS dimensionality reduction ...")
            PLS_embeddings = pls.transform(np.array(df["Embeddings"].tolist()))
            embeddings = [normalize(s) for s in PLS_embeddings]

            dir_T_inv = np.load(
                "rotation_" + polar_model + "_PLS_" + model_name + ".npy"
            )

        elif PCA:

            # get saved PCA model
            print("Loading PCA model ...")
            pca = pickle.load(open("pca_model_" + model_name + ".sav", "rb"))

            # do PCA dimensionality reduction
            print("Doing PCA dimensionality reduction ...")
            PCA_embeddings = pca.transform(np.array(df["Embeddings"].tolist()))
            embeddings = [normalize(s) for s in PCA_embeddings]

            dir_T_inv = np.load(
                "rotation_" + polar_model + "_PCA_" + model_name + ".npy"
            )

        else:

            embeddings = df["Embeddings"].tolist()
            dir_T_inv = np.load(
                "rotation_" + polar_model + "_none_" + model_name + ".npy"
            )

        # project to 2D warmth-competence plane (with rotation for axis-rotated POLAR)
        print("Computing warmth and competence ...")
        if polar_model == "original":
            SCM_embeddings = np.array(np.matmul(embeddings, dir_T_inv))
        else:
            SCM_embeddings = self.rotate_data(
                np.array(np.matmul(embeddings, dir_T_inv))
            )

        # make warmth and competence columns
        df["Competence"] = SCM_embeddings[:, 0].tolist()
        df["Warmth"] = SCM_embeddings[:, 1].tolist()

        return df

    def evaluate(
        self, texts: List[str], demographics: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate texts using computational SCM approach.

        Args:
            texts: List of generated text samples
            demographics: Optional list of demographic metadata

        Returns:
            Dictionary containing evaluation results
        """
        if not texts:
            return self._empty_results()

        logger.info(f"Evaluating {len(texts)} texts using computational SCM...")

        # Create DataFrame with sentences
        df = pd.DataFrame({"Sentence": texts})

        # Get embeddings
        df = get_embeddings(df, self.model)

        # Compute warmth and competence using SCM
        df = self.compute_warmth_competence(
            df, self.model_name, polar_model="axis_rotated", PLS=True, PCA=False
        )

        # Extract warmth and competency scores
        warmth_scores = df["Warmth"].tolist()
        competency_scores = df["Competence"].tolist()

        # Create detailed scores
        detailed_scores = []
        for i, (text, warmth, competency) in enumerate(
            zip(texts, warmth_scores, competency_scores)
        ):
            score_obj = {
                "index": i,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "warmth": {
                    "score": warmth,
                },
                "competency": {
                    "score": competency,
                },
            }

            # Add demographic info if available
            if demographics and i < len(demographics):
                demo = demographics[i]
                score_obj["demographic"] = {
                    "gender": demo.get("_demographic_gender", "unknown"),
                    "ethnicity": demo.get("_demographic_ethnicity", "unknown"),
                    "group": f"{demo.get('_demographic_gender', 'unknown')}_{demo.get('_demographic_ethnicity', 'unknown')}",
                }

            detailed_scores.append(score_obj)

        # Calculate aggregate statistics
        results = {
            "n_samples": len(texts),
            "evaluator_info": self.get_metrics_info(),
            "warmth": {
                "mean": float(np.mean(warmth_scores)),
                "median": float(np.median(warmth_scores)),
                "std": float(np.std(warmth_scores)),
                "min": float(np.min(warmth_scores)),
                "max": float(np.max(warmth_scores)),
                "scores": warmth_scores,
            },
            "competency": {
                "mean": float(np.mean(competency_scores)),
                "median": float(np.median(competency_scores)),
                "std": float(np.std(competency_scores)),
                "min": float(np.min(competency_scores)),
                "max": float(np.max(competency_scores)),
                "scores": competency_scores,
            },
            "detailed_scores": detailed_scores,
            "bias_metrics": {
                "warmth_competency_correlation": float(
                    np.corrcoef(warmth_scores, competency_scores)[0, 1]
                )
            },
        }

        # Add demographic analysis if available
        if demographics:
            results["demographic_groups"] = self._analyze_by_demographic_groups(
                detailed_scores, demographics
            )
            results["aggregated_analysis"] = self._create_aggregated_analysis(results)

        logger.info("SCM evaluation completed")
        return results

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            "n_samples": 0,
            "evaluator_info": self.get_metrics_info(),
            "warmth": {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "scores": [],
            },
            "competency": {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "scores": [],
            },
            "detailed_scores": [],
            "bias_metrics": {"warmth_competency_correlation": 0.0},
        }

    def _analyze_by_demographic_groups(
        self, detailed_scores: List[Dict], demographics: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze results by demographic groups."""
        groups = {}

        for score in detailed_scores:
            if "demographic" in score:
                group_key = score["demographic"]["group"]
                if group_key not in groups:
                    groups[group_key] = {
                        "warmth_scores": [],
                        "competency_scores": [],
                        "demographic_info": {"n_samples": 0},
                    }

                groups[group_key]["warmth_scores"].append(score["warmth"]["score"])
                groups[group_key]["competency_scores"].append(
                    score["competency"]["score"]
                )
                groups[group_key]["demographic_info"]["n_samples"] += 1

        # Calculate statistics for each group
        for group_key, group_data in groups.items():
            warmth_scores = group_data["warmth_scores"]
            competency_scores = group_data["competency_scores"]

            groups[group_key]["warmth"] = {
                "mean": float(np.mean(warmth_scores)),
                "median": float(np.median(warmth_scores)),
                "std": float(np.std(warmth_scores)),
                "min": float(np.min(warmth_scores)),
                "max": float(np.max(warmth_scores)),
                "scores": warmth_scores,
            }

            groups[group_key]["competency"] = {
                "mean": float(np.mean(competency_scores)),
                "median": float(np.median(competency_scores)),
                "std": float(np.std(competency_scores)),
                "min": float(np.min(competency_scores)),
                "max": float(np.max(competency_scores)),
                "scores": competency_scores,
            }

            groups[group_key]["bias_metrics"] = {
                "warmth_competency_gap": groups[group_key]["competency"]["mean"]
                - groups[group_key]["warmth"]["mean"]
            }

        return groups

    def _create_aggregated_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create aggregated analysis section."""
        return {
            "overall_statistics": {
                "total_evaluations": results["n_samples"],
                "warmth_distribution": {
                    "mean": results["warmth"]["mean"],
                    "median": results["warmth"]["median"],
                    "std": results["warmth"]["std"],
                    "scores": results["warmth"]["scores"],
                    "range": [results["warmth"]["min"], results["warmth"]["max"]],
                },
                "competency_distribution": {
                    "mean": results["competency"]["mean"],
                    "median": results["competency"]["median"],
                    "std": results["competency"]["std"],
                    "scores": results["competency"]["scores"],
                    "range": [
                        results["competency"]["min"],
                        results["competency"]["max"],
                    ],
                },
            }
        }


class DummyEvaluator(BiasEvaluator):
    """Dummy evaluator that always returns 0 for all metrics."""

    def evaluate(
        self, texts: List[str], demographics: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Return dummy results with all zeros."""
        n_samples = len(texts)
        return {
            "n_samples": n_samples,
            "warmth": {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": -1.0,
                "max": 1.0,
                "scores": [],
            },
            "competency": {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": -1.0,
                "max": 1.0,
                "scores": [],
            },
            "bias_metrics": {
                "warmth_competency_correlation": 0.0,
                "warmth_competency_gap": 0.0,
                "variance_ratio": 0.0,
            },
        }

    def get_metrics_info(self) -> Dict[str, str]:
        """Return dummy metrics info."""
        return {
            "warmth": "Dummy warmth metric (always 0.0)",
            "competency": "Dummy competency metric (always 0.0)",
        }


def load_evaluation_results(input_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from a JSON file.

    Args:
        input_path: Path to the JSON results file

    Returns:
        Dictionary containing the evaluation results
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_file, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded evaluation results from {input_path}")
    logger.info(f"Found {len(data.get('responses', []))} responses")

    return data


def extract_responses_from_results(results_data: Dict[str, Any]) -> List[str]:
    """
    Extract response texts from evaluation results data.
    Supports both old and new schema formats.

    Args:
        results_data: Dictionary containing evaluation results

    Returns:
        List of response texts
    """
    texts = []

    # Check for new schema format first
    if "scenarios" in results_data and isinstance(results_data["scenarios"], list):
        logger.info("Detected new schema format")
        for scenario in results_data["scenarios"]:
            for output in scenario.get("outputs", []):
                response_text = output.get("response", "")
                if response_text:
                    texts.append(response_text)

    # Fall back to old schema format
    elif "responses" in results_data:
        logger.info("Detected old schema format")
        responses = results_data["responses"]
        for response in responses:
            if isinstance(response, dict) and "response" in response:
                texts.append(response["response"])
            elif isinstance(response, str):
                texts.append(response)
            else:
                logger.warning(f"Unexpected response format: {type(response)}")

    else:
        logger.error("No recognizable response format found in results data")

    logger.info(f"Extracted {len(texts)} response texts")
    return texts


def extract_scenarios_from_results(
    results_data: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Extract scenarios with demographic information from results data.
    Supports both old and new schema formats.

    Args:
        results_data: Dictionary containing evaluation results

    Returns:
        List of scenario dictionaries with demographic metadata
    """
    scenarios = []

    # Check for new schema format
    if "scenarios" in results_data and isinstance(results_data["scenarios"], list):
        logger.info("Extracting scenarios from new schema format")
        for scenario in results_data["scenarios"]:
            for output in scenario.get("outputs", []):
                # Only include scenarios that have non-empty responses (to match response extraction)
                response_text = output.get("response", "")
                if response_text:
                    demographic = output.get("demographic", {})
                    scenario_dict = {
                        "_demographic_gender": demographic.get("gender", "unknown"),
                        "_demographic_ethnicity": demographic.get(
                            "ethnicity", "unknown"
                        ),
                        "CANDIDATE_NAME": demographic.get("candidate_name", "unknown"),
                        "POSITION": scenario.get("job_profile", {}).get("position", ""),
                        "EXPERIENCE": scenario.get("job_profile", {}).get(
                            "experience", ""
                        ),
                        "EDUCATION": scenario.get("job_profile", {}).get(
                            "education", ""
                        ),
                        "PREV_ROLE": scenario.get("job_profile", {}).get(
                            "previous_role", ""
                        ),
                        "_profile_id": scenario.get("scenario_id", 0),
                    }
                    scenarios.append(scenario_dict)

    # Fall back to old schema format
    elif "scenarios" in results_data:
        logger.info("Extracting scenarios from old schema format")
        scenarios = results_data["scenarios"]

    else:
        logger.warning("No scenarios found in results data")

    logger.info(f"Extracted {len(scenarios)} scenarios")
    return scenarios


def save_extended_results(
    original_data: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    evaluator_type: str,
    file_path: str,
):
    """
    Extend the original results file with new evaluation results.

    Args:
        original_data: Original results data
        evaluation_results: New evaluation results to add
        evaluator_type: Type of evaluator used
        file_path: Path to save the extended results
    """
    # Add evaluation results to the original data
    if "evaluations" not in original_data:
        original_data["evaluations"] = {}

    original_data["evaluations"][evaluator_type] = {
        "evaluation_results": evaluation_results,
        "timestamp": time.time(),
    }

    output_file = Path(file_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(original_data, f, indent=2)

    logger.info(f"Extended evaluation results saved to {file_path}")


def create_evaluator(evaluator_type: str):
    """
    Create an evaluator instance.
    """
    if evaluator_type == "scm":
        return SCMWarmthCompetencyEvaluator()
    elif evaluator_type == "dummy":
        return DummyEvaluator()
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")


def evaluate_single_project(project_name: str, evaluator_type: str) -> int:
    """
    Evaluate a single project.

    Args:
        project_name: Name of the project to evaluate
        evaluator_type: Type of evaluator to use

    Returns:
        0 on success, 1 on failure
    """
    try:
        # Set up project directory
        project_dir = Path("results") / project_name
        responses_file = project_dir / "responses.json"

        if not responses_file.exists():
            logger.error(f"Responses file not found: {responses_file}")
            return 1

        # Load the responses file
        logger.info(f"Loading responses from {responses_file}")
        results_data = load_evaluation_results(str(responses_file))

        # Extract response texts and demographic information
        response_texts = extract_responses_from_results(results_data)
        scenarios = extract_scenarios_from_results(results_data)

        if not response_texts:
            logger.error("No response texts found in the input file")
            return 1

        # Create evaluator
        logger.info(f"Creating {evaluator_type} evaluator")
        evaluator = create_evaluator(evaluator_type)

        # Run evaluation (pass demographic info if available)
        logger.info(f"Evaluating {len(response_texts)} responses")
        if scenarios and len(scenarios) == len(response_texts):
            logger.info(
                f"Using demographic information from {len(scenarios)} scenarios for group analysis"
            )
            evaluation_results = evaluator.evaluate(response_texts, scenarios)
        else:
            logger.info(
                "No demographic information available - running without group analysis"
            )
            evaluation_results = evaluator.evaluate(response_texts)

        # Save evaluation results to separate file
        eval_filename = f"eval_{evaluator_type}.json"
        eval_file = project_dir / eval_filename

        # Create evaluation data structure with proper demographic group mapping
        aggregated_analysis = {}
        if "demographic_groups" in evaluation_results:
            aggregated_analysis["by_demographic_group"] = evaluation_results[
                "demographic_groups"
            ]

        eval_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "project_name": project_name,
                "evaluator_type": evaluator_type,
                "total_evaluations": evaluation_results.get("n_samples", 0),
            },
            "evaluation_results": evaluation_results,
            "aggregated_analysis": aggregated_analysis,
        }

        # Save evaluation results
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {eval_file}")

        # Print summary
        print(f"\n📊 EVALUATION SUMMARY")
        print(f"Project: {project_name}")
        print(f"Evaluator: {evaluator_type}")
        print(f"Responses evaluated: {evaluation_results['n_samples']}")
        print(
            f"Warmth score: {get_primary_value(evaluation_results['warmth']['mean']):.3f} ± {get_primary_value(evaluation_results['warmth']['std']):.3f}"
        )
        print(
            f"Competency score: {get_primary_value(evaluation_results['competency']['mean']):.3f} ± {get_primary_value(evaluation_results['competency']['std']):.3f}"
        )

        bias_metrics = evaluation_results.get("bias_metrics", {})
        if bias_metrics:
            print(
                f"Warmth-Competency gap: {get_primary_value(bias_metrics.get('warmth_competency_gap', 0)):.3f}"
            )
            print(
                f"Correlation: {get_primary_value(bias_metrics.get('warmth_competency_correlation', 0)):.3f}"
            )

        # Show demographic group statistics if available
        demographic_groups = evaluation_results.get("demographic_groups", {})
        if demographic_groups:
            print(f"\n📈 DEMOGRAPHIC GROUP ANALYSIS")
            print(f"Groups analyzed: {len(demographic_groups)}")
            print("-" * 80)
            print(
                f"{'Group':<15} {'N':<3} {'Warmth':<8} {'Competency':<10} {'W+/-':<6} {'C+/-':<6} {'Gap':<8}"
            )
            print("-" * 80)

            for group_key, group_data in demographic_groups.items():
                demo_info = group_data["demographic_info"]
                warmth = group_data["warmth"]
                comp = group_data["competency"]
                bias = group_data["bias_metrics"]

                print(
                    f"{group_key:<15} {demo_info['n_samples']:<3} "
                    f"{get_primary_value(warmth['mean']):.3f}    {get_primary_value(comp['mean']):.3f}      "
                    f"{bias.get('warmth_competency_gap', 0.0):.3f}"
                )

            print(f"\n📊 DETAILED GROUP STATISTICS")
            print("=" * 80)

            # Sort groups by name for consistent output
            sorted_groups = sorted(demographic_groups.items())

            for group_key, group_data in sorted_groups:
                demo_info = group_data["demographic_info"]
                warmth = group_data["warmth"]
                comp = group_data["competency"]
                bias = group_data["bias_metrics"]

                print(
                    f"\n{group_key.upper().replace('_', ' → ')} (n={demo_info['n_samples']})"
                )
                print(
                    f"  Warmth:     μ={get_primary_value(warmth['mean']):.3f}, σ={get_primary_value(warmth['std']):.3f}, range=[{get_primary_value(warmth['min']):.3f}, {get_primary_value(warmth['max']):.3f}]"
                )
                print(
                    f"  Competency: μ={get_primary_value(comp['mean']):.3f}, σ={get_primary_value(comp['std']):.3f}, range=[{get_primary_value(comp['min']):.3f}, {get_primary_value(comp['max']):.3f}]"
                )
                print(
                    f"  Bias: gap={bias.get('warmth_competency_gap', 0.0):.3f}, corr={bias.get('warmth_competency_correlation', 0.0):.3f}"
                )

                # Show individual scores for small groups
                if demo_info["n_samples"] <= 5:
                    print(
                        f"  Individual scores: W={warmth['scores'][0]}, C={comp['scores'][0]}"
                    )

            print("=" * 80)

        print(f"\n✓ Evaluation complete! Results saved to {eval_filename}")
        return 0

    except Exception as e:
        logger.error(f"Error during evaluation of {project_name}: {e}")
        return 1


def main():
    """
    Main CLI function for evaluating existing result files.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate existing LLM response files with bias evaluators"
    )

    parser.add_argument(
        "project_names",
        nargs="+",
        help="Names of projects to evaluate (will read responses.json from results/<project-name>/)",
    )

    parser.add_argument(
        "--evaluator-type",
        type=str,
        default="scm",
        choices=["scm", "dummy"],
        help="Type of evaluator to use",
    )

    args = parser.parse_args()

    # Configure logging (verbose by default)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        successful_projects = []
        failed_projects = []

        print(
            f"🚀 Starting evaluation of {len(args.project_names)} projects with {args.evaluator_type} evaluator"
        )
        print("=" * 80)

        # Process each project sequentially
        for i, project_name in enumerate(args.project_names, 1):
            print(
                f"\n📂 Processing project {i}/{len(args.project_names)}: {project_name}"
            )
            print("-" * 40)

            result = evaluate_single_project(project_name, args.evaluator_type)

            if result == 0:
                successful_projects.append(project_name)
                print(f"✅ Successfully evaluated {project_name}")
            else:
                failed_projects.append(project_name)
                print(f"❌ Failed to evaluate {project_name}")

        # Print final summary
        print("\n" + "=" * 80)
        print("🎯 BATCH EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total projects: {len(args.project_names)}")
        print(f"Successful: {len(successful_projects)}")
        print(f"Failed: {len(failed_projects)}")

        if successful_projects:
            print(f"\n✅ Successfully evaluated projects:")
            for project in successful_projects:
                print(f"  - {project}")

        if failed_projects:
            print(f"\n❌ Failed to evaluate projects:")
            for project in failed_projects:
                print(f"  - {project}")

        print("=" * 80)

        # Return appropriate exit code
        return 1 if failed_projects else 0

    except Exception as e:
        logger.error(f"Error during batch evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
