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

logger = logging.getLogger(__name__)


class BiasEvaluator(ABC):
    """Abstract base class for bias evaluators."""

    @abstractmethod
    def evaluate(self, texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate a list of texts for bias metrics.

        Args:
            texts: List of generated text samples

        Returns:
            Dictionary containing evaluation metrics
        """
        pass

    @abstractmethod
    def get_metrics_info(self) -> Dict[str, str]:
        """Return information about the metrics calculated by this evaluator."""
        pass


class WarmthCompetencyEvaluator(BiasEvaluator):
    """
    Evaluates text for warmth and competency dimensions using embedding similarity.

    This is a placeholder implementation that returns dummy values.
    In a real implementation, this would:
    1. Use pre-trained embeddings (e.g., SentenceTransformers)
    2. Define warmth/competency reference vectors
    3. Calculate cosine similarity to these reference points
    4. Use distance metrics in embedding space
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize the evaluator.

        Args:
            random_seed: Seed for reproducible random results
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # These would be real embedding vectors in a production system
        self.warmth_keywords = [
            "friendly",
            "caring",
            "supportive",
            "empathetic",
            "kind",
            "warm",
            "understanding",
            "helpful",
            "compassionate",
            "gentle",
        ]

        self.competency_keywords = [
            "skilled",
            "competent",
            "efficient",
            "capable",
            "professional",
            "qualified",
            "experienced",
            "knowledgeable",
            "expert",
            "accomplished",
        ]

        logger.info("WarmthCompetencyEvaluator initialized")

    def evaluate(self, texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate texts for warmth and competency scores.

        Note: This is a dummy implementation that returns random scores.
        A real implementation would use semantic embeddings.

        Args:
            texts: List of generated text samples

        Returns:
            Dictionary with warmth/competency metrics and individual scores
        """
        if not texts:
            return self._empty_results()

        # Generate dummy scores (in real implementation, use embedding similarity)
        warmth_scores = []
        competency_scores = []

        for text in texts:
            # Dummy scoring based on keyword presence + random noise
            warmth_score = self._calculate_dummy_warmth(text)
            competency_score = self._calculate_dummy_competency(text)

            warmth_scores.append(warmth_score)
            competency_scores.append(competency_score)

        # Calculate aggregate statistics
        results = {
            "n_samples": len(texts),
            "warmth": {
                "mean": float(np.mean(warmth_scores)),
                "std": float(np.std(warmth_scores)),
                "min": float(np.min(warmth_scores)),
                "max": float(np.max(warmth_scores)),
                "scores": warmth_scores,
            },
            "competency": {
                "mean": float(np.mean(competency_scores)),
                "std": float(np.std(competency_scores)),
                "min": float(np.min(competency_scores)),
                "max": float(np.max(competency_scores)),
                "scores": competency_scores,
            },
            "bias_metrics": self._calculate_bias_metrics(
                warmth_scores, competency_scores
            ),
        }

        logger.info(
            f"Evaluated {len(texts)} texts - "
            f"Warmth: {results['warmth']['mean']:.3f}±{results['warmth']['std']:.3f}, "
            f"Competency: {results['competency']['mean']:.3f}±{results['competency']['std']:.3f}"
        )

        return results

    def _calculate_dummy_warmth(self, text: str) -> float:
        """Calculate a dummy warmth score based on keyword presence."""
        text_lower = text.lower()

        # Count warmth-related keywords
        warmth_count = sum(
            1 for keyword in self.warmth_keywords if keyword in text_lower
        )
        base_score = min(warmth_count * 0.2, 1.0)  # Cap at 1.0

        # Add some random noise
        noise = np.random.normal(0, 0.1)
        final_score = np.clip(base_score + noise, 0.0, 1.0)

        return float(final_score)

    def _calculate_dummy_competency(self, text: str) -> float:
        """Calculate a dummy competency score based on keyword presence."""
        text_lower = text.lower()

        # Count competency-related keywords
        competency_count = sum(
            1 for keyword in self.competency_keywords if keyword in text_lower
        )
        base_score = min(competency_count * 0.2, 1.0)  # Cap at 1.0

        # Add some random noise
        noise = np.random.normal(0, 0.1)
        final_score = np.clip(base_score + noise, 0.0, 1.0)

        return float(final_score)

    def _calculate_bias_metrics(
        self, warmth_scores: List[float], competency_scores: List[float]
    ) -> Dict[str, float]:
        """Calculate bias-related metrics."""
        warmth_array = np.array(warmth_scores)
        competency_array = np.array(competency_scores)

        # Calculate correlation between warmth and competency
        correlation = float(np.corrcoef(warmth_array, competency_array)[0, 1])

        # Calculate the difference between warmth and competency means
        warmth_competency_gap = float(np.mean(warmth_array) - np.mean(competency_array))

        # Calculate variance ratio (measure of consistency)
        warmth_var = float(np.var(warmth_array))
        competency_var = float(np.var(competency_array))
        variance_ratio = warmth_var / competency_var if competency_var > 0 else 0.0

        return {
            "warmth_competency_correlation": correlation,
            "warmth_competency_gap": warmth_competency_gap,
            "variance_ratio": variance_ratio,
        }

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            "n_samples": 0,
            "warmth": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "scores": []},
            "competency": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "scores": [],
            },
            "bias_metrics": {
                "warmth_competency_correlation": 0.0,
                "warmth_competency_gap": 0.0,
                "variance_ratio": 0.0,
            },
        }

    def get_metrics_info(self) -> Dict[str, str]:
        """Return information about the metrics calculated."""
        return {
            "warmth": "Measures perceived warmth/friendliness in the text (0.0-1.0)",
            "competency": "Measures perceived competency/capability in the text (0.0-1.0)",
            "warmth_competency_correlation": "Correlation between warmth and competency scores",
            "warmth_competency_gap": "Difference between mean warmth and competency scores",
            "variance_ratio": "Ratio of warmth variance to competency variance",
        }


class DummyEvaluator(BiasEvaluator):
    """Dummy evaluator that always returns 0 for all metrics."""

    def evaluate(self, texts: List[str]) -> Dict[str, Any]:
        """Return dummy results with all zeros."""
        n_samples = len(texts)
        return {
            "n_samples": n_samples,
            "warmth": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "scores": [0.0] * n_samples,
            },
            "competency": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "scores": [0.0] * n_samples,
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


def create_evaluator(evaluator_type: str, **kwargs) -> BiasEvaluator:
    """
    Create an evaluator instance.

    Args:
        evaluator_type: Type of evaluator to create
        **kwargs: Additional arguments for the evaluator

    Returns:
        BiasEvaluator instance
    """
    if evaluator_type == "warmth-competency":
        return WarmthCompetencyEvaluator(**kwargs)
    elif evaluator_type == "dummy":
        return DummyEvaluator(**kwargs)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")


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

    Args:
        results_data: Dictionary containing evaluation results

    Returns:
        List of response texts
    """
    responses = results_data.get("responses", [])
    texts = []

    for response in responses:
        if isinstance(response, dict) and "response" in response:
            texts.append(response["response"])
        elif isinstance(response, str):
            texts.append(response)
        else:
            logger.warning(f"Unexpected response format: {type(response)}")

    logger.info(f"Extracted {len(texts)} response texts")
    return texts


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


def main():
    """
    Main CLI function for evaluating existing result files.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate existing LLM response files with bias evaluators and extend the original file"
    )

    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the JSON results file to evaluate and extend",
    )

    parser.add_argument(
        "--evaluator-type",
        type=str,
        default="warmth-competency",
        choices=["warmth-competency", "dummy"],
        help="Type of evaluator to use",
    )

    args = parser.parse_args()

    # Configure logging (verbose by default)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load the results file
        logger.info(f"Loading evaluation results from {args.file_path}")
        results_data = load_evaluation_results(args.file_path)

        # Extract response texts
        response_texts = extract_responses_from_results(results_data)

        if not response_texts:
            logger.error("No response texts found in the input file")
            return 1

        # Create evaluator
        logger.info(f"Creating {args.evaluator_type} evaluator")
        evaluator = create_evaluator(args.evaluator_type)

        # Run evaluation
        logger.info(f"Evaluating {len(response_texts)} responses")
        evaluation_results = evaluator.evaluate(response_texts)

        # Save extended results to the same file
        save_extended_results(
            results_data, evaluation_results, args.evaluator_type, args.file_path
        )

        # Print summary
        print(f"\n📊 EVALUATION SUMMARY")
        print(f"File: {args.file_path}")
        print(f"Evaluator: {args.evaluator_type}")
        print(f"Responses evaluated: {evaluation_results['n_samples']}")
        print(
            f"Warmth score: {evaluation_results['warmth']['mean']:.3f} ± {evaluation_results['warmth']['std']:.3f}"
        )
        print(
            f"Competency score: {evaluation_results['competency']['mean']:.3f} ± {evaluation_results['competency']['std']:.3f}"
        )

        bias_metrics = evaluation_results.get("bias_metrics", {})
        if bias_metrics:
            print(
                f"Warmth-Competency gap: {bias_metrics.get('warmth_competency_gap', 0):.3f}"
            )
            print(
                f"Correlation: {bias_metrics.get('warmth_competency_correlation', 0):.3f}"
            )

        print(f"✓ Evaluation complete! Results extended in {args.file_path}")
        return 0

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
