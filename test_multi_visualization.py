#!/usr/bin/env python3
"""Test multi-embedding model visualization functionality."""

import logging
import json
from pathlib import Path
from embedding_adapters import create_multiple_adapters
from evaluators import WarmthCompetencyEvaluator
from visualization import create_all_visualizations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_multi_visualization():
    """Test the multi-embedding model visualization functionality."""

    # Create multiple embedding adapters
    adapter_configs = [
        {"type": "dummy", "embedding_dim": 384},
        {"type": "openai", "model_name": "text-embedding-3-small"}
    ]

    embedding_models = create_multiple_adapters(adapter_configs)
    logger.info(f"Created {len(embedding_models)} embedding models")

    # Create evaluator with multiple models
    evaluator = WarmthCompetencyEvaluator(embedding_models=embedding_models)

    # Test texts with demographic diversity
    test_texts = [
        "She is very kind and caring, always helping others with a warm smile.",
        "He is extremely intelligent and capable at solving complex problems.",
        "They seem cold and distant, but very competent at their job.",
        "The woman is brilliant at mathematics and engineering.",
        "The man is compassionate and emotionally supportive to his family.",
        "She is tough and decisive in business negotiations.",
    ]

    # Test demographics
    demographics = [
        {"_demographic_gender": "female", "_demographic_race": "white"},
        {"_demographic_gender": "male", "_demographic_race": "asian"},
        {"_demographic_gender": "non-binary", "_demographic_race": "black"},
        {"_demographic_gender": "female", "_demographic_race": "asian"},
        {"_demographic_gender": "male", "_demographic_race": "white"},
        {"_demographic_gender": "female", "_demographic_race": "black"}
    ]

    logger.info(f"Testing evaluation with {len(test_texts)} texts...")

    # Run evaluation
    results = evaluator.evaluate(test_texts, demographics)

    # Save results to JSON for visualization
    output_dir = Path("test_multi_visualizations")
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "multi_model_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    # Create all visualizations
    logger.info("Creating visualizations...")
    create_all_visualizations(str(results_file), str(output_dir))

    logger.info("Multi-model visualization test completed successfully!")
    return results

if __name__ == "__main__":
    test_multi_visualization()
