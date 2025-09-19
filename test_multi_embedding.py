#!/usr/bin/env python3
"""Test multi-embedding model functionality."""

import logging
from embedding_adapters import create_multiple_adapters
from evaluators import WarmthCompetencyEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_multi_embedding():
    """Test the multi-embedding model functionality."""

    # Create multiple embedding adapters
    adapter_configs = [
        {"type": "dummy", "embedding_dim": 384},
        {"type": "openai", "model_name": "text-embedding-3-small"}
    ]

    embedding_models = create_multiple_adapters(adapter_configs)
    logger.info(f"Created {len(embedding_models)} embedding models")

    # Create evaluator with multiple models
    evaluator = WarmthCompetencyEvaluator(embedding_models=embedding_models)

    # Test texts
    test_texts = [
        "She is very kind and caring, always helping others.",
        "He is extremely intelligent and capable at solving problems.",
        "They seem cold and distant, but very competent at their job."
    ]

    # Test demographics
    demographics = [
        {"_demographic_gender": "female", "_demographic_race": "white"},
        {"_demographic_gender": "male", "_demographic_race": "asian"},
        {"_demographic_gender": "non-binary", "_demographic_race": "black"}
    ]

    logger.info(f"Testing evaluation with {len(test_texts)} texts...")

    # Run evaluation
    results = evaluator.evaluate(test_texts, demographics)

    # Print results structure
    print(f"\nResults structure:")
    print(f"- n_samples: {results['n_samples']}")
    print(f"- embedding_models: {len(results['embedding_models'])} models")
    for i, model_info in enumerate(results['embedding_models']):
        print(f"  Model {i}: {model_info}")

    print(f"\nWarmth scores (shape: {len(results['warmth']['scores'])} texts × {len(results['warmth']['scores'][0])} models):")
    for i, text_scores in enumerate(results['warmth']['scores']):
        print(f"  Text {i}: {text_scores}")

    print(f"\nCompetency scores (shape: {len(results['competency']['scores'])} texts × {len(results['competency']['scores'][0])} models):")
    for i, text_scores in enumerate(results['competency']['scores']):
        print(f"  Text {i}: {text_scores}")

    print(f"\nAggregate statistics:")
    print(f"- Warmth means: {results['warmth']['mean']}")
    print(f"- Competency means: {results['competency']['mean']}")

    print(f"\nDetailed scores for first text:")
    first_score = results['detailed_scores'][0]
    print(f"- Text: {first_score['text']}")
    print(f"- Warmth scores: {first_score['warmth']['score']}")
    print(f"- Warmth orientations: {first_score['warmth']['orientation']}")
    print(f"- Competency scores: {first_score['competency']['score']}")
    print(f"- Competency orientations: {first_score['competency']['orientation']}")
    print(f"- Demographics: {first_score.get('demographic', 'None')}")

    logger.info("Multi-embedding model test completed successfully!")
    return results

if __name__ == "__main__":
    test_multi_embedding()
