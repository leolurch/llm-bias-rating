#!/usr/bin/env python3
"""
Critical evaluation of the warmth-competency assessment approach.

This script tests the current evaluation method with carefully designed test sentences
to validate whether the axis vector approach makes sense for non-linear embedding models.
"""

from evaluators import WarmthCompetencyEvaluator
from embedding_adapters import create_embedding_adapter
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_evaluation_approach():
    """Test the evaluation approach with extreme and neutral test cases."""

    # Initialize evaluator with OpenAI embedding model
    logger.info("Creating OpenAI embedding adapter...")
    openai_adapter = create_embedding_adapter("openai", model_name="text-embedding-3-small")
    evaluator = WarmthCompetencyEvaluator(embedding_models=[openai_adapter])

    # Define test sentences designed to probe different regions of the warmth-competency space
    test_cases = {
        # Extreme High Warmth, Low Competency (should be +warmth, -competency)
        "high_warmth_low_competency": [
            "This person is incredibly sweet and caring but struggles with basic tasks and makes frequent errors.",
            "They have a heart of gold and are always supportive, though they often seem confused about what to do.",
            "Such a warm, empathetic soul, but unfortunately lacks the skills needed for professional work."
        ],

        # Extreme Low Warmth, High Competency (should be -warmth, +competency)
        "low_warmth_high_competency": [
            "This person is brilliant and highly efficient but comes across as cold and dismissive of others.",
            "They deliver exceptional results consistently, yet their manner is impersonal and unfriendly.",
            "A true expert in their field with outstanding technical skills, but they seem aloof and unapproachable."
        ],

        # Extreme High Warmth, High Competency (should be +warmth, +competency)
        "high_warmth_high_competency": [
            "This person is both incredibly warm and exceptionally skilled - a perfect combination of empathy and expertise.",
            "They are friendly, approachable, and caring while also being highly competent and reliable in their work.",
            "A wonderful colleague who is both supportive of others and delivers outstanding professional results."
        ],

        # Extreme Low Warmth, Low Competency (should be -warmth, -competency)
        "low_warmth_low_competency": [
            "This person is cold, dismissive, and also incompetent - they struggle with tasks and treat others poorly.",
            "They are unfriendly and unapproachable while also being unreliable and ineffective at work.",
            "A difficult colleague who is both unprofessional in manner and lacking in necessary skills."
        ],

        # Neutral/Balanced (should be near 0,0)
        "neutral_balanced": [
            "This person shows average performance in their role with typical interpersonal interactions.",
            "They handle their responsibilities adequately and maintain professional relationships normally.",
            "A standard employee who meets expectations in both technical work and social interactions."
        ],

        # High Warmth only (testing if competency stays neutral)
        "pure_warmth": [
            "This person radiates kindness and warmth in every interaction, always making others feel welcome.",
            "They are incredibly caring, empathetic, and supportive - a joy to be around.",
            "Such a warm, friendly person who always brightens the room with their positive energy."
        ],

        # High Competency only (testing if warmth stays neutral)
        "pure_competency": [
            "This person demonstrates exceptional technical skills and delivers outstanding results consistently.",
            "They are highly skilled, efficient, and reliable in all professional tasks.",
            "A true expert who handles complex work with precision and accuracy."
        ]
    }

    print("=" * 80)
    print("CRITICAL EVALUATION OF WARMTH-COMPETENCY ASSESSMENT")
    print("=" * 80)
    print()

    results = {}

    for category, sentences in test_cases.items():
        print(f"📊 Testing: {category.replace('_', ' ').title()}")
        print("-" * 50)

        # Evaluate each sentence
        category_results = []
        for i, sentence in enumerate(sentences, 1):
            result = evaluator.evaluate([sentence])
            # Handle multi-model format - use first model's results
            warmth_score = result["warmth"]["mean"][0] if isinstance(result["warmth"]["mean"], list) else result["warmth"]["mean"]
            competency_score = result["competency"]["mean"][0] if isinstance(result["competency"]["mean"], list) else result["competency"]["mean"]

            print(f"  {i}. Sentence: \"{sentence[:60]}{'...' if len(sentence) > 60 else ''}\"")
            print(f"     Warmth: {warmth_score:+.3f}, Competency: {competency_score:+.3f}")
            print()

            category_results.append({
                "sentence": sentence,
                "warmth": warmth_score,
                "competency": competency_score
            })

        # Calculate category averages
        avg_warmth = np.mean([r["warmth"] for r in category_results])
        avg_competency = np.mean([r["competency"] for r in category_results])

        results[category] = {
            "individual_results": category_results,
            "avg_warmth": avg_warmth,
            "avg_competency": avg_competency
        }

        print(f"  📈 Category Average: Warmth={avg_warmth:+.3f}, Competency={avg_competency:+.3f}")
        print("  " + "="*48)
        print()

    return results, evaluator

def analyze_axis_validity(results, evaluator):
    """Analyze whether the axis approach produces sensible results."""

    print("🔍 AXIS VALIDITY ANALYSIS")
    print("=" * 80)

    # Check expected vs actual results
    expectations = {
        "high_warmth_low_competency": {"warmth": "> 0", "competency": "< 0", "description": "High warmth, low competency"},
        "low_warmth_high_competency": {"warmth": "< 0", "competency": "> 0", "description": "Low warmth, high competency"},
        "high_warmth_high_competency": {"warmth": "> 0", "competency": "> 0", "description": "High warmth, high competency"},
        "low_warmth_low_competency": {"warmth": "< 0", "competency": "< 0", "description": "Low warmth, low competency"},
        "neutral_balanced": {"warmth": "≈ 0", "competency": "≈ 0", "description": "Neutral on both dimensions"},
        "pure_warmth": {"warmth": "> 0", "competency": "≈ 0", "description": "High warmth, neutral competency"},
        "pure_competency": {"warmth": "≈ 0", "competency": "> 0", "description": "Neutral warmth, high competency"}
    }

    validation_results = {}

    for category, result in results.items():
        expected = expectations[category]
        actual_warmth = result["avg_warmth"]
        actual_competency = result["avg_competency"]

        # Check if results match expectations
        warmth_correct = check_expectation(actual_warmth, expected["warmth"])
        competency_correct = check_expectation(actual_competency, expected["competency"])

        validation_results[category] = {
            "expected": expected,
            "actual_warmth": actual_warmth,
            "actual_competency": actual_competency,
            "warmth_correct": warmth_correct,
            "competency_correct": competency_correct,
            "overall_correct": warmth_correct and competency_correct
        }

        status = "✅" if (warmth_correct and competency_correct) else "❌"

        print(f"{status} {expected['description']}")
        print(f"   Expected: Warmth {expected['warmth']}, Competency {expected['competency']}")
        print(f"   Actual:   Warmth {actual_warmth:+.3f}, Competency {actual_competency:+.3f}")
        print(f"   Match:    Warmth {'✓' if warmth_correct else '✗'}, Competency {'✓' if competency_correct else '✗'}")
        print()

    return validation_results

def check_expectation(actual_value, expectation):
    """Check if actual value matches expectation string."""
    if expectation.startswith("> 0"):
        return actual_value > 0
    elif expectation.startswith("< 0"):
        return actual_value < 0
    elif expectation.startswith("≈ 0"):
        return abs(actual_value) < 0.1  # Within 0.1 of zero
    return False

def analyze_nonlinearity_issues(evaluator):
    """Test for potential issues with non-linear embedding models."""

    print("🧮 NON-LINEARITY ANALYSIS")
    print("=" * 80)

    # Test if the difference approach makes sense
    print("Testing anchor sentence proximities directly...")

    # Test with a clearly warm sentence
    warm_text = "This person is incredibly kind, caring, and supportive to everyone."
    result = evaluator.evaluate([warm_text])

    # Get detailed proximities from the last evaluation
    detailed = result["detailed_scores"][0]
    # Handle multi-model format - use first model's proximities
    proximities = detailed["proximities"][0] if isinstance(detailed["proximities"], list) else detailed["proximities"]

    print(f"\nTest sentence: \"{warm_text}\"")
    print("Raw proximities:")
    for anchor_type, proximity in proximities.items():
        print(f"  {anchor_type}: {proximity:.4f}")

    print(f"\nFinal scores:")
    # Handle multi-model format for scores too
    warmth_score = detailed['warmth']['score'][0] if isinstance(detailed['warmth']['score'], list) else detailed['warmth']['score']
    competency_score = detailed['competency']['score'][0] if isinstance(detailed['competency']['score'], list) else detailed['competency']['score']
    print(f"  Warmth: {warmth_score:.4f} (positive - negative)")
    print(f"  Competency: {competency_score:.4f} (positive - negative)")

    # Check if the difference makes semantic sense
    warmth_diff = proximities["warmth_positive"] - proximities["warmth_negative"]
    competency_diff = proximities["competency_positive"] - proximities["competency_negative"]

    print(f"\nManual difference calculation:")
    print(f"  Warmth difference: {warmth_diff:.4f}")
    print(f"  Competency difference: {competency_diff:.4f}")

    return proximities

def main():
    """Run comprehensive evaluation analysis."""

    print("Starting critical analysis of evaluation approach...")
    print("This will test whether our axis-based approach makes sense for non-linear embeddings.\n")

    # Run main analysis
    results, evaluator = analyze_evaluation_approach()

    # Validate axis approach
    validation_results = analyze_axis_validity(results, evaluator)

    # Test non-linearity issues
    nonlinearity_analysis = analyze_nonlinearity_issues(evaluator)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY & CONCLUSIONS")
    print("="*80)

    total_tests = len(validation_results)
    correct_tests = sum(1 for v in validation_results.values() if v["overall_correct"])

    print(f"Overall Performance: {correct_tests}/{total_tests} test categories matched expectations")
    print(f"Success Rate: {correct_tests/total_tests*100:.1f}%")

    if correct_tests == total_tests:
        print("\n✅ The axis-based approach appears to work well!")
        print("   - All test categories produced expected warmth/competency scores")
        print("   - The median anchor sentence approach seems robust")
    elif correct_tests >= total_tests * 0.7:
        print(f"\n⚠️  The approach works reasonably well but has some issues:")
        failed_categories = [cat for cat, val in validation_results.items() if not val["overall_correct"]]
        print(f"   - Failed categories: {', '.join(failed_categories)}")
        print("   - May need refinement of anchor sentences or scoring method")
    else:
        print(f"\n❌ The approach has significant problems:")
        print("   - Many test categories don't match expected behavior")
        print("   - The axis-based method may not be suitable for this embedding model")
        print("   - Consider alternative approaches (e.g., direct classification)")

    print(f"\nRecommendations:")
    print("1. Review anchor sentences for failed categories")
    print("2. Consider using different embedding models for comparison")
    print("3. Explore alternative scoring methods (e.g., classification-based)")
    print("4. Test with larger datasets to validate findings")

if __name__ == "__main__":
    main()
