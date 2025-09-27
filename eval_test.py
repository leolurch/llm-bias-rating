#!/usr/bin/env python3
"""
Simple test script for evaluating single sentences with SCM warmth/competency evaluator.
Usage: python eval_test.py
"""

import sys
from evaluators import SCMWarmthCompetencyEvaluator

def main():
    """Interactive test for SCM evaluator."""
    print("SCM Warmth/Competency Evaluator Test")
    print("=" * 40)
    print("Enter sentences to evaluate (press Ctrl+C to exit)")
    print()

    # Initialize evaluator
    print("Initializing SCM evaluator...")
    evaluator = SCMWarmthCompetencyEvaluator()
    print("Ready!\n")

    try:
        while True:
            # Get input sentence
            sentence = input("Enter sentence: ").strip()
            if not sentence:
                continue

            # Evaluate
            print("Evaluating...")
            results = evaluator.evaluate([sentence])

            # Display results
            if results["detailed_scores"]:
                score = results["detailed_scores"][0]
                print(f"\nResults:")
                print(f"  Warmth:     {score['warmth']['score']:.3f} ({score['warmth']['orientation']})")
                print(f"  Competency: {score['competency']['score']:.3f} ({score['competency']['orientation']})")

                # Interpretation
                warmth = score['warmth']['score']
                competency = score['competency']['score']

                if warmth >= 0 and competency >= 0:
                    quadrant = "Warm & Competent"
                elif warmth >= 0 and competency < 0:
                    quadrant = "Warm & Incompetent"
                elif warmth < 0 and competency >= 0:
                    quadrant = "Cold & Competent"
                else:
                    quadrant = "Cold & Incompetent"

                print(f"  Quadrant:   {quadrant}")
            else:
                print("No results returned")

            print("-" * 40)

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()