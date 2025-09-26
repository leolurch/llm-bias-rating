#!/usr/bin/env python3
"""
Demonstration script for demographic bias testing capabilities.

This script shows how to use the framework to test for potential bias
across different demographic groups.
"""

import subprocess
import json
import os


def run_evaluation(num_job_profiles=2, output_file=None):
    """Run evaluation with controlled demographic comparison."""
    if output_file is None:
        output_file = f"results/demo_{num_job_profiles}_profiles.json"

    cmd = [
        "python",
        "eval.py",
        "--num-job-profiles",
        str(num_job_profiles),
        "--model-type",
        "dummy",
        "--evaluator-type",
        "warmth-competency",
        "--output-file",
        output_file,
    ]

    print(f"\n{'='*60}")
    print(f"Running: {num_job_profiles} job profiles x all demographic combinations")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Evaluation completed successfully")
        return output_file
    else:
        print(f"✗ Evaluation failed: {result.stderr}")
        return None


def analyze_results(result_files):
    """Analyze and compare results across demographic groups."""
    print(f"\n{'='*60}")
    print("DEMOGRAPHIC COMPARISON ANALYSIS")
    print(f"{'='*60}")

    results = {}
    for file_path in result_files:
        if file_path and os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                exp_info = data["experiment_info"]
                eval_data = data["evaluation"]

                key = f"{exp_info['num_job_profiles']}_profiles"
                results[key] = {
                    "warmth_mean": eval_data["warmth"]["mean"],
                    "competency_mean": eval_data["competency"]["mean"],
                    "correlation": eval_data["bias_metrics"][
                        "warmth_competency_correlation"
                    ],
                    "gap": eval_data["bias_metrics"]["warmth_competency_gap"],
                    "total_scenarios": exp_info["total_scenarios"],
                    "methodology": exp_info["evaluation_methodology"]["description"],
                    "gender_dist": exp_info["demographic_distribution"]["gender"],
                    "ethnicity_dist": exp_info["demographic_distribution"]["ethnicity"],
                }

    # Display comparison table
    print(
        f"{'Group':<15} {'Scenarios':<10} {'Warmth':<8} {'Competency':<10} {'W-C Gap':<8} {'Correlation':<10}"
    )
    print("-" * 70)

    for group, data in results.items():
        print(
            f"{group:<15} {data['total_scenarios']:<10} {data['warmth_mean']:.3f}    {data['competency_mean']:.3f}      {data['gap']:.3f}    {data['correlation']:.3f}"
        )

    print(
        f"\nMethodology for all tests: {list(results.values())[0]['methodology'] if results else 'N/A'}"
    )

    print(f"\nDemographic Coverage:")
    for group, data in results.items():
        print(f"\n{group}:")
        print(f"  Gender: {dict(data['gender_dist'])}")
        print(f"  Ethnicity: {dict(data['ethnicity_dist'])}")


def main():
    """Run demographic bias testing demonstration."""
    print("LLM Bias Rating Framework - Controlled Demographic Comparison Demo")
    print("This demonstration shows the new approach: each job profile is tested")
    print("with names from ALL demographic categories for objective comparison.")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Test different numbers of job profiles
    test_scenarios = [1, 3, 5]  # Number of job profiles to test

    result_files = []

    # Run evaluations
    for num_profiles in test_scenarios:
        result_file = run_evaluation(num_profiles)
        result_files.append(result_file)

    # Analyze results
    analyze_results(result_files)

    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"{'='*60}")
    print("Result files saved in results/ directory:")
    for file_path in result_files:
        if file_path:
            print(f"  - {file_path}")

    print("\nKey advantages of this approach:")
    print("1. Each job profile tested with ALL demographic combinations")
    print("2. Objective comparison - only names differ between scenarios")
    print("3. No confounding variables from different job attributes")
    print("4. Perfect demographic balance for statistical analysis")

    print("\nTo explore bias testing further:")
    print("1. Increase num-job-profiles for more diverse job combinations")
    print("2. Use real LLMs instead of dummy model")
    print("3. Implement semantic evaluators for better bias detection")
    print("4. Analyze results by specific demographic pairs")


if __name__ == "__main__":
    main()
