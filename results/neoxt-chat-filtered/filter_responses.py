#!/usr/bin/env python3
"""
Filter script to remove <human> and <bot> tags from response JSON files.
Truncates response text before the first occurrence of these tags.
"""

import json
import re
from pathlib import Path
import argparse
import logging

logger = logging.getLogger(__name__)


def filter_response_text(response_text: str) -> str:
    """
    Filter response text by truncating before first <human> or <bot> tag.

    Args:
        response_text: Original response text

    Returns:
        Filtered response text
    """
    # Find the first occurrence of either <human> or <bot>
    human_match = re.search(r"<human>", response_text, re.IGNORECASE)
    bot_match = re.search(r"<bot>", response_text, re.IGNORECASE)

    # Determine which tag appears first (if any)
    truncate_pos = None

    if human_match and bot_match:
        # Both tags found, use the earlier one
        truncate_pos = min(human_match.start(), bot_match.start())
    elif human_match:
        truncate_pos = human_match.start()
    elif bot_match:
        truncate_pos = bot_match.start()

    if truncate_pos is not None:
        # Truncate at the tag position and strip trailing whitespace
        filtered_text = response_text[:truncate_pos].rstrip()
        return filtered_text

    # No tags found, return original text
    return response_text


def filter_responses_file(input_file: str, output_file: str = None) -> int:
    """
    Filter responses in a JSON file.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (defaults to input_file with _filtered suffix)

    Returns:
        Number of responses that were filtered
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Determine output file path
    if output_file is None:
        output_path = (
            input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
        )
    else:
        output_path = Path(output_file)

    # Load the JSON data
    logger.info(f"Loading responses from {input_file}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_count = 0
    total_responses = 0

    # Process scenarios and outputs
    if "scenarios" in data:
        for scenario in data["scenarios"]:
            for output in scenario.get("outputs", []):
                if "response" in output:
                    total_responses += 1
                    original_response = output["response"]
                    filtered_response = filter_response_text(original_response)

                    if filtered_response != original_response:
                        filtered_count += 1
                        output["response"] = filtered_response
                        logger.debug(
                            f"Filtered response {output.get('output_id', 'unknown')}: "
                            f"'{original_response[:50]}...' -> '{filtered_response[:50]}...'"
                        )

    # Save the filtered data
    logger.info(f"Saving filtered responses to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Filtering complete: {filtered_count}/{total_responses} responses were filtered"
    )
    return filtered_count


def main():
    """Main entry point for the filtering script."""
    parser = argparse.ArgumentParser(
        description="Filter <human> and <bot> tags from response JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_file",
        help="Path to input JSON file (e.g., results/neoxt-chat/responses.json)",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: input_file with _filtered suffix)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    try:
        filtered_count = filter_responses_file(args.input_file, args.output)

        print(f"\n✅ Filtering completed successfully!")
        print(f"📊 {filtered_count} responses were filtered")

        if args.output:
            print(f"📁 Filtered file saved to: {args.output}")
        else:
            input_path = Path(args.input_file)
            output_path = (
                input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
            )
            print(f"📁 Filtered file saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error during filtering: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
