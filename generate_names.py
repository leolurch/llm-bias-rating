#!/usr/bin/env python3
"""
Generate names.json file from Popular Baby Names dataset.

This script reads the NYC Popular Baby Names CSV file, deduplicates it,
and combines first names with appropriate surnames to create a comprehensive
name database for bias testing.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Surname mappings based on demographic data
SURNAMES = {
    "white": ["YODER", "FRIEDMAN", "KRUEGER", "SCHWARTZ", "SCHMITT", "MUELLER", "WEISS", "NOVAK", "OCONNELL", "KLEIN"],
    "black": ["WASHINGTON", "JEFFERSON", "BOOKER", "BANKS", "JOSEPH", "MOSLEY", "JACKSON", "CHARLES", "DORSEY", "RIVERS"],
    "asian": ["XIONG", "ZHANG", "HUANG", "TRUONG", "YANG", "LI", "VANG", "HUYNH", "VU", "NGUYEN"],
    "hispanic": ["BARRAJAS", "ZAVALA", "VELAZQUEZ", "AVALOS", "OROZCO", "VAZQUEZ", "JUAREZ", "MEZA", "HUERTA", "IBARRA"],
}

# Mapping from NYC dataset ethnicity categories to our ethnicity categories
ETHNICITY_MAPPING = {
    "WHITE NON HISPANIC": "white",
    "WHITE NON HISP": "white",  # Truncated version in some years
    "BLACK NON HISPANIC": "black",
    "BLACK NON HISP": "black",  # Truncated version in some years
    "ASIAN AND PACIFIC ISLANDER": "asian",
    "ASIAN AND PACI": "asian",  # Truncated version in some years
    "HISPANIC": "hispanic",
}

def load_and_deduplicate_names(csv_path: Path) -> dict:
    """Load NYC baby names CSV and deduplicate entries."""
    logger.info(f"Loading names from {csv_path}")

    names_by_demo = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    seen_rows = set()
    total_rows = 0
    duplicate_rows = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_rows += 1

            # Create a tuple of all values to detect exact duplicates
            row_tuple = (
                row['Year of Birth'],
                row['Gender'],
                row['Ethnicity'],
                row["Child's First Name"],
                row['Count'],
                row['Rank']
            )

            if row_tuple in seen_rows:
                duplicate_rows += 1
                continue
            seen_rows.add(row_tuple)

            # Map ethnicity to our ethnicity categories
            ethnicity = row['Ethnicity'].strip()
            if ethnicity not in ETHNICITY_MAPPING:
                continue

            ethnicity_category = ETHNICITY_MAPPING[ethnicity]
            gender = row['Gender'].lower()
            name = row["Child's First Name"].strip()
            count = int(row['Count'])

            # Accumulate counts across all years for this name
            names_by_demo[ethnicity_category][gender][name] += count

    logger.info(f"Processed {total_rows} total rows, removed {duplicate_rows} duplicates")

    return names_by_demo

def create_full_names(names_by_demo: dict) -> dict:
    """Combine first names with surnames to create full names."""
    logger.info("Creating full names by combining first names with surnames")

    result = {}

    for ethnicity_category, genders in names_by_demo.items():
        if ethnicity_category not in SURNAMES:
            logger.warning(f"No surnames defined for ethnicity: {ethnicity_category}")
            continue

        result[ethnicity_category] = {}
        ethnicity_surnames = SURNAMES[ethnicity_category]

        for gender, names in genders.items():
            # Sort by total count and take top names
            sorted_names = sorted(names.items(), key=lambda x: x[1], reverse=True)[:50]

            # Create all combinations of first names with all surnames
            full_names = []
            for first_name, _ in sorted_names:
                for surname in ethnicity_surnames:
                    full_name = f"{first_name.upper()} {surname.upper()}"
                    full_names.append(full_name)

            result[ethnicity_category][gender] = full_names
            logger.info(f"Created {len(full_names)} full names for {ethnicity_category} {gender}")

    return result

def generate_names_file():
    """Main function to generate the names.json file."""
    csv_path = Path("data/Popular_Baby_Names.csv")
    output_path = Path("data/names.json")

    if not csv_path.exists():
        logger.error(f"Input file not found: {csv_path}")
        logger.error("Please download the Popular Baby Names CSV file from:")
        logger.error("https://catalog.data.gov/dataset/popular-baby-names")
        return False

    # Create data directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True)

    try:
        # Load and process data
        names_by_demo = load_and_deduplicate_names(csv_path)

        # Create full names
        full_names = create_full_names(names_by_demo)

        # Add metadata
        result = {
            'metadata': {
                'source': 'NYC Popular Baby Names Dataset',
                'url': 'https://catalog.data.gov/dataset/popular-baby-names',
                'surname_source': 'U.S. Census Bureau 2010 Surnames by Race/Ethnicity',
                'surname_url': 'https://www2.census.gov/topics/genealogy/2010surnames/',
                'generated_by': 'generate_names.py',
                'description': 'First names from NYC baby names data combined with demographically appropriate surnames from U.S. Census data',
                'ethnicities': list(full_names.keys()),
                'surname_categories': {
                    'white': 'Non-Hispanic White alone',
                    'black': 'Non-Hispanic Black or African American alone',
                    'asian': 'Non-Hispanic Asian and Native Hawaiian and Other Pacific Islander alone',
                    'hispanic': 'Hispanic or Latino origin'
                }
            },
            'names': full_names
        }

        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully generated {output_path}")

        # Print summary statistics
        total_names = 0
        for ethnicity_category, genders in full_names.items():
            for gender, names in genders.items():
                count = len(names)
                total_names += count
                logger.info(f"  {ethnicity_category} {gender}: {count} names")

        logger.info(f"Total names: {total_names}")
        return True

    except Exception as e:
        logger.error(f"Error generating names file: {e}")
        return False

if __name__ == "__main__":
    success = generate_names_file()
    if not success:
        exit(1)

    print("\n✅ Successfully generated data/names.json")
    print("This file contains demographically categorized names for bias testing.")
    print("You can now use this with eval.py for LLM bias evaluation.")
