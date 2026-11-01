"""
Utility functions for loading Bhopal class mappings

This module provides helper functions to load class names and mappings
from the CSV files for use in model configurations.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple


def load_62class_names(csv_path: str = None) -> List[str]:
    """
    Load the 62 training class names from CSV file.

    Args:
        csv_path: Path to Bhopal_class_mapping_V13_TRAINING.csv
                  If None, uses default location in class_mapping directory

    Returns:
        List of 62 class names in order (indices 0-61)
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / "Bhopal_class_mapping_V13_TRAINING.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Class mapping CSV not found: {csv_path}")

    class_names = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            training_code = int(row['training_class_code'])
            class_name = row['class_name']
            # Ensure sequential ordering
            assert training_code == len(class_names), \
                f"Expected training_class_code {len(class_names)}, got {training_code}"
            class_names.append(class_name)

    assert len(class_names) == 62, f"Expected 62 classes, got {len(class_names)}"
    return class_names


def load_12class_names(csv_path: str = None) -> Tuple[List[str], Dict[int, int]]:
    """
    Load the 12 parent class names and mapping from 62 classes.

    Args:
        csv_path: Path to Bhopal_class_mapping_V13_TRAINING.csv
                  If None, uses default location in class_mapping directory

    Returns:
        Tuple of:
        - List of 12 parent class names (indices 0-11)
        - Dict mapping from 62-class codes to 12-class codes
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / "Bhopal_class_mapping_V13_TRAINING.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Class mapping CSV not found: {csv_path}")

    # Map XX (major category) to parent class name and index
    # XX ranges from 1-12 (not 0-indexed in the CSV)
    xx_to_parent = {}
    xx_to_index = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            xx = int(row['XX'])
            zz = int(row['ZZ'])
            class_name = row['class_name']

            # Parent class is when ZZ=0
            if zz == 0:
                # XX starts from 1, but we want 0-indexed for model
                parent_idx = xx - 1
                xx_to_parent[xx] = class_name
                xx_to_index[xx] = parent_idx

    # Create ordered list of 12 parent class names
    parent_names = []
    for xx in range(1, 13):  # XX ranges 1-12
        if xx in xx_to_parent:
            parent_names.append(xx_to_parent[xx])
        else:
            raise ValueError(f"Missing parent class for XX={xx}")

    # Create mapping from 62-class codes to 12-class codes
    class_62_to_12 = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            training_code = int(row['training_class_code'])  # 0-61
            xx = int(row['XX'])  # 1-12
            parent_idx = xx - 1  # Convert to 0-11
            class_62_to_12[training_code] = parent_idx

    assert len(parent_names) == 12, f"Expected 12 parent classes, got {len(parent_names)}"
    assert len(class_62_to_12) == 62, f"Expected 62 mappings, got {len(class_62_to_12)}"

    return parent_names, class_62_to_12


def get_class_distribution_by_parent() -> Dict[str, List[str]]:
    """
    Get distribution of 62 classes grouped by 12 parent classes.

    Returns:
        Dict mapping parent class name to list of child class names
    """
    csv_path = Path(__file__).parent / "Bhopal_class_mapping_V13_TRAINING.csv"

    distribution = {}
    current_parent = None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            xx = int(row['XX'])
            zz = int(row['ZZ'])
            class_name = row['class_name']

            # New parent class
            if zz == 0:
                current_parent = class_name
                distribution[current_parent] = [class_name]
            else:
                # Child class
                if current_parent:
                    distribution[current_parent].append(class_name)

    return distribution


# ============================================================================
# Convenience function for direct use in configs
# ============================================================================

def get_62class_names_for_config():
    """
    Get class names formatted for direct use in config files.
    Returns Python code string that can be copied into configs.
    """
    class_names = load_62class_names()

    # Format as Python list with proper indentation
    lines = ["class_names = ["]
    for i in range(0, len(class_names), 4):
        batch = class_names[i:i+4]
        formatted_batch = ", ".join([f'"{name}"' for name in batch])
        lines.append(f"    {formatted_batch},")
    lines.append("]")

    return "\n".join(lines)


def get_12class_names_for_config():
    """
    Get parent class names formatted for direct use in config files.
    Returns Python code string that can be copied into configs.
    """
    parent_names, _ = load_12class_names()

    # Format as Python list
    formatted = ", ".join([f'"{name}"' for name in parent_names])
    return f"class_names = [{formatted}]"


if __name__ == "__main__":
    # Test the functions
    print("=" * 80)
    print("62-CLASS NAMES (Training Classes)")
    print("=" * 80)
    names_62 = load_62class_names()
    for i, name in enumerate(names_62):
        print(f"{i:2d}: {name}")

    print("\n" + "=" * 80)
    print("12-CLASS NAMES (Parent Classes)")
    print("=" * 80)
    names_12, mapping = load_12class_names()
    for i, name in enumerate(names_12):
        print(f"{i:2d}: {name}")

    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION BY PARENT")
    print("=" * 80)
    distribution = get_class_distribution_by_parent()
    for parent, children in distribution.items():
        print(f"\n{parent} ({len(children)} classes):")
        for child in children:
            print(f"  - {child}")

    print("\n" + "=" * 80)
    print("FORMATTED FOR CONFIG FILES")
    print("=" * 80)
    print("\n# 62-class configuration:")
    print(get_62class_names_for_config())
    print("\n# 12-class configuration:")
    print(get_12class_names_for_config())
