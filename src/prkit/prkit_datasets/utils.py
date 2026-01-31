"""
Utility functions for working with physical reasoning datasets.
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prkit.prkit_core.models.physics_dataset import PhysicalDataset


def sample_balanced(
    dataset: PhysicalDataset,
    field: str,
    samples_per_category: int,
    seed: Optional[int] = None,
) -> PhysicalDataset:
    """
    Sample a balanced subset from the dataset based on a categorical field.

    Args:
        dataset: Source dataset
        field: Field name to balance on (e.g., 'difficulty', 'subject')
        samples_per_category: Number of samples to take from each category
        seed: Random seed for reproducibility

    Returns:
        Balanced subset of the dataset
    """
    if seed is not None:
        random.seed(seed)

    # Group samples by the specified field
    categories = {}
    for sample in dataset:
        category = sample.get(field, "unknown")
        if category not in categories:
            categories[category] = []
        categories[category].append(sample)

    # Sample from each category
    balanced_samples = []
    for category, samples in categories.items():
        if len(samples) >= samples_per_category:
            selected = random.sample(samples, samples_per_category)
        else:
            selected = samples  # Take all if fewer than requested
        balanced_samples.extend(selected)

    # Create info
    info = dataset.get_info().copy()
    info.update(
        {
            "balanced_on": field,
            "samples_per_category": samples_per_category,
            "categories": list(categories.keys()),
            "category_counts": {
                cat: len(samples) for cat, samples in categories.items()
            },
        }
    )

    return PhysicalDataset(balanced_samples, info, dataset.split)


def get_statistics(dataset: PhysicalDataset) -> Dict[str, Any]:
    """
    Get statistics about the dataset.

    Args:
        dataset: Dataset to analyze

    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        "total_samples": len(dataset),
        "fields": list(dataset[0].keys()) if len(dataset) > 0 else [],
    }

    # Count categorical fields
    categorical_fields = [
        "domain",
        "subject",
        "difficulty",
        "tag",
        "language",
        "dataset",
    ]
    for field in categorical_fields:
        if len(dataset) > 0 and field in dataset[0]:
            values = [
                sample[field] if field in sample else "unknown" for sample in dataset
            ]
            stats[f"{field}_distribution"] = dict(Counter(values))

    # Count images
    if len(dataset) > 0:
        has_images_count = sum(
            1 for sample in dataset if ("has_images" in sample and sample["has_images"])
        )
        stats["samples_with_images"] = has_images_count
        stats["samples_without_images"] = len(dataset) - has_images_count

    return stats


def export_to_json(
    dataset: PhysicalDataset, output_path: Union[str, Path], include_info: bool = True
) -> None:
    """
    Export dataset to JSON file.

    Args:
        dataset: Dataset to export
        output_path: Path to save the JSON file
        include_info: Whether to include dataset info in the export
    """
    output_path = Path(output_path)

    export_data = {"samples": dataset.to_list()}

    if include_info:
        export_data["info"] = dataset.get_info()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)


def filter_by_keywords(
    dataset: PhysicalDataset,
    keywords: List[str],
    fields: List[str] = None,
    case_sensitive: bool = False,
) -> PhysicalDataset:
    """
    Filter dataset by keywords in specified fields.

    Args:
        dataset: Source dataset
        keywords: List of keywords to search for
        fields: List of field names to search in. If None, searches in common text fields
        case_sensitive: Whether the search should be case sensitive

    Returns:
        Filtered dataset
    """
    if fields is None:
        fields = ["question", "content", "solution", "answer"]

    if not case_sensitive:
        keywords = [kw.lower() for kw in keywords]

    def matches_keywords(sample):
        for field in fields:
            if field in sample:
                text = str(sample[field])
                if not case_sensitive:
                    text = text.lower()

                for keyword in keywords:
                    if keyword in text:
                        return True
        return False

    return dataset.filter(matches_keywords)


def create_cross_validation_splits(
    dataset: PhysicalDataset, n_splits: int = 5, seed: Optional[int] = None
) -> List[tuple[PhysicalDataset, PhysicalDataset]]:
    """
    Create cross-validation splits of the dataset.

    Args:
        dataset: Source dataset
        n_splits: Number of splits for cross-validation
        seed: Random seed for reproducibility

    Returns:
        List of (train, validation) dataset pairs
    """
    if seed is not None:
        random.seed(seed)

    # Shuffle indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Create splits
    splits = []
    fold_size = len(dataset) // n_splits

    for i in range(n_splits):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(dataset)

        val_indices = indices[start_idx:end_idx]
        train_indices = indices[:start_idx] + indices[end_idx:]

        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)

        splits.append((train_dataset, val_dataset))

    return splits


def validate_dataset_format(
    dataset: PhysicalDataset, required_fields: List[str] = None
) -> Dict[str, Any]:
    """
    Validate dataset format and check for consistency.

    Args:
        dataset: Dataset to validate
        required_fields: List of required field names

    Returns:
        Validation report
    """
    if required_fields is None:
        required_fields = ["problem_id", "question"]

    report = {"valid": True, "issues": [], "warnings": []}

    if len(dataset) == 0:
        report["valid"] = False
        report["issues"].append("Dataset is empty")
        return report

    # Check required fields
    first_sample = dataset[0]
    missing_fields = [field for field in required_fields if field not in first_sample]
    if missing_fields:
        report["valid"] = False
        report["issues"].append(f"Missing required fields: {missing_fields}")

    # Check field consistency across samples
    all_fields = set(first_sample.keys())
    for i, sample in enumerate(dataset):
        sample_fields = set(sample.keys())
        if sample_fields != all_fields:
            report["warnings"].append(f"Sample {i} has different fields than sample 0")

    # Check for duplicate problem IDs
    if "problem_id" in first_sample:
        problem_ids = []
        for sample in dataset:
            try:
                problem_ids.append(sample["problem_id"])
            except (KeyError, AttributeError):
                # Skip if problem_id is not accessible
                continue
        duplicates = [pid for pid in set(problem_ids) if problem_ids.count(pid) > 1]
        if duplicates:
            report["warnings"].append(
                f"Duplicate problem IDs found: {duplicates[:5]}..."
            )  # Show first 5

    return report
