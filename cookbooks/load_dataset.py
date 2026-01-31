#!/usr/bin/env python3
"""
Quick Start: Load or Download a Dataset

This cookbook demonstrates how to:
- List all available datasets
- Load a dataset (with automatic download if needed)
- Explore basic dataset information

Prerequisites:
- physical-reasoning-toolkit installed (pip install physical-reasoning-toolkit)

Usage:
    python cookbooks/load_dataset.py [dataset_name]
    
Examples:
    # Load default dataset (ugphysics)
    python cookbooks/load_dataset.py

    # Load specific dataset
    python cookbooks/load_dataset.py physreason

    # Load with auto-download
    python cookbooks/load_dataset.py phybench --split train --auto-download

    # Load with sample size
    python cookbooks/load_dataset.py ugphysics --sample-size 100
"""

import argparse
import sys
from pathlib import Path

# Import the dataset hub
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_core import PhysKitLogger

# Set up logger
logger = PhysKitLogger.get_logger(__name__)


def main():
    """Main function demonstrating dataset loading."""
    parser = argparse.ArgumentParser(
        description="Load or download a physical reasoning dataset"
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        default="ugphysics",
        help="Name of the dataset to load (default: ugphysics)",
    )
    parser.add_argument(
        "--variant",
        default="full",
        help="Dataset variant (default: full)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of problems to sample (default: all)",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Automatically download dataset if not found",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Physical Reasoning Toolkit - Dataset Loading")
    logger.info("=" * 60)

    # 1. List available datasets
    logger.info("📚 Listing available datasets...")
    try:
        available_datasets = DatasetHub.list_available()
        logger.info(f"Found {len(available_datasets)} available datasets:")
        for dataset in available_datasets:
            marker = "  ✓ " if dataset == args.dataset_name else "  • "
            logger.info(f"{marker}{dataset}")
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        sys.exit(1)

    # Check if requested dataset is available
    if args.dataset_name not in available_datasets:
        logger.error(
            f"Dataset '{args.dataset_name}' not found! "
            f"Available: {', '.join(available_datasets)}"
        )
        sys.exit(1)

    # 2. Get dataset information
    logger.info(f"\n🔍 Getting information for '{args.dataset_name}'...")
    try:
        dataset_info = DatasetHub.get_info(args.dataset_name)
        logger.info(f"  Name: {dataset_info.get('name', 'N/A')}")
        logger.info(f"  Description: {dataset_info.get('description', 'N/A')[:80]}...")
        logger.info(f"  Variants: {dataset_info.get('variants', [])}")
        logger.info(f"  Splits: {dataset_info.get('splits', [])}")
        logger.info(f"  Total problems: {dataset_info.get('total_problems', 'N/A')}")
    except Exception as e:
        logger.warning(f"Could not retrieve dataset info: {e}")

    # 3. Load the dataset
    logger.info(f"\n📖 Loading '{args.dataset_name}' dataset...")
    logger.info(f"  Variant: {args.variant}")
    logger.info(f"  Split: {args.split}")
    if args.sample_size:
        logger.info(f"  Sample size: {args.sample_size}")
    if args.auto_download:
        logger.info("  Auto-download: enabled")

    try:
        dataset = DatasetHub.load(
            dataset_name=args.dataset_name,
            variant=args.variant,
            split=args.split,
            sample_size=args.sample_size,
            auto_download=args.auto_download,
        )
        logger.info(f"✅ Successfully loaded {len(dataset)} problems")

        # 4. Explore sample problems
        logger.info("\n📝 Sample problems:")
        logger.info("-" * 60)
        for i, problem in enumerate(dataset[:3]):  # Show first 3 problems
            logger.info(f"\nProblem {i + 1}:")
            logger.info(f"  ID: {problem.get('problem_id', 'N/A')}")
            logger.info(f"  Domain: {problem.get('domain', 'N/A')}")
            logger.info(f"  Type: {problem.get('problem_type', 'N/A')}")
            question = problem.get("question", "")
            if question:
                preview = question[:100] + "..." if len(question) > 100 else question
                logger.info(f"  Question: {preview}")
            answer = problem.get("answer")
            if answer:
                answer_value = (
                    answer.value if hasattr(answer, "value") else str(answer)
                )
                logger.info(f"  Answer: {answer_value}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ Dataset loading completed successfully!")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"❌ Dataset not found: {e}")
        logger.info(
            "\n💡 Tip: Use --auto-download to automatically download the dataset"
        )
        logger.info(f"   Example: python {sys.argv[0]} {args.dataset_name} --auto-download")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
