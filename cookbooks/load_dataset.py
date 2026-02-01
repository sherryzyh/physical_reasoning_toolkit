#!/usr/bin/env python3
"""
Quick Start: Load or Download a Dataset

This cookbook demonstrates how to:
- List all available datasets
- Load a dataset (with automatic download if needed)
- Explore basic dataset information

Prerequisites:
- physical-reasoning-toolkit installed (pip install physical-reasoning-toolkit)
- For image display: Pillow (pip install Pillow) and optionally matplotlib (pip install matplotlib)

Usage:
    python cookbooks/load_dataset.py [dataset_name]
    
Examples:
    # Load specific dataset
    python cookbooks/load_dataset.py physreason

    # Load with auto-download
    python cookbooks/load_dataset.py phybench --auto-download

    # Load with sample size
    python cookbooks/load_dataset.py ugphysics --sample-size 3

    # Load and display images (requires Pillow and optionally matplotlib)
    python cookbooks/load_dataset.py seephys --display-image --sample-size 3
"""

import argparse
import sys
from pathlib import Path

# Import the dataset hub
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_core import PRKitLogger

# Set up logger
logger = PRKitLogger.get_logger(__name__)

# Try to import image display libraries
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mpimg = None


def display_image(image, problem_id, image_index):
    """
    Display an image using matplotlib or PIL's show() method.
    
    Args:
        image: PIL Image object
        problem_id: Problem ID for title
        image_index: Image index for title
    """
    if not PIL_AVAILABLE or image is None:
        logger.warning("  ⚠️  Cannot display image: PIL/Pillow not available")
        return
    
    if MATPLOTLIB_AVAILABLE:
        # Use matplotlib for better display control
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            ax.axis('off')  # Hide axes
            ax.set_title(f"Problem {problem_id} - Image {image_index}\nSize: {image.size[0]}x{image.size[1]} pixels", 
                        fontsize=12, pad=10)
            plt.tight_layout()
            
            # Display the image
            logger.info(f"  🖼️  Displaying image {image_index} for problem {problem_id}...")
            plt.show(block=False)  # Non-blocking display
            plt.pause(2)  # Show for 2 seconds
            plt.close()
            
        except Exception as e:
            logger.warning(f"  ⚠️  Error displaying with matplotlib: {e}")
            # Fallback to PIL's show()
            try:
                logger.info(f"  🖼️  Displaying image using PIL...")
                image.show()
            except Exception as e2:
                logger.warning(f"  ⚠️  Error displaying with PIL: {e2}")
    else:
        # Fallback to PIL's show() method
        try:
            logger.info(f"  🖼️  Displaying image {image_index} for problem {problem_id}...")
            logger.info(f"     (Install matplotlib for better display: pip install matplotlib)")
            image.show()
        except Exception as e:
            logger.warning(f"  ⚠️  Error displaying image: {e}")


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
        default=None,
        help="Dataset variant (uses dataset default if not specified)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split (uses dataset default if not specified)",
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
    parser.add_argument(
        "--display-image",
        action="store_true",
        help="Display images for problems that have images",
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

    # 2. Get dataset information (for display only, validation happens in hub)
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
        # Continue anyway - validation will happen during load

    # 3. Load the dataset
    logger.info(f"\n📖 Loading '{args.dataset_name}' dataset...")
    if args.variant:
        logger.info(f"  Variant: {args.variant}")
    if args.split:
        logger.info(f"  Split: {args.split}")
    if args.sample_size:
        logger.info(f"  Sample size: {args.sample_size}")
    if args.auto_download:
        logger.info("  Auto-download: enabled")

    try:
        # Prepare load arguments - only include variant/split if explicitly provided
        load_kwargs = {
            "dataset_name": args.dataset_name,
            "sample_size": args.sample_size,
            "auto_download": args.auto_download,
        }
        if args.variant is not None:
            load_kwargs["variant"] = args.variant
        if args.split is not None:
            load_kwargs["split"] = args.split
        
        dataset = DatasetHub.load(**load_kwargs)
        logger.info(f"✅ Successfully loaded {len(dataset)} problems")

        # 4. Explore sample problems
        logger.info("\n📝 Sample problems:")
        logger.info("-" * 60)
        
        # Get loader instance if display_image is enabled
        loader = None
        if args.display_image:
            try:
                loader = DatasetHub._get_loader(args.dataset_name)
            except Exception as e:
                logger.warning(f"Could not get loader for image display: {e}")
        
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
            
            # Display images if requested
            if args.display_image:
                image_path = problem.get("image_path")
                if image_path:
                    logger.info(f"\n  🖼️  Image paths found: {len(image_path) if isinstance(image_path, list) else 1}")
                    
                    if not PIL_AVAILABLE:
                        logger.warning("  ⚠️  PIL/Pillow not available. Install with: pip install Pillow")
                    elif loader is None:
                        logger.warning("  ⚠️  Could not get loader instance for image loading")
                    else:
                        try:
                            # Resolve data_dir using loader's resolve_data_dir method
                            data_dir = loader.resolve_data_dir(None, args.dataset_name)
                            
                            # Load images using the loader
                            images = loader.load_images_from_paths(
                                image_path,
                                data_dir=data_dir
                            )
                            
                            if images:
                                logger.info(f"  ✅ Loaded {len(images)} image(s)")
                                for img_idx, img in enumerate(images, 1):
                                    logger.info(f"    • Image {img_idx}: {img.size[0]}x{img.size[1]} pixels, mode: {img.mode}")
                                    display_image(img, problem.get('problem_id', f'Problem {i + 1}'), img_idx)
                            else:
                                logger.warning("  ⚠️  No images could be loaded (files may not exist)")
                        except ImportError:
                            logger.warning("  ⚠️  PIL/Pillow not installed. Install with: pip install Pillow")
                        except Exception as e:
                            logger.warning(f"  ⚠️  Error loading/displaying images: {e}")
                else:
                    logger.info("  📷 No images for this problem")

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
    except ValueError as e:
        # Validation errors from hub (invalid variant/split)
        logger.error(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
