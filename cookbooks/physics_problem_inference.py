#!/usr/bin/env python3
"""
Quick Start: Visual Inference with Vision Models

This cookbook demonstrates how to:
- Load visual physics problems from datasets (SeePhys by default)
- Run inference with vision models (qwen3-vl via Ollama)
- Process problems with images and text questions

Prerequisites:
- physical-reasoning-toolkit installed (pip install physical-reasoning-toolkit)
- Ollama installed and running (https://ollama.com/download)
- Model pulled in Ollama: `ollama pull qwen3-vl`
- Dataset downloaded (will auto-download if --auto-download is used)

Usage:
    python cookbooks/visual_inference.py [--dataset DATASET] [--model MODEL] [--sample-size N] [--auto-download]
    
Examples:
    # Run with default settings (seephys, qwen3-vl, 3 samples)
    python cookbooks/visual_inference.py
    
    # Use a different dataset
    python cookbooks/visual_inference.py --dataset physreason
    
    # Use a different model
    python cookbooks/visual_inference.py --model qwen2-vl
    
    # Use more samples
    python cookbooks/visual_inference.py --sample-size 5
    
    # Auto-download dataset if not found
    python cookbooks/visual_inference.py --auto-download
"""

import argparse
import sys
from pathlib import Path

# Import the model client and dataset hub
from prkit.prkit_core.model_clients import create_model_client
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_core import PRKitLogger

# Set up logger
logger = PRKitLogger.get_logger(__name__)


def run_visual_inference(
    dataset_name: str = "seephys",
    model_name: str = "qwen3-vl",
    sample_size: int = 3,
    auto_download: bool = False,
):
    """
    Run visual inference on dataset samples.
    
    Args:
        dataset_name: Name of the dataset to load (default: seephys)
        model_name: Name of the vision model to use (default: qwen3-vl)
        sample_size: Number of problems to process (default: 3)
        auto_download: Whether to auto-download dataset if not found
    """
    logger.info("=" * 60)
    logger.info("Visual Inference with Vision Models")
    logger.info("=" * 60)
    
    # 1. Create model client
    logger.info(f"\n🤖 Creating client for model: {model_name}")
    try:
        client = create_model_client(model_name)
        logger.info(f"✅ Client created successfully (provider: {client.provider})")
    except ConnectionError as e:
        logger.error(f"❌ Connection error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"❌ Model error: {e}")
        logger.info(f"\n💡 Tip: Pull the model first: `ollama pull {model_name}`")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to create client: {e}")
        sys.exit(1)
    
    # 2. Load dataset
    logger.info(f"\n📚 Loading {dataset_name} dataset (sample size: {sample_size})...")
    try:
        dataset = DatasetHub.load(
            dataset_name=dataset_name,
            sample_size=sample_size,
            auto_download=auto_download,
        )
        logger.info(f"✅ Successfully loaded {len(dataset)} problems")
    except FileNotFoundError as e:
        logger.error(f"❌ Dataset not found: {e}")
        logger.info("\n💡 Tip: Use --auto-download to automatically download the dataset")
        logger.info("   Example: python cookbooks/visual_inference.py --auto-download")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)
    
    # 3. Process each problem
    logger.info("\n" + "=" * 60)
    logger.info("Running Visual Inference")
    logger.info("=" * 60)
    
    for i, problem in enumerate(dataset, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Problem {i}/{len(dataset)}")
        logger.info(f"{'='*60}")
        
        # Display problem information
        problem_id = problem.get("problem_id", f"Problem-{i}")
        question = problem.get("question", "")
        domain = problem.get("domain", "Unknown")
        image_paths = problem.get("image_path", [])
        
        logger.info(f"\n📋 Problem ID: {problem_id}")
        logger.info(f"📐 Domain: {domain}")
        logger.info(f"❓ Question: {question[:200]}{'...' if len(question) > 200 else ''}")
        
        # Display image information if available
        if image_paths:
            logger.info(f"🖼️  Images: {len(image_paths)} image(s)")
            for idx, img_path in enumerate(image_paths, 1):
                logger.info(f"   • Image {idx}: {Path(img_path).name}")
        else:
            logger.info("📷 No images for this problem (text-only)")
        
        # Prepare prompt
        if image_paths:
            prompt = f"""You are a physics expert. Please analyze the image(s) and answer the following question.

Question: {question}

Please provide a detailed answer based on what you see in the image(s)."""
        else:
            prompt = f"""You are a physics expert. Please answer the following question.

Question: {question}

Please provide a detailed answer."""
        
        # Run inference
        logger.info(f"\n💬 Running inference with {model_name}...")
        try:
            # Model client handles both cases: with images or text-only
            response = client.chat(user_prompt=prompt, image_paths=image_paths)
            
            logger.info("\n" + "-" * 60)
            logger.info("Model Response:")
            logger.info("-" * 60)
            logger.info(response)
            logger.info("-" * 60)
            
            # Display ground truth answer if available
            answer = problem.get("answer")
            if answer:
                answer_value = (
                    answer.value if hasattr(answer, "value") else str(answer)
                )
                logger.info(f"\n✅ Ground Truth Answer: {answer_value}")
            
            logger.info("✅ Inference completed successfully!")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Image file not found: {e}")
            logger.info("   Skipping this problem...")
            continue
        except ConnectionError as e:
            logger.error(f"❌ Connection error during inference: {e}")
            logger.info("   Skipping this problem...")
            continue
        except ValueError as e:
            logger.error(f"❌ Model error: {e}")
            logger.info("   Skipping this problem...")
            continue
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            logger.info("   Skipping this problem...")
            continue
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Visual inference completed!")
    logger.info("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run visual inference on physics datasets with vision models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (seephys, qwen3-vl, 3 samples)
  python cookbooks/visual_inference.py
  
  # Use a different dataset
  python cookbooks/visual_inference.py --dataset physreason
  
  # Use a different model
  python cookbooks/visual_inference.py --model qwen2-vl
  
  # Use more samples
  python cookbooks/visual_inference.py --sample-size 5
  
  # Auto-download dataset if not found
  python cookbooks/visual_inference.py --auto-download
        """
    )
    
    parser.add_argument(
        "--dataset",
        "-d",
        default="seephys",
        help="Dataset name to load (default: seephys)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="qwen3-vl",
        help="Vision model name (default: qwen3-vl). Must be pulled in Ollama first.",
    )
    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        default=3,
        help="Number of problems to process (default: 3)",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Automatically download dataset if not found",
    )
    
    args = parser.parse_args()
    
    run_visual_inference(
        dataset_name=args.dataset,
        model_name=args.model,
        sample_size=args.sample_size,
        auto_download=args.auto_download,
    )


if __name__ == "__main__":
    main()
