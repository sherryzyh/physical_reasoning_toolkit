#!/usr/bin/env python3
"""
Single Problem Inference with Answer Tags

This cookbook runs inference on a single physics problem using a structured prompt
that asks the model to output reasoning in <think></think> tags and the final answer in <answer></answer> tags.

- Loads one problem from a dataset (SeePhys by default)
- Runs single inference with vision model (qwen3-vl via Ollama)
- Saves full model response as "model_response"
- Parses content between <answer> and </answer> as "model_answer"

Usage:
    python cookbooks/inference_single_with_answer_tags.py [--dataset DATASET] [--model MODEL] [--output-dir OUTPUT_DIR]

Examples:
    python cookbooks/inference_single_with_answer_tags.py
    python cookbooks/inference_single_with_answer_tags.py --dataset physreason --model qwen2-vl
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from prkit.prkit_core import PRKitLogger
from prkit.prkit_core.model_clients import create_model_client
from prkit.prkit_datasets import DatasetHub

logger = PRKitLogger.get_logger(__name__)

# Prompt instruction for structured output
ANSWER_PROMPT_INSTRUCTION = (
    "Please answer this question with reasoning. "
    "First output your reasoning process in <think></think> tags and then output the final answer in <answer> </answer> tags."
)


def parse_answer_from_response(response: str) -> Optional[str]:
    """
    Parse the content between <answer> and </answer> tags from the model response.

    Args:
        response: Full model response string

    Returns:
        Extracted answer string, or None if no <answer>...</answer> block found
    """
    if not response or not isinstance(response, str):
        return None
    # Match content between <answer> and </answer> (non-greedy, handles multiline)
    match = re.search(
        r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return None


def serialize_problem_field(value):
    """Serialize a problem field value to a JSON-serializable format."""
    if hasattr(value, "value"):
        return value.value
    elif hasattr(value, "__dict__"):
        return {k: serialize_problem_field(v) for k, v in value.__dict__.items()}
    elif isinstance(value, (list, tuple)):
        return [serialize_problem_field(item) for item in value]
    elif isinstance(value, dict):
        return {k: serialize_problem_field(v) for k, v in value.items()}
    elif isinstance(value, Path):
        return str(value)
    else:
        return value


def get_result_file_path(output_dir: Path, problem_id: str) -> Path:
    """Get the expected file path for a problem result."""
    safe_problem_id = str(problem_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    safe_problem_id = "".join(c for c in safe_problem_id if c.isalnum() or c in "._-")
    return output_dir / f"problem_{safe_problem_id}.json"


def run_single_inference(
    dataset_name: str = "seephys",
    model_name: str = "qwen3-vl",
    auto_download: bool = True,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run single inference on the first problem of a dataset.

    Args:
        dataset_name: Name of the dataset to load
        model_name: Vision model name
        auto_download: Whether to auto-download dataset if not found
        output_dir: Directory to save results
    """
    logger.info("=" * 60)
    logger.info("Single Problem Inference (Answer Tags)")
    logger.info("=" * 60)

    # 1. Create model client
    logger.info(f"\n🤖 Creating client for model: {model_name}")
    try:
        client = create_model_client(model_name)
        logger.info(f"✅ Client created successfully (provider: {client.provider})")
    except (ConnectionError, ValueError) as e:
        logger.error(f"❌ Failed to create client: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to create client: {e}")
        sys.exit(1)

    # 2. Load dataset and take first problem only
    logger.info(f"\n📖 Loading dataset: {dataset_name} (sample_size=1)")
    try:
        dataset = DatasetHub.load(
            dataset_name=dataset_name,
            sample_size=1,
            auto_download=auto_download,
        )
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        sys.exit(1)

    if len(dataset) == 0:
        logger.error("❌ Dataset is empty")
        sys.exit(1)

    problem = dataset[0]
    problem_id = problem.get("problem_id", "unknown")
    question = problem.get("question", "")

    logger.info(f"   Problem ID: {problem_id}")
    logger.info(f"   Question preview: {question[:80]}...")

    # 3. Resolve output directory
    if output_dir is None:
        output_dir = Path.cwd() / "inference_results"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n📁 Output directory: {output_dir}")

    # 4. Build prompt with instruction
    full_prompt = f"{ANSWER_PROMPT_INSTRUCTION}\n\n{question}"

    # 5. Resolve image paths
    image_paths = None
    img_path = problem.get("image_path")
    if img_path:
        if isinstance(img_path, str):
            image_paths = [img_path]
        elif isinstance(img_path, (list, tuple)):
            image_paths = [str(p) for p in img_path if p]
        if image_paths:
            logger.info(f"   Using {len(image_paths)} image(s)")

    # 6. Run inference
    logger.info("\n💬 Running inference...")
    result = {
        "problem_id": problem_id,
        "question": question,
        "model_response": None,
        "model_answer": None,
    }

    try:
        if image_paths:
            # Validate image files exist
            for p in image_paths:
                if not Path(p).exists():
                    raise FileNotFoundError(f"Image file not found: {p}")
        response = client.chat(user_prompt=full_prompt, image_paths=image_paths)
        result["model_response"] = response
        result["model_answer"] = parse_answer_from_response(response)
    except FileNotFoundError as e:
        logger.error(f"❌ Image file not found: {e}")
        result["error"] = f"Image file not found: {e}"
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        result["error"] = f"Inference failed: {e}"

    # 7. Add serializable problem fields for reference
    result["problem"] = {
        k: serialize_problem_field(v)
        for k, v in problem.items()
        if k not in ("image_path",)  # Skip large paths in problem copy if desired
    }
    result["problem"]["image_path"] = problem.get("image_path")

    # 8. Save result
    output_path = get_result_file_path(output_dir, problem_id)
    output_data = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "problem_index": 0,
        "problem_id": problem_id,
        "timestamp": datetime.now().isoformat(),
        **result,
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved result to {output_path}")
        logger.info("=" * 60)
        logger.info("✅ Done!")
    except Exception as e:
        logger.error(f"❌ Failed to save result to {output_path}: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run single inference with <think> and <answer> tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="seephys",
        help="Dataset name (default: seephys)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="qwen3-vl",
        help="Vision model name (default: qwen3-vl)",
    )
    parser.add_argument(
        "--auto-download",
        dest="auto_download",
        action="store_true",
        default=True,
        help="Auto-download dataset if not found (default: True)",
    )
    parser.add_argument(
        "--no-auto-download",
        dest="auto_download",
        action="store_false",
        help="Disable auto-download",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for results",
    )
    args = parser.parse_args()

    run_single_inference(
        dataset_name=args.dataset,
        model_name=args.model,
        auto_download=args.auto_download,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
