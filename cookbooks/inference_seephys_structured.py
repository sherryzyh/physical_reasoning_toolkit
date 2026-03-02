#!/usr/bin/env python3
"""
SeePhys Inference with Structured Output (Reason + Answer)

This cookbook runs inference on the SeePhys dataset using models that support
structured output (gpt-5-mini, gemini-2.5-flash). The model response is constrained
to two fields:
  - reason: Step-by-step reasoning process
  - answer: Final answer only (no extra text)

Uses native structured output for reliable schema adherence (OpenAI/Gemini).

Prerequisites:
  - physical-reasoning-toolkit installed
  - For OpenAI: OPENAI_API_KEY set
  - For Gemini: GEMINI_API_KEY or GOOGLE_API_KEY set
  - SeePhys dataset downloaded (use --auto-download)

Usage:
    python cookbooks/inference_seephys_structured.py [--model MODEL] [--max-problems N] [--output-dir DIR]

Examples:
    # Run on 1 problem with gpt-5-mini (default)
    python cookbooks/inference_seephys_structured.py

    # Run with gemini-2.5-flash on 10 problems
    python cookbooks/inference_seephys_structured.py --model gemini-2.5-flash --max-problems 10

    # Full dataset (may take a while)
    python cookbooks/inference_seephys_structured.py --model gpt-5-mini --max-problems 0
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from prkit.prkit_core import PRKitLogger
from prkit.prkit_core.model_clients import create_model_client
from prkit.prkit_datasets import DatasetHub

logger = PRKitLogger.get_logger(__name__)


# Structured output schema: reason (reasoning) + answer (final answer only)
class ReasonAndAnswer(BaseModel):
    """Response format with reasoning and final answer only."""

    reason: str = Field(
        description="Step-by-step reasoning process to arrive at the answer."
    )
    answer: str = Field(
        description="The final answer only, with no additional explanation or text."
    )


PROMPT_INSTRUCTION = (
    "You are a physics expert. Solve this physics problem step by step.\n\n"
    "First, explain your reasoning process in the 'reason' field.\n"
    "Then, provide ONLY the final answer in the 'answer' field — no extra text, "
    "explanations, or formatting. Just the final answer."
)


def get_image_paths(problem: Any) -> Optional[List[str]]:
    """Extract image paths from a problem (PhysicsProblem or dict)."""
    img = (
        problem.get("image_path")
        if hasattr(problem, "get")
        else getattr(problem, "image_path", None)
    )
    if not img:
        return None
    if isinstance(img, str):
        return [img] if img.strip() else None
    if isinstance(img, (list, tuple)):
        paths = [str(p) for p in img if p]
        return paths if paths else None
    return None


def run_inference(
    model_name: str = "gpt-5-mini",
    dataset_name: str = "seephys",
    max_problems: Optional[int] = 1,
    auto_download: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Run structured inference on SeePhys dataset.

    Args:
        model_name: Model to use (gpt-5-mini, gemini-2.5-flash, etc.)
        dataset_name: Dataset name (default: seephys)
        max_problems: Maximum number of problems to run (default: 1). None = all.
        auto_download: Auto-download dataset if not found
        output_dir: Directory to save results
    """
    logger.info("=" * 60)
    logger.info("SeePhys Inference with Structured Output (reason + answer)")
    logger.info("=" * 60)

    # 1. Create model client (OpenAI/Gemini support structured output)
    logger.info(f"\n🤖 Creating client for model: {model_name}")
    try:
        client = create_model_client(model_name)
        logger.info(f"✅ Client created (provider: {client.provider})")
    except (ConnectionError, ValueError) as e:
        logger.error(f"❌ Failed to create client: {e}")
        sys.exit(1)

    # 2. Load dataset (full load, no random sampling)
    logger.info(f"\n📖 Loading dataset: {dataset_name}")
    try:
        dataset = DatasetHub.load(
            dataset_name=dataset_name,
            sample_size=None,  # Load all, then take first n
            auto_download=auto_download,
        )
        if max_problems is not None:
            dataset = dataset.take(max_problems)
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        sys.exit(1)

    if len(dataset) == 0:
        logger.error("❌ Dataset is empty")
        sys.exit(1)

    logger.info(f"   Loaded first {len(dataset)} problems")

    # 3. Resolve output directory
    if output_dir is None:
        output_dir = Path.cwd() / "inference_seephys_results"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n📁 Output directory: {output_dir}")

    # 4. Run inference on each problem
    results: List[Dict[str, Any]] = []
    for i, problem in enumerate(dataset):
        problem_id = problem.get("problem_id", f"problem_{i}")
        question = problem.get("question", "")
        image_paths = get_image_paths(problem)

        logger.info(f"\n[{i + 1}/{len(dataset)}] Problem {problem_id}")

        full_prompt = f"{PROMPT_INSTRUCTION}\n\nQuestion:\n{question}"

        result_row: Dict[str, Any] = {
            "problem_id": problem_id,
            "question": question,
            "reason": None,
            "answer": None,
            "error": None,
        }

        try:
            # Validate image paths exist
            if image_paths:
                for p in image_paths:
                    if not Path(p).exists():
                        raise FileNotFoundError(f"Image not found: {p}")

            # Use higher max_output_tokens for Gemini (8192 default can truncate long physics reasoning)
            extra_kwargs = {}
            if client.provider == "google":
                extra_kwargs["max_output_tokens"] = 16384

            response_text = client.chat(
                user_prompt=full_prompt,
                image_paths=image_paths,
                response_format=ReasonAndAnswer,
                **extra_kwargs,
            )

            # Parse JSON response (response_text may be None from some providers)
            response_text = response_text or ""
            if not response_text.strip():
                raise ValueError("Model returned empty or null response")
            parsed = json.loads(response_text.strip())
            result_row["reason"] = parsed.get("reason") or ""
            result_row["answer"] = parsed.get("answer") or ""

            answer_preview = result_row["answer"]
            logger.info(
                f"   Answer: {answer_preview[:80]}{'...' if len(answer_preview) > 80 else ''}"
            )

        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
            result_row["error"] = str(e)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON: {e}")
            result_row["error"] = f"JSON parse error: {e}"
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            result_row["error"] = str(e)

        results.append(result_row)

    # 5. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"seephys_{model_name.replace('/', '_')}_{timestamp}.json"

    output_data = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "total_problems": len(results),
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n💾 Saved results to {output_file}")
    logger.info("=" * 60)
    logger.info("✅ Done!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SeePhys inference with structured output (reason + answer)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gpt-5-mini",
        help="Model name (default: gpt-5-mini). Use gemini-2.5-flash for Gemini.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="seephys",
        help="Dataset name (default: seephys)",
    )
    parser.add_argument(
        "--max-problems",
        "-n",
        type=int,
        default=1,
        help="Maximum number of problems to run (default: 1). Use 0 for full dataset.",
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

    max_problems = None if args.max_problems == 0 else args.max_problems

    run_inference(
        model_name=args.model,
        dataset_name=args.dataset,
        max_problems=max_problems,
        auto_download=args.auto_download,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
