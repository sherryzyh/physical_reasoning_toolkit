#!/usr/bin/env python3
"""
Evaluate inference results using accuracy evaluator with normalized match comparator.

This script:
1. Reads JSON result files from an inference results directory
2. Evaluates each result using AccuracyEvaluator with NormalizedMatchComparator
3. Outputs detailed statistics and a CSV file with comparison results including normalized values
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd

from prkit.prkit_core import PRKitLogger
from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_evaluation.evaluator.accuracy import AccuracyEvaluator
from prkit.prkit_evaluation.evaluator.base import BaseEvaluator
from prkit.prkit_evaluation.comparator.normalized_match import NormalizedMatchComparator
from prkit.prkit_evaluation.comparator.base import BaseComparator
from prkit.prkit_datasets.loaders.base_loader import detect_answer_type

# Set up logger
logger = PRKitLogger.get_logger(__name__)


def create_comparator(comparator_name: str) -> BaseComparator:
    """
    Create a comparator instance from a string name.
    
    Args:
        comparator_name: Name of the comparator (e.g., "normalized_match")
        
    Returns:
        Comparator instance
        
    Raises:
        ValueError: If comparator name is not recognized
    """
    comparator_name_lower = comparator_name.lower().replace("_", "").replace("-", "")
    
    if comparator_name_lower in ["normalizedmatch", "normalized_match"]:
        return NormalizedMatchComparator()
    else:
        raise ValueError(
            f"Unknown comparator: {comparator_name}. "
            f"Available comparators: normalized_match"
        )


def create_evaluator(evaluator_name: str, comparator: BaseComparator) -> BaseEvaluator:
    """
    Create an evaluator instance from a string name.
    
    Args:
        evaluator_name: Name of the evaluator (e.g., "accuracy")
        comparator: Comparator instance to use
        
    Returns:
        Evaluator instance
        
    Raises:
        ValueError: If evaluator name is not recognized
    """
    evaluator_name_lower = evaluator_name.lower().replace("_", "").replace("-", "")
    
    if evaluator_name_lower in ["accuracy", "accuracyevaluator"]:
        return AccuracyEvaluator(comparator=comparator)
    else:
        raise ValueError(
            f"Unknown evaluator: {evaluator_name}. "
            f"Available evaluators: accuracy"
        )


def create_answer_from_value(value: Any) -> Answer:
    """
    Create an Answer object from a value, auto-detecting the type.
    
    Args:
        value: The answer value (can be string, number, etc.)
        
    Returns:
        Answer object with detected type
    """
    if value is None:
        raise ValueError("Answer value cannot be None")
    
    # Convert to string for type detection
    value_str = str(value).strip()
    
    # Detect answer type
    answer_type = detect_answer_type(value_str)
    
    # For numerical answers, try to convert to float/int
    if answer_type.value == "numerical":
        try:
            # Try to parse as float first
            num_value = float(value_str)
            # If it's a whole number, use int
            if num_value.is_integer():
                num_value = int(num_value)
            return Answer(value=num_value, answer_type=answer_type)
        except ValueError:
            # If parsing fails, keep as string
            return Answer(value=value_str, answer_type=answer_type)
    
    # For other types, use string value
    return Answer(value=value_str, answer_type=answer_type)


def get_normalized_value(
    comparator: NormalizedMatchComparator,
    answer: Answer
) -> str:
    """
    Get the normalized value of an answer using the comparator.
    
    Args:
        comparator: The NormalizedMatchComparator instance
        answer: The Answer object to normalize
        
    Returns:
        String representation of the normalized value
    """
    answer_str = str(answer.value)
    category, normalized_value = comparator._normalize_answer(answer_str)
    
    # Convert normalized value to string
    if isinstance(normalized_value, float):
        # For floats, use a consistent format
        # If it's a whole number, display as int
        if normalized_value.is_integer():
            return str(int(normalized_value))
        else:
            # Use scientific notation for very small/large numbers, otherwise decimal
            if abs(normalized_value) < 1e-6 or abs(normalized_value) > 1e6:
                return f"{normalized_value:.6e}"
            else:
                return f"{normalized_value:.10f}".rstrip('0').rstrip('.')
    else:
        return str(normalized_value)


def process_json_file(json_path: Path) -> Dict[str, Any] | None:
    """
    Process a single JSON file and extract evaluation data.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary with evaluation data or None if processing fails
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)
        
        # Extract required fields
        problem_id = data.get("problem_id", json_path.stem)
        ground_truth_value = data.get("answer")
        model_answer_value = data.get("model_answer")
        
        if ground_truth_value is None:
            logger.warning(f"  ⚠️  Skipping {json_path.name} - missing 'answer' field")
            return None
        
        if model_answer_value is None:
            logger.warning(f"  ⚠️  Skipping {json_path.name} - missing 'model_answer' field")
            return None
        
        # Create Answer objects
        try:
            ground_truth_answer = create_answer_from_value(ground_truth_value)
            model_answer = create_answer_from_value(model_answer_value)
        except Exception as e:
            logger.error(f"  ❌ Failed to create Answer objects from {json_path.name}: {e}")
            return None
        
        return {
            "problem_id": problem_id,
            "ground_truth_answer": ground_truth_answer,
            "model_answer": model_answer,
            "ground_truth_value": str(ground_truth_value),
            "model_answer_value": str(model_answer_value),
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"  ❌ Invalid JSON in {json_path.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"  ❌ Error processing {json_path.name}: {e}")
        return None


def extract_metadata_from_results(results_dir: Path) -> Dict[str, Any]:
    """
    Extract metadata from inference result files.
    
    Args:
        results_dir: Directory containing JSON result files
        
    Returns:
        Dictionary with aggregated metadata
    """
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        return {}
    
    # Read first file to get common metadata
    try:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            first_data = json.load(f)
        
        metadata = {
            "dataset_name": first_data.get("dataset_name"),
            "model_name": first_data.get("model_name"),
            "inference_timestamp": first_data.get("inference_timestamp"),
            "timestamp": first_data.get("timestamp"),
        }
        
        # Count total files
        metadata["total_result_files"] = len(json_files)
        
        return {k: v for k, v in metadata.items() if v is not None}
    except Exception as e:
        logger.warning(f"Failed to extract metadata: {e}")
        return {}


def evaluate_results(
    results_dir: Path,
    output_dir: Path,
    comparator: BaseComparator,
    evaluator: BaseEvaluator,
    comparator_name: str,
    evaluator_name: str,
) -> None:
    """
    Evaluate all inference results and save outputs.
    
    Args:
        results_dir: Directory containing JSON result files
        output_dir: Directory to save evaluation outputs
        comparator: Comparator instance to use
        evaluator: Evaluator instance to use
        comparator_name: Name of the comparator (for config)
        evaluator_name: Name of the evaluator (for config)
    """
    # Ensure comparator is NormalizedMatchComparator
    if not isinstance(comparator, NormalizedMatchComparator):
        logger.error("❌ This script requires NormalizedMatchComparator")
        sys.exit(1)
    
    # Find all JSON files
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"⚠️  No JSON files found in {results_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON file(s) to process")
    
    # Extract metadata from results
    inference_metadata = extract_metadata_from_results(results_dir)
    
    # Process each file and collect results
    logger.info(f"\n📝 Processing {len(json_files)} file(s)...\n")
    
    all_results: List[Dict[str, Any]] = []
    successful = 0
    failed = 0
    
    for i, json_file in enumerate(json_files, 1):
        logger.info(f"[{i}/{len(json_files)}] Processing {json_file.name}...")
        
        # Extract data from JSON file
        file_data = process_json_file(json_file)
        if file_data is None:
            failed += 1
            continue
        
        # Evaluate the answer
        try:
            eval_result = evaluator.evaluate(
                predicted_answer=file_data["model_answer"],
                ground_truth_answer=file_data["ground_truth_answer"],
            )
            
            # Get normalized values
            normalized_answer = get_normalized_value(
                comparator, file_data["ground_truth_answer"]
            )
            normalized_model_answer = get_normalized_value(
                comparator, file_data["model_answer"]
            )
            
            # Collect result
            result = {
                "problem_id": file_data["problem_id"],
                "answer": file_data["ground_truth_value"],
                "model_answer": file_data["model_answer_value"],
                "normalized_answer": normalized_answer,
                "normalized_model_answer": normalized_model_answer,
                "accuracy_score": eval_result["accuracy_score"],
            }
            all_results.append(result)
            successful += 1
            
            if eval_result["accuracy_score"] == 1.0:
                logger.info(f"  ✅ Correct (score: {eval_result['accuracy_score']})")
            else:
                logger.info(f"  ❌ Incorrect (score: {eval_result['accuracy_score']})")
                
        except Exception as e:
            logger.error(f"  ❌ Evaluation failed for {json_file.name}: {e}")
            failed += 1
    
    # Calculate statistics
    if not all_results:
        logger.error("❌ No results to evaluate!")
        return
    
    accuracy_scores = [r["accuracy_score"] for r in all_results]
    overall_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    correct_count = sum(1 for score in accuracy_scores if score == 1.0)
    incorrect_count = len(accuracy_scores) - correct_count
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV file
    csv_path = output_dir / "evaluation_results.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"\n💾 Saved CSV results to {csv_path}")
    
    # Save statistics
    stats = {
        "overall_accuracy": overall_accuracy,
        "total_problems": len(all_results),
        "correct": correct_count,
        "incorrect": incorrect_count,
        "successful_evaluations": successful,
        "failed_evaluations": failed,
    }
    
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Saved statistics to {stats_path}")
    
    # Save detailed configuration
    config = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "evaluation_config": {
            "evaluator": evaluator_name,
            "evaluator_class": type(evaluator).__name__,
            "comparator": comparator_name,
            "comparator_class": type(comparator).__name__,
        },
        "inference_metadata": inference_metadata,
        "input_directory": str(results_dir.absolute()),
        "output_directory": str(output_dir.absolute()),
        "statistics": stats,
    }
    
    config_path = output_dir / "evaluation_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Saved evaluation configuration to {config_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    logger.info(f"Correct: {correct_count}")
    logger.info(f"Incorrect: {incorrect_count}")
    logger.info(f"Total Evaluated: {len(all_results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate inference results using accuracy evaluator with normalized match comparator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate results from seephys_qwen3-vl (default: normalized_match comparator, accuracy evaluator)
  python scripts/evaluate_inference_normalized_match.py \\
      --results-dir results/inference/seephys_qwen3-vl \\
      --output-dir results/evaluation/seephys_qwen3-vl_norm
  
  # Use custom comparator and evaluator
  python scripts/evaluate_inference_normalized_match.py \\
      --results-dir results/inference/seephys_qwen3-vl \\
      --output-dir results/evaluation/seephys_qwen3-vl_norm \\
      --comparator normalized_match \\
      --evaluator accuracy
  
  # Use custom paths
  python scripts/evaluate_inference_normalized_match.py \\
      --results-dir /path/to/inference/results \\
      --output-dir /path/to/evaluation/output
        """
    )
    
    parser.add_argument(
        "--results-dir",
        "-d",
        type=str,
        required=True,
        help="Directory containing JSON result files from inference",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save evaluation outputs (CSV and statistics)",
    )
    parser.add_argument(
        "--comparator",
        "-c",
        type=str,
        default="normalized_match",
        help="Comparator to use for answer comparison (default: normalized_match). "
             "Available: normalized_match",
    )
    parser.add_argument(
        "--evaluator",
        "-e",
        type=str,
        default="accuracy",
        help="Evaluator to use for evaluation (default: accuracy). "
             "Available: accuracy",
    )
    
    args = parser.parse_args()
    
    # Validate results directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"❌ Results directory does not exist: {results_dir}")
        sys.exit(1)
    
    if not results_dir.is_dir():
        logger.error(f"❌ Path is not a directory: {results_dir}")
        sys.exit(1)
    
    # Create output directory path
    output_dir = Path(args.output_dir)
    
    # Create comparator and evaluator
    logger.info("=" * 60)
    logger.info("Evaluate Inference Results (Normalized Match)")
    logger.info("=" * 60)
    logger.info(f"Input directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Comparator: {args.comparator}")
    logger.info(f"Evaluator: {args.evaluator}")
    logger.info("=" * 60)
    
    try:
        # Create comparator
        logger.info(f"\n🔧 Creating comparator: {args.comparator}")
        comparator = create_comparator(args.comparator)
        logger.info(f"✅ Comparator created: {type(comparator).__name__}")
        
        # Create evaluator
        logger.info(f"🔧 Creating evaluator: {args.evaluator}")
        evaluator = create_evaluator(args.evaluator, comparator)
        logger.info(f"✅ Evaluator created: {type(evaluator).__name__}")
        
        # Run evaluation
        evaluate_results(
            results_dir=results_dir,
            output_dir=output_dir,
            comparator=comparator,
            evaluator=evaluator,
            comparator_name=args.comparator,
            evaluator_name=args.evaluator,
        )
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
