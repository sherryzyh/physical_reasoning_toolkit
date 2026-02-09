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
    python cookbooks/physical_problem_inference.py [--dataset DATASET] [--model MODEL] [--max-problems N] [--auto-download] [--output-dir OUTPUT_DIR] [--workers N]
    
Examples:
    # Run with default settings (seephys, qwen3-vl, process all problems, 4 workers)
    python cookbooks/physical_problem_inference.py
    
    # Use a different dataset
    python cookbooks/physical_problem_inference.py --dataset physreason
    
    # Use a different model
    python cookbooks/physical_problem_inference.py --model qwen2-vl
    
    # Process first 50 problems with 8 parallel workers (faster)
    python cookbooks/physical_problem_inference.py --max-problems 50 --workers 8
    
    # Auto-download dataset if not found
    python cookbooks/physical_problem_inference.py --auto-download
    
    # Specify output directory (results saved one per file)
    python cookbooks/physical_problem_inference.py --output-dir my_results
    
    # Use more workers for faster processing (if GPU can handle it)
    python cookbooks/physical_problem_inference.py --workers 8
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

# Import the model client and dataset hub
from prkit.prkit_core.model_clients import create_model_client
from prkit.prkit_datasets import DatasetHub
from prkit.prkit_core import PRKitLogger

# Set up logger
logger = PRKitLogger.get_logger(__name__)


def serialize_problem_field(value):
    """
    Serialize a problem field value to a JSON-serializable format.
    
    Handles special cases like answer objects that may have a .value attribute.
    """
    if hasattr(value, "value"):
        # Handle answer objects with .value attribute
        return value.value
    elif hasattr(value, "__dict__"):
        # Handle other objects - convert to dict
        return {k: serialize_problem_field(v) for k, v in value.__dict__.items()}
    elif isinstance(value, (list, tuple)):
        # Handle lists/tuples
        return [serialize_problem_field(item) for item in value]
    elif isinstance(value, dict):
        # Handle nested dicts
        return {k: serialize_problem_field(v) for k, v in value.items()}
    elif isinstance(value, Path):
        # Convert Path objects to strings
        return str(value)
    else:
        # Return as-is if already serializable
        return value


def save_problem_result(result, output_dir, problem_id, index, dataset_name, model_name):
    """
    Save a single problem result to a JSON file.
    
    Args:
        result: Dictionary containing problem data and inference results
        output_dir: Directory to save the file
        problem_id: Problem ID for filename
        index: Problem index for filename
        dataset_name: Dataset name for metadata
        model_name: Model name for metadata
    """
    # Sanitize problem_id for filename (remove invalid characters)
    safe_problem_id = str(problem_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    safe_problem_id = "".join(c for c in safe_problem_id if c.isalnum() or c in "._-")
    
    # Create filename: problem_{problem_id}.json
    filename = f"problem_{safe_problem_id}.json"
    output_path = output_dir / filename
    
    # Prepare output data with metadata
    output_data = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "problem_index": index,
        "problem_id": problem_id,
        "timestamp": datetime.now().isoformat(),
        **result  # Include all problem fields and inference results
    }
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved result to {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save result to {output_path}: {e}")
        return False


def display_statistics(successful_count, failed_count, inference_times, current_problem_index, total_problems):
    """
    Display current statistics after each problem.
    
    Args:
        successful_count: Number of successful inferences
        failed_count: Number of failed inferences
        inference_times: List of inference times in seconds
        current_problem_index: Current problem index (1-based)
        total_problems: Total number of problems
    """
    total_processed = successful_count + failed_count
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    
    logger.info("\n" + "-" * 60)
    logger.info("📊 Current Statistics:")
    logger.info(f"   Problems processed: {current_problem_index}/{total_problems}")
    logger.info(f"   Solved (successful): {successful_count}")
    logger.info(f"   Failed: {failed_count}")
    logger.info(f"   Average inference time: {avg_time:.2f} seconds")
    if inference_times:
        logger.info(f"   Total inference time: {sum(inference_times):.2f} seconds")
    logger.info("-" * 60)


def save_final_statistics(output_dir, dataset_name, model_name, max_problems, successful_count, failed_count, inference_times, start_time):
    """
    Save final statistics to a JSON file in the output directory.
    
    Args:
        output_dir: Directory to save the statistics file
        dataset_name: Name of the dataset
        model_name: Name of the model
        max_problems: Maximum number of problems processed (None if all)
        successful_count: Number of successful inferences
        failed_count: Number of failed inferences
        inference_times: List of inference times in seconds
        start_time: Start time of the inference run
    """
    total_processed = successful_count + failed_count
    total_time = time.time() - start_time
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    min_inference_time = min(inference_times) if inference_times else 0.0
    max_inference_time = max(inference_times) if inference_times else 0.0
    
    statistics = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "max_problems": max_problems,
        "total_problems_processed": total_processed,
        "successful_inferences": successful_count,
        "failed_inferences": failed_count,
        "success_rate": (successful_count / total_processed * 100) if total_processed > 0 else 0.0,
        "timing": {
            "total_time_seconds": round(total_time, 2),
            "total_time_formatted": f"{total_time:.2f} seconds",
            "average_inference_time_seconds": round(avg_inference_time, 2),
            "average_inference_time_formatted": f"{avg_inference_time:.2f} seconds",
            "min_inference_time_seconds": round(min_inference_time, 2),
            "max_inference_time_seconds": round(max_inference_time, 2),
            "total_inference_time_seconds": round(sum(inference_times), 2),
        },
        "timestamp": datetime.now().isoformat(),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
    }
    
    stats_file = output_dir / "statistics.json"
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Final statistics saved to {stats_file.name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save statistics: {e}")
        return False


def process_single_problem(
    problem_data,
    index,
    total_problems,
    model_name,
    output_dir,
    dataset_name,
    stats_lock,
    stats_dict,
):
    """
    Process a single problem with inference (Step 1 and Step 2).
    
    This function is designed to be called in parallel by ThreadPoolExecutor.
    Each thread creates its own model client to avoid thread safety issues.
    
    Args:
        problem_data: Tuple of (problem dict, problem_id)
        index: Problem index (1-based)
        total_problems: Total number of problems
        model_name: Model name for creating client
        output_dir: Output directory for saving results
        dataset_name: Dataset name for metadata
        stats_lock: Thread lock for statistics
        stats_dict: Dictionary to store statistics (thread-safe)
        
    Returns:
        Tuple of (success: bool, inference_time: float, problem_id: str)
    """
    problem, problem_id = problem_data
    question = problem.get("question", "")
    image_paths = problem.get("image_path", [])
    
    # Create a client instance for this thread
    try:
        client = create_model_client(model_name)
    except Exception as e:
        with stats_lock:
            stats_dict["failed_count"] += 1
        logger.error(f"❌ Problem {index}: Failed to create client: {e}")
        return (False, 0.0, problem_id)
    
    inference_start_time = time.time()
    
    try:
        # Step 1: Run inference with question and images
        step1_response = client.chat(user_prompt=question, image_paths=image_paths)
        step1_time = time.time() - inference_start_time
        
        # Step 2: Extract answer from step 1 response
        step2_start_time = time.time()
        step2_prompt = f"""Question: {question}

Model's response: {step1_response}

Based on the question and the model's response above, provide only the final answer."""
        
        step2_response = client.chat(user_prompt=step2_prompt, image_paths=None)
        step2_time = time.time() - step2_start_time
        total_inference_time = step1_time + step2_time
        
        # Collect result
        result = {}
        for key, value in problem.items():
            result[key] = serialize_problem_field(value)
        result["model_response"] = step1_response
        result["model_answer"] = step2_response
        result["inference_timestamp"] = datetime.now().isoformat()
        result["inference_time_seconds"] = round(total_inference_time, 2)
        result["step1_time_seconds"] = round(step1_time, 2)
        result["step2_time_seconds"] = round(step2_time, 2)
        
        # Save result
        save_problem_result(result, output_dir, problem_id, index, dataset_name, model_name)
        
        # Update statistics (thread-safe)
        with stats_lock:
            stats_dict["successful_count"] += 1
            stats_dict["inference_times"].append(total_inference_time)
        
        logger.info(f"✅ Problem {index}/{total_problems} ({problem_id}): Completed in {total_inference_time:.2f}s")
        return (True, total_inference_time, problem_id)
        
    except FileNotFoundError as e:
        inference_time = time.time() - inference_start_time
        logger.error(f"❌ Problem {index}: Image file not found: {e}")
        result = {}
        for key, value in problem.items():
            result[key] = serialize_problem_field(value)
        result["model_response"] = None
        result["model_answer"] = None
        result["inference_timestamp"] = datetime.now().isoformat()
        result["inference_time_seconds"] = round(inference_time, 2)
        result["error"] = f"Image file not found: {e}"
        save_problem_result(result, output_dir, problem_id, index, dataset_name, model_name)
        with stats_lock:
            stats_dict["failed_count"] += 1
            stats_dict["inference_times"].append(inference_time)
        return (False, inference_time, problem_id)
        
    except Exception as e:
        inference_time = time.time() - inference_start_time
        logger.error(f"❌ Problem {index} ({problem_id}): Inference failed: {e}")
        result = {}
        for key, value in problem.items():
            result[key] = serialize_problem_field(value)
        result["model_response"] = None
        result["model_answer"] = None
        result["inference_timestamp"] = datetime.now().isoformat()
        result["inference_time_seconds"] = round(inference_time, 2)
        result["error"] = f"Inference failed: {e}"
        save_problem_result(result, output_dir, problem_id, index, dataset_name, model_name)
        with stats_lock:
            stats_dict["failed_count"] += 1
            stats_dict["inference_times"].append(inference_time)
        return (False, inference_time, problem_id)


def run_visual_inference(
    dataset_name: str = "seephys",
    model_name: str = "qwen3-vl",
    max_problems: int = None,
    auto_download: bool = True,
    output_dir: str = None,
    num_workers: int = 4,
):
    """
    Run visual inference on dataset samples.
    
    Args:
        dataset_name: Name of the dataset to load (default: seephys)
        model_name: Name of the vision model to use (default: qwen3-vl)
        max_problems: Maximum number of problems to process (first N problems sequentially) (default: None, process all)
        auto_download: Whether to auto-download dataset if not found
        output_dir: Directory to save results (one JSON file per problem) (default: auto-generated)
        num_workers: Number of parallel workers for processing (default: 4)
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
    
    # 2. Load dataset (full dataset, no random sampling)
    logger.info(f"\n📚 Loading {dataset_name} dataset...")
    logger.info(f"\n📚 Auto-download: {auto_download}")
    try:
        # Load full dataset without random sampling
        dataset = DatasetHub.load(
            dataset_name=dataset_name,
            sample_size=None,  # Load full dataset
            auto_download=auto_download,
        )
        logger.info(f"✅ Successfully loaded {len(dataset)} problems")
        
        # Limit to first N problems if max_problems is specified
        if max_problems is not None and max_problems > 0:
            original_count = len(dataset)
            dataset = dataset[:max_problems]
            logger.info(f"📊 Limited to first {len(dataset)} problems (out of {original_count} total)")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)
    
    # 3. Set up output directory
    if output_dir is None:
        # Generate default output directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"inference_results_{dataset_name}_{model_name}_{timestamp}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n📁 Output directory: {output_dir}")
    logger.info(f"   Results will be saved immediately after each problem")
    
    # 4. Process problems in parallel
    logger.info("\n" + "=" * 60)
    logger.info("Running Visual Inference")
    logger.info("=" * 60)
    logger.info(f"🚀 Using {num_workers} parallel workers for faster processing")
    
    # Prepare problem data with indices
    problem_list = []
    for i, problem in enumerate(dataset, 1):
        problem_id = problem.get("problem_id", f"Problem-{i}")
        problem_list.append((problem, problem_id, i))
    
    # Thread-safe statistics tracking
    stats_lock = Lock()
    stats_dict = {
        "successful_count": 0,
        "failed_count": 0,
        "inference_times": [],
        "processed_count": 0,
    }
    
    start_time = time.time()
    
    # Process problems in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_problem = {}
        for problem, problem_id, index in problem_list:
            future = executor.submit(
                process_single_problem,
                (problem, problem_id),
                index,
                len(dataset),
                model_name,
                output_dir,
                dataset_name,
                stats_lock,
                stats_dict,
            )
            future_to_problem[future] = (index, problem_id)
        
        # Process completed tasks and show progress
        completed = 0
        for future in as_completed(future_to_problem):
            index, problem_id = future_to_problem[future]
            completed += 1
            try:
                success, inference_time, pid = future.result()
                with stats_lock:
                    stats_dict["processed_count"] = completed
                    # Display progress every 10 problems or at the end
                    if completed % 10 == 0 or completed == len(dataset):
                        successful = stats_dict["successful_count"]
                        failed = stats_dict["failed_count"]
                        times = stats_dict["inference_times"]
                        avg_time = sum(times) / len(times) if times else 0.0
                        logger.info(
                            f"📊 Progress: {completed}/{len(dataset)} | "
                            f"✅ {successful} | ❌ {failed} | "
                            f"⏱️  Avg: {avg_time:.2f}s"
                        )
            except Exception as e:
                logger.error(f"❌ Problem {index} ({problem_id}): Unexpected error: {e}")
    
    # Extract final statistics
    successful_count = stats_dict["successful_count"]
    failed_count = stats_dict["failed_count"]
    inference_times = stats_dict["inference_times"]
    
    # Summary and save final statistics
    total_processed = successful_count + failed_count
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Visual inference completed!")
    logger.info("=" * 60)
    logger.info(f"\n📊 Final Summary:")
    logger.info(f"   Output directory: {output_dir}")
    logger.info(f"   Total problems processed: {total_processed}")
    logger.info(f"   Successful inferences: {successful_count}")
    logger.info(f"   Failed inferences: {failed_count}")
    if successful_count > 0:
        success_rate = (successful_count / total_processed * 100)
        logger.info(f"   Success rate: {success_rate:.1f}%")
    if inference_times:
        logger.info(f"   Average inference time: {avg_inference_time:.2f} seconds")
        logger.info(f"   Total inference time: {sum(inference_times):.2f} seconds")
        logger.info(f"   Min inference time: {min(inference_times):.2f} seconds")
        logger.info(f"   Max inference time: {max(inference_times):.2f} seconds")
    logger.info(f"   Total elapsed time: {total_time:.2f} seconds")
    logger.info(f"\n💾 All results saved individually in: {output_dir}")
    
    # Save final statistics to JSON file
    save_final_statistics(output_dir, dataset_name, model_name, max_problems, 
                         successful_count, failed_count, inference_times, start_time)
    
    logger.info("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run visual inference on physics datasets with vision models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (seephys, qwen3-vl, process all problems)
  python cookbooks/physical_problem_inference.py
  
  # Use a different dataset
  python cookbooks/physical_problem_inference.py --dataset physreason
  
  # Use a different model
  python cookbooks/physical_problem_inference.py --model qwen2-vl
  
  # Process first 50 problems with 8 parallel workers (faster)
  python cookbooks/physical_problem_inference.py --max-problems 50 --workers 8
  
  # Auto-download dataset if not found
  python cookbooks/physical_problem_inference.py --auto-download
  
  # Specify output directory (results saved one per file)
  python cookbooks/physical_problem_inference.py --output-dir my_results
  
  # Use more workers for faster processing (if GPU can handle it)
  python cookbooks/physical_problem_inference.py --workers 8
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
        "--max-problems",
        "-n",
        type=int,
        default=None,
        help="Maximum number of problems to process (first N problems sequentially, default: None, process all)",
    )
    parser.add_argument(
        "--auto-download",
        type=bool,
        default=True,
        help="Automatically download dataset if not found",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for results (one JSON file per problem) (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers for processing (default: 4). Increase for faster processing if GPU can handle it.",
    )
    
    args = parser.parse_args()
    
    run_visual_inference(
        dataset_name=args.dataset,
        model_name=args.model,
        max_problems=args.max_problems,
        auto_download=args.auto_download,
        output_dir=args.output_dir,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
