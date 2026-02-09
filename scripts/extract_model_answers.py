#!/usr/bin/env python3
"""
Extract final answers from model responses using qwen3.

This script:
1. Reads JSON result files from a results directory
2. For each file, uses qwen3 to extract the final answer from model_response
3. Saves the extracted answer to a "model_answer" field in each JSON file
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from prkit.prkit_core.model_clients import create_model_client
from prkit.prkit_core import PRKitLogger

# Set up logger
logger = PRKitLogger.get_logger(__name__)


def extract_answer_with_model(
    question: str,
    model_response: str,
    ground_truth_answer: str,
    model_client,
) -> str:
    """
    Use qwen3 model to extract the final answer from a model response.
    
    Args:
        question: The original question
        model_response: The full model response
        ground_truth_answer: The ground truth answer (for reference)
        model_client: The model client instance
        
    Returns:
        The extracted final answer
    """
    prompt = f"""You are a physics expert. Given a physics problem question and a detailed model response, extract ONLY the final numerical or textual answer.

Question: {question}

Model Response:
{model_response}

Ground Truth Answer (for reference): {ground_truth_answer}

Please extract and return ONLY the final answer from the model response. This should be:
- A single number (with appropriate precision)
- Or a short text answer
- Or the option letter if it's a multiple choice question

Do not include any explanation, reasoning, or additional text. Just return the answer itself."""

    try:
        extracted_answer = model_client.chat(user_prompt=prompt)
        # Clean up the response - remove any extra whitespace or formatting
        extracted_answer = extracted_answer.strip()
        return extracted_answer
    except Exception as e:
        logger.error(f"Error extracting answer with model: {e}")
        raise


def process_json_file(
    json_path: Path,
    output_path: Path,
    model_client,
    dry_run: bool = False,
) -> bool:
    """
    Process a single JSON file to extract and save model_answer.
    
    Args:
        json_path: Path to the input JSON file
        output_path: Path to save the output JSON file
        model_client: The model client instance
        dry_run: If True, don't actually save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)
        
        # Check if model_answer already exists
        if "model_answer" in data:
            logger.info(f"  ⏭️  Skipping {json_path.name} - model_answer already exists")
            # Still copy to output directory
            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        
        # Extract required fields
        question = data.get("question", "")
        model_response = data.get("model_response", "")
        ground_truth_answer = data.get("answer", "")
        
        if not question:
            logger.warning(f"  ⚠️  Skipping {json_path.name} - missing 'question' field")
            return False
        
        if not model_response:
            logger.warning(f"  ⚠️  Skipping {json_path.name} - missing 'model_response' field")
            return False
        
        # Extract the answer using the model
        logger.info(f"  🔍 Extracting answer from {json_path.name}...")
        try:
            model_answer = extract_answer_with_model(
                question=question,
                model_response=model_response,
                ground_truth_answer=ground_truth_answer,
                model_client=model_client,
            )
            
            logger.info(f"  ✅ Extracted answer: {model_answer}")
            
            # Add model_answer to the data
            data["model_answer"] = model_answer
            
            # Save the updated JSON file to output directory
            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"  💾 Saved to {output_path}")
            else:
                logger.info(f"  🔍 [DRY RUN] Would save to {output_path} with answer: {model_answer}")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Failed to extract answer from {json_path.name}: {e}")
            return False
            
    except json.JSONDecodeError as e:
        logger.error(f"  ❌ Invalid JSON in {json_path.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"  ❌ Error processing {json_path.name}: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract final answers from model responses using qwen3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all JSON files and save to results/seephys_qwen3-vl_with_answers
  python scripts/extract_model_answers.py --results-dir results/seephys_qwen3-vl
  
  # Test with first 3 files only
  python scripts/extract_model_answers.py --results-dir results/seephys_qwen3-vl --test 3
  
  # Use a different model
  python scripts/extract_model_answers.py --results-dir results/seephys_qwen3-vl --model qwen2.5
  
  # Specify custom output directory
  python scripts/extract_model_answers.py --results-dir results/seephys_qwen3-vl --output-dir results/seephys_qwen3-vl_processed
  
  # Dry run (don't save files)
  python scripts/extract_model_answers.py --results-dir results/seephys_qwen3-vl --dry-run
        """
    )
    
    parser.add_argument(
        "--results-dir",
        "-d",
        type=str,
        required=True,
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="qwen3",
        help="Model name to use for extraction (default: qwen3)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for updated JSON files (default: results/{input_dir_name}_with_answers)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually save files, just show what would be done",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        help="Test mode: only process first N files (for testing)",
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
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: create a new directory under results with "_with_answers" suffix
        output_dir = results_dir.parent / f"{results_dir.name}_with_answers"
    
    # Find all JSON files
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"⚠️  No JSON files found in {results_dir}")
        sys.exit(0)
    
    # Limit files if in test mode
    if args.test:
        json_files = json_files[:args.test]
        logger.info(f"🧪 TEST MODE: Processing only first {args.test} file(s)")
    
    logger.info("=" * 60)
    logger.info("Extract Model Answers")
    logger.info("=" * 60)
    logger.info(f"Input directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Found {len(json_files)} JSON file(s) to process")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)
    
    # Create model client
    logger.info(f"\n🤖 Creating client for model: {args.model}")
    try:
        model_client = create_model_client(args.model)
        logger.info(f"✅ Client created successfully (provider: {model_client.provider})")
    except ConnectionError as e:
        logger.error(f"❌ Connection error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"❌ Model error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to create client: {e}")
        sys.exit(1)
    
    # Process each JSON file
    logger.info(f"\n📝 Processing {len(json_files)} file(s)...\n")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for i, json_file in enumerate(json_files, 1):
        logger.info(f"[{i}/{len(json_files)}] Processing {json_file.name}...")
        
        # Determine output path
        output_path = output_dir / json_file.name
        
        # Check if already has model_answer in input file
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "model_answer" in data:
                logger.info(f"  ⏭️  Already has model_answer: {data['model_answer']}")
                # Still copy to output directory
                if not args.dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                skipped += 1
                continue
        except Exception:
            pass  # Will be handled in process_json_file
        
        if process_json_file(json_file, output_path, model_client, dry_run=args.dry_run):
            successful += 1
        else:
            failed += 1
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"⏭️  Skipped: {skipped}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Total: {len(json_files)}")
    logger.info("=" * 60)
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
