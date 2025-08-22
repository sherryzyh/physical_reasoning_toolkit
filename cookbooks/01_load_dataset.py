#!/usr/bin/env python3
"""
Cookbook 1: Load and Explore Datasets

This cookbook demonstrates how to:
- List all available datasets
- Load a specific dataset by name (passed as argument)
- Explore dataset information and structure
- Examine individual problems
- Access different dataset splits and variants

Prerequisites:
- physkit_datasets package installed
- Dataset files available in the data directory

Usage:
    python 01_load_dataset.py [dataset_name]
    
Examples:
    python 01_load_dataset.py ugphysics
    python 01_load_dataset.py phybench
    python 01_load_dataset.py seephys
"""

import os
import json
import sys
import pprint
import argparse
from pathlib import Path

# Add the physkit to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "physkit"))

# Import the dataset loading functionality
from physkit_datasets import DatasetHub


def arg_parser():
    """Main function demonstrating dataset loading and exploration."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load and explore PhysKit datasets")
    parser.add_argument("dataset_name", nargs="?", default="ugphysics", 
                       help="Name of the dataset to load (default: ugphysics)")
    parser.add_argument("--variant", default="full", 
                       help="Dataset variant (default: full)")
    parser.add_argument("--split", default="test", 
                       help="Dataset split (default: test)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Number of problems to sample (default: all)")
    parser.add_argument("--test-all", action="store_true", default=False,
                       help="Test all datasets")
    args = parser.parse_args()
    return args


def detailed_log_of_loading(args):

    print("🚀 PhysKit Dataset Loading Cookbook")
    print("=" * 50)
    print(f"📊 Testing dataset: {args.dataset_name}")
    print(f"🔧 Variant: {args.variant}")
    print(f"✂️  Split: {args.split}")
    if args.sample_size:
        print(f"📏 Sample size: {args.sample_size}")
    print()
    
    # 1. List all available datasets
    print("📚 Available Datasets:")
    print("-" * 30)
    try:
        available_datasets = DatasetHub.list_available()
        for dataset_name in available_datasets:
            marker = "  ✅ " if dataset_name == args.dataset_name else "  • "
            print(f"{marker}{dataset_name}")
    except Exception as e:
        print(f"  ❌ Error listing datasets: {e}")
        print("  Make sure physkit_datasets is properly installed")
        return
    
    # Check if requested dataset is available
    if args.dataset_name not in available_datasets:
        print(f"\n❌ Dataset '{args.dataset_name}' not found!")
        print(f"Available datasets: {', '.join(available_datasets)}")
        return
    
    # 2. Get detailed information about the requested dataset
    print(f"\n🔍 {args.dataset_name.upper()} Dataset Information:")
    print("-" * 40)
    try:
        dataset_info = DatasetHub.get_info(args.dataset_name)
        print(f"  Name: {dataset_info.get('name', 'N/A')}")
        print(f"  Description: {dataset_info.get('description', 'N/A')}")
        print(f"  Available variants: {dataset_info.get('variants', [])}")
        print(f"  Available splits: {dataset_info.get('splits', [])}")
        print(f"  Total problems: {dataset_info.get('total_problems', 'N/A')}")
        print(f"  Domains: {dataset_info.get('domains', [])}")
        print(f"  Languages: {dataset_info.get('languages', [])}")
        
        # Show additional loader information
        loader_info = DatasetHub.get_loader_info(args.dataset_name)
        print(f"  Loader class: {loader_info.get('loader_class', 'N/A')}")
        print(f"  Loader module: {loader_info.get('loader_module', 'N/A')}")
        
    except Exception as e:
        print(f"  Error getting info: {e}")
    
    # 3. Load the requested dataset using the unified DatasetHub interface
    print(f"\n📖 Loading {args.dataset_name.upper()} Dataset:")
    print("-" * 50)
    
    print(f"🔍 Loading {args.dataset_name} dataset ({args.variant} variant)...")
    try:
        # Use the unified DatasetHub.load() method
        load_kwargs = {
            "dataset_name": args.dataset_name,
            "sample_size": args.sample_size
        }
        
        # Add data_dir if specified
        if args.data_dir:
            load_kwargs["data_dir"] = args.data_dir
            
        # Add split and variant as kwargs
        if args.split:
            load_kwargs["split"] = args.split
        if args.variant:
            load_kwargs["variant"] = args.variant
            
        print(f"🔍 Loading dataset with parameters: {load_kwargs}")
        dataset = DatasetHub.load(
            **load_kwargs
        )
        
        print(f"✅ Successfully loaded {len(dataset)} problems")
        
        # Examine individual problems
        print("\n📝 Sample Problems:")
        print("-" * 30)
        
        for i, problem in enumerate(dataset[:1]):  # Show first problem
            pprint.pprint(problem.to_dict())
        
        # 8. Save dataset information to showcase output
        root_dir = os.path.dirname(os.path.dirname(__file__))
        output_dir = Path(root_dir) / "showcase_output" / "dataset_exploration"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset summary
        summary_file = output_dir / f"{args.dataset_name}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"{args.dataset_name.upper()} Dataset Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Dataset: {args.dataset_name}\n")
            f.write(f"Variant: {args.variant}\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Total problems: {len(dataset)}\n")
            f.write(f"Variants available: {dataset_info.get('variants', [])}\n")
            f.write(f"Splits available: {dataset_info.get('splits', [])}\n\n")
        
        print(f"\n💾 Dataset summary saved to: {summary_file}")
        
        # Save sample problems
        samples_file = output_dir / f"{args.dataset_name}_sample_problems.json"
        sample_data = []
        for i, problem in enumerate(dataset[:5]):  # Save first 5 problems
            try:
                problem_dict = {
                    "problem_id": problem['problem_id'],
                    "domain": str(problem.get('domain', 'N/A')),  # Convert to string to avoid serialization issues
                    "question": problem['question'],
                    "answer": problem.get('answer', '').to_dict() if hasattr(problem.get('answer', ''), 'to_dict') else problem.get('answer', ''),
                    "solution": problem.get('solution', None),
                    "all_fields": {k: str(v) if hasattr(v, '__dict__') else v for k, v in problem.items()}
                }
                sample_data.append(problem_dict)
            except Exception as e:
                print(f"    ⚠️  Error processing problem {i}: {e}")
                continue
        
        with open(samples_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"  Sample problems saved to: {samples_file}")
        
    except Exception as e:
        print(f"❌ Error loading {args.dataset_name} dataset: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide more specific error information
        if "Multiple choice problems must have options" in str(e):
            print("\n  🔍 Specific Issue: Multiple choice problems are missing required options")
            print("     This suggests the dataset format doesn't match the loader's expectations.")
            print("     Check if the dataset has been updated or if the loader needs modification.")
        
        if "not supported between instances" in str(e):
            print("\n  🔍 Specific Issue: PhysicsDomain objects cannot be compared/sorted")
            print("     The PhysicsDomain class needs to implement comparison methods.")
            print("     This is a package-level issue that needs to be fixed in the source code.")
        
        return
    
    print(f"\n✅ Dataset loading cookbook completed successfully for {args.dataset_name}!")
    print(f"📁 Check the showcase_output/dataset_exploration/ folder for saved files.")
    print(f"\n💡 Try testing other datasets:")
    for other_dataset in available_datasets:
        if other_dataset != args.dataset_name:
            print(f"    python 01_load_dataset.py {other_dataset}")


def iterate_all_datasets(args):
    """Test all available datasets with minimal logging and error reporting only."""
    print("🧪 Testing all available datasets...")
    
    available_datasets = DatasetHub.list_available()
    print(f"Found {len(available_datasets)} datasets to test")
    
    results = {"success": [], "failed": []}
    
    for dataset_name in available_datasets:
        try:
            # Test basic info retrieval
            dataset_info = DatasetHub.get_info(dataset_name)
            
            # Test dataset loading with minimal sample
            dataset = DatasetHub.load(
                dataset_name=dataset_name,
                variant="full" if "full" in dataset_info.get("variants", []) else dataset_info.get("variants", [""])[0],
                split="test" if "test" in dataset_info.get("splits", []) else dataset_info.get("splits")[0],
                sample_size=1  # Only load 1 problem to test functionality
            )
            
            # Test basic dataset operations
            problem_count = len(dataset)
            first_problem = dataset[0] if problem_count > 0 else None
            
            # Test problem access and basic fields
            if first_problem:
                # Test basic field access
                _ = first_problem.get('problem_id', '')
                _ = first_problem.get('question', '')
                _ = first_problem.get('domain', '')
            
            results["success"].append(dataset_name)
            
        except Exception as e:
            results["failed"].append((dataset_name, str(e)))
    
    # Print results summary
    print(f"\n✅ Successfully tested: {len(results['success'])} datasets")
    if results["failed"]:
        print(f"❌ Failed to test: {len(results['failed'])} datasets")
        print("\nErrors:")
        for dataset_name, error in results["failed"]:
            print(f"  {dataset_name}: {error}")
    
    return results
        

def main():
    args = arg_parser()
    
    if args.test_all:
        iterate_all_datasets(args)
    else:
        detailed_log_of_loading(args)

if __name__ == "__main__":
    main()
