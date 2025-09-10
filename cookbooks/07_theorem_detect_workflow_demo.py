#!/usr/bin/env python3
"""
Cookbook 7: Theorem Label Only Workflow on PHYBench Mechanics

This cookbook demonstrates how to:
- Load the PHYBench dataset
- Filter to only include mechanics problems (mechanics + classical_mechanics)
- Run the theorem_label_only_workflow on the filtered dataset
- Analyze workflow results and performance
- Display comprehensive execution statistics

The theorem_label_only_workflow identifies relevant physical theorems, 
principles, and equations for each physics problem.

Prerequisites:
- physkit_annotation package installed
- physkit_datasets package installed
- PHYBench dataset available
- OpenAI API key set (for LLM-based annotation)

Usage:
    python 07_theorem_detect_workflow_demo.py [sample_size]
    
Examples:
    python 07_theorem_detect_workflow_demo.py 5
    python 07_theorem_detect_workflow_demo.py 10
"""

import os
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Add the physkit to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "physkit"))

# Import the preset workflow and dataset functionality
from physkit_annotation.workflows.presets.theorem_label_only_workflow import TheoremLabelOnlyWorkflow
from physkit_datasets import DatasetHub


def arg_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Theorem Label Only Workflow on PHYBench Mechanics")
    parser.add_argument("sample_size", nargs="?", type=int, default=3,
                       help="Number of problems to sample (default: 3)")
    parser.add_argument("--model", default="o3-mini",
                       help="LLM model to use (default: o3-mini)")
    parser.add_argument("--output-dir", default="/Users/yinghuan/Projects/PhysicalReasoning/physical_reasoning_toolkit/showcase_output/theorem_detect_workflow_demo",
                       help="Output directory name (default: /Users/yinghuan/Projects/PhysicalReasoning/physical_reasoning_toolkit/showcase_output/theorem_detect_workflow_demo)")
    return parser.parse_args()


def display_workflow_header(args):
    """Display the workflow demo header."""
    print("🔬 PhysKit Theorem Label Only Workflow - PHYBench Mechanics")
    print("=" * 80)
    print(f"📊 Dataset: PHYBench (filtered to mechanics problems only)")
    print(f"📏 Sample size: {args.sample_size}")
    print(f"🤖 Model: {args.model}")
    print(f"📁 Output: {args.output_dir}")
    print()


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking Prerequisites:")
    print("-" * 30)
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("  ❌ OPENAI_API_KEY environment variable not set")
        print("     This cookbook requires an OpenAI API key for LLM-based annotation")
        print("     Please set your API key: export OPENAI_API_KEY='your-key-here'")
        print("     Or create a .env file with: OPENAI_API_KEY=your-key-here")
        return False
    
    print("  ✅ OpenAI API key found")
    
    # Check available datasets
    try:
        available_datasets = DatasetHub.list_available()
        if "phybench" not in available_datasets:
            print("  ❌ PHYBench dataset not available")
            print(f"     Available datasets: {', '.join(available_datasets)}")
            return False
        
        print(f"  ✅ PHYBench dataset available")
        return True
    except Exception as e:
        print(f"  ❌ Error checking datasets: {e}")
        return False


def load_and_filter_phybench_dataset(sample_size: int):
    """Load PHYBench dataset and filter to mechanics problems only."""
    print(f"📖 Loading PHYBench Dataset (Mechanics Problems Only):")
    print("-" * 60)
    
    try:
        # Load PHYBench dataset
        print("  🔄 Loading PHYBench dataset...")
        dataset = DatasetHub.load("phybench", variant="fullques", split="test")
        print(f"  ✅ Loaded PHYBench dataset with {len(dataset)} problems")
        
        # Get dataset statistics before filtering
        original_stats = dataset.get_statistics()
        print(f"  📊 Original dataset statistics:")
        print(f"    - Total problems: {original_stats.get('total_problems', 0)}")
        print(f"    - Available domains: {list(original_stats.get('domains', {}).keys())}")
        
        # Filter to mechanics problems (both "mechanics" and "classical_mechanics")
        print("  🔍 Filtering to mechanics problems (mechanics + classical_mechanics)...")
        filtered_dataset = dataset.filter_by_domains(["mechanics", "classical_mechanics"])
        
        if len(filtered_dataset) == 0:
            print("  ❌ No mechanics problems found!")
            print("     Available domains in PHYBench:")
            for domain, count in original_stats.get('domains', {}).items():
                print(f"       - {domain}: {count} problems")
            return None
        
        print(f"  ✅ Filtered dataset: {len(filtered_dataset)} mechanics problems")
        
        # Sample the filtered dataset
        if sample_size and sample_size < len(filtered_dataset):
            print(f"  📏 Sampling {sample_size} problems...")
            sampled_dataset = filtered_dataset.sample(sample_size)
            print(f"  ✅ Sampled {len(sampled_dataset)} problems")
            return sampled_dataset
        else:
            print(f"  📏 Using all {len(filtered_dataset)} filtered problems")
            return filtered_dataset
            
    except Exception as e:
        print(f"  ❌ Error loading/filtering dataset: {e}")
        return None


def setup_output_directory(output_dir_name: str) -> Path:
    """Setup output directory for workflow results."""
    print(f"📁 Setting up Output Directory:")
    print("-" * 40)
    
    try:
        # Create output directory
        output_dir = Path(output_dir_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing files (following user preference)
        for file_path in output_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                import shutil
                shutil.rmtree(file_path)
        
        print(f"  ✅ Output directory ready: {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"  ❌ Error setting up output directory: {e}")
        return Path("/Users/yinghuan/Projects/PhysicalReasoning/physical_reasoning_toolkit/showcase_output/theorem_label_phybench_mechanics")


def create_and_run_workflow(dataset, output_dir: Path, model: str) -> Dict[str, Any]:
    """Create and run the theorem label only workflow."""
    print(f"🚀 Creating and Running Theorem Label Only Workflow:")
    print("-" * 60)
    
    try:
        # Create the workflow
        print("  🔧 Creating theorem label only workflow...")
        print(f"  🔍 Using model: {model}")
        
        # Test TheoremDetector creation separately to isolate the issue
        try:
            print("  🧪 Testing TheoremDetector creation...")
            from physkit_annotation.workers import TheoremDetector
            test_detector = TheoremDetector(model=model)
            print("  ✅ TheoremDetector created successfully")
        except Exception as detector_error:
            print(f"  ❌ TheoremDetector creation failed: {detector_error}")
            print(f"  🔍 Error type: {type(detector_error).__name__}")
            import traceback
            print(f"  📋 Full traceback:")
            traceback.print_exc()
            return None
        
        workflow = TheoremLabelOnlyWorkflow(
            output_dir=str(output_dir),
            model=model
        )
        print(f"  ✅ Workflow created: {workflow}")
        
        # Run the workflow
        print("  🏃 Running workflow on filtered dataset...")
        workflow_results = workflow.run(dataset)
        
        if workflow_results is None:
            print("  ❌ Workflow returned no results")
            return None
        
        print("  ✅ Workflow completed successfully")
        return workflow_results
        
    except Exception as e:
        print(f"  ❌ Error creating/running workflow: {e}")
        print(f"  🔍 Error type: {type(e).__name__}")
        import traceback
        print(f"  📋 Full traceback:")
        traceback.print_exc()
        return None


def analyze_workflow_results(workflow_results: Dict[str, Any], dataset_size: int):
    """Analyze and display workflow results."""
    print(f"\n📊 Analyzing Workflow Results:")
    print("=" * 50)
    
    try:
        # Extract key information
        workflow_status = workflow_results.get('workflow_status', {})
        problem_stats = workflow_status.get('problem_stats', {})
        module_results = workflow_results.get('module_results', {})
        
        # Overall workflow statistics
        print("  📈 Overall Workflow Statistics:")
        print(f"    - Workflow name: {workflow_status.get('workflow_name', 'N/A')}")
        print(f"    - Total modules: {workflow_status.get('total_modules', 0)}")
        print(f"    - Modules executed: {workflow_status.get('modules_executed', 0)}")
        print(f"    - Total problems: {problem_stats.get('total', 0)}")
        print(f"    - Problems processed: {problem_stats.get('processed', 0)}")
        print(f"    - Successful: {problem_stats.get('successful', 0)}")
        print(f"    - Failed: {problem_stats.get('failed', 0)}")
        
        # Performance metrics
        execution_time = workflow_status.get('execution_time_seconds', 0)
        if execution_time > 0:
            print(f"    - Execution time: {execution_time:.2f} seconds")
            print(f"    - Problems per minute: {workflow_status.get('problems_per_minute', 0):.2f}")
        
        # Module-level analysis
        print(f"\n  📦 Theorem Detection Performance:")
        for module_name, module_data in module_results.items():
            total_problems = module_data.get('total_problems', 0)
            successful_problems = module_data.get('successful_problems', 0)
            failed_problems = module_data.get('failed_problems', 0)
            execution_time = module_data.get('execution_time_seconds', 0)
            
            print(f"    Module: {module_name}")
            print(f"      - Problems processed: {total_problems}")
            print(f"      - Completion rate: {successful_problems}/{total_problems}")
            print(f"      - Execution time: {execution_time:.2f}s")
            
            if total_problems > 0:
                success_rate = (successful_problems / total_problems) * 100
                print(f"      - Success rate: {success_rate:.1f}%")
        
        # Theorem detection statistics
        print(f"\n  🔬 Theorem Detection Statistics:")
        for module_name, module_data in module_results.items():
            if "theorem_detector" in module_name:
                theorems_detected = module_data.get('theorems_detected', 0)
                problems_with_theorems = module_data.get('problems_with_theorems', 0)
                problems_without_theorems = module_data.get('problems_without_theorems', 0)
                average_theorems = module_data.get('average_theorems_per_problem', 0.0)
                
                print(f"    - Total theorems detected: {theorems_detected}")
                print(f"    - Problems with theorems: {problems_with_theorems}")
                print(f"    - Problems without theorems: {problems_without_theorems}")
                print(f"    - Average theorems per problem: {average_theorems:.2f}")
        
    except Exception as e:
        print(f"  ❌ Error analyzing results: {e}")


def explore_output_files(output_dir: Path):
    """Explore and display information about generated output files."""
    print("\n📁 Generated Output Files:")
    print("=" * 40)
    
    try:
        # List all files in output directory
        output_files = list(output_dir.glob("*"))
        
        if not output_files:
            print("  ❌ No output files found")
            return
        
        print(f"  📂 Output directory: {output_dir}")
        print(f"  📄 Total files: {len(output_files)}")
        print()
        
        # Show key files
        for file_path in output_files:
            file_name = file_path.name
            file_size = file_path.stat().st_size if file_path.is_file() else 0
            
            if file_path.is_file():
                print(f"  📄 {file_name} ({file_size} bytes)")
                
                # Show content preview for JSON files
                if file_name.endswith('.json'):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        if 'results' in file_name:
                            results = data.get('results', [])
                            print(f"      - Type: Workflow results")
                            print(f"      - Total results: {len(results)}")
                            
                            # Show sample result
                            if results:
                                sample = results[0]
                                problem_id = sample.get('problem_id', 'Unknown')
                                status = sample.get('status', 'Unknown')
                                print(f"      - Sample: {problem_id} | {status}")
                                
                                # Show theorem detection info
                                theorems = sample.get('theorems')
                                if theorems and isinstance(theorems, list):
                                    num_theorems = len(theorems)
                                    print(f"      - Theorems detected: {num_theorems}")
                                    
                                    if num_theorems > 0:
                                        first_theorem = theorems[0]
                                        theorem_name = first_theorem.get('name', 'Unknown')
                                        print(f"      - Sample theorem: {theorem_name}")
                        
                        elif 'status' in file_name:
                            print(f"      - Type: Workflow status")
                            print(f"      - Contains execution metadata")
                            
                    except Exception as e:
                        print(f"      - Error reading file: {e}")
        
    except Exception as e:
        print(f"  ❌ Error exploring output files: {e}")


def demonstrate_workflow_control(workflow: TheoremLabelOnlyWorkflow):
    """Demonstrate workflow control and status capabilities."""
    print("\n🎛️  Workflow Control and Status:")
    print("=" * 40)
    
    try:
        # Get current workflow status
        status = workflow.get_status()
        print(f"  📊 Current Workflow Status:")
        print(f"    - Name: {status.get('workflow_name', 'N/A')}")
        print(f"    - Total modules: {status.get('total_modules', 0)}")
        print(f"    - Modules executed: {status.get('modules_executed', 0)}")
        
        # Show workflow information
        print(f"\n  📋 Workflow Information:")
        print(f"    - Output directory: {workflow.output_dir}")
        print(f"    - Model: {workflow.model}")
        
        # Show available methods
        print(f"\n  🛠️  Available Methods:")
        print(f"    - workflow.run(dataset): Execute the workflow")
        print(f"    - workflow.get_status(): Get current status")
        print(f"    - workflow.reset(): Reset workflow state")
        
    except Exception as e:
        print(f"  ❌ Error demonstrating workflow control: {e}")


def main():
    """Main function demonstrating the theorem label only workflow on PHYBench classical mechanics."""
    args = arg_parser()
    
    # Display header
    display_workflow_header(args)
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Load and filter dataset
    dataset = load_and_filter_phybench_dataset(args.sample_size)
    if dataset is None:
        return
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Create and run workflow
    workflow_results = create_and_run_workflow(dataset, output_dir, args.model)
    if workflow_results is None:
        return
    
    # Analyze results
    analyze_workflow_results(workflow_results, len(dataset))
    
    # Explore output files
    explore_output_files(output_dir)
    
    # Demonstrate workflow control
    try:
        workflow = TheoremLabelOnlyWorkflow(
            output_dir=str(output_dir),
            model=args.model
        )
        demonstrate_workflow_control(workflow)
    except Exception as e:
        print(f"  ❌ Error demonstrating workflow control: {e}")
    
    print(f"\n🎉 Theorem label only workflow on PHYBench mechanics completed successfully!")
    print(f"📁 Check the output directory for detailed results: {output_dir}")
    print(f"🔬 The workflow has identified relevant theorems and principles for mechanics problems")


def test_theorem_detector_creation():
    """Test function to isolate TheoremDetector creation issues."""
    print("🧪 Testing TheoremDetector Creation:")
    print("=" * 50)
    
    print(f"\n🔍 Testing model: o3-mini")
    try:
        from physkit_annotation.workers import TheoremDetector
        print("  📦 Importing TheoremDetector...")
        detector = TheoremDetector(model="o3-mini")
        print(f"  ✅ Successfully created TheoremDetector with o3-mini")
        
        # Test the work method with a simple question
        print("  🧪 Testing work method...")
        test_question = "A ball is thrown upward with initial velocity v0. What is the maximum height?"
        result = detector.work(test_question)
        print(f"  ✅ Work method completed successfully")
        print(f"  📊 Result type: {type(result)}")
        if hasattr(result, 'theorems'):
            print(f"  🔬 Number of theorems detected: {len(result.theorems)}")
        
    except Exception as e:
        print(f"  ❌ Failed to create or test TheoremDetector with o3-mini: {e}")
        print(f"  🔍 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Add command line argument to run test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_theorem_detector_creation()
    else:
        main()
