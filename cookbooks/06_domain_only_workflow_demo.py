#!/usr/bin/env python3
"""
Cookbook 6: Domain-Only Workflow Demo

This cookbook demonstrates how to use the DomainOnlyWorkflow preset:
- Initialize and configure the preset workflow
- Run domain annotation on physics problems
- Analyze workflow results and performance
- Display comprehensive execution statistics

The DomainOnlyWorkflow is a pre-configured workflow that handles domain
classification automatically without requiring manual module configuration.

Prerequisites:
- physkit_annotation package installed
- physkit_datasets package installed
- ugphysics dataset available (or any physics dataset)
- OpenAI API key set (for LLM-based annotation)

Usage:
    python 06_domain_only_workflow_demo.py [dataset_name] [sample_size]
    
Examples:
    python 06_domain_only_workflow_demo.py ugphysics 5
    python 06_domain_only_workflow_demo.py phybench 3
    python 06_domain_only_workflow_demo.py seephys 10
"""

import os
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Import the preset workflow and dataset functionality
from prkit_annotation.workflows.presets.domain_only_workflow import DomainOnlyWorkflow
from prkit_datasets import DatasetHub


def arg_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Domain-Only Workflow Demo")
    parser.add_argument("dataset_name", nargs="?", default="phybench", 
                       help="Name of the dataset to process (default: phybench)")
    parser.add_argument("sample_size", nargs="?", type=int, default=2,
                       help="Number of problems to sample (default: 2)")
    parser.add_argument("--model", default="o3-mini",
                       help="LLM model to use (default: o3-mini)")
    parser.add_argument("--output-dir", default="domain_workflow_output",
                       help="Output directory name (default: domain_workflow_output)")
    return parser.parse_args()


def display_workflow_header(args):
    """Display the workflow demo header."""
    print("🔬 PhysKit Domain-Only Workflow Demo")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset_name}")
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
        print(f"  ✅ Available datasets: {', '.join(available_datasets)}")
        return True
    except Exception as e:
        print(f"  ❌ Error checking datasets: {e}")
        return False


def load_dataset(dataset_name: str, sample_size: int) -> Any:
    """Load the specified dataset with sample size."""
    print(f"📖 Loading {dataset_name.upper()} Dataset:")
    print("-" * 40)
    
    try:
        # Load dataset with sample size
        dataset = DatasetHub.load(dataset_name, sample_size=sample_size)
        print(f"  ✅ Successfully loaded dataset!")
        print(f"  📊 Total problems: {len(dataset)}")
        
        # Show sample problems
        print(f"\n  📋 Sample Problems:")
        for i, problem in enumerate(dataset):
            problem_id = getattr(problem, 'problem_id', f'problem_{i}')
            domain = getattr(problem, 'domain', 'unknown')
            question = getattr(problem, 'question', getattr(problem, 'content', ''))
            
            print(f"    {i+1}. {problem_id}")
            print(f"       Domain: {domain}")
            print(f"       Question: {question[:300]}{'...' if len(question) > 300 else ''}")
            print()
        
        return dataset
        
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
        print(f"  Make sure the {dataset_name} dataset is available")
        return None


def setup_output_directory(output_dir_name: str) -> Path:
    """Set up the output directory."""
    print("📁 Setting up Output Directory:")
    print("-" * 35)
    
    # Get the root directory and create output path
    root_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = Path(root_dir) / "showcase_output" / output_dir_name
    
    # Clear output directory at the beginning (following user preference)
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print(f"  🗑️  Cleared existing output directory")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ Output directory created: {output_dir}")
    
    return output_dir


def create_and_run_workflow(dataset: Any, output_dir: Path, model: str) -> Dict[str, Any]:
    """Create and run the DomainOnlyWorkflow preset."""
    print("🔧 Creating and Running Domain-Only Workflow:")
    print("-" * 50)
    
    try:
        # Step 1: Initialize the preset workflow
        print("  Step 1: Initializing DomainOnlyWorkflow...")
        workflow = DomainOnlyWorkflow(
            output_dir=str(output_dir),
            model=model,
            config={
                "description": "Domain-only annotation workflow demo",
                "version": "1.0.0",
                "author": "PhysKit Demo"
            }
        )
        print(f"    ✅ Workflow initialized: {workflow}")
        
        # Step 2: Show workflow configuration
        print("  Step 2: Workflow Configuration:")
        workflow_status = workflow.get_status()
        print(f"    📋 Workflow name: {workflow_status['workflow_name']}")
        print(f"    📦 Total modules: {workflow_status['total_modules']}")
        print(f"    🤖 Model: {model}")
        print(f"    📁 Output directory: {workflow.output_dir}")
        
        # Step 3: Run the workflow
        print("\n  Step 3: Executing Workflow...")
        print("    Processing problems through domain assessment pipeline...")
        print("    (This preset workflow handles all module configuration automatically)")
        
        workflow_results = workflow.run(dataset)
        
        print(f"    ✅ Workflow execution completed successfully!")
        
        return workflow_results
        
    except Exception as e:
        print(f"  ❌ Error during workflow execution: {e}")
        print(f"  Check the log files for more details.")
        return None


def analyze_workflow_results(workflow_results: Dict[str, Any], dataset_size: int):
    """Analyze and display workflow results."""
    print("\n📊 Workflow Results Analysis:")
    print("=" * 50)
    
    if not workflow_results:
        print("  ❌ No workflow results to analyze")
        return
    
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
        
        # Module-level analysis (high-level, not exposing implementation details)
        print(f"\n  📦 Workflow Performance:")
        total_module_time = 0
        total_module_executions = 0
        
        for module_name, module_data in module_results.items():
            total_problems = module_data.get('total_problems', 0)
            successful_problems = module_data.get('successful_problems', 0)
            failed_problems = module_data.get('failed_problems', 0)
            execution_time = module_data.get('execution_time_seconds', 0)
            
            total_module_time += execution_time
            total_module_executions += total_problems
            
            print(f"    Module: {module_name}")
            print(f"      - Problems processed: {total_problems}")
            print(f"      - Completion rate: {successful_problems}/{total_problems}")
            print(f"      - Execution time: {execution_time:.2f}s")
            
            if total_problems > 0:
                success_rate = (successful_problems / total_problems) * 100
                print(f"      - Success rate: {success_rate:.1f}%")
        
        # Problem execution summary
        print(f"\n  🔄 Problem Processing Summary:")
        problem_stats = workflow_status.get('problem_stats', {})
        total_problems = problem_stats.get('total', 0)
        successful_problems = problem_stats.get('successful', 0)
        failed_problems = problem_stats.get('failed', 0)
        
        print(f"    - Total problems: {total_problems}")
        print(f"    - Successful: {successful_problems}")
        print(f"    - Failed: {failed_problems}")
        
        # Workflow summary
        workflow_summary = workflow_status.get('workflow_summary', {})
        if workflow_summary:
            print(f"\n  📋 Workflow Summary:")
            print(f"    - Total execution time: {workflow_summary.get('total_execution_time_minutes', 0):.2f} minutes")
            print(f"    - Success rate: {workflow_summary.get('success_rate_percentage', 0):.1f}%")
            print(f"    - Processing rate: {workflow_summary.get('problems_per_minute', 0):.2f} problems/minute")
            print(f"    - Average module time: {workflow_summary.get('average_module_execution_time', 0):.2f} seconds")
        
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
                            print(f"      - Type: Workflow results")
                            print(f"      - Content: {len(data)} items")
                        elif 'status' in file_name:
                            print(f"      - Type: Workflow status")
                            print(f"      - Contains: {list(data.keys())}")
                        else:
                            print(f"      - Type: JSON data")
                            print(f"      - Keys: {list(data.keys())}")
                    except Exception as e:
                        print(f"      - Error reading file: {e}")
            else:
                print(f"  📁 {file_name}/ (directory)")
        
        # Show file contents for key files
        print(f"\n  🔍 Key File Contents:")
        
        # Show workflow status
        status_file = output_dir / f"domain_only_annotation_status.json"
        if status_file.exists():
            print(f"    📊 Workflow Status ({status_file.name}):")
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)
                
                problem_stats = status.get('problem_stats', {})
                print(f"      - Problems: {problem_stats.get('total', 0)} total, {problem_stats.get('successful', 0)} successful")
                print(f"      - Execution time: {status.get('execution_time_seconds', 0):.2f}s")
                
            except Exception as e:
                print(f"      - Error reading status: {e}")
        
        # Show workflow results
        results_file = output_dir / f"domain_only_annotation_results.json"
        if results_file.exists():
            print(f"    📋 Workflow Results ({results_file.name}):")
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                print(f"      - Total results: {len(results)}")
                
                # Show sample result
                if results:
                    sample = results[0]
                    problem_id = sample.get('problem_id', 'Unknown')
                    status = sample.get('status', 'Unknown')
                    print(f"      - Sample: {problem_id} | {status}")
                    
                    # Show execution flow for sample
                    execution_flow = sample.get('execution_flow', [])
                    if execution_flow:
                        print(f"      - Execution flow: {len(execution_flow)} modules")
                        for module_exec in execution_flow:
                            module_name = module_exec.get('module_name', 'Unknown')
                            module_status = module_exec.get('status', 'Unknown')
                            print(f"        • {module_name}: {module_status}")
                
            except Exception as e:
                print(f"      - Error reading results: {e}")
        
    except Exception as e:
        print(f"  ❌ Error exploring output files: {e}")


def demonstrate_workflow_control(workflow: DomainOnlyWorkflow):
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
    """Main function demonstrating the DomainOnlyWorkflow preset."""
    args = arg_parser()
    
    # Display header
    display_workflow_header(args)
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Load dataset
    dataset = load_dataset(args.dataset_name, args.sample_size)
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
        workflow = DomainOnlyWorkflow(
            output_dir=str(output_dir),
            model=args.model
        )
        demonstrate_workflow_control(workflow)
    except Exception as e:
        print(f"  ❌ Error demonstrating workflow control: {e}")
    
    print(f"\n🎉 Domain-only workflow demo completed successfully!")
    print(f"📁 Check the output directory for detailed results: {output_dir}")


if __name__ == "__main__":
    main()
