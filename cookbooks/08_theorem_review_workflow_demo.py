#!/usr/bin/env python3
"""
Cookbook 8: Theorem Review Workflow Demo

This cookbook demonstrates how to:
- Load theorem detection results from previous experiments
- Create PhysicsProblem objects from annotated problem data
- Use ReviewTheoremModule to review predicted theorems
- Run a workflow with only theorem review functionality
- Analyze review results and human feedback

The ReviewTheoremModule allows human reviewers to evaluate predicted theorems for:
1. Relevance to the physics problem
2. Correctness of equations
3. Validity of conditions

Prerequisites:
- physkit_annotation package installed
- Theorem detection results available in experiments_outputs/theorem_detection/phybench_mechanics/results/
- OpenAI API key set (for LLM-based annotation)

Usage:
    python 08_theorem_review_workflow_demo.py [sample_size]
    
Examples:
    python 08_theorem_review_workflow_demo.py 3
    python 08_theorem_review_workflow_demo.py 5
"""

import os
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add the physkit to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "physkit"))

# Import the workflow composition functionality
from physkit_annotation.workflows import WorkflowComposer
from physkit_annotation.workflows.modules.review_theorem_module import ReviewTheoremModule
from physkit_core.models.physics_problem import PhysicsProblem


def arg_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Theorem Review Workflow Demo")
    parser.add_argument("sample_size", nargs="?", type=int, default=1,
                       help="Number of problems to review (default: 3)")
    parser.add_argument("--model", default="o3-mini",
                       help="LLM model to use (default: o3-mini)")
    parser.add_argument("--output-dir", default="theorem_review_output",
                       help="Output directory name (default: theorem_review_output)")
    parser.add_argument("--results-dir", 
                       default="/Users/yinghuan/Projects/PhysicalReasoning/physical_reasoning_toolkit/showcase_output/theorem_detect_workflow_demo/results",
                       help="Path to theorem detection results directory")
    return parser.parse_args()


def display_workflow_header(args):
    """Display the workflow demo header."""
    print("🔍 PhysKit Theorem Review Workflow Demo")
    print("=" * 60)
    print(f"📊 Sample size: {args.sample_size}")
    print(f"📁 Output: {args.output_dir}")
    print(f"📂 Results dir: {args.results_dir}")
    print()


def load_theorem_detection_results(results_dir: str, sample_size: int) -> List[Dict[str, Any]]:
    """Load theorem detection results from JSON files."""
    print("📖 Loading Theorem Detection Results:")
    print("-" * 50)
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"  ❌ Results directory not found: {results_path}")
        return []
    
    # Find all result JSON files
    result_files = list(results_path.glob("problem_*_result.json"))
    if not result_files:
        print(f"  ❌ No result files found in {results_path}")
        return []
    
    print(f"  📊 Found {len(result_files)} result files")
    
    # Load and sample results
    results = []
    for i, file_path in enumerate(result_files[:sample_size]):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # Extract the annotated problem
            annotated_problem = result_data.get('annotated_problem')
            if annotated_problem:
                results.append(annotated_problem)
                problem_id = annotated_problem.get('problem_id', f'problem_{i}')
                print(f"    ✅ Loaded {problem_id}")
            else:
                print(f"    ⚠️  No annotated_problem found in {file_path.name}")
                
        except Exception as e:
            print(f"    ❌ Error loading {file_path.name}: {e}")
    
    print(f"  📊 Successfully loaded {len(results)} problems for review")
    return results


def create_physics_problems_from_results(results: List[Dict[str, Any]]) -> List[PhysicsProblem]:
    """Convert result data to PhysicsProblem objects."""
    print("\n🔧 Creating PhysicsProblem Objects:")
    print("-" * 40)
    
    problems = []
    for i, result_data in enumerate(results):
        try:
            # Use PhysicsProblem.from_dict to create the object
            problem = PhysicsProblem.from_dict(result_data)
            problems.append(problem)
            
            problem_id = problem.problem_id
            domain = problem.domain
            theorem_count = len(problem.additional_fields.get('theorems', []))
            
            print(f"    ✅ Created {problem_id} ({domain}) with {theorem_count} theorems")
            
        except Exception as e:
            print(f"    ❌ Error creating problem {i}: {e}")
    
    print(f"  📊 Successfully created {len(problems)} PhysicsProblem objects")
    return problems


def setup_output_directory(output_dir_name: str) -> Path:
    """Set up the output directory."""
    print("\n📁 Setting up Output Directory:")
    print("-" * 35)
    
    # Get the root directory and create output path
    root_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = Path(root_dir) / "showcase_output" / output_dir_name
    
    # Clear output directory at the beginning (following user preference)
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print("  🗑️  Cleared existing output directory")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ Output directory created: {output_dir}")
    
    return output_dir


def create_and_run_workflow(problems: List[PhysicsProblem], output_dir: Path, model: str) -> Dict[str, Any]:
    """Create and run the theorem review workflow."""
    print("\n🔧 Creating and Running Theorem Review Workflow:")
    print("-" * 55)
    
    try:
        # Step 1: Initialize the workflow composer
        print("  Step 1: Initializing WorkflowComposer...")
        workflow = WorkflowComposer(
            name="theorem_review_demo",
            output_dir=output_dir,
            config={
                "description": "Theorem review workflow demo using ReviewTheoremModule",
                "version": "1.0.0",
                "author": "PhysKit Demo"
            }
        )
        print(f"    ✅ Workflow composer initialized: {workflow.name}")
        
        # Step 2: Create and configure the theorem review module
        print("  Step 2: Creating ReviewTheoremModule...")
        review_module = ReviewTheoremModule(
            name="theorem_reviewer",
            model=model,
            config={
                "enable_detailed_logging": True,
                "review_type": "theorem"
            }
        )
        print(f"    ✅ Review module created: {review_module.name}")
        print(f"    🤖 Using model: {review_module.model}")
        
        # Step 3: Add the module to the workflow
        print("  Step 3: Adding module to workflow...")
        workflow.add_module(review_module)
        print(f"    ✅ Module {review_module.name} added to workflow")
        print(f"    📊 Total modules in workflow: {len(workflow.modules)}")
        
        # Step 4: Show workflow configuration
        print("  Step 4: Workflow configuration:")
        print(f"    📋 Workflow name: {workflow.name}")
        print(f"    📁 Output directory: {workflow.output_dir}")
        print(f"    🔧 Configuration: {workflow.config}")
        
        # Show module details
        for i, module in enumerate(workflow.modules):
            print(f"    📦 Module {i+1}: {module.name} ({module.__class__.__name__})")
            print(f"       - Model: {module.model}")
        
        # Step 5: Run the workflow
        print("\n  Step 5: Executing Workflow...")
        print("    Processing problems through theorem review pipeline...")
        print("    ⚠️  Note: This will require human interaction for theorem review!")
        
        # Create a simple dataset-like object for the workflow
        class SimpleDataset:
            def __init__(self, problems):
                self.problems = problems
            
            def __iter__(self):
                return iter(self.problems)
            
            def __len__(self):
                return len(self.problems)
        
        dataset = SimpleDataset(problems)
        
        workflow_results = workflow.run(
            dataset=dataset,
            enable_progress_tracking=True,
            save_intermediate_results=True
        )
        
        print("    ✅ Workflow execution completed successfully!")
        
        # Return both the results and the workflow object for status demonstration
        return {
            'results': workflow_results,
            'workflow': workflow
        }
        
    except Exception as e:
        print(f"  ❌ Error during workflow execution: {e}")
        print("  Check the log files for more details.")
        return None


def analyze_workflow_results(workflow_data: Dict[str, Any], problem_count: int = 0):
    """Analyze and display workflow results."""
    print("\n📊 Workflow Results Analysis:")
    print("=" * 50)
    
    if not workflow_data:
        print("  ❌ No workflow data to analyze")
        return
    
    # Extract the actual workflow results
    workflow_results = workflow_data.get('results', {})
    if not workflow_results:
        print("  ❌ No workflow results to analyze")
        return
    
    try:
        # Extract key information from the correct structure
        workflow_status = workflow_results.get('workflow_status', {})
        problem_stats = workflow_status.get('problem_stats', {})
        
        # Overall workflow statistics
        print("  📈 Overall Workflow Statistics:")
        print(f"    - Workflow name: {workflow_results.get('workflow_name', 'N/A')}")
        print(f"    - Total modules: {workflow_status.get('total_modules', 0)}")
        print(f"    - Problems processed: {problem_stats.get('processed', 0)}")
        print(f"    - Successful: {problem_stats.get('successful', 0)}")
        print(f"    - Failed: {problem_stats.get('failed', 0)}")
        
        # Performance metrics
        execution_time = workflow_status.get('execution_time_seconds', 0)
        if execution_time > 0:
            print(f"    - Execution time: {execution_time:.2f} seconds")
            print(f"    - Problems per minute: {workflow_status.get('problems_per_minute', 0):.2f}")
        
        # Module-level analysis
        print("\n  📦 Module Performance:")
        module_results = workflow_results.get('module_results', {})
        
        for module_name, module_data in module_results.items():
            print(f"    Module: {module_name}")
            print(f"      - Problems processed: {module_data.get('total_problems', 0)}")
            print(f"      - Success rate: {module_data.get('successful_problems', 0)}/{module_data.get('total_problems', 0)}")
            
            # Get module-specific statistics directly from the executed workflow
            if workflow_data and 'workflow' in workflow_data:
                executed_workflow = workflow_data['workflow']
                # Find the module in the executed workflow
                for module in executed_workflow.modules:
                    if module.name == module_name:
                        module_status = module.get_status()
                        # Show theorem review specific statistics
                        if 'theorems_reviewed' in module_status:
                            print(f"      - Theorems reviewed: {module_status.get('theorems_reviewed', 0)}")
                            print(f"      - Relevant theorems: {module_status.get('relevant_theorems_count', 0)}")
                            print(f"      - Correct equations: {module_status.get('correct_equations_count', 0)}")
                            print(f"      - Valid conditions: {module_status.get('valid_conditions_count', 0)}")
                            print(f"      - Problems with relevant theorems: {module_status.get('problems_with_relevant_theorems', 0)}")
                            print(f"      - Problems without relevant theorems: {module_status.get('problems_without_relevant_theorems', 0)}")
                            # Missing theorem statistics
                            print(f"      - Missing theorems added: {module_status.get('missing_theorems_added', 0)}")
                            print(f"      - Problems with missing theorems: {module_status.get('problems_with_missing_theorems', 0)}")
                        break

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
                        with open(file_path, 'r', encoding='utf-8') as f:
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
        
        # Show workflow results
        results_file = output_dir / f"theorem_review_demo_results.json"
        if results_file.exists():
            print(f"    📋 Workflow Results ({results_file.name}):")
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
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




def main():
    """Main function demonstrating the theorem review workflow."""
    args = arg_parser()
    
    # Display header
    display_workflow_header(args)
    
    # Load theorem detection results
    results = load_theorem_detection_results(args.results_dir, args.sample_size)
    if not results:
        print("  ❌ No results loaded. Exiting.")
        return
    
    # Create PhysicsProblem objects
    problems = create_physics_problems_from_results(results)
    if not problems:
        print("  ❌ No problems created. Exiting.")
        return
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Create and run workflow
    workflow_data = create_and_run_workflow(problems, output_dir, args.model)
    if workflow_data is None:
        return
    
    # Analyze results
    analyze_workflow_results(workflow_data, len(problems))
    
    # Explore output files
    explore_output_files(output_dir)
    
    
    print(f"\n🎉 Theorem review workflow demo completed successfully!")
    print(f"📁 Check the output directory for detailed results: {output_dir}")
    print(f"💡 Tip: The review results include human feedback on theorem relevance, equation correctness, and condition validity!")


if __name__ == "__main__":
    main()

