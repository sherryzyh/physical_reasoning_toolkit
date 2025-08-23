#!/usr/bin/env python3
"""
Cookbook 6: Workflow Composition Demo

This cookbook demonstrates how to:
- Use WorkflowComposer to compose custom annotation workflows
- Add individual workflow modules to create annotation pipelines
- Run composed workflows and analyze results
- Chain modules together for complex annotation tasks

Prerequisites:
- physkit_annotation package installed
- physkit_datasets package installed
- phybench dataset available
- OpenAI API key set (for LLM-based annotation)

Usage:
    python 06_workflow_composition_demo.py
"""

import pprint
import sys
import os
# Add the physkit to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "physkit"))

import json
from pathlib import Path
from typing import Dict, Any

# Import the workflow composition functionality directly from modules
from physkit_annotation.workflows import WorkflowComposer
from physkit_annotation.workflows.modules import DomainAssessmentModule
from physkit_datasets import DatasetHub

def main():
    """Main function demonstrating workflow composition."""
    
    print("🔧 PhysKit Workflow Composition Demo")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set.")
        print("   This cookbook requires an OpenAI API key to run LLM-based annotation.")
        print("   Please set your API key: export OPENAI_API_KEY='your-key-here'")
        print("   Or create a .env file with: OPENAI_API_KEY=your-key-here")
        return
    
    # 1. Load the PHYBench dataset with limited sample
    print("\n📖 Loading PHYBench Dataset for Workflow Demo:")
    print("-" * 50)
    
    try:
        # Load only 2 problems for showcase
        dataset = DatasetHub.load("phybench", sample_size=2)
        print(f"  ✅ Successfully loaded dataset!")
        print(f"  📊 Total problems to process: {len(dataset)}")
        
        # Show the problems we'll be working with
        print("\n  Problems to be processed:")
        for i, problem in enumerate(dataset):
            problem_id = getattr(problem, 'problem_id', f'problem_{i}')
            domain = getattr(problem, 'domain', 'unknown')
            question = getattr(problem, 'question', getattr(problem, 'content', ''))
            print(f"    {i+1}. {problem_id} - {domain}")
            print(f"       Question: {question[:80]}{'...' if len(question) > 80 else ''}")
            
            # # Debug: Show detailed problem data structure
            # print(f"       Debug - Full problem data:")
            # problem_dict = problem.to_dict()
            # for key, value in problem_dict.items():
            #     if key == 'question' and len(str(value)) > 100:
            #         print(f"         {key}: {str(value)[:100]}... (type: {type(value)})")
            #     else:
            #         print(f"         {key}: {value} (type: {type(value)})")
            print()
        
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
        print(f"  Make sure the phybench dataset is available and physkit_datasets is properly installed.")
        return
    
    # 2. Set up output directory
    root_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = Path(root_dir) / "showcase_output" / "workflow_composition_demo"
    
    # Clear output directory at the beginning
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # 3. Create a custom workflow using WorkflowComposer
    print("\n🔧 Composing Custom Workflow with WorkflowComposer:")
    print("-" * 50)
    
    try:
        # Step 1: Initialize the workflow composer
        print("  Step 1: Initializing WorkflowComposer...")
        workflow = WorkflowComposer(
            name="domain_assessment_demo",
            output_dir=output_dir,
            config={
                "description": "Demo workflow showing domain assessment module composition",
                "version": "1.0.0",
                "author": "PhysKit Demo"
            }
        )
        print(f"    ✅ Workflow composer initialized: {workflow.name}")
        
        # Step 2: Create and configure the domain assessment module
        print("  Step 2: Creating DomainAssessmentModule...")
        domain_module = DomainAssessmentModule(
            name="physics_domain_classifier",
            model="o3-mini",
            config={
                "confidence_threshold": 0.7,
                "max_domains_per_problem": 3,
                "enable_detailed_logging": True
            }
        )
        print(f"    ✅ Domain module created: {domain_module.name}")
        print(f"    🤖 Using model: {domain_module.model}")
        
        # Step 3: Add the module to the workflow
        print("  Step 3: Adding module to workflow...")
        workflow.add_module(domain_module)
        print(f"    ✅ Module {domain_module.name} added to workflow")
        print(f"    📊 Total modules in workflow: {workflow.stats['total_modules']}")
        
        # Step 4: Show workflow configuration
        print("  Step 4: Workflow configuration:")
        print(f"    📋 Workflow name: {workflow.name}")
        print(f"    📁 Output directory: {workflow.output_dir}")
        print(f"    🔧 Configuration: {workflow.config}")
        
        # Show module details
        for i, module in enumerate(workflow.modules):
            print(f"    📦 Module {i+1}: {module.name} ({module.__class__.__name__})")
            print(f"       - Model: {module.model}")
        
    except Exception as e:
        print(f"  ❌ Error composing workflow: {e}")
        return
    
    # 4. Run the composed workflow
    print("\n🚀 Running Composed Workflow:")
    print("-" * 40)
    
    try:
        print("  Executing workflow with composed modules...")
        print("  This will process problems through the domain assessment pipeline...")
        
        # Run the composed workflow
        workflow_results = workflow.run(
            dataset=dataset,
            enable_progress_tracking=True,
            save_intermediate_results=True
        )
        
        print(f"  ✅ Workflow execution completed successfully!")
        print(f"  📊 Workflow summary:")
        stats = workflow_results.get('statistics', {})
        workflow_summary = stats.get('workflow_summary', {})
        print(f"    - Total modules: {workflow_summary.get('total_modules', 0)}")
        print(f"    - Problems processed: {stats.get('problem_stats', {}).get('processed', 0)}")
        print(f"    - Successful: {stats.get('successful', 0)}")
        print(f"    - Failed: {stats.get('failed', 0)}")
        print(f"    - Duration: {stats.get('duration_seconds', 0):.2f}s")
        print(f"    - Rate: {stats.get('problems_per_minute', 0):.1f} problems/min")
        
    except Exception as e:
        print(f"  ❌ Error during workflow execution: {e}")
        print(f"  Check the log files for more details.")
        return
    
    # 5. Analyze the workflow results
    print("\n📋 Workflow Results Analysis:")
    print("-" * 40)
    
    try:
        # Show workflow statistics
        stats = workflow_results["statistics"]
        print(f"  📊 Overall Workflow Statistics:")
        print(f"    - Total modules executed: {stats['modules_executed']}")
        print(f"    - Total problems processed: {stats['problem_stats']['processed']}")
        print(f"    - Successful: {stats['successful']}")
        print(f"    - Failed: {stats['failed']}")
        
        if stats.get('duration_seconds'):
            print(f"    - Execution time: {stats['duration_seconds']:.2f} seconds")
            print(f"    - Processing rate: {stats.get('problems_per_minute', 0):.2f} problems/minute")
        
        # Show data flow between modules
        print(f"\n  🔄 Data Flow Analysis:")
        for flow in workflow_results.get("data_flow", []):
            print(f"    Module {flow['module_index']+1}: {flow['module_name']}")
            print(f"      - Input size: {flow['input_size']}")
            print(f"      - Output size: {flow['output_size']}")
            print(f"      - Execution time: {flow['execution_time']:.2f}s")
        
        # Show individual module results
        print(f"\n  📦 Individual Module Results:")
        for module_name, module_result in workflow_results.get("module_results", {}).items():
            print(f"    Module: {module_name}")
            module_stats = module_result.get("statistics", {})
            print(f"      - Problems processed: {module_stats.get('problem_stats', {}).get('processed', 0)}")
            print(f"      - Success rate: {module_stats.get('problem_stats', {}).get('successful', 0)}/{module_stats.get('problem_stats', {}).get('processed', 0)}")
            
            # Show domain-specific statistics using generic method
            domain_stats = domain_module.get_statistics()
            if domain_stats.get('domains_labeled'):
                print(f"      - Domains labeled: {domain_stats['domains_labeled']}")
            if domain_stats.get('confidence_scores'):
                avg_confidence = sum(domain_stats['confidence_scores']) / len(domain_stats['confidence_scores']) if domain_stats['confidence_scores'] else 0
                print(f"      - Average confidence: {avg_confidence:.3f}")
        
    except Exception as e:
        print(f"  ❌ Error analyzing results: {e}")
    
    # 6. Explore the output files
    print("\n📁 Generated Output Files:")
    print("-" * 30)
    
    try:
        # Show only essential files
        print(f"  📂 Key files created:")
        
        # Show workflow statistics file
        workflow_stats_file = output_dir / "domain_assessment_demo_workflow_statistics.json"
        if workflow_stats_file.exists():
            print(f"    • {workflow_stats_file.relative_to(output_dir)}")
        
        # Show workflow results file
        results_file = output_dir / "domain_assessment_demo_results.json"
        if results_file.exists():
            print(f"    • {results_file.relative_to(output_dir)}")
            
            # Show brief summary
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                print(f"      - Total results: {len(results)}")
                
                # Show a sample result
                if results:
                    sample = results[0]
                    problem_id = sample.get('problem_id', 'Unknown')
                    status = sample.get('status', 'Unknown')
                    domain = sample.get('domain_to_proceed', 'Unknown')
                    print(f"      - Sample: {problem_id} | {status} | {domain}")
            except Exception as e:
                print(f"      - Error reading results: {e}")
        else:
            print("  ❌ No workflow results found")
        
    except Exception as e:
        print(f"  ❌ Error exploring output: {e}")
    
    # 7. Demonstrate workflow status and control
    print("\n🎛️  Workflow Control and Status:")
    print("-" * 40)
    
    try:
        # Get current workflow status
        status = workflow.get_workflow_status()
        print(f"  📊 Current Workflow Status:")
        print(f"    - Name: {status['workflow_name']}")
        print(f"    - Total modules: {status['total_modules']}")
        print(f"    - Modules executed: {status['modules_executed']}")
        print(f"    - Current status: {status['status']}")
        
        # Show module status
        print(f"\n  📦 Individual Module Status:")
        for module in workflow.modules:
            module_status = module.get_status()
            print(f"    - {module_status['module_name']}: {module_status['status']}")
            print(f"      * Problems processed: {module_status['statistics']['problem_stats']['processed']}")
            print(f"      * Success rate: {module_status['statistics']['problem_stats']['successful']}/{module_status['statistics']['problem_stats']['processed']}")
        
        # Demonstrate reset functionality
        print(f"\n  🔄 Workflow Reset Capability:")
        print(f"    - You can call workflow.reset() to reset all modules")
        print(f"    - This clears statistics and prepares for a new run")
        
    except Exception as e:
        print(f"  ❌ Error getting workflow status: {e}")
    
    # 8. Summary and next steps
    print("\n🎯 Summary and Next Steps:")
    print("-" * 40)
    
    print("  ✅ What we accomplished:")
    print("    • Created a custom workflow using WorkflowComposer")
    print("    • Added DomainAssessmentModule to the workflow")
    print("    • Executed the composed workflow successfully")
    print("    • Analyzed results and workflow performance")
    
    print("\n  🚀 How to extend this workflow:")
    print("    • Add more modules: workflow.add_module(NewModule(...))")
    print("    • Create conditional workflows based on module results")
    print("    • Add parallel processing for independent modules")
    print("    • Implement result validation and quality checks")
    
    print("\n  📚 Key concepts demonstrated:")
    print("    • WorkflowComposer for orchestration")
    print("    • Module composition and chaining")
    print("    • Result aggregation and statistics")
    print("    • Output management and file organization")
    
    print(f"\n🎉 Workflow composition demo completed successfully!")
    print(f"📁 Check the output directory for detailed results: {output_dir}")
    print(f"💡 Tip: The workflow statistics file contains a clean summary without redundant problem data!")

if __name__ == "__main__":
    main()
