#!/usr/bin/env python3
"""
Cookbook 2: Automated Annotation Workflow

This cookbook demonstrates how to:
- Run an automated annotation workflow without supervision
- Process physics problems through the complete annotation pipeline
- Use the ugphysics dataset with limited sample size for showcase
- Save results to organized output directories

Prerequisites:
- prkit_annotation package installed (via: pip install physical-reasoning-toolkit)
- prkit_datasets package installed (via: pip install physical-reasoning-toolkit)
- ugphysics dataset available
- OpenAI API key set (for LLM-based annotation)

Usage:
    python 02_automated_annotation.py
"""

import pprint
import sys
import os
import json
from pathlib import Path

# Import the annotation workflow functionality
from prkit_annotation.workflows.presets import PlainAutomaticWorkflow
from prkit_datasets import DatasetHub

def main():
    """Main function demonstrating automated annotation workflow."""
    
    print("🤖 PhysKit Automated Annotation Cookbook")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set.")
        print("   This cookbook requires an OpenAI API key to run LLM-based annotation.")
        print("   Please set your API key: export OPENAI_API_KEY='your-key-here'")
        print("   Or create a .env file with: OPENAI_API_KEY=your-key-here")
        return
    
    # 1. Load the ugphysics dataset with mini variant and limited sample
    print("\n📖 Loading UGPPhysics Dataset for Annotation:")
    print("-" * 50)
    
    try:
        # Load only 3 problems for showcase
        dataset = DatasetHub.load("ugphysics", sample_size=1)
        print(f"  ✅ Successfully loaded dataset!")
        print(f"  📊 Total problems to annotate: {len(dataset)}")
        
        # Show the problems we'll be working with
        print("\n  Problems to be annotated:")
        for i, problem in enumerate(dataset):
            print(f"    {i+1}. {problem.problem_id} - {problem.domain}")
            print(f"       Question: {problem.question[:80]}{'...' if len(problem.question) > 80 else ''}")
        
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
        print(f"  Make sure the ugphysics dataset is available and physkit_datasets is properly installed.")
        return
    
    # 2. Set up output directory
    root_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = Path(root_dir) / "showcase_output" / "automated_annotation"
    
    # Clear output directory at the beginning
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # 3. Initialize the automated annotation workflow
    print("\n🔧 Initializing Annotation Workflow:")
    print("-" * 40)
    
    try:
        # Initialize with o3-mini model (you can change this to other models)
        workflow = PlainAutomaticWorkflow(
            output_dir=output_dir,
            model="o3-mini"  # You can also use "gpt-4", "gpt-3.5-turbo", etc.
        )
        print(f"  ✅ Workflow initialized successfully!")
        print(f"  🤖 Using model: {workflow.model}")
        print(f"  📂 Output directory: {workflow.output_dir}")
        
    except Exception as e:
        print(f"  ❌ Error initializing workflow: {e}")
        return
    
    # 4. Run the automated annotation
    print("\n🚀 Starting Automated Annotation:")
    print("-" * 40)
    
    try:
        print("  Processing problems through the annotation pipeline...")
        print("  This may take a few minutes depending on the LLM response time...")
        
        # Run the annotation workflow
        summary = workflow.run(dataset=dataset)
        
        print(f"  ✅ Annotation completed successfully!")
        print(f"  📊 Results summary:")
        pprint.pprint(summary)
        
    except Exception as e:
        print(f"  ❌ Error during annotation: {e}")
        print(f"  Check the log file for more details: {workflow.output_dir}/annotation_workflow.log")
        return
    
    # 5. Explore the results
    print("\n📋 Annotation Results:")
    print("-" * 30)
    
    try:
        # Check what was created
        print(f"  📁 Output directory structure:")
        for item in output_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(output_dir)
                print(f"    • {rel_path}")
        
        # Show individual problem results
        annotation_dir = output_dir / "annotation"
        if annotation_dir.exists():
            print(f"\n  📝 Individual problem annotations:")
            for problem_file in annotation_dir.glob("*.json"):
                print(f"    • {problem_file.name}")
                
                # Load and show a brief summary of one result
                try:
                    with open(problem_file, 'r') as f:
                        problem_result = json.load(f)
                    
                    problem_id = problem_result.get('problem_id', 'Unknown')
                    domain = problem_result.get('annotations', {}).get('domain', {})
                    theorem = problem_result.get('annotations', {}).get('theorem', {})
                    
                    # Extract domain and theorem information from the new structure
                    domain_info = domain.get('primary_domain', 'Unknown') if hasattr(domain, 'get') else str(domain)
                    theorem_info = theorem.get('theorem', 'Unknown') if hasattr(theorem, 'get') else str(theorem)
                    
                    print(f"      - Problem ID: {problem_id}")
                    print(f"      - Domain: {domain_info}")
                    print(f"      - Theorem: {str(theorem_info)[:50]}{'...' if len(str(theorem_info)) > 50 else ''}")
                    break  # Only show first one to avoid too much output
                except Exception as e:
                    print(f"      - Error reading result: {e}")
        
        # Show workflow log
        log_file = output_dir / "annotation_workflow.log"
        if log_file.exists():
            print(f"\n  📋 Workflow log: {log_file}")
            print(f"    (Check this file for detailed processing information)")
        
    except Exception as e:
        print(f"  ❌ Error exploring results: {e}")
    
if __name__ == "__main__":
    main()
