#!/usr/bin/env python3
"""
Cookbook 2: Automated Annotation Workflow

This cookbook demonstrates how to:
- Run an automated annotation workflow without supervision
- Process physics problems through the complete annotation pipeline
- Use the ugphysics dataset with limited sample size for showcase
- Save results to organized output directories

Prerequisites:
- physkit_annotation package installed
- physkit_datasets package installed
- ugphysics dataset available
- OpenAI API key set (for LLM-based annotation)

Usage:
    python 02_automated_annotation.py
"""

import pprint
import sys
import os
# Add the physkit to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "physkit"))


import json
from pathlib import Path

# Import the annotation workflow functionality
from physkit_annotation import AnnotationWorkflow
from physkit_datasets import DatasetHub

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
    print(f"\n📁 Output directory: {output_dir}")
    
    # 3. Initialize the automated annotation workflow
    print("\n🔧 Initializing Annotation Workflow:")
    print("-" * 40)
    
    try:
        # Initialize with o3-mini model (you can change this to other models)
        workflow = AnnotationWorkflow(
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
        summary = workflow.run(
            dataset=dataset,
            max_problems=3,  # Ensure we only process 3 problems
            domain_filter=None  # Process all domains
        )
        
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
        annotation_dir = output_dir / "llm_annotation"
        if annotation_dir.exists():
            print(f"\n  📝 Individual problem annotations:")
            for problem_file in annotation_dir.glob("*.json"):
                print(f"    • {problem_file.name}")
                
                # Load and show a brief summary of one result
                try:
                    with open(problem_file, 'r') as f:
                        problem_result = json.load(f)
                    
                    problem_id = problem_result.get('problem_id', 'Unknown')
                    domain = problem_result.get('domain_annotation', {}).get('primary_domain', 'Unknown')
                    theorem = problem_result.get('theorem_annotation', {}).get('theorem', 'Unknown')
                    
                    print(f"      - Problem ID: {problem_id}")
                    print(f"      - Domain: {domain}")
                    print(f"      - Theorem: {theorem[:50]}{'...' if len(str(theorem)) > 50 else ''}")
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
