#!/usr/bin/env python3
"""
Cookbook 3: Supervised Annotation Workflow

This cookbook demonstrates how to:
- Run a supervised annotation workflow with human review
- Process physics problems with human-in-the-loop quality control
- Use the ugphysics dataset for showcase

Prerequisites:
- physkit_annotation package installed
- physkit_datasets package installed
- ugphysics dataset available
- OpenAI API key set (for LLM-based annotation)

Usage:
    python 03_supervised_annotation.py
"""

import os
import sys
# Add the physkit to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "physkit"))

from pathlib import Path

# Import the supervised annotation workflow functionality
from physkit_annotation import SupervisedAnnotationWorkflow
from physkit_datasets import DatasetHub

def main():
    """Main function demonstrating supervised annotation workflow."""
    
    print("PhysKit Supervised Annotation Cookbook")
    print("=" * 40)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    # 1. Load the ugphysics dataset
    print("Loading UGPPhysics Dataset...")
    
    try:
        dataset = DatasetHub.load("ugphysics", sample_size=5)
        print(f"Dataset loaded: {len(dataset)} problems")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 2. Set up output directory
    root_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = Path(root_dir) / "showcase_output" / "supervised_annotation"
    
    # 3. Initialize the supervised annotation workflow
    print("Initializing workflow...")
    
    try:
        workflow = SupervisedAnnotationWorkflow(
            output_dir=output_dir,
            model="gpt-4o"
        )
        print(f"Workflow initialized with model: {workflow.model}")
        
    except Exception as e:
        print(f"Error initializing workflow: {e}")
        return
    
    # 4. Run the supervised annotation workflow
    print("Running supervised annotation workflow...")
    
    try:
        summary = workflow.run(dataset=dataset)
        
        print(f"Workflow completed successfully!")
        print(f"Problems processed: {summary['problems_processed']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        
        # Show comprehensive domain annotation statistics
        domain_stats = workflow.stats.get('summary_of_domain', {})
        if domain_stats:
            print(f"\nDomain Annotation Statistics:")
            print(f"  Assessment Results:")
            assessment_results = domain_stats.get('assessment_results', {})
            for result, count in assessment_results.items():
                if count > 0:
                    print(f"    {result}: {count}")
            
            print(f"  Revision Types:")
            revision_types = domain_stats.get('revision_types', {})
            for rev_type, count in revision_types.items():
                if count > 0:
                    print(f"    {rev_type}: {count}")
            
            human_review_count = domain_stats.get('human_review_required', 0)
            print(f"  Human Review Required: {human_review_count}")
            
            # Calculate success rates
            total_assessed = sum(assessment_results.values())
            if total_assessed > 0:
                print(f"  Success Rates:")
                exact_rate = assessment_results.get("True", 0) / total_assessed
                semantic_rate = assessment_results.get("SemanticallyTrue", 0) / total_assessed
                human_approved_rate = assessment_results.get("HumanApproved", 0) / total_assessed
                human_corrected_rate = assessment_results.get("HumanCorrected", 0) / total_assessed
                overall_correct = (assessment_results.get("True", 0) + 
                                 assessment_results.get("SemanticallyTrue", 0) + 
                                 assessment_results.get("HumanApproved", 0)) / total_assessed
                
                print(f"    Exact Match: {exact_rate:.1%}")
                print(f"    Semantic Match: {semantic_rate:.1%}")
                print(f"    Human Approved: {human_approved_rate:.1%}")
                print(f"    Human Corrected: {human_corrected_rate:.1%}")
                print(f"    Overall Correct: {overall_correct:.1%}")
        
        # Disclaimer about human review functionality in showcase
        print(f"\nNote: This is a showcase demonstration.")
        print("Human approval and correction counts may not reflect actual human review.")
        print("The workflow simulates human review scenarios for demonstration purposes.")
        
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        return
    
    print("\nCookbook completed successfully!")

if __name__ == "__main__":
    main()
