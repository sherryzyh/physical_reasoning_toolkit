"""
Simple annotation workflow that chains four annotators step by step.

This workflow processes physics problems through a sequential pipeline:
1. Domain annotation
2. Theorem annotation (using domain output)
3. Variable annotation (using domain and theorem outputs)
4. Final answer computation (using all previous outputs)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator
from datetime import datetime

from physkit_annotation.annotators import (
    DomainAnnotator, TheoremAnnotator, VariableAnnotator, FinalAnswerAnnotator
)
from physkit_core.models import PhysicalDataset
from physkit_core import PhysKitLogger

from ..base import BaseWorkflow


class PlainAutomaticWorkflow(BaseWorkflow):
    """
    Simple automatic annotation workflow that chains four annotators step by step.
    
    Each step's output becomes the input for the next step, creating a
    sequential pipeline for physics problem annotation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_annotator = DomainAnnotator(model=self.model)
        self.theorem_annotator = TheoremAnnotator(model=self.model)
        self.variable_annotator = VariableAnnotator(model=self.model)
        self.final_answer_annotator = FinalAnswerAnnotator(model=self.model)

        # Create llm_annotation folder for individual problem results
        self.annotation_dir = self.output_dir / "annotation"
        self.annotation_dir.mkdir(exist_ok=True)
        
          
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current status of the annotation workflow."""
        return {
            "output_directory": str(self.output_dir),
            "processing_mode": "sequential",
            "statistics": self.stats.copy(),
            "workflow_components": {
                "domain_annotator": self.domain_annotator.__class__.__name__,
                "theorem_annotator": self.theorem_annotator.__class__.__name__,
                "variable_annotator": self.variable_annotator.__class__.__name__,
                "final_answer_annotator": self.final_answer_annotator.__class__.__name__
            },
        }
        
    def run(
        self,
        dataset: PhysicalDataset,
    ) -> Dict[str, Any]:
        self.logger.info(f"Dataset: {dataset.name}")
        self.logger.info(f"Dataset size: {len(dataset)} problems")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Process problems sequentially
        results = []
        self.stats["start_time"] = datetime.now()
        
        for i, problem in enumerate(dataset):
            # Convert problem to dictionary format
            problem_data = problem.to_dict()
            problem_id = problem_data.get("problem_id", f"problem_{i}")
            question = problem_data.get("question", problem_data.get("content", ""))
            domain_gt = problem_data.get("domain", "")
            
            self.logger.info(f"Processing problem {i+1}/{len(dataset)}: {problem_id}")
            
            try:
                # Step 1: Domain annotation
                self.logger.debug("  Step 1: Domain classification...")
                domain_anno = self.domain_annotator.annotate(question)
                self.stats["domain_annotated"] += 1
                
                # Step 2: Theorem annotation (using domain output)
                self.logger.debug("  Step 2: Theorem identification...")
                theorem_anno = self.theorem_annotator.annotate(question, domain_anno)
                self.stats["theorem_annotated"] += 1
                
                # Step 3: Variable annotation (using domain and theorem outputs)
                self.logger.debug("  Step 3: Variable extraction...")
                variable_anno = self.variable_annotator.annotate(question, domain_anno, theorem_anno)
                self.stats["variable_annotated"] += 1
                
                # Step 4: Final answer computation (using all previous outputs)
                self.logger.debug("  Step 4: Final computation...")
                final_answer = self.final_answer_annotator.annotate(
                    question, domain_anno, theorem_anno, variable_anno
                )
                self.stats["final_answer_computed"] += 1
                
                # Create complete annotation result as a dictionary
                result = {
                    "problem_id": problem_id,
                    "question": question,
                    "annotations": {
                        "domain": domain_anno,
                        "theorem": theorem_anno,
                        "variable": variable_anno,
                        "final_answer": final_answer
                    },
                    "ground_truth": {
                        "domain": domain_gt
                    },
                    "metadata": {
                        "model_used": self.model,
                        "annotation_method": "sequential_chain",
                        "pipeline_steps": ["domain", "theorem", "variable", "final_answer"],
                        "problem_number": i + 1,
                        "timestamp": datetime.now().isoformat(),
                        "status": "SUCCESS"
                    }
                }
                
                results.append(result)
                self.logger.info(f"Successfully annotated problem {problem_id}")
                
                # Rate limiting between problems to avoid API limits
                if i + 1 < len(dataset):
                    self.logger.info("Waiting 1 second before next problem...")
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Failed to annotate problem {problem_id}: {e}")
                # Create error result for failed problems
                error_result = {
                    "problem_id": problem_id,
                    "question": question,
                    "annotations": {},
                    "ground_truth": {
                        "domain": domain_gt
                    },
                    "metadata": {
                        "model_used": self.model,
                        "annotation_method": "sequential_chain",
                        "pipeline_steps": [],
                        "problem_number": i + 1,
                        "timestamp": datetime.now().isoformat(),
                        "status": "FAILED",
                        "error": str(e)
                    }
                }
                results.append(error_result)
        
        # Save all results
        self._save_annotation_results(results)
        
        self.stats["end_time"] = datetime.now()
        self.stats["total_processed"] = len(results)
        self.stats["successful"] = len([r for r in results if r["metadata"]["status"] == "SUCCESS"])
        self.stats["failed"] = len([r for r in results if r["metadata"]["status"] == "FAILED"])
        
        # workflow summary
        summary = self.get_workflow_status()
        return summary

    
    def _save_annotation_results(self, results: List[Any]) -> None:
        """Save annotation results to files."""
        for result in results:
            if result and isinstance(result, dict):
                problem_id = result.get("problem_id", "unknown")
                filename = f"{problem_id}.json"
                filepath = self.annotation_dir / filename
                
                # Save the result dictionary as JSON
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
