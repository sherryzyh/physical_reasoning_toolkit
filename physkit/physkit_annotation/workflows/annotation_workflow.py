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
import logging

from ..full_physics_annotation import FullPhysicsAnnotation
from ..annotators import (
    DomainAnnotator, TheoremAnnotator, VariableAnnotator, FinalAnswerAnnotator
)
from physkit_core.models import PhysicalDataset


class AnnotationWorkflow:
    """
    Simple annotation workflow that chains four annotators step by step.
    
    Each step's output becomes the input for the next step, creating a
    sequential pipeline for physics problem annotation.
    """
    
    def __init__(self, output_dir: Union[str, Path], model: str = "gpt-4o"):
        """
        Initialize the annotation workflow.
        
        Args:
            output_dir: Directory to save annotation results
            model: LLM model to use (default: gpt-4o)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create llm_annotation folder for individual problem results
        self.annotation_dir = self.output_dir / "llm_annotation"
        self.annotation_dir.mkdir(exist_ok=True)
        
        # Initialize individual annotators for the annotation pipeline
        self.domain_annotator = DomainAnnotator(model=model)
        self.theorem_annotator = TheoremAnnotator(model=model)
        self.variable_annotator = VariableAnnotator(model=model)
        self.final_answer_annotator = FinalAnswerAnnotator(model=model)
        
        self.model = model
        
        # Setup logging
        self._setup_logging()
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "domain_annotated": 0,
            "theorem_annotated": 0,
            "variable_annotated": 0,
            "final_answer_computed": 0,
            "start_time": None,
            "end_time": None
        }
    

    def _setup_logging(self):
        """Setup logging configuration."""
        # Use existing logger if already configured, otherwise create a basic one
        self.logger = logging.getLogger(__name__)
    
    def run(
        self,
        dataset: PhysicalDataset,
        max_problems: Optional[int] = None,
        domain_filter: Optional[str] = None,
        start_from: int = 0,
        auto_save: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        self.logger.info(f"Starting annotation workflow")
        self.logger.info(f"Dataset: {dataset.__class__.__name__}")
        self.logger.info(f"Dataset size: {len(dataset)} problems")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Apply domain filter if specified
        if domain_filter:
            dataset = dataset.filter(lambda x: domain_filter.lower() in x.get('domain', '').lower())
            self.logger.info(f"Filtered to {len(dataset)} problems in domain: {domain_filter}")
        
        # Limit number of problems if specified
        if max_problems and len(dataset) > max_problems:
            dataset = dataset.select(list(range(max_problems)))
            self.logger.info(f"Limited to {max_problems} problems")
        
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
                self.logger.info(f"  Step 1: Domain classification...")
                domain_anno = self.domain_annotator.annotate(question)
                self.stats["domain_annotated"] += 1
                
                # Step 2: Theorem annotation (using domain output)
                self.logger.info(f"  Step 2: Theorem identification...")
                theorem_anno = self.theorem_annotator.annotate(question, domain_anno)
                self.stats["theorem_annotated"] += 1
                
                # Step 3: Variable annotation (using domain and theorem outputs)
                self.logger.info(f"  Step 3: Variable extraction...")
                variable_anno = self.variable_annotator.annotate(question, domain_anno, theorem_anno)
                self.stats["variable_annotated"] += 1
                
                # Step 4: Final answer computation (using all previous outputs)
                self.logger.info(f"  Step 4: Final computation...")
                final_answer = self.final_answer_annotator.annotate(
                    question, domain_anno, theorem_anno, variable_anno
                )
                self.stats["final_answer_computed"] += 1
                
                # Create complete annotation result
                result = FullPhysicsAnnotation(
                    problem_id=problem_id,
                    question=question,
                    domain_annotation=domain_anno,
                    theorem_annotation=theorem_anno,
                    variable_annotation=variable_anno,
                    final_answer=final_answer,
                    domain_gt=domain_gt,
                    metadata={
                        "model_used": self.model,
                        "annotation_method": "sequential_chain",
                        "pipeline_steps": ["domain", "theorem", "variable", "final_answer"],
                        "problem_number": i + 1,
                        "timestamp": datetime.now().isoformat()
                    },
                    timestamp=datetime.now().isoformat()
                )
                
                results.append(result)
                self.logger.info(f"Successfully annotated problem {problem_id}")
                
                # Rate limiting between problems to avoid API limits
                if i + 1 < len(dataset):
                    self.logger.info("Waiting 1 second before next problem...")
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Failed to annotate problem {problem_id}: {e}")
                results.append(None)
        
        # Save all results
        self._save_annotation_results(results)
        
        self.stats["end_time"] = datetime.now()
        self.stats["total_processed"] = len(results)
        self.stats["successful"] = len([r for r in results if r is not None])
        self.stats["failed"] = len([r for r in results if r is None])
        
        # Save final statistics
        self._save_statistics()
        
        # workflow summary
        summary = self.get_workflow_status()
        return summary
    
    def _save_annotation_results(self, results: List[FullPhysicsAnnotation]):
        """Save annotation results to files."""
        for result in results:
            if result is not None:
                filename = f"{result.problem_id}.json"
                filepath = self.annotation_dir / filename
                result.save_to_file(str(filepath))
    
    def _save_statistics(self):
        """Save overall workflow statistics."""
        stats_file = self.output_dir / "annotation_workflow_statistics.json"
        
        # Calculate duration and performance metrics
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
            self.stats["duration_seconds"] = duration
            self.stats["problems_per_minute"] = self.stats["total_processed"] / (duration / 60)
        
        # Add workflow component information
        self.stats["workflow_components"] = {
            "domain_annotator": {
                "class": self.domain_annotator.__class__.__name__,
                "model": self.domain_annotator.model
            },
            "theorem_annotator": {
                "class": self.theorem_annotator.__class__.__name__,
                "model": self.theorem_annotator.model
            },
            "variable_annotator": {
                "class": self.variable_annotator.__class__.__name__,
                "model": self.variable_annotator.model
            },
            "final_answer_annotator": {
                "class": self.final_answer_annotator.__class__.__name__,
                "model": self.final_answer_annotator.model
            },
            "total_steps": 4
        }
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current status of the annotation workflow."""
        return {
            "output_directory": str(self.output_dir),
            "annotation_directory": str(self.annotation_dir),
            "processing_mode": "sequential",
            "statistics": self.stats.copy(),
            "workflow_components": {
                "domain_annotator": self.domain_annotator.__class__.__name__,
                "theorem_annotator": self.theorem_annotator.__class__.__name__,
                "variable_annotator": self.variable_annotator.__class__.__name__,
                "final_answer_annotator": self.final_answer_annotator.__class__.__name__
            },
            "is_workflow_valid": self._validate_workflow()
        }
    
    def _validate_workflow(self) -> bool:
        """Validate that all workflow components are properly initialized."""
        try:
            # Check that all annotators exist and have the required methods
            assert hasattr(self.domain_annotator, 'annotate')
            assert hasattr(self.theorem_annotator, 'annotate')
            assert hasattr(self.variable_annotator, 'annotate')
            assert hasattr(self.final_answer_annotator, 'annotate')
            return True
        except AssertionError:
            return False
    
    def get_dataset_compatibility_info(self) -> Dict[str, Any]:
        """Get information about dataset compatibility."""
        return {
            "supported_datasets": [
                "phybench",
                "physreason", 
                "seephys",
                "ugphysics",
                "mmlu_pro"
            ],
            "dataset_interface": "PhysicalDataset (from physical_datasets.base)",  # TODO: Enable when datasets package is ready
            "required_methods": [
                "__len__",
                "__getitem__", 
                "__iter__",
                "filter",
                "select"
            ],
            "sample_interface": "PhysicalDatasetSample (from physical_datasets.base)",  # TODO: Enable when datasets package is ready
            "required_sample_methods": [
                "to_dict",
                "__getitem__",
                "__contains__"
            ],
            "usage_example": """
            # from physical_datasets import load_dataset  # TODO: Enable when datasets package is ready
            dataset = load_dataset("phybench")
            results = workflow.annotate_dataset(dataset, max_problems=100)
            """
        }
    
    def resume_from_checkpoint(self, checkpoint_dir: Union[str, Path]) -> List[FullPhysicsAnnotation]:
        """
        Resume annotation workflow from a checkpoint directory.
        
        Args:
            checkpoint_dir: Path to checkpoint directory containing previous annotation results
            
        Returns:
            List of annotation results
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        self.logger.info(f"Resuming annotation workflow from checkpoint: {checkpoint_dir}")
        
        # Load existing results from llm_annotation folder
        annotation_dir = checkpoint_dir / "llm_annotation"
        if not annotation_dir.exists():
            self.logger.warning(f"No llm_annotation folder found in checkpoint: {checkpoint_dir}")
            return []
        
        existing_results = []
        for json_file in annotation_dir.glob("*.json"):
            try:
                result = FullPhysicsAnnotation.load_from_file(str(json_file))
                existing_results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to load result from {json_file}: {e}")
        
        self.logger.info(f"Loaded {len(existing_results)} existing results from checkpoint")
        return existing_results
