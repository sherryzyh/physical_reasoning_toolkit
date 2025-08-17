"""
Human-in-the-loop workflow for physics problem annotation with human review after each step.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator
from datetime import datetime
import logging
from collections import Counter


from ..assessment.domain_assessor import DomainAssessor
from ..revision.domain_revisor import DomainRevisor

from ..annotators.domain import DomainAnnotator, DomainAnnotation
from ..full_physics_annotation import FullPhysicsAnnotation
from physkit_datasets.base import PhysicalDataset


class SupervisedAnnotationWorkflow:
    """
    Supervised annotation workflow that involves human review after each step.
    Currently implements only the Domain Annotation step with human review.
    
    After processing, you can access the revised domain annotations using:
    
    # Get revised domain annotation for a specific problem
    revised_anno = workflow.get_revised_domain_annotation("problem_001")
    
    # Get all revised domain annotations
    all_revised = workflow.get_all_revised_domain_annotations()

    # Get statistics about revised domain annotations
    stats = workflow.get_revised_domain_statistics()
    
    # Access the revised domain annotation properties
    if revised_anno:
        print(f"Domain: {revised_anno.primary_domain.value}")
        print(f"Confidence: {revised_anno.confidence}")
        print(f"Reasoning: {revised_anno.reasoning}")
        print(f"Subdomains: {revised_anno.subdomains}")
    """
    
    def __init__(self, output_dir: Union[str, Path], model: str = "gpt-4o"):
        """
        Initialize the supervised annotation workflow.
        
        Args:
            output_dir: Directory to save workflow results
            model: LLM model to use (default: gpt-4o)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure for the workflow
        self._create_directory_structure()
        
        # Initialize the classes for the domain annotation
        self.domain_annotator = DomainAnnotator(model=model)
        self.domain_assessor = DomainAssessor(model=model)
        self.domain_revisor = DomainRevisor()
        self.model = model
        
        # Setup logging
        self._setup_logging()
        
        # Statistics with enhanced tracking
        self.stats = {
            "problems_successful": 0,
            "start_time": None,
            "end_time": None,
            "current_problem": 0,
            "current_step": "not_started",
            "steps_completed": [],
            "processing_times": [],
            "summary_of_domain": {
                "assessment_results": {
                    "True": 0,
                    "SemanticallyTrue": 0,
                    "False": 0,
                    "HumanApproved": 0,
                    "HumanCorrected": 0
                },
                "revision_types": {
                    "no_change": 0,
                    "golden_annotation": 0,
                    "human_answer": 0
                },
                "human_review_required": 0,
                "processing_times": []
            }
        }
    
    def _create_directory_structure(self):
        """Create the necessary directory structure for the workflow."""
        # Domain annotation step metadata directory (main output)
        self.domain_annotation_step_dir = self.output_dir / "domain_annotation_step"
        self.domain_annotation_step_dir.mkdir(exist_ok=True)
    
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
        """
        Main entry point to run the complete supervised annotation workflow.
        
        Args:
            dataset: PhysicalDataset instance to annotate
            max_problems: Maximum number of problems to process (None for all)
            domain_filter: Filter problems by specific domain (None for all domains)
            start_from: Index to start processing from (for resuming)
            auto_save: Whether to automatically save results
            
        Returns:
            Dictionary containing workflow results and summary
        """
        self.logger.info("🚀 Starting Supervised Annotation Workflow")
        self.logger.info(f"Dataset: {dataset.__class__.__name__}")
        self.logger.info(f"Dataset size: {len(dataset)} problems")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Starting from problem: {start_from}")
        
        # Apply domain filter if specified
        if domain_filter:
            dataset = dataset.filter(lambda x: domain_filter.lower() in x.get('domain', '').lower())
            self.logger.debug(f"Filtered to {len(dataset)} problems in domain: {domain_filter}")
        
        # Limit number of problems if specified
        if max_problems and len(dataset) > max_problems:
            dataset = dataset.select(list(range(max_problems)))
            self.logger.debug(f"Limited to {max_problems} problems")
        
        # Initialize workflow statistics
        self.stats["start_time"] = datetime.now()
        self.stats["current_step"] = "domain_annotation"
        self.stats["problems_successful"] = 0
        
        # Process problems through the workflow steps
        results = []
        total_problems = len(dataset)
        
        for i in range(start_from, total_problems):
            problem = dataset[i]
            problem_num = i + 1
            
            self.logger.debug(f"🔄 Processing problem {problem_num}/{total_problems}")
            self.stats["current_problem"] = problem_num
            
            try:
                # Convert problem to dictionary format
                problem_data = problem.to_dict()
                problem_id = problem_data.get("problem_id", f"problem_{i}")
                        
            except Exception as e:
                self.logger.error(f"❌ Failed to load problem {problem_num}: {e}")
                results.append({
                    "problem_id": problem_data.get("problem_id", f"problem_{i}"),
                    "status": "failed",
                })
                continue
                    
            # Process through domain annotation step
            self.logger.debug(f"  📍 Processing domain annotation for {problem_id}")
            try:
                revised_domain_anno = self.process_domain_annotation_step(
                    problem_data, problem_id, problem_start_time=time.time()
                )
                
                # After revision, all problems should have a final domain annotation
                # The revision step handles failures and provides fallbacks
                if revised_domain_anno:
                    # Add to results
                    results.append({
                        "problem_id": problem_id,
                        "status": "completed",
                    })
                    
                    self.logger.debug(f"  ✅ Successfully processed {problem_id}")
                else:
                    # This should not happen in normal operation
                    # The revision step should always provide a final annotation
                    self.logger.error(f"  ❌ Unexpected: No domain annotation after revision for {problem_id}")
                    results.append({
                        "problem_id": problem_id,
                        "status": "failed",
                    })
                    
            except Exception as e:
                self.logger.error(f"  ❌ Critical error in domain annotation step for {problem_id}: {e}")
                results.append({
                    "problem_id": problem_id,
                    "status": "failed",
                })

        
            self.stats["problems_successful"] += 1
            
            # Update progress bar if callback provided
            if progress_callback:
                progress_callback(1)
            
            # Auto-save statistics periodically
            if auto_save and (i + 1) % 10 == 0:
                self._save_statistics()
                self.logger.debug(f"💾 Auto-saved statistics after {i + 1} problems")
        
        # Complete workflow
        self.stats["end_time"] = datetime.now()
        self.stats["current_step"] = "completed"
        self.stats["steps_completed"] = ["domain_annotation"]
        
        # Save final statistics
        self._save_statistics()
        
        # Prepare workflow summary
        workflow_summary = {
            "workflow_status": "completed",
            "total_problems": total_problems,
            "problems_processed": len(results),
            "problems_successful": len([r for r in results if r["status"] == "completed"]),
            "problems_failed": len([r for r in results if r["status"] == "failed"]),
            "success_rate": len([r for r in results if r["status"] == "completed"]) / len(results) if results else 0,
            "steps_completed": self.stats["steps_completed"],
            "start_time": self.stats["start_time"].isoformat() if self.stats["start_time"] else None,
            "end_time": self.stats["end_time"].isoformat() if self.stats["end_time"] else None,
            "results": results
        }
        
        self.logger.info("🎉 Workflow completed successfully!")
        self.logger.info(f"Total processed: {self.stats['problems_successful']}")
        self.logger.info(f"Success rate: {workflow_summary['success_rate']:.1%}")
        
        return workflow_summary
    
    def process_domain_annotation_step(
        self,
        problem_data: Dict[str, Any],
        problem_id: str,
        problem_start_time: float
    ) -> Optional[DomainAnnotation]:
        """
        Process a single problem through the Domain Annotation step with human review.
        
        Args:
            problem_data: Problem data dictionary
            problem_id: Problem identifier
            problem_start_time: Start time for processing this problem
            
        Returns:
            Revised DomainAnnotation instance if successful, None otherwise
        """
        try:
            # Step 1: Automated Domain Annotation
            self.logger.debug("    Step 1.1: Automated domain classification")
            question = problem_data.get("question", problem_data.get("content", ""))
            
            # Get automated domain annotation
            domain_anno = self.domain_annotator.annotate(question)
            
            # Get golden truth domain
            domain_gt = problem_data.get("domain", "")
            
            # Step 2: Assessment of the domain annotation
            self.logger.debug("    Step 1.2: Assessment of the domain annotation")
            if not domain_gt:
                self.logger.debug("      Goes to human review as no golden truth domain is provided")
                assessment_result = self.domain_assessor.human_review_with_context(
                    llm_annotation=domain_anno,
                    context=question
                )
            else:
                self.logger.debug("      Goes to auto assessment as golden truth domain is provided")
                assessment_result = self.domain_assessor.assess(
                    llm_annotation=domain_anno,
                    golden_annotation=domain_gt,
                    context=question,
                )
            
            # Step 3: Revision of the domain annotation
            self.logger.debug("    Step 1.3: Revision of the domain annotation")
            revision_result = self.domain_revisor.revise(assessment_result)
            
            # Step 4: Create revised DomainAnnotation instance
            self.logger.debug("    Step 1.4: Creating revised domain annotation instance")
            revised_domain_anno = self._create_revised_domain_annotation(domain_anno, revision_result)
            
            # Step 5: Save the metadata and update statistics
            self.logger.debug("    Step 1.5: Saving metadata and updating statistics")
            self._save_domain_annotation_metadata(
                problem_id, problem_data, domain_anno, assessment_result, revision_result, revised_domain_anno
            )
            
            # Update statistics
            self._update_domain_statistics(
                assessment_result,
                revision_result,
                problem_start_time
            )
            
            return revised_domain_anno
            
        except Exception as e:
            self.logger.error(f"    ❌ Error in domain annotation step for {problem_id}: {e}")
            return None
    
    def _create_revised_domain_annotation(self, original_annotation: DomainAnnotation, revision_result) -> DomainAnnotation:
        """
        Create a revised DomainAnnotation instance based on the revision result.
        
        Args:
            original_annotation: The original DomainAnnotation from the LLM
            revision_result: The RevisionResult from the domain revisor
            
        Returns:
            A new DomainAnnotation instance with the revised domain
        """
        from physkit.models import PhysicsDomain
        
        # Get the revised domain string
        revised_domain_str = revision_result.revised_annotation
        
        # Map the string to PhysicsDomain enum
        try:
            # Try to find the matching enum value
            revised_domain = None
            for domain_enum in PhysicsDomain:
                if domain_enum.value == revised_domain_str:
                    revised_domain = domain_enum
                    break
            
            # If no exact match found, try to find a close match
            if revised_domain is None:
                for domain_enum in PhysicsDomain:
                    if revised_domain_str.lower() in domain_enum.value.lower() or domain_enum.value.lower() in revised_domain_str.lower():
                        revised_domain = domain_enum
                        break
            
            # If still no match, default to OTHER
            if revised_domain is None:
                revised_domain = PhysicsDomain.OTHER
                
        except Exception as e:
            self.logger.warning(f"Could not map revised domain '{revised_domain_str}' to enum, using OTHER: {e}")
            revised_domain = PhysicsDomain.OTHER
        
        # Create new DomainAnnotation with revised domain
        revised_annotation = DomainAnnotation(
            primary_domain=revised_domain,
            confidence=original_annotation.confidence,  # Keep original confidence
            reasoning=f"Revised from '{original_annotation.primary_domain.value}' to '{revised_domain.value}' based on {revision_result.revision_type}",
            subdomains=original_annotation.subdomains  # Keep original subdomains for now
        )
        
        return revised_annotation
    
    def _update_domain_statistics(
        self,
        assessment_result,
        revision_result,
        problem_start_time
    ):
        """Update domain annotation statistics."""
        # Log what we're actually getting for debugging
        self.logger.debug(f"    📊 Assessment result: {assessment_result.assessment_result}")
        self.logger.debug(f"    📊 Revision type: {revision_result.revision_type}")
        
        # Update assessment result counts
        assessment_result_type = assessment_result.assessment_result
        if assessment_result_type in self.stats["summary_of_domain"]["assessment_results"]:
            self.stats["summary_of_domain"]["assessment_results"][assessment_result_type] += 1
        else:
            self.logger.warning(f"    ⚠️  Unknown assessment result type: {assessment_result_type}")
        
        # Update revision type counts
        revision_type = revision_result.revision_type
        if revision_type in self.stats["summary_of_domain"]["revision_types"]:
            self.stats["summary_of_domain"]["revision_types"][revision_type] += 1
        else:
            self.logger.warning(f"    ⚠️  Unknown revision type: {revision_type}")
        
        # Update human review requirement
        if assessment_result.assessment_result in ["HumanApproved", "HumanCorrected"]:
            self.stats["summary_of_domain"]["human_review_required"] += 1
        
        # Update processing time
        problem_end_time = time.time()
        processing_time = problem_end_time - problem_start_time

        
        self.stats["processing_times"].append(processing_time)
    

    
    def _save_domain_annotation_metadata(
        self, 
        problem_id: str, 
        problem_data: Dict[str, Any], 
        domain_anno: DomainAnnotation, 
        assessment_result, 
        revision_result,
        revised_domain_anno: DomainAnnotation
    ):
        """Save comprehensive metadata for the domain annotation step in one consolidated file."""
        # Create one comprehensive file per problem, named directly as problem_id.json
        comprehensive_file = self.domain_annotation_step_dir / f"{problem_id}.json"
        
        # Organize data logically
        comprehensive_data = {
            # Problem identification and metadata
            "annotation_metadata": {
                "workflow_type": "supervised_annotation",
                "step": "domain_annotation",
                "model": self.model,
                "status": "completed"
            },
            
            # Original problem data
            "problem_data": problem_data,
            
            # LLM annotation results
            "raw_llm_annotation": domain_anno.to_dict(),
            
            # Quality assessment results
            "assessment": {
                "is_correct": assessment_result.is_correct,
                "assessment_result": assessment_result.assessment_result,
                "explanation": assessment_result.explanation,
                "suggested_correction": assessment_result.suggested_correction,
                "assessment_metadata": assessment_result.metadata
            },
            
            # Revision decisions and final result
            "revision": {
                "original_annotation": revision_result.original_annotation,
                "revised_annotation": revision_result.revised_annotation,
                "revision_type": revision_result.revision_type,
                "explanation": revision_result.explanation,
                "revision_metadata": revision_result.metadata
            },
            
            # Revised domain annotation instance
            "revised_domain_annotation": {
                "primary_domain": revised_domain_anno.primary_domain.value,
                "confidence": revised_domain_anno.confidence,
                "reasoning": revised_domain_anno.reasoning,
                "subdomains": revised_domain_anno.subdomains,
                "raw_annotation": revised_domain_anno.to_dict()
            },
            
            # Final domain classification
            "final_result": {
                "final_domain": revised_domain_anno.primary_domain.value,
                "confidence": revised_domain_anno.confidence,
                "requires_human_review": revision_result.revision_type == "human_answer"
            }
        }
        
        # Save the comprehensive file
        with open(comprehensive_file, 'w') as f:
            json.dump(comprehensive_data, f, indent=2, default=str)
    
    def _save_statistics(self):
        """Save overall workflow statistics with enhanced domain summary."""
        stats_file = self.output_dir / "supervised_anno_statistics.json"
        
        # Calculate duration and performance metrics
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
            self.stats["duration_seconds"] = duration
            self.stats["problems_per_minute"] = self.stats["problems_successful"] / (duration / 60)
            
        # Calculate additional domain statistics
        domain_stats = self.stats["summary_of_domain"]
            
        # Calculate success rates
        total_assessed = sum(domain_stats["assessment_results"].values())
        if total_assessed > 0:
            domain_stats["success_rate"] = {
                "exact_match": domain_stats["assessment_results"]["True"] / total_assessed,
                "semantic_match": domain_stats["assessment_results"]["SemanticallyTrue"] / total_assessed,
                "human_approved": domain_stats["assessment_results"]["HumanApproved"] / total_assessed,
                "human_corrected": domain_stats["assessment_results"]["HumanCorrected"] / total_assessed,
                "incorrect": domain_stats["assessment_results"]["False"] / total_assessed,
                "overall_correct": (domain_stats["assessment_results"]["True"] + 
                                  domain_stats["assessment_results"]["SemanticallyTrue"] + 
                                  domain_stats["assessment_results"]["HumanApproved"]) / total_assessed
            }
        
        # Add annotator information
        self.stats["domain_annotator"] = {
            "class": self.domain_annotator.__class__.__name__,
            "model": self.domain_annotator.model
        }
        
        # Add workflow efficiency metrics
        self.stats["efficiency_metrics"] = {
            "problems_per_minute": self.stats.get("problems_per_minute", 0),
            "human_review_rate": domain_stats["human_review_required"] / max(total_assessed, 1),
            "auto_correction_rate": domain_stats["revision_types"]["golden_annotation"] / max(total_assessed, 1),
            "no_change_rate": domain_stats["revision_types"]["no_change"] / max(total_assessed, 1)
        }
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
    
    def print_current_statistics(self):
        """Print current workflow statistics in a readable format."""
        print("\n" + "="*60)
        print("📊 SUPERVISED ANNOTATION WORKFLOW - CURRENT STATISTICS")
        print("="*60)
        
        # Basic workflow info
        print(f"🔄 Current Step: {self.stats['current_step']}")
        print(f"📈 Total Processed: {self.stats['problems_successful']}")
        
        if self.stats["start_time"]:
            print(f"⏰ Start Time: {self.stats['start_time']}")
        if self.stats["end_time"]:
            print(f"⏰ End Time: {self.stats['end_time']}")
        
        # Domain annotation statistics
        domain_stats = self.stats["summary_of_domain"]
        print(f"\n🔍 DOMAIN ANNOTATION STATISTICS:")
        print(f"   Assessment Results:")
        for result_type, count in domain_stats["assessment_results"].items():
            print(f"     - {result_type}: {count}")
        
        print(f"   Revision Types:")
        for revision_type, count in domain_stats["revision_types"].items():
            print(f"     - {revision_type}: {count}")
        
        print(f"   Human Review Required: {domain_stats['human_review_required']}")
        
        # Calculate success rates if we have data
        total_assessed = sum(domain_stats["assessment_results"].values())
        if total_assessed > 0:
            print(f"\n📊 SUCCESS RATES:")
            exact_rate = domain_stats["assessment_results"]["True"] / total_assessed
            semantic_rate = domain_stats["assessment_results"]["SemanticallyTrue"] / total_assessed
            human_approved_rate = domain_stats["assessment_results"]["HumanApproved"] / total_assessed
            human_corrected_rate = domain_stats["assessment_results"]["HumanCorrected"] / total_assessed
            incorrect_rate = domain_stats["assessment_results"]["False"] / total_assessed
            
            print(f"   Exact Match: {exact_rate:.1%}")
            print(f"   Semantic Match: {semantic_rate:.1%}")
            print(f"   Human Approved: {human_approved_rate:.1%}")
            print(f"   Human Corrected: {human_corrected_rate:.1%}")
            print(f"   Incorrect: {incorrect_rate:.1%}")
            print(f"   Overall Correct: {(exact_rate + semantic_rate + human_approved_rate):.1%}")
        
        print("="*60)
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current status of the supervised annotation workflow."""
        return {
            "output_directory": str(self.output_dir),
            "domain_annotation_step_directory": str(self.domain_annotation_step_dir),
            "model": self.model,
            "statistics": self.stats.copy(),
            "current_step": self.stats["current_step"],
            "workflow_type": "supervised_annotation"
        }
    
    def get_domain_annotation_metadata(self, problem_id: str = None) -> Dict[str, Any]:
        """Get metadata for domain annotation step."""
        if problem_id:
            # Get comprehensive metadata for specific problem
            comprehensive_file = self.domain_annotation_step_dir / f"{problem_id}.json"
            if comprehensive_file.exists():
                with open(comprehensive_file, 'r') as f:
                    return json.load(f)
            return None
        else:
            # Get summary of all metadata
            comprehensive_files = list(self.domain_annotation_step_dir.glob("*.json"))
            
            summary = {
                "total_problems": len(comprehensive_files),
                "comprehensive_files": [f.name for f in comprehensive_files],
                "directory": str(self.domain_annotation_step_dir)
            }
            return summary
    
    def get_revised_domain_annotation(self, problem_id: str) -> Optional[DomainAnnotation]:
        """
        Get the revised domain annotation for a specific problem.
        
        Args:
            problem_id: The ID of the problem
            
        Returns:
            DomainAnnotation instance if found, None otherwise
        """
        try:
            # Load the comprehensive metadata for the problem
            comprehensive_file = self.domain_annotation_step_dir / f"{problem_id}.json"
            if not comprehensive_file.exists():
                self.logger.warning(f"No metadata found for problem {problem_id}")
                return None
            
            with open(comprehensive_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract the revised domain annotation data
            revised_data = metadata.get("revised_domain_annotation", {})
            if not revised_data:
                self.logger.warning(f"No revised domain annotation found for problem {problem_id}")
                return None
            
            # Create DomainAnnotation instance from the data
            from physkit.models import PhysicsDomain
            
            # Map the domain string to enum
            domain_enum = None
            for domain in PhysicsDomain:
                if domain.value == revised_data["primary_domain"]:
                    domain_enum = domain
                    break
            
            if domain_enum is None:
                self.logger.warning(f"Could not map domain '{revised_data['primary_domain']}' to enum for problem {problem_id}")
                domain_enum = PhysicsDomain.OTHER
            
            # Create and return the DomainAnnotation instance
            revised_annotation = DomainAnnotation(
                primary_domain=domain_enum,
                confidence=revised_data.get("confidence", 1.0),
                reasoning=revised_data.get("reasoning", ""),
                subdomains=revised_data.get("subdomains", [])
            )
            
            return revised_annotation
            
        except Exception as e:
            self.logger.error(f"Error getting revised domain annotation for problem {problem_id}: {e}")
            return None
    
    def get_all_revised_domain_annotations(self) -> Dict[str, DomainAnnotation]:
        """
        Get all revised domain annotations for all processed problems.
        
        Returns:
            Dictionary mapping problem_id to DomainAnnotation instances
        """
        revised_annotations = {}
        
        try:
            # Get all comprehensive files
            comprehensive_files = list(self.domain_annotation_step_dir.glob("*.json"))
            
            for file_path in comprehensive_files:
                problem_id = file_path.stem  # Remove .json extension
                
                try:
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract the revised domain annotation data
                    revised_data = metadata.get("revised_domain_annotation", {})
                    if not revised_data:
                        continue
                    
                    # Create DomainAnnotation instance from the data
                    from physkit.models import PhysicsDomain
                    
                    # Map the domain string to enum
                    domain_enum = None
                    for domain in PhysicsDomain:
                        if domain.value == revised_data["primary_domain"]:
                            domain_enum = domain
                            break
                    
                    if domain_enum is None:
                        self.logger.warning(f"Could not map domain '{revised_data['primary_domain']}' to enum for problem {problem_id}")
                        domain_enum = PhysicsDomain.OTHER
                    
                    # Create the DomainAnnotation instance
                    revised_annotation = DomainAnnotation(
                        primary_domain=domain_enum,
                        confidence=revised_data.get("confidence", 1.0),
                        reasoning=revised_data.get("reasoning", ""),
                        subdomains=revised_data.get("subdomains", [])
                    )
                    
                    revised_annotations[problem_id] = revised_annotation
                    
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path} for problem {problem_id}: {e}")
                    continue
            
            self.logger.debug(f"Successfully loaded {len(revised_annotations)} revised domain annotations")
            return revised_annotations
            
        except Exception as e:
            self.logger.error(f"Error getting all revised domain annotations: {e}")
            return {}
    
    def get_revised_domain_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the revised domain annotations.
        
        Returns:
            Dictionary containing statistics about revised domain annotations
        """
        all_annotations = self.get_all_revised_domain_annotations()
        
        if not all_annotations:
            return {"error": "No revised domain annotations found"}
        
        # Count domains
        domain_counts = Counter()
        confidence_values = []
        reasoning_lengths = []
        subdomain_counts = []
        
        for problem_id, annotation in all_annotations.items():
            domain_counts[annotation.primary_domain.value] += 1
            confidence_values.append(annotation.confidence)
            reasoning_lengths.append(len(annotation.reasoning))
            subdomain_counts.append(len(annotation.subdomains))
        
        # Calculate statistics
        stats = {
            "total_problems": len(all_annotations),
            "domain_distribution": dict(domain_counts),
            "confidence_statistics": {
                "mean": sum(confidence_values) / len(confidence_values),
                "min": min(confidence_values),
                "max": max(confidence_values),
                "std_dev": self._calculate_std_dev(confidence_values)
            },
            "reasoning_statistics": {
                "mean_length": sum(reasoning_lengths) / len(reasoning_lengths),
                "min_length": min(reasoning_lengths),
                "max_length": max(reasoning_lengths)
            },
            "subdomain_statistics": {
                "mean_count": sum(subdomain_counts) / len(subdomain_counts),
                "problems_with_subdomains": sum(1 for count in subdomain_counts if count > 0),
                "problems_without_subdomains": sum(1 for count in subdomain_counts if count == 0)
            }
        }
        
        return stats
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def get_dataset_compatibility_info(self) -> Dict[str, Any]:
        """Get information about dataset compatibility."""
        return {
            "supported_datasets": [
                "phybench",
                "physreason", 
                "seephys",
                "ugphysics",
                "mmlu_pro",
                "phyx"
            ],
            "dataset_interface": "PhysicalDataset (from physkit_datasets.base)",
            "required_methods": [
                "__len__",
                "__getitem__", 
                "__iter__",
                "filter",
                "select"
            ],
            "sample_interface": "PhysicalDatasetSample (from physkit_datasets.base)",
            "required_sample_methods": [
                "to_dict",
                "__getitem__",
                "__contains__"
            ],
            "usage_example": """
            from physkit_datasets import load_dataset
            from physical_annotation.workflows.supervised_annotation_workflow import SupervisedAnnotationWorkflow
            
            # Initialize workflow
            workflow = SupervisedAnnotationWorkflow("output_directory")
            
            # Run the complete workflow
            results = workflow.run(dataset, max_problems=10)
            
            # Access revised domain annotations
            revised_anno = workflow.get_revised_domain_annotation("problem_001")
            """
        }
    
    def resume_from_problem(
        self,
        problem_index: int,
        dataset: PhysicalDataset,
        auto_save: bool = True
    ) -> Dict[str, Any]:
        """
        Resume workflow from a specific problem index.
        
        Args:
            problem_index: Index of the problem to start from
            dataset: The dataset to process
            
        Returns:
            Workflow results dictionary
        """
        self.logger.info(f"Resuming workflow from problem index: {problem_index}")
        
        # Continue processing from the specified index
        return self.run(dataset, start_from=problem_index, auto_save=auto_save)
