"""
Accuracy metric for evaluating physical reasoning answers.

This module provides the AccuracyMetric class which compares predicted
answers against ground truth answers using appropriate comparison strategies.
"""

from typing import List, Dict, Any, Union, Optional
from physkit_core.definitions.answer_types import AnswerType
from physkit_core.models.answer import Answer
from ..comparison import SmartAnswerComparator
from .base import BaseMetric


class AccuracyMetric(BaseMetric):
    """
    Accuracy metric for evaluating answer correctness.
    
    This metric compares predicted answers against ground truth answers
    and calculates accuracy based on the comparison results.
    """
    
    def __init__(self, comparator: Optional[SmartAnswerComparator] = None):
        """
        Initialize the accuracy metric.
        
        Args:
            comparator: AnswerComparator instance to use for comparisons.
                       If None, creates a new instance.
        """
        super().__init__(
            "Accuracy",
            "Measures the proportion of correct predictions"
        )
        self.comparator = comparator or SmartAnswerComparator()
    
    def compute(
        self,
        predictions: List[Answer], 
        ground_truths: List[Answer],
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute accuracy for a list of predictions and ground truths.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            return_details: Whether to return detailed comparison results
            
        Returns:
            Dictionary containing accuracy metrics and optional details
        """
        # Validate inputs using base class method
        self.validate_inputs(predictions, ground_truths)
        
        correct_count = 0
        total_count = len(predictions)
        comparison_details = []
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Compare the answers
            comparison_result = self.comparator.compare(pred, gt)
            
            # Check if the answer is correct
            is_correct = comparison_result.get("is_equal", False)
            if is_correct:
                correct_count += 1
            
            # Store comparison details if requested
            if return_details:
                detail = {
                    "sample_index": i,
                    "prediction": str(pred),
                    "ground_truth": str(gt),
                    "is_correct": is_correct,

                    "comparison_details": comparison_result.get("details", {})
                }
                comparison_details.append(detail)
        
        # Calculate accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        result = {
            "accuracy": accuracy,
            "total_samples": total_count,
            "correct_samples": correct_count,
            "incorrect_samples": total_count - correct_count
        }
        
        if return_details:
            result["details"] = comparison_details
        
        return result
    
    def compute_single(self, prediction: Answer, ground_truth: Answer) -> Dict[str, Any]:
        """
        Compute accuracy for a single prediction-ground truth pair.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary containing comparison results
        """
        comparison_result = self.comparator.compare(prediction, ground_truth)
        
        return {
            "is_correct": comparison_result.get("is_equal", False),

            "comparison_details": comparison_result.get("details", {}),
            "prediction": str(prediction),
            "ground_truth": str(ground_truth)
        }
    
    def get_accuracy_by_type(self, 
                            predictions: List[Answer], 
                            ground_truths: List[Answer]) -> Dict[str, Dict[str, Any]]:
        """
        Compute accuracy broken down by answer type.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            
        Returns:
            Dictionary with accuracy metrics for each answer type
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length")
        
        # Group by answer type
        type_groups = {}
        for pred, gt in zip(predictions, ground_truths):
            answer_type = pred.answer_type.value
            
            if answer_type not in type_groups:
                type_groups[answer_type] = {
                    "predictions": [],
                    "ground_truths": []
                }
            
            type_groups[answer_type]["predictions"].append(pred)
            type_groups[answer_type]["ground_truths"].append(gt)
        
        # Compute accuracy for each type
        results = {}
        for answer_type, group in type_groups.items():
            if group["predictions"]:  # Only compute if we have samples
                results[answer_type] = self.compute(
                    group["predictions"], 
                    group["ground_truths"]
                )
        
        return results
