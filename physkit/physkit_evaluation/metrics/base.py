"""
Base classes for evaluation metrics.

This module provides the foundation for all evaluation metrics
in the physical reasoning toolkit.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from physkit_core.definitions.answer_types import Answer


class BaseMetric(ABC):
    """
    Base class for all evaluation metrics.
    
    This class defines the interface that all metrics must implement
    and provides common functionality for metric evaluation.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the base metric.
        
        Args:
            name: Name of the metric
            description: Description of what the metric measures
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def compute(self, 
                predictions: List[Answer], 
                ground_truths: List[Answer],
                **kwargs) -> Dict[str, Any]:
        """
        Compute the metric value.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            **kwargs: Additional arguments specific to the metric
            
        Returns:
            Dictionary containing metric results
        """
        pass
    
    def validate_inputs(self, 
                       predictions: List[Answer], 
                       ground_truths: List[Answer]) -> None:
        """
        Validate input data for the metric.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(predictions, list) or not isinstance(ground_truths, list):
            raise ValueError("Predictions and ground truths must be lists")
        
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length")
        
        if not predictions:
            raise ValueError("Cannot compute metric on empty lists")
        
        # Validate that all items are Answer instances
        for i, pred in enumerate(predictions):
            if not isinstance(pred, Answer):
                raise ValueError(f"Prediction at index {i} is not an Answer instance")
        
        for i, gt in enumerate(ground_truths):
            if not isinstance(gt, Answer):
                raise ValueError(f"Ground truth at index {i} is not an Answer instance")
    
    def get_metric_info(self) -> Dict[str, str]:
        """
        Get information about the metric.
        
        Returns:
            Dictionary containing metric metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "class": self.__class__.__name__
        }
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}')"
