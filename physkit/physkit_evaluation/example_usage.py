"""
Example usage of the Physical Reasoning Evaluation Toolkit.

This script demonstrates how to use the AccuracyMetric with different
types of answers (symbolic, numerical, and textual).
"""

from physkit.definitions.answer_types import (
    SymbolicAnswer, NumericalAnswer, TextualAnswer
)
from .metrics import AccuracyMetric


def main():
    """Demonstrate the evaluation toolkit usage."""
    
    # Create some example predictions and ground truths
    predictions = [
        SymbolicAnswer("x^2 + 2x + 1"),
        NumericalAnswer(3.14, units="m/s"),
        TextualAnswer("The object accelerates due to gravity"),
        SymbolicAnswer("2x + 3"),
        NumericalAnswer(9.81, units="m/s^2")
    ]
    
    ground_truths = [
        SymbolicAnswer("(x + 1)^2"),  # Equivalent to x^2 + 2x + 1
        NumericalAnswer(3.14, units="m/s"),  # Exact match
        TextualAnswer("Gravity causes the object to accelerate"),  # Similar meaning
        SymbolicAnswer("2x + 3"),  # Exact match
        NumericalAnswer(9.8, units="m/s^2")  # Close but not exact
    ]
    
    # Initialize the accuracy metric
    accuracy_metric = AccuracyMetric()
    
    print("Physical Reasoning Evaluation Toolkit Example")
    print("=" * 50)
    
    # Compute overall accuracy
    result = accuracy_metric.compute(predictions, ground_truths, return_details=True)
    
    print(f"Overall Accuracy: {result['accuracy']:.2%}")
    print(f"Correct Answers: {result['correct_samples']}/{result['total_samples']}")
    print()
    
    # Show detailed results
    print("Detailed Results:")
    for detail in result['details']:
        print(f"Sample {detail['sample_index']}:")
        print(f"  Prediction: {detail['prediction']}")
        print(f"  Ground Truth: {detail['ground_truth']}")
        print(f"  Correct: {'✓' if detail['is_correct'] else '✗'}")

        print()
    
    # Compute accuracy by answer type
    type_results = accuracy_metric.get_accuracy_by_type(predictions, ground_truths)
    
    print("Accuracy by Answer Type:")
    for answer_type, type_result in type_results.items():
        print(f"  {answer_type.capitalize()}: {type_result['accuracy']:.2%}")
    
    # Test single comparison
    print("\nSingle Comparison Example:")
    single_result = accuracy_metric.compute_single(predictions[0], ground_truths[0])
    print(f"Symbolic comparison: {single_result['is_correct']}")



if __name__ == "__main__":
    main()
