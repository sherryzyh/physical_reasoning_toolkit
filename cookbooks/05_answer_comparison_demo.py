#!/usr/bin/env python3
"""
Cookbook 5: Enhanced Evaluation Toolkit Demo

This cookbook demonstrates comprehensive answer comparison capabilities:
1. Symbolic expressions with different formats (equations vs expressions)
2. Numerical values with various units and significant figures
3. Textual descriptions with semantic comparison
4. Edge cases and special scenarios
"""

import os
import sys

from prkit_evaluation.metrics import AccuracyMetric
from prkit_core.definitions.answer_types import AnswerType
from prkit_core.models.answer import Answer


def main():
    """Run the enhanced evaluation demo."""
    print("🚀 Enhanced PhysKit Evaluation Toolkit Demo")
    print("=" * 60)
    print("This cookbook demonstrates comprehensive comparison scenarios:")
    print("• Symbolic: Equations vs expressions, complex LaTeX")
    print("• Numerical: Unit conversions, significant figures, special cases")
    print("• Textual: Semantic similarity using LLM")
    print()
    
    # Create comprehensive test cases
    predictions = [
        # Symbolic comparisons - different scenarios
        Answer(
            value="v(t) = \\frac{k}{c-b}e^{-b t} - \\frac{g}{c} + \\left(-\\frac{k}{c-b}+\\frac{g}{c}\\right)e^{-c t}",
            answer_type=AnswerType.SYMBOLIC,
            metadata={"id": "sym_001"}
        ),
        Answer(
            value="F = ma",
            answer_type=AnswerType.SYMBOLIC,
            metadata={"id": "sym_002"}
        ),
        Answer(
            value="\\frac{g}{c}\\left(\\mathrm{e}^{-c t} - 1\\right) + \\frac{k}{c-b}\\left(\\mathrm{e}^{-b t} - \\mathrm{e}^{-c t}\\right)",
            answer_type=AnswerType.SYMBOLIC,
            metadata={"id": "sym_003"}
        ),
        Answer(
            value="E = mc^2",
            answer_type=AnswerType.SYMBOLIC,
            metadata={"id": "sym_004"}
        ),
        Answer(
            value="\\int_0^t v(\\tau) d\\tau = x(t)",
            answer_type=AnswerType.SYMBOLIC,
            metadata={"id": "sym_005"}
        ),
        
        # Numerical comparisons - various scenarios
        Answer(
            value=9.81,
            unit="m/s²",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_001"}
        ),
        Answer(
            value=35.316,
            unit="km/h",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_002"}
        ),
        Answer(
            value=3.14159,
            unit="rad",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_003"}
        ),
        Answer(
            value=1000.0,
            unit="g",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_004"}
        ),
        Answer(
            value=float('inf'),
            unit="m/s",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_005"}
        ),
        Answer(
            value=float('nan'),
            unit="kg",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_006"}
        ),
        Answer(
            value=0.0,
            unit="N",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_007"}
        ),
        Answer(
            value=212.0,
            unit="°F",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_008"}
        ),
        
        # Textual comparisons - different phrasings
        Answer(
            value="The object accelerates downward due to the force of gravity acting on it.",
            answer_type=AnswerType.TEXTUAL,
            metadata={"id": "txt_001"}
        ),
        Answer(
            value="Gravity causes the object to accelerate downward.",
            answer_type=AnswerType.TEXTUAL,
            metadata={"id": "txt_002"}
        ),
        Answer(
            value="When a force is applied to an object, it accelerates according to F = ma.",
            answer_type=AnswerType.TEXTUAL,
            metadata={"id": "txt_003"}
        ),
        Answer(
            value="Newton's second law states that acceleration equals force divided by mass.",
            answer_type=AnswerType.TEXTUAL,
            metadata={"id": "txt_004"}
        ),
        Answer(
            value="The object falls faster and faster because of Earth's gravitational pull.",
            answer_type=AnswerType.TEXTUAL,
            metadata={"id": "txt_005"}
        ),
        
        # Option comparisons - multiple choice scenarios
        Answer(
            value="B",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_001"}
        ),
        Answer(
            value="A",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_002"}
        ),
        Answer(
            value="ABC",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_003"}
        ),
        Answer(
            value="cab",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_004"}
        ),
        Answer(
            value="B, A",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_005"}
        ),
        Answer(
            value="A;B",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_006"}
        ),
        Answer(
            value="",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_007"}
        ),
    ]
    
    ground_truths = [
        # Symbolic ground truths
        Answer(
            value="\\boxed{\\frac{g}{c}\\left(\\mathrm{e}^{-c t} - 1\\right) + \\frac{k}{c-b}\\left(\\mathrm{e}^{-b t} - \\mathrm{e}^{-c t}\\right)}",
            answer_type=AnswerType.SYMBOLIC,
            metadata={"id": "sym_001"}
        ),
        Answer(
            value="F = m \\cdot a",
            answer_type=AnswerType.SYMBOLIC,
            metadata={"id": "sym_002"}
        ),
        Answer(
            value="\\frac{g}{c}\\left(\\mathrm{e}^{-c t} - 1\\right) + \\frac{k}{c-b}\\left(\\mathrm{e}^{-b t} - \\mathrm{e}^{-c t}\\right)",
            answer_type=AnswerType.SYMBOLIC,
            metadata={"id": "sym_003"}
        ),
        Answer(
            value="E = m c^2",
            answer_type=AnswerType.SYMBOLIC,
            metadata={"id": "sym_004"}
        ),
        Answer(
            value="\\int_0^t v(\\tau) d\\tau = x(t)",
            answer_type=AnswerType.SYMBOLIC,
            metadata={"id": "sym_005"}
        ),
        
        # Numerical ground truths
        Answer(
            value=9.80665,
            unit="m/s²",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_001"}
        ),
        Answer(
            value=9.81,
            unit="m/s",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_002"}
        ),
        Answer(
            value=3.14159,
            unit="rad",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_003"}
        ),
        Answer(
            value=1.0,
            unit="kg",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_004"}
        ),
        Answer(
            value=float('inf'),
            unit="m/s",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_005"}
        ),
        Answer(
            value=float('nan'),
            unit="kg",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_006"}
        ),
        Answer(
            value=0.0,
            unit="N",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_007"}
        ),
        Answer(
            value=100.0,
            unit="°C",
            answer_type=AnswerType.NUMERICAL,
            metadata={"id": "num_008"}
        ),
        
        # Textual ground truths
        Answer(
            value="Gravity causes the object to accelerate downward.",
            answer_type=AnswerType.TEXTUAL,
            metadata={"id": "txt_001"}
        ),
        Answer(
            value="Gravity causes the object to accelerate downward.",
            answer_type=AnswerType.TEXTUAL,
            metadata={"id": "txt_002"}
        ),
        Answer(
            value="Newton's second law states that acceleration equals force divided by mass.",
            answer_type=AnswerType.TEXTUAL,
            metadata={"id": "txt_003"}
        ),
        Answer(
            value="Newton's second law states that acceleration equals force divided by mass.",
            answer_type=AnswerType.TEXTUAL,
            metadata={"id": "txt_004"}
        ),
        Answer(
            value="The object accelerates downward due to the force of gravity acting on it.",
            answer_type=AnswerType.TEXTUAL,
            metadata={"id": "txt_005"}
        ),
        
        # Option ground truths
        Answer(
            value="B",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_001"}
        ),
        Answer(
            value="A",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_002"}
        ),
        Answer(
            value="ABC",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_003"}
        ),
        Answer(
            value="ABC",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_004"}
        ),
        Answer(
            value="A, B",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_005"}
        ),
        Answer(
            value="A;B",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_006"}
        ),
        Answer(
            value="",
            answer_type=AnswerType.OPTION,
            metadata={"id": "opt_007"}
        ),
    ]
    
    print("📋 Enhanced Problem Set Overview")
    print("-" * 50)
    print("Symbolic Problems (5):")
    print("  • Complex velocity function (different formats)")
    print("  • Newton's second law (equation vs expression)")
    print("  • Velocity function (expression vs expression)")
    print("  • Einstein's mass-energy relation")
    print("  • Integral equation")
    print()
    print("Numerical Problems (8):")
    print("  • Gravitational acceleration (significant figures)")
    print("  • Speed conversion (km/h vs m/s)")
    print("  • Angle values (radians)")
    print("  • Mass conversion (g vs kg)")
    print("  • Infinity comparison")
    print("  • NaN comparison")
    print("  • Zero comparison")
    print("  • Temperature conversion (°F vs °C)")
    print()
    print("Textual Problems (5):")
    print("  • Gravity explanation (different phrasings)")
    print("  • Newton's law (different phrasings)")
    print("  • Falling object (different phrasings)")
    print()
    print("Option Problems (7):")
    print("  • Single choice (B vs B)")
    print("  • Single choice (A vs A)")
    print("  • Multiple choice (ABC vs ABC)")
    print("  • Case-insensitive (cab vs ABC)")
    print("  • Order-independent (B, A vs A, B)")
    print("  • Different separators (A;B vs A;B)")
    print("  • Empty answers")
    print()
    
    # Run evaluation
    metric = AccuracyMetric()
    result = metric.compute(predictions, ground_truths, return_details=True)
    
    # Display results
    print("📊 Overall Results")
    print("-" * 50)
    print(f"Overall Accuracy: {result['accuracy']:.2%}")
    print(f"Correct Answers: {result['correct_samples']}/{result['total_samples']}")
    print()
    
    # Group results by type
    symbolic_results = []
    numerical_results = []
    textual_results = []
    option_results = []
    
    for detail in result['details']:
        # Extract ID from metadata if available, otherwise use index
        pred_id = detail.get('sample_index', 0)
        pred = predictions[pred_id]
        
        if pred.answer_type == AnswerType.SYMBOLIC:
            symbolic_results.append(detail)
        elif pred.answer_type == AnswerType.NUMERICAL:
            numerical_results.append(detail)
        elif pred.answer_type == AnswerType.OPTION:
            option_results.append(detail)
        else:
            textual_results.append(detail)
    
    # Display detailed results by type
    print("🔍 Symbolic Comparison Results")
    print("-" * 50)
    for detail in symbolic_results:
        pred_id = detail.get('sample_index', 0)
        pred = predictions[pred_id]
        gt_id = detail.get('sample_index', 0)
        gt = ground_truths[gt_id]
        status = "✓" if detail['is_correct'] else "✗"
        print(f"{status} {pred.metadata['id']}: {detail['is_correct']}")
        if not detail['is_correct'] and 'comparison_details' in detail:
            comp_details = detail['comparison_details']
            if 'sympy_eq1' in comp_details and 'sympy_eq2' in comp_details:
                print(f"    Pred: {comp_details['sympy_eq1']}")
                print(f"    GT:   {comp_details['sympy_eq2']}")
                if 'zero_exp' in comp_details:
                    print(f"    Diff: {comp_details['zero_exp']}")
        print()
    
    print("🔢 Numerical Comparison Results")
    print("-" * 50)
    for detail in numerical_results:
        pred_id = detail.get('sample_index', 0)
        pred = predictions[pred_id]
        gt_id = detail.get('sample_index', 0)
        gt = ground_truths[gt_id]
        status = "✓" if detail['is_correct'] else "✗"
        print(f"{status} {pred.metadata['id']}: {detail['is_correct']}")
        if 'comparison_details' in detail:
            comp_details = detail['comparison_details']
            if 'method' in comp_details:
                print(f"    Method: {comp_details['method']}")
            if 'explanation' in comp_details:
                exp = comp_details['explanation']
                if isinstance(exp, dict) and 'method' in exp:
                    print(f"    Comparison: {exp['method']}")
                    if exp['method'] == 'significant_figures_comparison':
                        print(f"    Sig Figs: {exp.get('comparison_significant_figures', 'N/A')}")
                        print(f"    Rounded: {exp.get('rounded_value1', 'N/A')} vs {exp.get('rounded_value2', 'N/A')}")
        print()
    
    print("📝 Textual Comparison Results")
    print("-" * 50)
    for detail in textual_results:
        pred_id = detail.get('sample_index', 0)
        pred = predictions[pred_id]
        gt_id = detail.get('sample_index', 0)
        gt = ground_truths[gt_id]
        status = "✓" if detail['is_correct'] else "✗"
        print(f"{status} {pred.metadata['id']}: {detail['is_correct']}")
        if 'comparison_details' in detail:
            comp_details = detail['comparison_details']
            if 'method' in comp_details:
                print(f"    Method: {comp_details['method']}")
        print()
    
    print("🔘 Option Comparison Results")
    print("-" * 50)
    for detail in option_results:
        pred_id = detail.get('sample_index', 0)
        pred = predictions[pred_id]
        gt_id = detail.get('sample_index', 0)
        gt = ground_truths[gt_id]
        status = "✓" if detail['is_correct'] else "✗"
        print(f"{status} {pred.metadata['id']}: {detail['is_correct']}")
        if 'comparison_details' in detail:
            comp_details = detail['comparison_details']
            if 'comparison_method' in comp_details:
                print(f"    Method: {comp_details['comparison_method']}")
            if 'is_multiple_choice' in comp_details:
                print(f"    Multiple Choice: {comp_details['is_multiple_choice']}")
            if 'normalized_answer1' in comp_details and 'normalized_answer2' in comp_details:
                print(f"    Normalized: {comp_details['normalized_answer1']} vs {comp_details['normalized_answer2']}")
        print()
    
    # Summary by type
    print("📈 Accuracy Breakdown by Answer Type")
    print("-" * 50)
    
    symbolic_accuracy = sum(1 for r in symbolic_results if r['is_correct']) / len(symbolic_results) if symbolic_results else 0
    numerical_accuracy = sum(1 for r in numerical_results if r['is_correct']) / len(numerical_results) if numerical_results else 0
    textual_accuracy = sum(1 for r in textual_results if r['is_correct']) / len(textual_results) if textual_results else 0
    option_accuracy = sum(1 for r in option_results if r['is_correct']) / len(option_results) if option_results else 0
    
    print(f"Symbolic:   {symbolic_accuracy:.2%} ({sum(1 for r in symbolic_results if r['is_correct'])}/{len(symbolic_results)})")
    print(f"Numerical:  {numerical_accuracy:.2%} ({sum(1 for r in numerical_results if r['is_correct'])}/{len(numerical_results)})")
    print(f"Textual:    {textual_accuracy:.2%} ({sum(1 for r in textual_results if r['is_correct'])}/{len(textual_results)})")
    print(f"Option:     {option_accuracy:.2%} ({sum(1 for r in option_results if r['is_correct'])}/{len(option_results)})")
    print()
    
    print("✅ Enhanced Evaluation Toolkit Demo Completed Successfully!")
    print()
    print("🎯 Key Features Demonstrated:")
    print("  • Symbolic: Handles equations vs expressions, complex LaTeX parsing")
    print("  • Numerical: Unit conversions, significant figures, special cases (inf, NaN)")
    print("  • Textual: Semantic similarity using LLM comparison")
    print("  • Option: Case-insensitive, order-independent multiple choice comparison")
    print("  • Comprehensive error handling and detailed comparison results")


if __name__ == "__main__":
    main()
