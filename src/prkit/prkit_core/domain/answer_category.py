"""
Answer category definitions for physical reasoning evaluation.

This module defines the different categories of answers that can be compared.
AnswerCategory provides granular semantics (number, equation, physical_quantity,
formula, text, option) that cover both content-based and format-based classification.
"""

from enum import Enum


class AnswerCategory(Enum):
    """
    Enumeration of answer categories for normalization and comparison.

    Covers both content semantics (from normalization) and format (option):
    - number: Dimensionless numeric value (e.g., 42, 3.14)
    - equation: Single-equation form (e.g., F = ma)
    - physical_quantity: Number with units (e.g., 9.8 m/s^2)
    - formula: Mathematical expression (e.g., x^2 + 1)
    - text: Text-based descriptive answer
    - option: Multiple choice selection (e.g., A, B, 1, 2)
    """

    NUMBER = "number"
    EQUATION = "equation"
    PHYSICAL_QUANTITY = "physical_quantity"
    FORMULA = "formula"
    TEXT = "text"
    OPTION = "option"
