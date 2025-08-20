"""
Symbolic comparator for mathematical expressions and equations.

This module provides comparison strategies for symbolic mathematical
answers using latex2sympy2_extended for robust LaTeX parsing.
"""

import re
from typing import Any, Dict, Optional
import sympy as sp
from physkit.definitions.answer_types import Answer, SymbolicAnswer
from .base import BaseComparator

from ..utils.phybench_latex_pre_process import master_convert, time_simplify

class SymbolicComparator(BaseComparator):
    """Comparator for symbolic mathematical expressions."""
    
    def _latex_to_sympy(self, latex_expr: str) -> sp.Basic:
        if latex_expr.startswith("\boxed{"):
            latex_expr=latex_expr[6:-1]
        answer_exp = master_convert(latex_expr)
        answer_exp, replacements=sp.posify(answer_exp)
        answer_exp=time_simplify(answer_exp)
        answer_exp=answer_exp.subs(replacements)
        return answer_exp
    
    def compare(
        self,
        answer1: Answer,
        answer2: Answer
    ) -> Dict[str, Any]:
        
        try:
            answer1_exp=self._latex_to_sympy(answer1.value)
            answer2_exp=self._latex_to_sympy(answer2.value)
            
            zero_exp=time_simplify(sp.expand(answer1_exp-answer2_exp))
            
            result={
                "is_equal": zero_exp==0 or answer1_exp==answer2_exp,
                "details":{
                    "method": "symbolic",
                    "parsed_eq1": answer1.value,
                    "parsed_eq2": answer2.value,
                    "sympy_eq1": answer1_exp,
                    "sympy_eq2": answer2_exp,
                    "zero_exp": zero_exp
                },
                "error": None
            }
        except Exception as e:
            return {
                "is_equal": False,
                "details":{
                    "method": "symbolic",
                    "parsed_eq1": answer1.value,
                    "parsed_eq2": answer2.value,
                    "sympy_eq1": None,
                    "sympy_eq2": None
                },
                "error": str(e)
            }
        

        return result

    # def _extract_equation_sides(self, equation: str) -> tuple[Optional[str], Optional[str]]:
    #     """
    #     Extract left and right sides of an equation.
        
    #     Args:
    #         equation: Equation string (e.g., "F = ma" or "$$ F = ma $$")
            
    #     Returns:
    #         Tuple of (left_side, right_side) or (None, None) if not an equation
    #     """
    #     # Remove surrounding delimiters
    #     equation = re.sub(r'^\$+|\$+$', '', equation.strip())
        
    #     # Split on equals sign
    #     if '=' in equation:
    #         parts = equation.split('=', 1)
    #         if len(parts) == 2:
    #             return parts[0].strip(), parts[1].strip()
        
    #     return None, None
    
    # def compare(self, answer1: Answer, answer2: Answer) -> Dict[str, Any]:
    #     """
    #     Compare two symbolic expressions for mathematical equivalence.
        
    #     This comparator handles:
    #     - Algebraic expressions
    #     - Mathematical formulas
    #     - Symbolic equations
    #     - Mixed equation/expression comparisons
    #     """
    #     if not self.can_compare(answer1, answer2):
    #         raise ValueError("Cannot compare non-symbolic answers with SymbolicComparator")
        
    #     result = {
    #         'is_equal': False,
    #         'method': 'failed',
    #         'error': None,
    #         'parsed_eq1': None,
    #         'parsed_eq2': None,
    #         'sympy_eq1': None,
    #         'sympy_eq2': None
    #     }
        
    #     try:
    #         # Get the LaTeX expressions from the answers
    #         latex1 = answer1.value
    #         latex2 = answer2.value
            
    #         # Extract equation parts
    #         left1, right1 = self._extract_equation_sides(latex1)
    #         left2, right2 = self._extract_equation_sides(latex2)
            
    #         # Handle different scenarios
    #         if left1 and right1 and left2 and right2:
    #             # Both are equations: compare (left1 - right1) with (left2 - right2)
    #             left1_expr = self._latex_to_sympy(left1)
    #             right1_expr = self._latex_to_sympy(right1)
    #             left2_expr = self._latex_to_sympy(left2)
    #             right2_expr = self._latex_to_sympy(right2)
                
    #             if all(expr is not None for expr in [left1_expr, right1_expr, left2_expr, right2_expr]):
    #                 expr1 = left1_expr - right1_expr
    #                 expr2 = left2_expr - right2_expr
                    
    #                 result['parsed_eq1'] = f"{left1} = {right1}"
    #                 result['parsed_eq2'] = f"{left2} = {right2}"
    #                 result['sympy_eq1'] = f"{left1_expr} = {right1_expr}"
    #                 result['sympy_eq2'] = f"{left2_expr} = {right2_expr}"
                    
    #                 # Check if the difference is zero
    #                 try:
    #                     diff = sp.simplify(expr1 - expr2)
    #                     result['is_equal'] = diff == 0
    #                     result['method'] = 'symbolic'
    #                 except Exception as e:
    #                     result['error'] = f"Symbolic simplification failed: {e}"
                        
    #             else:
    #                 result['error'] = "Failed to parse one or more equation parts"
                    
    #         elif (left1 and right1) or (left2 and right2):
    #             # One is an equation, one is an expression: compare the right sides
    #             if left1 and right1:
    #                 # First is equation, second is expression
    #                 right1_expr = self._latex_to_sympy(right1)
    #                 expr2 = self._latex_to_sympy(latex2)
                    
    #                 if right1_expr is not None and expr2 is not None:
    #                     result['parsed_eq1'] = f"{left1} = {right1}"
    #                     result['parsed_eq2'] = latex2
    #                     result['sympy_eq1'] = f"equation_right_side = {right1_expr}"
    #                     result['sympy_eq2'] = str(expr2)
                        
    #                     # Compare right side of equation with the expression
    #                     try:
    #                         diff = sp.simplify(right1_expr - expr2)
    #                         result['is_equal'] = diff == 0
    #                         result['method'] = 'symbolic_mixed'
    #                     except Exception as e:
    #                         result['error'] = f"Symbolic comparison failed: {e}"
    #                 else:
    #                     result['error'] = "Failed to parse one or both expressions"
                        
    #             else:
    #                 # First is expression, second is equation
    #                 expr1 = self._latex_to_sympy(latex1)
    #                 right2_expr = self._latex_to_sympy(right2)
                    
    #                 if expr1 is not None and right2_expr is not None:
    #                     result['parsed_eq1'] = latex1
    #                     result['parsed_eq2'] = f"{left2} = {right2}"
    #                     result['sympy_eq1'] = str(expr1)
    #                     result['sympy_eq2'] = f"equation_right_side = {right2_expr}"
                        
    #                     # Compare expression with right side of equation
    #                     try:
    #                         diff = sp.simplify(expr1 - right2_expr)
    #                         result['is_equal'] = diff == 0
    #                         result['method'] = 'symbolic_mixed'
    #                     except Exception as e:
    #                         result['error'] = f"Symbolic comparison failed: {e}"
    #                 else:
    #                     result['error'] = "Failed to parse one or both expressions"
                        
    #         else:
    #             # Both are expressions: compare directly
    #             expr1 = self._latex_to_sympy(latex1)
    #             expr2 = self._latex_to_sympy(latex2)
                
    #             if expr1 is not None and expr2 is not None:
    #                 result['parsed_eq1'] = latex1
    #                 result['parsed_eq2'] = latex2
    #                 result['sympy_eq1'] = str(expr1)
    #                 result['sympy_eq2'] = str(expr2)
                    
    #                 # Check if expressions are equal
    #                 try:
    #                     diff = sp.simplify(expr1 - expr2)
    #                     result['is_equal'] = diff == 0
    #                     result['method'] = 'symbolic'
    #                 except Exception as e:
    #                     result['error'] = f"Symbolic comparison failed: {e}"
    #             else:
    #                 result['error'] = "Failed to parse one or both expressions"
                    
    #     except Exception as e:
    #         result['error'] = f"General comparison error: {e}"
        
    #     return {
    #         "is_equal": result['is_equal'],
    #         "details": {
    #             "method": result['method'],
    #             "parsed_eq1": result['parsed_eq1'],
    #             "parsed_eq2": result['parsed_eq2'],
    #             "sympy_eq1": result['sympy_eq1'],
    #             "sympy_eq2": result['sympy_eq2'],
    #             "error": result['error']
    #         }
    #     }
    
    def can_compare(self, answer1: Answer, answer2: Answer) -> bool:
        """Check if both answers are symbolic."""
        return (isinstance(answer1, SymbolicAnswer) and 
                isinstance(answer2, SymbolicAnswer))
