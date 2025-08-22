"""
Symbolic comparator for mathematical expressions and equations.

This module provides comparison strategies for symbolic mathematical
answers using latex2sympy2_extended for robust LaTeX parsing.
"""

import re
from typing import Any, Dict, Optional
import sympy as sp
from physkit_core.definitions.answer_types import Answer, SymbolicAnswer
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
            
            # First try symbolic comparison
            zero_exp=time_simplify(sp.expand(answer1_exp-answer2_exp))
            symbolic_equal = zero_exp==0 or answer1_exp==answer2_exp
            
            # If symbolic comparison fails, try enhanced symbolic comparison
            enhanced_symbolic_equal = False
            if not symbolic_equal:
                enhanced_symbolic_equal = self._check_enhanced_symbolic_equivalence(answer1_exp, answer2_exp)
            
            # Only try numerical comparison if both symbolic methods fail AND expressions are numerical
            numerical_equal = False
            if not symbolic_equal and not enhanced_symbolic_equal:
                if self._are_expressions_numerical(answer1_exp, answer2_exp):
                    numerical_equal = self._check_numerical_equivalence(answer1_exp, answer2_exp)
            
            result={
                "is_equal": symbolic_equal or enhanced_symbolic_equal or numerical_equal,
                "details":{
                    "method": "symbolic" if symbolic_equal else ("enhanced_symbolic" if enhanced_symbolic_equal else "numerical"),
                    "parsed_eq1": answer1.value,
                    "parsed_eq2": answer2.value,
                    "sympy_eq1": answer1_exp,
                    "sympy_eq2": answer2_exp,
                    "zero_exp": zero_exp,
                    "symbolic_equal": symbolic_equal,
                    "enhanced_symbolic_equal": enhanced_symbolic_equal,
                    "numerical_equal": numerical_equal,
                    "expressions_numerical": self._are_expressions_numerical(answer1_exp, answer2_exp)
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
    
    def _check_numerical_equivalence(self, expr1: sp.Basic, expr2: sp.Basic, tolerance: float = 1e-10) -> bool:
        """
        Check if two expressions are numerically equivalent.
        
        Args:
            expr1: First SymPy expression
            expr2: Second SymPy expression
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if expressions are numerically equivalent within tolerance
        """
        try:
            # Try to evaluate both expressions numerically
            # First, check if they can be evaluated to numbers
            if expr1.is_number and expr2.is_number:
                # Both are numbers, compare directly
                return abs(float(expr1) - float(expr2)) < tolerance
            
            # Try to substitute common variables with random values and compare
            # This handles cases like comparing 0.8 vs 2/5
            if self._can_evaluate_numerically(expr1) and self._can_evaluate_numerically(expr2):
                # Evaluate both expressions
                val1 = self._evaluate_expression(expr1)
                val2 = self._evaluate_expression(expr2)
                
                if val1 is not None and val2 is not None:
                    return abs(val1 - val2) < tolerance
            
            return False
            
        except Exception:
            return False
    
    def _check_enhanced_symbolic_equivalence(self, expr1: sp.Basic, expr2: sp.Basic) -> bool:
        """
        Enhanced symbolic comparison for expressions that might not simplify to zero directly.
        
        This method handles cases where expressions are symbolically equivalent but
        don't simplify to zero through basic expansion and simplification.
        
        Args:
            expr1: First SymPy expression
            expr2: Second SymPy expression
            
        Returns:
            True if expressions are symbolically equivalent
        """
        try:
            # Method 1: Try different simplification strategies
            simplified1 = time_simplify(expr1)
            simplified2 = time_simplify(expr2)
            
            if simplified1 == simplified2:
                return True
            
            # Method 2: Check if they're structurally identical
            if expr1 == expr2:
                return True
            
            # Method 3: Try factoring and expanding in different ways
            try:
                factored1 = sp.factor(expr1)
                factored2 = sp.factor(expr2)
                if factored1 == factored2:
                    return True
            except:
                pass
            
            # Method 4: Check if they're both equations and compare sides
            if self._are_both_equations(expr1, expr2):
                return self._compare_equation_sides(expr1, expr2)
            
            # Method 5: Try to solve for variables and check equivalence
            if self._can_solve_for_equivalence(expr1, expr2):
                return self._solve_for_equivalence(expr1, expr2)
            
            return False
            
        except Exception:
            return False
    
    def _are_expressions_numerical(self, expr1: sp.Basic, expr2: sp.Basic) -> bool:
        """
        Check if both expressions can be evaluated to numerical values.
        
        Args:
            expr1: First SymPy expression
            expr2: Second SymPy expression
            
        Returns:
            True if both expressions are purely numerical (no variables)
        """
        try:
            # Check if both expressions contain no free symbols (variables)
            return len(expr1.free_symbols) == 0 and len(expr2.free_symbols) == 0
        except Exception:
            return False
    
    def _are_both_equations(self, expr1: sp.Basic, expr2: sp.Basic) -> bool:
        """Check if both expressions are equations."""
        try:
            return (hasattr(expr1, 'lhs') and hasattr(expr1, 'rhs') and 
                    hasattr(expr2, 'lhs') and hasattr(expr2, 'rhs'))
        except Exception:
            return False
    
    def _compare_equation_sides(self, eq1: sp.Basic, eq2: sp.Basic) -> bool:
        """Compare equation sides for equivalence."""
        try:
            # Compare left sides
            left_equal = time_simplify(sp.expand(eq1.lhs - eq2.lhs)) == 0
            # Compare right sides
            right_equal = time_simplify(sp.expand(eq1.rhs - eq2.rhs)) == 0
            return left_equal and right_equal
        except Exception:
            return False
    
    def _can_solve_for_equivalence(self, expr1: sp.Basic, expr2: sp.Basic) -> bool:
        """Check if we can solve expressions for equivalence."""
        try:
            # Only try solving if expressions are relatively simple
            complexity1 = self._get_expression_complexity(expr1)
            complexity2 = self._get_expression_complexity(expr2)
            return complexity1 < 10 and complexity2 < 10  # Arbitrary complexity threshold
        except Exception:
            return False
    
    def _solve_for_equivalence(self, expr1: sp.Basic, expr2: sp.Basic) -> bool:
        """Try to solve expressions for equivalence."""
        try:
            # Try to solve expr1 = expr2
            solution = sp.solve(expr1 - expr2)
            # If solution is empty list, expressions are never equal
            # If solution is a list with values, expressions can be equal
            return len(solution) > 0
        except Exception:
            return False
    
    def _get_expression_complexity(self, expr: sp.Basic) -> int:
        """Get a rough measure of expression complexity."""
        try:
            # Count operations and symbols
            count = 0
            for arg in expr.args:
                count += 1
                if hasattr(arg, 'args'):
                    count += self._get_expression_complexity(arg)
            return count
        except Exception:
            return 0
    
    def _can_evaluate_numerically(self, expr: sp.Basic) -> bool:
        """Check if expression can be evaluated to a number."""
        try:
            # Check if it's a simple expression that can be evaluated
            if expr.is_number:
                return True
            
            # Check if it's a fraction, decimal, or simple arithmetic
            if expr.is_rational or expr.is_float:
                return True
            
            # Check if it's a simple arithmetic expression
            if expr.is_Add or expr.is_Mul or expr.is_Pow:
                # Check if all sub-expressions are numbers
                return all(arg.is_number for arg in expr.args)
            
            return False
        except Exception:
            return False
    
    def _evaluate_expression(self, expr: sp.Basic) -> Optional[float]:
        """Evaluate expression to a numerical value."""
        try:
            if expr.is_number:
                return float(expr)
            
            # Try to evaluate the expression
            result = expr.evalf()
            if result.is_number:
                return float(result)
            
            return None
        except Exception:
            return None
    
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
