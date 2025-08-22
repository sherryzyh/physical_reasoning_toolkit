"""
Smart answer comparator that routes to appropriate comparison strategies.

This module provides the main SmartAnswerComparator class that automatically
selects and uses the appropriate comparator based on answer types with
enhanced error handling, validation, and debugging capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from physkit_core.definitions.answer_types import Answer, AnswerType, TextualAnswer
from .symbolic import SymbolicComparator
from .numerical import NumericalComparator
from .textual import TextualComparator
from .option import OptionComparator


class SmartAnswerComparator:
    """
    Enhanced comparator with intelligent routing, validation, and debugging.
    
    Features:
    - Automatic comparator selection based on answer types
    - Enhanced error handling with detailed diagnostics
    - Validation using can_compare() safety checks
    - Support for debugging and introspection
    - Fallback mechanisms for edge cases
    """
    
    def __init__(self):
        """
        Initialize the smart answer comparator.
        
        Always uses fallback mode with automatic TextualAnswer conversion
        for cross-type comparisons.
        """
        # Primary comparators mapped by answer type
        self.comparators = {
            AnswerType.SYMBOLIC: SymbolicComparator(),
            AnswerType.NUMERICAL: NumericalComparator(),
            AnswerType.TEXTUAL: TextualComparator(),
            AnswerType.OPTION: OptionComparator(),
        }
        
        # Textual comparator for fallback cross-type comparisons
        self.textual_comparator = TextualComparator()
    
    def compare(
        self,
        answer1: Union[Answer, str],
        answer2: Union[Answer, str],
    ) -> Dict[str, Any]:
        """
        Intelligently compare two answers with enhanced error handling.
        
        Args:
            answer1: First answer to compare (Answer instance or string)
            answer2: Second answer to compare (Answer instance or string)
            
        Returns:
            Comparison results with enhanced diagnostics
        """
        # Smart type conversion: if one is Answer and other is string, convert string to TextualAnswer
        answer1, answer2 = self._smart_type_conversion(answer1, answer2)
        
        # Input validation
        validation_result = self._validate_inputs(answer1, answer2)
        if not validation_result["valid"]:
            return {
                "is_equal": False,
                "details": {
                    "error": "Input validation failed",
                    "validation_details": validation_result,
                    "comparator_used": None
                }
            }
        
        # Try primary routing first (same type comparison)
        primary_result = self._try_primary_comparison(answer1, answer2)
        if primary_result["success"]:
            return primary_result["result"]
        
        # Try fallback routing with TextualAnswer conversion
        fallback_result = self._try_fallback_comparison(answer1, answer2)
        if fallback_result["success"]:
            return fallback_result["result"]
        
        # All attempts failed
        return self._create_failure_result(answer1, answer2, primary_result, fallback_result)
    
    def _smart_type_conversion(self, answer1: Union[Answer, str], answer2: Union[Answer, str]) -> tuple[Answer, Answer]:
        """
        Intelligently convert mixed types to ensure both are Answer instances.
        
        If one input is an Answer instance and the other is a string, convert the string
        to a TextualAnswer object for seamless comparison.
        
        Args:
            answer1: First answer (Answer instance or string)
            answer2: Second answer (Answer instance or string)
            
        Returns:
            Tuple of (answer1, answer2) where both are Answer instances
        """
        # Check if we have mixed types
        is_answer1 = isinstance(answer1, Answer)
        is_answer2 = isinstance(answer2, Answer)
        
        if is_answer1 and not is_answer2:
            # answer1 is Answer, answer2 is string - convert answer2 to TextualAnswer
            answer2 = self._string_to_textual_answer(answer2)
        elif not is_answer1 and is_answer2:
            # answer1 is string, answer2 is Answer - convert answer1 to TextualAnswer
            answer1 = self._string_to_textual_answer(answer1)
        elif not is_answer1 and not is_answer2:
            # Both are strings - convert both to TextualAnswer
            answer1 = self._string_to_textual_answer(answer1)
            answer2 = self._string_to_textual_answer(answer2)
        # If both are already Answer instances, no conversion needed
        
        return answer1, answer2
    
    def _string_to_textual_answer(self, string_value: str) -> TextualAnswer:
        """
        Convert a string to a TextualAnswer object.
        
        Args:
            string_value: String to convert
            
        Returns:
            TextualAnswer object with the string value
        """
        metadata = {
            "original_input_type": "string",
            "conversion_method": "automatic_string_to_textual",
            "source": "SmartAnswerComparator._string_to_textual_answer"
        }
        
        return TextualAnswer(
            value=str(string_value),
            metadata=metadata
        )
    
    def _validate_inputs(self, answer1: Union[Answer, str], answer2: Union[Answer, str]) -> Dict[str, Any]:
        """Validate input answers."""
        errors = []
        
        if answer1 is None:
            errors.append("answer1 is None")
        if answer2 is None:
            errors.append("answer2 is None")
            
        if not errors:
            # Check if answers have required attributes
            for i, answer in enumerate([answer1, answer2], 1):
                if not hasattr(answer, 'answer_type'):
                    errors.append(f"answer{i} missing answer_type attribute")
                if not hasattr(answer, 'value'):
                    errors.append(f"answer{i} missing value attribute")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "answer1_type": getattr(answer1, 'answer_type', None) if hasattr(answer1, 'answer_type') else type(answer1).__name__,
            "answer2_type": getattr(answer2, 'answer_type', None) if hasattr(answer2, 'answer_type') else type(answer2).__name__
        }
    
    def _try_primary_comparison(self, answer1: Union[Answer, str], answer2: Union[Answer, str]) -> Dict[str, Any]:
        """Try primary type-based routing (same type comparison)."""
        try:
            # Check if answer types match
            if answer1.answer_type != answer2.answer_type:
                return {
                    "success": False,
                    "error": "type_mismatch",
                    "details": {
                        "error": "Answer types do not match - will try fallback",
                        "type1": answer1.answer_type.value,
                        "type2": answer2.answer_type.value,
                        "suggestion": "Fallback will convert both to TextualAnswer for comparison"
                    }
                }
            
            # Get appropriate comparator
            comparator = self.comparators.get(answer1.answer_type)
            if comparator is None:
                return {
                    "success": False,
                    "error": "no_comparator",
                    "details": {
                        "error": f"No comparator available for {answer1.answer_type.value}",
                        "available_types": [t.value for t in self.comparators.keys()],
                        "suggestion": "Fallback will convert to TextualAnswer for comparison"
                    }
                }
            
            # Validate with can_compare
            if not comparator.can_compare(answer1, answer2):
                return {
                    "success": False,
                    "error": "validation_failed",
                    "details": {
                        "error": f"Comparator validation failed for {answer1.answer_type.value}",
                        "comparator": comparator.__class__.__name__,
                        "reason": "can_compare() returned False",
                        "suggestion": "Fallback will convert to TextualAnswer for comparison"
                    }
                }
            
            # Perform comparison
            result = comparator.compare(answer1, answer2)
            
            # Enhance result with metadata
            if "details" not in result:
                result["details"] = {}
            result["details"].update({
                "comparator_used": comparator.__class__.__name__,
                "routing_method": "primary",
                "answer_types": {
                    "answer1": answer1.answer_type.value,
                    "answer2": answer2.answer_type.value
                }
            })
            
            return {"success": True, "result": result}
            
        except Exception as e:
            return {
                "success": False,
                "error": "exception",
                "details": {
                    "error": f"Exception during primary comparison: {str(e)}",
                    "exception_type": type(e).__name__,
                    "comparator": comparator.__class__.__name__ if 'comparator' in locals() else None
                }
            }
    
    def _try_fallback_comparison(self, answer1: Union[Answer, str], answer2: Union[Answer, str]) -> Dict[str, Any]:
        """Try fallback routing by converting both answers to TextualAnswer."""
        try:
            # Convert both answers to TextualAnswer for cross-type comparison
            textual_answer1 = self._convert_to_textual_answer(answer1)
            textual_answer2 = self._convert_to_textual_answer(answer2)
            
            # Use TextualComparator for the converted answers
            result = self.textual_comparator.compare(textual_answer1, textual_answer2)
            
            # Enhance result with fallback metadata
            if "details" not in result:
                result["details"] = {}
            result["details"].update({
                "comparator_used": "TextualComparator",
                "routing_method": "fallback",
                "fallback_conversion": {
                    "answer1_original_type": answer1.answer_type.value,
                    "answer2_original_type": answer2.answer_type.value,
                    "converted_to": "TextualAnswer",
                    "conversion_method": "automatic_fallback"
                },
                "warning": "Used fallback routing with TextualAnswer conversion",
                "answer_types": {
                    "answer1": answer1.answer_type.value,
                    "answer2": answer2.answer_type.value
                }
            })
            
            return {"success": True, "result": result}
            
        except Exception as e:
            return {
                "success": False,
                "error": "fallback_exception",
                "details": {
                    "error": f"Exception during fallback comparison: {str(e)}",
                    "exception_type": type(e).__name__,
                    "fallback_method": "TextualAnswer conversion"
                }
            }
    
    def _convert_to_textual_answer(self, answer: Union[Answer, str]) -> TextualAnswer:
        """
        Convert any answer type or string to TextualAnswer for fallback comparison.
        
        Args:
            answer: Answer to convert (Answer instance or string)
            
        Returns:
            TextualAnswer with the converted content
        """
        # Handle string input
        if isinstance(answer, str):
            return self._string_to_textual_answer(answer)
        
        # Extract the value as a string representation
        if hasattr(answer, 'value'):
            value_str = str(answer.value)
        else:
            value_str = str(answer)
        
        # Create TextualAnswer with metadata about the conversion
        metadata = {
            "original_type": answer.answer_type.value,
            "converted_from": answer.__class__.__name__,
            "conversion_method": "fallback_textual_conversion"
        }
        
        # Preserve original metadata if available
        if hasattr(answer, 'metadata') and answer.metadata:
            metadata.update(answer.metadata)
        
        return TextualAnswer(
            value=value_str,
            metadata=metadata
        )
    
    def _create_failure_result(self, answer1: Union[Answer, str], answer2: Union[Answer, str], 
                             primary_result: Dict[str, Any], 
                             fallback_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create comprehensive failure result."""
        return {
            "is_equal": False,
            "details": {
                "error": "All comparison attempts failed",
                "primary_failure": primary_result,
                "fallback_failure": fallback_result,
                "answer_info": {
                    "answer1": {
                        "type": getattr(answer1, 'answer_type', 'unknown'),
                        "value_preview": str(getattr(answer1, 'value', 'N/A'))[:50] + "..." 
                                       if len(str(getattr(answer1, 'value', ''))) > 50 
                                       else str(getattr(answer1, 'value', 'N/A'))
                    },
                    "answer2": {
                        "type": getattr(answer2, 'answer_type', 'unknown'),
                        "value_preview": str(getattr(answer2, 'value', 'N/A'))[:50] + "..." 
                                       if len(str(getattr(answer2, 'value', ''))) > 50 
                                       else str(getattr(answer2, 'value', 'N/A'))
                    }
                },
                "suggestions": [
                    "Check if answer types are correct",
                    "Verify answer formats are valid",
                    "Try enabling fallback mode",
                    "Use specific comparator directly"
                ]
            }
        }
    
    def get_available_comparators(self) -> List[str]:
        """Get list of available comparator names."""
        return [comp.__class__.__name__ for comp in self.comparators.values()]
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported answer types."""
        return [t.value for t in self.comparators.keys()]
    
    def diagnose(self, answer1: Union[Answer, str], answer2: Union[Answer, str]) -> Dict[str, Any]:
        """
        Diagnose why comparison might fail without actually comparing.
        
        Returns detailed analysis of compatibility.
        """
        # Smart type conversion for diagnosis
        answer1, answer2 = self._smart_type_conversion(answer1, answer2)
        
        diagnosis = {
            "input_validation": self._validate_inputs(answer1, answer2),
            "type_compatibility": {},
            "comparator_compatibility": {},
            "recommendations": [],
            "smart_conversion_applied": False
        }
        
        # Check if smart conversion was applied
        if isinstance(answer1, TextualAnswer) and answer1.metadata.get("conversion_method") == "automatic_string_to_textual":
            diagnosis["smart_conversion_applied"] = True
            diagnosis["conversion_details"] = {
                "answer1_converted": True,
                "original_type": "string"
            }
        if isinstance(answer2, TextualAnswer) and answer2.metadata.get("conversion_method") == "automatic_string_to_textual":
            diagnosis["smart_conversion_applied"] = True
            if "conversion_details" not in diagnosis:
                diagnosis["conversion_details"] = {}
            diagnosis["conversion_details"]["answer2_converted"] = True
            diagnosis["conversion_details"]["original_type"] = "string"
        
        if diagnosis["input_validation"]["valid"]:
            # Check type compatibility
            types_match = answer1.answer_type == answer2.answer_type
            diagnosis["type_compatibility"] = {
                "types_match": types_match,
                "answer1_type": answer1.answer_type.value,
                "answer2_type": answer2.answer_type.value,
                "fallback_available": True
            }
            
            # Check each comparator's compatibility
            for answer_type, comparator in self.comparators.items():
                can_compare = False
                try:
                    can_compare = comparator.can_compare(answer1, answer2)
                except Exception as e:
                    can_compare = f"Error: {e}"
                
                diagnosis["comparator_compatibility"][answer_type.value] = {
                    "can_compare": can_compare,
                    "comparator_class": comparator.__class__.__name__
                }
            
            # Generate recommendations
            if types_match:
                if answer1.answer_type in self.comparators:
                    diagnosis["recommendations"].append("✅ Direct comparison should work")
                else:
                    diagnosis["recommendations"].append("❌ No comparator for this type")
            else:
                diagnosis["recommendations"].append("🔄 Types don't match - will use fallback with TextualAnswer conversion")
                
            diagnosis["recommendations"].append("🔄 Fallback mode always enabled - converts to TextualAnswer for cross-type comparison")
            diagnosis["recommendations"].append("🧠 Smart type conversion automatically handles mixed Answer/string inputs")
        
        return diagnosis
