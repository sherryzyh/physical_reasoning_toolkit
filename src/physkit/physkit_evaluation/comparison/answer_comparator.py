"""
Smart answer comparator that routes to appropriate comparison strategies.

This module provides the main SmartAnswerComparator class that automatically
selects and uses the appropriate comparator based on answer types with
enhanced error handling, validation, and debugging capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from physkit_core.definitions.answer_types import AnswerType
from physkit_core.models.answer import Answer
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
        
        Always uses fallback mode with automatic conversion to Answer(answer_type=TEXTUAL)
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
        # Smart type conversion:
        # if one is Answer and other is string, convert string to the type of the other answer
        # if both are strings, convert both to Answer(answer_type=TEXTUAL)
        answer1, answer2 = self._smart_type_conversion(answer1, answer2)
        
        # primary routing (same type comparison)
        primary_result = self._try_primary_comparison(answer1, answer2)
        if primary_result["success"]:
            return primary_result["result"]
        
        # Try fallback routing with Answer(answer_type=TEXTUAL) conversion
        fallback_result = self._try_fallback_comparison(answer1, answer2)
        if fallback_result["success"]:
            return fallback_result["result"]
        
        # All attempts failed
        return self._create_failure_result(answer1, answer2, primary_result, fallback_result)
    
    def _smart_type_conversion(
        self,
        answer1: Union[Answer, str],
        answer2: Union[Answer, str],
    ) -> tuple[Answer, Answer]:
        """
        Intelligently convert mixed types to ensure both are Answer instances.
        
        If one input is an Answer instance and the other is a string, convert the string
        to an Answer object with answer_type=TEXTUAL for seamless comparison.
        
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
            # answer1 is Answer, answer2 is string - convert answer2 to type of answer1
            answer2 = Answer(value=answer2, answer_type=answer1.answer_type)
        elif not is_answer1 and is_answer2:
            # answer1 is string, answer2 is Answer - convert answer1 to type of answer2
            answer1 = Answer(value=answer1, answer_type=answer2.answer_type)
        elif not is_answer1 and not is_answer2:
            # Both are strings - convert both to Answer(answer_type=TEXTUAL)
            answer1 = self._string_to_textual_answer(answer1)
            answer2 = self._string_to_textual_answer(answer2)
        # If both are already Answer instances, no conversion needed
        
        return answer1, answer2
    
    def _string_to_textual_answer(self, string_value: str) -> Answer:
        """
        Convert a string to an Answer object with answer_type=TEXTUAL.
        
        Args:
            string_value: String to convert
            
        Returns:
            Answer object with the string value and answer_type=TEXTUAL
        """
        metadata = {
            "original_input_type": "string",
            "conversion_method": "automatic_string_to_textual",
            "source": "SmartAnswerComparator._string_to_textual_answer"
        }
        return Answer(
            value=str(string_value),
            answer_type=AnswerType.TEXTUAL,
            metadata=metadata,
        )
    
    def _try_primary_comparison(
        self,
        answer1: Answer,
        answer2: Answer,
    ) -> Dict[str, Any]:
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
                        "suggestion": "Fallback will convert both to Answer(answer_type=TEXTUAL) for comparison"
                    }
                }
            
            # Get appropriate comparator
            shared_answer_type = answer1.answer_type
            comparator = self.comparators.get(shared_answer_type)
            
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
                "error": "Primary comparison exception",
                "details": {
                    "error": f"Exception during primary comparison: {str(e)}",
                    "exception_type": type(e).__name__,
                    "comparator": shared_answer_type.value
                }
            }
    
    def _try_fallback_comparison(
        self,
        answer1: Answer,
        answer2: Answer,
    ) -> Dict[str, Any]:
        """Try fallback routing by converting both answers to Answer(answer_type=TEXTUAL)."""
        try:
            # Convert both answers to Answer(answer_type=TEXTUAL) for cross-type comparison
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
                    "converted_to": "Answer",
                    "conversion_method": "automatic_fallback"
                },
                "warning": "Used fallback routing with Answer conversion",
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
                    "fallback_method": "Answer conversion"
                }
            }
    
    def _convert_to_textual_answer(self, answer: Answer) -> Answer:
        """
        Convert any answer type or string to Answer(answer_type=TEXTUAL) for fallback comparison.
        
        Args:
            answer: Answer to convert (Answer instance or string)
            
        Returns:
            Answer with the converted content
        """
        # Extract the value as a string representation
        value_str = str(answer.value)
        
        # Create Answer(answer_type=TEXTUAL) with metadata about the conversion
        metadata = {
            "original_type": answer.answer_type.value,
            "converted_from": answer.__class__.__name__,
            "conversion_method": "fallback_textual_conversion"
        }
        if hasattr(answer, 'metadata') and answer.metadata:
            metadata.update(answer.metadata)
        return Answer(
            value=value_str,
            answer_type=AnswerType.TEXTUAL,
            metadata=metadata,
        )
    
    def _create_failure_result(
        self,
        answer1: Answer,
        answer2: Answer,
        primary_result: Dict[str, Any],
        fallback_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create comprehensive failure result."""
        return {
            "is_equal": False,
            "details": {
                "error": "All comparison attempts failed",
                "primary_failure": primary_result,
                "fallback_failure": fallback_result,
                "answer_info": {
                    "answer1": {
                        "type": answer1.answer_type.value,
                        "value_preview": str(getattr(answer1, 'value', 'N/A'))[:50] + "..." 
                                       if len(str(getattr(answer1, 'value', ''))) > 50 
                                       else str(getattr(answer1, 'value', 'N/A'))
                    },
                    "answer2": {
                        "type": answer2.answer_type.value,
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
        return [t.value for t in AnswerType]
    
    def diagnose(self, answer1: Answer, answer2: Answer) -> Dict[str, Any]:
        """
        Diagnose why comparison might fail without actually comparing.
        
        Returns detailed analysis of compatibility.
        """
        # Smart type conversion for diagnosis
        answer1, answer2 = self._smart_type_conversion(
            answer1, answer2,
        )
        
        diagnosis = {
            "type_compatibility": {},
            "comparator_compatibility": {},
            "recommendations": [],
            "smart_conversion_applied": False
        }
        
        # The following diagnosis logic is kept for backward compatibility with previous code,
        # but TextualAnswer is now just Answer(answer_type=TEXTUAL).
        # The rest of the logic is unchanged.
        if (
            isinstance(answer1, Answer)
            and answer1.answer_type == AnswerType.TEXTUAL
            and getattr(answer1, "metadata", {}).get("conversion_method") == "automatic_string_to_textual"
        ):
            diagnosis["smart_conversion_applied"] = True
            diagnosis["conversion_details"] = {
                "answer1_converted": True,
                "original_type": "string"
            }
        if (
            isinstance(answer2, Answer)
            and answer2.answer_type == AnswerType.TEXTUAL
            and getattr(answer2, "metadata", {}).get("conversion_method") == "automatic_string_to_textual"
        ):
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
                diagnosis["recommendations"].append("🔄 Types don't match - will use fallback with Answer(answer_type=TEXTUAL) conversion")
                
            diagnosis["recommendations"].append("🔄 Fallback mode always enabled - converts to Answer(answer_type=TEXTUAL) for cross-type comparison")
            diagnosis["recommendations"].append("🧠 Smart type conversion automatically handles mixed Answer/string inputs")
        
        return diagnosis
