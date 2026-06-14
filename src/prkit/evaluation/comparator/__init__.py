"""Answer comparators for physical reasoning evaluation: exact, normalized, category, smart, LLM, and similarity."""

from .base import BaseComparator
from .by_module import (
    build_comparator,
    comparator_module_names,
    resolve_comparator_module,
    uses_openai_model,
    uses_rouge_threshold,
)
from .category_match import CategoryComparator
from .exact_match import ExactMatchComparator
from .normalized_match import NormalizedMatchComparator
from .record_match import RecordMatchComparator
from .similarity_match import SimilarityMatchComparator
from .smart_llm import SmartLLMComparator
from .smart_match import SmartMatchComparator
from .smart_pipeline import (
    SmartMatchPipelineHost,
    SmartPipelineResult,
    run_smart_pipeline,
)
from .typed_llm import TypedLLMComparator

__all__ = [
    "BaseComparator",
    "build_comparator",
    "CategoryComparator",
    "comparator_module_names",
    "resolve_comparator_module",
    "ExactMatchComparator",
    "NormalizedMatchComparator",
    "RecordMatchComparator",
    "SimilarityMatchComparator",
    "SmartLLMComparator",
    "SmartMatchComparator",
    "SmartMatchPipelineHost",
    "SmartPipelineResult",
    "TypedLLMComparator",
    "uses_openai_model",
    "uses_rouge_threshold",
    "run_smart_pipeline",
]
