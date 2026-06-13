"""
Construct a :class:`~prkit.prkit_evaluation.comparator.base.BaseComparator` from a
package submodule name (the Python module stem, e.g. ``\"smart_match\"`` for
``smart_match.py``).
"""

from __future__ import annotations

import importlib
from typing import Dict, FrozenSet, Optional, Type

from prkit.prkit_evaluation.comparator.base import BaseComparator

# Submodule name (``*.py`` stem) -> comparator class name in that module.
_MODULE_CLASS: Dict[str, str] = {
    "category_match": "CategoryComparator",
    "exact_match": "ExactMatchComparator",
    "normalized_match": "NormalizedMatchComparator",
    "record_match": "RecordMatchComparator",
    "similarity_match": "SimilarityMatchComparator",
    "smart_match": "SmartMatchComparator",
    "smart_llm": "SmartLLMComparator",
    "typed_llm": "TypedLLMComparator",
}

# Legacy / convenience aliases (e.g. old scripts referred to an LLM judge comparator).
_ALIASES: Dict[str, str] = {
    "llm_judge": "typed_llm",
}

_MODULES_ACCEPTING_MODEL: FrozenSet[str] = frozenset({"typed_llm", "smart_llm"})
_MODULES_WITH_ROUGE_THRESHOLD: FrozenSet[str] = frozenset({"similarity_match"})


def comparator_module_names() -> tuple[str, ...]:
    """Sorted valid submodule names (excluding aliases)."""
    return tuple(sorted(_MODULE_CLASS.keys()))


def resolve_comparator_module(name: str) -> str:
    """Map an alias or module name to the canonical submodule name."""
    n = name.strip()
    if not n:
        raise ValueError("Comparator name must be non-empty")
    return _ALIASES.get(n, n)


def uses_openai_model(module_name: str) -> bool:
    """Whether ``build_comparator(..., model=...)`` passes ``model`` to the class."""
    return resolve_comparator_module(module_name) in _MODULES_ACCEPTING_MODEL


def uses_rouge_threshold(module_name: str) -> bool:
    return resolve_comparator_module(module_name) in _MODULES_WITH_ROUGE_THRESHOLD


def build_comparator(
    module_name: str,
    *,
    model: Optional[str] = None,
    rouge_threshold: float = 0.5,
) -> BaseComparator:
    """
    Import ``prkit.prkit_evaluation.comparator.<module>`` and instantiate the
    registered comparator class.

    Parameters
    ----------
    module_name
        Submodule stem, e.g. ``smart_match``, or alias ``llm_judge`` → ``typed_llm``.
    model
        Passed to :class:`~prkit.prkit_evaluation.comparator.typed_llm.TypedLLMComparator`
        and :class:`~prkit.prkit_evaluation.comparator.smart_llm.SmartLLMComparator`.
        If omitted, ``prkit.prkit_evaluation.llm_judge.DEFAULT_MODEL`` is used.
    rouge_threshold
        Passed to :class:`~prkit.prkit_evaluation.comparator.similarity_match.SimilarityMatchComparator`.
    """
    from prkit.prkit_evaluation.llm_judge import DEFAULT_MODEL

    name = resolve_comparator_module(module_name)
    if name not in _MODULE_CLASS:
        valid = ", ".join(comparator_module_names())
        raise ValueError(
            f"Unknown comparator module {module_name!r}. Choose one of: {valid}"
        )

    mod = importlib.import_module(f"prkit.prkit_evaluation.comparator.{name}")
    cls: Type[BaseComparator] = getattr(mod, _MODULE_CLASS[name])

    if name in _MODULES_ACCEPTING_MODEL:
        m = model if model is not None else DEFAULT_MODEL
        return cls(model=m)

    if name in _MODULES_WITH_ROUGE_THRESHOLD:
        return cls(rouge_threshold=rouge_threshold)

    return cls()
