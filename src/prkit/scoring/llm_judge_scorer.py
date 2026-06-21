"""Model-graded :class:`prkit.api.Scorer` wrapping the OpenAI LLM-judge engine.

``LLMJudgeScorer`` is the one non-deterministic scorer in PRKit's reference set.
It adapts :class:`prkit.evaluation.llm_judge.OpenAIJudgeRunner` (the model-graded
physics judge engine, which stays in ``prkit.evaluation``) into the canonical
:class:`~prkit.core.verdict.Verdict`.

Import discipline (enforced by ``tests/prkit/verify/test_import_isolation.py``):
``import prkit.scoring`` must stay free of ``openai`` and the
``prkit.evaluation.llm_judge`` subpackage. Every judge import is therefore
deferred to a method body (``__init__`` / ``score``); annotations that need the
judge types reference them only under :data:`typing.TYPE_CHECKING`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prkit.core.domain.answer import Answer
from prkit.core.verdict import Verdict

if TYPE_CHECKING:  # annotations only — never imported at runtime by this module
    from prkit.evaluation.llm_judge.runner import OpenAIJudgeRunner
    from prkit.evaluation.llm_judge.types import LLMJudgeResult

#: Local scorer-wiring revision (the ``wrapN`` slot of the canonical provenance
#: format). The judged model is recorded per-verdict in ``details["model"]`` and
#: surfaced by ``get_info()``, not baked into this stamp.
_VERSION = "llm-judge/openai+wrap1"


class LLMJudgeScorer:
    """Model-graded :class:`prkit.api.Scorer` over the OpenAI physics judge.

    Args:
        model: The judge model name. Required to build the default runner; it may
            be omitted only when an explicit *runner* is injected.
        instructions: Optional grading-instruction override; the runner defaults
            it to the standard physics grading prompt when ``None``.
        runner: An injectable judge runner — anything exposing ``judge(payload)
            -> LLMJudgeResult`` and a ``model_name`` property. Defaults to a real
            :class:`~prkit.evaluation.llm_judge.OpenAIJudgeRunner`; tests inject a
            fake to avoid constructing an OpenAI client or hitting the network.
    """

    version: str = _VERSION

    def __init__(
        self,
        *,
        model: str | None = None,
        instructions: str | None = None,
        runner: OpenAIJudgeRunner | Any | None = None,
    ) -> None:
        if runner is None:
            if not model:
                raise ValueError(
                    "LLMJudgeScorer requires `model` when no `runner` is injected"
                )
            # Lazy import keeps ``import prkit.scoring`` free of openai /
            # prkit.evaluation.llm_judge (enforced by test_import_isolation).
            from prkit.evaluation.llm_judge.runner import OpenAIJudgeRunner

            runner = OpenAIJudgeRunner(model=model, instructions=instructions)
        self._runner = runner

    def score(
        self,
        prediction: Answer | str,
        reference: Answer | str,
        *,
        question: str | None = None,
        **kwargs: Any,
    ) -> Verdict:
        """Grade ``prediction`` against ``reference`` with the LLM judge.

        ``question``, when supplied, is embedded in the judge payload so the model
        has the problem context. Returns a binary :class:`Verdict` (``score`` is
        ``1.0`` for a ``"correct"`` judgement, else ``0.0``); the judge's
        natural-language explanation is surfaced in ``rationale``.
        """
        # Lazy import: must not pull prkit.evaluation.llm_judge at module load.
        from prkit.evaluation.llm_judge.payload import (
            build_standard_answer_judge_payload,
        )

        payload = build_standard_answer_judge_payload(prediction, reference, question)
        result = self._runner.judge(payload)
        return self._verdict_from_result(result)

    def _verdict_from_result(self, result: LLMJudgeResult) -> Verdict:
        """Map an :class:`LLMJudgeResult` onto the canonical :class:`Verdict`.

        ``Verdict`` is ``frozen`` / ``extra="forbid"``; every ``details`` value
        here is a JSON-serializable scalar/string. ``deterministic=False`` lives
        in ``get_info()``, never on the Verdict.
        """
        equivalent = result.verdict == "correct"
        return Verdict(
            equivalent=equivalent,
            correct=equivalent,
            score=1.0 if equivalent else 0.0,
            comparison_mode=f"llm_judge:{result.expected_answer_type}",
            scorer_version=self.version,
            partial_credit=None,
            rationale=result.reasoning,
            details={
                "confidence": result.confidence,
                "expected_answer_type": result.expected_answer_type,
                "raw_response": result.raw_response,
                "verdict_type": result.verdict_type,
                "model": self._runner.model_name,
            },
        )

    def get_info(self) -> dict[str, Any]:
        """Return scorer metadata; always includes ``version`` (non-deterministic)."""
        return {
            "name": "LLMJudgeScorer",
            "version": self.version,
            "engine": "openai_judge",
            "deterministic": False,
            "model": self._runner.model_name,
        }
