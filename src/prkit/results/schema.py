"""Versioned, portable per-(problem, sample) eval record (roadmap X4).

A :class:`PhysicsEvalResult` is the self-describing, reproducible, harness-portable
record the keystone "physics-correctness scoring layer" emits: *what model, decoded
how, judged by which scorer, on which dataset version, produced which raw/extracted
answer, with what verdict — at what token/dollar cost.*

Design decision (updated from the roadmap draft, which predated the canonical
``Verdict`` shipping): the ``verdict`` payload is the frozen contract type
:class:`prkit.core.verdict.Verdict` (``prkit.api.Verdict``) — version-stamped and
exactly what every :class:`prkit.api.Scorer` emits — not the engine-internal
``AnswerComparison``. Deep engine evidence, when wanted, rides in ``Verdict.details``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from prkit.core.verdict import Verdict

if TYPE_CHECKING:
    from prkit.core.domain import PhysicsSolution
    from prkit.cost import CallRecord

SCHEMA_VERSION = "prkit.eval_result/1"


class _ResultModel(BaseModel):
    """Strict base for every result model (mirrors the semantics layer's house style)."""

    model_config = ConfigDict(extra="forbid")


class ModelProvenance(_ResultModel):
    provider: str  # e.g. "openai" (BaseModelClient.provider)
    model_name: str  # exact id submitted, e.g. "gpt-5.1"
    model_version: str | None = None  # provider-reported snapshot/version
    backend: str | None = None  # "openai" | "anthropic" | "vllm" | "ollama" | ...
    base_url: str | None = None
    model_args: dict[str, Any] = Field(default_factory=dict)


class DecodeParams(_ResultModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    reasoning_effort: str | None = None
    thinking_budget: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class JudgeProvenance(_ResultModel):
    judge_kind: Literal["deterministic", "llm", "none"] = "deterministic"
    judge_name: str | None = None
    judge_version: str | None = None
    judge_model: ModelProvenance | None = None  # only when judge_kind == "llm"
    prompt_name: str | None = None
    prompt_version: str | None = None


class ScorerProvenance(_ResultModel):
    scorer_name: str  # registered scorer id under the Scorer protocol
    scorer_version: str  # maps from Verdict.scorer_version
    policy_mode: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class CroissantRef(_ResultModel):
    name: str  # schema.org Dataset name
    url: str | None = None
    version: str | None = None
    sha256: str | None = None  # FileObject checksum
    conforms_to: str = "http://mlcommons.org/croissant/1.0"


class DatasetProvenance(_ResultModel):
    dataset_name: str  # DatasetHub registry key, e.g. "ugphysics"
    dataset_version: str | None = None
    variant: str | None = None
    split: str | None = None
    revision: str | None = None
    croissant: CroissantRef | None = None


class UsageAndCost(_ResultModel):
    input_tokens: int | None = None  # aligns with Inspect ModelUsage.input_tokens
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    cost_usd: float | None = None  # nullable when pricing unknown
    pricing_source: str | None = None
    latency_ms: float | None = None

    @classmethod
    def from_call_record(cls, record: CallRecord) -> UsageAndCost:
        """Lift an N6 :class:`prkit.cost.CallRecord` into a usage/cost facet."""
        usage = record.usage
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            cost_usd=record.cost,
            pricing_source=f"{record.provider}:{record.model}",
        )


class PhysicsEvalResult(_ResultModel):
    """One scored, provenanced (problem, sample) evaluation row."""

    # Plain str (not Literal) so the major-version guard in from_jsonl_line — and a
    # future migration hook — own compatibility, rather than Pydantic hard-rejecting.
    schema_version: str = SCHEMA_VERSION
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    # identity / linkage
    result_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str | None = None
    problem_id: str
    sample_index: int = 0
    domain: str | None = None
    problem_type: str | None = None

    # answers
    reference_answer: str | None = None
    raw_response: str | None = None
    extracted_answer: str | None = None

    # verdict — the canonical, version-stamped contract type
    verdict: Verdict | None = None
    correct: bool | None = None
    score: float | None = None

    # provenance
    model: ModelProvenance
    decode: DecodeParams = Field(default_factory=DecodeParams)
    judge: JudgeProvenance = Field(default_factory=JudgeProvenance)
    scorer: ScorerProvenance
    dataset: DatasetProvenance
    usage: UsageAndCost = Field(default_factory=UsageAndCost)

    # integrity / governance
    contaminated: bool | None = None
    contamination_source: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------ bridges
    @classmethod
    def from_solution(
        cls,
        solution: PhysicsSolution,
        *,
        model: ModelProvenance,
        scorer: ScorerProvenance,
        dataset: DatasetProvenance,
        verdict: Verdict | None = None,
        **overrides: Any,
    ) -> PhysicsEvalResult:
        """Lift an unscored :class:`PhysicsSolution` into a provenanced+scored result."""
        fields: dict[str, Any] = {
            "problem_id": solution.problem_id,
            "raw_response": solution.agent_answer,
            "verdict": verdict,
            "correct": None if verdict is None else verdict.correct,
            "score": None if verdict is None else verdict.score,
            "model": model,
            "scorer": scorer,
            "dataset": dataset,
            "metadata": dict(solution.metadata or {}),
        }
        fields.update(overrides)
        return cls(**fields)

    def to_solution(self) -> PhysicsSolution:
        """Reverse bridge (lossy): provenance/verdict collapse into ``metadata``."""
        from prkit.core.domain import PhysicsProblem, PhysicsSolution

        problem = PhysicsProblem(
            problem_id=self.problem_id,
            question=str(self.metadata.get("question", "")),
        )
        meta = dict(self.metadata)
        meta.setdefault("eval_result_id", self.result_id)
        return PhysicsSolution(
            problem_id=self.problem_id,
            problem=problem,
            agent_answer=self.raw_response or "",
            metadata=meta,
        )

    # -------------------------------------------------------------- JSONL (canonical)
    def to_jsonl_line(self) -> str:
        """Serialize to one compact JSON object (no trailing newline)."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl_line(cls, line: str) -> PhysicsEvalResult:
        """Parse one JSONL line, guarding the schema major version."""
        record = cls.model_validate_json(line)
        _check_schema_version(record.schema_version)
        return record

    # ---------------------------------------------------------- parquet (analytics)
    def to_flat_row(self) -> dict[str, Any]:
        """Flatten to parquet-friendly scalar columns + a lossless ``_json`` blob.

        Promotes the columns analytics actually group/filter on; the full nested
        record (verdict, provenance, dicts) round-trips losslessly through
        ``_json`` (re-read with :meth:`from_flat_row`).
        """
        return {
            "result_id": self.result_id,
            "run_id": self.run_id,
            "problem_id": self.problem_id,
            "sample_index": self.sample_index,
            "domain": self.domain,
            "provider": self.model.provider,
            "model_name": self.model.model_name,
            "dataset_name": self.dataset.dataset_name,
            "scorer_name": self.scorer.scorer_name,
            "comparison_mode": (
                None if self.verdict is None else self.verdict.comparison_mode
            ),
            "correct": self.correct,
            "score": self.score,
            "cost_usd": self.usage.cost_usd,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "contaminated": self.contaminated,
            "error": self.error,
            "created_at": self.created_at,
            "_json": self.to_jsonl_line(),
        }

    @classmethod
    def from_flat_row(cls, row: dict[str, Any]) -> PhysicsEvalResult:
        """Reconstruct from a :meth:`to_flat_row` mapping (via the ``_json`` blob)."""
        return cls.from_jsonl_line(row["_json"])


def _check_schema_version(version: str) -> None:
    supported_major = SCHEMA_VERSION.rsplit("/", 1)[-1]
    incoming_major = str(version).rsplit("/", 1)[-1]
    if incoming_major != supported_major:
        raise ValueError(
            f"Unsupported result schema_version {version!r}; this build supports "
            f"{SCHEMA_VERSION!r}. (A migration hook would live here.)"
        )
