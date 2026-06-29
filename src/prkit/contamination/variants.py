"""Parametric variant generation, semantics-verified (X3 sub-feature C).

ABench-Physics ``Phy_B``-style memorization defense: perturb the numeric
constants in a templated problem and re-derive the answer, so a model cannot
win by having memorized the original. PRKit's twist (the **N1 dependency**):
when a template supplies *two* independent derivation paths (a Python
``answer_fn`` and a SymPy ``answer_expr``), each variant's re-derived answer is
cross-checked through the shipped semantics verifier
(:func:`prkit.verify.verify`), so broken templates surface as ``verified=False``
instead of silently emitting wrong gold answers.

``sympy`` (core dep) and :func:`prkit.verify.verify` (light-import) are imported
lazily so the bare module import stays cheap.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from prkit.contamination.provenance import ProblemProvenance, attach_problem_provenance
from prkit.core.domain import PhysicsDataset, PhysicsProblem
from prkit.core.domain.answer import PhysicsAnswer

if TYPE_CHECKING:
    from prkit.core.verdict import Verdict


@dataclass(frozen=True)
class ParamSpec:
    """Uniform sampling spec for a single template parameter."""

    low: float
    high: float
    integer: bool = False

    def sample(self, rng: random.Random) -> float:
        value = rng.uniform(self.low, self.high)
        return float(round(value)) if self.integer else value


@dataclass
class ParametricTemplate:
    """A problem with ``{param}`` placeholders and a closed-form answer rule."""

    base_problem_id: str
    question_template: str
    params: dict[str, ParamSpec]
    #: Python re-derivation: ``params -> answer`` (number, string, or PhysicsAnswer).
    answer_fn: Callable[[dict[str, float]], PhysicsAnswer | str | float] | None = None
    #: SymPy-evaluable expression over the param names, e.g. ``"m * g"``.
    answer_expr: str | None = None
    #: Unit attached to a numerically re-derived answer.
    unit: str | None = None
    #: Forwarded onto the synthetic problem (domain, problem_type, ...).
    base_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantResult:
    """One generated variant + its verification outcome."""

    problem: PhysicsProblem
    params: dict[str, float]
    verified: bool
    verdict_detail: Verdict | None = None


def generate_variants(
    template: ParametricTemplate,
    n: int = 5,
    *,
    seed: int | None = None,
    verify: bool = True,
) -> list[VariantResult]:
    """Generate *n* number-swapped variants of *template*.

    Each variant substitutes freshly-sampled parameter values into
    ``question_template`` and re-derives the answer. When both ``answer_fn`` and
    ``answer_expr`` are supplied and *verify* is True, the two derivations are
    cross-checked through :func:`prkit.verify.verify`; ``verified`` mirrors the
    verdict (``False`` when they disagree). With a single derivation path,
    ``verified`` is ``True`` (nothing to cross-check). Deterministic for a fixed
    *seed*.
    """
    if template.answer_fn is None and template.answer_expr is None:
        raise ValueError(
            "ParametricTemplate needs at least one of answer_fn / answer_expr."
        )

    rng = random.Random(seed)
    results: list[VariantResult] = []
    for index in range(n):
        values = {name: spec.sample(rng) for name, spec in template.params.items()}
        question = template.question_template.format(**values)

        fn_answer = _from_answer_fn(template, values)
        expr_answer = _from_answer_expr(template, values)
        primary = fn_answer if fn_answer is not None else expr_answer
        assert primary is not None  # guaranteed by the validation above

        verified, verdict = _verify_consistency(fn_answer, expr_answer, enabled=verify)

        problem = PhysicsProblem(
            problem_id=f"{template.base_problem_id}__var{index}",
            question=question,
            answer=primary,
            **template.base_fields,
        )
        attach_problem_provenance(
            problem,
            ProblemProvenance(
                source_dataset=template.base_fields.get("source_dataset", "synthetic"),
                is_synthetic=True,
                parent_problem_id=template.base_problem_id,
            ),
        )
        results.append(
            VariantResult(
                problem=problem,
                params=values,
                verified=verified,
                verdict_detail=verdict,
            )
        )
    return results


def variants_to_dataset(
    results: Sequence[VariantResult], *, name: str = "synthetic_variants"
) -> PhysicsDataset:
    """Package verified-or-not variants into a provenance-stamped ``eval`` dataset."""
    return PhysicsDataset(
        [r.problem for r in results],
        info={"name": name, "synthetic": True},
        split="eval",
    )


def _to_answer(value: PhysicsAnswer | str | float, unit: str | None) -> PhysicsAnswer:
    if isinstance(value, PhysicsAnswer):
        return value
    return PhysicsAnswer(value=_format_number(value), unit=unit)


def _format_number(value: str | float) -> str:
    if isinstance(value, str):
        return value
    # Render floats without spurious trailing zeros; keep ints exact.
    if value == int(value):
        return str(int(value))
    return repr(float(value))


def _from_answer_fn(
    template: ParametricTemplate, values: dict[str, float]
) -> PhysicsAnswer | None:
    if template.answer_fn is None:
        return None
    return _to_answer(template.answer_fn(values), template.unit)


def _from_answer_expr(
    template: ParametricTemplate, values: dict[str, float]
) -> PhysicsAnswer | None:
    if template.answer_expr is None:
        return None
    import sympy  # core dep; lazy to keep import light

    expr = sympy.sympify(template.answer_expr)
    substituted = expr.subs({sympy.Symbol(k): v for k, v in values.items()})
    evaluated = sympy.N(substituted)
    return _to_answer(_format_number(float(evaluated)), template.unit)


def _verify_consistency(
    fn_answer: PhysicsAnswer | None,
    expr_answer: PhysicsAnswer | None,
    *,
    enabled: bool,
) -> tuple[bool, Verdict | None]:
    if not enabled or fn_answer is None or expr_answer is None:
        return True, None
    from prkit.verify import verify  # light-import facade

    verdict = verify(fn_answer, expr_answer)
    return bool(verdict.correct), verdict
