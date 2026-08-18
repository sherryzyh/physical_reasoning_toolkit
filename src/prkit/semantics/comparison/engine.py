"""Recursive comparison engine for protocol-level physics answers."""

from __future__ import annotations

import re
from typing import Any

from ..part_labels import infer_multi_part_part_labels
from ..schema import (
    AnswerComparison,
    AnswerObjectKind,
    AnswerStructure,
    BridgeTier,
    ComparisonPolicyMode,
    ContractValidationStatus,
    OrderingPolicy,
    PhysicsAnswerSemantics,
    PhysicsEvaluationContract,
    PhysicsQuestionSemantics,
    QuestionSymbolicMode,
)
from .bridge_registry import (
    BRIDGE_REGISTRY,
    bridge_enabled_for_policy,
    bridge_spec_for,
)
from .coercion import (
    coerce_evaluation_contract,
    coerce_protocol_answer,
    coerce_question_semantics,
)
from .common import (
    _needs_surface_repair,
    available_texts,
    context_symbol_alias_map,
    effective_canonical_text,
    reparse_sources,
)
from .contract import (
    build_evaluation_contract,
    coerce_policy_mode,
    validate_answer_against_contract,
)
from .different_object_kind import compare_different_object_kinds
from .label_family_fallback import compare_label_family_fallback
from .same_object_kind import compare_same_object_kind
from .semantics import (
    canonicalize_qualitative_label,
    normalize_plain_text,
    parse_numeric_value,
)
from .sign_convention import (
    answer_directional_convention,
    compare_sign_convention,
    orientation_relation,
    vectors_exact_negation,
)
from .structure_canonicalization import canonicalize_structure


def compare_protocol_answers(
    pred: PhysicsAnswerSemantics | dict,
    ref: PhysicsAnswerSemantics | dict,
    *,
    contract: PhysicsEvaluationContract | dict | None = None,
    context: PhysicsQuestionSemantics | dict | None = None,
    policy_mode: ComparisonPolicyMode | str | None = None,
    _validate_top_level: bool = True,
    _allow_cross_kind_identical_text: bool = False,
) -> AnswerComparison:
    """Compare two protocol answers under a policy-controlled evaluation contract."""

    resolved_policy = coerce_policy_mode(policy_mode)
    resolved_context = coerce_question_semantics(context)
    pred_answer = _repair_answer_for_comparison(
        coerce_protocol_answer(pred),
        context=resolved_context,
    )
    ref_answer = _repair_answer_for_comparison(
        coerce_protocol_answer(ref),
        context=resolved_context,
    )
    resolved_contract = _resolve_comparison_contract(
        contract=contract,
        context=resolved_context,
        ref_answer=ref_answer,
    )
    resolved_context = resolved_contract.comparison_context

    pred_validation = None
    if _validate_top_level:
        pred_validation = validate_answer_against_contract(
            pred_answer, resolved_contract
        )
        ref_validation = validate_answer_against_contract(ref_answer, resolved_contract)
        if resolved_policy != ComparisonPolicyMode.PERMISSIVE:
            if ref_validation.status == ContractValidationStatus.VIOLATING:
                return AnswerComparison(
                    equivalent=False,
                    comparison_mode="reference_contract_violation",
                    diagnostics=ref_validation.diagnostics,
                    validation_status=ref_validation.status,
                    policy_mode=resolved_policy,
                )
            if pred_validation.status == ContractValidationStatus.VIOLATING:
                return AnswerComparison(
                    equivalent=False,
                    comparison_mode="contract_violation",
                    diagnostics=pred_validation.diagnostics,
                    validation_status=pred_validation.status,
                    policy_mode=resolved_policy,
                )
            if (
                resolved_policy == ComparisonPolicyMode.STRICT
                and pred_validation.status == ContractValidationStatus.COERCIBLE
            ):
                return AnswerComparison(
                    equivalent=False,
                    comparison_mode="contract_violation",
                    diagnostics=pred_validation.diagnostics,
                    validation_status=pred_validation.status,
                    policy_mode=resolved_policy,
                )

    if pred_answer.structure != ref_answer.structure:
        return _finalize_comparison(
            AnswerComparison(
                equivalent=False,
                comparison_mode="structure_mismatch",
                diagnostics=(
                    f"pred_structure={pred_answer.structure.value}",
                    f"ref_structure={ref_answer.structure.value}",
                ),
            ),
            validation_status=(
                pred_validation.status if pred_validation is not None else None
            ),
            policy_mode=resolved_policy,
        )

    if pred_answer.structure == AnswerStructure.ATOMIC:
        result = _compare_atomic(
            pred_answer,
            ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            policy_mode=resolved_policy,
            allow_cross_kind_identical_text=_allow_cross_kind_identical_text,
        )
        return _finalize_result(
            result,
            pred=pred_answer,
            ref=ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            validation_status=(
                pred_validation.status if pred_validation is not None else None
            ),
            policy_mode=resolved_policy,
        )

    if pred_answer.structure == AnswerStructure.MULTI_PART:
        if resolved_context.ordering == OrderingPolicy.UNORDERED:
            result = _compare_unordered_children(
                pred_answer,
                ref_answer,
                context=resolved_context,
                contract=resolved_contract,
                policy_mode=resolved_policy,
                mode="multi_part",
            )
        elif resolved_context.ordering == OrderingPolicy.PER_PART:
            result = _compare_per_part_children(
                pred_answer,
                ref_answer,
                context=resolved_context,
                contract=resolved_contract,
                policy_mode=resolved_policy,
                mode="multi_part",
            )
        else:
            result = _compare_ordered_children(
                pred_answer,
                ref_answer,
                context=resolved_context,
                contract=resolved_contract,
                policy_mode=resolved_policy,
                mode="multi_part",
            )
        return _finalize_result(
            result,
            pred=pred_answer,
            ref=ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            validation_status=(
                pred_validation.status if pred_validation is not None else None
            ),
            policy_mode=resolved_policy,
        )

    if pred_answer.structure == AnswerStructure.TUPLE:
        result = _compare_ordered_children(
            pred_answer,
            ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            policy_mode=resolved_policy,
            mode="tuple",
        )
        return _finalize_result(
            result,
            pred=pred_answer,
            ref=ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            validation_status=(
                pred_validation.status if pred_validation is not None else None
            ),
            policy_mode=resolved_policy,
        )

    if pred_answer.structure == AnswerStructure.SET:
        result = _compare_unordered_children(
            pred_answer,
            ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            policy_mode=resolved_policy,
            mode="set",
        )
        return _finalize_result(
            result,
            pred=pred_answer,
            ref=ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            validation_status=(
                pred_validation.status if pred_validation is not None else None
            ),
            policy_mode=resolved_policy,
        )

    if pred_answer.structure == AnswerStructure.INTERVAL:
        result = _compare_interval(
            pred_answer,
            ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            policy_mode=resolved_policy,
        )
        return _finalize_result(
            result,
            pred=pred_answer,
            ref=ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            validation_status=(
                pred_validation.status if pred_validation is not None else None
            ),
            policy_mode=resolved_policy,
        )

    if pred_answer.structure in {
        AnswerStructure.VECTOR,
        AnswerStructure.MATRIX,
        AnswerStructure.TENSOR,
    }:
        result = _compare_shaped(
            pred_answer,
            ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            policy_mode=resolved_policy,
        )
        return _finalize_result(
            result,
            pred=pred_answer,
            ref=ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            validation_status=(
                pred_validation.status if pred_validation is not None else None
            ),
            policy_mode=resolved_policy,
        )

    if pred_answer.structure == AnswerStructure.PIECEWISE:
        result = _compare_piecewise(
            pred_answer,
            ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            policy_mode=resolved_policy,
        )
        return _finalize_result(
            result,
            pred=pred_answer,
            ref=ref_answer,
            context=resolved_context,
            contract=resolved_contract,
            validation_status=(
                pred_validation.status if pred_validation is not None else None
            ),
            policy_mode=resolved_policy,
        )

    return _finalize_comparison(
        AnswerComparison(
            False, "unsupported_structure", (pred_answer.structure.value,)
        ),
        validation_status=(
            pred_validation.status if pred_validation is not None else None
        ),
        policy_mode=resolved_policy,
    )


def compare_protocol_answers_legacy(
    pred: PhysicsAnswerSemantics | dict,
    ref: PhysicsAnswerSemantics | dict,
    *,
    context: PhysicsQuestionSemantics | dict | None = None,
) -> AnswerComparison:
    """Legacy v1 comparison path kept for head-to-head benchmarking."""

    return compare_protocol_answers(
        pred,
        ref,
        context=context,
        policy_mode=ComparisonPolicyMode.PERMISSIVE,
        _allow_cross_kind_identical_text=True,
    )


def compare_predictions(
    a_i: PhysicsAnswerSemantics | dict,
    a_j: PhysicsAnswerSemantics | dict,
    *,
    context: PhysicsQuestionSemantics | dict | None = None,
    policy_mode: ComparisonPolicyMode | str | None = None,
) -> AnswerComparison:
    """Symmetric reference-free equivalence ``Eq(a_i, a_j ; q_prob)`` for clustering.

    Unlike :func:`compare_protocol_answers` — whose contract is co-built with a *gold*
    second argument — neither argument here is a reference, so deriving the expected
    kind/structure or the numeric precision bar from one of the two predictions is
    unsound (it makes the verdict depend on argument order and lets a prediction reject
    the very contract it defined). This entry point removes both hazards:

    - **Explicit ``q_prob`` contract.** A single contract is built from ``context``
      (``q_prob``) and shared by both directions, so the engine never infers
      ``expected_object_kind`` / ``expected_structure`` from one prediction. Its
      ``allowed_*`` and ``expected_*`` come from the question's declared admissibility
      (permissive when ``q_prob`` is unconstrained), so a self-derived
      ``reference_contract_violation`` cannot arise.
    - **Symmetrization.** The verdict is ``equivalent`` iff the engine accepts in *both*
      directions under that shared contract. This neutralizes the only remaining
      asymmetry — the reference-printed-precision steps of
      ``numbers_match_with_reference_precision`` (an order-sensitive accept that fires
      only one way is rejected) — without letting either side's printed precision set the
      bar. Relative ``q_prob.tolerance`` (symmetric by construction) still applies in both
      directions, so genuine numeric agreement is accepted both ways.

    The structure routing, per-kind criteria, and bridges are reused unchanged because
    they are already symmetric. Defaults to the permissive policy (no gold to gate
    against); an explicit ``policy_mode`` is honored and uses the shared ``q_prob``
    contract for any contract enforcement.

    See ``EQUIVALENCE.md`` §1.1 and §10.3 for the methodology.
    """

    resolved_policy = coerce_policy_mode(policy_mode)
    resolved_context = coerce_question_semantics(context)
    contract = _build_reference_free_contract(resolved_context)

    forward = _normalize_reference_free_mode(
        compare_protocol_answers(
            a_i,
            a_j,
            contract=contract,
            context=resolved_context,
            policy_mode=resolved_policy,
        )
    )
    if not forward.equivalent:
        return forward

    backward = compare_protocol_answers(
        a_j,
        a_i,
        contract=contract,
        context=resolved_context,
        policy_mode=resolved_policy,
    )
    if backward.equivalent:
        return forward
    return AnswerComparison(
        equivalent=False,
        comparison_mode=forward.comparison_mode,
        diagnostics=("asymmetric_match",) + backward.diagnostics,
        validation_status=forward.validation_status,
        policy_mode=resolved_policy,
    )


def _normalize_reference_free_mode(result: AnswerComparison) -> AnswerComparison:
    """Drop the incoherent ``reference_contract_violation`` mode when neither side is gold.

    ``compare_protocol_answers`` validates its *second* argument as the reference and labels
    a violation ``reference_contract_violation``. In reference-free comparison neither side
    is gold, so a contract violation of one prediction is just a ``contract_violation`` (a
    prediction failing the ``q_prob`` contract), never a *reference* violation. Remapping
    keeps the verdict symmetric and avoids the self-rejection audit #4 calls out — the
    contract was derived from ``q_prob``, not from the answer it rejects.
    """

    if result.comparison_mode != "reference_contract_violation":
        return result
    return result.model_copy(update={"comparison_mode": "contract_violation"})


def _build_reference_free_contract(
    context: PhysicsQuestionSemantics,
) -> PhysicsEvaluationContract:
    """Build a symmetric, ``q_prob``-derived contract for reference-free comparison.

    Expected kind/structure are taken from the question's *declared* admissibility
    (``allowed_object_kinds`` / ``allowed_structures``) rather than from either
    prediction: a singleton allowed set pins the expectation, while an unconstrained
    (default-permissive) set yields a permissive expectation that admits every kind /
    structure. This is the build-time analogue of ``q_prob`` carrying only ``allowed_*``
    and policy fields, never a realized answer.
    """

    expected_object_kind = _single_or_default(
        context.allowed_object_kinds,
        full=tuple(AnswerObjectKind),
        default=AnswerObjectKind.EXPRESSION,
    )
    expected_structure = _single_or_default(
        context.allowed_structures,
        full=tuple(AnswerStructure),
        default=AnswerStructure.ATOMIC,
    )
    return PhysicsEvaluationContract(
        question_semantics=context,
        expected_object_kind=expected_object_kind,
        expected_structure=expected_structure,
        target_variable=context.target_variable,
        enabled_bridge_ids=tuple(BRIDGE_REGISTRY),
        enabled_bridge_tiers=(BridgeTier.TIER1, BridgeTier.TIER2, BridgeTier.TIER3),
    )


def _single_or_default(
    values: tuple[Any, ...], *, full: tuple[Any, ...], default: Any
) -> Any:
    """Pin a singleton declared set, else fall back to a permissive default.

    A singleton ``allowed_*`` is an explicit, symmetric expectation. The default-permissive
    set (every member) carries no expectation, so a neutral ``default`` is used — under the
    permissive policy (the reference-free default) the contract's ``expected_*`` is never
    read, and under an explicit strict/audited override the neutral default keeps the gate
    from favoring one prediction's shape over the other's.
    """

    if len(values) == 1:
        return values[0]
    if values and set(values) != set(full):
        # A narrowed-but-not-singleton declaration; pick a member deterministically so the
        # shared contract is identical for both directions.
        return next(iter(values))
    return default


def _compare_atomic(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
    allow_cross_kind_identical_text: bool,
) -> AnswerComparison:
    """Compare two atomic answers, flagging any surface this engine had to recompute.

    The verdict itself comes from :func:`_compare_atomic_surfaces`. This wrapper only adds
    the ``canonical_text_repaired`` diagnostic, so an audit replay can measure how often a
    stored ``canonical_text`` was corrupt without re-deriving the condition.
    """

    result = _compare_atomic_surfaces(
        pred,
        ref,
        context=context,
        contract=contract,
        policy_mode=policy_mode,
        allow_cross_kind_identical_text=allow_cross_kind_identical_text,
    )
    if not (_needs_surface_repair(pred) or _needs_surface_repair(ref)):
        return result
    return result.model_copy(
        update={"diagnostics": (*result.diagnostics, "canonical_text_repaired")}
    )


def _compare_atomic_surfaces(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
    allow_cross_kind_identical_text: bool,
) -> AnswerComparison:
    """Compare two atomic answers using strict, bridged, and fallback logic."""

    if pred.object_kind == ref.object_kind:
        # Sign-convention reconciliation runs first for directional kinds so it can both
        # accept a global flip between opposite stated conventions and override a plain
        # equality that would otherwise miss the precision dual (opposite conventions,
        # equal values => physically opposite). It owns the verdict only on that concrete
        # evidence; otherwise it declines and the normal criterion decides.
        sign_convention = compare_sign_convention(pred, ref, context=context)
        if sign_convention is not None:
            if sign_convention.equivalent:
                return _apply_bridge_policy(
                    sign_convention,
                    pred=pred,
                    ref=ref,
                    contract=contract,
                    policy_mode=policy_mode,
                )
            return sign_convention
        strict = compare_same_object_kind(pred, ref, context=context)
        if strict.equivalent:
            return strict
        fallback = compare_label_family_fallback(pred, ref, context=context)
        if fallback is not None:
            return _apply_bridge_policy(
                fallback,
                pred=pred,
                ref=ref,
                contract=contract,
                policy_mode=policy_mode,
            )
        identical_text = _compare_identical_atomic_text(pred, ref)
        if identical_text is not None:
            return identical_text
        return strict

    if allow_cross_kind_identical_text:
        identical_text = _compare_identical_atomic_text(pred, ref)
        if identical_text is not None:
            return identical_text

    coerced = compare_different_object_kinds(pred, ref, context=context)
    if coerced is not None:
        return _apply_bridge_policy(
            coerced,
            pred=pred,
            ref=ref,
            contract=contract,
            policy_mode=policy_mode,
        )

    fallback = compare_label_family_fallback(pred, ref, context=context)
    if fallback is not None:
        return _apply_bridge_policy(
            fallback,
            pred=pred,
            ref=ref,
            contract=contract,
            policy_mode=policy_mode,
        )

    return AnswerComparison(
        equivalent=False,
        comparison_mode="object_kind_mismatch",
        diagnostics=(
            f"pred_kind={pred.object_kind.value}",
            f"ref_kind={ref.object_kind.value}",
        ),
    )


def _compare_identical_atomic_text(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
) -> AnswerComparison | None:
    """Rescue exact canonical-text matches regardless of auxiliary metadata.

    Compares the *repaired* surface: a stored ``canonical_text`` corrupted by latex2sympy
    singleton capture would otherwise make distinct answers identical here (``$I^2 R$`` and
    ``$I R$`` both store ``I*r``), and this shortcut accepts without consulting any criterion.
    """

    pred_text = effective_canonical_text(pred)
    ref_text = effective_canonical_text(ref)
    if pred_text and pred_text == ref_text:
        return AnswerComparison(True, "identical_text", surface_shortcut_used=True)
    return None


# Non-atomic comparison runs only when it provably reduces to atomic-vs-atomic element
# comparisons; anything else is TBD. By default TBD is a distinct non-equivalent sentinel so
# the hot path (batch eval / RL rewards) does not crash; flip this to raise instead.
STRICT_STRUCTURE_COMPARISON = False


def _structure_tbd(mode: str, *diagnostics: str) -> AnswerComparison:
    """Signal that a non-atomic comparison cannot yet be certified (TBD)."""

    if STRICT_STRUCTURE_COMPARISON:
        raise NotImplementedError(
            f"structure comparison not implemented ({mode}): {', '.join(diagnostics)}"
        )
    return AnswerComparison(False, "not_implemented", (mode,) + diagnostics)


def _children_all_atomic(*answers: PhysicsAnswerSemantics) -> bool:
    """Whether every child of every given answer is atomic."""

    return all(
        child.structure == AnswerStructure.ATOMIC
        for answer in answers
        for child in answer.children
    )


def _exact_element_key(answer: PhysicsAnswerSemantics) -> tuple:
    """A conservative exact-identity key for an atomic element (no tolerance).

    Numbers use their parsed value so ``1/2`` and ``0.5`` match exactly while ``1/3`` and
    ``0.3333`` do not (exact float equality, never a tolerance window). Everything else uses
    the repaired canonical text, so set and tuple elements are never merged on a surface that
    latex2sympy collapsed (``{Q/M, q/m}`` must stay two elements, not one).
    """

    if answer.object_kind in {
        AnswerObjectKind.NUMBER,
        AnswerObjectKind.PHYSICAL_QUANTITY,
    }:
        value = answer.numeric_value
        if value is None:
            value = parse_numeric_value(answer.numeric_text or answer.canonical_text)
        if value is not None:
            return ("num", value, answer.unit or "")
    return ("text", normalize_plain_text(effective_canonical_text(answer) or ""))


def _compare_ordered_children(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
    mode: str,
) -> AnswerComparison:
    """Compare structured children position by position (atomic elements only)."""

    if len(pred.children) != len(ref.children):
        return AnswerComparison(False, mode, ("different_child_count",))
    if not _children_all_atomic(pred, ref):
        return _structure_tbd(mode, "non_atomic_element")

    for index, (pred_child, ref_child) in enumerate(zip(pred.children, ref.children)):
        result = compare_protocol_answers(
            pred_child,
            ref_child,
            contract=contract,
            context=context,
            policy_mode=policy_mode,
            _validate_top_level=False,
        )
        if not result.equivalent:
            return AnswerComparison(
                False,
                mode,
                (f"child_{index}",) + result.diagnostics,
            )
    return AnswerComparison(True, mode)


def _compare_unordered_children(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
    mode: str,
) -> AnswerComparison:
    """Compare structured children as an order-insensitive multiset.

    Only the certain case is accepted: an *exact* multiset match of atomic elements (no
    tolerance, no ambiguous matching). A greedy/tolerant bijection is unsound under
    non-transitive tolerance (it accepts ``{1.0, 1.0}`` vs ``{1.0, 1.1}``), so every other
    case is TBD pending the sound matching algorithm (roadmap milestone; see STRUCTURE.md).
    """

    if len(pred.children) != len(ref.children):
        return AnswerComparison(False, mode, ("different_child_count",))
    if not _children_all_atomic(pred, ref):
        return _structure_tbd(mode, "non_atomic_element")

    pred_keys = sorted(_exact_element_key(child) for child in pred.children)
    ref_keys = sorted(_exact_element_key(child) for child in ref.children)
    if pred_keys == ref_keys:
        return AnswerComparison(True, mode)
    return _structure_tbd(mode, "inexact_unordered_match")


def _compare_per_part_children(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
    mode: str,
) -> AnswerComparison:
    """Compare children using inferred or explicit ``part_label`` alignment."""

    if len(pred.children) != len(ref.children):
        return AnswerComparison(False, mode, ("different_child_count",))
    if not _children_all_atomic(pred, ref):
        return _structure_tbd(mode, "non_atomic_part")

    pred_map = _part_child_map(pred, context=context)
    ref_map = _part_child_map(ref, context=context)
    # Require an explicit, aligned label set on both sides. The previous positional fallback
    # when labels did not align could compare mismatched parts, so it is retired to TBD.
    if pred_map is None or ref_map is None or set(pred_map) != set(ref_map):
        return _structure_tbd(mode, "part_labels_unaligned")

    for label in pred_map:
        result = compare_protocol_answers(
            pred_map[label],
            ref_map[label],
            contract=contract,
            context=context,
            policy_mode=policy_mode,
            _validate_top_level=False,
        )
        if not result.equivalent:
            return AnswerComparison(
                False, mode, (f"part={label}",) + result.diagnostics
            )
    return AnswerComparison(True, mode)


def _compare_interval(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
) -> AnswerComparison:
    """Compare interval endpoints after checking endpoint openness flags."""

    if (
        pred.interval_open_left != ref.interval_open_left
        or pred.interval_open_right != ref.interval_open_right
    ):
        return AnswerComparison(False, "interval", ("endpoint_openness_mismatch",))
    return _compare_ordered_children(
        pred,
        ref,
        context=context,
        contract=contract,
        policy_mode=policy_mode,
        mode="interval",
    )


def _reconcile_shaped_sign_convention(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
) -> AnswerComparison | None:
    """Reconcile a shaped (vector) pair under opposite stated frames; ``None`` to defer.

    Owns the verdict only when the question fixes no convention, both answers declare
    opposite (globally-reversed) directional conventions, and every cell is atomic. An exact
    component-wise negation accepts via the audited ``sign_convention`` bridge; a partial /
    non-negation rejects (the precision dual). Otherwise it declines so the existing frame
    gates (one-sided ⇒ TBD, genuinely-incompatible ⇒ mismatch, both-unset ⇒ per-cell) run.
    """

    if context.sign_convention or context.coordinate_frame:
        return None
    pred_convention = answer_directional_convention(pred)
    ref_convention = answer_directional_convention(ref)
    if not pred_convention or not ref_convention:
        return None
    if orientation_relation(pred_convention, ref_convention) != "opposite":
        return None
    if not _children_all_atomic(pred, ref):
        return None

    if vectors_exact_negation(pred, ref, context.tolerance):
        result = AnswerComparison(
            True,
            "sign_convention",
            ("global_sign_flip", f"kind={pred.structure.value}"),
        )
        return _apply_bridge_policy(
            result,
            pred=pred,
            ref=ref,
            contract=contract,
            policy_mode=policy_mode,
        )
    return AnswerComparison(
        False,
        "sign_convention",
        ("opposite_convention_not_global_negation",),
    )


def _compare_shaped(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
) -> AnswerComparison:
    """Compare vector-, matrix-, or tensor-like answers with shape metadata."""

    if pred.shape != ref.shape:
        return AnswerComparison(
            False,
            pred.structure.value,
            (f"shape_mismatch:{pred.shape}!={ref.shape}",),
        )

    # An unparsed shaped payload cannot be certified per-cell; the text-equality fallback is
    # retired to TBD.
    if not pred.children or not ref.children:
        return _structure_tbd(pred.structure.value, "unparsed_shaped_answer")

    # Sign-convention reconciliation (vectors): when the question fixes no convention and the
    # two answers declare opposite (globally-reversed) frames, an exact component-wise
    # negation is the same physical vector. This refines the both-set-incompatible branch
    # below — only the *opposite* sub-case reconciles; genuinely-incompatible frames still
    # reject, partial flips still reject.
    reconciled = _reconcile_shaped_sign_convention(
        pred,
        ref,
        context=context,
        contract=contract,
        policy_mode=policy_mode,
    )
    if reconciled is not None:
        return reconciled

    # Coordinate frame: both-unset ⇒ the problem's implicit shared frame (proceed); a
    # one-sided declaration is unresolved ⇒ TBD; both-set-incompatible is a real mismatch.
    pred_frame = pred.coordinate_frame or context.coordinate_frame
    ref_frame = ref.coordinate_frame or context.coordinate_frame
    if bool(pred_frame) != bool(ref_frame):
        return _structure_tbd(pred.structure.value, "coordinate_frame_unresolved")
    if (
        pred_frame
        and ref_frame
        and not _metadata_text_compatible(pred_frame, ref_frame)
    ):
        return AnswerComparison(
            False, pred.structure.value, ("coordinate_frame_mismatch",)
        )

    pred_sign = pred.sign_convention or context.sign_convention
    ref_sign = ref.sign_convention or context.sign_convention
    if bool(pred_sign) != bool(ref_sign):
        return _structure_tbd(pred.structure.value, "sign_convention_unresolved")
    if pred_sign and ref_sign and not _metadata_text_compatible(pred_sign, ref_sign):
        return AnswerComparison(
            False, pred.structure.value, ("sign_convention_mismatch",)
        )

    # Per-cell comparison via the ordered path, which itself gates non-atomic cells: a vector
    # of atomic cells is certified; a matrix/tensor (rows are non-atomic) falls to TBD.
    return _compare_ordered_children(
        pred,
        ref,
        context=context,
        contract=contract,
        policy_mode=policy_mode,
        mode=pred.structure.value,
    )


def _compare_piecewise(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
) -> AnswerComparison:
    """Compare piecewise answers case by case."""

    if len(pred.cases) != len(ref.cases):
        return AnswerComparison(False, "piecewise", ("different_case_count",))

    for index, (pred_case, ref_case) in enumerate(zip(pred.cases, ref.cases)):
        if any(
            part.structure != AnswerStructure.ATOMIC
            for part in (
                pred_case.expression,
                pred_case.condition,
                ref_case.expression,
                ref_case.condition,
            )
        ):
            return _structure_tbd("piecewise", f"non_atomic_case_{index}")
        expr_result = compare_protocol_answers(
            pred_case.expression,
            ref_case.expression,
            contract=contract,
            context=context,
            policy_mode=policy_mode,
            _validate_top_level=False,
        )
        cond_result = compare_protocol_answers(
            pred_case.condition,
            ref_case.condition,
            contract=contract,
            context=context,
            policy_mode=policy_mode,
            _validate_top_level=False,
        )
        if not expr_result.equivalent or not cond_result.equivalent:
            return AnswerComparison(False, "piecewise", (f"case_{index}_mismatch",))
    return AnswerComparison(True, "piecewise")


_COMPARISON_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "in",
        "of",
        "shown",
        "the",
        "to",
        "with",
    }
)


def _repair_answer_for_comparison(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> PhysicsAnswerSemantics:
    """Repair lightly malformed answers before comparison dispatch."""

    from ..normalization import enrich_answer_quantity_views

    repaired = enrich_answer_quantity_views(answer, context=context)
    repaired = _hydrate_structured_answer(repaired, context=context)
    repaired = _backfill_subject_to(repaired, context=context)
    # Collapse structural degeneracies LAST (dominates the tuple→vector promotion above), so
    # a degenerate wrapper reaches atomic before the structure gate; if it does, fall through
    # to the atomic repair.
    repaired = canonicalize_structure(repaired, context=context)
    if repaired.structure != AnswerStructure.ATOMIC:
        return repaired
    return _repair_atomic_answer(repaired, context=context)


def _hydrate_structured_answer(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> PhysicsAnswerSemantics:
    """Reparse structured answers whose child payloads were not persisted."""

    if answer.structure == AnswerStructure.ATOMIC or answer.children or answer.cases:
        return answer

    from ..normalization import normalize_physics_answer

    for source_text in reparse_sources(answer):
        reparsed = normalize_physics_answer(source_text, context=context)
        if reparsed.structure != answer.structure and answer.structure in {
            AnswerStructure.VECTOR,
            AnswerStructure.MATRIX,
            AnswerStructure.TENSOR,
        }:
            reparsed = _coerce_tuple_shaped_answer(reparsed) or reparsed
        if reparsed.structure != answer.structure or (
            not reparsed.children and not reparsed.cases
        ):
            continue

        return reparsed.model_copy(
            update={
                "canonical_text": (
                    answer.canonical_text
                    if answer.object_kind
                    in {AnswerObjectKind.EXPRESSION, AnswerObjectKind.RELATION}
                    and answer.canonical_text
                    else reparsed.canonical_text
                ),
                "canonical_latex": answer.canonical_latex or reparsed.canonical_latex,
                "raw_text": answer.raw_text or reparsed.raw_text,
                "target_variable": answer.target_variable or reparsed.target_variable,
                "coordinate_frame": (
                    answer.coordinate_frame or reparsed.coordinate_frame
                ),
                "sign_convention": (answer.sign_convention or reparsed.sign_convention),
                "provenance": dict(answer.provenance) or dict(reparsed.provenance),
                "diagnostics": answer.diagnostics or reparsed.diagnostics,
                "quantity_view": answer.quantity_view or reparsed.quantity_view,
                "subject_to": answer.subject_to or reparsed.subject_to,
            }
        )

    return answer


def _coerce_tuple_shaped_answer(
    answer: PhysicsAnswerSemantics,
) -> PhysicsAnswerSemantics | None:
    """Promote tuple-shaped payloads into vector/matrix/tensor semantics."""

    if answer.structure != AnswerStructure.TUPLE:
        return (
            answer
            if answer.structure
            in {
                AnswerStructure.ATOMIC,
                AnswerStructure.VECTOR,
                AnswerStructure.MATRIX,
                AnswerStructure.TENSOR,
            }
            else None
        )

    coerced_children: list[PhysicsAnswerSemantics] = []
    for child in answer.children:
        coerced_child = _coerce_tuple_shaped_answer(child)
        if coerced_child is None:
            return None
        coerced_children.append(coerced_child)

    child_tuple = tuple(coerced_children)
    if child_tuple and all(
        child.structure == AnswerStructure.ATOMIC for child in child_tuple
    ):
        return answer.model_copy(
            update={
                "structure": AnswerStructure.VECTOR,
                "shape": (len(child_tuple),),
                "children": child_tuple,
            }
        )

    if not child_tuple or not all(
        child.structure
        in {
            AnswerStructure.VECTOR,
            AnswerStructure.MATRIX,
            AnswerStructure.TENSOR,
        }
        for child in child_tuple
    ):
        return None

    child_shapes = {child.shape for child in child_tuple}
    if len(child_shapes) != 1:
        return None

    inner_shape = next(iter(child_shapes))
    if len(inner_shape) == 1:
        structure = AnswerStructure.MATRIX
    else:
        structure = AnswerStructure.TENSOR

    return answer.model_copy(
        update={
            "structure": structure,
            "shape": (len(child_tuple),) + inner_shape,
            "children": child_tuple,
        }
    )


def _repair_atomic_answer(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> PhysicsAnswerSemantics:
    """Reparse atomic answers when persisted fields look incomplete or mislabeled."""

    if (
        answer.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY
        and answer.numeric_value is not None
        and answer.unit
    ):
        return answer

    if answer.object_kind not in {
        AnswerObjectKind.NUMBER,
        AnswerObjectKind.PHYSICAL_QUANTITY,
        AnswerObjectKind.EXPRESSION,
        AnswerObjectKind.RELATION,
        AnswerObjectKind.QUALITATIVE_LABEL,
    }:
        return answer

    source_texts = list(reparse_sources(answer))
    if not source_texts:
        return answer
    if answer.object_kind == AnswerObjectKind.QUALITATIVE_LABEL and not any(
        _is_symbol_like_salvage_text(source_text, context=context)
        for source_text in source_texts
    ):
        return answer

    from ..normalization import normalize_physics_answer

    for source_text in source_texts:
        reparsed = normalize_physics_answer(source_text, context=context)
        if reparsed.structure != AnswerStructure.ATOMIC:
            continue
        if reparsed.object_kind not in {
            AnswerObjectKind.NUMBER,
            AnswerObjectKind.PHYSICAL_QUANTITY,
            AnswerObjectKind.EXPRESSION,
            AnswerObjectKind.RELATION,
        }:
            continue
        update: dict[str, Any] = {
            "canonical_text": (
                reparsed.canonical_text
                if reparsed.subject_to and not answer.subject_to
                else (
                    answer.canonical_text
                    if answer.object_kind
                    in {AnswerObjectKind.EXPRESSION, AnswerObjectKind.RELATION}
                    and answer.canonical_text
                    else reparsed.canonical_text
                )
            ),
            "raw_text": answer.raw_text or reparsed.raw_text,
            "canonical_latex": (
                reparsed.canonical_latex
                if reparsed.subject_to and not answer.subject_to
                else answer.canonical_latex or reparsed.canonical_latex
            ),
            "target_variable": answer.target_variable or reparsed.target_variable,
            "dimension": answer.dimension or reparsed.dimension,
            "provenance": dict(answer.provenance) or dict(reparsed.provenance),
            "diagnostics": answer.diagnostics or reparsed.diagnostics,
            "subject_to": answer.subject_to or reparsed.subject_to,
            # Preserve the answer's stated directional convention through the reparse so the
            # sign-convention lane can still read it (mirrors the structured reparse merge).
            "coordinate_frame": answer.coordinate_frame or reparsed.coordinate_frame,
            "sign_convention": answer.sign_convention or reparsed.sign_convention,
        }
        if reparsed.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY:
            update.update(
                {
                    "unit": reparsed.unit,
                    "quantity_view": reparsed.quantity_view,
                }
            )
        else:
            update.update(
                {
                    "unit": answer.unit or reparsed.unit,
                    "quantity_view": answer.quantity_view or reparsed.quantity_view,
                }
            )
        return reparsed.model_copy(update=update)

    return answer


def _backfill_subject_to(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> PhysicsAnswerSemantics:
    """Populate legacy side conditions from reparsed text surfaces when missing."""

    if answer.subject_to:
        return answer

    from ..normalization import normalize_physics_answer

    for source_text in available_texts(
        answer.raw_text,
        answer.canonical_latex,
        answer.canonical_text,
    ):
        reparsed = normalize_physics_answer(source_text, context=context)
        if reparsed.structure != answer.structure or not reparsed.subject_to:
            continue

        update: dict[str, Any] = {
            "canonical_text": reparsed.canonical_text,
            "raw_text": answer.raw_text or reparsed.raw_text,
            "canonical_latex": reparsed.canonical_latex,
            "target_variable": answer.target_variable or reparsed.target_variable,
            "provenance": dict(answer.provenance) or dict(reparsed.provenance),
            "diagnostics": answer.diagnostics or reparsed.diagnostics,
            "subject_to": reparsed.subject_to,
        }
        if answer.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY:
            update["unit"] = answer.unit or reparsed.unit
            update["quantity_view"] = answer.quantity_view or reparsed.quantity_view
        return answer.model_copy(update=update)

    return answer


def _subject_to_context(context: PhysicsQuestionSemantics) -> PhysicsQuestionSemantics:
    """Derive the comparison context used for one subject-to conjunction."""

    return context.model_copy(
        update={
            "question_symbolic_mode": QuestionSymbolicMode.RELATION,
            "required_parts": (),
        }
    )


def _compare_subject_to(
    result: AnswerComparison,
    *,
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    context: PhysicsQuestionSemantics,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
) -> AnswerComparison:
    """Enforce order-insensitive equivalence for conjunctive side conditions."""

    if not result.equivalent:
        return result
    if not pred.subject_to and not ref.subject_to:
        return result
    if len(pred.subject_to) != len(ref.subject_to):
        return AnswerComparison(
            False,
            result.comparison_mode,
            result.diagnostics + ("subject_to_count_mismatch",),
        )

    subject_context = _subject_to_context(context)
    unused = list(pred.subject_to)
    for index, ref_constraint in enumerate(ref.subject_to):
        match_index = None
        for candidate_index, pred_constraint in enumerate(unused):
            constraint_result = compare_protocol_answers(
                pred_constraint,
                ref_constraint,
                contract=contract,
                context=subject_context,
                policy_mode=policy_mode,
                _validate_top_level=False,
            )
            if constraint_result.equivalent:
                match_index = candidate_index
                break
        if match_index is None:
            return AnswerComparison(
                False,
                result.comparison_mode,
                result.diagnostics + (f"subject_to_unmatched:{index}",),
            )
        unused.pop(match_index)

    return result


_SYMBOL_LIKE_TEXT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\\]*$")
_MATH_LIKE_SALVAGE_RE = re.compile(r"[=<>+\-*/^()\[\]{}]")


def _is_symbol_like_salvage_text(
    text: str,
    *,
    context: PhysicsQuestionSemantics,
) -> bool:
    """Return whether a qualitative label is safe to reinterpret as symbolic."""

    stripped = text.strip()
    if not stripped:
        return False
    normalized = normalize_plain_text(stripped)
    if not normalized:
        return False
    if canonicalize_qualitative_label(stripped) != normalized:
        return False
    if _MATH_LIKE_SALVAGE_RE.search(stripped):
        return True

    compact = stripped.replace(" ", "")
    if not _SYMBOL_LIKE_TEXT_RE.fullmatch(compact):
        return False
    if len(stripped.split()) > 1:
        return False

    normalized_compact = normalize_plain_text(stripped).replace(" ", "")
    expected_symbols = _expected_symbol_tokens(context)
    if compact in expected_symbols or normalized_compact in expected_symbols:
        return True

    if "\\" in compact or any(char.isdigit() for char in compact):
        return True
    if any(char.isupper() for char in compact):
        return True
    return len(normalized_compact) <= 2


def _expected_symbol_tokens(context: PhysicsQuestionSemantics) -> set[str]:
    """Collect target-variable and alias tokens expected by the question."""

    expected: set[str] = set()

    def _add(text: str | None) -> None:
        if not text:
            return
        stripped = text.strip()
        normalized = normalize_plain_text(stripped).replace(" ", "")
        compact = stripped.replace(" ", "")
        if stripped:
            expected.add(stripped)
        if compact:
            expected.add(compact)
        if normalized:
            expected.add(normalized)

    _add(context.target_variable)
    for alias, canonical in context_symbol_alias_map(context).items():
        _add(alias)
        _add(canonical)
    return expected


def _metadata_text_compatible(left: str, right: str) -> bool:
    """Compare metadata strings by exact match or token containment."""

    if left == right:
        return True

    left_tokens = _metadata_tokens(left)
    right_tokens = _metadata_tokens(right)
    if not left_tokens or not right_tokens:
        return normalize_plain_text(left) == normalize_plain_text(right)

    smaller, larger = (
        (left_tokens, right_tokens)
        if len(left_tokens) <= len(right_tokens)
        else (right_tokens, left_tokens)
    )
    return smaller <= larger


def _metadata_tokens(text: str) -> set[str]:
    """Tokenize metadata text into a comparison-friendly normalized set."""

    normalized = (
        normalize_plain_text(text).replace("axes", "axis").replace("centre", "center")
    )
    return {
        token
        for token in normalized.split()
        if token and token not in _COMPARISON_STOPWORDS
    }


def _part_child_map(
    answer: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> dict[str, PhysicsAnswerSemantics] | None:
    """Map structured children by part label when per-part comparison is viable."""

    if not answer.children:
        return None

    surfaces = tuple(
        child.raw_text or child.canonical_text for child in answer.children
    )
    inferred_labels = infer_multi_part_part_labels(surfaces, context.required_parts)
    labels: list[str | None] = []
    for index, child in enumerate(answer.children):
        label = child.part_label or inferred_labels[index]
        if label is None and len(context.required_parts) == len(answer.children):
            label = context.required_parts[index]
        labels.append(label)

    if any(label is None for label in labels):
        return None

    canonical_labels = tuple(str(label) for label in labels if label is not None)
    if len(set(canonical_labels)) != len(canonical_labels):
        return None

    preferred_order = tuple(
        part for part in context.required_parts if part in canonical_labels
    )
    order = preferred_order or canonical_labels
    child_by_label = {
        str(label): child
        for label, child in zip(labels, answer.children)
        if label is not None
    }
    return {label: child_by_label[label] for label in order if label in child_by_label}


def _resolve_comparison_contract(
    *,
    contract: PhysicsEvaluationContract | dict | None,
    context: PhysicsQuestionSemantics,
    ref_answer: PhysicsAnswerSemantics,
) -> PhysicsEvaluationContract:
    """Resolve a comparison contract from explicit input or build one on the fly."""

    if contract is not None:
        return coerce_evaluation_contract(contract)
    return build_evaluation_contract(
        question_semantics=context,
        reference_answer_semantics=ref_answer,
    )


def _apply_bridge_policy(
    result: AnswerComparison,
    *,
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
) -> AnswerComparison:
    """Accept or reject a bridged comparison under the active evaluation policy."""

    if not result.equivalent:
        return result

    spec = bridge_spec_for(result.comparison_mode)
    if spec is None:
        return result

    if policy_mode == ComparisonPolicyMode.PERMISSIVE:
        return result.model_copy(
            update={
                "bridge_id": spec.bridge_id,
                "bridge_tier": spec.tier,
                "bridge_evidence": _bridge_evidence(
                    spec.bridge_id, pred, ref, contract
                ),
            }
        )

    if (
        contract.enabled_bridge_ids
        and spec.bridge_id not in contract.enabled_bridge_ids
    ):
        return AnswerComparison(
            equivalent=False,
            comparison_mode="bridge_blocked",
            diagnostics=(f"bridge_not_enabled:{spec.bridge_id}",),
        )
    if not bridge_enabled_for_policy(
        spec,
        contract=contract,
        policy_mode=policy_mode,
    ):
        return AnswerComparison(
            equivalent=False,
            comparison_mode="bridge_blocked",
            diagnostics=(f"bridge_disallowed_by_policy:{spec.bridge_id}",),
        )
    if not spec.predicate(pred, ref, contract):
        return AnswerComparison(
            equivalent=False,
            comparison_mode="bridge_blocked",
            diagnostics=(f"bridge_precondition_failed:{spec.bridge_id}",),
        )

    return result.model_copy(
        update={
            "bridge_id": spec.bridge_id,
            "bridge_tier": spec.tier,
            "bridge_evidence": _bridge_evidence(spec.bridge_id, pred, ref, contract),
        }
    )


def _bridge_evidence(
    bridge_id: str,
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    contract: PhysicsEvaluationContract,
) -> dict[str, str]:
    """Build a compact evidence payload for one accepted bridge."""

    evidence = {
        "pred_kind": pred.object_kind.value,
        "ref_kind": ref.object_kind.value,
        "expected_object_kind": contract.expected_object_kind.value,
        "expected_structure": contract.expected_structure.value,
    }
    if contract.target_variable:
        evidence["target_variable"] = contract.target_variable
    if contract.question_semantics.question_unit_policy.value != "not_applicable":
        evidence["question_unit_policy"] = (
            contract.question_semantics.question_unit_policy.value
        )
    if contract.question_semantics.question_symbolic_mode.value != "either":
        evidence["question_symbolic_mode"] = (
            contract.question_semantics.question_symbolic_mode.value
        )
    if (
        bridge_id in {"choice", "terminal_polarity_choice"}
        and contract.question_semantics.choice_space
    ):
        evidence["choice_space"] = ",".join(contract.question_semantics.choice_space)
    if bridge_id == "sign_convention":
        pred_convention = answer_directional_convention(pred)
        ref_convention = answer_directional_convention(ref)
        if pred_convention:
            evidence["pred_convention"] = pred_convention
        if ref_convention:
            evidence["ref_convention"] = ref_convention
        evidence["orientation"] = "opposite"
        evidence["reconciliation"] = "global_-1"
    return evidence


def _finalize_comparison(
    result: AnswerComparison,
    *,
    validation_status: ContractValidationStatus | None,
    policy_mode: ComparisonPolicyMode,
) -> AnswerComparison:
    """Attach policy and validation metadata to a comparison result."""

    update: dict[str, object] = {"policy_mode": policy_mode}
    if validation_status is not None:
        update["validation_status"] = validation_status
    return result.model_copy(update=update)


def _finalize_result(
    result: AnswerComparison,
    *,
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    context: PhysicsQuestionSemantics,
    contract: PhysicsEvaluationContract,
    validation_status: ContractValidationStatus | None,
    policy_mode: ComparisonPolicyMode,
) -> AnswerComparison:
    """Attach side-condition checks before policy and validation metadata."""

    return _finalize_comparison(
        _compare_subject_to(
            result,
            pred=pred,
            ref=ref,
            context=context,
            contract=contract,
            policy_mode=policy_mode,
        ),
        validation_status=validation_status,
        policy_mode=policy_mode,
    )


compare_physics_answers = compare_protocol_answers
