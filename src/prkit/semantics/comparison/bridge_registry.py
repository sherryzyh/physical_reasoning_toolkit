"""Bridge specifications and policy helpers for semantics comparison."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..schema import (
    AnswerObjectKind,
    AnswerStructure,
    BridgeTier,
    ComparisonPolicyMode,
    PhysicsAnswerSemantics,
    PhysicsEvaluationContract,
)

BridgePredicate = Callable[
    [PhysicsAnswerSemantics, PhysicsAnswerSemantics, PhysicsEvaluationContract], bool
]


@dataclass(frozen=True)
class BridgeSpec:
    """Static description of one comparison bridge."""

    bridge_id: str
    tier: BridgeTier
    predicate: BridgePredicate
    description: str


def _is_symbolic_bridge(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    contract: PhysicsEvaluationContract,
) -> bool:
    del pred, ref
    return contract.question_semantics.question_symbolic_mode.value in {
        "expression",
        "either",
    }


def _is_relation_rhs_bridge(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    contract: PhysicsEvaluationContract,
) -> bool:
    del pred, ref
    return bool(contract.target_variable)


def _has_fixed_question_unit(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    contract: PhysicsEvaluationContract,
) -> bool:
    del pred, ref
    question = contract.question_semantics
    return (
        question.question_unit_policy.value == "optional_if_question_fixed_unit"
        and bool(question.question_unit)
    )


def _has_choice_space(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    contract: PhysicsEvaluationContract,
) -> bool:
    del pred, ref
    return bool(contract.question_semantics.choice_space)


def _sign_convention_reconcilable(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    contract: PhysicsEvaluationContract,
) -> bool:
    del pred, ref
    question = contract.question_semantics
    return question.sign_convention is None and question.coordinate_frame is None


def _looks_like_change_question(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    contract: PhysicsEvaluationContract,
) -> bool:
    del pred, ref
    text = (contract.question_text or "").lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "change",
            "unchanged",
            "remain",
            "increase",
            "decrease",
            "same",
        )
    )


def _always(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    contract: PhysicsEvaluationContract,
) -> bool:
    del pred, ref, contract
    return True


BRIDGE_REGISTRY: dict[str, BridgeSpec] = {
    "quantity_to_number": BridgeSpec(
        bridge_id="quantity_to_number",
        tier=BridgeTier.TIER2,
        predicate=_has_fixed_question_unit,
        description="Allow bare numbers against unitful references only when the question fixes the output unit.",
    ),
    "expression_to_number": BridgeSpec(
        bridge_id="expression_to_number",
        tier=BridgeTier.TIER1,
        predicate=_always,
        description="Compare numerically evaluable expressions against scalar numbers.",
    ),
    "expression_quantity": BridgeSpec(
        bridge_id="expression_quantity",
        tier=BridgeTier.TIER2,
        predicate=_has_fixed_question_unit,
        description="Compare numerically evaluable expressions against unitful quantities when the unit is fixed by the question.",
    ),
    "relation_to_expression": BridgeSpec(
        bridge_id="relation_to_expression",
        tier=BridgeTier.TIER1,
        predicate=_is_symbolic_bridge,
        description="Reduce relation answers to the requested symbolic expression when the question allows expressions.",
    ),
    "relation_rhs": BridgeSpec(
        bridge_id="relation_rhs",
        tier=BridgeTier.TIER1,
        predicate=_is_relation_rhs_bridge,
        description="Reduce relation answers to their right-hand side when the target variable is known.",
    ),
    "choice": BridgeSpec(
        bridge_id="choice",
        tier=BridgeTier.TIER2,
        predicate=_has_choice_space,
        description="Map discrete answer labels through the question's choice space.",
    ),
    "terminal_polarity_choice": BridgeSpec(
        bridge_id="terminal_polarity_choice",
        tier=BridgeTier.TIER2,
        predicate=_has_choice_space,
        description="Map terminal-polarity language onto discrete choice labels when the question is choice-based.",
    ),
    "terminal_polarity": BridgeSpec(
        bridge_id="terminal_polarity",
        tier=BridgeTier.TIER2,
        predicate=_always,
        description="Normalize terminal-polarity language into a discrete directional label family.",
    ),
    "label_family_fallback": BridgeSpec(
        bridge_id="label_family_fallback",
        tier=BridgeTier.TIER3,
        predicate=_always,
        description="Fallback overlap check inside bounded sign/direction or qualitative label families.",
    ),
    "relation_to_qualitative_label": BridgeSpec(
        bridge_id="relation_to_qualitative_label",
        tier=BridgeTier.TIER3,
        predicate=_looks_like_change_question,
        description="Interpret a narrow sameness-style relation as a qualitative outcome only for change questions.",
    ),
    "qualitative_zero": BridgeSpec(
        bridge_id="qualitative_zero",
        tier=BridgeTier.TIER3,
        predicate=_looks_like_change_question,
        description="Treat no-change language as zero only for change-oriented questions.",
    ),
    "sign_convention": BridgeSpec(
        bridge_id="sign_convention",
        tier=BridgeTier.TIER2,
        predicate=_sign_convention_reconcilable,
        description="Reconcile a global sign flip between directional answers that declare "
        "opposite conventions, only when the question fixes none.",
    ),
    "implicit_unit_alias": BridgeSpec(
        bridge_id="implicit_unit_alias",
        tier=BridgeTier.TIER3,
        predicate=_always,
        description="Rescue a bare '<number> <token>' answer against a unitful reference when "
        "the token is a curated case-insensitive alias of the reference's unit.",
    ),
}


def bridge_spec_for(mode: str) -> BridgeSpec | None:
    """Return the bridge spec for one comparison mode when it exists."""

    return BRIDGE_REGISTRY.get(mode)


def bridge_enabled_for_policy(
    spec: BridgeSpec,
    *,
    contract: PhysicsEvaluationContract,
    policy_mode: ComparisonPolicyMode,
) -> bool:
    """Return whether a bridge tier is enabled under the policy and contract."""

    if policy_mode == ComparisonPolicyMode.PERMISSIVE:
        return True
    if policy_mode == ComparisonPolicyMode.STRICT:
        return False
    return spec.tier in contract.enabled_bridge_tiers


def _bridge_candidate_ids_for_atomic_kinds(
    pred_kind: AnswerObjectKind,
    ref_kind: AnswerObjectKind,
) -> tuple[str, ...]:
    """Return bridge ids that could plausibly connect two atomic answer kinds."""

    kinds = {pred_kind, ref_kind}
    candidates: list[str] = []
    if pred_kind == ref_kind and pred_kind in {
        AnswerObjectKind.NUMBER,
        AnswerObjectKind.PHYSICAL_QUANTITY,
        AnswerObjectKind.SIGN_DIRECTION,
    }:
        candidates.append("sign_convention")
    if kinds == {AnswerObjectKind.NUMBER, AnswerObjectKind.PHYSICAL_QUANTITY}:
        candidates.append("quantity_to_number")
    if kinds == {AnswerObjectKind.NUMBER, AnswerObjectKind.EXPRESSION}:
        candidates.append("expression_to_number")
    if kinds == {AnswerObjectKind.EXPRESSION, AnswerObjectKind.PHYSICAL_QUANTITY}:
        candidates.append("expression_quantity")
    if kinds == {AnswerObjectKind.RELATION, AnswerObjectKind.EXPRESSION}:
        candidates.append("relation_to_expression")
    if AnswerObjectKind.RELATION in kinds and kinds <= {
        AnswerObjectKind.RELATION,
        AnswerObjectKind.NUMBER,
        AnswerObjectKind.PHYSICAL_QUANTITY,
    }:
        candidates.append("relation_rhs")
    if kinds == {AnswerObjectKind.CHOICE, AnswerObjectKind.QUALITATIVE_LABEL}:
        candidates.append("choice")
    if kinds == {AnswerObjectKind.CHOICE, AnswerObjectKind.SIGN_DIRECTION}:
        candidates.extend(("choice", "terminal_polarity_choice"))
    if kinds == {AnswerObjectKind.QUALITATIVE_LABEL, AnswerObjectKind.SIGN_DIRECTION}:
        candidates.extend(("terminal_polarity", "label_family_fallback"))
    if kinds in (
        {AnswerObjectKind.NUMBER, AnswerObjectKind.QUALITATIVE_LABEL},
        {AnswerObjectKind.PHYSICAL_QUANTITY, AnswerObjectKind.QUALITATIVE_LABEL},
    ):
        candidates.append("qualitative_zero")
    if kinds == {
        AnswerObjectKind.PHYSICAL_QUANTITY,
        AnswerObjectKind.DESCRIPTIVE_TEXT,
    }:
        candidates.append("implicit_unit_alias")
    return tuple(dict.fromkeys(candidates))


def bridge_candidate_ids_for_answers(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
) -> tuple[str, ...]:
    """Return bridge ids that can plausibly connect the concrete answer shapes."""

    if (
        pred.structure != AnswerStructure.ATOMIC
        or ref.structure != AnswerStructure.ATOMIC
    ):
        return ()
    return _bridge_candidate_ids_for_atomic_kinds(pred.object_kind, ref.object_kind)
