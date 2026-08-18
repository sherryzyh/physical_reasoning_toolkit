"""Tests for the ``prkit.verify`` light-import facade (verify)."""

from __future__ import annotations

import pytest

from prkit.core.verdict import Verdict
from prkit.semantics import PhysicsAnswerSemantics, extract_prediction_answer_semantics
from prkit.verify import verify


class TestVerify:
    def test_returns_canonical_verdict(self):
        v = verify("3 m/s", "3 m/s")
        assert isinstance(v, Verdict)
        assert v.correct is True
        assert v.score == 1.0

    def test_unit_suffix_normalization_handled(self):
        # math-verify mishandles units; the squared-suffix variants normalize to
        # the same unit, so the dimensional check passes (units_ok True) even
        # though 9.8 != 9.81 numerically.
        v = verify("9.8 m/s^2", "9.8 m/s²")
        assert v.correct is True
        assert v.units_ok is True

    def test_unit_mismatch_is_not_equivalent(self):
        v = verify("3 m/s", "3 m")
        assert v.correct is False

    def test_numeric_tolerance_mismatch(self):
        v = verify("3 m/s", "5 m/s")
        assert v.correct is False
        assert v.numeric_within_tol is False

    def test_symbolic_commutativity_equivalent(self):
        # Cases math-verify mishandles: symbolic equivalence under reordering.
        v = verify("v = a t", "v = t a")
        assert v.correct is True
        assert v.symbolic_equiv is True

    def test_symbolic_relation_rearrangement_equivalent(self):
        # Equalities equivalent only after algebraic rearrangement across "=".
        v = verify("F = m a", "a = F/m")
        assert v.correct is True
        assert v.symbolic_equiv is True
        assert v.comparison_mode == "relation"

    def test_compact_product_with_coefficient_equivalent(self):
        # The reference states a product compactly; the prediction commutes one factor.
        # Both sides must survive preprocessing intact for SymPy to see them as equal.
        v = verify(r"\gamma B\left(2JS+\gamma B\right)", r"\gamma B(\gamma B+2JS)")
        assert v.correct is True
        assert v.symbolic_equiv is True
        assert v.comparison_mode == "expression"

    def test_thin_space_and_function_argument_equivalent(self):
        # Typographic spacing and a braced function argument are surface, not meaning.
        v = verify(r"\frac{V_0}{\ln(x)}", r"V_0\,\frac{1}{\ln x}")
        assert v.correct is True
        assert v.symbolic_equiv is True

    def test_number_vs_fraction_equivalent(self):
        v = verify("0.5", "1/2")
        assert v.correct is True

    def test_gold_pred_argument_order(self):
        # verify(gold, pred): the prediction surface is what gets extracted.
        v = verify("3 m/s", "4 m/s")
        assert v.extracted_answer is not None
        assert "4" in v.extracted_answer

    def test_partial_credit_true_returns_graded_verdict(self):
        # verify(gold, pred): a one-coefficient near-miss earns graded credit.
        v = verify(
            "2*m*g + 2*m*v0**2/l",
            "2*m*g + 4*m*v0**2/l",
            partial_credit=True,
        )
        assert isinstance(v, Verdict)
        assert 0.0 < v.score < 1.0
        assert v.partial_credit == v.score
        assert v.correct is False

    def test_partial_credit_exact_match_full_credit(self):
        v = verify("3 m/s", "3 m/s", partial_credit=True)
        assert v.partial_credit == 1.0
        assert v.correct is True

    def test_binary_path_leaves_partial_credit_and_rationale_none(self):
        # The default (binary) deterministic engine never emits a graded
        # partial-credit signal or a natural-language rationale; those stay None
        # so a near-miss reads as a hard 0.0, not a soft score. (X1 contract.)
        v = verify("2*m*g + 2*m*v0**2/l", "2*m*g + 4*m*v0**2/l")
        assert v.correct is False
        assert v.score == 0.0
        assert v.partial_credit is None
        assert v.rationale is None

    def test_partial_credit_unknown_unit_policy_raises(self):
        with pytest.raises(ValueError, match="unit_policy"):
            verify("a", "b", partial_credit=True, unit_policy="bogus")

    def test_unknown_unit_policy_raises(self):
        with pytest.raises(ValueError, match="unit_policy"):
            verify("a", "b", unit_policy="bogus")

    @pytest.mark.parametrize("policy", ["strict", "audited", "permissive"])
    def test_recognized_unit_policies_accepted(self, policy):
        v = verify("3 m/s", "3 m/s", unit_policy=policy)
        assert isinstance(v, Verdict)

    def test_tolerance_passthrough(self):
        # tolerance is a relative threshold; a generous value flips a near-miss
        # that the default (precision-driven) comparison rejects.
        strict = verify("100", "101")
        loose = verify("100", "101", tolerance=0.05)
        assert strict.correct is False
        assert loose.correct is True


def _positive_x_context():
    """A ``q_ref`` declaring ``x`` positive, unlocking domain-gated symbolic identities.

    Built lazily inside the test so the module-level import stays light (the schema
    types live behind the heavier ``prkit.semantics`` import).
    """
    from prkit.semantics import PhysicsQuestionSemantics
    from prkit.semantics.schema.enums import SymbolAssumption
    from prkit.semantics.schema.models import PhysicsSymbolAssumptionSemantics

    return PhysicsQuestionSemantics(
        symbol_assumptions=(
            PhysicsSymbolAssumptionSemantics(
                symbol="x", assumption=SymbolAssumption.POSITIVE
            ),
        )
    )


class TestVerifyThreadsQuestionContext:
    """WS C: ``verify(context=q_ref)`` reaches the judgement and can change a verdict.

    ``log(x**2) == 2*log(x)`` holds only for ``x > 0``. Under the empty default
    context (``context=None``) the engine cannot assume positivity and rejects the
    pair; supplying a ``q_ref`` that declares ``x`` positive unlocks the
    domain-gated symbolic accept. This is the wiring WS C closes — previously
    ``verify`` passed ``context=None`` unconditionally so a rich ``q_ref`` was inert.
    """

    GOLD = "log(x**2)"
    PRED = "2*log(x)"

    def test_default_context_rejects_domain_gated_identity(self):
        v = verify(self.GOLD, self.PRED)
        assert v.correct is False

    def test_supplying_q_ref_unlocks_domain_gated_accept(self):
        v = verify(self.GOLD, self.PRED, context=_positive_x_context())
        assert v.correct is True
        assert v.symbolic_equiv is True

    def test_verdict_changes_with_vs_without_context(self):
        without = verify(self.GOLD, self.PRED)
        with_q_ref = verify(self.GOLD, self.PRED, context=_positive_x_context())
        assert without.correct != with_q_ref.correct

    def test_dict_context_accepted(self):
        v = verify(
            self.GOLD,
            self.PRED,
            context={"symbol_assumptions": [{"symbol": "x", "assumption": "positive"}]},
        )
        assert v.correct is True

    def test_reference_artifact_context_unwrapped_to_question_semantics(self):
        # Anything exposing ``.question_semantics`` (e.g. a ReferenceSemanticsArtifact)
        # is duck-typed and unwrapped to its q_ref — without importing the heavy
        # inference artifact type into the light verify facade.
        class _ArtifactLike:
            question_semantics = _positive_x_context()

        v = verify(self.GOLD, self.PRED, context=_ArtifactLike())
        assert v.correct is True

    def test_none_context_preserves_default_behavior(self):
        explicit_none = verify("3 m/s", "3 m/s", context=None)
        implicit = verify("3 m/s", "3 m/s")
        assert explicit_none.correct == implicit.correct is True

    def test_partial_credit_path_also_threads_context(self):
        # The graded path must accept and thread q_ref too (it reaches the same
        # context-coercion plumbing); supplying it must not error and yields a
        # valid graded Verdict. The verdict-change assertion lives on the binary
        # path above, where the domain gate is decisive.
        v = verify(
            self.GOLD, self.PRED, partial_credit=True, context=_positive_x_context()
        )
        assert isinstance(v, Verdict)
        assert v.partial_credit == v.score


class TestExtractPredictionAnswerSemantics:
    """The deterministic extractor replaces the removed ``prkit.verify.parse``."""

    def test_returns_physics_answer_semantics(self):
        parsed = extract_prediction_answer_semantics("9.8 m/s^2")
        assert isinstance(parsed, PhysicsAnswerSemantics)
        assert parsed.unit == "m/s^2"


def _answer(payload):
    """Coerce a protocol-answer mapping into ``PhysicsAnswerSemantics`` for ``verify``."""
    from prkit.semantics import coerce_protocol_answer

    return coerce_protocol_answer(payload)


def _directional_quantity(text, value, unit, convention=None):
    payload = {
        "object_kind": "physical_quantity",
        "canonical_text": text,
        "numeric_value": value,
        "numeric_text": str(value),
        "unit": unit,
    }
    if convention is not None:
        payload["coordinate_frame"] = convention
    return _answer(payload)


def _directional_vector(components, convention=None):
    payload = {
        "object_kind": "number",
        "structure": "vector",
        "canonical_text": "(" + ", ".join(str(c) for c in components) + ")",
        "shape": [len(components)],
        "children": [
            {
                "object_kind": "number",
                "structure": "atomic",
                "canonical_text": str(c),
                "numeric_value": float(c),
                "numeric_text": str(c),
            }
            for c in components
        ],
    }
    if convention is not None:
        payload["coordinate_frame"] = convention
    return _answer(payload)


def _sign_label(label, convention=None):
    payload = {
        "object_kind": "sign_direction",
        "canonical_text": label,
        "sign_value": label,
    }
    if convention is not None:
        payload["sign_convention"] = convention
    return _answer(payload)


class TestVerifySignConvention:
    """End-to-end: ``verify`` reconciles a global sign flip between opposite conventions.

    The lane fires only under an enforcement policy that enables bridges
    (``unit_policy="audited"``) and only when both answers declare opposite,
    globally-reversed conventions and the question fixes none. The default
    (``"strict"``) keeps it off. Bridge metadata surfaces under ``Verdict.details``.
    """

    def test_velocity_flip_accepts_under_audited(self):
        gold = _directional_quantity("-20 m/s", -20.0, "m/s", "right-as-positive")
        pred = _directional_quantity("+20 m/s", 20.0, "m/s", "taking left as positive")
        v = verify(gold, pred, unit_policy="audited")
        assert v.correct is True
        assert v.comparison_mode == "sign_convention"
        assert v.details["bridge_id"] == "sign_convention"
        assert v.details["bridge_tier"] is not None

    def test_vector_global_negation_accepts_under_audited(self):
        gold = _directional_vector((-3.0, 4.0), "x to the left")
        pred = _directional_vector((3.0, -4.0), "x to the right")
        v = verify(gold, pred, unit_policy="audited")
        assert v.correct is True
        assert v.comparison_mode == "sign_convention"

    def test_sign_direction_resolves_under_audited(self):
        gold = _sign_label("positive", "taking left as positive")
        pred = _sign_label("negative", "right-as-positive")
        v = verify(gold, pred, unit_policy="audited")
        assert v.correct is True
        assert v.comparison_mode == "sign_convention"

    def test_default_strict_policy_does_not_reconcile(self):
        gold = _directional_quantity("-20 m/s", -20.0, "m/s", "right-as-positive")
        pred = _directional_quantity("+20 m/s", 20.0, "m/s", "taking left as positive")
        assert verify(gold, pred).correct is False

    def test_intrinsic_sign_never_reconciled(self):
        # Charge sign is intrinsic, not an axis convention: +5 C != -5 C.
        gold = _directional_quantity("+5 C", 5.0, "C")
        pred = _directional_quantity("-5 C", -5.0, "C")
        assert verify(gold, pred, unit_policy="audited").correct is False
