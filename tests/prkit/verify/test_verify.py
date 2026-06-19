"""Tests for the ``prkit.verify`` light-import facade (parse / verify)."""

from __future__ import annotations

import pytest

from prkit.core.verdict import Verdict
from prkit.semantics import PhysicsAnswerSemantics
from prkit.verify import parse, verify


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


class TestParse:
    def test_returns_physics_answer_semantics(self):
        parsed = parse("9.8 m/s^2")
        assert isinstance(parsed, PhysicsAnswerSemantics)
        assert parsed.unit == "m/s^2"

    def test_category_not_supported_yet(self):
        with pytest.raises(NotImplementedError):
            parse("9.8", category="number")
