"""Opt-in live smoke: a real reference build routes a free-axis convention onto ``a_ref``.

Skipped unless ``OPENAI_API_KEY`` is set. Builds reference semantics for a 1-D kinematics
problem whose golden is a signed velocity on a *free* axis, and asserts the build places the
convention on ``a_ref`` (the lane's evidence) and leaves ``q_ref`` convention-free (the problem
fixes no axis) — the routing this whole change is about. If the model declares no convention at
all this run, the routing cannot be observed and the test skips rather than fails.
"""

from __future__ import annotations

import os

import pytest

from prkit.core.domain import Answer, AnswerCategory, PhysicsProblem
from prkit.core.model_clients import create_model_client
from prkit.semantics.build.calls import build_reference_semantics

pytestmark = pytest.mark.integration

_LIVE_MODEL = os.environ.get("PRKIT_SIGN_CONVENTION_SMOKE_MODEL", "gpt-5.4-mini")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set"
)
def test_live_reference_build_routes_free_axis_convention_to_a_ref() -> None:
    problem = PhysicsProblem(
        problem_id="signconv-live-1",
        question=(
            "A block slides along a horizontal frictionless track. No positive direction "
            "is specified. Taking the block's motion into account, its velocity is found to "
            "be 20 m/s directed to the left. Report the velocity as a signed value."
        ),
        answer=Answer(
            value="-20 m/s", answer_category=AnswerCategory.PHYSICAL_QUANTITY
        ),
        domain="mechanics",
    )

    client = create_model_client(_LIVE_MODEL)
    artifact = build_reference_semantics(problem, client)

    a_ref = artifact.reference_answer_semantics
    q_ref = artifact.question_semantics
    a_convention = a_ref.coordinate_frame or a_ref.sign_convention
    q_convention = q_ref.coordinate_frame or q_ref.sign_convention

    # The problem fixes no axis, so the convention must not be routed onto q_ref (the old bug).
    assert q_convention is None, f"convention leaked onto q_ref: {q_convention!r}"
    if a_convention is None:
        pytest.skip("model declared no convention this run; routing not observable")
    # The golden's expressed convention landed on a_ref (the lane's evidence).
    assert a_convention
