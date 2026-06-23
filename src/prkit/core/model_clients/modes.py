"""Mode enums for the high-level ``solve_physics_problem`` client method."""

from __future__ import annotations

from enum import StrEnum


class _StrEnum(StrEnum):
    """Enum subclass with string values and friendly ``str()`` output."""

    def __str__(self) -> str:
        return str(self.value)


class PhysicsOutputMode(_StrEnum):
    """What form the model's answer should take.

    The input *kind* is determined by the runtime type of the argument passed to
    ``solve_physics_problem`` (plain ``str``, ``PhysicsProblem``, or — once
    supported — a physics ``PhysicsQuestionSemantics``), so there is no
    corresponding input-mode enum.
    """

    ANSWER_TEXT = "answer_text"
    # TODO: structured physics answer semantics output (needs prkit.semantics).
    ANSWER_SEMANTICS = "answer_semantics"
