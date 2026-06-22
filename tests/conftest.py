"""
Pytest configuration and shared fixtures for PRKit tests.
"""

import tempfile
from pathlib import Path

import pytest

from prkit.core.domain import (
    PhysicsAnswer,
    PhysicsDataset,
    PhysicsDomain,
    PhysicsProblem,
)


@pytest.fixture
def sample_answer_numerical():
    """Create a sample numerical answer."""
    return PhysicsAnswer(value="42.0", unit="m/s", source_type="NV")


@pytest.fixture
def sample_answer_symbolic():
    """Create a sample symbolic answer."""
    return PhysicsAnswer(value="x^2 + 2x + 1")


@pytest.fixture
def sample_answer_textual():
    """Create a sample textual answer."""
    return PhysicsAnswer(value="The force is equal to mass times acceleration")


@pytest.fixture
def sample_answer_option():
    """Create a sample option answer."""
    return PhysicsAnswer(value="A", source_type="MC")


@pytest.fixture
def sample_physics_problem():
    """Create a sample physics problem."""
    return PhysicsProblem(
        problem_id="test_001",
        question="What is the speed of light?",
        answer=PhysicsAnswer(value="3e8", unit="m/s"),
        solution="The speed of light in vacuum is approximately 3 × 10^8 m/s",
        domain=PhysicsDomain.CLASSICAL_MECHANICS,
        language="en",
        problem_type="OE",
    )


@pytest.fixture
def sample_physics_problem_mc():
    """Create a sample multiple choice physics problem."""
    return PhysicsProblem(
        problem_id="test_002",
        question="What is F = ma?",
        answer=PhysicsAnswer(value="A", source_type="MCQ"),
        options=[
            "Newton's second law",
            "Newton's first law",
            "Newton's third law",
            "Ohm's law",
        ],
        correct_option=0,
        problem_type="MC",
        domain=PhysicsDomain.CLASSICAL_MECHANICS,
    )


@pytest.fixture
def sample_dataset(sample_physics_problem, sample_physics_problem_mc):
    """Create a sample dataset with multiple problems."""
    problems = [sample_physics_problem, sample_physics_problem_mc]
    return PhysicsDataset(
        problems=problems, info={"name": "test_dataset", "version": "1.0"}, split="test"
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_problems_list():
    """Create a list of sample physics problems."""
    problems = []
    for i in range(5):
        problem = PhysicsProblem(
            problem_id=f"test_{i:03d}",
            question=f"Test question {i}",
            answer=PhysicsAnswer(value=str(i)),
            domain=(
                PhysicsDomain.CLASSICAL_MECHANICS
                if i % 2 == 0
                else PhysicsDomain.QUANTUM_MECHANICS
            ),
        )
        problems.append(problem)
    return problems


@pytest.fixture
def mock_llm_response(monkeypatch):
    """Mock LLM client responses for testing."""

    def mock_chat(self, messages):
        # Simple mock that returns "TRUE" for most cases
        user_content = messages[-1]["content"] if messages else ""
        if "TRUE" in user_content.upper() or "equal" in user_content.lower():
            return "TRUE"
        return "FALSE"

    # This would need to be patched at the model client level
    # For now, we'll handle it in individual test files
    return mock_chat
