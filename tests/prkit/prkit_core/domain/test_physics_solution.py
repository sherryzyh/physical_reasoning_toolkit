"""
Tests for PhysicsSolution model.
"""

import json
from datetime import datetime

import pytest

from prkit.prkit_core.domain import PhysicsProblem, PhysicsSolution


class TestPhysicsSolution:
    """Test cases for PhysicsSolution model."""

    def test_solution_creation_minimal(self, sample_physics_problem):
        """Test creating a minimal physics solution."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        assert solution.problem_id == "test_001"
        assert solution.problem == sample_physics_problem
        assert solution.agent_answer == "42 m/s"
        assert solution.metadata == {}
        assert solution.intermediate_steps == []

    def test_solution_creation_full(self, sample_physics_problem):
        """Test creating a full physics solution."""
        metadata = {"model": "gpt-4o", "timestamp": datetime.now()}
        intermediate_steps = [{"step": 1, "content": "Identify variables"}]
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
            metadata=metadata,
            intermediate_steps=intermediate_steps,
        )
        assert solution.metadata == metadata
        assert solution.intermediate_steps == intermediate_steps

    def test_solution_problem_id_mismatch(self, sample_physics_problem):
        """Test that problem ID mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Problem ID mismatch"):
            PhysicsSolution(
                problem_id="test_002",  # Different from problem's ID
                problem=sample_physics_problem,
                agent_answer="42 m/s",
            )

    def test_solution_get_domain(self, sample_physics_problem):
        """Test getting domain from solution."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        assert solution.get_domain() == "classical_mechanics"

    def test_solution_get_problem_type(self, sample_physics_problem, sample_physics_problem_mc):
        """Test getting problem type."""
        solution_oe = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        assert solution_oe.get_problem_type() == "OE"

        solution_mc = PhysicsSolution(
            problem_id="test_002",
            problem=sample_physics_problem_mc,
            agent_answer="A",
        )
        assert solution_mc.get_problem_type() == "MC"

    def test_solution_is_multiple_choice(self, sample_physics_problem, sample_physics_problem_mc):
        """Test checking if solution is for multiple choice problem."""
        solution_oe = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        assert solution_oe.is_multiple_choice() is False

        solution_mc = PhysicsSolution(
            problem_id="test_002",
            problem=sample_physics_problem_mc,
            agent_answer="A",
        )
        assert solution_mc.is_multiple_choice() is True

    def test_solution_is_open_ended(self, sample_physics_problem, sample_physics_problem_mc):
        """Test checking if solution is for open-ended problem."""
        solution_oe = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        assert solution_oe.is_open_ended() is True

        solution_mc = PhysicsSolution(
            problem_id="test_002",
            problem=sample_physics_problem_mc,
            agent_answer="A",
        )
        assert solution_mc.is_open_ended() is False

    def test_solution_is_answer_latex_formatted(self, sample_physics_problem):
        """Test checking if answer is LaTeX formatted."""
        solution_latex = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="$$x^2 + 2x + 1$$",
        )
        assert solution_latex.is_answer_latex_formatted() is True

        solution_not_latex = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="x^2 + 2x + 1",
        )
        assert solution_not_latex.is_answer_latex_formatted() is False

        # Test with single $ (should return False)
        solution_single_dollar = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="$x^2$",
        )
        assert solution_single_dollar.is_answer_latex_formatted() is False

    def test_solution_add_intermediate_step(self, sample_physics_problem):
        """Test adding intermediate steps."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        solution.add_intermediate_step(
            step_name="identify_variables",
            step_content="Identify v, u, a, t",
            step_type="reasoning",
        )
        assert len(solution.intermediate_steps) == 1
        assert solution.intermediate_steps[0]["step_name"] == "identify_variables"
        assert solution.intermediate_steps[0]["step_content"] == "Identify v, u, a, t"
        assert solution.intermediate_steps[0]["step_type"] == "reasoning"

    def test_solution_add_intermediate_step_with_tool_usage(self, sample_physics_problem):
        """Test adding intermediate step with tool usage."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        tool_usage = {"calculator": {"operation": "add", "result": 42}}
        solution.add_intermediate_step(
            step_name="calculate",
            step_content="Calculate result",
            tool_usage=tool_usage,
        )
        assert solution.intermediate_steps[0]["tool_usage"] == tool_usage

    def test_solution_add_intermediate_step_with_kwargs(self, sample_physics_problem):
        """Test adding intermediate step with additional kwargs."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        solution.add_intermediate_step(
            step_name="step1",
            step_content="Content",
            custom_field="custom_value",
            another_field=123,
        )
        assert solution.intermediate_steps[0]["custom_field"] == "custom_value"
        assert solution.intermediate_steps[0]["another_field"] == 123

    def test_solution_get_intermediate_step(self, sample_physics_problem):
        """Test getting a specific intermediate step."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        solution.add_intermediate_step("step1", "Content 1")
        solution.add_intermediate_step("step2", "Content 2")

        step = solution.get_intermediate_step("step1")
        assert step is not None
        assert step["step_name"] == "step1"
        assert step["step_content"] == "Content 1"

        step_none = solution.get_intermediate_step("nonexistent")
        assert step_none is None

    def test_solution_get_all_step_names(self, sample_physics_problem):
        """Test getting all step names."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        solution.add_intermediate_step("step1", "Content 1")
        solution.add_intermediate_step("step2", "Content 2")
        solution.add_intermediate_step("step3", "Content 3")

        step_names = solution.get_all_step_names()
        assert len(step_names) == 3
        assert "step1" in step_names
        assert "step2" in step_names
        assert "step3" in step_names

    def test_solution_add_metadata(self, sample_physics_problem):
        """Test adding metadata."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        solution.add_metadata("model", "gpt-4o")
        solution.add_metadata("temperature", 0.7)

        assert solution.metadata["model"] == "gpt-4o"
        assert solution.metadata["temperature"] == 0.7

    def test_solution_get_metadata(self, sample_physics_problem):
        """Test getting metadata."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
            metadata={"model": "gpt-4o", "temperature": 0.7},
        )

        assert solution.get_metadata("model") == "gpt-4o"
        assert solution.get_metadata("temperature") == 0.7
        assert solution.get_metadata("nonexistent") is None
        assert solution.get_metadata("nonexistent", "default") == "default"

    def test_solution_has_metadata(self, sample_physics_problem):
        """Test checking if metadata key exists."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
            metadata={"model": "gpt-4o"},
        )

        assert solution.has_metadata("model") is True
        assert solution.has_metadata("nonexistent") is False

    def test_solution_to_dict(self, sample_physics_problem):
        """Test converting solution to dictionary."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
            metadata={"model": "gpt-4o"},
        )
        solution.add_intermediate_step("step1", "Content 1")

        result = solution.to_dict()
        assert result["problem_id"] == "test_001"
        assert result["agent_answer"] == "42 m/s"
        assert result["metadata"]["model"] == "gpt-4o"
        assert len(result["intermediate_steps"]) == 1
        assert "problem" in result

    def test_solution_to_json(self, sample_physics_problem):
        """Test converting solution to JSON string."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        json_str = solution.to_json()
        assert isinstance(json_str, str)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["problem_id"] == "test_001"
        assert parsed["agent_answer"] == "42 m/s"

    def test_solution_from_dict(self, sample_physics_problem):
        """Test creating solution from dictionary."""
        data = {
            "problem_id": "test_001",
            "problem": sample_physics_problem,
            "agent_answer": "42 m/s",
            "metadata": {"model": "gpt-4o"},
            "intermediate_steps": [{"step_name": "step1", "step_content": "Content 1"}],
        }
        solution = PhysicsSolution.from_dict(data)

        assert solution.problem_id == "test_001"
        assert solution.agent_answer == "42 m/s"
        assert solution.metadata["model"] == "gpt-4o"
        assert len(solution.intermediate_steps) == 1

    def test_solution_from_dict_with_timestamp(self, sample_physics_problem):
        """Test creating solution from dict with timestamp string."""
        timestamp_str = datetime.now().isoformat()
        data = {
            "problem_id": "test_001",
            "problem": sample_physics_problem,
            "agent_answer": "42 m/s",
            "timestamp": timestamp_str,
        }
        solution = PhysicsSolution.from_dict(data)
        # Should not raise an error
        assert solution.problem_id == "test_001"

    def test_solution_get_summary(self, sample_physics_problem):
        """Test getting solution summary."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )
        solution.add_intermediate_step("step1", "Content 1")
        solution.add_intermediate_step("step2", "Content 2")

        summary = solution.get_summary()
        assert summary["problem_id"] == "test_001"
        assert summary["domain"] == "classical_mechanics"
        assert summary["problem_type"] == "OE"
        assert summary["intermediate_steps_count"] == 2

    def test_solution_str_repr(self, sample_physics_problem):
        """Test string representations."""
        solution = PhysicsSolution(
            problem_id="test_001",
            problem=sample_physics_problem,
            agent_answer="42 m/s",
        )

        str_repr = str(solution)
        assert "test_001" in str_repr
        assert "classical_mechanics" in str_repr

        repr_str = repr(solution)
        assert "test_001" in repr_str
        assert "classical_mechanics" in repr_str
        assert "42 m/s" in repr_str
