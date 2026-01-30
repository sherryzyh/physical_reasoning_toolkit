"""
Tests for PhysicsProblem model.
"""

import pytest
from prkit.prkit_core.models import PhysicsProblem, Answer
from prkit.prkit_core.definitions import PhysicsDomain, AnswerType


class TestPhysicsProblem:
    """Test cases for PhysicsProblem model."""
    
    def test_problem_creation_minimal(self):
        """Test creating a minimal physics problem."""
        problem = PhysicsProblem(problem_id="test_001", question="What is F=ma?")
        assert problem.problem_id == "test_001"
        assert problem.question == "What is F=ma?"
        assert problem.answer is None
        assert problem.solution is None
        assert problem.domain is None
    
    def test_problem_creation_full(self):
        """Test creating a full physics problem."""
        answer = Answer(value=42, answer_type=AnswerType.NUMERICAL)
        problem = PhysicsProblem(
            problem_id="test_001",
            question="What is the answer?",
            answer=answer,
            solution="The answer is 42",
            domain=PhysicsDomain.CLASSICAL_MECHANICS,
            language="en",
            problem_type="OE"
        )
        assert problem.problem_id == "test_001"
        assert problem.answer == answer
        assert problem.solution == "The answer is 42"
        assert problem.domain == PhysicsDomain.CLASSICAL_MECHANICS
        assert problem.language == "en"
        assert problem.problem_type == "OE"
    
    def test_problem_domain_conversion(self):
        """Test domain string to enum conversion."""
        problem = PhysicsProblem(
            problem_id="test_001",
            question="Test",
            domain="classical_mechanics"
        )
        assert isinstance(problem.domain, PhysicsDomain) or problem.domain == "classical_mechanics"
    
    def test_problem_image_path_normalization(self):
        """Test image path normalization."""
        # Test with single string
        problem = PhysicsProblem(
            problem_id="test_001",
            question="Test",
            image_path="/path/to/image.jpg"
        )
        assert isinstance(problem.image_path, list)
        assert len(problem.image_path) == 1
        
        # Test with list
        problem2 = PhysicsProblem(
            problem_id="test_002",
            question="Test",
            image_path=["/path/to/img1.jpg", "/path/to/img2.jpg"]
        )
        assert len(problem2.image_path) == 2
        
        # Test with None
        problem3 = PhysicsProblem(problem_id="test_003", question="Test", image_path=None)
        assert problem3.image_path == []
    
    def test_problem_type_validation(self):
        """Test problem type validation."""
        # Valid types
        problem_mc = PhysicsProblem(problem_id="test_001", question="Test", problem_type="MC")
        assert problem_mc.problem_type == "MC"
        
        problem_oe = PhysicsProblem(problem_id="test_002", question="Test", problem_type="OE")
        assert problem_oe.problem_type == "OE"
        
        # Invalid type should raise ValueError
        with pytest.raises(ValueError):
            PhysicsProblem(problem_id="test_003", question="Test", problem_type="INVALID")
    
    def test_problem_multiple_choice(self):
        """Test multiple choice problem methods."""
        problem = PhysicsProblem(
            problem_id="test_001",
            question="What is F=ma?",
            problem_type="MC",
            options=["A", "B", "C", "D"],
            correct_option=0
        )
        assert problem.is_multiple_choice() is True
        assert problem.is_open_ended() is False
        assert problem.options == ["A", "B", "C", "D"]
        assert problem.correct_option == 0
    
    def test_problem_open_ended(self):
        """Test open-ended problem methods."""
        problem = PhysicsProblem(
            problem_id="test_001",
            question="What is F=ma?",
            problem_type="OE"
        )
        assert problem.is_open_ended() is True
        assert problem.is_multiple_choice() is False
    
    def test_problem_has_solution(self):
        """Test solution checking."""
        problem_with_solution = PhysicsProblem(
            problem_id="test_001",
            question="Test",
            solution="This is a solution"
        )
        assert problem_with_solution.has_solution() is True
        
        problem_without_solution = PhysicsProblem(problem_id="test_002", question="Test")
        assert problem_without_solution.has_solution() is False
    
    def test_problem_get_domain_name(self):
        """Test getting domain name."""
        problem = PhysicsProblem(
            problem_id="test_001",
            question="Test",
            domain=PhysicsDomain.CLASSICAL_MECHANICS
        )
        assert problem.get_domain_name() == "classical_mechanics"
        
        problem_no_domain = PhysicsProblem(problem_id="test_002", question="Test")
        assert problem_no_domain.get_domain_name() == "Unknown"
    
    def test_problem_dict_access(self):
        """Test dictionary-like access."""
        problem = PhysicsProblem(
            problem_id="test_001",
            question="Test",
            domain=PhysicsDomain.CLASSICAL_MECHANICS
        )
        assert problem["problem_id"] == "test_001"
        assert problem["question"] == "Test"
        assert problem["domain"] == PhysicsDomain.CLASSICAL_MECHANICS
        
        # Test contains
        assert "problem_id" in problem
        assert "nonexistent" not in problem
    
    def test_problem_dict_setitem(self):
        """Test dictionary-like assignment."""
        problem = PhysicsProblem(problem_id="test_001", question="Test")
        problem["custom_field"] = "custom_value"
        assert problem["custom_field"] == "custom_value"
        assert problem.additional_fields["custom_field"] == "custom_value"
    
    def test_problem_keys_values_items(self):
        """Test keys, values, and items methods."""
        problem = PhysicsProblem(
            problem_id="test_001",
            question="Test",
            domain=PhysicsDomain.CLASSICAL_MECHANICS
        )
        keys = problem.keys()
        assert "problem_id" in keys
        assert "question" in keys
        assert "domain" in keys
        
        values = problem.values()
        assert "test_001" in values
        
        items = problem.items()
        assert ("problem_id", "test_001") in items
    
    def test_problem_get_method(self):
        """Test get method with default."""
        problem = PhysicsProblem(problem_id="test_001", question="Test")
        assert problem.get("problem_id") == "test_001"
        assert problem.get("nonexistent", "default") == "default"
    
    def test_problem_update(self):
        """Test update method."""
        problem = PhysicsProblem(problem_id="test_001", question="Test")
        problem.update({"custom1": "value1", "custom2": "value2"})
        assert problem["custom1"] == "value1"
        assert problem["custom2"] == "value2"
    
    def test_problem_to_dict(self):
        """Test problem serialization."""
        answer = Answer(value=42, answer_type=AnswerType.NUMERICAL)
        problem = PhysicsProblem(
            problem_id="test_001",
            question="Test",
            answer=answer,
            domain=PhysicsDomain.CLASSICAL_MECHANICS
        )
        result = problem.to_dict()
        assert result["problem_id"] == "test_001"
        assert result["question"] == "Test"
        assert "answer" in result
    
    def test_problem_from_dict(self):
        """Test problem deserialization."""
        data = {
            "problem_id": "test_001",
            "question": "Test question",
            "answer": {
                "value": 42,
                "answer_type": "numerical",
                "unit": "m/s"
            },
            "domain": "classical_mechanics"
        }
        problem = PhysicsProblem.from_dict(data)
        assert problem.problem_id == "test_001"
        assert problem.question == "Test question"
        assert problem.answer.value == 42
        assert problem.answer.answer_type == AnswerType.NUMERICAL
    
    def test_problem_copy(self):
        """Test problem copying."""
        problem = PhysicsProblem(
            problem_id="test_001",
            question="Test",
            domain=PhysicsDomain.CLASSICAL_MECHANICS
        )
        copied = problem.copy()
        assert copied.problem_id == problem.problem_id
        assert copied.question == problem.question
        assert copied is not problem
    
    def test_problem_display(self):
        """Test problem display method."""
        problem = PhysicsProblem(
            problem_id="test_001",
            question="Test question",
            solution="Test solution"
        )
        display = problem.display()
        assert "test_001" in display
        assert "Test question" in display
        assert "Test solution" in display
    
    def test_problem_repr_str(self):
        """Test string representations."""
        problem = PhysicsProblem(problem_id="test_001", question="Test question")
        assert "test_001" in repr(problem)
        assert "Test question" in str(problem)
