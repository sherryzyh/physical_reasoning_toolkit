"""
Tests for evaluation metrics: AccuracyMetric.
"""

import pytest

from prkit.prkit_core.definitions import AnswerType
from prkit.prkit_core.models import Answer
from prkit.prkit_evaluation.comparison import SmartAnswerComparator
from prkit.prkit_evaluation.metrics import AccuracyMetric, BaseMetric


class TestBaseMetric:
    """Test cases for BaseMetric base class."""

    def test_base_metric_initialization(self):
        """Test base metric initialization."""

        class TestMetric(BaseMetric):
            def compute(self, predictions, ground_truths, **kwargs):
                return {"result": 0.5}

        metric = TestMetric(name="test", description="Test metric")
        assert metric.name == "test"
        assert metric.description == "Test metric"

    def test_base_metric_validate_inputs_valid(self):
        """Test input validation with valid inputs."""

        class TestMetric(BaseMetric):
            def compute(self, predictions, ground_truths, **kwargs):
                return {"result": 0.5}

        metric = TestMetric(name="test")
        predictions = [Answer(value=1, answer_type=AnswerType.NUMERICAL)]
        ground_truths = [Answer(value=1, answer_type=AnswerType.NUMERICAL)]

        # Should not raise
        metric.validate_inputs(predictions, ground_truths)

    def test_base_metric_validate_inputs_invalid_length(self):
        """Test input validation with mismatched lengths."""

        class TestMetric(BaseMetric):
            def compute(self, predictions, ground_truths, **kwargs):
                return {"result": 0.5}

        metric = TestMetric(name="test")
        predictions = [Answer(value=1, answer_type=AnswerType.NUMERICAL)]
        ground_truths = [
            Answer(value=1, answer_type=AnswerType.NUMERICAL),
            Answer(value=2, answer_type=AnswerType.NUMERICAL),
        ]

        with pytest.raises(ValueError, match="same length"):
            metric.validate_inputs(predictions, ground_truths)

    def test_base_metric_validate_inputs_empty(self):
        """Test input validation with empty lists."""

        class TestMetric(BaseMetric):
            def compute(self, predictions, ground_truths, **kwargs):
                return {"result": 0.5}

        metric = TestMetric(name="test")
        with pytest.raises(ValueError, match="empty"):
            metric.validate_inputs([], [])

    def test_base_metric_validate_inputs_not_lists(self):
        """Test input validation with non-list inputs."""

        class TestMetric(BaseMetric):
            def compute(self, predictions, ground_truths, **kwargs):
                return {"result": 0.5}

        metric = TestMetric(name="test")
        with pytest.raises(ValueError, match="must be lists"):
            metric.validate_inputs("not a list", [])

    def test_base_metric_validate_inputs_not_answers(self):
        """Test input validation with non-Answer objects."""

        class TestMetric(BaseMetric):
            def compute(self, predictions, ground_truths, **kwargs):
                return {"result": 0.5}

        metric = TestMetric(name="test")
        with pytest.raises(ValueError, match="not an Answer instance"):
            metric.validate_inputs(
                ["not an answer"], [Answer(value=1, answer_type=AnswerType.NUMERICAL)]
            )

    def test_base_metric_get_metric_info(self):
        """Test getting metric info."""

        class TestMetric(BaseMetric):
            def compute(self, predictions, ground_truths, **kwargs):
                return {"result": 0.5}

        metric = TestMetric(name="test", description="Test description")
        info = metric.get_metric_info()
        assert info["name"] == "test"
        assert info["description"] == "Test description"
        assert "class" in info


class TestAccuracyMetric:
    """Test cases for AccuracyMetric."""

    def test_accuracy_metric_initialization(self):
        """Test accuracy metric initialization."""
        metric = AccuracyMetric()
        assert metric.name == "Accuracy"
        assert metric.description == "Measures the proportion of correct predictions"
        assert isinstance(metric.comparator, SmartAnswerComparator)

    def test_accuracy_metric_initialization_custom_comparator(self):
        """Test accuracy metric with custom comparator."""
        comparator = SmartAnswerComparator()
        metric = AccuracyMetric(comparator=comparator)
        assert metric.comparator == comparator

    def test_compute_all_correct(self):
        """Test computing accuracy with all correct predictions."""
        metric = AccuracyMetric()
        predictions = [
            Answer(value=1, answer_type=AnswerType.NUMERICAL),
            Answer(value=2, answer_type=AnswerType.NUMERICAL),
            Answer(value=3, answer_type=AnswerType.NUMERICAL),
        ]
        ground_truths = [
            Answer(value=1, answer_type=AnswerType.NUMERICAL),
            Answer(value=2, answer_type=AnswerType.NUMERICAL),
            Answer(value=3, answer_type=AnswerType.NUMERICAL),
        ]

        result = metric.compute(predictions, ground_truths)
        assert result["accuracy"] == 1.0
        assert result["total_samples"] == 3
        assert result["correct_samples"] == 3
        assert result["incorrect_samples"] == 0

    def test_compute_all_incorrect(self):
        """Test computing accuracy with all incorrect predictions."""
        metric = AccuracyMetric()
        predictions = [
            Answer(value=1, answer_type=AnswerType.NUMERICAL),
            Answer(value=2, answer_type=AnswerType.NUMERICAL),
        ]
        ground_truths = [
            Answer(value=10, answer_type=AnswerType.NUMERICAL),
            Answer(value=20, answer_type=AnswerType.NUMERICAL),
        ]

        result = metric.compute(predictions, ground_truths)
        assert result["accuracy"] == 0.0
        assert result["correct_samples"] == 0
        assert result["incorrect_samples"] == 2

    def test_compute_partial_correct(self):
        """Test computing accuracy with partial correctness."""
        metric = AccuracyMetric()
        predictions = [
            Answer(value=1, answer_type=AnswerType.NUMERICAL),
            Answer(value=20, answer_type=AnswerType.NUMERICAL),
            Answer(value=3, answer_type=AnswerType.NUMERICAL),
        ]
        ground_truths = [
            Answer(value=1, answer_type=AnswerType.NUMERICAL),
            Answer(value=2, answer_type=AnswerType.NUMERICAL),
            Answer(value=3, answer_type=AnswerType.NUMERICAL),
        ]

        result = metric.compute(predictions, ground_truths)
        assert result["accuracy"] == pytest.approx(2 / 3, rel=1e-6)
        assert result["correct_samples"] == 2
        assert result["incorrect_samples"] == 1

    def test_compute_with_details(self):
        """Test computing accuracy with return_details=True."""
        metric = AccuracyMetric()
        predictions = [
            Answer(value=1, answer_type=AnswerType.NUMERICAL),
            Answer(value=2, answer_type=AnswerType.NUMERICAL),
        ]
        ground_truths = [
            Answer(value=1, answer_type=AnswerType.NUMERICAL),
            Answer(value=20, answer_type=AnswerType.NUMERICAL),
        ]

        result = metric.compute(predictions, ground_truths, return_details=True)
        assert "details" in result
        assert len(result["details"]) == 2
        assert result["details"][0]["is_correct"] is True
        assert result["details"][1]["is_correct"] is False

    def test_compute_single_correct(self):
        """Test computing accuracy for a single correct prediction."""
        metric = AccuracyMetric()
        prediction = Answer(value=42, answer_type=AnswerType.NUMERICAL)
        ground_truth = Answer(value=42, answer_type=AnswerType.NUMERICAL)

        result = metric.compute_single(prediction, ground_truth)
        assert result["is_correct"] is True
        assert "comparison_details" in result
        assert "prediction" in result
        assert "ground_truth" in result

    def test_compute_single_incorrect(self):
        """Test computing accuracy for a single incorrect prediction."""
        metric = AccuracyMetric()
        prediction = Answer(value=42, answer_type=AnswerType.NUMERICAL)
        ground_truth = Answer(value=43, answer_type=AnswerType.NUMERICAL)

        result = metric.compute_single(prediction, ground_truth)
        assert result["is_correct"] is False

    def test_get_accuracy_by_type(self):
        """Test computing accuracy broken down by answer type."""
        metric = AccuracyMetric()
        predictions = [
            Answer(value=1, answer_type=AnswerType.NUMERICAL),
            Answer(value="A", answer_type=AnswerType.OPTION),
            Answer(value=3, answer_type=AnswerType.NUMERICAL),
        ]
        ground_truths = [
            Answer(value=1, answer_type=AnswerType.NUMERICAL),
            Answer(value="A", answer_type=AnswerType.OPTION),
            Answer(value=30, answer_type=AnswerType.NUMERICAL),
        ]

        result = metric.get_accuracy_by_type(predictions, ground_truths)
        assert "numerical" in result
        assert "option" in result
        assert result["numerical"]["accuracy"] == pytest.approx(0.5, rel=1e-6)
        assert result["option"]["accuracy"] == 1.0

    def test_compute_empty_list(self):
        """Test computing accuracy with empty lists."""
        metric = AccuracyMetric()
        with pytest.raises(ValueError):
            metric.compute([], [])

    def test_compute_mismatched_lengths(self):
        """Test computing accuracy with mismatched lengths."""
        metric = AccuracyMetric()
        predictions = [Answer(value=1, answer_type=AnswerType.NUMERICAL)]
        ground_truths = [
            Answer(value=1, answer_type=AnswerType.NUMERICAL),
            Answer(value=2, answer_type=AnswerType.NUMERICAL),
        ]

        with pytest.raises(ValueError):
            metric.compute(predictions, ground_truths)

    def test_compute_mixed_types(self):
        """Test computing accuracy with mixed answer types."""
        metric = AccuracyMetric()
        predictions = [
            Answer(value=42, answer_type=AnswerType.NUMERICAL),
            Answer(value="A", answer_type=AnswerType.OPTION),
        ]
        ground_truths = [
            Answer(value=42, answer_type=AnswerType.NUMERICAL),
            Answer(value="A", answer_type=AnswerType.OPTION),
        ]

        result = metric.compute(predictions, ground_truths)
        assert result["accuracy"] == 1.0
