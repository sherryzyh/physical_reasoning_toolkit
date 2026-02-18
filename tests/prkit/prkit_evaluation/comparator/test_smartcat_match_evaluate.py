"""
Unit tests for smartcat_match_evaluate script.

Tests cover:
- get_category_and_normalized_value_for_csv
- process_json_file (success, missing fields, invalid JSON)
- load_accuracy_scores_from_evaluation_csv
- extract_metadata_from_results
- evaluate_results (with temp JSON files)
"""

import importlib.util
import json
import pytest
from pathlib import Path

from prkit.prkit_evaluation.comparator.smart_match import SmartMatchComparator


# Load smartcat_match_evaluate module from script path
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT_PATH = _PROJECT_ROOT / "uncertainty_quantification_physical_reasoning" / "scripts" / "evaluate" / "smartcat_match_evaluate.py"


def _load_smartcat_module():
    """Load the smartcat_match_evaluate module dynamically."""
    spec = importlib.util.spec_from_file_location("smartcat_match_evaluate", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"smartcat_match_evaluate script not found at {_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def smartcat_module():
    """Load smartcat_match_evaluate module."""
    return _load_smartcat_module()


@pytest.fixture
def comparator():
    """SmartMatchComparator instance for tests."""
    return SmartMatchComparator()


class TestGetCategoryAndNormalizedValueForCsv:
    """Tests for get_category_and_normalized_value_for_csv."""

    def test_number_string(self, smartcat_module, comparator):
        """Number string returns NUMBER category and normalized value."""
        cat, norm = smartcat_module.get_category_and_normalized_value_for_csv(
            comparator, "42"
        )
        assert cat == "number"
        assert norm == "42"

    def test_physical_quantity_string(self, smartcat_module, comparator):
        """Physical quantity returns category and normalized value."""
        cat, norm = smartcat_module.get_category_and_normalized_value_for_csv(
            comparator, "9.8 m/s^2"
        )
        assert cat == "physical_quantity"
        assert "9.8" in norm or "9" in norm
        assert "m/s" in norm or "m" in norm

    def test_text_string(self, smartcat_module, comparator):
        """Text string returns TEXT category."""
        cat, norm = smartcat_module.get_category_and_normalized_value_for_csv(
            comparator, "hello world"
        )
        assert cat == "text"
        assert norm == "hello world"

    def test_float_formatting_small(self, smartcat_module, comparator):
        """Small floats use decimal format."""
        cat, norm = smartcat_module.get_category_and_normalized_value_for_csv(
            comparator, "3.14"
        )
        assert cat == "number"
        assert "3.14" in norm or "3" in norm

    def test_float_formatting_scientific(self, smartcat_module, comparator):
        """Very small/large numbers use scientific notation."""
        cat, norm = smartcat_module.get_category_and_normalized_value_for_csv(
            comparator, "1e-10"
        )
        assert cat == "number"
        assert "e" in norm.lower() or norm == "1e-10"


class TestProcessJsonFile:
    """Tests for process_json_file."""

    def test_success_valid_json(self, smartcat_module, temp_dir):
        """Valid JSON with required fields returns data."""
        json_path = temp_dir / "test.json"
        json_path.write_text(
            json.dumps({
                "problem_id": "prob_001",
                "answer": "42",
                "model_answer": "42",
            }),
            encoding="utf-8",
        )
        data, failure = smartcat_module.process_json_file(json_path)
        assert data is not None
        assert failure is None
        assert data["problem_id"] == "prob_001"
        assert data["ground_truth_str"] == "42"
        assert data["model_answer_str"] == "42"
        assert data["empty_model_answer"] is False

    def test_missing_answer(self, smartcat_module, temp_dir):
        """Missing 'answer' field returns failure."""
        json_path = temp_dir / "missing_answer.json"
        json_path.write_text(
            json.dumps({
                "problem_id": "prob_002",
                "model_answer": "42",
            }),
            encoding="utf-8",
        )
        data, failure = smartcat_module.process_json_file(json_path)
        assert data is None
        assert failure is not None
        assert "answer" in failure.get("error", "").lower() or "missing" in failure.get("error", "").lower()

    def test_missing_model_answer(self, smartcat_module, temp_dir):
        """Missing 'model_answer' field returns failure."""
        json_path = temp_dir / "missing_model.json"
        json_path.write_text(
            json.dumps({
                "problem_id": "prob_003",
                "answer": "42",
            }),
            encoding="utf-8",
        )
        data, failure = smartcat_module.process_json_file(json_path)
        assert data is None
        assert failure is not None

    def test_empty_model_answer(self, smartcat_module, temp_dir):
        """Empty model_answer (None or blank) is flagged but file processed."""
        json_path = temp_dir / "empty_model.json"
        json_path.write_text(
            json.dumps({
                "problem_id": "prob_004",
                "answer": "42",
                "model_answer": "",
            }),
            encoding="utf-8",
        )
        data, failure = smartcat_module.process_json_file(json_path)
        assert data is not None
        assert data["empty_model_answer"] is True
        assert data["model_answer_str"] == ""

    def test_invalid_json(self, smartcat_module, temp_dir):
        """Invalid JSON returns failure."""
        json_path = temp_dir / "invalid.json"
        json_path.write_text("{ invalid json }", encoding="utf-8")
        data, failure = smartcat_module.process_json_file(json_path)
        assert data is None
        assert failure is not None
        assert "error" in failure


class TestLoadAccuracyScoresFromEvaluationCsv:
    """Tests for load_accuracy_scores_from_evaluation_csv."""

    def test_file_not_found_returns_none(self, smartcat_module, temp_dir):
        """Non-existent path returns None."""
        result = smartcat_module.load_accuracy_scores_from_evaluation_csv(
            temp_dir, "nonexistent_exp", "exact"
        )
        assert result is None

    def test_load_existing_csv(self, smartcat_module, temp_dir):
        """Load accuracy_score from existing CSV."""
        eval_base = temp_dir
        csv_dir = (
            eval_base
            / "inference_with_answer_tag"
            / "exact_match"
            / "exp_exact"
        )
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "evaluation_results.csv"
        csv_path.write_text(
            "problem_id,accuracy_score\n0,1.0\n1,0.0\n", encoding="utf-8"
        )
        result = smartcat_module.load_accuracy_scores_from_evaluation_csv(
            eval_base, "exp", "exact"
        )
        assert result is not None
        assert result.get("0") == 1.0
        assert result.get("1") == 0.0


class TestExtractMetadataFromResults:
    """Tests for extract_metadata_from_results."""

    def test_empty_dir_returns_empty(self, smartcat_module, temp_dir):
        """Empty directory returns empty dict."""
        result = smartcat_module.extract_metadata_from_results(temp_dir)
        assert result == {}

    def test_no_json_files_returns_empty(self, smartcat_module, temp_dir):
        """Directory with no JSON files returns empty."""
        (temp_dir / "not_json.txt").write_text("x", encoding="utf-8")
        result = smartcat_module.extract_metadata_from_results(temp_dir)
        assert result == {}

    def test_extracts_metadata_from_first_file(self, smartcat_module, temp_dir):
        """Extracts metadata from first JSON file (skips statistics)."""
        (temp_dir / "result_001.json").write_text(
            json.dumps({
                "dataset_name": "test_dataset",
                "model_name": "gpt-4",
                "problem_id": "001",
                "answer": "42",
                "model_answer": "42",
            }),
            encoding="utf-8",
        )
        result = smartcat_module.extract_metadata_from_results(temp_dir)
        assert result.get("dataset_name") == "test_dataset"
        assert result.get("model_name") == "gpt-4"
        assert result.get("total_result_files") == 1


class TestEvaluateResults:
    """Tests for evaluate_results integration."""

    def test_evaluate_writes_output(self, smartcat_module, comparator, temp_dir):
        """evaluate_results processes JSON files and writes CSV/stats."""
        results_dir = temp_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "prob_001.json").write_text(
            json.dumps({
                "problem_id": "prob_001",
                "answer": "42",
                "model_answer": "42",
            }),
            encoding="utf-8",
        )
        output_dir = temp_dir / "output"
        smartcat_module.evaluate_results(
            results_dir=results_dir,
            output_dir=output_dir,
            comparator=comparator,
        )
        assert (output_dir / "evaluation_results.csv").exists()
        assert (output_dir / "statistics.json").exists()
        assert (output_dir / "evaluation_config.json").exists()
        stats = json.loads((output_dir / "statistics.json").read_text(encoding="utf-8"))
        assert stats["total_problems"] == 1
        assert stats["correct"] == 1
        assert stats["overall_accuracy"] == 1.0
