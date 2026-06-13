from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from prkit.annotation.workflows.modules.detect_theorem_module import (
    DetectTheoremModule,
)
from prkit.annotation.workflows.modules.domain_assessment_module import (
    DomainAssessmentModule,
)
from prkit.annotation.workflows.modules.review_theorem_module import (
    ReviewTheoremModule,
)
from prkit.annotation.workflows.presets.domain_only_workflow import (
    DomainOnlyWorkflow,
)
from prkit.annotation.workflows.presets.theorem_label_only_workflow import (
    TheoremLabelOnlyWorkflow,
)
from prkit.core.domain import PhysicsProblem


class FakeTheoremResult:
    def __init__(self, theorems):
        self.theorems = theorems

    def to_dict(self):
        return {"theorems": self.theorems}


@patch("prkit.annotation.workflows.modules.detect_theorem_module.TheoremDetector")
def test_detect_theorem_module_process_and_output(mock_detector_class):
    mock_detector = MagicMock()
    mock_detector.work.return_value = FakeTheoremResult([{"name": "Newton"}])
    mock_detector_class.return_value = mock_detector

    module = DetectTheoremModule(model="gpt-5.4-mini")
    problem = PhysicsProblem(problem_id="p1", question="What is force?")

    result = module.process(problem)
    output = module._form_output_as_a_problem(result, problem)

    assert result["status"] == "SUCCESS"
    assert result["theorems"] == [{"name": "Newton"}]
    assert output.additional_fields["theorems"] == [{"name": "Newton"}]
    assert module.get_status()["successful_problems"] == 1


@patch("prkit.annotation.workflows.modules.detect_theorem_module.TheoremDetector")
def test_detect_theorem_module_handles_missing_result(mock_detector_class):
    mock_detector = MagicMock()
    mock_detector.work.return_value = None
    mock_detector_class.return_value = mock_detector

    module = DetectTheoremModule(model="gpt-5.4-mini")
    result = module.process({"question": "What is force?", "problem_id": "p1"})

    assert result["status"] == "FAILED"
    assert module.get_status()["failed_problems"] == 1


@patch("prkit.annotation.workflows.modules.domain_assessment_module.DomainLabeler")
def test_domain_assessment_module_process_success(mock_labeler_class):
    mock_labeler = MagicMock()
    mock_labeler.work.return_value = SimpleNamespace(
        domains=["mechanics", "thermodynamics"],
        confidence=0.8,
    )
    mock_labeler_class.return_value = mock_labeler

    module = DomainAssessmentModule(model="gemini-2.5-pro")
    problem = PhysicsProblem(problem_id="p1", question="Question")

    result = module.process(problem)

    assert result["status"] == "SUCCESS"
    assert result["metadata"]["timestamp"] is None
    assert module.get_status()["domains_labeled"] == 2
    assert module.get_status()["problems_with_multiple_domains"] == 1


@patch("prkit.annotation.workflows.modules.domain_assessment_module.DomainLabeler")
def test_domain_assessment_module_failure_and_problem_output(mock_labeler_class):
    mock_labeler = MagicMock()
    mock_labeler.work.side_effect = RuntimeError("labeler failed")
    mock_labeler_class.return_value = mock_labeler

    module = DomainAssessmentModule(model="gemini-2.5-pro")
    problem = PhysicsProblem(problem_id="p1", question="Question")

    result = module.process(problem)
    passthrough = module._form_output_as_a_problem(result, {"not": "a problem"})

    assert result["status"] == "FAILED"
    assert module.get_status()["failed_problems"] == 1
    assert passthrough == result


@patch("prkit.annotation.workflows.modules.domain_assessment_module.DomainLabeler")
def test_domain_assessment_module_missing_result_and_attaches_metadata(
    mock_labeler_class,
):
    mock_labeler = MagicMock()
    mock_labeler.work.return_value = None
    mock_labeler_class.return_value = mock_labeler

    module = DomainAssessmentModule(model="gemini-2.5-pro")
    result = module.process({"question": "Question", "problem_id": "p1"})

    attached = module._form_output_as_a_problem(
        {
            "domain_labeling": {"domains": ["mechanics"]},
            "metadata": {"module_name": "domain_labeler"},
        },
        PhysicsProblem(problem_id="p2", question="Question"),
    )

    assert result["status"] == "FAILED"
    assert attached.additional_fields["domain_labeling"] == {"domains": ["mechanics"]}
    assert attached.additional_fields["domain_labeling_metadata"] == {
        "module_name": "domain_labeler"
    }


@patch("prkit.annotation.workflows.modules.review_theorem_module.TheoremDetector")
def test_review_theorem_module_get_human_feedback_and_single_review(
    mock_detector_class, monkeypatch
):
    mock_detector_class.return_value = MagicMock()
    module = ReviewTheoremModule(model="claude-sonnet-4-6")

    responses = iter(["maybe", "y", "y", "n", "Conditions are incomplete"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    theorem = {
        "name": "Newton's second law",
        "equations": ["F = ma"],
        "conditions": ["constant mass"],
    }
    reviewed = module._review_single_theorem(theorem, "Q", "S", 1, 1)

    assert reviewed["is_relevant"] is True
    assert reviewed["equations_correct"] is True
    assert reviewed["conditions_valid"] is False
    assert reviewed["conditions_feedback"] == "Conditions are incomplete"


@patch("prkit.annotation.workflows.modules.review_theorem_module.TheoremDetector")
def test_review_theorem_module_reviews_irrelevant_and_incorrect_equations(
    mock_detector_class, monkeypatch
):
    mock_detector_class.return_value = MagicMock()
    module = ReviewTheoremModule(model="claude-sonnet-4-6")

    theorem = {"name": "Theorem", "description": "Desc", "equations": ["F=ma"]}

    irrelevant_inputs = iter(["n", "Not used here"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(irrelevant_inputs))
    irrelevant = module._review_single_theorem(theorem, "Q", "", 1, 1)

    equation_inputs = iter(["y", "n", "Equation is wrong", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(equation_inputs))
    equation_issue = module._review_single_theorem(theorem, "Q", "S", 1, 1)

    assert irrelevant["is_relevant"] is False
    assert (
        irrelevant["equations_feedback"] == "Not applicable - theorem is not relevant"
    )
    assert equation_issue["equations_correct"] is False
    assert equation_issue["equations_feedback"] == "Equation is wrong"
    assert equation_issue["conditions_valid"] is True


@patch("prkit.annotation.workflows.modules.review_theorem_module.TheoremDetector")
def test_review_theorem_module_missing_theorems_and_process_without_predictions(
    mock_detector_class, monkeypatch
):
    mock_detector_class.return_value = MagicMock()
    module = ReviewTheoremModule(model="claude-sonnet-4-6")

    add_inputs = iter(
        [
            "Conservation of Energy",
            "Energy is conserved",
            "E1 + E2 = const",
            "",
            "mechanics",
            "closed system",
            "",
            "DONE",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(add_inputs))

    problem = PhysicsProblem(
        problem_id="p1",
        question="Question",
        additional_fields={},
    )

    missing = module._review_missing_theorems(problem)
    module._review_missing_theorems = lambda _problem: []
    result = module.process(problem)
    output = module._form_output_as_a_problem(
        {
            "theorems": [],
            "missing_theorems": missing,
            "review_metadata": {"problem_id": "p1"},
        },
        PhysicsProblem(
            problem_id="p1",
            question="Question",
            additional_fields={
                "theorems": [{"name": "Old theorem"}],
                "theorem_detection_metadata": {"source": "detector"},
            },
        ),
    )

    assert len(missing) == 1
    assert missing[0]["is_missing_theorem"] is True
    assert result["theorems"] == []
    assert output.additional_fields["missing_theorems"] == missing
    assert output.additional_fields["theorem_metadata"]["detection"] == {
        "source": "detector"
    }
    assert output.additional_fields["theorem_metadata"]["review"] == {
        "problem_id": "p1"
    }


@patch("prkit.annotation.workflows.modules.review_theorem_module.TheoremDetector")
def test_review_theorem_module_process_with_predictions_and_reset(
    mock_detector_class, monkeypatch
):
    mock_detector_class.return_value = MagicMock()
    module = ReviewTheoremModule(model="claude-sonnet-4-6")
    module._review_missing_theorems = lambda _problem: []

    reviewed = [
        {
            "name": "T1",
            "is_relevant": True,
            "equations_correct": True,
            "conditions_valid": False,
        },
        {
            "name": "T2",
            "is_relevant": False,
            "equations_correct": False,
            "conditions_valid": False,
        },
    ]
    monkeypatch.setattr(
        module,
        "_review_single_theorem",
        lambda theorem, **_kwargs: reviewed.pop(0),
    )

    result = module.process(
        PhysicsProblem(
            problem_id="p1",
            question="Question",
            solution="Solution",
            additional_fields={"theorems": [{"name": "T1"}, {"name": "T2"}]},
        )
    )

    status = module.get_status()
    assert result["review_metadata"]["reviewed_theorems"] == 2
    assert status["theorems_reviewed"] == 2
    assert status["problems_with_relevant_theorems"] == 1
    assert status["average_theorems_per_problem"] == 2.0

    module.reset()
    reset_status = module.get_status()
    assert reset_status["theorems_reviewed"] == 0
    assert reset_status["missing_theorems_added"] == 0


@patch("prkit.annotation.workflows.modules.review_theorem_module.TheoremDetector")
def test_review_theorem_module_feedback_handles_interrupts_and_io_errors(
    mock_detector_class, monkeypatch
):
    mock_detector_class.return_value = MagicMock()
    module = ReviewTheoremModule(model="claude-sonnet-4-6")

    attempts = iter([EOFError("eof"), "yes"])

    def fake_input(_prompt=""):
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr("builtins.input", fake_input)
    assert module._get_human_feedback("Prompt", ["yes"]) == "yes"

    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt="": (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt, match="Review interrupted by user"):
        module._get_human_feedback("Prompt", ["y"])


@patch("prkit.annotation.workflows.modules.detect_theorem_module.TheoremDetector")
def test_detect_theorem_module_exception_reset_and_passthrough(mock_detector_class):
    mock_detector = MagicMock()
    mock_detector.work.side_effect = RuntimeError("detector failed")
    mock_detector_class.return_value = mock_detector

    module = DetectTheoremModule(model="gpt-5.4-mini")
    result = module.process("raw question")

    assert result["status"] == "FAILED"
    assert module.get_status()["failed_problems"] == 1
    assert module._form_output_as_a_problem(result, {"not": "problem"}) == result

    module.reset()
    assert module.get_status()["theorems_detected"] == 0


class FakeComposer:
    def __init__(self, name, output_dir, config):
        self.name = name
        self.output_dir = output_dir
        self.config = config
        self.modules = []

    def add_module(self, module):
        self.modules.append(module)

    def run(self, dataset, **kwargs):
        return {"dataset": dataset, "kwargs": kwargs}

    def get_workflow_status(self):
        return {"modules": len(self.modules)}

    def reset(self):
        self.modules.clear()


def test_domain_only_workflow_uses_patched_composer_and_module(monkeypatch, tmp_path):
    fake_module = object()
    monkeypatch.setattr(
        "prkit.annotation.workflows.presets.domain_only_workflow.WorkflowComposer",
        FakeComposer,
    )
    monkeypatch.setattr(
        "prkit.annotation.workflows.presets.domain_only_workflow.DomainAssessmentModule",
        lambda model: fake_module,
    )

    workflow = DomainOnlyWorkflow(output_dir=str(tmp_path), model="gemini-2.5-pro")
    result = workflow.run("dataset")

    assert workflow.workflow.modules == [fake_module]
    assert result["dataset"] == "dataset"
    assert workflow.get_status()["modules"] == 1
    workflow.reset()
    assert workflow.get_status()["modules"] == 0


def test_theorem_label_only_workflow_uses_patched_composer_and_module(
    monkeypatch, tmp_path
):
    fake_module = object()
    monkeypatch.setattr(
        "prkit.annotation.workflows.presets.theorem_label_only_workflow.WorkflowComposer",
        FakeComposer,
    )
    monkeypatch.setattr(
        "prkit.annotation.workflows.presets.theorem_label_only_workflow.DetectTheoremModule",
        lambda name, model: fake_module,
    )

    workflow = TheoremLabelOnlyWorkflow(output_dir=str(tmp_path), model="gpt-5.4-mini")
    result = workflow.run("dataset")

    assert workflow.workflow.modules == [fake_module]
    assert result["dataset"] == "dataset"
    assert "TheoremLabelOnlyWorkflow" in repr(workflow)
