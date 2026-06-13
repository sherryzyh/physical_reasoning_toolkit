from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from prkit.annotation.workflows.modules.base_module import BaseWorkflowModule
from prkit.annotation.workflows import workflow_composer as workflow_composer_module
from prkit.annotation.workflows.workflow_composer import WorkflowComposer
from prkit.core.domain import PhysicalDataset, PhysicsProblem


class DummyWorkflowModule(BaseWorkflowModule):
    def process(self, problem: PhysicsProblem, **kwargs):
        if kwargs.get("return_none"):
            return None
        if kwargs.get("fail_in_process"):
            raise RuntimeError("process failed")
        return {
            "problem_id": problem.problem_id,
            "flag": kwargs.get("flag", "ok"),
        }

    def _form_output_as_a_problem(self, result, problem: PhysicsProblem) -> PhysicsProblem:
        updated = problem.copy()
        updated.additional_fields = dict(updated.additional_fields or {})
        updated.additional_fields["flag"] = result["flag"]
        return updated


class ExplodingWorkflowModule(DummyWorkflowModule):
    def run(self, problem, problem_as_output=True, **kwargs):
        raise RuntimeError("boom")


def _dataset() -> PhysicalDataset:
    return PhysicalDataset(
        problems=[
            PhysicsProblem(problem_id="p1", question="Question 1"),
            PhysicsProblem(problem_id="p2", question="Question 2"),
        ]
    )


def test_base_workflow_module_run_success_and_reset():
    module = DummyWorkflowModule("dummy", model="gpt-5.4-mini")
    problem = PhysicsProblem(problem_id="p1", question="Question 1")

    output = module.run(problem, flag="handled")

    assert isinstance(output, PhysicsProblem)
    assert output.additional_fields["flag"] == "handled"
    assert module.get_status()["execution_status"] == "SUCCESS"
    assert "DummyWorkflowModule" in repr(module)

    module.reset()
    assert module.get_status()["execution_status"] == "PENDING"


def test_base_workflow_module_run_returns_raw_result_and_handles_errors():
    module = DummyWorkflowModule("dummy", model="gpt-5.4-mini")
    problem = PhysicsProblem(problem_id="p1", question="Question 1")

    raw_output = module.run(problem, problem_as_output=False, flag="raw")
    assert raw_output == {"problem_id": "p1", "flag": "raw"}

    none_output = module.run(problem, return_none=True)
    assert isinstance(none_output, PhysicsProblem)

    failed_output = module.run(problem, fail_in_process=True)
    assert isinstance(failed_output, PhysicsProblem)
    assert module.get_status()["execution_status"] == "FAILED"

    with pytest.raises(ValueError, match="PhysicsProblem object"):
        module.run({"problem_id": "bad"})


def test_workflow_composer_add_remove_clear_modules(tmp_path):
    module = DummyWorkflowModule("dummy", model="gpt-5.4-mini")
    composer = WorkflowComposer(
        name="demo",
        output_dir=tmp_path,
        config={"show_progress": False},
    )

    composer.add_module(module)
    assert composer.get_module_status("dummy")["module_name"] == "dummy"

    composer.remove_module("dummy")
    assert composer.get_module_status("dummy") is None

    composer.add_modules([module])
    composer.clear_modules()
    assert composer.modules == []


def test_workflow_composer_run_saves_results_and_status(tmp_path):
    module = DummyWorkflowModule("dummy", model="gpt-5.4-mini")
    composer = WorkflowComposer(
        name="demo",
        output_dir=tmp_path,
        modules=[module],
        config={"show_progress": False},
    )

    result = composer.run(_dataset(), flag="done")

    assert result["workflow_name"] == "demo"
    assert result["workflow_status"]["problem_stats"]["successful"] == 2
    assert (tmp_path / "results" / "problem_p1_result.json").exists()
    assert (tmp_path / "demo_status.json").exists()
    assert composer.get_workflow_status()["modules_executed"] == 1
    assert "WorkflowComposer" in repr(composer)


def test_workflow_composer_handles_module_run_exceptions(tmp_path):
    module = ExplodingWorkflowModule("explode", model="gpt-5.4-mini")
    composer = WorkflowComposer(
        name="demo",
        output_dir=tmp_path,
        modules=[module],
        config={"show_progress": False},
    )

    result = composer.run(PhysicalDataset(problems=[PhysicsProblem(problem_id="p1", question="Q")]))

    assert result["workflow_status"]["problem_stats"]["failed"] == 1
    assert composer.get_module_status("explode")["failed_problems"] == 1


def test_workflow_composer_requires_modules(tmp_path):
    composer = WorkflowComposer(
        name="demo",
        output_dir=tmp_path,
        config={"show_progress": False},
    )

    with pytest.raises(ValueError, match="No modules in workflow"):
        composer.run(_dataset())


def test_workflow_composer_safe_to_dict_and_getters(tmp_path):
    composer = WorkflowComposer(
        name="demo",
        output_dir=tmp_path,
        config={"show_progress": False},
    )

    assert composer._safe_to_dict(None) is None
    assert composer._safe_to_dict({"x": 1}) == {"x": 1}
    assert composer._safe_to_dict(SimpleNamespace()) == "namespace()"
    assert composer.get_module_status("missing") is None


def test_workflow_composer_progress_bar_and_error_paths(tmp_path, monkeypatch):
    module = DummyWorkflowModule("dummy", model="gpt-5.4-mini")
    composer = WorkflowComposer(
        name="demo",
        output_dir=tmp_path,
        modules=[module],
        config={"show_progress": True},
    )

    class FakeTqdm:
        def __init__(self, iterable, **_kwargs):
            self._items = list(iterable)
            self.postfixes = []
            self.closed = False

        def __iter__(self):
            return iter(self._items)

        def set_postfix(self, values):
            self.postfixes.append(values)

        def close(self):
            self.closed = True

    fake_bar = FakeTqdm(_dataset())
    monkeypatch.setattr(workflow_composer_module, "TQDM_AVAILABLE", True)
    monkeypatch.setattr(workflow_composer_module, "tqdm", lambda iterable, **kwargs: fake_bar)

    result = composer.run(_dataset(), flag="done")

    assert result["workflow_status"]["problem_stats"]["successful"] == 2
    assert fake_bar.postfixes
    assert fake_bar.closed is True


def test_workflow_composer_run_records_workflow_errors_and_reset(tmp_path, monkeypatch):
    module = DummyWorkflowModule("dummy", model="gpt-5.4-mini")
    composer = WorkflowComposer(
        name="demo",
        output_dir=tmp_path,
        modules=[module],
        config={"show_progress": False},
    )

    monkeypatch.setattr(
        composer,
        "_process_problem_through_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken pipeline")),
    )

    result = composer.run(_dataset())
    assert result["workflow_status"]["workflow_errors"] == [
        "Workflow execution failed: broken pipeline"
    ]

    composer.reset()
    assert composer.get_workflow_status()["workflow_errors"] == []


def test_workflow_composer_save_helpers_log_errors(tmp_path, monkeypatch):
    module = DummyWorkflowModule("dummy", model="gpt-5.4-mini")
    composer = WorkflowComposer(
        name="demo",
        output_dir=tmp_path,
        modules=[module],
        config={"show_progress": False},
    )

    def explode_open(*_args, **_kwargs):
        raise OSError("no write")

    monkeypatch.setattr(workflow_composer_module, "open", explode_open, raising=False)

    with patch.object(composer.logger, "error") as mock_error:
        composer._save_problem_result("p1", {"status": "ok"})
        composer._save_workflow_results([{"status": "ok"}])
        composer._save_workflow_status()

    assert mock_error.call_count == 3
