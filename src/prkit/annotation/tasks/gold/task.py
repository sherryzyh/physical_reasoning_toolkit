"""The ``gold`` annotation task: a human expert authors gold attribute labels.

Currently only the ``domain`` subtype is implemented. Problems come from a registered
dataset (via :class:`DatasetHub`) or a directory of ``problem_*.json`` files, and gold
records are written to a dedicated, non-destructive output store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from ....core.domain.physics_problem import PhysicsProblem
from ...base import AnnotationTask, list_problem_jsons, load_json
from . import domain as domain_subtype
from .terminal import run_terminal_domain_annotation


class GoldAnnotationTask(AnnotationTask):
    """Expert-authored gold labels for intrinsic problem attributes."""

    name: ClassVar[str] = "gold"
    subtypes: ClassVar[tuple[str, ...]] = (domain_subtype.SUBTYPE,)

    def run(  # type: ignore[override]
        self,
        subtype: str | None = None,
        *,
        dataset: str | None = None,
        problems_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        annotator: str = "",
        **_: Any,
    ) -> int:
        """Run gold annotation for *subtype* over a dataset or a problems directory."""
        if subtype is None:
            raise ValueError(
                f"The 'gold' task needs a subtype. Supported: {', '.join(self.subtypes)}."
            )
        if subtype not in self.subtypes:
            raise ValueError(
                f"Unknown gold subtype {subtype!r}. Supported: {', '.join(self.subtypes)}."
            )

        problems, source_name = self._load_problems(dataset, problems_dir)
        if not problems:
            raise ValueError("No problems to annotate.")

        out = (
            Path(output_dir)
            if output_dir is not None
            else Path("annotations") / "gold" / subtype / source_name
        )
        return run_terminal_domain_annotation(problems, out, annotator=annotator)

    @staticmethod
    def _load_problems(
        dataset: str | None, problems_dir: str | Path | None
    ) -> tuple[list[PhysicsProblem], str]:
        """Resolve the problem list and a short source name for the output path."""
        if dataset:
            from ....datasets import DatasetHub  # local import: heavy optional dep

            loaded = DatasetHub.load(dataset)
            return list(loaded), dataset
        if problems_dir:
            folder = Path(problems_dir)
            if not folder.is_dir():
                raise ValueError(f"Not a directory: {folder}")
            problems = [
                PhysicsProblem.from_dict(load_json(path))
                for path in list_problem_jsons(folder)
            ]
            return problems, folder.name
        raise ValueError("Provide a dataset name or a problems directory.")
