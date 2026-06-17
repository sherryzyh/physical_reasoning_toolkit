"""``prkit annotate`` subcommand: parser registration and dispatch.

Kept separate from :mod:`prkit.cli` so the top-level CLI stays a thin shell. Building the
parser pulls in nothing heavy; the orchestrator is imported lazily only when a task runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_annotate_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``annotate`` (with ``gold`` / ``correctness`` task sub-subparsers)."""
    annotate = subparsers.add_parser(
        "annotate", help="Run a human annotation task over a dataset."
    )
    tasks = annotate.add_subparsers(dest="annotate_task")

    gold = tasks.add_parser("gold", help="Expert authors gold attribute labels.")
    gold.add_argument(
        "subtype", choices=["domain"], help="Attribute to annotate (domain)."
    )
    gold.add_argument("dataset", nargs="?", help="Dataset name to load via DatasetHub.")
    gold.add_argument(
        "--problems-dir",
        type=Path,
        help="Directory of problem_*.json to annotate instead of a dataset.",
    )
    gold.add_argument(
        "--output-dir",
        type=Path,
        help="Where to write gold records (default: ./annotations/gold/<subtype>/<source>).",
    )
    gold.add_argument(
        "--annotator", default="", help="Annotator name to stamp on records."
    )

    correctness = tasks.add_parser(
        "correctness", help="Judge model answers vs. the gold reference (Streamlit UI)."
    )
    correctness.add_argument(
        "answers_dir", help="Directory of model-answer problem_*.json files."
    )
    correctness.add_argument(
        "--annotator", default="", help="Annotator name to stamp on labels."
    )
    correctness.add_argument("--port", type=int, help="Streamlit server port.")
    correctness.add_argument(
        "--no-browser",
        action="store_true",
        help="Run Streamlit headless (do not open a browser).",
    )


def handle_annotate(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``annotate`` invocation; return a process exit code."""
    from .orchestrator import AnnotationOrchestrator

    task = getattr(args, "annotate_task", None)
    if task is None:
        print("Specify an annotation task: gold | correctness", file=sys.stderr)
        return 2

    orchestrator = AnnotationOrchestrator()
    if task == "gold":
        return orchestrator.dispatch(
            "gold",
            subtype=args.subtype,
            dataset=args.dataset,
            problems_dir=args.problems_dir,
            output_dir=args.output_dir,
            annotator=args.annotator,
        )
    if task == "correctness":
        return orchestrator.dispatch(
            "correctness",
            answers_dir=args.answers_dir,
            annotator=args.annotator,
            port=args.port,
            no_browser=args.no_browser,
        )
    print(f"Unknown annotation task: {task}", file=sys.stderr)
    return 2
