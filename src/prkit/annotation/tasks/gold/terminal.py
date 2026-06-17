"""Minimal, resumable in-terminal annotator for gold-domain labels.

This is the stop-gap interface until a richer UI lands. It walks the expert through each
problem, shows the existing dataset ``domain`` as a default, and writes one
``problem_<id>.json`` record per problem. IO is injectable so it can be unit-tested.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from ....core.domain.physics_problem import PhysicsProblem
from ...base import load_json, save_json
from .domain import DomainRecord, domain_choices, normalize_domain

_QUESTION_PREVIEW_CHARS = 1500


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _record_path(output_dir: Path, problem_id: str) -> Path:
    return output_dir / f"problem_{problem_id}.json"


def _is_done(output_dir: Path, problem_id: str) -> bool:
    """True if a non-empty gold-domain record already exists for *problem_id*."""
    path = _record_path(output_dir, problem_id)
    if not path.is_file():
        return False
    try:
        data = load_json(path)
    except (OSError, ValueError):
        return False
    return bool(str(data.get("gold_domain", "")).strip())


def run_terminal_domain_annotation(
    problems: Sequence[PhysicsProblem],
    output_dir: Path,
    annotator: str = "",
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    now_fn: Callable[[], str] = _now_iso,
) -> int:
    """Drive a terminal loop labelling each problem's gold physics domain.

    Args:
        problems: Problems to annotate, in display order.
        output_dir: Directory where ``problem_<id>.json`` records are written.
        annotator: Name stamped onto each record.
        input_fn/print_fn/now_fn: Injected IO/clock for testing.

    Returns:
        Process exit code (always ``0``; quitting early is not an error).

    Controls per problem: a menu number selects a domain, Enter accepts the suggested
    default, free text is normalized to a domain, ``s`` skips, ``b`` goes back, ``q`` quits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    choices = domain_choices()
    total = len(problems)

    already = sum(1 for p in problems if _is_done(output_dir, p.problem_id))
    print_fn(
        f"Gold domain annotation — {total} problem(s), {already} already labelled."
    )
    print_fn(f"Output: {output_dir}")
    print_fn("")

    def advance(start: int) -> int:
        index = start
        while index < total and _is_done(output_dir, problems[index].problem_id):
            index += 1
        return index

    i = advance(0)
    if i >= total:
        print_fn("Everything is already labelled. Nothing to do.")
        return 0

    while 0 <= i < total:
        problem = problems[i]
        pid = problem.problem_id
        suggested = problem.get_domain_name()
        suggested = "" if suggested == "Unknown" else suggested

        print_fn("=" * 72)
        labelled = sum(1 for p in problems if _is_done(output_dir, p.problem_id))
        print_fn(f"[{i + 1}/{total}] problem_id={pid}  (labelled {labelled}/{total})")
        question = (problem.question or "").strip()
        if len(question) > _QUESTION_PREVIEW_CHARS:
            question = question[:_QUESTION_PREVIEW_CHARS] + " …"
        print_fn(f"\nQuestion:\n{question}\n")
        if problem.image_path:
            print_fn(f"[has {len(problem.image_path)} image(s)]")
        print_fn(f"Suggested (dataset domain): {suggested or '(none)'}\n")

        for n, choice in enumerate(choices, start=1):
            marker = "  <- suggested" if choice == normalize_domain(suggested) else ""
            print_fn(f"  {n:2d}. {choice}{marker}")
        print_fn(
            "\nEnter number / domain text · [Enter]=accept suggested · "
            "s=skip · b=back · q=quit"
        )

        raw = input_fn("> ").strip()

        if raw.lower() == "q":
            print_fn("Stopping (progress saved).")
            break
        if raw.lower() == "s":
            print_fn("Skipped.\n")
            i = advance(i + 1)
            continue
        if raw.lower() == "b":
            i = max(0, i - 1)
            continue

        if raw == "":
            if not suggested:
                print_fn("No suggested default for this problem — pick a value.\n")
                continue
            gold = normalize_domain(suggested)
        elif raw.isdigit():
            choice_index = int(raw)
            if not 1 <= choice_index <= len(choices):
                print_fn("Number out of range.\n")
                continue
            gold = choices[choice_index - 1]
        else:
            gold = normalize_domain(raw)

        record = DomainRecord(
            problem_id=pid,
            gold_domain=gold,
            suggested_domain=suggested,
            question_preview=question,
            annotator=annotator,
            timestamp=now_fn(),
        )
        save_json(_record_path(output_dir, pid), record.to_dict())
        print_fn(f"Saved gold_domain={gold!r}.\n")
        i = advance(i + 1)

    final = sum(1 for p in problems if _is_done(output_dir, p.problem_id))
    print_fn(f"Done — {final}/{total} labelled in {output_dir}.")
    return 0
