#!/usr/bin/env python3
"""Replay stored answer-comparison records through the current equivalence engine.

The audit set (human-labelled correctness) lives in the consumer repo, not in PRKit
(toolkit independence -- see ``src/prkit/semantics/comparison/METHODOLOGY.md``). This tool
therefore locates nothing: both inputs are required arguments.

It re-runs :func:`compare_protocol_answers` over records that carry the exact engine inputs
(reference semantics, prediction semantics, evaluation contract, policy mode), scores the
verdicts against human labels, and -- given a baseline written by an earlier run -- reports
which pairs newly accept and which newly reject. A change to the relation is only a widening
if the newly-rejected set is empty.

Usage::

    python tools/audit_replay.py \
        --records  <dir of per-problem *.json records> \
        --annotations <*_annotations.json with human verdicts> \
        [--baseline before.json] [--out after.json] [--timeout 20]
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class ComparisonTimeout(BaseException):
    """Raised when a single comparison exceeds the per-record wall-clock budget.

    Deliberately a ``BaseException``: the engine catches ``Exception`` broadly around
    its symbolic work, so an ordinary exception raised from the signal handler is
    swallowed and the budget never takes effect.
    """


@contextmanager
def _time_budget(seconds: float) -> Iterator[None]:
    """Abort the enclosed block with :class:`ComparisonTimeout` after *seconds*."""

    if seconds <= 0:
        yield
        return

    def _fire(signum: int, frame: Any) -> None:
        raise ComparisonTimeout(f"exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--records",
        type=Path,
        required=True,
        help="Directory of per-problem comparison records (*.json).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Human annotation export with an 'annotations' list of {problem_id, verdict}.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Result file from an earlier run, to diff the accepted set against.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Write this run's per-problem results here."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-record wall-clock budget in seconds (0 disables).",
    )
    return parser


def load_human_labels(path: Path) -> dict[str, bool]:
    """Return ``{problem_id: human verdict}`` from an annotation export."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["problem_id"]): bool(row["verdict"])
        for row in payload["annotations"]
        if row.get("verdict") is not None
    }


def replay(records_dir: Path, timeout: float) -> list[dict[str, Any]]:
    """Re-run every stored record through the current engine."""

    from prkit.semantics import compare_protocol_answers

    rows: list[dict[str, Any]] = []
    for path in sorted(records_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        problem_id = str(record["problem"]["problem_id"])
        started = time.monotonic()
        try:
            with _time_budget(timeout):
                comparison = compare_protocol_answers(
                    record["prediction_answer_semantics"],
                    record["reference_answer_semantics"],
                    contract=record.get("evaluation_contract"),
                    policy_mode=record.get("policy_mode"),
                )
            row = {
                "problem_id": problem_id,
                "equivalent": bool(comparison.equivalent),
                "comparison_mode": comparison.comparison_mode,
                "diagnostics": list(comparison.diagnostics),
                "error": None,
            }
        except ComparisonTimeout as exc:
            row = {
                "problem_id": problem_id,
                "equivalent": False,
                "comparison_mode": "TIMEOUT",
                "diagnostics": [],
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - a crash is a verdict of "not equivalent"
            row = {
                "problem_id": problem_id,
                "equivalent": False,
                "comparison_mode": "ERROR",
                "diagnostics": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        row["recorded_equivalent"] = bool(record.get("comparison", {}).get("equivalent"))
        row["seconds"] = round(time.monotonic() - started, 3)
        rows.append(row)
        print(
            f"[{len(rows):>4}] {problem_id:>8} {str(row['equivalent']):>5} "
            f"{row['comparison_mode']:<32} {row['seconds']}s",
            file=sys.stderr,
        )
    return rows


def confusion(rows: list[dict[str, Any]], human: dict[str, bool]) -> dict[str, list[str]]:
    """Bucket every labelled row into TP / FP / TN / FN, keeping the problem ids."""

    buckets: dict[str, list[str]] = {"TP": [], "FP": [], "TN": [], "FN": []}
    for row in rows:
        label = human.get(row["problem_id"])
        if label is None:
            continue
        predicted = row["equivalent"]
        if label and predicted:
            buckets["TP"].append(row["problem_id"])
        elif label:
            buckets["FN"].append(row["problem_id"])
        elif predicted:
            buckets["FP"].append(row["problem_id"])
        else:
            buckets["TN"].append(row["problem_id"])
    return buckets


def _sorted_ids(ids: list[str]) -> list[str]:
    """Sort problem ids numerically when they are numeric, lexically otherwise."""

    return sorted(ids, key=lambda i: (0, int(i)) if i.isdigit() else (1, 0, i))


def main(argv: list[str] | None = None) -> int:
    """Run the replay and print the confusion matrix and the accepted-set diff."""

    args = build_parser().parse_args(argv)
    human = load_human_labels(args.annotations)
    rows = replay(args.records, args.timeout)
    buckets = confusion(rows, human)

    tp, fp, tn, fn = (len(buckets[k]) for k in ("TP", "FP", "TN", "FN"))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 1.0
    print(f"records {len(rows)}  labelled {tp + fp + tn + fn}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"recall={recall:.1%}  precision={precision:.1%}")
    print(f"FN ids: {' '.join(_sorted_ids(buckets['FN']))}")
    print(f"FP ids: {' '.join(_sorted_ids(buckets['FP']))}")

    slow = [r for r in rows if r["seconds"] >= 1.0]
    if slow:
        worst = sorted(slow, key=lambda r: -r["seconds"])[:5]
        print("slowest: " + ", ".join(f"{r['problem_id']}={r['seconds']}s" for r in worst))
    stuck = [r for r in rows if r["comparison_mode"] in {"TIMEOUT", "ERROR"}]
    if stuck:
        print(f"timeouts/errors ({len(stuck)}): " + ", ".join(
            f"{r['problem_id']}:{r['comparison_mode']}" for r in stuck
        ))

    if args.baseline and args.baseline.exists():
        before = {r["problem_id"]: r for r in json.loads(args.baseline.read_text("utf-8"))}
        gained = [r["problem_id"] for r in rows
                  if r["equivalent"] and not before.get(r["problem_id"], {}).get("equivalent")]
        lost = [r["problem_id"] for r in rows
                if not r["equivalent"] and before.get(r["problem_id"], {}).get("equivalent")]
        print(f"newly accepted ({len(gained)}): {' '.join(_sorted_ids(gained))}")
        print(f"newly REJECTED ({len(lost)}): {' '.join(_sorted_ids(lost))}")
        if lost:
            print("the accepted set shrank -- this is not a widening")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
