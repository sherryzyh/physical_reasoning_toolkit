#!/usr/bin/env python3
"""Phase-4 cross-commit behavior-diff harness.

Runs the adversarial corpus through prkit at a PRE commit (the pre-Phase-4 Run B
HEAD) and a POST commit (the current HEAD), capturing a deterministic, offline
fingerprint per entry, and emits every ``(input, original_output, new_output)``
triplet where the two differ -- the review artifact the owner ratifies.

Mechanics: the *same* driver + corpus run against both commits; only prkit
differs. Each commit is checked out into a throwaway detached worktree, and the
capture subprocess imports prkit from ``PYTHONPATH=<worktree>/src`` (which
precedes the editable-install ``.pth``, so the worktree wins -- the capture
asserts this). Both runs share the one venv, so pint is identical and only the
prkit source under test changes.

Usage (from the repo root):
    .venv/bin/python tools/phase4_diff.py                 # orchestrate PRE vs HEAD
    .venv/bin/python tools/phase4_diff.py --pre <sha>     # override PRE commit
    .venv/bin/python tools/phase4_diff.py --capture ...   # internal capture mode
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CORPUS = _REPO / "tests" / "fixtures" / "phase4_corpus.jsonl"
_OUT_MD = _REPO / "internal" / "PHASE4_BEHAVIOR_DIFF.md"
_OUT_JSONL = _REPO / "internal" / "PHASE4_BEHAVIOR_DIFF.jsonl"

# Precision-contract categories first (any diff there is a regression), then the
# intended-change categories.
_CATEGORY_ORDER = ["A", "B", "C", "E", "F", "I", "D", "G", "H", "J"]
_PRECISION_CONTRACT = {"A", "B", "C", "E", "F", "I"}


# --------------------------------------------------------------------------- #
# Capture mode: import prkit from PYTHONPATH and fingerprint each corpus entry. #
# --------------------------------------------------------------------------- #


def _load_corpus(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _capture(corpus_path: Path, out_path: Path, expect_root: str | None) -> None:
    import prkit  # resolved via PYTHONPATH

    if expect_root is not None:
        resolved = str(Path(prkit.__file__).resolve())
        assert resolved.startswith(
            str(Path(expect_root).resolve())
        ), f"prkit imported from {resolved}, expected under {expect_root}"

    from prkit.semantics import compare_protocol_answers, normalize_physics_answer
    from prkit.semantics.quantities.surface import _try_parse_physical_quantity

    def _record(text: str) -> object:
        try:
            answer = normalize_physics_answer(text)
        except Exception as exc:  # noqa: BLE001 - capture, never abort
            return f"<error:{type(exc).__name__}>"
        kind = getattr(answer.object_kind, "value", str(answer.object_kind))
        return {
            "object_kind": kind,
            "canonical_text": answer.canonical_text,
            "unit": answer.unit,
            "numeric_value": answer.numeric_value,
        }

    def _parse(text: str) -> object:
        try:
            return _try_parse_physical_quantity(text)
        except Exception as exc:  # noqa: BLE001
            return f"<error:{type(exc).__name__}>"

    def _verdict(pred: str, ref: str) -> object:
        try:
            cmp = compare_protocol_answers(
                normalize_physics_answer(pred), normalize_physics_answer(ref)
            )
            return [bool(cmp.equivalent), cmp.comparison_mode]
        except Exception as exc:  # noqa: BLE001
            return f"<error:{type(exc).__name__}>"

    captured: dict[str, dict] = {}
    for row in _load_corpus(corpus_path):
        rid = str(row["id"])
        if row["kind"] == "single":
            text = str(row["input"])
            captured[rid] = {
                "parse": _parse(text),
                "record": _record(text),
            }
        else:
            captured[rid] = {
                "verdict": _verdict(str(row["pred"]), str(row["ref"])),
            }
    out_path.write_text(json.dumps(captured, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestrator mode: two worktrees, run capture against each, diff, write docs. #
# --------------------------------------------------------------------------- #


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(_REPO), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _capture_at_commit(commit: str, label: str, tmpdir: Path) -> dict:
    worktree = tmpdir / f"prkit-{label}"
    _git("worktree", "add", "--detach", str(worktree), commit)
    try:
        out_json = tmpdir / f"{label}.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(worktree / "src")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--capture",
                "--corpus",
                str(_CORPUS),
                "--out",
                str(out_json),
                "--expect-root",
                str(worktree / "src"),
            ],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        return json.loads(out_json.read_text(encoding="utf-8"))
    finally:
        _git("worktree", "remove", "--force", str(worktree))


def _fingerprint(entry: dict) -> object:
    """The diff key: parse result for singles, verdict for pairs."""

    return entry["parse"] if "parse" in entry else entry.get("verdict")


def _fmt(value: object) -> str:
    if value is None:
        return "∅ (None)"
    return f"`{json.dumps(value, ensure_ascii=False)}`"


def _orchestrate(pre_commit: str) -> int:
    post_commit = _git("rev-parse", "HEAD")
    corpus = _load_corpus(_CORPUS)
    by_id = {str(r["id"]): r for r in corpus}

    with tempfile.TemporaryDirectory(prefix="phase4-diff-") as tmp:
        tmpdir = Path(tmp)
        print(f"PRE  = {pre_commit}\nPOST = {post_commit}\ncorpus rows = {len(corpus)}")
        pre = _capture_at_commit(pre_commit, "pre", tmpdir)
        post = _capture_at_commit(post_commit, "post", tmpdir)

    diffs: list[dict] = []
    for rid, row in by_id.items():
        before = _fingerprint(pre.get(rid, {}))
        after = _fingerprint(post.get(rid, {}))
        if before != after:
            diffs.append(
                {
                    "id": rid,
                    "category": row["category"],
                    "kind": row["kind"],
                    "input": row.get("input")
                    or f'{row.get("pred")} || {row.get("ref")}',
                    "expectation": row["expectation"],
                    "original_output": before,
                    "new_output": after,
                }
            )

    diffs.sort(key=lambda d: (_CATEGORY_ORDER.index(d["category"]), d["id"]))

    counts_total = {c: 0 for c in _CATEGORY_ORDER}
    counts_diff = {c: 0 for c in _CATEGORY_ORDER}
    for row in corpus:
        counts_total[row["category"]] += 1
    for d in diffs:
        counts_diff[d["category"]] += 1

    # The precision-contract gate covers *asserted* rows (SYMBOLIC/QUANTITY/
    # MATCH/NOMATCH) in A,B,C,E,F,I. OBSERVE rows -- even in those categories,
    # e.g. the category-I uppercase-vs-variable pairs -- are owner-reviewed
    # observations, not regressions, so they do not trip the gate.
    contract_violations = [
        d
        for d in diffs
        if d["category"] in _PRECISION_CONTRACT and d["expectation"] != "OBSERVE"
    ]

    _write_artifacts(
        pre_commit,
        post_commit,
        corpus,
        diffs,
        counts_total,
        counts_diff,
        len(contract_violations),
    )

    print(
        f"\ntotal diffs = {len(diffs)} | "
        f"precision-contract violations = {len(contract_violations)}"
    )
    for c in _CATEGORY_ORDER:
        flag = " <- PRECISION CONTRACT" if c in _PRECISION_CONTRACT else ""
        print(f"  {c}: {counts_diff[c]:3d} diffs / {counts_total[c]:3d} rows{flag}")
    for d in contract_violations:
        print(f"  !! contract violation: {d['id']} {d['input']!r}")
    print(f"\nwrote {_OUT_MD}\nwrote {_OUT_JSONL}")
    return 1 if contract_violations else 0


def _write_artifacts(
    pre_commit: str,
    post_commit: str,
    corpus: list[dict],
    diffs: list[dict],
    counts_total: dict[str, int],
    counts_diff: dict[str, int],
    contract_violations: int,
) -> None:
    _OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Phase 4 behavior diff (PRE vs POST)")
    lines.append("")
    lines.append(f"- PRE  commit: `{pre_commit}`")
    lines.append(f"- POST commit: `{post_commit}`")
    lines.append(f"- corpus rows: {len(corpus)}")
    lines.append(f"- total diffs: {len(diffs)}")
    lines.append(
        f"- precision-contract violations (asserted rows in A,B,C,E,F,I): "
        f"**{contract_violations}**"
    )
    lines.append(
        "  (OBSERVE rows -- e.g. the category-I uppercase-vs-variable pairs -- are "
        "owner-reviewed observations, not regressions.)"
    )
    lines.append("")
    lines.append("## Per-category counts")
    lines.append("")
    lines.append("| category | rows | diffs | role |")
    lines.append("|---|---|---|---|")
    for c in _CATEGORY_ORDER:
        role = "precision contract" if c in _PRECISION_CONTRACT else "intended change"
        lines.append(f"| {c} | {counts_total[c]} | {counts_diff[c]} | {role} |")
    lines.append("")
    lines.append("## Triplets (precision-contract categories first)")
    lines.append("")
    if not diffs:
        lines.append("_No behavior diffs._")
    else:
        lines.append(
            "| id | category | input | original output | new output | expectation |"
        )
        lines.append("|---|---|---|---|---|---|")
        for d in diffs:
            inp = str(d["input"]).replace("|", "\\|")
            lines.append(
                f"| {d['id']} | {d['category']} | `{inp}` | "
                f"{_fmt(d['original_output'])} | {_fmt(d['new_output'])} | "
                f"{d['expectation']} |"
            )
    lines.append("")
    _OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with _OUT_JSONL.open("w", encoding="utf-8") as handle:
        for d in diffs:
            handle.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")


def _default_pre_commit() -> str:
    """Resolve the PRE baseline to this branch's point off ``main``.

    Using the ``git merge-base`` keeps the default stable as history advances,
    rather than rotting against a hardcoded SHA.
    """

    result = subprocess.run(
        ["git", "merge-base", "HEAD", "main"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="internal capture mode")
    parser.add_argument("--corpus", type=Path, default=_CORPUS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expect-root", type=str, default=None)
    parser.add_argument(
        "--pre",
        type=str,
        default=None,
        help="PRE commit (default: git merge-base HEAD main)",
    )
    args = parser.parse_args()

    if args.capture:
        if args.out is None:
            parser.error("--capture requires --out")
        _capture(args.corpus, args.out, args.expect_root)
        return 0
    return _orchestrate(args.pre or _default_pre_commit())


if __name__ == "__main__":
    raise SystemExit(main())
