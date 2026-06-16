#!/usr/bin/env python3
"""CLI tool to backfill sampling-format problem JSONs from inference outputs for a given problem subset."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prkit.core.project_env import (
    find_toolkit_root,
    find_uq_root,
    find_uqps_root,
)

DEFAULT_REPO_ROOT = find_toolkit_root(__file__) or Path(__file__).resolve().parents[4]
DEFAULT_UQ_ROOT = find_uq_root(__file__) or (
    DEFAULT_REPO_ROOT.parent / "uncertainty_quantification_physical_reasoning"
)
DEFAULT_SAMPLING_ROOT = DEFAULT_UQ_ROOT / "experiment_results" / "sampling"
DEFAULT_INFERENCE_ROOT = (
    DEFAULT_UQ_ROOT / "experiment_results" / "inference" / "response_with_answer_tag"
)
_default_uqps_root = find_uqps_root(__file__)
DEFAULT_CANONICAL_SAMPLING_ROOT = (
    (_default_uqps_root / "baselines" / "sampling")
    if _default_uqps_root is not None
    else (
        DEFAULT_REPO_ROOT.parent
        / "uncertainty_quantification_via_physics_semantics"
        / "baselines"
        / "sampling"
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill sampling-format problem JSONs from response_with_answer_tag "
            "inference outputs for a subset of problem IDs."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset prefix used in directory names, e.g. seephys.",
    )
    parser.add_argument(
        "--subset-path",
        type=Path,
        required=True,
        help="JSON array of target problem IDs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names, e.g. gpt-5.4 gpt-4.1.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help=f"Toolkit repo root for reporting/debugging (default: {DEFAULT_REPO_ROOT}).",
    )
    parser.add_argument(
        "--sampling-root",
        type=Path,
        default=None,
        help="Override experiment_results/sampling root.",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=None,
        help="Override experiment_results/inference/response_with_answer_tag root.",
    )
    parser.add_argument(
        "--canonical-sampling-root",
        type=Path,
        default=None,
        help="Override uncertainty_quantification_via_physics_semantics/baselines/sampling root.",
    )
    parser.add_argument(
        "--missing-ids-dir",
        type=Path,
        default=None,
        help="Directory for generated missing-id JSON files (default: subset dir).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write missing sampling files to the configured destinations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-existing output files when --apply is set.",
    )
    return parser.parse_args()


def load_problem_ids(path: Path) -> list[str]:
    """Load a JSON array of problem ID strings from *path*."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON array")
    return [str(item).strip() for item in raw if str(item).strip()]


def problem_ids_from_dir(path: Path) -> set[str]:
    """Return the set of problem IDs found as ``problem_<id>.json`` files in *path*."""
    if not path.is_dir():
        return set()
    ids: set[str] = set()
    for file_path in path.glob("problem_*.json"):
        match = re.match(r"problem_(.+)\.json$", file_path.name)
        if match:
            ids.add(match.group(1))
    return ids


def sort_problem_ids(problem_ids: set[str] | list[str]) -> list[str]:
    """Sort problem IDs numerically when all-digit, lexicographically otherwise."""

    def key(value: str) -> tuple[int, Any]:
        return (0, int(value)) if value.isdigit() else (1, value)

    return sorted((str(item) for item in problem_ids), key=key)


def load_inference_doc(path: Path) -> tuple[str, str | None]:
    """Load an inference output JSON and return ``(raw_model_response, extracted_model_answer)``."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")

    raw = (
        data.get("raw_model_response")
        or data.get("model_response")
        or data.get("response")
        or data.get("full_response")
        or ""
    )
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        raw = str(raw)

    answer = data.get("extracted_model_answer")
    if answer is None:
        answer = data.get("model_answer")
    if answer is not None and not isinstance(answer, str):
        answer = str(answer)
    return raw, answer


def build_sampling_doc(
    *,
    dataset: str,
    model: str,
    problem_id: str,
    raw_model_response: str,
    extracted_model_answer: str | None,
) -> dict[str, Any]:
    """Build a single-response sampling document dict from inference output fields."""
    return {
        "problem_id": problem_id,
        "database_name": dataset,
        "model": model,
        "responses": [
            {
                "index_no": 0,
                "model": model,
                "raw_model_response": raw_model_response,
                "extracted_model_answer": extracted_model_answer,
                "source_model": "dataset",
            }
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    """Write *payload* as pretty-printed JSON to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_sampling_doc(path: Path, payload: dict[str, Any], overwrite: bool) -> bool:
    """Write a sampling document to *path*; return ``False`` (skip) when it already exists and *overwrite* is false."""
    if path.exists() and not overwrite:
        return False
    write_json(path, payload)
    return True


def command_for_model(
    script_path: Path,
    dataset: str,
    subset_path: Path,
    model: str,
    report_path: Path | None,
) -> str:
    """Build the shell command string to re-run this script for a single model."""
    parts = [
        "python",
        str(script_path),
        "--dataset",
        dataset,
        "--subset-path",
        str(subset_path),
        "--models",
        model,
        "--apply",
    ]
    if report_path is not None:
        parts.extend(["--report-path", str(report_path)])
    return " ".join(shlex.quote(part) for part in parts)


def model_report(
    *,
    dataset: str,
    model: str,
    subset_ids: list[str],
    subset_path: Path,
    missing_ids_dir: Path,
    inference_root: Path,
    sampling_root: Path,
    canonical_sampling_root: Path | None,
    apply: bool,
    overwrite: bool,
    report_path: Path | None,
) -> dict[str, Any]:
    """Compute the backfill status for one model and optionally write missing sampling documents."""
    run_name = f"{dataset}_{model}"
    inference_dir = inference_root / run_name
    sampling_dir = sampling_root / run_name
    canonical_dir = (
        canonical_sampling_root / run_name if canonical_sampling_root else None
    )

    subset_set = set(subset_ids)
    existing_sampling_ids = problem_ids_from_dir(sampling_dir)
    existing_subset_ids = subset_set & existing_sampling_ids
    extra_sampling_ids = existing_sampling_ids - subset_set
    missing_sampling_ids = sort_problem_ids(subset_set - existing_sampling_ids)

    inference_ids = problem_ids_from_dir(inference_dir)
    missing_inference_ids = sort_problem_ids(set(missing_sampling_ids) - inference_ids)

    subset_stem = subset_path.stem
    missing_sampling_ids_path = (
        missing_ids_dir / f"{subset_stem}_missing_in_sampling__{model}.json"
    )
    write_json(missing_sampling_ids_path, missing_sampling_ids)

    missing_inference_ids_path: Path | None = None
    if missing_inference_ids:
        missing_inference_ids_path = (
            missing_ids_dir
            / f"{subset_stem}_missing_in_response_with_answer_tag__{model}.json"
        )
        write_json(missing_inference_ids_path, missing_inference_ids)

    backfilled_sampling = 0
    backfilled_canonical = 0
    skipped_existing_sampling = 0
    skipped_existing_canonical = 0

    if apply:
        for problem_id in missing_sampling_ids:
            inference_path = inference_dir / f"problem_{problem_id}.json"
            if not inference_path.exists():
                continue

            raw_model_response, extracted_model_answer = load_inference_doc(
                inference_path
            )
            payload = build_sampling_doc(
                dataset=dataset,
                model=model,
                problem_id=problem_id,
                raw_model_response=raw_model_response,
                extracted_model_answer=extracted_model_answer,
            )

            sampling_path = sampling_dir / f"problem_{problem_id}.json"
            if write_sampling_doc(sampling_path, payload, overwrite=overwrite):
                backfilled_sampling += 1
            else:
                skipped_existing_sampling += 1

            if canonical_dir is not None:
                canonical_path = canonical_dir / f"problem_{problem_id}.json"
                if write_sampling_doc(canonical_path, payload, overwrite=overwrite):
                    backfilled_canonical += 1
                else:
                    skipped_existing_canonical += 1

    return {
        "run_name": run_name,
        "subset_problem_count": len(subset_ids),
        "subset_problem_ids_present_in_sampling_count": len(existing_subset_ids),
        "extra_sampling_problem_ids_count": len(extra_sampling_ids),
        "extra_sampling_problem_ids": sort_problem_ids(extra_sampling_ids),
        "missing_sampling_problem_ids_count": len(missing_sampling_ids),
        "missing_sampling_problem_ids": missing_sampling_ids,
        "missing_sampling_problem_ids_file": str(missing_sampling_ids_path),
        "missing_inference_problem_ids_count": len(missing_inference_ids),
        "missing_inference_problem_ids": missing_inference_ids,
        "missing_inference_problem_ids_file": (
            str(missing_inference_ids_path) if missing_inference_ids_path else None
        ),
        "batch_submission_required": bool(missing_inference_ids),
        "batch_submission_note": (
            "No new response_with_answer_tag batch is required; inference already exists "
            "for every missing sampling problem."
            if not missing_inference_ids
            else "Response_with_answer_tag inference is still missing for some problems."
        ),
        "inference_dir": str(inference_dir),
        "sampling_dir": str(sampling_dir),
        "canonical_sampling_dir": str(canonical_dir) if canonical_dir else None,
        "post_parse_sync_command": command_for_model(
            Path(__file__).resolve(),
            dataset,
            subset_path,
            model,
            report_path,
        ),
        "backfilled_sampling_count": backfilled_sampling,
        "backfilled_canonical_sampling_count": backfilled_canonical,
        "skipped_existing_sampling_count": skipped_existing_sampling,
        "skipped_existing_canonical_sampling_count": skipped_existing_canonical,
    }


def main() -> int:
    """Entry point: parse arguments, run backfill for each model, and output the report."""
    args = parse_args()

    repo_root = args.repo_root.resolve()
    subset_path = args.subset_path.resolve()
    sampling_root = (args.sampling_root or DEFAULT_SAMPLING_ROOT).resolve()
    inference_root = (args.inference_root or DEFAULT_INFERENCE_ROOT).resolve()
    canonical_sampling_root = (
        None
        if args.canonical_sampling_root == Path("")
        else (
            (args.canonical_sampling_root or DEFAULT_CANONICAL_SAMPLING_ROOT).resolve()
        )
    )
    missing_ids_dir = (args.missing_ids_dir or subset_path.parent).resolve()

    subset_ids = load_problem_ids(subset_path)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "subset_path": str(subset_path),
        "subset_problem_count": len(subset_ids),
        "apply": args.apply,
        "overwrite": args.overwrite,
        "repo_root": str(repo_root),
        "sampling_root": str(sampling_root),
        "inference_root": str(inference_root),
        "canonical_sampling_root": (
            str(canonical_sampling_root) if canonical_sampling_root else None
        ),
        "models": {},
    }

    for model in args.models:
        report["models"][model] = model_report(
            dataset=args.dataset,
            model=model,
            subset_ids=subset_ids,
            subset_path=subset_path,
            missing_ids_dir=missing_ids_dir,
            inference_root=inference_root,
            sampling_root=sampling_root,
            canonical_sampling_root=canonical_sampling_root,
            apply=args.apply,
            overwrite=args.overwrite,
            report_path=args.report_path.resolve() if args.report_path else None,
        )

    if args.report_path is not None:
        write_json(args.report_path.resolve(), report)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
