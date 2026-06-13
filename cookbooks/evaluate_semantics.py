#!/usr/bin/env python3
"""Evaluate saved reference and prediction semantics JSON artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from prkit.core import PRKitLogger
from prkit.semantics import (
    evaluate_saved_semantics,
    load_semantics_artifact,
    save_semantics_json,
)

logger = PRKitLogger.get_logger(__name__)

_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one pair of saved semantics JSON files, or two directories of saved semantics artifacts.",
    )
    parser.add_argument("--reference-file", type=str, default=None, help="Reference JSON artifact.")
    parser.add_argument("--prediction-file", type=str, default=None, help="Prediction JSON artifact.")
    parser.add_argument("--reference-dir", type=str, default=None, help="Directory of reference JSON artifacts.")
    parser.add_argument("--prediction-dir", type=str, default=None, help="Directory of prediction JSON artifacts.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation JSON files.",
    )
    args = parser.parse_args()

    single_file_mode = bool(args.reference_file and args.prediction_file)
    directory_mode = bool(args.reference_dir and args.prediction_dir)
    if single_file_mode == directory_mode:
        raise ValueError(
            "Provide either --reference-file with --prediction-file, "
            "or --reference-dir with --prediction-dir."
        )

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if single_file_mode:
        _evaluate_single_pair(
            reference_file=Path(args.reference_file),
            prediction_file=Path(args.prediction_file),
            output_dir=output_dir,
        )
        return

    _evaluate_directory_pair(
        reference_dir=Path(args.reference_dir),
        prediction_dir=Path(args.prediction_dir),
        output_dir=output_dir,
    )


def _evaluate_single_pair(
    *,
    reference_file: Path,
    prediction_file: Path,
    output_dir: Path,
) -> None:
    record = evaluate_saved_semantics(reference_file, prediction_file)
    output_path = output_dir / f"{_safe_filename(record.problem.problem_id)}.json"
    save_semantics_json(record, output_path)
    logger.info(
        "Problem %s equivalent=%s mode=%s",
        record.problem.problem_id,
        record.comparison.equivalent,
        record.comparison.comparison_mode,
    )
    logger.info("Saved evaluation record to %s", output_path)


def _evaluate_directory_pair(
    *,
    reference_dir: Path,
    prediction_dir: Path,
    output_dir: Path,
) -> None:
    reference_files = _collect_artifacts(reference_dir, expected_type="reference_semantics")
    prediction_files = _collect_artifacts(prediction_dir, expected_type="prediction_semantics")

    common_problem_ids = sorted(set(reference_files) & set(prediction_files))
    manifest_results: list[dict[str, Any]] = []

    for index, problem_id in enumerate(common_problem_ids, start=1):
        logger.info(
            "[%s/%s] Evaluating %s",
            index,
            len(common_problem_ids),
            problem_id,
        )
        try:
            record = evaluate_saved_semantics(
                reference_files[problem_id],
                prediction_files[problem_id],
            )
            output_path = output_dir / f"{_safe_filename(problem_id)}.json"
            save_semantics_json(record, output_path)
            manifest_results.append(
                {
                    "problem_id": problem_id,
                    "status": "ok",
                    "equivalent": record.comparison.equivalent,
                    "comparison_mode": record.comparison.comparison_mode,
                    "diagnostics": list(record.comparison.diagnostics),
                    "evaluation_path": str(output_path),
                }
            )
        except Exception as exc:
            logger.error("Failed on %s: %s", problem_id, exc)
            manifest_results.append(
                {
                    "problem_id": problem_id,
                    "status": "error",
                    "error": str(exc),
                }
            )

    missing_reference = sorted(set(prediction_files) - set(reference_files))
    missing_prediction = sorted(set(reference_files) - set(prediction_files))
    manifest = {
        "artifact_type": "semantics_evaluation_manifest",
        "created_at": datetime.now().isoformat(),
        "reference_dir": str(reference_dir),
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "evaluated_problem_count": len(common_problem_ids),
        "equivalent_count": sum(
            1
            for row in manifest_results
            if row["status"] == "ok" and row["equivalent"]
        ),
        "missing_reference_problem_ids": missing_reference,
        "missing_prediction_problem_ids": missing_prediction,
        "results": manifest_results,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved evaluation results to %s", output_dir)


def _collect_artifacts(directory: Path, *, expected_type: str) -> dict[str, Path]:
    artifact_map: dict[str, Path] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            artifact = load_semantics_artifact(path)
        except Exception:
            continue
        if artifact.artifact_type != expected_type:
            continue
        artifact_map[artifact.problem.problem_id] = path
    return artifact_map


def _resolve_output_dir(raw_output_dir: str | None) -> Path:
    if raw_output_dir:
        return Path(raw_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"semantics_evaluation_{timestamp}"


def _safe_filename(value: str) -> str:
    cleaned = _FILENAME_SANITIZE_RE.sub("_", value.strip())
    return cleaned or "artifact"


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)
