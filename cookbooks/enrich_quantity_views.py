#!/usr/bin/env python3
"""Backfill deterministic quantity views into saved semantics artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from prkit.core import PRKitLogger
from prkit.semantics.inference import (
    load_prediction_semantics_artifact,
    load_reference_semantics_artifact,
    load_semantics_artifact,
    load_semantics_evaluation_record,
    save_semantics_json,
)
from prkit.semantics.normalization import enrich_answer_quantity_views

logger = PRKitLogger.get_logger(__name__)

_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill deterministic quantity views into saved semantics artifacts.",
    )
    parser.add_argument("--prediction-dir", type=str, default=None)
    parser.add_argument("--reference-dir", type=str, default=None)
    parser.add_argument("--evaluation-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    selected_inputs = [
        bool(args.prediction_dir),
        bool(args.reference_dir),
        bool(args.evaluation_dir),
    ]
    if sum(selected_inputs) != 1:
        raise ValueError(
            "Provide exactly one of --prediction-dir, --reference-dir, or --evaluation-dir."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prediction_dir:
        _backfill_prediction_dir(Path(args.prediction_dir), output_dir=output_dir)
        return
    if args.reference_dir:
        _backfill_reference_dir(Path(args.reference_dir), output_dir=output_dir)
        return
    _backfill_evaluation_dir(Path(args.evaluation_dir), output_dir=output_dir)


def _backfill_prediction_dir(source_dir: Path, *, output_dir: Path) -> None:
    artifacts = _collect_artifacts(source_dir, expected_type="prediction_semantics")
    results: list[dict[str, Any]] = []

    for index, (problem_id, path) in enumerate(sorted(artifacts.items()), start=1):
        logger.info("[%s/%s] Backfilling prediction %s", index, len(artifacts), problem_id)
        artifact = load_prediction_semantics_artifact(path)
        enriched = artifact.model_copy(
            update={
                "prediction_answer_semantics": enrich_answer_quantity_views(
                    artifact.prediction_answer_semantics,
                    context=artifact.question_semantics,
                )
            }
        )
        output_path = output_dir / f"{_safe_filename(problem_id)}.json"
        save_semantics_json(enriched, output_path)
        results.append(
            {
                "problem_id": problem_id,
                "status": "ok",
                "output_path": str(output_path),
            }
        )

    _write_manifest(
        output_dir=output_dir,
        source_dir=source_dir,
        source_artifact_type="prediction_semantics",
        results=results,
    )


def _backfill_reference_dir(source_dir: Path, *, output_dir: Path) -> None:
    artifacts = _collect_artifacts(source_dir, expected_type="reference_semantics")
    results: list[dict[str, Any]] = []

    for index, (problem_id, path) in enumerate(sorted(artifacts.items()), start=1):
        logger.info("[%s/%s] Backfilling reference %s", index, len(artifacts), problem_id)
        artifact = load_reference_semantics_artifact(path)
        enriched = artifact.model_copy(
            update={
                "reference_answer_semantics": enrich_answer_quantity_views(
                    artifact.reference_answer_semantics,
                    context=artifact.question_semantics,
                )
            }
        )
        output_path = output_dir / f"{_safe_filename(problem_id)}.json"
        save_semantics_json(enriched, output_path)
        results.append(
            {
                "problem_id": problem_id,
                "status": "ok",
                "output_path": str(output_path),
            }
        )

    _write_manifest(
        output_dir=output_dir,
        source_dir=source_dir,
        source_artifact_type="reference_semantics",
        results=results,
    )


def _backfill_evaluation_dir(source_dir: Path, *, output_dir: Path) -> None:
    source_problem_dir = source_dir / "problems" if (source_dir / "problems").is_dir() else source_dir
    output_problem_dir = output_dir / "problems"
    output_problem_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    source_records: list[tuple[Path, Any]] = []
    for path in sorted(source_problem_dir.glob("*.json")):
        try:
            record = load_semantics_evaluation_record(path)
        except Exception:
            continue
        source_records.append((path, record))

    for index, (path, record) in enumerate(source_records, start=1):
        logger.info("[%s/%s] Backfilling evaluation %s", index, len(source_records), path.stem)
        enriched = record.model_copy(
            update={
                "reference_answer_semantics": enrich_answer_quantity_views(
                    record.reference_answer_semantics,
                    context=record.question_semantics,
                ),
                "prediction_answer_semantics": enrich_answer_quantity_views(
                    record.prediction_answer_semantics,
                    context=record.question_semantics,
                ),
            }
        )
        output_path = output_problem_dir / path.name
        save_semantics_json(enriched, output_path)
        results.append(
            {
                "problem_id": record.problem.problem_id,
                "status": "ok",
                "output_path": str(output_path),
            }
        )

    _write_manifest(
        output_dir=output_dir,
        source_dir=source_dir,
        source_artifact_type="semantics_evaluation_record",
        results=results,
    )


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


def _write_manifest(
    *,
    output_dir: Path,
    source_dir: Path,
    source_artifact_type: str,
    results: list[dict[str, Any]],
) -> None:
    manifest = {
        "artifact_type": "quantity_view_backfill_manifest",
        "created_at": datetime.now().isoformat(),
        "source_dir": str(source_dir),
        "source_artifact_type": source_artifact_type,
        "output_dir": str(output_dir),
        "total_records": len(results),
        "results": results,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _safe_filename(value: str) -> str:
    cleaned = _FILENAME_SANITIZE_RE.sub("_", value.strip())
    return cleaned or "artifact"


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)
