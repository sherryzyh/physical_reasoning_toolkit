#!/usr/bin/env python3
"""Solve a dataset with a model and save prediction-semantics JSON artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from prkit.core import PRKitLogger
from prkit.core.model_clients import create_model_client
from prkit.datasets import DatasetHub
from prkit.semantics import generate_prediction_semantics, save_semantics_json

logger = PRKitLogger.get_logger(__name__)

_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run physics reasoning with semantics output and save one prediction JSON per problem.",
    )
    parser.add_argument("--dataset", "-d", required=True, help="Dataset name.")
    parser.add_argument("--model", "-m", default="gpt-5-mini", help="Model name.")
    parser.add_argument("--variant", default=None, help="Dataset variant.")
    parser.add_argument("--split", default=None, help="Dataset split.")
    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        default=1,
        help="Number of problems to process. Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Automatically download the dataset if needed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for artifact JSON files.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional max_output_tokens override for the model call.",
    )
    args = parser.parse_args()

    client = create_model_client(args.model)
    dataset = _load_dataset(
        dataset_name=args.dataset,
        variant=args.variant,
        split=args.split,
        sample_size=args.sample_size,
        auto_download=args.auto_download,
    )

    output_dir = _resolve_output_dir(
        args.output_dir,
        prefix="prediction_semantics",
        dataset_name=args.dataset,
        model_name=args.model,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_results: list[dict[str, Any]] = []
    for index, problem in enumerate(dataset, start=1):
        logger.info(
            "[%s/%s] Generating prediction semantics for %s",
            index,
            len(dataset),
            problem.problem_id,
        )

        artifact_path = output_dir / f"{_safe_filename(problem.problem_id)}.json"
        try:
            artifact = generate_prediction_semantics(
                problem,
                client,
                max_output_tokens=args.max_output_tokens,
            )
            save_semantics_json(artifact, artifact_path)
            manifest_results.append(
                {
                    "problem_id": problem.problem_id,
                    "status": "ok",
                    "artifact_path": str(artifact_path),
                    "final_answer": artifact.final_answer,
                }
            )
        except Exception as exc:
            logger.error("Failed on %s: %s", problem.problem_id, exc)
            manifest_results.append(
                {
                    "problem_id": problem.problem_id,
                    "status": "error",
                    "error": str(exc),
                }
            )

    manifest = {
        "artifact_type": "prediction_semantics_manifest",
        "dataset_name": args.dataset,
        "variant": args.variant,
        "split": args.split,
        "model_name": args.model,
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "total_problems": len(manifest_results),
        "successful_problems": sum(
            1 for row in manifest_results if row["status"] == "ok"
        ),
        "results": manifest_results,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Saved prediction semantics to %s", output_dir)


def _load_dataset(
    *,
    dataset_name: str,
    variant: str | None,
    split: str | None,
    sample_size: int,
    auto_download: bool,
):
    load_kwargs: dict[str, Any] = {
        "dataset_name": dataset_name,
        "sample_size": None,
        "auto_download": auto_download,
    }
    if variant is not None:
        load_kwargs["variant"] = variant
    if split is not None:
        load_kwargs["split"] = split
    dataset = DatasetHub.load(**load_kwargs)
    if sample_size == 0:
        return dataset
    return dataset.take(sample_size)


def _resolve_output_dir(
    raw_output_dir: str | None,
    *,
    prefix: str,
    dataset_name: str,
    model_name: str,
) -> Path:
    if raw_output_dir:
        return Path(raw_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path.cwd()
        / f"{prefix}_{_safe_filename(dataset_name)}_{_safe_filename(model_name)}_{timestamp}"
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
