"""Batch-mode *submit* for physics-reasoning runs (N4 Stage 1).

This leaf turns a set of :class:`~prkit.core.domain.PhysicsProblem`\\s into
submitted provider batch jobs and returns one typed receipt
(:class:`BatchSubmission`) per batch. It preprocesses each problem **exactly**
like the synchronous ``solve_physics_problem`` path (via the client's
``build_problem_batch_request``), splits the dataset into provider batches of
``batch_size`` problems, writes each batch's requests as provider-correct JSONL
under one run folder per call, submits each batch, and writes a run-level
``metadata.json`` audit record **before** returning.

It is a *bounded submitter*: it stops at submitted receipts. Fetching /
polling / scoring / pricing is a later milestone that consumes these receipts.

Import discipline: at module load this imports only :mod:`prkit.core.domain`
and the standard library — never ``prkit.api``, the dataset hub, a scorer, the
cost meter, or a provider SDK. The model client is duck-typed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prkit.core.domain import PhysicsDataset, PhysicsProblem

__all__ = [
    "BatchSubmission",
    "BatchInputError",
    "BatchSubmitError",
    "submit_batch_physics_reasoning",
    "dumps_batch_jsonl",
    "write_batch_jsonl",
    "validate_batch_requests",
]

# Providers whose batch line correlates by ``key`` rather than ``custom_id``.
_KEY_ID_PROVIDERS = frozenset({"google", "gemini"})
# Anthropic restricts custom ids to this charset/length.
_ANTHROPIC_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MAX_ID_LEN = 64


@dataclass(frozen=True)
class BatchSubmission:
    """One submission receipt = one batch = one provider batch job.

    The normalized provider id is in :attr:`batch_id` (empty string when submit
    failed — the batch is then resumable via :attr:`input_file_path`, whose JSONL
    is always written). :attr:`id_map` maps each wire ``custom_id`` / ``key`` back
    to its ``problem_id`` so a later fetch step correlates results to problems
    even when surrogate ids are used.
    """

    provider: str
    model: str
    batch_id: str
    input_file_path: str
    submitted_at: datetime
    num_requests: int
    batch_index: int
    id_map: dict[str, str]
    endpoint: str | None = None
    completion_window: str | None = None
    metadata_path: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (``submitted_at`` rendered ISO 8601)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "batch_id": self.batch_id,
            "input_file_path": self.input_file_path,
            "submitted_at": self.submitted_at.isoformat(),
            "num_requests": self.num_requests,
            "batch_index": self.batch_index,
            "id_map": dict(self.id_map),
            "endpoint": self.endpoint,
            "completion_window": self.completion_window,
            "metadata_path": self.metadata_path,
            "metadata": dict(self.metadata),
            "error": self.error,
        }


class BatchInputError(ValueError):
    """Invalid submit input: empty problem set, duplicate/illegal id, or a
    pre-existing non-empty run folder (when ``overwrite`` is False)."""


class BatchSubmitError(RuntimeError):
    """A batch submission failed.

    Carries the receipts that succeeded before the failure plus the failed
    receipt so a consumer can resume by re-submitting the failed batch (its
    input JSONL already exists). The Stage-1 submit loop records per-batch
    failures on the receipts and returns the full list rather than raising; this
    type is the resume-carrying error reserved for that tooling.
    """

    def __init__(
        self,
        message: str,
        *,
        successes: list[BatchSubmission],
        failed: BatchSubmission,
    ) -> None:
        super().__init__(message)
        self.successes = successes
        self.failed = failed


def _slug(text: str) -> str:
    """Lowercase, collapse non-``[a-z0-9._-]`` runs to ``-``, trim edges.

    Falls back to ``"run"`` when the result would be empty.
    """
    slugged = re.sub(r"[^a-z0-9._-]+", "-", text.strip().lower()).strip("-")
    return slugged or "run"


def dumps_batch_jsonl(requests: Sequence[dict[str, Any]]) -> str:
    """Serialize *requests* as JSON Lines (one object per line).

    Uses ``ensure_ascii=False`` so non-ASCII (e.g. Chinese) physics problems stay
    readable in the artifact. This artifact is for inspection / resume; the
    provider upload re-serializes from the request list, so byte-identity is not
    required.
    """
    return "\n".join(json.dumps(request, ensure_ascii=False) for request in requests)


def write_batch_jsonl(requests: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write *requests* as JSONL to *path* (creating parents) and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_batch_jsonl(requests) + "\n", encoding="utf-8")
    return out


def validate_batch_requests(
    requests: Sequence[dict[str, Any]], *, provider: str
) -> None:
    """Fail-fast validation of correlation ids before any file/network I/O.

    Raises :class:`BatchInputError` on an empty request set, a missing/empty id,
    a duplicate id across the whole run, or an id that violates the active
    provider's length/charset rule (OpenAI ≤ 64 chars; Anthropic
    ``^[a-zA-Z0-9_-]{1,64}$``). Gemini and unknown providers require only a
    non-empty unique id.
    """
    if not requests:
        raise BatchInputError("No batch requests to submit (empty input).")

    id_field = "key" if provider in _KEY_ID_PROVIDERS else "custom_id"
    seen: set[str] = set()
    for index, request in enumerate(requests):
        raw = request.get(id_field)
        if not isinstance(raw, str) or not raw:
            raise BatchInputError(
                f"Batch request at index {index} has a missing/empty {id_field!r}."
            )
        if raw in seen:
            raise BatchInputError(
                f"Duplicate correlation id {raw!r} across the run; ids must be unique."
            )
        seen.add(raw)

        if provider == "anthropic":
            if not _ANTHROPIC_ID_RE.match(raw):
                raise BatchInputError(
                    f"Id {raw!r} is invalid for Anthropic; must match "
                    r"^[a-zA-Z0-9_-]{1,64}$."
                )
        elif provider == "openai":
            if len(raw) > _MAX_ID_LEN:
                raise BatchInputError(
                    f"Id {raw!r} exceeds OpenAI's {_MAX_ID_LEN}-char limit "
                    f"(len={len(raw)})."
                )


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Split *items* into consecutive chunks of at most *size*."""
    if size <= 0:
        raise BatchInputError(f"batch_size must be positive; got {size}.")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _prkit_api_version() -> str:
    """Return the contract version, read lazily to keep this module leaf-light."""
    try:
        from prkit.api import API_VERSION

        return API_VERSION
    except Exception:  # pragma: no cover - defensive; api import should not fail
        return "1.0"


def submit_batch_physics_reasoning(
    client: Any,
    problems: PhysicsDataset | Sequence[PhysicsProblem],
    *,
    output_dir: str | Path = "batch_runs",
    run_name: str | None = None,
    batch_size: int = 500,
    instructions: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    custom_id_fn: Callable[[PhysicsProblem], str] | None = None,
    display_name: str | None = None,
    metadata: dict[str, str] | None = None,
    overwrite: bool = False,
) -> list[BatchSubmission]:
    """Preprocess, split, write, and submit *problems* as provider batch jobs.

    Builds one free-text request per problem (mirroring the synchronous
    ``solve_physics_problem`` preprocessing via ``client.build_problem_batch_request``),
    splits them into batches of ``batch_size``, writes each batch's provider-correct
    JSONL under ``<output_dir>/<run_name>/inputs/batch_NNNN.jsonl``, submits each
    batch sequentially via ``client.submit_batch``, writes a run-level
    ``metadata.json`` (run header + submissions) **before** returning, and returns
    one :class:`BatchSubmission` per batch.

    A batch whose submit raises is recorded with ``error`` set and ``batch_id=""``
    (its input file remains for resume); the returned list always has one entry
    per batch.

    Raises:
        BatchInputError: empty input, an invalid/duplicate id, a non-positive
            ``batch_size``, or a pre-existing non-empty run folder when
            ``overwrite`` is False.
    """
    if isinstance(problems, PhysicsDataset):
        dataset_name: str | None = problems.name
        dataset_version: Any = problems.get_info().get("version")
        problem_list: list[PhysicsProblem] = list(problems)
    else:
        dataset_name = None
        dataset_version = None
        problem_list = list(problems)

    if not problem_list:
        raise BatchInputError("No problems to submit (empty dataset/sequence).")

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    if run_name is None:
        base = (
            f"{dataset_name}-{client.model}-{timestamp}"
            if dataset_name
            else f"{client.model}-{timestamp}"
        )
        run_name = _slug(base)
    else:
        run_name = _slug(run_name)
    display_name = display_name or run_name

    run_dir = Path(output_dir) / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise BatchInputError(
                f"Run folder {str(run_dir)!r} already exists and is non-empty; "
                "pass overwrite=True to write into it."
            )
        for stale in (run_dir / "inputs").glob("batch_*.jsonl"):
            stale.unlink()
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # Build one request per problem, keeping ids parallel for id_map + validation.
    requests: list[dict[str, Any]] = []
    request_ids: list[str] = []
    problem_ids: list[str] = []
    for problem in problem_list:
        request_id = custom_id_fn(problem) if custom_id_fn else problem.problem_id
        requests.append(
            client.build_problem_batch_request(
                problem,
                request_id=request_id,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        )
        request_ids.append(request_id)
        problem_ids.append(problem.problem_id)

    provider = client.provider
    validate_batch_requests(requests, provider=provider)

    request_chunks = _chunked(requests, batch_size)
    rid_chunks = _chunked(request_ids, batch_size)
    pid_chunks = _chunked(problem_ids, batch_size)

    # Write all JSONL artifacts first (cheap, local), then submit sequentially.
    file_paths: list[Path] = []
    for index, chunk in enumerate(request_chunks):
        file_paths.append(
            write_batch_jsonl(chunk, inputs_dir / f"batch_{index:04d}.jsonl")
        )

    metadata_path = run_dir / "metadata.json"
    merged_metadata = {**(metadata or {}), "display_name": display_name}

    submissions: list[BatchSubmission] = []
    for index, chunk in enumerate(request_chunks):
        id_map = dict(zip(rid_chunks[index], pid_chunks[index]))
        submitted_at = datetime.now(timezone.utc)
        batch_id = ""
        error: str | None = None
        try:
            batch_id = client.submit_batch(chunk, metadata=merged_metadata)
        except Exception as exc:  # noqa: BLE001 - recorded on the receipt for resume
            error = f"{type(exc).__name__}: {exc}"
        submissions.append(
            BatchSubmission(
                provider=provider,
                model=client.model,
                batch_id=batch_id,
                input_file_path=str(file_paths[index]),
                submitted_at=submitted_at,
                num_requests=len(chunk),
                batch_index=index,
                id_map=id_map,
                metadata_path=str(metadata_path),
                metadata=merged_metadata,
                error=error,
            )
        )

    dataset_field: dict[str, Any] | None
    if dataset_name is not None:
        dataset_field = {"name": dataset_name, "version": dataset_version}
    else:
        dataset_field = None

    run_record = {
        "run_id": run_name,
        "created_at": now.isoformat(),
        "provider": provider,
        "model": client.model,
        "dataset": dataset_field,
        "batch_size": batch_size,
        "total_problems": len(problem_list),
        "num_batches": len(request_chunks),
        "request_kind": "free_text",
        "prkit_api_version": _prkit_api_version(),
        "submissions": [submission.to_dict() for submission in submissions],
    }
    metadata_path.write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return submissions
