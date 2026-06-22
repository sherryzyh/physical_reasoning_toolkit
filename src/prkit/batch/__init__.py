"""Batch-mode *submit* for physics-reasoning runs (N4).

This leaf turns a set of :class:`~prkit.core.domain.PhysicsProblem`\\s into
submitted provider batch jobs and one whole-batch :class:`BatchSubmission` ledger.

**Vocabulary** (owner-set): a **batch** is the whole thing a user triggers over a
dataset (one :func:`submit_batch_physics_reasoning` call → one *run folder* → one
:class:`BatchSubmission` ledger). A **minibatch** is one ``minibatch_size``-problem
group = one provider batch job = one ``minibatch_XXXX.jsonl`` = one element of
:attr:`BatchSubmission.minibatches`. ``prkit.batch`` / ``submit_batch_*`` keep
"batch" because it names the *batch lane*, not a unit.

:func:`submit_batch_physics_reasoning` preprocesses each problem exactly like the
synchronous ``solve_physics_problem`` path (via the client's
``build_problem_batch_request``), splits the dataset into minibatches of
``minibatch_size`` problems, writes each minibatch's requests as provider-correct
JSONL under one run folder, submits each minibatch, saves the consolidated
``metadata.json`` ledger, and **returns the run-folder path** (a ``str``). The
ledger is mutable on purpose — it is the resume state store (each minibatch carries
a ``status``); disk is the source of truth across the ~24h provider window, so the
object is reconstructed on demand via :meth:`BatchSubmission.load`. Polling /
downloading / correlating the results is a later milestone that consumes the ledger.

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
    "submit_batch_physics_reasoning",
    "dumps_batch_jsonl",
    "write_batch_jsonl",
    "validate_batch_requests",
    # minibatch status constants (plain strings; kept off the BatchState enum to
    # keep the leaf light)
    "SUBMITTED",
    "RUNNING",
    "COMPLETED",
    "EXPIRED",
    "FAILED",
    "CANCELLED",
    "FETCHED",
    "SUBMIT_ERROR",
    "FETCH_ERROR",
]

# --------------------------------------------------------------------------- #
# Minibatch status constants                                                  #
# --------------------------------------------------------------------------- #
SUBMITTED = "submitted"  # accepted by the provider, not yet polled to a result
RUNNING = "running"  # provider is processing (pending / in-progress)
COMPLETED = "completed"  # provider finished; download not yet persisted
EXPIRED = "expired"  # window elapsed and nothing was retrievable
FAILED = "failed"  # batch-level provider failure
CANCELLED = "cancelled"  # cancelled at the provider
FETCHED = "fetched"  # results downloaded + persisted to outputs/ (terminal-good)
SUBMIT_ERROR = "submit_error"  # never submitted (batch_id == ""); re-submit via Stage 1
FETCH_ERROR = "fetch_error"  # retrieve raised; non-final, retried on the next pass

# Minibatches in these statuses are done with the fetch loop and skipped on the
# next pass (idempotent resume). EXPIRED is deliberately NOT here: an expired job
# is re-polled (cheap snapshot) until its results window truly closes — matches
# the design's skip set exactly.
_SKIP_FETCH_STATUSES = frozenset({FETCHED, SUBMIT_ERROR, FAILED, CANCELLED})

# Minibatches in these statuses are terminal for ``is_complete()`` — fetched, or
# terminal-failed with nothing left to retrieve.
_COMPLETE_STATUSES = frozenset({FETCHED, FAILED, CANCELLED, SUBMIT_ERROR, EXPIRED})

# Providers whose batch line correlates by ``key`` rather than ``custom_id``.
_KEY_ID_PROVIDERS = frozenset({"google", "gemini"})
# Anthropic restricts custom ids to this charset/length.
_ANTHROPIC_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MAX_ID_LEN = 64


@dataclass
class BatchSubmission:
    """The whole-batch ledger = one run folder = one ``metadata.json``.

    Mutable on purpose: the fetch step advances each minibatch's ``status`` and
    persists the ledger, which is what makes fetch idempotent/resumable across
    processes. Batch-level facts are stored once; the per-minibatch fetch state
    lives in :attr:`minibatches`, a list of plain dicts (one per provider job).

    Each ``minibatches[i]`` dict carries::

        {
            "index": int,                 # minibatch ordinal (0-based)
            "batch_id": str,              # provider job id ("" => submit failed)
            "status": str,                # one of the module status constants
            "input_file_path": str,       # <run_dir>/inputs/minibatch_XXXX.jsonl
            "num_requests": int,          # problems in this minibatch
            "id_map": dict[str, str],     # wire custom_id / key -> problem_id
            "submitted_at": str | None,   # ISO 8601
            "error": str | None,          # local submit error (with SUBMIT_ERROR)
            "output_path": str | None,    # <run_dir>/outputs/... (set on fetch)
            "fetched_at": str | None,     # ISO 8601 (set on fetch)
            "counts": dict[str, int],     # last poll's request_counts
            "endpoint": str | None,       # audit-only (provider parity)
            "completion_window": str | None,  # audit-only (provider parity)
        }
    """

    # ---- batch-level (stored once; never repeated per minibatch) ----
    run_name: str
    provider: str  # "openai" | "anthropic" | "google"
    model: str
    created_at: datetime  # UTC
    minibatch_size: int  # record cap per minibatch
    minibatch_count: int  # number of minibatches
    total_problems: int
    dataset: dict[str, Any] | None  # {"name", "version"} or None for a bare list
    request_kind: str  # "free_text"
    prkit_api_version: str
    display_name: str
    run_dir: str  # "<output_dir>/<run_name>"
    metadata: dict[str, str] = field(default_factory=dict)
    # ---- per-minibatch ledger (the mutable, fetch-updated part) ----
    minibatches: list[dict[str, Any]] = field(default_factory=list)

    # ---- pure state helpers (NO network) ----
    def minibatches_to_fetch(self) -> list[dict[str, Any]]:
        """Minibatches still in the fetch loop (status not in the skip set)."""
        return [
            mb for mb in self.minibatches if mb["status"] not in _SKIP_FETCH_STATUSES
        ]

    def set_status(self, index: int, status: str, **fields: Any) -> None:
        """Set ``minibatches[index]['status']`` (and any extra fields) in place.

        Looks up by the minibatch's own ``index`` field (not list position, though
        they coincide for ledgers built by :func:`submit_batch_physics_reasoning`).
        """
        for mb in self.minibatches:
            if mb["index"] == index:
                mb["status"] = status
                mb.update(fields)
                return
        raise KeyError(f"No minibatch with index {index}.")

    def is_complete(self) -> bool:
        """True once every minibatch is fetched or terminal-failed."""
        return all(mb["status"] in _COMPLETE_STATUSES for mb in self.minibatches)

    def status_counts(self) -> dict[str, int]:
        """Return ``{status: count}`` over the minibatches (backs the summary)."""
        counts: dict[str, int] = {}
        for mb in self.minibatches:
            counts[mb["status"]] = counts.get(mb["status"], 0) + 1
        return counts

    # ---- serialization ----
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (``created_at`` rendered ISO 8601).

        Per-minibatch timestamps are already stored as ISO strings, so only the
        batch-level ``created_at`` needs rendering.
        """
        return {
            "run_name": self.run_name,
            "provider": self.provider,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "minibatch_size": self.minibatch_size,
            "minibatch_count": self.minibatch_count,
            "total_problems": self.total_problems,
            "dataset": self.dataset,
            "request_kind": self.request_kind,
            "prkit_api_version": self.prkit_api_version,
            "display_name": self.display_name,
            "run_dir": self.run_dir,
            "metadata": dict(self.metadata),
            "minibatches": [dict(mb) for mb in self.minibatches],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchSubmission:
        """Reconstruct a ledger from :meth:`to_dict` output.

        ``created_at`` is parsed with :meth:`datetime.fromisoformat`; the resulting
        tzinfo compares **equal** to ``timezone.utc`` (``+00:00`` round-trips by
        equality, not identity), which is the contract Stage-1 tests assert.
        """
        created_raw = data["created_at"]
        created_at = (
            datetime.fromisoformat(created_raw)
            if isinstance(created_raw, str)
            else created_raw
        )
        return cls(
            run_name=data["run_name"],
            provider=data["provider"],
            model=data["model"],
            created_at=created_at,
            minibatch_size=data["minibatch_size"],
            minibatch_count=data["minibatch_count"],
            total_problems=data["total_problems"],
            dataset=data.get("dataset"),
            request_kind=data.get("request_kind", "free_text"),
            prkit_api_version=data.get("prkit_api_version", ""),
            display_name=data.get("display_name", data["run_name"]),
            run_dir=data["run_dir"],
            metadata=dict(data.get("metadata") or {}),
            minibatches=[dict(mb) for mb in data.get("minibatches", [])],
        )

    def save(self, run_dir: str | Path | None = None) -> Path:
        """Write ``<run_dir>/metadata.json`` and return its path.

        Defaults to :attr:`run_dir`. Creates the folder if missing so a fresh
        ledger can be persisted before any inputs are laid down.
        """
        target = Path(run_dir) if run_dir is not None else Path(self.run_dir)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "metadata.json"
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, run_dir_or_metadata_path: str | Path) -> BatchSubmission:
        """Read a ledger back from a run folder or a ``metadata.json`` path."""
        p = Path(run_dir_or_metadata_path)
        metadata_path = p if p.name == "metadata.json" else p / "metadata.json"
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


class BatchInputError(ValueError):
    """Invalid submit input: empty problem set, duplicate/illegal id, or a
    pre-existing non-empty run folder (when ``overwrite`` is False)."""


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
        raise BatchInputError(f"minibatch_size must be positive; got {size}.")
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
    minibatch_size: int = 500,
    instructions: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    custom_id_fn: Callable[[PhysicsProblem], str] | None = None,
    display_name: str | None = None,
    metadata: dict[str, str] | None = None,
    overwrite: bool = False,
) -> str:
    """Preprocess, split, write, and submit *problems* as provider batch jobs.

    Builds one free-text request per problem (mirroring the synchronous
    ``solve_physics_problem`` preprocessing via ``client.build_problem_batch_request``),
    splits them into minibatches of ``minibatch_size``, writes each minibatch's
    provider-correct JSONL under ``<output_dir>/<run_name>/inputs/minibatch_NNNN.jsonl``,
    submits each minibatch sequentially via ``client.submit_batch``, saves the
    consolidated :class:`BatchSubmission` ledger to ``<run_dir>/metadata.json``, and
    **returns the run-folder path** (a ``str``).

    Disk is the source of truth across the ~24h provider window, so the ledger is
    persisted rather than handed back as an object: reconstruct it on demand at
    fetch time via :meth:`BatchSubmission.load`.

    A minibatch whose submit raises is recorded with ``status=SUBMIT_ERROR``,
    ``error`` set, and ``batch_id=""`` (its input file remains for re-submission via
    Stage 1); the ledger always has one ``minibatches`` entry per minibatch.

    Raises:
        BatchInputError: empty input, an invalid/duplicate id, a non-positive
            ``minibatch_size``, or a pre-existing non-empty run folder when
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
        for stale in (run_dir / "inputs").glob("minibatch_*.jsonl"):
            stale.unlink()
        for stale in (run_dir / "outputs").glob("minibatch_*.jsonl"):
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

    request_chunks = _chunked(requests, minibatch_size)
    rid_chunks = _chunked(request_ids, minibatch_size)
    pid_chunks = _chunked(problem_ids, minibatch_size)

    # Write all JSONL artifacts first (cheap, local), then submit sequentially.
    file_paths: list[Path] = []
    for index, chunk in enumerate(request_chunks):
        file_paths.append(
            write_batch_jsonl(chunk, inputs_dir / f"minibatch_{index:04d}.jsonl")
        )

    merged_metadata = {**(metadata or {}), "display_name": display_name}

    minibatches: list[dict[str, Any]] = []
    for index, chunk in enumerate(request_chunks):
        id_map = dict(zip(rid_chunks[index], pid_chunks[index]))
        submitted_at = datetime.now(timezone.utc)
        batch_id = ""
        error: str | None = None
        try:
            batch_id = client.submit_batch(chunk, metadata=merged_metadata)
        except Exception as exc:  # noqa: BLE001 - recorded on the ledger for resume
            error = f"{type(exc).__name__}: {exc}"
        minibatches.append(
            {
                "index": index,
                "batch_id": batch_id,
                "status": SUBMITTED if batch_id else SUBMIT_ERROR,
                "input_file_path": str(file_paths[index]),
                "num_requests": len(chunk),
                "id_map": id_map,
                "submitted_at": submitted_at.isoformat(),
                "error": error,
                "output_path": None,
                "fetched_at": None,
                "counts": {},
                "endpoint": None,
                "completion_window": None,
            }
        )

    dataset_field: dict[str, Any] | None
    if dataset_name is not None:
        dataset_field = {"name": dataset_name, "version": dataset_version}
    else:
        dataset_field = None

    submission = BatchSubmission(
        run_name=run_name,
        provider=provider,
        model=client.model,
        created_at=now,
        minibatch_size=minibatch_size,
        minibatch_count=len(request_chunks),
        total_problems=len(problem_list),
        dataset=dataset_field,
        request_kind="free_text",
        prkit_api_version=_prkit_api_version(),
        display_name=display_name,
        run_dir=str(run_dir),
        metadata=dict(metadata or {}),
        minibatches=minibatches,
    )
    submission.save()

    return str(run_dir)
