"""Batch-mode *submit* and *fetch* for physics-reasoning runs (N4).

This leaf drives the discounted provider **batch lane** end to end while staying
a bounded helper — never an orchestrating runner.

**Vocabulary** (owner-set): a **batch** is the whole thing a user triggers over a
dataset (one :func:`submit_batch_physics_reasoning` call → one *run folder* → one
:class:`BatchSubmission` ledger). A **minibatch** is one ``minibatch_size``-problem
group = one provider batch job = one ``minibatch_XXXX.jsonl`` = one element of
:attr:`BatchSubmission.minibatches`. ``prkit.batch`` / ``submit_batch_*`` /
:func:`fetch_batch` keep "batch" because it names the *batch lane*, not a unit.

**Submit half** (Stage 1): :func:`submit_batch_physics_reasoning` preprocesses each
problem exactly like the synchronous ``solve_physics_problem`` path (via the
client's ``build_problem_batch_request``), splits the dataset into minibatches of
``minibatch_size`` problems, writes each minibatch's requests as provider-correct
JSONL under one run folder, submits each minibatch, saves the consolidated
``metadata.json`` ledger, and **returns the run-folder path** (a ``str``).

**Fetch half** (Stage 2): :func:`fetch_batch` reconstructs the ledger from that
path, polls each non-final minibatch once (or loops until terminal when
``wait=True``), downloads terminal-and-retrievable minibatches to
``outputs/minibatch_XXXX.jsonl``, and persists the advanced ledger. It is
idempotent/resumable: already-fetched and terminal-failed minibatches are skipped,
so re-runs never re-hit the network. :func:`iter_batch_results` is a pure offline
reader that correlates each persisted result back to its ``problem_id`` via the
minibatch's ``id_map``, emitting one ``(problem_id, BatchResult)`` per submitted
problem. Scoring and pricing are **not** done here — the consumer calls
``prkit.api.Scorer`` / ``Verdict`` directly (cost metering is N6's job).

Import discipline: at module load this imports only :mod:`prkit.core.domain` and
the standard library — never ``prkit.api``, the dataset hub, a scorer, the cost
meter, or a provider SDK. The model client is duck-typed. The batch-lifecycle
types (:class:`~prkit.core.model_clients.batch_types.BatchResult` etc.) live under
``prkit.core.model_clients`` (an import-isolation forbidden module), so they are
imported **lazily inside** :func:`fetch_batch` / :func:`iter_batch_results`, never
at module load. See ``tests/prkit/batch/test_import_isolation.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prkit.core.domain import PhysicsDataset, PhysicsProblem

if TYPE_CHECKING:
    from pydantic import BaseModel

    from prkit.core.model_clients.batch_types import BatchResult, BatchState
    from prkit.core.model_clients.structured_output import StructuredOutputPolicy

__all__ = [
    "BatchSubmission",
    "BatchInputError",
    "BatchFetchUnsupportedError",
    "BatchSubmitUnsupportedError",
    "BatchNotTerminalError",
    "submit_batch_physics_reasoning",
    "fetch_batch",
    "iter_batch_results",
    "consolidate_batch_results",
    "resubmit_failures",
    "batch_fetch_supported",
    "batch_submit_supported",
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
    "CONSOLIDATED",
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
CONSOLIDATED = "consolidated"  # results/ files written (post-FETCHED, terminal-good)
SUBMIT_ERROR = "submit_error"  # never submitted (batch_id == ""); resubmit target
FETCH_ERROR = "fetch_error"  # retrieve raised; non-final, retried on the next pass

# Minibatches in these statuses are done with the fetch loop and skipped on the
# next pass (idempotent resume). EXPIRED is deliberately NOT here: an expired job
# is re-polled (cheap snapshot) until its results window truly closes — matches
# the design's skip set exactly. CONSOLIDATED is FETCHED-and-finalized, so it is
# skipped too (fetch never re-polls a consolidated minibatch).
_SKIP_FETCH_STATUSES = frozenset(
    {FETCHED, CONSOLIDATED, SUBMIT_ERROR, FAILED, CANCELLED}
)

# Minibatches in these statuses are terminal for ``is_complete()`` — fetched (or
# consolidated), or terminal-failed with nothing left to retrieve. ``wait=True``
# stops once every minibatch is one of these.
_COMPLETE_STATUSES = frozenset(
    {FETCHED, CONSOLIDATED, FAILED, CANCELLED, SUBMIT_ERROR, EXPIRED}
)

# Minibatches that have a readable ``outputs/`` file on disk: FETCHED, and
# FETCHED-then-CONSOLIDATED (consolidation never deletes the outputs/ file). The
# offline readers (:func:`iter_batch_results` / :func:`consolidate_batch_results`)
# gate on this so re-scoring still works after finalize.
_HAS_OUTPUT_STATUSES = frozenset({FETCHED, CONSOLIDATED})

# Minibatches that :func:`resubmit_failures` re-submits: every terminal,
# non-consolidatable status — including CANCELLED (Stage 4 D1 reverses the Stage-3
# CANCELLED exclusion, so a cancelled job is re-driven into the loop instead of
# being a dead-end). This is exactly ``_COMPLETE_STATUSES − {FETCHED, CONSOLIDATED}``.
_RESUBMIT_STATUSES = frozenset({FAILED, SUBMIT_ERROR, EXPIRED, CANCELLED})

# Total submissions a single record gets — spanning BOTH whole-minibatch retries
# (each bumps the minibatch ``attempt``) and record-level retries (carried in
# ``record_attempts``) — before it is given up on. A record's submission count is
# ``record_attempts.get(cid, 0) + minibatch_attempt``; it is siphoned for retry
# while that is ``< MAX_ATTEMPTS`` and rewritten as MAX_ATTEMPTED once it reaches it.
# A fixed module constant by design (Stage 4 D3), never a per-call argument.
MAX_ATTEMPTS = 3

# Per-record ``BatchItemStatus`` *string values* (not enum members — so the siphon
# partition needs no ``batch_types`` import at module load, preserving leaf import
# discipline) that mark a recoverable per-record failure. ``MAX_ATTEMPTED`` is
# deliberately absent: an exhausted record is terminal and never re-siphoned.
_SIPHON_RECORD_STATUSES = frozenset({"errored", "expired", "canceled"})

# Providers with a batch lifecycle at all — submit, poll and retrieve. Gemini's
# provider string is "google" (not "gemini"); xAI / DeepSeek / Dashscope /
# Moonshot / Ollama have no batch surface and are intentionally absent. Submit
# and fetch capability are the same set today, and one set cannot drift.
_BATCH_CAPABLE_PROVIDERS = frozenset({"openai", "anthropic", "google"})

# Providers whose batch line correlates by ``key`` rather than ``custom_id``.
_KEY_ID_PROVIDERS = frozenset({"google", "gemini"})
# Anthropic restricts custom ids to this charset/length.
_ANTHROPIC_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MAX_ID_LEN = 64

# ``problem_id`` is not constrained to a filesystem-safe charset (only wire ids are
# validated), so consolidation renders it to a safe ``<stem>.json`` filename: runs
# of non-``[A-Za-z0-9._-]`` collapse to ``_`` and the stem is capped well under the
# 255-byte POSIX name limit. The mapping is lossy, so a collision is possible and
# guarded loudly (never a silent overwrite) by :func:`consolidate_batch_results`.
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_RESULTS_STEM_LEN = 200

# Leaf-light logger: flows through PRKitLogger handlers when the app configured
# them, plain stdlib logging otherwise. Used for the per-pass progress summary.
_logger = logging.getLogger("prkit.batch")


@dataclass
class BatchSubmission:
    """The whole-batch ledger = one run folder = one ``metadata.json``.

    Mutable on purpose: :func:`fetch_batch` advances each minibatch's ``status``
    and persists the ledger, which is what makes fetch idempotent/resumable across
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
            # ---- Stage-4 additive keys (round-trip for free via dict(mb)) ----
            "attempt": int,               # whole-minibatch submission count
                                          # (submit => 1; whole-minibatch resubmit => +1)
            "failed_records": list[dict], # pending siphoned per-record failures, each
                                          # {custom_id, problem_id, error, attempt};
                                          # present only when records have been siphoned
            # ---- on RETRY minibatches only (built by resubmit_failures' drain) ----
            "is_retry": bool,             # True for a record-drain minibatch
            "retry_sources": list[int],   # source minibatch indices pooled into it
            "record_attempts": dict[str, int],  # per-record prior submission count
                                          # (custom_id -> count), carried forward
        }

    Invariants: ``num_requests == len(id_map)`` on every minibatch (the siphon
    decrements both in lockstep); a record's submission count is
    ``record_attempts.get(custom_id, 0) + attempt``.
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
    # Resolved structured-output plan for the run: {mode, strategy,
    # native_schema_enforced}, or None for a free-text run. Recorded because a
    # run that demoted to prompt-only on one provider and stayed native on
    # another is not comparable with one that did not, and nothing else on disk
    # would say which happened.
    structured_output: dict[str, Any] | None = None
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
            "structured_output": self.structured_output,
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
            structured_output=data.get("structured_output"),
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


class BatchFetchUnsupportedError(BatchInputError):
    """The client's provider has no batch fetch lifecycle (poll + retrieve).

    Raised **up front** by :func:`fetch_batch` for providers outside
    :data:`_BATCH_CAPABLE_PROVIDERS` — never a raw ``NotImplementedError`` from
    partway through a sweep.
    """


class BatchSubmitUnsupportedError(BatchInputError):
    """The client's provider has no batch submit surface.

    Raised **up front** by :func:`submit_batch_physics_reasoning`, before any run
    folder is created — never a raw ``NotImplementedError`` from inside the
    per-problem build loop, which would leave an orphan run folder behind.
    """


class BatchNotTerminalError(BatchInputError):
    """The ledger is not terminal, so a terminal-gated finalize step refuses.

    Raised **up front** by :func:`resubmit_failures` when some minibatch
    is still ``SUBMITTED`` / ``RUNNING`` / ``COMPLETED`` / ``FETCH_ERROR``. Run
    :func:`fetch_batch` first to drive every minibatch terminal (which also resolves
    any transient ``FETCH_ERROR`` by re-downloading), then resubmit the failures.
    """


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
    response_format: type[BaseModel] | dict[str, Any] | None = None,
    structured_policy: StructuredOutputPolicy = "best_effort",
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
    fetch time via :meth:`BatchSubmission.load` (or just pass the ``run_dir`` to
    :func:`fetch_batch` / :func:`iter_batch_results`).

    A minibatch whose submit raises is recorded with ``status=SUBMIT_ERROR``,
    ``error`` set, and ``batch_id=""`` (its input file remains for re-submission via
    Stage 1); the ledger always has one ``minibatches`` entry per minibatch.

    Pass *response_format* to request native structured output; the resolved
    plan is recorded on the ledger as ``structured_output`` so a demoted run is
    distinguishable from a natively enforced one after the fact.

    Raises:
        BatchSubmitUnsupportedError: up front, if the client's provider has no
            batch submit surface (never a raw ``NotImplementedError`` from
            inside the build loop, which would leave an orphan run folder).
        ValueError: up front, if *structured_policy* is ``'native_required'``
            and the provider cannot enforce the schema.
        BatchInputError: empty input, an invalid/duplicate id, a non-positive
            ``minibatch_size``, or a pre-existing non-empty run folder when
            ``overwrite`` is False.
    """
    if not batch_submit_supported(client):
        provider = getattr(client, "provider", None) or "unknown"
        raise BatchSubmitUnsupportedError(
            f"Provider {provider!r} has no batch submit surface; batch-capable "
            f"providers are {sorted(_BATCH_CAPABLE_PROVIDERS)}. Use the "
            "synchronous path for this provider, or check "
            "batch_submit_supported(client) first when sweeping several models."
        )

    # Resolve the plan once, before the run folder exists: it fixes what gets
    # recorded on the ledger, and makes a native_required policy the provider
    # cannot satisfy fail with nothing written to disk.
    structured_plan = (
        None
        if response_format is None
        else client.resolve_structured_output_plan(
            response_format, structured_policy=structured_policy
        )
    )

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

    now = datetime.now(UTC)
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
        # Also clear a stale Stage-3 results set, so reusing a run folder for a fresh
        # submit cannot leave old per-problem answers next to a new ledger. This is
        # the *only* auto-clear of results/ (consolidate/resubmit never clear it).
        results_dir = run_dir / "results"
        if results_dir.is_dir():
            for stale in results_dir.iterdir():
                if stale.is_file():
                    stale.unlink()
        stale_manifest = run_dir / "results_manifest.json"
        if stale_manifest.exists():
            stale_manifest.unlink()
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
                response_format=response_format,
                structured_policy=structured_policy,
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
        submitted_at = datetime.now(UTC)
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
                "attempt": 1,
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
        request_kind="free_text" if structured_plan is None else "structured",
        prkit_api_version=_prkit_api_version(),
        display_name=display_name,
        run_dir=str(run_dir),
        metadata=dict(metadata or {}),
        structured_output=(
            None
            if structured_plan is None
            else {
                "mode": structured_plan.mode,
                "strategy": structured_plan.strategy,
                "native_schema_enforced": structured_plan.native_schema_enforced,
            }
        ),
        minibatches=minibatches,
    )
    submission.save()
    _log_next_after_submit(submission)

    return str(run_dir)


# --------------------------------------------------------------------------- #
# Fetch half (Stage 2)                                                        #
# --------------------------------------------------------------------------- #
def batch_submit_supported(client: Any) -> bool:
    """True only for providers with a batch submit surface (in the allow-list).

    Callers sweeping a mix of models can branch on this to skip the ones with no
    batch lane, rather than catching :class:`BatchSubmitUnsupportedError`.
    """
    return getattr(client, "provider", None) in _BATCH_CAPABLE_PROVIDERS


def batch_fetch_supported(client: Any) -> bool:
    """True only for providers with a full fetch lifecycle (in the allow-list)."""
    return getattr(client, "provider", None) in _BATCH_CAPABLE_PROVIDERS


def fetch_batch(
    client: Any,
    submission: BatchSubmission | str | Path,
    *,
    wait: bool = False,
    poll_interval: float = 10.0,
    timeout: float | None = None,
    outputs_dirname: str = "outputs",
    progress: bool = True,
) -> BatchSubmission:
    """Poll, download, and persist a submitted batch run; return the ledger.

    Loads the ledger if given a ``run_dir`` (else uses *submission* as-is), then
    polls each non-final minibatch once — downloading terminal-and-retrievable ones
    to ``<run_dir>/<outputs_dirname>/minibatch_XXXX.jsonl`` and advancing + saving
    the ledger after each change. With ``wait=True`` the pass repeats every
    ``poll_interval`` seconds until every minibatch is terminal (or ``timeout``
    seconds elapse). Idempotent: ``FETCHED`` / terminal-failed / ``SUBMIT_ERROR``
    minibatches are skipped, so re-runs never re-hit the network for finished work.

    ``EXPIRED`` minibatches are still retrieved (they can carry a completed subset);
    only ``FAILED`` / ``CANCELLED`` have nothing to fetch. When ``progress=True``
    (the default), a one-line status summary is logged at INFO after each pass.

    Raises:
        BatchFetchUnsupportedError: up front, if the client's provider has no fetch
            lifecycle (never a raw ``NotImplementedError`` mid-sweep).
    """
    if not batch_fetch_supported(client):
        provider = getattr(client, "provider", None) or "unknown"
        raise BatchFetchUnsupportedError(
            f"Provider {provider!r} has no batch fetch lifecycle; fetch-capable "
            f"providers are {sorted(_BATCH_CAPABLE_PROVIDERS)}."
        )

    sub = (
        submission
        if isinstance(submission, BatchSubmission)
        else BatchSubmission.load(submission)
    )
    # Lazy import: batch_types sits under prkit.core.model_clients (forbidden at
    # leaf load time). The caller already holds a live client, so the package is
    # loaded by now anyway.
    from prkit.core.model_clients.batch_types import BatchState

    outputs_dir = Path(sub.run_dir) / outputs_dirname
    start = time.monotonic()
    while True:
        newly_fetched = _run_fetch_pass(client, sub, outputs_dir, BatchState)
        if progress:
            _log_progress(sub, newly_fetched)
        if not wait or sub.is_complete():
            break
        if timeout is not None and (time.monotonic() - start) >= timeout:
            break
        time.sleep(poll_interval)
    if progress:
        _log_next_after_fetch(sub)
    return sub


def _run_fetch_pass(
    client: Any,
    sub: BatchSubmission,
    outputs_dir: Path,
    batch_state: type[BatchState],
) -> int:
    """One poll-and-download pass over the non-final minibatches; return Δ fetched."""
    newly_fetched = 0
    for mb in sub.minibatches_to_fetch():
        index = mb["index"]
        old_status = mb["status"]
        st = client.poll_batch(mb["batch_id"])
        counts = dict(st.counts)

        if st.state in (batch_state.COMPLETED, batch_state.EXPIRED):
            try:
                results = list(client.retrieve_batch_results(mb["batch_id"]))
            except Exception as exc:  # noqa: BLE001 - retried on the next pass
                sub.set_status(
                    index,
                    FETCH_ERROR,
                    counts=counts,
                    error=f"{type(exc).__name__}: {exc}",
                )
                _log_transition(index, old_status, FETCH_ERROR)
                sub.save()
                continue
            # COMPLETED always persists (an empty file still marks the minibatch
            # final; iter_batch_results synthesizes failures for the missing ids).
            # EXPIRED persists only when it carried a partial subset.
            if results or st.state == batch_state.COMPLETED:
                output_path = outputs_dir / f"minibatch_{index:04d}.jsonl"
                # Stage 4: partition + siphon rides this one download pass — write
                # succeeded-only (+ exhausted MAX_ATTEMPTED) to outputs/, siphon the
                # recoverable failures onto the ledger (pruning id_map/num_requests
                # in lockstep) and refresh the derived accumulator.
                _siphon_minibatch(sub, mb, results, output_path)
                sub.set_status(
                    index,
                    FETCHED,
                    counts=counts,
                    output_path=str(output_path),
                    fetched_at=_now_iso(),
                )
                newly_fetched += 1
                _log_transition(index, old_status, FETCHED, len(results))
            else:
                sub.set_status(index, EXPIRED, counts=counts)
                _log_transition(index, old_status, EXPIRED)
        else:
            new_status = _status_for_state(st.state, batch_state)
            if new_status is None:  # UNKNOWN: keep prior status, refresh counts
                sub.set_status(index, old_status, counts=counts)
            else:
                sub.set_status(index, new_status, counts=counts)
                if new_status != old_status:
                    _log_transition(index, old_status, new_status)
        sub.save()
    return newly_fetched


def _status_for_state(state: BatchState, batch_state: type[BatchState]) -> str | None:
    """Map a poll's ``BatchState`` to a minibatch status.

    Returns ``None`` for ``UNKNOWN`` (keep the prior status and keep polling).
    ``COMPLETED`` / ``EXPIRED`` are handled by the retrieve path, not here.
    """
    return {
        batch_state.PENDING: RUNNING,
        batch_state.IN_PROGRESS: RUNNING,
        batch_state.FAILED: FAILED,
        batch_state.CANCELLED: CANCELLED,
    }.get(state)


def _write_results(path: Path, results: Sequence[BatchResult]) -> None:
    """Write normalized ``BatchResult`` lines as JSONL to *path* (creating parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "custom_id": r.custom_id,
                "status": str(r.status),
                "text": r.text,
                "error": r.error,
            },
            ensure_ascii=False,
        )
        for r in results
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _partition_results(
    results: Sequence[BatchResult],
    id_map: dict[str, str],
) -> tuple[list[BatchResult], list[BatchResult], int]:
    """Partition one retrieved minibatch into ``(succeeded, failed, uncorrelated)``.

    Pure, in-memory (no I/O), and a structural mirror of the correlation in
    :func:`_iter_minibatch_results` (synthetic-ERRORED completeness + uncorrelated
    counting). For each ``custom_id`` in *id_map* (input order): the correlated
    result is "succeeded" unless its status is a recoverable per-record failure
    (:data:`_SIPHON_RECORD_STATUSES`); an id the provider never returned is
    synthesized as an ERRORED "failed" record. Any returned id absent from *id_map*
    is an uncorrelated extra — counted, not partitioned. ``batch_types`` is imported
    lazily here to keep the leaf light (import discipline, §10).
    """
    from prkit.core.model_clients.batch_types import BatchItemStatus, BatchResult

    results_by_cid: dict[str, BatchResult] = {}
    uncorrelated = 0
    for r in results:
        if r.custom_id in id_map:
            results_by_cid[r.custom_id] = r
        else:
            uncorrelated += 1

    succeeded: list[BatchResult] = []
    failed: list[BatchResult] = []
    for cid in id_map:
        correlated = results_by_cid.get(cid)
        if correlated is None:
            correlated = BatchResult(
                custom_id=cid,
                status=BatchItemStatus.ERRORED,
                error="No result returned by the provider for this request.",
            )
        if str(correlated.status) in _SIPHON_RECORD_STATUSES:
            failed.append(correlated)
        else:
            succeeded.append(correlated)
    return succeeded, failed, uncorrelated


def _siphon_minibatch(
    sub: BatchSubmission,
    mb: dict[str, Any],
    results: Sequence[BatchResult],
    output_path: Path,
) -> None:
    """Partition a retrieved minibatch, siphon recoverable failures, write outputs.

    The Stage-4 record-recovery step that rides :func:`fetch_batch`'s single
    poll-and-download pass (it adds no new sweep). Partition *results* in memory
    (:func:`_partition_results`); write the succeeded records — plus any *exhausted*
    failure rewritten as :class:`BatchItemStatus.MAX_ATTEMPTED` — to *output_path*
    (succeeded-only on the wire). Each failure under the :data:`MAX_ATTEMPTS` bound
    is **siphoned**: appended to ``mb["failed_records"]`` and pruned from
    ``mb["id_map"]`` with ``mb["num_requests"]`` decremented in lockstep (so
    ``num_requests == len(id_map)`` holds). A record whose submission count
    (``record_attempts.get(cid, 0) + attempt``) has reached ``MAX_ATTEMPTS`` is NOT
    re-siphoned — it is rewritten as MAX_ATTEMPTED and kept in ``id_map`` so it
    consolidates terminally (distinct from a transient ERRORED). Finally refresh the
    derived run-level accumulator (:func:`_rewrite_failed_records_input`); the ledger
    ``failed_records`` key is the crash-safe source of truth.
    """
    from prkit.core.model_clients.batch_types import BatchItemStatus

    succeeded, failed, uncorrelated = _partition_results(results, mb["id_map"])
    if uncorrelated:
        mb["uncorrelated_count"] = uncorrelated

    output_records = list(succeeded)
    attempt = mb.get("attempt", 1)
    record_attempts: dict[str, int] = mb.get("record_attempts") or {}
    for r in failed:
        cid = r.custom_id
        submissions = record_attempts.get(cid, 0) + attempt
        if submissions < MAX_ATTEMPTS:
            mb.setdefault("failed_records", []).append(
                {
                    "custom_id": cid,
                    "problem_id": mb["id_map"][cid],
                    "error": r.error,
                    "attempt": submissions,
                }
            )
            del mb["id_map"][cid]
            mb["num_requests"] -= 1
        else:
            output_records.append(replace(r, status=BatchItemStatus.MAX_ATTEMPTED))

    _write_results(output_path, output_records)
    _rewrite_failed_records_input(sub)


def _rewrite_failed_records_input(sub: BatchSubmission) -> None:
    """Rewrite ``<run_dir>/failed-records-batch-input.jsonl`` from the ledger (derived).

    The authoritative pending set is each minibatch's ``failed_records`` ledger key;
    this file is a **derived** atomic full-rewrite of the provider request lines for
    every pending failed record, re-read from each source minibatch's never-pruned
    ``inputs/`` file and correlated by the provider id field (``key`` for Gemini,
    else ``custom_id``) — never a positional zip, which would mis-correlate on order
    drift. A full-rewrite (not append) keeps it crash-safe and idempotent: a re-run
    can never double-count, and resubmit's correctness depends only on the ledger
    plus the persisted ``inputs/`` files. When nothing is pending the stale file is
    removed, so the artifact appears exactly when record recovery is in play.
    """
    id_field = "key" if sub.provider in _KEY_ID_PROVIDERS else "custom_id"
    path = Path(sub.run_dir) / "failed-records-batch-input.jsonl"
    lines: list[str] = []
    for mb in sub.minibatches:
        pending = mb.get("failed_records")
        if not pending:
            continue
        lines_by_cid = {
            json.loads(line)[id_field]: line
            for line in _read_jsonl(mb["input_file_path"])
        }
        for entry in pending:
            line = lines_by_cid.get(entry["custom_id"])
            if line is not None:
                lines.append(line)
    if lines:
        _atomic_write_text(path, "\n".join(lines) + "\n")
    elif path.exists():
        path.unlink()


def iter_batch_results(
    submission: BatchSubmission | str | Path,
) -> Iterator[tuple[str, BatchResult]]:
    """Pure offline reader: yield ``(problem_id, BatchResult)`` in input order.

    Loads the ledger if given a ``run_dir`` (else uses *submission* as-is). For
    every minibatch with a persisted ``outputs/`` file (``FETCHED`` *or*
    ``CONSOLIDATED`` — consolidation keeps the file, so re-scoring still works
    post-finalize), reads it and correlates each line's ``custom_id`` back to its
    ``problem_id`` via that minibatch's ``id_map``, emitting one result per
    ``id_map`` entry (input order). A synthetic ERRORED :class:`BatchResult` is
    emitted for any submitted id the provider never returned (completeness);
    extra/uncorrelated ids are dropped and counted on the minibatch
    (``uncorrelated_count``). Reads no network — safe to re-run for re-scoring.
    """
    sub = (
        submission
        if isinstance(submission, BatchSubmission)
        else BatchSubmission.load(submission)
    )
    for mb in sub.minibatches:
        if mb["status"] in _HAS_OUTPUT_STATUSES:
            yield from _iter_minibatch_results(mb)


def _iter_minibatch_results(
    mb: dict[str, Any],
) -> Iterator[tuple[str, BatchResult]]:
    """Correlate one minibatch's persisted ``outputs/`` file back to ``problem_id``.

    Reads ``mb["output_path"]``, maps each line's ``custom_id`` to its ``problem_id``
    via ``mb["id_map"]``, and yields one ``(problem_id, BatchResult)`` per ``id_map``
    entry in input order — synthesizing an ERRORED result for any id the provider
    never returned, and dropping/counting extras on ``mb["uncorrelated_count"]``.
    Shared by :func:`iter_batch_results` and :func:`consolidate_batch_results`; the
    caller gates on status (this helper assumes a readable ``outputs/`` file). Reads
    no network. ``batch_types`` is imported lazily here to keep the leaf light.
    """
    from prkit.core.model_clients.batch_types import BatchItemStatus, BatchResult

    id_map: dict[str, str] = mb.get("id_map") or {}
    output_path = mb.get("output_path")

    results_by_cid: dict[str, BatchResult] = {}
    uncorrelated = 0
    if output_path and Path(output_path).exists():
        for line in _read_jsonl(output_path):
            obj = json.loads(line)
            cid = str(obj.get("custom_id", ""))
            result = BatchResult(
                custom_id=cid,
                status=_coerce_item_status(obj.get("status"), BatchItemStatus),
                text=obj.get("text"),
                error=obj.get("error"),
            )
            if cid in id_map:
                results_by_cid[cid] = result
            else:
                uncorrelated += 1
    if uncorrelated:
        mb["uncorrelated_count"] = uncorrelated

    for cid, problem_id in id_map.items():
        correlated = results_by_cid.get(cid)
        if correlated is None:
            correlated = BatchResult(
                custom_id=cid,
                status=BatchItemStatus.ERRORED,
                error="No result returned by the provider for this request.",
            )
        yield problem_id, correlated


def _coerce_item_status(value: Any, batch_item_status: type[Any]) -> Any:
    """Map a persisted status string back to ``BatchItemStatus`` (ERRORED on miss)."""
    try:
        return batch_item_status(value)
    except (ValueError, KeyError):
        return batch_item_status.ERRORED


def _read_jsonl(path: str | Path) -> Iterator[str]:
    """Yield non-empty stripped lines from a JSONL file."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            yield stripped


def _now_iso() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _atomic_write_text(path: str | Path, text: str) -> None:
    """Write *text* to *path* atomically (temp file in the same dir + ``os.replace``).

    The temp file is created in the destination directory so ``os.replace`` is a
    same-filesystem rename (atomic on POSIX and Windows); a crash mid-write therefore
    never leaves a truncated or corrupt file at *path*. Used for every
    ``results/<problem_id>.json`` and ``results_manifest.json`` write so finalize is
    crash-safe.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    except BaseException:
        # Best-effort cleanup if the replace never happened (temp file orphaned).
        try:
            os.unlink(tmp_name)
        except OSError:  # pragma: no cover - defensive
            pass
        raise


def _safe_results_filename(problem_id: str) -> str:
    """Render *problem_id* as a filesystem-safe ``<stem>.json`` results filename.

    Collapses runs of non-``[A-Za-z0-9._-]`` characters to ``_``, strips leading
    ``._-`` (so the result is never a dotfile or empty), caps the stem length, and
    falls back to ``"problem"`` when nothing usable remains. The mapping is lossy, so
    two distinct ids can collide on one name — :func:`consolidate_batch_results`
    guards that by refusing to overwrite a file that holds a *different*
    ``problem_id`` (it never silently clobbers).
    """
    stem = _UNSAFE_FILENAME_RE.sub("_", problem_id).strip("._-") or "problem"
    if len(stem) > _MAX_RESULTS_STEM_LEN:
        stem = stem[:_MAX_RESULTS_STEM_LEN]
    return f"{stem}.json"


# --------------------------------------------------------------------------- #
# Progress reporting (per fetch pass)                                         #
# --------------------------------------------------------------------------- #
def _log_progress(sub: BatchSubmission, newly_fetched: int) -> None:
    """Emit the one-line INFO status summary (or completion line) for a pass."""
    counts = sub.status_counts()
    n = sub.minibatch_count
    fetched = counts.get(FETCHED, 0)
    running = (
        counts.get(SUBMITTED, 0)
        + counts.get(RUNNING, 0)
        + counts.get(COMPLETED, 0)
        + counts.get(FETCH_ERROR, 0)
    )
    failed = counts.get(FAILED, 0) + counts.get(CANCELLED, 0) + counts.get(EXPIRED, 0)
    not_submitted = counts.get(SUBMIT_ERROR, 0)

    if sub.is_complete():
        _logger.info(
            "✓ Batch %r complete — %d/%d minibatches fetched, %d failed, "
            "%d not submitted.",
            sub.run_name,
            fetched,
            n,
            failed,
            not_submitted,
        )
        return

    problems_done = sum(
        mb["num_requests"] for mb in sub.minibatches if mb["status"] == FETCHED
    )
    _logger.info(
        "Batch %r [%s/%s] — fetched %d/%d minibatches (+%d this pass) · "
        "running %d · failed %d · not-submitted %d  |  results %d/%d",
        sub.run_name,
        sub.provider,
        sub.model,
        fetched,
        n,
        newly_fetched,
        running,
        failed,
        not_submitted,
        problems_done,
        sub.total_problems,
    )


def _log_transition(
    index: int, old_status: str, new_status: str, num_results: int | None = None
) -> None:
    """Emit a per-minibatch DEBUG transition line (off unless the logger is DEBUG)."""
    if num_results is None:
        _logger.debug("minibatch %d: %s → %s", index, old_status, new_status)
    else:
        _logger.debug(
            "minibatch %d: %s → %s (%d results)",
            index,
            old_status,
            new_status,
            num_results,
        )


# --------------------------------------------------------------------------- #
# Finalize half (Stage 3): consolidate + resubmit                             #
# --------------------------------------------------------------------------- #
def consolidate_batch_results(
    submission: BatchSubmission | str | Path,
    *,
    results_dirname: str = "results",
) -> BatchSubmission:
    """Stream every FETCHED-not-yet-CONSOLIDATED minibatch's results to disk.

    For each such minibatch, correlate each output line's ``custom_id`` back to its
    ``problem_id`` (via the minibatch ``id_map``) and write
    ``<run_dir>/<results_dirname>/<problem_id>.json`` =
    ``{problem_id, custom_id, status, text, error}`` **atomically**, then mark the
    minibatch ``CONSOLIDATED`` and persist the ledger. This is incremental and
    resumable: a re-run skips already-``CONSOLIDATED`` minibatches and a crash
    resumes from the first non-consolidated FETCHED one. Results are **streamed** —
    one record is held in memory at a time, never the whole run.

    Offline (no client, like :func:`iter_batch_results`) and **lenient**: it WARNS
    (does not raise) when some minibatches are not yet succeeded, consolidating the
    succeeded subset; re-run after resubmit + fetch to complete. A refreshed
    ``<run_dir>/results_manifest.json`` summary is written last (atomic).

    Raises:
        BatchInputError: an empty ledger (no minibatches), or a filename collision
            (two problems sanitizing to the same file — never a silent overwrite).
    """
    sub = (
        submission
        if isinstance(submission, BatchSubmission)
        else BatchSubmission.load(submission)
    )
    if not sub.minibatches:
        raise BatchInputError(
            "Ledger has no minibatches to consolidate (empty/degenerate run)."
        )

    pending = [mb for mb in sub.minibatches if mb["status"] not in _HAS_OUTPUT_STATUSES]
    if pending:
        _logger.warning(
            "Consolidating the succeeded subset: %d/%d minibatches not yet succeeded "
            "(resubmit + fetch them, then re-run consolidate).",
            len(pending),
            len(sub.minibatches),
        )

    results_dir = Path(sub.run_dir) / results_dirname
    results_dir.mkdir(parents=True, exist_ok=True)  # NEVER cleared here (incremental)

    status_counts: dict[str, int] = {}
    results_written = 0
    uncorrelated_total = 0
    for mb in sub.minibatches:
        if mb["status"] != FETCHED:  # CONSOLIDATED already done; the rest are skipped
            continue
        for problem_id, result in _iter_minibatch_results(mb):
            path = results_dir / _safe_results_filename(problem_id)
            existing_pid = _existing_problem_id(path)
            if existing_pid is not None and existing_pid != problem_id:
                raise BatchInputError(
                    f"Results filename collision: {path.name!r} already holds "
                    f"problem_id {existing_pid!r}, cannot also write {problem_id!r}. "
                    "Two problems sanitize to the same file; disambiguate their ids."
                )
            status_str = str(result.status)
            _atomic_write_text(
                path,
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "custom_id": result.custom_id,
                        "status": status_str,
                        "text": result.text,
                        "error": result.error,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            status_counts[status_str] = status_counts.get(status_str, 0) + 1
            results_written += 1
        uncorrelated_total += int(mb.get("uncorrelated_count", 0) or 0)
        sub.set_status(mb["index"], CONSOLIDATED)
        sub.save()  # crash-safe per minibatch (ledger owns resumability)

    _write_results_manifest(sub, status_counts, results_written, uncorrelated_total)
    _log_next_after_consolidate(sub, results_dirname)
    return sub


def _existing_problem_id(path: Path) -> str | None:
    """Return the ``problem_id`` recorded in an existing results file, else ``None``.

    Lets :func:`consolidate_batch_results` tell a genuine filename collision (the
    file holds a *different* problem) from an idempotent re-write of the same problem
    on a resume. A missing/malformed/unreadable file returns ``None`` (a safe
    overwrite target — the atomic re-write replaces it cleanly).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = data.get("problem_id")
    return pid if isinstance(pid, str) else None


def _write_results_manifest(
    sub: BatchSubmission,
    status_counts: dict[str, int],
    results_written: int,
    uncorrelated_total: int,
) -> None:
    """Refresh ``<run_dir>/results_manifest.json`` (atomic, written last each call).

    A human/consumer-facing at-a-glance summary at the run-dir root (``results/``
    holds only per-problem files); crash-safety/resumability is owned by the ledger
    (``metadata.json``), not this marker. ``minibatches_consolidated`` /
    ``fully_consolidated`` are ledger-derived (cumulative); ``results_written`` /
    ``status_counts`` / ``uncorrelated_total`` reflect *this call's* newly
    consolidated minibatches (per the design's streaming counters). A failed or
    CANCELLED minibatch keeps ``fully_consolidated`` false (it is not yet consolidated).

    ``pending_failed_records`` (Stage 4 D4) is the run-wide count of siphoned records
    still awaiting recovery (``Σ len(mb["failed_records"])``); ``fully_consolidated``
    is gated on it being zero, so a ``while not fully_consolidated`` consumer keeps
    looping until every record is recovered or exhausted. An exhausted MAX_ATTEMPTED
    record is *not* in ``failed_records``, so a run with only permanent failures is
    legitimately "done."
    """
    consolidated = sub.status_counts().get(CONSOLIDATED, 0)
    pending_failed_records = sum(
        len(mb.get("failed_records") or []) for mb in sub.minibatches
    )
    manifest = {
        "run_name": sub.run_name,
        "provider": sub.provider,
        "model": sub.model,
        "total_problems": sub.total_problems,
        "results_written": results_written,
        "status_counts": status_counts,
        "minibatches_consolidated": consolidated,
        "minibatches_total": sub.minibatch_count,
        "pending_failed_records": pending_failed_records,
        "fully_consolidated": (
            consolidated == sub.minibatch_count and pending_failed_records == 0
        ),
        "uncorrelated_total": uncorrelated_total,
        "consolidated_at": _now_iso(),
    }
    _atomic_write_text(
        Path(sub.run_dir) / "results_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False),
    )


def resubmit_failures(
    client: Any,
    submission: BatchSubmission | str | Path,
) -> BatchSubmission:
    """Re-drive a terminal batch run's failures: whole minibatches **and** records.

    Does both bounded-recovery jobs in one call (Stage 4 D5):

    (a) **Whole-minibatch retries.** Re-submit each FAILED / SUBMIT_ERROR / EXPIRED /
    **CANCELLED** minibatch in place — re-reading its persisted
    ``inputs/minibatch_XXXX.jsonl`` and calling ``client.submit_batch`` with the run's
    merged metadata, then — **only after** the submit returns a new ``batch_id`` —
    resetting that ledger entry (status → ``SUBMITTED``, new ``batch_id``, cleared
    ``output_path`` / ``fetched_at`` / ``counts`` / ``error``, ``attempt`` bumped) in
    memory and on disk. A single minibatch's submit failure is recorded as
    ``SUBMIT_ERROR`` (``batch_id=""``, ``error`` set) and the loop continues. CANCELLED
    is now resubmitted like any other failure (Stage 4 D1 reverses the Stage-3
    exclusion), so a cancelled job is no longer a dead-end.

    (b) **Record drain.** Drain the run-level failed-records accumulator into fresh
    minibatch(es): each chunk of ``<= minibatch_size`` pending siphoned records is
    re-read from its source ``inputs/`` file (correlated by the provider id field, not
    a positional zip), validated, written to a fresh ``inputs/minibatch_XXXX.jsonl``,
    submitted, and appended to the ledger as a SUBMITTED retry minibatch (fresh
    monotonic ``index``, ``is_retry=True``, ``attempt=1``, ``retry_sources``, and a
    ``record_attempts`` carrying each record's prior submission count). ``minibatch_count``
    is bumped per appended retry minibatch; the consumed entries are removed from their
    source ``failed_records`` in the same atomic save; the derived accumulator file is
    refreshed last.

    Returns the updated ledger; the consumer then re-runs :func:`fetch_batch` on the
    new jobs (and re-consolidates). The recovery loop terminates because every record
    either succeeds or reaches :data:`MAX_ATTEMPTS` and is consolidated as a terminal
    MAX_ATTEMPTED result, so the pending count strictly decreases to zero.

    Raises:
        BatchFetchUnsupportedError: up front, for a provider with no batch lifecycle.
        BatchNotTerminalError: if the ledger is not terminal (run ``fetch_batch``
            first to resolve any RUNNING / FETCH_ERROR minibatch).
    """
    if not batch_fetch_supported(client):
        provider = getattr(client, "provider", None) or "unknown"
        raise BatchFetchUnsupportedError(
            f"Provider {provider!r} has no batch lifecycle; batch-capable providers "
            f"are {sorted(_BATCH_CAPABLE_PROVIDERS)}."
        )

    sub = (
        submission
        if isinstance(submission, BatchSubmission)
        else BatchSubmission.load(submission)
    )
    if not sub.is_complete():
        raise BatchNotTerminalError(
            f"Batch {sub.run_name!r} is not terminal; run fetch_batch first to drive "
            "every minibatch terminal (resolving any FETCH_ERROR), then resubmit."
        )

    targets = [mb for mb in sub.minibatches if mb["status"] in _RESUBMIT_STATUSES]
    has_pending = any(mb.get("failed_records") for mb in sub.minibatches)
    if not targets and not has_pending:
        _log_next_after_resubmit(sub, resubmitted=0, still_failed=0)
        return sub

    # The ledger stores the un-merged user metadata + display_name separately, so
    # reconstruct the provider-side merged metadata exactly as submit did.
    merged_metadata = {**sub.metadata, "display_name": sub.display_name}
    outputs_dir = Path(sub.run_dir) / "outputs"
    resubmitted = 0
    still_failed = 0

    # (a) Whole-minibatch failures — now incl. CANCELLED (D1); each bumps ``attempt``.
    for mb in targets:
        index = mb["index"]
        old_status = mb["status"]
        try:
            requests = [json.loads(line) for line in _read_jsonl(mb["input_file_path"])]
            new_id = client.submit_batch(requests, metadata=merged_metadata)
        except Exception as exc:  # noqa: BLE001 - recorded on the ledger; loop goes on
            sub.set_status(
                index,
                SUBMIT_ERROR,
                batch_id="",
                error=f"{type(exc).__name__}: {exc}",
                output_path=None,
                fetched_at=None,
                counts={},
            )
            still_failed += 1
            _log_transition(index, old_status, SUBMIT_ERROR)
            sub.save()
            continue
        # Submit succeeded: drop a stale downloaded artifact (defensive — targets
        # normally have output_path=None), then reset the entry to SUBMITTED.
        stale_output = outputs_dir / f"minibatch_{index:04d}.jsonl"
        if stale_output.exists():
            stale_output.unlink()
        sub.set_status(
            index,
            SUBMITTED,
            batch_id=new_id,
            submitted_at=_now_iso(),
            error=None,
            output_path=None,
            fetched_at=None,
            counts={},
            attempt=mb.get("attempt", 1) + 1,
        )
        resubmitted += 1
        _log_transition(index, old_status, SUBMITTED)
        sub.save()  # crash-safe per minibatch

    # (b) Drain the run-level failed-records accumulator into fresh minibatch(es).
    resub_drain, drain_failed = _drain_failed_records(client, sub, merged_metadata)
    resubmitted += resub_drain
    still_failed += drain_failed

    _log_next_after_resubmit(sub, resubmitted=resubmitted, still_failed=still_failed)
    return sub


def _drain_failed_records(
    client: Any,
    sub: BatchSubmission,
    merged_metadata: dict[str, str],
) -> tuple[int, int]:
    """Drain pending siphoned records into fresh retry minibatches; return Δ counts.

    Reuse, not re-derive: each pending record's provider request line is re-read from
    its source minibatch's never-pruned ``inputs/`` file (correlated by the id field,
    not a positional zip — blocker #1). Each chunk of ``<= minibatch_size`` records
    becomes a fresh SUBMITTED retry minibatch appended to the ledger with a monotonic
    ``index``, rebuilt ``id_map`` / ``record_attempts``, and ``minibatch_count`` bumped;
    the drained entries are consumed from their source ``failed_records`` in the same
    crash-safe save. Returns ``(submitted_count, submit_error_count)``.
    """
    pending = [
        (mb, entry)
        for mb in sub.minibatches
        for entry in list(mb.get("failed_records") or [])
    ]
    if not pending:
        return 0, 0

    id_field = "key" if sub.provider in _KEY_ID_PROVIDERS else "custom_id"
    inputs_dir = Path(sub.run_dir) / "inputs"
    # Cache each source minibatch's never-pruned input lines, keyed by wire id.
    source_lines: dict[int, dict[str, str]] = {}
    for mb in sub.minibatches:
        if mb.get("failed_records"):
            source_lines[mb["index"]] = {
                json.loads(line)[id_field]: line
                for line in _read_jsonl(mb["input_file_path"])
            }

    resubmitted = 0
    still_failed = 0
    for chunk in _chunked(pending, sub.minibatch_size):
        new_index = max(m["index"] for m in sub.minibatches) + 1
        requests: list[dict[str, Any]] = []
        id_map: dict[str, str] = {}
        record_attempts: dict[str, int] = {}
        retry_sources: set[int] = set()
        for src_mb, entry in chunk:
            cid = entry["custom_id"]
            requests.append(json.loads(source_lines[src_mb["index"]][cid]))
            id_map[cid] = entry["problem_id"]
            record_attempts[cid] = entry["attempt"]
            retry_sources.add(src_mb["index"])

        validate_batch_requests(requests, provider=sub.provider)
        retry_input_path = inputs_dir / f"minibatch_{new_index:04d}.jsonl"
        write_batch_jsonl(requests, retry_input_path)

        batch_id = ""
        error: str | None = None
        try:
            batch_id = client.submit_batch(requests, metadata=merged_metadata)
        except Exception as exc:  # noqa: BLE001 - recorded on the ledger; loop goes on
            error = f"{type(exc).__name__}: {exc}"

        sub.minibatches.append(
            {
                "index": new_index,
                "batch_id": batch_id,
                "status": SUBMITTED if batch_id else SUBMIT_ERROR,
                "input_file_path": str(retry_input_path),
                "num_requests": len(requests),
                "id_map": id_map,
                "submitted_at": _now_iso(),
                "error": error,
                "output_path": None,
                "fetched_at": None,
                "counts": {},
                "endpoint": None,
                "completion_window": None,
                "attempt": 1,
                "is_retry": True,
                "retry_sources": sorted(retry_sources),
                "record_attempts": record_attempts,
            }
        )
        sub.minibatch_count += 1  # counts submit + retry minibatch jobs
        if batch_id:
            resubmitted += 1
        else:
            still_failed += 1
        # Consume the drained entries from their source ledger lists in the same save.
        for src_mb, entry in chunk:
            src_mb["failed_records"].remove(entry)
        sub.save()  # crash-safe per chunk

    _rewrite_failed_records_input(sub)  # now empty / shrunk (derived)
    return resubmitted, still_failed


# --------------------------------------------------------------------------- #
# Next-command guidance (one INFO line telling the human what to run next)     #
# --------------------------------------------------------------------------- #
def _log_next_after_submit(sub: BatchSubmission) -> None:
    """After submit: point at fetch (a fetch routes any SUBMIT_ERROR afterward)."""
    _logger.info(
        'Submitted %d minibatches. Next: client.fetch_batch_physics_reasoning("%s").',
        sub.minibatch_count,
        sub.run_dir,
    )


def _log_next_after_fetch(sub: BatchSubmission) -> None:
    """After a fetch pass: the owner's 3-way next-command prompt (§5.6)."""
    counts = sub.status_counts()
    n = sub.minibatch_count
    succeeded = counts.get(FETCHED, 0) + counts.get(CONSOLIDATED, 0)

    if not sub.is_complete():
        running = (
            counts.get(SUBMITTED, 0)
            + counts.get(RUNNING, 0)
            + counts.get(COMPLETED, 0)
            + counts.get(FETCH_ERROR, 0)
        )
        _logger.info(
            "Batch in progress (%d running). Re-run "
            'fetch_batch_physics_reasoning("%s") later to continue.',
            running,
            sub.run_dir,
        )
        return

    if succeeded == n:
        _logger.info(
            'All %d minibatches succeeded. Next: consolidate_batch_results("%s") '
            "to write results/.",
            n,
            sub.run_dir,
        )
        return

    # Terminal, some failed. Every terminal-not-good status (incl. CANCELLED, D1) is
    # resubmittable, so there is no dead-end branch.
    failed_total = n - succeeded
    if succeeded:
        _logger.info(
            '%d minibatches failed. Next: client.resubmit_failures("%s"),'
            " then fetch again. (%d already succeeded — consolidate_batch_results"
            " can capture them now.)",
            failed_total,
            sub.run_dir,
            succeeded,
        )
    else:
        _logger.info(
            '%d minibatches failed. Next: client.resubmit_failures("%s"),'
            " then fetch again.",
            failed_total,
            sub.run_dir,
        )


def _log_next_after_resubmit(
    sub: BatchSubmission, *, resubmitted: int, still_failed: int
) -> None:
    """After resubmit: the K-resubmitted / M-failed summary + the fetch-next hint."""
    if resubmitted == 0 and still_failed == 0:
        _logger.info(
            "Nothing to resubmit — all minibatches already succeeded and no records "
            'are pending. Next: consolidate_batch_results("%s").',
            sub.run_dir,
        )
        return
    _logger.info(
        "Resubmitted %d minibatches (%d still failed). "
        'Next: client.fetch_batch_physics_reasoning("%s").',
        resubmitted,
        still_failed,
        sub.run_dir,
    )


def _log_next_after_consolidate(sub: BatchSubmission, results_dirname: str) -> None:
    """After consolidate: 'done', the record-recovery hint (D4), or the resume hint."""
    consolidated = sub.status_counts().get(CONSOLIDATED, 0)
    n = sub.minibatch_count
    pending_failed = sum(len(mb.get("failed_records") or []) for mb in sub.minibatches)
    if consolidated == n and pending_failed == 0:
        _logger.info("Done — results in %s/%s/.", sub.run_dir, results_dirname)
    elif pending_failed:
        # D4: minibatch-completeness alone is blind to pending siphoned records, so a
        # `while not fully_consolidated` consumer would stop without record-retry.
        _logger.info(
            "Done consolidating %d minibatches, but %d records still failed — run "
            'client.resubmit_failures("%s"), then fetch + consolidate again.',
            consolidated,
            pending_failed,
            sub.run_dir,
        )
    else:
        _logger.info(
            "Consolidated %d/%d minibatches; the rest are not yet succeeded — "
            'resubmit + fetch them, then re-run consolidate_batch_results("%s").',
            consolidated,
            n,
            sub.run_dir,
        )
