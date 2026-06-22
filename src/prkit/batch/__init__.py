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
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prkit.core.domain import PhysicsDataset, PhysicsProblem

if TYPE_CHECKING:
    from prkit.core.model_clients.batch_types import BatchResult, BatchState

__all__ = [
    "BatchSubmission",
    "BatchInputError",
    "BatchFetchUnsupportedError",
    "submit_batch_physics_reasoning",
    "fetch_batch",
    "iter_batch_results",
    "batch_fetch_supported",
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
# terminal-failed with nothing left to retrieve. ``wait=True`` stops once every
# minibatch is one of these.
_COMPLETE_STATUSES = frozenset({FETCHED, FAILED, CANCELLED, SUBMIT_ERROR, EXPIRED})

# Providers with a full batch fetch lifecycle (poll + retrieve). Gemini's
# provider string is "google" (not "gemini"); xAI / DeepSeek / Dashscope / Ollama
# have no batch fetch surface and are intentionally absent.
_FETCH_CAPABLE_PROVIDERS = frozenset({"openai", "anthropic", "google"})

# Providers whose batch line correlates by ``key`` rather than ``custom_id``.
_KEY_ID_PROVIDERS = frozenset({"google", "gemini"})
# Anthropic restricts custom ids to this charset/length.
_ANTHROPIC_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MAX_ID_LEN = 64

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


class BatchFetchUnsupportedError(BatchInputError):
    """The client's provider has no batch fetch lifecycle (poll + retrieve).

    Raised **up front** by :func:`fetch_batch` for providers outside
    :data:`_FETCH_CAPABLE_PROVIDERS` — never a raw ``NotImplementedError`` from
    partway through a sweep.
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


# --------------------------------------------------------------------------- #
# Fetch half (Stage 2)                                                        #
# --------------------------------------------------------------------------- #
def batch_fetch_supported(client: Any) -> bool:
    """True only for providers with a full fetch lifecycle (in the allow-list)."""
    return getattr(client, "provider", None) in _FETCH_CAPABLE_PROVIDERS


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
            f"providers are {sorted(_FETCH_CAPABLE_PROVIDERS)}."
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
                _write_results(output_path, results)
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


def iter_batch_results(
    submission: BatchSubmission | str | Path,
) -> Iterator[tuple[str, BatchResult]]:
    """Pure offline reader: yield ``(problem_id, BatchResult)`` in input order.

    Loads the ledger if given a ``run_dir`` (else uses *submission* as-is). For
    every ``FETCHED`` minibatch, reads its persisted ``outputs/`` file and
    correlates each line's ``custom_id`` back to its ``problem_id`` via that
    minibatch's ``id_map``, emitting one result per ``id_map`` entry (input order).
    A synthetic ERRORED :class:`BatchResult` is emitted for any submitted id the
    provider never returned (completeness); extra/uncorrelated ids are dropped and
    counted on the minibatch (``uncorrelated_count``). Reads no network — safe to
    re-run for re-scoring.
    """
    from prkit.core.model_clients.batch_types import BatchItemStatus, BatchResult

    sub = (
        submission
        if isinstance(submission, BatchSubmission)
        else BatchSubmission.load(submission)
    )
    for mb in sub.minibatches:
        if mb["status"] != FETCHED:
            continue
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
    return datetime.now(timezone.utc).isoformat()


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
