"""Provider-agnostic types for the asynchronous Batch API lifecycle.

Batch processing is shaped differently from a synchronous chat call: many
requests are submitted at once, the provider returns a job id, the caller polls
until the job reaches a *terminal* state, then downloads per-request results
(each of which may independently succeed or fail). These types give OpenAI,
Anthropic, and Gemini batches one common vocabulary so a consumer can drive all
three through the same interface (see ``BaseModelClient.submit_batch`` /
``poll_batch`` / ``retrieve_batch_results``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .modes import _StrEnum


class BatchState(_StrEnum):
    """Provider-agnostic state of a whole batch job."""

    PENDING = "pending"  # accepted / validating, not yet running
    IN_PROGRESS = "in_progress"  # running (incl. finalizing / cancelling)
    COMPLETED = "completed"  # terminal: finished, results available
    FAILED = "failed"  # terminal: batch-level failure
    EXPIRED = "expired"  # terminal: window elapsed before completion
    CANCELLED = "cancelled"  # terminal: cancelled by the user
    UNKNOWN = "unknown"  # unrecognized provider status


TERMINAL_STATES = frozenset(
    {
        BatchState.COMPLETED,
        BatchState.FAILED,
        BatchState.EXPIRED,
        BatchState.CANCELLED,
    }
)


class BatchItemStatus(_StrEnum):
    """Per-request outcome within a batch that has finished processing."""

    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    EXPIRED = "expired"
    CANCELED = "canceled"


@dataclass(frozen=True)
class BatchStatus:
    """A snapshot of a batch job's state from a single poll.

    *output_ref* / *error_ref* carry whatever handle the provider needs to fetch
    results later (e.g. an OpenAI output file id, a Gemini destination file
    name); they may be ``None`` until the job is terminal.
    """

    batch_id: str
    state: BatchState
    provider: str
    raw_status: str
    counts: dict[str, int] = field(default_factory=dict)
    output_ref: str | None = None
    error_ref: str | None = None

    @property
    def is_terminal(self) -> bool:
        """True once the job has stopped processing (success or failure)."""
        return self.state in TERMINAL_STATES


@dataclass(frozen=True)
class BatchResult:
    """One request's result, correlated back to its input by ``custom_id``.

    On success *text* holds the model's free-text output (the same string a
    synchronous ``response()`` would return). On any non-success outcome *text*
    is ``None`` and *error* describes the failure.
    """

    custom_id: str
    status: BatchItemStatus
    text: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is BatchItemStatus.SUCCEEDED
