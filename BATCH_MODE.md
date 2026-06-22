# Batch Mode (N4) — Design & Usage Reference

**Single source of truth** for prkit's discounted provider **batch lane**. This doc is the rolled-up,
up-to-date design: it supersedes the per-stage design notes in `internal/N4_BATCH_DESIGN_STAGE_{1,2,3}.md`
(kept only as historical rationale). When a future stage is designed, **roll its committed decisions into this
file** rather than leaving them stranded in a stage doc — see [Maintaining this doc](#maintaining-this-doc).

## Implementation status (at a glance)

| Stage | Capability | Status | Public surface |
|---|---|---|---|
| **1 — Submit** | preprocess → split → write JSONL → submit → ledger | ✅ **Implemented & usable** | `submit_batch_physics_reasoning`, `build_problem_batch_request` |
| **2 — Fetch** | poll → download → correlate → resume | ✅ **Implemented & usable** | `fetch_batch`, `iter_batch_results`, `batch_fetch_supported` |
| **3 — Finalize** | consolidate per-problem results · resubmit failed minibatches | ✅ **Implemented & usable** | `consolidate_batch_results`, `resubmit_failed_minibatches` |

> **All three stages are implemented and callable** as of 2026-06-22 (104 batch tests pass on
> `feat/n4-batch-mode`). Everything below describes the live API.

---

## Concepts & vocabulary

The vocabulary was fixed in Stage 2 (it renamed Stage 1's terms — see [Design evolution](#design-evolution)).

- **batch** — the *whole thing* a user triggers over a dataset. One `submit_batch_physics_reasoning(...)` call
  → one **run folder** → one `BatchSubmission` **ledger** (`metadata.json`).
- **minibatch** — one `minibatch_size`-problem group = one provider batch job = one `minibatch_XXXX.jsonl` =
  one element of `BatchSubmission.minibatches`. **The minibatch is the unit of success**: it either succeeded
  (`FETCHED`) or failed as a whole. (Record-level failures within a fetched minibatch surface as `ERRORED`
  results, not as minibatch failures.)
- **run folder** — the consumer-owned directory holding the ledger, the input JSONL, and (after fetch) the
  output JSONL. **Disk is the source of truth** across the provider's ~24h window.
- The names `prkit.batch` / `submit_batch_*` / `fetch_batch` keep the word "batch" because it names the
  discounted **batch lane**, not a unit.

### What batch mode is (and is not)

`prkit.batch` is a **bounded helper**, not an end-to-end runner. It owns the tedious, provider-specific glue a
hub should own once — preprocessing parity, splitting, JSONL writing, submission, polling, downloading,
`problem_id` correlation, and a resumable ledger — and hands control back to the consumer at every seam.

It deliberately does **not** own:

- **An end-to-end runner / outer loop.** No dataset loading, no inference orchestration, no auto-chaining. The
  consumer drives `submit → fetch → [resubmit → fetch]* → consolidate`.
- **Scoring.** No scorer adapter. Batch mode stops at correlated `BatchResult`s; the consumer calls
  `prkit.api.Scorer` / `Verdict` directly (see [Scoring seam](#scoring-seam)).
- **Pricing.** That is N6's job (the cost meter). Batch mode forks no pricing and imports no `prkit.cost`. The
  ~50% batch discount is realized at the provider regardless; N6 only makes it *visible*
  (see [Cost-meter seam](#cost-meter-n6-seam)).
- **A hidden state store.** The only state is the consumer-owned ledger + output files inside the run folder.
  No checkpoint DB, no response cache.
- **Concurrency orchestration.** Minibatches are submitted / polled / fetched sequentially. A consumer wanting
  parallelism runs the sync call in its own threads.

---

## Provider support

| Provider | `client.provider` | Batch input transport | Fetch lifecycle |
|---|---|---|---|
| OpenAI | `"openai"` | JSONL **file** upload | ✅ poll + retrieve (output **and** error file) |
| Anthropic | `"anthropic"` | **inline list** (no file upload) | ✅ poll + retrieve (`.results()` stream) |
| Gemini | **`"google"`** | keyed JSONL **file** upload (`key` → `custom_id`) | ✅ poll + retrieve |
| xAI / DeepSeek / Dashscope / Ollama | — | — | ❌ no batch surface |

- Gemini's provider string is **`"google"`**, not `"gemini"` — relevant if you key off `client.provider`.
- Fetch capability is gated by `batch_fetch_supported(client)` against `{"openai", "anthropic", "google"}`;
  `fetch_batch` raises `BatchFetchUnsupportedError` **up front** for anything else (never a raw
  `NotImplementedError` mid-sweep).
- A local `.jsonl` artifact is written under `inputs/` for **all three** providers (reproducibility / inspection
  / resume), even Anthropic, whose `submit_batch` actually sends an inline list.

---

## Quick start

```python
from prkit.datasets import DatasetHub
from prkit.core.model_clients import create_model_client
from prkit.batch import iter_batch_results

dataset = DatasetHub.load("physreason", variant="full", split="test")
client  = create_model_client("gpt-5.1")

# 1) SUBMIT — preprocess (matches solve_physics_problem), split, write JSONL, submit.
#    Returns the run-folder PATH (a str). The ledger is saved to <run_dir>/metadata.json.
run_dir = client.submit_batch_physics_reasoning(dataset)
#    Default run folder: ./batch_runs/<dataset>-<model>-<UTC-timestamp>/
#    Override: client.submit_batch_physics_reasoning(dataset, output_dir="/data/runs", run_name="eval-001")

# ... the provider holds results for ~24h. Fetch can happen later, in a different process —
#     only the run_dir string is needed (disk is the source of truth). ...

# 2) FETCH — poll, download ready minibatches, persist, return the fresh ledger.
#    Idempotent/resumable: re-run to make progress; fetched minibatches are skipped.
sub = client.fetch_batch_physics_reasoning(run_dir, wait=True)   # wait=True loops until terminal
print(sub.is_complete(), sub.status_counts())

# 3) READ + SCORE — offline correlation back to problem_id (no network; re-runnable).
for problem_id, result in iter_batch_results(run_dir):           # accepts run_dir or the ledger
    if result.succeeded:
        verdict = scorer.score(result.text, gold[problem_id])    # consumer's scoring seam (no adapter)
    # non-success: result.status is ERRORED/EXPIRED/CANCELED, result.text is None, result.error set
```

`fetch_batch` without `wait=True` does **one** poll-and-download pass and returns — ideal for a cron / `/loop`
driver that re-invokes until `sub.is_complete()`. Each pass logs a one-line INFO summary on the `prkit.batch`
logger (suppress with `progress=False`).

---

## Public API (implemented)

All symbols live in the import-light leaf `prkit.batch` (`src/prkit/batch/__init__.py`). The `BaseModelClient`
facades are the one-line ergonomic entry points; the module functions are the same thing without the facade.

### Submit

```python
# Module function:
prkit.batch.submit_batch_physics_reasoning(
    client,                                  # duck-typed BaseModelClient (openai / anthropic / google)
    problems,                                # PhysicsDataset | Sequence[PhysicsProblem]
    *,
    output_dir: str | Path = "batch_runs",   # root dir that holds run folders
    run_name: str | None = None,             # default: slug("<dataset>-<model>-<UTC-timestamp>")
    minibatch_size: int = 500,               # problems per provider job; safely under all provider ceilings
    instructions: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    custom_id_fn: Callable[[PhysicsProblem], str] | None = None,  # default: problem.problem_id
    display_name: str | None = None,         # provider-facing label; defaults to run_name
    metadata: dict[str, str] | None = None,
    overwrite: bool = False,                 # reuse a non-empty run folder (clears stale minibatch_*.jsonl)
) -> str                                     # returns the run-folder path

# Facade (mirrors solve_physics_problem ergonomics):
client.submit_batch_physics_reasoning(problems, **kwargs) -> str
```

- Builds **one free-text request per problem** via `client.build_problem_batch_request`, which reuses the same
  prompt builder + images as the synchronous `solve_physics_problem` path — so **batch prompts ≡ sync prompts**.
  Free-text only today (mirrors the `ANSWER_TEXT`-only sync path).
- Writes all `inputs/minibatch_XXXX.jsonl` first (cheap, local), then submits sequentially. A minibatch whose
  submit raises is recorded with `status=SUBMIT_ERROR`, `error` set, `batch_id=""`, and its input file present —
  so the run is resumable.
- **Correlation ids** default to `problem_id`. Validated up front (fail-fast) for uniqueness and per-provider
  limits: OpenAI ≤ 64 chars; Anthropic `^[a-zA-Z0-9_-]{1,64}$`. Pass `custom_id_fn` to map a problem → a
  surrogate id; the ledger's per-minibatch `id_map` recovers `problem_id` at read time regardless.

```python
# Lower-level client method submit uses internally (also useful directly):
client.build_problem_batch_request(problem, *, request_id=None, instructions=None,
                                   max_output_tokens=None, temperature=None, **kwargs) -> dict
```

### Fetch

```python
prkit.batch.fetch_batch(
    client,
    submission,                              # BatchSubmission | run_dir str | Path
    *,
    wait: bool = False,                      # False: one poll-and-download pass; True: loop to completion
    poll_interval: float = 10.0,             # seconds between polls when wait=True
    timeout: float | None = None,            # max seconds when wait=True; None = no cap
    outputs_dirname: str = "outputs",
    progress: bool = True,                   # emit the one-line INFO summary after each pass
) -> BatchSubmission                         # the freshly-loaded, updated ledger

client.fetch_batch_physics_reasoning(run_dir_or_submission, **kwargs) -> BatchSubmission
```

- Polls each non-final minibatch once; on `COMPLETED`/`EXPIRED` it downloads results to
  `outputs/minibatch_XXXX.jsonl` and marks the minibatch `FETCHED`. The ledger is saved after **every** change
  (crash-safe).
- **Idempotent/resumable:** `FETCHED` / `SUBMIT_ERROR` / `FAILED` / `CANCELLED` (and `CONSOLIDATED`) minibatches
  are skipped, so re-runs never re-hit the network for finished work. `EXPIRED` is still *re-polled* (cheap) and
  still *retrieved* (it can carry a completed subset).
- A retrieve that raises is recorded as `FETCH_ERROR` (non-terminal) and retried on the next pass.

### Read results (offline)

```python
prkit.batch.iter_batch_results(submission) -> Iterator[tuple[str, BatchResult]]
```

- Pure offline reader (no network, re-runnable for re-scoring). Accepts a `BatchSubmission` or a `run_dir`.
- For every minibatch with a downloaded output file, correlates each line's `custom_id` → `problem_id` via the
  minibatch's `id_map`, yielding **one `(problem_id, BatchResult)` per submitted problem in input order**.
- **Completeness:** any submitted id the provider never returned yields a synthetic `ERRORED` `BatchResult`;
  extra/uncorrelated ids are dropped and counted on the minibatch (`uncorrelated_count`).

### Capability check

```python
prkit.batch.batch_fetch_supported(client) -> bool   # True for openai / anthropic / google
```

### Result types (from `prkit.core.model_clients.batch_types`)

```python
@dataclass(frozen=True)
class BatchResult:
    custom_id: str
    status: BatchItemStatus          # SUCCEEDED | ERRORED | EXPIRED | CANCELED
    text: str | None = None          # the model's free-text output on success; None otherwise
    error: str | None = None         # failure description on any non-success outcome
    @property
    def succeeded(self) -> bool: ...
```

### Errors

| Error | Base | Raised when |
|---|---|---|
| `BatchInputError` | `ValueError` | empty input, duplicate/illegal id, pre-existing non-empty run folder without `overwrite` |
| `BatchFetchUnsupportedError` | `BatchInputError` | `fetch_batch` (or resubmit) called on a non-batch provider |
| `BatchNotTerminalError` | `BatchInputError` | *(Stage 3)* `resubmit_failed_minibatches` called on a non-terminal ledger |

---

## The ledger (`BatchSubmission` / `metadata.json`)

One run folder = one `metadata.json` = `BatchSubmission.to_dict()`. The ledger is a **mutable status record**
(Stage 1's frozen per-batch receipt was reshaped into this in Stage 2). `submit` builds it, saves it, and
returns the `run_dir`; `fetch` loads, advances, and re-saves it.

```python
@dataclass                       # mutable: it is a status ledger
class BatchSubmission:
    # ---- batch-level (stored once) ----
    run_name: str
    provider: str                # "openai" | "anthropic" | "google"
    model: str
    created_at: datetime         # UTC
    minibatch_size: int
    minibatch_count: int
    total_problems: int
    dataset: dict | None         # {"name", "version"} or None for a bare problem list
    request_kind: str            # "free_text"
    prkit_api_version: str
    display_name: str
    run_dir: str
    metadata: dict[str, str]
    # ---- per-minibatch ledger (mutable, fetch-updated) ----
    minibatches: list[dict]

    # pure state helpers (NO network):
    def minibatches_to_fetch(self) -> list[dict]: ...   # status not in the fetch-skip set
    def set_status(self, index: int, status: str, **fields) -> None: ...
    def is_complete(self) -> bool: ...                  # every minibatch terminal
    def status_counts(self) -> dict[str, int]: ...      # {status: count}
    # serialization:
    def to_dict(self) -> dict: ...                      # JSON-safe (datetimes -> ISO 8601)
    @classmethod
    def from_dict(cls, data: dict) -> "BatchSubmission": ...
    def save(self, run_dir=None) -> Path: ...           # write <run_dir>/metadata.json
    @classmethod
    def load(cls, run_dir_or_metadata_path) -> "BatchSubmission": ...
```

Each `minibatches[i]` dict:

```python
{
    "index": int,                 # 0-based ordinal
    "batch_id": str,              # provider job id ("" => submit failed)
    "status": str,                # one of the status constants below
    "input_file_path": str,       # <run_dir>/inputs/minibatch_XXXX.jsonl
    "num_requests": int,
    "id_map": dict[str, str],     # wire custom_id / key -> problem_id (per minibatch)
    "submitted_at": str | None,   # ISO 8601
    "error": str | None,          # local submit error (with SUBMIT_ERROR)
    "output_path": str | None,    # <run_dir>/outputs/... (set on fetch)
    "fetched_at": str | None,     # ISO 8601 (set on fetch)
    "counts": dict[str, int],     # last poll's request_counts
    "endpoint": str | None,       # audit-only (provider parity)
    "completion_window": str | None,  # audit-only
    # "uncorrelated_count": int   # added by iter_batch_results when extra ids were dropped
}
```

### Run-folder layout

```
<output_dir>/<run_name>/
    metadata.json                 # the BatchSubmission ledger (mutable; advanced by fetch)
    inputs/
        minibatch_0000.jsonl      # one provider-correct request line per problem
        minibatch_0001.jsonl
        ...
    outputs/                      # written by fetch: one normalized-BatchResult JSONL per fetched minibatch
        minibatch_0000.jsonl      # lines: {"custom_id", "status", "text", "error"}
        ...
    results/                      # written by consolidate (Stage 3): one file per problem
        <problem_id>.json
    results_manifest.json         # written by consolidate (Stage 3): consolidation summary
```

---

## Status lifecycle

Minibatch status constants (`prkit.batch`, plain strings):
`SUBMITTED`, `RUNNING`, `COMPLETED`, `EXPIRED`, `FAILED`, `CANCELLED`, `FETCHED`, `CONSOLIDATED`,
`SUBMIT_ERROR`, `FETCH_ERROR`.

```
SUBMITTED ─poll→ RUNNING ─poll→ COMPLETED ─retrieve→ FETCHED ─consolidate→ CONSOLIDATED   (terminal-good)
                          └poll→ EXPIRED(partial+results) → FETCHED                       (counts as success)
                          └poll→ EXPIRED(empty) ─────────┐
                          └poll→ FAILED ─────────────────┤ resubmit (Stage 3) ─→ SUBMITTED  (re-enters loop)
SUBMIT_ERROR ────────────────────────────────────────────┘
                          └poll→ CANCELLED         (terminal; NOT auto-resubmitted — see Stage 3 limitation)
FETCH_ERROR (non-terminal) ─re-run fetch_batch→ ...
```

**Poll state → minibatch status** (the fetch-pass mapping):

| `BatchState` from poll | fetch action | resulting status |
|---|---|---|
| `PENDING` / `IN_PROGRESS` | none | `RUNNING` |
| `COMPLETED` | retrieve → write `outputs/` (empty file still marks it final) | `FETCHED` |
| `EXPIRED` | retrieve; persist if any results returned | `FETCHED` (partial) if any, else `EXPIRED` |
| `FAILED` | none | `FAILED` |
| `CANCELLED` | none | `CANCELLED` |
| `UNKNOWN` | none (keep polling) | unchanged |
| retrieve raised | record error | `FETCH_ERROR` (retried next pass) |
| local submit failure (`batch_id == ""`) | skip | `SUBMIT_ERROR` |

Key status sets (internal, but they explain behavior):

- `_SKIP_FETCH_STATUSES = {FETCHED, CONSOLIDATED, SUBMIT_ERROR, FAILED, CANCELLED}` — skipped by the fetch loop.
  `EXPIRED` is intentionally **not** here (re-polled until its window truly closes).
- `_COMPLETE_STATUSES = {FETCHED, CONSOLIDATED, FAILED, CANCELLED, SUBMIT_ERROR, EXPIRED}` — `is_complete()` /
  `wait=True` stop once every minibatch is one of these.
- `_HAS_OUTPUT_STATUSES = {FETCHED, CONSOLIDATED}` — minibatches with a readable `outputs/` file.

---

## Stage 3 — Finalize

Stage 3 closes the loop with two **bounded** verbs over a fetched ledger. It is still **not** an end-to-end
runner — the consumer owns the outer loop; each verb does one step and returns the ledger.

### `consolidate_batch_results` (offline, incremental)

```python
prkit.batch.consolidate_batch_results(
    submission,                      # BatchSubmission | run_dir str | Path
    *,
    results_dirname: str = "results",
) -> BatchSubmission
```

- Streams every `FETCHED`-but-not-yet-`CONSOLIDATED` minibatch's results into **per-problem files**
  `results/<problem_id>.json` = `{problem_id, custom_id, status, text, error}`, then marks each minibatch
  `CONSOLIDATED` and persists the ledger (so a re-run skips it). Writes a `results_manifest.json` summary at the
  run-dir root.
- **Lenient + incremental:** consolidates the succeeded subset and *warns* (does not raise) about minibatches
  not yet succeeded; safe to run mid-flight and re-run after the rest land. Streams one record at a time — no
  in-memory aggregate. **Offline, no client** (mirrors `iter_batch_results`).
- **Filename safety:** `problem_id` is sanitized to a filesystem-safe name; a collision (two problems mapping to
  the same file) raises `BatchInputError` rather than silently overwriting.

### `resubmit_failed_minibatches` (needs client, terminal-gated)

```python
prkit.batch.resubmit_failed_minibatches(client, submission) -> BatchSubmission
client.resubmit_failed_minibatches(run_dir_or_submission, **kwargs) -> BatchSubmission   # facade
```

- Re-submits each `FAILED` / `SUBMIT_ERROR` / `EXPIRED` minibatch (`_RESUBMIT_STATUSES`) by re-reading its
  persisted `inputs/minibatch_XXXX.jsonl` and calling `client.submit_batch`, resetting the ledger entry to
  `SUBMITTED` with a new `batch_id` **only after submit succeeds** (a failed submit → `SUBMIT_ERROR`, loop
  continues). The consumer then re-runs `fetch_batch` on the new jobs.
- **Excludes `CANCELLED`** (a cancel can be deliberate; auto-resubmit would fight intent).
- Requires a terminal ledger (`is_complete()`), else raises `BatchNotTerminalError` (run `fetch_batch` first);
  raises `BatchFetchUnsupportedError` up front for a non-batch provider.

### Intended end-to-end loop (consumer-driven)

```python
run_dir = client.submit_batch_physics_reasoning(dataset)
sub = client.fetch_batch_physics_reasoning(run_dir, wait=True)
while not all(mb["status"] in ("fetched", "consolidated") for mb in sub.minibatches):
    sub = client.resubmit_failed_minibatches(run_dir)            # Stage 3
    sub = client.fetch_batch_physics_reasoning(run_dir, wait=True)
sub = consolidate_batch_results(run_dir)                          # Stage 3 -> results/<problem_id>.json
for problem_id, result in iter_batch_results(run_dir):
    if result.succeeded:
        verdict = scorer.score(result.text, gold[problem_id])
```

### Known limitation (Stage 3)

A `CANCELLED` minibatch is neither resubmitted nor consolidatable, so a batch containing one can never reach
"fully consolidated" and a naïve `while not all-done` loop would spin. Surfaced via the manifest's
`fully_consolidated` flag and the next-command guidance; a future stage may add an opt-in
`force_resubmit_cancelled` path.

### Status → finalize action (single source of truth)

| minibatch `status` | consolidate | resubmit |
|---|---|---|
| `FETCHED` | write `results/<problem_id>.json` → `CONSOLIDATED` | skip |
| `CONSOLIDATED` | skip (done) | skip |
| `FAILED` / `SUBMIT_ERROR` / `EXPIRED` | skip + warn | re-read input → `submit_batch` → `SUBMITTED` (or `SUBMIT_ERROR`) |
| `CANCELLED` | skip + warn (dead-end) | skip (known limitation) |
| `RUNNING` / `SUBMITTED` / `COMPLETED` / `FETCH_ERROR` | skip + warn | precondition `is_complete()` fails → raise |

---

## Scoring seam

Batch mode stops at correlated `BatchResult`s. The consumer scores by reading `outputs/` via
`iter_batch_results` (or the per-problem `results/<problem_id>.json` files after consolidation) and calling
`prkit.api.Scorer.score(prediction, reference) -> Verdict` directly. The reference/gold answers live on the
consumer's `PhysicsProblem`s, not in the ledger. `prkit.batch` imports no `prkit.api`, accepts no `Scorer`, and
owns no references — scoring is genuinely the consumer's.

## Cost-meter (N6) seam

Pricing is N6's job; batch mode forks none and imports no `prkit.cost`. **Forward dependency:** `BatchResult`
carries only `custom_id` / `status` / `text` / `error` — **no token usage** — so batch-path pricing is not
possible until N6 ships and adds a usage channel to the batch path. The ~50% discount is realized at the
provider regardless; N6 only makes it *visible*.

## Import discipline (leaf-light)

`prkit.batch` imports only the standard library + `prkit.core.domain` at module load — never `prkit.api`, the
dataset hub, a scorer, the cost meter, or a provider SDK. Batch-lifecycle types (`BatchResult`,
`BatchItemStatus`, `BatchState`) live under `prkit.core.model_clients` (an import-isolation forbidden module),
so they are imported **lazily inside** `fetch_batch` / `iter_batch_results`. The `BaseModelClient` facades
likewise lazily import `prkit.batch`. The client is duck-typed throughout. Enforced by
`tests/prkit/batch/test_import_isolation.py` (load-time cleanliness only).

---

## Design evolution

Decisions that changed across stages — recorded so the renames and contract shifts are unambiguous.

| Stage 1 (original) | Current (Stage 2+) | Why |
|---|---|---|
| "run" = whole submit; "batch" = unit | **"batch"** = whole submit; **"minibatch"** = unit | clearer naming; "batch" names the lane |
| `batch_size`, `num_batches`, `batch_index`, `batch_*.jsonl` | `minibatch_size`, `minibatch_count`, `index`, `minibatch_*.jsonl` | follows the rename |
| `BatchSubmission` = frozen per-batch receipt | **mutable whole-batch ledger** (+ `from_dict`/`save`/`load`/status helpers) | fetch needs a resumable status record |
| submit returns `list[BatchSubmission]` | submit returns **`run_dir` str** | disk is the source of truth across the ~24h gap; an in-memory object would be stale by fetch time |
| `metadata.json` = immutable audit record, "never read back" | **mutable resume ledger** prkit reads & updates | this is what makes fetch idempotent (boundary preserved: still consumer-owned, no *hidden* store) |
| xAI = "build-only" batch | xAI = **no batch surface** | Stage-1 cleanup deleted its only (unused, structured) batch method |
| `iter_batch_results` gates on `== FETCHED` | *(Stage 3)* widens to `_HAS_OUTPUT_STATUSES` so re-scoring still works after consolidation | consolidate keeps `outputs/` files |
| `submit(overwrite=True)` clears `inputs/`+`outputs/` | *(Stage 3)* also clears `results/`+`results_manifest.json` | a reused folder must not leave a stale results set |

Also removed during Stage 1 (no longer part of the contract): the reserved `api.Runner` Protocol, and the
unused structured-batch request/response wrappers. The structured-output *engine* (`StructuredOutputPlan` /
`.parse()`) was untouched.

---

## Maintaining this doc

This file is the rollup. When a new stage is designed and its decisions are owner-approved:

1. Write/keep the detailed rationale in `internal/N4_BATCH_DESIGN_STAGE_<n>.md` (the working design + Q&A).
2. **Fold the committed surface into this file**: update the status table, [Public API](#public-api-implemented),
   the ledger/layout, the lifecycle, and add a row to [Design evolution](#design-evolution) for anything that
   *changes* an earlier decision.
3. Flip a capability from 🟡 designed to ✅ implemented **only** when it is actually callable in
   `src/prkit/batch/__init__.py` (and its facade, if any, in `base.py`) — re-verify against the code, not the
   stage doc.

### Sources (historical rationale)

- `internal/N4_BATCH_DESIGN_STAGE_1.md` — submit half; Q1–Q6; `Runner` + structured-batch cleanup.
- `internal/N4_BATCH_DESIGN_STAGE_2.md` — fetch half; ledger reshape; Stage-1 amendments.
- `internal/N4_BATCH_DESIGN_STAGE_3.md` — finalize (consolidate + resubmit); Stage-1/2 amendments.
- Code: `src/prkit/batch/__init__.py`, `src/prkit/core/model_clients/base.py` (facades),
  `src/prkit/core/model_clients/batch_types.py` (`BatchResult` / `BatchItemStatus` / `BatchState`).
- Roadmap: `internal/DEVELOPMENT_ROADMAP.md` §N4, §N6.
