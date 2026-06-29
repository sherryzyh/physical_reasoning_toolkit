# `prkit.contamination` — contamination / freshness (roadmap X3)

A score means little if the test was already in the model's training data, or if
a "new" benchmark is a rephrase of an existing one. This package adds the three
signals that let a consumer trust a physics score.

## Sub-features

### A — Provenance / release-date metadata (foundation)
`provenance.py` — normalized, JSON-safe provenance attached at load:

- **Dataset-level** `DatasetProvenance` → `PhysicalDataset.info['provenance']`.
- **Per-problem** `ProblemProvenance` → `PhysicsProblem.additional_fields['provenance']`.

`DatasetHub.load(...)` always stamps best-effort provenance (cheap, metadata
only) via `BaseDatasetLoader.get_provenance()`, which derives from the
`name`/`paper_url`/`repository_url`/`license_spdx`/`year` keys loaders already
expose. Both objects round-trip through `to_dict`/`from_dict` (ISO dates), so
`save_to_json` is unaffected. A per-problem `release_date` is what makes
model-cutoff-aware ("evaluate after the model's training cut-off") evaluation
possible.

### B — n-gram (+ embedding) overlap report
`overlap.py` — `compute_overlap_report(target, references=None, ...) -> OverlapReport`:

- **Stage 1 — n-gram containment** (default-on, zero extra deps): high-order
  word n-gram shingles of normalized question text, containment =
  `|shingles(a) ∩ shingles(b)| / |shingles(a)|`, with inverted-index candidate
  pruning. `references=None` runs self-overlap (intra-dataset duplicates).
- **Stage 2 — embedding cosine** (opt-in): catches *rephrased* duplicates n-gram
  misses. Inject any `Embedder`, or install the **`freshness`** extra for the
  default `sentence-transformers` model.

`DatasetHub.load(..., contamination_check=True, contamination_refs=[...])`
attaches the report at `info['overlap_report']` (default OFF — no behavior
change).

### C — Parametric variants (semantics-verified)
`variants.py` — `generate_variants(template, n, seed=, verify=) -> list[VariantResult]`:

ABench-Physics `Phy_B`-style: perturb a templated problem's numeric constants
and re-derive the answer. When a `ParametricTemplate` supplies two independent
derivation paths (`answer_fn` and `answer_expr`), each variant is cross-checked
through the shipped verifier (`prkit.verify.verify`, the **N1** dependency), so a
broken template surfaces as `verified=False` rather than emitting a wrong gold.
Variants are emitted as `is_synthetic=True` problems carrying `parent_problem_id`.

## Method lineage (cited)

- **n-gram decontamination** — classic GPT-3/Llama high-order n-gram overlap.
- **llm-decontaminator** (arXiv:2311.04850) — embedding top-k recall for rephrases.
- **Min-K% Prob** (arXiv:2310.16789) — token-level membership inference; a future
  per-model freshness probe (needs logprobs, not uniformly exposed today).
- **ABench-Physics** (arXiv:2507.04766) — the `Phy_B` parametric-variation idea.

## Import discipline

Bare `import prkit.contamination` pulls **core deps only**. The text normalizer
(SymPy-importing module), `numpy`, `sympy`, and `sentence-transformers` are all
imported lazily inside the functions that need them.
