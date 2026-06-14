# Evaluation Subset Selection Memo

Date: 2026-04-14

This memo summarizes the current evaluation-subset selection policy reflected in the repository, excluding `phybench`.

## Current dataset-by-dataset status

| Dataset | Current evaluation subset | Status |
| --- | --- | --- |
| `seephys` | Custom 100-problem subset from `uncertainty_quantification_physical_reasoning/perturbations/seephys/problem_ids_for_perturbation.json` | `seephys` does not expose an official held-out split in the loader; the repo only records `train`. |
| `phyx` | Custom 100-problem subset from `uncertainty_quantification_physical_reasoning/perturbations/phyx/problem_ids_for_perturbation.json` | `phyx` exposes `test_mini` (1000 problems) in the loader; the saved evaluation slice is a custom 100-problem subset of that split. |
| `physreason` | Official `mini` variant on the `test` split, 200 problems | Confirmed by the loader and by exact ID match against local `PhysReason-mini`. |
| `physbench` | Official `val` split, 200 problems | Confirmed by the loader and by exact ID match against local PhysBench `val`. |
| `physics` | Official `eval` split, 297 problems | Confirmed by loader metadata; the saved sample-set run also contains exactly 297 problems. |
| `ugphysics` | Custom English 130-problem subset, 10 problems per domain across 13 domains | This is a custom balanced-by-domain slice, not an official held-out split. |

## Important correction on `seephys` and `phyx`

The repository does not currently support the claim that the saved `seephys` and `phyx` subsets are a clean `50 both-correct + 50 both-incorrect` split under `gpt-5.2` and `gemini-2.5-pro`.

Using the stored `correctness_smart_llm` labels in:

- `uncertainty_quantification_physical_reasoning/exp_evaluation_set/sample_set/inference/response_with_answer_tag/seephys_gpt-5.2`
- `uncertainty_quantification_physical_reasoning/exp_evaluation_set/sample_set/inference/response_with_answer_tag/seephys_gemini-2.5-pro`
- `uncertainty_quantification_physical_reasoning/exp_evaluation_set/sample_set/inference/response_with_answer_tag/phyx_gpt-5.2`
- `uncertainty_quantification_physical_reasoning/exp_evaluation_set/sample_set/inference/response_with_answer_tag/phyx_gemini-2.5-pro`

the current saved subsets break down as:

- `seephys`: 65 both-correct, 21 both-incorrect, 14 disagreement
- `phyx`: 44 both-correct, 36 both-incorrect, 20 disagreement

So the safest wording is:

- `seephys` and `phyx` use fixed custom 100-problem evaluation subsets.
- The repo does not preserve a reproducible subset-generation script or note that proves a `50/50` correctness-balanced construction for the current saved slices.

## Recommended short description for papers / reports

For evaluation subsets, we use official held-out splits when the benchmark provides them. Concretely, `physreason` uses the `mini` test set, `physbench` uses the `val` split, and `physics` uses the `eval` split. For datasets without a suitable official held-out slice, we use fixed repository-tracked subsets defined by `problem_ids_for_perturbation.json`.

For the custom subsets, `seephys` uses a fixed 100-problem subset, `phyx` uses a fixed 100-problem subset drawn from `test_mini`, and `ugphysics` uses a fixed English 130-problem subset with 10 problems from each of the 13 domains. If future custom subsets are regenerated, the selection rule should be saved alongside the ID file so the provenance is explicit.

## Paper-safe wording

Recommended wording for `seephys` and `phyx`:

> For benchmarks without a suitable official held-out split, we evaluate on fixed repository-tracked subsets and keep the same subset for all models. In particular, for SeePhys we use a fixed 100-problem subset from the dataset, and for PhyX we use a fixed 100-problem subset from `test_mini`. The exact problem IDs are included in the repository for reproducibility.

Recommended wording for the full benchmark paragraph:

> We use official held-out splits when available. Specifically, we use the PhysReason-mini test set, the PhysBench validation split, and the PHYSICS eval split. For benchmarks without a suitable official held-out split, we use fixed repository-tracked subsets: a 100-problem subset for SeePhys, a 100-problem subset from PhyX `test_mini`, and a 130-problem English subset for UGPhysics.

Avoid claiming any of the following unless a provenance script or note is recovered:

- random sampling
- stratified sampling by correctness
- a `50` both-correct / `50` both-incorrect construction
- balancing by GPT-5.2 / Gemini-2.5-Pro agreement

## Repo anchors

- Official split metadata lives in:
  - `src/prkit/datasets/loaders/seephys_loader.py`
  - `src/prkit/datasets/loaders/phyx_loader.py`
  - `src/prkit/datasets/loaders/physreason_loader.py`
  - `src/prkit/datasets/loaders/physbench_loader.py`
  - `src/prkit/datasets/loaders/physics_loader.py`
  - `src/prkit/datasets/loaders/ugphysics_loader.py`
- Fixed evaluation ID lists live in:
  - `uncertainty_quantification_physical_reasoning/perturbations/<dataset>/problem_ids_for_perturbation.json`
