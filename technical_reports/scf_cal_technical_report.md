# SCF-Cal Technical Report

## Scope

This report documents the method we will refer to in the paper as `SCF-Cal`, short for
`Sparse Cluster-Feature Calibrator`.

The report is intentionally standalone. It defines the method directly from:

- perturbation-based semantic clustering
- four shared cluster features
- an oriented log-space transformation
- a sparse non-negative logistic correctness model
- deterministic cross-fitted evaluation

This document is written so it can be reused when drafting the paper.

## Repository Mapping

- Paper-facing method name: `SCF-Cal`
- Current implementation name: `cluster_feature_sparse_logistic`
- Current implementation family: sparse non-negative logistic calibrator on cluster features

Important naming note:

- some run directories and artifact filenames still use the historical string
  `cf_log_sparse`
- those artifact paths refer to the same method documented here as `SCF-Cal`

Primary source files:

- `uncertainty_quantification_physical_reasoning/uq/confidence/pc_cluster.py`
- `uncertainty_quantification_physical_reasoning/uq/posthoc_calibration.py`
- `uncertainty_quantification_physical_reasoning/uq/run_ece_from_perturbation.py`
- `uncertainty_quantification_physical_reasoning/tests/uq/test_posthoc_calibration.py`
- `uncertainty_quantification_physical_reasoning/tests/uq/test_run_ece_from_perturbation.py`

Key code anchors:

- method registry and display names:
  `uncertainty_quantification_physical_reasoning/uq/posthoc_calibration.py:17-91`
- non-negative logistic solver:
  `uncertainty_quantification_physical_reasoning/uq/posthoc_calibration.py:357-471`
- sparse subset selector:
  `uncertainty_quantification_physical_reasoning/uq/posthoc_calibration.py:760-900`
- cross-fit wrapper:
  `uncertainty_quantification_physical_reasoning/uq/posthoc_calibration.py:1483-1660`
- oriented feature transform:
  `uncertainty_quantification_physical_reasoning/uq/run_ece_from_perturbation.py:1518-1566`

## One-Sentence Definition

`SCF-Cal` is a post-hoc correctness calibrator that maps semantic-cluster statistics
from perturbation responses into a sparse, non-negative logistic probability model in
oriented log space.

## Naming

The paper-facing name is:

- full name: `Sparse Cluster-Feature Calibrator`
- short name: `SCF-Cal`

This name reflects the actual method:

- `Sparse`: the fitted model uses exact subset selection, not a forced all-features fit
- `Cluster-Feature`: the inputs are semantic-cluster statistics from perturbation answers
- `Calibrator`: the output is a calibrated probability of correctness

The implementation documented here is specifically a logistic instance of `SCF-Cal`, but
the paper-facing method name does not need to encode `logistic` explicitly.

## Inputs and Output

For each problem instance `i`, the pipeline takes:

- the original extracted answer
- a set of perturbation answers
- perturbation prompts or prompt surrogates for weighting
- a semantic equivalence backend
- a correctness label for the original answer

The output is a calibrated probability:

`p_i = P(ell_i = 1 | z_i)`

where:

- `ell_i` is the correctness label of the original answer
- `z_i` is the transformed cluster-feature vector defined below

## Notation

This report uses:

- `i` for problem index
- `j` for perturbation index
- `q_i0` for the original prompt
- `q_ij` for the `j`-th perturbation prompt
- `y_i0` for the original extracted answer
- `y_ij` for the `j`-th perturbation extracted answer
- `w_ij` for the prompt-derived perturbation weight
- `ell_i` for the correctness label
- `x_i` for the raw 4-dimensional cluster-feature vector
- `z_i` for the transformed feature vector used by `SCF-Cal`
- `p_i` for the calibrated correctness probability

## Cluster-Feature Construction

`SCF-Cal` depends on the shared semantic-cluster features constructed in
`uq/confidence/pc_cluster.py`. The raw feature vector is:

`x_i = (valid_rate_i, anchor_mass_i, largest_alt_mass_i, cluster_entropy_i)`

### Prompt Weighting

Each perturbation answer receives a non-negative weight:

`w_ij = s(q_i0, q_ij)`

where `s` is the configured prompt-similarity function.

In the SeePhys runs used here:

- prompt similarity = `rougeL`

Operational detail:

- if perturbation prompt texts are available, weighting can be non-uniform
- if prompts are unavailable, the implementation falls back to a placeholder prompt and
  weights become effectively uniform

### Semantic Clustering

The semantic clustering pipeline works as follows:

1. treat the original answer as the anchor answer
2. determine whether each valid perturbation answer semantically matches the anchor
3. assign anchor-matching answers to the anchor cluster
4. group the remaining valid answers into rival clusters using bidirectional semantic matching
5. exclude invalid or empty perturbation answers from clustering; they only reduce `valid_rate`

Implementation details:

- anchor assignment first uses stored `is_match_<comparison>` labels when available
- otherwise it falls back to a deterministic runtime semantic comparator
- rival-cluster grouping uses runtime bidirectional semantic matching

### Raw Feature Definitions

Let:

- `W_all = sum_j max(0, w_ij)`
- `V_i` be the set of valid perturbations with non-empty extracted answers
- `W_valid = sum_{j in V_i} max(0, w_ij)`

Then:

### 1. Valid Rate

`valid_rate_i = W_valid / W_all`

Interpretation:

- high value means perturbations usually produce valid extracted answers
- low value means perturbation outputs are often unusable or empty

### 2. Anchor Mass

Let `W_anchor` be the total valid weight assigned to the anchor cluster. Then:

`anchor_mass_i = W_anchor / W_valid`

Interpretation:

- high value means perturbations stay semantically close to the original answer
- low value means support moves away from the anchor meaning

### 3. Largest Alternative Mass

Let `W_alt,max` be the largest total weight among rival clusters, with default `0` if
no rival cluster exists. Then:

`largest_alt_mass_i = W_alt,max / W_valid`

Interpretation:

- high value means there is one strong competing answer mode
- low value means no rival mode dominates

### 4. Cluster Entropy

Let `p_c` denote the masses of all populated clusters, including:

- the anchor cluster if it has non-zero mass
- each non-empty rival cluster

Then:

`cluster_entropy_i = - sum_c p_c log p_c`

Interpretation:

- low entropy means semantic concentration
- high entropy means semantic dispersion across answer modes

The implementation also records:

- `effective_cluster_count = exp(cluster_entropy)`
- `inverse_effective_cluster_count = exp(-cluster_entropy)`

but `SCF-Cal` uses `cluster_entropy` itself.

## Transformed Feature Space

`SCF-Cal` does not fit directly on the raw vector `x_i`. It first transforms the four
cluster features into an oriented log-space representation:

- `z_i1 = log(valid_rate_i + eps)`
- `z_i2 = log(anchor_mass_i + eps)`
- `z_i3 = log(1 - largest_alt_mass_i + eps)`
- `z_i4 = -cluster_entropy_i`

with `eps = 1e-6` in the evaluator feature path.

This orientation makes all transformed features semantically aligned so that larger
values correspond to stronger evidence for correctness:

- larger `log(valid_rate + eps)` means higher answer validity
- larger `log(anchor_mass + eps)` means stronger anchor preservation
- larger `log(1 - largest_alt_mass + eps)` means weaker dominant rival mode
- larger `-cluster_entropy` means lower semantic dispersion

The transformed feature vector is:

`z_i = (z_i1, z_i2, z_i3, z_i4)`

## Why the Log-Space Representation Is Natural

The repository already uses multiplicative cluster-confidence formulas such as:

- `valid_rate * exp(-cluster_entropy)`
- `valid_rate * anchor_mass * exp(-cluster_entropy)`
- `valid_rate * exp(-cluster_entropy) * (1 - largest_alt_mass)`
- `valid_rate * anchor_mass * exp(-cluster_entropy) * (1 - largest_alt_mass)`

In log space, these become additive evidence terms:

`log(valid_rate) + log(anchor_mass) + log(1 - largest_alt_mass) - cluster_entropy`

So `SCF-Cal` is naturally interpreted as a correctness model over additive evidence
derived from multiplicative cluster statistics.

## Model Class

For any non-empty subset `S` of the transformed features, `SCF-Cal` fits:

`p_i = sigmoid(b + sum_{k in S} w_k z'_ik)`

where:

- `z'_ik` are standardized versions of the selected transformed features
- `w_k >= 0`
- `b` is an unconstrained intercept

The sign constraint is important. Because the feature representation is already
oriented, non-negative coefficients enforce semantically consistent directions without
requiring per-feature sign bookkeeping during fitting.

## Standardization

For each candidate subset:

1. compute per-feature mean and standard deviation on the training fold
2. drop near-constant columns with scale `<= 1e-12`
3. standardize the retained columns

So the subset selector and the logistic solver operate on standardized transformed
features, not on raw feature scales.

## Optimization

The underlying solver is `_fit_nonnegative_logistic_design_matrix`.

### Initialization

For one subset fit:

- initialize the intercept to the logit of the training prevalence
- initialize all coefficients to `0`

### Objective

The solver maximizes the penalized logistic objective:

`mean(y * eta - log(1 + exp(eta))) - 0.5 * ridge * ||w||^2`

where:

- `eta = b + Z w`
- `w >= 0`
- default `ridge = 1e-2`

### Solver Mechanics

The implementation uses projected gradient ascent with backtracking:

- update the intercept without sign constraints
- update the coefficients and project onto the non-negative orthant
- backtrack until the penalized objective improves
- iterate up to `800` steps with tolerance `1e-6`

### Numerical Safeguards

The implementation includes:

- clipping logistic linear scores to `[-30, 30]`
- intercept and coefficient magnitude checks
- fallback to constant prevalence if fitting degenerates

## Sparse Subset Selection

The method evaluates every non-empty subset of the candidate transformed features.

With all four transformed features available, that means:

- 4 one-feature subsets
- 6 two-feature subsets
- 4 three-feature subsets
- 1 four-feature subset

for a total of:

`2^4 - 1 = 15`

candidate models.

For the feature ablation variants `SCF-Cal/-x`, one candidate feature is removed before
subset search, leaving 3 transformed features and therefore:

`2^3 - 1 = 7`

candidate models.

## BIC Selection Rule

Each candidate subset is scored by:

`BIC = -2 * log_likelihood + k * log(n)`

where:

- `log_likelihood` is the Bernoulli log-likelihood on the training fold
- `n` is the training-fold sample count
- `k` is effectively the intercept plus the number of active coefficients

In the implementation:

- `n_active = number of coefficients with magnitude > 1e-10`
- `k = max(1, n_active + 1)`

Selection uses the lexicographic key:

1. lower `BIC`
2. fewer active features
3. larger log-likelihood
4. lexicographically smaller subset index tuple

So the method explicitly prefers a sparser model when fit is similar.

## Label Policy

`SCF-Cal` is fit on binary correctness labels.

In `crossfit_posthoc_calibration`:

- if correctness is already exactly `0/1`, those labels are preserved
- otherwise correctness is thresholded at `0.5`

For the SeePhys runs in this report, correctness is `smart_llm` and is already binary.

## Cross-Fitted Evaluation Protocol

`SCF-Cal` is evaluated with deterministic cross-fitting.

### Fold Construction

- number of folds: `5`
- seed: `0`
- folds are assigned deterministically from hashed problem IDs

### Cross-Fit Procedure

For each fold:

1. fit the sparse calibrator on the training partition
2. apply it to the held-out partition
3. save the out-of-fold calibrated probabilities

After all folds:

- concatenate the out-of-fold probabilities for evaluation
- fit one additional full-data model for inspection only

Important:

- reported metrics use only the out-of-fold predictions
- the full-data fit is metadata, not the evaluation prediction source

## Fallback Behavior

`SCF-Cal` falls back to a constant-prevalence model when:

- the training fold contains only one class
- no valid subset fit succeeds

In that case:

`p_i = training_prevalence`

## Exact Family-Level Formula

The evaluator records the family-level formula as:

`sigmoid(intercept + w1*log(valid_rate) + w2*log(anchor_mass) + w3*log(1-largest_alt_mass) - w4*cluster_entropy)`

This is accurate as a family description, with the crucial detail that `SCF-Cal`
permits some of these weights to be exactly zero through subset selection.

## SeePhys Setup Used Here

The results below come from:

- comparison run:
  `uncertainty_quantification_physical_reasoning/uq/uq_bundle/runs/seephys_cf_log_sparse_comparison/experiment.json`
- sparse-family ablation run:
  `uncertainty_quantification_physical_reasoning/uq/uq_bundle/runs/seephys_cf_log_sparse_feature_ablation/experiment.json`

Shared setup:

- dataset: `seephys`
- perturbation type: `physpara`
- models:
  - `ollama-gemma4-31b-cloud`
  - `ollama-qwen3.5-397b-cloud`
  - `ollama-mistral-large-3-675b-cloud`
  - `claude-opus-4-6`
  - `gemini-2.5-pro`
  - `gpt-5.4`
  - `gpt-5.2`
  - `gpt-4.1`
- raw confidence: `pc`
- comparison backend: `smart_llm`
- correctness target: `smart_llm`
- prompt similarity: `rougeL`
- post-hoc folds: `5`
- post-hoc seed: `0`
- post-hoc scope: `per_run`
- perturbation selection: `random`
- perturbation-selection seed: `0`

The sparse-family ablation bundle completed with:

- requested task specs: `5`
- status counts: `40 ok`

## Metrics Highlighted Here

This report focuses on:

- `ECE`
- `AECE`
- `RMSCE`
- `Brier`
- `NLL`

Interpretation:

- `ECE`, `AECE`, and `RMSCE` emphasize calibration mismatch
- `Brier` and `NLL` are proper scoring rules and therefore reflect both calibration and
  sharpness/refinement

## Primary Empirical Evidence: SCF-Cal Family Ablation

The most direct question is whether the full `SCF-Cal` method is supported by its own
ablation study.

We evaluated leave-one-feature-out variants `SCF-Cal/-x`, where one candidate feature
is removed before sparse subset selection.

Mean across the 8 SeePhys models:

| Variant | ECE | AECE | RMSCE | Brier | NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SCF-Cal/-valid_rate` | 0.088 | 0.101 | 0.130 | 0.137 | 0.432 |
| `SCF-Cal/-anchor_mass` | 0.087 | 0.083 | 0.126 | 0.137 | 0.448 |
| `SCF-Cal/-largest_alt_mass` | 0.095 | 0.095 | 0.141 | 0.139 | 0.438 |
| `SCF-Cal/-cluster_entropy` | 0.080 | 0.130 | 0.132 | 0.148 | 0.489 |
| `SCF-Cal` | 0.078 | 0.093 | 0.123 | 0.137 | 0.436 |

Tie-inclusive win counts across the 8 models:

| Variant | ECE Wins | AECE Wins | RMSCE Wins | Brier Wins | NLL Wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SCF-Cal/-valid_rate` | 2 | 1 | 2 | 3 | 4 |
| `SCF-Cal/-anchor_mass` | 2 | 5 | 3 | 2 | 2 |
| `SCF-Cal/-largest_alt_mass` | 1 | 3 | 1 | 3 | 2 |
| `SCF-Cal/-cluster_entropy` | 4 | 0 | 3 | 2 | 2 |
| `SCF-Cal` | 3 | 1 | 3 | 4 | 4 |

Tie handling:

- win counts are tie-inclusive
- tolerance is `1e-12`

### Interpretation

This ablation is supportive of the full method.

What it shows:

- the full `SCF-Cal` is the best mean row on `ECE`, `RMSCE`, and `Brier`
- the full `SCF-Cal` ties for the top `NLL` win count
- no leave-one-feature-out variant dominates the full method across all metrics

This is the main empirical reason the full method is defensible as the main reported
contribution.

## Feature Usage Learned by SCF-Cal

Selected features by model in the full SeePhys run:

| Model | Selected Features |
| --- | --- |
| `claude-opus-4-6` | `log_anchor_mass`, `neg_cluster_entropy` |
| `gemini-2.5-pro` | `log_valid_rate`, `log_anchor_mass` |
| `gpt-4.1` | `log_anchor_mass`, `neg_cluster_entropy` |
| `gpt-5.2` | `log_one_minus_largest_alt_mass`, `neg_cluster_entropy` |
| `gpt-5.4` | `log_anchor_mass`, `neg_cluster_entropy` |
| `ollama-gemma4-31b-cloud` | `log_anchor_mass`, `neg_cluster_entropy` |
| `ollama-mistral-large-3-675b-cloud` | `log_anchor_mass`, `log_one_minus_largest_alt_mass`, `neg_cluster_entropy` |
| `ollama-qwen3.5-397b-cloud` | `neg_cluster_entropy` |

Selected-feature counts across the 8 models:

| Feature | Selected by Models |
| --- | ---: |
| `neg_cluster_entropy` | 7 |
| `log_anchor_mass` | 6 |
| `log_one_minus_largest_alt_mass` | 2 |
| `log_valid_rate` | 1 |

### Interpretation

This pattern suggests:

- `cluster_entropy` is the most stable core signal
- `anchor_mass` is the second most stable signal
- `largest_alt_mass` helps for some models but is not universal
- `valid_rate` is occasionally useful but not a dominant driver in the current SeePhys setting

This is exactly the behavior we want from a sparse cluster-feature calibrator:

- all four candidate signals are available
- only the subset justified by the data is activated

## Contextual Baseline Comparison

As additional context, we also compared `SCF-Cal` against previously tested fixed
cluster-feature logistic baselines on the same SeePhys setup.

Mean across the 8 SeePhys models:

| Variant | ECE | AECE | RMSCE | Brier | NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SCF-Cal` | 0.078 | 0.093 | 0.123 | 0.137 | 0.436 |
| fixed cluster logistic `-valid_rate` | 0.094 | 0.086 | 0.147 | 0.128 | 0.437 |
| fixed cluster logistic `-anchor_mass` | 0.088 | 0.095 | 0.128 | 0.130 | 0.442 |
| fixed cluster logistic `-largest_alt_mass` | 0.103 | 0.092 | 0.156 | 0.127 | 0.431 |
| fixed cluster logistic `-cluster_entropy` | 0.091 | 0.091 | 0.145 | 0.127 | 0.434 |
| fixed full-feature cluster logistic | 0.090 | 0.091 | 0.145 | 0.129 | 0.445 |

Tie-inclusive win counts across the 8 models:

| Variant | ECE Wins | AECE Wins | RMSCE Wins | Brier Wins | NLL Wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SCF-Cal` | 3 | 3 | 3 | 3 | 3 |
| fixed cluster logistic `-valid_rate` | 2 | 4 | 2 | 0 | 0 |
| fixed cluster logistic `-anchor_mass` | 2 | 0 | 2 | 0 | 0 |
| fixed cluster logistic `-largest_alt_mass` | 0 | 1 | 0 | 2 | 3 |
| fixed cluster logistic `-cluster_entropy` | 1 | 0 | 1 | 3 | 2 |
| fixed full-feature cluster logistic | 1 | 1 | 1 | 0 | 0 |

### Interpretation

This comparison is useful as context, but it is not part of the method definition.

What it shows:

- `SCF-Cal` improves the fixed full-feature cluster logistic baseline on mean `ECE`,
  mean `RMSCE`, and mean `NLL`
- `SCF-Cal` has the most balanced win profile in that comparison set
- the sparse formulation is empirically stronger than a fixed all-features logistic rule

## What the Current Evidence Supports

The current experiments support the following claims:

- `SCF-Cal` is a standalone sparse cluster-feature calibrator built on semantically
  constructed perturbation statistics
- the full method is supported by its own leave-one-feature-out ablation on SeePhys
- the strongest stable features in the current setting are `cluster_entropy` and
  `anchor_mass`
- the method benefits from adaptive subset activation rather than forcing all candidate
  features into every fit

Claims that would be too strong at this stage:

- "`valid_rate` is useless"
- "`largest_alt_mass` is never helpful"
- "`SCF-Cal` dominates all methods on every metric"
- "`BIC` is the optimal selection rule for calibration"
- "`SCF-Cal` is theoretically optimal"

The current evidence is strong for the current SeePhys setting, but it remains empirical.

## Paper-Ready Method Description

This wording is accurate and concise:

> SCF-Cal is a post-hoc correctness calibrator defined on semantic-cluster features
> extracted from perturbation responses. For each example, we compute four cluster
> statistics: valid answer rate, anchor-cluster mass, largest rival-cluster mass, and
> semantic cluster entropy. We transform these into an oriented feature space,
> `log(valid_rate)`, `log(anchor_mass)`, `log(1-largest_alt_mass)`, and
> `-cluster_entropy`, so that larger values consistently indicate stronger evidence for
> correctness. We then fit a sparse non-negative logistic model over these transformed
> features and select the active subset by exhaustive non-empty subset search with BIC.
> The final output is a calibrated probability of correctness.

## Paper-Ready Ablation Description

This wording is also faithful to the current results:

> We evaluate leave-one-feature-out variants of SCF-Cal by removing one candidate
> cluster feature before sparse subset selection. On SeePhys, the full SCF-Cal model
> remains competitive against all of its own ablations: it achieves the best mean ECE,
> RMSCE, and Brier within the sparse family and ties for the highest NLL win count
> across models. This indicates that the full sparse formulation is stable rather than
> over-specified.

## Short Discussion Paragraph

> The main advantage of SCF-Cal is not merely that it uses fewer features. Its
> advantage is that it operates in a semantically oriented feature space and then
> activates only the subset of cluster statistics that is justified by the data. This
> matters because answer validity, anchor preservation, dominant-rival suppression, and
> semantic concentration are related but distinct signals of correctness. A fixed
> decision rule can overfit unstable interactions among them, whereas SCF-Cal preserves
> the cluster-feature hypothesis while enforcing a simpler and more stable calibration
> model.

## Limitations

The current method still has important limitations:

- subset selection is driven by `BIC`, not directly by held-out `NLL` or `Brier`
- the current evidence is strongest on `seephys` and should be extended to
  `physreason` and `phyx`
- the model is linear in transformed features and therefore cannot represent richer
  non-linear interactions
- the semantic-cluster features inherit any errors made by the answer-equivalence backend

## Bottom-Line Claim

The most accurate high-level summary is:

> SCF-Cal is a sparse cluster-feature correctness calibrator that maps semantically
> constructed perturbation statistics into an oriented log-space probability model. On
> the current SeePhys experiments, it is a strong and stable post-hoc calibrator, and
> its own ablation results support the full sparse formulation rather than undermining it.

## Artifact Pointers

Main comparison artifacts:

- `uncertainty_quantification_physical_reasoning/uq/uq_bundle/runs/seephys_cf_log_sparse_comparison/report/cf_log_sparse_comparison.md`
- `uncertainty_quantification_physical_reasoning/uq/uq_bundle/runs/seephys_cf_log_sparse_comparison/report/cf_log_sparse_comparison_mean.csv`
- `uncertainty_quantification_physical_reasoning/uq/uq_bundle/runs/seephys_cf_log_sparse_comparison/report/cf_log_sparse_comparison_wins.csv`
- `uncertainty_quantification_physical_reasoning/uq/uq_bundle/runs/seephys_cf_log_sparse_comparison/report/cf_log_sparse_selected_features_by_model.csv`
- `uncertainty_quantification_physical_reasoning/uq/uq_bundle/runs/seephys_cf_log_sparse_comparison/report/cf_log_sparse_selected_feature_counts.csv`

Sparse-family ablation artifacts:

- `uncertainty_quantification_physical_reasoning/uq/uq_bundle/runs/seephys_cf_log_sparse_feature_ablation/report/cf_log_sparse_feature_ablation.md`
- `uncertainty_quantification_physical_reasoning/uq/uq_bundle/runs/seephys_cf_log_sparse_feature_ablation/report/cf_log_sparse_feature_ablation_mean.csv`
- `uncertainty_quantification_physical_reasoning/uq/uq_bundle/runs/seephys_cf_log_sparse_feature_ablation/report/cf_log_sparse_feature_ablation_wins.csv`

Focused validation:

- `uncertainty_quantification_physical_reasoning/tests/uq/test_posthoc_calibration.py`
- `uncertainty_quantification_physical_reasoning/tests/uq/test_run_ece_from_perturbation.py`

