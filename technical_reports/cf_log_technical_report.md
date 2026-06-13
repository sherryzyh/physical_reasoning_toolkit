# CF-Log Technical Report

## Scope and Code Mapping

This report documents the method called `CF-Log` in the paper assets and the method implemented as `cluster_feature_logistic` in the uncertainty quantification code.

This document is intended to be self-contained. A reader should be able to understand:

- the raw confidence methods that feed the calibration stack
- how perturbation weighting and semantic clustering work
- what `CF-Log` is learning
- how the repository evaluates calibration, discrimination, and selective prediction

without opening the implementation files.

There is no literal `CL-Log` method name in the repository. The repository-backed mapping is:

- paper-facing label: `CF-Log`
- implementation name: `cluster_feature_logistic`
- related legacy exploratory variant: `multifeature_logistic`

The safest repository-faithful interpretation is:

> CF-Log = Cluster-Feature Logistic

Relevant source files:

- `uncertainty_quantification_physical_reasoning/uq/seephys_rerun_paper_assets/seephys_experimental_setup.md`
- `uncertainty_quantification_physical_reasoning/uq/posthoc_calibration.py`
- `uncertainty_quantification_physical_reasoning/uq/confidence/pc_cluster.py`
- `uncertainty_quantification_physical_reasoning/uq/confidence/perturbation_consistency.py`
- `uncertainty_quantification_physical_reasoning/uq/run_ece_from_perturbation.py`
- `uncertainty_quantification_physical_reasoning/uq/run_full_uq_evaluation_from_perturbation.py`
- `uncertainty_quantification_physical_reasoning/uq/metrics/ece.py`
- `uncertainty_quantification_physical_reasoning/uq/metrics/calibration_extended.py`
- `uncertainty_quantification_physical_reasoning/uq/metrics/brier.py`
- `uncertainty_quantification_physical_reasoning/uq/metrics/auroc.py`
- `uncertainty_quantification_physical_reasoning/uq/metrics/auprc.py`
- `uncertainty_quantification_physical_reasoning/uq/metrics/risk_coverage.py`
- `uncertainty_quantification_physical_reasoning/tests/uq/test_posthoc_calibration.py`
- `uncertainty_quantification_physical_reasoning/tests/uq/test_run_ece_from_perturbation.py`

## What CF-Log Is

CF-Log is a post-hoc probabilistic correctness model built on top of a shared feature vector extracted from semantic clustering over perturbation answers.

It is not a raw confidence constructor such as:

- `pc`
- `physics_semantic_consistency`
- `semantic_entropy`
- `anchored_semantic_entropy`
- `contrastive_semantic_entropy`
- `contrastive_anchored_semantic_entropy`

Instead, it takes a four-dimensional perturbation-cluster feature vector and learns a logistic mapping to the probability that the original answer is correct.

In the repository paper assets, CF-Log is the proposed method for the calibration and proper-scoring dimensions.

## Notation

This report uses the following symbols consistently:

- problem index: $i$
- perturbation index within one problem: $j$
- original prompt: $q_{i0}$
- perturbed prompt: $q_{ij}$
- original extracted answer: $\hat{y}_{i0}$
- perturbed extracted answer: $\hat{y}_{ij}$
- raw confidence score from a chosen confidence constructor: $c_i$
- correctness label for the original answer: $\ell_i \in [0,1]$
- post-hoc calibrated probability after CF-Log: $p_i$

In the main SeePhys CF-Log setting, $\ell_i$ is binary and comes from the `smart_llm`
correctness channel. Some other repository evaluations allow fractional correctness
values such as token F1; this report explicitly states when a metric uses fractional
labels directly and when labels are binarized.

## Inputs and Outputs

For each problem instance, the pipeline starts from:

- the original extracted answer
- a set of perturbation answers
- optional perturbation prompt texts for weighting
- per-instance correctness labels for the original answer

CF-Log outputs a scalar probability:

$$
p_i = P(\ell_i = 1 \mid x_i)
$$

where:

- $\ell_i = 1$ means the original answer is correct
- $x_i$ is the shared cluster-feature vector for instance $i$

Equivalently, CF-Log estimates the probability that the original answer
$\hat{y}_{i0}$ is correct, conditional on the perturbation-cluster feature vector
derived from the perturbed answers $\hat{y}_{ij}$.

## Confidence Methods and Naming

The repository contains both raw confidence methods and post-hoc calibrators. The
main confidence names relevant to CF-Log are:

- `pc` = plain perturbation consistency (PC)
- `physics_semantic_consistency` = cluster-aware perturbation confidence; legacy alias `pc_cluster`
- `semantic_entropy` = semantic entropy (SE)
- `anchored_semantic_entropy` = anchored semantic entropy (ASE)
- `contrastive_semantic_entropy` = contrastive semantic entropy (CSE)
- `contrastive_anchored_semantic_entropy` = contrastive anchored semantic entropy (CASE)
- `cluster_feature_logistic` = CF-Log, the post-hoc logistic model over shared cluster features

Two names that are easy to confuse must be separated:

- plain `pc` is the simple anchor-agreement score
- `physics_semantic_consistency` is the cluster-aware method that historically came from the old alias `pc_cluster`

So in current repository terms, `pc` and `pc_cluster` are not the same method.

## Shared Cluster Features

The feature vector used by CF-Log is:

$$
x_i = (v_i, a_i, r_i, H_i)
$$

with:

- $v_i$ = `valid_rate`
- $a_i$ = `anchor_mass`
- $r_i$ = `largest_alt_mass`
- $H_i$ = `cluster_entropy`

These features are produced by the semantic-cluster code in `uq/confidence/pc_cluster.py` and surfaced into `confidence_detail_rows` in `uq/run_ece_from_perturbation.py`.

### Semantic Comparison and Cluster Construction

The cluster-family methods do not compare answers by raw string identity. They use a
binary semantic comparator, typically `smart_llm` in the paper-facing runs and
optionally `typed_llm` in other runs.

The cluster construction logic is:

1. treat the original answer $\hat{y}_{i0}$ as the anchor answer
2. decide whether each valid perturbation answer matches the anchor
3. assign anchor-matching answers to the anchor cluster
4. group the remaining valid non-anchor answers into rival clusters by bidirectional semantic equivalence
5. leave empty or invalid extracted answers unclustered; they only reduce `valid_rate`

Anchor assignment uses precomputed `is_match_<comparison>` labels when those are
available in the perturbation JSON. If they are unavailable, the runtime falls back
to a deterministic semantic matcher. Pairwise decisions may be cached on disk for
speed, but caching does not change the mathematics of the score.

### 1. Prompt Weighting

Each perturbation answer is assigned a nonnegative weight:

$$
w_{ij} = s(q_{i0}, q_{ij})
$$

where `s` is the configured prompt-similarity function.

Prompt weighting is used by:

- `isc`
- `physics_semantic_consistency`
- `semantic_entropy`
- `anchored_semantic_entropy`
- `contrastive_semantic_entropy`
- `contrastive_anchored_semantic_entropy`

Prompt weighting is not used by plain `pc`; in plain `pc`, each perturbation counts
equally.

The available prompt-similarity functions are:

- `uniform`
- `rougeL`
- `bertscore`

The repository default is `rougeL`, and the historical SeePhys reference bundle also
uses `rougeL`.

Operationally, this means:

- if perturbation prompt texts are available, `rougeL` or `bertscore` can assign non-uniform weights based on prompt closeness to the original prompt
- if the method is configured as `uniform`, all perturbations receive equal weight
- if prompt metadata cannot be recovered, the code falls back to the base prompt for every perturbation, so even `rougeL` or `bertscore` collapse to effectively equal weights

So the intended default is prompt-aware weighting, but the realized behavior can
still become uniform when prompt texts are unavailable.

### 2. Valid Rate

Let:

- $W_{all} = \sum_j \max(0, w_{ij})$
- $V_i$ be the set of perturbations whose extracted answers are non-empty
- $W_{valid} = \sum_{j \in V_i} \max(0, w_{ij})$

Then:

$$
v_i = \frac{W_{valid}}{W_{all}}
$$

Interpretation:

- high `valid_rate` means perturbations reliably produce parseable answers
- low `valid_rate` means many perturbations yield empty or invalid extracted answers

### 3. Anchor Mass

The code defines an anchor cluster corresponding to the original answer.

Anchor membership is determined by:

- stored `is_match_<comparison>` labels when available
- otherwise a deterministic runtime comparator

Let $W_{anchor}$ be the total valid perturbation weight assigned to the anchor cluster. Then:

$$
a_i = \frac{W_{anchor}}{W_{valid}}
$$

Interpretation:

- high `anchor_mass` means perturbations stay with the original answer
- low `anchor_mass` means perturbations drift away from the original answer

### 4. Largest Alternative Mass

All valid non-anchor perturbation answers are grouped into rival clusters using bidirectional semantic matching.

If each populated rival cluster $c$ has weight $W_c$, then:

$$
r_i = \max_{c \neq anchor} \frac{W_c}{W_{valid}}
$$

Interpretation:

- high `largest_alt_mass` means one strong rival answer mode exists
- low `largest_alt_mass` means no dominant competing answer cluster exists

### 5. Cluster Entropy

Let the populated cluster masses over valid perturbations be $\{p_c\}$. Then:

$$
H_i = -\sum_c p_c \log p_c
$$

The code also computes:

- `normalized_entropy`
- `effective_cluster_count = exp(H_i)`
- `inverse_effective_cluster_count = exp(-H_i)`

but strict CF-Log uses only `cluster_entropy` itself.

Interpretation:

- low entropy means perturbations concentrate on one semantic meaning
- high entropy means perturbations spread across multiple semantic meanings

## Raw Confidence Scores Compared to CF-Log

Several raw confidence methods in the repository are fixed formulas over either:

- direct anchor agreement (`pc`), or
- the shared cluster ingredients $(v, a, r, H)$ defined above.

### Plain Perturbation Consistency (PC)

In the paper-facing binary-comparison setting, plain `pc` is:

$$
c_i^{pc}
=
\frac{\#\{j : \hat{y}_{ij} \text{ is valid and matches } \hat{y}_{i0}\}}
{\#\{j : \hat{y}_{ij} \text{ is valid}\}}
$$

Text meaning:

- confidence is high when most parseable perturbations preserve the original answer
- confidence is low when perturbations often change the answer
- plain `pc` does not distinguish a single strong rival answer mode from scattered disagreement
- plain `pc` does not use prompt weights

The code also permits `comparison=rougeL` for `pc`. In that lexical mode, `pc`
becomes an unweighted mean pairwise similarity score rather than a binary agreement
fraction. The paper-facing `smart_llm` / `typed_llm` setting is the binary one above.

### Physics Semantic Consistency (legacy `pc_cluster`)

$$
c_i^{psc} = v_i a_i (1-r_i)
$$

Text meaning:

- high `valid_rate` rewards perturbations that still yield parseable answers
- high `anchor_mass` rewards staying with the original answer cluster
- low `largest_alt_mass` rewards the absence of a strong competing answer cluster

### Semantic Entropy

$$
c_i^{se} = v_i e^{-H_i}
$$

Text meaning:

- confidence is high when valid perturbation answers collapse onto one semantic meaning
- confidence is low when answers fragment across multiple semantic meanings
- this score is anchor-agnostic: it cares about semantic concentration, not whether the dominant cluster is the original answer

### Anchored Semantic Entropy (ASE)

$$
c_i^{ase} = v_i a_i e^{-H_i}
$$

Text meaning:

- confidence is high when perturbations stay near the original answer and remain semantically concentrated
- confidence drops if answers drift away from the anchor, even if they collapse onto some other stable meaning

### Contrastive Semantic Entropy (CSE)

$$
c_i^{cse} = v_i e^{-H_i}(1-r_i)
$$

Text meaning:

- confidence is high when the answer distribution is semantically concentrated
- confidence is additionally penalized when one strong rival answer mode emerges
- unlike ASE, this score still does not require the dominant cluster to be the original answer

### Contrastive Anchored Semantic Entropy (CASE)

$$
c_i^{case} = v_i a_i e^{-H_i}(1-r_i)
$$

Text meaning:

- confidence is high only when perturbations stay with the original answer, remain semantically concentrated, and avoid a strong rival answer cluster
- among the raw cluster-family scores, this is the most anchor-preserving and rival-sensitive

CF-Log differs from all of these raw methods because it does not hard-code one
multiplicative formula. It learns a data-driven mapping from the shared feature vector
to correctness probability.

## Formal Model

### Conceptual Paper Form

The paper asset writes this as $P(y=1 \mid x)$. In this report we rename that target
to $\ell$ so it does not collide with the answer-string notation $\hat{y}$. The same
conceptual model is therefore:

$$
P(\ell=1 \mid x) = \sigma(b_0 + b_v v + b_a a + b_r r + b_H H)
$$

where:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### Exact Implementation Form

The implementation in `uq/posthoc_calibration.py` standardizes each feature inside the training fold:

$$
z_{ik} = \frac{x_{ik} - \mu_k}{s_k}
$$

where:

- $\mu_k$ is the fold-specific feature mean
- $s_k$ is the fold-specific feature standard deviation
- if $s_k \le 10^{-12}$, the code replaces it with `1.0`

The fitted logit is:

$$
\eta_i = \beta_0 + \beta_v z_{iv} + \beta_a z_{ia} + \beta_r z_{ir} + \beta_H z_{iH}
$$

and the output probability is:

$$
p_i = \sigma(\eta_i)
$$

with the implementation clipping $\eta_i$ to `[-30, 30]` before applying the sigmoid.

So the exact operational formula is:

$$
p_i
=
\sigma\left(
\beta_0
+
\beta_v \frac{v_i-\mu_v}{s_v}
+
\beta_a \frac{a_i-\mu_a}{s_a}
+
\beta_r \frac{r_i-\mu_r}{s_r}
+
\beta_H \frac{H_i-\mu_H}{s_H}
\right)
$$

## Training Objective

The implementation fits a ridge-regularized logistic regression.

Equivalent objective:

$$
\min_{\beta_0,\beta}
-\sum_i \left[\ell_i \log p_i + (1-\ell_i)\log(1-p_i)\right]
+
\frac{\lambda}{2}\|\beta\|_2^2
$$

where:

- $p_i = \sigma(\beta_0 + \beta^T z_i)$
- $\lambda = 10^{-2}$
- the intercept is not penalized

Implementation details from `uq/posthoc_calibration.py`:

- solver: Newton-style iterative updates using gradient and Hessian
- `max_iter = 50`
- `tol = 1e-6`
- intercept initialized from the empirical training prevalence logit
- coefficient magnitudes above `1e3` are treated as unstable

## Labels Used for Calibration

CF-Log is fit on binary correctness labels.

The label policy is:

- if correctness values are already exactly `0/1`, keep them
- otherwise threshold at `0.5`

This logic is implemented in `binary_calibration_labels()` in `uq/posthoc_calibration.py`.

In the main UQ setup, the common label choice is:

- `correctness = smart_llm`

meaning that CF-Log learns the probability that the original answer is correct under that correctness criterion.

## Cross-Fitting Protocol

The repository does not evaluate CF-Log on in-sample probabilities from the full fit. It evaluates out-of-fold predictions from grouped cross-fitting.

### Fold Assignment

Fold assignment is deterministic by `problem_id`:

1. compute `sha256(f"{seed}:{problem_id}")`
2. sort unique problem IDs by digest
3. assign fold index by `rank % n_folds`

This ensures:

- repeated occurrences of the same `problem_id` share a fold
- cross-fitting is deterministic for a given seed

### Default Settings

Common settings in the repo:

- `folds = 5`
- `seed = 0`
- `scope = per_run`
- `num_bins = 10`
- `correctness = smart_llm` in the main SeePhys bundle
- `comparison = smart_llm` for the cluster-family methods
- `prompt_similarity = rougeL`

### What Is Evaluated

For each fold:

1. fit the logistic model on the training folds
2. predict probabilities on the held-out fold

The code then concatenates all out-of-fold predictions and evaluates those probabilities using:

- `ECE`
- `AECE`
- `Brier`
- `NLL`
- `AUROC`
- `AUPRC`
- `risk-coverage`
- `AURC`

The metadata explicitly notes that `full_fit` is saved for inspection or reuse but is not used for evaluation.

## How Many Models Are Learned in One CF-Log Evaluation?

For the standard repository setting with `posthoc_calibration_folds = 5`, the reported
CF-Log evaluation uses five learned fold-specific models:

1. one model trained on folds 2-5 and evaluated on fold 1
2. one model trained on folds 1, 3, 4, 5 and evaluated on fold 2
3. one model trained on folds 1, 2, 4, 5 and evaluated on fold 3
4. one model trained on folds 1, 2, 3, 5 and evaluated on fold 4
5. one model trained on folds 1-4 and evaluated on fold 5

So the repository does not report one single in-sample fit. It reports the union of
five out-of-fold prediction blocks.

After those fold-specific predictions are produced, the code also fits one additional
full-data model on all samples in the current calibration scope. That full fit is saved
for inspection or future reuse, but it is not the object used to produce the reported
ECE, Brier, NLL, AUROC, AUPRC, or AURC values.

This detail matters for interpretation:

- evaluation-time CF-Log = 5 fold-specific models in the default setting
- archival full fit = 1 extra model saved after evaluation

## Is CF-Log Being Evaluated on the Same Run?

Yes, but the phrase "same run" needs to be stated precisely.

There are two distinct notions:

- same-sample fit-and-score
- same-distribution in-run cross-fitting

CF-Log is **not** evaluated by fitting on a sample and then reporting that same
sample's fitted probability. Every reported sample-level probability is out-of-fold.
So there is no direct same-sample leakage in the default CF-Log metrics.

However, CF-Log **is** learned on the same evaluation slice distribution when
`scope = per_run`. In practical terms, the current paper-facing and consolidated runs
use the same dataset/model/perturbation slice to:

- collect features
- fit the fold-specific logistic models
- evaluate the held-out predictions

That means CF-Log is best interpreted as an **in-run learned calibrator**. Its strong
results are legitimate within-run out-of-fold results, but they are not yet the same as
evidence of cross-run transfer.

For the current consolidated bundle runs, the relevant defaults are:

- `posthoc_calibration_folds = 5`
- `posthoc_calibration_seed = 0`
- `posthoc_calibration_scope = per_run`
- `correctness = smart_llm`

So the main question is not "is this direct leakage?" but rather:

> how much of the observed gain is a stable modeling advantage, and how much is
> adaptation to the current run's feature and label distribution?

## Evaluation Metrics

This section defines the repository metrics in self-contained form.

### Label Policy by Metric

The repository uses two label regimes:

- calibration-style metrics can consume correctness values in $[0,1]$
- discrimination and selective-prediction metrics require binary labels

Concretely:

- `ECE`, `AECE`, and `Brier` use correctness values directly, even if they are fractional
- `AUROC`, `AUPRC`, `AURC`, binary `NLL`, calibration intercept/slope, and `ICI` use binary labels

If correctness labels are already exactly `0/1`, the code keeps them. Otherwise it
binarizes with the rule:

$$
\ell_i^{bin} = \mathbf{1}[\ell_i \ge 0.5]
$$

In the main CF-Log setting with `correctness = smart_llm`, labels are already binary,
so this thresholding step is usually inactive.

### Calibration Metrics

#### Expected Calibration Error (ECE)

The repository ECE uses equal-width bins over $[0,1]$, with $M = 10$ by default.
Confidence value `1.0` is included in the last bin.

Let $B_m$ be bin $m$. Then:

$$
\mathrm{ECE}
=
\sum_{m=1}^M
\frac{|B_m|}{N}
\left|
\mathrm{acc}(B_m) - \mathrm{conf}(B_m)
\right|
$$

where:

- $\mathrm{acc}(B_m)$ is mean correctness in bin $m$
- $\mathrm{conf}(B_m)$ is mean confidence in bin $m$

Lower is better.

#### Adaptive ECE (AECE)

AECE uses the same absolute calibration gap, but after sorting samples by confidence
and splitting them into equal-mass bins rather than equal-width bins:

$$
\mathrm{AECE}
=
\sum_{m=1}^M
\frac{|B_m|}{N}
\left|
\mathrm{acc}(B_m) - \mathrm{conf}(B_m)
\right|
$$

with $B_m$ now defined by equal-count partitions of the confidence-sorted sample list.
Lower is better.

#### MCE and RMSCE

The extended calibration diagnostics also report:

$$
\mathrm{MCE} = \max_m |\mathrm{acc}(B_m) - \mathrm{conf}(B_m)|
$$

and

$$
\mathrm{RMSCE}
=
\sqrt{
\sum_{m=1}^M \frac{|B_m|}{N}
\left(\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right)^2
}
$$

using the same equal-width bins as ECE.

#### Brier Score

The Brier score is:

$$ 
\mathrm{Brier}
=
\frac{1}{N}\sum_{i=1}^N (p_i - \ell_i)^2
$$

Lower is better.

#### Binary Negative Log-Likelihood (NLL)

For binary evaluation, probabilities are clipped to $[\varepsilon, 1-\varepsilon]$
with $\varepsilon = 10^{-12}$, then:

$$
\mathrm{NLL}
=
-\frac{1}{N}\sum_{i=1}^N
\left[
\ell_i^{bin}\log p_i + (1-\ell_i^{bin})\log(1-p_i)
\right]
$$

Lower is better.

### Discrimination and Selective-Prediction Metrics

#### AUROC

`AUROC` measures ranking quality for binary correctness labels. In words, it is the
probability that a randomly chosen correct sample receives a higher confidence than a
randomly chosen incorrect sample, with tied scores handled by average ranks.

Higher is better.

#### AUPRC

`AUPRC` is average precision for the binary task "correct" vs "incorrect". Samples
are sorted by decreasing confidence, and the metric averages the precision achieved
at the ranks where a correct sample appears.

Higher is better. The natural baseline is the positive prevalence.

#### Risk-Coverage Curve

For selective prediction, samples are sorted by decreasing confidence. For the top-$k$
prefix:

$$
\mathrm{coverage}_k = \frac{k}{N}
$$

$$
\mathrm{risk}_k
=
\frac{1}{k}\sum_{j=1}^k (1-\ell_{(j)}^{bin})
$$

where $(j)$ denotes the $j$-th sample in descending-confidence order.

Interpretation:

- coverage asks how much of the dataset is retained
- risk asks what error rate remains on the retained subset

#### Area Under the Risk-Coverage Curve (AURC)

`AURC` is the trapezoidal area under the repository's risk-coverage curve over the
discrete points $(\mathrm{coverage}_k, \mathrm{risk}_k)$ for $k = 1,\ldots,N$.

Lower is better.

### Additional Reported Summaries

Repository summaries also commonly report:

- `accuracy = mean(\ell_i^{bin})`
- `mean_confidence = mean(p_i)` or `mean(c_i)` depending on whether the score is calibrated

The full evaluation diagnostics can additionally report calibration intercept,
calibration slope, and ICI. Those are useful diagnostics but are not the defining
metrics of CF-Log itself.

### Which Stage Produces Which Metrics

The repository evaluation stack has two layers:

1. `run_ece_from_perturbation.py` computes raw or post-hoc confidence values, writes
   per-sample outputs, and computes equal-width ECE.
2. `run_full_uq_evaluation_from_perturbation.py` and the bundle analysis scripts read
   those per-sample outputs and add `AECE`, `Brier`, `NLL`, `AUROC`, `AUPRC`,
   `risk-coverage`, `AURC`, and the extended calibration diagnostics.

## Degeneracy and Fallback Behavior

If a training fold cannot support a stable logistic fit, the code falls back to a constant prevalence predictor.

This happens when:

- the fold contains only one class
- fitted parameters become non-finite
- coefficient magnitudes exceed the instability threshold

The fallback predictor is:

$$
p_i = \frac{1}{n_{train}} \sum_{j \in train} \ell_j
$$

Metadata recorded in `posthoc_calibration.json` includes:

- `n_constant_prevalence_folds`
- `n_constant_output_folds`
- `is_partially_degenerate`
- `is_fully_degenerate`
- `degeneracy_reasons`

## Scope: Per-Run vs Shared-Model

The implementation supports two calibration scopes:

### `per_run`

The calibrator is fit only on the current run:

- one dataset
- one model
- one perturbation type
- one confidence run

This is the default and the paper-facing setting.

### `shared_model`

The calibrator pools sibling perturbation types for the same dataset/model before fitting.

The repository paper assets treat this as exploratory and not the preferred main setting. The discussion notes that naive pooling did not outperform the per-run CF-Log fit.

## Bias Caveats with `smart_llm`

The main repository setting commonly uses:

- `comparison = smart_llm` to build semantic clustering features
- `correctness = smart_llm` as the supervision target

This creates two distinct caveats.

### 1. Proxy-Label Bias

If cross-fitting is done correctly, training and evaluating on `smart_llm` does not
automatically create leakage. But it does mean that CF-Log is optimized for the
`smart_llm` notion of correctness.

So a strong CF-Log result should be read as:

> CF-Log predicts `smart_llm`-defined correctness well in an out-of-fold sense.

This is not automatically identical to:

> CF-Log predicts true physical correctness equally well.

### 2. Shared-Judge Bias

There is also a same-judge or shared-judge bias risk. The cluster features are
constructed using `smart_llm`-style semantic matching, while the supervision label can
also be `smart_llm` correctness. In that case the feature generator and the target
labeler are not statistically independent.

This is not classical data leakage, but it can make the method look more optimistic
with respect to that particular proxy.

The clean interpretation is therefore:

- CF-Log is not same-sample cheating in the default cross-fit protocol
- CF-Log may still benefit from proxy-specific alignment when both features and labels depend on `smart_llm`

## Why CF-Log Can Look Strong

CF-Log has two structural advantages over the raw score-family methods:

1. It directly learns a probability map from the shared cluster features instead of
   fixing one multiplicative formula by hand.
2. It learns that map on the current run distribution under grouped cross-fitting.

This combination explains why CF-Log often looks strongest on:

- `ECE`
- `Brier`
- `NLL`

while the hand-crafted score variants can remain stronger on:

- `AUROC`
- `AUPRC`
- `AURC`

The raw score family is often stronger at ranking, while CF-Log is better at mapping
feature patterns into probability scale.

## Geometric-Score Interpretation of the Raw Score Family

The strongest raw discrimination methods in this repository family are:

- `PSC`
- `ASE`
- `CSE`
- `CASE`

All of them can be viewed as multiplicative confidence constructors over the same
shared ingredients `(v, a, r, H)`.

For example:

$$
c_i^{psc} = v_i a_i (1-r_i)
$$

$$
c_i^{ase} = v_i a_i e^{-H_i}
$$

$$
c_i^{cse} = v_i e^{-H_i}(1-r_i)
$$

$$
c_i^{case} = v_i a_i e^{-H_i}(1-r_i)
$$

This is important because it suggests that strong discrimination is coming from a
useful multiplicative geometry over the shared cluster features, not just from a raw
linear model in the original feature space.

## Proposed Follow-Up Variant

If the goal is to preserve the discrimination strength of the multiplicative score
family while retaining CF-Log-style probabilistic outputs, the most principled next
variant is a log-feature constrained logistic model.

Define:

$$
q_i = \alpha \log(v_i + \varepsilon)
+ \beta \log(a_i + \varepsilon)
+ \gamma \log(1-r_i + \varepsilon)
- \delta H_i
$$

Equivalently,

$$
s_i = (v_i + \varepsilon)^\alpha
(a_i + \varepsilon)^\beta
(1-r_i + \varepsilon)^\gamma
e^{-\delta H_i}
$$

with

$$
q_i = \log s_i
$$

Since `log` is monotone, `q_i` and `s_i` induce the same ranking.

The cleaner one-stage probabilistic form is:

$$
p_i =
\sigma\left(
b_0
+ b_v \log(v_i + \varepsilon)
+ b_a \log(a_i + \varepsilon)
+ b_r \log(1-r_i + \varepsilon)
- b_H H_i
\right)
$$

with monotone-sign constraints:

- $b_v \ge 0$
- $b_a \ge 0$
- $b_r \ge 0$
- $b_H \ge 0$

This variant has three attractive properties:

1. it preserves the multiplicative structure behind `PSC` / `ASE` / `CSE` / `CASE`
2. it directly outputs a probability, like CF-Log
3. it does not require an extra Platt-scaling stage because the logistic output is already probabilistic

In other words, it is a natural bridge between:

- the strong ranking behavior of the raw score family
- the strong calibration and proper-scoring behavior of CF-Log

## Recommended Review Protocol for a Next Variant

To distinguish true method improvement from run-specific adaptation, a follow-up
variant should be reviewed under several settings rather than only the default
`per_run` cross-fit setup.

Recommended evaluation layers:

1. **In-run grouped cross-fit**
   - same protocol as current CF-Log
   - useful for apples-to-apples comparison

2. **Cross-model transfer**
   - train on some models, evaluate on a held-out model

3. **Cross-dataset transfer**
   - train on some datasets, evaluate on a held-out dataset

4. **Stronger-label review when available**
   - if any adjudicated or stricter correctness subset exists, evaluate there as well

The main purpose of these stronger protocols is to answer:

> is the learned mapping robust beyond the run on which it was fit?

## Reproducible Command

From `uncertainty_quantification_physical_reasoning/`:

```bash
PYTHONPATH=".:../src" python -m uq.run_ece_from_perturbation \
  --perturbation-dir experiment_results/perturbation/physpara/seephys_gpt-5.4 \
  --confidence physics_semantic_consistency \
  --comparison smart_llm \
  --prompt-similarity rougeL \
  --correctness smart_llm \
  --posthoc-calibration cluster_feature_logistic \
  --posthoc-calibration-folds 5 \
  --posthoc-calibration-seed 0 \
  --posthoc-calibration-scope per_run
```

This is the most direct repository-backed reproduction of the CF-Log pipeline.

## Saved Artifacts

A run writes:

- `config.json`
- `posthoc_calibration.json`
- `ece_per_sample.csv`
- `ece_statistics.json`

Useful fields include:

- `posthoc_calibration = "cluster_feature_logistic"`
- `posthoc_calibration_feature_names = ["valid_rate", "anchor_mass", "largest_alt_mass", "cluster_entropy"]`
- per-sample `raw_confidence`
- per-sample `posthoc_calibration_fold`
- per-sample cluster feature columns such as:
  - `confidence_valid_rate`
  - `confidence_anchor_mass`
  - `confidence_largest_alt_mass`
  - `confidence_cluster_entropy`
  - `confidence_cluster_count`

## What CF-Log Is Not

CF-Log should not be conflated with:

- `physics_semantic_consistency` itself
- the legacy alias `pc_cluster`
- `multifeature_logistic`
- the direct supervised risk model in `uq/direct_risk_model.py`

The strict paper-facing CF-Log object is the four-feature post-hoc logistic model implemented as `cluster_feature_logistic`.

## Recommended Short Definition

A repository-faithful one-sentence definition is:

> CF-Log is a cross-fitted, ridge-regularized logistic correctness model over the shared perturbation-cluster feature vector `(valid_rate, anchor_mass, largest_alt_mass, cluster_entropy)`, producing an out-of-fold probability that the original answer is correct.

An interpretation-faithful extension is:

> CF-Log is a legitimate in-run, out-of-fold learned calibrator, not a same-sample fit-and-score method; its strongest claims are within-run probability quality, while cross-run robustness should be tested separately.
