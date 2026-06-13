# SCF-Cal Learned Model Analysis Report

Generated from saved calibration artifacts on 2026-04-15.

## Scope and Artifact Inventory

This report analyzes the learned model structure of the base `SCF-Cal` method
(`cluster_feature_sparse_logistic`) from saved post-hoc calibration artifacts only.

Included artifacts are runs matching:

- `**/posthoc_cluster_feature_sparse_logistic_crossfit_5_seed_0/*/*/posthoc_calibration.json`

Excluded artifacts:

- leave-one-feature-out ablation variants such as `cluster_feature_sparse_logistic_drop_*`
- methods from other post-hoc calibration families

- total analyzed runs: `16`
- datasets covered: `physreason, phyx, seephys`
- model names covered: `claude-opus-4-6, gemini-2.5-pro, gpt-4.1, gpt-5.2, gpt-5.4, ollama-qwen3.5-397b-cloud`

Primary per-run source files:

- `posthoc_calibration.json` for selected subsets, coefficients, BIC, and fold summaries
- `ece_per_sample.csv` for actual out-of-fold calibrated confidence and cluster features
- `uq_metrics_full.json` for ECE / NLL / Brier / AUROC summaries

## Main Findings

1. Across `16` saved base `SCF-Cal` runs, the most common `full_fit` subset is `{log_anchor_mass, neg_cluster_entropy}`; it appears in `5` runs.
2. The strongest globally stable feature is `neg_cluster_entropy`, selected in `13/16` `full_fit`s. `log_anchor_mass` is second at `7/16`.
3. No analyzed base run selected all four features in `full_fit` (`0` cases), and no fold-level fit selected all four features either (`0` cases). In this artifact set, the sparse search never judged the full 4-feature model necessary.
4. Model-level stability is highly heterogeneous. `gpt-4.1` is perfectly stable across the three datasets and always selects `{log_anchor_mass, neg_cluster_entropy}`. `gpt-5.4` is stable on an entropy-centered family, while `claude-opus-4-6` and `gemini-2.5-pro` are strongly dataset-adaptive.
5. Fold-level subset selection is not always identical even when `full_fit` is concise: `6/16` runs have exactly one fold-level subset, while the others show two or three distinct fold-level subsets. The sparse rule is stable in its low-dimensionality, but not always in the exact same active set on every fold.

## Per-Run Learned Model Summary

| Dataset | Model | Full-Fit Subset | Coefficients | BIC | Distinct Fold Subsets | Fold-Consistent | ECE | NLL | Mean Abs Shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| physreason | `claude-opus-4-6` | `{log_valid_rate, neg_cluster_entropy}` | `log_valid_rate=1.319; neg_cluster_entropy=0.650` | 170.436 | 1 | yes | 0.038 | 0.400 | 0.204 |
| physreason | `gemini-2.5-pro` | `{log_valid_rate, log_one_minus_largest_alt_mass, neg_cluster_entropy}` | `log_valid_rate=1.986; log_one_minus_largest_alt_mass=0.432; neg_cluster_entropy=0.628` | 162.138 | 2 | no | 0.045 | 0.404 | 0.233 |
| physreason | `gpt-4.1` | `{log_anchor_mass, neg_cluster_entropy}` | `log_anchor_mass=0.895; neg_cluster_entropy=1.039` | 214.327 | 1 | yes | 0.047 | 0.503 | 0.147 |
| physreason | `gpt-5.2` | `{log_anchor_mass, log_one_minus_largest_alt_mass, neg_cluster_entropy}` | `log_anchor_mass=1.126; log_one_minus_largest_alt_mass=1.211; neg_cluster_entropy=0.774` | 181.688 | 3 | no | 0.067 | 0.443 | 0.119 |
| physreason | `gpt-5.4` | `{log_anchor_mass, neg_cluster_entropy}` | `log_anchor_mass=0.822; neg_cluster_entropy=0.936` | 199.248 | 2 | no | 0.081 | 0.501 | 0.141 |
| phyx | `claude-opus-4-6` | `{log_valid_rate}` | `log_valid_rate=1.585` | 72.805 | 3 | no | 0.048 | 0.367 | 0.110 |
| phyx | `gemini-2.5-pro` | `{log_valid_rate}` | `log_valid_rate=2.104` | 69.130 | 1 | yes | 0.032 | 0.305 | 0.414 |
| phyx | `gpt-4.1` | `{log_anchor_mass, neg_cluster_entropy}` | `log_anchor_mass=0.621; neg_cluster_entropy=1.021` | 110.455 | 2 | no | 0.126 | 0.559 | 0.184 |
| phyx | `gpt-5.2` | `{log_one_minus_largest_alt_mass, neg_cluster_entropy}` | `log_one_minus_largest_alt_mass=1.292; neg_cluster_entropy=1.096` | 108.649 | 2 | no | 0.106 | 0.562 | 0.189 |
| phyx | `gpt-5.4` | `{neg_cluster_entropy}` | `neg_cluster_entropy=0.951` | 113.506 | 2 | no | 0.112 | 0.621 | 0.166 |
| seephys | `claude-opus-4-6` | `{log_anchor_mass, neg_cluster_entropy}` | `log_anchor_mass=1.708; neg_cluster_entropy=0.781` | 87.584 | 3 | no | 0.061 | 0.434 | 0.173 |
| seephys | `gemini-2.5-pro` | `{log_anchor_mass}` | `log_anchor_mass=1.293` | 91.413 | 2 | no | 0.042 | 0.450 | 0.232 |
| seephys | `gpt-4.1` | `{log_anchor_mass, neg_cluster_entropy}` | `log_anchor_mass=1.261; neg_cluster_entropy=1.237` | 92.749 | 1 | yes | 0.065 | 0.411 | 0.095 |
| seephys | `gpt-5.2` | `{neg_cluster_entropy}` | `neg_cluster_entropy=1.115` | 101.271 | 3 | no | 0.115 | 0.482 | 0.165 |
| seephys | `gpt-5.4` | `{neg_cluster_entropy}` | `neg_cluster_entropy=1.033` | 106.285 | 1 | yes | 0.067 | 0.499 | 0.166 |
| seephys | `ollama-qwen3.5-397b-cloud` | `{neg_cluster_entropy}` | `neg_cluster_entropy=1.515` | 91.779 | 1 | yes | 0.062 | 0.441 | 0.151 |

## Cross-Run Selection Patterns

- analyzed base `SCF-Cal` runs: `16`
- runs with 4-feature `full_fit`: `0`
- fold-level fits with 4-feature selection: `0`

### Full-Fit Subset Frequency

| Subset | Runs |
| --- | --- |
| `{log_anchor_mass, neg_cluster_entropy}` | 5 |
| `{neg_cluster_entropy}` | 4 |
| `{log_valid_rate}` | 2 |
| `{log_valid_rate, log_one_minus_largest_alt_mass, neg_cluster_entropy}` | 1 |
| `{log_valid_rate, neg_cluster_entropy}` | 1 |
| `{log_anchor_mass, log_one_minus_largest_alt_mass, neg_cluster_entropy}` | 1 |
| `{log_one_minus_largest_alt_mass, neg_cluster_entropy}` | 1 |
| `{log_anchor_mass}` | 1 |

### Full-Fit Feature Usage

| Feature | Runs Selected | Coverage |
| --- | --- | --- |
| `neg_cluster_entropy` | 13 | 13/16 |
| `log_anchor_mass` | 7 | 7/16 |
| `log_valid_rate` | 4 | 4/16 |
| `log_one_minus_largest_alt_mass` | 3 | 3/16 |

### Feature Count Distribution

_Full-fit subset size_

| Selected Features | Runs |
| --- | --- |
| 1 | 7 |
| 2 | 7 |
| 3 | 2 |

_Fold-level subset size_

| Selected Features | Fold Fits |
| --- | --- |
| 1 | 33 |
| 2 | 42 |
| 3 | 5 |

### Fold-Level Selection Variability

| Distinct Fold Subsets Within One Run | Runs |
| --- | --- |
| 1 | 6 |
| 2 | 6 |
| 3 | 4 |

## Cross-Dataset Stability by Model

### `claude-opus-4-6`

| Dataset | Full-Fit Subset | Fold Subsets | # Features | Fold-Consistent | Mean Abs Shift | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| physreason | `{log_valid_rate, neg_cluster_entropy}` | `{log_valid_rate, neg_cluster_entropy} x5` | 2 | yes | 0.204 | 0.038 |
| phyx | `{log_valid_rate}` | `{log_valid_rate} x2; {log_valid_rate, neg_cluster_entropy} x2; {log_anchor_mass} x1` | 1 | no | 0.110 | 0.048 |
| seephys | `{log_anchor_mass, neg_cluster_entropy}` | `{log_anchor_mass} x3; {log_anchor_mass, neg_cluster_entropy} x1; {log_one_minus_largest_alt_mass, neg_cluster_entropy} x1` | 2 | no | 0.173 | 0.061 |

- universal features across datasets: `{}`
- union of selected features across datasets: `{log_anchor_mass, log_valid_rate, neg_cluster_entropy}`
- pairwise Jaccard similarity of full-fit subsets: `0.500, 0.333, 0.000`

### `gemini-2.5-pro`

| Dataset | Full-Fit Subset | Fold Subsets | # Features | Fold-Consistent | Mean Abs Shift | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| physreason | `{log_valid_rate, log_one_minus_largest_alt_mass, neg_cluster_entropy}` | `{log_valid_rate, log_one_minus_largest_alt_mass, neg_cluster_entropy} x3; {log_valid_rate, neg_cluster_entropy} x2` | 3 | no | 0.233 | 0.045 |
| phyx | `{log_valid_rate}` | `{log_valid_rate} x5` | 1 | yes | 0.414 | 0.032 |
| seephys | `{log_anchor_mass}` | `{log_anchor_mass} x4; {log_valid_rate, log_one_minus_largest_alt_mass} x1` | 1 | no | 0.232 | 0.042 |

- universal features across datasets: `{}`
- union of selected features across datasets: `{log_anchor_mass, log_one_minus_largest_alt_mass, log_valid_rate, neg_cluster_entropy}`
- pairwise Jaccard similarity of full-fit subsets: `0.333, 0.000, 0.000`

### `gpt-4.1`

| Dataset | Full-Fit Subset | Fold Subsets | # Features | Fold-Consistent | Mean Abs Shift | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| physreason | `{log_anchor_mass, neg_cluster_entropy}` | `{log_anchor_mass, neg_cluster_entropy} x5` | 2 | yes | 0.147 | 0.047 |
| phyx | `{log_anchor_mass, neg_cluster_entropy}` | `{log_anchor_mass, neg_cluster_entropy} x4; {neg_cluster_entropy} x1` | 2 | no | 0.184 | 0.126 |
| seephys | `{log_anchor_mass, neg_cluster_entropy}` | `{log_anchor_mass, neg_cluster_entropy} x5` | 2 | yes | 0.095 | 0.065 |

- universal features across datasets: `{log_anchor_mass, neg_cluster_entropy}`
- union of selected features across datasets: `{log_anchor_mass, neg_cluster_entropy}`
- pairwise Jaccard similarity of full-fit subsets: `1.000, 1.000, 1.000`

### `gpt-5.2`

| Dataset | Full-Fit Subset | Fold Subsets | # Features | Fold-Consistent | Mean Abs Shift | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| physreason | `{log_anchor_mass, log_one_minus_largest_alt_mass, neg_cluster_entropy}` | `{log_anchor_mass, log_one_minus_largest_alt_mass, neg_cluster_entropy} x2; {log_anchor_mass, neg_cluster_entropy} x2; {log_anchor_mass, log_one_minus_largest_alt_mass} x1` | 3 | no | 0.119 | 0.067 |
| phyx | `{log_one_minus_largest_alt_mass, neg_cluster_entropy}` | `{log_one_minus_largest_alt_mass, neg_cluster_entropy} x3; {log_anchor_mass, neg_cluster_entropy} x2` | 2 | no | 0.189 | 0.106 |
| seephys | `{neg_cluster_entropy}` | `{log_one_minus_largest_alt_mass, neg_cluster_entropy} x2; {neg_cluster_entropy} x2; {log_anchor_mass, neg_cluster_entropy} x1` | 1 | no | 0.165 | 0.115 |

- universal features across datasets: `{neg_cluster_entropy}`
- union of selected features across datasets: `{log_anchor_mass, log_one_minus_largest_alt_mass, neg_cluster_entropy}`
- pairwise Jaccard similarity of full-fit subsets: `0.667, 0.333, 0.500`

### `gpt-5.4`

| Dataset | Full-Fit Subset | Fold Subsets | # Features | Fold-Consistent | Mean Abs Shift | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| physreason | `{log_anchor_mass, neg_cluster_entropy}` | `{log_anchor_mass, neg_cluster_entropy} x4; {neg_cluster_entropy} x1` | 2 | no | 0.141 | 0.081 |
| phyx | `{neg_cluster_entropy}` | `{neg_cluster_entropy} x4; {log_one_minus_largest_alt_mass, neg_cluster_entropy} x1` | 1 | no | 0.166 | 0.112 |
| seephys | `{neg_cluster_entropy}` | `{neg_cluster_entropy} x5` | 1 | yes | 0.166 | 0.067 |

- universal features across datasets: `{neg_cluster_entropy}`
- union of selected features across datasets: `{log_anchor_mass, neg_cluster_entropy}`
- pairwise Jaccard similarity of full-fit subsets: `0.500, 0.500, 1.000`

## Feature-Effect Analysis on Learned Outputs

This section combines two views of feature importance:

- `mean_abs_prob_delta_if_removed` is computed from the saved `full_fit` only. It measures how much the inspection model's probability would move, on average, if one selected feature were removed while keeping the other learned terms fixed.
- `q1/q4` confidence, accuracy, and shift statistics are computed from the actual out-of-fold `ece_per_sample.csv` outputs. They show how the realized calibrated outputs differ between low-feature and high-feature slices of the run.
- Negative `Q4-Q1` values can appear even for selected features. That does not contradict non-negative coefficients: these slice statistics are computed on the realized out-of-fold outputs, where multiple selected features interact and the exact fold-specific fit may differ slightly from the saved `full_fit`.

### Aggregate Feature Effects

| Feature | Runs Selected | Median Remove-MAD | Median (Q4-Q1) OOF Confidence | Median (Q4-Q1) OOF Accuracy | Median (Q4-Q1) Shift |
| --- | --- | --- | --- | --- | --- |
| `log_anchor_mass` | 7 | 0.119 | 0.663 | 0.689 | -0.274 |
| `log_one_minus_largest_alt_mass` | 3 | 0.044 | 0.570 | 0.700 | -0.338 |
| `log_valid_rate` | 4 | 0.216 | 0.364 | 0.416 | 0.086 |
| `neg_cluster_entropy` | 13 | 0.159 | 0.502 | 0.504 | -0.184 |

### Per-Run Feature Effects

| Dataset | Model | Feature | Coef | Remove-MAD | Q1 Conf | Q4 Conf | Q1 Acc | Q4 Acc | Q1 Shift | Q4 Shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| physreason | `claude-opus-4-6` | `log_valid_rate` | 1.319 | 0.110 | 0.765 | 0.823 | 0.765 | 0.831 | -0.026 | 0.002 |
| physreason | `claude-opus-4-6` | `neg_cluster_entropy` | 0.650 | 0.074 | 0.658 | 0.798 | 0.640 | 0.818 | 0.313 | -0.150 |
| physreason | `gemini-2.5-pro` | `log_valid_rate` | 1.986 | 0.284 | 0.180 | 0.816 | 0.140 | 0.875 | -0.158 | -0.013 |
| physreason | `gemini-2.5-pro` | `log_one_minus_largest_alt_mass` | 0.432 | 0.020 | 0.697 | 0.614 | 0.580 | 0.620 | 0.299 | -0.094 |
| physreason | `gemini-2.5-pro` | `neg_cluster_entropy` | 0.628 | 0.068 | 0.666 | 0.614 | 0.640 | 0.609 | 0.259 | -0.073 |
| physreason | `gpt-4.1` | `log_anchor_mass` | 0.895 | 0.093 | 0.134 | 0.797 | 0.120 | 0.809 | 0.071 | -0.203 |
| physreason | `gpt-4.1` | `neg_cluster_entropy` | 1.039 | 0.163 | 0.149 | 0.780 | 0.120 | 0.786 | -0.008 | -0.192 |
| physreason | `gpt-5.2` | `log_anchor_mass` | 1.126 | 0.122 | 0.143 | 0.853 | 0.060 | 0.874 | 0.098 | -0.147 |
| physreason | `gpt-5.2` | `log_one_minus_largest_alt_mass` | 1.211 | 0.044 | 0.283 | 0.853 | 0.140 | 0.874 | 0.115 | -0.147 |
| physreason | `gpt-5.2` | `neg_cluster_entropy` | 0.774 | 0.094 | 0.307 | 0.809 | 0.220 | 0.826 | 0.089 | -0.136 |
| physreason | `gpt-5.4` | `log_anchor_mass` | 0.822 | 0.062 | 0.393 | 0.868 | 0.340 | 0.913 | 0.158 | -0.132 |
| physreason | `gpt-5.4` | `neg_cluster_entropy` | 0.936 | 0.126 | 0.375 | 0.852 | 0.400 | 0.880 | -0.008 | -0.102 |
| phyx | `claude-opus-4-6` | `log_valid_rate` | 1.585 | 0.148 | 0.788 | 0.880 | 0.790 | 0.888 | -0.062 | -0.066 |
| phyx | `gemini-2.5-pro` | `log_valid_rate` | 2.104 | 0.382 | 0.094 | 0.889 | 0.071 | 0.971 | -0.731 | -0.103 |
| phyx | `gpt-4.1` | `log_anchor_mass` | 0.621 | 0.055 | 0.293 | 0.838 | 0.280 | 0.895 | -0.017 | -0.159 |
| phyx | `gpt-4.1` | `neg_cluster_entropy` | 1.021 | 0.163 | 0.299 | 0.838 | 0.360 | 0.864 | -0.186 | -0.126 |
| phyx | `gpt-5.2` | `log_one_minus_largest_alt_mass` | 1.292 | 0.161 | 0.243 | 0.826 | 0.160 | 0.860 | 0.188 | -0.151 |
| phyx | `gpt-5.2` | `neg_cluster_entropy` | 1.096 | 0.183 | 0.259 | 0.662 | 0.320 | 0.679 | -0.076 | -0.088 |
| phyx | `gpt-5.4` | `neg_cluster_entropy` | 0.951 | 0.158 | 0.421 | 0.842 | 0.400 | 0.841 | 0.061 | -0.112 |
| seephys | `claude-opus-4-6` | `log_anchor_mass` | 1.708 | 0.252 | 0.091 | 0.801 | 0.032 | 0.917 | 0.091 | -0.199 |
| seephys | `claude-opus-4-6` | `neg_cluster_entropy` | 0.781 | 0.079 | 0.165 | 0.718 | 0.120 | 0.815 | 0.095 | -0.171 |
| seephys | `gemini-2.5-pro` | `log_anchor_mass` | 1.293 | 0.202 | 0.304 | 0.868 | 0.240 | 0.883 | 0.294 | -0.009 |
| seephys | `gpt-4.1` | `log_anchor_mass` | 1.261 | 0.119 | 0.106 | 0.879 | 0.080 | 0.912 | 0.071 | -0.121 |
| seephys | `gpt-4.1` | `neg_cluster_entropy` | 1.237 | 0.163 | 0.111 | 0.861 | 0.040 | 0.886 | 0.001 | -0.110 |
| seephys | `gpt-5.2` | `neg_cluster_entropy` | 1.115 | 0.177 | 0.345 | 0.883 | 0.360 | 0.911 | 0.160 | -0.117 |
| seephys | `gpt-5.4` | `neg_cluster_entropy` | 1.033 | 0.159 | 0.354 | 0.831 | 0.360 | 0.853 | 0.044 | -0.169 |
| seephys | `ollama-qwen3.5-397b-cloud` | `neg_cluster_entropy` | 1.515 | 0.222 | 0.228 | 0.856 | 0.160 | 0.866 | -0.047 | -0.144 |

## Representative Case Studies

These cases use the saved `full_fit` to expose feature contributions (`eta = intercept + sum_j contribution_j`) while also reporting the actual out-of-fold confidence from `ece_per_sample.csv`. The `full_fit` is inspection-only, so the point is mechanistic interpretation rather than exact reconstruction of the evaluation confidence.

### Anchor + Entropy archetype: `physreason_gpt-4.1`

- full-fit selected subset: `{log_anchor_mass, neg_cluster_entropy}`
- coefficients: `log_anchor_mass=0.895; neg_cluster_entropy=1.039`
- fold subsets: `{log_anchor_mass, neg_cluster_entropy} x5`
- mean raw confidence = `0.603`, mean calibrated confidence = `0.505`, accuracy = `0.505`

| Feature | Coef | Remove-MAD | Q1 Conf | Q4 Conf | Q1 Acc | Q4 Acc |
| --- | --- | --- | --- | --- | --- | --- |
| `log_anchor_mass` | 0.895 | 0.093 | 0.134 | 0.797 | 0.120 | 0.809 |
| `neg_cluster_entropy` | 1.039 | 0.163 | 0.149 | 0.780 | 0.120 | 0.786 |

| Case | Problem ID | Raw Conf | OOF Conf | Correctness | Eta (Full-Fit) | P (Full-Fit) | Selected Feature Values | Feature Contributions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_eta | `cal_problem_00035_1` | 1.000 | 0.808 | 1.0 | 1.369 | 0.797 | `log_anchor_mass=0.000; neg_cluster_entropy=0.000` | `log_anchor_mass=0.460; neg_cluster_entropy=1.087` |
| min_eta | `cal_problem_01267_1` | 0.000 | 0.015 | 0.0 | -4.487 | 0.011 | `log_anchor_mass=-13.816; neg_cluster_entropy=-2.079` | `log_anchor_mass=-2.197; neg_cluster_entropy=-2.112` |

### Entropy-only archetype: `seephys_gpt-5.4`

- full-fit selected subset: `{neg_cluster_entropy}`
- coefficients: `neg_cluster_entropy=1.033`
- fold subsets: `{neg_cluster_entropy} x5`
- mean raw confidence = `0.811`, mean calibrated confidence = `0.705`, accuracy = `0.710`

| Feature | Coef | Remove-MAD | Q1 Conf | Q4 Conf | Q1 Acc | Q4 Acc |
| --- | --- | --- | --- | --- | --- | --- |
| `neg_cluster_entropy` | 1.033 | 0.159 | 0.354 | 0.831 | 0.360 | 0.853 |

| Case | Problem ID | Raw Conf | OOF Conf | Correctness | Eta (Full-Fit) | P (Full-Fit) | Selected Feature Values | Feature Contributions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_eta | `1148` | 1.000 | 0.817 | 0.0 | 1.602 | 0.832 | `neg_cluster_entropy=0.000` | `neg_cluster_entropy=0.597` |
| min_eta | `340` | 0.000 | 0.086 | 0.0 | -1.772 | 0.145 | `neg_cluster_entropy=-2.078` | `neg_cluster_entropy=-2.778` |

### Single-feature validity archetype: `phyx_gemini-2.5-pro`

- full-fit selected subset: `{log_valid_rate}`
- coefficients: `log_valid_rate=2.104`
- fold subsets: `{log_valid_rate} x5`
- mean raw confidence = `0.828`, mean calibrated confidence = `0.549`, accuracy = `0.550`

| Feature | Coef | Remove-MAD | Q1 Conf | Q4 Conf | Q1 Acc | Q4 Acc |
| --- | --- | --- | --- | --- | --- | --- |
| `log_valid_rate` | 2.104 | 0.382 | 0.094 | 0.889 | 0.071 | 0.971 |

| Case | Problem ID | Raw Conf | OOF Conf | Correctness | Eta (Full-Fit) | P (Full-Fit) | Selected Feature Values | Feature Contributions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_eta | `128` | 1.000 | 0.906 | 1.0 | 2.080 | 0.889 | `log_valid_rate=0.000` | `log_valid_rate=1.869` |
| min_eta | `101` | 1.000 | 0.104 | 0.0 | -2.257 | 0.095 | `log_valid_rate=-13.816` | `log_valid_rate=-2.469` |

### Three-feature archetype with rival suppression: `physreason_gemini-2.5-pro`

- full-fit selected subset: `{log_valid_rate, log_one_minus_largest_alt_mass, neg_cluster_entropy}`
- coefficients: `log_valid_rate=1.986; log_one_minus_largest_alt_mass=0.432; neg_cluster_entropy=0.628`
- fold subsets: `{log_valid_rate, log_one_minus_largest_alt_mass, neg_cluster_entropy} x3; {log_valid_rate, neg_cluster_entropy} x2`
- mean raw confidence = `0.625`, mean calibrated confidence = `0.646`, accuracy = `0.640`

| Feature | Coef | Remove-MAD | Q1 Conf | Q4 Conf | Q1 Acc | Q4 Acc |
| --- | --- | --- | --- | --- | --- | --- |
| `log_valid_rate` | 1.986 | 0.284 | 0.180 | 0.816 | 0.140 | 0.875 |
| `log_one_minus_largest_alt_mass` | 0.432 | 0.020 | 0.697 | 0.614 | 0.580 | 0.620 |
| `neg_cluster_entropy` | 0.628 | 0.068 | 0.666 | 0.614 | 0.640 | 0.609 |

| Case | Problem ID | Raw Conf | OOF Conf | Correctness | Eta (Full-Fit) | P (Full-Fit) | Selected Feature Values | Feature Contributions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_eta | `cal_problem_00035_1` | 1.000 | 0.888 | 1.0 | 2.162 | 0.897 | `log_valid_rate=0.000; log_one_minus_largest_alt_mass=0.000; neg_cluster_entropy=0.000` | `log_valid_rate=1.092; log_one_minus_largest_alt_mass=0.087; neg_cluster_entropy=0.364` |
| min_eta | `cal_problem_00049_1` | 0.000 | 0.036 | 1.0 | -2.773 | 0.059 | `log_valid_rate=-13.816; log_one_minus_largest_alt_mass=0.000; neg_cluster_entropy=-0.000` | `log_valid_rate=-3.843; log_one_minus_largest_alt_mass=0.087; neg_cluster_entropy=0.364` |

## Interpretation

- The saved artifacts support the practical claim that SCF-Cal behaves like an adaptive sparse family, not like a fixed full-feature logistic rule with occasional shrinkage.
- The artifact set does **not** support the stronger claim that one universal subset works for every model and every dataset. Instead, the stable story is hierarchical: `neg_cluster_entropy` is the most reliable core feature, `log_anchor_mass` is the most common companion, and the other two features turn on when a dataset/model combination makes them useful.
- The output-level analyses show that feature selection matters operationally: the chosen features can move probabilities by large amounts (`Remove-MAD` frequently above `0.1`, and much higher for some single-feature runs), and the actual out-of-fold confidences differ sharply between low-feature and high-feature slices of the data.
- The saved `full_fit` is only an inspection fit, so any per-feature contribution analysis should be read as a mechanistic approximation of the learned geometry. The exact evaluation predictions remain the out-of-fold confidences stored in `ece_per_sample.csv`.
