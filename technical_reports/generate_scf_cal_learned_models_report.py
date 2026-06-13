#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "uncertainty_quantification_physical_reasoning" / "uq" / "uq_bundle" / "runs"
OUTPUT_REPORT = REPO_ROOT / "technical_reports" / "scf_cal_learned_models_report_20260415.md"
RUN_GLOB = "**/posthoc_cluster_feature_sparse_logistic_crossfit_5_seed_0/*/*/posthoc_calibration.json"
EPS = 1e-6

REPRESENTATIVE_CASES: list[tuple[str, str, str]] = [
    ("physreason", "gpt-4.1", "Anchor + Entropy archetype"),
    ("seephys", "gpt-5.4", "Entropy-only archetype"),
    ("phyx", "gemini-2.5-pro", "Single-feature validity archetype"),
    ("physreason", "gemini-2.5-pro", "Three-feature archetype with rival suppression"),
]


@dataclass
class ExampleRow:
    label: str
    problem_id: str
    raw_confidence: float
    calibrated_confidence: float
    correctness: float
    eta_full_fit: float
    p_full_fit: float
    contribution_text: str
    feature_text: str


@dataclass
class FeatureEffect:
    feature: str
    coefficient: float
    mean_abs_prob_delta_if_removed: float
    q1_mean_confidence: float
    q4_mean_confidence: float
    q1_mean_accuracy: float
    q4_mean_accuracy: float
    q1_mean_shift: float
    q4_mean_shift: float


@dataclass
class RunAnalysis:
    dataset: str
    model: str
    source_path: Path
    results_dir: Path
    subset: tuple[str, ...]
    coefficient_by_feature: dict[str, float]
    selected_subset_bic: float | None
    selected_subset_log_likelihood: float | None
    n_candidate_subsets: int | None
    fold_subsets: list[tuple[str, ...]]
    fold_subset_counts: Counter[tuple[str, ...]]
    ece: float | None
    nll: float | None
    brier: float | None
    auroc: float | None
    mean_raw_confidence: float
    mean_calibrated_confidence: float
    overall_accuracy: float
    mean_abs_shift: float
    max_positive_shift: float
    max_negative_shift: float
    upshift_fraction: float
    downshift_fraction: float
    feature_effects: list[FeatureEffect]
    inspection_examples: list[ExampleRow]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def _md_code(text: str) -> str:
    return f"`{text}`"


def _set_str(items: Iterable[str]) -> str:
    items_list = list(items)
    if not items_list:
        return "{}"
    return "{" + ", ".join(items_list) + "}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _parse_dataset_model(model_dir: str) -> tuple[str, str]:
    dataset, model = model_dir.split("_", 1)
    return dataset, model


def _feature_arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    valid_rate = np.array([float(row["confidence_valid_rate"]) for row in rows], dtype=float)
    anchor_mass = np.array([float(row["confidence_anchor_mass"]) for row in rows], dtype=float)
    largest_alt_mass = np.array(
        [float(row["confidence_largest_alt_mass"]) for row in rows],
        dtype=float,
    )
    cluster_entropy = np.array([float(row["confidence_cluster_entropy"]) for row in rows], dtype=float)
    return {
        "log_valid_rate": np.log(valid_rate + EPS),
        "log_anchor_mass": np.log(anchor_mass + EPS),
        "log_one_minus_largest_alt_mass": np.log((1.0 - largest_alt_mass) + EPS),
        "neg_cluster_entropy": -cluster_entropy,
    }


def _counter_to_compact_strings(counter: Counter[tuple[str, ...]]) -> list[str]:
    items = sorted(
        counter.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return [f"{_set_str(subset)} x{count}" for subset, count in items]


def _median_or_none(values: Iterable[float]) -> float | None:
    values_list = list(values)
    if not values_list:
        return None
    return float(median(values_list))


def _mean_or_none(values: Iterable[float]) -> float | None:
    values_list = list(values)
    if not values_list:
        return None
    return float(sum(values_list) / len(values_list))


def _build_run_analysis(path: Path) -> RunAnalysis:
    calibration = _load_json(path)
    results_dir = path.parent
    metrics = _load_json(results_dir / "uq_metrics_full.json")
    with open(results_dir / "ece_per_sample.csv", "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    dataset, model = _parse_dataset_model(results_dir.name)
    full_fit = dict(calibration.get("full_fit", {}))
    subset = tuple(str(x) for x in full_fit.get("selected_feature_names", []))
    coefficient_by_feature = {
        str(k): float(v) for k, v in dict(full_fit.get("coefficient_by_feature", {})).items()
    }
    fold_subsets = [
        tuple(str(x) for x in fold.get("selected_feature_names", []))
        for fold in calibration.get("folds", [])
    ]
    fold_subset_counts: Counter[tuple[str, ...]] = Counter(fold_subsets)

    calibrated_confidence = np.array([float(row["confidence"]) for row in rows], dtype=float)
    raw_confidence = np.array([float(row["raw_confidence"]) for row in rows], dtype=float)
    correctness = np.array([float(row["correctness"]) for row in rows], dtype=float)
    delta_confidence = calibrated_confidence - raw_confidence

    feature_arrays = _feature_arrays(rows)
    feature_names = [str(x) for x in full_fit.get("feature_names", [])]
    coefficients = np.array(full_fit.get("coefficients", []), dtype=float)
    means = np.array(full_fit.get("feature_means", []), dtype=float)
    scales = np.array(full_fit.get("feature_scales", []), dtype=float)
    intercept = float(full_fit.get("intercept", 0.0))

    feature_effects: list[FeatureEffect] = []
    examples: list[ExampleRow] = []
    if feature_names and coefficients.size:
        feature_matrix = np.column_stack([feature_arrays[name] for name in feature_names])
        standardized = (feature_matrix - means) / scales
        contributions = standardized * coefficients
        eta = intercept + np.sum(contributions, axis=1)
        p_full_fit = _sigmoid(eta)

        for feature_idx, feature_name in enumerate(feature_names):
            coef = float(coefficients[feature_idx])
            if abs(coef) <= 1e-12:
                continue
            feature_values = feature_arrays[feature_name]
            q1 = float(np.quantile(feature_values, 0.25))
            q4 = float(np.quantile(feature_values, 0.75))
            q1_mask = feature_values <= q1
            q4_mask = feature_values >= q4
            p_without = _sigmoid(eta - contributions[:, feature_idx])
            feature_effects.append(
                FeatureEffect(
                    feature=feature_name,
                    coefficient=coef,
                    mean_abs_prob_delta_if_removed=float(np.mean(np.abs(p_full_fit - p_without))),
                    q1_mean_confidence=float(np.mean(calibrated_confidence[q1_mask])),
                    q4_mean_confidence=float(np.mean(calibrated_confidence[q4_mask])),
                    q1_mean_accuracy=float(np.mean(correctness[q1_mask])),
                    q4_mean_accuracy=float(np.mean(correctness[q4_mask])),
                    q1_mean_shift=float(np.mean(delta_confidence[q1_mask])),
                    q4_mean_shift=float(np.mean(delta_confidence[q4_mask])),
                )
            )

        for label, idx in (
            ("max_eta", int(np.argmax(eta))),
            ("min_eta", int(np.argmin(eta))),
        ):
            row = rows[idx]
            contribution_chunks: list[str] = []
            feature_chunks: list[str] = []
            for feature_idx, feature_name in enumerate(feature_names):
                coef = float(coefficients[feature_idx])
                if abs(coef) <= 1e-12:
                    continue
                contribution_chunks.append(
                    f"{feature_name}={contributions[idx, feature_idx]:.3f}"
                )
                feature_chunks.append(
                    f"{feature_name}={feature_arrays[feature_name][idx]:.3f}"
                )
            examples.append(
                ExampleRow(
                    label=label,
                    problem_id=str(row["problem_id"]),
                    raw_confidence=float(row["raw_confidence"]),
                    calibrated_confidence=float(row["confidence"]),
                    correctness=float(row["correctness"]),
                    eta_full_fit=float(eta[idx]),
                    p_full_fit=float(p_full_fit[idx]),
                    contribution_text="; ".join(contribution_chunks),
                    feature_text="; ".join(feature_chunks),
                )
            )

    return RunAnalysis(
        dataset=dataset,
        model=model,
        source_path=path,
        results_dir=results_dir,
        subset=subset,
        coefficient_by_feature=coefficient_by_feature,
        selected_subset_bic=(
            float(full_fit["selected_subset_bic"])
            if full_fit.get("selected_subset_bic") is not None
            else None
        ),
        selected_subset_log_likelihood=(
            float(full_fit["selected_subset_log_likelihood"])
            if full_fit.get("selected_subset_log_likelihood") is not None
            else None
        ),
        n_candidate_subsets=(
            int(full_fit["n_candidate_subsets"])
            if full_fit.get("n_candidate_subsets") is not None
            else None
        ),
        fold_subsets=fold_subsets,
        fold_subset_counts=fold_subset_counts,
        ece=float(metrics["ece"]["ece"]) if metrics.get("ece") else None,
        nll=float(metrics["nll"]["nll"]) if metrics.get("nll") else None,
        brier=float(metrics["brier"]["score"]) if metrics.get("brier") else None,
        auroc=float(metrics["discrimination"]["auroc"]) if metrics.get("discrimination") else None,
        mean_raw_confidence=float(np.mean(raw_confidence)),
        mean_calibrated_confidence=float(np.mean(calibrated_confidence)),
        overall_accuracy=float(np.mean(correctness)),
        mean_abs_shift=float(np.mean(np.abs(delta_confidence))),
        max_positive_shift=float(np.max(delta_confidence)),
        max_negative_shift=float(np.min(delta_confidence)),
        upshift_fraction=float(np.mean(delta_confidence > 1e-12)),
        downshift_fraction=float(np.mean(delta_confidence < -1e-12)),
        feature_effects=feature_effects,
        inspection_examples=examples,
    )


def _coeff_summary(run: RunAnalysis) -> str:
    if not run.subset:
        return ""
    return "; ".join(
        f"{feature}={run.coefficient_by_feature[feature]:.3f}"
        for feature in run.subset
    )


def _run_summary_table(runs: list[RunAnalysis]) -> str:
    headers = [
        "Dataset",
        "Model",
        "Full-Fit Subset",
        "Coefficients",
        "BIC",
        "Distinct Fold Subsets",
        "Fold-Consistent",
        "ECE",
        "NLL",
        "Mean Abs Shift",
    ]
    rows: list[list[str]] = []
    for run in sorted(runs, key=lambda r: (r.dataset, r.model)):
        rows.append(
            [
                run.dataset,
                _md_code(run.model),
                _md_code(_set_str(run.subset)),
                _md_code(_coeff_summary(run)),
                _fmt_float(run.selected_subset_bic, 3),
                str(len(set(run.fold_subsets))),
                _fmt_bool(len(set(run.fold_subsets)) == 1),
                _fmt_float(run.ece, 3),
                _fmt_float(run.nll, 3),
                _fmt_float(run.mean_abs_shift, 3),
            ]
        )
    return _table(headers, rows)


def _global_summary_section(runs: list[RunAnalysis]) -> str:
    subset_counter = Counter(run.subset for run in runs)
    feature_counter = Counter(feature for run in runs for feature in run.subset)
    full_fit_size_counter = Counter(len(run.subset) for run in runs)
    fold_size_counter = Counter(len(subset) for run in runs for subset in run.fold_subsets)
    distinct_fold_subset_counter = Counter(len(set(run.fold_subsets)) for run in runs)

    subset_rows = [
        [_md_code(_set_str(subset)), str(count)]
        for subset, count in subset_counter.most_common()
    ]
    feature_rows = [
        [_md_code(feature), str(count), f"{count}/{len(runs)}"]
        for feature, count in feature_counter.most_common()
    ]
    full_size_rows = [[str(k), str(v)] for k, v in sorted(full_fit_size_counter.items())]
    fold_size_rows = [[str(k), str(v)] for k, v in sorted(fold_size_counter.items())]
    distinct_rows = [[str(k), str(v)] for k, v in sorted(distinct_fold_subset_counter.items())]

    all_four_full = sum(1 for run in runs if len(run.subset) == 4)
    all_four_folds = sum(1 for run in runs for subset in run.fold_subsets if len(subset) == 4)

    lines = [
        "## Cross-Run Selection Patterns",
        "",
        f"- analyzed base `SCF-Cal` runs: `{len(runs)}`",
        f"- runs with 4-feature `full_fit`: `{all_four_full}`",
        f"- fold-level fits with 4-feature selection: `{all_four_folds}`",
        "",
        "### Full-Fit Subset Frequency",
        "",
        _table(["Subset", "Runs"], subset_rows),
        "",
        "### Full-Fit Feature Usage",
        "",
        _table(["Feature", "Runs Selected", "Coverage"], feature_rows),
        "",
        "### Feature Count Distribution",
        "",
        "_Full-fit subset size_",
        "",
        _table(["Selected Features", "Runs"], full_size_rows),
        "",
        "_Fold-level subset size_",
        "",
        _table(["Selected Features", "Fold Fits"], fold_size_rows),
        "",
        "### Fold-Level Selection Variability",
        "",
        _table(
            ["Distinct Fold Subsets Within One Run", "Runs"],
            distinct_rows,
        ),
        "",
    ]
    return "\n".join(lines)


def _model_stability_section(runs: list[RunAnalysis]) -> str:
    by_model: dict[str, list[RunAnalysis]] = defaultdict(list)
    for run in runs:
        by_model[run.model].append(run)

    lines = ["## Cross-Dataset Stability by Model", ""]
    for model, model_runs in sorted(by_model.items()):
        if len(model_runs) < 2:
            continue
        model_runs = sorted(model_runs, key=lambda r: r.dataset)
        subsets = [set(run.subset) for run in model_runs]
        universal = set.intersection(*subsets) if subsets else set()
        union = set.union(*subsets) if subsets else set()
        pairwise_jaccards: list[float] = []
        for idx in range(len(model_runs)):
            for jdx in range(idx + 1, len(model_runs)):
                a = set(model_runs[idx].subset)
                b = set(model_runs[jdx].subset)
                pairwise_jaccards.append(len(a & b) / len(a | b))
        rows = []
        for run in model_runs:
            rows.append(
                [
                    run.dataset,
                    _md_code(_set_str(run.subset)),
                    _md_code("; ".join(_counter_to_compact_strings(run.fold_subset_counts))),
                    str(len(run.subset)),
                    _fmt_bool(len(set(run.fold_subsets)) == 1),
                    _fmt_float(run.mean_abs_shift, 3),
                    _fmt_float(run.ece, 3),
                ]
            )
        lines.extend(
            [
                f"### `{model}`",
                "",
                _table(
                    [
                        "Dataset",
                        "Full-Fit Subset",
                        "Fold Subsets",
                        "# Features",
                        "Fold-Consistent",
                        "Mean Abs Shift",
                        "ECE",
                    ],
                    rows,
                ),
                "",
                f"- universal features across datasets: `{_set_str(sorted(universal))}`",
                f"- union of selected features across datasets: `{_set_str(sorted(union))}`",
                f"- pairwise Jaccard similarity of full-fit subsets: `{', '.join(_fmt_float(v, 3) for v in pairwise_jaccards)}`",
                "",
            ]
        )
    return "\n".join(lines)


def _feature_effect_section(runs: list[RunAnalysis]) -> str:
    per_feature_rows: list[list[str]] = []
    per_feature_aggregate: dict[str, list[FeatureEffect]] = defaultdict(list)
    for run in sorted(runs, key=lambda r: (r.dataset, r.model)):
        for effect in run.feature_effects:
            per_feature_aggregate[effect.feature].append(effect)
            per_feature_rows.append(
                [
                    run.dataset,
                    _md_code(run.model),
                    _md_code(effect.feature),
                    _fmt_float(effect.coefficient, 3),
                    _fmt_float(effect.mean_abs_prob_delta_if_removed, 3),
                    _fmt_float(effect.q1_mean_confidence, 3),
                    _fmt_float(effect.q4_mean_confidence, 3),
                    _fmt_float(effect.q1_mean_accuracy, 3),
                    _fmt_float(effect.q4_mean_accuracy, 3),
                    _fmt_float(effect.q1_mean_shift, 3),
                    _fmt_float(effect.q4_mean_shift, 3),
                ]
            )

    aggregate_rows: list[list[str]] = []
    for feature, effects in sorted(per_feature_aggregate.items()):
        aggregate_rows.append(
            [
                _md_code(feature),
                str(len(effects)),
                _fmt_float(_median_or_none(e.mean_abs_prob_delta_if_removed for e in effects), 3),
                _fmt_float(_median_or_none(e.q4_mean_confidence - e.q1_mean_confidence for e in effects), 3),
                _fmt_float(_median_or_none(e.q4_mean_accuracy - e.q1_mean_accuracy for e in effects), 3),
                _fmt_float(_median_or_none(e.q4_mean_shift - e.q1_mean_shift for e in effects), 3),
            ]
        )

    lines = [
        "## Feature-Effect Analysis on Learned Outputs",
        "",
        "This section combines two views of feature importance:",
        "",
        "- `mean_abs_prob_delta_if_removed` is computed from the saved `full_fit` only. It measures how much the inspection model's probability would move, on average, if one selected feature were removed while keeping the other learned terms fixed.",
        "- `q1/q4` confidence, accuracy, and shift statistics are computed from the actual out-of-fold `ece_per_sample.csv` outputs. They show how the realized calibrated outputs differ between low-feature and high-feature slices of the run.",
        "- Negative `Q4-Q1` values can appear even for selected features. That does not contradict non-negative coefficients: these slice statistics are computed on the realized out-of-fold outputs, where multiple selected features interact and the exact fold-specific fit may differ slightly from the saved `full_fit`.",
        "",
        "### Aggregate Feature Effects",
        "",
        _table(
            [
                "Feature",
                "Runs Selected",
                "Median Remove-MAD",
                "Median (Q4-Q1) OOF Confidence",
                "Median (Q4-Q1) OOF Accuracy",
                "Median (Q4-Q1) Shift",
            ],
            aggregate_rows,
        ),
        "",
        "### Per-Run Feature Effects",
        "",
        _table(
            [
                "Dataset",
                "Model",
                "Feature",
                "Coef",
                "Remove-MAD",
                "Q1 Conf",
                "Q4 Conf",
                "Q1 Acc",
                "Q4 Acc",
                "Q1 Shift",
                "Q4 Shift",
            ],
            per_feature_rows,
        ),
        "",
    ]
    return "\n".join(lines)


def _case_studies_section(runs: list[RunAnalysis]) -> str:
    run_by_key = {(run.dataset, run.model): run for run in runs}
    lines = [
        "## Representative Case Studies",
        "",
        "These cases use the saved `full_fit` to expose feature contributions (`eta = intercept + sum_j contribution_j`) while also reporting the actual out-of-fold confidence from `ece_per_sample.csv`. The `full_fit` is inspection-only, so the point is mechanistic interpretation rather than exact reconstruction of the evaluation confidence.",
        "",
    ]
    for dataset, model, title in REPRESENTATIVE_CASES:
        run = run_by_key.get((dataset, model))
        if run is None:
            continue
        lines.extend(
            [
                f"### {title}: `{dataset}_{model}`",
                "",
                f"- full-fit selected subset: `{_set_str(run.subset)}`",
                f"- coefficients: `{_coeff_summary(run)}`",
                f"- fold subsets: `{'; '.join(_counter_to_compact_strings(run.fold_subset_counts))}`",
                f"- mean raw confidence = `{_fmt_float(run.mean_raw_confidence, 3)}`, mean calibrated confidence = `{_fmt_float(run.mean_calibrated_confidence, 3)}`, accuracy = `{_fmt_float(run.overall_accuracy, 3)}`",
                "",
            ]
        )
        feature_rows = []
        for effect in run.feature_effects:
            feature_rows.append(
                [
                    _md_code(effect.feature),
                    _fmt_float(effect.coefficient, 3),
                    _fmt_float(effect.mean_abs_prob_delta_if_removed, 3),
                    _fmt_float(effect.q1_mean_confidence, 3),
                    _fmt_float(effect.q4_mean_confidence, 3),
                    _fmt_float(effect.q1_mean_accuracy, 3),
                    _fmt_float(effect.q4_mean_accuracy, 3),
                ]
            )
        lines.extend(
            [
                _table(
                    [
                        "Feature",
                        "Coef",
                        "Remove-MAD",
                        "Q1 Conf",
                        "Q4 Conf",
                        "Q1 Acc",
                        "Q4 Acc",
                    ],
                    feature_rows,
                ),
                "",
            ]
        )
        example_rows = []
        for example in run.inspection_examples:
            example_rows.append(
                [
                    example.label,
                    _md_code(example.problem_id),
                    _fmt_float(example.raw_confidence, 3),
                    _fmt_float(example.calibrated_confidence, 3),
                    _fmt_float(example.correctness, 1),
                    _fmt_float(example.eta_full_fit, 3),
                    _fmt_float(example.p_full_fit, 3),
                    _md_code(example.feature_text),
                    _md_code(example.contribution_text),
                ]
            )
        lines.extend(
            [
                _table(
                    [
                        "Case",
                        "Problem ID",
                        "Raw Conf",
                        "OOF Conf",
                        "Correctness",
                        "Eta (Full-Fit)",
                        "P (Full-Fit)",
                        "Selected Feature Values",
                        "Feature Contributions",
                    ],
                    example_rows,
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _inventory_section(runs: list[RunAnalysis]) -> str:
    datasets = sorted({run.dataset for run in runs})
    models = sorted({run.model for run in runs})
    lines = [
        "## Scope and Artifact Inventory",
        "",
        "This report analyzes the learned model structure of the base `SCF-Cal` method",
        "(`cluster_feature_sparse_logistic`) from saved post-hoc calibration artifacts only.",
        "",
        "Included artifacts are runs matching:",
        "",
        f"- `{RUN_GLOB}`",
        "",
        "Excluded artifacts:",
        "",
        "- leave-one-feature-out ablation variants such as `cluster_feature_sparse_logistic_drop_*`",
        "- methods from other post-hoc calibration families",
        "",
        f"- total analyzed runs: `{len(runs)}`",
        f"- datasets covered: `{', '.join(datasets)}`",
        f"- model names covered: `{', '.join(models)}`",
        "",
        "Primary per-run source files:",
        "",
        "- `posthoc_calibration.json` for selected subsets, coefficients, BIC, and fold summaries",
        "- `ece_per_sample.csv` for actual out-of-fold calibrated confidence and cluster features",
        "- `uq_metrics_full.json` for ECE / NLL / Brier / AUROC summaries",
        "",
    ]
    return "\n".join(lines)


def _main_findings_section(runs: list[RunAnalysis]) -> str:
    subset_counter = Counter(run.subset for run in runs)
    feature_counter = Counter(feature for run in runs for feature in run.subset)
    full_fit_all4 = sum(1 for run in runs if len(run.subset) == 4)
    fold_all4 = sum(1 for run in runs for subset in run.fold_subsets if len(subset) == 4)
    fold_consistent = sum(1 for run in runs if len(set(run.fold_subsets)) == 1)
    lines = [
        "## Main Findings",
        "",
        f"1. Across `{len(runs)}` saved base `SCF-Cal` runs, the most common `full_fit` subset is `{_set_str(['log_anchor_mass', 'neg_cluster_entropy'])}`; it appears in `{subset_counter[('log_anchor_mass', 'neg_cluster_entropy')]}` runs.",
        f"2. The strongest globally stable feature is `neg_cluster_entropy`, selected in `{feature_counter['neg_cluster_entropy']}/{len(runs)}` `full_fit`s. `log_anchor_mass` is second at `{feature_counter['log_anchor_mass']}/{len(runs)}`.",
        f"3. No analyzed base run selected all four features in `full_fit` (`{full_fit_all4}` cases), and no fold-level fit selected all four features either (`{fold_all4}` cases). In this artifact set, the sparse search never judged the full 4-feature model necessary.",
        f"4. Model-level stability is highly heterogeneous. `gpt-4.1` is perfectly stable across the three datasets and always selects `{_set_str(['log_anchor_mass', 'neg_cluster_entropy'])}`. `gpt-5.4` is stable on an entropy-centered family, while `claude-opus-4-6` and `gemini-2.5-pro` are strongly dataset-adaptive.",
        f"5. Fold-level subset selection is not always identical even when `full_fit` is concise: `{fold_consistent}/{len(runs)}` runs have exactly one fold-level subset, while the others show two or three distinct fold-level subsets. The sparse rule is stable in its low-dimensionality, but not always in the exact same active set on every fold.",
        "",
    ]
    return "\n".join(lines)


def build_report(runs: list[RunAnalysis]) -> str:
    lines = [
        "# SCF-Cal Learned Model Analysis Report",
        "",
        "Generated from saved calibration artifacts on 2026-04-15.",
        "",
        _inventory_section(runs),
        _main_findings_section(runs),
        "## Per-Run Learned Model Summary",
        "",
        _run_summary_table(runs),
        "",
        _global_summary_section(runs),
        _model_stability_section(runs),
        _feature_effect_section(runs),
        _case_studies_section(runs),
        "## Interpretation",
        "",
        "- The saved artifacts support the practical claim that SCF-Cal behaves like an adaptive sparse family, not like a fixed full-feature logistic rule with occasional shrinkage.",
        "- The artifact set does **not** support the stronger claim that one universal subset works for every model and every dataset. Instead, the stable story is hierarchical: `neg_cluster_entropy` is the most reliable core feature, `log_anchor_mass` is the most common companion, and the other two features turn on when a dataset/model combination makes them useful.",
        "- The output-level analyses show that feature selection matters operationally: the chosen features can move probabilities by large amounts (`Remove-MAD` frequently above `0.1`, and much higher for some single-feature runs), and the actual out-of-fold confidences differ sharply between low-feature and high-feature slices of the data.",
        "- The saved `full_fit` is only an inspection fit, so any per-feature contribution analysis should be read as a mechanistic approximation of the learned geometry. The exact evaluation predictions remain the out-of-fold confidences stored in `ece_per_sample.csv`.",
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    run_paths = sorted(RUNS_ROOT.glob(RUN_GLOB))
    runs = [_build_run_analysis(path) for path in run_paths]
    report = build_report(runs)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
