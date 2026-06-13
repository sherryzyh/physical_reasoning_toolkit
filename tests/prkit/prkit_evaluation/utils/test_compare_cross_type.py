from prkit.prkit_core.domain.answer_category import AnswerCategory
from prkit.prkit_evaluation.utils import cross_typed_match
from prkit.prkit_evaluation.utils import compare_cross_type as compare_cross_type_module


def test_split_respecting_parens_and_expand_gt_set():
    assert compare_cross_type_module.split_respecting_parens("a,(b,c),[d,e]") == [
        "a",
        "(b,c)",
        "[d,e]",
    ]
    assert compare_cross_type_module.expand_gt_set("{ a , Eq(x, y) }") == [
        "a",
        "Eq(x, y)",
    ]
    assert compare_cross_type_module.expand_gt_set("single") == ["single"]


def test_strip_unbalanced_parens_and_extract_formula_candidates():
    assert compare_cross_type_module.strip_unbalanced_parens("((x+y)") == "(x+y)"
    assert compare_cross_type_module.strip_unbalanced_parens("[value]]") == "[value]"
    assert compare_cross_type_module.extract_formula_candidates(", F = (ma ; [p]]") == [
        "F = (ma",
        "(ma",
        "ma",
        "[p]]",
        "[p]",
    ]


def test_compare_text_against_formula_or_equation_gt_matches_symbolic_equation(
    monkeypatch,
):
    monkeypatch.setattr(
        compare_cross_type_module,
        "extract_formula_candidates",
        lambda _text: ["3*t"],
    )
    monkeypatch.setattr(
        compare_cross_type_module,
        "normalize_answer",
        lambda _candidate: (AnswerCategory.FORMULA, "3*t"),
    )
    monkeypatch.setattr(
        compare_cross_type_module,
        "compare_formula",
        lambda candidate_norm, gt_norm: candidate_norm == gt_norm == "3*t",
    )

    assert (
        compare_cross_type_module.compare_text_against_formula_or_equation_gt(
            "The conserved quantity is 3*t.",
            "Eq(v, 3*t)",
        )
        is True
    )


def test_compare_text_against_formula_or_equation_gt_handles_errors_and_quantity_path(
    monkeypatch,
):
    monkeypatch.setattr(
        compare_cross_type_module,
        "extract_formula_candidates",
        lambda _text: ["", "bad", "Eq(x, 2)", "10 m/s", "words"],
    )

    def fake_normalize_answer(candidate: str):
        if candidate == "bad":
            raise ValueError("bad candidate")
        if candidate == "Eq(x, 2)":
            return AnswerCategory.EQUATION, "Eq(x, 2)"
        if candidate == "10 m/s":
            return AnswerCategory.PHYSICAL_QUANTITY, "10 m/s"
        return AnswerCategory.TEXT, candidate

    monkeypatch.setattr(
        compare_cross_type_module,
        "normalize_answer",
        fake_normalize_answer,
    )
    monkeypatch.setattr(
        compare_cross_type_module,
        "extract_rhs_and_category",
        lambda normalized, _category: (normalized.rsplit(",", 1)[-1].strip(" )"), AnswerCategory.PHYSICAL_QUANTITY),
    )

    def fake_compare_formula(candidate_norm: str, gt_norm: str) -> bool:
        if gt_norm == "2":
            raise ValueError("force quantity fallback")
        return False

    monkeypatch.setattr(
        compare_cross_type_module,
        "compare_formula",
        fake_compare_formula,
    )
    monkeypatch.setattr(
        compare_cross_type_module,
        "compare_physical_quantity",
        lambda candidate_norm, gt_norm: candidate_norm == gt_norm == "10 m/s",
    )

    assert (
        compare_cross_type_module.compare_text_against_formula_or_equation_gt(
            "unused",
            "{2, 10 m/s}",
        )
        is True
    )


def test_cross_typed_match_shim_reexports_compare_helpers(monkeypatch):
    monkeypatch.setattr(
        compare_cross_type_module,
        "extract_formula_candidates",
        lambda _text: ["x + y"],
    )
    monkeypatch.setattr(
        compare_cross_type_module,
        "normalize_answer",
        lambda _candidate: (AnswerCategory.FORMULA, "x + y"),
    )
    monkeypatch.setattr(
        compare_cross_type_module,
        "compare_formula",
        lambda candidate_norm, gt_norm: candidate_norm == gt_norm == "x + y",
    )

    assert (
        cross_typed_match.compare_text_against_formula_or_equation_gt(
            "Answer: x + y",
            "Eq(z, x + y)",
        )
        is True
    )
