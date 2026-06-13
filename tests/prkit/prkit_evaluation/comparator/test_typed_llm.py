from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_category import AnswerCategory
from prkit.prkit_evaluation.comparator import typed_llm as typed_llm_module
from prkit.prkit_evaluation.comparator.typed_llm import (
    DEFAULT_MODEL,
    TypedLLMComparator,
    _compare_formula_or_equation_as_expressions,
    _compare_physical_quantity_same_unit_pool_placeholder,
    _contains_latex_text_macro,
    _normalized_equation_rhs_string,
    _plain_text_true_else_llm,
    _symbolic_operand_for_expression_compare,
    _typed_category_and_value,
    infer_symbolic_answer_is_expression,
)
from prkit.prkit_evaluation.llm_judge import (
    RESULT_SOURCE_LLM_JUDGE,
    RESULT_SOURCE_SKIPPED_LLM,
    RESULT_SOURCE_TYPED_MATCH,
    parse_judge_response,
)


class _DummyResponses:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_params = None

    def create(self, **kwargs):
        self.last_params = kwargs

        class _Resp:
            output_text = self.response_text

        return _Resp()


class _DummyClient:
    def __init__(self, response_text: str):
        self.responses = _DummyResponses(response_text)


def test_default_model_name():
    dummy = _DummyClient(
        '{"verdict":"correct","confidence":0.9,"expected_answer_type":"numeric_value","reasoning":"ok"}'
    )
    comp = TypedLLMComparator(client=dummy)
    assert comp.model_name == DEFAULT_MODEL


def test_compare_uses_question_in_payload():
    dummy = _DummyClient(
        '{"verdict":"correct","confidence":0.8,"expected_answer_type":"physical_quantity","reasoning":"unit inferred"}'
    )
    comp = TypedLLMComparator(client=dummy)

    result = comp.compare("10", "10 N", question="Find the force in N")
    assert result is True
    payload_text = dummy.responses.last_params["input"][0]["content"][0]["text"]
    assert '"expectations_from_question"' not in payload_text
    assert "Find the force in N" in payload_text
    assert "instructions" in dummy.responses.last_params


def test_quick_option_case_insensitive_shortcut():
    dummy = _DummyClient(
        '{"verdict":"incorrect","confidence":0.1,"expected_answer_type":"multiple_choice","reasoning":"should not be called"}'
    )
    comp = TypedLLMComparator(client=dummy)
    pred = Answer(value="A", answer_category=AnswerCategory.OPTION)
    gt = Answer(value="a", answer_category=AnswerCategory.OPTION)
    assert comp.compare(pred, gt) is True
    assert comp.last_result is not None
    assert comp.last_result.raw_response == "local_shortcut"
    assert comp.last_result.expected_answer_type == "other"
    assert comp.last_result.verdict_type == RESULT_SOURCE_TYPED_MATCH


def test_same_type_mismatch_uses_quick_typed_path_not_llm():
    dummy = _DummyClient(
        '{"verdict":"correct","confidence":0.99,"expected_answer_type":"numeric_value","reasoning":"should not be called"}'
    )
    comp = TypedLLMComparator(client=dummy)
    assert comp.compare("10", "11") is False
    assert comp.last_result is not None
    assert comp.last_result.raw_response == "local_shortcut"
    assert dummy.responses.last_params is None


def test_text_answers_route_to_llm_not_quick_path():
    dummy = _DummyClient(
        '{"verdict":"incorrect","confidence":0.77,"expected_answer_type":"textual_concept","reasoning":"extra items included"}'
    )
    comp = TypedLLMComparator(client=dummy)
    assert (
        comp.compare(
            "fine structure, lamb shift",
            "fine structure, hyperfine structure",
            question="Which corrections apply? Answer in the name of the corrections.",
        )
        is False
    )
    assert comp.last_result is not None
    assert comp.last_result.raw_response != "local_shortcut"
    assert dummy.responses.last_params is not None


def test_text_exact_match_uses_local_plaintext_shortcut():
    dummy = _DummyClient(
        '{"verdict":"incorrect","confidence":0.2,"expected_answer_type":"textual_concept","reasoning":"should not be called"}'
    )
    comp = TypedLLMComparator(client=dummy)
    pred = Answer(value="mechanical energy is conserved", answer_category=AnswerCategory.TEXT)
    gt = Answer(value="mechanical energy is conserved", answer_category=AnswerCategory.TEXT)
    assert comp.compare(pred, gt) is True
    assert comp.last_result is not None
    assert comp.last_result.raw_response == "local_shortcut"
    assert dummy.responses.last_params is None


def test_formula_with_text_macro_routes_to_llm_not_quick_path():
    dummy = _DummyClient(
        '{"verdict":"correct","confidence":0.88,"expected_answer_type":"symbolic_expression","reasoning":"gradient and slope are equivalent labels"}'
    )
    comp = TypedLLMComparator(client=dummy)
    assert comp.compare(r"M = \frac{\text{slope}}{G}", r"M=\frac{\text{gradient}}{G}") is True
    assert comp.last_result is not None
    assert comp.last_result.raw_response != "local_shortcut"
    assert dummy.responses.last_params is not None


def test_formula_compare_false_then_plaintext_true_shortcuts_locally(monkeypatch):
    dummy = _DummyClient(
        '{"verdict":"incorrect","confidence":0.2,"expected_answer_type":"symbolic_expression","reasoning":"should not be called"}'
    )
    comp = TypedLLMComparator(client=dummy)

    def _always_false_formula(*_args, **_kwargs):
        return False

    monkeypatch.setattr(typed_llm_module, "compare_formula", _always_false_formula)

    pred = Answer(value="v = a + b", answer_category=AnswerCategory.FORMULA)
    gt = Answer(value="a + b", answer_category=AnswerCategory.FORMULA)
    assert comp.compare(pred, gt) is True
    assert comp.last_result is not None
    assert comp.last_result.raw_response == "local_shortcut"
    assert dummy.responses.last_params is None


def test_parse_response_with_json_and_fallback():
    parsed = parse_judge_response(
        '{"verdict":"incorrect","confidence":0.2,"expected_answer_type":"direction_or_sign","reasoning":"wrong sign"}'
    )
    assert parsed.verdict == "incorrect"
    assert parsed.confidence == 0.2
    assert parsed.expected_answer_type == "direction_or_sign"
    assert parsed.verdict_type == RESULT_SOURCE_LLM_JUDGE

    fallback = parse_judge_response("This is correct.")
    assert fallback.verdict == "correct"
    assert fallback.expected_answer_type == "other"
    assert fallback.verdict_type == RESULT_SOURCE_LLM_JUDGE


def test_parse_response_contradictory_reasoning_forces_incorrect():
    parsed = parse_judge_response(
        '{"verdict":"correct","confidence":0.99,"expected_answer_type":"symbolic_expression","reasoning":"The model answer is incorrect because it is missing a factor of 2."}'
    )
    assert parsed.verdict == "incorrect"
    assert parsed.confidence <= 0.35


def test_accuracy_score_is_binary_from_verdict():
    dummy = _DummyClient(
        '{"verdict":"correct","confidence":0.73,"expected_answer_type":"symbolic_expression","reasoning":"equivalent"}'
    )
    comp = TypedLLMComparator(client=dummy)
    score = comp.accuracy_score("x=2", "2", question="Find x")
    assert score == 1.0

    dummy_bad = _DummyClient(
        '{"verdict":"incorrect","confidence":0.9,"expected_answer_type":"symbolic_expression","reasoning":"no"}'
    )
    comp_bad = TypedLLMComparator(client=dummy_bad)
    assert comp_bad.accuracy_score("x=2", "3", question="Find x") == 0.0


def test_parse_unknown_expected_type_falls_back_to_other():
    parsed = parse_judge_response(
        '{"verdict":"correct","confidence":0.95,"expected_answer_type":"free_text_blob","reasoning":"ok"}'
    )
    assert parsed.expected_answer_type == "other"


def test_expression_question_formula_vs_equation_matches_without_llm():
    """Ground-truth expression vs model with v=...; question asks for expression in terms of."""
    dummy = _DummyClient(
        '{"verdict":"incorrect","confidence":0.1,"expected_answer_type":"symbolic_expression","reasoning":"should not be called"}'
    )
    comp = TypedLLMComparator(client=dummy)
    gt = r"$\sqrt{C(T_{1}+T_{2}-2 T)}$"
    pred = r"$v = \sqrt{C(T_1 + T_2 - 2T)}$"
    q = (
        "What is the speed of the jet in terms of $T_{1}, T_{2}$ and $T$, where $T$ is the "
        "temperature of water in the jet?"
    )
    assert comp.compare(pred, gt, question=q) is True
    assert comp.last_result is not None
    assert comp.last_result.raw_response == "local_shortcut"
    assert dummy.responses.last_params is None


def test_equation_question_formula_vs_equation_can_shortcut_via_plaintext():
    """For equation-style question, if plaintext fallback is affirmative, keep local shortcut."""
    dummy = _DummyClient(
        '{"verdict":"correct","confidence":0.9,"expected_answer_type":"symbolic_expression","reasoning":"ok"}'
    )
    comp = TypedLLMComparator(client=dummy)
    gt = r"$\sqrt{C(T_{1}+T_{2}-2 T)}$"
    pred = r"$v = \sqrt{C(T_1 + T_2 - 2T)}$"
    q = "Derive the equation relating the jet speed to $T_1$, $T_2$, and $T$."
    assert comp.compare(pred, gt, question=q) is True
    assert comp.last_result is not None
    assert comp.last_result.raw_response == "local_shortcut"
    assert dummy.responses.last_params is None


def test_equation_question_formula_vs_equation_routes_to_llm_when_plaintext_not_affirmative():
    dummy = _DummyClient(
        '{"verdict":"incorrect","confidence":0.9,"expected_answer_type":"symbolic_expression","reasoning":"not equivalent"}'
    )
    comp = TypedLLMComparator(client=dummy)
    gt = r"$\sqrt{C(T_{1}+T_{2}-2 T)}$"
    pred = r"$v = \sqrt{C(T_1 + T_2 + 2T)}$"
    q = "Derive the equation relating the jet speed to $T_1$, $T_2$, and $T$."
    assert comp.compare(pred, gt, question=q) is False
    assert comp.last_result is not None
    assert comp.last_result.raw_response != "local_shortcut"
    assert dummy.responses.last_params is not None


def test_infer_symbolic_answer_is_expression():
    assert infer_symbolic_answer_is_expression("What is the speed in terms of $a$ and $b$?") is True
    assert infer_symbolic_answer_is_expression("Write the equation of motion for the system.") is False
    assert infer_symbolic_answer_is_expression("Solve the problem.") is None
    assert (
        infer_symbolic_answer_is_expression(
            "Determine the equation for $V$ in terms of $t$, where $V$ is in volts."
        )
        is True
    )
    assert (
        infer_symbolic_answer_is_expression(
            "Derive an equation for the distribution of intensity $I(x, y)$ in the plane."
        )
        is False
    )
    assert infer_symbolic_answer_is_expression("图中（I）是 $t=0$ 时的波形图，写出波动方程的表达式。") is True
    assert infer_symbolic_answer_is_expression("图示为两个简谐振动的 $x-t$ 曲线，试分别写出其简谐振动方程。") is False


def test_symbolic_answer_is_expression_kwarg_overrides_question():
    dummy = _DummyClient(
        '{"verdict":"incorrect","confidence":0.1,"expected_answer_type":"symbolic_expression","reasoning":"should not be called"}'
    )
    comp = TypedLLMComparator(client=dummy)
    gt = r"$\sqrt{C(T_{1}+T_{2}-2 T)}$"
    pred = r"$v = \sqrt{C(T_1 + T_2 - 2T)}$"
    q = "Derive the equation relating the jet speed to temperatures."
    assert (
        comp.compare(
            pred,
            gt,
            question=q,
            symbolic_answer_is_expression=True,
        )
        is True
    )
    assert comp.last_result.raw_response == "local_shortcut"
    assert dummy.responses.last_params is None


def test_ambiguous_physical_quantity_units_route_to_llm():
    dummy = _DummyClient(
        '{"verdict":"correct","confidence":0.81,"expected_answer_type":"physical_quantity","reasoning":"equivalent units"}'
    )
    comp = TypedLLMComparator(client=dummy)
    assert comp.compare("22 rad/s", "22 1/rads", question="Angular frequency") is True
    assert comp.last_result is not None
    assert comp.last_result.raw_response != "local_shortcut"
    assert dummy.responses.last_params is not None


def test_skip_llm_returns_dummy_without_api_call():
    dummy = _DummyClient(
        '{"verdict":"correct","confidence":0.9,"expected_answer_type":"textual_concept","reasoning":"should not be called"}'
    )
    comp = TypedLLMComparator(client=dummy)
    assert (
        comp.compare(
            "a",
            "b",
            question="Which corrections apply?",
            skip_llm=True,
        )
        is False
    )
    assert comp.last_result is not None
    assert comp.last_result.verdict_type == RESULT_SOURCE_SKIPPED_LLM
    assert comp.last_result.raw_response == "skipped_llm"
    assert dummy.responses.last_params is None


def test_skip_llm_still_uses_typed_match_shortcut():
    dummy = _DummyClient(
        '{"verdict":"incorrect","confidence":0.1,"expected_answer_type":"multiple_choice","reasoning":"should not be called"}'
    )
    comp = TypedLLMComparator(client=dummy)
    pred = Answer(value="A", answer_category=AnswerCategory.OPTION)
    gt = Answer(value="a", answer_category=AnswerCategory.OPTION)
    assert comp.compare(pred, gt, skip_llm=True) is True
    assert comp.last_result.verdict_type == RESULT_SOURCE_TYPED_MATCH
    assert dummy.responses.last_params is None


def test_accuracy_score_forwards_skip_llm():
    dummy = _DummyClient(
        '{"verdict":"correct","confidence":0.9,"expected_answer_type":"textual_concept","reasoning":"should not be called"}'
    )
    comp = TypedLLMComparator(client=dummy)
    assert (
        comp.accuracy_score(
            "a",
            "b",
            question="Which corrections apply?",
            skip_llm=True,
        )
        == 0.0
    )
    assert comp.last_result is not None
    assert comp.last_result.verdict_type == RESULT_SOURCE_SKIPPED_LLM
    assert dummy.responses.last_params is None


def test_comparator_exception_routes_to_llm(monkeypatch):
    dummy = _DummyClient(
        '{"verdict":"correct","confidence":0.7,"expected_answer_type":"numeric_value","reasoning":"LLM fallback on comparator exception"}'
    )
    comp = TypedLLMComparator(client=dummy)

    def _raise_compare_number(*_args, **_kwargs):
        raise ValueError("forced test error")

    monkeypatch.setattr(typed_llm_module, "compare_number", _raise_compare_number)

    pred = Answer(value="10", answer_category=AnswerCategory.NUMBER)
    gt = Answer(value="10", answer_category=AnswerCategory.NUMBER)
    assert comp.compare(pred, gt, question="Compute the value") is True
    assert comp.last_result is not None
    assert comp.last_result.raw_response != "local_shortcut"
    assert dummy.responses.last_params is not None


def test_helper_functions_cover_fallback_paths(monkeypatch):
    assert _contains_latex_text_macro(r"\text{speed}") is True
    assert _contains_latex_text_macro("plain text") is False
    assert _normalized_equation_rhs_string("Eq(x, 2)") == "2"
    assert _normalized_equation_rhs_string("not an equation") is None
    assert (
        _symbolic_operand_for_expression_compare(AnswerCategory.FORMULA, "a + b")
        == "a + b"
    )
    assert (
        _symbolic_operand_for_expression_compare(AnswerCategory.EQUATION, "Eq(v, t)")
        == "t"
    )
    assert (
        _symbolic_operand_for_expression_compare(AnswerCategory.TEXT, "ignored") is None
    )
    assert _compare_physical_quantity_same_unit_pool_placeholder("1", "m", "100", "cm") == (
        False,
        False,
    )

    monkeypatch.setattr(
        typed_llm_module,
        "normalize_answer",
        lambda _answer: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert _typed_category_and_value(" ?? ") == (None, "??")


def test_compare_formula_or_equation_as_expressions_handles_non_symbolic_cases(monkeypatch):
    assert (
        _compare_formula_or_equation_as_expressions(
            AnswerCategory.TEXT,
            "x",
            AnswerCategory.FORMULA,
            "x",
            "x",
            "x",
        )
        is None
    )
    assert (
        _compare_formula_or_equation_as_expressions(
            AnswerCategory.FORMULA,
            "x",
            AnswerCategory.TEXT,
            "x",
            "x",
            "x",
        )
        is None
    )
    assert (
        _compare_formula_or_equation_as_expressions(
            AnswerCategory.FORMULA,
            "x",
            AnswerCategory.EQUATION,
            "Eq(y, x)",
            r"\text{x}",
            "Eq(y, x)",
        )
        is None
    )

    monkeypatch.setattr(typed_llm_module, "compare_formula", lambda *_args: True)
    assert (
        _compare_formula_or_equation_as_expressions(
            AnswerCategory.FORMULA,
            "x + y",
            AnswerCategory.EQUATION,
            "Eq(z, y + x)",
            "x + y",
            "Eq(z, y + x)",
        )
        is True
    )

    monkeypatch.setattr(
        typed_llm_module,
        "compare_formula",
        lambda *_args: (_ for _ in ()).throw(ValueError("boom")),
    )
    assert (
        _compare_formula_or_equation_as_expressions(
            AnswerCategory.FORMULA,
            "x",
            AnswerCategory.FORMULA,
            "x",
            "x",
            "x",
        )
        is None
    )


def test_plain_text_true_else_llm_returns_none_for_false_or_errors(monkeypatch):
    assert _plain_text_true_else_llm("x", "y") is None

    monkeypatch.setattr(
        typed_llm_module,
        "compare_plain_text",
        lambda *_args: (_ for _ in ()).throw(TypeError("boom")),
    )
    assert _plain_text_true_else_llm("x", "x") is None


def test_quick_typed_match_helper_branches(monkeypatch):
    monkeypatch.setattr(
        typed_llm_module,
        "normalize_answer",
        lambda _answer: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert TypedLLMComparator._quick_typed_match("bad", "still bad") is None

    monkeypatch.setattr(
        typed_llm_module,
        "_compare_physical_quantity_same_unit_pool_placeholder",
        lambda *_args: (True, True),
    )
    assert (
        TypedLLMComparator._quick_typed_match(
            Answer(value="22 rad/s", answer_category=AnswerCategory.PHYSICAL_QUANTITY),
            Answer(value="22 1/rads", answer_category=AnswerCategory.PHYSICAL_QUANTITY),
        )
        is True
    )

    monkeypatch.setattr(
        typed_llm_module,
        "compare_formula",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad formula")),
    )
    assert (
        TypedLLMComparator._quick_typed_match(
            Answer(value="x + y", answer_category=AnswerCategory.FORMULA),
            Answer(value="x + y", answer_category=AnswerCategory.FORMULA),
        )
        is True
    )

    monkeypatch.setattr(typed_llm_module, "compare_plain_text", lambda *_args: False)
    assert (
        TypedLLMComparator._quick_typed_match(
            Answer(value="Eq(x, 1)", answer_category=AnswerCategory.EQUATION),
            Answer(value="Eq(x, 2)", answer_category=AnswerCategory.EQUATION),
        )
        is None
    )
