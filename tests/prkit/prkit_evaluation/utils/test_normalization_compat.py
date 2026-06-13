from prkit.prkit_evaluation.utils import normalization as current_normalization
from prkit.prkit_evaluation.utils import normalization_v1, normalization_v2


def test_normalization_v1_reexports_current_helpers():
    assert normalization_v1.normalize_text("  hello  ") == current_normalization.normalize_text("  hello  ")
    assert normalization_v1.normalize_number("4") == current_normalization.normalize_number("4")
    assert normalization_v1.classify_expression("x + y") == current_normalization.classify_expression("x + y")


def test_normalization_v2_reexports_current_helpers_and_constants():
    assert normalization_v2.normalize_answer("4") == current_normalization.normalize_answer("4")
    assert normalization_v2._UNIT_ALIASES["meter"] == current_normalization._UNIT_ALIASES["meter"]
    assert normalization_v2._NUM_TOKEN == current_normalization._NUM_TOKEN
