import importlib.util

_uq_available = (
    importlib.util.find_spec("uncertainty_quantification_physical_reasoning")
    is not None
)

collect_ignore: list[str] = []
if not _uq_available:
    collect_ignore = [
        "test_batch_prepare_common.py",
        "test_extract_answer_parsers.py",
        "test_single_inference_with_answer_tags.py",
    ]
