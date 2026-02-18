from typing import Optional, Union

from prkit.prkit_core import PRKitLogger
from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_core.domain.answer_category import AnswerCategory
from prkit.prkit_evaluation.utils.answer_utils import same_comparison_category
from prkit.prkit_evaluation.utils.normalization import normalize_answer, normalize_text
from prkit.prkit_evaluation.utils.smartcompare_by_type import (
    compare_number,
    compare_plain_text,
    compare_physical_quantity,
    compare_formula,
    parse_physical_quantity,
)

from .base import BaseComparator


class SmartMatchComparator(BaseComparator):
    """
    Comparator that uses a smart match algorithm to compare answers.
    """
    DEFAULT_COMPARATORS = {
        AnswerCategory.NUMBER: compare_number,
        AnswerCategory.PHYSICAL_QUANTITY: compare_physical_quantity,
        AnswerCategory.FORMULA: compare_formula,
        AnswerCategory.TEXT: compare_plain_text,
        AnswerCategory.OPTION: compare_plain_text,
    }
    
    def __init__(self):
        """Initialize with default category comparators."""
        self._comparators = dict(self.DEFAULT_COMPARATORS)
        self.logger = PRKitLogger.get_logger(__name__)

    def _compare_by_category(
        self,
        category: AnswerCategory,
        predicted_norm: Union[float, str],
        ground_truth_norm: Union[float, str],
    ) -> bool:
        comparator = self._comparators.get(category, compare_plain_text)
        try:
            comparator_result = comparator(
                predicted_norm,
                ground_truth_norm,
            )
        except Exception as e:
            self.logger.warning(f"{category} comparator failed: {e}. Falling back to plain text comparison.")
            return compare_plain_text(
                predicted_norm,
                ground_truth_norm,
            )
        return comparator_result

    def compare(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
    ) -> bool:
        
        if isinstance(answer1, Answer):
            pred_norm = str(answer1.value)
            pred_cat = answer1.answer_category
        else:
            pred_cat, pred_norm = normalize_answer(answer1)
        if isinstance(answer2, Answer):
            gt_norm = str(answer2.value)
            gt_cat = answer2.answer_category
        else:
            gt_cat, gt_norm = normalize_answer(answer2)
            
        if same_comparison_category(gt_cat, pred_cat):
            return self._compare_by_category(gt_cat, pred_norm, gt_norm)

        # Extract RHS from equation-like pred (EQUATION cat or string with "lhs = rhs")
        if pred_cat == AnswerCategory.EQUATION or (
            "=" in str(pred_norm) and str(pred_norm).count("=") == 1
        ):
            parts = str(pred_norm).split("=", 1)
            pred_norm = parts[1].strip() if len(parts) > 1 else pred_norm
            pred_cat, pred_norm = normalize_answer(str(pred_norm))
        if gt_cat == AnswerCategory.EQUATION:
            parts = str(gt_norm).split("=", 1)
            gt_norm = parts[1].strip() if len(parts) > 1 else gt_norm
            gt_cat, gt_norm = normalize_answer(str(gt_norm))
        if same_comparison_category(gt_cat, pred_cat):
            self.logger.info(f"Same RHS category - answer: {gt_norm} and model answer: {pred_norm}")
            return self._compare_by_category(gt_cat, pred_norm, gt_norm)

        # Cross-category combinations (gt_cat != pred_cat), neither can be EQUATION
        if gt_cat == AnswerCategory.NUMBER and pred_cat == AnswerCategory.PHYSICAL_QUANTITY:
            # Compare the numerical part of pred_norm with gt_norm
            pred_num: Optional[float]
            pred_unit: str
            pred_num, pred_unit = parse_physical_quantity(str(pred_norm))
            self.logger.info(f"GT Answer: {gt_norm} ({gt_cat})")
            self.logger.info(f"Pred Answer: {pred_norm} ({pred_cat})")
            if pred_num is None:
                return False
            return compare_number(pred_num, gt_norm)
        elif gt_cat == AnswerCategory.NUMBER and pred_cat == AnswerCategory.FORMULA:
            # TODO:
            pass
        elif gt_cat == AnswerCategory.NUMBER and pred_cat == AnswerCategory.TEXT:
            self.logger.info(f"GT Answer: {gt_norm} ({gt_cat})")
            self.logger.info(f"Pred Answer: {pred_norm} ({pred_cat})")
            pred_str = str(pred_norm)
            gt_str = str(gt_norm)
            if gt_str in pred_str:
                return True
            # For whole numbers, also check int form (e.g. 42.0 -> "42")
            try:
                gt_val = float(gt_norm)
                if gt_val == int(gt_val):
                    if str(int(gt_val)) in pred_str:
                        return True
            except (ValueError, TypeError):
                pass
            return False
        elif gt_cat == AnswerCategory.NUMBER and pred_cat == AnswerCategory.OPTION:
            pass
        elif gt_cat == AnswerCategory.PHYSICAL_QUANTITY and pred_cat == AnswerCategory.NUMBER:
            # Missing unit is not allowed
            self.logger.info(f"GT Answer: {gt_norm} ({gt_cat})")
            self.logger.info(f"Pred Answer: {pred_norm} ({pred_cat})")
            return False
        elif gt_cat == AnswerCategory.PHYSICAL_QUANTITY and pred_cat == AnswerCategory.FORMULA:
            # TODO:
            pass
        elif gt_cat == AnswerCategory.PHYSICAL_QUANTITY and pred_cat == AnswerCategory.TEXT:
            # TODO: llm-as-judge
            pass
        elif gt_cat == AnswerCategory.PHYSICAL_QUANTITY and pred_cat == AnswerCategory.OPTION:
            pass
        elif gt_cat == AnswerCategory.FORMULA and pred_cat == AnswerCategory.NUMBER:
            # TODO:
            pass
        elif gt_cat == AnswerCategory.FORMULA and pred_cat == AnswerCategory.PHYSICAL_QUANTITY:
            # TODO: verify
            return False
        elif gt_cat == AnswerCategory.FORMULA and pred_cat == AnswerCategory.TEXT:
            # TODO:
            pass
        elif gt_cat == AnswerCategory.FORMULA and pred_cat == AnswerCategory.OPTION:
            pass
        elif gt_cat == AnswerCategory.TEXT and pred_cat == AnswerCategory.NUMBER:
            # TODO:
            pass
        elif gt_cat == AnswerCategory.TEXT and pred_cat == AnswerCategory.PHYSICAL_QUANTITY:
            # TODO: verify
            pred_num: Optional[float]
            pred_unit: str
            pred_num, pred_unit = parse_physical_quantity(str(pred_norm))
            self.logger.info(f"GT Answer: {gt_norm} ({gt_cat})")
            self.logger.info(f"Pred Answer: {pred_norm} ({pred_cat})")
            if pred_num is None:
                return False
            if str(pred_num) in str(gt_norm) and str(pred_unit) in str(gt_norm):
                return True
            return False
        elif gt_cat == AnswerCategory.TEXT and pred_cat == AnswerCategory.FORMULA:
            # TODO: verify
            self.logger.info(f"GT Answer: {gt_norm} ({gt_cat})")
            self.logger.info(f"Pred Answer: {pred_norm} ({pred_cat})")
            if str(pred_norm) in str(gt_norm):
                return True
            return False
        elif gt_cat == AnswerCategory.TEXT and pred_cat == AnswerCategory.OPTION:
            pass
        elif gt_cat == AnswerCategory.OPTION and pred_cat == AnswerCategory.NUMBER:
            pass
        elif gt_cat == AnswerCategory.OPTION and pred_cat == AnswerCategory.PHYSICAL_QUANTITY:
            pass
        elif gt_cat == AnswerCategory.OPTION and pred_cat == AnswerCategory.FORMULA:
            pass
        elif gt_cat == AnswerCategory.OPTION and pred_cat == AnswerCategory.TEXT:
            pass
        return False

    def accuracy_score(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
    ) -> float:
        is_match = self.compare(answer1, answer2)
        return 1.0 if is_match else 0.0
