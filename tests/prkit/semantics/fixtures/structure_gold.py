"""Gold corpus for answer-structure classification + canonicalization.

Each row pairs an answer surface (under an optional question context) with the structure
and object kind it *should* classify as. ``gate_structure`` / ``gate_kind`` mark rows whose
label is confirmed AND currently classified correctly, so the harness asserts them as a
regression lock; rows with a gate set to ``False`` carry a ``note`` describing a known
classifier gap and are reported (not asserted).

NOTE (provenance): these labels were bootstrapped against the current classifier and the
structure tie-break rules in ``comparison/STRUCTURE.md``. They are the seed corpus — rows
used as gates should be human-audited before being relied on as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StructureGoldRow:
    """One labeled structure-classification example."""

    answer: str
    expected_structure: str
    expected_object_kind: str
    context: dict[str, Any] = field(default_factory=dict)
    gate_structure: bool = True
    gate_kind: bool = True
    note: str = ""


_CHOICE_CTX = {"choice_space": ("A", "B", "C", "D")}
_PARTS_CTX = {"required_parts": ("magnitude", "direction"), "ordering": "per_part"}


STRUCTURE_GOLD: tuple[StructureGoldRow, ...] = (
    # --- atomic object kinds ---
    StructureGoldRow("3 m/s", "atomic", "physical_quantity"),
    StructureGoldRow("9.8 m/s^2", "atomic", "physical_quantity"),
    StructureGoldRow("1/2", "atomic", "number"),
    StructureGoldRow("9.8", "atomic", "number"),
    StructureGoldRow("v = a t", "atomic", "relation"),
    StructureGoldRow("x^2 + 1", "atomic", "expression"),
    StructureGoldRow("B", "atomic", "choice", context=_CHOICE_CTX),
    StructureGoldRow("yes", "atomic", "boolean"),
    StructureGoldRow("true", "atomic", "boolean"),
    StructureGoldRow("clockwise", "atomic", "sign_direction"),
    StructureGoldRow(
        "increases",
        "atomic",
        "qualitative_label",
        gate_kind=False,
        note="known gap: classifier returns object_kind=expression for bare verbs",
    ),
    # --- tuple (ordered coordinate of one object) ---
    StructureGoldRow("(1, 2)", "tuple", "number"),
    StructureGoldRow("(1, 2, 3)", "tuple", "number"),
    # boundary: a bare finite (a, b) is a TUPLE, not an INTERVAL
    StructureGoldRow("(2, 3)", "tuple", "number", note="boundary: bare finite ⇒ tuple"),
    # --- set (unordered collection) ---
    StructureGoldRow("{1, 2}", "set", "number"),
    StructureGoldRow("{1, 2, 3}", "set", "number"),
    # --- interval (connected range) ---
    StructureGoldRow("[2, 3]", "interval", "number"),
    StructureGoldRow(
        "[2, 3)", "interval", "number", note="boundary: bracket ⇒ interval"
    ),
    StructureGoldRow("[0, 1]", "interval", "number"),
    StructureGoldRow(
        "(2, oo)",
        "interval",
        "number",
        gate_kind=False,
        note="boundary: infinity ⇒ interval; aggregate kind currently expression",
    ),
    # --- shaped ---
    StructureGoldRow("<1, 2, 3>", "vector", "number"),
    StructureGoldRow(r"2\hat{i} - 3\hat{j}", "vector", "number"),
    StructureGoldRow(
        r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}", "matrix", "number"
    ),
    # --- multi_part ---
    StructureGoldRow(
        "3 m/s, east",
        "multi_part",
        "physical_quantity",
        context=_PARTS_CTX,
        gate_kind=False,
        note="aggregate kind over heterogeneous parts",
    ),
    StructureGoldRow(
        "x = 1; y = 2",
        "multi_part",
        "relation",
        gate_kind=False,
        note="aggregate kind over parts",
    ),
    # --- subject_to (atomic answer carrying a side condition) ---
    StructureGoldRow(
        "E = k/r, a < r < b",
        "atomic",
        "relation",
        note="trailing constraint parsed as subject_to, answer stays atomic",
    ),
)


# Adversarial pairs that MUST stay non-equivalent (precision floor). Each holds today and
# must keep holding after structure canonicalization lands.
ADVERSARIAL_DISTINCT: tuple[tuple[str, str, str], ...] = (
    ("{1, 2}", "(1, 2)", "set vs tuple: different structure"),
    ("(1, 2)", "(2, 1)", "tuple is ordered"),
    ("[2, 3]", "(2, 3)", "interval vs tuple: different structure"),
    ("[2, 3]", "[2, 3)", "closed vs half-open interval"),
    ("{1, 2}", "{1, 3}", "different set elements"),
    ("(1, 2, 3)", "(1, 2)", "different cardinality"),
)
