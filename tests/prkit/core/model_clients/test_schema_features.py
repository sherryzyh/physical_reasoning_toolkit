"""Characterization tests for ``inspect_schema_features`` and ``SchemaFeatures``.

These pin the behaviour of every ``SchemaFeatures`` field, which had no coverage
at all. Written before the ``$ref`` walker is corrected, so the reference-flag
assertions here describe today's behaviour — including the defect that repeated
use of one ``$def`` is reported as recursion.
"""

from unittest.mock import MagicMock

import pytest

from prkit.core.model_clients.anthropic import AnthropicModel
from prkit.core.model_clients.structured_output import inspect_schema_features
from prkit.results.schema import PhysicsEvalResult
from prkit.semantics.build.strict_models import StrictPredictionSemanticsResponse
from prkit.semantics.schema.models import (
    PhysicalQuantityView,
    PhysicsEvaluationContract,
)


def _closed_object(properties: dict, required: list[str]) -> dict:
    """Build a closed object schema, the shape every provider transform forces."""
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class TestSchemaFeatureCounts:
    def test_counts_optional_and_union_fields(self):
        schema = _closed_object(
            {
                "kept": {"type": "string"},
                "optional_one": {"type": "string"},
                "optional_two": {"type": "integer"},
                "either": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            ["kept"],
        )

        features = inspect_schema_features(schema)

        assert features.optional_field_count == 3
        assert features.union_field_count == 1

    def test_flags_allof_prefix_items_and_numeric_bounds(self):
        schema = _closed_object(
            {
                "score": {"type": "number", "minimum": 0, "maximum": 10},
                "pair": {
                    "type": "array",
                    "prefixItems": [{"type": "string"}, {"type": "integer"}],
                },
                "merged": {
                    "allOf": [_closed_object({"a": {"type": "string"}}, ["a"])],
                },
            },
            ["score", "pair", "merged"],
        )

        features = inspect_schema_features(schema)

        assert features.has_allof is True
        assert features.has_prefix_items is True
        assert features.has_numeric_bounds is True
        assert features.has_string_constraints is False

    def test_flags_string_constraints(self):
        schema = _closed_object(
            {"answer": {"type": "string", "minLength": 1}}, ["answer"]
        )

        assert inspect_schema_features(schema).has_string_constraints is True


class TestOpenObjects:
    def test_object_without_declared_properties_is_reported_open(self):
        """An object with no ``properties`` is reported open even when closed.

        ``additionalProperties: false`` here says the opposite, so this conflates
        "the author allows extra keys" with "the object declares no keys". Pinned
        as-is; callers must not read ``has_open_objects`` as author intent.
        """
        schema = {"type": "object", "additionalProperties": False}

        assert inspect_schema_features(schema).has_open_objects is True

    def test_additional_properties_true_is_reported_open(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": True,
        }

        assert inspect_schema_features(schema).has_open_objects is True

    def test_closed_object_with_properties_is_not_open(self):
        schema = _closed_object({"a": {"type": "string"}}, ["a"])

        assert inspect_schema_features(schema).has_open_objects is False


class TestRootAnyOf:
    def test_root_level_anyof_sets_the_flag(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}

        assert inspect_schema_features(schema).has_root_anyof is True

    def test_nested_anyof_does_not_set_the_root_flag(self):
        schema = _closed_object(
            {"either": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            ["either"],
        )

        features = inspect_schema_features(schema)

        assert features.has_root_anyof is False
        assert features.union_field_count == 1


class TestReferenceFlags:
    def test_repeated_ref_is_reported_as_recursion(self):
        """Two fields of the same nested type set ``has_recursive_refs``.

        This is reuse, not recursion — the ``$ref`` graph has no cycle. The flag
        is set purely because the same ``$ref`` string appears twice. Pinned here
        so that correcting the walker is a visible, reviewable change.
        """
        schema = {
            "$defs": {"Item": _closed_object({"name": {"type": "string"}}, ["name"])},
            **_closed_object(
                {
                    "findings": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Item"},
                    },
                    "rebuttals": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Item"},
                    },
                },
                ["findings", "rebuttals"],
            ),
        }

        assert inspect_schema_features(schema).has_recursive_refs is True

    def test_single_use_ref_is_not_flagged(self):
        schema = {
            "$defs": {"Item": _closed_object({"name": {"type": "string"}}, ["name"])},
            **_closed_object({"finding": {"$ref": "#/$defs/Item"}}, ["finding"]),
        }

        assert inspect_schema_features(schema).has_recursive_refs is False

    def test_self_referencing_definition_is_flagged(self):
        """A genuine self-loop is flagged — but only because the ``$ref`` repeats."""
        schema = {
            "$defs": {
                "Node": _closed_object(
                    {
                        "child": {"$ref": "#/$defs/Node"},
                        "sibling": {"$ref": "#/$defs/Node"},
                    },
                    ["child", "sibling"],
                )
            },
            **_closed_object({"root": {"$ref": "#/$defs/Node"}}, ["root"]),
        }

        assert inspect_schema_features(schema).has_recursive_refs is True


class TestAnthropicPlanModeBaseline:
    """Pins the Anthropic plan mode for models spanning the demotion reasons.

    ``_anthropic_native_schema_incompatibility`` demotes on recursive refs, on
    more than 24 optional fields, or on more than 16 union-typed fields. These
    four models cover a reference-only demotion, a mixed one, a genuinely
    recursive one, and a field-count one.
    """

    @staticmethod
    def _anthropic_client() -> AnthropicModel:
        client = object.__new__(AnthropicModel)
        client.model = "claude-sonnet-4-6"
        client.provider = "anthropic"
        client.logger = MagicMock()
        return client

    @pytest.mark.parametrize(
        ("response_model", "expected_mode"),
        [
            (PhysicalQuantityView, "prompt_only"),
            (PhysicsEvaluationContract, "prompt_only"),
            (StrictPredictionSemanticsResponse, "prompt_only"),
            (PhysicsEvalResult, "prompt_only"),
        ],
    )
    def test_anthropic_plan_mode_for_representative_models(
        self, response_model, expected_mode
    ):
        plan = self._anthropic_client().resolve_structured_output_plan(
            response_model,
            structured_policy="best_effort",
        )

        assert plan.mode == expected_mode
