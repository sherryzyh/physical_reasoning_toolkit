"""Characterization tests for ``inspect_schema_features`` and ``SchemaFeatures``.

These pin the behaviour of every ``SchemaFeatures`` field, which had no coverage
at all, and cover the ``$ref`` graph walk that distinguishes a genuinely
recursive schema from one that merely reuses a definition.
"""

from unittest.mock import MagicMock

import pytest

from prkit.core.model_clients.anthropic import AnthropicModel
from prkit.core.model_clients.structured_output import (
    inspect_schema_features,
    schema_has_circular_refs,
)
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
    def test_repeated_ref_is_reuse_not_recursion(self):
        """Two fields of the same nested type reuse a definition; they do not recurse.

        The ``$ref`` graph here has no cycle, so ``has_circular_refs`` stays
        ``False`` while ``has_repeated_refs`` records the reuse.
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

        features = inspect_schema_features(schema)

        assert features.has_repeated_refs is True
        assert features.has_circular_refs is False

    def test_single_use_ref_is_not_flagged(self):
        schema = {
            "$defs": {"Item": _closed_object({"name": {"type": "string"}}, ["name"])},
            **_closed_object({"finding": {"$ref": "#/$defs/Item"}}, ["finding"]),
        }

        features = inspect_schema_features(schema)

        assert features.has_repeated_refs is False
        assert features.has_circular_refs is False

    def test_self_referencing_definition_is_circular(self):
        """A definition that references itself is genuinely recursive."""
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

        assert inspect_schema_features(schema).has_circular_refs is True


class TestCircularRefDetection:
    """Direct coverage of the ``$ref`` graph walk behind ``has_circular_refs``."""

    def test_mutual_recursion_is_circular(self):
        schema = {
            "$defs": {
                "A": _closed_object({"b": {"$ref": "#/$defs/B"}}, ["b"]),
                "B": _closed_object({"a": {"$ref": "#/$defs/A"}}, ["a"]),
            },
            **_closed_object({"root": {"$ref": "#/$defs/A"}}, ["root"]),
        }

        assert schema_has_circular_refs(schema) is True

    def test_deep_non_recursive_chain_is_not_circular(self):
        schema = {
            "$defs": {
                "A": _closed_object({"b": {"$ref": "#/$defs/B"}}, ["b"]),
                "B": _closed_object({"c": {"$ref": "#/$defs/C"}}, ["c"]),
                "C": _closed_object({"leaf": {"type": "string"}}, ["leaf"]),
            },
            **_closed_object({"root": {"$ref": "#/$defs/A"}}, ["root"]),
        }

        assert schema_has_circular_refs(schema) is False

    def test_root_pointer_self_reference_is_circular(self):
        schema = _closed_object({"self": {"$ref": "#"}}, ["self"])

        assert schema_has_circular_refs(schema) is True

    def test_unresolvable_ref_is_not_circular(self):
        """prkit cannot follow a dangling pointer, so it must not claim recursion."""
        schema = _closed_object({"x": {"$ref": "#/$defs/Missing"}}, ["x"])

        assert schema_has_circular_refs(schema) is False

    def test_external_ref_is_not_circular(self):
        schema = _closed_object({"x": {"$ref": "https://example.com/s.json#/A"}}, ["x"])

        assert schema_has_circular_refs(schema) is False

    def test_json_pointer_escapes_are_resolved(self):
        """A definition named ``a/b`` is addressed as ``#/$defs/a~1b`` (RFC 6901)."""
        schema = {
            "$defs": {"a/b": _closed_object({"c": {"$ref": "#/$defs/a~1b"}}, ["c"])},
            **_closed_object({"x": {"type": "string"}}, ["x"]),
        }

        assert schema_has_circular_refs(schema) is True

    def test_cycle_in_unreferenced_definition_is_detected(self):
        """Providers validate the whole document, so an orphan cycle still counts."""
        schema = {
            "$defs": {"Loop": _closed_object({"c": {"$ref": "#/$defs/Loop"}}, ["c"])},
            **_closed_object({"x": {"type": "string"}}, ["x"]),
        }

        assert schema_has_circular_refs(schema) is True

    def test_nested_defs_do_not_create_a_phantom_edge(self):
        """A definition's own ``$defs`` must not lend edges to whatever holds it."""
        schema = {
            "$defs": {
                "A": {
                    **_closed_object({"x": {"type": "string"}}, ["x"]),
                    "$defs": {"Inner": {"$ref": "#/$defs/A"}},
                }
            },
            **_closed_object({"root": {"$ref": "#/$defs/A"}}, ["root"]),
        }

        assert schema_has_circular_refs(schema) is False

    def test_draft_07_definitions_container_is_walked(self):
        schema = {
            "definitions": {
                "N": _closed_object({"c": {"$ref": "#/definitions/N"}}, ["c"])
            },
            **_closed_object({"root": {"$ref": "#/definitions/N"}}, ["root"]),
        }

        assert schema_has_circular_refs(schema) is True

    def test_non_dict_schema_is_not_circular(self):
        assert schema_has_circular_refs("not a schema") is False

    def test_strict_prediction_semantics_response_is_genuinely_circular(self):
        """Pins that this model does not flip: its ``$defs`` contain a real cycle."""
        schema = StrictPredictionSemanticsResponse.model_json_schema()

        assert schema_has_circular_refs(schema) is True


class TestAnthropicPlanModeBaseline:
    """Pins the Anthropic plan mode for models spanning the demotion reasons.

    ``_anthropic_native_schema_incompatibility`` demotes on circular refs, on
    more than 24 optional fields, or on more than 16 union-typed fields. The
    first two models reuse a definition without recursing and are now enforced
    natively; the third is genuinely recursive and the fourth exceeds the
    optional-field limit, so both still demote.
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
            (PhysicalQuantityView, "json_schema"),
            (PhysicsEvaluationContract, "json_schema"),
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
