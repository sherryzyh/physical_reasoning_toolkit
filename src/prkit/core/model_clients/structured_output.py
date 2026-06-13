"""
Provider-neutral structured output helpers.

This module intentionally avoids provider-specific request semantics. It only
owns schema coercion, schema inspection, and generic JSON cleanup/extraction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Generic, Literal, TypeVar, Union

from pydantic import BaseModel

StructuredOutputFormat = Dict[str, Any]
StructuredOutputMode = Literal["json_schema", "json_object", "prompt_only"]
StructuredOutputPolicy = Literal["native_required", "best_effort"]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class SchemaFeatures:
    has_root_anyof: bool = False
    has_allof: bool = False
    has_prefix_items: bool = False
    has_recursive_refs: bool = False
    optional_field_count: int = 0
    union_field_count: int = 0
    has_numeric_bounds: bool = False
    has_string_constraints: bool = False
    has_open_objects: bool = False


@dataclass(frozen=True)
class StructuredOutputSpec:
    name: str
    schema: dict[str, Any]
    description: str | None = None
    strict: bool = True
    source_model: type[BaseModel] | None = None
    schema_features: SchemaFeatures | None = None

    def with_schema(self, schema: dict[str, Any]) -> "StructuredOutputSpec":
        return StructuredOutputSpec(
            name=self.name,
            schema=schema,
            description=self.description,
            strict=self.strict,
            source_model=self.source_model,
            schema_features=inspect_schema_features(schema),
        )


@dataclass(frozen=True)
class StructuredOutputPlan:
    mode: StructuredOutputMode
    strategy: str
    native_schema_enforced: bool
    accepted_artifact_modes: tuple[str, ...]
    accepted_artifact_strategies: tuple[str, ...]
    response_format: dict[str, Any] | type[BaseModel] | None
    prompt_suffix: str | None = None


@dataclass(frozen=True)
class StructuredCallResult(Generic[T]):
    parsed: T | None
    raw_text: str | None
    raw_payload: dict[str, Any] | list[Any] | None
    validation_error: str | None
    structured_output_mode: StructuredOutputMode
    structured_output_strategy: str
    native_schema_enforced: bool
    provider: str
    model_name: str

    def require_parsed(self) -> T:
        if self.parsed is not None:
            return self.parsed
        detail = self.validation_error or "Structured output could not be validated."
        raise ValueError(detail)


def sanitize_schema_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(name))
    cleaned = cleaned[:64]
    return cleaned or "response"


def structured_output_spec_to_response_format(
    spec: StructuredOutputSpec,
) -> StructuredOutputFormat:
    return {
        "type": "json_schema",
        "name": spec.name,
        "schema": spec.schema,
        "strict": spec.strict,
        "description": spec.description,
    }


def coerce_structured_output_spec(
    response_format: Union[StructuredOutputFormat, type[BaseModel]],
) -> StructuredOutputSpec:
    """
    Coerce a Pydantic model class or json_schema dict into a neutral spec.
    """

    if isinstance(response_format, dict):
        if response_format.get("type") != "json_schema":
            raise ValueError(
                "Structured output only supports type='json_schema'. "
                f"Got: {response_format.get('type')}"
            )
        if "schema" not in response_format:
            raise ValueError("response_format must contain 'schema' key")
        if "name" not in response_format:
            raise ValueError("response_format must contain 'name' key")
        schema = response_format["schema"]
        if not isinstance(schema, dict):
            raise ValueError("response_format['schema'] must be a dict")
        return StructuredOutputSpec(
            name=sanitize_schema_name(str(response_format["name"])),
            schema=schema,
            description=response_format.get("description"),
            strict=bool(response_format.get("strict", True)),
            source_model=None,
            schema_features=inspect_schema_features(schema),
        )

    try:
        schema = response_format.model_json_schema()
    except AttributeError:
        raise ValueError(
            "response_format must be a dict with type/name/schema or a Pydantic BaseModel. "
            f"Got: {type(response_format)}"
        ) from None

    return StructuredOutputSpec(
        name=sanitize_schema_name(
            getattr(response_format, "__name__", response_format.__class__.__name__)
        ),
        schema=schema,
        description=None,
        strict=True,
        source_model=response_format,
        schema_features=inspect_schema_features(schema),
    )


def normalize_response_format(
    response_format: Union[StructuredOutputFormat, type[BaseModel]],
) -> StructuredOutputFormat:
    """
    Backward-compatible wrapper that returns the canonical json_schema dict.
    """

    return structured_output_spec_to_response_format(
        coerce_structured_output_spec(response_format)
    )


def extract_schema_for_gemini(normalized_format: StructuredOutputFormat) -> Dict[str, Any]:
    return normalized_format["schema"]


def build_json_schema_prompt_suffix(schema: dict[str, Any]) -> str:
    return (
        "\n\nReturn ONLY valid JSON that matches this JSON Schema exactly.\n"
        "Do not include markdown fences, commentary, or any extra keys.\n"
        "JSON Schema:\n"
        + json.dumps(schema, indent=2, ensure_ascii=False)
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    payload = extract_json_payload(text)
    return payload if isinstance(payload, dict) else None


def extract_json_payload(text: str) -> dict[str, Any] | list[Any] | None:
    stripped = (text or "").strip()
    if not stripped:
        return None

    direct = _try_parse_json_payload(stripped)
    if direct is not None:
        return direct

    for match in _JSON_FENCE_RE.finditer(stripped):
        fenced_text = match.group(1).strip()
        direct = _try_parse_json_payload(fenced_text)
        if direct is not None:
            return direct
        for candidate in _iter_balanced_json_candidates(fenced_text):
            parsed_candidate = _try_parse_json_payload(candidate)
            if parsed_candidate is not None:
                return parsed_candidate

    for candidate in _iter_balanced_json_candidates(stripped):
        parsed_candidate = _try_parse_json_payload(candidate)
        if parsed_candidate is not None:
            return parsed_candidate

    return None


def inspect_schema_features(schema: dict[str, Any]) -> SchemaFeatures:
    optional_field_count = 0
    union_field_count = 0
    has_allof = False
    has_prefix_items = False
    has_recursive_refs = False
    has_numeric_bounds = False
    has_string_constraints = False
    has_open_objects = False

    seen_refs: set[str] = set()

    def visit(node: Any) -> None:
        nonlocal optional_field_count
        nonlocal union_field_count
        nonlocal has_allof
        nonlocal has_prefix_items
        nonlocal has_recursive_refs
        nonlocal has_numeric_bounds
        nonlocal has_string_constraints
        nonlocal has_open_objects

        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                if ref in seen_refs:
                    has_recursive_refs = True
                seen_refs.add(ref)

            if node.get("allOf"):
                has_allof = True
            if "prefixItems" in node:
                has_prefix_items = True
            if any(key in node for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")):
                has_numeric_bounds = True
            if any(key in node for key in ("minLength", "maxLength", "pattern")):
                has_string_constraints = True

            node_type = node.get("type")
            if node_type == "object":
                properties = node.get("properties")
                additional_properties = node.get("additionalProperties")
                if additional_properties not in (False, None):
                    has_open_objects = True
                if not isinstance(properties, dict) or not properties:
                    has_open_objects = True
                else:
                    required = set(node.get("required") or [])
                    optional_field_count += len(
                        [key for key in properties if key not in required]
                    )
                    for value in properties.values():
                        visit(value)

            for union_key in ("anyOf", "oneOf"):
                union_value = node.get(union_key)
                if isinstance(union_value, list):
                    union_field_count += 1
                    for item in union_value:
                        visit(item)
            for key, value in node.items():
                if key in {"properties", "anyOf", "oneOf"}:
                    continue
                visit(value)
            return

        if isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return SchemaFeatures(
        has_root_anyof=isinstance(schema, dict) and isinstance(schema.get("anyOf"), list),
        has_allof=has_allof,
        has_prefix_items=has_prefix_items,
        has_recursive_refs=has_recursive_refs,
        optional_field_count=optional_field_count,
        union_field_count=union_field_count,
        has_numeric_bounds=has_numeric_bounds,
        has_string_constraints=has_string_constraints,
        has_open_objects=has_open_objects,
    )


def schema_has_open_objects(schema: Any) -> bool:
    return inspect_schema_features(schema if isinstance(schema, dict) else {}).has_open_objects


def schema_contains_keyword(schema: Any, keyword: str) -> bool:
    if isinstance(schema, dict):
        if keyword in schema:
            return True
        return any(schema_contains_keyword(value, keyword) for value in schema.values())
    if isinstance(schema, list):
        return any(schema_contains_keyword(item, keyword) for item in schema)
    return False


def strip_schema_keywords(schema: Any, *, keywords: set[str] | frozenset[str]) -> Any:
    if isinstance(schema, dict):
        return {
            key: strip_schema_keywords(value, keywords=keywords)
            for key, value in schema.items()
            if key not in keywords
        }
    if isinstance(schema, list):
        return [strip_schema_keywords(item, keywords=keywords) for item in schema]
    return schema


def _try_parse_json_payload(candidate: str) -> dict[str, Any] | list[Any] | None:
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def _iter_balanced_json_candidates(text: str):
    for start, start_char in (
        (index, char)
        for index, char in enumerate(text)
        if char in "{["
    ):
        closing_char = "}" if start_char == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == start_char:
                depth += 1
            elif char == closing_char:
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


__all__ = [
    "SchemaFeatures",
    "StructuredCallResult",
    "StructuredOutputFormat",
    "StructuredOutputMode",
    "StructuredOutputPlan",
    "StructuredOutputPolicy",
    "StructuredOutputSpec",
    "build_json_schema_prompt_suffix",
    "coerce_structured_output_spec",
    "extract_json_object",
    "extract_json_payload",
    "extract_schema_for_gemini",
    "inspect_schema_features",
    "normalize_response_format",
    "sanitize_schema_name",
    "schema_contains_keyword",
    "schema_has_open_objects",
    "strip_schema_keywords",
    "structured_output_spec_to_response_format",
]
