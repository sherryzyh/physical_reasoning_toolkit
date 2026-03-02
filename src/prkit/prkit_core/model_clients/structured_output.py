"""
Structured output utilities for model clients.

This module provides a canonical format (OpenAI-style) for structured output schemas
and helpers to convert/normalize them for different providers.

OpenAI format (canonical):
    {
        "type": "json_schema",
        "name": str,
        "schema": dict,  # JSON Schema object
        "strict": bool,
        "description": str | None  # optional
    }

Providers:
- OpenAI: Uses format directly (text.format parameter in Responses API)
- Gemini: Extracts schema dict, uses response_mime_type + response_json_schema
- Others: Not supported - will warn and raise or fall back
"""

from typing import Any, Dict, Optional, Union

# Type for the canonical structured output format (OpenAI-style)
StructuredOutputFormat = Dict[str, Any]


def _ensure_additional_properties_false(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively add additionalProperties: false to all object schemas.
    Required by OpenAI's strict structured output mode.
    """
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()

    if result.get("type") == "object":
        if "additionalProperties" not in result:
            result["additionalProperties"] = False

        if "properties" in result:
            result["properties"] = {
                k: _ensure_additional_properties_false(v)
                for k, v in result["properties"].items()
            }

    if "items" in result:
        result["items"] = _ensure_additional_properties_false(result["items"])

    if "$defs" in result:
        result["$defs"] = {
            k: _ensure_additional_properties_false(v)
            for k, v in result["$defs"].items()
        }

    return result


def normalize_response_format(
    response_format: Union[StructuredOutputFormat, type],
) -> StructuredOutputFormat:
    """
    Normalize response_format to the canonical OpenAI-style dict format.

    Args:
        response_format: Either:
            - A dict with keys: type, name, schema, strict (and optionally description)
            - A Pydantic BaseModel class (will extract schema via model_json_schema)

    Returns:
        Normalized dict with type, name, schema, strict keys

    Raises:
        ValueError: If format is invalid or not a supported type
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
        schema = _ensure_additional_properties_false(response_format["schema"])
        return {
            "type": "json_schema",
            "name": response_format["name"],
            "schema": schema,
            "strict": response_format.get("strict", True),
            "description": response_format.get("description"),
        }

    # Assume Pydantic BaseModel
    try:
        schema = response_format.model_json_schema()
    except AttributeError:
        raise ValueError(
            "response_format must be a dict with type/name/schema or a Pydantic BaseModel. "
            f"Got: {type(response_format)}"
        ) from None

    # Use model name as schema name (sanitize for OpenAI: a-z, A-Z, 0-9, underscore, dash)
    name = getattr(
        response_format, "__name__", response_format.__class__.__name__
    )
    name = "".join(
        c if c.isalnum() or c in "_-" else "_" for c in str(name)
    )[:64] or "response"

    schema = _ensure_additional_properties_false(schema)
    return {
        "type": "json_schema",
        "name": name,
        "schema": schema,
        "strict": True,
        "description": None,
    }


def extract_schema_for_gemini(normalized_format: StructuredOutputFormat) -> Dict[str, Any]:
    """
    Extract the raw JSON Schema dict for Gemini's response_json_schema.

    Gemini uses standard JSON Schema; we pass the schema object directly.
    """
    return normalized_format["schema"]
