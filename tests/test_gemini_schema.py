"""Offline tests for the Gemini schema conversion (no network)."""

import json

from src.gemini_llm import to_gemini_schema
from src.tools.registry import REGISTRY


def _walk(node):
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


class TestGeminiSchemaConversion:
    def test_every_tool_schema_converts_cleanly(self):
        for spec in REGISTRY.values():
            converted = to_gemini_schema(spec.input_model.model_json_schema())
            text = json.dumps(converted)
            assert "$ref" not in text, spec.name
            assert "$defs" not in text, spec.name
            assert "additionalProperties" not in text, spec.name
            assert converted["type"] == "object"

    def test_nested_refs_are_inlined(self):
        schema = REGISTRY["build_compliance_matrix"].input_model.model_json_schema()
        converted = to_gemini_schema(schema)
        # MatrixRequirement fields must appear inline under requirements.items
        item_props = converted["properties"]["requirements"]["items"]["properties"]
        assert {"requirement_id", "title", "kind"} <= set(item_props)

    def test_required_and_enum_preserved(self):
        schema = REGISTRY["build_compliance_matrix"].input_model.model_json_schema()
        converted = to_gemini_schema(schema)
        assert set(converted["required"]) == {"requirements", "findings"}
        status = converted["properties"]["findings"]["items"]["properties"]["status"]
        assert set(status["enum"]) == {"compliant", "partial", "missing"}
