# prompts/article_schemas.py
from __future__ import annotations

ARTICLE_SCHEMA_BASE = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "infobox": {
            "type": "object",
            "additionalProperties": {"type": "string"}
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["heading", "content_md"],
                "properties": {
                    "heading": {"type": "string"},
                    "content_md": {"type": "string"}
                }
            }
        },
        "references": {
            "type": "array",
            "items": {"type": "string"}
        },
        "categories": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["title", "summary", "sections"],
    "additionalProperties": False
}

ARTICLE_SCHEMA_CAL = {
    **ARTICLE_SCHEMA_BASE,
    "properties": {
        **ARTICLE_SCHEMA_BASE["properties"],
        "overall_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["heading", "content_md"],
                "properties": {
                    "heading": {"type": "string"},
                    "content_md": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                }
            }
        }
    }
}
