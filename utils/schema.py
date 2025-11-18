from __future__ import annotations

# JSON schema for a single LLMPedia article response.
# Models must return STRICT JSON that validates this schema.

LLMPEDIA_ARTICLE_SCHEMA = {
    "type": "object",
    "properties": {
        "article": {
            "type": "object",
            "properties": {
                # Canonical page title (we let the model echo/normalize the subject)
                "subject": {"type": "string", "minLength": 1},

                # The full page body in MediaWiki (Wiki) syntax
                "wikitext": {"type": "string", "minLength": 1},

                # Optional: list of outgoing links (page titles). If omitted, we’ll
                # fall back to parsing [[...]] from wikitext.
                "links": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True
                }
            },
            "required": ["subject", "wikitext"],
            "additionalProperties": False
        }
    },
    "required": ["article"],
    "additionalProperties": False
}
