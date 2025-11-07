# JSON Schemas you can pass as response_format for strict JSON.
ELICITATION_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["subject", "predicate", "object"],
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"}
                }
            }
        }
    },
    "required": ["facts"],
    "additionalProperties": False
}

ELICITATION_WITH_CONFIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["subject", "predicate", "object", "confidence"],
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                }
            }
        }
    },
    "required": ["facts"],
    "additionalProperties": False
}

NER_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "type", "keep"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["NE", "Literal", "Noise"]},
                    "keep": {"type": "boolean"}
                }
            }
        }
    },
    "required": ["entities"],
    "additionalProperties": False
}
