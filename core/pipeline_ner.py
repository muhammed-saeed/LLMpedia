from __future__ import annotations
from typing import Dict, Any
from llm.factory import make_llm_from_config
from .prompt_loader import load_messages_from_prompt_json
from prompts.schemas import NER_SCHEMA

def run_ner(
    cfg,
    prompt_path: str,
    phrases_block: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    extra_inputs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Loads the prompt JSON, formats system+user, and calls the LLM with a strict JSON schema.
    """
    llm = make_llm_from_config(cfg)

    messages = load_messages_from_prompt_json(
        prompt_path,
        phrases_block=phrases_block
    )

    out = llm(
        messages,
        json_schema=NER_SCHEMA
    )
    return out
