# llm/factory.py
from __future__ import annotations
from typing import Any, Dict
import os

from llm.config import ModelConfig
from llm.openai_client import OpenAIClient
from llm.replicate_client import ReplicateLLM
from llm.deepseek_client import DeepSeekClient

def make_llm_from_config(cfg: ModelConfig):
    prov = (cfg.provider or "").lower()

    if prov == "openai":
        api_key = os.getenv(cfg.api_key_env or "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY.")
        return OpenAIClient(
            model=cfg.model,
            api_key=api_key,
            base_url=cfg.base_url,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_responses_api=bool(getattr(cfg, "use_responses_api", False)),
            extra_inputs=getattr(cfg, "extra_inputs", None),
        )

    if prov == "replicate":
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise RuntimeError("Missing REPLICATE_API_TOKEN.")
        # Pass model defaults (prompt_template, stop_sequences, etc.) through
        return ReplicateLLM(
            model=cfg.model,
            api_token=token,
            default_extra=getattr(cfg, "extra_inputs", None),
        )

    if prov == "deepseek":
        api_key = os.getenv(cfg.api_key_env or "DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY.")
        return DeepSeekClient(
            model=cfg.model,
            api_key=api_key,
            base_url=cfg.base_url,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            extra_inputs=getattr(cfg, "extra_inputs", None),
        )

    if prov == "unsloth":
        raise RuntimeError("Unsloth backend not available in this environment.")

    raise ValueError(f"Unknown provider: {prov}")
