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

    # ---------------- OpenAI ----------------
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

    # ---------------- SCADSAI (OpenAI-compatible) ----------------
    if prov == "scadsai":
        # Prefer cfg.api_key_env if set in settings (recommended),
        # otherwise fall back to SCADSAI_API_KEY.
        api_key = os.getenv(cfg.api_key_env or "SCADSAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing SCADSAI_API_KEY (or cfg.api_key_env).")

        # Prefer cfg.base_url if set in settings, otherwise env, otherwise default.
        base_url = cfg.base_url or os.getenv("SCADSAI_BASE_URL") or "https://llm.scads.ai/v1"

        return OpenAIClient(
            model=cfg.model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_responses_api=bool(getattr(cfg, "use_responses_api", False)),
            extra_inputs=getattr(cfg, "extra_inputs", None),
            max_requests_per_minute=float(os.getenv("SCADSAI_RATE_LIMIT", "0")),
        )

    # ---------------- Replicate ----------------
    if prov == "replicate":
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise RuntimeError("Missing REPLICATE_API_TOKEN.")
        return ReplicateLLM(
            model=cfg.model,
            api_token=token,
            default_extra=getattr(cfg, "extra_inputs", None),
        )

    # ---------------- DeepSeek ----------------
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


