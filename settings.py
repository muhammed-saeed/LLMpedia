#settings.py
from __future__ import annotations
import os
from typing import Dict
from pydantic import BaseModel
from llm.config import ModelConfig

SCADSAI_DEFAULT_BASE_URL = os.getenv("SCADSAI_BASE_URL", "https://llm.scads.ai/v1")
# ---------- JSON Schemas (triples/ner) ----------

ELICIT_SCHEMA_BASE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
            },
        }
    },
    "required": ["facts"],
}

ELICIT_SCHEMA_CAL = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["subject", "predicate", "object", "confidence"],
            },
        }
    },
    "required": ["facts"],
}

NER_SCHEMA_BASE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "phrases": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "phrase": {"type": "string"},
                    "is_ne": {"type": "boolean"},
                },
                "required": ["phrase", "is_ne"],
            },
        }
    },
    "required": ["phrases"],
}

NER_SCHEMA_CAL = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "phrases": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "phrase": {"type": "string"},
                    "is_ne": {"type": "boolean"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["phrase", "is_ne", "confidence"],
            },
        }
    },
    "required": ["phrases"],
}

# ---------- JSON Schemas (articles namespace) ----------

ARTICLE_SCHEMA_BASE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "article": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subject": {"type": "string"},
                "wikitext": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["subject", "wikitext"],
        }
    },
    "required": ["article"],
}

ARTICLE_NER_SCHEMA_BASE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "phrases": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "phrase": {"type": "string"},
                    "is_ne": {"type": "boolean"},
                },
                "required": ["phrase", "is_ne"],
            },
        }
    },
    "required": ["phrases"],
}

# ---------- SQLite DDL ----------

QUEUE_DDL = """
CREATE TABLE IF NOT EXISTS queue (
    subject        TEXT NOT NULL,
    subject_norm   TEXT NOT NULL,
    subject_canon  TEXT NOT NULL,
    hop            INTEGER NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',  -- pending | working | done | failed
    retries        INTEGER NOT NULL DEFAULT 0,
    created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(subject_canon)
);
"""

FACTS_DDL = """
CREATE TABLE IF NOT EXISTS triples_accepted (
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    hop         INTEGER NOT NULL,
    model_name  TEXT,
    strategy    TEXT,
    confidence  REAL,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS triples_sink (
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    hop         INTEGER NOT NULL,
    model_name  TEXT,
    strategy    TEXT,
    confidence  REAL,
    reason      TEXT,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

# ---------- Token Pricing (per token, in USD) ----------

OPENAI_TOKEN_PRICES: Dict[str, Dict[str, float]] = {
    "gpt-5.1": {
        "input": 1.250 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
        "output": 10.000 / 1_000_000,
        "reasoning_input": 0.0,
        "reasoning_output": 0.0,
    },

    # ----- GPT-5 Family -----
    "gpt-5": {
        "input": 1.250 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
        "output": 10.000 / 1_000_000,
        "reasoning_input": 0.0,
        "reasoning_output": 0.0,
    },
    "gpt-5-mini": {
        "input": 0.250 / 1_000_000,
        "cached_input": 0.025 / 1_000_000,
        "output": 2.000 / 1_000_000,
        "reasoning_input": 0.0,
        "reasoning_output": 0.0,
    },
    "gpt-5-nano": {
        "input": 0.050 / 1_000_000,
        "cached_input": 0.005 / 1_000_000,
        "output": 0.400 / 1_000_000,
        "reasoning_input": 0.0,
        "reasoning_output": 0.0,
    },
    "gpt-5-pro": {
        "input": 15.00 / 1_000_000,
        "cached_input": 15.00 / 1_000_000,
        "output": 120.00 / 1_000_000,
        "reasoning_input": 0.0,
        "reasoning_output": 0.0,
    },

    # ----- GPT-4.1 Family -----
    "gpt-4.1": {
        "input": 3.00 / 1_000_000,
        "cached_input": 0.75 / 1_000_000,
        "output": 12.00 / 1_000_000,
        "reasoning_input": 0.0,
        "reasoning_output": 0.0,
    },
    "gpt-4.1-mini": {
        "input": 0.80 / 1_000_000,
        "cached_input": 0.20 / 1_000_000,
        "output": 3.20 / 1_000_000,
        "reasoning_input": 0.0,
        "reasoning_output": 0.0,
    },
    "gpt-4.1-nano": {
        "input": 0.20 / 1_000_000,
        "cached_input": 0.05 / 1_000_000,
        "output": 0.80 / 1_000_000,
        "reasoning_input": 0.0,
        "reasoning_output": 0.0,
    },

    # ----- o4-mini -----
    "o4-mini": {
        "input": 4.00 / 1_000_000,
        "cached_input": 1.00 / 1_000_000,
        "output": 16.00 / 1_000_000,
        "reasoning_input": 0.0,
        "reasoning_output": 0.0,
    },
}

# ---------- Settings ----------


class Settings(BaseModel):
    CONCURRENCY: int = 8
    MAX_DEPTH: int = 2
    NER_BATCH_SIZE: int = 5
    MAX_FACTS_HINT: int = 50

    MODELS: Dict[str, ModelConfig] = {
        # -------- OpenAI (Chat Completions) --------
        "gpt4o": ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),
        "gpt4o-mini": ModelConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),
        "gpt4-turbo": ModelConfig(
            provider="openai",
            model="gpt-4-turbo",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),

        # -------- OpenAI (Chat Completions) — GPT-4.1 family --------
        "gpt-4.1": ModelConfig(
            provider="openai",
            model="gpt-4.1",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),
        "gpt-4.1-mini": ModelConfig(
            provider="openai",
            model="gpt-4.1-mini",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),
        "gpt-4.1-nano": ModelConfig(
            provider="openai",
            model="gpt-4.1-nano",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),

        # -------- OpenAI (Responses API) — GPT-5 family --------
        "gpt-5.1": ModelConfig(
            provider="openai",
            model="gpt-5.1",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0,      # now defaulted, not None
            top_p=1.0,
            max_tokens=128_000,
            use_responses_api=True,
            extra_inputs={
                "reasoning": {"effort": "none"},
                "text": {"verbosity": "medium"},
            },
        ),
        "gpt-5": ModelConfig(
            provider="openai",
            model="gpt-5",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=True,
            extra_inputs={
                "reasoning": {"effort": "medium"},
                "text": {"verbosity": "medium"},
            },
        ),
        "gpt-5-mini": ModelConfig(
            provider="openai",
            model="gpt-5-mini",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=True,
            extra_inputs={
                "reasoning": {"effort": "low"},
                "text": {"verbosity": "low"},
            },
        ),
        "gpt-5-nano": ModelConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=True,
            extra_inputs={
                "reasoning": {"effort": "low"},   # valid per docs
                "text": {"verbosity": "low"},
            },
        ),
        #scadsai models
        "scads-alias-ha": ModelConfig(
            provider="scadsai",
            model="alias-ha",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),
        "scads-alias-reasoning": ModelConfig(
            provider="scadsai",
            model="alias-reasoning",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),
        "scads-alias-code": ModelConfig(
            provider="scadsai",
            model="alias-code",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),

        "scads-llama-3.3-70b": ModelConfig(
            provider="scadsai",
            model="meta-llama/Llama-3.3-70B-Instruct",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.2,
            top_p=0.95,
            max_tokens=4096,
            use_responses_api=False,
        ),
        "scads-llama-3.1-8b": ModelConfig(
            provider="scadsai",
            model="meta-llama/Llama-3.1-8B-Instruct",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.2,
            top_p=0.95,
            max_tokens=4096,
            use_responses_api=False,
        ),
        "scads-llama-4-scout-17b": ModelConfig(
            provider="scadsai",
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.2,
            top_p=0.95,
            max_tokens=4096,
            use_responses_api=False,
        ),

        "scads-teuken-7b": ModelConfig(
            provider="scadsai",
            model="openGPT-X/Teuken-7B-instruct-research-v0.4",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.2,
            top_p=0.95,
            max_tokens=4096,
            use_responses_api=False,
        ),

        # Alternative DeepSeek model on SCADSAI (often more reliable than V3.2-Exp)
        "scads-deepseek-coder-v2-lite": ModelConfig(
            provider="scadsai",
            model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),

        # Your “non-working” one (keep it, but you can switch away in CLI)
        "scads-DeepSeek-V3.2": ModelConfig(
            provider="scadsai",
            model="deepseek-ai/DeepSeek-V3.2",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),

        # GPT-OSS on SCADSAI
        "scads-gpt-oss-120b": ModelConfig(
            provider="scadsai",
            model="openai/gpt-oss-120b",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),

        "scads-qwen3-coder-30b-a3b": ModelConfig(
            provider="scadsai",
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.2,
            top_p=0.95,
            max_tokens=4096,
            use_responses_api=False,
        ),
    
        "scads-minimax-m2.5": ModelConfig(
            provider="scadsai",
            model="MiniMaxAI/MiniMax-M2.5",
            api_key_env="SCADSAI_API_KEY",
            base_url=SCADSAI_DEFAULT_BASE_URL,
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            use_responses_api=False,
        ),


        # -------- DeepSeek --------
        "deepseek": ModelConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            temperature=0.0,
            top_p=0.95,
            max_tokens=8192,
        ),
        "deepseek-reasoner": ModelConfig(
            provider="deepseek",
            model="deepseek-reasoner",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            temperature=0.0,
            top_p=0.95,
            max_tokens=4096,
        ),

        # -------- Replicate (various) --------
        "llama3-70b-instruct": ModelConfig(
            provider="replicate",
            model="meta/meta-llama-3-70b-instruct",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=8192,
            extra_inputs={
                "system_prompt": "You are a helpful assistant",
                "prompt_template": (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                ),
                "length_penalty": 1,
                "presence_penalty": 1.15,
                "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
                "log_performance_metrics": False,
            },
        ),
        "llama3-8b-instruct": ModelConfig(
            provider="replicate",
            model="meta/meta-llama-3-8b-instruct",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.7,
            top_p=0.95,
            top_k=0,
            max_tokens=4096,
            extra_inputs={
                "system_prompt": "You are a helpful assistant",
                "prompt_template": (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                ),
                "length_penalty": 1,
                "presence_penalty": 0,
                "max_new_tokens": 512,
                "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
                "log_performance_metrics": False,
            },
        ),
        "llama3-8b": ModelConfig(
            provider="replicate",
            model="meta/meta-llama-3-8b",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6,
            top_p=0.9,
            top_k=0,
            max_tokens=4096,
            extra_inputs={
                "system_prompt": "You are a helpful assistant that returns STRICT JSON per schema.",
                "prompt_template": "{system_prompt}\n\n{prompt}",
                "stop_sequences": ["<|end_of_text|>"],
                "length_penalty": 1,
                "presence_penalty": 0,
            },
        ),
        "llama3-70b": ModelConfig(
            provider="replicate",
            model="meta/meta-llama-3-70b",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6,
            top_p=0.9,
            top_k=0,
            max_tokens=4096,
            extra_inputs={
                "system_prompt": "You are a helpful assistant that returns STRICT JSON per schema.",
                "prompt_template": "{system_prompt}\n\n{prompt}",
                "stop_sequences": ["<|end_of_text|>"],
                "length_penalty": 1,
                "presence_penalty": 0,
            },
        ),
        "llama405b": ModelConfig(
            provider="replicate",
            model="meta/meta-llama-3.1-405b-instruct",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            max_tokens=4096,
            extra_inputs={
                "system_prompt": "You are a helpful assistant.",
                "prompt_template": "",
            },
        ),
        "mistral7b": ModelConfig(
            provider="replicate",
            model="mistralai/mistral-7b-instruct",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            max_tokens=4096,
            extra_inputs={
                "system_prompt": "You are a helpful assistant.",
                "prompt_template": "",
            },
        ),
        "mixtral8x7b": ModelConfig(
            provider="replicate",
            model="mistralai/mixtral-8x7b-instruct",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            max_tokens=4096,
            extra_inputs={
                "system_prompt": "You are a helpful assistant.",
                "prompt_template": "",
            },
        ),
        "gemini-flash": ModelConfig(
            provider="replicate",
            model="google/gemini-2.5-flash",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.0,
            top_p=0.9,
            max_tokens=4096,
            extra_inputs={
                "prefer": "prompt",
                "dynamic_thinking": False,
            },
        ),
        "grok4": ModelConfig(
            provider="replicate",
            model="xai/grok-4",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.1,
            top_p=1.0,
            max_tokens=2048,
            extra_inputs={
                "system_prompt": "You are a helpful assistant.",
                "prompt_template": "",
            },
        ),
        "claude35h": ModelConfig(
            provider="replicate",
            model="anthropic/claude-3.5-haiku",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.3,
            top_p=0.9,
            max_tokens=8192,
            extra_inputs={
                "system_prompt": "You are a concise and creative assistant.",
                "prompt_template": "",
            },
        ),
        "claude37s": ModelConfig(
            provider="replicate",
            model="anthropic/claude-3.7-sonnet",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.0,
            top_p=0.9,
            max_tokens=8192,
            extra_inputs={
                "extended_thinking": False,
                "max_image_resolution": 0.5,
                "thinking_budget_tokens": 1024,
                "system_prompt": "Return ONLY strict JSON; no prose; no fences.",
            },
        ),
        "granite8b": ModelConfig(
            provider="replicate",
            model="ibm-granite/granite-3.3-8b-instruct",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            max_tokens=4096,
            extra_inputs={
                "system_prompt": "Return ONLY strict JSON that validates against the provided schema.",
            },
        ),
        "gpt-oss-20b": ModelConfig(
            provider="replicate",
            model="openai/gpt-oss-20b",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.1,
            top_p=1.0,
            max_tokens=4096,
        ),
        "gpt-oss-120b": ModelConfig(
            provider="replicate",
            model="openai/gpt-oss-120b",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.1,
            top_p=1.0,
            max_tokens=4096,
        ),
        "qwen3-235b": ModelConfig(
            provider="replicate",
            model="qwen/qwen3-235b-a22b-instruct-2507",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.3,
            top_p=0.9,
            max_tokens=1536,
            extra_inputs={
                "system_prompt": "Return ONLY strict JSON per schema; no prose; no fences.",
            },
        ),
        "granite20b": ModelConfig(
            provider="replicate",
            model="ibm-granite/granite-20b-code-instruct-8k",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            max_tokens=512,
            extra_inputs={
                "system_prompt": "",
                "prompt_template": "",
            },
        ),

        # -------- Local via Unsloth (optional) --------
        "smollm2-1.7b": ModelConfig(
            provider="unsloth",
            model="unsloth/SmolLM2-1.7B-Instruct-bnb-4bit",
            api_key_env=None,
            temperature=0.0,
            top_p=0.95,
            top_k=40,
            max_tokens=800,
            extra_inputs={
                "max_seq_length": 2048,
                "load_in_4bit": False,
                "dtype": "float16",
                "device": "mps",
            },
        ),
        "smollm2-360m": ModelConfig(
            provider="unsloth",
            model="unsloth/SmolLM2-360M-Instruct-bnb-4bit",
            api_key_env=None,
            temperature=0.0,
            top_p=0.95,
            top_k=40,
            max_tokens=512,
            extra_inputs={
                "max_seq_length": 2048,
                "load_in_4bit": True,
            },
        ),
    }

    ELICIT_MODEL_KEY: str = "gpt4o-mini"
    NER_MODEL_KEY: str = "gpt4o-mini"
    ARTICLE_MODEL_KEY: str = "gpt4o-mini"
    ARTICLE_NER_MODEL_KEY: str = "gpt4o-mini"

    TOKEN_PRICES: Dict[str, Dict[str, float]] = OPENAI_TOKEN_PRICES


settings = Settings()



