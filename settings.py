from __future__ import annotations
from typing import Dict
from pydantic import BaseModel
from llm.config import ModelConfig

# ---------- JSON Schemas ----------

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
          "subject":   {"type": "string"},
          "predicate": {"type": "string"},
          "object":    {"type": "string"}
        },
        "required": ["subject", "predicate", "object"]
      }
    }
  },
  "required": ["facts"]
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
          "subject":    {"type": "string"},
          "predicate":  {"type": "string"},
          "object":     {"type": "string"},
          "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["subject", "predicate", "object", "confidence"]
      }
    }
  },
  "required": ["facts"]
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
          "is_ne":  {"type": "boolean"}
        },
        "required": ["phrase", "is_ne"]
      }
    }
  },
  "required": ["phrases"]
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
          "phrase":     {"type": "string"},
          "is_ne":      {"type": "boolean"},
          "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["phrase", "is_ne", "confidence"]
      }
    }
  },
  "required": ["phrases"]
}

# ---------- SQLite DDL ----------

QUEUE_DDL = """
CREATE TABLE IF NOT EXISTS queue(
  subject        TEXT NOT NULL,
  subject_norm   TEXT NOT NULL,
  subject_canon  TEXT NOT NULL DEFAULT '',
  hop            INT  NOT NULL DEFAULT 0,
  status         TEXT NOT NULL DEFAULT 'pending',
  retries        INT  NOT NULL DEFAULT 0,
  created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

FACTS_DDL = """
CREATE TABLE IF NOT EXISTS triples_accepted(
  subject     TEXT, 
  predicate   TEXT, 
  object      TEXT,
  hop         INT, 
  model_name  TEXT, 
  strategy    TEXT, 
  confidence  REAL,
  PRIMARY KEY(subject, predicate, object, hop)
);

CREATE TABLE IF NOT EXISTS triples_sink(
  subject     TEXT, 
  predicate   TEXT, 
  object      TEXT,
  hop         INT, 
  model_name  TEXT, 
  strategy    TEXT, 
  confidence  REAL, 
  reason      TEXT
);
"""

# ---------- Settings ----------

class Settings(BaseModel):
    CONCURRENCY: int = 8
    MAX_DEPTH: int = 2
    NER_BATCH_SIZE: int = 50
    MAX_FACTS_HINT: int = 50

    MODELS: Dict[str, ModelConfig] = {
        # -------- OpenAI (Chat Completions) --------
        "gpt4o": ModelConfig(
            provider="openai", model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0, top_p=1.0, max_tokens=4096,
            use_responses_api=False
        ),
        "gpt4o-mini": ModelConfig(
            provider="openai", model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0, top_p=1.0, max_tokens=4096,
            use_responses_api=False
        ),
        "gpt4-turbo": ModelConfig(
            provider="openai", model="gpt-4-turbo",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0, top_p=1.0, max_tokens=4096,
            use_responses_api=False
        ),

        # -------- OpenAI (Responses API) — GPT-5 family --------
        "gpt-5": ModelConfig(
            provider="openai",
            model="gpt-5",
            api_key_env="OPENAI_API_KEY",
            temperature=None, top_p=None, max_tokens=4096,
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
            temperature=None, top_p=None, max_tokens=4096,
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
            use_responses_api=True,
            extra_inputs={
                "reasoning": {"effort": "minimal"},
                "text": {"verbosity": "low"},
            },
            max_tokens=4096,
        ),

        # -------- DeepSeek --------
        "deepseek": ModelConfig(
            provider="deepseek", model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            temperature=0.0, top_p=0.95, max_tokens=4096
        ),
        "deepseek-reasoner": ModelConfig(
            provider="deepseek", model="deepseek-reasoner",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            temperature=0.0, top_p=0.95, max_tokens=4096
        ),

        # -------- Replicate (various) --------

        # Add these inside Settings.MODELS in settings.py

        # ------- Replicate (Meta Llama-3 70B Instruct) -------
        "llama3-70b-instruct": ModelConfig(
            provider="replicate",
            model="meta/meta-llama-3-70b-instruct",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6,
            top_p=0.9,
            top_k=0,
            max_tokens=4096,  # you can lower per-run; example snippet used 512
            extra_inputs={
                # keep this EXACTLY — your pipeline will pass {system_prompt} and {prompt}
                "system_prompt": "You are a helpful assistant",
                "prompt_template": (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                ),
                "length_penalty": 1,
                "presence_penalty": 1.15,
                # Replicate runners typically prefer list form; your example showed a CSV string —
                # we include both-friendly variant; your factory should map to what the runner expects.
                "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
                "log_performance_metrics": False,
            },
        ),

        # ------- Replicate (Meta Llama-3 8B Instruct) -------
        "llama3-8b-instruct": ModelConfig(
            provider="replicate",
            model="meta/meta-llama-3-8b-instruct",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.7,
            top_p=0.95,
            top_k=0,
            max_tokens=4096,  # example used 512; keep 4096 default and cap per-run if needed
            extra_inputs={
                # keep this EXACTLY — note this template uses {system_prompt} placeholder too
                "system_prompt": "You are a helpful assistant",
                "prompt_template": (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                ),
                "length_penalty": 1,
                "presence_penalty": 0,
                "max_new_tokens": 512,  # optional; some runners accept both max_tokens + max_new_tokens
                "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
                "log_performance_metrics": False,
            },
        ),
        # ------- Replicate (Meta Llama-3 8B — BASE, non-instruct) -------
"llama3-8b": ModelConfig(
    provider="replicate",
    model="meta/meta-llama-3-8b",           # BASE model (no -instruct)
    api_key_env="REPLICATE_API_TOKEN",
    temperature=0.6,
    top_p=0.9,
    top_k=0,
    max_tokens=4096,
    extra_inputs={
        # Your factory should format: prompt_template.format(system_prompt=..., prompt=...)
        # For BASE models we emulate chat by concatenating system + user.
        "system_prompt": "You are a helpful assistant that returns STRICT JSON per schema.",
        "prompt_template": "{system_prompt}\n\n{prompt}",
        # Runners typically accept list or string for stop; list is safer:
        "stop_sequences": ["<|end_of_text|>"],
        "length_penalty": 1,
        "presence_penalty": 0,
        "log_performance_metrics": False,
    },
),

# ------- Replicate (Meta Llama-3 70B — BASE, non-instruct) -------
"llama3-70b": ModelConfig(
    provider="replicate",
    model="meta/meta-llama-3-70b",          # BASE model (no -instruct)
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
        "log_performance_metrics": False,
    },
),

        "llama405b": ModelConfig(
            provider="replicate", model="meta/meta-llama-3.1-405b-instruct",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6, top_p=0.9, top_k=50, max_tokens=4096,
            extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
        ),
        "mistral7b": ModelConfig(
            provider="replicate", model="mistralai/mistral-7b-instruct",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6, top_p=0.95, top_k=50, max_tokens=4096,
            extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
        ),
        "mixtral8x7b": ModelConfig(
            provider="replicate", model="mistralai/mixtral-8x7b-instruct",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6, top_p=0.95, top_k=50, max_tokens=4096,
            extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
        ),

        "gemini-flash": ModelConfig(
            provider="replicate",
            model="google/gemini-2.5-flash",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.2,
            top_p=0.9,
            max_tokens=4096,
            extra_inputs={"prefer": "prompt", "dynamic_thinking": False},
        ),
        "grok4": ModelConfig(
            provider="replicate",
            model="xai/grok-4",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.1, top_p=1.0, max_tokens=2048,
            extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""},
        ),
        "claude35h": ModelConfig(
            provider="replicate",
            model="anthropic/claude-3.5-haiku",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.3, top_p=0.9, max_tokens=8192,
            extra_inputs={"system_prompt": "You are a concise and creative assistant.", "prompt_template": ""},
        ),
        "claude37s": ModelConfig(
            provider="replicate",
            model="anthropic/claude-3.7-sonnet",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.2, top_p=0.9, max_tokens=8192,
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
            temperature=0.6, top_p=0.9, top_k=50, max_tokens=4096,
            extra_inputs={
                "system_prompt": "Return ONLY strict JSON that validates against the provided schema.",
            },
        ),
        "gpt-oss-20b": ModelConfig(
            provider="replicate",
            model="openai/gpt-oss-20b",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.1, top_p=1.0, max_tokens=4096,
        ),
        "gpt-oss-120b": ModelConfig(
    provider="replicate",
    model="openai/gpt-oss-120b",
    api_key_env="REPLICATE_API_TOKEN",
    temperature=0.1, top_p=1.0, max_tokens=4096,
),

        "qwen3-235b": ModelConfig(
            provider="replicate",
            model="qwen/qwen3-235b-a22b-instruct-2507",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.3, top_p=0.9, max_tokens=1536,
            extra_inputs={"system_prompt": "Return ONLY strict JSON per schema; no prose; no fences."},
        ),

        "granite20b": ModelConfig(
            provider="replicate",
            model="ibm-granite/granite-20b-code-instruct-8k",
            api_key_env="REPLICATE_API_TOKEN",
            temperature=0.6, top_p=0.9, top_k=50, max_tokens=512,
            extra_inputs={"system_prompt": "", "prompt_template": ""},
        ),

        # -------- Local via Unsloth (optional) --------
        "smollm2-1.7b": ModelConfig(
            provider="unsloth",
            model="unsloth/SmolLM2-1.7B-Instruct-bnb-4bit",
            api_key_env=None,
            temperature=0.2, top_p=0.95, top_k=40, max_tokens=800,
            extra_inputs={"max_seq_length": 2048, "load_in_4bit": False, "dtype": "float16", "device": "mps"},
        ),
        "smollm2-360m": ModelConfig(
            provider="unsloth",
            model="unsloth/SmolLM2-360M-Instruct-bnb-4bit",
            api_key_env=None,
            temperature=0.2, top_p=0.95, top_k=40, max_tokens=512,
            extra_inputs={"max_seq_length": 2048, "load_in_4bit": True},
        ),
    }

    # defaults; override via CLI
    ELICIT_MODEL_KEY: str = "gpt4o-mini"
    NER_MODEL_KEY: str = "gpt4o-mini"

settings = Settings()




