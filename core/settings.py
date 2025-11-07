from __future__ import annotations
from llm.config import ModelConfig

# Choose your default provider/model here. You can switch per script/run.
# OpenAI example (Responses or Chat Completions handled internally by your clients):
OPENAI_GENERAL = ModelConfig(
    provider="openai",
    model="gpt-4o-mini",          # or "gpt-5-nano" if you want Responses API automatically
    api_key_env="OPENAI_API_KEY",
    base_url=None,                 # or a compatible gateway
    temperature=0.0,
    top_p=1.0,
    max_tokens=4096,
    use_responses_api=False,       # True auto for gpt-5* via your OpenAIClient anyway
    extra_inputs=None
)

# DeepSeek example:
DEEPSEEK_GENERAL = ModelConfig(
    provider="deepseek",
    model="deepseek-chat",
    api_key_env="DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com",
    temperature=0.2,
    top_p=1.0,
    max_tokens=4096
)

# Replicate example (adjust model slug as needed)
REPLICATE_GENERAL = ModelConfig(
    provider="replicate",
    model="meta/meta-llama-3-8b-instruct",
    api_key_env=None,
    temperature=0.2,
    top_p=0.9,
    max_tokens=2048
)

# Unsloth (local) example
UNSLOTH_LOCAL = ModelConfig(
    provider="unsloth",
    model="unsloth/Meta-Llama-3-8B-Instruct",
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    extra_inputs={
        "max_seq_length": 4096,
        "dtype": "float16",      # or "bfloat16"
        "load_in_4bit": False,   # set True if CUDA + bitsandbytes
        "device": None
    }
)
