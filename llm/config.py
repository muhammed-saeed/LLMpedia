from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel

class ModelConfig(BaseModel):
    provider: str
    model: str
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    extra_inputs: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    use_responses_api: bool = False
