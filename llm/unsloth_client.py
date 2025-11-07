# llm/unsloth_client.py
from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional

# --- Dependency gate with helpful error message --------------------------------
try:
    import torch
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer  # noqa: imported for side-effects / tokenizer consistency
except Exception as e:
    raise ImportError(
        "Unsloth backend not available.\n"
        "Install:\n"
        "  pip install -U unsloth unsloth_zoo transformers accelerate safetensors\n"
        "If you have an NVIDIA GPU (CUDA):\n"
        "  pip install bitsandbytes  &&  install a CUDA build of torch\n"
        f"\nOriginal import error: {e}"
    )

# --- Helpers -------------------------------------------------------------------

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _pick_device() -> str:
    """
    Choose device with environment override:
      export UNSLOTH_DEVICE={cuda|mps|cpu}
    """
    env = (os.getenv("UNSLOTH_DEVICE") or "").strip().lower()
    if env in {"cuda", "mps", "cpu"}:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_torch_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    """
    Map string dtype names to torch dtypes. None -> auto.
    Accepts: "float16", "bfloat16", "float32"
    """
    if name is None:
        return None
    name = str(name).lower()
    if name in {"float16", "fp16", "f16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32", "f32"}:
        return torch.float32
    # fallback: ignore unknown and let Unsloth decide
    return None


def _chat_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Convert OpenAI-like chat messages into a single instruction prompt
    that works well with *-Instruct local models.
    """
    sys_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    user_parts = [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]
    sys_txt = ("\n".join(sys_parts)).strip()
    usr_txt = ("\n\n".join(user_parts)).strip()

    if sys_txt:
        return (
            "Below is a system rule and an instruction. Follow the system rule strictly.\n\n"
            f"### System:\n{sys_txt}\n\n"
            f"### Instruction:\n{usr_txt}\n\n"
            "### Response:\n"
        )
    else:
        return (
            "Below is an instruction. Follow it strictly.\n\n"
            f"### Instruction:\n{usr_txt}\n\n"
            "### Response:\n"
        )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON extractor for local model outputs:
    1) Prefer fenced ```json blocks
    2) Otherwise, try first balanced {...} region
    """
    m = JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Try first balanced { ... }
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    break
    return None


# --- Main client ----------------------------------------------------------------

class UnslothLLM:
    """
    Minimal local LLM wrapper using Unsloth + HF Transformers.

    Usage parity with your other backends:
      out = client.generate(messages, json_schema=..., temperature=..., top_p=..., top_k=..., max_tokens=..., seed=...)

    Notes for Apple Silicon (MPS):
      - Set load_in_4bit=False (bitsandbytes is CUDA-only)
      - Use dtype="float16" and device="mps" for best speed
    """

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[str] = None,        # "float16" | "bfloat16" | "float32" | None (auto)
        load_in_4bit: bool = True,          # CUDA only; set False on Mac/CPU
        device: Optional[str] = None,       # "cuda" | "mps" | "cpu" | None (auto)
        trust_remote_code: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.device = device or _pick_device()
        self.max_seq_length = max_seq_length
        self.load_in_4bit = bool(load_in_4bit)
        self.dtype = _to_torch_dtype(dtype)
        self.trust_remote_code = trust_remote_code
        self.extra = extra or {}

        # If not on CUDA, disable 4-bit to avoid bitsandbytes requirement.
        if self.device != "cuda" and self.load_in_4bit:
            self.load_in_4bit = False

        # Load model + tokenizer via Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,             # None => auto
            load_in_4bit=self.load_in_4bit,
            trust_remote_code=self.trust_remote_code,
        )
        FastLanguageModel.for_inference(self.model)  # enable fused kernels where available

        # Place model on device
        if self.device == "cuda":
            self.model = self.model.to("cuda")
        elif self.device == "mps":
            self.model = self.model.to("mps")
        else:
            self.model = self.model.to("cpu")

    def generate(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          - If json_schema is provided: a parsed dict (or {"_raw": "..."} if parsing failed)
          - Otherwise: {"text": "..."} with raw string
        """
        cfg_extra = extra or self.extra or {}
        gen_kwargs: Dict[str, Any] = dict(
            do_sample=(temperature and temperature > 0.0) or (top_p is not None and top_p < 1.0) or (top_k is not None),
            temperature=temperature if temperature is not None else 0.0,
            top_p=top_p if top_p is not None else 1.0,
            max_new_tokens=max_tokens if max_tokens is not None else 512,
            repetition_penalty=cfg_extra.get("repetition_penalty", 1.0),
        )
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        if seed is not None:
            try:
                torch.manual_seed(int(seed))
            except Exception:
                pass

        prompt = _chat_to_prompt(messages)

        # Strong nudge for JSON when schema requested
        if json_schema is not None:
            prompt += "\nReturn ONLY valid JSON. No prose, no code fences.\n"

        inputs = self.tokenizer([prompt], return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        elif self.device == "mps":
            # MPS: tensors need to be moved individually
            for k in inputs:
                inputs[k] = inputs[k].to("mps")

        # Generate (no streaming to keep API consistent with cloud backends)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Keep only the assistant segment after the response marker, if present
        if "### Response:" in text:
            text = text.split("### Response:", 1)[-1].strip()

        if json_schema is not None:
            parsed = _extract_json(text)
            if parsed is None:
                # Return raw output for debugging; caller can decide how to handle
                return {"_raw": text}
            return parsed

        return {"text": text}
