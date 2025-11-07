# llm/anthropic_client.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load .env so ANTHROPIC_API_KEY is available
load_dotenv()

try:
    import anthropic
except Exception:
    anthropic = None


class AnthropicLLM:
    """
    Minimal Anthropic wrapper with thinking constraints.

    - Reads ANTHROPIC_API_KEY from env (or pass api_key= explicitly).
    - Accepts messages like [{"role":"system","content":"..."}, {"role":"user","content":"..."}].
      We collect system messages into `system=` and pass the rest to `messages=`.
    - Extended thinking via thinking={"type":"enabled","budget_tokens":...} (alias: reasoning=).
      When thinking is enabled:
        * temperature is FORCED to 1 (non-overridable).
        * max_tokens is FORCED to be >= 1024.
        * if budget_tokens is provided, max_tokens is FORCED to be > budget_tokens.
    - Returns {"text": <string>, "_raw": <sdk_response>}.
    """

    def __init__(self, api_key: Optional[str] = None, *, max_retries: int = 3, debug: bool = False):
        if anthropic is None:
            raise ImportError("anthropic SDK not installed. Run: pip install anthropic python-dotenv")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY. Set it in your .env or pass api_key=.")

        ClientClass = getattr(anthropic, "Anthropic", None) or getattr(anthropic, "Client", None)
        if ClientClass is None:
            raise RuntimeError("Anthropic SDK missing Anthropic/Client class.")

        self.client = ClientClass(api_key=self.api_key)
        self.max_retries = max(1, int(max_retries))
        self.debug = bool(debug or os.getenv("ANTHROPIC_DEBUG") == "1")

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        return self.generate(messages, **kwargs)

    # ---------- helpers ----------

    def _log(self, *a):
        if self.debug:
            print("[AnthropicLLM]", *a, flush=True)

    @staticmethod
    def _split_system_and_dialog(messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        """
        Returns (system_text, dialog_messages_without_system).
        Concatenates multiple system messages with blank lines.
        """
        sys_parts: List[str] = []
        dialog: List[Dict[str, str]] = []
        for m in messages or []:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "")
            if role == "system":
                if content:
                    sys_parts.append(str(content))
            elif role in ("user", "assistant"):
                dialog.append({"role": role, "content": str(content)})
            else:
                dialog.append({"role": "user", "content": str(content)})
        return "\n\n".join(sys_parts).strip(), dialog

    @staticmethod
    def _extract_text(resp: Any) -> str:
        # Messages API: resp.content is list of blocks
        try:
            content = getattr(resp, "content", None)
            if isinstance(content, list):
                parts: List[str] = []
                for blk in content:
                    if hasattr(blk, "text"):
                        parts.append(str(getattr(blk, "text") or ""))
                    elif isinstance(blk, dict) and blk.get("type") == "text":
                        parts.append(str(blk.get("text") or ""))
                    elif isinstance(blk, str):
                        parts.append(blk)
                return " ".join(p for p in parts if p).strip()
            if isinstance(content, str):
                return content
        except Exception:
            pass

        # Legacy completion style
        try:
            comp = getattr(resp, "completion", None)
            if isinstance(comp, str):
                return comp
        except Exception:
            pass

        # Dict-like fallback
        if isinstance(resp, dict):
            c = resp.get("content")
            if isinstance(c, list):
                texts: List[str] = []
                for blk in c:
                    if isinstance(blk, dict) and "text" in blk:
                        texts.append(str(blk["text"] or ""))
                    elif isinstance(blk, str):
                        texts.append(blk)
                return " ".join(t for t in texts if t).strip()
            if isinstance(c, str):
                return c
            if isinstance(resp.get("completion"), str):
                return str(resp["completion"])

        try:
            return str(resp)
        except Exception:
            return ""

    # ---------- main call ----------

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: Optional[int] = 512,
        temperature: Optional[float] = 0.0,
        reasoning: Optional[Dict[str, Any]] = None,  # alias for thinking
        thinking: Optional[Dict[str, Any]] = None,
        **extra,
    ) -> Dict[str, Any]:
        """
        Calls anthropic.messages.create() with enforced constraints when thinking is enabled.
        """
        system_text, dialog = self._split_system_and_dialog(messages)

        thinking_payload = thinking or reasoning

        # ----- Enforce MUSTs when thinking is enabled -----
        if thinking_payload:
            # 1) Force temperature=1, not changeable
            if temperature != 1:
                self._log("forcing temperature=1 (thinking enabled)")
            temperature = 1

            # 2) Force max_tokens >= 1024
            if max_tokens is None or max_tokens < 1024:
                self._log(f"bumping max_tokens to >=1024 (was {max_tokens})")
                max_tokens = 1024

            # 3) Ensure max_tokens > thinking budget_tokens (if provided)
            budget = None
            if isinstance(thinking_payload, dict):
                budget = thinking_payload.get("budget_tokens")
            if isinstance(budget, (int, float)):
                budget = int(budget)
                if max_tokens <= budget:
                    new_max = budget + 1
                    self._log(f"bumping max_tokens to > budget ({budget}); setting max_tokens={new_max}")
                    max_tokens = new_max
        # --------------------------------------------------

        # Build call kwargs
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": int(max_tokens if max_tokens is not None else 512),
            "messages": dialog,
        }
        if system_text:
            call_kwargs["system"] = system_text

        if thinking_payload:
            call_kwargs["thinking"] = thinking_payload
            call_kwargs["temperature"] = 1  # double-assert
        else:
            if temperature is not None:
                call_kwargs["temperature"] = temperature

        # Pass through any supported extras (avoid overriding our enforced keys)
        for k, v in (extra or {}).items():
            if k in ("model", "messages", "system", "max_tokens", "temperature", "thinking", "reasoning"):
                continue
            if v is not None:
                call_kwargs[k] = v

        # Retry on transient errors
        last_err: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._log(
                    f"messages.create attempt={attempt} model={model} "
                    f"thinking={'yes' if thinking_payload else 'no'} "
                    f"temperature={call_kwargs.get('temperature')} max_tokens={call_kwargs.get('max_tokens')}"
                )
                resp = self.client.messages.create(**call_kwargs)
                text = self._extract_text(resp)
                return {"text": text, "_raw": resp}
            except Exception as e:
                last_err = e
                self._log(f"error: {type(e).__name__}: {e}")
                if attempt == self.max_retries:
                    break
                time.sleep(min(10.0, 0.6 * (2 ** (attempt - 1))))

        raise RuntimeError(f"Anthropic call failed after {self.max_retries} attempts: {last_err}")
