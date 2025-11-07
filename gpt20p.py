from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("ANTHROPIC_API_KEY"))

from llm.anthropic_client import AnthropicLLM

client = AnthropicLLM()  # loads ANTHROPIC_API_KEY from .env

msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me 3 facts about Saturn."},
]

print("=== plain text call ===")
out = client(msgs, model="claude-sonnet-4-5-20250929", max_tokens=400)
print(out.get("text") if isinstance(out, dict) else out)

print("\n=== reasoning/thinking call ===")
think = {"type": "enabled", "budget_tokens": 1024}
out2 = client(msgs, model="claude-sonnet-4-5-20250929", max_tokens=2000, reasoning=think)
print(out2.get("text") if isinstance(out2, dict) else out2)


