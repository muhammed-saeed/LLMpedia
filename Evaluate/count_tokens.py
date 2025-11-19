import json
from pathlib import Path

import tiktoken

# ---------- FILE PATHS (YOUR EXACT ONES) ----------

ARTICLES_PATH = Path(
    "/Users/muhammedsaeed/Downloads/TU Dresden/LLMpedia/runsTest1/"
    "AncientCityofBabylonbatch_calibrate_no_selfrag/articles.jsonl"
)

ELICITATION_PATH = Path(
    "/Users/muhammedsaeed/Downloads/TU Dresden/LLMpedia/prompts/topic/baseline/elicitation.json"
)

NER_PATH = Path(
    "/Users/muhammedsaeed/Downloads/TU Dresden/LLMpedia/prompts/topic/baseline/ner.json"
)

# Choose the model whose tokenization you care about for cost
MODEL_NAME = "gpt-4o-mini"  # change if you want another model


# ---------- TOKENIZER SETUP ----------

try:
    encoding = tiktoken.encoding_for_model(MODEL_NAME)
except KeyError:
    # Fallback for newer/unknown models
    encoding = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in a text string."""
    return len(encoding.encode(text))


# ---------- HELPERS ----------

def iter_json_strings(obj):
    """
    Recursively yield all string values inside a JSON object
    (dicts, lists, nested structures, etc.).
    """
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_json_strings(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_json_strings(item)
    # ignore other primitive types


def count_wikitext_tokens_from_jsonl(path: Path) -> int:
    """
    Read a .jsonl file where each line is a JSON object
    and sum tokens only from the 'wikitext' field.
    """
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            wikitext = record.get("wikitext", "")
            if isinstance(wikitext, str):
                total += count_tokens(wikitext)
    return total


def count_tokens_in_json_file(path: Path) -> int:
    """
    Load a JSON file and count tokens in ALL string fields.
    This will include user/system message texts if that's
    how they're stored.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    for s in iter_json_strings(data):
        total += count_tokens(s)
    return total


# ---------- MAIN ----------

def main():
    wiki_tokens = count_wikitext_tokens_from_jsonl(ARTICLES_PATH)
    elicitation_tokens = count_tokens_in_json_file(ELICITATION_PATH)
    ner_tokens = count_tokens_in_json_file(NER_PATH)

    grand_total = wiki_tokens + elicitation_tokens + ner_tokens

    print("Token counts (model: {})".format(MODEL_NAME))
    print("---------------------------------")
    print(f"articles.jsonl (wikitext only): {wiki_tokens}")
    print(f"elicitation.json (all strings): {elicitation_tokens}")
    print(f"ner.json (all strings):         {ner_tokens}")
    print("---------------------------------")
    print(f"TOTAL tokens:                   {grand_total}")


if __name__ == "__main__":
    main()
