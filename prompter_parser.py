# # prompter_parser.py
# from __future__ import annotations

# import json
# import os
# import re
# from typing import Dict, List, Any

# # -----------------------------------------------------------------------------
# # Configuration
# # -----------------------------------------------------------------------------

# # Base directory for prompt files (you can change to an absolute path if needed)
# PROMPTS_BASE_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# # Allowed placeholders you may use in prompt JSON files:
# # Add more here as your prompts grow.
# _ALLOWED_KEYS = {
#     "subject_name",
#     "phrases_block",
#     "root_subject",
#     "root_topic",  # <-- NEW: support {root_topic}
#     "max_facts_hint",
#     "conf_threshold",
#     "min_sections",
#     "max_sections",
#     "avg_words_per_article",
#     "article_block",
#     "triples_block",
#     "web_context",
# }

# # -----------------------------------------------------------------------------
# # Utilities
# # -----------------------------------------------------------------------------

# _PLACEHOLDER_RX = re.compile(r"\{([a-zA-Z0-9_]+)\}")

# def _render_placeholders(template: str, variables: Dict[str, Any], *,
#                          allow_missing: bool = True) -> str:
#     """
#     Replace {placeholder} with variables[placeholder] (converted to str).
#     Only replaces names in _ALLOWED_KEYS.
#     If allow_missing=False, raises KeyError on a missing placeholder that is allowed.
#     Placeholders not in _ALLOWED_KEYS are left as-is (so JSON braces remain safe).
#     """
#     def repl(m: re.Match) -> str:
#         name = m.group(1)
#         if name not in _ALLOWED_KEYS:
#             # Not a managed placeholder → leave as-is (so JSON { } stay intact)
#             return m.group(0)
#         if name not in variables:
#             if allow_missing:
#                 # Leave placeholder as-is (useful when the same prompt works for both domains)
#                 return m.group(0)
#             raise KeyError(f"Missing variable for placeholder: {{{name}}}")
#         val = variables[name]
#         return str(val)

#     return _PLACEHOLDER_RX.sub(repl, template)


# def _load_json(path: str) -> Dict[str, Any]:
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Prompt file not found: {path}")
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except json.JSONDecodeError as e:
#         raise ValueError(f"Invalid JSON in prompt file: {path}\n{e}") from e
#     if not isinstance(data, dict):
#         raise ValueError(f"Prompt file must contain a JSON object with 'system' and 'user' keys: {path}")
#     if "system" not in data or "user" not in data:
#         raise ValueError(f"Prompt file missing 'system' or 'user' keys: {path}")
#     if not isinstance(data["system"], str) or not isinstance(data["user"], str):
#         raise ValueError(f"'system' and 'user' must be strings in: {path}")
#     return data


# def _prompt_path(base_dir: str, domain: str, strategy: str, ptype: str) -> str:
#     """
#     Build path: <base_dir>/<domain>/<strategy>/<ptype>.json
#     where ptype is 'elicitation', 'ner', or 'encyclopedia' (case-insensitive).
#     """
#     ptype = ptype.lower()
#     if ptype not in {"elicitation", "ner", "encyclopedia"}:
#         raise ValueError(f"ptype must be 'elicitation', 'ner', or 'encyclopedia', got: {ptype}")
#     return os.path.join(base_dir, domain, strategy, f"{ptype}.json")


# # -----------------------------------------------------------------------------
# # Public API
# # -----------------------------------------------------------------------------

# def get_prompt_messages(
#     strategy: str,
#     ptype: str,
#     *,
#     domain: str = "topic",
#     variables: Dict[str, Any] | None = None,
#     base_dir: str = PROMPTS_BASE_DIR,
#     allow_missing_vars: bool = True,
# ) -> List[Dict[str, str]]:
#     """
#     Load and render a prompt as messages for an LLM.

#     - strategy: one of "baseline", "calibrate", "dont_know", "icl" (free-form)
#     - ptype: "elicitation", "ner", or "encyclopedia"
#     - domain: "topic" or "general" (forms the folder path)
#     - variables: mapping used to fill placeholders like {subject_name}, {root_subject}, {root_topic}, ...
#     - base_dir: root folder (default ./prompts)
#     - allow_missing_vars: if False, raises when an allowed placeholder is present but not passed

#     Returns:
#       [
#         {"role": "system", "content": "..."},
#         {"role": "user",   "content": "..."}
#       ]
#     """
#     variables = variables or {}
#     path = _prompt_path(base_dir, domain, strategy, ptype)
#     raw = _load_json(path)

#     system_txt = _render_placeholders(raw["system"], variables, allow_missing=allow_missing_vars)
#     user_txt   = _render_placeholders(raw["user"],   variables, allow_missing=allow_missing_vars)

#     return [
#         {"role": "system", "content": system_txt},
#         {"role": "user",   "content": user_txt},
#     ]


# def build_elicitation_messages_for_subject(
#     *,
#     domain: str,
#     strategy: str,
#     subject_name: str,
#     seed: str | None = None,
#     root_topic: str | None = None,          # <-- NEW ARG
#     conf_threshold: float | None = None,
#     max_facts_hint: int | None = None,
#     min_sections: int | None = None,
#     max_sections: int | None = None,
#     avg_words_per_article: int | None = None,
#     web_context: str | None = None,
#     base_dir: str = PROMPTS_BASE_DIR,
#     allow_missing_vars: bool = True,
# ) -> List[Dict[str, str]]:
#     """
#     Convenience builder for ELICITATION messages that handles domain differences:

#     - domain == "topic": root topic is usually the global seed (the main topic).
#     - domain == "general": root_topic defaults to the current subject unless overridden.

#     Both {root_subject} and {root_topic} are supported placeholders and get the same value.
#     """

#     variables: Dict[str, Any] = {
#         "subject_name": subject_name,
#     }

#     # Decide root topic value
#     # Priority: explicit root_topic arg > seed > subject_name
#     topic_val: str | None = root_topic or seed or subject_name

#     if topic_val:
#         # For topic domain we keep the historical name root_subject,
#         # but we also expose root_topic for the new prompts.
#         variables["root_subject"] = topic_val
#         variables["root_topic"] = topic_val

#     if conf_threshold is not None:
#         variables["conf_threshold"] = f"{conf_threshold:.2f}"
#     if max_facts_hint is not None:
#         variables["max_facts_hint"] = str(max_facts_hint)
#     if min_sections is not None:
#         variables["min_sections"] = str(min_sections)
#     if max_sections is not None:
#         variables["max_sections"] = str(max_sections)
#     if avg_words_per_article is not None:
#         variables["avg_words_per_article"] = str(avg_words_per_article)
#     if web_context:
#         variables["web_context"] = web_context

#     return get_prompt_messages(
#         strategy=strategy,
#         ptype="elicitation",
#         domain=domain,
#         variables=variables,
#         base_dir=base_dir,
#         allow_missing_vars=allow_missing_vars,
#     )


# def build_ner_messages_for_phrases(
#     *,
#     domain: str,
#     strategy: str,
#     subject_name: str,
#     seed: str | None,
#     phrases: List[str],
#     root_topic: str | None = None,          # <-- optional, consistent with elicitation
#     base_dir: str = PROMPTS_BASE_DIR,
#     allow_missing_vars: bool = True,
# ) -> List[Dict[str, str]]:
#     """
#     Convenience builder for NER messages (category-based NER):

#     - domain == "topic": injects a shared root topic (seed, unless overridden).
#     - domain == "general": root_topic defaults to the current subject unless overridden.

#     Both {root_subject} and {root_topic} placeholders are supported.
#     """
#     variables: Dict[str, Any] = {
#         "subject_name": subject_name,
#         "phrases_block": "\n".join(phrases or []),
#     }

#     topic_val: str | None = root_topic or seed or subject_name

#     if topic_val:
#         # Keep root_subject for backwards-compatible prompts,
#         # but also support the new root_topic name.
#         variables["root_subject"] = topic_val
#         variables["root_topic"] = topic_val

#     return get_prompt_messages(
#         strategy=strategy,
#         ptype="ner",
#         domain=domain,
#         variables=variables,
#         base_dir=base_dir,
#         allow_missing_vars=allow_missing_vars,
#     )

# prompter_parser.py
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Any

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Base directory for prompt files (you can change to an absolute path if needed)
PROMPTS_BASE_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Allowed placeholders you may use in prompt JSON files:
_ALLOWED_KEYS = {
    "subject_name",
    "phrases_block",
    "root_subject",
    "root_topic",  # support {root_topic}
    "max_facts_hint",
    "conf_threshold",
    "min_sections",
    "max_sections",
    "avg_words_per_article",
    "article_block",
    "triples_block",
    "web_context",
    "persona_block",  # persona placeholder
}

_PLACEHOLDER_RX = re.compile(r"\{([a-zA-Z0-9_]+)\}")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _render_placeholders(
    template: str,
    variables: Dict[str, Any],
    *,
    allow_missing: bool = True,
) -> str:
    """
    Replace {placeholder} with variables[placeholder] (converted to str).
    Only replaces names in _ALLOWED_KEYS.
    Placeholders not in _ALLOWED_KEYS are left as-is.
    """

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in _ALLOWED_KEYS:
            # Not a managed placeholder → leave as-is (e.g. JSON braces)
            return m.group(0)
        if name not in variables:
            if allow_missing:
                return m.group(0)
            raise KeyError(f"Missing variable for placeholder: {{{name}}}")
        return str(variables[name])

    return _PLACEHOLDER_RX.sub(repl, template)


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in prompt file: {path}\n{e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file must contain a JSON object with 'system' and 'user' keys: {path}")
    if "system" not in data or "user" not in data:
        raise ValueError(f"Prompt file missing 'system' or 'user' keys: {path}")
    if not isinstance(data["system"], str) or not isinstance(data["user"], str):
        raise ValueError(f"'system' and 'user' must be strings in: {path}")
    return data


def _prompt_path(base_dir: str, domain: str, strategy: str, ptype: str) -> str:
    """
    Build path: <base_dir>/<domain>/<strategy>/<ptype>.json
    where ptype is 'elicitation', 'ner', or 'encyclopedia'.
    """
    ptype = ptype.lower()
    if ptype not in {"elicitation", "ner", "encyclopedia"}:
        raise ValueError(f"ptype must be 'elicitation', 'ner', or 'encyclopedia', got: {ptype}")
    return os.path.join(base_dir, domain, strategy, f"{ptype}.json")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_prompt_messages(
    strategy: str,
    ptype: str,
    *,
    domain: str = "topic",
    variables: Dict[str, Any] | None = None,
    base_dir: str = PROMPTS_BASE_DIR,
    allow_missing_vars: bool = True,
) -> List[Dict[str, str]]:
    """
    Load and render a prompt as messages for an LLM.
    """
    variables = variables or {}
    path = _prompt_path(base_dir, domain, strategy, ptype)
    raw = _load_json(path)

    system_txt = _render_placeholders(raw["system"], variables, allow_missing=allow_missing_vars)
    user_txt   = _render_placeholders(raw["user"],   variables, allow_missing=allow_missing_vars)

    return [
        {"role": "system", "content": system_txt},
        {"role": "user",   "content": user_txt},
    ]


def build_elicitation_messages_for_subject(
    *,
    domain: str,
    strategy: str,
    subject_name: str,
    seed: str | None = None,
    root_topic: str | None = None,
    conf_threshold: float | None = None,
    max_facts_hint: int | None = None,
    min_sections: int | None = None,
    max_sections: int | None = None,
    avg_words_per_article: int | None = None,
    web_context: str | None = None,
    persona_block: str | None = None,   # persona input
    base_dir: str = PROMPTS_BASE_DIR,
    allow_missing_vars: bool = True,
) -> List[Dict[str, str]]:
    """
    Convenience builder for ELICITATION messages that handles domain differences
    and injects persona text when provided.
    """

    variables: Dict[str, Any] = {
        "subject_name": subject_name,
    }

    # root topic
    topic_val: str | None = root_topic or seed or subject_name
    if topic_val:
        variables["root_subject"] = topic_val
        variables["root_topic"] = topic_val

    if conf_threshold is not None:
        variables["conf_threshold"] = f"{conf_threshold:.2f}"
    if max_facts_hint is not None:
        variables["max_facts_hint"] = str(max_facts_hint)
    if min_sections is not None:
        variables["min_sections"] = str(min_sections)
    if max_sections is not None:
        variables["max_sections"] = str(max_sections)
    if avg_words_per_article is not None:
        variables["avg_words_per_article"] = str(avg_words_per_article)
    if web_context:
        variables["web_context"] = web_context
    # IMPORTANT: even empty string overwrites {persona_block}
    if persona_block is not None:
        variables["persona_block"] = persona_block

    return get_prompt_messages(
        strategy=strategy,
        ptype="elicitation",
        domain=domain,
        variables=variables,
        base_dir=base_dir,
        allow_missing_vars=allow_missing_vars,
    )


def build_ner_messages_for_phrases(
    *,
    domain: str,
    strategy: str,
    subject_name: str,
    seed: str | None,
    phrases: List[str],
    root_topic: str | None = None,
    persona_block: str | None = None,   # persona input for NER
    base_dir: str = PROMPTS_BASE_DIR,
    allow_missing_vars: bool = True,
) -> List[Dict[str, str]]:
    """
    Convenience builder for NER messages with optional persona support.
    """

    variables: Dict[str, Any] = {
        "subject_name": subject_name,
        "phrases_block": "\n".join(phrases or []),
    }

    topic_val: str | None = root_topic or seed or subject_name
    if topic_val:
        variables["root_subject"] = topic_val
        variables["root_topic"] = topic_val
    if persona_block is not None:
        variables["persona_block"] = persona_block

    return get_prompt_messages(
        strategy=strategy,
        ptype="ner",
        domain=domain,
        variables=variables,
        base_dir=base_dir,
        allow_missing_vars=allow_missing_vars,
    )
