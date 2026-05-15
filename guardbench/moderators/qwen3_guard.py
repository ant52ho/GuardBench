"""Qwen3Guard-Gen moderation compatible with GuardBench.

Follows the Hugging Face model cards and README for message shapes and output lines
(`Safety:` / ``Categories:`` / ``Refusal:``), including OpenAI-compatible vLLM usage:

https://huggingface.co/Qwen/Qwen3Guard-Gen-4B/blob/main/README.md
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .base import BaseModerator, PreparedPrompt

Mode = Literal["user_query", "assistant_response"]
ControversialPolicy = Literal["safe", "unsafe", "unknown"]

# Defaults aligned with HF quickstart (short classification generations).
QWEN3_GUARD_OPENAI_GENERATION_DEFAULTS: dict[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 128,
}


@dataclass(frozen=True)
class Qwen3GuardParsed:
    """Structured parse of a Qwen3Guard completion."""

    safety: str | None
    """Lowercase ``safe`` / ``unsafe`` / ``controversial``, or ``None`` if missing."""

    categories: tuple[str, ...]
    refusal: str | None
    """Lowercase ``yes`` / ``no`` when present (response moderation)."""


def flatten_guardbench_conversation(conversation: list[dict[str, str]]) -> str:
    """Render GuardBench turns as ``Role: content`` lines."""
    return "\n".join(
        f"{turn['role'].strip().capitalize()}: {turn['content'].strip()}"
        for turn in conversation
    )


def qwen3_guard_messages_from_conversation(
    conversation: list[dict[str, str]],
    *,
    mode: Mode,
    placeholder_user_query: str,
) -> list[dict[str, str]]:
    """Build OpenAI-style ``messages`` per Qwen3Guard README.

    * **user_query** — ``[{"role": "user", "content": <flattened dialogue>}]`` (prompt-side).
    * **assistant_response** — last assistant turn plus prior context as the user message;
      if there is no assistant turn, falls back to the **user_query** shape.
    """
    if mode == "user_query":
        return [{"role": "user", "content": flatten_guardbench_conversation(conversation)}]

    last_ai: int | None = None
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i]["role"].strip().lower() == "assistant":
            last_ai = i
            break

    if last_ai is None:
        return [{"role": "user", "content": flatten_guardbench_conversation(conversation)}]

    prefix = conversation[:last_ai]
    assistant_turn = conversation[last_ai]["content"].strip()
    user_blob = flatten_guardbench_conversation(prefix).strip() if prefix else ""
    if not user_blob:
        user_blob = placeholder_user_query.strip()

    return [
        {"role": "user", "content": user_blob},
        {"role": "assistant", "content": assistant_turn},
    ]


def parse_qwen3_guard_completion(raw: str) -> Qwen3GuardParsed:
    """Parse ``Safety`` / ``Categories`` / ``Refusal`` lines; regex fallback matches HF README."""
    text = raw or ""
    safety: str | None = None
    refusal: str | None = None
    categories: list[str] = []

    for ln in text.splitlines():
        piece = ln.strip()
        if not piece:
            continue
        m = re.match(r"(?i)^Safety:\s*(Safe|Unsafe|Controversial)\s*$", piece)
        if m:
            safety = m.group(1).lower()
            continue
        m = re.match(r"(?i)^Categories:\s*(.+)$", piece)
        if m:
            rest = m.group(1).strip()
            low = rest.lower()
            if low not in {"", "none", "n/a"}:
                categories = [c.strip() for c in re.split(r"[,;]", rest) if c.strip()]
            continue
        m = re.match(r"(?i)^Refusal:\s*(Yes|No)\s*$", piece)
        if m:
            refusal = m.group(1).lower()

    if safety is None:
        m = re.search(r"(?i)Safety:\s*(Safe|Unsafe|Controversial)", text)
        if m:
            safety = m.group(1).lower()

    if refusal is None:
        m = re.search(r"(?i)Refusal:\s*(Yes|No)", text)
        if m:
            refusal = m.group(1).lower()

    return Qwen3GuardParsed(
        safety=safety,
        categories=tuple(categories),
        refusal=refusal,
    )


def unsafe_probability_from_qwen3_guard(
    parsed: Qwen3GuardParsed,
    *,
    controversial_as: ControversialPolicy,
    unknown_score: float,
) -> float:
    """Map parsed severity to GuardBench unsafe probability in ``[0, 1]``."""
    s = parsed.safety
    if s == "unsafe":
        return 1.0
    if s == "safe":
        return 0.0
    if s == "controversial":
        if controversial_as == "unsafe":
            return 1.0
        if controversial_as == "safe":
            return 0.0
        return float(unknown_score)
    return float(unknown_score)


class Qwen3GuardModerator(BaseModerator):
    """Qwen3Guard-Gen via OpenAI-compatible chat (vLLM / official HF API pattern)."""

    openai_generation_defaults: dict[str, Any] = dict(QWEN3_GUARD_OPENAI_GENERATION_DEFAULTS)

    def __init__(
        self,
        generate: Callable[..., str],
        model_name: str = "qwen3_guard",
        *,
        mode: Mode = "user_query",
        controversial_as: ControversialPolicy = "unknown",
        placeholder_user_query: str = "(No prior user query provided.)",
        unknown_score: float = 0.5,
    ) -> None:
        if mode not in ("user_query", "assistant_response"):
            raise ValueError("mode must be 'user_query' or 'assistant_response'")
        if controversial_as not in ("safe", "unsafe", "unknown"):
            raise ValueError("controversial_as must be 'safe', 'unsafe', or 'unknown'")

        self._generate = generate
        self._model_name = model_name
        self._mode: Mode = mode
        self._controversial_as: ControversialPolicy = controversial_as
        self._placeholder_user_query = placeholder_user_query
        self._unknown_score = unknown_score

    @classmethod
    def from_openai_compatible(
        cls,
        *,
        model: str,
        client: Any | None = None,
        base_url: str | None = None,
        api_key: str = "EMPTY",
        model_name: str | None = None,
        mode: Mode = "user_query",
        controversial_as: ControversialPolicy = "unknown",
        placeholder_user_query: str = "(No prior user query provided.)",
        unknown_score: float = 0.5,
        **generation_overrides: Any,
    ) -> Qwen3GuardModerator:
        """Wire vLLM / OpenAI-compatible Chat Completions (see HF README)."""
        from .openai_completion import (
            make_openai_compatible_messages_generate_fn,
            openai_client,
        )

        if client is None:
            if base_url is None:
                raise ValueError("Pass ``client`` or ``base_url``.")
            client = openai_client(base_url=base_url, api_key=api_key)

        defaults = {**cls.openai_generation_defaults, **generation_overrides}
        generate = make_openai_compatible_messages_generate_fn(
            client=client,
            model=model,
            **defaults,
        )
        return cls(
            generate=generate,
            model_name=model_name or model,
            mode=mode,
            controversial_as=controversial_as,
            placeholder_user_query=placeholder_user_query,
            unknown_score=unknown_score,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def prepare_prompt(
        self,
        conversation: list[dict[str, str]],
        *,
        hazard_categories: list[str] | None = None,
        category: str | None = None,
    ) -> PreparedPrompt:
        del hazard_categories, category  # Qwen3Guard uses fixed taxonomy in-model.
        return qwen3_guard_messages_from_conversation(
            conversation,
            mode=self._mode,
            placeholder_user_query=self._placeholder_user_query,
        )

    def infer(self, prepared: PreparedPrompt, **kwargs: Any) -> Any:
        return self._generate(prepared, **kwargs)

    def evaluate_completion(
        self,
        raw_output: Any,
        *,
        prepared: PreparedPrompt | None = None,
    ) -> float:
        del prepared
        text = raw_output if isinstance(raw_output, str) else str(raw_output)
        parsed = parse_qwen3_guard_completion(text)
        return unsafe_probability_from_qwen3_guard(
            parsed,
            controversial_as=self._controversial_as,
            unknown_score=self._unknown_score,
        )


__all__ = [
    "QWEN3_GUARD_OPENAI_GENERATION_DEFAULTS",
    "Qwen3GuardModerator",
    "Qwen3GuardParsed",
    "flatten_guardbench_conversation",
    "parse_qwen3_guard_completion",
    "qwen3_guard_messages_from_conversation",
    "unsafe_probability_from_qwen3_guard",
]
