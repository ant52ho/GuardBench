"""OpenAI Chat Completions-shaped calls against vLLM (or any OpenAI-compatible server).

Uses the ``openai`` Python SDK. Install with ``pip install 'guardbench[openai]'``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048


def _openai_module():
    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "The openai package is required for OpenAI-compatible inference. "
            "Install with: pip install 'guardbench[openai]'"
        ) from e
    return openai


def openai_client(
    *,
    base_url: str,
    api_key: str = "EMPTY",
    timeout: float | None = None,
) -> Any:
    """Build an ``OpenAI`` HTTP client aimed at vLLM's ``/v1`` endpoint.

    Example ``base_url``: ``http://127.0.0.1:8000/v1`` (include the trailing ``/v1``).
    """
    openai = _openai_module()
    kw: dict[str, Any] = {"base_url": base_url, "api_key": api_key}
    if timeout is not None:
        kw["timeout"] = timeout
    return openai.OpenAI(**kw)


def openai_compatible_chat_completion(
    *,
    client: Any,
    model: str,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    repetition_penalty: float | None = None,
    stop: list[str] | None = None,
    extra_body: dict[str, Any] | None = None,
    **extra_create_kwargs: Any,
) -> str:
    """Single-turn chat completion; returns assistant message text.

    Parameters such as ``repetition_penalty`` are merged into ``extra_body`` for vLLM and
    similar servers that accept extra JSON fields on the Chat Completions API.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    merged_extra: dict[str, Any] = dict(extra_body or ())
    if repetition_penalty is not None:
        merged_extra["repetition_penalty"] = repetition_penalty

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **extra_create_kwargs,
    }
    if stop is not None:
        params["stop"] = stop
    if merged_extra:
        params["extra_body"] = merged_extra

    resp = client.chat.completions.create(**params)
    choice = resp.choices[0].message
    content = getattr(choice, "content", None)
    return (content or "").strip()


def openai_compatible_chat_completion_messages(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    repetition_penalty: float | None = None,
    stop: list[str] | None = None,
    extra_body: dict[str, Any] | None = None,
    **extra_create_kwargs: Any,
) -> str:
    """Chat completion with caller-supplied ``messages`` (Qwen3Guard, multi-turn APIs, …)."""
    merged_extra: dict[str, Any] = dict(extra_body or ())
    if repetition_penalty is not None:
        merged_extra["repetition_penalty"] = repetition_penalty

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **extra_create_kwargs,
    }
    if stop is not None:
        params["stop"] = stop
    if merged_extra:
        params["extra_body"] = merged_extra

    resp = client.chat.completions.create(**params)
    choice = resp.choices[0].message
    content = getattr(choice, "content", None)
    return (content or "").strip()


def make_openai_compatible_generate_fn(
    *,
    client: Any,
    model: str,
    system_prompt: str | None = None,
    **generation_defaults: Any,
) -> Callable[..., str]:
    """Return ``generate(prompt: str, **kwargs) -> str`` for use with moderators.

    Keyword arguments on each call override ``generation_defaults``. Both are forwarded
    to :func:`openai_compatible_chat_completion` (except ``client`` / ``model``, which
    stay fixed).
    """

    def generate(prompt: str, **kwargs: Any) -> str:
        merged = dict(generation_defaults)
        merged.update(kwargs)
        sp = merged.pop("system_prompt", system_prompt)
        return openai_compatible_chat_completion(
            client=client,
            model=model,
            user_prompt=prompt,
            system_prompt=sp,
            **merged,
        )

    return generate


def make_openai_compatible_messages_generate_fn(
    *,
    client: Any,
    model: str,
    **generation_defaults: Any,
) -> Callable[..., str]:
    """Return ``generate(messages: list[dict[str, str]], **kwargs) -> str``.

    Used when the backend expects OpenAI-style chat messages (e.g. Qwen3Guard per HF README).
    """

    def generate(messages: list[dict[str, str]], **kwargs: Any) -> str:
        merged = dict(generation_defaults)
        merged.update(kwargs)
        return openai_compatible_chat_completion_messages(
            client=client,
            model=model,
            messages=messages,
            **merged,
        )

    return generate


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "make_openai_compatible_generate_fn",
    "make_openai_compatible_messages_generate_fn",
    "openai_client",
    "openai_compatible_chat_completion",
    "openai_compatible_chat_completion_messages",
]
