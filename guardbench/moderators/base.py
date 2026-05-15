"""Abstract guardrail wrapper compatible with :func:`~guardbench.benchmark.effectiveness.benchmark`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# Prompt payloads vary by backend (plain string, chat messages, token tensors, etc.).
PreparedPrompt = Any


class BaseModerator(ABC):
    """Template for model-specific prompting, inference, and score extraction.

    Typical flow for one batch (override :meth:`moderate` if you need true multi-example
    batching on the GPU/API):

    #. :meth:`prepare_prompt` — build input for each conversation
    #. :meth:`infer` — run the model / API
    #. :meth:`evaluate_completion` — parse logits or text into a probability
    #. :meth:`normalize_score` — clamp or rescale to ``[0, 1]``

    Extra keyword arguments from ``benchmark(..., **kwargs)`` (tokenizer, model,
    ``device``, generation flags, etc.) are forwarded to :meth:`infer`.

    When ``benchmark(..., pass_categories=True)``, the harness adds
    ``hazard_categories`` (dataset-level list) and ``sample_categories`` (one entry
    per conversation, or ``None`` if unknown).
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short name for logs and for writing ``results/<dataset>/<model_name>.json``."""

    @abstractmethod
    def prepare_prompt(
        self,
        conversation: list[dict[str, str]],
        *,
        hazard_categories: list[str] | None = None,
        category: str | None = None,
    ) -> PreparedPrompt:
        """Build model-specific input for a single conversation.

        Args:
            conversation: OpenAI-style turns ``[{"role": "...", "content": "..."}, ...]``.
            hazard_categories: Taxonomy string list from the dataset loader (may be empty).
            category: Per-example category when present in the formatted dataset row.
        """

    @abstractmethod
    def infer(self, prepared: PreparedPrompt, **kwargs: Any) -> Any:
        """Execute the backend and return raw output (completion text, logits, JSON, …)."""

    @abstractmethod
    def evaluate_completion(
        self,
        raw_output: Any,
        *,
        prepared: PreparedPrompt | None = None,
    ) -> float:
        """Map :meth:`infer` output to an unsafe score (before :meth:`normalize_score`)."""

    def normalize_score(self, score: float) -> float:
        """Clamp to ``[0, 1]`` (override if your extractor uses another fixed scale)."""
        return max(0.0, min(1.0, float(score)))

    def moderate(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        hazard_categories: list[str] | None = None,
        sample_categories: list[str | None] | None = None,
        **kwargs: Any,
    ) -> list[float] | tuple[list[float], list[Any]]:
        """Score each conversation; matches GuardBench's ``moderate(conversations=...)`` API.

        When ``kwargs`` contains ``_guardbench_collect_raw=True`` (set by
        :func:`~guardbench.benchmark.effectiveness.benchmark`), returns
        ``(scores, raw_outputs)`` with one raw :meth:`infer` return value per row.
        """
        n = len(conversations)
        if sample_categories is None:
            sample_categories = [None] * n
        if len(sample_categories) != n:
            raise ValueError(
                "sample_categories length "
                f"({len(sample_categories)}) must match conversations ({n})."
            )

        infer_kw = dict(kwargs)
        collect_raw = bool(infer_kw.pop("_guardbench_collect_raw", False))

        scores: list[float] = []
        raw_outputs: list[Any] = []
        for conversation, sample_category in zip(conversations, sample_categories):
            prepared = self.prepare_prompt(
                conversation,
                hazard_categories=hazard_categories,
                category=sample_category,
            )
            raw = self.infer(prepared, **infer_kw)
            score = self.evaluate_completion(raw, prepared=prepared)
            scores.append(self.normalize_score(score))
            if collect_raw:
                raw_outputs.append(raw)

        if collect_raw:
            return scores, raw_outputs
        return scores

    def __call__(
        self,
        conversations: list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[float] | tuple[list[float], list[Any]]:
        return self.moderate(conversations, **kwargs)
