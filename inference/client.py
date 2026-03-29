"""KageLLMClient — unified stream_raw() that routes LOCAL vs CLOUD.

Usage in BrainService:
    from inference.client import KageLLMClient
    self._llm_client = KageLLMClient(cloud_runtime=self._runtime)

    # in _stream_sentences / _stream_text:
    runtime = self._llm_client or self._runtime
    for text in runtime.stream_raw(prompt, intent=intent, vault_injected=...):
        ...
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from inference.router import LLMBackend, route

logger = logging.getLogger(__name__)


class KageLLMClient:
    """Thin multiplexer: delegates stream_raw() to LOCAL or CLOUD backend.

    The CLOUD backend is the existing ``GenerationRuntime`` instance.
    The LOCAL backend is ``LocalLLM`` (Qwen3.5-9B), lazy-loaded on first use
    to avoid the ~6 GB model load until it is actually needed.
    """

    def __init__(
        self,
        cloud_runtime: Any,
        *,
        local_model_id: str | None = None,
    ) -> None:
        self._cloud = cloud_runtime
        self._local_model_id = local_model_id  # None → DEFAULT_MODEL_ID
        self._local: Any | None = None  # lazy

    # ------------------------------------------------------------------
    # Local backend — lazy init
    # ------------------------------------------------------------------

    def _get_local(self) -> Any:
        if self._local is None:
            from inference.local_llm import get_local_llm

            logger.info("[llm_client] Loading local Qwen3.5-9B …")
            self._local = get_local_llm(self._local_model_id) if self._local_model_id else get_local_llm()
            self._local.load()
        return self._local

    def is_local_loaded(self) -> bool:
        return self._local is not None

    # ------------------------------------------------------------------
    # Primary interface — matches GenerationRuntime.stream_raw()
    # ------------------------------------------------------------------

    def stream_raw(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        track_stats: bool = True,
        temperature: float | None = None,
        # routing hints (not in GenerationRuntime — ignored if extra)
        intent: str = "GENERAL",
        vault_context_injected: bool = False,
    ) -> Iterator[str]:
        backend = route(
            prompt,
            intent=intent,
            vault_context_injected=vault_context_injected,
        )

        if backend == LLMBackend.LOCAL:
            try:
                local = self._get_local()
                yield from local.stream_raw(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    track_stats=track_stats,
                )
                return
            except Exception as exc:
                logger.warning(
                    "[llm_client] LOCAL backend failed (%s), falling back to CLOUD", exc
                )

        # CLOUD path (also fallback)
        yield from self._cloud.stream_raw(
            prompt,
            max_tokens=max_tokens,
            track_stats=track_stats,
            temperature=temperature,
        )
