"""LLM router — decides LOCAL vs CLOUD for each request.

LOCAL  = Qwen3.5-9B via MLX  (deep context, vault, personal)
CLOUD  = existing Kage GenerationRuntime (fast, tool-heavy, general)

Routing table (first match wins):
  1. Privacy-sensitive intents         → LOCAL
  2. Vault context was injected        → LOCAL
  3. Long context (> 8 000 tokens)     → LOCAL
  4. Tool/code/web intents             → CLOUD
  5. Default                           → LOCAL
"""
from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)

# Intent strings from core/second_brain/planner.py
_LOCAL_INTENTS = frozenset(
    {
        "RECALL_REQUEST",
        "PLANNING_REQUEST",
        "PROFILE_UPDATE",
        "PREFERENCE",
        "TASK_CAPTURE",
        "COMMITMENT",
    }
)

# Rough token estimation: 1 token ≈ 4 chars
_CHARS_PER_TOKEN = 4
_LONG_CONTEXT_TOKEN_THRESHOLD = 8_000


class LLMBackend(Enum):
    LOCAL = "local"   # Qwen3.5-9B via MLX
    CLOUD = "cloud"   # existing GenerationRuntime (Qwen3-8B / mlx_vlm)


def route(
    prompt: str,
    *,
    intent: str = "GENERAL",
    vault_context_injected: bool = False,
) -> LLMBackend:
    """Return which backend should handle this request.

    Parameters
    ----------
    prompt:
        The fully-assembled chat-template prompt (used for length estimate).
    intent:
        Intent string from RouteDecision.intent, or "GENERAL" if unknown.
    vault_context_injected:
        True when the system prompt contains vault/NeuroCache context.
    """
    # Personal / memory intents → need deep context, keep local
    if intent in _LOCAL_INTENTS:
        logger.debug("[router] LOCAL — intent=%s", intent)
        return LLMBackend.LOCAL

    # Vault context was fetched → model must understand the notes locally
    if vault_context_injected:
        logger.debug("[router] LOCAL — vault context injected")
        return LLMBackend.LOCAL

    # Long prompts → local model handles the full window better
    estimated_tokens = len(prompt) // _CHARS_PER_TOKEN
    if estimated_tokens > _LONG_CONTEXT_TOKEN_THRESHOLD:
        logger.debug("[router] LOCAL — long context (%d est. tokens)", estimated_tokens)
        return LLMBackend.LOCAL

    # Default: fast cloud model for general / tool-heavy queries
    logger.debug("[router] CLOUD — intent=%s", intent)
    return LLMBackend.CLOUD
