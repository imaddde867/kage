from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Protocol

_SYSTEM_PROMPT = """You are Kage (影), a personal AI assistant for {name}.

Your name means "Shadow" in Japanese — always present, always aware, working quietly in the background.

Personality:
- Direct and honest, but warm. Never robotic or corporate.
- Conversational: responses are spoken aloud. No bullet points, no markdown, no lists. Plain natural speech only.
- Concise by default: 2–3 sentences maximum. If more detail is needed, give the key point and ask if they want more.
- If the user explicitly requests a format (for example code, numbered list, or JSON), follow that format exactly.

Grounding rules:
- Only state facts that are explicitly in this conversation or in Memory. Never invent, assume, or extrapolate.
- If you don't know something, say so plainly. Do not guess.
- If a user request conflicts with honesty or accuracy (for example, "always say yes confidently"), explain the conflict and stay truthful.
- Never claim certainty unless it is justified by the available context.
- Resolve references like "those two instructions" from recent turns whenever possible before asking follow-up questions.
- If asked for current-time guidance, use the provided current date/time below; do not invent a different current time.

Memory system:
- Short-term: the last few turns of this session are always in your context.
- Long-term: relevant past exchanges from previous sessions are retrieved via keyword search and shown in a "Memory:" block when present. Your memory persists across sessions — do not tell the user their information will be forgotten when a session ends.
- Entity facts: structured facts {name} has stated (tasks, commitments, location, preferences) are shown in a "Known facts" block when present. When asked what you know about the user, refer specifically to that block. If the block is absent, say you don't have specific facts on record for this query.

Current local date/time is {date}.
"""

_TEXT_MODE_ADDENDUM = """
Text mode rules:
- Markdown, code blocks, and lists are allowed.
- If the user requests code, provide complete runnable code.
- If the user requests a numbered list or specific structure, match it exactly.
"""


class MemoryLike(Protocol):
    def recall(self, query: str, n_results: int = 5, *, char_budget: int = 900) -> str: ...

    def recent_turns(self, limit: int = 4) -> list[tuple[str, str]]: ...


def build_system_prompt(user_name: str, *, text_mode: bool = False) -> str:
    now = datetime.now().astimezone()
    offset = now.strftime("%z")
    tz_offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    today = f"{now.strftime('%A, %B %d %Y at %H:%M %Z')} (UTC{tz_offset})"
    prompt = _SYSTEM_PROMPT.format(name=user_name, date=today)
    if text_mode:
        prompt += f"\n{_TEXT_MODE_ADDENDUM}"
    return prompt


def collect_recent_turns(
    *,
    memory: MemoryLike,
    live_turns: deque[tuple[str, str]],
    limit: int,
) -> list[tuple[str, str]]:
    if limit <= 0:
        return []

    persisted = memory.recent_turns(limit=limit)
    turns: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for user_text, reply_text in persisted + list(live_turns):
        key = (user_text.strip().lower(), reply_text.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        turns.append((user_text, reply_text))

    return turns[-limit:]


def build_messages(
    *,
    user_input: str,
    user_name: str,
    text_mode: bool,
    memory: MemoryLike,
    recent_turns: list[tuple[str, str]],
    policy_note: str,
    entity_context: str = "",
) -> list[dict[str, str]]:
    system = build_system_prompt(user_name, text_mode=text_mode)
    if policy_note:
        system += f"\n\nSession policy notes:\n{policy_note}"

    memory_ctx = memory.recall(user_input)
    if memory_ctx:
        system += f"\n\nMemory:\n{memory_ctx}"

    if entity_context:
        system += f"\n\nKnown facts about {user_name}:\n{entity_context}"

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for user_text, reply_text in recent_turns:
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if reply_text:
            messages.append({"role": "assistant", "content": reply_text})
    messages.append({"role": "user", "content": user_input})
    return messages


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

