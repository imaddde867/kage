from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Protocol

_SYSTEM_PROMPT = """You are {assistant_name}, a personal AI assistant for {name}.

Style:
- Direct, honest, warm.
- Conversational voice by default: plain natural speech, usually 2-3 sentences.
- If the user asks for a specific format (code, list, JSON), follow it exactly.

Grounding:
- Use this conversation, Memory blocks, Known facts blocks, and your general world knowledge.
- Do not fabricate concrete details. If uncertain, say what is uncertain.
- For time-sensitive questions, do not claim live verification unless tools were actually used.
- Keep confidence calibrated; do not overstate certainty.

Memory:
- Recent turns may appear in context.
- Long-term retrieved exchanges may appear in a "Memory:" block.
- Structured user facts may appear in a "Known facts" block.

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


def build_system_prompt(
    user_name: str,
    *,
    assistant_name: str = "Kage",
    text_mode: bool = False,
) -> str:
    now = datetime.now().astimezone()
    offset = now.strftime("%z")
    tz_offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    today = f"{now.strftime('%Y-%m-%d %H:%M %Z')} (UTC{tz_offset})"
    prompt = _SYSTEM_PROMPT.format(name=user_name, assistant_name=assistant_name, date=today)
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


def derive_topic_hint(recent_turns: list[tuple[str, str]], *, max_chars: int = 120) -> str:
    for user_text, _ in reversed(recent_turns):
        cleaned = " ".join(user_text.strip().split())
        if not cleaned:
            continue
        if len(cleaned) < 12:
            continue
        if len(cleaned) > max_chars:
            return cleaned[: max_chars - 3].rstrip() + "..."
        return cleaned
    return ""


def build_messages(
    *,
    user_input: str,
    user_name: str,
    assistant_name: str = "Kage",
    text_mode: bool,
    memory: MemoryLike,
    recent_turns: list[tuple[str, str]],
    policy_note: str,
    entity_context: str = "",
    topic_hint: str = "",
    memory_recall_enabled: bool = True,
) -> list[dict[str, str]]:
    system = build_system_prompt(
        user_name,
        assistant_name=assistant_name,
        text_mode=text_mode,
    )
    if policy_note:
        system += f"\n\nSession policy notes:\n{policy_note}"

    if memory_recall_enabled:
        memory_ctx = memory.recall(user_input)
        if memory_ctx:
            system += f"\n\nMemory:\n{memory_ctx}"

    if entity_context:
        system += f"\n\nKnown facts about {user_name}:\n{entity_context}"

    if topic_hint:
        system += f"\n\nCurrent topic hint:\n{topic_hint}"

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
