from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    name: str
    argument: str = ""


_COMMAND_ALIASES = {
    "new": "new_chat",
    "reset": "new_chat",
    "clear": "new_chat",
    "sidebar": "toggle_sidebar",
    "toggle-sidebar": "toggle_sidebar",
    "sources": "show_sources",
    "source": "show_sources",
    "memory": "show_memory",
    "copy": "copy_last_answer",
    "copy-last": "copy_last_answer",
    "copy-last-answer": "copy_last_answer",
    "quit": "quit",
    "exit": "quit",
    "q": "quit",
}


def parse_slash_command(text: str) -> SlashCommand | None:
    raw = (text or "").strip()
    if not raw.startswith("/"):
        return None

    body = raw[1:].strip()
    if not body:
        return SlashCommand(name="help")

    name, _, argument = body.partition(" ")
    normalized = _COMMAND_ALIASES.get(name.strip().lower(), name.strip().lower())
    return SlashCommand(name=normalized, argument=argument.strip())
