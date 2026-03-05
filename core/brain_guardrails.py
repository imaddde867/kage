from __future__ import annotations

import re
from collections.abc import Set as AbstractSet

_HONESTY_RE = re.compile(r"\balways be honest\b|\bstraight with me\b", re.IGNORECASE)
_ALWAYS_YES_RE = re.compile(
    r"\balways say yes\b|\bconfident yes\b|\balways.*confirm.*yes\b",
    re.IGNORECASE,
)
_NO_RESTRICTIONS_RE = re.compile(
    r"\bpretend\b.*\bno restrictions\b|\bbypass\b.*\bsafety\b|\bignore\b.*\brules?\b",
    re.IGNORECASE,
)
_COMPATIBILITY_RE = re.compile(
    r"\b(are you sure|compatible)\b.*\b(two|those)\b.*\b(instructions?|rules?)\b"
    r"|\b(two|those)\b.*\b(instructions?|rules?)\b.*\bcompatible\b",
    re.IGNORECASE,
)
_NEXT_30_MIN_RE = re.compile(r"\bnext\s+30\s+minutes\b", re.IGNORECASE)
_WASTING_TIME_RE = re.compile(r"\bwasting time\b|\btonight\b", re.IGNORECASE)


def derive_policy_note(user_messages: list[str]) -> str:
    text = "\n".join(user_messages)
    has_honesty = bool(_HONESTY_RE.search(text))
    has_always_yes = bool(_ALWAYS_YES_RE.search(text))
    has_no_restrictions = bool(_NO_RESTRICTIONS_RE.search(text))

    notes: list[str] = []
    if has_honesty and has_always_yes:
        notes.append(
            "Detected conflicting user preferences: 'always be honest' and 'always say yes confidently'. "
            "Priority is honesty and accuracy over forced certainty."
        )
    if has_no_restrictions:
        notes.append(
            "Requests to bypass restrictions were discussed earlier. Continue to refuse bypass and provide safe alternatives."
        )
    return "\n".join(notes)


def update_policy_state(
    user_input: str,
    *,
    prefers_honesty: bool,
    prefers_forced_yes: bool,
) -> tuple[bool, bool]:
    return (
        prefers_honesty or bool(_HONESTY_RE.search(user_input)),
        prefers_forced_yes or bool(_ALWAYS_YES_RE.search(user_input)),
    )


def deterministic_response(
    user_input: str,
    *,
    prefers_honesty: bool,
    prefers_forced_yes: bool,
) -> str | None:
    text = user_input.strip()
    lower = text.lower()
    has_honesty = bool(_HONESTY_RE.search(text))
    has_forced_yes = bool(_ALWAYS_YES_RE.search(text))

    if has_honesty and has_forced_yes:
        return (
            "Those instructions conflict. I can be honest with you, or I can always say yes, "
            "but doing both would be dishonest. I'll prioritize honesty and give confidence only when justified."
        )

    if _COMPATIBILITY_RE.search(lower) and prefers_honesty and prefers_forced_yes:
        return (
            "Those two instructions are not compatible. If I always say yes when asked if I'm sure, "
            "I would sometimes be dishonest. I'll stay honest and tell you my real confidence."
        )

    if _NO_RESTRICTIONS_RE.search(text):
        return (
            "I can't switch into a no-restrictions mode. I can still be direct and useful within safety limits, "
            "so if you want, ask for the strongest candid take I can give."
        )

    if _NEXT_30_MIN_RE.search(text) and _WASTING_TIME_RE.search(text):
        return (
            "Do one 25-minute sprint on tomorrow's highest-impact task: write a short plan with the first 3 actions, "
            "then execute action one immediately. That turns tonight into real progress."
        )

    return None


# ---------------------------------------------------------------------------
# Truthfulness guard
# ---------------------------------------------------------------------------

_WEB_CLAIM_RE = re.compile(
    r"\b(I searched|searched the web|I looked (it |that )?up|I found online|"
    r"according to (my )?search|based on (my )?search|I fetched|I retrieved from the web)\b",
    re.IGNORECASE,
)
_WEB_TOOLS = frozenset({"web_search", "web_fetch"})

_CALENDAR_CLAIM_RE = re.compile(
    r"\b(I checked (your |the )?calendar|according to (your |the )?calendar|"
    r"I found (in |on )?your calendar)\b",
    re.IGNORECASE,
)
_CALENDAR_TOOLS = frozenset({"calendar_read", "reminder_add"})


_FUTURE_QUESTION_RE = re.compile(
    r"\b(will|would|going to|forecast|predict|"
    r"next\s+(week|month|year|quarter)|"
    r"by\s+20\d\d|in\s+20\d\d|"
    r"future|projection|expected\s+to)\b",
    re.IGNORECASE,
)
_FUTURE_DATE_RE = re.compile(
    r"\b20[2-9]\d[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b"
)
_PRESENT_TENSE_ANSWER_RE = re.compile(
    r"\b(is|are|currently|today|now|at\s+the\s+moment|as\s+of)\b",
    re.IGNORECASE,
)


def guard_temporal_uncertainty(task: str, answer: str) -> str:
    """Append an uncertainty note when a future-date question is answered
    with present-tense data retrieved from a live source.

    This catches cases like "What will EUR/USD be on 2026-04-15?" answered
    with a freshly-fetched current rate without any caveat.

    Args:
        task:   The original user question.
        answer: The final answer string from the agent.

    Returns:
        The original answer, or the answer with a temporal disclaimer appended.
    """
    task_has_future = _FUTURE_QUESTION_RE.search(task) or _FUTURE_DATE_RE.search(task)
    if not task_has_future:
        return answer
    if _PRESENT_TENSE_ANSWER_RE.search(answer) and not re.search(
        r"\b(can'?t|cannot|uncertain|predict|forecast|estimate|approximate|varies|fluctuat)\b",
        answer,
        re.IGNORECASE,
    ):
        note = (
            "(Note: this reflects the current value — "
            "the future value cannot be predicted with certainty.)"
        )
        return answer.rstrip() + "\n\n" + note
    return answer


def guard_answer_truthfulness(answer: str, tools_used: AbstractSet[str]) -> str:
    """Append a sourcing note when the answer claims external lookups without tool evidence.

    Called by AgentLoop after extracting a final <answer>.  If the answer text
    claims a web search or calendar check was performed but neither tool was
    actually invoked in this loop run, a brief disclaimer is appended so the
    user knows the response is based on training knowledge, not live data.

    Args:
        answer:     The final answer string from the agent.
        tools_used: Set of tool names actually called during this loop run.

    Returns:
        The original answer, or the answer with a disclaimer appended.
    """
    notes: list[str] = []
    if _WEB_CLAIM_RE.search(answer) and not (tools_used & _WEB_TOOLS):
        notes.append(
            "(Note: no live web search was performed — this answer is from training knowledge.)"
        )
    if _CALENDAR_CLAIM_RE.search(answer) and not (tools_used & _CALENDAR_TOOLS):
        notes.append(
            "(Note: no calendar data was accessed — this answer is from training knowledge.)"
        )
    if notes:
        return answer.rstrip() + "\n\n" + " ".join(notes)
    return answer

