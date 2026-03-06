from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SignalRule:
    intent: str
    pattern: str
    weight: float = 1.0


class IntentSignals:
    """Lightweight intent scorer using compiled weighted regex rules.

    This keeps routing logic modular and data-driven instead of spreading ad-hoc
    if/regex chains across services.
    """

    def __init__(self, rules: tuple[SignalRule, ...]) -> None:
        grouped: dict[str, list[tuple[re.Pattern[str], float]]] = defaultdict(list)
        for rule in rules:
            grouped[rule.intent].append(
                (re.compile(rule.pattern, re.IGNORECASE), float(rule.weight))
            )
        self._compiled: dict[str, tuple[tuple[re.Pattern[str], float], ...]] = {
            intent: tuple(entries) for intent, entries in grouped.items()
        }
        self._intents: tuple[str, ...] = tuple(self._compiled.keys())

    def intents(self) -> tuple[str, ...]:
        return self._intents

    def score(self, text: str, intent: str) -> float:
        value = text or ""
        total = 0.0
        for pattern, weight in self._compiled.get(intent, ()):
            if pattern.search(value):
                total += weight
        return total

    def has(self, text: str, intent: str, *, threshold: float = 1.0) -> bool:
        return self.score(text, intent) >= threshold

    def scores(self, text: str, intents: Iterable[str] | None = None) -> dict[str, float]:
        target_intents = tuple(intents) if intents is not None else self._intents
        return {intent: self.score(text, intent) for intent in target_intents}

    def weighted_score(self, text: str, intent_weights: dict[str, float]) -> float:
        total = 0.0
        for intent, multiplier in intent_weights.items():
            total += self.score(text, intent) * float(multiplier)
        return total


_DEFAULT_RULES: tuple[SignalRule, ...] = (
    SignalRule("capability_query", r"\b(what|which)\s+(connectors?|tools?|capabilities)\b", 1.0),
    SignalRule("capability_query", r"\b(what|which)\s+(connectors?|tools?)\s+can\s+you\s+use\b", 1.0),
    SignalRule("capability_query", r"\bwhat do you have access to\b", 1.0),
    SignalRule("capability_query", r"\bwhat can you access\b", 1.0),
    SignalRule("task_context", r"\b(my tasks?|open tasks?|todo|to-?do|commitments?)\b", 1.0),
    SignalRule("task_context", r"\b(what should i do next|what should i work on|what'?s next)\b", 1.0),
    SignalRule("task_context", r"\b(continue planning|my trip plan|on my plate)\b", 1.0),
    SignalRule("calendar_lookup", r"\b(calendar|appointment|appointments|schedule|meeting|meetings)\b", 1.0),
    SignalRule("calendar_lookup", r"\b(next dentist visit|dentist appointment|next appointment)\b", 1.0),
    SignalRule("live_web", r"\b(latest|recent updates?|current|today'?s?|right now|breaking news|live)\b", 1.0),
    SignalRule("needs_tools", r"\b(weather|price|stock|flight|tickets?|bookings?)\b", 1.0),
    SignalRule("needs_tools", r"\b(search|look up|fetch|online|web)\b", 1.0),
    SignalRule("needs_tools", r"\b(calendar|appointment|schedule|dentist|meeting)\b", 1.0),
    SignalRule("needs_tools", r"\b(latest|recent updates?|current|today'?s?|right now|breaking news|live)\b", 1.0),
    SignalRule(
        "needs_tools",
        r"\b(compare|comparison|vs\.?|versus|better|faster)\b.*\b(specs?|performance|benchmark|model|macbook|laptop|machine|computer|local)\b",
        1.0,
    ),
    SignalRule(
        "needs_tools",
        r"\b(macbook|laptop|machine|computer|local)\b.*\b(compare|comparison|vs\.?|versus|better|faster)\b",
        1.0,
    ),
    SignalRule("needs_tools", r"\b(my local (mac|machine|computer)|this (mac|machine|computer)|system specs?|hardware specs?)\b", 1.0),
)


DEFAULT_SIGNALS = IntentSignals(_DEFAULT_RULES)
