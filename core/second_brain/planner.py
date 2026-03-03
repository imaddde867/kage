from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    intent: str
    inject_entities: bool
    should_extract: bool
    proactive_ok: bool


_ROUTES: dict[str, RouteDecision] = {
    "TASK_CAPTURE": RouteDecision("TASK_CAPTURE", inject_entities=True, should_extract=True, proactive_ok=False),
    "COMMITMENT": RouteDecision("COMMITMENT", inject_entities=True, should_extract=True, proactive_ok=True),
    "PLANNING_REQUEST": RouteDecision("PLANNING_REQUEST", inject_entities=True, should_extract=False, proactive_ok=True),
    "RECALL_REQUEST": RouteDecision("RECALL_REQUEST", inject_entities=True, should_extract=False, proactive_ok=False),
    "PROFILE_UPDATE": RouteDecision("PROFILE_UPDATE", inject_entities=False, should_extract=True, proactive_ok=False),
    "PREFERENCE": RouteDecision("PREFERENCE", inject_entities=False, should_extract=True, proactive_ok=False),
    "GENERAL": RouteDecision("GENERAL", inject_entities=False, should_extract=False, proactive_ok=False),
}

_TASK_RE = re.compile(
    r"\b(remind me|add a task|i need to|to-?do)\b",
    re.IGNORECASE,
)
_COMMITMENT_RE = re.compile(
    r"\b(i have a meeting|i have a call|i have a standup|i have an? appointment|i promised|i committed|i agreed)\b",
    re.IGNORECASE,
)
_PLANNING_RE = re.compile(
    r"\b(what should i|what(?:'s| is) next|help me plan|what do i (?:need to )?do|what(?:'s| is) on my)\b",
    re.IGNORECASE,
)
_RECALL_RE = re.compile(
    r"\b(do you remember|what did i tell you|did i mention|have i told you"
    r"|what do you know about me|what have you learned about me"
    r"|what do you know about my|what(?:'s| is) in my memory)\b",
    re.IGNORECASE,
)
_PROFILE_RE = re.compile(
    r"\b(i live in|i(?:'m| am) based in|my timezone|i(?:'m| am) from)\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(i prefer|i don'?t like|i like|i always|i never)\b",
    re.IGNORECASE,
)


class IntentRouter:
    def classify(self, text: str) -> RouteDecision:
        if _TASK_RE.search(text):
            return _ROUTES["TASK_CAPTURE"]
        if _COMMITMENT_RE.search(text):
            return _ROUTES["COMMITMENT"]
        if _PLANNING_RE.search(text):
            return _ROUTES["PLANNING_REQUEST"]
        if _RECALL_RE.search(text):
            return _ROUTES["RECALL_REQUEST"]
        if _PROFILE_RE.search(text):
            return _ROUTES["PROFILE_UPDATE"]
        if _PREFERENCE_RE.search(text):
            return _ROUTES["PREFERENCE"]
        return _ROUTES["GENERAL"]
