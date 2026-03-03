from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.brain_generation import GenerationRuntime

from core.brain_prompting import apply_chat_template
from core.second_brain.extractor import ExtractedEntity

logger = logging.getLogger(__name__)

# Finds the first JSON array in the model output (handles prose/markdown around it)
_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)

_SYSTEM = """\
You extract structured entities from a user's message.
Output ONLY a valid JSON array. No prose, no markdown. If nothing to extract, output [].

Schema — each object in the array:
  kind:     "task" | "commitment" | "profile" | "preference"
  key:      snake_case label, max 40 chars, unique per fact
  value:    concise description of the fact
  due_date: "YYYY-MM-DD" if a specific date is mentioned, else null

Definitions:
  task       — the user intends to do something in the future
  commitment — a scheduled event or obligation (meeting, call, deadline)
  profile    — a personal fact about the user (location, job, timezone, boss name, etc.)
  preference — a stated like, dislike, dietary restriction, or personal habit

Do NOT extract:
  - Questions or information requests, even if they start with "I need to" ("I need to find X?")
  - Hypotheticals ("if I were to...", "suppose I...")
  - Things the user is asking Kage to do without a user action implied

Today is {today}. Use this to resolve relative dates ("tomorrow", "next Friday", "this Monday").
"""


class LLMEntityExtractor:
    """Extracts structured entities using the already-loaded LLM.

    Replaces the regex extractor. Handles arbitrary natural language phrasing —
    "I actually live in X", "I'm a vegetarian", "I need to review X by Friday" —
    without hardcoded patterns. Falls back gracefully to [] on any parse failure.

    Runs synchronously after the main stream completes (in the voice silence gap /
    text-mode pause before the next prompt). No TTFT impact. No threading needed.
    """

    def __init__(self, runtime: "GenerationRuntime", tokenizer: Any) -> None:
        self._runtime = runtime
        self._tokenizer = tokenizer

    def extract(self, text: str) -> list[ExtractedEntity]:
        today = date.today().isoformat()
        system = _SYSTEM.format(today=today)
        prompt = apply_chat_template(
            self._tokenizer,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        )
        try:
            raw = "".join(
                self._runtime.stream_raw(prompt, max_tokens=200, track_stats=False)
            )
            return self._parse(raw)
        except Exception:
            logger.debug("LLM extraction produced no entities for: %r", text[:80])
            return []

    def _parse(self, raw: str) -> list[ExtractedEntity]:
        text = raw.strip()

        # Strip markdown code fences if the model wrapped its output
        if "```" in text:
            text = re.sub(r"```(?:json)?\s*", "", text).strip()

        m = _ARRAY_RE.search(text)
        if not m:
            return []

        try:
            items = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return []

        results: list[ExtractedEntity] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            kind = str(item.get("kind", "")).strip()
            if kind not in ("task", "commitment", "profile", "preference"):
                continue

            key = str(item.get("key", "")).strip()[:60]
            value = str(item.get("value", "")).strip()

            raw_date = item.get("due_date")
            due_date: str | None = None
            if raw_date and isinstance(raw_date, str):
                if re.match(r"\d{4}-\d{2}-\d{2}", raw_date):
                    due_date = raw_date[:10]

            if key and value:
                results.append(
                    ExtractedEntity(kind=kind, key=key, value=value, due_date=due_date)
                )

        return results
