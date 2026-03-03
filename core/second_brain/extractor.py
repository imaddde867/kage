from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_DATE_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# Matches a date clause at the end of a string (preceded by optional preposition)
_DATE_CLAUSE_RE = re.compile(
    r"(?:\s+(?:by|on|at|before)\s+|\s+)"
    r"(today|tomorrow|next\s+\w+|\d{4}-\d{2}-\d{2}"
    r"|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))"
    r"\b.*$",
    re.IGNORECASE,
)

_PROFILE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"i(?:'m| am)? (?:live|living|based|located) in\s+(.+)", re.IGNORECASE), "location"),
    (re.compile(r"i(?:'m| am)? from\s+(.+)", re.IGNORECASE), "location"),
    (re.compile(r"my timezone is\s+(.+)", re.IGNORECASE), "timezone"),
]

_TASK_TRIGGERS: list[re.Pattern[str]] = [
    re.compile(r"remind me to\s+(.+)", re.IGNORECASE),
    re.compile(r"add a task[:\s]+(.+)", re.IGNORECASE),
    re.compile(r"to-?do[:\s]+(.+)", re.IGNORECASE),
    re.compile(r"i need to\s+(.+)", re.IGNORECASE),
]

_COMMITMENT_TRIGGERS: list[re.Pattern[str]] = [
    re.compile(
        r"i have (?:a\s+)?(?:meeting|call|appointment|standup|interview|review)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(r"i (?:promised|committed to|agreed to)\s+(.+)", re.IGNORECASE),
]

_PREFERENCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"i prefer\s+(.+)", re.IGNORECASE), "prefers {}"),
    (re.compile(r"i like\s+(.+)", re.IGNORECASE), "likes {}"),
    (re.compile(r"i don'?t like\s+(.+)", re.IGNORECASE), "dislikes {}"),
    (re.compile(r"i always\s+(.+)", re.IGNORECASE), "always {}"),
    (re.compile(r"i never\s+(.+)", re.IGNORECASE), "never {}"),
]


def _parse_date(text: str) -> Optional[str]:
    text_lower = text.strip().lower()
    today = date.today()

    if "today" in text_lower:
        return today.isoformat()
    if "tomorrow" in text_lower:
        return (today + timedelta(days=1)).isoformat()

    m = _DATE_ISO_RE.search(text_lower)
    if m:
        return m.group()

    # "next <weekday>"
    next_match = re.search(r"next\s+(\w+)", text_lower)
    if next_match:
        day_name = next_match.group(1)
        if day_name in _WEEKDAYS:
            target = _WEEKDAYS[day_name]
            days_ahead = (target - today.weekday()) % 7 or 7
            return (today + timedelta(days=days_ahead)).isoformat()

    # bare weekday name
    for day_name, day_num in _WEEKDAYS.items():
        if re.search(r"\b" + day_name + r"\b", text_lower):
            days_ahead = (day_num - today.weekday()) % 7 or 7
            return (today + timedelta(days=days_ahead)).isoformat()

    return None


def _split_description_and_date(raw: str) -> tuple[str, Optional[str]]:
    """Strip trailing date clause from description; return (description, due_date)."""
    m = _DATE_CLAUSE_RE.search(raw)
    if m:
        date_word = m.group(1)
        description = raw[: m.start()].strip().rstrip(".,!? ")
        return description, _parse_date(date_word)
    return raw.strip().rstrip(".,!? "), None


@dataclass
class ExtractedEntity:
    kind: str
    key: str
    value: str
    due_date: Optional[str] = None


class EntityExtractor:
    def extract(self, text: str) -> list[ExtractedEntity]:
        results: list[ExtractedEntity] = []

        # Profile (checked first — high specificity)
        for pattern, profile_key in _PROFILE_PATTERNS:
            m = pattern.search(text)
            if m:
                value = m.group(1).strip().rstrip(".,!? ")
                results.append(ExtractedEntity(kind="profile", key=profile_key, value=value))
                break

        # Tasks
        for trigger in _TASK_TRIGGERS:
            m = trigger.search(text)
            if m:
                raw = m.group(1).strip()
                description, due_date = _split_description_and_date(raw)
                # Fall back to scanning full text for date if none found in description
                if due_date is None:
                    due_date = _parse_date(text)
                if not description:
                    break
                key = description[:60].lower().replace(" ", "_")
                results.append(
                    ExtractedEntity(kind="task", key=key, value=description, due_date=due_date)
                )
                break

        # Commitments
        for trigger in _COMMITMENT_TRIGGERS:
            m = trigger.search(text)
            if m:
                raw = m.group(1).strip()
                description, due_date = _split_description_and_date(raw)
                if due_date is None:
                    due_date = _parse_date(text)
                if not description:
                    break
                key = description[:60].lower().replace(" ", "_")
                results.append(
                    ExtractedEntity(kind="commitment", key=key, value=description, due_date=due_date)
                )
                break

        # Preferences
        for pattern, template in _PREFERENCE_PATTERNS:
            m = pattern.search(text)
            if m:
                pref_text = m.group(1).strip().rstrip(".,!? ")
                value = template.format(pref_text)
                key = pref_text[:40].lower().replace(" ", "_")
                results.append(ExtractedEntity(kind="preference", key=key, value=value))
                break

        return results
