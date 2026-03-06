from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractedEntity:
    kind: str
    key: str
    value: str
    due_date: Optional[str] = None

