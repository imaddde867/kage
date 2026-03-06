from __future__ import annotations

import time
from datetime import date
from typing import Any

from core.platform.models import ProactiveOpportunity
from core.platform.storage.schema import connect_db, ensure_schema


class ProactivePolicyEngine:
    def __init__(self) -> None:
        self._last_emit_monotonic = 0.0

    def suggest_from_reply(
        self,
        *,
        entity_store: Any,
        settings: Any,
        reply: str,
        proactive_ok: bool,
    ) -> str | None:
        if not proactive_ok:
            return None

        debounce = getattr(settings, "proactive_debounce_seconds", 60)
        now = time.monotonic()
        if now - self._last_emit_monotonic < debounce:
            return None

        entities = []
        entities.extend(entity_store.get_by_kind("task", status="active"))
        entities.extend(entity_store.get_by_kind("commitment", status="active"))
        if not entities:
            return None

        lowered_reply = (reply or "").lower()
        for entity in entities:
            value = (entity.value or "").strip()
            if not value:
                continue
            if value.lower() in lowered_reply:
                continue
            if entity.due_date:
                suggestion = f"By the way, you have {value} due {entity.due_date}."
            else:
                suggestion = f"By the way, you have an open task: {value}."
            self._last_emit_monotonic = now
            db_path = getattr(entity_store, "db_path", None)
            if db_path is not None:
                try:
                    self._audit_opportunities(
                        db_path=db_path,
                        opportunities=[
                            ProactiveOpportunity(
                                kind="reply_followup",
                                message=suggestion,
                                reason="post_reply_contextual_hint",
                                due_date=getattr(entity, "due_date", None),
                                entity_id=getattr(entity, "id", None),
                            )
                        ],
                    )
                except Exception:
                    pass
            return suggestion

        return None

    def due_opportunities(
        self,
        *,
        entity_store: Any,
    ) -> list[ProactiveOpportunity]:
        today = date.today().isoformat()
        opportunities: list[ProactiveOpportunity] = []
        for kind in ("task", "commitment"):
            for entity in entity_store.get_by_kind(kind, status="active"):
                due_date = getattr(entity, "due_date", None)
                if not due_date or due_date > today:
                    continue
                label = "overdue" if due_date < today else "due today"
                opportunities.append(
                    ProactiveOpportunity(
                        kind=kind,
                        message=f"{entity.value} ({label})",
                        reason="due_or_overdue",
                        due_date=due_date,
                        entity_id=getattr(entity, "id", None),
                    )
                )
        return opportunities

    def _audit_opportunities(self, *, db_path: Any, opportunities: list[ProactiveOpportunity]) -> None:
        if not opportunities:
            return
        ensure_schema(db_path)
        with connect_db(db_path) as conn:
            for item in opportunities:
                conn.execute(
                    """
                    INSERT INTO proactive_opportunities (kind, message, reason, due_date, entity_id, created_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (item.kind, item.message, item.reason, item.due_date, item.entity_id),
                )

    def compose_due_digest(
        self,
        *,
        entity_store: Any,
        user_name: str,
    ) -> str | None:
        opportunities = self.due_opportunities(entity_store=entity_store)
        db_path = getattr(entity_store, "db_path", None)
        if db_path is not None:
            try:
                self._audit_opportunities(db_path=db_path, opportunities=opportunities)
            except Exception:
                pass
        due_items = [o.message for o in opportunities]
        if not due_items:
            return None
        if len(due_items) == 1:
            return f"Hey {user_name}, just a reminder: {due_items[0]}."
        joined = "; ".join(due_items[:3])
        return f"Hey {user_name}, you have {len(due_items)} things due: {joined}."
