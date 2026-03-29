from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.platform.models import RiskTier
from core.platform.storage import ApprovalStore

_VALID_POLICY_MODES = {"strict", "hybrid", "owner_fast"}
_VALID_TIERS = {tier.value for tier in RiskTier}
_DEFAULT_REQUIRED_TIERS = {RiskTier.MODERATE_CHANGE.value, RiskTier.HIGH_IMPACT.value}

_TOOL_RISK_TIERS: dict[str, RiskTier] = {
    "web_search": RiskTier.SAFE_READ,
    "web_fetch": RiskTier.SAFE_READ,
    "browser_fetch": RiskTier.SAFE_READ,
    "local_find_files": RiskTier.SAFE_READ,
    "local_read_text": RiskTier.SAFE_READ,
    "local_extract_pdf": RiskTier.SAFE_READ,
    "local_extract_docx": RiskTier.SAFE_READ,
    "local_extract_sheet": RiskTier.SAFE_READ,
    "list_open_tasks": RiskTier.SAFE_READ,
    "list_scheduled_reminders": RiskTier.SAFE_READ,
    "calendar_read": RiskTier.MODERATE_CHANGE,
    "schedule_reminder": RiskTier.MODERATE_CHANGE,
    "cancel_reminder": RiskTier.MODERATE_CHANGE,
    "reminder_add": RiskTier.MODERATE_CHANGE,
    "mark_task_done": RiskTier.MODERATE_CHANGE,
    "update_fact": RiskTier.MODERATE_CHANGE,
    "forget_fact": RiskTier.MODERATE_CHANGE,
    "notify": RiskTier.MODERATE_CHANGE,
    "speak": RiskTier.MODERATE_CHANGE,
    "shell": RiskTier.HIGH_IMPACT,
    "shell_mutation": RiskTier.HIGH_IMPACT,
}


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    tier: RiskTier
    requires_approval: bool
    reason_code: str
    message: str = ""
    approval_keys: tuple[str, ...] = ()


class PolicyEngine:
    def __init__(self, *, settings: Any, approval_store: ApprovalStore) -> None:
        self._settings = settings
        self._approval_store = approval_store

    def mode(self) -> str:
        raw = str(getattr(self._settings, "agent_policy_mode", "strict")).strip().lower()
        return raw if raw in _VALID_POLICY_MODES else "strict"

    def configured_required_tiers(self) -> tuple[str, ...]:
        raw = getattr(self._settings, "agent_approval_required_tiers", ())
        if not isinstance(raw, tuple):
            raw = tuple(raw) if isinstance(raw, list) else ()
        normalized = tuple(sorted({str(item).strip().lower() for item in raw if str(item).strip()}))
        valid = tuple(tier for tier in normalized if tier in _VALID_TIERS)
        return valid or tuple(sorted(_DEFAULT_REQUIRED_TIERS))

    def required_tiers(self) -> set[str]:
        mode = self.mode()
        if mode == "owner_fast":
            return set()
        if mode == "hybrid":
            return {RiskTier.HIGH_IMPACT.value}
        configured = set(self.configured_required_tiers())
        return configured

    def tier_for_tool(self, tool_name: str) -> RiskTier | None:
        key = (tool_name or "").strip().lower()
        return _TOOL_RISK_TIERS.get(key)

    def evaluate(self, *, tool_name: str, args: dict[str, Any]) -> PolicyDecision:
        del args
        normalized_name = (tool_name or "").strip().lower()
        tier = self.tier_for_tool(normalized_name)
        if tier is None:
            return PolicyDecision(
                allowed=False,
                tier=RiskTier.HIGH_IMPACT,
                requires_approval=False,
                reason_code="policy_unknown_tool",
                message=(
                    f"[POLICY_BLOCK:blocked] Tool '{normalized_name}' is blocked because it has no policy tier mapping."
                ),
            )

        required = self.required_tiers()
        requires_approval = tier.value in required
        if not requires_approval:
            return PolicyDecision(
                allowed=True,
                tier=tier,
                requires_approval=False,
                reason_code="policy_allowed",
            )

        tool_key = f"tool:{normalized_name}"
        tier_key = f"tier:{tier.value}"
        if self._approval_store.is_granted_key(tool_key) or self._approval_store.is_granted_key(tier_key):
            return PolicyDecision(
                allowed=True,
                tier=tier,
                requires_approval=True,
                reason_code="policy_approved",
                approval_keys=(tool_key, tier_key),
            )

        return PolicyDecision(
            allowed=False,
            tier=tier,
            requires_approval=True,
            reason_code="policy_awaiting_approval",
            approval_keys=(tool_key, tier_key),
            message=(
                "[POLICY_BLOCK:awaiting_approval] "
                f"Tool '{normalized_name}' is {tier.value} and needs approval in mode '{self.mode()}'. "
                f"Grant one of: `kage approvals grant tool {normalized_name}` or "
                f"`kage approvals grant tier {tier.value}`."
            ),
        )


def valid_policy_mode(value: str) -> bool:
    return str(value).strip().lower() in _VALID_POLICY_MODES


def valid_risk_tier(value: str) -> bool:
    return str(value).strip().lower() in _VALID_TIERS
