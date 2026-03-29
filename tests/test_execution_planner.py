from __future__ import annotations

import unittest

import config
from core.platform.capability_catalog import CapabilityCatalog
from core.platform.execution_planner import ExecutionPlanner
from core.platform.models import RiskTier, Strategy


class ExecutionPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = ExecutionPlanner()
        self.settings = config.get()

    def test_capability_query_short_circuits_to_direct(self) -> None:
        catalog = CapabilityCatalog.build(
            settings=self.settings,
            tool_names=["web_search", "web_fetch"],
        )
        decision = self.planner.plan(
            user_input="What connectors can you use?",
            agent_enabled=True,
            catalog=catalog,
            classify_ambiguous=lambda _text: True,
        )
        self.assertEqual(decision.strategy, Strategy.DIRECT_ANSWER)
        self.assertTrue(decision.capability_query)
        self.assertFalse(decision.use_agent)

    def test_live_web_with_degraded_fetch_adds_reason_code(self) -> None:
        catalog = CapabilityCatalog.build(
            settings=self.settings,
            tool_names=["web_search", "web_fetch"],
            tool_health_fn=lambda name: 0.2 if name == "web_fetch" else 1.0,
        )
        decision = self.planner.plan(
            user_input="Search for the latest Bitcoin price right now",
            agent_enabled=True,
            catalog=catalog,
            classify_ambiguous=lambda _text: True,
        )
        self.assertTrue(decision.use_agent)
        self.assertIn("fresh_data_required", decision.reason_codes)
        self.assertIn("web_fetch_degraded", decision.reason_codes)

    def test_ambiguous_request_calls_classifier(self) -> None:
        calls = {"count": 0}

        def classifier(_text: str) -> bool:
            calls["count"] += 1
            return False

        catalog = CapabilityCatalog.build(
            settings=self.settings,
            tool_names=["web_search"],
        )
        decision = self.planner.plan(
            user_input="Give me one sentence about productivity",
            agent_enabled=True,
            catalog=catalog,
            classify_ambiguous=classifier,
        )
        self.assertEqual(calls["count"], 1)
        self.assertEqual(decision.strategy, Strategy.DIRECT_ANSWER)

    def test_tools_required_but_agent_disabled_sets_tooling_unavailable(self) -> None:
        catalog = CapabilityCatalog.build(
            settings=self.settings,
            tool_names=["web_search", "shell"],
        )
        decision = self.planner.plan(
            user_input="Use shell to show current directory and date",
            agent_enabled=False,
            catalog=catalog,
            classify_ambiguous=lambda _text: True,
        )
        self.assertFalse(decision.use_agent)
        self.assertTrue(decision.tooling_unavailable)
        self.assertIn("tools_unavailable", decision.reason_codes)

    def test_build_execution_intent_marks_shell_requests_high_impact(self) -> None:
        catalog = CapabilityCatalog.build(
            settings=self.settings,
            tool_names=["web_search", "shell"],
        )
        decision = self.planner.plan(
            user_input="Use shell to show current directory and date",
            agent_enabled=True,
            catalog=catalog,
            classify_ambiguous=lambda _text: True,
        )
        intent = self.planner.build_execution_intent(
            user_input="Use shell to show current directory and date",
            decision=decision,
        )
        self.assertEqual(intent.risk_tier, RiskTier.HIGH_IMPACT)
        self.assertEqual(intent.action, "run_agent")

    def test_build_execution_intent_marks_live_web_as_safe_read(self) -> None:
        catalog = CapabilityCatalog.build(
            settings=self.settings,
            tool_names=["web_search", "web_fetch"],
        )
        decision = self.planner.plan(
            user_input="Search for the latest Bitcoin price right now",
            agent_enabled=True,
            catalog=catalog,
            classify_ambiguous=lambda _text: True,
        )
        intent = self.planner.build_execution_intent(
            user_input="Search for the latest Bitcoin price right now",
            decision=decision,
            required_approval_tiers=("moderate_change", "high_impact"),
        )
        self.assertEqual(intent.risk_tier, RiskTier.SAFE_READ)
        self.assertFalse(intent.requires_approval)

    def test_build_execution_intent_marks_local_artifact_request_as_safe_read(self) -> None:
        catalog = CapabilityCatalog.build(
            settings=self.settings,
            tool_names=["local_find_files", "local_extract_pdf"],
        )
        decision = self.planner.plan(
            user_input="There is a PDF called Lasku in Downloads, check specs.",
            agent_enabled=True,
            catalog=catalog,
            classify_ambiguous=lambda _text: True,
        )
        intent = self.planner.build_execution_intent(
            user_input="There is a PDF called Lasku in Downloads, check specs.",
            decision=decision,
            required_approval_tiers=("moderate_change", "high_impact"),
        )
        self.assertEqual(intent.risk_tier, RiskTier.SAFE_READ)
        self.assertFalse(intent.requires_approval)

    def test_build_execution_intent_requires_approval_when_tier_is_configured(self) -> None:
        catalog = CapabilityCatalog.build(
            settings=self.settings,
            tool_names=["calendar_read"],
        )
        decision = self.planner.plan(
            user_input="Check my calendar for tomorrow's meetings",
            agent_enabled=True,
            catalog=catalog,
            classify_ambiguous=lambda _text: True,
        )
        intent = self.planner.build_execution_intent(
            user_input="Check my calendar for tomorrow's meetings",
            decision=decision,
            required_approval_tiers=("moderate_change",),
        )
        self.assertEqual(intent.risk_tier, RiskTier.MODERATE_CHANGE)
        self.assertTrue(intent.requires_approval)


if __name__ == "__main__":
    unittest.main()
