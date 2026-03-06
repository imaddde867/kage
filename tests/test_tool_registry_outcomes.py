from __future__ import annotations

import unittest

from core.agent.tool_base import Tool, ToolCall, ToolResult
from core.agent.tool_registry import ToolRegistry


class _TraceSink:
    def __init__(self) -> None:
        self.calls = []

    def record_tool_result(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _EvidenceSink:
    def __init__(self) -> None:
        self.calls = []

    def record(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _SimpleTool(Tool):
    name = "simple_tool"
    description = "Simple test tool"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, *, text: str, **kwargs):
        del kwargs
        return ToolResult(tool_name=self.name, content=f"ok:{text}")


class ToolRegistryOutcomeTests(unittest.TestCase):
    def test_registry_attaches_outcome_and_records_trace_and_evidence(self) -> None:
        trace = _TraceSink()
        evidence = _EvidenceSink()
        registry = ToolRegistry(trace_store=trace, evidence_store=evidence)
        registry.register(_SimpleTool())

        result = registry.execute(ToolCall(name="simple_tool", args={"text": "hello"}))
        self.assertFalse(result.is_error)
        self.assertIsNotNone(result.outcome)
        assert result.outcome is not None
        self.assertEqual(result.outcome.status, "ok")
        self.assertGreaterEqual(result.outcome.latency_ms or 0.0, 0.0)
        self.assertEqual(len(trace.calls), 1)
        self.assertEqual(len(evidence.calls), 1)


if __name__ == "__main__":
    unittest.main()

