"""Unit tests for core/agent/loop.py with a scripted mock runtime (no LLM)."""
import unittest
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import config
from core.agent.loop import AgentLoop
from core.agent.tool_base import Tool, ToolResult
from core.agent.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _ScriptedRuntime:
    """Returns pre-scripted XML responses in order; repeats the last one when exhausted."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def stream_raw(
        self, prompt: str, *, max_tokens: int = 256, track_stats: bool = True
    ) -> Iterator[str]:
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        yield text


def _mock_tokenizer() -> Any:
    tok = MagicMock()
    tok.apply_chat_template = lambda msgs, **kw: " ".join(m["content"] for m in msgs)
    return tok


def _registry(*tools: Tool) -> ToolRegistry:
    r = ToolRegistry()
    for t in tools:
        r.register(t)
    return r


class _UpperTool(Tool):
    name = "upper"
    description = "Returns the uppercased text"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, *, text: str, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, content=text.upper())


class _ErrorTool(Tool):
    name = "bad_tool"
    description = "Always fails"
    parameters: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, content="Something went wrong.", is_error=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentLoopDirectAnswer(unittest.TestCase):
    """Model answers without calling any tools."""

    def _loop(self, responses: list[str], tools: list[Tool] | None = None) -> AgentLoop:
        return AgentLoop(
            runtime=_ScriptedRuntime(responses),
            tokenizer=_mock_tokenizer(),
            registry=_registry(*(tools or [])),
            settings=config.get(),
        )

    def test_direct_answer(self) -> None:
        loop = self._loop(["<answer>Paris is the capital of France.</answer>"])
        result = "".join(loop.run("What is the capital of France?"))
        self.assertEqual(result, "Paris is the capital of France.")

    def test_answer_with_thought(self) -> None:
        raw = "<thought>I know this.</thought>\n<answer>42 is the answer.</answer>"
        loop = self._loop([raw])
        result = "".join(loop.run("What is the answer to life?"))
        self.assertEqual(result, "42 is the answer.")

    def test_plain_text_fallback(self) -> None:
        """No XML tags at all — raw text should be yielded as the answer."""
        loop = self._loop(["I just know this without tools."])
        result = "".join(loop.run("Tell me something."))
        self.assertEqual(result, "I just know this without tools.")

    def test_empty_response_hits_max_steps(self) -> None:
        # empty raw → no XML, no text → loop exhausts and yields cap message
        loop = self._loop([""])
        result = "".join(loop.run("", max_steps=1))
        self.assertIn("step limit", result.lower())


class TestAgentLoopToolUse(unittest.TestCase):
    """Model calls a tool then answers."""

    def _loop(self, responses: list[str], tools: list[Tool]) -> AgentLoop:
        return AgentLoop(
            runtime=_ScriptedRuntime(responses),
            tokenizer=_mock_tokenizer(),
            registry=_registry(*tools),
            settings=config.get(),
        )

    def test_tool_then_answer(self) -> None:
        responses = [
            '<thought>Let me uppercase this.</thought>\n<tool>upper</tool>\n<input>{"text": "hello"}</input>',
            "<answer>The result is HELLO.</answer>",
        ]
        loop = self._loop(responses, [_UpperTool()])
        result = "".join(loop.run("Uppercase hello"))
        self.assertEqual(result, "The result is HELLO.")

    def test_tool_error_still_continues(self) -> None:
        """If tool returns is_error, the loop continues to the next step."""
        responses = [
            '<tool>bad_tool</tool>\n<input>{}</input>',
            "<answer>Could not complete that task.</answer>",
        ]
        loop = self._loop(responses, [_ErrorTool()])
        result = "".join(loop.run("Do something bad"))
        self.assertEqual(result, "Could not complete that task.")

    def test_unknown_tool_call_continues(self) -> None:
        """Calling a non-registered tool should return error result and let loop continue."""
        responses = [
            '<tool>does_not_exist</tool>\n<input>{}</input>',
            "<answer>I couldn't use that tool.</answer>",
        ]
        loop = self._loop(responses, [])
        result = "".join(loop.run("Use unknown tool"))
        self.assertEqual(result, "I couldn't use that tool.")

    def test_max_steps_cap(self) -> None:
        """Loop hitting max steps yields a graceful failure message."""
        # runtime always returns a tool call — never a final answer
        loop = self._loop(
            ['<tool>upper</tool>\n<input>{"text": "x"}</input>'] * 20,
            [_UpperTool()],
        )
        result = "".join(loop.run("Loop forever", max_steps=3))
        self.assertIn("step limit", result.lower())

    def test_multi_step_two_tool_calls(self) -> None:
        responses = [
            '<tool>upper</tool>\n<input>{"text": "foo"}</input>',
            '<tool>upper</tool>\n<input>{"text": "bar"}</input>',
            "<answer>Done: FOO and BAR.</answer>",
        ]
        loop = self._loop(responses, [_UpperTool()])
        result = "".join(loop.run("Uppercase foo and bar"))
        self.assertEqual(result, "Done: FOO and BAR.")


class TestAgentLoopHistoryStructure(unittest.TestCase):
    """Verify that the prompt builder receives correct role alternation."""

    def test_history_passed_to_tokenizer(self) -> None:
        """The mock tokenizer joins all message contents; we verify tool result appears."""
        collected_prompts: list[str] = []

        class _CapturingTokenizer:
            def apply_chat_template(self, msgs: list[dict], **kw: Any) -> str:
                prompt = " ".join(m["content"] for m in msgs)
                collected_prompts.append(prompt)
                return prompt

        responses = [
            '<tool>upper</tool>\n<input>{"text": "world"}</input>',
            "<answer>WORLD it is.</answer>",
        ]
        loop = AgentLoop(
            runtime=_ScriptedRuntime(responses),
            tokenizer=_CapturingTokenizer(),
            registry=_registry(_UpperTool()),
            settings=config.get(),
        )
        "".join(loop.run("Uppercase world"))

        # Second prompt (step 2) must contain the tool observation
        self.assertGreater(len(collected_prompts), 1)
        self.assertIn("WORLD", collected_prompts[1])


class TestAgentLoopContextInjection(unittest.TestCase):
    """Entity context block appears in the system prompt."""

    def test_context_in_prompt(self) -> None:
        captured: list[str] = []

        class _CapturingTokenizer:
            def apply_chat_template(self, msgs: list[dict], **kw: Any) -> str:
                prompt = " ".join(m["content"] for m in msgs)
                captured.append(prompt)
                return prompt

        loop = AgentLoop(
            runtime=_ScriptedRuntime(["<answer>ok</answer>"]),
            tokenizer=_CapturingTokenizer(),
            registry=_registry(),
            settings=config.get(),
        )
        "".join(loop.run("task", context="Tasks: finish report (due today)"))
        self.assertIn("finish report", captured[0])


if __name__ == "__main__":
    unittest.main()
