"""Unit tests for core/agent/loop.py with a scripted mock runtime (no LLM).

Strategy
--------
_ScriptedRuntime replaces GenerationRuntime and returns pre-written XML strings
one-by-one so tests are deterministic and run in milliseconds.  _mock_tokenizer
concatenates all message contents so prompts can be inspected as plain strings.

Test classes:
    TestAgentLoopDirectAnswer   — model answers immediately with no tool calls
    TestAgentLoopToolUse        — model calls tools, then answers; error paths
    TestAgentLoopHistoryStructure — verifies role alternation in the built prompt
    TestAgentLoopContextInjection — entity context appears in system prompt
"""
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
    """Fake GenerationRuntime that yields pre-written responses in order.

    Responses are consumed one per stream_raw() call.  Once the list is
    exhausted the last response is repeated — this keeps infinite-loop tests
    from crashing while letting the max_steps cap trigger as expected.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def stream_raw(
        self, prompt: str, *, max_tokens: int = 256, track_stats: bool = True
    ) -> Iterator[str]:
        """Yield the next scripted response as a single chunk."""
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        yield text


def _mock_tokenizer() -> Any:
    """Return a tokenizer mock whose apply_chat_template joins message contents.

    This lets tests inspect the assembled prompt as a plain string by checking
    that expected substrings are present (e.g. tool observation, entity context).
    """
    tok = MagicMock()
    tok.apply_chat_template = lambda msgs, **kw: " ".join(m["content"] for m in msgs)
    return tok


def _registry(*tools: Tool) -> ToolRegistry:
    """Convenience helper — builds a ToolRegistry pre-loaded with the given tools."""
    r = ToolRegistry()
    for t in tools:
        r.register(t)
    return r


class _UpperTool(Tool):
    """Returns the uppercased version of its text argument — simple, deterministic."""
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
    """Always returns an error ToolResult — used to verify loop continues after errors."""
    name = "bad_tool"
    description = "Always fails"
    parameters: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, content="Something went wrong.", is_error=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentLoopDirectAnswer(unittest.TestCase):
    """Model answers without calling any tools.

    These tests verify the happy path where a single generation step produces
    a final <answer> (or falls back to plain text).  No tools are invoked.
    """

    def _loop(self, responses: list[str], tools: list[Tool] | None = None) -> AgentLoop:
        """Create an AgentLoop with a scripted runtime and optional tools."""
        return AgentLoop(
            runtime=_ScriptedRuntime(responses),
            tokenizer=_mock_tokenizer(),
            registry=_registry(*(tools or [])),
            settings=config.get(),
        )

    def test_direct_answer(self) -> None:
        """A plain <answer> tag on the first step yields the answer and stops."""
        loop = self._loop(["<answer>Paris is the capital of France.</answer>"])
        result = "".join(loop.run("What is the capital of France?"))
        self.assertEqual(result, "Paris is the capital of France.")

    def test_answer_with_thought(self) -> None:
        """<thought> is silently discarded; only the <answer> content is yielded."""
        raw = "<thought>I know this.</thought>\n<answer>42 is the answer.</answer>"
        loop = self._loop([raw])
        result = "".join(loop.run("What is the answer to life?"))
        self.assertEqual(result, "42 is the answer.")

    def test_plain_text_fallback(self) -> None:
        """No XML tags at all — raw text is used as the answer via the plain-text fallback."""
        loop = self._loop(["I just know this without tools."])
        result = "".join(loop.run("Tell me something."))
        self.assertEqual(result, "I just know this without tools.")

    def test_empty_response_hits_max_steps(self) -> None:
        """Empty generation output produces no answer and no tool call; max_steps=1 triggers cap."""
        # empty raw → no XML, no text → loop exhausts and yields cap message
        loop = self._loop([""])
        result = "".join(loop.run("", max_steps=1))
        self.assertIn("step limit", result.lower())


class TestAgentLoopToolUse(unittest.TestCase):
    """Model calls a tool then answers — verifies the full ReAct cycle.

    The scripted runtime produces tool-call XML on early steps and a final
    <answer> at the end.  These tests verify that the loop:
      - dispatches the tool call correctly
      - passes the observation back into the next prompt
      - continues after tool errors or unknown tool names
      - terminates when max_steps is hit
    """

    def _loop(self, responses: list[str], tools: list[Tool]) -> AgentLoop:
        """Create an AgentLoop pre-loaded with the given scripted responses and tools."""
        return AgentLoop(
            runtime=_ScriptedRuntime(responses),
            tokenizer=_mock_tokenizer(),
            registry=_registry(*tools),
            settings=config.get(),
        )

    def test_tool_then_answer(self) -> None:
        """One tool call followed by a final answer — the basic ReAct step."""
        responses = [
            '<thought>Let me uppercase this.</thought>\n<tool>upper</tool>\n<input>{"text": "hello"}</input>',
            "<answer>The result is HELLO.</answer>",
        ]
        loop = self._loop(responses, [_UpperTool()])
        result = "".join(loop.run("Uppercase hello"))
        self.assertEqual(result, "The result is HELLO.")

    def test_tool_error_still_continues(self) -> None:
        """If tool returns is_error=True, the loop feeds the error as an observation and continues."""
        responses = [
            '<tool>bad_tool</tool>\n<input>{}</input>',
            "<answer>Could not complete that task.</answer>",
        ]
        loop = self._loop(responses, [_ErrorTool()])
        result = "".join(loop.run("Do something bad"))
        self.assertEqual(result, "Could not complete that task.")

    def test_unknown_tool_call_continues(self) -> None:
        """Calling a non-registered tool returns an error observation; loop continues normally."""
        responses = [
            '<tool>does_not_exist</tool>\n<input>{}</input>',
            "<answer>I couldn't use that tool.</answer>",
        ]
        loop = self._loop(responses, [])
        result = "".join(loop.run("Use unknown tool"))
        self.assertEqual(result, "I couldn't use that tool.")

    def test_max_steps_cap(self) -> None:
        """Loop terminates gracefully when tool calls never produce an answer.

        When every response is a tool call with varying args, the repeated-tool
        guard does not fire.  After max_steps iterations the step-limit message
        is returned.
        """
        # Varying args prevent the repeated-tool guard from firing; only max_steps caps it
        responses = [
            f'<tool>upper</tool>\n<input>{{"text": "step{i}"}}</input>' for i in range(20)
        ]
        loop = self._loop(responses, [_UpperTool()])
        result = "".join(loop.run("Loop forever", max_steps=3))
        self.assertIn("step limit", result.lower())

    def test_multi_step_two_tool_calls(self) -> None:
        """Two consecutive tool calls followed by a final answer — verifies loop continues correctly."""
        responses = [
            '<tool>upper</tool>\n<input>{"text": "foo"}</input>',
            '<tool>upper</tool>\n<input>{"text": "bar"}</input>',
            "<answer>Done: FOO and BAR.</answer>",
        ]
        loop = self._loop(responses, [_UpperTool()])
        result = "".join(loop.run("Uppercase foo and bar"))
        self.assertEqual(result, "Done: FOO and BAR.")


class TestAgentLoopHistoryStructure(unittest.TestCase):
    """Verify that the prompt builder receives correct role alternation.

    After a tool call, the observation must appear in the *next* prompt.
    The capturing tokenizer records every call to apply_chat_template so
    tests can assert on the exact content of each step's prompt.
    """

    def test_history_passed_to_tokenizer(self) -> None:
        """Tool observation (WORLD) must appear in the second step's assembled prompt.

        This confirms that _build_prompt() correctly injects the prior
        assistant output and the tool result as history before calling the
        tokenizer for step 2.
        """
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


class TestAgentLoopRepeatedToolGuard(unittest.TestCase):
    """When the same tool+args is called 3 times the loop bails with a message.

    This prevents infinite fetch loops where the agent keeps calling web_fetch
    on the same URL without making progress.
    """

    def _loop(self, responses: list[str], tools: list[Tool]) -> AgentLoop:
        return AgentLoop(
            runtime=_ScriptedRuntime(responses),
            tokenizer=_mock_tokenizer(),
            registry=_registry(*tools),
            settings=config.get(),
        )

    def test_repeated_same_call_triggers_guard(self) -> None:
        """Third identical (tool, args) call yields the 'making progress' bail message."""
        # Every response is the same tool call with the same args
        loop = self._loop(
            ['<tool>upper</tool>\n<input>{"text": "x"}</input>'] * 10,
            [_UpperTool()],
        )
        result = "".join(loop.run("Uppercase x forever", max_steps=10))
        self.assertIn("progress", result.lower())

    def test_different_args_not_counted_together(self) -> None:
        """Calls with different arguments are tracked separately — not treated as repeats."""
        responses = [
            '<tool>upper</tool>\n<input>{"text": "a"}</input>',
            '<tool>upper</tool>\n<input>{"text": "b"}</input>',
            '<tool>upper</tool>\n<input>{"text": "c"}</input>',
            "<answer>Done: A B C.</answer>",
        ]
        loop = self._loop(responses, [_UpperTool()])
        result = "".join(loop.run("Uppercase a, b, c", max_steps=10))
        self.assertEqual(result, "Done: A B C.")


class TestTruthfulnessGuard(unittest.TestCase):
    """guard_answer_truthfulness rewrites answers that falsely claim web/calendar lookups.

    When the agent produces an answer claiming a search was performed but no
    relevant tool was called, a disclaimer is appended.
    """

    def _loop(self, responses: list[str]) -> AgentLoop:
        return AgentLoop(
            runtime=_ScriptedRuntime(responses),
            tokenizer=_mock_tokenizer(),
            registry=_registry(),
            settings=config.get(),
        )

    def test_false_web_claim_gets_disclaimer(self) -> None:
        """Answer claiming 'I searched' without any web tool adds a note."""
        loop = self._loop(["<answer>I searched the web and found that Python 4 is out.</answer>"])
        result = "".join(loop.run("Has Python 4 been released?"))
        self.assertIn("Note:", result)
        self.assertIn("training knowledge", result)

    def test_real_web_search_no_disclaimer(self) -> None:
        """When web_search was actually called, no disclaimer is added to the answer."""
        responses = [
            '<tool>web_search</tool>\n<input>{"query": "Python 4"}</input>',
            "<answer>I searched and found Python 4 is not out yet.</answer>",
        ]

        class _FakeWebSearchTool(Tool):
            name = "web_search"
            description = "Search"
            parameters: dict = {}

            def execute(self, **kwargs: Any) -> ToolResult:
                return ToolResult(tool_name=self.name, content="No results for Python 4.")

        loop = AgentLoop(
            runtime=_ScriptedRuntime(responses),
            tokenizer=_mock_tokenizer(),
            registry=_registry(_FakeWebSearchTool()),
            settings=config.get(),
        )
        result = "".join(loop.run("Python 4 status?"))
        self.assertNotIn("Note:", result)

    def test_plain_answer_unchanged(self) -> None:
        """Answers making no external claims pass through without modification."""
        loop = self._loop(["<answer>The capital of France is Paris.</answer>"])
        result = "".join(loop.run("Capital of France?"))
        self.assertEqual(result, "The capital of France is Paris.")


class TestAgentLoopContextInjection(unittest.TestCase):
    """Entity context block passed to run() appears in the system prompt.

    BrainService passes the user's known facts/tasks as a context string.
    This test verifies that string ends up in the first step's prompt so
    the model can reference it when reasoning.
    """

    def test_context_in_prompt(self) -> None:
        """The context kwarg is included in the first prompt's system message."""
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
