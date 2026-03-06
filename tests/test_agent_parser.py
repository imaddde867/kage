"""Unit tests for core/agent/parser.py — no LLM required.

All tests call parse_step() with pre-written XML strings and assert on the
returned ParsedStep fields.  No model, no network, no file I/O.

Coverage checklist:
    - <answer> only
    - <answer> + <thought>
    - <answer> spanning multiple lines (re.DOTALL required)
    - <tool> + <thought> with JSON args
    - <tool> without <thought>
    - <tool> with nested JSON args
    - Malformed JSON args → empty dict (no crash)
    - Plain text with no XML → used as answer verbatim
    - Empty string → (None, None)
    - Whitespace-only → (None, None)
    - Both <answer> and <tool> present → <answer> wins (priority rule)
    - <tool> present but <input> tag missing → args = {}
    - Return type is always ParsedStep
"""
import unittest

from core.agent.parser import ParsedStep, parse_step


class TestParseStep(unittest.TestCase):
    """Verify parse_step() handles all XML patterns the agent loop produces."""

    # --- final answer ---

    def test_answer_only(self):
        """A bare <answer> tag with no other tags yields the answer text."""
        raw = "<answer>The sky is blue because of Rayleigh scattering.</answer>"
        step = parse_step(raw)
        self.assertIsNone(step.tool_call)
        self.assertEqual(step.answer, "The sky is blue because of Rayleigh scattering.")
        self.assertIsNone(step.thought)

    def test_answer_with_thought(self):
        """<thought> is captured separately; <answer> is still the final result."""
        raw = "<thought>I already know this.</thought>\n<answer>Paris is the capital.</answer>"
        step = parse_step(raw)
        self.assertIsNone(step.tool_call)
        self.assertEqual(step.answer, "Paris is the capital.")
        self.assertEqual(step.thought, "I already know this.")

    def test_answer_multiline(self):
        """Newlines inside <answer> are preserved (requires re.DOTALL in regex)."""
        raw = "<answer>First line.\nSecond line.</answer>"
        step = parse_step(raw)
        self.assertEqual(step.answer, "First line.\nSecond line.")

    # --- tool calls ---

    def test_tool_call_with_thought(self):
        """Full tool-call step: thought + tool name + JSON args all parsed correctly."""
        raw = (
            "<thought>I should search for this.</thought>\n"
            "<tool>web_search</tool>\n"
            '<input>{"query": "Python 3.13 release"}</input>'
        )
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.name, "web_search")
        self.assertEqual(step.tool_call.args["query"], "Python 3.13 release")
        self.assertEqual(step.thought, "I should search for this.")
        self.assertIsNone(step.answer)

    def test_tool_call_no_thought(self):
        """A tool call without a <thought> tag — thought field is None."""
        raw = "<tool>list_open_tasks</tool>\n<input>{}</input>"
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.name, "list_open_tasks")
        self.assertEqual(step.tool_call.args, {})
        self.assertIsNone(step.thought)

    def test_tool_call_nested_json(self):
        """Multi-field JSON args are parsed into a dict with all keys intact."""
        raw = '<tool>update_fact</tool>\n<input>{"kind": "task", "key": "report", "value": "finish Q1 report"}</input>'
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.args["kind"], "task")
        self.assertEqual(step.tool_call.args["key"], "report")

    # --- fallback / malformed ---

    def test_malformed_json_args_defaults_to_empty_dict(self):
        """Invalid JSON inside <input> must not crash — falls back to {} args."""
        raw = "<tool>web_search</tool>\n<input>not valid json</input>"
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.args, {})

    def test_no_xml_plain_text(self):
        """Raw text with no XML tags is used as the answer verbatim (plain-text fallback)."""
        raw = "Sorry, I could not find any information."
        step = parse_step(raw)
        self.assertIsNone(step.tool_call)
        self.assertEqual(step.answer, raw)

    def test_empty_string_returns_none_answer(self):
        """Empty model output → both tool_call and answer are None."""
        step = parse_step("")
        self.assertIsNone(step.tool_call)
        self.assertIsNone(step.answer)

    def test_whitespace_only_returns_none_answer(self):
        """Whitespace-only output strips to empty → both fields are None."""
        step = parse_step("   \n  ")
        self.assertIsNone(step.tool_call)
        self.assertIsNone(step.answer)

    def test_answer_takes_priority_over_tool(self):
        """When both <answer> and <tool> are present, <answer> wins (no tool call made)."""
        raw = (
            "<tool>web_search</tool>\n"
            "<input>{}</input>\n"
            "<answer>Actually, I already know.</answer>"
        )
        step = parse_step(raw)
        self.assertIsNone(step.tool_call)
        self.assertEqual(step.answer, "Actually, I already know.")

    def test_tool_without_input_tag(self):
        """<tool> present but <input> missing → ToolCall created with empty args dict."""
        raw = "<tool>list_open_tasks</tool>"
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.name, "list_open_tasks")
        self.assertEqual(step.tool_call.args, {})

    def test_web_fetch_with_json_input(self):
        raw = '<tool>web_fetch</tool><input>{"url":"https://x"}</input>'
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.name, "web_fetch")
        self.assertEqual(step.tool_call.args.get("url"), "https://x")

    def test_tool_body_inline_attrs(self):
        raw = '<tool>web_fetch url="https://x"</tool>'
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.name, "web_fetch")
        self.assertEqual(step.tool_call.args.get("url"), "https://x")

    def test_tool_name_attribute_form(self):
        raw = '<tool name="web_search" query="agadir things to do" />'
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.name, "web_search")
        self.assertEqual(step.tool_call.args.get("query"), "agadir things to do")

    def test_tool_open_inline_without_closing_tag(self):
        raw = '<tool>web_fetch url="https://example.com">'
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.name, "web_fetch")
        self.assertEqual(step.tool_call.args.get("url"), "https://example.com")

    def test_malformed_tool_like_output_routes_to_invalid_tool_call(self):
        raw = "<tool name=>"
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.name, "invalid_tool_call")
        self.assertIn("raw", step.tool_call.args)

    def test_web_fetch_without_input_yields_empty_args_for_registry_validation(self):
        step = parse_step("<tool>web_fetch</tool>")
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.name, "web_fetch")
        self.assertEqual(step.tool_call.args, {})

    def test_json_tool_envelope_with_prefix_and_extra_brace_is_recovered(self):
        raw = (
            'Tool call:\n'
            '{"type":"tool","thought":"run shell","tool":"shell","args":{"command":"uname -a"}}}'
        )
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        assert step.tool_call is not None
        self.assertEqual(step.tool_call.name, "shell")
        self.assertEqual(step.tool_call.args.get("command"), "uname -a")

    def test_json_answer_envelope_inside_code_fence(self):
        raw = '```json\n{"type":"answer","answer":"ok"}\n```'
        step = parse_step(raw)
        self.assertIsNone(step.tool_call)
        self.assertEqual(step.answer, "ok")

    def test_return_type_is_parsed_step(self):
        """parse_step always returns a ParsedStep dataclass instance."""
        self.assertIsInstance(parse_step("<answer>hi</answer>"), ParsedStep)


if __name__ == "__main__":
    unittest.main()
