"""Unit tests for core/agent/parser.py — no LLM required."""
import unittest

from core.agent.parser import ParsedStep, parse_step


class TestParseStep(unittest.TestCase):
    # --- final answer ---

    def test_answer_only(self):
        raw = "<answer>The sky is blue because of Rayleigh scattering.</answer>"
        step = parse_step(raw)
        self.assertIsNone(step.tool_call)
        self.assertEqual(step.answer, "The sky is blue because of Rayleigh scattering.")
        self.assertIsNone(step.thought)

    def test_answer_with_thought(self):
        raw = "<thought>I already know this.</thought>\n<answer>Paris is the capital.</answer>"
        step = parse_step(raw)
        self.assertIsNone(step.tool_call)
        self.assertEqual(step.answer, "Paris is the capital.")
        self.assertEqual(step.thought, "I already know this.")

    def test_answer_multiline(self):
        raw = "<answer>First line.\nSecond line.</answer>"
        step = parse_step(raw)
        self.assertEqual(step.answer, "First line.\nSecond line.")

    # --- tool calls ---

    def test_tool_call_with_thought(self):
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
        raw = "<tool>list_open_tasks</tool>\n<input>{}</input>"
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.name, "list_open_tasks")
        self.assertEqual(step.tool_call.args, {})
        self.assertIsNone(step.thought)

    def test_tool_call_nested_json(self):
        raw = '<tool>update_fact</tool>\n<input>{"kind": "task", "key": "report", "value": "finish Q1 report"}</input>'
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.args["kind"], "task")
        self.assertEqual(step.tool_call.args["key"], "report")

    # --- fallback / malformed ---

    def test_malformed_json_args_defaults_to_empty_dict(self):
        raw = "<tool>web_search</tool>\n<input>not valid json</input>"
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.args, {})

    def test_no_xml_plain_text(self):
        raw = "Sorry, I could not find any information."
        step = parse_step(raw)
        self.assertIsNone(step.tool_call)
        self.assertEqual(step.answer, raw)

    def test_empty_string_returns_none_answer(self):
        step = parse_step("")
        self.assertIsNone(step.tool_call)
        self.assertIsNone(step.answer)

    def test_whitespace_only_returns_none_answer(self):
        step = parse_step("   \n  ")
        self.assertIsNone(step.tool_call)
        self.assertIsNone(step.answer)

    def test_answer_takes_priority_over_tool(self):
        # If both are present (malformed output), <answer> wins
        raw = (
            "<tool>web_search</tool>\n"
            "<input>{}</input>\n"
            "<answer>Actually, I already know.</answer>"
        )
        step = parse_step(raw)
        self.assertIsNone(step.tool_call)
        self.assertEqual(step.answer, "Actually, I already know.")

    def test_tool_without_input_tag(self):
        # Tool present but no <input> — args should be empty dict
        raw = "<tool>list_open_tasks</tool>"
        step = parse_step(raw)
        self.assertIsNotNone(step.tool_call)
        self.assertEqual(step.tool_call.name, "list_open_tasks")
        self.assertEqual(step.tool_call.args, {})

    def test_return_type_is_parsed_step(self):
        self.assertIsInstance(parse_step("<answer>hi</answer>"), ParsedStep)


if __name__ == "__main__":
    unittest.main()
