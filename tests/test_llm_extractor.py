from __future__ import annotations

import unittest

from core.second_brain.llm_extractor import LLMEntityExtractor


class LLMExtractorParseTests(unittest.TestCase):
    """Tests for the _parse() method only — no LLM inference required."""

    def _parse(self, raw: str):
        # Build a minimal stub — _parse doesn't touch runtime or tokenizer
        extractor = LLMEntityExtractor.__new__(LLMEntityExtractor)
        return extractor._parse(raw)

    # --- happy-path cases ---

    def test_parse_task_with_due_date(self):
        raw = '[{"kind":"task","key":"review_pr","value":"review the PR","due_date":"2026-03-05"}]'
        entities = self._parse(raw)
        self.assertEqual(len(entities), 1)
        e = entities[0]
        self.assertEqual(e.kind, "task")
        self.assertEqual(e.key, "review_pr")
        self.assertEqual(e.value, "review the PR")
        self.assertEqual(e.due_date, "2026-03-05")

    def test_parse_profile_entity(self):
        raw = '[{"kind":"profile","key":"location","value":"Turku Finland","due_date":null}]'
        entities = self._parse(raw)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].kind, "profile")
        self.assertEqual(entities[0].value, "Turku Finland")
        self.assertIsNone(entities[0].due_date)

    def test_parse_preference_entity(self):
        raw = '[{"kind":"preference","key":"diet","value":"vegetarian","due_date":null}]'
        entities = self._parse(raw)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].kind, "preference")

    def test_parse_multiple_entities(self):
        raw = (
            '[{"kind":"task","key":"submit_report","value":"submit the quarterly report","due_date":"2026-03-07"},'
            '{"kind":"profile","key":"location","value":"Helsinki","due_date":null}]'
        )
        entities = self._parse(raw)
        self.assertEqual(len(entities), 2)

    # --- empty / no-extract cases ---

    def test_parse_empty_array_returns_empty_list(self):
        self.assertEqual(self._parse("[]"), [])

    def test_parse_whitespace_only_returns_empty_list(self):
        self.assertEqual(self._parse("   "), [])

    def test_parse_no_array_returns_empty_list(self):
        self.assertEqual(self._parse("I have nothing to extract from this."), [])

    # --- robustness / malformed output ---

    def test_parse_invalid_json_returns_empty_list(self):
        self.assertEqual(self._parse("[{bad json}]"), [])

    def test_parse_markdown_fences_stripped(self):
        raw = "```json\n[{\"kind\":\"task\",\"key\":\"write_tests\",\"value\":\"write unit tests\",\"due_date\":null}]\n```"
        entities = self._parse(raw)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].key, "write_tests")

    def test_parse_prose_before_array_ignored(self):
        raw = 'Sure! Here are the entities: [{"kind":"commitment","key":"standup","value":"daily standup","due_date":"2026-03-05"}]'
        entities = self._parse(raw)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].kind, "commitment")

    def test_parse_invalid_kind_skipped(self):
        raw = '[{"kind":"unknown","key":"foo","value":"bar","due_date":null}]'
        self.assertEqual(self._parse(raw), [])

    def test_parse_missing_key_skipped(self):
        raw = '[{"kind":"task","key":"","value":"something","due_date":null}]'
        self.assertEqual(self._parse(raw), [])

    def test_parse_missing_value_skipped(self):
        raw = '[{"kind":"task","key":"my_task","value":"","due_date":null}]'
        self.assertEqual(self._parse(raw), [])

    def test_parse_invalid_due_date_format_stored_as_none(self):
        raw = '[{"kind":"task","key":"my_task","value":"do something","due_date":"next Friday"}]'
        entities = self._parse(raw)
        self.assertEqual(len(entities), 1)
        self.assertIsNone(entities[0].due_date)

    def test_parse_non_dict_items_skipped(self):
        raw = '[null, "string", {"kind":"profile","key":"name","value":"Imad","due_date":null}]'
        entities = self._parse(raw)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].key, "name")

    def test_parse_key_truncated_to_60_chars(self):
        long_key = "a" * 80
        raw = f'[{{"kind":"profile","key":"{long_key}","value":"test","due_date":null}}]'
        entities = self._parse(raw)
        self.assertEqual(len(entities), 1)
        self.assertEqual(len(entities[0].key), 60)


if __name__ == "__main__":
    unittest.main()
