from __future__ import annotations

import unittest
from datetime import date, timedelta

from core.second_brain.extractor import EntityExtractor


class EntityExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = EntityExtractor()

    def test_extract_profile_location(self) -> None:
        entities = self.extractor.extract("I live in Turku Finland")
        profiles = [e for e in entities if e.kind == "profile"]
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].key, "location")
        self.assertIn("Turku", profiles[0].value)

    def test_extract_task_remind_me(self) -> None:
        entities = self.extractor.extract("remind me to review the PR tomorrow")
        tasks = [e for e in entities if e.kind == "task"]
        self.assertEqual(len(tasks), 1)
        self.assertIn("review", tasks[0].value.lower())
        self.assertIsNotNone(tasks[0].due_date)

    def test_extract_commitment_meeting(self) -> None:
        entities = self.extractor.extract("I have a meeting with the team tomorrow at 9am")
        commitments = [e for e in entities if e.kind == "commitment"]
        self.assertEqual(len(commitments), 1)
        self.assertIsNotNone(commitments[0].due_date)

    def test_extract_preference_likes(self) -> None:
        entities = self.extractor.extract("I prefer concise answers")
        prefs = [e for e in entities if e.kind == "preference"]
        self.assertEqual(len(prefs), 1)
        self.assertIn("concise", prefs[0].value.lower())

    def test_extract_returns_empty_for_generic_chat(self) -> None:
        entities = self.extractor.extract("What is the capital of France?")
        self.assertEqual(entities, [])

    def test_extract_date_tomorrow_resolves_correctly(self) -> None:
        entities = self.extractor.extract("remind me to call John tomorrow")
        tasks = [e for e in entities if e.kind == "task"]
        self.assertEqual(len(tasks), 1)
        expected = (date.today() + timedelta(days=1)).isoformat()
        self.assertEqual(tasks[0].due_date, expected)


if __name__ == "__main__":
    unittest.main()
