from __future__ import annotations

import time
import unittest

from core.second_brain.planner import IntentRouter


class IntentRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = IntentRouter()

    def test_classify_task_capture(self) -> None:
        route = self.router.classify("remind me to submit the report by Friday")
        self.assertEqual(route.intent, "TASK_CAPTURE")
        self.assertTrue(route.inject_entities)
        self.assertTrue(route.should_extract)
        self.assertFalse(route.proactive_ok)

    def test_classify_planning_request_injects_entities(self) -> None:
        route = self.router.classify("what should I work on next?")
        self.assertEqual(route.intent, "PLANNING_REQUEST")
        self.assertTrue(route.inject_entities)
        self.assertTrue(route.proactive_ok)

    def test_classify_factual_question_no_injection(self) -> None:
        route = self.router.classify("What is the capital of France?")
        self.assertEqual(route.intent, "GENERAL")
        self.assertFalse(route.inject_entities)
        self.assertFalse(route.proactive_ok)

    def test_classify_commitment_proactive_ok(self) -> None:
        route = self.router.classify("I have a meeting with the team tomorrow")
        self.assertEqual(route.intent, "COMMITMENT")
        self.assertTrue(route.proactive_ok)
        self.assertTrue(route.inject_entities)

    def test_classify_what_do_you_know_is_recall(self) -> None:
        route = self.router.classify("What do you know about me?")
        self.assertEqual(route.intent, "RECALL_REQUEST")
        self.assertTrue(route.inject_entities)

    def test_classify_runs_under_1ms(self) -> None:
        iterations = 100
        t0 = time.perf_counter()
        for _ in range(iterations):
            self.router.classify("remind me to review the PR tomorrow")
        elapsed_ms = (time.perf_counter() - t0) * 1000 / iterations
        self.assertLess(elapsed_ms, 1.0)


if __name__ == "__main__":
    unittest.main()
