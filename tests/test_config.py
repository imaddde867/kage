"""Unit tests for config module."""

import unittest
from unittest.mock import patch


class TestConfig(unittest.TestCase):
    def test_get_returns_settings(self) -> None:
        import config

        settings = config.get()
        self.assertIsNotNone(settings)
        self.assertTrue(hasattr(settings, "llm_backend"))
        self.assertTrue(hasattr(settings, "shell_confirm_token"))

    def test_get_is_cached(self) -> None:
        import config

        settings1 = config.get()
        settings2 = config.get()
        self.assertIs(settings1, settings2)

    def test_clear_cache_invalidates(self) -> None:
        import config

        settings1 = config.get()
        config.clear_cache()
        settings2 = config.get()
        self.assertIsNot(settings1, settings2)

    def test_reload_returns_new_settings(self) -> None:
        import config

        settings1 = config.get()
        settings2 = config.reload()
        self.assertIsNot(settings1, settings2)
        settings3 = config.get()
        self.assertIs(settings2, settings3)

    def test_shell_confirm_token_has_default(self) -> None:
        import config

        settings = config.get()
        self.assertTrue(settings.shell_confirm_token)
        self.assertEqual(
            settings.shell_confirm_token, "YES_I_UNDERSTAND_LOCAL_MUTATION"
        )

    def test_shell_confirm_token_from_env(self) -> None:
        import config

        config.clear_cache()
        with patch.dict("os.environ", {"SHELL_CONFIRM_TOKEN": "custom_token"}):
            settings = config.get()
            self.assertEqual(settings.shell_confirm_token, "custom_token")
        config.clear_cache()
