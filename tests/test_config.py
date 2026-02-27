import unittest

import config


class ConfigImportTests(unittest.TestCase):
    def setUp(self) -> None:
        config.get_settings.cache_clear()

    def tearDown(self) -> None:
        config.get_settings.cache_clear()

    def test_defaults_load_without_dotenv_dependency(self) -> None:
        settings = config.get_settings()
        self.assertEqual(settings.ollama_base_url, "http://localhost:11434")
        self.assertEqual(settings.ollama_model, "qwen3:8b")


if __name__ == "__main__":
    unittest.main()
