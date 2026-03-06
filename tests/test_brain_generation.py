import unittest

from core.brain_generation import GenerationRuntime, _is_vlm_checkpoint_mismatch


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text


class TestMlxStreamCompatibility(unittest.TestCase):
    def _runtime_stub(self, stream_fn):
        runtime = GenerationRuntime.__new__(GenerationRuntime)
        runtime._mlx_model = object()  # type: ignore[attr-defined]
        runtime.tokenizer = object()  # type: ignore[attr-defined]
        runtime._mlx_draft_model = None  # type: ignore[attr-defined]
        runtime._mlx_stream = stream_fn  # type: ignore[attr-defined]
        runtime._mlx_temperature_arg = None  # type: ignore[attr-defined]
        runtime._mlx_make_sampler = None  # type: ignore[attr-defined]
        return runtime

    def test_prefers_sampler_when_available(self) -> None:
        captured = {}

        def make_sampler(temp):  # noqa: ANN001
            captured["temp"] = temp

            def sampler_fn(logits):  # noqa: ANN001
                return logits

            captured["sampler"] = sampler_fn
            return sampler_fn

        def stream_fn(model, tokenizer, *, prompt, max_tokens, draft_model, sampler):  # noqa: ANN001
            _ = (model, tokenizer, prompt, max_tokens, draft_model)
            assert sampler is captured["sampler"]
            yield _Chunk("ok")

        runtime = self._runtime_stub(stream_fn)
        runtime._mlx_make_sampler = make_sampler  # type: ignore[attr-defined]
        chunks = list(runtime._iter_mlx_stream(prompt="hi", max_tokens=8, temperature=0.2))
        self.assertEqual([c.text for c in chunks], ["ok"])
        self.assertEqual(captured["temp"], 0.2)
        self.assertEqual(runtime._mlx_temperature_arg, "sampler")

    def test_prefers_temperature_when_supported(self) -> None:
        def stream_fn(model, tokenizer, *, prompt, max_tokens, draft_model, temperature):  # noqa: ANN001
            _ = (model, tokenizer, prompt, max_tokens, draft_model, temperature)
            yield _Chunk("ok")

        runtime = self._runtime_stub(stream_fn)
        chunks = list(runtime._iter_mlx_stream(prompt="hi", max_tokens=8, temperature=0.2))
        self.assertEqual([c.text for c in chunks], ["ok"])
        self.assertEqual(runtime._mlx_temperature_arg, "temperature")

    def test_falls_back_to_temp_when_temperature_is_rejected(self) -> None:
        def stream_fn(model, tokenizer, *, prompt, max_tokens, draft_model, temp):  # noqa: ANN001
            _ = (model, tokenizer, prompt, max_tokens, draft_model, temp)
            yield _Chunk("ok")

        runtime = self._runtime_stub(stream_fn)
        chunks = list(runtime._iter_mlx_stream(prompt="hi", max_tokens=8, temperature=0.2))
        self.assertEqual([c.text for c in chunks], ["ok"])
        self.assertEqual(runtime._mlx_temperature_arg, "temp")

    def test_non_keyword_typeerror_is_not_suppressed(self) -> None:
        def stream_fn(*args, **kwargs):  # noqa: ANN001
            _ = (args, kwargs)
            raise TypeError("boom")
            yield  # pragma: no cover

        runtime = self._runtime_stub(stream_fn)
        with self.assertRaisesRegex(TypeError, "boom"):
            list(runtime._iter_mlx_stream(prompt="hi", max_tokens=8, temperature=0.2))


class TestMlxBackendMismatchHint(unittest.TestCase):
    def test_detects_vision_tower_weight_mismatch(self) -> None:
        exc = ValueError(
            "Received 333 parameters not in model: language_model.vision_tower.blocks.0.attn.qkv.weight"
        )
        self.assertTrue(_is_vlm_checkpoint_mismatch(exc))

    def test_non_vlm_value_error_not_marked_as_mismatch(self) -> None:
        exc = ValueError("unknown tensor dtype")
        self.assertFalse(_is_vlm_checkpoint_mismatch(exc))


if __name__ == "__main__":
    unittest.main()
