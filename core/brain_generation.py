from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import config

_BACKEND_MLX = "mlx"
_BACKEND_MLX_VLM = "mlx_vlm"


class GenerationRuntime:
    def __init__(self, *, settings: config.Settings) -> None:
        self.settings = settings
        self.backend = self.settings.llm_backend.strip().lower()
        self.last_stats: dict[str, Any] = {}
        self.tokenizer: Any | None = None

        if self.backend == _BACKEND_MLX_VLM:
            self._init_mlx_vlm()
        elif self.backend == _BACKEND_MLX:
            self._init_mlx()
        else:
            raise ValueError(
                f"Unsupported LLM_BACKEND '{self.settings.llm_backend}'. "
                f"Use one of: {_BACKEND_MLX_VLM}, {_BACKEND_MLX}."
            )

    def _init_mlx_vlm(self) -> None:
        from mlx_vlm import stream_generate  # type: ignore[import]
        from mlx_vlm.tokenizer_utils import load_tokenizer as load_detokenizer  # type: ignore[import]
        from mlx_vlm.utils import StoppingCriteria  # type: ignore[import]
        from mlx_vlm.utils import get_model_path, load_model  # type: ignore[import]
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging

        model_path = get_model_path(self.settings.mlx_model)
        print(f"  Loading {self.settings.mlx_model}…", flush=True)
        model = load_model(model_path)

        prev_hf_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        finally:
            hf_logging.set_verbosity(prev_hf_level)

        detokenizer_class = load_detokenizer(model_path, return_tokenizer=False)
        tokenizer.detokenizer = detokenizer_class(tokenizer)

        eos_ids = getattr(model.config, "eos_token_id", None) or tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        tokenizer.stopping_criteria = StoppingCriteria(eos_ids, tokenizer)

        self._vlm_model = model
        self._vlm_processor = tokenizer
        self._vlm_stream = stream_generate
        self.tokenizer = tokenizer

    def _init_mlx(self) -> None:
        from mlx_lm import load, stream_generate  # type: ignore[import]

        print(f"  Loading {self.settings.mlx_model}…", flush=True)
        model, tokenizer = load(self.settings.mlx_model)
        self._mlx_model = model
        self._mlx_stream = stream_generate

        self._mlx_draft_model = None
        if self.settings.mlx_draft_model:
            print(f"  Loading draft model {self.settings.mlx_draft_model}…", flush=True)
            draft_model, _ = load(self.settings.mlx_draft_model)
            self._mlx_draft_model = draft_model

        self.tokenizer = tokenizer

    def warmup(self, prompt: str, *, max_tokens: int = 5) -> None:
        print("  Warming up (compiling MLX graphs)…", flush=True)
        for _ in self.stream_raw(prompt, max_tokens=max_tokens):
            pass
        print("  Ready.\n", flush=True)

    def stream_raw(self, prompt: str, *, max_tokens: int | None = None) -> Iterator[str]:
        tokens = max_tokens if max_tokens is not None else self.settings.mlx_max_tokens
        if self.backend == _BACKEND_MLX_VLM:
            gen_iter = iter(
                self._vlm_stream(
                    self._vlm_model,
                    self._vlm_processor,
                    prompt=prompt,
                    max_tokens=tokens,
                    temperature=self.settings.temperature,
                )
            )
        elif self.backend == _BACKEND_MLX:
            gen_iter = iter(
                self._mlx_stream(
                    self._mlx_model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=tokens,
                    draft_model=self._mlx_draft_model,
                    temp=self.settings.temperature,
                )
            )
        else:
            raise RuntimeError(f"Unsupported LLM backend: {self.backend}")

        total_tokens = 0
        pure_gen_s = 0.0
        try:
            while True:
                t_tok = time.perf_counter()
                try:
                    chunk = next(gen_iter)
                except StopIteration:
                    break
                pure_gen_s += time.perf_counter() - t_tok
                text = chunk.text if hasattr(chunk, "text") else str(chunk)
                total_tokens = getattr(chunk, "generation_tokens", total_tokens + 1)
                if text:
                    yield text
        finally:
            self.last_stats.clear()
            self.last_stats.update(
                {
                    "backend": self.backend,
                    "tokens": total_tokens,
                    "gen_seconds": pure_gen_s,
                    "tok_per_sec": total_tokens / pure_gen_s if pure_gen_s > 0 else 0.0,
                }
            )

