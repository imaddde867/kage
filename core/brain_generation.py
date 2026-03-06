from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import config

_BACKEND_MLX = "mlx"
_BACKEND_MLX_VLM = "mlx_vlm"
logger = logging.getLogger(__name__)


def _is_vlm_checkpoint_mismatch(exc: Exception) -> bool:
    """Detect common MLX-LM load failures caused by VLM checkpoints."""
    message = str(exc).lower()
    if "vision_tower" in message:
        return True
    return "parameters not in model" in message and "language_model." in message


class GenerationRuntime:
    def __init__(self, *, settings: config.Settings) -> None:
        self.settings = settings
        self.backend = self.settings.llm_backend.strip().lower()
        self.last_stats: dict[str, Any] = {}
        self.tokenizer: Any | None = None
        self._mlx_temperature_arg: str | None = None
        self._mlx_make_sampler: Any | None = None

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
        logger.info("Loading model %s", self.settings.mlx_model)
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
        try:
            from mlx_lm.sample_utils import make_sampler  # type: ignore[import]
        except Exception:
            make_sampler = None

        logger.info("Loading model %s", self.settings.mlx_model)
        try:
            model, tokenizer = load(self.settings.mlx_model)
        except ValueError as exc:
            if _is_vlm_checkpoint_mismatch(exc):
                raise ValueError(
                    f"Model '{self.settings.mlx_model}' appears to be a VLM checkpoint "
                    "but LLM_BACKEND=mlx uses mlx_lm (text-only). "
                    "Set LLM_BACKEND=mlx_vlm for this model, or choose a text-only model for mlx."
                ) from exc
            raise
        self._mlx_model = model
        self._mlx_stream = stream_generate
        self._mlx_make_sampler = make_sampler

        self._mlx_draft_model = None
        if self.settings.mlx_draft_model:
            logger.info("Loading draft model %s", self.settings.mlx_draft_model)
            draft_model, _ = load(self.settings.mlx_draft_model)
            self._mlx_draft_model = draft_model

        self.tokenizer = tokenizer

    def warmup(self, prompt: str, *, max_tokens: int = 5) -> None:
        logger.info("Warming up runtime")
        for _ in self.stream_raw(prompt, max_tokens=max_tokens):
            pass
        logger.info("Runtime ready")

    def _iter_mlx_stream(self, *, prompt: str, max_tokens: int, temperature: float) -> Iterator[Any]:
        """Yield chunks from mlx_lm stream_generate across API variants."""
        preferred_args: list[str] = []
        if self._mlx_temperature_arg in {"sampler", "temperature", "temp"}:
            preferred_args.append(self._mlx_temperature_arg)
        for candidate in ("sampler", "temperature", "temp"):
            if candidate not in preferred_args:
                preferred_args.append(candidate)

        last_kw_error: TypeError | None = None
        base_kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "draft_model": self._mlx_draft_model,
        }
        sampler = None
        if callable(self._mlx_make_sampler):
            sampler = self._mlx_make_sampler(temperature)

        for temp_arg in preferred_args:
            kwargs = dict(base_kwargs)
            if temp_arg == "sampler":
                if sampler is None:
                    continue
                kwargs[temp_arg] = sampler
            else:
                kwargs[temp_arg] = temperature
            try:
                gen_iter = iter(
                    self._mlx_stream(
                        self._mlx_model,
                        self.tokenizer,
                        **kwargs,
                    )
                )
                first = next(gen_iter)
            except StopIteration:
                self._mlx_temperature_arg = temp_arg
                return
            except TypeError as exc:
                message = str(exc)
                if "unexpected keyword argument" in message and f"'{temp_arg}'" in message:
                    last_kw_error = exc
                    continue
                raise

            self._mlx_temperature_arg = temp_arg
            yield first
            yield from gen_iter
            return

        if last_kw_error is not None:
            raise last_kw_error
        raise RuntimeError("Unable to initialize MLX stream generation.")

    def stream_raw(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        track_stats: bool = True,
        temperature: float | None = None,
    ) -> Iterator[str]:
        tokens = max_tokens if max_tokens is not None else self.settings.mlx_max_tokens
        temp = temperature if temperature is not None else self.settings.temperature
        if self.backend == _BACKEND_MLX_VLM:
            gen_iter = iter(
                self._vlm_stream(
                    self._vlm_model,
                    self._vlm_processor,
                    prompt=prompt,
                    max_tokens=tokens,
                    temperature=temp,
                )
            )
        elif self.backend == _BACKEND_MLX:
            gen_iter = iter(
                self._iter_mlx_stream(
                    prompt=prompt,
                    max_tokens=tokens,
                    temperature=temp,
                )
            )
        else:
            raise RuntimeError(f"Unsupported LLM backend: {self.backend}")

        total_tokens = 0
        pure_gen_s = 0.0
        prompt_tokens: int | None = None
        prompt_tps: float | None = None
        generation_tps: float | None = None
        peak_memory_gb: float | None = None
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
                if hasattr(chunk, "prompt_tokens"):
                    value = getattr(chunk, "prompt_tokens", None)
                    if isinstance(value, int):
                        prompt_tokens = value
                if hasattr(chunk, "prompt_tps"):
                    value = getattr(chunk, "prompt_tps", None)
                    if isinstance(value, (int, float)):
                        prompt_tps = float(value)
                if hasattr(chunk, "generation_tps"):
                    value = getattr(chunk, "generation_tps", None)
                    if isinstance(value, (int, float)):
                        generation_tps = float(value)
                if hasattr(chunk, "peak_memory"):
                    value = getattr(chunk, "peak_memory", None)
                    if isinstance(value, (int, float)):
                        peak_memory_gb = float(value)
                if text:
                    yield text
        finally:
            if track_stats:
                self.last_stats.clear()
                self.last_stats.update(
                    {
                        "backend": self.backend,
                        "tokens": total_tokens,
                        "gen_seconds": pure_gen_s,
                        "tok_per_sec": total_tokens / pure_gen_s if pure_gen_s > 0 else 0.0,
                    }
                )
                if prompt_tokens is not None:
                    self.last_stats["prompt_tokens"] = prompt_tokens
                if prompt_tps is not None:
                    self.last_stats["prompt_tps"] = prompt_tps
                if generation_tps is not None:
                    self.last_stats["generation_tps"] = generation_tps
                if peak_memory_gb is not None:
                    self.last_stats["peak_memory_gb"] = peak_memory_gb
