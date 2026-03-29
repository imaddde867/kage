"""Local LLM inference wrapper — MLX.

Wraps mlx_vlm (for VLM checkpoints like Qwen3.5) and provides the same
stream_raw() interface as Kage's GenerationRuntime so it can be swapped
in via LLM_BACKEND config without touching brain.py.

Model:          mlx-community/Qwen3.5-9B-MLX-4bit  (VLM, pre-quantized)
Expected perf:  ~15–25 tok/s, <2s TTFT for typical queries on M4 16GB
RAM footprint:  ~4.5 GB weights (well within 16 GB)

Backend selection:
  Qwen3.5 is a VLM architecture — mlx_vlm is required.
  mlx_lm would raise "vision_tower / parameters not in model" on load.
  LocalLLM detects this and always uses mlx_vlm for this model family.

Interface contract (matching GenerationRuntime.stream_raw):
    stream_raw(prompt, *, max_tokens, temperature) -> Iterator[str]

Extras:
    chat(messages, ...)  — apply chat template then stream
    embed(text)          — mean-pool token embeddings (no separate model)
    last_stats           — tokens, tok_per_sec, ram_delta_gb

# TODO: revisit KV cache compression when MLX adds cache hooks
#       (see git history for turbo_quant.py — PolarQuant + QJL impl).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Pre-quantized 4-bit MLX build — no conversion needed, ~4.5 GB on disk.
DEFAULT_MODEL_ID = "mlx-community/Qwen3.5-9B-MLX-4bit"


class LocalLLM:
    """MLX VLM-backed local LLM.

    Lazy-loads the model on first call so import is always fast.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.last_stats: dict[str, Any] = {}

        self._model: Any = None
        self._processor: Any = None       # mlx_vlm tokenizer/processor
        self._tokenizer: Any = None       # HF tokenizer for chat template
        self._stream_generate: Any = None
        self._stopping_criteria: Any = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Lazy-load model weights into unified memory. Idempotent."""
        if self._loaded:
            return

        from mlx_vlm.utils import get_model_path, load_model  # type: ignore[import]
        from mlx_vlm.tokenizer_utils import load_tokenizer as load_detokenizer  # type: ignore[import]
        from mlx_vlm.utils import StoppingCriteria  # type: ignore[import]
        from mlx_vlm import stream_generate  # type: ignore[import]
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging

        logger.info("[LocalLLM] Loading %s", self.model_id)
        ram_before = _ram_gb()
        t0 = time.perf_counter()

        # get_model_path() only accepts HF repo IDs — bypass it for local paths.
        # A "local path" is anything starting with ~, /, or . (relative).
        _is_local_path = self.model_id.startswith(("~", "/", "./", "../"))
        if _is_local_path:
            local = Path(self.model_id).expanduser().resolve()
            if not local.is_dir():
                raise FileNotFoundError(
                    f"Local model path not found: {local}\n"
                    "To use the pre-downloaded HF model, pass the repo ID instead:\n"
                    "  LocalLLM('mlx-community/Qwen3.5-9B-MLX-4bit')"
                )
            model_path = str(local)
        else:
            model_path = get_model_path(self.model_id)
        self._model = load_model(model_path)

        hf_logging.set_verbosity_error()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        finally:
            hf_logging.set_verbosity_warning()

        detokenizer_cls = load_detokenizer(model_path, return_tokenizer=False)
        self._tokenizer.detokenizer = detokenizer_cls(self._tokenizer)

        eos_ids = getattr(self._model.config, "eos_token_id", None) or self._tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        self._stopping_criteria = StoppingCriteria(eos_ids, self._tokenizer)
        self._tokenizer.stopping_criteria = self._stopping_criteria

        self._stream_generate = stream_generate
        self._loaded = True

        load_time = time.perf_counter() - t0
        ram_delta = _ram_gb() - ram_before
        logger.info(
            "[LocalLLM] Ready — load %.1fs, +%.2f GB RAM.",
            load_time, ram_delta,
        )

    # ------------------------------------------------------------------
    # Core streaming — matches GenerationRuntime.stream_raw() signature
    # ------------------------------------------------------------------

    def stream_raw(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        track_stats: bool = True,
    ) -> Iterator[str]:
        """Stream raw tokens from the model.

        Drop-in for GenerationRuntime.stream_raw() — same signature,
        same yielded strings, same last_stats dict.
        """
        self.load()

        tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        temp = temperature if temperature is not None else self.default_temperature

        total_tokens = 0
        pure_gen_s = 0.0
        ram_before = _ram_gb()

        try:
            gen_iter = iter(
                self._stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=tokens,
                    temperature=temp,
                )
            )
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
            if track_stats:
                self.last_stats = {
                    "backend": "mlx_vlm",
                    "tokens": total_tokens,
                    "gen_seconds": pure_gen_s,
                    "tok_per_sec": total_tokens / pure_gen_s if pure_gen_s > 0 else 0.0,
                    "ram_delta_gb": _ram_gb() - ram_before,
                }

    # ------------------------------------------------------------------
    # Chat helper — applies the model's chat template
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = True,
    ) -> str | Iterator[str]:
        """Apply chat template and stream (or collect) a response.

        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        self.load()
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if stream:
            return self.stream_raw(prompt, max_tokens=max_tokens, temperature=temperature)
        return "".join(
            self.stream_raw(prompt, max_tokens=max_tokens, temperature=temperature)
        )

    # ------------------------------------------------------------------
    # Embeddings — mean-pool over token embedding layer
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Generate a dense embedding vector via mean-pooled token embeddings.

        Fast (~5 ms on M4) and consistent with the generation model space.
        Note: for production retrieval quality prefer a dedicated embedding
        model (e.g. nomic-embed-text). This is the zero-extra-model option.
        """
        self.load()
        import mlx.core as mx

        token_ids = self._tokenizer.encode(text)
        tokens = mx.array(token_ids)

        # Access the language model's embedding layer directly
        lm = getattr(self._model, "language_model", self._model)
        embed_layer = getattr(lm, "model", lm)
        hidden = embed_layer.embed_tokens(tokens)  # (seq_len, d_model)
        vector = hidden.mean(axis=0)               # (d_model,)
        mx.eval(vector)
        return vector.tolist()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ram_used_gb(self) -> float:
        return _ram_gb()


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_local_llm: LocalLLM | None = None


def get_local_llm(model_id: str = DEFAULT_MODEL_ID) -> LocalLLM:
    """Return (and lazily create) the process-wide LocalLLM instance."""
    global _local_llm
    if _local_llm is None:
        _local_llm = LocalLLM(model_id)
    return _local_llm


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ram_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return 0.0
