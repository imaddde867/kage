from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

load_dotenv()


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "y"}:
        return True
    if value in {"0", "false", "no", "off", "n"}:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    # LLM
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    ollama_think: bool

    # Wake word / STT
    wake_word: str
    wake_word_model: str
    wake_word_threshold: float
    whisper_model: str

    # TTS
    tts_backend: str
    tts_voice: str
    kittentts_profile: str
    kittentts_model: str
    kittentts_sample_rate: int
    say_fallback_voice: str

    # Memory
    memory_dir: str

    # User
    user_name: str

    # Audio
    sample_rate: int
    wake_word_chunk_size: int
    record_chunk_size: int
    silence_threshold: int
    silence_duration: float
    max_record_seconds: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_env_str("OLLAMA_MODEL", "qwen3:8b"),
        ollama_timeout_seconds=_env_int("OLLAMA_TIMEOUT_SECONDS", 60),
        ollama_think=_env_bool("OLLAMA_THINK", False),
        wake_word=_env_str("WAKE_WORD", "hey jarvis"),
        wake_word_model=_env_str("WAKE_WORD_MODEL", "hey_jarvis"),
        wake_word_threshold=_env_float("WAKE_WORD_THRESHOLD", 0.5),
        whisper_model=_env_str("WHISPER_MODEL", "base"),
        tts_backend=_env_str("TTS_BACKEND", "kittentts"),
        tts_voice=_env_str("TTS_VOICE", "Jasper"),
        kittentts_profile=_env_str("KITTENTTS_PROFILE", "nano"),
        kittentts_model=_env_str("KITTENTTS_MODEL", ""),
        kittentts_sample_rate=_env_int("KITTENTTS_SAMPLE_RATE", 24000),
        say_fallback_voice=_env_str("MACOS_SAY_VOICE", _env_str("SAY_FALLBACK_VOICE", "Daniel")),
        memory_dir=_env_str("MEMORY_DIR", "./data/memory"),
        user_name=_env_str("USER_NAME", "Imad"),
        sample_rate=_env_int("SAMPLE_RATE", 16000),
        wake_word_chunk_size=_env_int("WAKE_WORD_CHUNK_SIZE", 1280),
        record_chunk_size=_env_int("RECORD_CHUNK_SIZE", 1024),
        silence_threshold=_env_int("SILENCE_THRESHOLD", 500),
        silence_duration=_env_float("SILENCE_DURATION", 1.5),
        max_record_seconds=_env_int("MAX_RECORD_SECONDS", 30),
    )
