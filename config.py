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
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    return parts if parts else default


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_backend: str
    mlx_model: str
    mlx_draft_model: str  # empty string = disabled; same-family smaller model for speculative decoding
    mlx_max_tokens: int
    temperature: float  # generation temperature; lower = less hallucination

    # Wake word
    wake_word: str
    wake_word_model: str
    wake_word_threshold: float

    # STT
    stt_backend: str
    whisper_model: str
    stt_streaming_enabled: bool
    stt_stream_interval_seconds: float
    stt_stream_min_chars: int

    # TTS
    kokoro_model: str
    kokoro_voice: str
    kokoro_speed: float
    kokoro_lang_code: str

    # Memory
    memory_dir: str
    recent_turns: int

    # User
    user_name: str
    assistant_name: str

    # Audio
    sample_rate: int
    wake_word_chunk_size: int
    record_chunk_size: int
    silence_threshold: int
    silence_duration: float
    max_record_seconds: int

    # Turn-taking
    allow_barge_in: bool
    interrupt_min_score: float
    interrupt_hold_ms: int
    interrupt_debounce_ms: int
    post_tts_guard_ms: int

    # Pronunciation controls
    tts_name_override_enabled: bool
    tts_name_pronunciation: str
    stt_name_normalization_enabled: bool
    stt_name_variants: tuple[str, ...]

    # Text mode UX
    text_mode_tts_enabled: bool

    # Second Brain
    second_brain_enabled: bool
    entity_recall_budget: int
    proactive_debounce_seconds: int
    extraction_enabled: bool

    # Agent — autonomous tool-using ReAct loop (core/agent/).
    # When agent_enabled is True, BrainService classifies each request with a fast
    # 8-token routing call.  Requests that need tools are handled by AgentLoop;
    # all others continue on the existing fast conversational path unchanged.
    agent_enabled: bool  # AGENT_ENABLED — master switch; False disables all tool use
    agent_max_steps: int  # AGENT_MAX_STEPS — hard cap on ReAct iterations per request
    agent_temperature: (
        float  # AGENT_TEMPERATURE — sampling temperature for tool-mode generations
    )
    agent_entity_mode: (
        str  # AGENT_ENTITY_MODE — personal_only | relevance_filtered | full
    )
    agent_history_char_budget: (
        int  # AGENT_HISTORY_CHAR_BUDGET — max chars retained in loop history
    )
    agent_observation_max_chars: (
        int  # AGENT_OBSERVATION_MAX_CHARS — per-observation compression cap
    )
    agent_tool_max_retries: (
        int  # AGENT_TOOL_MAX_RETRIES — retries for retryable tool failures
    )
    agent_tool_retry_backoff_seconds: (
        float  # AGENT_TOOL_RETRY_BACKOFF_SECONDS — exponential backoff base
    )
    agent_tool_cooldown_failures: (
        int  # AGENT_TOOL_COOLDOWN_FAILURES — failures before temporary tool cooldown
    )
    agent_tool_cooldown_seconds: (
        float  # AGENT_TOOL_COOLDOWN_SECONDS — cooldown duration per tool
    )
    agent_policy_mode: str  # AGENT_POLICY_MODE — strict | hybrid | owner_fast
    agent_approval_required_tiers: tuple[
        str, ...
    ]  # AGENT_APPROVAL_REQUIRED_TIERS — CSV of tiers requiring approval
    agent_autonomy_max_horizon_steps: int  # AGENT_AUTONOMY_MAX_HORIZON_STEPS — cap for long-running autonomous task plans
    agent_reflection_enabled: bool  # AGENT_REFLECTION_ENABLED — append failure reflections into tool observations
    shell_confirm_token: (
        str  # SHELL_CONFIRM_TOKEN — token required for shell mutation commands
    )

    # Heartbeat — proactive background daemon (core/agent/heartbeat.py).
    # A daemon thread wakes every heartbeat_interval_seconds, scans EntityStore for
    # due/overdue items, and speaks a reminder if conditions allow (IDLE, not DND,
    # debounce cleared).  The thread is a daemon so it auto-terminates with the process.
    heartbeat_enabled: (
        bool  # HEARTBEAT_ENABLED — starts HeartbeatAgent on voice mode startup
    )
    heartbeat_interval_seconds: (
        int  # HEARTBEAT_INTERVAL_SECONDS — seconds between wakeup checks
    )

    # Do Not Disturb — suppresses heartbeat speech during quiet hours.
    # Overnight windows (e.g. dnd_start_hour=23, dnd_end_hour=7) are handled correctly:
    # the check wraps midnight so any hour >= 23 OR < 7 is treated as DND.
    # Same-day windows (e.g. dnd_start_hour=9, dnd_end_hour=17) also work naturally.
    dnd_start_hour: int  # DND_START_HOUR — 24h hour (0–23) when quiet period begins
    dnd_end_hour: int  # DND_END_HOUR   — 24h hour (0–23) when quiet period ends

    # OpenAI-compatible server (used when LLM_BACKEND=openai_compat)
    llm_base_url: str  # LLM_BASE_URL — base URL of the OpenAI-compatible server
    llm_api_key: str  # LLM_API_KEY  — API key (any string; mlx_lm.server ignores it)

    # Web fetch TLS mode — controls SSL certificate verification in web_fetch.
    # 'strict' (default): refuse connections with invalid/self-signed certificates.
    # 'allow_insecure_fallback': retry with verify=False on SSL failure and annotate
    # the result.  Use only when fetching trusted internal URLs with self-signed certs.
    web_fetch_tls_mode: (
        str  # WEB_FETCH_TLS_MODE — 'strict' or 'allow_insecure_fallback'
    )
    # CSV allowlist of domains where insecure TLS fallback is permitted.
    # Empty by default; must be explicitly set by the user.
    web_fetch_insecure_fallback_domains: tuple[str, ...]
    # If True, httpx retries SSL failures once using certifi's CA bundle before
    # considering insecure fallback.
    web_fetch_tls_retry_with_certifi: bool

    # Local artifact access policy — read-only file discovery and extraction.
    # Paths must stay within LOCAL_ARTIFACT_SAFE_ROOTS and outside
    # LOCAL_ARTIFACT_DENY_ROOTS.
    local_artifact_safe_roots: tuple[str, ...]
    local_artifact_deny_roots: tuple[str, ...]
    local_artifact_max_chars: int
    local_artifact_max_file_bytes: int
    local_artifact_find_max_files: int

    # Cron daemon — fires scheduled reminders stored via ScheduleReminderTool.
    # A daemon thread polls EntityStore every cron_poll_interval_seconds and
    # speaks any reminder whose due_date has passed.
    # Requires SECOND_BRAIN_ENABLED=true to have an active EntityStore.
    cron_enabled: bool  # CRON_ENABLED — starts CronDaemon on voice mode startup
    cron_poll_interval_seconds: (
        int  # CRON_POLL_INTERVAL_SECONDS — seconds between polls
    )

    # Conversation session — after wake-word activation, stay in listen mode
    # until the user goes silent for this many seconds.  0 = disabled (original
    # one-shot behavior: break back to wake word after every response).
    conversation_timeout_seconds: float

    # Calendar connector runtime tuning.
    calendar_read_timeout_seconds: int
    calendar_read_retry_count: int
    calendar_read_retry_delay_seconds: float

    # Web search rate limiting — minimum seconds between DuckDuckGo calls.
    web_search_min_interval_seconds: float  # WEB_SEARCH_MIN_INTERVAL_SECONDS

    # Telegram bridge — opt-in messaging connector (connectors/telegram_bridge.py).
    # Requires: pip install python-telegram-bot
    telegram_bot_token: str  # TELEGRAM_BOT_TOKEN — bot token from @BotFather; empty = disabled
    telegram_allowed_chat_ids: tuple[str, ...]  # TELEGRAM_ALLOWED_CHAT_IDS — CSV of numeric chat IDs

    # NeuroCache — Obsidian vault memory via second-brain (connectors/neurocache_connector.py).
    # Requires second-brain server running: cd second-brain && uvicorn api.server:app --port 8765
    neurocache_enabled: bool    # NEUROCACHE_ENABLED — master switch; injects vault context into every response
    neurocache_api_url: str     # NEUROCACHE_API_URL — base URL of the second-brain server
    neurocache_vault_inbox: str # NEUROCACHE_VAULT_INBOX — absolute path to Obsidian inbox folder for write-back (empty = disabled)


@lru_cache(maxsize=1)
def get() -> Settings:
    return Settings(
        llm_backend=_env_str("LLM_BACKEND", "mlx"),
        mlx_model=_env_str("MLX_MODEL", "mlx-community/Qwen3-8B-4bit"),
        mlx_draft_model=_env_str("MLX_DRAFT_MODEL", ""),
        mlx_max_tokens=_env_int("MLX_MAX_TOKENS", 160),
        temperature=_env_float("TEMPERATURE", 0.0),
        wake_word=_env_str("WAKE_WORD", "hey jarvis"),
        wake_word_model=_env_str("WAKE_WORD_MODEL", "hey_jarvis"),
        wake_word_threshold=_env_float("WAKE_WORD_THRESHOLD", 0.5),
        stt_backend=_env_str("STT_BACKEND", "apple"),
        whisper_model=_env_str("WHISPER_MODEL", "base"),
        stt_streaming_enabled=_env_bool("STT_STREAMING_ENABLED", False),
        stt_stream_interval_seconds=max(
            0.2, _env_float("STT_STREAM_INTERVAL_SECONDS", 0.8)
        ),
        stt_stream_min_chars=max(0, _env_int("STT_STREAM_MIN_CHARS", 6)),
        kokoro_model=_env_str("KOKORO_MODEL", "mlx-community/Kokoro-82M-bf16"),
        kokoro_voice=_env_str("KOKORO_VOICE", "af_heart"),
        kokoro_speed=_env_float("KOKORO_SPEED", 1.0),
        kokoro_lang_code=_env_str("KOKORO_LANG_CODE", "en-us"),
        memory_dir=_env_str("MEMORY_DIR", "./data/memory"),
        recent_turns=max(0, _env_int("RECENT_TURNS", 0)),
        user_name=_env_str("USER_NAME", "Imad"),
        assistant_name=_env_str("ASSISTANT_NAME", "Kage"),
        sample_rate=_env_int("SAMPLE_RATE", 16000),
        wake_word_chunk_size=_env_int("WAKE_WORD_CHUNK_SIZE", 1280),
        record_chunk_size=_env_int("RECORD_CHUNK_SIZE", 1024),
        silence_threshold=_env_int("SILENCE_THRESHOLD", 500),
        silence_duration=_env_float("SILENCE_DURATION", 1.5),
        max_record_seconds=_env_int("MAX_RECORD_SECONDS", 30),
        allow_barge_in=_env_bool("ALLOW_BARGE_IN", True),
        interrupt_min_score=_env_float("INTERRUPT_MIN_SCORE", 0.55),
        interrupt_hold_ms=_env_int("INTERRUPT_HOLD_MS", 220),
        interrupt_debounce_ms=_env_int("INTERRUPT_DEBOUNCE_MS", 500),
        post_tts_guard_ms=_env_int("POST_TTS_GUARD_MS", 250),
        tts_name_override_enabled=_env_bool("TTS_NAME_OVERRIDE_ENABLED", True),
        tts_name_pronunciation=_env_str("TTS_NAME_PRONUNCIATION", "Kah-gay"),
        stt_name_normalization_enabled=_env_bool(
            "STT_NAME_NORMALIZATION_ENABLED", True
        ),
        stt_name_variants=_env_csv(
            "STT_NAME_VARIANTS",
            ("kage", "cage", "kaj", "kaige", "kahge", "ka-geh"),
        ),
        text_mode_tts_enabled=_env_bool("TEXT_MODE_TTS_ENABLED", False),
        second_brain_enabled=_env_bool("SECOND_BRAIN_ENABLED", False),
        entity_recall_budget=_env_int("ENTITY_RECALL_BUDGET", 400),
        proactive_debounce_seconds=_env_int("PROACTIVE_DEBOUNCE_SECONDS", 60),
        extraction_enabled=_env_bool("EXTRACTION_ENABLED", False),
        agent_enabled=_env_bool("AGENT_ENABLED", False),
        agent_max_steps=_env_int("AGENT_MAX_STEPS", 8),
        agent_temperature=_env_float("AGENT_TEMPERATURE", 0.0),
        agent_entity_mode=_env_str("AGENT_ENTITY_MODE", "relevance_filtered"),
        agent_history_char_budget=max(
            1000, _env_int("AGENT_HISTORY_CHAR_BUDGET", 8000)
        ),
        agent_observation_max_chars=max(
            500, _env_int("AGENT_OBSERVATION_MAX_CHARS", 1800)
        ),
        agent_tool_max_retries=max(0, _env_int("AGENT_TOOL_MAX_RETRIES", 1)),
        agent_tool_retry_backoff_seconds=max(
            0.0, _env_float("AGENT_TOOL_RETRY_BACKOFF_SECONDS", 0.25)
        ),
        agent_tool_cooldown_failures=max(
            1, _env_int("AGENT_TOOL_COOLDOWN_FAILURES", 2)
        ),
        agent_tool_cooldown_seconds=max(
            0.0, _env_float("AGENT_TOOL_COOLDOWN_SECONDS", 15.0)
        ),
        agent_policy_mode=_env_str("AGENT_POLICY_MODE", "strict"),
        agent_approval_required_tiers=_env_csv(
            "AGENT_APPROVAL_REQUIRED_TIERS",
            ("moderate_change", "high_impact"),
        ),
        agent_autonomy_max_horizon_steps=max(
            1, _env_int("AGENT_AUTONOMY_MAX_HORIZON_STEPS", 24)
        ),
        agent_reflection_enabled=_env_bool("AGENT_REFLECTION_ENABLED", True),
        shell_confirm_token=_env_str(
            "SHELL_CONFIRM_TOKEN", "YES_I_UNDERSTAND_LOCAL_MUTATION"
        ),
        heartbeat_enabled=_env_bool("HEARTBEAT_ENABLED", False),
        heartbeat_interval_seconds=_env_int("HEARTBEAT_INTERVAL_SECONDS", 300),
        dnd_start_hour=_env_int("DND_START_HOUR", 23),
        dnd_end_hour=_env_int("DND_END_HOUR", 7),
        llm_base_url=_env_str("LLM_BASE_URL", "http://localhost:8080/v1"),
        llm_api_key=_env_str("LLM_API_KEY", "none"),
        web_fetch_tls_mode=_env_str("WEB_FETCH_TLS_MODE", "strict"),
        web_fetch_insecure_fallback_domains=_env_csv(
            "WEB_FETCH_INSECURE_FALLBACK_DOMAINS",
            (),
        ),
        web_fetch_tls_retry_with_certifi=_env_bool(
            "WEB_FETCH_TLS_RETRY_WITH_CERTIFI", True
        ),
        local_artifact_safe_roots=_env_csv(
            "LOCAL_ARTIFACT_SAFE_ROOTS",
            ("./", "~"),
        ),
        local_artifact_deny_roots=_env_csv(
            "LOCAL_ARTIFACT_DENY_ROOTS",
            (
                "~/.ssh",
                "~/.gnupg",
                "~/.aws",
                "~/.kube",
                "~/Library/Keychains",
                "~/Library/Application Support",
                "/etc",
                "/private",
                "/System",
                "/usr",
                "/bin",
                "/sbin",
            ),
        ),
        local_artifact_max_chars=max(500, _env_int("LOCAL_ARTIFACT_MAX_CHARS", 8000)),
        local_artifact_max_file_bytes=max(
            1024, _env_int("LOCAL_ARTIFACT_MAX_FILE_BYTES", 8 * 1024 * 1024)
        ),
        local_artifact_find_max_files=max(
            500, _env_int("LOCAL_ARTIFACT_FIND_MAX_FILES", 50000)
        ),
        conversation_timeout_seconds=max(
            0.0, _env_float("CONVERSATION_TIMEOUT_SECONDS", 45.0)
        ),
        cron_enabled=_env_bool("CRON_ENABLED", False),
        cron_poll_interval_seconds=max(10, _env_int("CRON_POLL_INTERVAL_SECONDS", 60)),
        calendar_read_timeout_seconds=max(
            1, _env_int("CALENDAR_READ_TIMEOUT_SECONDS", 10)
        ),
        calendar_read_retry_count=max(0, _env_int("CALENDAR_READ_RETRY_COUNT", 1)),
        calendar_read_retry_delay_seconds=max(
            0.0, _env_float("CALENDAR_READ_RETRY_DELAY_SECONDS", 0.4)
        ),
        web_search_min_interval_seconds=max(
            0.0, _env_float("WEB_SEARCH_MIN_INTERVAL_SECONDS", 1.0)
        ),
        telegram_bot_token=_env_str("TELEGRAM_BOT_TOKEN", ""),
        telegram_allowed_chat_ids=_env_csv("TELEGRAM_ALLOWED_CHAT_IDS", ()),
        neurocache_enabled=_env_bool("NEUROCACHE_ENABLED", False),
        neurocache_api_url=_env_str("NEUROCACHE_API_URL", "http://127.0.0.1:8765"),
        neurocache_vault_inbox=_env_str("NEUROCACHE_VAULT_INBOX", ""),
    )


def clear_cache() -> None:
    get.cache_clear()


def reload() -> Settings:
    clear_cache()
    return get()
