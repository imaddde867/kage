# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Is Kage

Kage is a fully local, always-on personal AI for macOS. The voice loop is: wake word (`hey_jarvis` via openwakeword) → STT (faster-whisper) → LLM (Ollama) → TTS (KittenTTS or macOS `say`). Memory persists across sessions in SQLite (`data/memory/kage_memory.db`). Live context (Calendar, Reminders, Notes) is injected into the LLM prompt via AppleScript connectors.

## Commands

**Setup:**
```bash
micromamba create -n kage python=3.11 pip -y && micromamba activate kage
pip install -r requirements.txt
cp .env.example .env
ollama serve          # in a separate terminal
ollama pull qwen3.5:9b
```

**Run:**
```bash
python main.py          # voice mode (wake word → listen → speak)
python main.py --text   # text chat mode (no mic/speaker required)
```

**Tests:**
```bash
python -m pytest tests/           # run all tests
python -m pytest tests/test_memory.py  # single test file
python -m py_compile main.py config.py core/*.py connectors/*.py  # syntax check
```

No build step, no linter config committed. If adding a linter, use `ruff` + `black`.

## Architecture

`AssistantRuntime` (`main.py`) owns and wires all services:

- **`config.py`** — frozen `Settings` dataclass, loaded once via `@lru_cache`. All env vars are read here. Add new settings here, not scattered through modules.
- **`core/brain.py`** — `BrainService.think_stream()` builds the Ollama prompt (system prompt + optional live context + memory recall), streams `/api/chat` token-by-token, yields complete sentences for immediate TTS, then persists the exchange. Context injection is gated on keyword hints in `_LIVE_CONTEXT_HINTS`. Non-streaming `think()` remains for internal fallback.
- **`core/memory.py`** — `MemoryStore` wraps a SQLite DB with two tables: `conversations` and `facts`. Recall is keyword-based (no embeddings yet). The DB path respects `~` expansion via `MEMORY_DIR`.
- **`core/listener.py`** — `ListenerService` handles wake-word detection (openwakeword) and speech recording + transcription. STT backend is `apple` (macOS native, hardware-accelerated) by default, with `whisper` (faster-whisper) as fallback. Set via `STT_BACKEND` env var.
- **`core/speaker.py`** — `SpeakerService` supports two backends: `macos_say` (default, macOS `say` with neural voices) and `kittentts` (local neural TTS). Falls back to `macos_say` automatically if KittenTTS fails. Set via `TTS_BACKEND` env var.
- **`connectors/`** — Each connector (`calendar.py`, `reminders.py`, `notes.py`) exposes `get_context() -> str` using AppleScript. `ConnectorManager` aggregates them and silently skips failures.

## Key Configuration

All settings are controlled via `.env` (see `.env.example`). Notable vars:
- `OLLAMA_MODEL` — default `qwen3.5:9b`
- `OLLAMA_THINK` — default `false` (faster direct answers; set `true` for deeper reasoning)
- `STT_BACKEND` — `apple` (macOS native, default) or `whisper` (local fallback)
- `TTS_BACKEND` — `macos_say` (default, Siri-quality neural voice) or `kittentts`
- `MACOS_SAY_VOICE` — default `Ava (Enhanced)`; list options with `say -v '?'`
- `WAKE_WORD` / `WAKE_WORD_MODEL` — default `hey jarvis` / `hey_jarvis`
- `MEMORY_DIR` — default `./data/memory`, supports `~`

## Conventions

- Python 3.11, 4-space indentation, `snake_case` for functions/variables, `PascalCase` for service classes.
- Each service class (`BrainService`, `ListenerService`, etc.) also exposes module-level singleton functions (`think()`, `speak()`, etc.) for convenience.
- New assistant logic goes in `core/`; external integrations go in `connectors/`.
- Never commit `.env` or `data/memory/*.db`.
- Commit messages use imperative form with scope: e.g. `core: split brain service and Ollama client`.
