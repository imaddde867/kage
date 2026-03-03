# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Is Kage

Kage is a fully local, always-on personal AI for macOS. The voice loop is: wake word (`hey_jarvis` via openwakeword) → STT (macOS native or faster-whisper) → LLM (Ollama) → TTS (macOS `say`). Memory persists across sessions in SQLite (`data/memory/kage_memory.db`).

## Commands

**Setup:**
```bash
micromamba create -n kage python=3.11 pip -y && micromamba activate kage
pip install -r requirements.txt
cp .env.example .env
ollama serve          # separate terminal
ollama pull qwen3.5:9b
```

**Run:**
```bash
python main.py          # voice mode (wake word → listen → respond aloud)
python main.py --text   # text chat mode (no mic/speaker required)
```

**Syntax check (no test suite):**
```bash
python -m py_compile main.py config.py core/*.py
python -c "import config; print(config.get())"   # verify settings load
```

No build step, no linter config. If adding a linter, use `ruff` + `black`.

## Architecture

`main.py` owns two flat loops — `run_voice()` and `run_text()` — and wires the services directly. No runtime class.

```
Voice loop: wait_for_wake_word() → record_until_silence() → transcribe() → think_stream() → speak()
Text loop:  input() → think_stream() → print + speak()
```

**`config.py`** — frozen `Settings` dataclass, loaded once via `config.get()` (lru_cache). All env vars are read here. Add new settings here only.

**`core/brain.py`** — `BrainService.think_stream(user_input)` builds the Ollama prompt (system prompt + memory recall), streams `/api/chat` token-by-token, and yields complete sentences. Persists the exchange to memory when done. It's a generator — callers iterate over sentences.

**`core/memory.py`** — `MemoryStore` wraps SQLite with one table: `conversations`. `recall(query)` does keyword matching over recent rows. DB path respects `~` expansion via `MEMORY_DIR`.

**`core/listener.py`** — `ListenerService` handles wake-word detection (openwakeword) and recording + transcription. STT backend is `apple` (macOS native via `SpeechRecognition`) by default; falls back to `faster-whisper` on failure or if `STT_BACKEND=whisper`.

**`core/speaker.py`** — Single `speak(text)` function using `subprocess.run(["say", "-v", voice, text])`. No class, no session management.

## Key Configuration

All settings via `.env` (see `.env.example`):
- `OLLAMA_MODEL` — default `qwen3.5:9b`
- `STT_BACKEND` — `apple` (default) or `whisper`
- `MACOS_SAY_VOICE` — default `Ava (Enhanced)`; list options with `say -v '?'`
- `WAKE_WORD_MODEL` — openwakeword model name (no extension), default `hey_jarvis`
- `MEMORY_DIR` — default `./data/memory`, supports `~`

## Conventions

- Python 3.11, 4-space indentation, `snake_case` for functions/variables, `PascalCase` for service classes.
- `config.get()` is the single entry point for all settings — never read `os.getenv` directly in service modules.
- New assistant logic goes in `core/`; external integrations (calendar, web search, etc.) go in `connectors/` as opt-in, not wired into the core loop.
- Never commit `.env` or `data/memory/*.db`.
- Commit messages use imperative form with scope: e.g. `core: simplify brain streaming`.

## What Not To Do

- Do not add fallback chains (e.g. "try KittenTTS, fall back to say"). One backend per concern.
- Do not inject live context (connectors) into the default hot path — it belongs in opt-in connectors.
- Do not add a `facts` table or other memory tables without a concrete use case driving it.
- Do not wrap module functions in singleton globals unless they're called from 3+ places.
