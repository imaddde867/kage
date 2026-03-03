# Kage (影)

A fully local, always-on personal AI for macOS. Wake word activates it, you speak, it responds aloud. No cloud, no subscriptions, no data leaving your machine.

**Current state:** functional voice loop. Memory persists across sessions. No calendar/reminders integration yet (removed — see roadmap).

---

## Stack

| Layer | Tool | Why |
|-------|------|-----|
| LLM | [Ollama](https://ollama.com) | Local inference, simple HTTP API |
| Wake word | [openwakeword](https://github.com/dscripka/openWakeWord) | Lightweight, CPU-only, works offline |
| STT | macOS native (`SpeechRecognition`) | Hardware-accelerated, zero latency |
| STT fallback | [faster-whisper](https://github.com/guillaumekleeven/faster-whisper) | CPU-only, no GPU needed |
| TTS | macOS `say` | Built-in, no deps, Siri-quality neural voices |
| Memory | SQLite | No server, simple, durable |
| Audio | [sounddevice](https://python-sounddevice.readthedocs.io) | Clean cross-platform mic access |

---

## Setup

**Prerequisites:** Python 3.11, [Ollama](https://ollama.com) installed and running.

```bash
# 1. Create environment
micromamba create -n kage python=3.11 pip -y && micromamba activate kage

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — at minimum set OLLAMA_MODEL and MACOS_SAY_VOICE

# 4. Pull the model
ollama pull qwen3.5:9b

# 5. Run
python main.py --text     # text mode (no mic needed)
python main.py            # voice mode (wake word → speak → respond)
```

**List available voices:**
```bash
say -v '?'
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen3.5:9b` | Model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `MACOS_SAY_VOICE` | `Ava (Enhanced)` | TTS voice |
| `STT_BACKEND` | `apple` | `apple` or `whisper` |
| `WAKE_WORD` | `hey jarvis` | Display name |
| `WAKE_WORD_MODEL` | `hey_jarvis` | openwakeword model file |
| `WAKE_WORD_THRESHOLD` | `0.5` | Detection sensitivity (0–1) |
| `USER_NAME` | `Imad` | Your name (used in prompts) |
| `MEMORY_DIR` | `./data/memory` | SQLite DB location |

---

## Architecture

```
main.py
├── voice mode: ListenerService → BrainService → speak()
└── text mode:  input()         → BrainService → speak()

core/
├── brain.py     BrainService.think_stream() — Ollama streaming, yields sentences
├── listener.py  ListenerService — wake word + record + STT
├── memory.py    MemoryStore — SQLite conversations, keyword recall
└── speaker.py   speak() — macOS say subprocess

config.py        Settings dataclass, loaded once via lru_cache
data/memory/     kage_memory.db (gitignored)
```

---

## Development principles

These rules exist to avoid accumulating the complexity that was just removed:

1. **One backend per concern.** No fallback chains. Pick one and make it work.
2. **No feature until it works end-to-end.** Don't wire up half-built code.
3. **No abstraction until it's needed 3+ times.** Duplicate twice; abstract on the third.
4. **Connectors are opt-in plugins, not core.** Calendar/Reminders should not be in the hot path.
5. **Test before wiring.** If you can't test it in isolation, don't add it.

---

## Roadmap

Ordered by value delivered:

- [ ] **Better memory recall** — BM25 or embeddings instead of keyword matching
- [ ] **Streaming STT** — start transcribing while the user is still speaking
- [ ] **Calendar connector** — read-only, opt-in, only injected when relevant
- [ ] **Web search** — on-demand via a tool call, not always-on
- [ ] **Multi-turn context** — keep last N turns in the prompt window
- [ ] **Wake word customization** — train a custom openwakeword model

---

## What was removed

KittenTTS, AVSpeech, phonemizer, connectors (calendar/reminders/notes/things), facts table, multi-backend TTS switching, streaming session complexity. These added code weight without proportional value at this stage. They can be reintroduced one at a time when the foundation is solid.
