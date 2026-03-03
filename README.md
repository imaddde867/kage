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
| TTS | [mlx-audio](https://github.com/Blaizzy/mlx-audio) + Kokoro-82M | Local MLX synthesis on Apple Silicon |
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
# Edit .env — set your model and Kokoro voice preset

# 4. Pull the model
ollama pull qwen3.5:9b

# 5. Run
python main.py --text     # text mode (no mic needed)
python main.py            # voice mode (wake word → speak → respond)
```

**Popular Kokoro voices:** `af_heart` (US), `bf_emma` (UK), `bm_george` (UK)

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen3.5:9b` | Model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `KOKORO_MODEL` | `mlx-community/Kokoro-82M-bf16` | Kokoro model repo |
| `KOKORO_VOICE` | `af_heart` | Voice preset |
| `KOKORO_LANG_CODE` | `en-us` | Accent/language (`en-us`, `en-gb`, `ja`, `zh`) |
| `KOKORO_SPEED` | `1.0` | Speaking rate multiplier |
| `STT_BACKEND` | `apple` | `apple` or `whisper` |
| `WAKE_WORD` | `hey jarvis` | Display name |
| `WAKE_WORD_MODEL` | `hey_jarvis` | openwakeword model file |
| `WAKE_WORD_THRESHOLD` | `0.5` | Detection sensitivity (0–1) |
| `ALLOW_BARGE_IN` | `true` | Allow interrupting Kage while it speaks |
| `INTERRUPT_POLICY` | `wake_word_then_speech` | Barge-in trigger policy |
| `INTERRUPT_MIN_SCORE` | `0.55` | Wake score threshold during TTS |
| `INTERRUPT_HOLD_MS` | `220` | Required speech duration after wake hit |
| `INTERRUPT_DEBOUNCE_MS` | `500` | Minimum gap between accepted interrupts |
| `POST_TTS_GUARD_MS` | `250` | Delay before opening mic after TTS stops |
| `USER_NAME` | `Imad` | Your name (used in prompts) |
| `ASSISTANT_NAME` | `Kage` | Canonical assistant name kept in text |
| `TTS_NAME_OVERRIDE_ENABLED` | `true` | Apply spoken-name replacement before TTS |
| `TTS_NAME_PRONUNCIATION` | `Kah-gay` | Spoken alias to force hard-g pronunciation |
| `STT_NAME_NORMALIZATION_ENABLED` | `true` | Normalize recognized variants back to canonical name |
| `STT_NAME_VARIANTS` | `kage,cage,kaj,kaige,kahge,ka-geh` | Comma-separated variants mapped to `ASSISTANT_NAME` |
| `MEMORY_DIR` | `./data/memory` | SQLite DB location |

---

## Architecture

```
main.py
├── voice mode: ListenerService → BrainService → speak()
└── text mode:  input()         → BrainService → speak()

core/
├── audio_coordinator.py state machine for listen/think/speak + barge-in guards
├── brain.py     BrainService.think_stream() — Ollama streaming, yields sentences
├── listener.py  ListenerService — wake word + record + STT
├── memory.py    MemoryStore — SQLite conversations, keyword recall
└── speaker.py   speak() — mlx-audio Kokoro synthesis + playback

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
