# Kage (影) — A Personal AI That Lives With You

Kage is an always-on, fully local personal AI for your Mac.

The goal is simple: build something like Siri, but actually smart, actually useful, and actually aware of your life over time.

Kage should know your calendar, reminders, notes, goals, habits, and the promises you made to yourself. It should remember what you said last week, connect dots across time, and nudge you when it matters. Not just answer questions, but think with you.

Long term, Kage is a second brain: part coach, part best friend, part technical collaborator, part memory system. And it runs on your machine. No cloud. No surveillance. It’s yours.

## Current Status (What This Repo Does Today)

This repository is an early local runtime for Kage on macOS:

- Voice loop: wake word -> speech-to-text -> LLM -> text-to-speech
- Local memory stored in SQLite (`data/memory/kage_memory.db`)
- Live context injection from Apple Calendar, Reminders, and Notes
- Ollama-backed local LLM responses
- Persistent memory recall across sessions (keyword-based for now)

> Current wake word is **"Hey Jarvis"** (not "Hey Kage" yet) because there is no pre-trained `hey_kage` wake-word model available. Custom wake-word training is planned.

## Why Kage

Most assistants start from zero every conversation. Kage should do the opposite.
If you say you’re about to spend money, Kage should remember what you told yourself two months ago. If you’re drifting from a goal, it should notice. If you forget something important, it should catch it. The point is continuity.

## Privacy

- Runs locally on your machine
- Uses local Ollama models
- Stores memory locally in SQLite
- No external servers required for core behavior

## Stack (Current)

| Layer        | Tool                                        |
| ------------ | ------------------------------------------- |
| LLM          | `qwen3:8b` via Ollama                       |
| STT          | `faster-whisper`                            |
| TTS          | `KittenTTS` or macOS `say` (offline voices) |
| Wake word    | `openwakeword` (`hey_jarvis`)               |
| Memory       | SQLite                                      |
| Audio I/O    | `sounddevice`                               |
| Integrations | AppleScript (Calendar / Reminders / Notes)  |

## Setup (macOS, local only)

### 1. Install and start Ollama

```bash
ollama serve
# in another terminal
ollama pull qwen3:8b
```

### 2. Create Python environment (recommended: Python 3.11)

Micromamba:

```bash
micromamba create -n kage python=3.11 pip -y
micromamba activate kage
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Review `.env` and set your name, model, voice, etc.

To use macOS TTS as your primary engine (offline, real-time):

```bash
TTS_BACKEND=macos_say
SAY_FALLBACK_VOICE="Eddy (English (US))"
```

To disable reasoning for faster responses on reasoning-capable models (for example `qwen3:8b`), set:

```bash
OLLAMA_THINK=false
```

List available local voices:

```bash
say -v '?'
```

### 5. Run Kage

```bash
python main.py
```

## Usage

Say your configured wake word (default: **"Hey Jarvis"**) -> Kage responds -> speak naturally -> Kage replies aloud.

Kage pulls live context from Apple apps and combines it with local memory before sending your prompt to the local LLM.

## Project Structure

```text
kage/
├── main.py                 # Runtime orchestration loop
├── config.py               # Typed settings + env loading
├── core/
│   ├── listener.py         # Wake word, recording, Whisper STT
│   ├── speaker.py          # TTS backend (KittenTTS or macOS say)
│   ├── brain.py            # Prompt building + Ollama client + memory writes
│   └── memory.py           # SQLite memory store + recall
├── connectors/
│   ├── __init__.py         # ConnectorManager aggregation
│   ├── _apple.py           # Shared AppleScript execution helpers
│   ├── calendar.py         # Apple Calendar context
│   ├── reminders.py        # Apple Reminders context
│   └── notes.py            # Apple Notes context
├── data/
│   └── memory/             # Local SQLite DB
└── requirements.txt
```

## Roadmap

- [x] Local voice loop (wake -> listen -> think -> speak)
- [x] Persistent memory (SQLite)
- [x] Calendar / Reminders / Notes connectors
- [x] Local TTS + local STT
- [ ] Custom **"Hey Kage"** wake word
- [ ] Semantic memory (embeddings + retrieval)
- [ ] Proactive nudges / scheduled check-ins
- [ ] More life context (goals, finances, messaging, projects)
- [ ] Better long-term memory modeling (not just keyword recall)
- [ ] Multi-device access while staying local-first

## Philosophy

Kage shouldn't be a chatbot with a voice. It’s a local intelligence layer for your life: always present, memory-persistent, proactive, and honest.
