# Kage (影) — A Personal AI That Lives With You

Kage is an always-on, fully local personal AI for your Mac.

The goal is simple: build something like Siri, but actually smart, actually useful, and actually aware of your life over time.

Kage should know your calendar, reminders, notes, goals, habits, and the promises you made to yourself. It should remember what you said last week, connect dots across time, and nudge you when it matters. Not just answer questions, but think with you.

Long term, Kage is a second brain: part coach, part best friend, part technical collaborator, part memory system. And it runs on your machine. No cloud. No surveillance. It's yours.

## Current Status

- Voice loop: wake word → speech-to-text → LLM → text-to-speech
- **Streaming responses** — first sentence spoken within ~1s of Ollama starting to reply
- **Apple native STT** — hardware-accelerated, no model download, instant start
- **macOS `say` TTS** — Siri-quality neural voices, zero latency
- Local memory stored in SQLite across sessions
- Live context injected from Apple Calendar, Reminders, Notes, and **Things 3**
- Ollama-backed local LLM (qwen3.5:9b)

> Current wake word is **"Hey Jarvis"** — no pre-trained `hey_kage` model exists yet.

## Why Kage

Most assistants start from zero every conversation. Kage should do the opposite.
If you say you're about to spend money, Kage should remember what you told yourself two months ago. If you're drifting from a goal, it should notice. If you forget something important, it should catch it. The point is continuity.

## Privacy

- Runs fully locally on your machine
- Local Ollama models — no OpenAI, no cloud
- Memory stored locally in SQLite
- Apple STT runs on-device via the Neural Engine

## Stack

| Layer        | Tool                                                       |
| ------------ | ---------------------------------------------------------- |
| LLM          | `qwen3.5:9b` via Ollama (streaming)                        |
| STT          | macOS native (`SpeechRecognition`) → Whisper fallback      |
| TTS          | macOS `say` with neural voice (Siri-quality)               |
| Wake word    | `openwakeword` (`hey_jarvis`)                              |
| Memory       | SQLite                                                     |
| Audio I/O    | `sounddevice`                                              |
| Integrations | AppleScript (Calendar, Reminders, Notes, Things 3)         |

## Setup (macOS, local only)

### 1. Install and start Ollama

```bash
ollama serve
# in another terminal
ollama pull qwen3.5:9b
```

### 2. Create Python environment

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

Key settings in `.env`:

```bash
OLLAMA_MODEL=qwen3.5:9b
OLLAMA_THINK=false        # faster direct answers

STT_BACKEND=apple         # macOS native — no model load, instant start
                          # set to "whisper" for fully offline fallback

TTS_BACKEND=macos_say
MACOS_SAY_VOICE=Ava (Enhanced)   # list voices with: say -v '?'
```

### 5. Run Kage

```bash
python main.py          # voice mode (wake word → listen → speak)
python main.py --text   # text chat mode (no mic/speaker needed)
```

## Usage

Say **"Hey Jarvis"** → Kage acknowledges → speak naturally → Kage replies aloud, sentence by sentence as the model generates.

Kage pulls live context from your Apple apps (Calendar, Reminders, Notes, Things 3) when relevant and combines it with local memory before calling the LLM.

## Project Structure

```text
kage/
├── main.py                 # Runtime orchestration loop
├── config.py               # Typed settings + env loading
├── core/
│   ├── listener.py         # Wake word, recording, STT (Apple native or Whisper)
│   ├── speaker.py          # TTS backend (macOS say or KittenTTS)
│   ├── brain.py            # Prompt building + streaming Ollama client + memory writes
│   └── memory.py           # SQLite memory store + recall
├── connectors/
│   ├── __init__.py         # ConnectorManager aggregation
│   ├── _apple.py           # Shared AppleScript helpers
│   ├── calendar.py         # Apple Calendar context
│   ├── reminders.py        # Apple Reminders context
│   ├── notes.py            # Apple Notes context
│   └── things.py           # Things 3 context (Today + Inbox)
├── data/
│   └── memory/             # Local SQLite DB
└── requirements.txt
```

## Roadmap

- [x] Local voice loop (wake → listen → think → speak)
- [x] Persistent memory (SQLite)
- [x] Calendar / Reminders / Notes connectors
- [x] Things 3 connector
- [x] Streaming LLM → TTS (sentence-by-sentence, ~1s first-word latency)
- [x] Apple native STT (hardware-accelerated, instant start)
- [x] macOS neural TTS (Siri-quality voice)
- [ ] Custom **"Hey Kage"** wake word
- [ ] Semantic memory (embeddings + retrieval)
- [ ] Proactive nudges / scheduled check-ins
- [ ] More life context (goals, finances, messaging, projects)
- [ ] Better long-term memory modeling (not just keyword recall)
- [ ] Multi-device access while staying local-first

## Philosophy

Kage shouldn't be a chatbot with a voice. It's a local intelligence layer for your life: always present, memory-persistent, proactive, and honest.
