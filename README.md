<h1 align="center">Kage (影)</h1>
<p align="center">
  <img src="kage.gif" alt="Kage the black horse" width="420" />
</p>

Kage is a fully local personal AI for macOS: wake word -> speech -> local reasoning -> spoken response, with persistent memory and optional tool use.

No cloud inference. No subscriptions. Data stays on your machine.

## Current State

Kage currently runs three layers:

1. Core assistant loop (voice/text, STT, LLM, TTS).
2. Second brain (structured entity memory: tasks, commitments, profile facts, preferences).
3. Agent layer (tool-calling loop with connectors + heartbeat reminders).

## Stack

| Layer        | Tool                                                                                           | Why                                      |
| ------------ | ---------------------------------------------------------------------------------------------- | ---------------------------------------- |
| LLM          | [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) / [MLX-LM](https://github.com/ml-explore/mlx-lm) | Fast local inference on Apple Silicon    |
| Wake word    | [openwakeword](https://github.com/dscripka/openWakeWord)                                       | Lightweight, CPU-only, offline           |
| STT          | `SpeechRecognition` (Apple backend)                                                            | Native macOS recognition                 |
| STT fallback | [faster-whisper](https://github.com/guillaumekleeven/faster-whisper)                           | Local fallback                           |
| TTS          | [mlx-audio](https://github.com/Blaizzy/mlx-audio) + Kokoro-82M                                 | Local speech synthesis                   |
| Memory       | SQLite                                                                                         | Durable local storage                    |
| Web tools    | `duckduckgo-search`, `scrapling[fetchers]`, `httpx`, `trafilatura`                             | Agent web search + adaptive page fetch    |

## Setup

Prerequisite: Python 3.11 on Apple Silicon.

```bash
# 1) Create environment
micromamba create -n kage python=3.11 pip -y
micromamba activate kage

# 2) Install dependencies (includes connector deps)
pip install -r requirements.txt

# 3) Configure
cp .env.example .env
# Edit .env as needed

# 4) Run
python3 main.py --text   # text mode
python3 main.py          # voice mode
```

## Connectors and Tools

When `AGENT_ENABLED=true`, the agent can use these tools via `ToolRegistry`:

- `web_search`: DuckDuckGo text search with source URLs (`duckduckgo-search`)
- `web_fetch`: Scrapling-first URL fetch + readable extraction with safe fallback (`scrapling[fetchers]`, `httpx`, `trafilatura`)
- `shell`: allowlisted local shell commands only
- `notify`: macOS notification via `osascript`
- `speak`: direct TTS output
- `calendar_read`: read upcoming events from macOS Calendar (`osascript`)
- `reminder_add`: add reminder in macOS Reminders (`osascript`)
- `mark_task_done`, `update_fact`, `list_open_tasks`: second-brain memory tools

Notes:

- Calendar/Reminders/notifications require macOS and AppleScript permissions.
- `shell` is restricted to a small allowlist and blocks pipes/redirection/operators.
- `web_fetch` prefers Scrapling fetchers and falls back to `httpx` if needed.

## Configuration (`.env`)

### Core Inference

| Variable          | Default                             | Description                                      |
| ----------------- | ----------------------------------- | ------------------------------------------------ |
| `LLM_BACKEND`     | `mlx_vlm`                           | `mlx_vlm` or `mlx`                               |
| `MLX_MODEL`       | `mlx-community/Qwen3.5-4B-MLX-4bit` | Main model                                       |
| `MLX_DRAFT_MODEL` | ``                                  | Optional speculative draft model (`mlx` backend) |
| `MLX_MAX_TOKENS`  | `250`                               | Generation cap                                   |
| `TEMPERATURE`     | `0.3`                               | Sampling temperature for non-agent conversational responses |

### Voice / Audio

| Variable              | Default                         | Description              |
| --------------------- | ------------------------------- | ------------------------ |
| `WAKE_WORD`           | `hey jarvis`                    | Display wake phrase      |
| `WAKE_WORD_MODEL`     | `hey_jarvis`                    | openwakeword model name  |
| `WAKE_WORD_THRESHOLD` | `0.5`                           | Wake detection threshold |
| `STT_BACKEND`         | `apple`                         | `apple` or `whisper`     |
| `WHISPER_MODEL`       | `base`                          | Whisper model size       |
| `KOKORO_MODEL`        | `mlx-community/Kokoro-82M-bf16` | TTS model                |
| `KOKORO_VOICE`        | `af_heart`                      | Voice preset             |
| `KOKORO_SPEED`        | `1.0`                           | Speech speed             |
| `KOKORO_LANG_CODE`    | `en-us`                         | Language/accent          |

### Audio Tuning

| Variable               | Default | Description                       |
| ---------------------- | ------- | --------------------------------- |
| `SAMPLE_RATE`          | `16000` | Input sample rate                 |
| `WAKE_WORD_CHUNK_SIZE` | `1280`  | Wake detector chunk size          |
| `RECORD_CHUNK_SIZE`    | `1024`  | Recording chunk size              |
| `SILENCE_THRESHOLD`    | `500`   | Silence cutoff                    |
| `SILENCE_DURATION`     | `1.5`   | Seconds of silence to end capture |
| `MAX_RECORD_SECONDS`   | `30`    | Max per-turn record duration      |

### Turn Taking / Barge-In

| Variable                | Default | Description                              |
| ----------------------- | ------- | ---------------------------------------- |
| `ALLOW_BARGE_IN`        | `true`  | Allow interruption while TTS is speaking |
| `INTERRUPT_MIN_SCORE`   | `0.55`  | Wake score threshold during TTS          |
| `INTERRUPT_HOLD_MS`     | `220`   | Required speech hold after wake hit      |
| `INTERRUPT_DEBOUNCE_MS` | `500`   | Minimum gap between interrupts           |
| `POST_TTS_GUARD_MS`     | `250`   | Guard delay before reopening mic         |

### Identity / Text Mode

| Variable                         | Default                            | Description                                 |
| -------------------------------- | ---------------------------------- | ------------------------------------------- |
| `USER_NAME`                      | `Imad`                             | User name used in prompts                   |
| `ASSISTANT_NAME`                 | `Kage`                             | Assistant display name                      |
| `TTS_NAME_OVERRIDE_ENABLED`      | `true`                             | Apply spoken-name replacement               |
| `TTS_NAME_PRONUNCIATION`         | `Kah-gay`                          | Spoken alias                                |
| `STT_NAME_NORMALIZATION_ENABLED` | `true`                             | Normalize variants back to `ASSISTANT_NAME` |
| `STT_NAME_VARIANTS`              | `kage,cage,kaj,kaige,kahge,ka-geh` | Accepted STT variants                       |
| `TEXT_MODE_TTS_ENABLED`          | `false`                            | Speak responses in `--text` mode            |

### Memory / Second Brain

| Variable                     | Default         | Description                                 |
| ---------------------------- | --------------- | ------------------------------------------- |
| `MEMORY_DIR`                 | `./data/memory` | SQLite location                             |
| `RECENT_TURNS`               | `4`             | Recent turn buffer                          |
| `SECOND_BRAIN_ENABLED`       | `true`          | Master switch for entity memory layer       |
| `EXTRACTION_ENABLED`         | `true`          | Run entity extraction after each turn       |
| `ENTITY_RECALL_BUDGET`       | `400`           | Max characters of entity context in prompts |
| `PROACTIVE_DEBOUNCE_SECONDS` | `60`            | Debounce for proactive suggestions          |

### Agent / Heartbeat

| Variable                     | Default | Description                                             |
| ---------------------------- | ------- | ------------------------------------------------------- |
| `AGENT_ENABLED`              | `true`  | Enable tool-using agent loop                            |
| `AGENT_MAX_STEPS`            | `8`     | Max ReAct iterations per request                        |
| `AGENT_TEMPERATURE`          | `0.0`   | Sampling temperature for tool-mode generations          |
| `AGENT_ENTITY_MODE`          | `relevance_filtered` | Entity recall mode: `personal_only`, `relevance_filtered`, `full` |
| `AGENT_HISTORY_CHAR_BUDGET`  | `8000`  | Max combined chars retained in agent step history       |
| `AGENT_OBSERVATION_MAX_CHARS`| `1800`  | Per-tool-observation compression cap                    |
| `HEARTBEAT_ENABLED`          | `true`  | Start background proactive reminder daemon (voice mode) |
| `HEARTBEAT_INTERVAL_SECONDS` | `300`   | Heartbeat tick interval                                 |
| `DND_START_HOUR`             | `23`    | Do-not-disturb start hour (24h)                         |
| `DND_END_HOUR`               | `7`     | Do-not-disturb end hour (24h)                           |

## Architecture

```text
main.py
├── voice mode: ListenerService -> BrainService -> speak()
└── text mode:  input()         -> BrainService -> print/(optional speak)

BrainService request flow
1) guardrails state update
2) if AGENT_ENABLED and classifier says "tools needed":
     AgentLoop (ReAct XML format) -> ToolRegistry -> connectors
   else:
     classic conversational path
     (intent route -> prompt build -> LLM stream)
3) persist exchange to SQLite
4) optional entity extraction (LLMEntityExtractor -> EntityStore)
5) optional proactive suggestion

Voice mode only
- HeartbeatAgent daemon wakes every HEARTBEAT_INTERVAL_SECONDS
- checks DND + audio-idle + debounce
- speaks due/overdue task reminders
```

Key modules:

- `core/brain.py`: orchestration and routing between classic vs agent path
- `core/intent_signals.py`: modular weighted intent scoring used by routing/context decisions
- `core/agent/loop.py`: multi-step tool loop
- `core/agent/tool_registry.py`: connector dispatch
- `connectors/*.py`: individual tool implementations
- `core/second_brain/entity_store.py`: structured memory persistence
- `core/second_brain/llm_extractor.py`: LLM-based entity extraction
- `core/agent/heartbeat.py`: proactive background reminders

## Testing

Run the test suite:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Current suite size (as of 2026-03-04): 148 tests.

Useful sanity checks:

```bash
# Verify loaded flags
python3 -c "import config; s=config.get(); print(s.agent_enabled, s.heartbeat_enabled, s.second_brain_enabled)"

# Inspect entity table
sqlite3 data/memory/kage_memory.db ".schema entities"
sqlite3 data/memory/kage_memory.db "SELECT kind,key,value,due_date,status FROM entities;"
```

---

## Roadmap

Ordered by value delivered:

- [x] **Second brain entity memory** — tasks, commitments, profile, preferences extracted and recalled
- [x] **Intent routing** — classify before LLM call; inject entities only when needed
- [x] **Proactive suggestions** — debounced, mention-aware reminders

Near-term (high confidence):

- [ ] **Close the loop on tasks** — detect completion phrases ("done", "finished") and call `mark_task_done` reliably, with tests for false positives
- [ ] **Agent reliability pass** — harden tool-failure handling (timeouts, retries, clear fallback responses) so tool mode is predictable in daily use
- [ ] **Memory quality before scale** — add entity dedup/merge rules and conflict handling ("new value replaces old value") to keep recall clean
- [ ] **On-demand external context** — expose web + calendar lookups only when asked, and include lightweight source attribution in responses

Mid-term (higher effort, still practical):

- [ ] **Streaming STT** — partial transcript while speaking to reduce turn latency and improve interruption handling
- [ ] **Context budget control** — token-aware budgeting across recent turns, recalled exchanges, and entity context per request
- [ ] **Recall upgrade path** — introduce BM25 first, then evaluate embeddings only if BM25 quality is insufficient

Later (optional / exploratory):

- [ ] **Wake word customization** — user-trained wake word support once core voice reliability stabilizes
