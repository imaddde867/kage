<h1 align="center">Kage (影)</h1>
<p align="center">
  <img src="kage.gif" alt="Kage the black horse" width="420" />
</p>

Kage is a fully local personal AI for macOS. Say the wake word, it listens, reasons locally, and responds aloud. No cloud inference, no subscriptions, no data leaving your machine.

Kage is intentionally narrow:
- single-user
- macOS-first
- voice-first
- private-by-default

Kage is intentionally not a multi-channel assistant platform. It does not aim to be a WhatsApp/Telegram/Slack/Discord gateway, a browser automation framework, or a remote multi-device control plane.

## How it works

```
wake word → transcribe → think → speak
```

Voice mode listens for "hey Jarvis", captures your speech, runs it through a local LLM, and speaks the response. Text mode skips the mic and speaker — same reasoning, just a terminal chat UI.

Memory persists in SQLite. An optional agent layer adds tool use (web search, calendar, reminders, shell). An optional heartbeat daemon speaks proactive task reminders when you're idle.

## Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) / [MLX-LM](https://github.com/ml-explore/mlx-lm) | Fast local inference on Apple Silicon |
| Wake word | [openwakeword](https://github.com/dscripka/openWakeWord) | Lightweight, CPU-only, offline |
| STT | macOS `SpeechRecognition` | Native; zero latency overhead |
| STT fallback | [faster-whisper](https://github.com/guillaumekleeven/faster-whisper) | Local fallback when Apple STT fails |
| TTS | [mlx-audio](https://github.com/Blaizzy/mlx-audio) + Kokoro-82M | Local, low-latency speech synthesis |
| Memory | SQLite | Durable, zero-dependency local storage |
| Web tools | `ddgs`, `scrapling`, `httpx`, `trafilatura` | Agent web search + page extraction |

## Setup

Requires Python 3.11 on Apple Silicon.

```bash
micromamba create -n kage python=3.11 pip -y
micromamba activate kage
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Edit `.env` to set your name and preferred model. Everything else works out of the box.

## Running

```bash
kage chat          # full-screen Textual chat UI (default)
kage chat --plain  # plain terminal fallback
kage voice         # voice mode (wake word → listen → respond aloud)
kage bench         # run an inference benchmark and exit
kage doctor        # check environment and dependency status
kage doctor --agent # include policy + approvals diagnostics
kage approvals list
kage approvals grant tool shell
kage approvals revoke tool shell
kage backup create # create a compressed local backup (.env + memory)
kage backup verify output/kage-backup-*.tar.gz
kage service install   # install + start launchd voice daemon (macOS)
kage service status    # inspect launchd daemon status
kage service stop      # stop daemon
kage service start     # start daemon
kage service uninstall # remove daemon + plist
```

Legacy shims still work: `python main.py --text` → chat, `python main.py` → voice.

## Configuration

All settings are in `.env`. Copy `.env.example` to get started.

### Model

| Variable | Default | Notes |
|---|---|---|
| `LLM_BACKEND` | `mlx_vlm` | `mlx_vlm` for Qwen3.5 VLM checkpoints; `mlx` for text-only |
| `MLX_MODEL` | `mlx-community/Qwen3.5-4B-MLX-4bit` | Swap this to change models |
| `MLX_MAX_TOKENS` | `160` | Generation cap per response |
| `TEMPERATURE` | `0.0` | 0 = deterministic; raise for more creative outputs |

Tested model profiles (M4 16GB, benchmarked):

| Model | Backend | tok/s | TTFT |
|---|---|---:|---:|
| `Qwen3.5-4B-MLX-4bit` | `mlx_vlm` | 39.6 | ~350ms |
| `Qwen3.5-9B-MLX-4bit` | `mlx_vlm` | 16.5 | ~535ms |
| `Qwen2.5-7B-Instruct-4bit` | `mlx` | 24.1 | ~400ms |

The 4B is the default — same quality as 9B for conversational use, 2.4× faster.

> Note: Qwen3.5 checkpoints include vision weights and must use `LLM_BACKEND=mlx_vlm`. For a text-only model with lower memory, use a `Qwen2.5-*-Instruct` checkpoint with `LLM_BACKEND=mlx`.

### Voice

| Variable | Default | Notes |
|---|---|---|
| `WAKE_WORD` | `hey jarvis` | Displayed phrase |
| `WAKE_WORD_MODEL` | `hey_jarvis` | openwakeword model file |
| `WAKE_WORD_THRESHOLD` | `0.5` | Detection confidence threshold |
| `STT_BACKEND` | `apple` | `apple` or `whisper` |
| `WHISPER_MODEL` | `base` | Only used when `STT_BACKEND=whisper` |
| `STT_STREAMING_ENABLED` | `false` | Emit partial transcripts while listening (voice mode) |
| `STT_STREAM_INTERVAL_SECONDS` | `0.8` | Interval between partial transcript updates |
| `KOKORO_VOICE` | `af_heart` | Voice preset (`bf_emma`, `bm_george`, etc.) |
| `KOKORO_SPEED` | `1.0` | Speech rate multiplier |
| `KOKORO_LANG_CODE` | `en-us` | Language/accent (`en-gb`, `ja`, `zh`) |

### Identity

| Variable | Default | Notes |
|---|---|---|
| `USER_NAME` | `Imad` | Shown in prompts |
| `ASSISTANT_NAME` | `Kage` | Display name |
| `TTS_NAME_PRONUNCIATION` | `Kah-gay` | How the name is spoken aloud |

### Optional features (all off by default)

| Variable | Default | What it does |
|---|---|---|
| `AGENT_ENABLED` | `false` | Enable tool-using agent loop |
| `AGENT_MAX_STEPS` | `8` | Max tool steps per request |
| `AGENT_TOOL_MAX_RETRIES` | `1` | Retry count for retryable tool failures |
| `AGENT_TOOL_COOLDOWN_SECONDS` | `15` | Temporary per-tool cooldown after repeated failures |
| `AGENT_POLICY_MODE` | `strict` | Policy mode: `strict`, `hybrid`, or `owner_fast` |
| `AGENT_APPROVAL_REQUIRED_TIERS` | `moderate_change,high_impact` | Risk tiers requiring approval in policy mode |
| `AGENT_REFLECTION_ENABLED` | `true` | Adds failure reflection hints after tool errors |
| `SECOND_BRAIN_ENABLED` | `false` | Enable structured entity memory (tasks, facts, preferences) |
| `EXTRACTION_ENABLED` | `false` | Extract entities from each conversation turn |
| `HEARTBEAT_ENABLED` | `false` | Proactive reminders daemon (voice mode only) |
| `HEARTBEAT_INTERVAL_SECONDS` | `300` | How often the heartbeat checks for due tasks |
| `DND_START_HOUR` | `23` | Start of quiet hours (24h) |
| `DND_END_HOUR` | `7` | End of quiet hours |
| `TEXT_MODE_TTS_ENABLED` | `false` | Speak responses aloud in text/chat mode |

See `.env.example` for the full list including audio tuning, barge-in, TLS, and calendar settings.

## Tools (when `AGENT_ENABLED=true`)

| Tool | What it does |
|---|---|
| `web_search` | DuckDuckGo search — no API key needed |
| `web_fetch` | Fetch and extract readable text from a URL |
| `shell` | Read-only allowlisted shell commands |
| `notify` | macOS notification banner |
| `speak` | Trigger TTS mid-chain |
| `calendar_read` | Read upcoming events from macOS Calendar |
| `reminder_add` | Add a reminder to macOS Reminders |
| `mark_task_done` | Mark a second-brain task as complete |
| `update_fact` | Update a stored fact or preference |
| `list_open_tasks` | List active tasks from second-brain memory |
| `forget_fact` | Remove a stored fact or preference from memory |

Calendar, Reminders, and notifications require macOS and AppleScript/Accessibility permissions. `shell` blocks all write operations and pipes by default. In strict mode, moderate/high-risk tools require explicit approval grants.

## Architecture

```
main.py → core.cli
├── chat:  SessionController → Textual UI / plain shell → BrainService
├── voice: ListenerService → BrainService → speak()
└── bench: run_bench() → BrainService

BrainService request flow
1. Guardrails check — deterministic overrides (temporal uncertainty, safety)
2. RequestOrchestrator — plans strategy, retrieves context, selects capabilities
3a. Agent path (if tools needed): AgentLoop → ToolRegistry → connectors
3b. Direct path: prompt build → LLM stream → sentence splitter
4. Persist exchange to SQLite
5. Entity extraction (optional: LLMEntityExtractor → EntityStore)
6. Proactive suggestion check (optional)
```

Key files:

- `core/brain.py` — compatibility facade; entry point for all LLM requests
- `core/platform/orchestrator.py` — routes each request through context + strategy planning
- `core/platform/execution_planner.py` — decides direct vs. agent path
- `core/agent/loop.py` — multi-step ReAct tool loop
- `core/session.py` — typed event bridge between BrainService and terminal UIs
- `core/listener.py` — wake word detection and STT
- `core/speaker.py` — Kokoro TTS playback
- `core/memory.py` — SQLite conversation memory with token-overlap recall
- `core/second_brain/entity_store.py` — structured entity persistence

## Testing

```bash
~/micromamba/envs/kage/bin/python -m unittest discover -s tests -p 'test_*.py'
```

293 tests. No external services required — all connectors are mocked.

Quick sanity checks:

```bash
# Verify settings load correctly
python -c "import config; s = config.get(); print(s.llm_backend, s.mlx_model)"

# Check which optional features are on
python -c "import config; s = config.get(); print('agent:', s.agent_enabled, '| second_brain:', s.second_brain_enabled)"

# Inspect stored entities
sqlite3 data/memory/kage_memory.db "SELECT kind, key, value, status FROM entities;"

# Create and verify a local backup
kage backup create
kage backup verify output/kage-backup-YYYYMMDD-HHMMSS.tar.gz
```

## Roadmap

Done:
- [x] Voice loop — wake word, STT, LLM, TTS, barge-in
- [x] Persistent conversation memory with semantic recall
- [x] Second brain — tasks, facts, commitments, preferences
- [x] Agent layer — tool-using ReAct loop with web, shell, calendar, reminders
- [x] Textual chat UI with session management

Near-term:
- [x] Task completion detection — reliably call `mark_task_done` on "done / finished" phrases
- [x] Agent reliability pass — handle tool timeouts and failures gracefully
- [x] Entity dedup — merge conflicting facts instead of stacking them
- [x] Source attribution — include source URLs when web results inform a response

Later:
- [x] Streaming STT — partial transcript during speech for lower turn latency
- [ ] Token-aware context budgeting across turns, memory, and entity recall
- [ ] User-trained wake word support
