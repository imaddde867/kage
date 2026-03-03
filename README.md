# Kage (影)

A fully local, always-on personal AI for macOS. Wake word activates it, you speak, it responds aloud. No cloud, no subscriptions, no data leaving your machine.

**Current state:** functional voice + text loop with persistent second-brain memory. Kage now extracts structured entities (tasks, commitments, profile facts, preferences) from conversation, routes intent before every LLM call, injects relevant context into prompts, and proactively surfaces open items when appropriate — all without extra cloud calls.

---

## Stack

| Layer | Tool | Why |
|-------|------|-----|
| LLM | [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) / [MLX-LM](https://github.com/ml-explore/mlx-lm) | Fast local inference on Apple Silicon |
| Wake word | [openwakeword](https://github.com/dscripka/openWakeWord) | Lightweight, CPU-only, works offline |
| STT | macOS native (`SpeechRecognition`) | Hardware-accelerated, zero latency |
| STT fallback | [faster-whisper](https://github.com/guillaumekleeven/faster-whisper) | CPU-only, no GPU needed |
| TTS | [mlx-audio](https://github.com/Blaizzy/mlx-audio) + Kokoro-82M | Local MLX synthesis on Apple Silicon |
| Memory | SQLite | No server, simple, durable |
| Audio | [sounddevice](https://python-sounddevice.readthedocs.io) | Clean cross-platform mic access |

---

## Setup

**Prerequisites:** Python 3.11 on Apple Silicon.

```bash
# 1. Create environment
micromamba create -n kage python=3.11 pip -y && micromamba activate kage

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — set your model and Kokoro voice preset

# 4. Run
python main.py --text     # text mode (no mic needed)
python main.py            # voice mode (wake word → speak → respond)
```

**Popular Kokoro voices:** `af_heart` (US), `bf_emma` (UK), `bm_george` (UK)

---

## Configuration (`.env`)

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `mlx_vlm` | `mlx_vlm` for Qwen3.5, `mlx` for text-only MLX-LM models |
| `MLX_MODEL` | `mlx-community/Qwen3.5-4B-MLX-4bit` | LLM model repo |
| `MLX_MAX_TOKENS` | `250` | Max generation length (bumped from 150 to accommodate entity context) |
| `TEMPERATURE` | `0.3` | Generation temperature; lower = less hallucination |
| `KOKORO_MODEL` | `mlx-community/Kokoro-82M-bf16` | Kokoro model repo |
| `KOKORO_VOICE` | `af_heart` | Voice preset |
| `KOKORO_LANG_CODE` | `en-us` | Accent/language (`en-us`, `en-gb`, `ja`, `zh`) |
| `KOKORO_SPEED` | `1.0` | Speaking rate multiplier |
| `STT_BACKEND` | `apple` | `apple` or `whisper` |
| `WAKE_WORD` | `hey jarvis` | Display name |
| `WAKE_WORD_MODEL` | `hey_jarvis` | openwakeword model file |
| `WAKE_WORD_THRESHOLD` | `0.5` | Detection sensitivity (0–1) |

### Barge-in / Turn-taking

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOW_BARGE_IN` | `true` | Allow interrupting Kage while it speaks |
| `INTERRUPT_MIN_SCORE` | `0.55` | Wake score threshold during TTS |
| `INTERRUPT_HOLD_MS` | `220` | Required speech duration after wake hit |
| `INTERRUPT_DEBOUNCE_MS` | `500` | Minimum gap between accepted interrupts |
| `POST_TTS_GUARD_MS` | `250` | Delay before opening mic after TTS stops |

### Identity

| Variable | Default | Description |
|----------|---------|-------------|
| `USER_NAME` | `Imad` | Your name (used in prompts and entity facts) |
| `ASSISTANT_NAME` | `Kage` | Canonical assistant name kept in text |
| `TTS_NAME_OVERRIDE_ENABLED` | `true` | Apply spoken-name replacement before TTS |
| `TTS_NAME_PRONUNCIATION` | `Kah-gay` | Spoken alias to force hard-g pronunciation |
| `STT_NAME_NORMALIZATION_ENABLED` | `true` | Normalize recognized variants back to canonical name |
| `STT_NAME_VARIANTS` | `kage,cage,kaj,kaige,kahge,ka-geh` | Comma-separated variants mapped to `ASSISTANT_NAME` |
| `TEXT_MODE_TTS_ENABLED` | `false` | Speak responses in `--text` mode |

### Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_DIR` | `./data/memory` | SQLite DB location (`~` supported) |
| `RECENT_TURNS` | `4` | Latest turns injected as short-term chat context |

### Second Brain

| Variable | Default | Description |
|----------|---------|-------------|
| `SECOND_BRAIN_ENABLED` | `true` | Master switch — `false` gives identical behavior to pre-second-brain |
| `EXTRACTION_ENABLED` | `true` | Run entity extraction in background thread after each turn |
| `ENTITY_RECALL_BUDGET` | `400` | Max chars of entity context injected into each prompt |
| `PROACTIVE_DEBOUNCE_SECONDS` | `60` | Minimum seconds between consecutive proactive suggestions |

---

## Architecture

```
main.py
├── voice mode: ListenerService → BrainService → speak()
└── text mode:  input()         → BrainService → print (optional speak)

core/
├── audio_coordinator.py   state machine for listen/think/speak + barge-in guards
├── brain.py               BrainService orchestration:
│                            update_policy_state → deterministic_response →
│                            IntentRouter.classify → _build_messages (entity injection) →
│                            LLM stream → _persist_exchange (background extraction) →
│                            ProactiveEngine.suggest
├── brain_generation.py    backend loading + raw token streaming + perf stats
├── brain_guardrails.py    deterministic policy conflict/safety responses
├── brain_prompting.py     prompt templates + recent-turn + entity context assembly
├── listener.py            ListenerService — wake word + record + STT
├── memory.py              MemoryStore — SQLite conversations + entities schema,
│                            bounded token-overlap recall
├── second_brain/
│   ├── entity_store.py    EntityStore — SQLite CRUD for profile/tasks/commitments/preferences
│   ├── extractor.py       EntityExtractor — regex extraction, runs in daemon thread
│   ├── planner.py         IntentRouter — keyword/regex intent classification, no LLM
│   └── proactive.py       ProactiveEngine — debounced next-action suggestions
└── speaker.py             speak() — mlx-audio Kokoro synthesis + playback

config.py        Settings dataclass, loaded once via lru_cache
data/memory/     kage_memory.db (gitignored)
```

### Turn control flow

```
User input
  → update_policy_state()          [guardrails state update]
  → deterministic_response()       [short-circuit if policy conflict triggered]
  → IntentRouter.classify()        [pure Python, no LLM — assigns intent + flags]
  → _build_messages()              [injects entity context when route.inject_entities]
  → LLM stream                     [yields sentences to caller]
  → _persist_exchange()            [stores to DB; spawns extraction thread if route.should_extract]
  → ProactiveEngine.suggest()      [appended only when route.proactive_ok, max 1 per turn]
```

### Intent routing table

| Intent | Example triggers | Injects entities | Extracts | Proactive |
|--------|-----------------|-----------------|----------|-----------|
| `TASK_CAPTURE` | "remind me to", "add a task", "I need to" | Yes | Yes | No |
| `COMMITMENT` | "I have a meeting", "I promised", "I agreed" | Yes | Yes | Yes |
| `PLANNING_REQUEST` | "what should I", "what's next", "help me plan" | Yes | No | Yes |
| `RECALL_REQUEST` | "do you remember", "what did I tell you" | Yes | No | No |
| `PROFILE_UPDATE` | "I live in", "my timezone is" | No | Yes | No |
| `PREFERENCE` | "I prefer", "I don't like", "I always" | No | Yes | No |
| `GENERAL` | everything else | No | No | No |

### Entity schema (`entities` table)

```sql
CREATE TABLE entities (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,   -- 'profile' | 'task' | 'commitment' | 'preference'
    key        TEXT NOT NULL,   -- short label (e.g. 'location', task slug)
    value      TEXT NOT NULL,   -- always text; what gets shown in prompts
    status     TEXT DEFAULT 'active',  -- 'active' | 'done' | 'cancelled'
    due_date   TEXT,            -- ISO date or NULL
    source_id  TEXT,            -- FK to conversations.id
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

The entity block injected into prompts looks like:

```
Known facts about Imad:
Tasks: review the PR (due 2026-03-04), finish report draft (due 2026-03-07)
Commitments: team standup (2026-03-05)
Profile: location=Turku Finland
Preferences: prefers concise answers
```

### Proactive suggestions

Proactive suggestions fire only for `COMMITMENT` and `PLANNING_REQUEST` intents, at most once per turn, at most once every `PROACTIVE_DEBOUNCE_SECONDS`, and never if the entity is already mentioned in the reply. They are prefixed `"By the way, ..."` so they read naturally in voice.

---

## Development principles

1. **One backend per concern.** No fallback chains. Pick one and make it work.
2. **No feature until it works end-to-end.** Don't wire up half-built code.
3. **No abstraction until it's needed 3+ times.** Duplicate twice; abstract on the third.
4. **Connectors are opt-in plugins, not core.** Calendar/Reminders should not be in the hot path.
5. **Test before wiring.** If you can't test it in isolation, don't add it.
6. **Second brain is a master switch.** `SECOND_BRAIN_ENABLED=false` produces identical behavior to the pre-second-brain codebase. All 45 tests pass either way.

---

## Testing

See [`TESTING.md`](TESTING.md) for a comprehensive test protocol covering:
- Entity extraction and memory recall
- Intent routing edge cases
- Proactive suggestion behavior
- Reasoning, logic, and calibration
- Instruction following and honesty
- Self-awareness and contradiction traps
- Coding assistance
- Context recall across turns

Quick smoke test after setup:

```bash
# All 45 unit tests
python -m unittest discover -s tests -p 'test_*.py'

# Verify config loads with new fields
python -c "import config; s = config.get(); print(s.second_brain_enabled, s.entity_recall_budget)"

# Inspect entity DB after a conversation
sqlite3 data/memory/kage_memory.db ".schema entities"
sqlite3 data/memory/kage_memory.db "SELECT kind, key, value, due_date, status FROM entities;"
```

---

## Roadmap

Ordered by value delivered:

- [x] **Second brain entity memory** — tasks, commitments, profile, preferences extracted and recalled
- [x] **Intent routing** — classify before LLM call; no entity injection on generic queries
- [x] **Proactive suggestions** — debounced, mention-checked, voice-natural
- [ ] **Mark done via speech** — "I finished the report" → entity `status=done`
- [ ] **Better memory recall** — BM25 or embeddings instead of keyword matching
- [ ] **Streaming STT** — start transcribing while the user is still speaking
- [ ] **Calendar connector** — read-only, opt-in, only injected when relevant
- [ ] **Web search** — on-demand via a tool call, not always-on
- [ ] **Context budgeting** — tune recent-turn and long-term memory token budgets adaptively
- [ ] **Wake word customization** — train a custom openwakeword model

---

## What was removed

KittenTTS, AVSpeech, phonemizer, connectors (calendar/reminders/notes/things), multi-backend TTS switching, streaming session complexity. These added code weight without proportional value at this stage. They can be reintroduced one at a time when the foundation is solid.
