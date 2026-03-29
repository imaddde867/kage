<h1 align="center">Kage (影)</h1>
<p align="center">
  <img src="kage.gif" alt="Kage" width="420" />
</p>
<p align="center">
  A fully local, private AI assistant for macOS — with long-term memory from your Obsidian vault.
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue" />
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black" />
  <img src="https://img.shields.io/badge/cloud-optional-green" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

---

## What is this?

**Kage** is your always-on AI layer — voice or text, local or cloud, tool-using agent when needed:
- Responds via **wake word → voice** or **Textual terminal UI**
- **Episodic memory** across sessions (SQLite)
- Tracks **tasks, preferences, and facts** about you
- **Multi-step agent** with web search, calendar, shell, reminders
- Routes personal/long-context queries to a **local Qwen model**; cloud for heavy tool use

**Second Brain** (the `second_brain/` subfolder) is the knowledge layer:
- Indexes your entire **Obsidian vault** into a hybrid graph + vector store
- **Automatically injects** relevant notes into Kage's context before every response
- **Hybrid retrieval**: semantic vector search + BM25 keyword search, fused with RRF
- Standalone **web UI** and **CLI** for direct vault queries

Together they form **NeuroCache** — Kage answers from *your own knowledge*, not just training data.

---

## Architecture

```
┌──────────────────────────── Kage ──────────────────────────────┐
│                                                                  │
│  Voice  : wake word ──► STT ──► brain ──► TTS                  │
│  Text   : terminal UI (Textual) ──► brain                      │
│                                                                  │
│  brain.py                                                       │
│  ├── orchestrator      route ──► direct answer │ agent loop    │
│  ├── context_planner   inject memory + vault notes             │
│  ├── inference/        local Qwen3.5-9B (MLX) or cloud LLM    │
│  └── connectors/       web, calendar, shell, neurocache …      │
│                                   │                             │
└───────────────────────────────────┼─────────────────────────── ┘
                                    │ HTTP :8765
┌───────────────────── second_brain ▼ ──────────────────────────┐
│                                                                  │
│  FastAPI server (:8765) ◄──► QueryEngine                       │
│                               ├── vector  (ChromaDB)           │
│                               ├── keyword (BM25)               │
│                               └── graph   (Kuzu traversal)     │
│  Ingestion                                                      │
│  ├── Obsidian watcher    live re-index on save                 │
│  ├── Markdown parser     frontmatter · tags · [[links]]        │
│  └── Code ingestor       tree-sitter: Python / JS / TS         │
│                                                                  │
│  Web UI  (React/Vite, :5173)  ◄── optional                     │
└─────────────────────────────────────────────────────────────── ┘
```

---

## Hardware

| | Minimum | Recommended |
|---|---|---|
| Chip | Apple Silicon M1 | M3 / M4 |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB free |
| OS | macOS 13 | macOS 14+ |

> **M4 16 GB**: Qwen3.5-9B-4bit uses ~4.5 GB. Both Kage and the second_brain server run comfortably in parallel with ~9 GB headroom.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/imaddde867/kage.git && cd kage

# 2. Kage environment
micromamba create -n kage python=3.11 -y && micromamba activate kage
pip install -r requirements.txt
cp .env.example .env                            # edit at minimum: USER_NAME, ASSISTANT_NAME

# 3. Second Brain environment  (separate venv — different deps)
cd second_brain
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env                            # set OBSIDIAN_VAULT=/path/to/vault
deactivate && cd ..

# 4. Ollama models  (for second_brain embeddings + optional LLM answers)
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# 5. Index your vault
cd second_brain && source .venv/bin/activate
brain index ~/Documents/your-vault
deactivate && cd ..

# 6. Launch everything
bash neurocache_start.sh
```

---

## Setup

### Kage

```bash
micromamba create -n kage python=3.11 -y
micromamba activate kage
pip install -r requirements.txt
cp .env.example .env
```

Minimum `.env` edits:
```bash
LLM_BACKEND=mlx_vlm
MLX_MODEL=mlx-community/Qwen3-8B-4bit   # or Qwen3.5-9B for more RAM
USER_NAME=YourName
ASSISTANT_NAME=Kage
NEUROCACHE_ENABLED=true
NEUROCACHE_VAULT_INBOX=~/path/to/vault/00-inbox/_kage
```

### Second Brain

```bash
cd second_brain
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`second_brain/.env`:
```bash
OBSIDIAN_VAULT=~/Documents/your-vault-name    # required
CORTEX_CHAT_MODEL=qwen2.5:7b                  # for /api/ask (optional)
```

Install as a background service (auto-starts on login):
```bash
bash second_brain/scripts/install_service.sh
```

---

## Configuration reference

### Kage (`.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `mlx_vlm` | `mlx_vlm` · `mlx` · `openai_compat` |
| `MLX_MODEL` | `mlx-community/Qwen3-8B-4bit` | HF repo ID or local path |
| `USER_NAME` | `User` | Your name |
| `ASSISTANT_NAME` | `Kage` | Assistant name |
| `NEUROCACHE_ENABLED` | `false` | Inject vault context before every response |
| `NEUROCACHE_API_URL` | `http://127.0.0.1:8765` | Second brain server |
| `NEUROCACHE_VAULT_INBOX` | _(empty)_ | Where Kage writes notes back to vault |
| `AGENT_ENABLED` | `false` | Multi-step tool-using agent |
| `SECOND_BRAIN_ENABLED` | `false` | Entity/task extraction on each turn |
| `WAKE_WORD_MODEL` | `hey_jarvis` | openwakeword model name |
| `STT_BACKEND` | `apple` | `apple` · `whisper` · `parakeet` |
| `MEMORY_DIR` | `./data/memory` | Episodic memory (SQLite) |

Full reference: `.env.example` (214 settings).

### Second Brain (`second_brain/.env`)

| Variable | Default | Description |
|---|---|---|
| `OBSIDIAN_VAULT` | _(required)_ | Absolute path to your vault |
| `CORTEX_CHAT_MODEL` | `qwen2.5:7b` | Ollama model for `/api/ask` |

---

## Usage

### Launch everything

```bash
bash neurocache_start.sh         # starts second_brain server, indexes vault, launches Kage
```

What it does:
1. Checks if second_brain server is already running via launchd — starts it if not
2. Runs a one-shot vault sync (`brain index`)
3. Pre-warms the local Qwen model (if `NEUROCACHE_ENABLED=true`)
4. Launches Kage; cleans up on exit

### Kage

```bash
kage chat               # Textual TUI  (default)
kage voice              # Voice mode — wake word → STT → respond → TTS
kage bench              # Local model benchmark
kage doctor             # Environment / dependency check
kage service install    # Install Kage as a launchd daemon
kage service status
kage approvals list     # Tool approval policy
kage approvals grant <tool>
```

### Second Brain CLI

```bash
# Activate second_brain venv first, or use the installed `brain` command
brain index ~/Documents/your-vault         # index / re-index vault
brain index-code /path/to/repo            # index a code repository
brain watch ~/Documents/your-vault        # live re-index on file save
brain search "your query"                 # hybrid semantic + keyword search
brain search-code "your query"            # search indexed code
brain ask "what projects am I working on?" # LLM answer with vault citations
brain stats                               # vault + code graph counts
```

### Second Brain API (`:8765`)

```bash
# Hybrid search (vector + BM25 + RRF)
curl -X POST http://127.0.0.1:8765/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "knowledge graphs research", "top_k": 5}'

# LLM answer with vault citations
curl -X POST http://127.0.0.1:8765/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what am I working on this week?"}'

# Stats
curl http://127.0.0.1:8765/api/stats
```

### Web UI (`:5173`)

```bash
cd second_brain/cortex-ui
npm install && npm run dev
```

---

## How vault context injection works

Every message you send to Kage:

1. `context_planner.py` checks `NEUROCACHE_ENABLED`
2. `neurocache_connector.py` posts your message as a query to `:8765/api/search`
3. Second brain runs hybrid retrieval → top-k notes (≤ 3000 chars)
4. Notes are injected into the system prompt as `## Relevant notes from your vault:`
5. The router sees vault context was injected → sends to **local** Qwen model
6. Kage responds using knowledge from your own writing

**Write-back**: set `NEUROCACHE_VAULT_INBOX` and Kage can save generated notes to your vault directly via the agent's `VaultSearchTool`.

---

## Retrieval — how it works

Three signals, fused with **Reciprocal Rank Fusion (RRF)**:

| Signal | Engine | Good for |
|---|---|---|
| Semantic | ChromaDB + nomic-embed-text (768-dim) | Concepts, paraphrases, ideas |
| Keyword | BM25Okapi (in-memory, built at startup) | Exact names, acronyms, dates |
| Graph | Kuzu — `[[wiki-links]]` as edges | Related notes, topic expansion |

---

## Project structure

```
kage/
├── main.py                      # entry point
├── config.py                    # all settings (single source of truth)
├── neurocache_start.sh          # unified launcher
├── com.imad.neurocache.plist    # LaunchAgent for second_brain server
├── pyproject.toml
├── requirements.txt
├── .env.example                 # full config reference
│
├── core/                        # brain + orchestration
│   ├── brain.py                 # BrainService — main coordination
│   ├── brain_generation.py      # LLM backend abstraction
│   ├── brain_guardrails.py      # policy + safety
│   ├── brain_prompting.py       # prompt assembly
│   ├── memory.py                # episodic memory (SQLite)
│   ├── listener.py              # wake word + STT
│   ├── speaker.py               # TTS (Kokoro / AVFoundation)
│   ├── chat_shell.py            # plain text loop
│   ├── textual_chat.py          # Textual TUI
│   ├── agent/                   # ReAct agent loop + tool registry
│   ├── platform/                # orchestrator · planner · storage
│   └── second_brain/            # entity extraction · intent routing
│
├── connectors/                  # opt-in tools
│   ├── neurocache_connector.py  # vault search + write-back
│   ├── web_search.py
│   ├── web_fetch.py
│   ├── apple_calendar.py
│   ├── shell.py
│   └── ...
│
├── inference/                   # local LLM
│   ├── local_llm.py             # MLX VLM wrapper (Qwen3.5-9B)
│   ├── router.py                # LOCAL vs CLOUD routing
│   ├── client.py                # unified client
│   └── benchmark.py
│
├── tests/                       # 380 tests (all mocked)
│
└── second_brain/                # knowledge layer — separate Python env
    ├── pyproject.toml           # deps: ollama · chromadb · kuzu · rank-bm25
    ├── .env.example
    ├── api/server.py            # FastAPI :8765
    ├── graph/store.py           # Kuzu graph + ChromaDB vector store
    ├── ingestion/               # parser · watcher · code ingestor
    ├── query/                   # hybrid engine · bm25 index
    ├── interface/cli.py         # `brain` CLI
    ├── cortex-ui/               # React/Vite web UI (optional)
    └── scripts/install_service.sh
```

---

## Testing

```bash
# Kage — from repo root, kage env active
python -m pytest tests/ -q
# → 380 passed

# Second Brain — from second_brain/, venv active
python -m unittest discover -s tests -p "test_*.py" -v

# Live check
curl http://127.0.0.1:8765/api/stats
kage doctor
python -m inference.benchmark
```

---

## Roadmap

**Second Brain**
- [ ] PDF, email, and calendar ingestors
- [ ] spaCy NER for automatic entity extraction from notes
- [ ] Pre-meeting context daemon (surface relevant notes before calendar events)
- [ ] Streaming responses in API
- [ ] Cross-file function call graph

**Kage**
- [ ] Automatic vault write-back trigger on important answers
- [ ] KV cache compression once MLX exposes cache hooks
- [ ] Obsidian plugin for direct Kage interaction from vault

---

## Author

**Imad Eddine El Mouss**
[github.com/imaddde867](https://github.com/imaddde867) · Data & AI Engineer · Turku, Finland
