# second_brain

The knowledge layer of Kage — indexes your Obsidian vault and code into a hybrid graph + vector store, then serves relevant context to Kage before every response.

Part of the [Kage](../README.md) monorepo.

---

## What it does

- **Indexes** Obsidian markdown notes and code repositories (Python, JS, TS via tree-sitter)
- **Hybrid retrieval**: semantic vector search (ChromaDB) + BM25 keyword search, fused with RRF
- **Graph**: `[[wiki-links]]` and `#tags` stored as typed edges in Kuzu — enables topic expansion
- **Serves** a FastAPI query API on `:8765` (consumed by Kage's `neurocache_connector.py`)
- **Web UI** (React/Vite) for standalone vault search and LLM-powered Q&A
- **Live watcher**: re-indexes notes automatically on save

Everything runs locally — no cloud, no API keys.

---

## Setup

```bash
# from the kage/ root
cd second_brain
python -m venv .venv && source .venv/bin/activate
pip install -e .

# pull Ollama models
ollama pull nomic-embed-text          # embeddings (required)
ollama pull qwen2.5:7b               # for /api/ask (optional)

# configure
cp .env.example .env
# edit .env: set OBSIDIAN_VAULT=/absolute/path/to/your/vault

# index your vault
brain index ~/Documents/your-vault

# start the server
uvicorn api.server:app --port 8765 --host 127.0.0.1
```

Or install as a background service that auto-starts on login:
```bash
bash scripts/install_service.sh          # install
bash scripts/install_service.sh status  # check
bash scripts/install_service.sh stop    # stop
```

---

## Configuration

`second_brain/.env`:

| Variable | Default | Description |
|---|---|---|
| `OBSIDIAN_VAULT` | _(required)_ | Absolute path to your Obsidian vault |
| `CORTEX_CHAT_MODEL` | `qwen2.5:7b` | Ollama model used by `/api/ask` |

---

## CLI reference

```bash
brain index <vault-path>              # index or re-index vault
brain index-code <repo-path>         # index a code repository
brain watch <vault-path>             # live re-index on file changes
brain search "<query>"               # hybrid search (vector + BM25)
brain search-code "<query>"          # search code nodes only
brain ask "<question>"               # LLM answer with vault citations
brain stats                          # vault + code graph counts
```

---

## API reference

Server: `http://127.0.0.1:8765`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/search` | Hybrid search — `{"query": str, "top_k": int}` |
| `POST` | `/api/ask` | LLM answer — `{"question": str, "model": str\|null}` |
| `GET` | `/api/stats` | Note / tag / entity counts |
| `GET` | `/api/graph/entities` | All named entities |
| `GET` | `/api/graph/tags` | Tags with note counts |
| `POST` | `/api/search-code` | Code-only search — `{"query": str, "top_k": int}` |
| `GET` | `/api/code-stats` | Repos / files / functions / classes |

---

## Retrieval details

`QueryEngine.search()` runs three signals and fuses them with **Reciprocal Rank Fusion (RRF, k=60)**:

1. **Vector** — `ChromaDB.query()` with nomic-embed-text embeddings (768-dim, cosine)
2. **BM25** — `BM25Okapi` index built in-memory at startup from ChromaDB documents
3. **Graph** — Kuzu `[[LINKS_TO]]` traversal expands results via wiki-link relationships

The `BM25Index` handles empty corpora gracefully (returns `[]` before the vault is indexed).

---

## Graph schema

**Nodes**: `Note` · `Tag` · `Entity` · `CodeFile` · `Function` · `Class` · `Module` · `Repository`

**Edges**: `TAGGED` · `LINKS_TO` · `RELATED_TO` · `CONTAINS` · `CONTAINS_CLASS` · `HAS_METHOD` · `IMPORTS` · `IN_REPO` · `CALLS`

---

## Structure

```
second_brain/
├── pyproject.toml          # deps: ollama · chromadb · kuzu · rank-bm25 · watchdog
├── .env.example
├── api/
│   └── server.py           # FastAPI application (7 endpoints)
├── graph/
│   └── store.py            # BrainStore — Kuzu graph + ChromaDB, MERGE-based upserts
├── ingestion/
│   ├── parser.py           # ParsedNote from markdown (frontmatter · tags · links)
│   ├── watcher.py          # VaultHandler (watchdog) + watch_vault()
│   └── code_ingestor.py    # tree-sitter AST extraction + graph indexing
├── query/
│   ├── engine.py           # QueryEngine: hybrid search · RRF · LLM answers
│   └── bm25_index.py       # BM25Index: in-memory keyword index
├── interface/
│   └── cli.py              # `brain` Typer CLI
├── cortex-ui/              # React/Vite web UI (optional)
│   └── src/App.jsx         # chat interface + sidebar stats
├── scripts/
│   └── install_service.sh  # LaunchAgent management
└── tests/
    ├── test_query_engine_models.py
    ├── test_api_ask_model.py
    └── test_cli_ask_model.py
```

---

## Testing

```bash
# from second_brain/, venv active
python -m unittest discover -s tests -p "test_*.py" -v

# live check (server must be running)
curl http://127.0.0.1:8765/api/stats
curl -X POST http://127.0.0.1:8765/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "active projects", "top_k": 3}'
```

---

## Data location

Runtime data is stored in `second_brain/data/` (gitignored):
- `data/graph.kuzu` — Kuzu database (graph nodes + edges)
- `data/chroma/` — ChromaDB persistent store (embeddings)

To reset the index: `rm -rf second_brain/data/ && brain index <vault-path>`
