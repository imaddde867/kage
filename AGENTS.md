# Repository Guidelines

## Project Structure

Kage is a local Python voice assistant for macOS.

- `main.py`: entrypoint, `run_voice()` and `run_text()` loops
- `config.py`: environment loading and typed `Settings` dataclass (`config.get()`)
- `core/`: assistant services (`brain.py`, `listener.py`, `memory.py`, `speaker.py`)
- `data/memory/`: local SQLite database (`kage_memory.db`, gitignored)

New assistant logic goes in `core/`. External integrations (calendar, search, etc.) belong in `connectors/` as opt-in modules — not wired into the core loop by default.

## Commands

```bash
pip install -r requirements.txt
python main.py --text        # text mode
python main.py               # voice mode
python -m py_compile main.py config.py core/*.py   # syntax check
python -c "import config; print(config.get())"     # verify settings
```

## Coding Conventions

- Python 3.11, 4-space indentation
- `snake_case` for functions/variables, `PascalCase` for service classes
- Read all settings through `config.get()` — never `os.getenv` in service modules
- One backend per concern — no fallback chains
- Commit messages: imperative, scoped (e.g. `core: simplify brain streaming`)

## What Not To Do

- Do not add fallback TTS/STT chains
- Do not inject connector context into the core hot path
- Do not commit `.env` or `data/memory/*.db`
