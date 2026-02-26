# Repository Guidelines

## Project Structure & Module Organization

Kage is a local Python voice assistant for macOS. Keep runtime code small and modular.

- `main.py`: app entrypoint and orchestration loop
- `config.py`: environment loading and typed settings
- `core/`: assistant services (`listener.py`, `speaker.py`, `brain.py`, `memory.py`)
- `connectors/`: Apple app integrations (`calendar.py`, `reminders.py`, `notes.py`) plus
  shared AppleScript helper (`_apple.py`)
- `data/memory/`: local SQLite memory database (`kage_memory.db`, legacy `jarvis_memory.db` may exist)
- `README.md`: setup, usage, and product vision

Put new assistant logic in `core/`; add external integrations under `connectors/`.

## Build, Test, and Development Commands

There is no build step. Run locally with Python 3.11.

- `micromamba create -n kage python=3.11 pip -y && micromamba activate kage`: create env
- `pip install -r requirements.txt`: install dependencies
- `cp .env.example .env`: create local config
- `ollama serve` and `ollama pull qwen3:8b`: start local model backend
- `python main.py`: run Kage
- `python -m py_compile main.py config.py core/*.py connectors/*.py`: quick syntax check

## Coding Style & Naming Conventions

- Python, 4-space indentation, no tabs
- `snake_case` for functions/modules/variables
- `PascalCase` for service classes (for example `BrainService`, `ListenerService`)
- `UPPER_SNAKE_CASE` for module constants and compatibility aliases in `config.py`
- Prefer small methods, explicit dependency injection, and compatibility wrappers when
  refactoring

No formatter/linter config is committed yet. If added, prefer `ruff` + `black`.

## Testing Guidelines

No automated test suite exists yet. For new work:

- Add tests under `tests/` using `pytest`
- Name files `test_<module>.py`
- Prioritize `core/memory.py`, `core/brain.py`, and connector parsing/error handling
- Include manual smoke-test notes for audio/wake-word flows (`python main.py`)

## Commit & Pull Request Guidelines

Current git history is minimal (single initial commit: `first comm`), so there is no
strong convention yet. Use concise, imperative commit messages with scope when possible
(example: `core: split brain service and Ollama client`).

PRs should include: summary, behavior impact, config/env changes, manual test steps, and
logs/screenshots if audio or connector behavior changed.

## Security & Configuration Tips

Never commit `.env` or `data/memory/*.db`. Kage handles personal context
(Calendar, Reminders, Notes), so keep processing local-first and avoid adding cloud
dependencies without explicit documentation.
