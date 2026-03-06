# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the entry point for both voice and text modes. Keep core orchestration and assistant logic in `core/` (brain, agent loop, memory, listener/speaker). Put external integrations and tool adapters in `connectors/` (web, shell, calendar, notifications).  
Tests live in `tests/` and follow `test_*.py` naming. Benchmark harnesses are in `benchmarks/`, with generated outputs under `benchmarks/reports/`. Runtime data (for example SQLite memory) is stored in `data/`. Long-form notes and tuning docs belong in `docs/`.

## Build, Test, and Development Commands
No build step is required; this is a Python 3.11 project.

```bash
micromamba create -n kage python=3.11 pip -y
micromamba activate kage
pip install -r requirements.txt
cp .env.example .env
python3 main.py --text
python3 main.py
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m py_compile main.py config.py core/*.py tests/*.py
python3 main.py --bench
python3 benchmarks/deep_capability_benchmark.py
```

Use `--text` for fast local iteration; run the unittest command before opening a PR.

## Coding Style & Naming Conventions
Use 4-space indentation and standard Python naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.  
Centralize environment parsing in `config.py`; use `config.get()` instead of reading `os.getenv()` in service modules. Keep modules focused: reusable assistant behavior in `core/`, integration boundaries in `connectors/`.

## Testing Guidelines
The suite uses `unittest` (not pytest as the primary runner). Add tests in `tests/test_<feature>.py` and keep them deterministic by mocking LLM/runtime, AppleScript, and network surfaces where possible.  
For bug fixes, include a regression test that fails before the fix and passes after it.

## Commit & Pull Request Guidelines
Recent history mixes styles, but scoped imperative messages are preferred (for example: `agent: tighten tool-call guard`, `benchmarks: add edge-case scenario`). Keep commits single-purpose.  
PRs should include: what changed, why it changed, config/env impacts, and exact test commands run. Link related issues and include sample terminal output when behavior changes.

## Security & Configuration Tips
Do not commit `.env`, secrets, or `data/memory/*.db`. Update `.env.example` when adding config keys. Document any connector that expands command, web, or OS-level permission scope.
