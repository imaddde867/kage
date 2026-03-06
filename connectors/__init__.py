"""Optional connectors — web search, memory ops, notifications, shell, calendar.

Each connector is a Tool subclass.  Tools are registered at startup in
BrainService._build_tool_registry() and become available to AgentLoop.

Connector catalogue
-------------------
web_search.py       DuckDuckGo text search, no API key.
                    Requires: pip install duckduckgo-search

web_fetch.py        Fetch a URL and extract readable text.
                    Requires: pip install "scrapling[fetchers]" httpx trafilatura

memory_ops.py       Read/write EntityStore from agent turns:
                      mark_task_done  — mark a task or commitment as done
                      update_fact     — upsert any entity (profile, preference, …)
                      list_open_tasks — dump all active tasks/commitments

notify.py           macOS system notification (NotifyTool) and
                    direct TTS speech (SpeakTool) via core.speaker.

shell.py            Read-only shell connector (`shell`) plus gated mutation
                    connector (`shell_mutation`) with explicit confirmation.

apple_calendar.py   Read upcoming Calendar events and add Reminders,
                    both via osascript (macOS only).

Adding a new connector
----------------------
1. Create a file in this package with a Tool subclass.
2. Register it in BrainService._build_tool_registry().
   Use a try/except ImportError if the connector has optional dependencies.
"""
