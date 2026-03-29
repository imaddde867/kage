"""Single source of truth for tool name aliases used by AgentLoop.

Both the parser (for XML tag normalization) and the registry (for dispatch
lookup) import this dict.  If you add a new alias, add it here only.
"""
from __future__ import annotations

TOOL_ALIASES: dict[str, str] = {
    "search": "web_search",
    "web_search": "web_search",
    "fetch": "web_fetch",
    "web_fetch": "web_fetch",
    "calendar": "calendar_read",
    "reminder": "reminder_add",
    "shell": "shell",
    "notify": "notify",
    "speak": "speak",
    "local_find_files": "local_find_files",
    "local_read_text": "local_read_text",
    "local_extract_pdf": "local_extract_pdf",
    "local_extract_docx": "local_extract_docx",
    "local_extract_sheet": "local_extract_sheet",
    "find_file": "local_find_files",
    "read_file": "local_read_text",
    "extract_pdf": "local_extract_pdf",
    "extract_docx": "local_extract_docx",
    "extract_sheet": "local_extract_sheet",
}
