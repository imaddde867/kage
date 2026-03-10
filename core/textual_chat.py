from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static, TextArea

import config
from core.chat_commands import parse_slash_command
from core.chat_shell import create_session_controller, handle_command
from core.clipboard import ClipboardError, copy_to_clipboard
from core.session import SessionController, SessionEvent


@dataclass
class _TurnWidgets:
    assistant: "TranscriptBubble | None" = None


class Composer(TextArea):
    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def on_mount(self) -> None:
        self.show_line_numbers = False
        self.tab_behavior = "indent"
        self.soft_wrap = True

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            self.action_submit()
        elif event.key == "ctrl+j":
            event.stop()
            self.action_newline()

    def action_submit(self) -> None:
        text = self.text.rstrip()
        if text.strip():
            self.post_message(self.Submitted(text))

    def action_newline(self) -> None:
        self.insert("\n")


class StatusBar(Static):
    def update_state(
        self,
        *,
        mode_label: str,
        backend: str,
        model: str,
        agent_enabled: bool,
        memory_enabled: bool,
        request_status: str,
    ) -> None:
        agent = "agent:on" if agent_enabled else "agent:off"
        memory = "memory:on" if memory_enabled else "memory:off"
        content = Text()
        content.append(" Kage ", style="bold black on #d2b48c")
        content.append(f" {mode_label} ", style="bold #f3eadf on #36515d")
        content.append(f" {backend} ", style="#f3eadf on #1e2a33")
        content.append(f" {model.split('/')[-1]} ", style="#d8cbb5 on #152028")
        content.append(f" {agent} ", style="#efe2c6 on #4f5f2f")
        content.append(f" {memory} ", style="#efe2c6 on #55422b")
        content.append(f" {request_status} ", style="bold #111111 on #e0c097")
        self.update(content)


class TranscriptBubble(Static):
    def __init__(self, *, title: str, accent: str, content: str = "", classes: str = "") -> None:
        super().__init__(classes=classes)
        self._title = title
        self._accent = accent
        self._content = content
        self.set_content(content)

    def set_content(self, content: str) -> None:
        self._content = content
        body = content if content.strip() else "_..._"
        self.update(
            Panel(
                RichMarkdown(body),
                title=self._title,
                border_style=self._accent,
                padding=(0, 1),
            )
        )


class ToolRow(Static):
    def __init__(self, *, tool_name: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.set_state("running", f"Running {tool_name}")

    def set_state(self, status: str, detail: str) -> None:
        style = "#d0b07a" if status == "running" else "#7abf8e" if status == "ok" else "#d97b66"
        self.update(Panel(detail, title=f"tool:{self.tool_name}", border_style=style, padding=(0, 1)))


class EmptyState(Static):
    def set_content(self, *, assistant_name: str, backend: str, model: str) -> None:
        self.update(
            Panel(
                RichMarkdown(
                    "\n".join(
                        [
                            f"# {assistant_name} Terminal",
                            "A chat-first shell for local reasoning, tools, and memory.",
                            "",
                            "## Suggested prompts",
                            "- Summarize my open tasks and tell me what matters first.",
                            "- Search the web for the latest Apple Silicon MLX updates.",
                            "- Draft a concise reply to a message and keep my style direct.",
                            "",
                            "## Key hints",
                            "- `Enter` send",
                            "- `Ctrl+J` newline",
                            "- `Ctrl+B` toggle sidebar",
                            "- `Ctrl+S` sources",
                            "- `Ctrl+M` memory",
                            "- `Ctrl+Y` copy last answer",
                            "- `Ctrl+N` new chat",
                            "",
                            f"Backend: `{backend}`  Model: `{model}`",
                        ]
                    )
                ),
                title="Ready",
                border_style="#4a7285",
                padding=(1, 2),
            )
        )


class SidebarPanel(Static):
    def set_panel(self, *, title: str, body: str) -> None:
        text = body.strip() or "No data yet."
        self.update(
            Panel(
                RichMarkdown(text),
                title=title,
                border_style="#5c7d6e",
                padding=(0, 1),
            )
        )


class KageChatApp(App[None]):
    CSS = """
    Screen {
        background: #0d141a;
        color: #f4ece2;
    }

    #status-bar {
        height: 1;
        dock: top;
    }

    #body {
        height: 1fr;
        layout: horizontal;
    }

    #main-column {
        width: 1fr;
        layout: vertical;
    }

    #transcript {
        height: 1fr;
        padding: 1 2;
    }

    #composer-wrap {
        height: 9;
        border-top: solid #32434d;
        padding: 1 2;
        background: #111b22;
    }

    #composer {
        height: 100%;
        border: round #56717f;
        background: #13212a;
    }

    #sidebar {
        width: 38;
        min-width: 30;
        border-left: solid #314652;
        padding: 1;
        background: #10191f;
    }

    #sidebar.-hidden {
        display: none;
    }

    .bubble-user {
        margin: 0 0 1 0;
    }

    .bubble-assistant {
        margin: 0 0 1 0;
    }

    .notice {
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [
        ("ctrl+n", "new_chat", "New Chat"),
        ("ctrl+b", "toggle_sidebar", "Sidebar"),
        ("ctrl+s", "show_sources", "Sources"),
        ("ctrl+h", "show_history", "History"),
        ("ctrl+m", "show_memory", "Memory"),
        ("ctrl+y", "copy_last_answer", "Copy Last"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, *, controller: SessionController, timing: bool = False) -> None:
        super().__init__()
        self.controller = controller
        self.timing = timing
        self._sidebar_mode = "sources"
        self._turn_widgets = _TurnWidgets()
        self._tool_rows: list[ToolRow] = []
        self._request_status = "idle"
        self._last_metrics: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar")
        with Horizontal(id="body"):
            with Vertical(id="main-column"):
                yield VerticalScroll(id="transcript")
                with Vertical(id="composer-wrap"):
                    yield Composer(id="composer")
            yield SidebarPanel(id="sidebar", classes="-hidden")

    def on_mount(self) -> None:
        self.set_interval(0.05, self._drain_events)
        self._refresh_empty_state()
        self._refresh_status_bar()
        self.query_one(Composer).focus()

    async def on_composer_submitted(self, event: Composer.Submitted) -> None:
        composer = self.query_one(Composer)
        command = parse_slash_command(event.text)
        composer.clear()
        if command is not None:
            await self._run_command(command)
            composer.focus()
            return
        if self.controller.is_busy:
            self.notify("Kage is still responding.", title="Busy")
            composer.focus()
            return

        await self._append_bubble(title="You", accent="#8fb7c5", content=event.text, classes="bubble-user")
        self._remove_empty_state()
        self._turn_widgets = _TurnWidgets()
        self._tool_rows = []
        composer.disabled = True
        self.controller.submit(event.text)
        self._request_status = "thinking"
        self._refresh_status_bar()
        self._scroll_transcript()

    async def _run_command(self, command) -> None:
        outcome = handle_command(self.controller, command)
        if outcome.should_exit:
            self.exit()
            return
        if command.name == "new_chat":
            self._clear_transcript()
            self._turn_widgets = _TurnWidgets()
            self._tool_rows = []
            self._request_status = "idle"
            self._refresh_status_bar()
            self._refresh_empty_state()
        elif command.name == "show_sources":
            self.action_show_sources()
        elif command.name == "show_memory":
            self.action_show_memory()
        elif command.name == "toggle_sidebar":
            self.action_toggle_sidebar()
        elif command.name == "copy_last_answer" and not outcome.message:
            self.notify("Copied the last answer.", title="Clipboard")
            return

        if outcome.message:
            await self._append_notice(outcome.message, accent="#7e9cac")
        self._scroll_transcript()

    def action_new_chat(self) -> None:
        if self.controller.is_busy:
            self.notify("Wait for the current reply to finish before resetting.", title="Busy")
            return
        self.controller.reset()
        self._clear_transcript()
        self._turn_widgets = _TurnWidgets()
        self._tool_rows = []
        self._request_status = "idle"
        self._refresh_status_bar()
        self._refresh_empty_state(force=True)

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar", SidebarPanel)
        if sidebar.has_class("-hidden"):
            sidebar.remove_class("-hidden")
            self._refresh_sidebar()
        else:
            sidebar.add_class("-hidden")

    def action_show_sources(self) -> None:
        self._sidebar_mode = "sources"
        sidebar = self.query_one("#sidebar", SidebarPanel)
        sidebar.remove_class("-hidden")
        self._refresh_sidebar()

    def action_show_history(self) -> None:
        self._sidebar_mode = "history"
        sidebar = self.query_one("#sidebar", SidebarPanel)
        sidebar.remove_class("-hidden")
        self._refresh_sidebar()

    def action_show_memory(self) -> None:
        self._sidebar_mode = "memory"
        sidebar = self.query_one("#sidebar", SidebarPanel)
        sidebar.remove_class("-hidden")
        self._refresh_sidebar()

    def action_copy_last_answer(self) -> None:
        if not self.controller.last_answer.strip():
            self.notify("No completed answer is available yet.", title="Clipboard")
            return
        try:
            copy_to_clipboard(self.controller.last_answer)
        except ClipboardError as exc:
            self.notify(str(exc), title="Clipboard")
            return
        self.notify("Copied the last answer.", title="Clipboard")

    def _drain_events(self) -> None:
        while True:
            event = self.controller.next_event(timeout=0.0)
            if event is None:
                break
            self._handle_session_event(event)

    def _handle_session_event(self, event: SessionEvent) -> None:
        if event.kind == "status_changed":
            self._request_status = event.data.get("status", "idle")
            self._refresh_status_bar(detail=event.text)
            return

        if event.kind == "assistant_chunk":
            self._update_assistant_bubble(event.text)
            return

        if event.kind == "assistant_done":
            self._request_status = "done"
            self._finish_assistant_turn(event)
            return

        if event.kind == "tool_started":
            self._add_tool_row(event)
            return

        if event.kind == "tool_finished":
            self._finish_tool_row(event)
            return

        if event.kind == "source_added":
            if self._sidebar_mode == "sources":
                self._refresh_sidebar()
            return

        if event.kind == "metrics_updated":
            self._last_metrics = dict(event.data)
            self._refresh_status_bar()
            return

        if event.kind == "error":
            self._show_error_notice(event.text)
            return

    async def _append_bubble(self, *, title: str, accent: str, content: str, classes: str) -> TranscriptBubble:
        bubble = TranscriptBubble(title=title, accent=accent, content=content, classes=classes)
        await self.query_one("#transcript", VerticalScroll).mount(bubble)
        return bubble

    async def _append_notice(self, message: str, *, accent: str) -> None:
        panel = TranscriptBubble(title="Notice", accent=accent, content=message, classes="notice")
        await self.query_one("#transcript", VerticalScroll).mount(panel)

    def _update_assistant_bubble(self, chunk: str) -> None:
        if self._turn_widgets.assistant is None:
            bubble = TranscriptBubble(title="Kage", accent="#d2b48c", content="", classes="bubble-assistant")
            self.query_one("#transcript", VerticalScroll).mount(bubble)
            self._turn_widgets.assistant = bubble
        current = getattr(self._turn_widgets.assistant, "_content", "")
        self._turn_widgets.assistant.set_content(current + chunk)
        self._scroll_transcript()

    def _finish_assistant_turn(self, event: SessionEvent) -> None:
        composer = self.query_one(Composer)
        if self._turn_widgets.assistant is None and event.text.strip():
            bubble = TranscriptBubble(title="Kage", accent="#d2b48c", content=event.text, classes="bubble-assistant")
            self.query_one("#transcript", VerticalScroll).mount(bubble)
            self._turn_widgets.assistant = bubble
        elif self._turn_widgets.assistant is not None:
            self._turn_widgets.assistant.set_content(event.text or self._turn_widgets.assistant._content)
        composer.disabled = False
        composer.focus()
        self._refresh_sidebar()
        self._refresh_status_bar()
        self._scroll_transcript()

    def _add_tool_row(self, event: SessionEvent) -> None:
        row = ToolRow(tool_name=event.data.get("tool_name", event.text or "tool"))
        self._tool_rows.append(row)
        self.query_one("#transcript", VerticalScroll).mount(row)
        self._scroll_transcript()

    def _finish_tool_row(self, event: SessionEvent) -> None:
        tool_name = event.data.get("tool_name", event.text)
        for row in reversed(self._tool_rows):
            if row.tool_name == tool_name:
                status = event.data.get("status", "ok")
                preview = event.data.get("preview", "")
                if status == "ok":
                    detail = preview[:120] if preview else f"{tool_name} finished"
                else:
                    detail = preview[:160] if preview else f"{tool_name} failed"
                row.set_state(status, detail)
                break
        self._scroll_transcript()

    def _show_error_notice(self, message: str) -> None:
        self.notify(message, title="Kage")
        self.query_one("#transcript", VerticalScroll).mount(
            TranscriptBubble(title="Error", accent="#d97b66", content=message, classes="notice")
        )
        composer = self.query_one(Composer)
        composer.disabled = False
        composer.focus()
        self._scroll_transcript()

    def _clear_transcript(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        for child in list(transcript.children):
            child.remove()

    def _refresh_empty_state(self, *, force: bool = False) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        if not force and len(transcript.children) > 0:
            return
        empty = EmptyState()
        empty.set_content(
            assistant_name=getattr(self.controller.settings, "assistant_name", "Kage"),
            backend=getattr(self.controller.settings, "llm_backend", "unknown"),
            model=getattr(self.controller.settings, "mlx_model", "unknown"),
        )
        transcript.mount(empty)

    def _remove_empty_state(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        for child in list(transcript.children):
            if isinstance(child, EmptyState):
                child.remove()

    def _refresh_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar", SidebarPanel)
        if sidebar.has_class("-hidden"):
            return
        if self._sidebar_mode == "sources":
            sources = self.controller.current_sources()
            body = "\n".join(f"- {source}" for source in sources) or "No sources captured in this turn yet."
            sidebar.set_panel(title="Sources", body=body)
            return
        if self._sidebar_mode == "history":
            turns = self.controller.recent_history(limit=6)
            parts: list[str] = []
            for idx, (user_text, reply_text) in enumerate(turns, 1):
                parts.append(f"### Turn {idx}\nUser: {user_text}\n\nKage: {reply_text}")
            sidebar.set_panel(title="Recent History", body="\n\n".join(parts) or "No recent history available.")
            return
        body = self.controller.memory_summary() + "\n\n## Connectors\n" + self.controller.connector_summary()
        sidebar.set_panel(title="Memory + Connectors", body=body)

    def _refresh_status_bar(self, *, detail: str = "") -> None:
        status = self._request_status
        if detail and status not in {"done", "idle"}:
            status = f"{status}: {detail}"
        if self.timing and self._last_metrics:
            tok_s = self._last_metrics.get("tok_per_sec")
            if tok_s:
                status = f"{status} | {float(tok_s):.1f} tok/s"
        self.query_one(StatusBar).update_state(
            mode_label="Textual chat",
            backend=getattr(self.controller.settings, "llm_backend", "unknown"),
            model=getattr(self.controller.settings, "mlx_model", "unknown"),
            agent_enabled=bool(getattr(self.controller.settings, "agent_enabled", False)),
            memory_enabled=bool(getattr(self.controller.settings, "second_brain_enabled", False)),
            request_status=status,
        )

    def _scroll_transcript(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.scroll_end(animate=False)


def run_textual_chat(
    settings: config.Settings,
    *,
    timing: bool = False,
    brain: Any | None = None,
) -> None:
    controller = create_session_controller(settings=settings, brain=brain)
    try:
        KageChatApp(controller=controller, timing=timing).run()
    finally:
        controller.close()
