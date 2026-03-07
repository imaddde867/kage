from __future__ import annotations

import sys

import config
from core.brain import BrainService
from core.chat_commands import SlashCommand, parse_slash_command
from core.clipboard import ClipboardError, copy_to_clipboard
from core.session import SessionController, SessionEvent


class CommandOutcome:
    def __init__(self, *, should_exit: bool = False, message: str = "") -> None:
        self.should_exit = should_exit
        self.message = message


def handle_command(controller: SessionController, command: SlashCommand) -> CommandOutcome:
    if command.name == "new_chat":
        controller.reset()
        return CommandOutcome(message="Started a new chat session.")
    if command.name == "toggle_sidebar":
        return CommandOutcome(message="Sidebar is only available in the full-screen chat UI.")
    if command.name == "show_sources":
        sources = controller.current_sources()
        if not sources:
            return CommandOutcome(message="No sources captured in this turn.")
        return CommandOutcome(message="Sources:\n" + "\n".join(f"- {source}" for source in sources))
    if command.name == "show_memory":
        return CommandOutcome(
            message=controller.memory_summary() + "\n\nConnector status:\n" + controller.connector_summary()
        )
    if command.name == "copy_last_answer":
        if not controller.last_answer.strip():
            return CommandOutcome(message="There is no completed answer to copy yet.")
        try:
            copy_to_clipboard(controller.last_answer)
        except ClipboardError as exc:
            return CommandOutcome(message=str(exc))
        return CommandOutcome(message="Copied the last answer to the clipboard.")
    if command.name == "quit":
        return CommandOutcome(should_exit=True)
    return CommandOutcome(message="Available commands: /new, /sources, /memory, /copy, /quit")


def create_session_controller(*, settings: config.Settings, brain: BrainService | None = None) -> SessionController:
    active_brain = brain or BrainService(settings=settings)
    return SessionController(brain=active_brain)


def run_plain_chat(
    settings: config.Settings,
    *,
    timing: bool = False,
    brain: BrainService | None = None,
) -> None:
    controller = create_session_controller(settings=settings, brain=brain)
    _drain_startup_events(controller)
    print(
        f"  {settings.assistant_name} plain chat. Type your message. '/quit' to exit. "
        f"[backend: {settings.llm_backend} | model: {settings.mlx_model}]\n"
    )

    while True:
        try:
            raw = input("[You]: ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n[{settings.assistant_name}] Going offline.")
            return

        user_text = raw.strip()
        if not user_text:
            continue

        command = parse_slash_command(user_text)
        if command is not None:
            outcome = handle_command(controller, command)
            if outcome.message:
                print(f"[{settings.assistant_name}] {outcome.message}\n")
            if outcome.should_exit:
                return
            continue

        controller.submit(user_text)
        _render_plain_turn(controller, assistant_name=settings.assistant_name, timing=timing)


def _drain_startup_events(controller: SessionController) -> None:
    while True:
        event = controller.next_event(timeout=0.0)
        if event is None:
            return


def _render_plain_turn(controller: SessionController, *, assistant_name: str, timing: bool) -> None:
    printed_reply = False
    while controller.is_busy:
        event = controller.next_event(timeout=0.05)
        if event is None:
            continue
        printed_reply = _handle_plain_event(event, assistant_name=assistant_name, timing=timing, printed_reply=printed_reply)

    while True:
        event = controller.next_event(timeout=0.0)
        if event is None:
            break
        printed_reply = _handle_plain_event(event, assistant_name=assistant_name, timing=timing, printed_reply=printed_reply)

    if printed_reply:
        print()


def _handle_plain_event(
    event: SessionEvent,
    *,
    assistant_name: str,
    timing: bool,
    printed_reply: bool,
) -> bool:
    if event.kind == "assistant_chunk":
        if not printed_reply:
            sys.stdout.write(f"[{assistant_name}]: ")
            printed_reply = True
        sys.stdout.write(event.text)
        sys.stdout.flush()
        return printed_reply

    if event.kind == "tool_started":
        print(f"\n[{assistant_name}] Using {event.data.get('tool_name', event.text)}...")
        return printed_reply

    if event.kind == "tool_finished" and event.data.get("status") == "error":
        print(f"[{assistant_name}] Tool warning: {event.data.get('preview', '')}")
        return printed_reply

    if event.kind == "metrics_updated" and timing:
        tokens = event.data.get("tokens", "?")
        tps = event.data.get("tok_per_sec", "?")
        backend = event.data.get("backend", "?")
        print(f"\n[{assistant_name}] metrics: backend={backend} tokens={tokens} tok/s={tps}")
        return printed_reply

    if event.kind == "error":
        print(f"\n[{assistant_name}] Error: {event.text}")
        return printed_reply

    return printed_reply
