"""
Kage (影) — Personal AI System
────────────────────────────────

Entry point. Runs the voice loop or a terminal text chat loop.

Usage:
    cd ~/Documents/GitHub/personal/kage
    source .venv/bin/activate
    python main.py
    python main.py --text

Say your configured wake word (default: "Hey Jarvis") to activate.
Kage responds, you speak, and it replies aloud.
"""

from __future__ import annotations

import argparse
import logging
import time

import config
from connectors import ConnectorManager
from core.brain import BrainService
from core.listener import ListenerService
from core.memory import MemoryStore
from core.speaker import SpeakerService

logger = logging.getLogger(__name__)
_POST_SPEECH_WAKE_COOLDOWN_SECONDS = 0.6


class AssistantRuntime:
    def __init__(self) -> None:
        self.settings = config.get_settings()
        self.memory = MemoryStore()
        self.connectors = ConnectorManager()
        self.brain = BrainService(
            settings=self.settings,
            memory_store=self.memory,
            connector_manager=self.connectors,
        )
        self.listener = ListenerService(settings=self.settings)
        self.speaker = SpeakerService(settings=self.settings)

    def initialize_voice_services(self) -> None:
        self.listener.load_models()
        self.speaker.load_engine()

    def initialize_text_services(self) -> None:
        self.speaker.load_engine()

    def boot(self, *, text_mode: bool = False) -> None:
        wake_word_display = self.settings.wake_word.title()
        print()
        print("  ╔══════════════════════════════════╗")
        print("  ║     KAGE (影)  —  Online         ║")
        print("  ║     Shadow AI — Always Here      ║")
        print(f"  ║  Model : {self.settings.ollama_model:<24}║")
        print(f"  ║  User  : {self.settings.user_name:<24}║")
        print("  ╚══════════════════════════════════╝")
        print()
        if text_mode:
            print("  Type your message and press Enter. Type 'exit' to quit.\n")
        else:
            print(f"  Say '{wake_word_display}' to activate.\n")
        self.speaker.speak(
            f"Kage is online. I'm here whenever you need me, {self.settings.user_name}."
        )

    def shutdown(self, *, speak: bool = True) -> None:
        print("\n[Kage] Going offline.")
        if not speak:
            return
        try:
            self.speaker.speak("Going dark. I'll be here when you need me.")
        except KeyboardInterrupt:
            pass

    def respond(self, user_text: str) -> None:
        sentences: list[str] = []

        def on_sentence(s: str) -> None:
            sentences.append(s)
            self.speaker.speak(s, display=False)

        self.brain.think_stream(user_text, on_sentence)
        print(f"\n[Kage]: {' '.join(sentences)}\n")

    def run_once(self) -> None:
        self.listener.wait_for_wake_word()
        self.speaker.speak("Yeah?")

        user_text = self.listener.listen_and_transcribe()
        if not user_text:
            self.speaker.speak("Didn't catch that.")
            time.sleep(_POST_SPEECH_WAKE_COOLDOWN_SECONDS)
            return

        self.respond(user_text)
        time.sleep(_POST_SPEECH_WAKE_COOLDOWN_SECONDS)

    def run_forever(self) -> int:
        self.initialize_voice_services()
        self.boot()

        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                self.shutdown(speak=False)
                return 0
            except Exception:
                logger.exception("Unhandled runtime loop error")
                time.sleep(1)

    def run_text_forever(self) -> int:
        self.initialize_text_services()
        self.boot(text_mode=True)

        while True:
            try:
                user_text = input("[You]: ").strip()
            except EOFError:
                self.shutdown()
                return 0
            except KeyboardInterrupt:
                self.shutdown(speak=False)
                return 0

            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit", ":q"}:
                self.shutdown()
                return 0

            try:
                self.respond(user_text)
            except KeyboardInterrupt:
                self.shutdown(speak=False)
                return 0
            except Exception:
                logger.exception("Unhandled text loop error")
                time.sleep(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kage in voice or text mode.")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Use terminal text input instead of wake-word audio.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.ERROR, format="[%(levelname)s] %(name)s: %(message)s"
    )
    args = parse_args()
    runtime = AssistantRuntime()
    return runtime.run_text_forever() if args.text else runtime.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
