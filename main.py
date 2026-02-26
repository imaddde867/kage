"""
Kage (影) — Personal AI System
────────────────────────────────
"Shadow" in Japanese. Always present, always aware.

Entry point. Runs the always-on voice loop.

Usage:
    cd ~/Documents/GitHub/personal/jarvis
    source .venv/bin/activate
    python main.py

Say "Hey Jarvis" to activate → Kage responds → you speak → it responds aloud.
(Custom wake word "Hey Kage" coming in a future update.)
"""

from __future__ import annotations

import logging
import time

import config
from connectors import ConnectorManager
from core.brain import BrainService
from core.listener import ListenerService
from core.memory import MemoryStore
from core.speaker import SpeakerService

logger = logging.getLogger(__name__)


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

    def initialize_services(self) -> None:
        self.listener.load_models()
        self.speaker.load_engine()

    def boot(self) -> None:
        print()
        print("  ╔══════════════════════════════════╗")
        print("  ║     KAGE (影)  —  Online         ║")
        print("  ║     Shadow AI — Always Here      ║")
        print(f"  ║  Model : {self.settings.ollama_model:<24}║")
        print(f"  ║  User  : {self.settings.user_name:<24}║")
        print("  ╚══════════════════════════════════╝")
        print()
        print("  Say 'Hey Jarvis' to activate.\n")
        self.speaker.speak(f"Kage is online. I'm here whenever you need me, {self.settings.user_name}.")

    def shutdown(self) -> None:
        print("\n[Kage] Going offline.")
        self.speaker.speak("Going dark. I'll be here when you need me.")

    def run_once(self) -> None:
        self.listener.wait_for_wake_word()
        self.speaker.speak("Yeah?")

        user_text = self.listener.listen_and_transcribe()
        if not user_text:
            self.speaker.speak("Didn't catch that.")
            return

        response = self.brain.think(user_text)
        self.speaker.speak(response)

    def run_forever(self) -> int:
        self.initialize_services()
        self.boot()

        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                self.shutdown()
                return 0
            except Exception:
                logger.exception("Unhandled runtime loop error")
                time.sleep(1)


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="[%(levelname)s] %(name)s: %(message)s")
    runtime = AssistantRuntime()
    return runtime.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
