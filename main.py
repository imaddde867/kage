"""
Kage (影) — Local Personal AI for macOS
Voice: wake word → STT → LLM → say
Text: input() → LLM → print + say
"""

from __future__ import annotations

import argparse
import logging
import time

import config
from core.brain import BrainService
from core.listener import ListenerService
from core.speaker import speak

logging.basicConfig(level=logging.ERROR, format="[%(levelname)s] %(name)s: %(message)s")


def respond(brain: BrainService, user_text: str) -> None:
    print("\n[Kage]: ", end="", flush=True)
    for sentence in brain.think_stream(user_text):
        print(sentence, end=" ", flush=True)
        speak(sentence)
    print("\n")


def run_voice(settings: config.Settings) -> None:
    listener = ListenerService(settings=settings)
    listener.load_models()
    brain = BrainService(settings=settings)

    print(f"  Kage online. Say '{settings.wake_word.title()}' to activate.\n")

    while True:
        try:
            listener.wait_for_wake_word()
            speak("Yeah?")
            audio = listener.record_until_silence()
            user_text = listener.transcribe(audio)
            if not user_text:
                speak("Didn't catch that.")
                continue
            print(f"[You]: {user_text}")
            respond(brain, user_text)
            time.sleep(0.6)
        except KeyboardInterrupt:
            print("\n[Kage] Going offline.")
            return
        except Exception:
            logging.exception("Voice loop error")
            time.sleep(1)


def run_text(settings: config.Settings) -> None:
    brain = BrainService(settings=settings)
    print("  Kage online. Type your message. 'exit' to quit.\n")

    while True:
        try:
            user_text = input("[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Kage] Going offline.")
            return

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", ":q"}:
            return

        try:
            respond(brain, user_text)
        except KeyboardInterrupt:
            print("\n[Kage] Going offline.")
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Kage — local personal AI")
    parser.add_argument("--text", action="store_true", help="Text mode instead of voice")
    args = parser.parse_args()

    settings = config.get()
    if args.text:
        run_text(settings)
    else:
        run_voice(settings)


if __name__ == "__main__":
    main()
