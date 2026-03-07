from __future__ import annotations

import argparse
import importlib.util
import sys
from typing import Sequence

import config


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def launch_textual_chat(*, settings: config.Settings, timing: bool = False) -> None:
    from core.textual_chat import run_textual_chat

    run_textual_chat(settings, timing=timing)


def launch_plain_chat(*, settings: config.Settings, timing: bool = False) -> None:
    from core.chat_shell import run_plain_chat

    run_plain_chat(settings, timing=timing)


def launch_voice(*, settings: config.Settings, timing: bool = False) -> None:
    from core.app_runner import run_voice

    run_voice(settings, timing=timing)


def launch_bench(*, settings: config.Settings) -> None:
    from core.app_runner import run_bench

    run_bench(settings)


def run_doctor(*, settings: config.Settings) -> None:
    checks = [
        ("textual", _module_available("textual")),
        ("sounddevice", _module_available("sounddevice")),
        ("SpeechRecognition", _module_available("speech_recognition")),
        ("faster_whisper", _module_available("faster_whisper")),
        ("ddgs", _module_available("ddgs") or _module_available("duckduckgo_search")),
        ("scrapling", _module_available("scrapling")),
        ("httpx", _module_available("httpx")),
        ("trafilatura", _module_available("trafilatura")),
    ]

    print("Kage doctor")
    print(f"- backend: {settings.llm_backend}")
    print(f"- model: {settings.mlx_model}")
    print(f"- agent_enabled: {settings.agent_enabled}")
    print(f"- second_brain_enabled: {settings.second_brain_enabled}")
    print(f"- text_mode_tts_enabled: {settings.text_mode_tts_enabled}")
    print(f"- memory_dir: {settings.memory_dir}")
    print("- dependencies:")
    for name, ok in checks:
        print(f"  - {name}: {'ok' if ok else 'missing'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kage — local personal AI")
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat", help="Launch the text chat UI")
    chat.add_argument("--plain", action="store_true", help="Use the plain terminal fallback instead of Textual")
    chat.add_argument("--timing", action="store_true", help="Show throughput metrics in the chat UI")

    voice = subparsers.add_parser("voice", help="Launch voice mode")
    voice.add_argument("--timing", action="store_true", help="Print latency breakdown after each response")

    bench = subparsers.add_parser("bench", help="Run inference benchmark without TTS and exit")
    bench.add_argument("--timing", action="store_true", help=argparse.SUPPRESS)
    subparsers.add_parser("doctor", help="Print environment and dependency diagnostics")
    return parser


def normalize_legacy_argv(argv: Sequence[str]) -> list[str]:
    args = list(argv)
    if "--bench" in args:
        return ["bench", *[arg for arg in args if arg != "--bench"]]
    if "--text" in args:
        return ["chat", *[arg for arg in args if arg != "--text"]]
    if args and args[0] in {"chat", "voice", "bench", "doctor"}:
        return args
    return ["voice", *args]


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    normalized = normalize_legacy_argv(raw_argv)
    parser = build_parser()
    args = parser.parse_args(normalized)
    settings = config.get()

    if args.command == "chat":
        if args.plain:
            launch_plain_chat(settings=settings, timing=args.timing)
            return 0
        try:
            launch_textual_chat(settings=settings, timing=args.timing)
        except ImportError:
            print("Textual UI is unavailable. Install dependencies with: pip install -r requirements.txt")
            return 1
        return 0

    if args.command == "bench":
        launch_bench(settings=settings)
        return 0

    if args.command == "doctor":
        run_doctor(settings=settings)
        return 0

    launch_voice(settings=settings, timing=getattr(args, "timing", False))
    return 0
