"""
Kage (影) — Local Personal AI for macOS
Voice: wake word → STT → LLM → Kokoro TTS
Text: input() → LLM → print + Kokoro TTS
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

import numpy as np
import sounddevice as sd

import config
from core.audio_coordinator import AudioCoordinator, AudioState
from core.brain import BrainService
from core.listener import ListenerService
from core.speaker import speak, stop_speaking

logging.basicConfig(level=logging.ERROR, format="[%(levelname)s] %(name)s: %(message)s")


def _monitor_barge_in(
    listener: ListenerService,
    coordinator: AudioCoordinator,
    stop_event: threading.Event,
) -> None:
    chunk = listener.settings.wake_word_chunk_size
    listener.reset_interrupt_detector()
    try:
        with sd.InputStream(
            samplerate=listener.settings.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk,
        ) as stream:
            while not stop_event.is_set():
                audio, _ = stream.read(chunk)
                flat = np.asarray(audio, dtype=np.int16).reshape(-1)
                if listener.detect_interrupt(flat):
                    if coordinator.request_interrupt():
                        stop_speaking()
                    stop_event.set()
                    return
    except Exception:
        logging.exception("Barge-in monitor error")


def respond(
    brain: BrainService,
    user_text: str,
    timing: bool = False,
    *,
    listener: ListenerService | None = None,
    coordinator: AudioCoordinator | None = None,
) -> bool:
    print("\n[Kage]: ", end="", flush=True)
    t0 = time.perf_counter()
    t_first_sentence: float | None = None
    tts_total = 0.0
    interrupted = False

    stream = brain.think_stream(user_text)
    try:
        for sentence in stream:
            if t_first_sentence is None:
                t_first_sentence = time.perf_counter()
            print(sentence, end=" ", flush=True)

            if listener is not None and coordinator is not None and coordinator.allow_barge_in:
                coordinator.begin_speaking()
                monitor_stop = threading.Event()
                monitor_thread = threading.Thread(
                    target=_monitor_barge_in,
                    args=(listener, coordinator, monitor_stop),
                    daemon=True,
                )
                monitor_thread.start()
                t_speak = time.perf_counter()
                result = speak(sentence, cancel_token=coordinator.cancel_token)
                tts_total += time.perf_counter() - t_speak
                monitor_stop.set()
                monitor_thread.join(timeout=0.3)
                coordinator.end_speaking(interrupted=result.interrupted)

                if result.interrupted:
                    interrupted = True
                    break
            else:
                t_speak = time.perf_counter()
                speak(sentence)
                tts_total += time.perf_counter() - t_speak
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    t_end = time.perf_counter()
    print("\n")

    if timing:
        ttfs = (t_first_sentence - t0) if t_first_sentence else 0.0
        stats = brain.last_stats
        tok = stats.get("tokens", "?")
        tps = stats.get("tok_per_sec", 0.0)
        backend = stats.get("backend", "?")
        tps_str = f"{tps:.1f} tok/s" if tps else "?"
        print(
            f"  ⏱  [{backend}] first sentence: {ttfs:.2f}s | "
            f"gen: {tok} tok @ {tps_str} | "
            f"tts: {tts_total:.2f}s | "
            f"total: {t_end - t0:.2f}s",
            flush=True,
        )

    return interrupted


def run_voice(settings: config.Settings, timing: bool = False) -> None:
    listener = ListenerService(settings=settings)
    listener.load_models()
    brain = BrainService(settings=settings)
    coordinator = AudioCoordinator(settings=settings)

    print(f"  Kage online. Say '{settings.wake_word.title()}' to activate.\n")

    while True:
        try:
            coordinator.transition(AudioState.IDLE)
            listener.wait_for_wake_word()
            speak("Yeah?")
            coordinator.end_speaking()

            while True:
                coordinator.transition(AudioState.LISTENING)
                coordinator.wait_for_listen_window()
                audio = listener.record_until_silence()
                user_text = listener.transcribe(audio)
                if not user_text:
                    coordinator.transition(AudioState.IDLE)
                    speak("Didn't catch that.")
                    coordinator.end_speaking()
                    break

                print(f"[You]: {user_text}")
                coordinator.transition(AudioState.THINKING)
                interrupted = respond(
                    brain,
                    user_text,
                    timing=timing,
                    listener=listener,
                    coordinator=coordinator,
                )
                if interrupted:
                    print("[Kage] Listening...")
                    continue

                coordinator.transition(AudioState.IDLE)
                time.sleep(0.6)
                break
        except KeyboardInterrupt:
            print("\n[Kage] Going offline.")
            return
        except Exception:
            logging.exception("Voice loop error")
            time.sleep(1)


def run_text(settings: config.Settings, timing: bool = False) -> None:
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
            respond(brain, user_text, timing=timing)
        except KeyboardInterrupt:
            print("\n[Kage] Going offline.")
            return


_BENCH_PROMPTS = [
    "What is 15% of 340?",
    "Name the capital of Japan.",
    "Give me one sentence about why the sky is blue.",
]


def run_bench(settings: config.Settings) -> None:
    """Pure inference benchmark — no TTS, no memory, prints accurate tok/s."""
    print(f"  Backend    : {settings.llm_backend}")
    print(f"  Model      : {settings.ollama_model if settings.llm_backend == 'ollama' else settings.mlx_model}")
    if settings.llm_backend == "mlx" and settings.mlx_draft_model:
        print(f"  Draft      : {settings.mlx_draft_model}  (speculative decoding)")

    # BrainService.__init__ loads + warms up the model (prints its own status lines)
    brain = BrainService(settings=settings)

    if settings.llm_backend in {"mlx", "mlx_vlm"}:
        import mlx.core as mx
        print(f"  MLX device : {mx.default_device()}")
        try:
            active_gb = mx.get_active_memory() / 1e9
            print(f"  MLX memory : {active_gb:.2f} GB active")
        except Exception:
            pass
    print()

    results = []
    for i, prompt in enumerate(_BENCH_PROMPTS, 1):
        t0 = time.perf_counter()
        t_first: float | None = None
        tokens_out: list[str] = []

        for sentence in brain.think_stream(prompt):
            if t_first is None:
                t_first = time.perf_counter()
            tokens_out.append(sentence)
            # no speak() — pure inference measurement

        stats = brain.last_stats
        ttft = (t_first - t0) if t_first else 0.0
        tps = stats.get("tok_per_sec", 0.0)
        tok = stats.get("tokens", "?")
        results.append(tps)

        response_preview = " ".join(tokens_out)[:72]
        print(f"  [{i}] {prompt}")
        print(f"       → {response_preview}")
        print(f"       TTFT: {ttft:.2f}s | {tok} tok @ {tps:.1f} tok/s\n")

    if results:
        avg = sum(results) / len(results)
        print(f"  avg throughput : {avg:.1f} tok/s")
        model_name = settings.mlx_model.split("/")[-1]
        if "1.5B" in model_name or "1.5b" in model_name:
            expected = "~40–60 tok/s"
        elif "3B" in model_name or "3b" in model_name:
            expected = "~20–35 tok/s"
        else:
            expected = "~8–15 tok/s"
        print(f"  expected (M4)  : {expected} for {model_name}\n")

    print("  To confirm Neural Engine is active, run in a separate terminal while bench runs:")
    print("    sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -n 8\n")
    print("  ANE Power > 0 mW during inference = Neural Engine in use.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kage — local personal AI")
    parser.add_argument("--text", action="store_true", help="Text mode instead of voice")
    parser.add_argument("--timing", action="store_true", help="Print latency breakdown after each response")
    parser.add_argument("--bench", action="store_true", help="Run inference benchmark without TTS and exit")
    args = parser.parse_args()

    settings = config.get()
    if args.bench:
        run_bench(settings)
    elif args.text:
        run_text(settings, timing=args.timing)
    else:
        run_voice(settings, timing=args.timing)


if __name__ == "__main__":
    main()
