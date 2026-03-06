"""
Kage (影) — Local Personal AI for macOS
Voice: wake word → STT → LLM → Kokoro TTS
Text: input() → LLM → print (optional Kokoro TTS)
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import threading
import time

import numpy as np
try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional for text/bench-only mode
    sd = None  # type: ignore[assignment]

import config
from core.audio_coordinator import AudioCoordinator, AudioState
from core.brain import BrainService
from core.listener import ListenerService
from core.speaker import speak, stop_speaking

logging.basicConfig(level=logging.ERROR, format="[%(levelname)s] %(name)s: %(message)s")


def _print_timing_stats(
    *,
    label: str,
    t0: float,
    t_first: float | None,
    t_end: float,
    tts_total: float,
    stats: dict[str, object],
) -> None:
    ttfs = (t_first - t0) if t_first else 0.0
    tok = stats.get("tokens", "?")
    tps = stats.get("tok_per_sec", 0.0)
    backend = stats.get("backend", "?")
    tps_str = f"{float(tps):.1f} tok/s" if isinstance(tps, (int, float)) and tps else "?"
    print(
        f"  ⏱  [{backend}] {label}: {ttfs:.2f}s | "
        f"gen: {tok} tok @ {tps_str} | "
        f"tts: {tts_total:.2f}s | "
        f"total: {t_end - t0:.2f}s",
        flush=True,
    )


def _monitor_barge_in(
    listener: ListenerService,
    coordinator: AudioCoordinator,
    stop_event: threading.Event,
) -> None:
    if sd is None:
        logging.warning("sounddevice unavailable; disabling barge-in monitor")
        return

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


def _say(text: str, *, coordinator: AudioCoordinator | None = None) -> None:
    speak(text)
    if coordinator is not None:
        coordinator.end_speaking()


def _listen_once(listener: ListenerService, coordinator: AudioCoordinator) -> str:
    coordinator.transition(AudioState.LISTENING)
    coordinator.wait_for_listen_window()
    audio = listener.record_until_silence()
    return listener.transcribe(audio).strip()


def respond(
    brain: BrainService,
    user_text: str,
    timing: bool = False,
    *,
    listener: ListenerService | None = None,
    coordinator: AudioCoordinator | None = None,
    speak_enabled: bool = True,
) -> bool:
    print("\n[Kage]: ", end="", flush=True)
    t0 = time.perf_counter()
    t_first_sentence: float | None = None
    tts_total = 0.0
    interrupted = False
    monitor_stop: threading.Event | None = None
    monitor_thread: threading.Thread | None = None
    cancel_token: threading.Event | None = None

    if not speak_enabled:
        stream = brain.think_text_stream(user_text)
        try:
            for chunk in stream:
                if t_first_sentence is None:
                    t_first_sentence = time.perf_counter()
                print(chunk, end="", flush=True)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        t_end = time.perf_counter()
        print("\n")
        if timing:
            _print_timing_stats(
                label="first chunk",
                t0=t0,
                t_first=t_first_sentence,
                t_end=t_end,
                tts_total=0.0,
                stats=brain.last_stats,
            )
        return False

    if listener is not None and coordinator is not None and coordinator.allow_barge_in:
        coordinator.begin_speaking()
        cancel_token = coordinator.cancel_token
        monitor_stop = threading.Event()
        monitor_thread = threading.Thread(
            target=_monitor_barge_in,
            args=(listener, coordinator, monitor_stop),
            daemon=True,
        )
        monitor_thread.start()

    stream = brain.think_stream(user_text)
    try:
        for sentence in stream:
            if t_first_sentence is None:
                t_first_sentence = time.perf_counter()
            print(sentence, end=" ", flush=True)

            if cancel_token is not None:
                t_speak = time.perf_counter()
                result = speak(sentence, cancel_token=cancel_token)
                tts_total += time.perf_counter() - t_speak
                if result.interrupted:
                    interrupted = True
                    break
            else:
                t_speak = time.perf_counter()
                speak(sentence)
                tts_total += time.perf_counter() - t_speak
    finally:
        if monitor_stop is not None:
            monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=0.3)
        if cancel_token is not None and coordinator is not None:
            coordinator.end_speaking(interrupted=interrupted)
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    t_end = time.perf_counter()
    print("\n")

    if timing:
        _print_timing_stats(
            label="first sentence",
            t0=t0,
            t_first=t_first_sentence,
            t_end=t_end,
            tts_total=tts_total,
            stats=brain.last_stats,
        )

    return interrupted


def run_voice(settings: config.Settings, timing: bool = False) -> None:
    listener = ListenerService(settings=settings)
    listener.load_models()
    brain = BrainService(settings=settings)
    coordinator = AudioCoordinator(settings=settings)

    # Start the proactive heartbeat daemon when enabled.  The daemon is a background
    # thread that wakes periodically (heartbeat_interval_seconds) to check EntityStore
    # for due/overdue tasks and speaks them aloud when conditions allow (idle audio,
    # outside DND hours, debounce cleared).  It is a daemon thread so it terminates
    # automatically when the main process exits — no explicit cleanup needed.
    if getattr(settings, "heartbeat_enabled", True):
        from core.agent.heartbeat import HeartbeatAgent
        HeartbeatAgent(brain, coordinator, settings).start()

    print(f"  Kage online. Say '{settings.wake_word.title()}' to activate.\n")

    while True:
        try:
            coordinator.transition(AudioState.IDLE)
            listener.wait_for_wake_word()
            _say("Yeah?", coordinator=coordinator)

            while True:
                user_text = _listen_once(listener, coordinator)
                if not user_text:
                    coordinator.transition(AudioState.IDLE)
                    _say("Didn't catch that.", coordinator=coordinator)
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
    tts_mode = "on" if settings.text_mode_tts_enabled else "off"
    print(f"  Kage online. Type your message. 'exit' to quit. [text-mode TTS: {tts_mode}]\n")

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
            respond(
                brain,
                user_text,
                timing=timing,
                speak_enabled=settings.text_mode_tts_enabled,
            )
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
    print(f"  Model      : {settings.mlx_model}")
    if settings.llm_backend == "mlx" and settings.mlx_draft_model:
        print(f"  Draft      : {settings.mlx_draft_model}  (speculative decoding)")

    # Keep bench focused on model inference by disabling feature-path overhead.
    bench_settings = settings
    if settings.agent_enabled or settings.second_brain_enabled or settings.recent_turns > 0:
        bench_settings = replace(
            settings,
            agent_enabled=False,
            second_brain_enabled=False,
            recent_turns=0,
            temperature=0.0,
        )
        print("  Bench mode : agent/second-brain/history disabled for cleaner model timing")

    # BrainService.__init__ loads + warms up the model (prints its own status lines)
    brain = BrainService(settings=bench_settings)

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
        chunks_out: list[str] = []

        for chunk in brain.think_text_stream(prompt):
            if t_first is None:
                t_first = time.perf_counter()
            chunks_out.append(chunk)
            # no speak() — pure inference measurement

        t_end = time.perf_counter()
        stats = brain.last_stats
        ttft = (t_first - t0) if t_first else 0.0
        total_s = t_end - t0
        tps = stats.get("tok_per_sec", 0.0)
        decode_tps = stats.get("generation_tps")
        prompt_tps = stats.get("prompt_tps")
        prompt_tokens = stats.get("prompt_tokens")
        tok = stats.get("tokens", "?")
        metric_tps = (
            float(decode_tps)
            if isinstance(decode_tps, (int, float))
            else float(tps)
            if isinstance(tps, (int, float))
            else None
        )
        if metric_tps is not None:
            results.append(metric_tps)

        response_preview = "".join(chunks_out).replace("\n", " ")[:72]
        print(f"  [{i}] {prompt}")
        print(f"       → {response_preview}")
        details = f"       TTFT: {ttft:.2f}s | total: {total_s:.2f}s | {tok} tok"
        if metric_tps is not None:
            details += f" | decode @ {metric_tps:.1f} tok/s"
        print(details)
        if isinstance(prompt_tokens, int) and isinstance(prompt_tps, (int, float)):
            print(f"       prefill: {prompt_tokens} tok @ {float(prompt_tps):.1f} tok/s")
        print()

    if results:
        avg = sum(results) / len(results)
        print(f"  avg throughput : {avg:.1f} tok/s")
        model_name = bench_settings.mlx_model.split("/")[-1]
        if bench_settings.llm_backend == "mlx_vlm":
            expected = "~lower than mlx-lm on text-only tasks (VLM backend overhead)"
        elif "1.5B" in model_name or "1.5b" in model_name:
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
