from __future__ import annotations

import subprocess
import time


def escape_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def run_osascript(script: str, *, timeout: int = 10) -> tuple[str, bool]:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return result.stderr.strip() or "osascript returned non-zero exit code.", True
        return result.stdout.strip(), False
    except FileNotFoundError:
        return "osascript is not available (non-macOS system).", True
    except subprocess.TimeoutExpired:
        return "osascript timed out.", True


def run_osascript_with_retry(
    script: str,
    *,
    timeout: int,
    retry_count: int,
    retry_delay: float,
) -> tuple[str, bool, int]:
    attempts = max(0, int(retry_count)) + 1
    delay = max(0.0, float(retry_delay))
    last_output = ""
    for index in range(attempts):
        output, is_error = run_osascript(script, timeout=timeout)
        last_output = output
        if not is_error:
            return output, False, index + 1
        if "timed out" not in output.lower():
            return output, True, index + 1
        if index < attempts - 1 and delay > 0:
            time.sleep(delay)
    return last_output, True, attempts

