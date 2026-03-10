"""Shared AppleScript utilities used by all macOS connectors.

Centralises two concerns:
  - String escaping before AppleScript interpolation (injection prevention)
  - subprocess wrapper for osascript with timeout and retry support

Connectors that need osascript should import from here rather than
calling subprocess directly, so error handling stays consistent.
"""
from __future__ import annotations

import subprocess
import time


def escape_applescript(text: str) -> str:
    """Escape a Python string for safe embedding in an AppleScript string literal.

    AppleScript string literals are delimited by double quotes. The only
    characters that need escaping inside them are:
        \\  →  \\\\   (backslash must be doubled first)
        "   →  \\"    (double quote must be escaped)

    Args:
        text: Raw Python string to embed in AppleScript.

    Returns:
        Escaped string safe to place between AppleScript double quotes.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def run_osascript(script: str, *, timeout: int = 10) -> tuple[str, bool]:
    """Execute an AppleScript string and return (output, is_error).

    Args:
        script:  Complete AppleScript text to run.
        timeout: Seconds before the subprocess is killed.

    Returns:
        ``(stdout.strip(), False)`` on success.
        ``(error_message, True)`` on non-zero exit, timeout, or missing osascript.
    """
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
    """Execute osascript and retry on timeout failures.

    Only retries when the failure message contains "timed out". Non-timeout
    errors (permissions, syntax) are returned immediately without retrying.

    Args:
        script:      AppleScript to execute.
        timeout:     Per-attempt timeout in seconds.
        retry_count: Number of *extra* attempts after the first (0 = no retry).
        retry_delay: Seconds to wait between attempts.

    Returns:
        ``(output, is_error, attempts_used)`` where ``attempts_used`` is the
        number of times the script was actually executed.
    """
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
