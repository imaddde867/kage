from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def run_osascript(script: str, timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.debug("AppleScript timed out after %ss", timeout)
        return ""
    except FileNotFoundError:
        logger.debug("osascript not available on this system")
        return ""
    except Exception:
        logger.exception("Unexpected AppleScript execution failure")
        return ""

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if stderr:
            logger.debug("AppleScript failed (%s): %s", result.returncode, stderr)
        return ""

    return (result.stdout or "").strip()


def non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def format_bulleted_section(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    formatted = "\n".join(f"  • {line}" for line in lines)
    return f"{title}\n{formatted}"
