from __future__ import annotations

import subprocess


class ClipboardError(RuntimeError):
    pass


def copy_to_clipboard(text: str) -> None:
    try:
        subprocess.run(
            ["pbcopy"],
            input=text,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise ClipboardError("pbcopy is not available on this machine.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise ClipboardError(f"Clipboard copy failed: {detail}") from exc
