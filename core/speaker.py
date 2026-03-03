from __future__ import annotations

import re
import subprocess

import config


def _sanitize(text: str) -> str:
    text = re.sub(r"[*#`_]", "", text)
    text = text.replace("...", ".").replace("—", ", ").replace("–", ", ")
    return re.sub(r"\s+", " ", text).strip()


def speak(text: str) -> None:
    clean = _sanitize(text)
    if not clean:
        return
    voice = config.get().say_voice
    subprocess.run(["say", "-v", voice, clean], check=False)
