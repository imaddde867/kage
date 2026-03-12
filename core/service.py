from __future__ import annotations

import os
import plistlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DEFAULT_VOICE_SERVICE_LABEL = "com.kage.voice"

_ALREADY_LOADED_MARKERS = (
    "already loaded",
    "already bootstrapped",
    "in progress",
)
_NOT_LOADED_MARKERS = (
    "could not find service",
    "no such process",
    "service cannot be found",
    "not loaded",
)
_PID_RE = re.compile(r"\bpid\s*=\s*(\d+)\b")


@dataclass(frozen=True)
class VoiceServiceStatus:
    label: str
    plist_path: Path
    installed: bool
    loaded: bool
    pid: int | None
    detail: str


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _launch_agents_dir(*, home: Path | None = None) -> Path:
    base = home.expanduser() if home is not None else Path.home()
    return base / "Library" / "LaunchAgents"


def _logs_dir(*, home: Path | None = None) -> Path:
    base = home.expanduser() if home is not None else Path.home()
    return base / "Library" / "Logs" / "kage"


def voice_service_plist_path(
    *,
    label: str = DEFAULT_VOICE_SERVICE_LABEL,
    home: Path | None = None,
) -> Path:
    return _launch_agents_dir(home=home) / f"{label}.plist"


def _launchctl_domain(*, uid: int | None = None) -> str:
    user_id = os.getuid() if uid is None else int(uid)
    return f"gui/{user_id}"


def _launchctl_target(*, label: str, uid: int | None = None) -> str:
    return f"{_launchctl_domain(uid=uid)}/{label}"


def _launchctl_message(proc: Any) -> str:
    stdout = str(getattr(proc, "stdout", "") or "").strip()
    stderr = str(getattr(proc, "stderr", "") or "").strip()
    if stderr:
        return stderr
    if stdout:
        return stdout
    return f"launchctl exited with code {int(getattr(proc, 'returncode', 1))}"


def _is_already_loaded(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _ALREADY_LOADED_MARKERS)


def _is_not_loaded(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _NOT_LOADED_MARKERS)


def _parse_pid(text: str) -> int | None:
    match = _PID_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _run_launchctl(
    args: list[str],
    *,
    runner: Callable[..., Any] | None = None,
) -> Any:
    run = runner or subprocess.run
    return run(["launchctl", *args], capture_output=True, text=True)


def _voice_plist_payload(
    *,
    label: str,
    project_root: Path,
    python_executable: str,
    stdout_path: Path,
    stderr_path: Path,
) -> bytes:
    payload = {
        "Label": label,
        "ProgramArguments": [python_executable, str(project_root / "main.py"), "voice"],
        "WorkingDirectory": str(project_root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def install_voice_service(
    *,
    label: str = DEFAULT_VOICE_SERVICE_LABEL,
    project_root: Path | None = None,
    home: Path | None = None,
    python_executable: str | None = None,
    uid: int | None = None,
    runner: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "launchd services are only supported on macOS."

    root = (project_root or _project_root()).resolve()
    py_exec = python_executable or sys.executable
    plist_path = voice_service_plist_path(label=label, home=home)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = _logs_dir(home=home)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "voice.out.log"
    stderr_path = log_dir / "voice.err.log"

    payload = _voice_plist_payload(
        label=label,
        project_root=root,
        python_executable=py_exec,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    plist_path.write_bytes(payload)

    domain = _launchctl_domain(uid=uid)
    _run_launchctl(["bootout", domain, str(plist_path)], runner=runner)
    boot = _run_launchctl(["bootstrap", domain, str(plist_path)], runner=runner)
    boot_msg = _launchctl_message(boot)
    if int(getattr(boot, "returncode", 1)) != 0 and not _is_already_loaded(boot_msg):
        return False, f"launchctl bootstrap failed: {boot_msg}"

    target = _launchctl_target(label=label, uid=uid)
    kick = _run_launchctl(["kickstart", "-k", target], runner=runner)
    kick_msg = _launchctl_message(kick)
    if int(getattr(kick, "returncode", 1)) != 0:
        return False, f"launchctl kickstart failed: {kick_msg}"

    return True, f"Installed and started '{label}' ({plist_path})."


def start_voice_service(
    *,
    label: str = DEFAULT_VOICE_SERVICE_LABEL,
    home: Path | None = None,
    uid: int | None = None,
    runner: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "launchd services are only supported on macOS."

    plist_path = voice_service_plist_path(label=label, home=home)
    if not plist_path.is_file():
        return False, f"Service plist is not installed: {plist_path}"

    domain = _launchctl_domain(uid=uid)
    boot = _run_launchctl(["bootstrap", domain, str(plist_path)], runner=runner)
    boot_msg = _launchctl_message(boot)
    if int(getattr(boot, "returncode", 1)) != 0 and not _is_already_loaded(boot_msg):
        return False, f"launchctl bootstrap failed: {boot_msg}"

    target = _launchctl_target(label=label, uid=uid)
    kick = _run_launchctl(["kickstart", "-k", target], runner=runner)
    kick_msg = _launchctl_message(kick)
    if int(getattr(kick, "returncode", 1)) != 0:
        return False, f"launchctl kickstart failed: {kick_msg}"

    return True, f"Started '{label}'."


def stop_voice_service(
    *,
    label: str = DEFAULT_VOICE_SERVICE_LABEL,
    home: Path | None = None,
    uid: int | None = None,
    runner: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "launchd services are only supported on macOS."

    plist_path = voice_service_plist_path(label=label, home=home)
    domain = _launchctl_domain(uid=uid)
    args = ["bootout", domain, str(plist_path)] if plist_path.exists() else ["bootout", _launchctl_target(label=label, uid=uid)]
    stop = _run_launchctl(args, runner=runner)
    stop_msg = _launchctl_message(stop)
    if int(getattr(stop, "returncode", 1)) != 0 and not _is_not_loaded(stop_msg):
        return False, f"launchctl bootout failed: {stop_msg}"
    return True, f"Stopped '{label}'."


def uninstall_voice_service(
    *,
    label: str = DEFAULT_VOICE_SERVICE_LABEL,
    home: Path | None = None,
    uid: int | None = None,
    runner: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "launchd services are only supported on macOS."

    ok, message = stop_voice_service(label=label, home=home, uid=uid, runner=runner)
    if not ok:
        return False, message

    plist_path = voice_service_plist_path(label=label, home=home)
    if plist_path.exists():
        plist_path.unlink()
    return True, f"Uninstalled '{label}'."


def voice_service_status(
    *,
    label: str = DEFAULT_VOICE_SERVICE_LABEL,
    home: Path | None = None,
    uid: int | None = None,
    runner: Callable[..., Any] | None = None,
) -> VoiceServiceStatus:
    plist_path = voice_service_plist_path(label=label, home=home)
    installed = plist_path.is_file()
    if sys.platform != "darwin":
        return VoiceServiceStatus(
            label=label,
            plist_path=plist_path,
            installed=installed,
            loaded=False,
            pid=None,
            detail="launchd services are only supported on macOS.",
        )

    target = _launchctl_target(label=label, uid=uid)
    status = _run_launchctl(["print", target], runner=runner)
    message = _launchctl_message(status)
    loaded = int(getattr(status, "returncode", 1)) == 0
    pid = _parse_pid(f"{getattr(status, 'stdout', '')}\n{getattr(status, 'stderr', '')}") if loaded else None
    if not loaded and _is_not_loaded(message):
        message = "not loaded"
    return VoiceServiceStatus(
        label=label,
        plist_path=plist_path,
        installed=installed,
        loaded=loaded,
        pid=pid,
        detail=message,
    )
