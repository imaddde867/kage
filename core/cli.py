from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


def _memory_db_path(*, settings: config.Settings) -> Path:
    return Path(settings.memory_dir).expanduser() / "kage_memory.db"


def _looks_like_vlm_model(model_name: str) -> bool:
    lowered = model_name.strip().lower()
    return any(token in lowered for token in ("qwen3.5", "vlm", "vision"))


def collect_doctor_checks(*, settings: config.Settings) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []

    backend = settings.llm_backend.strip().lower()
    if backend in {"mlx_vlm", "mlx", "openai_compat"}:
        checks.append(DoctorCheck("backend", "ok", f"Using backend '{backend}'"))
    else:
        checks.append(
            DoctorCheck(
                "backend",
                "error",
                f"Unsupported backend '{settings.llm_backend}'. Expected mlx_vlm, mlx, or openai_compat.",
            )
        )

    if backend == "mlx" and _looks_like_vlm_model(settings.mlx_model):
        checks.append(
            DoctorCheck(
                "model_compat",
                "warning",
                "Model name looks like a VLM checkpoint while LLM_BACKEND=mlx expects a text-only model.",
            )
        )
    else:
        checks.append(
            DoctorCheck("model_compat", "ok", f"Configured model: {settings.mlx_model}")
        )

    memory_dir = Path(settings.memory_dir).expanduser()
    memory_parent = memory_dir if memory_dir.exists() else memory_dir.parent
    if (
        memory_dir.exists()
        and memory_dir.is_dir()
        and os.access(memory_dir, os.R_OK | os.W_OK)
    ):
        checks.append(
            DoctorCheck("memory_dir", "ok", f"Readable and writable: {memory_dir}")
        )
    elif memory_dir.exists() and not memory_dir.is_dir():
        checks.append(
            DoctorCheck(
                "memory_dir",
                "error",
                f"Path exists but is not a directory: {memory_dir}",
            )
        )
    elif memory_parent.exists() and os.access(memory_parent, os.W_OK):
        checks.append(
            DoctorCheck(
                "memory_dir",
                "warning",
                f"Directory does not exist yet but can be created on first write: {memory_dir}",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                "memory_dir",
                "error",
                f"Directory is not writable and cannot be created: {memory_dir}",
            )
        )

    dep_checks = {
        "textual": _module_available("textual"),
        "sounddevice": _module_available("sounddevice"),
        "openwakeword": _module_available("openwakeword"),
        "SpeechRecognition": _module_available("speech_recognition"),
        "faster_whisper": _module_available("faster_whisper"),
        "nano_parakeet": _module_available("nano_parakeet"),
        "ddgs": _module_available("ddgs") or _module_available("duckduckgo_search"),
        "scrapling": _module_available("scrapling"),
        "httpx": _module_available("httpx"),
        "trafilatura": _module_available("trafilatura"),
        "pypdf": _module_available("pypdf"),
        "python_docx": _module_available("docx"),
        "openpyxl": _module_available("openpyxl"),
    }
    for name, ok in dep_checks.items():
        status = "ok" if ok else "warning"
        detail = "available" if ok else "missing"
        checks.append(DoctorCheck(f"dependency:{name}", status, detail))

    if dep_checks["sounddevice"] and dep_checks["openwakeword"]:
        checks.append(
            DoctorCheck(
                "voice_stack", "ok", "sounddevice and openwakeword are available"
            )
        )
    else:
        missing = [
            name for name in ("sounddevice", "openwakeword") if not dep_checks[name]
        ]
        checks.append(
            DoctorCheck(
                "voice_stack",
                "warning",
                f"Voice mode is incomplete; missing {', '.join(missing)}",
            )
        )

    stt_backend = settings.stt_backend.strip().lower()
    if stt_backend == "apple":
        if dep_checks["SpeechRecognition"]:
            checks.append(
                DoctorCheck("stt_backend", "ok", "Apple SpeechRecognition is available")
            )
        elif dep_checks["faster_whisper"]:
            checks.append(
                DoctorCheck(
                    "stt_backend",
                    "warning",
                    "Apple SpeechRecognition is missing; Whisper fallback is available.",
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    "stt_backend",
                    "error",
                    "Apple SpeechRecognition is missing and no Whisper fallback is installed.",
                )
            )
    elif stt_backend == "whisper":
        status = "ok" if dep_checks["faster_whisper"] else "error"
        detail = (
            "Whisper backend is available"
            if dep_checks["faster_whisper"]
            else "Whisper backend selected but faster_whisper is not installed"
        )
        checks.append(DoctorCheck("stt_backend", status, detail))
    elif stt_backend == "parakeet":
        status = "ok" if dep_checks["nano_parakeet"] else "error"
        detail = (
            "Parakeet V3 is available"
            if dep_checks["nano_parakeet"]
            else "Parakeet selected but nano-parakeet is not installed"
        )
        checks.append(DoctorCheck("stt_backend", status, detail))
    else:
        checks.append(
            DoctorCheck(
                "stt_backend",
                "warning",
                f"Unknown STT backend '{settings.stt_backend}'",
            )
        )

    if settings.agent_enabled:
        missing = [name for name in ("ddgs",) if not dep_checks[name]]
        if missing:
            checks.append(
                DoctorCheck(
                    "agent_mode",
                    "warning",
                    f"Agent mode is enabled but some core web tooling is missing: {', '.join(missing)}",
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    "agent_mode",
                    "ok",
                    "Agent mode has its core web dependency available",
                )
            )

    raw_safe_roots = tuple(
        getattr(settings, "local_artifact_safe_roots", ("./", "~")) or ("./", "~")
    )
    safe_roots = [Path(root).expanduser() for root in raw_safe_roots]
    if not safe_roots:
        checks.append(
            DoctorCheck("local_artifact_roots", "error", "No safe roots configured")
        )
    else:
        readable = [
            str(root)
            for root in safe_roots
            if root.exists() and os.access(root, os.R_OK)
        ]
        if readable:
            checks.append(
                DoctorCheck(
                    "local_artifact_roots",
                    "ok",
                    f"Readable roots: {', '.join(readable[:3])}",
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    "local_artifact_roots",
                    "warning",
                    "Configured safe roots do not currently exist or are unreadable",
                )
            )

    extractor_missing = [
        name for name in ("pypdf", "python_docx", "openpyxl") if not dep_checks[name]
    ]
    if extractor_missing:
        checks.append(
            DoctorCheck(
                "local_artifact_extractors",
                "warning",
                f"Missing optional extractor deps: {', '.join(extractor_missing)}",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                "local_artifact_extractors",
                "ok",
                "PDF, DOCX, and sheet extractors are available",
            )
        )

    on_macos = sys.platform == "darwin"
    osascript_available = shutil.which("osascript") is not None
    if on_macos and osascript_available:
        checks.append(
            DoctorCheck(
                "apple_automation",
                "manual",
                "Calendar, Reminders, Notifications, and Accessibility permissions must be granted when prompted.",
            )
        )
    elif on_macos:
        checks.append(
            DoctorCheck(
                "apple_automation",
                "warning",
                "osascript is unavailable on this macOS install",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                "apple_automation",
                "warning",
                "Apple automation features are unavailable because this is not macOS.",
            )
        )

    return checks


def collect_agent_doctor_checks(*, settings: config.Settings) -> list[DoctorCheck]:
    from core.platform.policy_engine import valid_policy_mode, valid_risk_tier
    from core.platform.storage import ApprovalStore

    checks: list[DoctorCheck] = []
    mode = str(getattr(settings, "agent_policy_mode", "strict")).strip().lower()
    if valid_policy_mode(mode):
        checks.append(
            DoctorCheck("agent_policy_mode", "ok", f"Configured mode '{mode}'")
        )
    else:
        checks.append(
            DoctorCheck(
                "agent_policy_mode",
                "error",
                f"Unsupported mode '{mode}'. Expected strict, hybrid, owner_fast, or interactive.",
            )
        )

    raw_tiers = getattr(settings, "agent_approval_required_tiers", ())
    if not isinstance(raw_tiers, tuple):
        raw_tiers = tuple(raw_tiers) if isinstance(raw_tiers, list) else ()
    normalized_tiers = tuple(
        sorted({str(t).strip().lower() for t in raw_tiers if str(t).strip()})
    )
    invalid_tiers = [tier for tier in normalized_tiers if not valid_risk_tier(tier)]
    if invalid_tiers:
        checks.append(
            DoctorCheck(
                "agent_approval_required_tiers",
                "error",
                f"Unsupported tier(s): {', '.join(invalid_tiers)}",
            )
        )
    else:
        detail = ", ".join(normalized_tiers) if normalized_tiers else "(none)"
        checks.append(DoctorCheck("agent_approval_required_tiers", "ok", detail))

    try:
        store = ApprovalStore(_memory_db_path(settings=settings))
        count = store.count_entries()
        checks.append(
            DoctorCheck("agent_approvals_store", "ok", f"{count} persisted approval(s)")
        )
    except Exception as exc:
        checks.append(
            DoctorCheck(
                "agent_approvals_store",
                "error",
                f"Unable to access approvals store: {exc}",
            )
        )
    return checks


def format_doctor_report(
    *, settings: config.Settings, include_agent: bool = False
) -> str:
    lines = [
        "Kage doctor",
        f"- backend: {settings.llm_backend}",
        f"- model: {settings.mlx_model}",
        f"- agent_enabled: {settings.agent_enabled}",
        f"- second_brain_enabled: {settings.second_brain_enabled}",
        f"- text_mode_tts_enabled: {settings.text_mode_tts_enabled}",
        f"- memory_dir: {settings.memory_dir}",
        "- checks:",
    ]
    checks = collect_doctor_checks(settings=settings)
    if include_agent:
        checks.extend(collect_agent_doctor_checks(settings=settings))
    for check in checks:
        lines.append(f"  - {check.name}: {check.status} ({check.detail})")
    return "\n".join(lines)


def run_doctor(*, settings: config.Settings, include_agent: bool = False) -> None:
    print(format_doctor_report(settings=settings, include_agent=include_agent))


def run_approvals_list(*, settings: config.Settings) -> int:
    from core.platform.storage import ApprovalStore

    store = ApprovalStore(_memory_db_path(settings=settings))
    entries = store.list_entries(limit=500)
    print("Kage approvals")
    if not entries:
        print("- none")
        return 0
    for entry in entries:
        print(
            f"- {entry.approval_key} (by={entry.granted_by}, updated_at={entry.updated_at}, note={entry.note or '-'})"
        )
    return 0


def run_approvals_grant(
    *,
    settings: config.Settings,
    scope_kind: str,
    scope_name: str,
    note: str = "manual_cli",
) -> int:
    from core.platform.storage import ApprovalStore

    store = ApprovalStore(_memory_db_path(settings=settings))
    key = store.grant(
        scope_kind=scope_kind,
        scope_name=scope_name,
        note=note,
        granted_by="cli",
    )
    print(f"Granted approval: {key}")
    return 0


def run_approvals_revoke(
    *, settings: config.Settings, scope_kind: str, scope_name: str
) -> int:
    from core.platform.storage import ApprovalStore

    store = ApprovalStore(_memory_db_path(settings=settings))
    removed = store.revoke(scope_kind=scope_kind, scope_name=scope_name)
    if removed:
        print(f"Revoked approval: {scope_kind}:{scope_name}")
    else:
        print(f"No approval found for: {scope_kind}:{scope_name}")
    return 0


def run_memory_stats(*, settings: config.Settings) -> int:
    from core.platform.storage import ConversationStore, EvidenceStore, TraceStore

    db_path = _memory_db_path(settings=settings)
    conversations = ConversationStore(db_path).count()
    evidence = EvidenceStore(db_path).count()
    traces = TraceStore(db_path).count()
    print("Kage memory stats")
    print(f"- conversations: {conversations}")
    print(f"- evidence:      {evidence}")
    print(f"- traces:        {traces}")
    print(f"- db:            {db_path}")
    return 0


def run_memory_prune(
    *,
    settings: config.Settings,
    keep_conversations: int,
    keep_evidence: int,
    keep_traces: int,
) -> int:
    from core.platform.storage import ConversationStore, EvidenceStore, TraceStore

    db_path = _memory_db_path(settings=settings)
    deleted_c = ConversationStore(db_path).prune(keep_conversations)
    deleted_e = EvidenceStore(db_path).prune(keep_evidence)
    deleted_t = TraceStore(db_path).prune(keep_traces)
    print(f"Pruned {deleted_c} conversation(s), {deleted_e} evidence record(s), {deleted_t} trace(s)")
    return 0


def run_backup_create(*, settings: config.Settings, output: str | None = None) -> int:
    from core.backup import create_backup

    path = create_backup(
        settings=settings,
        output_path=Path(output).expanduser() if output else None,
    )
    print(f"Created backup: {path}")
    return 0


def run_backup_verify(*, archive: str) -> int:
    from core.backup import verify_backup

    ok, lines = verify_backup(Path(archive))
    for line in lines:
        print(line)
    return 0 if ok else 1


def run_service_install() -> int:
    from core.service import install_voice_service

    ok, message = install_voice_service()
    print(message)
    return 0 if ok else 1


def run_service_uninstall() -> int:
    from core.service import uninstall_voice_service

    ok, message = uninstall_voice_service()
    print(message)
    return 0 if ok else 1


def run_service_start() -> int:
    from core.service import start_voice_service

    ok, message = start_voice_service()
    print(message)
    return 0 if ok else 1


def run_service_stop() -> int:
    from core.service import stop_voice_service

    ok, message = stop_voice_service()
    print(message)
    return 0 if ok else 1


def run_service_status() -> int:
    from core.service import voice_service_status

    status = voice_service_status()
    pid_text = str(status.pid) if status.pid is not None else "n/a"
    print("Kage service status")
    print(f"- label: {status.label}")
    print(f"- plist: {status.plist_path}")
    print(f"- installed: {status.installed}")
    print(f"- loaded: {status.loaded}")
    print(f"- pid: {pid_text}")
    print(f"- detail: {status.detail}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kage — local personal AI")
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat", help="Launch the text chat UI")
    chat.add_argument(
        "--plain",
        action="store_true",
        help="Use the plain terminal fallback instead of Textual",
    )
    chat.add_argument(
        "--timing", action="store_true", help="Show throughput metrics in the chat UI"
    )

    voice = subparsers.add_parser("voice", help="Launch voice mode")
    voice.add_argument(
        "--timing",
        action="store_true",
        help="Print latency breakdown after each response",
    )

    bench = subparsers.add_parser(
        "bench", help="Run inference benchmark without TTS and exit"
    )
    bench.add_argument("--timing", action="store_true", help=argparse.SUPPRESS)
    doctor = subparsers.add_parser(
        "doctor", help="Print environment and dependency diagnostics"
    )
    doctor.add_argument(
        "--agent",
        action="store_true",
        help="Include agent policy and approval diagnostics",
    )

    approvals = subparsers.add_parser(
        "approvals", help="Manage policy approvals for autonomous actions"
    )
    approvals_subparsers = approvals.add_subparsers(
        dest="approvals_command", required=True
    )
    approvals_subparsers.add_parser("list", help="List granted approvals")
    approvals_grant = approvals_subparsers.add_parser(
        "grant", help="Grant approval for a tool or tier"
    )
    approvals_grant.add_argument(
        "scope_kind", choices=["tool", "tier"], help="Approval scope kind"
    )
    approvals_grant.add_argument("scope_name", help="Tool name or risk tier")
    approvals_grant.add_argument(
        "--note", default="manual_cli", help="Optional audit note"
    )
    approvals_revoke = approvals_subparsers.add_parser(
        "revoke", help="Revoke approval for a tool or tier"
    )
    approvals_revoke.add_argument(
        "scope_kind", choices=["tool", "tier"], help="Approval scope kind"
    )
    approvals_revoke.add_argument("scope_name", help="Tool name or risk tier")

    memory = subparsers.add_parser("memory", help="Inspect and prune local memory stores")
    memory_subparsers = memory.add_subparsers(dest="memory_command", required=True)
    memory_subparsers.add_parser("stats", help="Print row counts for each memory store")
    memory_prune = memory_subparsers.add_parser(
        "prune", help="Delete oldest records, keeping only the N most recent"
    )
    memory_prune.add_argument(
        "--keep-conversations",
        type=int,
        default=5000,
        help="Conversations to keep (default: 5000)",
    )
    memory_prune.add_argument(
        "--keep-evidence",
        type=int,
        default=5000,
        help="Evidence records to keep (default: 5000)",
    )
    memory_prune.add_argument(
        "--keep-traces",
        type=int,
        default=5000,
        help="Trace records to keep (default: 5000)",
    )

    backup = subparsers.add_parser(
        "backup", help="Create or verify local state backups"
    )
    backup_subparsers = backup.add_subparsers(dest="backup_command", required=True)
    backup_create = backup_subparsers.add_parser(
        "create", help="Create a compressed local backup archive"
    )
    backup_create.add_argument(
        "--output", help="Optional output archive path (.tar.gz)"
    )
    backup_verify = backup_subparsers.add_parser(
        "verify", help="Verify a local backup archive"
    )
    backup_verify.add_argument("archive", help="Path to the backup archive to verify")

    service = subparsers.add_parser(
        "service", help="Manage launchd voice daemon on macOS"
    )
    service_subparsers = service.add_subparsers(dest="service_command", required=True)
    service_subparsers.add_parser(
        "install", help="Install and start the launchd voice service"
    )
    service_subparsers.add_parser(
        "uninstall", help="Stop and remove the launchd voice service"
    )
    service_subparsers.add_parser(
        "start", help="Start the installed launchd voice service"
    )
    service_subparsers.add_parser("stop", help="Stop the launchd voice service")
    service_subparsers.add_parser("status", help="Show launchd voice service status")
    return parser


def normalize_legacy_argv(argv: Sequence[str]) -> list[str]:
    args = list(argv)
    if not args:
        return ["voice"]
    if "--bench" in args:
        return ["bench", *[arg for arg in args if arg != "--bench"]]
    if "--text" in args:
        return ["chat", *[arg for arg in args if arg != "--text"]]
    if args[0].startswith("-"):
        # Preserve global argparse behavior for flags like --help.
        return args
    if args and args[0] in {
        "chat",
        "voice",
        "bench",
        "doctor",
        "approvals",
        "memory",
        "backup",
        "service",
    }:
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
            print(
                "Textual UI is unavailable. Install dependencies with: pip install -r requirements.txt"
            )
            return 1
        return 0

    if args.command == "bench":
        launch_bench(settings=settings)
        return 0

    if args.command == "doctor":
        run_doctor(settings=settings, include_agent=bool(getattr(args, "agent", False)))
        return 0

    if args.command == "approvals":
        if args.approvals_command == "list":
            return run_approvals_list(settings=settings)
        if args.approvals_command == "grant":
            return run_approvals_grant(
                settings=settings,
                scope_kind=args.scope_kind,
                scope_name=args.scope_name,
                note=args.note,
            )
        if args.approvals_command == "revoke":
            return run_approvals_revoke(
                settings=settings,
                scope_kind=args.scope_kind,
                scope_name=args.scope_name,
            )
        return 0

    if args.command == "memory":
        if args.memory_command == "stats":
            return run_memory_stats(settings=settings)
        if args.memory_command == "prune":
            return run_memory_prune(
                settings=settings,
                keep_conversations=args.keep_conversations,
                keep_evidence=args.keep_evidence,
                keep_traces=args.keep_traces,
            )
        return 0

    if args.command == "backup":
        if args.backup_command == "create":
            return run_backup_create(
                settings=settings, output=getattr(args, "output", None)
            )
        if args.backup_command == "verify":
            return run_backup_verify(archive=args.archive)
        parser.error("backup requires a subcommand")

    if args.command == "service":
        if args.service_command == "install":
            return run_service_install()
        if args.service_command == "uninstall":
            return run_service_uninstall()
        if args.service_command == "start":
            return run_service_start()
        if args.service_command == "stop":
            return run_service_stop()
        if args.service_command == "status":
            return run_service_status()
        parser.error("service requires a subcommand")

    launch_voice(settings=settings, timing=getattr(args, "timing", False))
    return 0
