from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_file(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def _default_backup_path(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return root / "output" / f"kage-backup-{stamp}.tar.gz"


def _manifest_entry(*, archive_path: str, source_path: str, data: bytes) -> dict[str, Any]:
    return {
        "archive_path": archive_path,
        "source_path": source_path,
        "size": len(data),
        "sha256": _sha256_bytes(data),
    }


def create_backup(*, settings: config.Settings, output_path: Path | None = None, root: Path | None = None) -> Path:
    repo_root = (root or Path.cwd()).resolve()
    destination = (output_path or _default_backup_path(repo_root)).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    env_path = repo_root / ".env"
    memory_dir = Path(settings.memory_dir).expanduser()
    if not memory_dir.is_absolute():
        memory_dir = (repo_root / memory_dir).resolve()

    files: list[tuple[str, Path]] = []
    if env_path.is_file():
        files.append(("config/.env", env_path))
    if memory_dir.exists():
        for path in sorted(memory_dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(memory_dir).as_posix()
                files.append((f"memory/{relative}", path))

    manifest_files: list[dict[str, Any]] = []
    with tarfile.open(destination, "w:gz") as archive:
        for archive_path, source_path in files:
            data = _read_file(source_path)
            info = tarfile.TarInfo(name=archive_path)
            info.size = len(data)
            info.mtime = source_path.stat().st_mtime
            archive.addfile(info, io.BytesIO(data))
            manifest_files.append(
                _manifest_entry(
                    archive_path=archive_path,
                    source_path=str(source_path),
                    data=data,
                )
            )

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "assistant_name": settings.assistant_name,
            "memory_dir": str(memory_dir),
            "files": manifest_files,
        }
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(payload)
        info.mtime = int(datetime.now(timezone.utc).timestamp())
        archive.addfile(info, io.BytesIO(payload))

    return destination


def verify_backup(archive_path: Path) -> tuple[bool, list[str]]:
    target = archive_path.expanduser().resolve()
    if not target.is_file():
        return False, [f"Backup file not found: {target}"]

    try:
        with tarfile.open(target, "r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()}
            manifest_member = members.get("manifest.json")
            if manifest_member is None:
                return False, [f"Backup missing manifest.json: {target}"]
            manifest_handle = archive.extractfile(manifest_member)
            if manifest_handle is None:
                return False, [f"Backup manifest is unreadable: {target}"]
            manifest = json.load(manifest_handle)

            lines = [f"Verified backup: {target}"]
            ok = True
            for entry in manifest.get("files", []):
                archive_name = str(entry.get("archive_path", "")).strip()
                member = members.get(archive_name)
                if member is None:
                    ok = False
                    lines.append(f"- missing: {archive_name}")
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    ok = False
                    lines.append(f"- unreadable: {archive_name}")
                    continue
                data = handle.read()
                expected_size = int(entry.get("size", -1))
                expected_hash = str(entry.get("sha256", "")).strip()
                if len(data) != expected_size or _sha256_bytes(data) != expected_hash:
                    ok = False
                    lines.append(f"- corrupted: {archive_name}")
                    continue
                lines.append(f"- ok: {archive_name}")
            return ok, lines
    except (OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        return False, [f"Backup verification failed: {exc}"]
