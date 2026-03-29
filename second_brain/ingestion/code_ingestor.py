"""Code ingestor — walks repositories and indexes code files into the knowledge graph."""

import hashlib
import logging
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable

from ingestion.code_parser import (
    EXTENSION_LANGUAGE,
    ParsedCodeFile,
    parse_code_file,
)

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDES = {
    "node_modules", ".venv", "venv", "__pycache__", ".git",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    ".eggs", ".next", ".nuxt", ".cache", "coverage",
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".min.js", ".map", ".bundle.js",
    ".wasm", ".so", ".dylib", ".dll", ".exe", ".o",
}

MAX_FILE_SIZE = 500_000  # 500 KB


# ── File discovery ───────────────────────────────────────────────────

def discover_code_files(
    root: Path,
    languages: set[str] | None = None,
    excludes: set[str] | None = None,
    max_depth: int = 10,
) -> list[Path]:
    """Recursively find code files in a directory tree."""
    if excludes is None:
        excludes = DEFAULT_EXCLUDES

    allowed_exts = set()
    for ext, lang in EXTENSION_LANGUAGE.items():
        if languages is None or lang in languages:
            allowed_exts.add(ext)

    gitignore_patterns = _read_gitignore(root)

    files: list[Path] = []
    _walk(root, root, allowed_exts, excludes, gitignore_patterns, max_depth, 0, files)
    return sorted(files)


def _read_gitignore(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []
    patterns = []
    try:
        for line in gitignore.read_text(errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line.rstrip("/"))
    except Exception:
        pass
    return patterns


def _matches_gitignore(path: Path, root: Path, patterns: list[str]) -> bool:
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        return False
    name = path.name
    for pattern in patterns:
        if fnmatch(name, pattern) or fnmatch(relative, pattern):
            return True
    return False


def _walk(
    current: Path, root: Path, allowed_exts: set[str],
    excludes: set[str], gitignore: list[str],
    max_depth: int, depth: int, out: list[Path],
):
    if depth > max_depth:
        return
    try:
        entries = sorted(current.iterdir())
    except PermissionError:
        return

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            if entry.name in excludes:
                continue
            if gitignore and _matches_gitignore(entry, root, gitignore):
                continue
            _walk(entry, root, allowed_exts, excludes, gitignore,
                  max_depth, depth + 1, out)
        elif entry.is_file():
            if entry.suffix.lower() not in allowed_exts:
                continue
            if entry.suffix.lower() in SKIP_EXTENSIONS:
                continue
            if gitignore and _matches_gitignore(entry, root, gitignore):
                continue
            try:
                if entry.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            out.append(entry)


# ── Repository detection ─────────────────────────────────────────────

def detect_repo(path: Path) -> tuple[str, str]:
    """Walk up from a path to find the repo root (.git directory).

    Returns (repo_path, repo_name). Falls back to the given path itself.
    """
    current = path if path.is_dir() else path.parent
    while current != current.parent:
        if (current / ".git").exists():
            return str(current), current.name
        current = current.parent
    return str(path), path.name


# ── Embedding generation ─────────────────────────────────────────────

def _generate_embeddings(parsed: ParsedCodeFile, engine, repo_name: str) -> dict:
    """Generate embeddings for a parsed code file.

    Returns dict with keys: file, functions, classes, methods.
    """
    # File-level embedding
    func_names = [f.name for f in parsed.functions]
    class_names = [c.name for c in parsed.classes]
    file_text = (
        f"File: {parsed.filename}\n"
        f"Repo: {repo_name}\n"
        f"Language: {parsed.language}\n"
    )
    if parsed.module_docstring:
        file_text += f"Description: {parsed.module_docstring}\n"
    if func_names:
        file_text += f"Functions: {', '.join(func_names)}\n"
    if class_names:
        file_text += f"Classes: {', '.join(class_names)}\n"
    if parsed.constants:
        file_text += f"Constants: {', '.join(parsed.constants)}\n"

    file_embedding = engine.embed(file_text[:1000])

    # Function-level embeddings
    func_embeddings = {}
    for func in parsed.functions:
        text = f"Function: {func.name}"
        if func.params:
            text += f"({', '.join(func.params)})"
        if func.return_type:
            text += f" -> {func.return_type}"
        if func.docstring:
            text += f"\n{func.docstring}"
        func_embeddings[func.name] = engine.embed(text[:500])

    # Class-level embeddings
    class_embeddings = {}
    for cls in parsed.classes:
        text = f"Class: {cls.name}"
        if cls.parents:
            text += f" extends {', '.join(cls.parents)}"
        if cls.docstring:
            text += f"\n{cls.docstring}"
        if cls.methods:
            text += f"\nMethods: {', '.join(m.name for m in cls.methods)}"
        class_embeddings[cls.name] = engine.embed(text[:500])

    # Method-level embeddings
    method_embeddings = {}
    for cls in parsed.classes:
        for method in cls.methods:
            key = f"{cls.name}::{method.name}"
            text = f"{cls.name}.{method.name}"
            if method.params:
                text += f"({', '.join(method.params)})"
            if method.docstring:
                text += f"\n{method.docstring}"
            method_embeddings[key] = engine.embed(text[:500])

    return {
        "file": file_embedding,
        "functions": func_embeddings,
        "classes": class_embeddings,
        "methods": method_embeddings,
    }


# ── Main indexing entry point ─────────────────────────────────────────

def index_code_directory(
    path: Path,
    store,
    engine,
    languages: set[str] | None = None,
    excludes: set[str] | None = None,
    max_depth: int = 10,
    on_progress: Callable[[str, str], None] | None = None,
) -> tuple[int, int, int]:
    """Index all code files in a directory.

    Returns (indexed_count, skipped_count, error_count).
    """
    path = path.expanduser().resolve()

    repo_path, repo_name = detect_repo(path)
    repo_id = hashlib.md5(repo_path.encode()).hexdigest()

    files = discover_code_files(path, languages, excludes, max_depth)

    indexed = 0
    skipped = 0
    errors = 0

    for file_path in files:
        try:
            parsed = parse_code_file(file_path)
            if parsed is None:
                skipped += 1
                continue

            existing_checksum = store.get_code_checksum(str(file_path))
            if existing_checksum == parsed.checksum:
                skipped += 1
                if on_progress:
                    on_progress("skip", str(file_path))
                continue

            embeddings = _generate_embeddings(parsed, engine, repo_name)
            store.upsert_code_file(parsed, embeddings, repo_id, repo_name, repo_path)

            indexed += 1
            if on_progress:
                on_progress("index", str(file_path))

        except Exception as e:
            errors += 1
            logger.warning(f"Error indexing {file_path}: {e}")
            if on_progress:
                on_progress("error", f"{file_path}: {e}")

    return indexed, skipped, errors
