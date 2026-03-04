"""Web fetch connector — Scrapling-first fetching with robust fallback.

Strategy:
1) Try Scrapling fetchers first (fast, resilient web retrieval).
2) If that fails, fall back to httpx.
3) Extract readable text with trafilatura when available, otherwise
   use a lightweight HTML-to-text fallback.

Install:
    pip install "scrapling[fetchers]" httpx trafilatura
"""
from __future__ import annotations

import json as _json
import re
import warnings
from urllib.parse import urlparse

import config as _config
from core.agent.tool_base import Tool, ToolResult

try:
    from scrapling.fetchers import Fetcher as _ScraplingFetcher  # type: ignore[import]
except Exception:
    _ScraplingFetcher = None  # type: ignore[assignment]

try:
    import httpx as _HTTPX  # type: ignore[import]
except ImportError:
    _HTTPX = None  # type: ignore[assignment]

try:
    # Suppress noisy lxml / cssselect UserWarnings emitted on import
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import trafilatura as _TRAFILATURA  # type: ignore[import]
except ImportError:
    _TRAFILATURA = None  # type: ignore[assignment]

_DEFAULT_MAX_CHARS = 4000
_MAX_ALLOWED_CHARS = 12000
_TIMEOUT_SECONDS = 12

_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?</\1>")
_TAG_RE = re.compile(r"(?is)<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _normalize_url(url: str) -> str:
    value = url.strip()
    if not value:
        return value

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return value

    if not parsed.scheme and not parsed.netloc and "." in parsed.path:
        return f"https://{value}"

    return value


def _condense(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _extract_text_from_html(html: str) -> str:
    if _TRAFILATURA is not None:
        try:
            text = _TRAFILATURA.extract(html) or ""
            if text.strip():
                return _condense(text)
        except Exception:
            pass

    cleaned = _SCRIPT_STYLE_RE.sub(" ", html)
    cleaned = _TAG_RE.sub(" ", cleaned)
    return _condense(cleaned)


def _extract_text_from_scrapling_response(response: object) -> str:
    raw_body = getattr(response, "body", None)
    html = ""
    if isinstance(raw_body, (bytes, bytearray)):
        html = raw_body.decode("utf-8", errors="ignore")
    elif isinstance(raw_body, str):
        html = raw_body

    if not html:
        raw_text = getattr(response, "text", None)
        if raw_text:
            html = str(raw_text)

    if html:
        text = _extract_text_from_html(html)
        if text:
            return text

    css = getattr(response, "css", None)
    if callable(css):
        try:
            nodes = css("body ::text")
            get_all = getattr(nodes, "getall", None)
            if callable(get_all):
                values = [v for v in get_all() if isinstance(v, str)]
                if values:
                    return _condense(" ".join(values))
        except Exception:
            pass

    return ""


def _is_json_content_type(headers: object) -> bool:
    """Return True when the response Content-Type indicates JSON."""
    try:
        ct = str(headers.get("content-type", "")).lower()  # type: ignore[union-attr]
        return "application/json" in ct or "text/json" in ct
    except Exception:
        return False


def _try_parse_json(text: str) -> str | None:
    """If `text` looks like JSON, return a pretty-printed version; otherwise None."""
    stripped = text.strip()
    if not stripped or stripped[0] not in ("{", "["):
        return None
    try:
        obj = _json.loads(stripped)
        return _json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return None


def _is_ssl_error(exc: Exception) -> bool:
    """Return True when the exception message suggests an SSL/TLS failure."""
    msg = str(exc).lower()
    return any(k in msg for k in ("ssl", "certificate", "tls", "handshake", "verify"))


def _clamp_max_chars(max_chars: int) -> int:
    try:
        requested = int(max_chars)
    except Exception:
        return _DEFAULT_MAX_CHARS
    return max(500, min(requested, _MAX_ALLOWED_CHARS))


class WebFetchTool(Tool):
    """Fetch a URL and return readable text content.

    Useful when the agent needs to read a specific page — e.g. documentation,
    an article, or a product page — rather than search across many pages.
    """
    name = "web_fetch"
    description = "Fetch a URL and return readable text (Scrapling-first, fallback-safe)"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "max_chars": {
                "type": "integer",
                "description": "Max characters returned (500-12000, default 4000)",
            },
        },
        "required": ["url"],
    }

    def execute(self, *, url: str, max_chars: int = _DEFAULT_MAX_CHARS, **kwargs) -> ToolResult:
        """Fetch the URL and return readable text, truncated to max_chars.

        JSON responses are detected by Content-Type or by attempting to parse
        the body; when detected, the raw JSON is returned instead of HTML extraction.

        TLS failures trigger a fallback retry with verify=False when
        WEB_FETCH_TLS_MODE=allow_insecure_fallback; the result is annotated.
        """
        del kwargs
        normalized_url = _normalize_url(url)
        parsed = urlparse(normalized_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ToolResult(
                tool_name=self.name,
                content=f"Invalid URL: {url!r}. Provide a full http/https URL.",
                is_error=True,
            )

        limit = _clamp_max_chars(max_chars)
        tls_mode = _config.get().web_fetch_tls_mode
        attempts: list[str] = []

        if _ScraplingFetcher is not None:
            try:
                response = _ScraplingFetcher.get(
                    normalized_url,
                    timeout=_TIMEOUT_SECONDS,
                    follow_redirects=True,
                )
                text = _extract_text_from_scrapling_response(response)
                if text:
                    return ToolResult(
                        tool_name=self.name,
                        content=f"URL: {normalized_url}\n{text[:limit]}",
                    )
                attempts.append("Scrapling returned no readable content.")
            except Exception as exc:
                attempts.append(f"Scrapling fetch failed: {exc}")
        else:
            attempts.append('Scrapling fetchers unavailable (install "scrapling[fetchers]").')

        if _HTTPX is not None:
            result = self._httpx_fetch(
                normalized_url, limit=limit, tls_mode=tls_mode, attempts=attempts
            )
            if result is not None:
                return result
        else:
            attempts.append("httpx unavailable.")

        details = " | ".join(attempts)
        return ToolResult(
            tool_name=self.name,
            content=(
                f"Fetch failed for {normalized_url}. {details} "
                'Install with: pip install "scrapling[fetchers]" httpx trafilatura'
            ),
            is_error=True,
        )

    def _httpx_fetch(
        self,
        url: str,
        *,
        limit: int,
        tls_mode: str,
        attempts: list[str],
        insecure: bool = False,
    ) -> "ToolResult | None":
        """Attempt one httpx GET and return a ToolResult on success, None on failure.

        When `insecure=True` the request is made with verify=False (TLS fallback
        path).  Appends failure descriptions to `attempts` so the caller can
        surface them in a consolidated error message.
        """
        try:
            response = _HTTPX.get(
                url,
                timeout=_TIMEOUT_SECONDS,
                follow_redirects=True,
                verify=not insecure,
            )
            response.raise_for_status()
            final_url = str(getattr(response, "url", url))
            annotation = " [TLS verification disabled]" if insecure else ""

            # Prefer raw JSON when Content-Type signals it, or body parses cleanly.
            if _is_json_content_type(response.headers):
                json_text = _try_parse_json(response.text) or response.text
                return ToolResult(
                    tool_name=self.name,
                    content=f"URL: {final_url}{annotation}\n{json_text[:limit]}",
                )

            json_parsed = _try_parse_json(response.text)
            if json_parsed is not None:
                return ToolResult(
                    tool_name=self.name,
                    content=f"URL: {final_url}{annotation}\n{json_parsed[:limit]}",
                )

            text = _extract_text_from_html(response.text)
            if text:
                return ToolResult(
                    tool_name=self.name,
                    content=f"URL: {final_url}{annotation}\n{text[:limit]}",
                )
            attempts.append("HTTP fallback returned no readable content.")
            return None

        except Exception as exc:
            if (
                not insecure
                and tls_mode == "allow_insecure_fallback"
                and _is_ssl_error(exc)
            ):
                attempts.append(f"HTTP fetch SSL error ({exc}) — retrying without verification")
                return self._httpx_fetch(
                    url, limit=limit, tls_mode=tls_mode, attempts=attempts, insecure=True
                )
            attempts.append(f"HTTP fallback failed: {exc}")
            return None
