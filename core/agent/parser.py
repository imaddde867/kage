"""XML tag parser for the agent's tool-call format.

Small local models frequently drift from the canonical schema. This parser
accepts the canonical format first and then a few practical variants:

Canonical:
    <tool>web_search</tool><input>{"query":"..."}</input>

Common drift:
    <tool name="web_search" query="..." />
    <tool>web_fetch url="https://example.com"></tool>
    <tool>web_fetch url="https://example.com">   (missing closing tag)
    <search>{"query":"..."}</search>
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core.agent.tool_base import ToolCall

_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)
_TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL | re.IGNORECASE)
_INPUT_RE = re.compile(r"<input>(.*?)</input>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_ANSWER_OPEN_RE = re.compile(r"<answer>(.*)", re.DOTALL | re.IGNORECASE)

# <tool name="web_search" query="..."/> / <tool tool="web_fetch" url="...">
_TOOL_ATTR_RE = re.compile(r"<tool\s+([^>]*?)(?:/?>)", re.DOTALL | re.IGNORECASE)
# <tool>web_fetch url="https://example.com">  (no closing tag)
_TOOL_INLINE_OPEN_RE = re.compile(
    r"<tool>\s*([A-Za-z0-9_]+)\s+([^>\n]+?)>",
    re.DOTALL | re.IGNORECASE,
)

_TOOL_ALIASES: dict[str, str] = {
    "search": "web_search",
    "web_search": "web_search",
    "fetch": "web_fetch",
    "web_fetch": "web_fetch",
    "calendar": "calendar_read",
    "reminder": "reminder_add",
    "shell": "shell",
    "notify": "notify",
    "speak": "speak",
}

_ALT_BODY_RE = re.compile(
    r"<(" + "|".join(re.escape(k) for k in _TOOL_ALIASES) + r")>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)
_ALT_ATTR_RE = re.compile(
    r"<(" + "|".join(re.escape(k) for k in _TOOL_ALIASES) + r")\s+([^>]+?)(?:/>|>)",
    re.DOTALL | re.IGNORECASE,
)
_JSON_CALL_RE = re.compile(r"^\s*(\{.*\})\s*$", re.DOTALL)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\S+))')
_JSON_ENVELOPE_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_OPEN_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_CODE_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$", re.IGNORECASE)


def _normalize_tool_name(name: str) -> str:
    key = (name or "").strip().lower()
    return _TOOL_ALIASES.get(key, key)


def _parse_attrs(attr_string: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in _ATTR_RE.finditer(attr_string):
        key = match.group(1)
        value = match.group(2) or match.group(3) or match.group(4) or ""
        result[key] = value
    return result


def _args_from_body(body: str, tool_name: str) -> dict:
    body = body.strip()
    if body:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    primary_arg: dict[str, str] = {
        "web_search": "query",
        "web_fetch": "url",
        "shell": "command",
        "notify": "message",
        "speak": "message",
        "calendar_read": "days",
        "reminder_add": "title",
    }
    key = primary_arg.get(tool_name)
    if key and body:
        return {key: body}
    return {}


def _parse_tool_body(body: str) -> tuple[str, dict]:
    content = body.strip()
    if not content:
        return "", {}

    first, sep, remainder = content.partition(" ")
    tool_name = _normalize_tool_name(first)
    if not sep:
        return tool_name, {}

    rest = remainder.strip()
    inline_attrs = _parse_attrs(rest)
    if inline_attrs:
        return tool_name, inline_attrs

    if rest.startswith("{") and rest.endswith("}"):
        try:
            payload = json.loads(rest)
            if isinstance(payload, dict):
                return tool_name, payload
        except json.JSONDecodeError:
            pass

    inferred = _args_from_body(rest, tool_name)
    return tool_name, inferred


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = _CODE_FENCE_OPEN_RE.sub("", stripped, count=1)
    stripped = _CODE_FENCE_CLOSE_RE.sub("", stripped, count=1)
    return stripped.strip()


def _decode_json_object(candidate: str, *, max_trim: int = 3) -> dict | None:
    text = _strip_code_fence(candidate)
    for _ in range(max(0, int(max_trim)) + 1):
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            if text.endswith("}"):
                text = text[:-1].rstrip()
                continue
            return None
        if isinstance(parsed, dict):
            return parsed
        return None
    return None


def _parse_json_envelope(raw: str) -> ParsedStep | None:
    candidates: list[str] = []
    stripped = raw.strip()
    stripped = _strip_code_fence(stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    match = _JSON_ENVELOPE_RE.search(raw)
    if match:
        candidate = match.group(0).strip()
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        payload = _decode_json_object(candidate)
        if payload is None:
            continue

        thought = payload.get("thought")
        thought_text = str(thought).strip() if isinstance(thought, str) and thought.strip() else None
        step_type = str(payload.get("type", "")).strip().lower()

        if step_type in {"answer", "final_answer", "final"} or ("answer" in payload and "tool" not in payload):
            answer = payload.get("answer") or payload.get("final_answer") or payload.get("content")
            if isinstance(answer, str) and answer.strip():
                return ParsedStep(thought=thought_text, tool_call=None, answer=answer.strip())

        if step_type in {"tool", "action", "tool_call"} or "tool" in payload or "name" in payload:
            name = payload.get("tool") or payload.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            args = payload.get("args") or payload.get("input") or payload.get("arguments") or {}
            if isinstance(args, str):
                try:
                    decoded = json.loads(args)
                    if isinstance(decoded, dict):
                        args = decoded
                except json.JSONDecodeError:
                    args = _args_from_body(args, _normalize_tool_name(name))
            if not isinstance(args, dict):
                args = {}
            return ParsedStep(
                thought=thought_text,
                tool_call=ToolCall(name=_normalize_tool_name(name), args=args),
                answer=None,
            )

    return None


@dataclass
class ParsedStep:
    """The structured result of parsing one AgentLoop generation step.

    Exactly one of tool_call or answer will be non-None for a valid step.
    Both are None only when the model produced an empty string.

    Attributes:
        thought:   Optional reasoning text from <thought>…</thought>.
                   Logged for debugging; not shown to the user.
        tool_call: Populated when the model wants to invoke a tool.
                   The AgentLoop executes it and feeds the result back.
        answer:    Populated when the model is done reasoning and ready
                   to respond.  The AgentLoop yields this to the caller.
    """
    thought: str | None
    tool_call: ToolCall | None  # None when this is a final answer step
    answer: str | None          # None when this is a tool-call step


def parse_step(raw: str) -> ParsedStep:
    """Parse one ReAct step from a complete (buffered) model output string.

    The full output is buffered before calling this function because the
    XML tags may span multiple streaming chunks; partial matches would give
    incorrect results if parsed incrementally.

    Priority: <answer> wins over <tool> if both appear (handles rare cases
    where a confused model outputs both in the same response).

    Args:
        raw: The complete raw string produced by the LLM for this step.

    Returns:
        A ParsedStep describing what the model wants to do next.
    """
    # Always extract the optional <thought> block first for debug logging.
    thought_match = _THOUGHT_RE.search(raw)
    thought = thought_match.group(1).strip() if thought_match else None

    # 0) Preferred structured contract (JSON envelope), with XML fallback.
    json_step = _parse_json_envelope(raw)
    if json_step is not None:
        if json_step.thought is None and thought is not None:
            return ParsedStep(thought=thought, tool_call=json_step.tool_call, answer=json_step.answer)
        return json_step

    # 1) Canonical <answer>...</answer>
    answer_match = _ANSWER_RE.search(raw)
    if answer_match:
        return ParsedStep(thought=thought, tool_call=None, answer=answer_match.group(1).strip())

    # 2) Truncated <answer> (missing closing tag due cutoff)
    answer_open = _ANSWER_OPEN_RE.search(raw)
    if answer_open:
        content = answer_open.group(1).strip()
        if content:
            return ParsedStep(thought=thought, tool_call=None, answer=content)

    # 3) Canonical <tool>...</tool> with optional <input> JSON
    tool_match = _TOOL_RE.search(raw)
    input_match = _INPUT_RE.search(raw)
    if tool_match:
        tool_name, inline_args = _parse_tool_body(tool_match.group(1))
        args: dict = {}
        if input_match:
            try:
                args = json.loads(input_match.group(1).strip())
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if inline_args:
            merged = dict(inline_args)
            merged.update(args)
            args = merged
        if tool_name:
            return ParsedStep(thought=thought, tool_call=ToolCall(name=tool_name, args=args), answer=None)

    # 4) Canonical attribute form: <tool name="..." ... />
    tool_attr = _TOOL_ATTR_RE.search(raw)
    if tool_attr:
        attrs = _parse_attrs(tool_attr.group(1))
        raw_name = attrs.pop("name", "") or attrs.pop("tool", "")
        tool_name = _normalize_tool_name(raw_name)
        if tool_name:
            return ParsedStep(thought=thought, tool_call=ToolCall(name=tool_name, args=attrs), answer=None)

    # 5) Broken open form: <tool>web_fetch url="...">
    inline_open = _TOOL_INLINE_OPEN_RE.search(raw)
    if inline_open:
        tool_name = _normalize_tool_name(inline_open.group(1))
        args = _parse_attrs(inline_open.group(2))
        return ParsedStep(thought=thought, tool_call=ToolCall(name=tool_name, args=args), answer=None)

    # 6) Tool aliases: <search>...</search> and <fetch url="...">
    alt_attr = _ALT_ATTR_RE.search(raw)
    if alt_attr:
        raw_name = alt_attr.group(1).lower()
        tool_name = _normalize_tool_name(raw_name)
        args = _parse_attrs(alt_attr.group(2))
        return ParsedStep(thought=thought, tool_call=ToolCall(name=tool_name, args=args), answer=None)

    alt_body = _ALT_BODY_RE.search(raw)
    if alt_body:
        raw_name = alt_body.group(1).lower()
        tool_name = _normalize_tool_name(raw_name)
        args = _args_from_body(alt_body.group(2), tool_name)
        return ParsedStep(thought=thought, tool_call=ToolCall(name=tool_name, args=args), answer=None)

    # 7) Bare JSON call object
    json_match = _JSON_CALL_RE.match(raw)
    if json_match:
        try:
            obj = json.loads(json_match.group(1))
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("tool")
                arguments = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
                if name and isinstance(arguments, dict):
                    return ParsedStep(
                        thought=thought,
                        tool_call=ToolCall(name=_normalize_tool_name(str(name)), args=arguments),
                        answer=None,
                    )
                if "query" in obj:
                    return ParsedStep(
                        thought=thought,
                        tool_call=ToolCall(name="web_search", args={"query": obj["query"]}),
                        answer=None,
                    )
                if "url" in obj:
                    return ParsedStep(
                        thought=thought,
                        tool_call=ToolCall(name="web_fetch", args={"url": obj["url"]}),
                        answer=None,
                    )
        except (json.JSONDecodeError, TypeError):
            pass

    # 8) Tool-like malformed output: turn into a synthetic tool error path.
    if raw.lstrip().lower().startswith("<tool"):
        return ParsedStep(
            thought=thought,
            tool_call=ToolCall(name="invalid_tool_call", args={"raw": raw.strip()[:400]}),
            answer=None,
        )

    # 9) Plain-text fallback — strip internal XML tags that leaked outside a
    # proper <answer> block so they are never shown to the user as plain text.
    plain = _THOUGHT_RE.sub("", raw).strip()
    return ParsedStep(thought=thought, tool_call=None, answer=plain if plain else None)
