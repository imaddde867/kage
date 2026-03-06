"""AgentLoop — multi-step ReAct (Reason + Act) engine.

How one request flows through this module
------------------------------------------
1. BrainService._needs_tools() decides the request requires tools.
2. BrainService.agent_stream() calls AgentLoop.run(task, context=...).
3. run() enters a for loop capped at agent_max_steps iterations:

     a. Build a chat-template prompt from:
          - The agent system prompt (with tool schemas injected)
          - The original user task
          - All prior (assistant_output, observation) history pairs
     b. Call GenerationRuntime.stream_raw() and buffer the full output.
        The full buffer is needed because XML tags may span chunk boundaries.
     c. parse_step(buffer) extracts <thought>, <tool>/<input>, or <answer>.
     d. If <answer> → yield the answer text and return.
        If <tool>   → execute the tool, append (raw, observation) to history,
                       loop back to step a.
        If neither  → yield raw text as a plain answer and return.

4. If the loop exhausts max_steps, attempt one forced finalization pass.
   If that still fails, yield a graceful cap message.

History / role alternation
---------------------------
Each history entry is a (assistant_output, observation) tuple.  When building
the next prompt the loop appends:

    {"role": "assistant", "content": assistant_output}
    {"role": "user",      "content": observation}

This gives the chat template correct role alternation, which is essential for
models that use separate KV-cache slots per role.  Consecutive same-role
messages confuse most chat-fine-tuned models.

Extending the loop
------------------
- To change the system prompt format, edit _AGENT_SYSTEM_PROMPT.
- To change how observations are formatted, edit _OBSERVATION_TEMPLATE.
- To add new tools, register them via ToolRegistry; no changes needed here.
- To add streaming mid-generation, change _generate() to yield and adjust
  the call sites.  The current design buffers intentionally for XML parsing.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from urllib.parse import urlparse, urlunparse

import config
from core.agent.parser import parse_step
from core.agent.tool_registry import ToolRegistry
from core.brain_guardrails import guard_answer_truthfulness, guard_temporal_uncertainty
from core.brain_prompting import apply_chat_template
from core.intent_signals import DEFAULT_SIGNALS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template
#
# {tool_schemas}          — filled by ToolRegistry.schema_block()
# {entity_context_block}  — filled with EntityStore facts when second_brain
#                           is enabled; empty string otherwise
# Double-braces {{ }} are literal braces in the formatted output (escaping
# Python's str.format() substitution).
# ---------------------------------------------------------------------------
_AGENT_SYSTEM_PROMPT = """\
You are {assistant_name}, an autonomous AI agent for {user_name}. Today is {today}.

Use tools step by step to complete tasks. For each step output exactly one JSON object:

Tool call:
{{"type":"tool","thought":"short reasoning","tool":"tool_name","args":{{"arg":"value"}}}}

Final answer:
{{"type":"answer","answer":"Your spoken response (no markdown)"}}

Available tools:
{tool_schemas}

Rules:
- One tool per step
- Keep answers concise and natural for speech
- For online research, prefer web_search first, then web_fetch on relevant URLs
- Do not fetch more than 2 URLs in one request
- For comparison tasks, gather evidence for each side before concluding
- For "my local machine" questions, use shell commands for system facts instead of guessing
- shell only accepts a single read-only command (no &&, pipes, redirects, or semicolons)
- For comparison/performance questions, do not finalize until you have at least one relevant tool result
- When reporting web facts, always cite the source URL in your answer
- Do not claim to have searched or fetched data unless a tool result supports it
- If a tool fails, try an alternative or explain the limitation
- XML fallback (<tool>/<input>/<answer>) is accepted for compatibility, but prefer JSON objects
- Max {max_steps} steps

{entity_context_block}"""

# Template used to format a tool result before appending it to history as a
# "user" message.  The model sees this text in the next generation step and
# uses it to decide whether to call another tool or produce a final answer.
_OBSERVATION_TEMPLATE = "[Tool result from {name}]:\n{content}"
_ANSWER_PREFIX_RE = re.compile(r"^(final answer|answer|response)\s*[:\-]\s*", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s\])>]+", re.IGNORECASE)
_URL_STRIP_CHARS = "\"'`<>{}[]().,;:!?"
# Strips <thought>, <tool>, <input> blocks that leak into the fallback answer path.
_INTERNAL_TAGS_RE = re.compile(
    r"<(?:thought|tool|input)>.*?</(?:thought|tool|input)>\s*",
    re.DOTALL | re.IGNORECASE,
)
_DANGLING_INTERNAL_RE = re.compile(
    r"<(?:thought|tool|input)\b[^>]*>.*$",
    re.DOTALL | re.IGNORECASE,
)
_INTERNAL_TAG_TOKEN_RE = re.compile(r"</?(?:thought|tool|input)\b[^>]*>", re.IGNORECASE)


@dataclass
class _LoopRunState:
    history: list[tuple[str, str]] = field(default_factory=list)
    tool_call_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    tools_used: set[str] = field(default_factory=set)
    fetched_urls: set[str] = field(default_factory=set)
    checked_urls: list[str] = field(default_factory=list)
    checked_url_keys: set[str] = field(default_factory=set)
    source_urls: list[str] = field(default_factory=list)
    source_url_keys: set[str] = field(default_factory=set)
    fetch_count: int = 0
    tool_calls: int = 0
    total_prompt_chars: int = 0
    web_attempts: int = 0
    web_successes: int = 0


class AgentLoop:
    """Multi-step ReAct engine that orchestrates tool use for a single request.

    One AgentLoop instance is created per BrainService and reused across
    requests (it holds no per-request state between run() calls).

    Args:
        runtime:  GenerationRuntime — the loaded LLM backend.
        tokenizer: The model tokenizer used to build chat-template prompts.
        registry:  ToolRegistry containing all available tools.
        settings:  Loaded Settings (agent_max_steps, mlx_max_tokens, etc.).
    """

    def __init__(
        self,
        runtime: Any,          # GenerationRuntime; typed as Any to avoid circular import
        tokenizer: Any,        # HF/MLX tokenizer; typed as Any for the same reason
        registry: ToolRegistry,
        settings: config.Settings,
    ) -> None:
        self._runtime = runtime
        self._tokenizer = tokenizer
        self._registry = registry
        self._settings = settings

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _system_prompt(self, *, entity_context: str, max_steps: int) -> str:
        """Build the agent system prompt for this request.

        entity_context is injected verbatim when non-empty so the model
        has access to the user's tasks, commitments, and profile facts.
        """
        entity_block = f"Known context:\n{entity_context}" if entity_context else ""
        return _AGENT_SYSTEM_PROMPT.format(
            assistant_name=self._settings.assistant_name,
            user_name=self._settings.user_name,
            today=date.today().isoformat(),
            tool_schemas=self._registry.schema_block(),
            max_steps=max_steps,
            entity_context_block=entity_block,
        )

    def _build_prompt(
        self, system: str, task: str, history: list[tuple[str, str]]
    ) -> str:
        """Assemble a chat-template prompt with correct role alternation.

        Messages layout:
            system   → agent system prompt
            user     → original task
            assistant → step N model output  ⎫ repeated for
            user     → step N tool result    ⎭ each prior step

        Using "user" for tool results (rather than a second "assistant"
        message) ensures correct role alternation for all chat models.

        Args:
            system:  Formatted system prompt string.
            task:    The original user request (stays constant each step).
            history: Prior (assistant_output, observation) pairs accumulated
                     by run() as the loop progresses.

        Returns:
            A single string ready to pass to GenerationRuntime.stream_raw().
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        for assistant_output, observation in history:
            messages.append({"role": "assistant", "content": assistant_output})
            messages.append({"role": "user", "content": observation})
        return apply_chat_template(self._tokenizer, messages)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """Accumulate the full model output for one step into a single string.

        We buffer the entire output before calling parse_step() because
        JSON objects or XML fallback tags may span multiple streaming chunks.
        Parsing a partial buffer would give incorrect results.

        track_stats=False avoids overwriting brain.last_stats, which is used
        by the timing display in main.py.
        """
        chunks: list[str] = []
        temperature = float(getattr(self._settings, "agent_temperature", 0.0))
        for chunk in self._runtime.stream_raw(
            prompt,
            max_tokens=self._settings.mlx_max_tokens,
            track_stats=False,
            temperature=temperature,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    def _compress_observation(self, tool_name: str, content: str) -> str:
        max_chars = int(getattr(self._settings, "agent_observation_max_chars", 1800))
        max_chars = max(500, max_chars)
        text = content.strip()
        if not text:
            return ""

        if tool_name == "web_search":
            try:
                payload = json.loads(text)
                if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                    compact_results: list[dict[str, Any]] = []
                    for item in payload["results"][:3]:
                        if not isinstance(item, dict):
                            continue
                        compact: dict[str, Any] = {}
                        rank = item.get("rank")
                        if isinstance(rank, int):
                            compact["rank"] = rank
                        title = str(item.get("title", "")).strip()
                        if title:
                            compact["title"] = title[:120]
                        url = str(item.get("url", "")).strip()
                        if url:
                            compact["url"] = url
                        snippet = str(item.get("snippet", "")).strip()
                        if snippet:
                            compact["snippet"] = snippet[:120]
                        if compact:
                            compact_results.append(compact)
                    compact_payload = {"results": compact_results}
                    text = json.dumps(compact_payload, ensure_ascii=False)
            except Exception:
                pass

        if tool_name == "web_fetch":
            lines = text.splitlines()
            url_line = lines[0].strip() if lines else ""
            if url_line.lower().startswith("url:"):
                body = " ".join(line.strip() for line in lines[1:] if line.strip())
                if body:
                    text = f"{url_line}\n{body}"
                else:
                    text = url_line

        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _history_chars(self, history: list[tuple[str, str]]) -> int:
        return sum(len(a) + len(b) for a, b in history)

    def _append_history(
        self, history: list[tuple[str, str]], assistant_raw: str, tool_name: str, content: str
    ) -> None:
        observation = _OBSERVATION_TEMPLATE.format(
            name=tool_name,
            content=self._compress_observation(tool_name, content),
        )
        history.append((assistant_raw, observation))
        budget = int(getattr(self._settings, "agent_history_char_budget", 8000))
        budget = max(1000, budget)
        while history and self._history_chars(history) > budget:
            history.pop(0)

    def _extract_urls(self, text: str) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()

        def _add(candidate: str) -> None:
            cleaned = candidate.strip().strip(_URL_STRIP_CHARS)
            if not cleaned.lower().startswith(("http://", "https://")):
                return
            if cleaned in seen:
                return
            seen.add(cleaned)
            urls.append(cleaned)

        stripped = text.strip()
        if stripped and stripped[0] in {"{", "["}:
            try:
                payload = json.loads(stripped)
                if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                    for item in payload["results"]:
                        if not isinstance(item, dict):
                            continue
                        url = str(item.get("url", "")).strip()
                        if url:
                            _add(url)
            except Exception:
                pass

        first_line = stripped.splitlines()[0].strip() if stripped else ""
        if first_line.lower().startswith("url:"):
            token = first_line[4:].strip().split(" ", 1)[0]
            if token:
                _add(token)

        for match in _URL_RE.findall(text):
            _add(match)

        return urls

    def _is_live_web_task(self, task: str) -> bool:
        return DEFAULT_SIGNALS.has(task, "live_web")

    def _append_sources_if_missing(self, answer: str, source_urls: list[str]) -> str:
        if not source_urls:
            return answer
        if _URL_RE.search(answer):
            return answer
        joined = ", ".join(source_urls[:3])
        return f"{answer.rstrip()}\n\nSources: {joined}"

    def _sanitize_user_answer_text(self, text: str) -> str:
        cleaned = _INTERNAL_TAGS_RE.sub("", text or "")
        cleaned = _DANGLING_INTERNAL_RE.sub("", cleaned)
        cleaned = _INTERNAL_TAG_TOKEN_RE.sub("", cleaned)
        cleaned = _ANSWER_PREFIX_RE.sub("", cleaned.strip())
        return cleaned.strip()

    def _canonical_web_url(self, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            return ""
        if "://" not in candidate and "." in candidate.split("/", 1)[0]:
            candidate = f"https://{candidate}"
        parsed = urlparse(candidate)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        if scheme not in {"http", "https"} or not host:
            return candidate
        try:
            port = parsed.port
        except ValueError:
            port = None
        default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        netloc = host if port is None or default_port else f"{host}:{port}"
        path = parsed.path or "/"
        return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))

    def _normalized_tool_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(args)
        if tool_name == "web_fetch":
            url = str(normalized.get("url", "")).strip()
            if url:
                normalized["url"] = self._canonical_web_url(url)
        elif tool_name == "web_search":
            query = str(normalized.get("query", "")).strip()
            if query:
                normalized["query"] = " ".join(query.split())
        return normalized

    def _tool_call_key(self, tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
        normalized = self._normalized_tool_args(tool_name, args)
        try:
            serialized = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        except TypeError:
            serialized = repr(normalized)
        return tool_name, serialized

    def _live_web_unverified_message(self, checked_urls: list[str]) -> str:
        fallback = (
            "I couldn't verify reliable live updates because the available sources were "
            "blocked or returned unusable pages."
        )
        if checked_urls:
            fallback += f" Sources checked: {', '.join(checked_urls[:3])}."
        return fallback

    def _append_unique_url(self, url: str, *, keys: set[str], values: list[str]) -> None:
        cleaned = url.strip()
        if not cleaned:
            return
        key = self._canonical_web_url(cleaned) or cleaned
        if key in keys:
            return
        keys.add(key)
        values.append(cleaned)

    def _record_web_result(self, *, state: _LoopRunState, result: Any) -> None:
        state.web_attempts += 1
        urls = []
        if result.outcome is not None and result.outcome.sources:
            urls = list(result.outcome.sources)
        if not urls:
            urls = self._extract_urls(result.content)
        for url in urls:
            self._append_unique_url(url, keys=state.checked_url_keys, values=state.checked_urls)
        if not result.is_error and urls:
            state.web_successes += 1
            for url in urls:
                self._append_unique_url(url, keys=state.source_url_keys, values=state.source_urls)

    def _answer_with_guards(
        self,
        *,
        task: str,
        answer_text: str,
        tools_used: set[str],
        web_successes: int,
        source_urls: list[str],
    ) -> str:
        answer = guard_answer_truthfulness(answer_text, tools_used)
        answer = guard_temporal_uncertainty(task, answer)
        if web_successes > 0:
            answer = self._append_sources_if_missing(answer, source_urls)
        return answer

    def _handle_parsed_answer(
        self,
        *,
        parsed_answer: str,
        raw: str,
        task: str,
        state: _LoopRunState,
    ) -> str | None:
        if self._is_live_web_task(task) and state.web_attempts > 0 and state.web_successes == 0:
            return self._live_web_unverified_message(state.checked_urls)

        answer_text = self._sanitize_user_answer_text(parsed_answer)
        if not answer_text:
            self._append_history(
                state.history,
                raw,
                "format_guard",
                "Invalid final answer format. Return a JSON answer object with user-facing content only.",
            )
            return None

        return self._answer_with_guards(
            task=task,
            answer_text=answer_text,
            tools_used=state.tools_used,
            web_successes=state.web_successes,
            source_urls=state.source_urls,
        )

    def _handle_tool_call(
        self,
        *,
        tool_call: Any,
        raw: str,
        state: _LoopRunState,
    ) -> str | None:
        logger.debug("Tool call: %s(%s)", tool_call.name, tool_call.args)
        call_key = self._tool_call_key(tool_call.name, tool_call.args)
        state.tool_call_counts[call_key] += 1
        if state.tool_call_counts[call_key] >= 3:
            logger.warning("Repeated tool call detected (%s x3) — forcing answer", call_key[0])
            logger.debug(
                "AgentLoop bailed after repeated calls (tool_calls=%d total_prompt_chars=%d)",
                state.tool_calls,
                state.total_prompt_chars,
            )
            return (
                "I kept retrieving the same data without making progress. "
                "I don't have a reliable answer for that right now."
            )

        if tool_call.name == "web_fetch":
            raw_fetch_url = str(tool_call.args.get("url", "")).strip()
            fetch_url = self._canonical_web_url(raw_fetch_url)
            if fetch_url and fetch_url in state.fetched_urls:
                shown_url = raw_fetch_url or fetch_url
                self._append_history(
                    state.history,
                    raw,
                    "web_fetch",
                    f"Already fetched {shown_url} earlier in this request. Use the previous result.",
                )
                return None
            if state.fetch_count >= 2:
                self._append_history(
                    state.history,
                    raw,
                    "web_fetch",
                    "Fetch limit reached (2 URLs). Synthesize a final answer now using the data already fetched.",
                )
                return None
            if fetch_url:
                state.fetched_urls.add(fetch_url)
            state.fetch_count += 1

        result = self._registry.execute(tool_call)
        state.tool_calls += 1
        state.tools_used.add(result.tool_name)
        if result.tool_name in {"web_search", "web_fetch"}:
            self._record_web_result(state=state, result=result)
        logger.debug("Tool result (error=%s): %s", result.is_error, result.content[:200])
        self._append_history(state.history, raw, result.tool_name, result.content)
        return None

    def _handle_plain_response(self, *, raw: str, task: str, state: _LoopRunState) -> str | None:
        plain = self._sanitize_user_answer_text(raw)
        if not plain:
            self._append_history(
                state.history,
                raw,
                "format_guard",
                (
                    "Malformed output. Next step must be either a valid tool call "
                    "or a JSON answer object with no internal tags."
                ),
            )
            return None
        if self._is_live_web_task(task) and state.web_attempts > 0 and state.web_successes == 0:
            return self._live_web_unverified_message(state.checked_urls)
        return self._answer_with_guards(
            task=task,
            answer_text=plain.strip(),
            tools_used=state.tools_used,
            web_successes=state.web_successes,
            source_urls=state.source_urls,
        )

    def _forced_finalize_answer(
        self,
        *,
        task: str,
        history: list[tuple[str, str]],
        tools_used: set[str],
        source_urls: list[str],
        checked_urls: list[str],
        web_attempts: int,
        web_successes: int,
    ) -> str:
        """Run one last no-tools generation pass to avoid step-limit dead-ends."""
        if self._is_live_web_task(task) and web_attempts > 0 and web_successes == 0:
            return self._live_web_unverified_message(checked_urls)

        system = (
            f"You are {self._settings.assistant_name}. Return the final user-facing answer now.\n"
            "Do not call tools. Output plain text only.\n"
            "Use only the user request and prior tool results in this conversation.\n"
            "If evidence is limited, state the limitation briefly and answer with best supported guidance."
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        for assistant_output, observation in history:
            messages.append({"role": "assistant", "content": assistant_output})
            messages.append({"role": "user", "content": observation})
        messages.append(
            {
                "role": "user",
                "content": "Provide the final answer now. No tool calls.",
            }
        )

        prompt = apply_chat_template(self._tokenizer, messages)
        raw = self._generate(prompt)
        parsed = parse_step(raw)
        candidate = parsed.answer if parsed.answer is not None else raw
        text = self._sanitize_user_answer_text(candidate)
        if not text:
            base = "I couldn't produce a final answer within the step budget."
            if checked_urls:
                return f"{base} Sources checked: {', '.join(checked_urls[:3])}."
            return f"{base} Please try narrowing the request."

        return self._answer_with_guards(
            task=task,
            answer_text=text,
            tools_used=tools_used,
            web_successes=web_successes,
            source_urls=source_urls,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self, task: str, *, context: str = "", max_steps: int | None = None
    ) -> Iterator[str]:
        """Execute the ReAct loop for a single user request.

        This is a generator: it yields one or more string chunks that together
        form the final answer.  Currently it yields exactly once (the full
        answer string) because we buffer generation for XML parsing, but
        callers should treat it as a stream for forward compatibility.

        Args:
            task:      The user's request (verbatim text).
            context:   Pre-formatted entity context string from EntityStore.
                       Injected into the system prompt; empty string if
                       second_brain is disabled.
            max_steps: Override for settings.agent_max_steps.  Useful in
                       tests and for callers that want a tighter budget.

        Yields:
            String chunks of the final answer, or a cap-exceeded message.
        """
        if max_steps is None:
            max_steps = self._settings.agent_max_steps

        started_at = time.perf_counter()
        system = self._system_prompt(entity_context=context, max_steps=max_steps)
        state = _LoopRunState()

        for step_num in range(max_steps):
            logger.debug("AgentLoop step %d/%d", step_num + 1, max_steps)
            prompt = self._build_prompt(system, task, state.history)
            state.total_prompt_chars += len(prompt)
            logger.debug(
                "AgentLoop metrics step=%d prompt_chars=%d history_chars=%d tool_calls=%d web_fetches=%d",
                step_num + 1,
                len(prompt),
                self._history_chars(state.history),
                state.tool_calls,
                state.fetch_count,
            )
            raw = self._generate(prompt)
            parsed = parse_step(raw)

            if parsed.thought:
                logger.debug("Thought: %s", parsed.thought)

            # --- Final answer: yield and exit ---
            if parsed.answer is not None:
                answer = self._handle_parsed_answer(
                    parsed_answer=parsed.answer,
                    raw=raw,
                    task=task,
                    state=state,
                )
                if answer is None:
                    continue
                logger.debug(
                    "AgentLoop completed in %.0fms (steps=%d tool_calls=%d web_fetches=%d total_prompt_chars=%d)",
                    (time.perf_counter() - started_at) * 1000,
                    step_num + 1,
                    state.tool_calls,
                    state.fetch_count,
                    state.total_prompt_chars,
                )
                yield answer
                return

            # --- Tool call: execute, record observation, loop again ---
            if parsed.tool_call is not None:
                bail = self._handle_tool_call(tool_call=parsed.tool_call, raw=raw, state=state)
                if bail is not None:
                    yield bail
                    return
                continue

            # --- No XML tags: treat raw text as a plain answer ---
            # Some model responses (especially for simple tasks) skip the
            # XML format entirely.  Accept them rather than forcing a retry.
            if raw.strip():
                answer = self._handle_plain_response(raw=raw, task=task, state=state)
                if answer is None:
                    continue
                logger.debug(
                    "AgentLoop plain-text completion in %.0fms (steps=%d tool_calls=%d total_prompt_chars=%d)",
                    (time.perf_counter() - started_at) * 1000,
                    step_num + 1,
                    state.tool_calls,
                    state.total_prompt_chars,
                )
                yield answer
                return

        # Exhausted all steps without a final answer: force one finalization pass.
        if state.history:
            logger.debug("AgentLoop max-steps reached — attempting forced finalization")
            yield self._forced_finalize_answer(
                task=task,
                history=state.history,
                tools_used=state.tools_used,
                source_urls=state.source_urls,
                checked_urls=state.checked_urls,
                web_attempts=state.web_attempts,
                web_successes=state.web_successes,
            )
            return

        # No history at all (e.g. empty model outputs each step): graceful cap.
        logger.debug(
            "AgentLoop hit max steps in %.0fms (max_steps=%d tool_calls=%d web_fetches=%d total_prompt_chars=%d)",
            (time.perf_counter() - started_at) * 1000,
            max_steps,
            state.tool_calls,
            state.fetch_count,
            state.total_prompt_chars,
        )
        yield "I reached the step limit without a final answer. Please try rephrasing your request."
