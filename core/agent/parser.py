"""XML tag parser for the agent's tool-call format.

Why XML tags instead of JSON function-calling?
----------------------------------------------
Local 4 B models are unreliable with strict JSON schemas.  Loose XML tags
are much easier for small models to produce consistently, and re.DOTALL
regex matching is forgiving of minor whitespace/formatting variations.

Expected model output formats
------------------------------
Tool call (model wants to use a tool before answering):

    <thought>I need to look this up.</thought>
    <tool>web_search</tool>
    <input>{"query": "Python 3.13 release date"}</input>

Final answer (model is done and ready to respond to the user):

    <answer>Python 3.13 was released in October 2024.</answer>

Parsing rules (in priority order)
-----------------------------------
1. If <answer> is found → ParsedStep with answer set, tool_call=None.
2. If <tool> is found (and no <answer>) → ParsedStep with tool_call set.
3. If neither is found → the raw text itself is treated as a plain answer.

The <thought> tag is optional and informational; it is always extracted when
present but does not change the routing decision.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core.agent.tool_base import ToolCall

# All patterns use re.DOTALL so they match content that spans multiple lines.
# The non-greedy .*? stops at the first closing tag, preventing over-capture
# when multiple tagged blocks appear in the same output.
_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
_TOOL_RE    = re.compile(r"<tool>(.*?)</tool>",       re.DOTALL)
_INPUT_RE   = re.compile(r"<input>(.*?)</input>",     re.DOTALL)
_ANSWER_RE  = re.compile(r"<answer>(.*?)</answer>",   re.DOTALL)


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

    # <answer> takes priority — if found, this step is a final response.
    answer_match = _ANSWER_RE.search(raw)
    if answer_match:
        return ParsedStep(thought=thought, tool_call=None, answer=answer_match.group(1).strip())

    # <tool> present (and no <answer>) — model wants to call a tool.
    tool_match = _TOOL_RE.search(raw)
    input_match = _INPUT_RE.search(raw)
    if tool_match:
        tool_name = tool_match.group(1).strip()
        args: dict = {}
        if input_match:
            try:
                args = json.loads(input_match.group(1).strip())
            except json.JSONDecodeError:
                # Malformed JSON — use empty args dict; the tool will either
                # accept defaults or return an informative error result.
                args = {}
        return ParsedStep(thought=thought, tool_call=ToolCall(name=tool_name, args=args), answer=None)

    # No XML structure at all — treat the entire output as a plain answer.
    # This is the graceful fallback for models that ignore the XML format.
    plain = raw.strip()
    return ParsedStep(thought=thought, tool_call=None, answer=plain if plain else None)
