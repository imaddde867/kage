"""XML tag parser for agent tool-call format.

The model outputs one of:

    <thought>reasoning</thought>
    <tool>tool_name</tool>
    <input>{"arg": "value"}</input>

or a final answer:

    <answer>spoken response here</answer>
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core.agent.tool_base import ToolCall

_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
_TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
_INPUT_RE = re.compile(r"<input>(.*?)</input>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class ParsedStep:
    thought: str | None
    tool_call: ToolCall | None  # None when this is a final answer
    answer: str | None          # None when this is a tool call


def parse_step(raw: str) -> ParsedStep:
    """Parse one ReAct step from buffered model output."""
    thought_match = _THOUGHT_RE.search(raw)
    thought = thought_match.group(1).strip() if thought_match else None

    answer_match = _ANSWER_RE.search(raw)
    if answer_match:
        return ParsedStep(thought=thought, tool_call=None, answer=answer_match.group(1).strip())

    tool_match = _TOOL_RE.search(raw)
    input_match = _INPUT_RE.search(raw)
    if tool_match:
        tool_name = tool_match.group(1).strip()
        args: dict = {}
        if input_match:
            try:
                args = json.loads(input_match.group(1).strip())
            except json.JSONDecodeError:
                args = {}
        return ParsedStep(thought=thought, tool_call=ToolCall(name=tool_name, args=args), answer=None)

    # No XML found — treat the whole output as a plain answer
    plain = raw.strip()
    return ParsedStep(thought=thought, tool_call=None, answer=plain if plain else None)
