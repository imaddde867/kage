"""AgentLoop — multi-step ReAct engine.

Implements the think→act→observe cycle using buffered LLM calls
and XML tag tool-call parsing. History is kept as (assistant_output, observation)
pairs so the chat template always sees correct role alternation.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date
from typing import Any

import config
from core.agent.parser import parse_step
from core.agent.tool_registry import ToolRegistry
from core.brain_prompting import apply_chat_template

logger = logging.getLogger(__name__)

_AGENT_SYSTEM_PROMPT = """\
You are {assistant_name}, an autonomous AI agent for {user_name}. Today is {today}.

Use tools step by step to complete tasks. For each step output exactly one of:

Tool call:
<thought>Your reasoning</thought>
<tool>tool_name</tool>
<input>{{"arg": "value"}}</input>

Final answer:
<answer>Your spoken response (no markdown)</answer>

Available tools:
{tool_schemas}

Rules:
- One tool per step
- Keep answers concise and natural for speech
- If a tool fails, try an alternative or explain the limitation
- Max {max_steps} steps

{entity_context_block}"""

_OBSERVATION_TEMPLATE = "[Tool result from {name}]:\n{content}"


class AgentLoop:
    def __init__(
        self,
        runtime: Any,
        tokenizer: Any,
        registry: ToolRegistry,
        settings: config.Settings,
    ) -> None:
        self._runtime = runtime
        self._tokenizer = tokenizer
        self._registry = registry
        self._settings = settings

    def _system_prompt(self, *, entity_context: str, max_steps: int) -> str:
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
        """Build chat-template prompt with correct role alternation.

        history entries are (assistant_output, observation) pairs where
        observation is delivered as a user message so roles alternate correctly.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        for assistant_output, observation in history:
            messages.append({"role": "assistant", "content": assistant_output})
            messages.append({"role": "user", "content": observation})
        return apply_chat_template(self._tokenizer, messages)

    def _generate(self, prompt: str) -> str:
        """Accumulate full model output before parsing."""
        chunks: list[str] = []
        for chunk in self._runtime.stream_raw(
            prompt,
            max_tokens=self._settings.mlx_max_tokens,
            track_stats=False,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    def run(
        self, task: str, *, context: str = "", max_steps: int | None = None
    ) -> Iterator[str]:
        if max_steps is None:
            max_steps = self._settings.agent_max_steps

        system = self._system_prompt(entity_context=context, max_steps=max_steps)
        history: list[tuple[str, str]] = []

        for step_num in range(max_steps):
            logger.debug("AgentLoop step %d/%d", step_num + 1, max_steps)
            prompt = self._build_prompt(system, task, history)
            raw = self._generate(prompt)
            parsed = parse_step(raw)

            if parsed.thought:
                logger.debug("Thought: %s", parsed.thought)

            if parsed.answer is not None:
                yield parsed.answer
                return

            if parsed.tool_call is not None:
                logger.debug(
                    "Tool call: %s(%s)", parsed.tool_call.name, parsed.tool_call.args
                )
                result = self._registry.execute(parsed.tool_call)
                logger.debug(
                    "Tool result (error=%s): %s", result.is_error, result.content[:200]
                )
                observation = _OBSERVATION_TEMPLATE.format(
                    name=result.tool_name, content=result.content
                )
                history.append((raw, observation))
                continue

            # No XML tags — treat raw text as a plain answer
            if raw.strip():
                yield raw.strip()
                return

        yield "I reached the step limit without a final answer. Please try rephrasing your request."
