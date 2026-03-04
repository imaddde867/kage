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

4. If the loop exhausts max_steps, yield a graceful cap message.

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

import logging
from collections.abc import Iterator
from datetime import date
from typing import Any

import config
from core.agent.parser import parse_step
from core.agent.tool_registry import ToolRegistry
from core.brain_prompting import apply_chat_template

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

# Template used to format a tool result before appending it to history as a
# "user" message.  The model sees this text in the next generation step and
# uses it to decide whether to call another tool or produce a final answer.
_OBSERVATION_TEMPLATE = "[Tool result from {name}]:\n{content}"


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

        We buffer the entire output before calling parse_step() because the
        XML tags (<tool>, <answer>, etc.) often span multiple streaming chunks.
        Parsing a partial buffer would give incorrect results.

        track_stats=False avoids overwriting brain.last_stats, which is used
        by the timing display in main.py.
        """
        chunks: list[str] = []
        for chunk in self._runtime.stream_raw(
            prompt,
            max_tokens=self._settings.mlx_max_tokens,
            track_stats=False,
        ):
            chunks.append(chunk)
        return "".join(chunks)

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

        system = self._system_prompt(entity_context=context, max_steps=max_steps)
        # history grows as (assistant_output, observation) pairs each step
        history: list[tuple[str, str]] = []

        for step_num in range(max_steps):
            logger.debug("AgentLoop step %d/%d", step_num + 1, max_steps)
            prompt = self._build_prompt(system, task, history)
            raw = self._generate(prompt)
            parsed = parse_step(raw)

            if parsed.thought:
                logger.debug("Thought: %s", parsed.thought)

            # --- Final answer: yield and exit ---
            if parsed.answer is not None:
                yield parsed.answer
                return

            # --- Tool call: execute, record observation, loop again ---
            if parsed.tool_call is not None:
                logger.debug(
                    "Tool call: %s(%s)", parsed.tool_call.name, parsed.tool_call.args
                )
                result = self._registry.execute(parsed.tool_call)
                logger.debug(
                    "Tool result (error=%s): %s", result.is_error, result.content[:200]
                )
                # Format the result as an observation string that the model
                # will see in the "user" role on the next iteration.
                observation = _OBSERVATION_TEMPLATE.format(
                    name=result.tool_name, content=result.content
                )
                history.append((raw, observation))
                continue

            # --- No XML tags: treat raw text as a plain answer ---
            # Some model responses (especially for simple tasks) skip the
            # XML format entirely.  Accept them rather than forcing a retry.
            if raw.strip():
                yield raw.strip()
                return

        # Exhausted all steps without a final answer.
        yield "I reached the step limit without a final answer. Please try rephrasing your request."
