"""Agent subsystem — ReAct loop, tool infrastructure, and heartbeat daemon.

Architecture overview
---------------------
This package turns Kage from a reactive assistant into an autonomous agent.
The main pieces and how they connect:

    BrainService._needs_tools(input)
        │  Single 8-token LLM call that decides whether the request
        │  needs tools ("yes") or can be answered from knowledge alone ("no").
        ▼
    BrainService.agent_stream(input)   ←── entity context from EntityStore
        │
        ▼
    AgentLoop.run(task)                ← core/agent/loop.py
        │  ReAct loop: generate → parse → execute → observe, repeat.
        │
        ├── parse_step(raw)            ← core/agent/parser.py
        │     Extracts <thought>, <tool>, <input>, <answer> XML tags
        │     from the raw LLM output buffer.
        │
        └── ToolRegistry.execute(call) ← core/agent/tool_registry.py
              Looks up the tool by name and calls tool.execute(**args).
              Each tool is a subclass of Tool (core/agent/tool_base.py).

    HeartbeatAgent                     ← core/agent/heartbeat.py
        Daemon thread. Wakes every N seconds, checks for due/overdue
        entities in EntityStore, and calls speak() proactively if the
        audio channel is idle and DND hours are not active.

Adding a new tool
-----------------
1. Create a class that subclasses Tool in connectors/ (or anywhere).
2. Set class attributes: name, description, parameters (JSON Schema dict).
3. Implement execute(**kwargs) → ToolResult.
4. Register it in BrainService._build_tool_registry().
"""
