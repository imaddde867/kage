## Kage Stability & Capability Hardening Plan

### Summary
- Ensure explicit routing requests trigger tools while keeping LLM-based decisioning.
- Harden reminder creation with full date/time support and zero false completions.
- Improve web search/fetch reliability (JSON handling, TLS toggle, suppression of rename warnings).
- Add truthfulness guard, logging cleanups, and configurable TLS fallback.

### 1) Routing Reliability
- Keep yes/no classifier but decode deterministically (`temperature=0`, strict output parsing).
- Retry routing prompt once on invalid output, default to tools when classifier is inconclusive.
- Expand routing system message with concrete examples (web, calendar, reminder, shell).

### 2) Truthfulness Guard
- Detect final answers claiming external actions without tool evidence and rewrite to honest phrasing.
- Update system prompts to require explicit sourcing when tool results exist.

### 3) Reminder `due_date`/`due_datetime`
- Support ISO date/datetime inputs, normalizing to local timezone and defaulting time when absent.
- Generate AppleScript via component setters to avoid locale parsing issues.
- Extend schema docs + tests, keep reminder_add return string with normalized timestamp.

### 4) Web Experience Improvements
- Import `ddgs` cleanly to avoid rename warnings.
- Detect JSON responses in `web_fetch` and return raw JSON when appropriate; keep HTML fallback.
- Track repeated tool calls in `AgentLoop` to break infinite fetch loops and prioritize answers once a valid source is found.
- Force final answers to cite source URLs.

### 5) TLS Fallback Toggle
- Add `WEB_FETCH_TLS_MODE` config (`strict`/`allow_insecure_fallback`, default `strict`).
- On verification failure, optionally retry with `verify=False` and annotate output when fallback used.

### 6) Logging & UX
- Suppress `duckduckgo_search` rename warning and noisy `lxml` warnings in user-facing output.
- Keep debug logs routed to logger only; final responses should only contain user text.

### Testing
- Unit: routing determinism, reminder date parsing, TLS fallback, JSON vs HTML detection, repeated tool call guard.
- Integration: sample CLI text session hitting memory/web/calendar/reminder paths; Bitcoin prompt rerun consistency (no step-limit).

### Assumptions
- Routing remains LLM-only (no hard keyword overrides).
- TLS default stays strict with optional toggle.
- Date-only reminders default to 09:00 local time when no time is supplied.
