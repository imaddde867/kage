# Kage Deep Benchmark Summary (March 4, 2026)

## Runs

- `deep_benchmark_20260304_213407`
- `deep_benchmark_20260304_213741`
- `deep_benchmark_20260304_214357`

## Aggregate Scores

- Weighted overall score (prompts 60%, connectors 25%, unit tests 15%): **95.4% mean**
  - Run 1: 95.7%
  - Run 2: 95.3%
  - Run 3: 95.3%
- Prompt benchmark score: **95.5% mean**
- Connector smoke score: **92.4% mean**
- Unit test reliability: **100%** (213/213 passed in every run)

## Stable Strengths (3/3 runs)

- Short-term memory recall (`ORBIT-741`) was correct.
- Long-term memory recall across BrainService restart (`Neovim`) was correct.
- Reasoning/logic prompts (arithmetic + syllogism) were correct.
- Instruction following for strict JSON output was exact.
- Honesty guardrail refusal ("no restrictions" prompt) was consistent and explicit.
- Calibration on uncertain future event stayed non-overconfident.
- Self-awareness/tool listing correctly surfaced available connectors.
- Internet live-data task used web tools and cited sources.
- Shell, notify, speak, reminder_add, update_fact, mark_task_done, list_open_tasks connectors worked in direct smoke tests.

## Known Weaknesses / Variability

- Coding quality plateau: `coding_is_prime` scored **0.70** in all runs.
  - Output was usable but not consistently complete/clean by rubric.
- Internet fetch edge case variability:
  - `internet_fetch_example` varied between **0.70** and **1.00**.
  - One run hit repeated `web_fetch` loop guard and bailed with "no reliable answer."
- Calendar reliability is unstable:
  - Prompt-level `connector_calendar_read` ranged **0.60–1.00**.
  - Direct `calendar_read` connector smoke had `osascript timed out` in 2/3 runs.
- Direct `web_fetch` connector consistently returned SSL cert-chain failures (curl error 60) in smoke tests:
  - Scored **0.50** in all runs.
  - Agent path still often succeeded by combining search + alternate fetch paths.

## Practical Verdict

Kage is strong on core assistant behavior (reasoning, memory, instruction-following, honesty, calibration) and most connectors, but has reliability debt in two external-integration areas:

1. `web_fetch` TLS/certificate handling robustness.
2. `calendar_read` AppleScript timeout/permission stability.

These two issues are the main blockers to calling the system fully production-stable for connector-heavy workflows.
