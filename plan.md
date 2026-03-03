╭──────────────────────────────────────────────────────────────────────────────────────────╮
│ Plan to implement                                                                        │
│                                                                                          │
│ Kage Second Brain MVP — Implementation Plan                                              │
│                                                                                          │
│ Context                                                                                  │
│                                                                                          │
│ Kage currently functions as a reactive voice/text chatbot with single-table SQLite       │
│ transcript recall. The goal is to transform it into a proactive "second brain" by adding │
│  structured entity memory (tasks, commitments, profile, preferences), lightweight intent │
│  routing, and controlled proactive suggestions. The live DB already has an orphaned      │
│ facts table (0 rows, unused) — this confirms the intent was always there but never wired │
│  up.                                                                                     │
│                                                                                          │
│ ---                                                                                      │
│ Repository Findings                                                                      │
│                                                                                          │
│ Architecture: Flat loops in main.py → BrainService orchestrates                          │
│ guardrails/prompting/generation → MemoryStore (single conversations table). No           │
│ connectors directory.                                                                    │
│                                                                                          │
│ Key extension points:                                                                    │
│ - MemoryLike protocol in core/brain_prompting.py — duck-typed; any object with recall()  │
│ + recent_turns() slots in cleanly                                                        │
│ - _persist_exchange() in core/brain.py — post-stream hook; safe place to trigger         │
│ background extraction                                                                    │
│ - build_messages() in core/brain_prompting.py — already appends a Memory: block; same    │
│ pattern for Known facts: entity block                                                    │
│ - _build_messages() in core/brain.py — the single caller; owns what gets injected        │
│                                                                                          │
│ Gaps:                                                                                    │
│ - No entity extraction — conversations never lift structured data into queryable rows    │
│ - store_exchange() doesn't return the inserted UUID (needed to link extractions back to  │
│ source)                                                                                  │
│ - No intent classification before LLM call — every turn takes the same path              │
│ - No proactive output — Kage never volunteers reminders or suggestions                   │
│ - mlx_max_tokens=150 will need a bump to ~250 for turns with proactive suggestions       │
│                                                                                          │
│ Live DB reality: facts table exists with 0 rows. Leave it untouched; add entities        │
│ alongside it.                                                                            │
│                                                                                          │
│ ---                                                                                      │
│ Proposed MVP Architecture                                                                │
│                                                                                          │
│ New directory: core/second_brain/                                                        │
│                                                                                          │
│ core/second_brain/                                                                       │
│     __init__.py        (empty)                                                           │
│     entity_store.py    (EntityStore: SQLite CRUD for                                     │
│ profile/tasks/commitments/preferences)                                                   │
│     extractor.py       (EntityExtractor: regex-based extraction, no LLM call)            │
│     planner.py         (IntentRouter: keyword/regex intent classification)               │
│     proactive.py       (ProactiveEngine: next-action suggestions from open entities)     │
│                                                                                          │
│ SQLite schema (added to kage_memory.db)                                                  │
│                                                                                          │
│ CREATE TABLE IF NOT EXISTS entities (                                                    │
│     id         TEXT PRIMARY KEY,                                                         │
│     kind       TEXT NOT NULL,   -- 'profile' | 'task' | 'commitment' | 'preference'      │
│     key        TEXT NOT NULL,   -- short label, e.g. 'location', 'remind_me', 'meeting'  │
│     value      TEXT NOT NULL,   -- always text; JSON string when structured              │
│     status     TEXT DEFAULT 'active',  -- 'active' | 'done' | 'cancelled'                │
│     due_date   TEXT,            -- ISO date or NULL                                      │
│     source_id  TEXT,            -- FK to conversations.id                                │
│     created_at TEXT NOT NULL,                                                            │
│     updated_at TEXT NOT NULL                                                             │
│ );                                                                                       │
│ CREATE INDEX IF NOT EXISTS idx_entities_kind   ON entities(kind);                        │
│ CREATE INDEX IF NOT EXISTS idx_entities_status ON entities(status);                      │
│ CREATE INDEX IF NOT EXISTS idx_entities_key    ON entities(kind, key);                   │
│                                                                                          │
│ Turn control flow (revised)                                                              │
│                                                                                          │
│ User input                                                                               │
│   → update_policy_state()          [existing, unchanged]                                 │
│   → deterministic_response()       [existing, short-circuits if triggered]               │
│   → IntentRouter.classify()        [NEW — pure Python, no LLM]                           │
│   → _build_messages()              [extended: injects entity_context when                │
│ route.inject_entities]                                                                   │
│   → LLM stream                     [unchanged]                                           │
│   → yield sentences to caller      [unchanged]                                           │
│   → _persist_exchange()            [extended: spawns background EntityExtractor thread]  │
│   → ProactiveEngine.suggest()      [NEW — only when route.proactive_ok, appended as      │
│ final sentence]                                                                          │
│                                                                                          │
│ ---                                                                                      │
│ Phased Implementation Plan                                                               │
│                                                                                          │
│ Phase 1 — Schema + EntityStore (foundation, no behavior change)                          │
│                                                                                          │
│ Goal: Add entities table to the DB; implement and test EntityStore in isolation.         │
│                                                                                          │
│ Files to create:                                                                         │
│ - core/second_brain/__init__.py (empty)                                                  │
│ - core/second_brain/entity_store.py                                                      │
│                                                                                          │
│ Files to touch:                                                                          │
│ - core/memory.py — add _init_schema_entities() called from _init_schema();               │
│ store_exchange() returns the UUID string                                                 │
│ - config.py — add 4 new frozen fields: second_brain_enabled: bool (True),                │
│ entity_recall_budget: int (400), proactive_debounce_seconds: int (60),                   │
│ extraction_enabled: bool (True)                                                          │
│                                                                                          │
│ EntityStore public API:                                                                  │
│ def upsert(kind, key, value, *, status="active", due_date=None, source_id=None) -> str   │
│ def get_by_kind(kind, *, status="active") -> list[Entity]                                │
│ def get_by_key(kind, key) -> Entity | None                                               │
│ def mark_done(entity_id) -> None                                                         │
│ def recall_for_prompt(*, char_budget=400) -> str  # plain text, voice-safe               │
│                                                                                          │
│ recall_for_prompt() output format:                                                       │
│ Tasks: review the PR (due 2026-03-04), finish report draft (due 2026-03-07)              │
│ Commitments: team meeting (2026-03-05 10:00)                                             │
│ Profile: location=Turku Finland                                                          │
│ Preferences: prefers concise answers                                                     │
│                                                                                          │
│ Tests to add (tests/test_entity_store.py):                                               │
│ - test_upsert_creates_new_entity                                                         │
│ - test_upsert_updates_existing_entity_by_kind_and_key                                    │
│ - test_get_by_kind_filters_status                                                        │
│ - test_mark_done_changes_status                                                          │
│ - test_recall_for_prompt_respects_char_budget                                            │
│ - test_recall_for_prompt_excludes_done_entities                                          │
│                                                                                          │
│ Exit criteria: All 6 tests pass; kage_memory.db gains entities table on next start;      │
│ existing tests unchanged.                                                                │
│                                                                                          │
│ ---                                                                                      │
│ Phase 2 — EntityExtractor (regex, no LLM)                                                │
│                                                                                          │
│ Goal: After each exchange, extract structured entities from user input and store them in │
│  background thread.                                                                      │
│                                                                                          │
│ Files to create: core/second_brain/extractor.py                                          │
│                                                                                          │
│ Files to touch: core/brain.py — __init__ adds self._extractor and self._entity_store;    │
│ _persist_exchange() spawns daemon thread for extraction                                  │
│                                                                                          │
│ # Extraction patterns (EntityExtractor._extract_* methods)                               │
│ profile:      "I live in X", "I'm based in X", "my timezone is X"                        │
│ tasks:        "remind me to X", "add a task X", "I need to X", "to-do: X"                │
│ commitments:  "I have a meeting X", "I promised X", "I'll X by Y"                        │
│ preferences:  "I prefer X", "I like X", "I don't like X", "I always X"                   │
│ # Date parsing: "today", "tomorrow", "next [weekday]", explicit ISO dates                │
│                                                                                          │
│ Background wiring in _persist_exchange():                                                │
│ exchange_id = self.memory.store_exchange(user_input, reply)  # now returns str           │
│ if self.settings.extraction_enabled:                                                     │
│     threading.Thread(                                                                    │
│         target=self._extract_and_store,                                                  │
│         args=(user_input, reply, exchange_id),                                           │
│         daemon=True,                                                                     │
│     ).start()                                                                            │
│                                                                                          │
│ Tests to add (tests/test_extractor.py):                                                  │
│ - test_extract_profile_location                                                          │
│ - test_extract_task_remind_me                                                            │
│ - test_extract_commitment_meeting                                                        │
│ - test_extract_preference_likes                                                          │
│ - test_extract_returns_empty_for_generic_chat                                            │
│ - test_extract_date_tomorrow_resolves_correctly                                          │
│                                                                                          │
│ Exit criteria: "remind me to review the PR tomorrow" → entities row with kind=task,      │
│ due_date=<tomorrow's ISO date>; TTFT unchanged (extraction is background).               │
│                                                                                          │
│ ---                                                                                      │
│ Phase 3 — IntentRouter                                                                   │
│                                                                                          │
│ Goal: Classify user intent before LLM call to gate entity injection and proactive        │
│ output.                                                                                  │
│                                                                                          │
│ Files to create: core/second_brain/planner.py                                            │
│                                                                                          │
│ Routing table:                                                                           │
│                                                                                          │
│ ┌──────────────────┬────────────────┬─────────────────┬────────────────┬──────────────┐  │
│ │      Intent      │    Triggers    │ inject_entities │ should_extract │ proactive_ok │  │
│ ├──────────────────┼────────────────┼─────────────────┼────────────────┼──────────────┤  │
│ │                  │ "remind me",   │                 │                │              │  │
│ │ TASK_CAPTURE     │ "add a task",  │ True            │ True           │ False        │  │
│ │                  │ "I need to"    │                 │                │              │  │
│ ├──────────────────┼────────────────┼─────────────────┼────────────────┼──────────────┤  │
│ │                  │ "I have a      │                 │                │              │  │
│ │ COMMITMENT       │ meeting", "I   │ True            │ True           │ True         │  │
│ │                  │ promised"      │                 │                │              │  │
│ ├──────────────────┼────────────────┼─────────────────┼────────────────┼──────────────┤  │
│ │                  │ "what should   │                 │                │              │  │
│ │ PLANNING_REQUEST │ I", "what's    │ True            │ False          │ True         │  │
│ │                  │ next", "help   │                 │                │              │  │
│ │                  │ me plan"       │                 │                │              │  │
│ ├──────────────────┼────────────────┼─────────────────┼────────────────┼──────────────┤  │
│ │                  │ "do you        │                 │                │              │  │
│ │ RECALL_REQUEST   │ remember",     │ True            │ False          │ False        │  │
│ │                  │ "what did I    │                 │                │              │  │
│ │                  │ tell you"      │                 │                │              │  │
│ ├──────────────────┼────────────────┼─────────────────┼────────────────┼──────────────┤  │
│ │ PROFILE_UPDATE   │ "I live in",   │ False           │ True           │ False        │  │
│ │                  │ "my timezone"  │                 │                │              │  │
│ ├──────────────────┼────────────────┼─────────────────┼────────────────┼──────────────┤  │
│ │ PREFERENCE       │ "I prefer", "I │ False           │ True           │ False        │  │
│ │                  │  don't like"   │                 │                │              │  │
│ ├──────────────────┼────────────────┼─────────────────┼────────────────┼──────────────┤  │
│ │ GENERAL          │ everything     │ False           │ False          │ False        │  │
│ │ (default)        │ else           │                 │                │              │  │
│ └──────────────────┴────────────────┴─────────────────┴────────────────┴──────────────┘  │
│                                                                                          │
│ Files to touch: core/brain.py — __init__ adds self._router = IntentRouter();             │
│ think_stream()/think_text_stream() call classify() and pass route downstream.            │
│                                                                                          │
│ Tests to add (tests/test_planner.py):                                                    │
│ - test_classify_task_capture                                                             │
│ - test_classify_planning_request_injects_entities                                        │
│ - test_classify_factual_question_no_injection                                            │
│ - test_classify_commitment_proactive_ok                                                  │
│ - test_classify_runs_under_1ms                                                           │
│                                                                                          │
│ Exit criteria: All routing tests pass; classify() is pure Python with no LLM calls;      │
│ existing streaming behavior unchanged for GENERAL intent.                                │
│                                                                                          │
│ ---                                                                                      │
│ Phase 4 — Entity Context Injection into Prompts                                          │
│                                                                                          │
│ Goal: When route.inject_entities, inject EntityStore.recall_for_prompt() into the system │
│  prompt as a Known facts: block.                                                         │
│                                                                                          │
│ Files to touch:                                                                          │
│ - core/brain_prompting.py — build_messages() gains entity_context: str = ""; appended    │
│ after the existing Memory: block                                                         │
│ - core/brain.py — _build_messages() accepts route: RouteDecision, calls                  │
│ entity_store.recall_for_prompt() when needed                                             │
│                                                                                          │
│ # In build_messages():                                                                   │
│ if entity_context:                                                                       │
│     system += f"\n\nKnown facts about {user_name}:\n{entity_context}"                    │
│                                                                                          │
│ Tests to add (extend tests/test_brain_policy.py):                                        │
│ - test_entity_context_injected_when_route_requests_it                                    │
│ - test_entity_context_absent_for_general_intent                                          │
│ - test_total_system_prompt_under_1800_chars_with_entity_block                            │
│                                                                                          │
│ Exit criteria: Planning request includes Known facts: block; factual question does not;  │
│ total system prompt ≤ 1800 chars (verified in test with populated store).                │
│                                                                                          │
│ ---                                                                                      │
│ Phase 5 — ProactiveEngine                                                                │
│                                                                                          │
│ Goal: After the main response stream, optionally yield one proactive sentence for        │
│ COMMITMENT and PLANNING_REQUEST intents.                                                 │
│                                                                                          │
│ Files to create: core/second_brain/proactive.py                                          │
│                                                                                          │
│ Gating rules (critical):                                                                 │
│ - Only runs when route.proactive_ok is True                                              │
│ - Skip if main reply already mentions the entity (substring match)                       │
│ - Debounce: no repeat within settings.proactive_debounce_seconds (default 60s)           │
│ - Maximum one suggestion per turn                                                        │
│ - Prefix with "By the way, ..." for natural voice delivery                               │
│                                                                                          │
│ Files to touch: core/brain.py — __init__ adds self._proactive =                          │
│ ProactiveEngine(self._entity_store, self.settings); think_stream() calls suggest() after │
│  collecting all parts.                                                                   │
│                                                                                          │
│ Tests to add (tests/test_proactive.py):                                                  │
│ - test_suggest_returns_due_today_task                                                    │
│ - test_suggest_skipped_when_already_mentioned_in_reply                                   │
│ - test_suggest_debounce_prevents_rapid_repeat                                            │
│ - test_suggest_returns_none_when_no_open_entities                                        │
│ - test_suggest_none_for_general_intent (proactive_ok=False)                              │
│                                                                                          │
│ Exit criteria: After "I have a standup tomorrow at 9am" → next planning request appends  │
│ "By the way, you have a standup tomorrow at 9am."; proactive never fires for GENERAL     │
│ intent; no repeat within 60s.                                                            │
│                                                                                          │
│ ---                                                                                      │
│ Acceptance Criteria                                                                      │
│                                                                                          │
│ 1. After three sessions, entities table contains at least one profile entry without      │
│ explicit prompting                                                                       │
│ 2. "remind me to submit the report by Friday" → task entity with correct due_date in DB  │
│ within 2s                                                                                │
│ 3. "what should I work on next?" response references active tasks/commitments from       │
│ entities table                                                                           │
│ 4. User saying "I finished the report" → entity row status=done, removed from            │
│ suggestions                                                                              │
│ 5. SECOND_BRAIN_ENABLED=false → identical behavior to current codebase (all existing     │
│ tests pass)                                                                              │
│ 6. TTFT for voice turn with entity injection ≤ 3s on M-series Mac (4B model)             │
│ 7. Background extraction completes within 500ms (logged at DEBUG level)                  │
│ 8. All existing 5 test modules pass with zero modifications                              │
│ 9. New 4 test modules achieve 100% branch coverage of their modules                      │
│ 10. Proactive suggestions never appear for GENERAL, FACTUAL_QUESTION, RECALL_REQUEST, or │
│  PROFILE_UPDATE intents                                                                  │
│                                                                                          │
│ ---                                                                                      │
│ Critical Files                                                                           │
│                                                                                          │
│ ┌───────────────────────────────────┬──────────────────────────────────────────────────┐ │
│ │               File                │                      Change                      │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │ core/memory.py                    │ _init_schema_entities(), store_exchange()        │ │
│ │                                   │ returns UUID                                     │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │                                   │ Wire EntityStore, EntityExtractor, IntentRouter, │ │
│ │ core/brain.py                     │  ProactiveEngine into __init__, think_stream,    │ │
│ │                                   │ think_text_stream, _build_messages,              │ │
│ │                                   │ _persist_exchange                                │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │ core/brain_prompting.py           │ build_messages() gains entity_context: str = ""  │ │
│ │                                   │ param                                            │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │ config.py                         │ 4 new frozen fields for second-brain control     │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │ core/second_brain/entity_store.py │ New                                              │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │ core/second_brain/extractor.py    │ New                                              │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │ core/second_brain/planner.py      │ New                                              │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │ core/second_brain/proactive.py    │ New                                              │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │ tests/test_entity_store.py        │ New                                              │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │ tests/test_extractor.py           │ New                                              │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │ tests/test_planner.py             │ New                                              │ │
│ ├───────────────────────────────────┼──────────────────────────────────────────────────┤ │
│ │ tests/test_proactive.py           │ New                                              │ │
│ └───────────────────────────────────┴──────────────────────────────────────────────────┘ │
│                                                                                          │
│ ---                                                                                      │
│ Open Questions (Decide Before Implementation)                                            │
│                                                                                          │
│ 1. store_exchange() return value: Return UUID directly (minimal change) or use a         │
│ callback/hook pattern? Recommendation: return UUID.                                      │
│ 2. EntityStore DB connection: Share MemoryStore's connection or own a separate           │
│ connection to same db_path? Recommendation: separate connection (simpler to test).       │
│ 3. Proactive suggestion timing: Yield in-stream as final sentence, or defer to next      │
│ wake-word activation? Recommendation: in-stream, prefixed with "By the way, ...".        │
│ 4. facts table: Drop it or leave it? Recommendation: leave it (avoid migration risk on   │
│ live DB), add entities alongside it.                                                     │
│ 5. Task capture gesture: Automatic extraction only, or also explicit "Kage, add task: X" │
│  command? Affects TASK_CAPTURE trigger patterns.                                         │
│ 6. mlx_max_tokens: Current default is 150. Should this be bumped globally or only when   │
│ proactive_ok? Recommendation: bump default to 250 in config.                             │
│                                                                                          │
│ ---                                                                                      │
│ Verification                                                                             │
│                                                                                          │
│ # After implementation:                                                                  │
│ python -m unittest discover -s tests -p 'test_*.py'   # all tests pass                   │
│ python -c "import config; print(config.get())"         # new fields visible              │
│ python main.py --text                                   # smoke test entity flow         │
│ sqlite3 data/memory/kage_memory.db ".schema entities"  # table exists                    │
│ sqlite3 data/memory/kage_memory.db "SELECT * FROM entities LIMIT 5;"  # rows appear      │
│ after conversation
