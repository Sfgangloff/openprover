# Integrate LeanExplore & Worker Actions into OpenProver

## Context

Workers are currently pure single-turn reasoning with no side effects. This change makes workers optionally agentic — they can verify Lean code and search Lean declarations via LeanExplore while working on proofs. This also renames `<OPENPROVER_TOML>` to `<OPENPROVER_ACTION>` everywhere (no backward compatibility).

## Files to Modify

1. **openprover/prompts.py** — tag rename, worker prompt function, action parsing
2. **openprover/prover.py** — multi-turn worker loop, action execution, LeanExplore init, new init param
3. **openprover/cli.py** — new `--lean-worker-actions` CLI argument
4. **openprover/tui.py** — replace old tag names with new ones in all regexes

## Changes

### 1. Rename `<OPENPROVER_TOML>` → `<OPENPROVER_ACTION>` (prompts.py)

- Rename constants: `_TOML_OPEN_TAG` → `_ACTION_OPEN_TAG`, `_TOML_CLOSE_TAG` → `_ACTION_CLOSE_TAG`
- Update all references in `_build_toml_fields()`, `planner_system_prompt()`, `format_planner_retry()`
- In `parse_planner_toml()`, update regex to use new tag names only

### 2. Rename tags in prover.py

- Line 418-420: update error message from `<OPENPROVER_TOML>` to `<OPENPROVER_ACTION>`

### 3. Rename tags in tui.py

Replace `OPENPROVER_TOML` with `OPENPROVER_ACTION` in:
- `_strip_toml_block` (line 2015): replace `OPENPROVER_TOML` with `OPENPROVER_ACTION`, also add `OPENPROVER_ACTION_OUTPUT`
- `_iter_toml_segments` (lines 2028-2030): replace `("<openprover_toml>", "</openprover_toml>")` with `("<openprover_action>", "</openprover_action>")`, add `("<openprover_action_output>", "</openprover_action_output>")`
- `_split_toml_stream_segments` (lines 2080-2082): replace `"<OPENPROVER_TOML>": "</OPENPROVER_TOML>"` with `"<OPENPROVER_ACTION>": "</OPENPROVER_ACTION>"`, add `"<OPENPROVER_ACTION_OUTPUT>": "</OPENPROVER_ACTION_OUTPUT>"`

### 4. Worker system prompt function (prompts.py)

Replace `WORKER_SYSTEM_PROMPT` constant with `worker_system_prompt(*, lean_worker_actions: bool = False)` function.

When `lean_worker_actions=True`, append action documentation:
- **lean_verify**: Verify standalone Lean 4 code (must include imports). Format:
  ```
  <OPENPROVER_ACTION>
  action = "lean_verify"
  code = """..."""
  </OPENPROVER_ACTION>
  ```
- **lean_search**: Search Lean 4 declarations (Batteries, Init, Lean, Mathlib, Std). Format:
  ```
  <OPENPROVER_ACTION>
  action = "lean_search"
  query = "..."
  </OPENPROVER_ACTION>
  ```
- Output returned in `<OPENPROVER_ACTION_OUTPUT>` tags
- Advise: only use when producing a proof in Lean. Max 5 actions per task.

### 5. Worker action parsing (prompts.py)

Add `parse_worker_action(text: str) -> dict | None`:
- Find the LAST `<OPENPROVER_ACTION>...</OPENPROVER_ACTION>` block in worker output
- Parse TOML content inside (reuse `_parse_toml_minimal`)
- Validate `action` is in `["lean_verify", "lean_search"]`
- Return parsed dict or None

### 6. Multi-turn worker loop (prover.py)

Modify `_run_worker()` — currently single-turn (lines 1042-1072) — to support an action loop:

```
MAX_WORKER_ACTION_ROUNDS = 5

for round in range(MAX_WORKER_ACTION_ROUNDS):
    call LLM with prompt + system_prompt
    accumulate cost/duration

    if not self.lean_worker_actions → break

    parse_worker_action(response) → action or None
    if action is None → break

    execute action → action_output

    rebuild prompt: original + response + <OPENPROVER_ACTION_OUTPUT>output</OPENPROVER_ACTION_OUTPUT> + "Continue."
```

The prompt grows with conversation history on each iteration (stateless LLM, so we replay the full context). Return the final turn's response to the planner.

### 7. Action execution methods (prover.py)

**`_execute_worker_action(action, worker_id)`** — dispatch by `action["action"]`

**`_worker_action_lean_verify(action, worker_id)`**:
- Extract `code` from action
- Write to temp file via `self.lean_work_dir.make_file("worker-verify", code)`
- Run `run_lean_check(path, self.lean_project_dir)` (reuse from lean.py:86)
- Return "succeeded" or "failed\n\n{feedback}"

**`_worker_action_lean_search(action, worker_id)`**:
- Extract `query` from action
- Call `self.lean_explore_service.search(query, limit=10, packages=["Batteries", "Init", "Lean", "Mathlib", "Std"])`
- The `Service.search()` is async → use `asyncio.run()` in the worker thread
- Format results: name, module, docstring, source_text

### 8. LeanExplore initialization (prover.py)

When `lean_worker_actions=True` at `Prover.__init__`:
- Lazy import: `from lean_explore.search import SearchEngine, Service`
- Initialize once: `SearchEngine(use_local_data=False)` → `Service(engine=engine)`
- Store as `self.lean_explore_service`
- If import fails → `SystemExit("pip install lean-explore[local]")`
- If data files missing (FileNotFoundError) → `SystemExit("lean-explore data fetch")`
- The `Service` instance is shared across worker threads

### 9. CLI argument (cli.py)

Add `--lean-worker-actions` following the `--lean-items` pattern:
```python
parser.add_argument("--lean-worker-actions", action=argparse.BooleanOptionalAction, default=None,
                    help="Enable lean_verify and lean_search actions for workers (auto-enabled with --lean-project)")
```

Validation (after `--lean-items` validation):
- Default None → auto-enable when `--lean-project` is set
- Requires `--lean-project`

Pass `lean_worker_actions=args.lean_worker_actions` to `Prover(...)`.

### 10. Prover init (prover.py)

Add `lean_worker_actions: bool = False` parameter to `__init__`, store as `self.lean_worker_actions`.

## Verification

1. Run with `--lean-project` and confirm `--lean-worker-actions` auto-enables
2. Run with `--no-lean-worker-actions` and confirm workers don't see action instructions
3. Verify planner still works with renamed `<OPENPROVER_ACTION>` tags
4. Test `lean_verify` with valid Lean code → "succeeded"
5. Test `lean_verify` with invalid Lean code → failure with compiler output
6. Test `lean_search` with lean-explore installed and data fetched → formatted results
7. Test `lean_search` without lean-explore installed → clear error at startup
8. Verify multi-turn cost accumulates correctly in step_meta.toml
