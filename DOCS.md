# OpenProver internals

## Architecture

OpenProver uses a **planner-worker** architecture. A single planner LLM coordinates the proof search by maintaining a whiteboard and repository, spawning focused worker tasks that run in parallel. All state lives on disk for resumability.

```
cli.py          Parse args, setup TUI, run prover, print cost
prover.py       Planner loop, step dispatch, action handlers, Repo
llm.py          LLMClient (Claude CLI), HFClient (HuggingFace HTTP)
prompts.py      All prompt templates, TOML parser, actions enum
tui.py          Full-screen terminal UI with tabs, streaming, key handling
```

## Planner-worker model

**Planner** (runs every step):
- Maintains a **whiteboard**: terse mathematical scratchpad (LaTeX, abbreviations)
- Manages a **repository** of items: lemmas, observations, failed attempts, literature findings
- Receives: whiteboard + repo index (one-line summaries) + previous workers' output
- Outputs: TOML decision with action, summary, updated whiteboard, and action-specific fields

**Workers** (spawned on demand, parallel):
- Receive a task description from the planner
- Can reference repo items via `[[wikilink]]` syntax (resolved before sending)
- Do pure reasoning (no tools) — only the planner spawns literature search workers
- Report free-form results back to the planner

**Repository** (`repo/` directory):
- Each item is a `.md` file: `Summary: One sentence.\n\nFull content`
- Planner can create, read, update, or delete items
- Used to persist proven lemmas, observations, literature reviews, failed approaches

## Modules

### `cli.py`

Entry point. Parses arguments, creates a `Prover` and `TUI`, installs a SIGINT handler (first Ctrl+C = graceful shutdown, second = force quit), runs the prover, prints cost summary on exit (`X calls · $Y.ZZZZ`).

The LLM client is constructed via a factory pattern: `Prover` calls `make_llm(archive_dir)` after setting up the work directory, so the archive path is correct from the start.

### `prover.py`

The `Prover` class owns the proving loop and all state.

**Init:** Creates or resumes a run directory (`runs/<slug>-<timestamp>/`). Loads or initializes the whiteboard. Creates the `Repo` instance. Resume is detected by checking for existing `WHITEBOARD.md` + `THEOREM.md`; step count inferred from `step_NNN` directories.

**Step flow** (`run` → `_do_step`):

1. **Planner call**: Build prompt from whiteboard + repo index + prev worker output. LLM call with streaming (shows in planner tab). Response must contain a `` ```toml `` block.

2. **Parse TOML**: Extract action, summary, whiteboard update, and action-specific fields (`[[tasks]]`, `[[items]]`, `search_query`, etc.) via `parse_planner_toml()`.

3. **Dispatch** to action handler based on the action field.

4. **Save step**: Write `planner.toml`, worker tasks/results, archive LLM calls, update `WHITEBOARD.md` on disk.

**Action handlers:**

| Handler | What it does |
|---------|-------------|
| `_handle_spawn` | Run worker tasks in parallel via `ThreadPoolExecutor` (up to `--parallelism`). Each worker gets its task description with wikilinks resolved. Results aggregated as `prev_output` for next step. |
| `_handle_literature_search` | Spawn a web-enabled worker (Claude CLI with `WebSearch` + `WebFetch` tools). Results fed back to planner. |
| `_handle_read_items` | Fetch full content of requested repo items, set as `prev_output`. |
| `_handle_write_items` | Create/update/delete repo items. Content of `null` deletes. |
| `_handle_proof_found` | Save proof to `PROOF.md`, terminate session. |
| `_handle_give_up` | Terminate. Only allowed after 80% of step budget used. |

**`Repo` class** (also in `prover.py`):
- `list_summaries()`: Returns index of all items (name + first-line summary)
- `read_item(slug)` / `read_items(slugs)`: Fetch content
- `write_item(slug, content)`: Create, update, or delete
- `resolve_wikilinks(text)`: Find `[[slug]]` references, return resolved text + appended reference materials

**Other methods:**
- `_write_discussion()`: Post-session analysis via LLM call
- `is_finished`: Check if run completed (`PROOF.md` or `DISCUSSION.md` exists)
- `inspect()`: Browse a historical run in read-only mode
- `_load_history()`: Restore step history from disk for inspect mode

### `llm.py`

Two LLM client implementations with the same interface.

**`LLMClient`** (Claude CLI wrapper):

Non-streaming:
```
claude -p --model <model> --system-prompt <...> --output-format json --tools ""
```

Streaming:
```
claude -p --model <model> --system-prompt <...> --output-format stream-json --verbose --include-partial-messages --tools ""
```
Uses `Popen` + `readline()` (not the line iterator, which has read-ahead buffering that defeats real-time streaming). Parses NDJSON lines, dispatches `content_block_delta` text to the callback.

Web search: When `web_search=True`, replaces `--tools ""` with `--permission-mode bypassPermissions --allowedTools WebSearch WebFetch`.

Archiving: Every call saved to `archive/calls/call_NNN.json` with full prompt, system prompt, schema, response, cost, timing, and errors.

**`HFClient`** (HuggingFace-compatible HTTP):
- Calls an OpenAI-compatible API at `--hf-url`
- Health check on init (`/health` endpoint)
- Streaming via chunked transfer encoding
- Same interface as `LLMClient` (web_search and json_schema ignored)
- Cost always 0.0 (local model)
- Automatically enforces `--isolation`

**Key gotchas:**
- `--json-schema` puts structured output in `raw["structured_output"]`, not `raw["result"]`
- `--tools ""` disables all tools (pure reasoning mode)
- Cost tracking uses `total_cost_usd` from the Claude CLI response

### `prompts.py`

All prompt templates, the TOML parser, and the actions enum.

**Actions:**
```python
ACTIONS = ["proof_found", "give_up", "read_items", "write_items", "spawn", "literature_search"]
ACTIONS_NO_SEARCH = [a for a in ACTIONS if a != "literature_search"]
```

**System prompts:**
- `PLANNER_SYSTEM_PROMPT`: Instructs the planner to coordinate proof search, maintain whiteboard, manage repo, delegate to workers. Includes 10 principles (keep simple, write clear tasks, catch doomed avenues early, etc.). Key rules: `proof_found` terminates the session (must have verified proof), `give_up` only after 80% of steps.
- `WORKER_SYSTEM_PROMPT`: Instructs worker to complete its task rigorously. If verifying, be skeptical and end with `VERDICT: CORRECT` or `VERDICT: INCORRECT`.
- `SEARCH_SYSTEM_PROMPT`: Instructs literature search worker.

**Prompt formatters:**
- `format_planner_prompt(whiteboard, repo_index, prev_output, ...)`: Planner input
- `format_worker_prompt(task_description, resolved_refs)`: Worker input
- `format_search_prompt(query, context)`: Literature search prompt
- `format_initial_whiteboard(theorem)`: Template with Goal / Strategy / Status / Tried sections
- `format_discussion_prompt(...)`: Post-session analysis

**TOML parser** (`parse_planner_toml`):
- Extracts TOML from `` ```toml...``` `` fenced block (or bare TOML at end of response)
- Uses `tomllib` (Python 3.11+) or `tomli` (3.10 fallback)
- Falls back to a minimal regex-based parser if neither available
- Handles `[[tasks]]` and `[[items]]` array-of-tables syntax

### `tui.py`

Full-screen terminal UI using ANSI escape codes and scroll regions.

**Layout:** Rows 1–4 are a fixed header (theorem name, step counter, model, tab bar). Row 5+ is the scrolling content area.

**Tab system:**
- Always has a `planner` tab (fixed)
- Spawn and search steps create worker tabs dynamically (`worker_step_N_i`, `search_step_N`)
- Tab bar shows status indicators: `✓` = done, `…` = streaming
- Left/right arrows switch tabs instantly

**Views:** `main` (log + streaming trace), `whiteboard`, `help`, `step_detail`, `input` (worker task on worker tabs).

**Key handling:** Background thread reads stdin in cbreak mode via `select()` + `os.read()`. Instant keys (work during streaming): `t`, `w`, `?`, `a`, `←/→`, `PgUp/PgDn`. Queued keys (during confirmation): `↑/↓`, `Tab`, `Enter`, `s`, `p`, `q`, `Esc`.

**Streaming:** Braille spinner while waiting for first token (~12 FPS). Toggleable trace text with `t`. Worker tabs show their own streaming output independently.

**Confirmation UI:** Two-option selector (accept / give feedback) with text input. Up/down browse step history; Enter on a history entry opens detail view.

**Thread safety:** All stdout writes protected by `_write_lock`. Background key thread doesn't hold the lock during I/O.

## Run directory

```
runs/<slug>-<timestamp>/
  THEOREM.md                   - immutable copy of input
  WHITEBOARD.md                - latest whiteboard state (enables resume)
  PROOF.md                     - written only if proof found
  DISCUSSION.md                - post-session analysis
  repo/
    *.md                       - repository items (lemmas, observations, etc.)
  steps/
    step_001/
      planner.toml             - planner's TOML decision
      workers/
        task_0.md              - worker task description
        result_0.md            - worker output
        worker_0_call.json     - archived LLM call
    step_002/...
  archive/
    calls/
      call_001.json            - full LLM call record
```

**Slug format:** First 40 chars of theorem, lowercased, non-alphanumeric replaced with hyphens. Example: `sqrt2-irrational-20260220-143706`.

**Resume:** If `--run-dir` contains `WHITEBOARD.md` + `THEOREM.md`, the prover picks up from the last completed step.

## Verification

Workers can be tasked with verification by the planner. A verifier worker sees only the proof text (not the reasoning that produced it) and must end its response with `VERDICT: CORRECT` or `VERDICT: INCORRECT`.

The planner is instructed to verify proofs before calling `proof_found`. If verification fails, it can spawn repair workers or try a different approach.

## Wikilinks

Task descriptions can reference repository items via `[[slug]]` syntax. Before a worker receives its task, `repo.resolve_wikilinks()` finds all references, fetches the content, and appends it as a "Referenced Materials" section. This lets the planner share proven lemmas, observations, or literature findings with workers without duplicating content in every task.

## Adding a new action

1. Add the action name to `ACTIONS` in `prompts.py`
2. Describe it in `PLANNER_SYSTEM_PROMPT` (format and when to use it)
3. Handle it in `Prover._do_step()` (add a `_handle_<action>` method)
4. Optionally add a color in `ACTION_STYLE` in `tui.py`
