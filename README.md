# OpenProver

Theorem prover powered by language models. A **planner** coordinates proof search by maintaining a whiteboard and repository, delegating focused tasks to **parallel workers** via Claude CLI or HuggingFace-compatible models.

## How it works

You give it a theorem statement (a `.md` file). A planner LLM maintains a **whiteboard** (terse mathematical scratchpad) and a **repository** of items (lemmas, observations, literature findings). Each step, the planner decides what to do: spawn workers to explore proof avenues, search literature, read/write repository items, or declare the proof found.

Workers run in parallel (up to `-P` at a time), each focused on a single task. They can reference repository items via `[[wikilinks]]`. Results flow back to the planner, which updates the whiteboard and decides the next step.

Two modes:
- **Interactive** (default): see each step's plan, accept or give feedback
- **Autonomous** (`--autonomous`): runs hands-off until proof found or step budget exhausted

## Requirements

- Python 3.10+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude` command on PATH)

## Install

```bash
git clone https://github.com/yourusername/openprover.git
cd openprover
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Interactive mode
openprover examples/sqrt2_irrational.md

# Autonomous with Opus, 100 step budget, 3 parallel workers
openprover examples/erdos_838.md --model opus --max-steps 100 --autonomous -P 3

# Resume an interrupted run
openprover --run-dir runs/sqrt2-irrational-20260217-143012

# Offline mode (no web searches)
openprover examples/cauchy_schwarz.md --isolation

# Use a HuggingFace-compatible model
openprover examples/infinite_primes.md --model qed-nano --hf-url http://localhost:8000
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `sonnet` | Model: `sonnet`, `opus`, or `qed-nano` |
| `--hf-url` | `http://localhost:8000` | Server URL for HuggingFace models |
| `--max-steps` | `50` | Step budget |
| `--autonomous` | off | Run without human confirmation |
| `--run-dir` | | Resume from an existing run directory |
| `--isolation` | off | Disable literature search / web access |
| `-P, --parallelism` | `1` | Max parallel workers per step |
| `--verbose` | off | Show full LLM responses |

### TUI controls

| Key | Action |
|-----|--------|
| `t` | Toggle reasoning trace |
| `w` | Toggle whiteboard view |
| `a` | Toggle autonomous mode |
| `←/→` | Switch between planner/worker tabs |
| `PgUp/PgDn` | Scroll content |
| `?` | Help overlay |
| `s` | Summarize progress |
| `p` | Pause (resume later with `--run-dir`) |
| `q` | Quit |

When confirming a step: Tab switches between accept/feedback, up/down browses step history, Enter on a history entry shows its detail.

## Planner actions

Each step, the planner chooses one action:

| Action | Description |
|--------|-------------|
| `spawn` | Delegate tasks to parallel workers |
| `literature_search` | Web-enabled search for relevant papers/results |
| `read_items` | Retrieve full content of repository items |
| `write_items` | Create, update, or delete repository items |
| `proof_found` | Declare the proof complete (terminates session) |
| `give_up` | Abandon search (only allowed after 80% of steps used) |

## Output

Each run creates a directory under `runs/`:

```
runs/<slug>-<timestamp>/
  THEOREM.md           - original theorem statement
  WHITEBOARD.md        - current whiteboard state
  PROOF.md             - final proof (if found)
  DISCUSSION.md        - post-session analysis
  repo/                - repository items (lemmas, observations, etc.)
  steps/step_NNN/      - per-step planner decisions and worker results
  archive/calls/       - raw LLM call logs with cost/timing
```

All state lives on disk, so runs can be interrupted and resumed.

## Example theorems

The `examples/` directory has theorem statements at various difficulty levels:

| File | Difficulty |
|------|-----------|
| `sqrt2_irrational.md` | Easy |
| `infinite_primes.md` | Easy |
| `e_irrational.md` | Medium |
| `cauchy_schwarz.md` | Medium |
| `erdos_205.md` | Hard (open) |
| `erdos_838.md` | Hard (open) |
| `collatz.md` | Hard (open) |
