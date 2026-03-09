# OpenProver

Theorem prover powered by language models. A **planner** coordinates proof search by maintaining a whiteboard and repository, delegating focused tasks to **parallel workers** via Claude CLI or local models (vLLM).

## How it works

You give it a theorem statement (a `.md` file). A planner LLM maintains a **whiteboard** (terse mathematical scratchpad) and a **repository** of items (lemmas, observations, literature findings). Each step, the planner decides what to do: spawn workers to explore proof avenues, search literature, read/write repository items, or submit the proof.

Workers run in parallel (up to `-P` at a time), each focused on a single task. They can reference repository items via `[[wikilinks]]`. Results flow back to the planner, which updates the whiteboard and decides the next step.

With `--lean-project`, workers get access to **lean_verify** (compile Lean 4 code) and **lean_search** (search Mathlib/Lean declarations) tools, enabling interactive formal proof development.

Modes:
- **Interactive** (default): see each step's plan, accept or give feedback
- **Autonomous** (`--autonomous`): runs hands-off until proof found or step budget exhausted
- **Formal verification** (`--lean-project`): proof attempts are verified via `lake env lean`, workers can verify code and search Lean libraries

## Requirements

- Python 3.10+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (`claude` command on PATH)

## Install

```bash
git clone https://github.com/yourusername/openprover.git
cd openprover
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

For Lean search support (optional):
```bash
openprover fetch-lean-data
```

## Usage

```bash
# Interactive mode
openprover examples/sqrt2_irrational.md

# Autonomous with Opus, 100 step budget, 3 parallel workers
openprover examples/erdos_838.md --model opus --max-steps 100 --autonomous -P 3

# Resume an interrupted run
openprover --run-dir runs/sqrt2-irrational-20260217-143012

# Use different models for planner and worker
openprover examples/cauchy_schwarz.md --planner-model opus --worker-model sonnet

# Enable web searches (disabled by default)
openprover examples/cauchy_schwarz.md --no-isolation

# Use a local model (via vLLM)
openprover examples/infinite_primes.md --model minimax-m2.5 --provider-url http://localhost:8000

# Prove and formalize in Lean 4
openprover examples/addition.md \
  --lean-project ~/mathlib4 \
  --lean-theorem examples/addition.lean

# Formalize an existing proof
openprover examples/addition.md \
  --lean-project ~/mathlib4 \
  --lean-theorem examples/addition.lean \
  --proof runs/addition-20260223/PROOF.md
```

### Subcommands

| Command | Description |
|---------|-------------|
| `openprover <theorem.md>` | Run the prover (main command) |
| `openprover inspect [run_dir]` | Browse prompts and outputs from a run |
| `openprover fetch-lean-data` | Download Lean Explore search data and models |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `sonnet` | Model for both planner and worker |
| `--planner-model` | | Override model for planner |
| `--worker-model` | | Override model for worker |
| `--max-steps` | `50` | Step budget |
| `--autonomous` | off | Run without human confirmation |
| `-P, --parallelism` | `1` | Max parallel workers per step |
| `--run-dir` | | Resume from an existing run directory |
| `--no-isolation` | | Enable literature search / web access (isolation is on by default) |
| `--give-up-after` | `0.5` | Fraction of steps before give_up is allowed |
| `--lean-project` | | Path to Lean project with lakefile |
| `--lean-theorem` | | Path to THEOREM.lean (requires `--lean-project`) |
| `--proof` | | Path to existing PROOF.md (formalize-only mode) |
| `--lean-worker-actions` | auto | Worker tool calls via MCP/vLLM (auto-enabled with `--lean-project` + capable worker) |
| `--headless` | off | Non-interactive mode (logs to stdout, implies `--autonomous`) |
| `--verbose` | off | Show full LLM responses |
| `--provider-url` | `http://localhost:8000` | Server URL for local models |
| `--answer-reserve` | `4096` | Tokens reserved for answer after thinking (local models) |

Available models: `sonnet`, `opus`, `qed-nano`, `qwen3-4b`, `minimax-m2.5`

### TUI controls

| Key | Action |
|-----|--------|
| `r` | Toggle reasoning trace |
| `i` | Show worker input (on worker tabs) |
| `w` | Toggle whiteboard view |
| `a` | Toggle autonomous mode |
| `←/→` | Switch between planner/worker tabs |
| `↑/↓` | Browse step history / navigate worker actions |
| `PgUp/PgDn` | Scroll content |
| `?` | Help overlay |

When confirming a step: Tab switches between accept/feedback, Enter confirms or opens detail view, Esc dismisses. In autonomous mode: `s` summarizes, `p` pauses, `q` quits.

## Planner actions

Each step, the planner chooses one action:

| Action | Description |
|--------|-------------|
| `spawn` | Delegate tasks to parallel workers |
| `literature_search` | Web-enabled search for relevant papers/results |
| `read_items` | Retrieve full content of repository items |
| `write_items` | Create, update, or delete repository items (lean items auto-verified) |
| `write_whiteboard` | Update the whiteboard without spawning workers |
| `read_theorem` | Re-read theorem statement(s) and any provided proof |
| `submit_proof` | Submit proof (informal and/or formal Lean) |
| `give_up` | Abandon search (only allowed after give-up threshold) |

## Worker tools

When `--lean-project` is set with a tool-capable worker model, workers get access to:

| Tool | Description |
|------|-------------|
| `lean_verify(code)` | Compile Lean 4 code via `lake env lean`, returns OK or compiler errors |
| `lean_search(query)` | Search Mathlib/Lean declarations by natural language query |

Tools are provided via MCP (Claude workers) or native tool calling (vLLM workers). Actions are shown in the worker tab and can be browsed with arrow keys.

## Output

Each run creates a directory under `runs/`:

```
runs/<slug>-<timestamp>/
  THEOREM.md           - original theorem statement
  THEOREM.lean         - formal Lean statement (if provided)
  WHITEBOARD.md        - current whiteboard state
  PROOF.md             - final proof (if found)
  PROOF.lean           - formal Lean proof (if lean mode)
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
