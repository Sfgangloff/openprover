"""Prompt templates for OpenProver — planner/worker architecture."""

import re

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


# ── Action types ────────────────────────────────────────────

ACTIONS = [
    "proof_found", "give_up", "read_items", "write_items",
    "spawn", "literature_search",
]
ACTIONS_NO_SEARCH = [a for a in ACTIONS if a != "literature_search"]


# ── System prompts ──────────────────────────────────────────

_TQ = '"""'  # triple-quote for embedding in prompts

def planner_system_prompt(*, isolation: bool = False, allow_give_up: bool = True) -> str:
    """Build the planner system prompt, conditionally omitting actions."""
    actions = (
        "- **spawn**: Send tasks to workers (they do the actual math / verification / exploration). Workers are pure reasoning - they only see the context you provide to them.\n"
        "- **read_items**: Request full content of repo items (you only see one-line summaries by default).\n"
        "- **write_items**: Create, update, or delete one or more repo items.\n"
        "- **proof_found**: Declare success. **This terminates the session.** You must be confident the proof is correct - it must have been independently verified by a worker.\n"
    )
    if allow_give_up:
        actions += "- **give_up**: Declare failure.\n"
    if not isolation:
        actions += (
            "- **literature_search**: Search the web for relevant mathematical literature. Spawns one web-enabled worker.\n"
        )

    principles = (
        "- Keep it simple when possible (some proofs might be easy). Be brief and focused.\n"
        "- You don't have to spawn workers when it is not useful. Use common sense."
        "- Write clear, direct task descriptions. State exactly what the worker should do.\n"
        "- You decide the proof strategy. Balance exploration and direct proof attempts based on the problem. Don't be afraid to attempt a full proof early to see where it fails.\n"
        "- Store failed attempts in the repo - they prevent repeating mistakes.\n"
        "- Verify the proof with an independent worker before declaring proof_found.\n"
        "- One focused task per worker. Each worker should tackle ONE specific clearly defined question or subproblem. "
        "It's your job as the senior researcher to come up with the general direction.\n"
        "- Don't get stuck. If the first proof avenue does not work, try others.\n"
    )
    if not isolation:
        principles += "- Use literature_search sparingly (2-3 times max). Store results in the repo immediately.\n"

    toml_fields = (
        f"**proof_found**: `proof = {_TQ}...{_TQ}`\n"
        f'**read_items**: `read = ["slug-1", "slug-2"]`\n'
        "**write_items**: one or more `[[items]]` sections:\n"
        "```toml\n"
        "[[items]]\n"
        f'slug = "item-slug"\n'
        f"content = {_TQ}\n"
        "Summary: One sentence.\n"
        "\n"
        "Full content here.\n"
        f"{_TQ}\n"
        "\n"
        "[[items]]\n"
        f'slug = "another-item"\n'
        "# omit content to delete\n"
        "```\n"
        f"**spawn**: one or more `[[tasks]]` sections with `description = {_TQ}...{_TQ}`\n"
    )
    if not isolation:
        toml_fields += f'**literature_search**: `search_query = "..."` and `search_context = {_TQ}...{_TQ}`\n'


    return (
        "You are a senior research mathematician coordinating a proof effort. "
        "You manage a whiteboard and a repository of useful items, and you delegate mathematical work to workers.\n"
        "\n"
        "## Your Role\n"
        "\n"
        "You are the PLANNER. You are responsible for the high-level plan and delegate low-level stuff to workers. Each step you choose one action:\n"
        "\n"
        f"{actions}"
        "\n"
        "## Principles\n"
        "\n"
        f"{principles}"
        "\n"
        "## Whiteboard Style\n"
        "\n"
        "Terse, dense, like shorthand on a real whiteboard:\n"
        "- Sections: Goal, Strategy, Status, Open Questions, Tried\n"
        "- Use LaTeX: $inline$ and $$display$$\n"
        "- Abbreviations and arrows freely\n"
        '"WLOG assume $p,q$ coprime" not "Without loss of generality..."\n'
        "\n"
        "## Repo Items\n"
        "\n"
        "Items in the repo are [[slug]]-referenced markdown files. Each has format:\n"
        "```\n"
        "Summary: One sentence.\n"
        "\n"
        "<full content>\n"
        "```\n"
        "\n"
        "Store: proven lemmas, failed attempts (brief), key observations, literature findings.\n"
        "Each item should be self-contained and atomic - one logical thing per item.\n"
        "Don't store: trivial facts, work-in-progress that belongs on the whiteboard.\n"
        "\n"
        "## CRITICAL: proof_found\n"
        "\n"
        "NEVER use proof_found unless you have a COMPLETE, RIGOROUS proof that has been VERIFIED by an independent worker. "
        "proof_found **terminates the session** - there is no going back. The proof field must contain the full proof text.\n"
        "\n"
        "## Output Format\n"
        "\n"
        "Think step by step, then end your response with a TOML decision block:\n"
        "\n"
        "```toml\n"
        'action = "spawn"\n'
        'summary = "One-line description for the log"\n'
        f"whiteboard = {_TQ}\n"
        f"Updated whiteboard (COMPLETE, replaces previous)\n"
        f"{_TQ}\n"
        "\n"
        "# Action-specific fields below (include only what's relevant)\n"
        "```\n"
        "\n"
        "### Action-specific TOML fields:\n"
        "\n"
        f"{toml_fields}"
    )

WORKER_SYSTEM_PROMPT = (
    "You are a research mathematician working on a specific task.\n"
    "\n"
    "Complete the task thoroughly and report your findings. "
    "If you fail to complete the task, be specific about what failed and why.\n"
    "\n"
    "If asked to verify a proof: be rigorous. Check every step. "
    "Don't fill in gaps yourself. End your response with exactly one of:\n"
    "VERDICT: CORRECT\n"
    "VERDICT: INCORRECT\n"
    "\n"
    "Write in concise mathematical style. Use $inline$ and $$display$$ LaTeX.\n"
)

SEARCH_SYSTEM_PROMPT = (
    "You are a mathematical research assistant. Search for relevant mathematical "
    "literature and results. Report findings concisely with precise mathematical content."
)


# ── Prompt formatters ───────────────────────────────────────

def format_planner_prompt(
    whiteboard: str,
    repo_index: str,
    prev_output: str,
    step_num: int,
    max_steps: int,
    parallelism: int = 1,
) -> str:
    parts = [f"# Whiteboard\n\n{whiteboard}"]
    if repo_index:
        parts.append(f"\n\n# Repository\n\n{repo_index}")
    if prev_output:
        parts.append(f"\n\n# Output from Previous Step\n\n{prev_output}")
    parts.append(f"\n\nMax {parallelism} worker(s) per spawn. What's the most productive next move?")
    return "".join(parts)


def format_worker_prompt(task_description: str, resolved_refs: str) -> str:
    parts = [f"# Task\n\n{task_description}"]
    if resolved_refs:
        parts.append(f"\n\n# Referenced Materials\n\n{resolved_refs}")
    return "\n".join(parts)


def format_search_prompt(query: str, context: str) -> str:
    parts = [f"# Literature Search\n\nSearch query: {query}"]
    if context:
        parts.append(f"\n\nContext: {context}")
    parts.append(
        "\n\nSearch the web for relevant theorems, proof techniques, known results, "
        "or partial progress. Report concisely: what's known, what techniques are used, "
        "any useful references. Focus on mathematical content."
    )
    return "\n".join(parts)


def format_initial_whiteboard(theorem: str) -> str:
    return (
        f"## Goal\n\n{theorem.strip()}\n\n"
        "## Strategy\n\nTBD — analyze first.\n\n"
        "## Status\n\nStarting.\n\n"
        "## Tried\n\n(none)\n"
    )


def format_discussion_prompt(
    theorem: str,
    whiteboard: str,
    repo_index: str,
    steps_taken: int,
    max_steps: int,
    proof: str = "",
) -> str:
    parts = [
        f"# Theorem\n\n{theorem}",
        f"\n\n# Final Whiteboard\n\n{whiteboard}",
    ]
    if repo_index:
        parts.append(f"\n\n# Repository\n\n{repo_index}")
    if proof:
        parts.append(f"\n\n# Proof\n\n{proof}")
    parts.append(f"\n\n{steps_taken}/{max_steps} steps used.")
    parts.append(
        "\n\nWrite a brief discussion: result, approaches tried, key insights, "
        "open gaps, recommendations. Use $ and $$ for math."
    )
    return "".join(parts)


# ── TOML parser ─────────────────────────────────────────────

def parse_planner_toml(text: str) -> dict | None:
    """Extract and parse the TOML decision block from planner output."""
    # Find ```toml ... ``` block
    match = re.search(r'```toml\s*\n(.*?)```', text, re.DOTALL)
    if not match:
        # Fallback: try to find TOML-like content at the end
        match = re.search(r'(action\s*=\s*"[^"]+".*)$', text, re.DOTALL)
        if not match:
            return None

    toml_text = match.group(1)

    if tomllib is None:
        # Minimal fallback parser for when tomllib/tomli unavailable
        return _parse_toml_minimal(toml_text)

    try:
        return tomllib.loads(toml_text)
    except Exception:
        return _parse_toml_minimal(toml_text)


def _parse_toml_minimal(text: str) -> dict | None:
    """Minimal TOML-ish parser for our specific format.

    Handles: top-level key = "value", triple-quoted multiline strings,
    key = [...] arrays, [[tasks]] and [[items]] array-of-tables.
    """
    result: dict = {}
    # Array-of-tables: [[tasks]] and [[items]]
    array_tables: dict[str, list[dict]] = {"tasks": [], "items": []}
    current_table: dict | None = None

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            i += 1
            continue

        # [[tasks]] or [[items]] — start a new table entry
        if line in ('[[tasks]]', '[[items]]'):
            table_name = line[2:-2]
            current_table = {}
            array_tables[table_name].append(current_table)
            i += 1
            continue

        # key = value
        m = re.match(r'(\w+)\s*=\s*(.*)', line)
        if not m:
            i += 1
            continue

        key = m.group(1)
        rest = m.group(2).strip()
        target = current_table if current_table is not None else result

        # Triple-quoted multiline string
        if rest.startswith('"""'):
            content_parts = [rest[3:]]
            i += 1
            while i < len(lines):
                if '"""' in lines[i]:
                    before = lines[i].split('"""')[0]
                    content_parts.append(before)
                    break
                content_parts.append(lines[i])
                i += 1
            target[key] = '\n'.join(content_parts).strip()
            i += 1
            continue

        # Single-line string
        if rest.startswith('"') and rest.endswith('"'):
            target[key] = rest[1:-1]
            i += 1
            continue

        # Array
        if rest.startswith('['):
            arr_text = rest
            while arr_text.count('[') > arr_text.count(']') and i + 1 < len(lines):
                i += 1
                arr_text += lines[i].strip()
            items = re.findall(r'"([^"]*)"', arr_text)
            target[key] = items
            i += 1
            continue

        # Boolean
        if rest in ('true', 'false'):
            target[key] = rest == 'true'
            i += 1
            continue

        # Bare value
        target[key] = rest
        i += 1

    for name, entries in array_tables.items():
        if entries:
            result[name] = entries

    return result if 'action' in result else None
