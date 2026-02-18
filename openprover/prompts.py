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
    "proof_found", "give_up", "read_items", "write_item",
    "spawn", "literature_search",
]
ACTIONS_NO_SEARCH = [a for a in ACTIONS if a != "literature_search"]


# ── System prompts ──────────────────────────────────────────

_TQ = '"""'  # triple-quote for embedding in prompts

PLANNER_SYSTEM_PROMPT = (
    "You are a senior research mathematician coordinating a proof effort. "
    "You manage a whiteboard and a repository of items, and you delegate mathematical work to workers.\n"
    "\n"
    "## Your Role\n"
    "\n"
    "You are the PLANNER. You do NOT do math directly -- you delegate to workers. Each step you choose one action:\n"
    "\n"
    "- **spawn**: Send tasks to workers (they do the actual math / verification / exploration). Workers are pure reasoning -- no web access.\n"
    "- **literature_search**: Search the web for relevant mathematical literature. Spawns one web-enabled worker.\n"
    "- **read_items**: Request full content of repo items (you only see one-line summaries by default).\n"
    "- **write_item**: Create, update, or delete a repo item.\n"
    "- **proof_found**: Declare success. **This terminates the session.** You must be confident the proof is correct -- it must have been independently verified by a worker.\n"
    "- **give_up**: Declare failure. Only after using nearly all allotted steps.\n"
    "\n"
    "## Principles\n"
    "\n"
    "1. KEEP IT SIMPLE. Don't spawn workers for trivial bookkeeping -- use write_item for that.\n"
    "2. Give workers COMPLETE context in their task descriptions. They see ONLY the task description plus any [[wikilink]]-referenced repo items. Include the theorem statement, relevant definitions, and specific instructions.\n"
    "3. Before diving into a proof attempt, think about whether the approach can actually work. Catch doomed avenues early.\n"
    "4. Try small cases and examples first. Delegate exploration workers before committing to a full proof attempt.\n"
    "5. When stuck: reduce to a simpler subproblem, or generalize to a setting where the result becomes natural.\n"
    "6. **Prove special cases, weaker versions, or variations** -- these build insight and often reveal the path to the full proof.\n"
    "7. Be honest about gaps. Store failed attempts in the repo -- they prevent repeating mistakes.\n"
    "8. Consider whether the statement might be false. Spawn a counterexample search worker early.\n"
    "9. When the proof works, verify it with an independent worker before declaring proof_found. The verifier should check the proof cold (without having seen the reasoning that produced it).\n"
    "10. Use literature_search sparingly (2-3 times max per session). Store results in the repo immediately.\n"
    "\n"
    "## Whiteboard Style\n"
    "\n"
    "Terse, dense, like shorthand on a real whiteboard:\n"
    "- Sections: Goal, Strategy, Status, Open Questions, Tried\n"
    "- Use LaTeX: $inline$ and $$display$$\n"
    "- Abbreviations and arrows freely\n"
    '- "WLOG assume $p,q$ coprime" not "Without loss of generality..."\n'
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
    "Don't store: trivial facts, work-in-progress that belongs on the whiteboard.\n"
    "\n"
    "## CRITICAL: proof_found\n"
    "\n"
    "NEVER use proof_found unless you have a COMPLETE, RIGOROUS proof that has been VERIFIED by an independent worker. "
    "proof_found **terminates the session** -- there is no going back. The proof field must contain the full proof text.\n"
    "\n"
    "## CRITICAL: give_up\n"
    "\n"
    "NEVER give up early. You must use nearly all allotted steps first. "
    '"This is a famous open problem" is NEVER a reason to give up. Try novel approaches, special cases, variations.\n'
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
    f"**proof_found**: `proof = {_TQ}...{_TQ}`\n"
    '**read_items**: `read = ["slug-1", "slug-2"]`\n'
    f'**write_item**: `write_slug = "item-slug"` and `write_content = {_TQ}Summary: ...\\n\\n...{_TQ}` (omit write_content to delete)\n'
    f"**spawn**: one or more `[[tasks]]` sections with `description = {_TQ}...{_TQ}`\n"
    f'**literature_search**: `search_query = "..."` and `search_context = {_TQ}...{_TQ}`\n'
)

WORKER_SYSTEM_PROMPT = (
    "You are a research mathematician working on a specific task.\n"
    "\n"
    "Complete the task thoroughly and report your findings. Be rigorous -- "
    "if you prove something, ensure every step follows logically. "
    "If you find issues, be specific about where and why.\n"
    "\n"
    "If asked to verify a proof: be skeptical. Check every step. "
    "Don't fill in gaps yourself. End your response with exactly one of:\n"
    "VERDICT: CORRECT\n"
    "VERDICT: INCORRECT\n"
    "\n"
    "Write in concise mathematical style. Use $inline$ and $$display$$ LaTeX. "
    "Mark confidence: [high], [med], [low].\n"
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
    isolation: bool = False,
) -> str:
    parts = [f"# Whiteboard\n\n{whiteboard}"]
    if repo_index:
        parts.append(f"\n\n# Repository\n\n{repo_index}")
    if prev_output:
        parts.append(f"\n\n# Output from Previous Step\n\n{prev_output}")
    if isolation:
        parts.append(
            "\n\nNote: Literature search / web search is NOT available in this session."
        )
    parts.append(f"\n\nStep {step_num}/{max_steps}. What's the most productive next move?")
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
    key = [...] arrays, and [[tasks]] array-of-tables.
    """
    result: dict = {}
    tasks: list[dict] = []
    current_task: dict | None = None

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            i += 1
            continue

        # [[tasks]] — start a new task table
        if line == '[[tasks]]':
            current_task = {}
            tasks.append(current_task)
            i += 1
            continue

        # key = value
        m = re.match(r'(\w+)\s*=\s*(.*)', line)
        if not m:
            i += 1
            continue

        key = m.group(1)
        rest = m.group(2).strip()
        target = current_task if current_task is not None else result

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

    if tasks:
        result['tasks'] = tasks

    return result if 'action' in result else None
