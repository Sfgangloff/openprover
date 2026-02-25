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
    "submit_lean_proof", "read_theorem",
]
ACTIONS_NO_SEARCH = [a for a in ACTIONS if a != "literature_search"]


# ── System prompts ──────────────────────────────────────────

_TQ = '"""'  # triple-quote for embedding in prompts

def planner_system_prompt(*, isolation: bool = False, allow_give_up: bool = True,
                          lean_mode: str = "prove", num_sorries: int = 0,
                          step_num: int = 1) -> str:
    """Build the planner system prompt, conditionally omitting actions."""
    has_lean = lean_mode in ("prove_and_formalize", "formalize_only")

    actions = (
        "- **spawn**: Send tasks to workers (they do the actual math / verification / exploration). Workers are pure reasoning - they only see the context you provide to them.\n"
        "- **read_items**: Request full content of repo items (you only see one-line summaries by default).\n"
        "- **write_items**: Create, update, or delete one or more repo items.\n"
        "- **read_theorem**: Re-read the original theorem statement(s) and any provided proof.\n"
    )
    if lean_mode == "formalize_only":
        actions += (
            "- **proof_found**: NOT AVAILABLE in formalize-only mode. Use submit_lean_proof instead.\n"
        )
    else:
        actions += (
            "- **proof_found**: Declare success with an informal proof. **This terminates the session** (unless a formal Lean proof is also required). "
            "You must be confident the proof is correct - it must have been independently verified by a worker.\n"
        )
    if allow_give_up:
        actions += "- **give_up**: Declare failure.\n"
    if not isolation:
        actions += (
            "- **literature_search**: Search the web for relevant mathematical literature. Spawns one web-enabled worker.\n"
        )
    if has_lean:
        actions += (
            f"- **submit_lean_proof**: Submit a formal Lean 4 proof. Provide {num_sorries} replacement block(s) "
            f"(one per `sorry` in THEOREM.lean), plus optional context. No `import` statements allowed in injected code. "
            f"The completed file is automatically verified via `lake env lean`. **If verification succeeds, PROOF.lean is written.**\n"
        )

    principles = (
        "- You are the project leader. Your job is to provide the high-level directions and delegate the rest to workers. Use parallel workers to your advantage.\n"
        "- Keep it simple when possible (some proofs might be easy). Be brief and focused.\n"
        "- Write clear, direct task descriptions for the workers. State exactly what the worker should do.\n"
        "- You decide the proof strategy. Balance exploration and direct proof attempts based on the problem. Don't be afraid to attempt a full proof early to see where it fails.\n"
        "- Store failed attempts in the repo - they prevent repeating mistakes.\n"
        "- Verify the proof with an independent worker before declaring proof_found.\n"
        "- One focused task per worker. Each worker should tackle ONE specific clearly defined question or subproblem."
        "- Don't get stuck. If the first proof avenue does not work, try others.\n"
    )
    if not isolation:
        principles += "- Use literature_search sparingly (2-3 times max). Store results in the repo immediately.\n"
    if lean_mode == "prove_and_formalize":
        principles += (
            "- A formal Lean 4 proof (PROOF.lean) is also required. The session ends only when BOTH proof_found AND submit_lean_proof succeed.\n"
            "- Use write_items with format=\"lean\" to test Lean snippets (they get auto-verified by `lake env lean`).\n"
            "- After you found a proof in English, use the read_theorem action to see the formal theorem statement in Lean."
            "- Use submit_lean_proof for the final formal proof submission.\n"
        )
    elif lean_mode == "formalize_only":
        principles += (
            "- An informal proof (PROOF.md) is already provided. Your only goal is to produce PROOF.lean.\n"
            "- Use read_theorem to view the informal proof and Lean theorem statement.\n"
            "- Use write_items with format=\"lean\" to iteratively develop and test Lean code snippets.\n"
            "- Use submit_lean_proof to submit the final formalized proof. The session ends when it succeeds.\n"
        )

    toml_fields = ""
    if lean_mode != "formalize_only":
        toml_fields += f"**proof_found**: `proof = {_TQ}...{_TQ}`\n"
    toml_fields += (
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
    if has_lean:
        toml_fields += (
            f"\n**write_items** (lean format — auto-verified): add `format = \"lean\"` to the item:\n"
            "```toml\n"
            "[[items]]\n"
            'slug = "helper-lemma"\n'
            'format = "lean"\n'
            f"content = {_TQ}\n"
            "-- Lean 4 code here\n"
            f"{_TQ}\n"
            "```\n"
            f"\n**submit_lean_proof**: provide {num_sorries} `[[lean_blocks]]` section(s) + optional `lean_context`:\n"
            "```toml\n"
            f"lean_context = {_TQ}\n"
            "optional helper definitions (placed after imports, no import statements allowed)\n"
            f"{_TQ}\n"
            "\n"
        )
        for i in range(min(num_sorries, 3)):
            toml_fields += (
                f"[[lean_blocks]]\n"
                f"code = {_TQ}\n"
                f"replacement for sorry #{i}\n"
                f"{_TQ}\n"
                "\n"
            )
        if num_sorries > 3:
            toml_fields += f"# ... up to {num_sorries} [[lean_blocks]] total\n\n"
        toml_fields += "```\n"


    return (
        "You are a senior research mathematician coordinating a proof effort."
        "Important: Don't do any math yourself - delegate that to workers."
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
        f"Updated whiteboard {'(optional on first step)' if step_num == 1 else '(REQUIRED — COMPLETE, replaces previous)'}\n"
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
    prev_outputs: list[str],
    step_num: int,
    max_steps: int,
    parallelism: int = 1,
) -> str:
    parts = [f"# Whiteboard\n\n{whiteboard}"]
    if repo_index:
        parts.append(f"\n\n# Repository\n\n{repo_index}")
    if prev_outputs:
        n = len(prev_outputs)
        for i, output in enumerate(prev_outputs):
            label = f"Step -{n - i}" if n > 1 else "Previous Step"
            parts.append(f"\n\n# Output from {label}\n\n{output}")
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


def format_initial_whiteboard(theorem: str, mode: str = "prove") -> str:
    goal = theorem.strip()
    if mode == "prove_and_formalize":
        goal += (
            "\n\n**Mode: Prove and Formalize**\n"
            "Produce both an informal proof (PROOF.md) and a formal Lean 4 proof (PROOF.lean).\n"
            "Formal statement available - use read_theorem to view THEOREM.lean."
        )
    elif mode == "formalize_only":
        goal += (
            "\n\n**Mode: Formalize Only**\n"
            "An informal proof is provided. Formalize it in Lean 4 (produce PROOF.lean).\n"
            "Use read_theorem to view PROOF.md and THEOREM.lean."
        )
    return (
        f"## Goal\n\n{goal}\n\n"
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


# ── Planner retry ──────────────────────────────────────────

def format_planner_retry(
    original_prompt: str,
    raw_output: str,
    error: str,
    attempt: int,
) -> str:
    """Build a retry prompt with error feedback appended to the original."""
    excerpt = raw_output[-500:] if len(raw_output) > 500 else raw_output
    if len(raw_output) > 500:
        excerpt = "..." + excerpt
    return (
        f"{original_prompt}\n\n"
        f"---\n\n"
        f"**RETRY ({attempt}/2)** — Your previous response could not be parsed.\n\n"
        f"Error: {error}\n\n"
        f"Your previous output ended with:\n"
        f"```\n{excerpt}\n```\n\n"
        f"Please respond again with a valid ```toml decision block."
    )


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
        parsed = _parse_toml_minimal(toml_text)
    else:
        try:
            parsed = tomllib.loads(toml_text)
        except Exception:
            parsed = _parse_toml_minimal(toml_text)

    if parsed is not None:
        _collect_lean_blocks(parsed)

    return parsed


def _collect_lean_blocks(parsed: dict):
    """Normalize lean_block_N numbered keys into a lean_blocks list.

    Supports two styles from the LLM:
    1. ``[[lean_blocks]]`` array-of-tables with ``code`` field (preferred)
    2. ``lean_block_0``, ``lean_block_1``, ... numbered top-level keys

    Also rescues ``lean_context`` if it ended up inside a lean_blocks entry
    (TOML puts keys after [[lean_blocks]] into that table).
    """
    # Style 1: already collected by parser as list of dicts with "code"
    if "lean_blocks" in parsed and isinstance(parsed["lean_blocks"], list):
        blocks = parsed["lean_blocks"]
        if blocks and isinstance(blocks[0], dict):
            # Rescue lean_context from last entry if present
            for b in blocks:
                if "lean_context" in b and "lean_context" not in parsed:
                    parsed["lean_context"] = b.pop("lean_context")
            parsed["lean_blocks"] = [b.get("code", "") for b in blocks]
        return

    # Style 2: numbered keys
    lean_blocks = []
    i = 0
    while f"lean_block_{i}" in parsed:
        lean_blocks.append(parsed.pop(f"lean_block_{i}"))
        i += 1
    if lean_blocks:
        parsed["lean_blocks"] = lean_blocks


def _parse_toml_minimal(text: str) -> dict | None:
    """Minimal TOML-ish parser for our specific format.

    Handles: top-level key = "value", triple-quoted multiline strings,
    key = [...] arrays, [[tasks]] and [[items]] array-of-tables.
    """
    result: dict = {}
    # Array-of-tables: [[tasks]], [[items]], [[lean_blocks]]
    array_tables: dict[str, list[dict]] = {"tasks": [], "items": [], "lean_blocks": []}
    current_table: dict | None = None

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            i += 1
            continue

        # [[tasks]], [[items]], or [[lean_blocks]] — start a new table entry
        if line in ('[[tasks]]', '[[items]]', '[[lean_blocks]]'):
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
