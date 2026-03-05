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
    "submit_proof", "give_up", "read_items", "write_items",
    "spawn", "literature_search",
    "submit_lean_proof", "read_theorem",
]
ACTIONS_NO_SEARCH = [a for a in ACTIONS if a != "literature_search"]


# ── System prompts ──────────────────────────────────────────

_TQ = '"""'  # triple-quote for embedding in prompts
_TOML_OPEN_TAG = "<OPENPROVER_ACTION>"
_TOML_CLOSE_TAG = "</OPENPROVER_ACTION>"


def _build_actions(*, lean_mode: str, has_lean: bool,
                   allow_give_up: bool, isolation: bool,
                   num_sorries: int) -> str:
    """Build the actions list section of the system prompt."""
    actions = (
        "- **spawn**: Send tasks to workers (they do the actual math / verification / exploration). Workers are pure reasoning - they only see the context you provide to them.\n"
        "- **read_items**: Read the full content of repo items (you only see one-line summaries by default).\n"
        "- **write_items**: Create, update, or delete one or more repo items.\n"
    )
    if has_lean:
        actions += "- **read_theorem**: Re-read the original theorem statement, formal Lean statement, and any provided proof.\n"
    else:
        actions += "- **read_theorem**: Re-read the original theorem statement.\n"

    if lean_mode != "formalize_only":
        if has_lean:
            actions += (
                "- **submit_proof**: Submit an informal proof (writes PROOF.md). "
                "The session ends when both PROOF.md and PROOF.lean are written. "
                "Can be re-submitted to refine, but only submit after at least one independent worker has verified the proof.\n"
            )
        else:
            actions += (
                "- **submit_proof**: Submit the proof (writes PROOF.md). **This terminates the session.** "
                "The proof must be complete, rigorous, and independently verified by a worker before submission.\n"
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
            f"(one per `sorry` in THEOREM.lean), plus optional context. No `import` statements allowed. "
            f"Auto-verified with Lean. **If verification succeeds, PROOF.lean is written.**\n"
        )
    return actions


def _build_principles(*, lean_mode: str, has_lean: bool,
                      isolation: bool, lean_items: bool) -> str:
    """Build the principles section of the system prompt."""
    principles = (
        "- You are the project leader. Delegate all mathematical work to workers - including problem analysis, exploring structure, checking special cases, and brainstorming strategies. Use parallel workers aggressively.\n"
        "- On step 1, immediately spawn workers to analyze the problem, explore key cases, and identify promising approaches. Do not spend too much time exploring the problem yourself.\n"
        "- Some problems require finding an answer before proving something about it (e.g. \"find all n such that...\").\n"
        "- Some problems are easy — that's OK. Don't overcomplicate things. A single worker might solve it in one shot.\n"
        "- Write clear, direct task descriptions for the workers. State exactly what the worker should do. Include all relevant context — workers only see what you give them.\n"
        "- You decide the proof strategy based on worker results. Balance exploration and direct proof attempts.\n"
        "- Store failed attempts in the repo - they prevent repeating mistakes.\n"
        "- One focused task per worker. Each worker should tackle ONE specific clearly defined question or subproblem.\n"
        "- Don't get stuck. If the first proof avenue does not work, try others.\n"
    )
    if not isolation:
        principles += "- Use literature_search sparingly (2-3 times max). Store results in the repo immediately.\n"
    if lean_items:
        principles += (
            "- Use write_items with format=\"lean\" to develop and test Lean code snippets. "
            "They are saved as .lean files in the repo and auto-verified by `lake env lean`. "
            "Items that fail verification are NOT saved.\n"
        )
    if lean_mode == "prove_and_formalize":
        principles += (
            "- Both an informal proof (PROOF.md) and a formal Lean 4 proof (PROOF.lean) are required. "
            "The session ends only when both are written.\n"
            "- After you have a proof in English, use read_theorem to see the formal theorem statement in Lean.\n"
            "- Before calling submit_proof, run at least one independent verification worker that checks the full informal proof end-to-end.\n"
            "- Use submit_lean_proof for the formal proof. Use submit_proof for the informal proof.\n"
        )
    elif lean_mode == "formalize_only":
        principles += (
            "- An informal proof (PROOF.md) is already provided. Your only goal is to produce PROOF.lean.\n"
            "- Use read_theorem to view the informal proof and Lean theorem statement.\n"
            "- Use submit_lean_proof to submit the final formalized proof. The session ends when it succeeds.\n"
        )
    elif lean_mode == "prove":
        principles += (
            "- Have the proof independently verified by a worker before calling submit_proof.\n"
        )
    return principles


def _build_toml_fields(*, lean_mode: str, has_lean: bool,
                       isolation: bool, lean_items: bool,
                       num_sorries: int) -> str:
    """Build the TOML fields reference section."""
    fields = ""
    if lean_mode != "formalize_only":
        fields += f"**submit_proof**: `proof = {_TQ}...{_TQ}`\n"
    fields += (
        f'**read_items**: `read = ["slug-1", "slug-2"]`\n'
        "**write_items**: one or more `[[items]]` sections:\n"
        f"{_TOML_OPEN_TAG}\n"
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
        f"{_TOML_CLOSE_TAG}\n"
        f"**spawn**: one or more `[[tasks]]` sections with `description = {_TQ}...{_TQ}`\n"
    )
    if not isolation:
        fields += f'**literature_search**: `search_query = "..."` and `search_context = {_TQ}...{_TQ}`\n'
    if lean_items:
        fields += (
            f"\n**write_items** (lean format — auto-verified): add `format = \"lean\"` to the item. "
            f"Include natural language descriptions of all theorems, definitions, and proofs as `--` comments. "
            f"The first comment line is the summary.\n"
            f"{_TOML_OPEN_TAG}\n"
            "[[items]]\n"
            'slug = "helper-lemma"\n'
            'format = "lean"\n'
            f"content = {_TQ}\n"
            "-- Summary: n * 0 = 0 for all natural numbers.\n"
            "\n"
            "-- Multiplying any natural number by zero yields zero.\n"
            "-- Proof: by induction on n, using the definition of Nat.mul.\n"
            "theorem mul_zero (n : Nat) : n * 0 = 0 := by\n"
            "  induction n with\n"
            "  | zero => rfl\n"
            "  | succ n ih => simp [Nat.succ_mul, ih]\n"
            f"{_TQ}\n"
            f"{_TOML_CLOSE_TAG}\n"
            "Items that fail Lean verification are NOT saved to the repo.\n"
        )
    if has_lean:
        fields += (
            f"\n**submit_lean_proof**: provide {num_sorries} `[[lean_blocks]]` section(s) + optional `lean_context`:\n"
            f"{_TOML_OPEN_TAG}\n"
            f"lean_context = {_TQ}\n"
            "optional helper definitions (placed after imports, no import statements allowed)\n"
            f"{_TQ}\n"
            "\n"
        )
        for i in range(min(num_sorries, 3)):
            fields += (
                f"[[lean_blocks]]\n"
                f"code = {_TQ}\n"
                f"replacement for sorry #{i}\n"
                f"{_TQ}\n"
                "\n"
            )
        if num_sorries > 3:
            fields += f"# ... up to {num_sorries} [[lean_blocks]] total\n\n"
        fields += f"{_TOML_CLOSE_TAG}\n"
    return fields


def _build_repo_items_section(*, lean_items: bool) -> str:
    """Build the Repo Items documentation section."""
    section = (
        "Items in the repo are [[slug]]-referenced files. Markdown items have format:\n"
        "```\n"
        "Summary: One sentence.\n"
        "\n"
        "<full content>\n"
        "```\n"
        "\n"
    )
    if lean_items:
        section += (
            "Lean items (format=\"lean\") are `.lean` files. The first `-- ` comment line is the summary:\n"
            "```lean\n"
            "-- Summary: One sentence.\n"
            "\n"
            "-- Natural language description of each definition/theorem/proof as comments.\n"
            "theorem foo : ... := by ...\n"
            "```\n"
            "\n"
        )
    section += (
        "Store: proven lemmas, failed attempts (brief), key observations, literature findings.\n"
        "Each item should be self-contained and atomic - one logical thing per item.\n"
        "Don't store: trivial facts, work-in-progress that belongs on the whiteboard.\n"
    )
    return section


def _build_submit_proof_section(*, lean_mode: str, has_lean: bool) -> str:
    """Build the CRITICAL: submit_proof section."""
    if lean_mode == "formalize_only":
        return ""
    if has_lean:
        return (
            "## submit_proof\n"
            "\n"
            "Only use submit_proof when you have a COMPLETE, RIGOROUS proof. "
            "Have it independently verified at least once by a worker first. "
            "You can re-submit to refine the proof if needed.\n"
        )
    return (
        "## CRITICAL: submit_proof\n"
        "\n"
        "NEVER use submit_proof unless you have a COMPLETE, RIGOROUS proof that has been VERIFIED by an independent worker. "
        "submit_proof **terminates the session** — there is no going back. The proof field must contain the full proof text.\n"
    )


def planner_system_prompt(*, isolation: bool = False, allow_give_up: bool = True,
                          lean_mode: str = "prove", num_sorries: int = 0,
                          step_num: int = 1, lean_items: bool = False) -> str:
    """Build the planner system prompt, conditionally omitting actions."""
    has_lean = lean_mode in ("prove_and_formalize", "formalize_only")

    actions = _build_actions(
        lean_mode=lean_mode, has_lean=has_lean, allow_give_up=allow_give_up,
        isolation=isolation, num_sorries=num_sorries,
    )
    principles = _build_principles(
        lean_mode=lean_mode, has_lean=has_lean,
        isolation=isolation, lean_items=lean_items,
    )
    toml_fields = _build_toml_fields(
        lean_mode=lean_mode, has_lean=has_lean, isolation=isolation,
        lean_items=lean_items, num_sorries=num_sorries,
    )
    repo_items = _build_repo_items_section(lean_items=lean_items)
    submit_proof_section = _build_submit_proof_section(
        lean_mode=lean_mode, has_lean=has_lean,
    )

    return (
        "You are a senior research mathematician coordinating a proof effort.\n"
        "\n"
        "## Your Role\n"
        "\n"
        "You are the PLANNER. You decide WHAT to do and workers do the DOING. "
        "Never do mathematical reasoning, analysis, or problem-solving yourself - not even "
        "\"just to understand the problem\" or \"just to get started\". "
        "If you need to understand the problem structure, explore special cases, identify useful lemmas, "
        "or brainstorm proof strategies - spawn workers for that. "
        "Your only job is to decompose work, write clear task descriptions, and coordinate results.\n"
        "\n"
        "Each step you choose one action:\n"
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
        "- Use LaTeX (will be displayed via MathJax): $inline$ and $$display$$\n"
        "- Abbreviations and arrows freely\n"
        '"WLOG assume $p,q$ coprime" not "Without loss of generality..."\n'
        "\n"
        f"## Repo Items\n"
        "\n"
        f"{repo_items}"
        "\n"
        f"{submit_proof_section}"
        "\n"
        "## Output Format\n"
        "\n"
        "Think step by step, then end your response with a TOML decision block wrapped in exact tags:\n"
        "\n"
        f"{_TOML_OPEN_TAG}\n"
        'action = "spawn"\n'
        'summary = "One-line description for the log"\n'
        f"whiteboard = {_TQ}\n"
        f"Updated whiteboard {'(optional on first step)' if step_num == 1 else '(REQUIRED — COMPLETE, replaces previous)'}\n"
        f"{_TQ}\n"
        "\n"
        "# Action-specific fields below (include only what's relevant)\n"
        f"{_TOML_CLOSE_TAG}\n"
        "\n"
        "Whiteboard rule: include a complete `whiteboard` field on every step except for step 1, "
        "including terminal actions like `submit_proof` and `submit_lean_proof`.\n"
        "\n"
        "### Action-specific TOML fields:\n"
        "\n"
        f"{toml_fields}"
    )

def worker_system_prompt(*, lean_worker_actions: bool = False) -> str:
    """Build worker system prompt, optionally documenting tool actions."""
    base = (
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
    if lean_worker_actions:
        base += (
            "\n"
            "## Available Tools\n"
            "\n"
            "You have access to the following tools:\n"
            "\n"
            "- **lean_verify(code)**: Verify standalone Lean 4 code. "
            "The code must include all necessary imports (e.g. `import Mathlib`). "
            "Returns 'OK' on success or compiler errors on failure.\n"
            "\n"
            "- **lean_search(query)**: Search Lean 4 declarations across Batteries, "
            "Init, Lean, Mathlib, and Std. Returns matching names, types, and docs.\n"
            "\n"
            "Use these tools to check Lean code and find relevant lemmas.\n"
        )
    return base


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
    *,
    has_lean_theorem: bool = False,
    has_proof_md: bool = False,
    has_proof_lean: bool = False,
) -> str:
    parts = [f"# Whiteboard\n\n{whiteboard}"]

    # Status indicators
    status_lines = [f"- Theorem statement: already present"]
    if has_lean_theorem:
        status_lines.append(f"- Formal Lean statement of theorem: already present")
        status_lines.append(f"- Proof in natural language: {'already present' if has_proof_md else 'missing'}")
        status_lines.append(f"- Formal Lean proof (verified): {'already present' if has_proof_lean else 'missing'}")
    else:
        status_lines.append(f"- Proof: {'already present' if has_proof_md else 'missing'}")
    parts.append(f"\n\n# What we have\n\n" + "\n".join(status_lines))

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


def format_initial_whiteboard(theorem: str, mode: str) -> str:
    theorem_text = theorem.strip()
    if mode == "prove_and_formalize":
        return f"""## Goal

**Prove and Formalize**
Produce both an informal proof and a formal Lean 4 proof of this theorem:

<theorem>
{theorem_text}
</theorem>

To see the formal theorem statement in Lean, use read_theorem. Only do that after you figured out the informal proof in English.

## Plan

- [ ] Find a proof in natural language.
- [ ] Formalize the proof in Lean 4.

## Notes

(none)
"""
    # TODO: also inject the proof!
    if mode == "formalize_only":
        return f"""## Goal

**Formalize**
Formalize the proof of the following theorem in Lean 4:

<theorem>
{theorem_text}
</theorem>

## Plan

- [ ] Formalize the proof in Lean 4.

## Notes

(none)
"""
    if mode == "prove":
        return f"""## Goal

Produce a proof of this theorem:

<theorem>
{theorem_text}
</theorem>

## Plan

- [ ] Find a proof of the theorem.

## Notes

(none)
"""
    raise ValueError(f"Invalid mode: {mode}")


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
        f"Please respond again with a valid TOML decision block wrapped in "
        f"{_TOML_OPEN_TAG} ... {_TOML_CLOSE_TAG}."
    )


def format_planner_truncated(
    original_prompt: str,
    truncated_output: str,
) -> str:
    """Build a Phase 2 prompt when planner output was truncated."""
    excerpt = truncated_output[-2000:] if len(truncated_output) > 2000 else truncated_output
    if len(truncated_output) > 2000:
        excerpt = "..." + excerpt
    return (
        f"{original_prompt}\n\n"
        f"---\n\n"
        f"Your previous response was cut off. Here is what you generated:\n\n"
        f"```\n{excerpt}\n```\n\n"
        f"Produce ONLY the {_TOML_OPEN_TAG} ... {_TOML_CLOSE_TAG} decision block. "
        f"Do not reason further."
    )


# ── TOML parser ─────────────────────────────────────────────

def parse_planner_toml(text: str) -> dict | None:
    """Extract and parse the TOML decision block from planner output."""
    # Find <OPENPROVER_ACTION> ... </OPENPROVER_ACTION> block
    match = re.search(
        rf"{re.escape(_TOML_OPEN_TAG)}\s*\n?(.*?){re.escape(_TOML_CLOSE_TAG)}",
        text,
        re.DOTALL,
    )
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

    Supports ``[[lean_blocks]]`` array-of-tables with ``code`` field.
    Also rescues ``lean_context`` if it ended up inside a lean_blocks entry
    (TOML puts keys after [[lean_blocks]] into that table).
    """
    if "lean_blocks" in parsed and isinstance(parsed["lean_blocks"], list):
        blocks = parsed["lean_blocks"]
        if blocks and isinstance(blocks[0], dict):
            # Rescue lean_context from last entry if present
            for b in blocks:
                if "lean_context" in b and "lean_context" not in parsed:
                    parsed["lean_context"] = b.pop("lean_context")
            parsed["lean_blocks"] = [b.get("code", "") for b in blocks]


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
