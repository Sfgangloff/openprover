"""Prompt templates for OpenProver - planner/worker architecture."""

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
    "submit_proof", "submit_lean_proof", "give_up", "read_items", "write_items",
    "spawn", "literature_search",
    "read_theorem", "write_whiteboard",
]
ACTIONS_NO_SEARCH = [a for a in ACTIONS if a != "literature_search"]


# ── System prompts ──────────────────────────────────────────

_TQ = '"""'  # triple-quote for embedding in prompts
_TOML_OPEN_TAG = "<OPENPROVER_ACTION>"
_TOML_CLOSE_TAG = "</OPENPROVER_ACTION>"


def _build_actions(*, lean_mode: str, has_lean: bool,
                   allow_give_up: bool, isolation: bool) -> str:
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
    actions += "- **write_whiteboard**: Update the whiteboard with new information.\n"

    if lean_mode == "prove":
        actions += (
            "- **submit_proof**: Submit the proof by referencing a repo item slug. **This terminates the session.** "
            "The proof must be complete, rigorous, and independently verified by a worker before submission.\n"
        )
    elif lean_mode == "prove_and_formalize":
        actions += (
            "- **submit_proof**: Submit the informal proof by referencing a repo item slug (proof_slug). "
            "The session ends when both informal and formal proofs are accepted.\n"
            "- **submit_lean_proof**: Submit the formal Lean 4 proof by referencing a lean repo item slug (lean_proof_slug). "
            "Auto-verified with Lean. "
            "The session ends when both informal and formal proofs are accepted.\n"
        )
    elif lean_mode == "formalize_only":
        actions += (
            "- **submit_lean_proof**: Submit the formal Lean 4 proof by referencing a lean repo item slug (lean_proof_slug). "
            "Auto-verified with Lean. **If verification succeeds, the session ends.**\n"
        )
    if allow_give_up:
        actions += "- **give_up**: Declare failure.\n"
    if not isolation:
        actions += (
            "- **literature_search**: Search the web for relevant mathematical literature. Spawns one web-enabled worker.\n"
        )
    return actions


def _build_principles(*, lean_mode: str, has_lean: bool,
                      isolation: bool, lean_items: bool) -> str:
    """Build the principles section of the system prompt."""
    principles = (
        "- You are the project leader. Delegate all mathematical work to workers - including problem analysis, exploring structure, checking special cases, and brainstorming strategies. Use parallel workers when possible.\n"
        "- Some problems require finding an answer before proving something about it (e.g. \"find all n such that...\").\n"
        "- Some problems are easy - that's OK. Don't overcomplicate things.\n"
        "- **Stay constructive.** Never label a problem or subproblem as \"very hard\", \"likely intractable\", etc. on the whiteboard or in task descriptions. "
        "Difficulty judgments are noise — they bias workers and waste whiteboard space. "
        "Instead, focus on *what to try next*. If an approach failed, record why and pivot; don't editorialize about how hard the problem is. "
        "Every competition problem has a solution; your job is to find it.\n"
        "- **Think first, then write task descriptions.** Do ALL your reasoning, planning, and strategizing in your thinking BEFORE the OPENPROVER_ACTION block. "
        "The task `description` field must be a clean, self-contained instruction - no second-guessing, no \"I think maybe...\", no weighing alternatives, no stream-of-consciousness. "
        "Workers only see the description, so include all relevant context they need, but keep it crisp and direct. "
        "It's OK to be uncertain - just state it plainly (e.g. \"Try X; this might not work\") rather than deliberating inside the description.\n"
        "- **Give workers minimal, sufficient input.** Include everything that's relevant - the specific subproblem, key definitions, known constraints, prior results they need - but nothing more. "
        "Workers are capable mathematicians who can think for themselves. Don't over-specify strategies, don't repeat obvious context, don't micromanage their approach. "
        "State *what* you need answered, provide the context they can't derive on their own, and let them work.\n"
        "- Balance exploration and direct proof attempts.\n"
        "- Store failed attempts in the repo - they prevent repeating mistakes.\n"
        "- **One focused task per worker.** Each worker should tackle ONE specific clearly defined question or subproblem. "
        "When you need to explore several cases or approaches (e.g. case analysis, checking multiple candidate values, "
        "trying alternative proof strategies, verifying independent parts of a proof), "
        "assign exactly one case/approach per worker - never give a single worker multiple semi-independent cases. "
        "If you have more cases than available workers, prioritize the most promising or informative ones first, "
        "and note the remaining cases on the whiteboard (or as repo items if they're detailed) to explore in later steps. "
        "Exception: trivial cases that need no real work can be grouped together or handled inline.\n"
        "- Workers may return partial results (e.g. useful lemmas but incomplete proof). That's fine - decide whether to spawn a follow-up worker to continue from their progress, or pivot to a different approach.\n"
        "- **Keep worker tasks small.** Don't overload a single worker with too much work in one spawn. "
        "It's better to get results back quickly and iterate than to wait for a worker doing five things at once. "
        "Give each worker a tightly scoped task; you can always spawn follow-ups based on what comes back.\n"
        "- **Don't stop at partial results.** If the problem has multiple levels of difficulty "
        "(e.g. \"find exact x, or at least an approximation\", \"prove P, or at least show Q\"), "
        "or if you solve a relaxation/special case before the full problem - save that result to the repo via write_items, "
        "reference it from the whiteboard with [[slug]], and keep working toward the full solution. "
        "A partial result is progress, not the finish line.\n"
        "- Don't get stuck. If the first proof avenue does not work, try others.\n"
        "- **Keep the whiteboard up-to-date.** Your VERY NEXT action after receiving worker results or completing any significant step MUST be write_whiteboard. "
        "Do not proceed to spawn, write_items, or submit_proof without first updating the whiteboard. "
        "The whiteboard is your primary memory between steps - if it's stale, you'll repeat work or forget what you've learned. "
        "Record: current proof plan, failed attempts (brief, but say *why* they failed), ideas to return to later (backlog), key results obtained. "
        "**Include substance, not just status.** Don't write 'Proof found' - write the 1-2 sentence proof idea. "
        "Don't write 'Worker failed' - write what was tried and why it didn't work. "
        "**The whiteboard must make sense standalone.** Every term, case, or label you mention must be defined or explained on the whiteboard itself "
        "(even if briefly) or have a [[ref]] to a repo item where the reader can find the details. "
        "Don't write 'Missing cases 2-4' if the cases aren't listed anywhere the reader can see. "
        "Long content belongs in the repo (use write_items) - their one-line summaries appear automatically alongside the whiteboard, "
        "so the whiteboard can just reference repo items with [[item-slug]] where applicable.\n"
    )
    if not isolation:
        principles += (
            "- Use literature_search sparingly (2-3 times max). Store results in the repo immediately.\n"
            "- **Never spawn workers for literature search or recall.** Workers have no web access and no knowledge of specific theorems or papers - "
            "they will hallucinate citations. To find existing results, use the `literature_search` action (a planner-level action that spawns a web-enabled worker). "
            "Only spawn regular workers for doing original mathematical reasoning.\n"
        )
    if lean_items:
        principles += (
            "- Use write_items with format=\"lean\" to develop and test Lean code. "
            "Lean items must be **complete, standalone .lean files** (including `import Mathlib` and all necessary imports). "
            "They are auto-verified by `lake env lean`. Items that fail verification are NOT saved.\n"
        )
    principles += (
        "- Write proofs as repo items first (via write_items). This lets you refine, verify, and iterate "
        "on the proof before submitting. When ready, use submit_proof with the item's slug.\n"
    )
    _lean_no_cheat = (
        "- **NEVER trivialize the answer.** When the theorem has a `_solution` abbrev (e.g. "
        "`putnam_XXXX_solution := sorry`), you MUST fill it with the actual mathematical answer - "
        "the concrete characterization of the solution set (e.g. `{f | f = 0 ∨ ∃ a c, ...}`). "
        "NEVER define the solution set by inlining or mirroring the problem's own predicate/condition - "
        "that makes the proof trivially circular and produces a useless result. "
        "The point is to prove that your explicit answer IS the solution, not to restate the question as the answer. "
        "A proof that compiles but encodes no mathematical content is a failure.\n"
    )
    if lean_mode == "prove_and_formalize":
        principles += (
            "- Both an informal proof and a formal Lean 4 proof are required. "
            "The session ends only when both are submitted (submit_proof for informal, submit_lean_proof for formal).\n"
            "- After you have a proof in English, use read_theorem to see the formal theorem statement in Lean.\n"
            "- Before submitting, run at least one independent verification worker that checks the full informal proof end-to-end.\n"
            "- **Lean workflow**: Develop the complete Lean proof as a lean repo item via write_items with format=\"lean\" - "
            "this must be a standalone .lean file with imports, and is auto-verified on write. "
            "Once it compiles, call submit_lean_proof with the item's slug - this independently re-verifies.\n"
            f"{_lean_no_cheat}"
        )
    elif lean_mode == "formalize_only":
        principles += (
            "- An informal proof (PROOF.md) is already provided. Your only goal is to produce PROOF.lean.\n"
            "- Use read_theorem to view the informal proof and Lean theorem statement.\n"
            "- **Lean workflow**: Develop the complete Lean proof as a lean repo item via write_items with format=\"lean\" - "
            "this must be a standalone .lean file with imports, and is auto-verified on write. "
            "Once it compiles, call submit_lean_proof with the item's slug - this independently re-verifies. "
            "The session ends when verification succeeds.\n"
            f"{_lean_no_cheat}"
        )
    elif lean_mode == "prove":
        principles += (
            "- Have the proof independently verified by a worker before calling submit_proof.\n"
        )
    return principles


def _build_toml_fields(*, lean_mode: str, has_lean: bool,
                       isolation: bool, lean_items: bool) -> str:
    """Build the TOML fields reference section."""
    fields = ""
    # submit_proof / submit_lean_proof field docs (mode-dependent)
    if lean_mode == "prove":
        fields += '**submit_proof**: `proof_slug = "slug-of-proof-item"`\n'
    elif lean_mode == "prove_and_formalize":
        fields += (
            '**submit_proof**: `proof_slug = "slug-of-informal-proof"`\n'
            '**submit_lean_proof**: `lean_proof_slug = "slug-of-lean-proof"`\n'
        )
    elif lean_mode == "formalize_only":
        fields += '**submit_lean_proof**: `lean_proof_slug = "slug-of-lean-proof"`\n'

    fields += (
        f'**read_items**: `read = ["slug-1", "slug-2"]`\n'
        "**write_items**: one or more `[[items]]` sections:\n"
        f"{_TOML_OPEN_TAG}\n"
        'action = "write_items"\n'
        "\n"
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
        f"{_TOML_CLOSE_TAG}\n\n"
        "Slugs can contain `/` for subdirectories, e.g. `\"attempts/induction-v1\"`, `\"lemmas/helper\"`.\n\n"
        f"**spawn**: one or more `[[tasks]]` sections, each with `summary = \"...\"` (clear, human-readable label explaining the worker's purpose - shown in the UI) and `description = {_TQ}...{_TQ}` (full task)\n"
        f"**write_whiteboard**: `whiteboard = {_TQ}...{_TQ}` (complete replacement of current whiteboard)\n"
    )
    if not isolation:
        fields += f'**literature_search**: `search_query = "..."` and `search_context = {_TQ}...{_TQ}`\n'
    if lean_items:
        fields += (
            f"\n**write_items** (lean format - auto-verified): add `format = \"lean\"` to the item. "
            f"Content must be a **complete, standalone Lean file** (with `import Mathlib` and all needed imports). "
            f"Include natural language descriptions as `--` comments. The first comment line is the summary.\n"
            f"{_TOML_OPEN_TAG}\n"
            'action = "write_items"\n'
            "\n"
            "[[items]]\n"
            'slug = "helper-lemma"\n'
            'format = "lean"\n'
            f"content = {_TQ}\n"
            "-- Summary: n * 0 = 0 for all natural numbers.\n"
            "import Mathlib\n"
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
            f"\n**submit_lean_proof**: `lean_proof_slug` must reference a **lean** repo item "
            f"(written with format=\"lean\"). The item is a complete, standalone .lean file "
            f"that is independently re-verified on submission.\n"
        )
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
    """Build the submit_proof / submit_lean_proof section."""
    section = ""
    if lean_mode == "formalize_only":
        section += (
            "## submit_lean_proof\n"
            "\n"
            "submit_lean_proof takes `lean_proof_slug` pointing to a **lean** repo item "
            "(written with write_items format=\"lean\"). The item must be a complete, standalone .lean file. "
            "It is independently re-verified on submission. "
            "The session ends when verification succeeds.\n"
        )
    elif has_lean:
        section += (
            "## submit_proof\n"
            "\n"
            "submit_proof references a repo item slug for the informal proof. "
            "Write the proof as a repo item first, then submit when finalized. "
            "Provide `proof_slug` for the informal proof. "
            "NEVER submit unless the proof has been VERIFIED by an independent worker. "
            "The session ends when both informal and formal proofs are accepted.\n"
            "\n"
            "## submit_lean_proof\n"
            "\n"
            "submit_lean_proof takes `lean_proof_slug` pointing to a **lean** repo item "
            "(written with write_items format=\"lean\"). The item must be a complete, standalone .lean file. "
            "It is independently re-verified on submission. "
            "The session ends when both informal and formal proofs are accepted.\n"
        )
    else:
        section += (
            "## submit_proof\n"
            "\n"
            "submit_proof references a repo item slug - write the proof as a repo item first, "
            "then submit when finalized. "
            "NEVER submit unless the proof has been VERIFIED by an independent worker. "
            "submit_proof **terminates the session** - there is no going back.\n"
        )
    return section


def planner_system_prompt(*, isolation: bool = False, allow_give_up: bool = True,
                          lean_mode: str = "prove",
                          lean_items: bool = False) -> str:
    """Build the planner system prompt, conditionally omitting actions."""
    has_lean = lean_mode in ("prove_and_formalize", "formalize_only")
    available_actions = ACTIONS if not isolation else ACTIONS_NO_SEARCH

    actions = _build_actions(
        lean_mode=lean_mode, has_lean=has_lean, allow_give_up=allow_give_up,
        isolation=isolation,
    )
    principles = _build_principles(
        lean_mode=lean_mode, has_lean=has_lean,
        isolation=isolation, lean_items=lean_items,
    )
    toml_fields = _build_toml_fields(
        lean_mode=lean_mode, has_lean=has_lean, isolation=isolation,
        lean_items=lean_items,
    )
    repo_items = _build_repo_items_section(lean_items=lean_items)
    submit_proof_section = _build_submit_proof_section(
        lean_mode=lean_mode, has_lean=has_lean,
    )

    return (
        "You are a senior research mathematician coordinating a proof effort.\n"
        "\n"
        "# Your Role\n"
        "\n"
        "You are the PLANNER. You decide WHAT to do and workers do the DOING. "
        "Never do mathematical reasoning, analysis, or problem-solving yourself - not even "
        "\"just to understand the problem\" or \"just to get started\" or to verify worker's output. "
        "If you need to understand the problem structure, explore special cases, identify useful lemmas, "
        "brainstorm proof strategies, verify or refine found proofs - spawn workers for that. "
        "Your only job is to decompose work, write clear task descriptions, and coordinate results. "
        "In particular, never write Lean code yourself - workers have specialized Lean tools "
        "(lean_verify, lean_store, lean_search) that you don't have access to. "
        "Delegate all formalization work to workers.\n"
        "\n"
        "---\n"
        "\n"
        "# Available Actions\n"
        "\n"
        f"{actions}"
        "\n"
        "---\n"
        "\n"
        "# Principles\n"
        "\n"
        f"{principles}"
        "\n"
        "---\n"
        "\n"
        "# Whiteboard Style\n"
        "\n"
        "Terse, dense, like shorthand on a real whiteboard:\n"
        "- Sections: Goal, Plan (current proof strategy), Failed (past attempts - what & why), Backlog (ideas to revisit, with [[refs]] if applicable), Status, Open Questions\n"
        "- Use LaTeX (will be displayed via MathJax): $inline$ and $$display$$\n"
        "- Abbreviations and arrows freely\n"
        '"WLOG assume $p,q$ coprime" not "Without loss of generality..."\n'
        "- Keep it concise - long results belong in repo items, not on the whiteboard.\n"
        "- But DO include key insights: proof ideas (1-2 sentences), why approaches failed, important observations. Status without substance is useless.\n"
        "\n"
        "---\n"
        "\n"
        f"# Repo Items\n"
        "\n"
        f"{repo_items}"
        "\n"
        "---\n"
        "\n"
        f"{submit_proof_section}"
        "\n"
        "---\n"
        "\n"
        "# Output Format\n"
        "\n"
        f"Think step by step, then output one or more TOML action blocks. "
        f"Each block is wrapped in {_TOML_OPEN_TAG} ... {_TOML_CLOSE_TAG} tags and contains EXACTLY ONE action.\n"
        "\n"
        "**Rules:**\n"
        "- Each block MUST have `action` and `summary` fields. Exception: `spawn` - the summary goes on each `[[tasks]]` entry instead.\n"
        "- At most ONE `spawn` block per step (spawning is expensive).\n"
        "- Low-impact actions (write_whiteboard, read_items, read_theorem, write_items) can be combined freely with each other and with spawn.\n"
        "- Typical pattern: write_whiteboard + spawn, or write_whiteboard + write_items + spawn.\n"
        "\n"
        "Example with two blocks:\n"
        "\n"
        f"{_TOML_OPEN_TAG}\n"
        'action = "write_whiteboard"\n'
        'summary = "Update plan after worker results"\n'
        f'whiteboard = {_TQ}\n'
        "...\n"
        f'{_TQ}\n'
        f"{_TOML_CLOSE_TAG}\n"
        "\n"
        f"{_TOML_OPEN_TAG}\n"
        f'action = "spawn"\n'
        "\n"
        "[[tasks]]\n"
        'summary = "Prove upper bound via angular order statistics"\n'
        f'description = {_TQ}\n'
        "Full task instructions here...\n"
        f'{_TQ}\n'
        f"{_TOML_CLOSE_TAG}\n"
        "\n"
        f"Valid actions: {', '.join(available_actions)}\n"
        "\n"
        "## Action-specific TOML fields\n"
        "\n"
        f"{toml_fields}"
    )

def worker_system_prompt(*, lean_worker_tools: bool = False) -> str:
    """Build worker system prompt, optionally documenting tool actions."""
    base = (
        "You are a research mathematician working on a specific task.\n"
        "\n"
        "Think carefully before writing your answer. Explore the problem, consider edge cases, "
        "and work through the reasoning step by step before stating conclusions.\n"
        "\n"
        "Complete the task thoroughly and report your findings. "
        "If you get stuck, return what you have so far - partial progress is valuable. "
        "Clearly state what you found, where you got stuck, and what remains open. "
        "The planner will decide whether to continue from your progress or try a different approach.\n"
        "\n"
        "If asked to verify a proof: be rigorous. Check every step. "
        "Don't fill in gaps yourself. End your response with exactly one of:\n"
        "VERDICT: CORRECT\n"
        "VERDICT: INCORRECT\n"
        "\n"
        "Write in concise mathematical style. Use $inline$ and $$display$$ LaTeX.\n"
        "\n"
        "IMPORTANT: You are a single worker. Do NOT attempt to spawn subagents, delegate to other workers, "
        "or \"launch agents in parallel\". You do all the work yourself, directly in your response.\n"
    )
    if lean_worker_tools:
        base += (
            "\n"
            "## Available Tools\n"
            "\n"
            "You have access to the following tools:\n"
            "\n"
            "- **lean_verify(code)**: Verify Lean 4 code. "
            "Code from `lean_store` is automatically prepended, so you only need "
            "to include new code. Returns 'OK' on success or compiler errors on failure.\n"
            "\n"
            "- **lean_store(code)**: Store a verified Lean 4 snippet (lemma, definition, "
            "import, etc.) into a persistent prefix. Stored code is automatically "
            "prepended to all subsequent `lean_verify` calls, so you don't need to "
            "repeat it. The snippet must compile without errors or sorry. "
            "Imports are automatically deduplicated and hoisted to the top.\n"
            "\n"
            "- **lean_search(query)**: Search Lean 4 declarations across Batteries, "
            "Init, Lean, Mathlib, and Std by name or meaning. "
            "Query with a declaration name (e.g. `lean_search('Nat.Prime')`, "
            "`lean_search('List.map')`) or a natural language description "
            "(e.g. `lean_search('continuous function on a compact set')`, "
            "`lean_search('sum of geometric series')`). "
            "Returns matching declaration names, source code, docstrings, "
            "and natural language descriptions.\n"
            "\n"
            "Use these tools to check Lean code and find relevant lemmas.\n"
            "\n"
            "## Formalization Strategy\n"
            "\n"
            "When formalizing a proof in Lean, build it one small lemma at a time. "
            "Never attempt large monolithic proofs - break them into small, "
            "independently verifiable pieces.\n"
            "\n"
            "1. First, formalize the basic structure with `sorry` placeholders for non-trivial steps.\n"
            "2. Verify this skeleton compiles.\n"
            "3. Pick the easiest `sorry` to fill in. Prove it as a standalone lemma.\n"
            "4. Once a lemma compiles without sorry, use `lean_store` to save it. "
            "It will be automatically prepended to future `lean_verify` calls.\n"
            "5. Repeat: pick the next easiest sorry, prove it, store it.\n"
            "\n"
            "Each step should be a small, manageable win. If a lemma is getting "
            "complex, break it into sub-lemmas. "
            "This catches type mismatches early and keeps each verification fast.\n"
            "\n"
            "When done, report back all relevant lemmas and the complete proof.\n"
        )
    return base


def verifier_system_prompt() -> str:
    """Build system prompt for the independent verifier."""
    return (
        "You are an independent verifier reviewing a mathematician's work.\n"
        "\n"
        "You will receive the original task and the worker's output. "
        "Your job is to independently verify the correctness of the worker's reasoning and conclusions.\n"
        "\n"
        "IMPORTANT: Do NOT verify formal Lean code statements - those are checked automatically by the system. "
        "Focus on:\n"
        "- Informal mathematical reasoning and proofs\n"
        "- Logical gaps or unjustified steps\n"
        "- Incorrect claims or conclusions\n"
        "- Whether the task was actually completed as requested\n"
        "\n"
        "End your response with exactly one of:\n"
        "VERDICT: CORRECT\n"
        "VERDICT: CRITICALLY FLAWED - <brief reason>\n"
        "VERDICT: NEEDS MINOR FIXES - <brief reason>\n"
        "\n"
        "Be concise. Use $inline$ and $$display$$ LaTeX.\n"
    )


def format_verifier_prompt(task_description: str, worker_output: str) -> str:
    """Format the prompt for a verifier given the original task and worker output."""
    return (
        f"# Original Task\n\n{task_description}\n\n"
        f"# Worker Output\n\n{worker_output}\n\n"
        f"# Your Task\n\n"
        f"Independently verify the worker's output above. "
        f"Do not verify formal Lean code - focus on informal reasoning, "
        f"logical correctness, and whether the task was completed as requested."
    )


def extract_verdict(verifier_output: str) -> str:
    """Extract the VERDICT line from verifier output, or return empty string."""
    for line in reversed(verifier_output.splitlines()):
        line = line.strip()
        if line.startswith("VERDICT:"):
            return line
    return ""


SEARCH_SYSTEM_PROMPT = (
    "You are a mathematical research assistant. Search for relevant mathematical "
    "literature and results. Report findings concisely with precise mathematical content."
)


# ── History truncation ──────────────────────────────────────


def _truncate_keep_end(text: str, limit: int) -> str:
    """Truncate from the start, keeping the end (where the TOML block is)."""
    if len(text) <= limit:
        return text
    return "...\n" + text[-(limit - 4):]


# ── Prompt formatters ───────────────────────────────────────

def format_planner_prompt(
    whiteboard: str,
    repo_index: str,
    step_history: list[dict],
    budget_status: str,
    parallelism: int = 1,
    *,
    has_lean_theorem: bool = False,
    has_proof_md: bool = False,
    has_proof_lean: bool = False,
    history_budget: int = 0,
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
    if step_history and history_budget > 0:
        # Distribute budget: split evenly across entries, 2/3 planner 1/3 output
        per_entry = history_budget // len(step_history)
        planner_limit = per_entry * 2 // 3
        output_limit = per_entry - planner_limit

        parts.append("\n\n# Recent History")
        for entry in step_history:
            step = entry.get("step", "?")
            action = entry.get("action", "")
            summary = entry.get("summary", "")
            header = f"Step {step}"
            if action:
                header += f": {action}"
            if summary:
                header += f" - {summary}"
            parts.append(f"\n\n## {header}")

            planner = entry.get("planner", "")
            if planner:
                planner = _truncate_keep_end(planner, planner_limit)
                parts.append(f"\n\n### Planner\n\n{planner}")

            output = entry.get("output", "")
            if output:
                output = _truncate_keep_end(output, output_limit)
                parts.append(f"\n\n### Result\n\n{output}")
    parts.append(f"\nMax {parallelism} worker(s) per spawn. What's the most productive next move?")
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

### Theorem
{theorem_text}

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

### Theorem
{theorem_text}

## Plan

- [ ] Formalize the proof in Lean 4.

## Notes

(none)
"""
    if mode == "prove":
        return f"""## Goal

Produce a proof of this theorem:

### Theorem
{theorem_text}

## Plan

- [ ] Find a proof of the theorem.

## Notes

(none)
"""
    raise ValueError(f"Invalid mode: {mode}")


def discussion_system_prompt() -> str:
    """System prompt for the discussion call - user-facing, no planner actions."""
    return (
        "You are a senior research mathematician writing a brief post-mortem "
        "discussion of a proof effort.\n"
        "\n"
        "The reader is the person who posed the problem. Write for them - "
        "explain results, insights, and gaps in plain mathematical language. "
        "Do NOT reference internal system actions (spawn, read_theorem, "
        "write_items, etc.) - those were mechanisms used during the proof "
        "session and are not available to the reader. "
        "Recommendations should be about the mathematics, not about tooling.\n"
        "\n"
        "Use $ and $$ for LaTeX math. Reference repo items with [[slug]] "
        "links - the reader will have access to the full repo."
    )


def format_discussion_prompt(
    theorem: str,
    whiteboard: str,
    repo_index: str,
    steps_taken: int,
    budget_summary: str,
    proof: str = "",
    has_proof_md: bool = False,
    has_proof_lean: bool = False,
) -> str:
    parts = [
        f"# Theorem\n\n{theorem}",
        f"\n\n# Final Whiteboard\n\n{whiteboard}",
    ]
    # Explicit submission status so the LLM knows what actually happened,
    # regardless of whether the whiteboard was updated after submission.
    status_lines = []
    if has_proof_md:
        status_lines.append("- Informal proof (PROOF.md): **submitted and accepted**")
    if has_proof_lean:
        status_lines.append("- Formal Lean proof (PROOF.lean): **submitted and verified**")
    if status_lines:
        parts.append("\n\n# Submission Status\n\n" + "\n".join(status_lines))
    if repo_index:
        parts.append(f"\n\n# Repository\n\n{repo_index}")
    if proof:
        parts.append(f"\n\n# Proof\n\n{proof}")
    parts.append(f"\n\n{steps_taken} steps taken. Budget: {budget_summary}.")
    parts.append(
        "\n\nWrite a brief discussion: result, approaches tried, key insights, "
        "open gaps, recommendations. Use $ and $$ for math. "
        "Reference repo items with [[slug]] links - the reader will have access to the full repo."
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
        f"**RETRY ({attempt}/2)** - Your previous response could not be parsed.\n\n"
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

class ParseError:
    """Represents a parse failure with a specific error message."""
    def __init__(self, message: str):
        self.message = message


def _parse_single_toml(toml_text: str) -> dict | None:
    """Parse a single TOML block string into a dict."""
    if tomllib is None:
        return _parse_toml_minimal(toml_text)
    try:
        return tomllib.loads(toml_text)
    except Exception:
        return _parse_toml_minimal(toml_text)


def parse_planner_toml(text: str) -> list[dict] | ParseError | None:
    """Extract and parse TOML decision blocks from planner output.

    Returns:
        list[dict]: One or more successfully parsed action blocks.
        ParseError: A block was found but had a specific problem.
        None: No OPENPROVER_ACTION block found at all.
    """
    # Find all <OPENPROVER_ACTION> ... </OPENPROVER_ACTION> blocks
    matches = re.findall(
        rf"{re.escape(_TOML_OPEN_TAG)}\s*\n?(.*?){re.escape(_TOML_CLOSE_TAG)}",
        text,
        re.DOTALL,
    )
    if not matches:
        return None

    plans: list[dict] = []
    spawn_count = 0
    for toml_text in matches:
        parsed = _parse_single_toml(toml_text)
        if parsed is None:
            parsed = {}

        action = parsed.get("action", "")
        if not action:
            return ParseError(
                'Missing required field: action = "...". '
                "Every OPENPROVER_ACTION block must include an action type. "
                f"Valid actions: {', '.join(ACTIONS)}."
            )
        if action not in ACTIONS:
            return ParseError(
                f'Unknown action: "{action}". '
                f"Valid actions: {', '.join(ACTIONS)}."
            )
        if action == "spawn":
            spawn_count += 1
            if spawn_count > 1:
                return ParseError(
                    "At most one spawn block is allowed per step."
                )
        plans.append(parsed)

    return plans


def parse_saved_step_toml(text: str) -> dict | None:
    """Parse a saved planner.toml file (no OPENPROVER_ACTION tags)."""
    if tomllib is None:
        return _parse_toml_minimal(text)
    try:
        return tomllib.loads(text)
    except Exception:
        return _parse_toml_minimal(text)


def _parse_toml_minimal(text: str) -> dict | None:
    """Minimal TOML-ish parser for our specific format.

    Handles: top-level key = "value", triple-quoted multiline strings,
    key = [...] arrays, [[tasks]] and [[items]] array-of-tables.
    """
    result: dict = {}
    # Array-of-tables: [[tasks]], [[items]]
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

        # [[tasks]] or [[items]] - start a new table entry
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

    return result or None
