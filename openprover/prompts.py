"""Prompt templates for OpenProver."""

ACTIONS = [
    "continue", "explore_avenue", "prove_lemma", "verify",
    "check_counterexample", "literature_search", "replan",
    "declare_proof", "declare_stuck",
]

SYSTEM_PROMPT = r"""
You are a senior research mathematician. You prove theorems.

Your workspace is a WHITEBOARD — terse, dense, like a real whiteboard. Update it every step.

## Principles

1. KEEP IT SIMPLE. If a step is trivial, just do it inline. Don't create lemmas for trivial facts.
2. Only extract a lemma when it's genuinely reusable or the proof is non-trivial.
3. Before diving into a proof attempt, think about whether the approach can actually work. Catch doomed avenues early.
4. Try small cases and examples first. Build intuition before formalism.
5. When stuck, consider: can you reduce to a simpler subproblem, or generalize to a setting where the result becomes natural? Changing perspective beats pushing harder on a stuck approach.
6. Understand *why* things work, not just *that* they work. If you can't explain the proof idea in one sentence, you don't understand it yet.
7. Be honest about gaps. A proof sketch with a clearly marked gap is worth more than a hand-wavy "proof." Mark confidence: [high], [med], [low].
8. Consider whether the statement might be false. A quick counterexample search can save hours.
9. When the proof works, write it up cleanly and declare it. Don't keep going.
10. After a literature search, store truly relevant known results as lemmas using the LEMMA section. Cite the source. These become available tools for your proof.

## CRITICAL: declare_proof rules

NEVER use declare_proof unless you have a COMPLETE, RIGOROUS proof with NO gaps. A proof means: every logical step follows from the previous one, all the way from hypotheses to conclusion. The following are NOT proofs:
- Summarizing what is known about a problem
- Citing existing results without connecting them into a complete argument
- A proof sketch with gaps marked "this can be shown" or "it is known that"
- Describing an approach without carrying it out

If the theorem is hard or open, THAT IS YOUR JOB. You are here to ATTEMPT PROOFS, not to report on the state of human knowledge. The fact that a problem is famous or open is IRRELEVANT — you must try anyway. Try novel approaches, combine known techniques in new ways, explore unconventional angles. Investigate special cases, find partial results, build toward a proof.

## CRITICAL: declare_stuck rules

NEVER use declare_stuck early. You must use nearly all your allotted steps before even considering it. "This problem is open" or "this is a famous unsolved problem" is NEVER a valid reason to declare_stuck. Instead:
- Try a different approach
- Investigate special cases or weaker versions
- Look for novel angles no one has considered
- Build partial results that constrain the solution
- Replan from scratch with a completely different strategy

## Whiteboard Style

Write like shorthand on a real whiteboard — abbreviated, dense, no fluff:
- "WLOG assume $p,q$ coprime" not "Without loss of generality, we may assume that p and q are coprime integers"
- "$p^2=2q^2 \Rightarrow p$ even $\Rightarrow p=2k \Rightarrow 4k^2=2q^2 \Rightarrow q$ even. Contradiction." not paragraphs explaining each step
- Use LaTeX for all math: $inline$ and $$display$$
- Use arrows, abbreviations freely
- Sections: Goal, Plan, Work, Observations, Tried (keep failed approaches as one-liners)

You respond in JSON as specified.
"""

# Used when literature_search is unavailable (--isolation mode)
ACTIONS_NO_SEARCH = [a for a in ACTIONS if a != "literature_search"]

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ACTIONS,
        },
        "summary": {
            "type": "string",
            "description": "One-line summary of what you plan to do.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief: why this is the most productive next step.",
        },
        "target": {
            "type": "string",
            "description": "For prove_lemma: lemma name. For verify: what to verify. For literature_search: search query. Otherwise empty.",
        },
    },
    "required": ["action", "summary", "reasoning"],
    "additionalProperties": False,
}

# Step output uses text sections instead of JSON schema (more reliable for long math)
STEP_OUTPUT_FORMAT = """
At the END of your response, output the following sections (these are REQUIRED):

ACTION: <one of: {actions}>
SUMMARY: <one-line summary of what you did>
WHITEBOARD:
<COMPLETE updated whiteboard — replaces previous. Keep terse and dense.>
END_WHITEBOARD

Optional sections (only if applicable):
PROOF:
<If action=declare_proof: the COMPLETE, RIGOROUS proof. Not a sketch. A real proof.>
END_PROOF
VERIFY_TARGET: <what to verify>
VERIFY_CONTENT:
<self-contained statement+proof to check>
END_VERIFY_CONTENT
SEARCH_QUERY: <what to search for>
LEMMA_NAME: <name of lemma to store>
LEMMA_SOURCE: <citation/URL if from literature, omit if derived>
LEMMA_CONTENT:
<precise statement and proof of the lemma>
END_LEMMA_CONTENT
"""


def _make_plan_schema(actions: list) -> dict:
    """Return plan_schema with the given action list."""
    import copy
    ps = copy.deepcopy(PLAN_SCHEMA)
    ps["properties"]["action"]["enum"] = actions
    return ps


def step_output_instructions(actions: list) -> str:
    """Return the step output format instructions with the given action list."""
    return STEP_OUTPUT_FORMAT.format(actions=", ".join(actions))


def format_plan_prompt(
    theorem: str,
    whiteboard: str,
    lemma_index: str,
    step_num: int,
    max_steps: int,
    human_feedback: str = "",
    verification_result: str = "",
    search_result: str = "",
) -> str:
    parts = [
        f"# Theorem\n\n{theorem}",
        f"\n\n# Whiteboard\n\n{whiteboard}",
    ]
    if lemma_index:
        parts.append(f"\n\n# Lemmas\n\n{lemma_index}")
    if verification_result:
        parts.append(f"\n\n# Verification Result\n\n{verification_result}")
    if search_result:
        parts.append(f"\n\n# Literature Search Result\n\n{search_result}")
    if human_feedback:
        parts.append(f"\n\n# Human Feedback\n\n{human_feedback}")
    parts.append(f"\n\nStep {step_num}/{max_steps}. What's the most productive next move?")
    return "".join(parts)


def format_step_prompt(
    theorem: str,
    whiteboard: str,
    lemma_index: str,
    step_num: int,
    max_steps: int,
    actions: list,
    plan: dict | None = None,
    human_feedback: str = "",
    verification_result: str = "",
    search_result: str = "",
) -> str:
    parts = [
        f"# Theorem\n\n{theorem}",
        f"\n\n# Whiteboard\n\n{whiteboard}",
    ]
    if lemma_index:
        parts.append(f"\n\n# Lemmas\n\n{lemma_index}")
    if verification_result:
        parts.append(f"\n\n# Verification Result\n\n{verification_result}")
    if search_result:
        parts.append(f"\n\n# Literature Search Result\n\n{search_result}")
    if human_feedback:
        parts.append(f"\n\n# Human Feedback\n\n{human_feedback}")
    parts.append(f"\n\nStep {step_num}/{max_steps}.")
    if plan:
        parts.append(
            f" Plan: {plan['action']} — {plan['summary']}\n\nExecute this plan. Update the whiteboard with your work."
        )
    else:
        parts.append(
            " Decide and execute the best next step. Update the whiteboard."
        )
    parts.append(
        "\n\nKeep the whiteboard terse. Do NOT declare_stuck or declare_proof unless you have genuinely exhausted approaches or have a complete proof."
    )
    parts.append(step_output_instructions(actions))
    return "".join(parts)


def format_literature_search_prompt(query: str, theorem: str) -> str:
    return (
        "# Literature Search\n\n"
        f"We are working on proving: {theorem.strip().split(chr(10))[0]}\n\n"
        f"Search the web for: {query}\n\n"
        "Find relevant theorems, proof techniques, known results, or partial progress. "
        "Report back concisely: what's known, what techniques are used, any useful references. "
        "Focus on mathematical content, not summaries of summaries."
    )

LITERATURE_SEARCH_SYSTEM_PROMPT = (
    "You are a mathematical research assistant. Search for relevant mathematical "
    "literature and results. Report findings concisely with precise mathematical content."
)


def format_verify_prompt(statement: str, proof: str) -> str:
    return (
        "# Independent Verification\n\n"
        "You have NOT seen the reasoning that produced this proof. "
        "Check it cold.\n\n"
        f"## Statement\n\n{statement}\n\n"
        f"## Proof\n\n{proof}\n\n"
        "Is this proof correct? Be specific about any gaps or errors.\n\n"
        "End your response with exactly one of these verdicts on its own line:\n"
        "VERDICT: CORRECT\n"
        "VERDICT: INCORRECT"
    )

VERIFY_SYSTEM_PROMPT = (
    "You are a rigorous proof checker. Be skeptical. "
    "Flag gaps, errors, unjustified steps. Don't fill in gaps yourself."
)


def format_discussion_prompt(
    theorem: str,
    whiteboard: str,
    lemma_index: str,
    steps_taken: int,
    max_steps: int,
    proof: str = "",
) -> str:
    parts = [
        f"# Theorem\n\n{theorem}",
        f"\n\n# Final Whiteboard\n\n{whiteboard}",
    ]
    if lemma_index:
        parts.append(f"\n\n# Lemmas\n\n{lemma_index}")
    if proof:
        parts.append(f"\n\n# Proof\n\n{proof}")
    parts.append(f"\n\n{steps_taken}/{max_steps} steps used.")
    parts.append(
        "\n\nWrite a brief discussion: result, approaches tried, key insights, "
        "open gaps, recommendations. Use $ and $$ for math."
    )
    return "".join(parts)


def format_summary_prompt(
    theorem: str,
    whiteboard: str,
    lemma_index: str,
    step_num: int,
    max_steps: int,
) -> str:
    parts = [
        f"# Theorem\n\n{theorem}",
        f"\n\n# Whiteboard (step {step_num}/{max_steps})\n\n{whiteboard}",
    ]
    if lemma_index:
        parts.append(f"\n\n# Lemmas\n\n{lemma_index}")
    parts.append(
        "\n\nBriefly summarize progress: what approaches have been tried, "
        "what's working, what's the current plan. 3-5 sentences max."
    )
    return "".join(parts)


def parse_step_output(text: str) -> dict | None:
    """Parse the section-based step output format into a dict."""
    result = {}

    # Extract ACTION
    for line in text.splitlines():
        if line.strip().startswith("ACTION:"):
            result["action"] = line.split(":", 1)[1].strip().lower()
            break

    # Extract SUMMARY
    for line in text.splitlines():
        if line.strip().startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()
            break

    # Extract WHITEBOARD
    result["whiteboard"] = _extract_section(text, "WHITEBOARD:", "END_WHITEBOARD")

    # Extract optional PROOF
    proof = _extract_section(text, "PROOF:", "END_PROOF")
    if proof:
        result["proof"] = proof

    # Extract optional VERIFY fields
    for line in text.splitlines():
        if line.strip().startswith("VERIFY_TARGET:"):
            result["verify_target"] = line.split(":", 1)[1].strip()
            break
    verify_content = _extract_section(text, "VERIFY_CONTENT:", "END_VERIFY_CONTENT")
    if verify_content:
        result["verify_content"] = verify_content

    # Extract optional SEARCH_QUERY
    for line in text.splitlines():
        if line.strip().startswith("SEARCH_QUERY:"):
            result["search_query"] = line.split(":", 1)[1].strip()
            break

    # Extract optional LEMMA fields
    for line in text.splitlines():
        if line.strip().startswith("LEMMA_NAME:"):
            result["lemma_name"] = line.split(":", 1)[1].strip()
            break
    for line in text.splitlines():
        if line.strip().startswith("LEMMA_SOURCE:"):
            result["lemma_source"] = line.split(":", 1)[1].strip()
            break
    lemma_content = _extract_section(text, "LEMMA_CONTENT:", "END_LEMMA_CONTENT")
    if lemma_content:
        result["lemma_content"] = lemma_content

    if "action" not in result or "whiteboard" not in result:
        return None
    result.setdefault("summary", "")
    return result


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """Extract content between start_marker line and end_marker line."""
    lines = text.splitlines()
    capturing = False
    content = []
    for line in lines:
        if line.strip() == end_marker:
            capturing = False
            continue
        if capturing:
            content.append(line)
        if line.strip().startswith(start_marker) and not capturing:
            # If there's content on the same line as the marker, grab it
            rest = line.strip()[len(start_marker):].strip()
            if rest:
                content.append(rest)
            capturing = True
    return "\n".join(content).strip()


def format_initial_whiteboard(theorem: str) -> str:
    first_line = theorem.strip().split("\n")[0]
    return (
        f"## Goal\n\n{first_line}\n\n"
        "## Plan\n\nTBD — analyze first.\n\n"
        "## Work\n\n(start)\n\n"
        "## Tried\n\n(none)\n"
    )
