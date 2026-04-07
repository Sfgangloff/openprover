#!/usr/bin/env python3
"""Baseline prover: single conversation with lean_verify tool calling.

The model receives the theorem statement and can call lean_verify as many
times as it wants within the time budget.  No planner/worker decomposition,
no repository, no whiteboard — just raw model + verifier.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

from openprover.lean.core import LeanWorkDir, lean_has_errors, run_lean_check, strip_code_fences

logger = logging.getLogger("baseline")

MISTRAL_BASE_URL = "https://api.mistral.ai"
MISTRAL_MODEL_MAP = {"leanstral": "labs-leanstral-2603"}

LEAN_VERIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "lean_verify",
        "description": "Verify Lean 4 code. Returns compiler output (errors/warnings or OK).",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Lean 4 source code to verify.",
                },
            },
            "required": ["code"],
        },
    },
}


def _parse_duration(s: str) -> float:
    """Parse a duration string like '10m', '2h', '1h30m' into seconds."""
    total = 0.0
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*([hms])", s):
        val, unit = float(match.group(1)), match.group(2)
        total += val * {"h": 3600, "m": 60, "s": 1}[unit]
    if total == 0:
        total = float(s)
    return total


def _lean_verify(code: str, lean_work_dir: LeanWorkDir, lean_project_dir: Path) -> str:
    """Run lean_verify and return the result string."""
    code = strip_code_fences(code)
    if not code:
        return "Error: no code provided."
    path = lean_work_dir.make_file("baseline_verify", code)
    success, feedback, _ = run_lean_check(path, lean_project_dir)
    if success:
        return "OK"
    if lean_has_errors(feedback):
        return feedback
    if "sorry" in feedback.lower():
        return feedback + "\n\nNote: code contains sorry — proof has gaps."
    return feedback


def _mistral_request(payload: dict, api_key: str,
                     conversation_id: str | None = None,
                     timeout: int = 600) -> dict:
    """Make a request to the Mistral Conversations API."""
    import urllib.request
    import urllib.error

    url = f"{MISTRAL_BASE_URL}/v1/conversations"
    if conversation_id:
        url = f"{url}/{conversation_id}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _parse_assistant_output(raw: dict) -> dict:
    """Extract text, thinking, and tool_calls from Mistral response.

    The Conversations API returns tool calls as separate output entries
    with type="function.call", not inside the assistant message.
    """
    result_text = ""
    thinking_text = ""
    tool_calls = []

    for entry in raw.get("outputs", []):
        etype = entry.get("type", "")

        # Tool calls are separate entries
        if etype == "function.call":
            tool_calls.append({
                "id": entry.get("id", ""),
                "tool_call_id": entry.get("tool_call_id", ""),
                "name": entry.get("name", ""),
                "arguments": entry.get("arguments", "{}"),
            })
            continue

        if entry.get("role") != "assistant":
            continue

        content = entry.get("content", "")
        if isinstance(content, str):
            result_text = content
        elif isinstance(content, list):
            thinking_parts = []
            output_parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "thinking":
                    for tp in part.get("thinking", []):
                        thinking_parts.append(tp.get("text", ""))
                else:
                    for cp in part.get("content", []):
                        output_parts.append(cp.get("text", ""))
                    if not output_parts and part.get("text"):
                        output_parts.append(part["text"])
            result_text = "".join(output_parts)
            thinking_text = "".join(thinking_parts)
        reasoning = entry.get("reasoning", "")
        if reasoning and not thinking_text:
            thinking_text = reasoning

        # Some responses have tool_calls on the assistant entry too
        tc = entry.get("tool_calls")
        if tc:
            for t in tc:
                tool_calls.append({
                    "id": t.get("id", ""),
                    "tool_call_id": t.get("tool_call_id", t.get("id", "")),
                    "name": t.get("name", t.get("function", {}).get("name", "")),
                    "arguments": t.get("arguments", t.get("function", {}).get("arguments", "{}")),
                })

    return {
        "text": result_text,
        "thinking": thinking_text,
        "tool_calls": tool_calls or None,
        "conversation_id": raw.get("conversation_id"),
    }


def run_baseline(
    name: str,
    theorem_lean: str,
    theorem_informal: str,
    lean_project_dir: Path,
    model: str,
    max_time: float,
    run_dir: Path,
    verbose: bool = False,
) -> dict:
    """Run the baseline prover on a single problem.

    Returns {"status": "proved"|"not_proved"|"error", "elapsed": float,
             "turns": int, "verifications": int, "error": str}.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        return {"status": "error", "elapsed": 0, "turns": 0,
                "verifications": 0, "error": "MISTRAL_API_KEY not set"}

    mistral_model = MISTRAL_MODEL_MAP.get(model, model)
    lean_work_dir = LeanWorkDir(lean_project_dir)

    system_prompt = (
        "You are a Lean 4 theorem prover. Your goal is to prove the following theorem.\n\n"
        f"## Informal statement\n{theorem_informal}\n\n"
        f"## Formal statement (Lean 4)\n```lean\n{theorem_lean}\n```\n\n"
        "The theorem file already has these imports:\n"
        "```\nimport MiniF2F.ProblemImports\n"
        "open scoped Real Nat Topology Polynomial\n```\n\n"
        "Use the `lean_verify` tool to check your proof attempts. "
        "Your code should include the full theorem statement with your proof replacing `sorry`. "
        "Keep trying different approaches if verification fails. "
        "When verification succeeds (returns OK), you're done."
    )

    # Save theorem
    (run_dir / "THEOREM.lean").write_text(theorem_lean + "\n")
    if theorem_informal:
        (run_dir / "THEOREM.md").write_text(theorem_informal + "\n")

    # Conversation state
    conversation_id = None
    turns = 0
    verifications = 0
    proved = False
    start = time.monotonic()
    log_lines: list[str] = []

    def log(msg: str):
        log_lines.append(msg)
        print(f"  [{name}] {msg}", flush=True)

    log(f"starting (model={model}, budget={max_time:.0f}s)")

    try:
        while True:
            elapsed = time.monotonic() - start
            remaining = max_time - elapsed
            if remaining <= 0:
                log(f"time budget exhausted after {turns} turns, {verifications} verifications")
                break

            turns += 1

            # Build request
            if conversation_id is None:
                # First turn
                payload = {
                    "model": mistral_model,
                    "inputs": [{"role": "user", "content": "Prove the theorem. Use lean_verify to check your proof."}],
                    "instructions": system_prompt,
                    "tools": [LEAN_VERIFY_TOOL],
                    "stream": False,
                    "completion_args": {
                        "temperature": 1.0,
                        "max_tokens": 256_000,
                        "top_p": 1,
                        "reasoning_effort": "high",
                    },
                }
            else:
                # Continuation: send tool results
                payload = {
                    "inputs": pending_inputs,
                    "stream": False,
                    "completion_args": {
                        "temperature": 1.0,
                        "max_tokens": 256_000,
                        "top_p": 1,
                        "reasoning_effort": "high",
                    },
                }

            # Call API
            raw = _mistral_request(payload, api_key,
                                   conversation_id=conversation_id,
                                   timeout=max(int(remaining) + 30, 60))
            conversation_id = raw.get("conversation_id", conversation_id)
            parsed = _parse_assistant_output(raw)

            # Archive raw response
            (run_dir / f"turn_{turns:03d}.json").write_text(
                json.dumps(raw, indent=2, ensure_ascii=False))

            if parsed["text"]:
                log(f"turn {turns}: {parsed['text'][:120]}...")

            if not parsed["tool_calls"]:
                if proved:
                    break
                # Model didn't call a tool — restart conversation with
                # stronger nudge (Conversations API doesn't allow user
                # messages after thinking-only output).
                log(f"turn {turns}: no tool call, restarting conversation")
                conversation_id = None
                continue

            # Process tool calls
            pending_inputs = []
            for tc in parsed["tool_calls"]:
                tc_id = tc.get("tool_call_id") or tc.get("id", "")
                fn_name = tc.get("name", "")
                fn_args_raw = tc.get("arguments", "{}")
                if isinstance(fn_args_raw, str):
                    try:
                        fn_args = json.loads(fn_args_raw)
                    except json.JSONDecodeError:
                        fn_args = {"code": fn_args_raw}
                else:
                    fn_args = fn_args_raw

                if fn_name == "lean_verify":
                    verifications += 1
                    code = fn_args.get("code", "")
                    result = _lean_verify(code, lean_work_dir, lean_project_dir)
                    status_short = "OK" if result == "OK" else "error"
                    log(f"turn {turns}: lean_verify #{verifications} -> {status_short}")

                    if result == "OK":
                        proved = True
                        (run_dir / "PROOF.lean").write_text(code + "\n")

                    pending_inputs.append({
                        "tool_call_id": tc_id,
                        "result": result,
                        "type": "function.result",
                    })
                else:
                    pending_inputs.append({
                        "tool_call_id": tc_id,
                        "result": f"Unknown tool: {fn_name}",
                        "type": "function.result",
                    })

            if proved:
                log(f"PROVED in {turns} turns, {verifications} verifications, "
                    f"{time.monotonic() - start:.0f}s")
                break

    except Exception as e:
        elapsed = time.monotonic() - start
        log(f"error: {e}")
        (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n")
        return {"status": "error", "elapsed": elapsed, "turns": turns,
                "verifications": verifications, "error": str(e)}

    elapsed = time.monotonic() - start
    status = "proved" if proved else "not_proved"
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n")
    (run_dir / "result.json").write_text(json.dumps({
        "name": name, "status": status, "elapsed": elapsed,
        "turns": turns, "verifications": verifications,
    }, indent=2) + "\n")

    return {"status": status, "elapsed": elapsed, "turns": turns,
            "verifications": verifications, "error": ""}


# ── CLI ──────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:40] or "theorem"


def _main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        prog="baseline",
        description="Baseline prover: single conversation with lean_verify",
    )
    parser.add_argument("run_dir", nargs="?",
                        help="Working directory (auto-created if omitted)")
    parser.add_argument("--theorem", type=Path, metavar="FILE",
                        help="Path to informal theorem statement (.md)")
    parser.add_argument("--lean-theorem", type=Path, metavar="FILE", required=True,
                        help="Path to formal theorem stub (.lean)")
    parser.add_argument("--lean-project", type=Path, metavar="DIR", required=True,
                        help="Path to Lean project with built .lake (lake build first)")
    parser.add_argument("--model", default="leanstral",
                        help="Model name (default: leanstral)")

    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--max-time", default="10m", metavar="DURATION",
                        help="Wall-clock budget per problem (default: 10m)")
    args = parser.parse_args()

    # Validate inputs
    if not args.lean_theorem.is_file():
        parser.error(f"--lean-theorem not found: {args.lean_theorem}")
    if not args.lean_project.is_dir():
        parser.error(f"--lean-project not found: {args.lean_project}")
    if not (args.lean_project / ".lake").is_dir():
        parser.error(f"{args.lean_project / '.lake'} not found — run `lake build` first")
    if args.theorem and not args.theorem.is_file():
        parser.error(f"--theorem not found: {args.theorem}")

    theorem_lean = args.lean_theorem.read_text()
    theorem_informal = args.theorem.read_text() if args.theorem else ""

    # Theorem name: extract from `theorem NAME ...` in the lean file
    name_match = re.search(r"^theorem\s+(\S+)", theorem_lean, re.MULTILINE)
    name = name_match.group(1) if name_match else args.lean_theorem.stem

    # Resolve run_dir (auto-create if not provided)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / f"baseline-{_slugify(name)}-{ts}"
        counter = 1
        while run_dir.exists():
            counter += 1
            run_dir = Path("runs") / f"baseline-{_slugify(name)}-{ts}-{counter}"

    budget_secs = _parse_duration(args.max_time)

    print(f"Theorem:  {name}")
    print(f"Model:    {args.model}")
    print(f"Budget:   {args.max_time} ({budget_secs:.0f}s)")
    print(f"Run dir:  {run_dir}")
    print("─" * 60)

    result = run_baseline(
        name=name,
        theorem_lean=theorem_lean,
        theorem_informal=theorem_informal,
        lean_project_dir=args.lean_project.resolve(),
        model=args.model,
        max_time=budget_secs,
        run_dir=run_dir,
    )

    print("─" * 60)
    print(f"Result:        {result['status']}")
    print(f"Elapsed:       {result['elapsed']:.0f}s")
    print(f"Turns:         {result['turns']}")
    print(f"Verifications: {result['verifications']}")
    if result.get("error"):
        print(f"Error:         {result['error']}")
    raise SystemExit(0 if result["status"] == "proved" else 1)


if __name__ == "__main__":
    _main()
