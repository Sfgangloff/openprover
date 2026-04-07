#!/usr/bin/env python3
"""Baseline prover: plain conversation, no tools.

The model is asked to write a Lean 4 proof inside <lean>...</lean> tags.
Each turn we extract the block, run lean_verify, and append the result
as a follow-up user message.  Loops until the proof verifies or the
time budget runs out.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from openprover.lean.core import (
    LeanTheorem, LeanWorkDir, lean_has_errors, run_lean_check,
)
from openprover.llm import MistralClient

logger = logging.getLogger("baseline")

MISTRAL_MODEL_MAP = {"leanstral": "labs-leanstral-2603"}

# Match ```lean ... ``` (or ```lean4 ... ```) markdown code fences.
LEAN_FENCE_RE = re.compile(
    r"```lean[0-9]*\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_duration(s: str) -> float:
    """Parse '10m', '2h', '1h30m' into seconds."""
    total = 0.0
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*([hms])", s):
        val, unit = float(match.group(1)), match.group(2)
        total += val * {"h": 3600, "m": 60, "s": 1}[unit]
    if total == 0:
        total = float(s)
    return total


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:40] or "theorem"


def _extract_lean_blocks(text: str) -> list[str]:
    """Return the contents of every ```lean ... ``` block in text, in order."""
    return [m.strip() for m in LEAN_FENCE_RE.findall(text)]


def _verify(code: str, work_dir: LeanWorkDir, project_dir: Path) -> tuple[bool, str]:
    """Run lean_verify and return (success, feedback)."""
    path = work_dir.make_file("baseline_verify", code)
    success, feedback, _ = run_lean_check(path, project_dir)
    if success:
        return True, "OK"
    if "sorry" in feedback.lower() and not lean_has_errors(feedback):
        return False, feedback + "\n\nNote: code contains sorry — proof has gaps."
    return False, feedback


# ── Core loop ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a Lean 4 theorem prover. You will be given a theorem statement \
whose proof is `sorry`. You must produce a Lean 4 proof term/tactic \
block that replaces `sorry` so the theorem compiles.

Each time you reply, output ONE ```lean ... ``` markdown code fence \
containing JUST the replacement for `sorry` — no `theorem` line, no \
imports, no opens. The framework splices your block into the original \
theorem and runs `lake env lean` on it. If verification fails, the \
compiler errors are reported back; try a different approach.

If the theorem has multiple `sorry`s, output one ```lean ... ``` fence \
per `sorry`, in order. Only the LAST set of fences in your reply is \
used, so put your final attempt at the end.

You may NOT use: `sorry`, `axiom`, `unsafe`, `set_option`, \
`native_decide`, or `import` inside your replacement.\
"""

INITIAL_USER_MSG = """\
Prove this theorem. Output a single ```lean ... ``` block containing \
the proof body that replaces `sorry`.

## Informal statement
{informal}

## Theorem (you must replace `sorry` with your proof)
```lean
{formal}
```

The theorem above has {num_sorries} `sorry` placeholder(s). Output \
{num_sorries} ```lean ... ``` block(s), one per placeholder, in order.
"""


def run_baseline(
    name: str,
    theorem_lean: str,
    theorem_informal: str,
    lean_project_dir: Path,
    model: str,
    run_dir: Path,
    max_time: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
) -> dict:
    """Run the baseline on a single theorem.

    Exactly one of max_time (seconds) or max_tokens (completion tokens)
    must be set. Returns {"status", "elapsed", "turns", "verifications",
    "tokens", "error"}.
    """
    if (max_time is None) == (max_tokens is None):
        raise ValueError("specify exactly one of max_time or max_tokens")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "THEOREM.lean").write_text(theorem_lean + "\n")
    if theorem_informal:
        (run_dir / "THEOREM.md").write_text(theorem_informal + "\n")

    archive_dir = run_dir / "calls"
    mistral_model = MISTRAL_MODEL_MAP.get(model, model)
    client = MistralClient(model=mistral_model, archive_dir=archive_dir)
    work_dir = LeanWorkDir(lean_project_dir)

    # Parse the input theorem so we can splice proof bodies into its
    # original sorries — the model can never edit the statement.
    parsed = LeanTheorem(theorem_lean)
    if parsed.num_sorries == 0:
        return {"status": "error", "elapsed": 0, "turns": 0,
                "verifications": 0, "tokens": 0,
                "error": "input theorem contains no `sorry` placeholder"}

    initial_user = INITIAL_USER_MSG.format(
        informal=theorem_informal or "(none provided)",
        formal=theorem_lean,
        num_sorries=parsed.num_sorries,
    )

    # Conversation as a single accumulated user prompt (we restart the
    # context each turn — no tool/function machinery, no API state).
    transcript: list[str] = [initial_user]
    log_lines: list[str] = []
    turns = 0
    verifications = 0
    tokens = 0
    proved = False
    start = time.monotonic()

    def log(msg: str):
        log_lines.append(msg)
        print(f"  [{name}] {msg}", flush=True)

    budget_str = (f"{max_time:.0f}s" if max_time is not None
                  else f"{max_tokens} tokens")
    log(f"starting (model={model}, budget={budget_str})")

    # Streaming callback (used by MistralClient if stream=True)
    dim = "\033[2m" if sys.stdout.isatty() else ""
    reset = "\033[0m" if sys.stdout.isatty() else ""
    state = {"in_thinking": False}

    def stream_cb(text: str, kind: str):
        if kind == "thinking":
            if not state["in_thinking"]:
                sys.stdout.write(dim)
                state["in_thinking"] = True
        else:
            if state["in_thinking"]:
                sys.stdout.write(reset)
                state["in_thinking"] = False
        sys.stdout.write(text)
        sys.stdout.flush()

    try:
        while True:
            elapsed = time.monotonic() - start
            if max_time is not None and elapsed >= max_time:
                log(f"time budget exhausted after {turns} turns, "
                    f"{verifications} verifications, {tokens} tokens")
                break
            if max_tokens is not None and tokens >= max_tokens:
                log(f"token budget exhausted after {turns} turns, "
                    f"{verifications} verifications, {tokens} tokens")
                break

            turns += 1
            prompt = "\n\n".join(transcript)

            if stream:
                print(f"\n  ─── turn {turns} ───", flush=True)
                resp = client.call(prompt=prompt, system_prompt=SYSTEM_PROMPT,
                                   stream_callback=stream_cb,
                                   label=f"baseline-{name}-t{turns}")
                if state["in_thinking"]:
                    sys.stdout.write(reset)
                    state["in_thinking"] = False
                print(flush=True)
            else:
                resp = client.call(prompt=prompt, system_prompt=SYSTEM_PROMPT,
                                   label=f"baseline-{name}-t{turns}")

            assistant_text = resp.get("result", "") or ""
            thinking_text = resp.get("thinking", "") or ""
            (run_dir / f"turn_{turns:03d}.txt").write_text(assistant_text + "\n")

            # Track completion tokens (use API usage if available, else
            # approximate via char count / 4 — works for streaming path
            # which doesn't expose usage).
            usage = (resp.get("raw") or {}).get("usage") or {}
            turn_tokens = usage.get("completion_tokens")
            if turn_tokens is None:
                turn_tokens = (len(assistant_text) + len(thinking_text)) // 4
            tokens += turn_tokens

            blocks = _extract_lean_blocks(assistant_text)
            # Take the LAST `num_sorries` blocks (matches the system prompt:
            # the model puts its final attempt at the end).
            replacements = blocks[-parsed.num_sorries:] if blocks else []

            if len(replacements) != parsed.num_sorries:
                log(f"turn {turns}: expected {parsed.num_sorries} ```lean "
                    f"block(s), got {len(blocks)} "
                    f"(reply: {len(assistant_text)} chars, "
                    f"{turn_tokens} tokens)")
                transcript.append(
                    f"# Assistant turn {turns}\n{assistant_text}\n\n"
                    f"# User\nExpected {parsed.num_sorries} ```lean ... ``` "
                    f"code fence(s) (one per `sorry`), got {len(blocks)}. "
                    "Output exactly the right number of fenced blocks, in "
                    "order, each containing only the proof body that "
                    "replaces the corresponding `sorry`."
                )
                continue

            # Splice into the original theorem. assemble_proof rejects
            # banned constructs (sorry, axiom, import, ...).
            try:
                full_code = parsed.assemble_proof(replacements)
            except ValueError as e:
                log(f"turn {turns}: assembly rejected: {e}")
                transcript.append(
                    f"# Assistant turn {turns}\n{assistant_text}\n\n"
                    f"# User\nYour proof block was rejected: {e}. "
                    "Output a fresh ```lean ... ``` block containing only "
                    "the proof body."
                )
                continue

            verifications += 1
            log(f"turn {turns}: verifying ({sum(len(r) for r in replacements)}"
                f" chars of proof, {turn_tokens} tokens this turn, {tokens} total)")
            verify_start = time.monotonic()
            success, feedback = _verify(full_code, work_dir, lean_project_dir)
            verify_secs = time.monotonic() - verify_start

            if success:
                proved = True
                (run_dir / "PROOF.lean").write_text(full_code + "\n")
                log(f"turn {turns}: lean_verify #{verifications} -> OK "
                    f"({verify_secs:.1f}s)")
                log(f"PROVED in {turns} turns, {verifications} verifications, "
                    f"{tokens} tokens, {time.monotonic() - start:.0f}s")
                break

            log(f"turn {turns}: lean_verify #{verifications} -> failed "
                f"({verify_secs:.1f}s)")
            if stream:
                # Show the compiler feedback in interactive mode so the user
                # can follow along.
                preview = feedback.strip()
                if len(preview) > 1500:
                    preview = preview[:1500] + "\n  ... (truncated)"
                for line in preview.splitlines():
                    print(f"  │ {line}", flush=True)

            transcript.append(
                f"# Assistant turn {turns}\n{assistant_text}\n\n"
                f"# User\nlean_verify failed:\n```\n{feedback}\n```\n"
                "Try again. Output the full Lean source inside a "
                "```lean ... ``` code fence."
            )

    except Exception as e:
        elapsed = time.monotonic() - start
        log(f"error: {e}")
        (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n")
        return {"status": "error", "elapsed": elapsed, "turns": turns,
                "verifications": verifications, "tokens": tokens, "error": str(e)}

    elapsed = time.monotonic() - start
    status = "proved" if proved else "not_proved"
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n")
    (run_dir / "result.json").write_text(json.dumps({
        "name": name, "status": status, "elapsed": elapsed,
        "turns": turns, "verifications": verifications, "tokens": tokens,
    }, indent=2) + "\n")

    return {"status": status, "elapsed": elapsed, "turns": turns,
            "verifications": verifications, "tokens": tokens, "error": ""}


# ── CLI ──────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        prog="baseline",
        description="Baseline Lean prover: single conversation, no tools",
    )
    parser.add_argument("run_dir", nargs="?",
                        help="Working directory (auto-created under runs/ if omitted)")
    parser.add_argument("--theorem", type=Path, metavar="FILE",
                        help="Path to informal theorem statement (.md)")
    parser.add_argument("--lean-theorem", type=Path, metavar="FILE", required=True,
                        help="Path to formal theorem stub (.lean)")
    parser.add_argument("--lean-project", type=Path, metavar="DIR", required=True,
                        help="Path to Lean project with built .lake (lake build first)")
    parser.add_argument("--model", default="leanstral",
                        help="Model name (default: leanstral)")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--max-time", default=None, metavar="DURATION",
                        help="Wall-clock budget per problem (e.g. '10m', '2h')")
    budget.add_argument("--max-tokens", type=int, default=None, metavar="N",
                        help="Output token budget per problem")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Stream model output to console "
                             "(thinking shown dimmed)")
    args = parser.parse_args()

    if args.max_time is None and args.max_tokens is None:
        args.max_time = "10m"  # default

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

    name_match = re.search(r"^theorem\s+(\S+)", theorem_lean, re.MULTILINE)
    name = name_match.group(1) if name_match else args.lean_theorem.stem

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / f"baseline-{_slugify(name)}-{ts}"
        counter = 1
        while run_dir.exists():
            counter += 1
            run_dir = Path("runs") / f"baseline-{_slugify(name)}-{ts}-{counter}"

    budget_secs = _parse_duration(args.max_time) if args.max_time else None

    print(f"Theorem:  {name}")
    print(f"Model:    {args.model}")
    if budget_secs is not None:
        print(f"Budget:   {args.max_time} ({budget_secs:.0f}s)")
    else:
        print(f"Budget:   {args.max_tokens} tokens")
    print(f"Run dir:  {run_dir}")
    print("─" * 60)

    result = run_baseline(
        name=name,
        theorem_lean=theorem_lean,
        theorem_informal=theorem_informal,
        lean_project_dir=args.lean_project.resolve(),
        model=args.model,
        max_time=budget_secs,
        max_tokens=args.max_tokens,
        run_dir=run_dir,
        stream=args.stream,
    )

    print("─" * 60)
    print(f"Result:        {result['status']}")
    print(f"Elapsed:       {result['elapsed']:.0f}s")
    print(f"Turns:         {result['turns']}")
    print(f"Verifications: {result['verifications']}")
    print(f"Tokens:        {result['tokens']}")
    if result.get("error"):
        print(f"Error:         {result['error']}")
    raise SystemExit(0 if result["status"] == "proved" else 1)


if __name__ == "__main__":
    _main()
