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

from openprover.lean.core import LeanWorkDir, lean_has_errors, run_lean_check
from openprover.llm import MistralClient

logger = logging.getLogger("baseline")

MISTRAL_MODEL_MAP = {"leanstral": "labs-leanstral-2603"}

LEAN_TAG_RE = re.compile(r"<lean>\s*\n?(.*?)\n?\s*</lean>", re.DOTALL | re.IGNORECASE)


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


def _extract_lean(text: str) -> str | None:
    """Pull the contents of the last <lean>...</lean> block from text."""
    matches = LEAN_TAG_RE.findall(text)
    return matches[-1].strip() if matches else None


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
ending in `sorry` and you must replace `sorry` with a complete proof.

Each time you reply, output the FULL Lean 4 source (imports, opens, and \
the theorem with your proof) inside a single <lean>...</lean> block. \
After your reply, the user will run `lake env lean` on your code and \
report the result. If verification fails, try a different approach.

Output ONLY the <lean>...</lean> block (and any short reasoning before \
it). Do not use markdown code fences inside the tag.\
"""

INITIAL_USER_MSG = """\
Prove this theorem. Output the full Lean source inside <lean>...</lean>.

## Informal statement
{informal}

## Formal statement
<lean>
{formal}
</lean>
"""


def run_baseline(
    name: str,
    theorem_lean: str,
    theorem_informal: str,
    lean_project_dir: Path,
    model: str,
    max_time: float,
    run_dir: Path,
    stream: bool = False,
) -> dict:
    """Run the baseline on a single theorem.

    Returns {"status": ..., "elapsed": ..., "turns": ..., "verifications": ...,
             "error": ...}.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "THEOREM.lean").write_text(theorem_lean + "\n")
    if theorem_informal:
        (run_dir / "THEOREM.md").write_text(theorem_informal + "\n")

    archive_dir = run_dir / "calls"
    mistral_model = MISTRAL_MODEL_MAP.get(model, model)
    client = MistralClient(model=mistral_model, archive_dir=archive_dir)
    work_dir = LeanWorkDir(lean_project_dir)

    initial_user = INITIAL_USER_MSG.format(
        informal=theorem_informal or "(none provided)",
        formal=theorem_lean,
    )

    # Conversation as a single accumulated user prompt (we restart the
    # context each turn — no tool/function machinery, no API state).
    transcript: list[str] = [initial_user]
    log_lines: list[str] = []
    turns = 0
    verifications = 0
    proved = False
    start = time.monotonic()

    def log(msg: str):
        log_lines.append(msg)
        print(f"  [{name}] {msg}", flush=True)

    log(f"starting (model={model}, budget={max_time:.0f}s)")

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
            if elapsed >= max_time:
                log(f"time budget exhausted after {turns} turns, "
                    f"{verifications} verifications")
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
            (run_dir / f"turn_{turns:03d}.txt").write_text(assistant_text + "\n")

            code = _extract_lean(assistant_text)
            if not code:
                log(f"turn {turns}: no <lean> block found")
                transcript.append(
                    f"# Assistant turn {turns}\n{assistant_text}\n\n"
                    "# User\nYour reply did not contain a <lean>...</lean> "
                    "block. Please output the full Lean source inside "
                    "<lean>...</lean>."
                )
                continue

            verifications += 1
            success, feedback = _verify(code, work_dir, lean_project_dir)
            status_short = "OK" if success else "error"
            log(f"turn {turns}: lean_verify #{verifications} -> {status_short}")

            if success:
                proved = True
                (run_dir / "PROOF.lean").write_text(code + "\n")
                log(f"PROVED in {turns} turns, {verifications} verifications, "
                    f"{time.monotonic() - start:.0f}s")
                break

            transcript.append(
                f"# Assistant turn {turns}\n{assistant_text}\n\n"
                f"# User\nlean_verify failed:\n```\n{feedback}\n```\n"
                "Try again. Output the full Lean source inside "
                "<lean>...</lean>."
            )

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
    parser.add_argument("--max-time", default="10m", metavar="DURATION",
                        help="Wall-clock budget per problem (default: 10m)")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Stream model output to console "
                             "(thinking shown dimmed)")
    args = parser.parse_args()

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
        stream=args.stream,
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
