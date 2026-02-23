"""CLI entry point for OpenProver."""

import argparse
import signal
from pathlib import Path

from openprover import __version__
from .llm import LLMClient, HFClient
from .prover import Prover
from .tui import TUI, HeadlessTUI


def main():
    parser = argparse.ArgumentParser(
        prog="openprover",
        description="Theorem prover powered by language models",
    )
    parser.add_argument("theorem", nargs="?", help="Path to theorem statement file (.md)")
    parser.add_argument("--model", default="sonnet", choices=["sonnet", "opus", "qed-nano"], help="Model to use (default: sonnet)")
    parser.add_argument("--hf-url", default="http://localhost:8000", help="HF server URL for qed-nano (default: http://localhost:8000)")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum number of proving steps (default: 50)")
    parser.add_argument("--autonomous", action="store_true", help="Start in autonomous mode (default: interactive)")
    parser.add_argument("--run-dir", help="Working directory (resumes if it contains an existing run)")
    parser.add_argument("--isolation", action=argparse.BooleanOptionalAction, default=True, help="Disable web searches (no literature_search action)")
    parser.add_argument("-P", "--parallelism", type=int, default=1, help="Max parallel workers per spawn step (default: 1)")
    parser.add_argument("--give-up-after", type=float, default=0.5, metavar="RATIO", help="Fraction of steps before give_up action is allowed (default: 0.5)")
    parser.add_argument("--answer-reserve", type=int, default=4096, metavar="TOKENS", help="Tokens reserved for answer after thinking (qed-nano, default: 4096)")
    parser.add_argument("--headless", action="store_true", help="Non-interactive mode (logs to stdout, errors to stderr)")
    parser.add_argument("--verbose", action="store_true", help="Show full LLM responses")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Lean verification
    parser.add_argument("--lean-project-dir", type=Path, metavar="DIR",
                        help="Path to Lean project with lakefile (enables formal verification)")
    parser.add_argument("--lean-theorem", type=Path, metavar="FILE",
                        help="Path to THEOREM.lean file (requires --lean-project-dir)")
    parser.add_argument("--proof", type=Path, metavar="FILE",
                        help="Path to existing PROOF.md (formalize-only mode, requires --lean-theorem)")
    parser.add_argument("--repl-dir", type=Path, metavar="DIR",
                        help="Path to lean-repl directory (reserved for future use)")

    args = parser.parse_args()

    if not args.theorem and not args.run_dir:
        parser.error("theorem is required (or use --run-dir to resume an existing run)")

    # Lean validation
    if args.lean_theorem and not args.lean_project_dir:
        parser.error("--lean-theorem requires --lean-project-dir")
    if args.proof and not args.lean_theorem:
        parser.error("--proof requires --lean-theorem (formalize-only mode needs a Lean theorem)")
    if args.lean_project_dir and not args.lean_project_dir.is_dir():
        parser.error(f"--lean-project-dir not found: {args.lean_project_dir}")
    if args.lean_theorem and not args.lean_theorem.is_file():
        parser.error(f"--lean-theorem not found: {args.lean_theorem}")
    if args.proof and not args.proof.is_file():
        parser.error(f"--proof not found: {args.proof}")

    # QED-Nano has no web search capability — force isolation
    if args.model == "qed-nano" and not args.isolation:
        args.isolation = True

    if args.headless:
        args.autonomous = True
        tui = HeadlessTUI()
    else:
        tui = TUI()

    # Construct LLM client — Prover.setup_work_dir needs to run first to know
    # the archive dir, so we pass a factory that Prover calls after setup.
    def make_llm(archive_dir):
        if args.model == "qed-nano":
            return HFClient("lm-provers/QED-Nano", archive_dir, base_url=args.hf_url, answer_reserve=args.answer_reserve)
        return LLMClient(args.model, archive_dir)

    prover = Prover(
        theorem_path=args.theorem,
        make_llm=make_llm,
        model_name=args.model,
        max_steps=args.max_steps,
        autonomous=args.autonomous,
        verbose=args.verbose,
        tui=tui,
        isolation=args.isolation,
        run_dir=args.run_dir,
        parallelism=args.parallelism,
        give_up_ratio=args.give_up_after,
        lean_project_dir=args.lean_project_dir,
        lean_theorem_path=args.lean_theorem,
        proof_path=args.proof,
    )

    # Check if this is a finished run → inspect mode
    if prover.is_finished and not args.theorem:
        try:
            prover.inspect()
        finally:
            tui.cleanup()
            print(f"  {prover.work_dir}")
        return

    # Signal handling: ctrl+c interrupts the active LLM call
    def handle_sigint(signum, frame):
        prover.request_interrupt()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        prover.run()
    finally:
        cost = prover.llm.total_cost
        calls = prover.llm.call_count
        tui.cleanup()
        # Print summary to regular stdout after TUI is gone
        has_proof = ((prover.work_dir / "PROOF.md").exists()
                     or (prover.work_dir / "PROOF.lean").exists())
        print(f"  {calls} calls · ${cost:.4f}")
        if (prover.work_dir / "PROOF.md").exists():
            print(f"  PROOF.md  → {prover.work_dir / 'PROOF.md'}")
        if (prover.work_dir / "PROOF.lean").exists():
            print(f"  PROOF.lean → {prover.work_dir / 'PROOF.lean'}")
        print(f"  {prover.work_dir}")
        if args.headless:
            print(f"[result] {'proved' if has_proof else 'not_proved'}")
