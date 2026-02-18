"""CLI entry point for OpenProver."""

import argparse
import signal
import sys

from openprover import __version__
from .prover import Prover
from .tui import TUI


def main():
    parser = argparse.ArgumentParser(
        prog="openprover",
        description="Theorem prover powered by language models",
    )
    parser.add_argument("theorem", nargs="?", help="Path to theorem statement file (.md)")
    parser.add_argument("--model", default="sonnet", choices=["sonnet", "opus"],
                        help="Model to use (default: sonnet)")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Maximum number of proving steps (default: 50)")
    parser.add_argument("--autonomous", action="store_true",
                        help="Start in autonomous mode (default: interactive)")
    parser.add_argument("--run-dir",
                        help="Working directory (resumes if it contains an existing run)")
    parser.add_argument("--isolation", action="store_true",
                        help="Disable web searches (no literature_search action)")
    parser.add_argument("-P", "--parallelism", type=int, default=1,
                        help="Max parallel workers per spawn step (default: 1)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full LLM responses")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    if not args.theorem and not args.run_dir:
        parser.error("theorem is required (or use --run-dir to resume an existing run)")

    tui = TUI()

    prover = Prover(
        theorem_path=args.theorem,
        model=args.model,
        max_steps=args.max_steps,
        autonomous=args.autonomous,
        verbose=args.verbose,
        tui=tui,
        isolation=args.isolation,
        run_dir=args.run_dir,
        parallelism=args.parallelism,
    )

    # Signal handling: first ctrl+c → graceful shutdown, second → immediate exit
    def handle_sigint(signum, frame):
        if prover.shutting_down:
            tui.cleanup()
            print("\nForce quit.")
            sys.exit(1)
        prover.request_shutdown()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        prover.run()
    finally:
        cost = prover.llm.total_cost
        calls = prover.llm.call_count
        tui.cleanup()
        # Print summary to regular stdout after TUI is gone
        print(f"  {calls} calls · ${cost:.4f}")
        if (prover.work_dir / "PROOF.md").exists():
            print(f"  PROOF.md → {prover.work_dir / 'PROOF.md'}")
        print(f"  {prover.work_dir}")
