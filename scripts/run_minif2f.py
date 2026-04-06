#!/usr/bin/env python3
"""Run openprover on MiniF2F problems."""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VALID_URL = "https://raw.githubusercontent.com/google-deepmind/miniF2F/refs/heads/main/MiniF2F/Valid.lean"
TEST_URL = "https://raw.githubusercontent.com/google-deepmind/miniF2F/refs/heads/main/MiniF2F/Test.lean"

# Preamble that every extracted theorem file needs.
LEAN_PREAMBLE = """\
import MiniF2F.ProblemImports

open scoped Real
open scoped Nat
open scoped Topology
open scoped Polynomial

"""


def _check_tool(name: str) -> None:
    if shutil.which(name) is None:
        print(f"Error: '{name}' not found on PATH.", file=sys.stderr)
        sys.exit(1)


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _fetch_lean_file(split: str) -> str:
    """Download the MiniF2F lean file for the given split."""
    url = VALID_URL if split == "valid" else TEST_URL
    print(f"  Fetching {split.capitalize()}.lean from GitHub...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode()


def _parse_theorems(lean_source: str) -> dict[str, dict[str, str]]:
    """Parse theorem names, informal statements, and formal code from a MiniF2F .lean file.

    Returns {name: {"informal": ..., "formal": ...}}.
    """
    theorems: dict[str, dict[str, str]] = {}

    # Split into blocks: each theorem starts with /-- or theorem
    # We look for pattern: optional docstring + theorem declaration + by sorry
    pattern = re.compile(
        r"(?:(?P<comment>/--.*?-/)\s*)?"          # optional docstring
        r"^(?P<theorem>theorem\s+(?P<name>\S+).*?:=\s*by\s*\n\s*sorry)",
        re.MULTILINE | re.DOTALL,
    )

    for m in pattern.finditer(lean_source):
        name = m.group("name")
        # Extract informal statement from docstring
        comment = m.group("comment") or ""
        informal = re.sub(r"^/--\s*", "", comment)
        informal = re.sub(r"\s*-/$", "", informal)
        informal = informal.strip()

        # Formal statement is the theorem block
        formal = m.group("theorem")

        theorems[name] = {"informal": informal, "formal": formal}

    return theorems


def _run_problem(
    name: str,
    info: dict[str, str],
    lean_project: Path | None,
    args: argparse.Namespace,
) -> tuple[str, str, float, str]:
    """Run openprover on a single MiniF2F problem.

    Returns (name, status, elapsed_seconds, error_message).
    """
    tmpfiles: list[str] = []
    try:
        # Write informal statement
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(f"{info['informal']}\n")
            theorem_path = f.name
            tmpfiles.append(f.name)

        cmd = [
            "openprover",
            "--theorem", theorem_path,
            "--model", args.model,
            "--max-time", args.max_time,
            "--headless",
            "-P", str(args.parallelism),
        ]

        if args.planner_model:
            cmd.extend(["--planner-model", args.planner_model])
        if args.worker_model:
            cmd.extend(["--worker-model", args.worker_model])

        # Set up Lean theorem file if we have a project
        if lean_project and not args.informal:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", delete=False, dir=lean_project
            ) as f:
                f.write(LEAN_PREAMBLE)
                f.write(info["formal"])
                f.write("\n")
                tmpfiles.append(f.name)
            cmd.extend(["--lean-project", str(lean_project)])
            cmd.extend(["--lean-theorem", f.name])

        hf_models = {"minimax-m2.5"}
        used_models = {args.model, args.planner_model, args.worker_model} - {None}
        if used_models & hf_models:
            cmd.extend(["--provider-url", args.provider_url])
        if args.isolation:
            cmd.append("--isolation")
        else:
            cmd.append("--no-isolation")

        start = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.monotonic() - start

        if result.returncode != 0:
            lines = result.stderr.strip().splitlines()[-3:] if result.stderr else ["unknown error"]
            return (name, "error", elapsed, "\n".join(lines))
        if "[result] proved" in result.stdout:
            return (name, "proved", elapsed, "")
        return (name, "not_proved", elapsed, "")
    except Exception as e:
        return (name, "error", time.monotonic() - start, str(e))
    finally:
        for p in tmpfiles:
            Path(p).unlink(missing_ok=True)


def _run_all(
    problems: dict[str, dict[str, str]],
    lean_project: Path | None,
    args: argparse.Namespace,
) -> None:
    """Run problems with progress tracking."""
    total = len(problems)
    proved = 0
    not_proved = 0
    errors = 0
    completed = 0
    pad = len(str(total))
    name_width = max(len(n) for n in problems) if problems else 20

    planner = args.planner_model or args.model
    worker = args.worker_model or args.model
    model_label = planner if planner == worker else f"{planner}/{worker}"
    mode = "informal" if args.informal or lean_project is None else "formal"
    print(f"  MiniF2F {args.split}: {total} problems, parallelism={args.problem_parallelism},"
          f" model={model_label}, mode={mode}\n")

    with ThreadPoolExecutor(max_workers=args.problem_parallelism) as pool:
        futures = {
            pool.submit(_run_problem, name, info, lean_project, args): name
            for name, info in problems.items()
        }

        for future in as_completed(futures):
            name, status, elapsed, error = future.result()
            completed += 1

            if status == "proved":
                proved += 1
            elif status == "not_proved":
                not_proved += 1
            else:
                errors += 1

            running = min(args.problem_parallelism, total - completed)
            elapsed_str = _format_time(elapsed)
            status_display = status.replace("_", " ")
            counts = f"P:{proved} F:{not_proved} E:{errors}"
            if running > 0:
                counts += f" R:{running}"

            print(f"  [{completed:>{pad}}/{total}]"
                  f"  {name:<{name_width}}"
                  f"  {status_display:<12}"
                  f"  {elapsed_str:>7}"
                  f"  ({counts})")

            if error:
                for line in error.strip().splitlines():
                    print(f"           {line}", file=sys.stderr)

    print(f"\n  Results: {proved} proved, {not_proved} not proved,"
          f" {errors} errors (of {total})")


def main():
    parser = argparse.ArgumentParser(description="Run openprover on MiniF2F problems")
    parser.add_argument("split", nargs="?", default="valid", choices=["valid", "test"],
                        help="Which MiniF2F split to run (default: valid)")
    parser.add_argument("--repo-path", type=Path, default=None,
                        help="Path to cloned MiniF2F repository (for Lean verification)")
    parser.add_argument("--problem", help="Specific theorem name to run (e.g., amc12a_2019_p21)")
    parser.add_argument("--limit", type=int, help="Limit number of problems to run")
    parser.add_argument("--problem-parallelism", type=int, default=1,
                        help="Number of concurrent openprover instances (default: 1)")
    parser.add_argument("-P", "--parallelism", type=int, default=1,
                        help="Max parallel workers per spawn step inside openprover (default: 1)")
    model_choices = ["sonnet", "opus", "minimax-m2.5", "leanstral"]
    parser.add_argument("--model", default="sonnet", choices=model_choices)
    parser.add_argument("--planner-model", choices=model_choices, default=None,
                        help="Override model for planner (defaults to --model)")
    parser.add_argument("--worker-model", choices=model_choices, default=None,
                        help="Override model for worker (defaults to --model)")
    parser.add_argument("--provider-url", default="http://localhost:8000",
                        help="Server URL for local models (default: http://localhost:8000)")
    parser.add_argument("--max-time", default="4h", metavar="DURATION",
                        help="Wall-clock time budget per problem, e.g. '30m', '2h' (default: 4h)")
    parser.add_argument("--informal", action="store_true",
                        help="Skip Lean verification; prove informally only")
    parser.add_argument("--isolation", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    # Resolve Lean project
    lean_project: Path | None = None
    if args.repo_path:
        lean_project = args.repo_path.resolve()
        if not (lean_project / "lakefile.lean").is_file() and not (lean_project / "lakefile.toml").is_file():
            print(f"Error: {lean_project} does not look like a Lean project"
                  " (no lakefile found).", file=sys.stderr)
            sys.exit(1)
        if not (lean_project / ".lake").is_dir():
            print(f"Error: {lean_project / '.lake'} not found."
                  " Build the project first with `lake build`.", file=sys.stderr)
            sys.exit(1)

    # Load theorems: prefer local file from repo, fall back to GitHub
    if args.repo_path:
        filename = "Valid.lean" if args.split == "valid" else "Test.lean"
        lean_path = args.repo_path.resolve() / "MiniF2F" / filename
        if not lean_path.is_file():
            print(f"Error: {lean_path} not found.", file=sys.stderr)
            sys.exit(1)
        lean_source = lean_path.read_text()
    else:
        lean_source = _fetch_lean_file(args.split)

    problems = _parse_theorems(lean_source)
    if not problems:
        print("Error: No theorems found in the Lean file.", file=sys.stderr)
        sys.exit(1)

    print(f"  Parsed {len(problems)} theorems from MiniF2F {args.split}")

    # Check required tools
    all_models = {args.model, args.planner_model, args.worker_model} - {None}
    if all_models & {"sonnet", "opus"}:
        _check_tool("claude")

    # Filter
    if args.problem:
        if args.problem not in problems:
            print(f"Error: unknown theorem '{args.problem}'", file=sys.stderr)
            available = [n for n in problems if args.problem in n]
            if available:
                print(f"  Did you mean: {', '.join(available[:5])}", file=sys.stderr)
            sys.exit(1)
        problems = {args.problem: problems[args.problem]}

    if args.limit is not None:
        problems = dict(list(problems.items())[:args.limit])

    _run_all(problems, lean_project, args)


if __name__ == "__main__":
    main()
