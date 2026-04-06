#!/usr/bin/env python3
"""Benchmark openprover (or baseline) against MiniF2F problems."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

VALID_URL = "https://raw.githubusercontent.com/google-deepmind/miniF2F/refs/heads/main/MiniF2F/Valid.lean"
TEST_URL = "https://raw.githubusercontent.com/google-deepmind/miniF2F/refs/heads/main/MiniF2F/Test.lean"

CLAUDE_MODELS = {"sonnet", "opus"}
RATE_LIMIT_WAIT = 600  # seconds to wait before retrying after rate limit

# Preamble that every extracted theorem file needs.
LEAN_PREAMBLE = """\
import MiniF2F.ProblemImports

open scoped Real
open scoped Nat
open scoped Topology
open scoped Polynomial

"""


# ── Helpers ──────────────────────────────────────────────────────────

def _check_tool(name: str) -> None:
    if shutil.which(name) is None:
        print(f"Error: '{name}' not found on PATH.", file=sys.stderr)
        sys.exit(1)


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _parse_duration(s: str) -> float:
    """Parse '10m', '2h', '1h30m' into seconds."""
    total = 0.0
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*([hms])", s):
        val, unit = float(match.group(1)), match.group(2)
        total += val * {"h": 3600, "m": 60, "s": 1}[unit]
    if total == 0:
        total = float(s)
    return total


# ── Theorem parsing ──────────────────────────────────────────────────

def _fetch_lean_file(split: str) -> str:
    url = VALID_URL if split == "valid" else TEST_URL
    print(f"  Fetching {split.capitalize()}.lean from GitHub...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode()


def _parse_theorems(lean_source: str) -> dict[str, dict[str, str]]:
    """Parse theorem names, informal statements, and formal code.

    Returns {name: {"informal": ..., "formal": ...}}.
    """
    theorems: dict[str, dict[str, str]] = {}
    pattern = re.compile(
        r"(?:(?P<comment>/--.*?-/)\s*)?"
        r"^(?P<theorem>theorem\s+(?P<name>\S+).*?:=\s*by\s*\n\s*sorry)",
        re.MULTILINE | re.DOTALL,
    )
    for m in pattern.finditer(lean_source):
        name = m.group("name")
        comment = m.group("comment") or ""
        informal = re.sub(r"^/--\s*", "", comment)
        informal = re.sub(r"\s*-/$", "", informal).strip()
        theorems[name] = {"informal": informal, "formal": m.group("theorem")}
    return theorems


# ── Results tracking ─────────────────────────────────────────────────

def _save_results(bench_dir: Path, results: list[dict]) -> None:
    """Write results.json atomically."""
    tmp = bench_dir / "results.json.tmp"
    tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    tmp.rename(bench_dir / "results.json")


# ── OpenProver runner ────────────────────────────────────────────────

def _run_openprover(
    name: str, info: dict[str, str],
    lean_project: Path | None, bench_dir: Path,
    args: argparse.Namespace,
) -> dict:
    tmpfiles: list[str] = []
    start = time.monotonic()
    try:
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

        if lean_project and not args.informal:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", delete=False, dir=lean_project,
            ) as f:
                f.write(LEAN_PREAMBLE + info["formal"] + "\n")
                tmpfiles.append(f.name)
            cmd.extend(["--lean-project", str(lean_project)])
            cmd.extend(["--lean-theorem", f.name])

        hf_models = {"minimax-m2.5"}
        used = {args.model, args.planner_model, args.worker_model} - {None}
        if used & hf_models:
            cmd.extend(["--provider-url", args.provider_url])
        cmd.append("--isolation" if args.isolation else "--no-isolation")

        # For Claude models: exit on rate limit so the benchmark can retry
        if used & CLAUDE_MODELS:
            cmd.extend(["--on-rate-limited", "exit"])

        # Hard timeout: budget + 2 min grace for final LLM call / Lean check
        hard_timeout = _parse_duration(args.max_time) + 120

        while True:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        timeout=hard_timeout)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                return {"name": name, "status": "not_proved", "elapsed": elapsed,
                        "error": f"hard timeout ({hard_timeout:.0f}s)"}

            elapsed = time.monotonic() - start

            if result.returncode != 0:
                err = "\n".join((result.stderr or "unknown").strip().splitlines()[-3:])
                return {"name": name, "status": "error", "elapsed": elapsed, "error": err}

            if "[result] rate_limited" in result.stdout:
                print(f"  ⏳ {name}: rate limited, waiting {RATE_LIMIT_WAIT // 60}m...",
                      flush=True)
                time.sleep(RATE_LIMIT_WAIT)
                continue  # retry the same problem

            status = "proved" if "[result] proved" in result.stdout else "not_proved"
            return {"name": name, "status": status, "elapsed": elapsed, "error": ""}

    except Exception as e:
        return {"name": name, "status": "error", "elapsed": time.monotonic() - start,
                "error": str(e)}
    finally:
        for p in tmpfiles:
            Path(p).unlink(missing_ok=True)


# ── Baseline runner ──────────────────────────────────────────────────

def _run_baseline(
    name: str, info: dict[str, str],
    lean_project: Path | None, bench_dir: Path,
    args: argparse.Namespace,
) -> dict:
    if lean_project is None:
        return {"name": name, "status": "error", "elapsed": 0,
                "error": "baseline requires --repo-path for Lean verification"}

    from baseline import run_baseline

    run_dir = bench_dir / "runs" / name
    budget_secs = _parse_duration(args.max_time)
    result = run_baseline(
        name=name,
        theorem_lean=info["formal"],
        theorem_informal=info["informal"],
        lean_project_dir=lean_project,
        model=args.model,
        max_time=budget_secs,
        run_dir=run_dir,
    )
    return {
        "name": name,
        "status": result["status"],
        "elapsed": result["elapsed"],
        "turns": result.get("turns", 0),
        "verifications": result.get("verifications", 0),
        "error": result.get("error", ""),
    }


# ── Main loop ────────────────────────────────────────────────────────

def _run_all(
    problems: dict[str, dict[str, str]],
    lean_project: Path | None,
    bench_dir: Path,
    args: argparse.Namespace,
) -> None:
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
    print(f"  MiniF2F {args.split}: {total} problems, method={args.method},"
          f" parallelism={args.problem_parallelism},"
          f" model={model_label}, mode={mode}")
    print(f"  Output: {bench_dir}\n")

    runner = _run_openprover if args.method == "openprover" else _run_baseline
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.problem_parallelism) as pool:
        futures = {
            pool.submit(runner, name, info, lean_project, bench_dir, args): name
            for name, info in problems.items()
        }

        for future in as_completed(futures):
            entry = future.result()
            completed += 1
            results.append(entry)
            _save_results(bench_dir, results)

            status = entry["status"]
            if status == "proved":
                proved += 1
            elif status == "not_proved":
                not_proved += 1
            else:
                errors += 1

            running = min(args.problem_parallelism, total - completed)
            elapsed_str = _format_time(entry["elapsed"])
            counts = f"P:{proved} F:{not_proved} E:{errors}"
            if running > 0:
                counts += f" R:{running}"

            extra = ""
            if "verifications" in entry and entry["verifications"]:
                extra = f"  v={entry['verifications']}"

            print(f"  [{completed:>{pad}}/{total}]"
                  f"  {entry['name']:<{name_width}}"
                  f"  {status.replace('_', ' '):<12}"
                  f"  {elapsed_str:>7}"
                  f"  ({counts}){extra}")

            if entry.get("error"):
                for line in entry["error"].strip().splitlines():
                    print(f"           {line}", file=sys.stderr)

    print(f"\n  Results: {proved} proved, {not_proved} not proved,"
          f" {errors} errors (of {total})")
    print(f"  Saved to {bench_dir / 'results.json'}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark openprover or baseline against MiniF2F")
    parser.add_argument("split", nargs="?", default="valid",
                        choices=["valid", "test"],
                        help="Which MiniF2F split (default: valid)")
    parser.add_argument("--method", default="openprover",
                        choices=["openprover", "baseline"],
                        help="Proving method (default: openprover)")
    parser.add_argument("--repo-path", type=Path, default=None,
                        help="Path to cloned MiniF2F repository (Lean verification)")
    parser.add_argument("--problem",
                        help="Run a single theorem (e.g., amc12a_2019_p21)")
    parser.add_argument("--limit", type=int,
                        help="Limit number of problems")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N problems (resume interrupted runs)")
    parser.add_argument("--problem-parallelism", type=int, default=1,
                        help="Concurrent problem instances (default: 1)")
    parser.add_argument("-P", "--parallelism", type=int, default=1,
                        help="Parallel workers per openprover step (default: 1)")

    model_choices = ["sonnet", "opus", "minimax-m2.5", "leanstral"]
    parser.add_argument("--model", default="sonnet", choices=model_choices)
    parser.add_argument("--planner-model", choices=model_choices, default=None)
    parser.add_argument("--worker-model", choices=model_choices, default=None)
    parser.add_argument("--provider-url", default="http://localhost:8000")
    parser.add_argument("--max-time", default="4h", metavar="DURATION",
                        help="Time budget per problem (default: 4h)")
    parser.add_argument("--informal", action="store_true",
                        help="Skip Lean verification (informal only)")
    parser.add_argument("--isolation",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--name",
                        help="Custom benchmark name (default: auto-generated)")
    args = parser.parse_args()

    # ── Resolve Lean project ──
    lean_project: Path | None = None
    if args.repo_path:
        lean_project = args.repo_path.resolve()
        if not (lean_project / "lakefile.lean").is_file() \
                and not (lean_project / "lakefile.toml").is_file():
            print(f"Error: {lean_project} has no lakefile.", file=sys.stderr)
            sys.exit(1)
        if not (lean_project / ".lake").is_dir():
            print(f"Error: {lean_project / '.lake'} not found."
                  " Run `lake build` first.", file=sys.stderr)
            sys.exit(1)

    # ── Load theorems ──
    if args.repo_path:
        filename = "Valid.lean" if args.split == "valid" else "Test.lean"
        lean_path = lean_project / "MiniF2F" / filename
        if not lean_path.is_file():
            print(f"Error: {lean_path} not found.", file=sys.stderr)
            sys.exit(1)
        lean_source = lean_path.read_text()
    else:
        lean_source = _fetch_lean_file(args.split)

    problems = _parse_theorems(lean_source)
    if not problems:
        print("Error: No theorems found.", file=sys.stderr)
        sys.exit(1)
    print(f"  Parsed {len(problems)} theorems from MiniF2F {args.split}")

    # ── Check tools ──
    all_models = {args.model, args.planner_model, args.worker_model} - {None}
    if all_models & {"sonnet", "opus"}:
        _check_tool("claude")

    # ── Filter ──
    if args.problem:
        if args.problem not in problems:
            print(f"Error: unknown theorem '{args.problem}'", file=sys.stderr)
            matches = [n for n in problems if args.problem in n]
            if matches:
                print(f"  Did you mean: {', '.join(matches[:5])}",
                      file=sys.stderr)
            sys.exit(1)
        problems = {args.problem: problems[args.problem]}
    if args.skip > 0:
        problems = dict(list(problems.items())[args.skip:])
        print(f"  Skipping first {args.skip}, {len(problems)} remaining")
    if args.limit is not None:
        problems = dict(list(problems.items())[:args.limit])

    # ── Create benchmark dir ──
    benchmarks_root = Path("benchmarks")
    benchmarks_root.mkdir(exist_ok=True)
    if args.name:
        bench_name = args.name
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_tag = args.model
        bench_name = f"{args.method}-{args.split}-{model_tag}-{ts}"
    bench_dir = benchmarks_root / bench_name
    if bench_dir.exists():
        print(f"Error: {bench_dir} already exists. Use --name to pick a"
              " different name, or delete the existing directory.",
              file=sys.stderr)
        sys.exit(1)
    bench_dir.mkdir(parents=True)

    # Save config
    config = {
        "method": args.method,
        "split": args.split,
        "model": args.model,
        "planner_model": args.planner_model,
        "worker_model": args.worker_model,
        "max_time": args.max_time,
        "parallelism": args.parallelism,
        "problem_parallelism": args.problem_parallelism,
        "informal": args.informal,
        "total_problems": len(problems),
        "skip": args.skip,
        "limit": args.limit,
    }
    (bench_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n")

    _run_all(problems, lean_project, bench_dir, args)


if __name__ == "__main__":
    main()
