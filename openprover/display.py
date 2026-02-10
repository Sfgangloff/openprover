"""Terminal display formatting for OpenProver."""

import sys
import termios
from openprover import __version__


# Soft, readable palette
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
WHITE = "\033[97m"
BLUE = "\033[38;5;75m"
GREEN = "\033[38;5;114m"
YELLOW = "\033[38;5;222m"
RED = "\033[38;5;174m"
MAGENTA = "\033[38;5;183m"
CYAN = "\033[38;5;116m"

ACTION_STYLE = {
    "continue": CYAN,
    "explore_avenue": BLUE,
    "prove_lemma": GREEN,
    "verify": YELLOW,
    "check_counterexample": YELLOW,
    "literature_search": MAGENTA,
    "replan": YELLOW,
    "declare_proof": GREEN,
    "declare_stuck": RED,
}


def _oneline(text: str) -> str:
    """Collapse multiline text into one line, joining with spaces."""
    return " ".join(text.split())


def header(theorem_text: str, work_dir: str):
    theorem_oneline = _oneline(theorem_text)
    # Truncate if very long
    if len(theorem_oneline) > 120:
        theorem_oneline = theorem_oneline[:117] + "..."
    print()
    print(f"  {BOLD}OpenProver{RESET} {DIM}v{__version__}{RESET}")
    print(f"  {DIM}{'─' * 50}{RESET}")
    print(f"  {WHITE}{theorem_oneline}{RESET}")
    print(f"  {DIM}{work_dir}{RESET}")
    print()


def step_start(step_num: int, max_steps: int, action: str, summary: str):
    color = ACTION_STYLE.get(action, "")
    print(f"  {color}■{RESET} {BOLD}{action}{RESET} {DIM}—{RESET} {summary}")


def lemma_stored(name: str, source: str = ""):
    if source:
        print(f"    {GREEN}+{RESET} {DIM}lemma{RESET} {name} {DIM}[{source}]{RESET}")
    else:
        print(f"    {GREEN}+{RESET} {DIM}lemma{RESET} {name}")
    print()


def proposal(step_num: int, max_steps: int, plan: dict):
    color = ACTION_STYLE.get(plan.get("action", ""), "")
    print(f"  {color}▸{RESET} {BOLD}{plan['action']}{RESET} {DIM}—{RESET} {plan['summary']}")
    if plan.get("reasoning"):
        print(f"    {DIM}{plan['reasoning']}{RESET}")


def interactive_prompt() -> str:
    """Show prompt and get user input. Returns the input string."""
    # Flush buffered stdin to prevent phantom accepts from keypresses during thinking
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (termios.error, OSError, ValueError):
        pass
    try:
        response = input(f"    {DIM}[enter/feedback/s/a/p/r/q]{RESET} {DIM}▸{RESET} ")
        return response.strip()
    except EOFError:
        return "q"


def feedback_note(feedback: str):
    print(f"    {YELLOW}Replanning...{RESET}")
    print()


def literature_search(query: str):
    print(f"    {MAGENTA}Searching:{RESET} {query}")


def verification_start(target: str):
    print(f"    {YELLOW}Verifying:{RESET} {target}")


def verification_result(passed: bool):
    if passed:
        print(f"    {GREEN}✓ Passed{RESET}")
    else:
        print(f"    {RED}✗ Issues found{RESET}")
    print()


def proof_found():
    print(f"\n  {GREEN}{BOLD}✓ Proof found{RESET}")
    print()


def proof_rejected(reason: str):
    print(f"\n  {RED}✗ Proof rejected:{RESET} {reason}")
    print()


def step_error():
    print(f"  {RED}Step failed — retrying...{RESET}")
    print()


def stuck():
    print(f"\n  {YELLOW}Stuck — no more ideas.{RESET}")
    print()


def stuck_rejected(step_num: int, max_steps: int):
    print(f"\n  {YELLOW}Not giving up yet — only {step_num}/{max_steps} steps used. Keep trying.{RESET}")
    print()


def discussion_written(path: str):
    print(f"  {DIM}→ {path}{RESET}")


def proof_written(path: str):
    print(f"  {DIM}→ {path}{RESET}")


def pausing():
    print(f"\n  {YELLOW}Paused.{RESET}")
    print()


def resuming(step_num: int, max_steps: int):
    print(f"  {CYAN}Resuming from step {step_num}/{max_steps}{RESET}")
    print()


def restarting():
    print(f"\n  {CYAN}Restarting...{RESET}")
    print()


def shutting_down():
    print(f"\n  {YELLOW}Interrupted — writing discussion...{RESET}")


def cost_summary(total_cost: float, num_calls: int):
    print(f"  {DIM}{num_calls} calls · ${total_cost:.4f}{RESET}")
    print()


def mode_switch(mode: str):
    print(f"    {DIM}→ {mode} mode{RESET}")
    print()


def thinking():
    """Print a thinking indicator."""
    sys.stdout.write(f"    {DIM}thinking...{RESET}")
    sys.stdout.flush()


def thinking_done(duration_ms: int):
    s = duration_ms / 1000
    sys.stdout.write(f"\r    {DIM}{s:.1f}s{RESET}         \n")
    sys.stdout.flush()


def stream_text(text: str):
    """Write a chunk of streamed text."""
    sys.stdout.write(text)
    sys.stdout.flush()


def stream_start():
    """Begin a streamed text block."""
    sys.stdout.write(f"    {DIM}")
    sys.stdout.flush()


def stream_end():
    """End a streamed text block."""
    sys.stdout.write(f"{RESET}\n\n")
    sys.stdout.flush()
