"""ANSI color constants and style maps for OpenProver TUI."""

# 256-color palette
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
GRAY = "\033[38;5;245m"

SPINNER = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

ACTION_STYLE = {
    "spawn": BLUE,
    "literature_search": MAGENTA,
    "read_items": CYAN,
    "write_items": CYAN,
    "read_theorem": CYAN,
    "submit_proof": GREEN,
    "submit_lean_proof": GREEN,
    "give_up": RED,
}

TOOL_STYLE = {
    "lean_verify": CYAN,
    "lean_search": MAGENTA,
}

COLOR_MAP = {
    "red": RED, "green": GREEN, "blue": BLUE,
    "yellow": YELLOW, "magenta": MAGENTA, "cyan": CYAN,
}

HEADER_ROWS = 4

HELP_TEXT = f"""\
  {BOLD}Controls{RESET}

  {DIM}Instant keys (work any time):{RESET}
    r           toggle reasoning
    i           show worker input (on worker tabs)
    w           toggle whiteboard view
    a           toggle autonomous mode
    {DIM}\u2190/\u2192{RESET}         switch tabs
    {DIM}\u2191/\u2193{RESET}         scroll chat history
    pgup/pgdn   scroll chat history (page)
    ?           this help
    esc/enter   dismiss overlay

  {DIM}When confirming a plan:{RESET}
    {DIM}up/down{RESET}     browse step history
    tab         switch accept / feedback
    enter       confirm or view step detail
    esc         close detail / deselect
    s           summarize progress

  {DIM}In autonomous mode all keys are instant.{RESET}
  {DIM}Press ? or enter to dismiss.{RESET}
"""
