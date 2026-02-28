"""Terminal UI for OpenProver — ANSI scroll regions, fixed header, tabs."""

import os
import queue
import re
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty

from openprover import __version__

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

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

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

HELP_TEXT = f"""\
  {BOLD}Controls{RESET}

  {DIM}Instant keys (work any time):{RESET}
    r           toggle reasoning
    i           show worker input (on worker tabs)
    w           toggle whiteboard view
    a           toggle autonomous mode
    {DIM}←/→{RESET}         switch tabs
    {DIM}↑/↓{RESET}         scroll chat history
    pgup/pgdn   scroll chat history (page)
    ?           this help
    esc/enter   dismiss overlay

  {DIM}When confirming a plan:{RESET}
    {DIM}up/down{RESET}     browse step history
    tab         switch accept / feedback
    enter       confirm or view step detail
    esc         close detail / deselect
    s           summarize progress
    p           pause (resume with --run-dir)
    q           quit

  {DIM}In autonomous mode all keys are instant.{RESET}
  {DIM}Press ? or enter to dismiss.{RESET}
"""

COLOR_MAP = {
    "red": RED, "green": GREEN, "blue": BLUE,
    "yellow": YELLOW, "magenta": MAGENTA, "cyan": CYAN,
}

HEADER_ROWS = 4


class _LogEntry:
    """A line in the log. step_idx >= 0 marks completed-step lines."""
    __slots__ = ("text", "step_idx", "is_trace", "is_output")

    def __init__(self, text: str, step_idx: int = -1, is_trace: bool = False,
                 is_output: bool = False):
        self.text = text
        self.step_idx = step_idx
        self.is_trace = is_trace
        self.is_output = is_output


class _Tab:
    """A tab with its own log buffer and streaming state."""
    __slots__ = ("id", "label", "log_lines", "trace_buf", "output_buf",
                 "scroll_offset",
                 "streaming", "spinner_label", "spinner_tick", "spinner_time",
                 "spinner_start", "spinner_tokens", "last_trace", "last_output",
                 "toml_pending", "toml_close_tag", "output_non_toml_seen",
                 "output_toml_seen", "is_waiting",
                 "done", "task_description")

    def __init__(self, tab_id: str, label: str, task_description: str = ""):
        self.id = tab_id
        self.label = label
        self.log_lines: list[_LogEntry] = []
        self.trace_buf: list[str] = []
        self.output_buf: list[str] = []
        self.scroll_offset = 0
        self.streaming = False
        self.spinner_label = ""
        self.spinner_tick = 0
        self.spinner_time = 0.0
        self.spinner_start = 0.0
        self.spinner_tokens = 0
        self.last_trace = ""
        self.last_output = ""
        self.toml_pending = ""
        self.toml_close_tag = ""
        self.output_non_toml_seen = False
        self.output_toml_seen = False
        self.is_waiting = False
        self.done = False
        self.task_description = task_description


class TUI:
    supports_streaming = True

    def __init__(self):
        self.rows = 0
        self.cols = 0
        self.trace_visible = True
        self.view = "main"
        self.whiteboard = ""
        self.pending_action: str | None = None
        self.autonomous = False
        self._old_termios = None
        self._active = False
        self.theorem_name = ""
        self.step_num = 0
        self.max_steps = 0
        self._old_sigwinch = None
        # Background key reader
        self._key_queue: queue.Queue[str] = queue.Queue()
        self._bg_thread: threading.Thread | None = None
        self._bg_stop = False
        # Step history includes action/summary/detail plus status flags/feedback.
        self.step_entries: list[dict] = []
        self._nav_step = -1  # -1 = options focused, 0..N-1 = step index
        self._step_detail_text = ""
        self._step_detail_title = ""
        self._step_detail_idx = -1
        self._step_detail_scroll = 0
        # Confirmation state
        self._confirming = False
        self._browsing = False
        self._confirm_accept_label = "accept"
        self._confirm_selected = 0
        self._confirm_buf: list[str] = []
        # Thread safety for stdout
        self._write_lock = threading.Lock()
        self._key_process_lock = threading.Lock()
        # Tabs
        self.tabs: list[_Tab] = [_Tab("planner", "Planner"), _Tab("logs", "Logs")]
        self.active_tab_idx = 0
        # Saved worker tabs (when navigating history)
        self._saved_worker_tabs: list[_Tab] | None = None
        # Proposal log range (for cleanup after confirmation)
        self._proposal_log_start: int = -1
        # Transient feedback/replan notice in planner tab.
        self._replan_notice_entry: _LogEntry | None = None
        # Run parameters (shown in help view)
        self.run_params: dict[str, str] = {}

    _content_start = HEADER_ROWS + 1

    @property
    def _active_tab(self) -> _Tab:
        if self.active_tab_idx < len(self.tabs):
            return self.tabs[self.active_tab_idx]
        return self.tabs[0]

    def _find_tab(self, tab_id: str) -> _Tab:
        for tab in self.tabs:
            if tab.id == tab_id:
                return tab
        return self.tabs[0]

    def _find_tab_or_none(self, tab_id: str) -> _Tab | None:
        for tab in self.tabs:
            if tab.id == tab_id:
                return tab
        return None

    def setup(self, theorem_name: str, work_dir: str,
              step_num: int = 0, max_steps: int = 50,
              model_name: str = ""):
        self.theorem_name = theorem_name
        self.work_dir = work_dir
        self.step_num = step_num
        self.max_steps = max_steps
        self.model_name = model_name
        size = shutil.get_terminal_size()
        self.cols, self.rows = size.columns, size.lines

        try:
            self._old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except (termios.error, OSError):
            self._old_termios = None

        with self._write_lock:
            self._write_raw('\033[?1049h\033[2J\033[?1000h\033[?1006h')
            self._draw_header()
            self._write_raw(f'\033[{self._content_start};{self.rows}r')
            sys.stdout.flush()
        self._write(f'\033[{self._content_start};1H\033[?25l')
        self._active = True

        self._old_sigwinch = signal.getsignal(signal.SIGWINCH)
        signal.signal(signal.SIGWINCH, self._on_resize)

        self._bg_stop = False
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

    def cleanup(self):
        if not self._active:
            return
        self._active = False
        self._bg_stop = True
        if self._bg_thread:
            self._bg_thread.join(timeout=0.2)
            self._bg_thread = None
        self._write('\033[?1000l\033[?1006l\033[r\033[?1049l\033[?25h')
        if self._old_termios:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            except (termios.error, OSError):
                pass
            self._old_termios = None
        if self._old_sigwinch is not None:
            try:
                signal.signal(signal.SIGWINCH, self._old_sigwinch)
            except (OSError, ValueError):
                pass

    def _on_resize(self, signum, frame):
        size = shutil.get_terminal_size()
        self.cols, self.rows = size.columns, size.lines
        self._write('\033[2J')
        self._write(f'\033[{self._content_start};{self.rows}r')
        self._redraw()

    # ── Low-level output ────────────────────────────────────────

    def _write(self, data: str):
        with self._write_lock:
            sys.stdout.write(data)
            sys.stdout.flush()

    def _write_raw(self, data: str):
        """Write without lock — caller must hold _write_lock."""
        sys.stdout.write(data)

    # ── Header (rows 1-4) ──────────────────────────────────────

    def _draw_header(self):
        w = self.cols
        step = f"step {self.step_num}/{self.max_steps}" if self.step_num else ""

        # Row 1
        model = getattr(self, 'model_name', '') or ''
        self._write_raw('\033[1;1H\033[2K')
        self._write_raw(f'{BLUE}╭─{RESET} {BOLD}OpenProver{RESET} {DIM}v{__version__}{RESET}')
        if step:
            self._write_raw(f' {BLUE}──{RESET} {DIM}{step}{RESET}')
        if model:
            self._write_raw(f' {BLUE}·{RESET} {YELLOW}{model}{RESET}')
        r1_text = f"─ OpenProver v{__version__}"
        if step:
            r1_text += f" ── {step}"
        if model:
            r1_text += f" · {model}"
        r1_text += " "
        fill1 = max(w - len(r1_text) - 2, 0)
        self._write_raw(f' {BLUE}{"─" * fill1}╮{RESET}')

        # Row 2 — theorem
        name = (self.theorem_name or "").replace("\n", " ").replace("\r", "")
        max_name = max(w - 4, 10)
        if len(name) > max_name:
            display_name = name[:max_name - 3] + "..."
        else:
            display_name = name
        name_pad = max(w - 3 - len(display_name), 0)
        self._write_raw('\033[2;1H\033[2K')
        self._write_raw(f'{BLUE}│{RESET} {WHITE}{display_name}{RESET}')
        self._write_raw(f'{" " * name_pad}{BLUE}│{RESET}')

        # Row 3 — hints
        help_style = BOLD if self.view == "help" else DIM
        auto_style = BOLD if self.autonomous else DIM
        wb_style = BOLD if self.view == "whiteboard" else DIM
        trace_style = BOLD if self.trace_visible else DIM
        if self.active_tab_idx > 0:
            input_style = BOLD if self.view == "input" else DIM
            hints_styled = (f'{help_style}? help{RESET} {DIM}·{RESET} '
                            f'{trace_style}r reasoning{RESET} {DIM}·{RESET} '
                            f'{input_style}i input{RESET} {DIM}·{RESET} '
                            f'{auto_style}a autonomous{RESET}')
            hints_len = len("? help · r reasoning · i input · a autonomous")
        else:
            hints_styled = (f'{help_style}? help{RESET} {DIM}·{RESET} '
                            f'{trace_style}r reasoning{RESET} {DIM}·{RESET} '
                            f'{wb_style}w whiteboard{RESET} {DIM}·{RESET} '
                            f'{auto_style}a autonomous{RESET}')
            hints_len = len("? help · r reasoning · w whiteboard · a autonomous")
        pad = max(w - 2 - hints_len - 1, 0)
        self._write_raw('\033[3;1H\033[2K')
        self._write_raw(f'{BLUE}│{RESET}{" " * pad}{hints_styled} {BLUE}│{RESET}')

        # Row 4 — bottom border + run dir + tab bar
        self._write_raw('\033[4;1H\033[2K')
        run_dir = getattr(self, 'work_dir', '') or ''
        tab_parts = []
        visible_len = 0
        for i, tab in enumerate(self.tabs):
            name = tab.label
            if len(name) > 20:
                name = name[:17] + "..."
            if tab.done:
                name += " ✓"
            elif self._tab_shows_spinner(tab):
                name += f" {SPINNER[tab.spinner_tick]}"
            bracket = f"[{name}]"
            visible_len += len(bracket) + 1
            if i == self.active_tab_idx:
                tab_parts.append(f'{BOLD}{WHITE}{bracket}{RESET}')
            else:
                tab_parts.append(f'{DIM}{bracket}{RESET}')
        tab_str = " ".join(tab_parts)
        if run_dir:
            # Show tabs on the left, run dir on the right
            dir_text = run_dir
            max_dir = w - visible_len - 6  # leave room for borders + tabs
            if len(dir_text) > max_dir:
                dir_text = "…" + dir_text[-(max_dir - 1):]
            fill = max(w - 2 - len(dir_text) - 1 - visible_len, 0)
            self._write_raw(
                f'{BLUE}╰{RESET} {tab_str}'
                f'{BLUE}{"─" * fill}{RESET}'
                f' {DIM}{dir_text}{RESET}{BLUE}╯{RESET}')
        else:
            fill = max(w - 2 - visible_len, 0)
            self._write_raw(f'{BLUE}╰{RESET} {tab_str}{BLUE}{"─" * fill}╯{RESET}')

    def update_step(self, step_num: int, max_steps: int):
        self.step_num = step_num
        self.max_steps = max_steps
        with self._write_lock:
            self._write_raw('\033[s')
            self._draw_header()
            self._write_raw('\033[u')
            sys.stdout.flush()

    # ── Tab management ──────────────────────────────────────────

    def add_worker_tab(self, tab_id: str, label: str, task_description: str = ""):
        tab = _Tab(tab_id, label, task_description)
        # Insert before logs tab (always last)
        if self.tabs and self.tabs[-1].id == "logs":
            self.tabs.insert(len(self.tabs) - 1, tab)
        else:
            self.tabs.append(tab)
        self._redraw_header()

    def mark_worker_done(self, tab_id: str):
        for tab in self.tabs:
            if tab.id == tab_id:
                tab.done = True
                tab.streaming = False
                tab.is_waiting = False
                break
        self._redraw_header()

    def snapshot_worker_tabs(self, step_num: int):
        """Store current worker tabs in the corresponding step entry."""
        worker_tabs = [t for t in self.tabs[1:] if t.id != "logs"]
        for entry in self.step_entries:
            if entry["step_num"] == step_num:
                entry["worker_tabs"] = worker_tabs
                break

    def set_waiting_status(self, text: str):
        """Show or clear a waiting-for-workers spinner on the planner tab."""
        planner = self.tabs[0]
        if text:
            planner.spinner_label = text
            planner.streaming = True
            planner.is_waiting = True
            planner.spinner_start = time.monotonic()
            planner.spinner_time = 0.0
            planner.spinner_tick = 0
            planner.spinner_tokens = 0
            if planner is self._active_tab and self.view == "main":
                ch = SPINNER[0]
                self._write(f'  {DIM}{ch} {text} {self._spinner_status(0, 0)}{RESET}')
        else:
            planner.streaming = False
            planner.is_waiting = False
            planner.spinner_label = ""
            if planner is self._active_tab and self.view == "main":
                self._write('\r\033[2K')
        self._redraw_header()

    def worker_output(self, tab_id: str, text: str):
        """Display worker result in its tab as regular (always-visible) content."""
        tab = self._find_tab(tab_id)
        sep_text = self._dim_separator()
        tab.log_lines.append(_LogEntry(sep_text))
        for line in text.splitlines():
            tab.log_lines.append(_LogEntry(line))
        if len(tab.log_lines) > 500:
            tab.log_lines = tab.log_lines[-500:]
        if tab is self._active_tab and self.view == "main":
            self._redraw()

    def clear_worker_tabs(self):
        """Remove all worker tabs, keeping planner and logs."""
        logs = [t for t in self.tabs if t.id == "logs"]
        self.tabs = [self.tabs[0]] + logs
        self.active_tab_idx = 0
        self._redraw_header()

    def _switch_tab(self, delta: int):
        if len(self.tabs) <= 1:
            return
        self.active_tab_idx = (self.active_tab_idx + delta) % len(self.tabs)
        self._redraw()

    def _redraw_header(self):
        with self._write_lock:
            self._write_raw('\033[s')
            self._draw_header()
            self._write_raw('\033[u')
            sys.stdout.flush()

    # ── Content area (scroll region) ───────────────────────────

    def _tab_log(self, tab: _Tab, text: str, step_idx: int = -1):
        entry = _LogEntry(text, step_idx)
        tab.log_lines.append(entry)
        if len(tab.log_lines) > 500:
            tab.log_lines = tab.log_lines[-500:]
        if tab is self._active_tab and self.view == "main":
            if tab.scroll_offset > 0:
                tab.scroll_offset = 0
                self._redraw()
            else:
                self._write(f' {text}\n')

    def log(self, text: str, color: str = "", bold: bool = False, dim: bool = False):
        self._tab_log(self.tabs[0], self._style(text, color, bold, dim))

    def tab_log(self, tab_id: str, text: str, color: str = "", dim: bool = False):
        """Log a line to a specific tab."""
        tab = self._find_tab(tab_id)
        self._tab_log(tab, self._style(text, color, dim=dim))

    def log_trace(self, text: str):
        """Log a trace message to the logs tab (dim style)."""
        tab = self._find_tab("logs")
        if tab.id == "logs":
            self._tab_log(tab, f'{DIM}{text}{RESET}')

    def show_proposal(self, plan: dict):
        planner = self.tabs[0]
        self.clear_replan_notice()
        self._proposal_log_start = len(planner.log_lines)
        sep = self._dim_separator()
        self._tab_log(planner, sep)
        self._tab_log(planner, f'{DIM}Next step:{RESET}')

        action = plan.get("action", "")
        summary = plan.get("summary", "")
        color = ACTION_STYLE.get(action, "")
        self._tab_log(planner, f'{color}▸{RESET} {BOLD}{action}{RESET} {DIM}—{RESET} {summary}')

        # Show task descriptions for spawn
        tasks = plan.get("tasks", [])
        for i, task in enumerate(tasks):
            desc = task.get("description", "").strip()
            if desc:
                lines = desc.splitlines()
                self._tab_log(planner, f'  {DIM}[{i}]{RESET} {lines[0]}')
                for line in lines[1:]:
                    self._tab_log(planner, f'      {line}')
            else:
                self._tab_log(planner, f'  {DIM}[{i}]{RESET} (no description)')

        # Show search details for literature_search
        if action == "literature_search":
            query = plan.get("search_query", "")
            context = plan.get("search_context", "")
            if query:
                self._tab_log(planner, f'  {DIM}Query:{RESET}   {query}')
            if context:
                self._tab_log(planner, f'  {DIM}Context:{RESET} {context.strip().splitlines()[0]}')
                for line in context.strip().splitlines()[1:]:
                    self._tab_log(planner, f'          {line}')

    def _format_step_line(self, entry: dict) -> str:
        action = entry.get("action", "")
        summary = entry.get("summary", "")
        color = ACTION_STYLE.get(action, "")
        line = f'{color}■{RESET} {BOLD}{action}{RESET} {DIM}—{RESET} {summary}'
        labels: list[str] = []
        feedback = (entry.get("feedback") or "").strip()
        if entry.get("rejected"):
            if feedback:
                labels.append(f"{YELLOW}rejected with feedback:{RESET} {GREEN}{feedback}{RESET}")
            else:
                labels.append(f"{YELLOW}rejected{RESET}")
        elif feedback:
            labels.append(f"{YELLOW}feedback:{RESET} {GREEN}{feedback}{RESET}")
        if entry.get("interrupted"):
            labels.append(f"{YELLOW}interrupted{RESET}")
        if labels:
            line += "\n" + "  " + f' {DIM}·{RESET} '.join(labels)
        return line

    def _sync_step_log_line(self, step_idx: int):
        if not (0 <= step_idx < len(self.step_entries)):
            return
        line = self._format_step_line(self.step_entries[step_idx])
        planner = self.tabs[0]
        for log_entry in planner.log_lines:
            if log_entry.step_idx == step_idx:
                log_entry.text = line
                break

    def step_complete(self, step_num: int, max_steps: int,
                      action: str, summary: str, detail: str = "",
                      rejected: bool = False, interrupted: bool = False,
                      feedback: str = "") -> int:
        planner = self.tabs[0]
        trace = planner.last_trace
        output = planner.last_output
        planner.last_trace = ""
        planner.last_output = ""
        idx = len(self.step_entries)
        entry = {
            "action": action,
            "summary": summary,
            "step_num": step_num,
            "detail": detail,
            "trace": trace,
            "output": output,
            "action_output": "",
            "rejected": rejected,
            "interrupted": interrupted,
            "feedback": feedback.strip(),
        }
        line = self._format_step_line(entry)
        self._tab_log(planner, line, step_idx=idx)
        self.step_entries.append(entry)
        self.update_step(step_num, max_steps)
        if self.view == "main":
            self._redraw()
        return idx

    def update_step_detail(self, step_idx: int, detail: str):
        if 0 <= step_idx < len(self.step_entries):
            self.step_entries[step_idx]["detail"] = detail

    def update_step_status(
            self,
            step_idx: int,
            *,
            rejected: bool | None = None,
            interrupted: bool | None = None,
            feedback: str | None = None,
            detail_append: str = "",
    ):
        if not (0 <= step_idx < len(self.step_entries)):
            return
        entry = self.step_entries[step_idx]
        if rejected is not None:
            entry["rejected"] = rejected
        if interrupted is not None:
            entry["interrupted"] = interrupted
        if feedback is not None:
            entry["feedback"] = feedback.strip()
        if detail_append:
            base = entry.get("detail", "")
            entry["detail"] = f"{base}\n\n{detail_append}".strip() if base else detail_append
        self._sync_step_log_line(step_idx)
        if self.view == "main":
            self._redraw()

    def append_step_action_output(self, step_num: int, text: str):
        """Append action-produced output to a step entry by step number."""
        if not text:
            return
        for entry in reversed(self.step_entries):
            if entry.get("step_num") == step_num:
                prev = entry.get("action_output", "")
                entry["action_output"] = (
                    f"{prev}\n\n{text}".strip() if prev else text
                )
                break

    # ── Spinner ─────────────────────────────────────────────────

    @staticmethod
    def _spinner_status(elapsed: int, tokens: int) -> str:
        """Format the elapsed time + token count suffix for spinner display."""
        parts = [f"{elapsed}s"]
        if tokens > 0:
            if tokens >= 1000:
                parts.append(f"{tokens / 1000:.1f}k tok")
            else:
                parts.append(f"{tokens} tok")
        return " · ".join(parts)

    def _update_spinner(self):
        tab = self._active_tab
        now = time.monotonic()
        if self.view == "main":
            ch = SPINNER[tab.spinner_tick]
            elapsed = int(now - tab.spinner_start)
            status = self._spinner_status(elapsed, tab.spinner_tokens)
            with self._write_lock:
                if (not tab.spinner_label
                        or self._has_visible_stream_content(tab)):
                    return
                self._write_raw(f'\r\033[2K  {DIM}{ch} {tab.spinner_label} {status}{RESET}')
                sys.stdout.flush()

    # ── Streaming ───────────────────────────────────────────────

    def stream_start(self, label: str = "thinking", tab: str = "planner"):
        target = self._find_tab_or_none(tab)
        if target is None:
            return
        target.trace_buf = []
        target.output_buf = []
        target.toml_pending = ""
        target.toml_close_tag = ""
        target.output_non_toml_seen = False
        target.output_toml_seen = False
        target.streaming = True
        target.is_waiting = False
        target.done = False
        target.spinner_label = label
        target.spinner_tick = 0
        target.spinner_tokens = 0
        target.spinner_time = time.monotonic()
        target.spinner_start = target.spinner_time
        self._redraw_header()
        if target is self._active_tab and self.view == "main":
            self._write(f'  {DIM}{SPINNER[0]} {label} {self._spinner_status(0, 0)}{RESET}')

    def stream_text(self, text: str, kind: str = "text", tab: str = "planner"):
        self._check_keys()
        target = self._find_tab_or_none(tab)
        if target is None:
            return
        # Ignore stale chunks that arrive after a stream has ended.
        if not target.streaming:
            return
        # Planner should never emit model chunks while just waiting for workers.
        if target.id == "planner" and target.is_waiting:
            return
        target.spinner_tokens += 1
        is_active = target is self._active_tab
        at_bottom = target.scroll_offset == 0
        is_thinking = kind == "thinking"

        # Was there visible content before this chunk?
        had_visible = self._has_visible_stream_content(target)
        had_visible_output = (
            target.output_non_toml_seen
            or (target.output_toml_seen and self.trace_visible)
        )

        output_segments: list[tuple[bool, str]] = []
        output_shown = False

        if is_thinking:
            target.trace_buf.append(text)
        else:
            target.output_buf.append(text)
            output_segments = self._split_toml_stream_segments(target, text)
            for is_toml, seg in output_segments:
                if not seg:
                    continue
                if is_toml:
                    target.output_toml_seen = True
                    if self.trace_visible:
                        output_shown = True
                else:
                    target.output_non_toml_seen = True
                    output_shown = True

        has_visible = self._has_visible_stream_content(target)

        # Clear spinner on first visible content
        if (not had_visible and has_visible
                and self.view == "main" and is_active and at_bottom):
            with self._write_lock:
                self._write_raw('\r\033[2K')
                sys.stdout.flush()

        trace_needs_newline = (
            output_shown
            and not had_visible_output
            and self.trace_visible
            and bool(target.trace_buf)
            and not target.trace_buf[-1].endswith("\n")
        )

        should_display = (is_thinking and self.trace_visible) or (not is_thinking and output_shown)
        if should_display and self.view == "main" and is_active and at_bottom:
            if trace_needs_newline and self.trace_visible:
                self._write("\n")
            if is_thinking:
                self._write(f'{DIM}{text}{RESET}')
            else:
                for is_toml, seg in output_segments:
                    if not seg:
                        continue
                    if is_toml:
                        if self.trace_visible:
                            self._write(f'{DIM}{seg}{RESET}')
                    else:
                        self._write(seg)

    def stream_end(self, tab: str = "planner"):
        target = self._find_tab_or_none(tab)
        if target is None:
            return
        target.streaming = False
        target.is_waiting = False
        target.spinner_label = ""

        if target.trace_buf:
            target.last_trace = "".join(target.trace_buf)
            target.log_lines.append(_LogEntry(target.last_trace, is_trace=True))
        else:
            target.last_trace = ""

        if target.output_buf:
            target.last_output = "".join(target.output_buf)
            target.log_lines.append(
                _LogEntry(target.last_output, is_output=True))
        else:
            target.last_output = ""

        if len(target.log_lines) > 500:
            target.log_lines = target.log_lines[-500:]

        target.trace_buf = []
        target.output_buf = []

        is_active = target is self._active_tab
        had_visible = ((target.last_trace and self.trace_visible)
                       or target.last_output)
        if is_active and self.view == "main":
            if target.scroll_offset > 0:
                self._redraw()
            elif had_visible:
                self._write('\n')
            else:
                self._write('\r\033[2K')
        self._redraw_header()

    # ── Background thread (key reader + spinner) ────────────────

    def _bg_loop(self):
        fd = sys.stdin.fileno()
        while not self._bg_stop:
            try:
                self._advance_tab_spinners()
                tab = self._active_tab
                if tab.spinner_label and tab.streaming:
                    self._update_spinner()

                # Process queued keys when idle (no streaming, no confirmation)
                if not self._confirming and not tab.streaming:
                    if not self._key_queue.empty():
                        self._check_keys()

                if not select.select([fd], [], [], 0.04)[0]:
                    continue
                data = os.read(fd, 32)
                if not data:
                    continue

                i = 0
                while i < len(data):
                    b = data[i]
                    if b == 0x1b:
                        if i + 2 < len(data) and data[i + 1] == 0x5b:
                            # SGR mouse: \033[<Btn;Col;RowM/m
                            if data[i + 2] == 0x3c:
                                j = i + 3
                                while j < len(data) and data[j] not in (0x4d, 0x6d):
                                    j += 1
                                if j < len(data):
                                    try:
                                        params = data[i+3:j].decode('ascii')
                                        btn = int(params.split(';')[0])
                                        if btn in (64, 65):
                                            key = 'scroll_up' if btn == 64 else 'scroll_down'
                                            if self._can_handle_directly():
                                                self._process_key(key)
                                            else:
                                                self._key_queue.put(key)
                                    except (ValueError, IndexError):
                                        pass
                                    i = j + 1
                                else:
                                    i = len(data)
                                continue
                            # Check for CSI N ~ sequences (Page Up/Down etc.)
                            if (i + 3 < len(data)
                                    and 0x30 <= data[i + 2] <= 0x39
                                    and data[i + 3] == 0x7e):
                                seq = chr(0x1b) + '[' + chr(data[i + 2]) + '~'
                                if self._can_handle_directly() and seq in ('\x1b[5~', '\x1b[6~'):
                                    self._process_key(seq)
                                else:
                                    self._key_queue.put(seq)
                                i += 4
                                continue
                            # Arrow keys — handle directly during streaming
                            arrow = chr(data[i + 2])
                            seq = chr(0x1b) + '[' + arrow
                            if self._can_handle_directly() and arrow in ('A', 'B', 'C', 'D'):
                                self._process_key(seq)
                                i += 3
                                continue
                            self._key_queue.put(seq)
                            i += 3
                            continue
                        if self._can_handle_directly() and self.view != "main":
                            self.view = "main"
                            self._redraw()
                        else:
                            self._key_queue.put('\x1b')
                        i += 1
                        continue

                    ch = chr(b)
                    if self._can_handle_directly():
                        if ch in ('r', 'i', 'w', '?', 'a'):
                            self._process_key(ch)
                            i += 1
                            continue
                        if ch in ('\n', '\r') and (
                            self.view != "main"
                            or self._active_tab.scroll_offset > 0
                        ):
                            self._process_key(ch)
                            i += 1
                            continue

                    self._key_queue.put(ch)
                    i += 1
            except (OSError, ValueError):
                break

    @staticmethod
    def _tab_shows_spinner(tab: _Tab) -> bool:
        return tab.streaming and not (tab.id == "planner" and tab.is_waiting)

    def _advance_tab_spinners(self):
        now = time.monotonic()
        updated = False
        for tab in self.tabs:
            if not self._tab_shows_spinner(tab):
                continue
            if now - tab.spinner_time < 0.08:
                continue
            tab.spinner_time = now
            tab.spinner_tick = (tab.spinner_tick + 1) % len(SPINNER)
            updated = True
        if updated:
            self._redraw_header()

    def _can_handle_directly(self) -> bool:
        return self._active_tab.streaming and not self._confirming

    # ── Key handling ────────────────────────────────────────────

    def _check_keys(self):
        if not self._key_process_lock.acquire(blocking=False):
            return
        try:
            while True:
                try:
                    ch = self._key_queue.get_nowait()
                except queue.Empty:
                    break
                self._process_key(ch)
        finally:
            self._key_process_lock.release()

    def _process_key(self, ch: str):
        if ch == 'r':
            self._toggle_trace()
        elif ch == 'i':
            if self.active_tab_idx > 0:
                self._toggle_view("input")
        elif ch == 'w':
            if self.active_tab_idx == 0:
                self._toggle_view("whiteboard")
        elif ch == '?':
            self._toggle_view("help")
        elif ch == 'a':
            self.autonomous = not self.autonomous
            self._redraw()
        elif ch == '\x1b[C':  # right arrow — next tab
            self._switch_tab(1)
        elif ch == '\x1b[D':  # left arrow — prev tab
            self._switch_tab(-1)
        elif ch == '\x1b[5~':
            self._scroll_up()
        elif ch == '\x1b[6~':
            self._scroll_down()
        elif ch == '\x1b[A':  # up arrow — planner history only
            if self.view == "main" and self.active_tab_idx == 0:
                if self.step_entries:
                    self._nav_up()
                    self._redraw()
            else:
                self._scroll_lines_up()
        elif ch == '\x1b[B':  # down arrow — planner history only
            if self.view == "main" and self.active_tab_idx == 0:
                if self.step_entries:
                    self._nav_down()
                    self._redraw()
            else:
                self._scroll_lines_down()
        elif ch == 'scroll_up':
            self._scroll_lines_up()
        elif ch == 'scroll_down':
            self._scroll_lines_down()
        elif ch == '\x1b':
            if self.view != "main":
                self.view = "main"
                self._redraw()
            elif self._nav_step >= 0:
                self._nav_step = -1
                self._restore_worker_tabs()
                self._redraw()
        elif ch in ('\n', '\r'):
            if self.view != "main":
                self.view = "main"
                self._redraw()
            elif self._nav_step >= 0:
                self._open_selected_step_detail()
                self._redraw()
            elif self._active_tab.scroll_offset > 0:
                self._active_tab.scroll_offset = 0
                self._redraw()
        elif self.autonomous and ch in ('q', 'p', 's'):
            self.pending_action = {
                'q': 'quit', 'p': 'pause',
                's': 'summarize',
            }[ch]

    def _open_selected_step_detail(self):
        if self._nav_step < 0:
            return
        self._step_detail_idx = self._nav_step
        self._step_detail_scroll = 0
        self._refresh_step_detail()
        self.view = "step_detail"

    @staticmethod
    def _step_detail_section_title(action: str) -> str:
        return {
            "spawn": "Worker Plan",
            "literature_search": "Literature Search",
            "read_items": "Reading Notes",
            "write_items": "Writing Notes",
            "read_theorem": "Theorem Analysis",
            "submit_proof": "Submission",
            "submit_lean_proof": "Lean Submission",
            "give_up": "Termination",
        }.get(action, "Step Details")

    def _refresh_step_detail(self):
        if not (0 <= self._step_detail_idx < len(self.step_entries)):
            self._step_detail_title = "Step Detail"
            self._step_detail_text = "(no detail)"
            return

        entry = self.step_entries[self._step_detail_idx]
        action = entry.get("action", "")
        summary = entry.get("summary", "")
        action_color = ACTION_STYLE.get(action, WHITE)
        self._step_detail_title = (
            f"Step {entry.get('step_num', '?')}: "
            f"{action_color}{action}{RESET} {DIM}—{RESET} {summary}"
        )

        parts: list[str] = []

        def add_section(title: str, lines: list[str], color: str = BLUE):
            if not lines:
                return
            if parts:
                parts.append(f"  {DIM}{'─' * 40}{RESET}")
                parts.append("")
            parts.append(f"  {color}{BOLD}{title}{RESET}")
            for line in lines:
                parts.append(f"  {line}" if line else "")

        feedback = (entry.get("feedback") or "").strip()
        status_lines: list[str] = []
        if entry.get("rejected"):
            if feedback:
                status_lines.append(
                    f"{YELLOW}● rejected with feedback:{RESET} {GREEN}{feedback}{RESET}"
                )
            else:
                status_lines.append(f"{YELLOW}● rejected{RESET}")
        elif feedback:
            status_lines.append(f"{YELLOW}● feedback:{RESET} {GREEN}{feedback}{RESET}")
        if entry.get("interrupted"):
            status_lines.append(f"{YELLOW}● execution interrupted{RESET}")
        if not status_lines:
            status_lines.append(f"{GREEN}● completed{RESET}")
        add_section("Status", status_lines, color=YELLOW)

        detail = (entry.get("detail") or "").strip()
        if detail:
            add_section(self._step_detail_section_title(action), detail.splitlines(),
                        color=CYAN)

        output_lines: list[str] = []
        output = (entry.get("output") or "").rstrip()
        if output:
            for is_toml, segment in self._iter_toml_segments(output):
                if is_toml and not self.trace_visible:
                    continue
                for line in segment.splitlines():
                    output_lines.append(f"{DIM}{line}{RESET}" if is_toml else line)
        if output_lines:
            add_section("Planner Output", output_lines, color=BLUE)

        trace = (entry.get("trace") or "").rstrip()
        if trace:
            if self.trace_visible:
                add_section("Reasoning", [f"{DIM}{line}{RESET}"
                                          for line in trace.splitlines()],
                            color=GREEN)
            else:
                add_section("Reasoning",
                            [f"{DIM}hidden (press r to show){RESET}"],
                            color=GREEN)

        if action == "spawn":
            worker_sections: list[str] = []
            worker_tabs = entry.get("worker_tabs") or []
            for tab in worker_tabs:
                label = getattr(tab, "label", "").strip() or "Worker"
                task_description = getattr(tab, "task_description", "").strip()
                result_lines: list[str] = []
                log_lines = getattr(tab, "log_lines", []) or []
                for log_entry in log_lines:
                    text = getattr(log_entry, "text", "")
                    if not text or text == self._dim_separator():
                        continue
                    result_lines.append(text)
                if not result_lines:
                    continue
                worker_sections.append(f"{BOLD}{label}{RESET}")
                if task_description:
                    worker_sections.append(f"{DIM}task:{RESET} {task_description}")
                worker_sections.extend(result_lines)
                worker_sections.append("")
            if worker_sections:
                # Trim trailing blank line between worker blocks.
                if worker_sections[-1] == "":
                    worker_sections.pop()
                add_section("Worker Outputs", worker_sections, color=MAGENTA)

        action_output = (entry.get("action_output") or "").rstrip()
        if action_output and action != "spawn":
            add_section("Action Output", action_output.splitlines(), color=MAGENTA)

        self._step_detail_text = "\n".join(parts) if parts else "  (no detail)"
        self._step_detail_scroll = min(
            self._step_detail_scroll,
            self._step_detail_max_scroll(),
        )

    def get_pending_action(self) -> str | None:
        self._check_keys()
        action = self.pending_action
        self.pending_action = None
        return action

    def interrupt(self):
        """Cancel any pending confirmation by injecting ctrl+c into key queue."""
        self._key_queue.put('\x03')

    # ── Confirmation UI ────────────────────────────────────────

    def get_confirmation(self) -> str:
        self._confirming = True
        self._confirm_accept_label = "accept"
        self._confirm_selected = 0
        self._confirm_buf = []
        self._nav_step = -1
        self._scroll_selection_into_view()
        self._redraw()
        self._write('\033[?25h')

        try:
            while True:
                try:
                    ch = self._key_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if ch == '\x1b':
                    if self.view != "main":
                        self.view = "main"
                    elif self._nav_step >= 0:
                        self._nav_step = -1
                        self._restore_worker_tabs()
                        self._scroll_selection_into_view()
                    else:
                        self._confirm_buf.clear()
                    self._redraw()
                    continue

                if ch in ('scroll_up', 'scroll_down'):
                    if ch == 'scroll_up':
                        self._scroll_lines_up()
                    else:
                        self._scroll_lines_down()
                    continue

                if len(ch) >= 3 and ch[:2] == '\x1b[':
                    if ch == '\x1b[A':
                        if self._nav_step == -1:
                            if self._confirm_selected == 0 and self.step_entries:
                                self._nav_up()
                            else:
                                self._confirm_selected = 0
                                self._scroll_selection_into_view()
                        else:
                            self._nav_up()
                        self._redraw()
                    elif ch == '\x1b[B':
                        if self._nav_step == -1:
                            self._confirm_selected = 1 - self._confirm_selected
                            self._scroll_selection_into_view()
                        else:
                            self._nav_down()
                        self._redraw()
                    elif ch == '\x1b[C':
                        self._switch_tab(1)
                    elif ch == '\x1b[D':
                        self._switch_tab(-1)
                    elif ch == '\x1b[5~':
                        self._scroll_up()
                    elif ch == '\x1b[6~':
                        self._scroll_down()
                    continue

                if ch in ('\n', '\r'):
                    if self.view != "main":
                        self.view = "main"
                        self._redraw()
                        continue
                    if self._nav_step >= 0:
                        self._open_selected_step_detail()
                        self._redraw()
                        continue
                    if self._confirm_selected == 0:
                        return ""
                    return "".join(self._confirm_buf)

                if ch in ('\x7f', '\x08'):
                    if self._confirm_selected == 1 and self._confirm_buf:
                        self._confirm_buf.pop()
                        self._redraw()
                    continue

                if ch in ('\x03', '\x04'):
                    return "q"

                if ch == '\t':
                    if self._nav_step >= 0:
                        self._nav_step = -1
                        self._restore_worker_tabs()
                        self._scroll_selection_into_view()
                    else:
                        self._confirm_selected = 1 - self._confirm_selected
                    self._redraw()
                    continue

                can_toggle = (self._confirm_selected == 0 or not self._confirm_buf)
                if can_toggle and ch in ('r', 'i', 'w', '?'):
                    self._process_key(ch)
                    continue

                if (self._nav_step == -1 and self._confirm_selected == 0
                        and ch in ('s', 'p', 'q')):
                    return ch

                if (ch == 'a' and self._nav_step == -1
                        and self._confirm_selected == 0):
                    self.autonomous = True
                    self._redraw()
                    return "a"

                if ch.isprintable():
                    if self._nav_step >= 0:
                        self._nav_step = -1
                        self._restore_worker_tabs()
                    if self._confirm_selected == 0:
                        self._confirm_selected = 1
                    self._confirm_buf.append(ch)
                    self._redraw()

        finally:
            self._confirming = False
            self._nav_step = -1
            self._restore_worker_tabs()
            self._clear_proposal()
            self._redraw()
            self._write('\033[?25l')

    def _clear_proposal(self):
        """Remove detailed proposal lines, keeping only a compact summary."""
        if self._proposal_log_start < 0:
            return
        planner = self.tabs[0]
        planner.log_lines = planner.log_lines[:self._proposal_log_start]
        self._proposal_log_start = -1

    def show_replan_notice(self, text: str):
        """Show a transient notice cleared when next proposal appears."""
        planner = self.tabs[0]
        self.clear_replan_notice()
        entry = _LogEntry(self._style(text, color="yellow"))
        planner.log_lines.append(entry)
        if len(planner.log_lines) > 500:
            planner.log_lines = planner.log_lines[-500:]
            if entry not in planner.log_lines:
                self._replan_notice_entry = None
            else:
                self._replan_notice_entry = entry
        else:
            self._replan_notice_entry = entry
        if planner is self._active_tab and self.view == "main":
            if planner.scroll_offset > 0:
                planner.scroll_offset = 0
                self._redraw()
            else:
                self._write(f' {entry.text}\n')

    def clear_replan_notice(self):
        entry = self._replan_notice_entry
        if entry is None:
            return
        planner = self.tabs[0]
        try:
            planner.log_lines.remove(entry)
        except ValueError:
            pass
        self._replan_notice_entry = None
        if planner is self._active_tab and self.view == "main":
            self._redraw()

    def show_interrupt_options(self):
        """Show continue / give feedback after CTRL+C interruption."""
        # Drain any stale keys from the queue (including the ctrl+c itself)
        while not self._key_queue.empty():
            try:
                self._key_queue.get_nowait()
            except queue.Empty:
                break

    def get_interrupt_response(self) -> str:
        """Block for user input: '' = continue, 'q' = quit, else = feedback text.

        Feedback is selected by default (unlike get_confirmation where accept is default).
        """
        self._confirming = True
        self._confirm_accept_label = "continue"
        self._confirm_selected = 1  # feedback selected by default
        self._confirm_buf = []
        self._nav_step = -1
        self._redraw()
        self._write('\033[?25h')

        try:
            while True:
                try:
                    ch = self._key_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if ch == '\x1b':
                    if self.view != "main":
                        self.view = "main"
                    else:
                        self._confirm_buf.clear()
                    self._redraw()
                    continue

                if ch in ('scroll_up', 'scroll_down'):
                    if ch == 'scroll_up':
                        self._scroll_lines_up()
                    else:
                        self._scroll_lines_down()
                    continue

                if len(ch) >= 3 and ch[:2] == '\x1b[':
                    if ch == '\x1b[C':
                        self._switch_tab(1)
                    elif ch == '\x1b[D':
                        self._switch_tab(-1)
                    elif ch == '\x1b[5~':
                        self._scroll_up()
                    elif ch == '\x1b[6~':
                        self._scroll_down()
                    continue

                if ch in ('\n', '\r'):
                    if self.view != "main":
                        self.view = "main"
                        self._redraw()
                        continue
                    if self._confirm_selected == 0:
                        return ""  # continue
                    return "".join(self._confirm_buf)

                if ch in ('\x7f', '\x08'):
                    if self._confirm_selected == 1 and self._confirm_buf:
                        self._confirm_buf.pop()
                        self._redraw()
                    continue

                if ch == '\x03':
                    # Second ctrl+c during interrupt prompt → quit
                    return "q"

                if ch == '\t':
                    self._confirm_selected = 1 - self._confirm_selected
                    self._redraw()
                    continue

                can_toggle = (self._confirm_selected == 0 or not self._confirm_buf)
                if can_toggle and ch in ('r', 'w', '?'):
                    self._process_key(ch)
                    continue

                if self._confirm_selected == 0 and ch == 'q':
                    return "q"

                if ch.isprintable():
                    if self._confirm_selected == 0:
                        self._confirm_selected = 1
                    self._confirm_buf.append(ch)
                    self._redraw()

        finally:
            self._confirming = False
            self._redraw()
            self._write('\033[?25l')

    def browse(self):
        """Interactive browse mode for inspect. Blocks until user presses q."""
        self._confirming = True  # prevent bg_loop from stealing keys
        self._browsing = True
        self._nav_step = -1
        self._redraw()

        try:
            while True:
                try:
                    ch = self._key_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if ch in ('q', '\x03', '\x04'):
                    break

                if ch in ('scroll_up', 'scroll_down'):
                    if ch == 'scroll_up':
                        self._scroll_lines_up()
                    else:
                        self._scroll_lines_down()
                    continue

                if len(ch) >= 3 and ch[:2] == '\x1b[':
                    if ch == '\x1b[A':
                        self._nav_up()
                        self._redraw()
                    elif ch == '\x1b[B':
                        self._nav_down()
                        self._redraw()
                    elif ch == '\x1b[C':
                        self._switch_tab(1)
                    elif ch == '\x1b[D':
                        self._switch_tab(-1)
                    elif ch == '\x1b[5~':
                        self._scroll_up()
                    elif ch == '\x1b[6~':
                        self._scroll_down()
                    continue

                if ch == '\x1b':
                    if self.view != "main":
                        self.view = "main"
                        self._redraw()
                    elif self._nav_step >= 0:
                        self._nav_step = -1
                        self._restore_worker_tabs()
                        self._redraw()
                    continue

                if ch in ('\n', '\r'):
                    if self.view != "main":
                        self.view = "main"
                        self._redraw()
                    elif self._nav_step >= 0:
                        self._open_selected_step_detail()
                        self._redraw()
                    elif self._active_tab.scroll_offset > 0:
                        self._active_tab.scroll_offset = 0
                        self._redraw()
                    continue

                self._process_key(ch)

        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self._confirming = False
            self._browsing = False
            self._nav_step = -1
            self._restore_worker_tabs()

    def _nav_up(self):
        if self._nav_step == -1:
            if self.step_entries:
                # Save current worker tabs before entering history (exclude logs)
                self._saved_worker_tabs = [t for t in self.tabs[1:] if t.id != "logs"]
                self._nav_step = len(self.step_entries) - 1
                self._load_historical_workers()
                self._scroll_selection_into_view()
        elif self._nav_step > 0:
            self._nav_step -= 1
            self._load_historical_workers()
            self._scroll_selection_into_view()

    def _nav_down(self):
        if self._nav_step >= 0:
            if self._nav_step < len(self.step_entries) - 1:
                self._nav_step += 1
                self._load_historical_workers()
                self._scroll_selection_into_view()
            else:
                self._nav_step = -1
                self._restore_worker_tabs()
                self._scroll_selection_into_view()

    def _load_historical_workers(self):
        """Load worker tabs from the selected historical step."""
        if self._nav_step < 0:
            return
        entry = self.step_entries[self._nav_step]
        historical = entry.get("worker_tabs", [])
        logs = [t for t in self.tabs if t.id == "logs"]
        self.tabs = [self.tabs[0]] + list(historical) + logs
        if self.active_tab_idx >= len(self.tabs):
            self.active_tab_idx = 0

    def _restore_worker_tabs(self):
        """Restore current worker tabs after leaving history."""
        if self._saved_worker_tabs is not None:
            logs = [t for t in self.tabs if t.id == "logs"]
            self.tabs = [self.tabs[0]] + self._saved_worker_tabs + logs
            self._saved_worker_tabs = None
            if self.active_tab_idx >= len(self.tabs):
                self.active_tab_idx = 0

    def _main_avail_rows(self, tab: _Tab | None = None) -> int:
        if tab is None:
            tab = self._active_tab
        cs = self._content_start
        confirm_rows = 3 if (self._confirming and not self._browsing and self.active_tab_idx == 0) else 0
        spinner_active = (tab.streaming and tab.spinner_label
                          and not (tab.trace_buf and self.trace_visible))
        spinner_rows = 1 if spinner_active else 0
        return max(self.rows - cs + 1 - confirm_rows - spinner_rows, 1)

    def _entry_render_lines(self, tab: _Tab, entry: _LogEntry, max_w: int) -> int:
        if entry.is_trace:
            if not self.trace_visible:
                return 0
            src = entry.text.splitlines() or [""]
            return sum(
                len(self._wrap_visual_text(f'  {DIM}{line}{RESET}', max_w))
                for line in src
            )
        if entry.is_output:
            src = entry.text.splitlines() or [""]
            return sum(
                len(self._wrap_visual_text(f'  {line}', max_w))
                for line in src
            )
        base = f' {entry.text}'
        continuation = " " * self._leading_visible_spaces(base)
        return len(self._wrap_visual_text(
            base, max_w, continuation_prefix=continuation
        ))

    @staticmethod
    def _planner_live_start(tab: _Tab) -> int:
        if tab.id != "planner":
            return 0
        last_step = -1
        for idx, entry in enumerate(tab.log_lines):
            if entry.step_idx >= 0:
                last_step = idx
        return last_step + 1

    @staticmethod
    def _wrap_visual_text(
            text: str, max_w: int, continuation_prefix: str = "") -> list[str]:
        """Wrap text by visible width while preserving ANSI sequences."""
        if max_w <= 0:
            return [text]
        cont = continuation_prefix
        cont_w = len(cont)
        if cont_w >= max_w:
            cont = ""
            cont_w = 0
        parts: list[str] = []
        buf: list[str] = []
        visible = 0
        i = 0
        n = len(text)
        while i < n:
            if text[i] == '\x1b':
                m = re.match(r'\x1b\[[0-9;?]*[ -/]*[@-~]', text[i:])
                if m:
                    buf.append(m.group(0))
                    i += len(m.group(0))
                    continue
            ch = text[i]
            buf.append(ch)
            i += 1
            visible += 1
            if visible >= max_w:
                parts.append("".join(buf))
                if i < n and cont:
                    buf = [cont]
                    visible = cont_w
                else:
                    buf = []
                    visible = 0
        if buf or not parts:
            parts.append("".join(buf))
        return parts

    @staticmethod
    def _leading_visible_spaces(text: str) -> int:
        """Count visible leading spaces while ignoring ANSI escapes."""
        i = 0
        n = len(text)
        spaces = 0
        while i < n:
            if text[i] == '\x1b':
                m = re.match(r'\x1b\[[0-9;?]*[ -/]*[@-~]', text[i:])
                if m:
                    i += len(m.group(0))
                    continue
            if text[i] != " ":
                break
            spaces += 1
            i += 1
        return spaces

    def _selection_render_range(self, tab: _Tab) -> tuple[int, int] | None:
        max_w = max(self.cols - 4, 20)
        planner_live_start = self._planner_live_start(tab)
        line_idx = 0
        if self._nav_step >= 0:
            for idx, entry in enumerate(tab.log_lines):
                if (tab.id == "planner"
                        and (entry.is_trace or entry.is_output)
                        and idx < planner_live_start):
                    continue
                rendered = self._entry_render_lines(tab, entry, max_w)
                if rendered <= 0:
                    continue
                if entry.step_idx == self._nav_step:
                    return (line_idx, line_idx + rendered - 1)
                line_idx += rendered
            return None
        if not (self._confirming and tab.id == "planner" and self._proposal_log_start >= 0):
            return None
        start = None
        for idx, entry in enumerate(tab.log_lines):
            if (tab.id == "planner"
                    and (entry.is_trace or entry.is_output)
                    and idx < planner_live_start):
                continue
            rendered = self._entry_render_lines(tab, entry, max_w)
            if rendered <= 0:
                continue
            if idx >= self._proposal_log_start and start is None:
                start = line_idx
            line_idx += rendered
        if start is None:
            return None
        return (start, max(line_idx - 1, start))

    def _scroll_selection_into_view(self):
        if self.view != "main":
            return
        tab = self._active_tab
        lines = self._build_main_lines(tab)
        if not lines:
            tab.scroll_offset = 0
            return
        sel = self._selection_render_range(tab)
        if sel is None:
            return

        avail = self._main_avail_rows(tab)
        total = len(lines)
        max_off = max(total - avail, 0)
        if tab.scroll_offset > max_off:
            tab.scroll_offset = max_off
        end = total - tab.scroll_offset
        start = max(end - avail, 0)
        target_start, target_end = sel

        if target_start < start:
            new_end = min(total, target_start + avail)
            tab.scroll_offset = max(total - new_end, 0)
        elif target_end >= end:
            tab.scroll_offset = max(total - (target_end + 1), 0)

        if tab.scroll_offset > max_off:
            tab.scroll_offset = max_off

    def _draw_confirmation(self):
        fb = "".join(self._confirm_buf)
        lbl = self._confirm_accept_label
        self._write_raw('\n')
        if self._nav_step >= 0:
            self._write_raw(f' {DIM}○ {lbl}{RESET}\n')
            self._write_raw(f' {DIM}○ give feedback{RESET}')
        elif self._confirm_selected == 0:
            self._write_raw(f' {GREEN}●{RESET} {BOLD}{lbl}{RESET}\n')
            self._write_raw(f' {DIM}○ give feedback{RESET}')
        else:
            self._write_raw(f' {DIM}○ {lbl}{RESET}\n')
            self._write_raw(f' {GREEN}●{RESET} {fb}')

    # ── View toggles ────────────────────────────────────────────

    def _toggle_trace(self):
        if self.view == "step_detail":
            old_lines = self._build_step_detail_lines()
            old_max = max(len(old_lines) - self._step_detail_avail_rows(), 0)
            old_ratio = (self._step_detail_scroll / old_max) if old_max > 0 else 0.0
            self.trace_visible = not self.trace_visible
            self._refresh_step_detail()
            new_lines = self._build_step_detail_lines()
            new_max = max(len(new_lines) - self._step_detail_avail_rows(), 0)
            if old_max > 0 and new_max > 0:
                self._step_detail_scroll = int(round(old_ratio * new_max))
            else:
                self._step_detail_scroll = min(self._step_detail_scroll, new_max)
            self._redraw()
            return

        tab = self._active_tab
        old_total = len(self._build_main_lines(tab))
        old_max_off = max(old_total - self._main_avail_rows(tab), 0)
        old_ratio = (tab.scroll_offset / old_max_off) if old_max_off > 0 else 0.0
        self.trace_visible = not self.trace_visible
        new_total = len(self._build_main_lines(tab))
        new_max_off = max(new_total - self._main_avail_rows(tab), 0)
        if old_max_off > 0 and new_max_off > 0:
            tab.scroll_offset = int(round(old_ratio * new_max_off))
        else:
            tab.scroll_offset = min(tab.scroll_offset, new_max_off)
        self._scroll_selection_into_view()
        self._redraw()

    def _toggle_view(self, target: str):
        self.view = "main" if self.view == target else target
        self._redraw()

    # ── Scrolling ────────────────────────────────────────────────

    def _scroll_up(self):
        if self.view == "step_detail":
            page = max(self._step_detail_avail_rows() - 1, 1)
            self._step_detail_scroll = max(self._step_detail_scroll - page, 0)
            self._redraw()
            return
        tab = self._active_tab
        page = max(self._main_avail_rows(tab) - 1, 1)
        lines = self._build_main_lines(tab)
        max_off = max(len(lines) - self._main_avail_rows(tab), 0)
        tab.scroll_offset = min(tab.scroll_offset + page, max_off)
        self._redraw()

    def _scroll_down(self):
        if self.view == "step_detail":
            page = max(self._step_detail_avail_rows() - 1, 1)
            max_scroll = self._step_detail_max_scroll()
            self._step_detail_scroll = min(self._step_detail_scroll + page, max_scroll)
            self._redraw()
            return
        tab = self._active_tab
        page = max(self._main_avail_rows(tab) - 1, 1)
        tab.scroll_offset = max(tab.scroll_offset - page, 0)
        self._redraw()

    def _scroll_lines_up(self, n: int = 3):
        if self.view == "step_detail":
            self._step_detail_scroll = max(self._step_detail_scroll - n, 0)
            self._redraw()
            return
        tab = self._active_tab
        lines = self._build_main_lines(tab)
        max_off = max(len(lines) - self._main_avail_rows(tab), 0)
        tab.scroll_offset = min(tab.scroll_offset + n, max_off)
        self._redraw()

    def _scroll_lines_down(self, n: int = 3):
        if self.view == "step_detail":
            max_scroll = self._step_detail_max_scroll()
            self._step_detail_scroll = min(self._step_detail_scroll + n, max_scroll)
            self._redraw()
            return
        tab = self._active_tab
        tab.scroll_offset = max(tab.scroll_offset - n, 0)
        self._redraw()

    def _build_main_lines(self, tab: _Tab | None = None) -> list[str]:
        """Build flat list of rendered lines for the active tab."""
        if tab is None:
            tab = self._active_tab
        lines: list[str] = []
        max_w = max(self.cols - 4, 20)
        planner_live_start = self._planner_live_start(tab)
        for idx, entry in enumerate(tab.log_lines):
            if entry.is_trace:
                if tab.id == "planner" and idx < planner_live_start:
                    continue
                if not self.trace_visible:
                    continue
                for tline in entry.text.splitlines():
                    text = f'  {DIM}{tline}{RESET}'
                    for wrapped in self._wrap_visual_text(text, max_w):
                        lines.append(wrapped)
                if not entry.text.splitlines():
                    text = f'  {DIM}{RESET}'
                    for wrapped in self._wrap_visual_text(text, max_w):
                        lines.append(wrapped)
            elif entry.is_output:
                if tab.id == "planner" and idx < planner_live_start:
                    continue
                output_text = entry.text
                if (tab.id == "planner"
                        and self._confirming
                        and self._proposal_log_start >= 0):
                    output_text = self._strip_toml_block(output_text)
                rendered_any = False
                for is_toml, seg in self._iter_toml_segments(output_text):
                    if not seg:
                        continue
                    if is_toml and not self.trace_visible:
                        continue
                    for tline in seg.splitlines():
                        text = f'  {DIM}{tline}{RESET}' if is_toml else f'  {tline}'
                        for wrapped in self._wrap_visual_text(text, max_w):
                            lines.append(wrapped)
                    if not seg.splitlines():
                        text = f'  {DIM}{RESET}' if is_toml else '  '
                        for wrapped in self._wrap_visual_text(text, max_w):
                            lines.append(wrapped)
                    rendered_any = True
                if not rendered_any:
                    continue
            else:
                is_step = entry.step_idx >= 0
                base = f' {entry.text}'
                continuation = " " * self._leading_visible_spaces(base)
                wrapped_lines = self._wrap_visual_text(
                    base, max_w, continuation_prefix=continuation
                )
                if is_step and entry.step_idx == self._nav_step:
                    for wrapped in wrapped_lines:
                        lines.append(f' {GREEN}▎{RESET}{wrapped}')
                else:
                    lines.extend(wrapped_lines)
        # Active streaming content (not yet baked)
        if tab.streaming:
            if tab.trace_buf and self.trace_visible:
                joined = "".join(tab.trace_buf)
                for tline in joined.splitlines():
                    while len(tline) > max_w:
                        lines.append(f'  {DIM}{tline[:max_w]}{RESET}')
                        tline = tline[max_w:]
                    lines.append(f'  {DIM}{tline}{RESET}')
            if tab.output_buf:
                joined = "".join(tab.output_buf)
                for is_toml, seg in self._iter_toml_segments(joined):
                    if not seg:
                        continue
                    if is_toml and not self.trace_visible:
                        continue
                    for tline in seg.splitlines():
                        while len(tline) > max_w:
                            if is_toml:
                                lines.append(f'  {DIM}{tline[:max_w]}{RESET}')
                            else:
                                lines.append(f'  {tline[:max_w]}')
                            tline = tline[max_w:]
                        if is_toml:
                            lines.append(f'  {DIM}{tline}{RESET}')
                        else:
                            lines.append(f'  {tline}')
        return lines

    def _build_step_detail_lines(self) -> list[str]:
        max_w = max(self.cols - 2, 20)
        lines: list[str] = []
        for dline in self._step_detail_text.splitlines() or [""]:
            lines.extend(self._wrap_visual_text(dline, max_w))
        return lines

    def _step_detail_avail_rows(self) -> int:
        # One title line + one separator line.
        return max(self.rows - self._content_start + 1 - 2, 1)

    def _step_detail_max_scroll(self) -> int:
        return max(len(self._build_step_detail_lines()) - self._step_detail_avail_rows(), 0)

    # ── Redraw ──────────────────────────────────────────────────

    def _redraw(self):
        with self._write_lock:
            self._write_raw('\033[?25l')
            self._draw_header()
            cs = self._content_start
            for row in range(cs, self.rows + 1):
                self._write_raw(f'\033[{row};1H\033[2K')
            self._write_raw(f'\033[{cs};1H')

            if self.view == "main":
                tab = self._active_tab
                lines = self._build_main_lines(tab)
                spinner_active = (tab.streaming and tab.spinner_label
                                  and not self._has_visible_stream_content(tab))
                avail = self._main_avail_rows(tab)

                # Clamp scroll offset
                max_off = max(len(lines) - avail, 0)
                if tab.scroll_offset > max_off:
                    tab.scroll_offset = max_off

                # Viewport window
                end = len(lines) - tab.scroll_offset
                start = max(end - avail, 0)
                if tab.scroll_offset >= max_off and max_off > 0:
                    start = 0
                    end = min(avail, len(lines))
                for line in lines[start:end]:
                    self._write_raw(f'{line}\n')

                # Spinner (no \n so _update_spinner can overwrite in place)
                if spinner_active:
                    ch = SPINNER[tab.spinner_tick]
                    elapsed = int(time.monotonic() - tab.spinner_start)
                    status = self._spinner_status(elapsed, tab.spinner_tokens)
                    self._write_raw(f'  {DIM}{ch} {tab.spinner_label} {status}{RESET}')

                # Scroll indicator
                if tab.scroll_offset > 0:
                    indicator = f' {DIM}↓ {tab.scroll_offset} more lines below{RESET}'
                    self._write_raw(f'\033[{self.rows};1H\033[2K{indicator}')

                if self._confirming and not self._browsing and self.active_tab_idx == 0:
                    self._draw_confirmation()
                    self._write_raw('\033[?25h')
            elif self.view == "whiteboard":
                self._write_raw(f'  {BOLD}Whiteboard{RESET} {DIM}(esc to return){RESET}\n')
                self._write_raw(f'  {DIM}{"─" * 40}{RESET}\n')
                sections: list[str] = []
                current_title = "Notes"
                current_lines: list[str] = []

                def flush_whiteboard_section():
                    if not current_lines:
                        return
                    if sections:
                        sections.append(f'  {DIM}{"─" * 40}{RESET}')
                        sections.append("")
                    sections.append(f"  {CYAN}{BOLD}{current_title}{RESET}")
                    for line in current_lines:
                        sections.append(f"  {line}" if line else "")

                source_lines = self.whiteboard.splitlines() or ["(whiteboard is empty)"]
                for wline in source_lines:
                    stripped = wline.strip()
                    if stripped.startswith("## "):
                        flush_whiteboard_section()
                        current_title = stripped[3:].strip() or "Notes"
                        current_lines = []
                        continue
                    current_lines.append(wline)
                flush_whiteboard_section()
                if not sections:
                    sections.append("  (whiteboard is empty)")

                for iline in sections:
                    self._write_raw(f'{iline}\n')
            elif self.view == "input":
                tab = self._active_tab
                self._write_raw(f'  {BOLD}Worker Input{RESET} {DIM}(esc to return){RESET}\n')
                self._write_raw(f'  {DIM}{"─" * 40}{RESET}\n')
                sections: list[str] = []

                def add_input_section(title: str, lines: list[str], color: str = BLUE):
                    if not lines:
                        return
                    if sections:
                        sections.append(f'  {DIM}{"─" * 40}{RESET}')
                        sections.append("")
                    sections.append(f"  {color}{BOLD}{title}{RESET}")
                    for line in lines:
                        sections.append(f"  {line}" if line else "")

                status_line = (
                    f"{GREEN}● completed{RESET}" if tab.done else f"{CYAN}● running{RESET}"
                )
                add_input_section("Status", [status_line], color=YELLOW)
                add_input_section("Worker", [tab.label], color=MAGENTA)

                desc = (tab.task_description or "").strip()
                add_input_section(
                    "Input",
                    desc.splitlines() if desc else ["(no task description)"],
                    color=CYAN,
                )

                for iline in sections:
                    self._write_raw(f'{iline}\n')
            elif self.view == "help":
                self._write_raw(HELP_TEXT)
                if self.run_params:
                    self._write_raw(f'\n  {BOLD}Parameters{RESET}\n\n')
                    for key, val in self.run_params.items():
                        self._write_raw(f'    {DIM}{key:<16}{RESET}{val}\n')
            elif self.view == "step_detail":
                self._write_raw(f'  {BOLD}{self._step_detail_title}{RESET}')
                self._write_raw(f' {DIM}(esc to return){RESET}\n')
                self._write_raw(f'  {DIM}{"─" * 40}{RESET}\n')
                lines = self._build_step_detail_lines()
                avail = self._step_detail_avail_rows()
                max_scroll = max(len(lines) - avail, 0)
                if self._step_detail_scroll > max_scroll:
                    self._step_detail_scroll = max_scroll
                start = self._step_detail_scroll
                end = min(start + avail, len(lines))
                for dline in lines[start:end]:
                    self._write_raw(f'{dline}\n')

                above = start
                below = max(len(lines) - end, 0)
                if above > 0 or below > 0:
                    indicator = f' {DIM}↑ {above} above · ↓ {below} below{RESET}'
                    self._write_raw(f'\033[{self.rows};1H\033[2K{indicator}')

            sys.stdout.flush()

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _style(text: str, color: str = "", bold: bool = False,
               dim: bool = False) -> str:
        prefix = ""
        if color:
            prefix += COLOR_MAP.get(color, "")
        if bold:
            prefix += BOLD
        if dim:
            prefix += DIM
        return f'{prefix}{text}{RESET}' if prefix else text

    @staticmethod
    def _strip_toml_block(text: str) -> str:
        """Hide planner TOML decision blocks from rendered output."""
        cleaned = re.sub(
            r"<(?:OPENPROVER_TOML|TOML_OUTPUT)>\s*\n?.*?</(?:OPENPROVER_TOML|TOML_OUTPUT)>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip("\n")

    @staticmethod
    def _iter_toml_segments(text: str) -> list[tuple[bool, str]]:
        """Split output into plain vs TOML-tagged blocks."""
        segments: list[tuple[bool, str]] = []
        lowers = text.lower()
        open_close = (
            ("<toml_output>", "</toml_output>"),
            ("<openprover_toml>", "</openprover_toml>"),
        )
        i = 0
        while i < len(text):
            next_idx = -1
            next_open = ""
            next_close = ""
            for open_tag, close_tag in open_close:
                idx = lowers.find(open_tag, i)
                if idx >= 0 and (next_idx < 0 or idx < next_idx):
                    next_idx = idx
                    next_open = open_tag
                    next_close = close_tag

            if next_idx < 0:
                if i < len(text):
                    segments.append((False, text[i:]))
                break

            if next_idx > i:
                segments.append((False, text[i:next_idx]))

            close_idx = lowers.find(next_close, next_idx + len(next_open))
            if close_idx < 0:
                # Unclosed TOML block: treat until end as TOML output.
                segments.append((True, text[next_idx:]))
                break

            end = close_idx + len(next_close)
            segments.append((True, text[next_idx:end]))
            i = end

        return segments

    @staticmethod
    def _longest_partial_tag_suffix(text: str, tags: tuple[str, ...]) -> str:
        """Return longest suffix that is a prefix of any tag."""
        best = ""
        for tag in tags:
            max_len = min(len(text), len(tag) - 1)
            for n in range(max_len, 0, -1):
                if text.endswith(tag[:n]):
                    if n > len(best):
                        best = text[-n:]
                    break
        return best

    def _split_toml_stream_segments(self, tab: _Tab,
                                    chunk: str) -> list[tuple[bool, str]]:
        """Stream-safe split that preserves partial TOML tags across chunks."""
        open_to_close = {
            "<TOML_OUTPUT>": "</TOML_OUTPUT>",
            "<OPENPROVER_TOML>": "</OPENPROVER_TOML>",
        }
        open_tags = tuple(open_to_close.keys())
        close_tags = tuple(open_to_close.values())
        tags_all = open_tags + close_tags

        data = tab.toml_pending + chunk
        tab.toml_pending = ""
        out: list[tuple[bool, str]] = []
        i = 0

        while i < len(data):
            if tab.toml_close_tag:
                close_tag = tab.toml_close_tag
                close_idx = data.find(close_tag, i)
                if close_idx < 0:
                    tail = data[i:]
                    keep = self._longest_partial_tag_suffix(tail, (close_tag,))
                    emit = tail[:-len(keep)] if keep else tail
                    if emit:
                        out.append((True, emit))
                    tab.toml_pending = keep
                    return out
                end = close_idx + len(close_tag)
                out.append((True, data[i:end]))
                i = end
                tab.toml_close_tag = ""
                continue

            next_open_idx = -1
            next_open_tag = ""
            for open_tag in open_tags:
                idx = data.find(open_tag, i)
                if idx >= 0 and (next_open_idx < 0 or idx < next_open_idx):
                    next_open_idx = idx
                    next_open_tag = open_tag

            if next_open_idx < 0:
                tail = data[i:]
                keep = self._longest_partial_tag_suffix(tail, tags_all)
                emit = tail[:-len(keep)] if keep else tail
                if emit:
                    out.append((False, emit))
                tab.toml_pending = keep
                return out

            if next_open_idx > i:
                out.append((False, data[i:next_open_idx]))

            close_tag = open_to_close[next_open_tag]
            close_idx = data.find(close_tag, next_open_idx + len(next_open_tag))
            if close_idx < 0:
                tab.toml_close_tag = close_tag
                tail = data[next_open_idx:]
                keep = self._longest_partial_tag_suffix(tail, (close_tag,))
                emit = tail[:-len(keep)] if keep else tail
                if emit:
                    out.append((True, emit))
                tab.toml_pending = keep
                return out

            end = close_idx + len(close_tag)
            out.append((True, data[next_open_idx:end]))
            i = end

        return out

    def _has_visible_stream_content(self, tab: _Tab) -> bool:
        if tab.output_non_toml_seen:
            return True
        if tab.output_toml_seen and self.trace_visible:
            return True
        return bool(tab.trace_buf and self.trace_visible)

    def _max_log_text_width(self) -> int:
        """Visible width available for plain log entry text."""
        # Regular log entries are rendered with one leading visible space.
        return max(max(self.cols - 4, 20) - 1, 1)

    def _dim_separator(self) -> str:
        """Separator line that never wraps as a regular log entry."""
        return f'{DIM}{"─" * self._max_log_text_width()}{RESET}'


class HeadlessTUI:
    """Non-interactive TUI that prints logs to stdout and errors to stderr."""

    supports_streaming = False

    def __init__(self):
        self._autonomous = True
        self.whiteboard = ""
        self.step_entries: list[dict] = []
        self.pending_action: str | None = None
        self.trace_visible = False

    @property
    def autonomous(self) -> bool:
        return True

    @autonomous.setter
    def autonomous(self, value: bool):
        pass

    def setup(self, theorem_name: str, work_dir: str,
              step_num: int = 0, max_steps: int = 50,
              model_name: str = ""):
        print(f"[openprover] {theorem_name}", flush=True)
        print(f"[openprover] {work_dir} | {model_name}", flush=True)

    def cleanup(self):
        pass

    def log(self, text: str, color: str = "", bold: bool = False,
            dim: bool = False):
        if color == "red":
            print(f"[error] {text}", file=sys.stderr, flush=True)
        else:
            print(f"[log] {text}", flush=True)

    def tab_log(self, tab_id: str, text: str, color: str = "",
                dim: bool = False):
        pass

    def log_trace(self, text: str):
        pass

    def stream_start(self, label: str = "thinking", tab: str = "planner"):
        pass

    def stream_text(self, text: str, kind: str = "text",
                    tab: str = "planner"):
        pass

    def stream_end(self, tab: str = "planner"):
        pass

    def step_complete(self, step_num: int, max_steps: int,
                      action: str, summary: str, detail: str = "",
                      rejected: bool = False, interrupted: bool = False,
                      feedback: str = "") -> int:
        suffix = []
        if rejected:
            suffix.append("rejected")
        if interrupted:
            suffix.append("interrupted")
        if feedback.strip():
            suffix.append(f"feedback: {feedback.strip()}")
        tail = f" [{' | '.join(suffix)}]" if suffix else ""
        print(f"[step {step_num}/{max_steps}] {action} — {summary}{tail}",
              flush=True)
        idx = len(self.step_entries)
        self.step_entries.append({
            "action": action, "summary": summary,
            "step_num": step_num, "detail": detail,
            "action_output": "",
            "rejected": rejected, "interrupted": interrupted,
            "feedback": feedback.strip(),
        })
        return idx

    def update_step(self, step_num: int, max_steps: int):
        pass

    def update_step_detail(self, step_idx: int, detail: str):
        if 0 <= step_idx < len(self.step_entries):
            self.step_entries[step_idx]["detail"] = detail

    def update_step_status(
            self,
            step_idx: int,
            *,
            rejected: bool | None = None,
            interrupted: bool | None = None,
            feedback: str | None = None,
            detail_append: str = "",
    ):
        if not (0 <= step_idx < len(self.step_entries)):
            return
        entry = self.step_entries[step_idx]
        if rejected is not None:
            entry["rejected"] = rejected
        if interrupted is not None:
            entry["interrupted"] = interrupted
        if feedback is not None:
            entry["feedback"] = feedback.strip()
        if detail_append:
            base = entry.get("detail", "")
            entry["detail"] = f"{base}\n\n{detail_append}".strip() if base else detail_append

    def append_step_action_output(self, step_num: int, text: str):
        if not text:
            return
        for entry in reversed(self.step_entries):
            if entry.get("step_num") == step_num:
                prev = entry.get("action_output", "")
                entry["action_output"] = (
                    f"{prev}\n\n{text}".strip() if prev else text
                )
                break

    def show_proposal(self, plan: dict):
        pass

    def show_replan_notice(self, text: str):
        print(f"[log] {text}", flush=True)

    def clear_replan_notice(self):
        pass

    def get_confirmation(self) -> str:
        return ""

    def get_pending_action(self) -> str | None:
        return None

    def show_interrupt_options(self):
        pass

    def get_interrupt_response(self) -> str:
        return ""

    def add_worker_tab(self, tab_id: str, label: str,
                       task_description: str = ""):
        pass

    def mark_worker_done(self, tab_id: str):
        pass

    def snapshot_worker_tabs(self, step_num: int):
        pass

    def set_waiting_status(self, text: str):
        pass

    def worker_output(self, tab_id: str, text: str):
        pass

    def clear_worker_tabs(self):
        pass

    def browse(self):
        pass

    def interrupt(self):
        pass
