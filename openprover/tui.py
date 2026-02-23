"""Terminal UI for OpenProver — ANSI scroll regions, fixed header, tabs."""

import os
import queue
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
    "proof_found": GREEN,
    "submit_lean_proof": GREEN,
    "give_up": RED,
}

HELP_TEXT = f"""\
  {BOLD}Controls{RESET}

  {DIM}Instant keys (work any time):{RESET}
    t           toggle reasoning trace
    i           show worker input (on worker tabs)
    w           toggle whiteboard view
    a           toggle autonomous mode
    {DIM}←/→{RESET}         switch tabs
    pgup/pgdn   scroll chat history
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
        # Step history — each entry: {action, summary, step_num, detail, trace, worker_tabs}
        self.step_entries: list[dict] = []
        self._nav_step = -1  # -1 = options focused, 0..N-1 = step index
        self._step_detail_text = ""
        self._step_detail_title = ""
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
                            f'{trace_style}t trace{RESET} {DIM}·{RESET} '
                            f'{input_style}i input{RESET} {DIM}·{RESET} '
                            f'{auto_style}a autonomous{RESET}')
            hints_len = len("? help · t trace · i input · a autonomous")
        else:
            hints_styled = (f'{help_style}? help{RESET} {DIM}·{RESET} '
                            f'{trace_style}t trace{RESET} {DIM}·{RESET} '
                            f'{wb_style}w whiteboard{RESET} {DIM}·{RESET} '
                            f'{auto_style}a autonomous{RESET}')
            hints_len = len("? help · t trace · w whiteboard · a autonomous")
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
            elif tab.streaming:
                name += " …"
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
            planner.spinner_start = time.monotonic()
            planner.spinner_time = 0.0
            planner.spinner_tick = 0
            planner.spinner_tokens = 0
            if planner is self._active_tab and self.view == "main":
                ch = SPINNER[0]
                self._write(f'  {DIM}{ch} {text} {self._spinner_status(0, 0)}{RESET}')
        else:
            planner.streaming = False
            planner.spinner_label = ""
            if planner is self._active_tab and self.view == "main":
                self._write('\r\033[2K')

    def worker_output(self, tab_id: str, text: str):
        """Display worker result in its tab as regular (always-visible) content."""
        tab = self._find_tab(tab_id)
        sep_text = f'{DIM}{"─" * max(self.cols - 4, 20)}{RESET}'
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
        self._proposal_log_start = len(planner.log_lines)
        sep = f'{DIM}{"─" * max(self.cols - 4, 20)}{RESET}'
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

    def step_complete(self, step_num: int, max_steps: int,
                      action: str, summary: str, detail: str = ""):
        planner = self.tabs[0]
        color = ACTION_STYLE.get(action, "")
        line = f'{color}■{RESET} {BOLD}{action}{RESET} {DIM}—{RESET} {summary}'
        trace = planner.last_trace
        output = planner.last_output
        planner.last_trace = ""
        planner.last_output = ""
        idx = len(self.step_entries)
        self._tab_log(planner, line, step_idx=idx)
        self.step_entries.append({
            "action": action, "summary": summary,
            "step_num": step_num, "detail": detail,
            "trace": trace, "output": output,
        })
        self.update_step(step_num, max_steps)
        if self.view == "main":
            self._redraw()

    def update_step_detail(self, step_idx: int, detail: str):
        if 0 <= step_idx < len(self.step_entries):
            self.step_entries[step_idx]["detail"] = detail

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
        if now - tab.spinner_time < 0.08:
            return
        tab.spinner_time = now
        tab.spinner_tick = (tab.spinner_tick + 1) % len(SPINNER)
        if self.view == "main":
            ch = SPINNER[tab.spinner_tick]
            elapsed = int(now - tab.spinner_start)
            status = self._spinner_status(elapsed, tab.spinner_tokens)
            with self._write_lock:
                if (not tab.spinner_label or tab.output_buf
                        or (tab.trace_buf and self.trace_visible)):
                    return
                self._write_raw(f'\r\033[2K  {DIM}{ch} {tab.spinner_label} {status}{RESET}')
                sys.stdout.flush()

    # ── Streaming ───────────────────────────────────────────────

    def stream_start(self, label: str = "thinking", tab: str = "planner"):
        target = self._find_tab(tab)
        target.trace_buf = []
        target.output_buf = []
        target.streaming = True
        target.spinner_label = label
        target.spinner_tick = 0
        target.spinner_tokens = 0
        target.spinner_time = time.monotonic()
        target.spinner_start = target.spinner_time
        if target is self._active_tab and self.view == "main":
            self._write(f'  {DIM}{SPINNER[0]} {label} {self._spinner_status(0, 0)}{RESET}')

    def stream_text(self, text: str, kind: str = "text", tab: str = "planner"):
        self._check_keys()
        target = self._find_tab(tab)
        target.spinner_tokens += 1
        is_active = target is self._active_tab
        at_bottom = target.scroll_offset == 0
        is_thinking = kind == "thinking"

        # Was there visible content before this chunk?
        had_visible = (target.output_buf
                       or (target.trace_buf and self.trace_visible))

        if is_thinking:
            target.trace_buf.append(text)
        else:
            target.output_buf.append(text)

        has_visible = (target.output_buf
                       or (target.trace_buf and self.trace_visible))

        # Clear spinner on first visible content
        if (not had_visible and has_visible
                and self.view == "main" and is_active and at_bottom):
            with self._write_lock:
                self._write_raw('\r\033[2K')
                sys.stdout.flush()

        should_display = not is_thinking or self.trace_visible
        if should_display and self.view == "main" and is_active and at_bottom:
            if is_thinking:
                self._write(f'{DIM}{text}{RESET}')
            else:
                self._write(text)

    def stream_end(self, tab: str = "planner"):
        target = self._find_tab(tab)
        target.streaming = False
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

    # ── Background thread (key reader + spinner) ────────────────

    def _bg_loop(self):
        fd = sys.stdin.fileno()
        while not self._bg_stop:
            try:
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
                            # Arrow keys — handle ←/→ directly for tab switching
                            arrow = chr(data[i + 2])
                            seq = chr(0x1b) + '[' + arrow
                            if self._can_handle_directly() and arrow in ('C', 'D'):
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
                        if ch in ('t', 'i', 'w', '?', 'a'):
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
        if ch == 't':
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
        elif ch == 'scroll_up':
            self._scroll_lines_up()
        elif ch == 'scroll_down':
            self._scroll_lines_down()
        elif ch == '\x1b' and self.view != "main":
            self.view = "main"
            self._redraw()
        elif ch in ('\n', '\r'):
            if self.view != "main":
                self.view = "main"
                self._redraw()
            elif self._active_tab.scroll_offset > 0:
                self._active_tab.scroll_offset = 0
                self._redraw()
        elif self.autonomous and ch in ('q', 'p', 's'):
            self.pending_action = {
                'q': 'quit', 'p': 'pause',
                's': 'summarize',
            }[ch]

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

                if ch in ('\n', '\r'):
                    if self.view != "main":
                        self.view = "main"
                        self._redraw()
                        continue
                    if self._nav_step >= 0:
                        entry = self.step_entries[self._nav_step]
                        self._step_detail_title = (
                            f"Step {entry['step_num']}: {entry['action']}"
                            f" — {entry['summary']}"
                        )
                        parts = []
                        trace = entry.get("trace", "")
                        if trace and self.trace_visible:
                            parts.append(trace.rstrip())
                            parts.append("")
                        output = entry.get("output", "")
                        if output:
                            parts.append(output.rstrip())
                            parts.append("")
                        detail = entry.get("detail", "")
                        if detail:
                            parts.append(detail)
                        self._step_detail_text = "\n".join(parts) if parts else "(no detail)"
                        self.view = "step_detail"
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
                    else:
                        self._confirm_selected = 1 - self._confirm_selected
                    self._redraw()
                    continue

                can_toggle = (self._confirm_selected == 0 or not self._confirm_buf)
                if can_toggle and ch in ('t', 'i', 'w', '?'):
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
                if can_toggle and ch in ('t', 'w', '?'):
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
                        entry = self.step_entries[self._nav_step]
                        self._step_detail_title = (
                            f"Step {entry['step_num']}: {entry['action']}"
                            f" — {entry['summary']}"
                        )
                        parts = []
                        trace = entry.get("trace", "")
                        if trace and self.trace_visible:
                            parts.append(trace.rstrip())
                            parts.append("")
                        output = entry.get("output", "")
                        if output:
                            parts.append(output.rstrip())
                            parts.append("")
                        detail = entry.get("detail", "")
                        if detail:
                            parts.append(detail)
                        self._step_detail_text = (
                            "\n".join(parts) if parts else "(no detail)")
                        self.view = "step_detail"
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
        elif self._nav_step > 0:
            self._nav_step -= 1
            self._load_historical_workers()

    def _nav_down(self):
        if self._nav_step >= 0:
            if self._nav_step < len(self.step_entries) - 1:
                self._nav_step += 1
                self._load_historical_workers()
            else:
                self._nav_step = -1
                self._restore_worker_tabs()

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
        self.trace_visible = not self.trace_visible
        if self.view == "main":
            self._redraw()

    def _toggle_view(self, target: str):
        self.view = "main" if self.view == target else target
        self._redraw()

    # ── Scrolling ────────────────────────────────────────────────

    def _scroll_up(self):
        tab = self._active_tab
        page = max(self.rows - self._content_start - 2, 1)
        tab.scroll_offset += page
        self._redraw()

    def _scroll_down(self):
        tab = self._active_tab
        page = max(self.rows - self._content_start - 2, 1)
        tab.scroll_offset = max(tab.scroll_offset - page, 0)
        self._redraw()

    def _scroll_lines_up(self, n: int = 3):
        self._active_tab.scroll_offset += n
        self._redraw()

    def _scroll_lines_down(self, n: int = 3):
        tab = self._active_tab
        tab.scroll_offset = max(tab.scroll_offset - n, 0)
        self._redraw()

    def _build_main_lines(self, tab: _Tab | None = None) -> list[str]:
        """Build flat list of rendered lines for the active tab."""
        if tab is None:
            tab = self._active_tab
        lines: list[str] = []
        max_w = max(self.cols - 4, 20)
        for entry in tab.log_lines:
            if entry.is_trace:
                if not self.trace_visible:
                    continue
                for tline in entry.text.splitlines():
                    while len(tline) > max_w:
                        lines.append(f'  {DIM}{tline[:max_w]}{RESET}')
                        tline = tline[max_w:]
                    lines.append(f'  {DIM}{tline}{RESET}')
            elif entry.is_output:
                if not self.trace_visible:
                    continue
                for tline in entry.text.splitlines():
                    while len(tline) > max_w:
                        lines.append(f'  {tline[:max_w]}')
                        tline = tline[max_w:]
                    lines.append(f'  {tline}')
            else:
                is_step = entry.step_idx >= 0
                if is_step and self._confirming and entry.step_idx == self._nav_step:
                    lines.append(f' {GREEN}▎{RESET}{entry.text}')
                else:
                    lines.append(f' {entry.text}')
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
                for tline in joined.splitlines():
                    while len(tline) > max_w:
                        lines.append(f'  {tline[:max_w]}')
                        tline = tline[max_w:]
                    lines.append(f'  {tline}')
        return lines

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
                confirm_rows = 3 if (self._confirming and not self._browsing and self.active_tab_idx == 0) else 0
                spinner_active = (tab.streaming and tab.spinner_label
                                  and not (tab.trace_buf and self.trace_visible))
                spinner_rows = 1 if spinner_active else 0
                avail = self.rows - cs + 1 - confirm_rows - spinner_rows

                # Clamp scroll offset
                max_off = max(len(lines) - avail, 0)
                if tab.scroll_offset > max_off:
                    tab.scroll_offset = max_off

                # Viewport window
                end = len(lines) - tab.scroll_offset
                start = max(end - avail, 0)
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
                for wline in self.whiteboard.splitlines():
                    self._write_raw(f'  {wline}\n')
            elif self.view == "input":
                tab = self._active_tab
                self._write_raw(f'  {BOLD}Worker Input{RESET} {DIM}(esc to return){RESET}\n')
                self._write_raw(f'  {DIM}{"─" * 40}{RESET}\n')
                desc = tab.task_description or "(no task description)"
                for tline in desc.splitlines():
                    self._write_raw(f'  {tline}\n')
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
                for dline in self._step_detail_text.splitlines():
                    self._write_raw(f'  {dline}\n')

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
                      action: str, summary: str, detail: str = ""):
        print(f"[step {step_num}/{max_steps}] {action} — {summary}",
              flush=True)
        self.step_entries.append({
            "action": action, "summary": summary,
            "step_num": step_num, "detail": detail,
        })

    def update_step(self, step_num: int, max_steps: int):
        pass

    def update_step_detail(self, step_idx: int, detail: str):
        if 0 <= step_idx < len(self.step_entries):
            self.step_entries[step_idx]["detail"] = detail

    def show_proposal(self, plan: dict):
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
