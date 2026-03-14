"""Tab management for the TUI."""

import sys
import time

from ._colors import DIM, BOLD, RESET, WHITE, GREEN, YELLOW, SPINNER, TOOL_STYLE
from ._types import _LogEntry, _Tab


class TabsMixin:

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
            # Waiting is a standalone status line; clear any stale live-stream
            # visibility flags so redraws keep showing this spinner.
            planner.trace_buf = []
            planner.output_buf = []
            planner.output_non_toml_seen = False
            planner.output_toml_seen = False
            planner.spinner_label = text
            planner.streaming = True
            planner.is_waiting = True
            planner.spinner_start = time.monotonic()
            planner.spinner_time = 0.0
            planner.spinner_tick = 0
            planner.spinner_tokens = 0
            if planner is self._active_tab and self._main_visible:
                if self.view == "whiteboard_split":
                    self._redraw()
                else:
                    ch = SPINNER[0]
                    with self._write_lock:
                        self._write_raw(f'  {DIM}{ch} {text} {self._spinner_status(0, 0)}{RESET}')
                        sys.stdout.flush()
        else:
            planner.streaming = False
            planner.is_waiting = False
            planner.spinner_label = ""
            if planner is self._active_tab and self._main_visible:
                if self.view == "whiteboard_split":
                    self._redraw()
                else:
                    with self._write_lock:
                        self._write_raw('\r\033[2K')
                        sys.stdout.flush()
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
        if tab is self._active_tab and self._main_visible:
            self._redraw()

    def start_worker_action(self, tab_id: str, tool: str, args: dict):
        """Show a tool call that is about to start executing, with a spinner."""
        tab = self._find_tab(tab_id)
        entry = {
            "type": "action",
            "tool": tool,
            "args": args,
            "result": "",
            "status": "running",
            "duration_ms": 0,
        }
        idx = len(tab.entries)
        tab.entries.append(entry)
        line = self._format_action_line(entry)
        self._tab_log(tab, line, step_idx=idx)
        tab.pending_action_log_idx = len(tab.log_lines) - 1
        # Enable spinner for tool execution.
        # If streaming is already active (MCP path), just update the label;
        # otherwise (vLLM path), set up a fresh spinner.
        tab.prev_spinner_label = tab.spinner_label
        tab.spinner_label = f"{tool}\u2026"
        tab.spinner_tick = 0
        if not tab.streaming:
            tab.spinner_start = time.monotonic()
            tab.spinner_time = 0.0
            tab.spinner_tokens = 0
            tab.streaming = True
            tab.is_waiting = False
            tab.trace_buf = []
            tab.output_buf = []
            tab.output_non_toml_seen = False
            tab.output_toml_seen = False
        self._redraw_header()
        if tab is self._active_tab and self._main_visible:
            if self.view == "whiteboard_split":
                self._redraw()
            elif not self._has_visible_stream_content(tab):
                ch = SPINNER[0]
                with self._write_lock:
                    self._write_raw(f'  {DIM}{ch} {tool}\u2026 {self._spinner_status(0, 0)}{RESET}')
                    sys.stdout.flush()

    def add_worker_action(self, tab_id: str, tool: str, args: dict,
                          result: str, status: str, duration_ms: int = 0):
        """Record a completed tool call in a worker tab as a navigable entry."""
        tab = self._find_tab(tab_id)

        # Complete a pending action: update the existing entry and log line
        if tab.pending_action_log_idx >= 0:
            log_idx = tab.pending_action_log_idx
            tab.pending_action_log_idx = -1
            if 0 <= log_idx < len(tab.log_lines):
                entry_idx = tab.log_lines[log_idx].step_idx
                if 0 <= entry_idx < len(tab.entries):
                    entry = tab.entries[entry_idx]
                    entry["result"] = result
                    entry["status"] = status
                    entry["duration_ms"] = duration_ms
                    tab.log_lines[log_idx].text = self._format_action_line(entry)
                    # Clear spinner
                    tab.spinner_label = tab.prev_spinner_label
                    tab.prev_spinner_label = ""
                    if not tab.spinner_label:
                        tab.streaming = False
                    # Clear the spinner line and redraw
                    if tab is self._active_tab and self._main_visible:
                        if self.view == "whiteboard_split":
                            self._redraw()
                        else:
                            with self._write_lock:
                                self._write_raw('\r\033[2K')
                                sys.stdout.flush()
                            self._redraw()
                    self._redraw_header()
                    return

        # Fallback: no pending action, append as new entry
        entry = {
            "type": "action",
            "tool": tool,
            "args": args,
            "result": result,
            "status": status,
            "duration_ms": duration_ms,
        }
        idx = len(tab.entries)
        tab.entries.append(entry)
        line = self._format_action_line(entry)
        self._tab_log(tab, line, step_idx=idx)

    @staticmethod
    def _format_action_line(entry: dict) -> str:
        tool = entry.get("tool", "?")
        status = entry.get("status", "")
        color = TOOL_STYLE.get(tool, WHITE)
        if status == "ok":
            icon = f"{GREEN}\u2713{RESET}"
        elif status == "running":
            icon = ""
        else:
            icon = f"{YELLOW}\u2717{RESET}"
        # Short summary from args
        args = entry.get("args", {})
        if "query" in args:
            summary = args["query"][:60]
        elif "code" in args:
            first = args["code"].strip().split("\n")[0][:60]
            summary = first
        else:
            summary = ""
        duration_ms = entry.get("duration_ms", 0)
        if duration_ms:
            dur = f" {DIM}({duration_ms / 1000:.1f}s){RESET}"
        else:
            dur = ""
        if status == "running":
            return f'{color}\u25b8{RESET} {BOLD}{tool}{RESET} {DIM}\u2014{RESET} {summary}'
        return f'{color}\u25b8{RESET} {BOLD}{tool}{RESET} {icon}{dur} {DIM}\u2014{RESET} {summary}'

    def clear_worker_tabs(self):
        """Remove all worker tabs, keeping planner and logs."""
        logs = [t for t in self.tabs if t.id == "logs"]
        self.tabs = [self.tabs[0]] + logs
        self.active_tab_idx = 0
        self._redraw_header()

    def _redraw_header(self):
        with self._write_lock:
            self._write_raw('\033[s')
            self._draw_header()
            self._write_raw('\033[u')
            sys.stdout.flush()

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
