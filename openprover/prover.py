"""Core proving loop for OpenProver — planner/worker architecture."""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

from . import prompts
from .lean import LeanTheorem, LeanWorkDir, run_lean_check
from .llm import Interrupted
from .tui import TUI

logger = logging.getLogger("openprover")


WORKER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lean_verify",
            "description": "Verify Lean 4 code. Returns compiler output (errors/warnings or OK).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Lean 4 source code to verify.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lean_search",
            "description": "Search Mathlib and Lean 4 declarations by natural language query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:50].strip("-")


class Repo:
    """Manages the repo/ directory of markdown and lean items."""

    def __init__(self, repo_dir: Path):
        self.dir = repo_dir
        self.dir.mkdir(exist_ok=True)

    def _resolve_path(self, slug: str) -> Path | None:
        """Find existing item file (.lean preferred over .md)."""
        lean = self.dir / f"{slug}.lean"
        if lean.exists():
            return lean
        md = self.dir / f"{slug}.md"
        if md.exists():
            return md
        return None

    @staticmethod
    def _extract_summary(path: Path) -> str:
        """Extract summary line from a repo item file."""
        text = path.read_text()
        if path.suffix == ".lean":
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped.startswith("-- "):
                    return stripped[3:].removeprefix("Summary:").strip()
            return "(no summary)"
        first_line = text.split("\n", 1)[0]
        return first_line.removeprefix("Summary:").strip()

    def list_summaries(self) -> str:
        """Return index: '- [[slug]]: summary' for each item."""
        entries = []
        files = sorted(
            list(self.dir.glob("*.md")) + list(self.dir.glob("*.lean")),
            key=lambda f: f.stem,
        )
        seen = set()
        for f in files:
            if f.stem in seen:
                continue
            seen.add(f.stem)
            summary = self._extract_summary(f)
            tag = " (lean)" if f.suffix == ".lean" else ""
            entries.append(f"- [[{f.stem}]]{tag}: {summary}")
        return "\n".join(entries)

    def read_item(self, slug: str) -> str | None:
        path = self._resolve_path(slug)
        return path.read_text() if path else None

    def read_items(self, slugs: list[str]) -> str:
        """Read multiple items, return formatted for prev_output."""
        parts = []
        for slug in slugs:
            content = self.read_item(slug)
            if content:
                parts.append(f"## [[{slug}]]\n\n{content}")
            else:
                parts.append(f"## [[{slug}]]\n\n(not found)")
        return "\n\n".join(parts)

    def write_item(self, slug: str, content: str | None, fmt: str = "markdown"):
        """Create/update item, or delete if content is None/empty."""
        ext = ".lean" if fmt == "lean" else ".md"
        path = self.dir / f"{slug}{ext}"
        if not content:
            (self.dir / f"{slug}.md").unlink(missing_ok=True)
            (self.dir / f"{slug}.lean").unlink(missing_ok=True)
        else:
            other_ext = ".md" if fmt == "lean" else ".lean"
            (self.dir / f"{slug}{other_ext}").unlink(missing_ok=True)
            path.write_text(content)

    def resolve_wikilinks(self, text: str) -> str:
        """Find [[slug]] references, return formatted appendix."""
        slugs = re.findall(r'\[\[([a-z0-9_-]+)\]\]', text)
        if not slugs:
            return ""
        parts = []
        seen = set()
        for slug in slugs:
            if slug in seen:
                continue
            seen.add(slug)
            content = self.read_item(slug)
            if content:
                parts.append(f"## [[{slug}]]\n\n{content}")
            else:
                parts.append(f"## [[{slug}]]\n\n(not found)")
        return "\n\n".join(parts)


class _TUILogHandler(logging.Handler):
    """Logging handler that forwards messages to the TUI logs tab."""

    def __init__(self, tui: TUI):
        super().__init__()
        self.tui = tui

    def emit(self, record):
        try:
            self.tui.log_trace(self.format(record))
        except Exception:
            pass


class Prover:
    def __init__(self, theorem_path: str | None, make_llm, model_name: str,
                 max_steps: int,
                 autonomous: bool, verbose: bool, tui: TUI,
                 isolation: bool = False, run_dir: str | None = None,
                 parallelism: int = 1, give_up_ratio: float = 0.5,
                 lean_project_dir: Path | None = None,
                 lean_theorem_path: Path | None = None,
                 proof_path: Path | None = None,
                 make_worker_llm=None,
                 lean_items: bool = False,
                 lean_worker_actions: bool = False):
        self.model = model_name
        self._make_llm = make_llm
        self._make_worker_llm = make_worker_llm or make_llm
        self.lean_items = lean_items
        self.lean_worker_actions = lean_worker_actions
        self.max_steps = max_steps
        self.autonomous = autonomous
        self.verbose = verbose
        self.isolation = isolation
        self.give_up_ratio = give_up_ratio
        self.tui = tui
        self.parallelism = parallelism
        self.shutting_down = False
        self.step_num = 0
        self.prev_outputs: list[str] = []  # rolling window of last 3 outputs
        self.proof_text = ""
        self.resumed = False

        # Lean configuration
        self.lean_project_dir = lean_project_dir
        self.lean_theorem: LeanTheorem | None = None
        self.lean_theorem_text: str = ""
        self.lean_work_dir: LeanWorkDir | None = None
        self.proof_md_text: str = ""

        if lean_theorem_path:
            self.lean_theorem_text = lean_theorem_path.read_text()
            self.lean_theorem = LeanTheorem(self.lean_theorem_text)
            self.lean_work_dir = LeanWorkDir(lean_project_dir)
        if proof_path:
            self.proof_md_text = proof_path.read_text()

        # Operating mode
        if lean_theorem_path and proof_path:
            self.mode = "formalize_only"
        elif lean_theorem_path:
            self.mode = "prove_and_formalize"
        else:
            self.mode = "prove"

        # Set up working directory
        if run_dir:
            self.work_dir = Path(run_dir)
        else:
            theorem_text = Path(theorem_path).read_text()
            first_line = theorem_text.strip().split("\n")[0][:40]
            slug = slugify(first_line) or "theorem"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.work_dir = Path("runs") / f"{slug}-{timestamp}"

        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "steps").mkdir(exist_ok=True)

        # Repo
        self.repo = Repo(self.work_dir / "repo")

        # Check for resume
        whiteboard_path = self.work_dir / "WHITEBOARD.md"
        theorem_file = self.work_dir / "THEOREM.md"
        if whiteboard_path.exists() and theorem_file.exists():
            self.whiteboard = whiteboard_path.read_text()
            self.theorem_text = theorem_file.read_text()
            steps_dir = self.work_dir / "steps"
            existing = [d for d in steps_dir.iterdir()
                        if d.is_dir() and d.name.startswith("step_")]
            self.step_num = len(existing)
            self.resumed = True
        else:
            if not theorem_path:
                raise SystemExit(
                    f"Error: no WHITEBOARD.md in {self.work_dir} — "
                    "provide a theorem file to start a new run"
                )
            self.theorem_text = Path(theorem_path).read_text()
            (self.work_dir / "THEOREM.md").write_text(self.theorem_text)
            if self.lean_theorem_text:
                (self.work_dir / "THEOREM.lean").write_text(self.lean_theorem_text)
            if self.proof_md_text and self.mode == "formalize_only":
                (self.work_dir / "PROOF.md").write_text(self.proof_md_text)
            self.whiteboard = prompts.format_initial_whiteboard(
                self.theorem_text, mode=self.mode,
            )
            whiteboard_path.write_text(self.whiteboard)

        # Logging
        self._setup_logging()
        logger.info("Mode: %s, Model: %s", self.mode, self.model)
        if self.resumed:
            logger.info("Resuming from step %d/%d", self.step_num, self.max_steps)

        # LLM clients (archive_dir is fallback; per-call paths used for step calls)
        archive_dir = self.work_dir / "archive"
        self.planner_llm = self._make_llm(archive_dir)
        self.worker_llm = self._make_worker_llm(archive_dir)
        # Unified view for cost/call tracking
        self.llm = self.planner_llm

        # LeanExplore for worker tool calls
        self.lean_explore_service = None
        if self.lean_worker_actions:
            if not getattr(self.worker_llm, 'vllm', False):
                raise ValueError("--lean-worker-actions requires a vLLM worker model")
            try:
                from lean_explore.search import SearchEngine, Service
                engine = SearchEngine(use_local_data=False)
                self.lean_explore_service = Service(engine=engine)
                logger.info("LeanExplore service initialized")
            except ImportError:
                logger.warning("lean_explore not installed — lean_search tool disabled")
            except Exception as e:
                logger.warning("LeanExplore init failed: %s", e)

        # Derive theorem name for header
        lines = self.theorem_text.strip().splitlines()
        parts = []
        for line in lines:
            stripped = line.lstrip("#").strip()
            if stripped:
                parts.append(stripped)
        self.theorem_name = " ".join(parts)

    def _stream_cb(self, tab: str):
        if not self.tui.supports_streaming:
            return None
        return lambda t, k="text": self.tui.stream_text(t, kind=k, tab=tab)

    def run(self):
        self.tui.setup(
            theorem_name=self.theorem_name,
            work_dir=str(self.work_dir),
            step_num=self.step_num,
            max_steps=self.max_steps,
            model_name=self.model,
        )
        self._setup_tui_logging()
        self.tui.autonomous = self.autonomous
        self.tui.whiteboard = self.whiteboard
        self.tui.run_params = {
            "model": self.model,
            "max_steps": str(self.max_steps),
            "parallelism": str(self.parallelism),
            "give_up_after": f"{self.give_up_ratio:.0%}",
            "isolation": "on" if self.isolation else "off",
            "mode": self.mode,
        }

        if self.resumed:
            self.tui.log(
                f"Resuming from step {self.step_num}/{self.max_steps}",
                color="cyan",
            )

        while self.step_num < self.max_steps and not self.shutting_down:
            self.step_num += 1
            result = self._do_step()
            if result == "stop":
                break
            if result == "pause":
                self.tui.log("Paused.", color="yellow")
                break

        if not self.shutting_down and self.tui.step_entries:
            self._write_discussion()

    def _push_output(self, text: str):
        """Add output to the rolling window (last 3 kept)."""
        if text:
            self.prev_outputs.append(text)
            if len(self.prev_outputs) > 3:
                self.prev_outputs = self.prev_outputs[-3:]
            self.tui.append_step_action_output(self.step_num, text)

    def _apply_whiteboard_from_plan(self, plan: dict):
        """Apply planner whiteboard update if provided."""
        if plan.get("whiteboard"):
            self.whiteboard = plan["whiteboard"]
            (self.work_dir / "WHITEBOARD.md").write_text(self.whiteboard)
            self.tui.whiteboard = self.whiteboard

    def _setup_logging(self):
        """Configure file logging to trace.log in the run directory."""
        root = logging.getLogger("openprover")
        root.setLevel(logging.INFO)
        # Remove any stale handlers (e.g. from a previous init)
        for h in list(root.handlers):
            root.removeHandler(h)
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(self.work_dir / "trace.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    def _setup_tui_logging(self):
        """Add TUI log handler (call after tui.setup)."""
        root = logging.getLogger("openprover")
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        handler = _TUILogHandler(self.tui)
        handler.setFormatter(fmt)
        root.addHandler(handler)

    def _do_step(self) -> str:
        """Execute one planner step. Returns 'continue', 'stop', 'pause'."""
        logger.info("Step %d/%d", self.step_num, self.max_steps)
        self.autonomous = self.tui.autonomous

        # Check for autonomous mode actions
        if self.autonomous:
            action = self.tui.get_pending_action()
            if action == "quit":
                self.shutting_down = True
                return "stop"
            if action == "pause":
                return "pause"
            if action == "summarize":
                pass  # TODO: on-demand summary

        # Clear previous step's worker tabs
        self.tui.clear_worker_tabs()

        # Save step input
        step_dir = self.work_dir / "steps" / f"step_{self.step_num:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Build planner prompt
        repo_index = self.repo.list_summaries()
        prompt = prompts.format_planner_prompt(
            whiteboard=self.whiteboard,
            repo_index=repo_index,
            prev_outputs=list(self.prev_outputs),
            step_num=self.step_num,
            max_steps=self.max_steps,
            parallelism=self.parallelism,
            has_lean_theorem=bool(self.lean_theorem_text),
            has_proof_md=(self.work_dir / "PROOF.md").exists(),
            has_proof_lean=(self.work_dir / "PROOF.lean").exists(),
        )

        # Planner LLM call (with up to 2 retries on parse failure)
        MAX_PARSE_RETRIES = 2
        system_prompt = prompts.planner_system_prompt(
            isolation=self.isolation,
            allow_give_up=self.step_num >= self.max_steps * self.give_up_ratio,
            lean_mode=self.mode,
            num_sorries=self.lean_theorem.num_sorries if self.lean_theorem else 0,
            step_num=self.step_num,
            lean_items=self.lean_items,
        )
        plan = None
        parse_error = ""
        last_resp = None

        for attempt in range(MAX_PARSE_RETRIES + 1):
            if attempt > 0:
                call_prompt = prompts.format_planner_retry(
                    original_prompt=prompt,
                    raw_output=last_resp["result"],
                    error=parse_error,
                    attempt=attempt,
                )
            else:
                call_prompt = prompt

            retry_suffix = "" if attempt == 0 else f"_retry_{attempt}"
            label = "planning" if attempt == 0 else f"retrying ({attempt}/{MAX_PARSE_RETRIES})"
            self.tui.stream_start(label, tab="planner")
            try:
                resp = self.planner_llm.call(
                    prompt=call_prompt,
                    system_prompt=system_prompt,
                    label=f"planner_step_{self.step_num}{retry_suffix}",
                    stream_callback=self._stream_cb("planner"),
                    archive_path=step_dir / f"planner_call{retry_suffix}.json",
                )
            except Interrupted:
                self.tui.stream_end(tab="planner")
                logger.info("Planner interrupted")
                return self._handle_interrupt(step_dir)
            except RuntimeError as e:
                self.tui.stream_end(tab="planner")
                logger.error("Planner error: %s", e)
                self.tui.log(f"Error: {e}", color="red")
                self._save_step_meta(step_dir, status="llm_error", error=str(e))
                return "continue"
            self.tui.stream_end(tab="planner")
            last_resp = resp
            logger.info("Planner: %dms $%.4f",
                         resp.get("duration_ms", 0), resp.get("cost", 0))

            # Parse TOML decision
            plan = prompts.parse_planner_toml(resp["result"])
            if plan is None:
                # Check if truncated — try Phase 2 forced output
                finish = resp.get("finish_reason", "")
                if finish in ("length", "max_tokens") and attempt == 0:
                    logger.info("Planner truncated (finish_reason=%s) — Phase 2", finish)
                    self.tui.log("Planner output truncated — forcing decision...", color="yellow")
                    phase2_prompt = prompts.format_planner_truncated(prompt, resp["result"])
                    self.tui.stream_start("forcing decision", tab="planner")
                    try:
                        phase2_max = getattr(self.planner_llm, 'answer_reserve', 4096)
                        resp = self.planner_llm.call(
                            prompt=phase2_prompt,
                            system_prompt=system_prompt,
                            label=f"planner_step_{self.step_num}_phase2",
                            stream_callback=self._stream_cb("planner"),
                            archive_path=step_dir / "planner_call_phase2.json",
                            **({"max_tokens": phase2_max} if hasattr(self.planner_llm, 'answer_reserve') else {}),
                        )
                    except Interrupted:
                        self.tui.stream_end(tab="planner")
                        return self._handle_interrupt(step_dir)
                    except RuntimeError as e:
                        self.tui.stream_end(tab="planner")
                        logger.error("Phase 2 error: %s", e)
                        self.tui.log(f"Error: {e}", color="red")
                        self._save_step_meta(step_dir, status="llm_error", error=str(e))
                        return "continue"
                    self.tui.stream_end(tab="planner")
                    last_resp = resp
                    plan = prompts.parse_planner_toml(resp["result"])
                    if plan is not None:
                        break

                parse_error = (
                    "Failed to parse TOML output. Your response must end with "
                    "an <OPENPROVER_ACTION>...</OPENPROVER_ACTION> block containing "
                    "action = \"...\" and other required fields."
                )
                remaining = MAX_PARSE_RETRIES - attempt
                self.tui.log(
                    f"Failed to parse planner output — "
                    f"{'retrying' if remaining else 'giving up'}...",
                    color="red",
                )
                logger.info("Parse error (attempt %d/%d)", attempt + 1, MAX_PARSE_RETRIES + 1)
                continue

            # Whiteboard: required on step 2+, optional on step 1
            if not plan.get("whiteboard") and self.step_num > 1:
                parse_error = (
                    "TOML block parsed but missing required 'whiteboard' field. "
                    'You must include whiteboard = """...""" with the full updated whiteboard.'
                )
                remaining = MAX_PARSE_RETRIES - attempt
                self.tui.log(
                    f"Planner output missing whiteboard — "
                    f"{'retrying' if remaining else 'giving up'}...",
                    color="red",
                )
                plan = None
                continue

            break  # success

        if plan is None:
            self._save_step_meta(step_dir, status="parse_error", resp=last_resp,
                                 error=parse_error)
            return "continue"

        action = plan.get("action", "")
        summary = plan.get("summary", "")
        logger.info("Action: %s — %s", action, summary)

        # Apply whiteboard immediately only when there is no interactive
        # accept/reject decision gate for this action.
        deferred_whiteboard = (
            action in ("spawn", "literature_search") and not self.autonomous
        )
        if not deferred_whiteboard:
            self._apply_whiteboard_from_plan(plan)

        # Log non-interactive steps immediately. For actions that require
        # confirmation, record history only after the user accepts the proposal.
        if action not in ("spawn", "literature_search"):
            self.tui.step_complete(
                self.step_num, self.max_steps, action, summary,
            )

        # Save planner output
        self._save_step(step_dir, plan)

        # Dispatch
        if action == "submit_proof":
            self._save_step_meta(step_dir, status="ok", action=action, resp=resp)
            return self._handle_submit_proof(plan)
        if action == "give_up":
            self._save_step_meta(step_dir, status="ok", action=action, resp=resp)
            return self._handle_give_up()
        if action == "read_items":
            self._save_step_meta(step_dir, status="ok", action=action, resp=resp)
            return self._handle_read_items(plan)
        if action == "write_items":
            self._save_step_meta(step_dir, status="ok", action=action, resp=resp)
            return self._handle_write_items(plan, step_dir)
        if action == "spawn":
            return self._handle_spawn(plan, step_dir, resp)
        if action == "literature_search":
            return self._handle_literature_search(plan, step_dir, resp)
        if action == "submit_lean_proof":
            self._save_step_meta(step_dir, status="ok", action=action, resp=resp)
            return self._handle_submit_lean_proof(plan, step_dir)
        if action == "read_theorem":
            self._save_step_meta(step_dir, status="ok", action=action, resp=resp)
            return self._handle_read_theorem()

        self.tui.log(f"Unknown action: {action}", color="red")
        self._save_step_meta(step_dir, status="unknown_action", action=action,
                             resp=resp, error=f"Unknown action: {action}")
        return "continue"

    # ── Action handlers ──────────────────────────────────────

    def _handle_interrupt(self, step_dir: Path) -> str:
        """Handle CTRL+C during planner/worker call.

        Shows continue/feedback options (feedback selected by default).
        If autonomous, switches to manual mode.
        """
        self._save_step_meta(step_dir, status="interrupted")
        self.step_num -= 1  # don't count interrupted step
        self.planner_llm.clear_interrupt()
        self.worker_llm.clear_interrupt()

        if self.autonomous:
            self.autonomous = False
            self.tui.autonomous = False
            self.tui.log("Interrupted — switching to manual mode", color="yellow")
        else:
            self.tui.log("Interrupted", color="yellow")

        # Show continue / give feedback (feedback selected by default)
        self.tui.show_interrupt_options()
        while True:
            user_resp = self.tui.get_interrupt_response()
            if user_resp == "":
                return "continue"  # continue
            if user_resp == "q":
                self.shutting_down = True
                return "stop"
            # Feedback
            text = user_resp.strip()
            if text:
                self.tui.log(f"Feedback: {text}", color="yellow")
            self._push_output(f"Human feedback: {user_resp}")
            self.tui.show_replan_notice("Feedback noted — will replan next step")
            return "continue"

    def _handle_submit_proof(self, plan: dict) -> str:
        if self.mode == "formalize_only":
            self.tui.log("submit_proof not available in formalize-only mode", color="red")
            self._push_output(
                "submit_proof rejected: in formalize-only mode, use submit_lean_proof instead."
            )
            return "continue"

        proof = plan.get("proof", "")
        if not proof:
            self.tui.log("submit_proof but no proof text — continuing", color="red")
            self._push_output("submit_proof rejected: no proof text provided.")
            return "continue"
        self.proof_text = proof
        (self.work_dir / "PROOF.md").write_text(proof)
        proof_path = self.work_dir / "PROOF.md"
        self.tui.log(f"PROOF.md written  {proof_path}", color="green")
        logger.info("PROOF.md written")

        if self.mode == "prove_and_formalize":
            if not (self.work_dir / "PROOF.lean").exists():
                self.tui.log("PROOF.lean not yet produced — continuing", color="yellow")
                self._push_output(
                    "PROOF.md written. Now formalize the proof in Lean via submit_lean_proof."
                )
                return "continue"
            self.tui.log("Both PROOF.md and PROOF.lean complete!", color="green", bold=True)
            logger.info("Both PROOF.md and PROOF.lean complete")

        return "stop"

    def _handle_give_up(self) -> str:
        if self.step_num < self.max_steps * max(self.give_up_ratio, 0.8):
            self.tui.log(
                f"Not giving up — only {self.step_num}/{self.max_steps} steps used",
                color="yellow",
            )
            self._push_output("give_up rejected: too many steps remaining. Keep trying.")
            return "continue"
        logger.info("Giving up at step %d/%d", self.step_num, self.max_steps)
        self.tui.log("Stuck — no more ideas.", color="yellow")
        return "stop"

    def _handle_read_items(self, plan: dict) -> str:
        slugs = plan.get("read", [])
        if not slugs:
            self.tui.log("read_items but no slugs specified", color="yellow")
            return "continue"
        self._push_output(self.repo.read_items(slugs))
        self.tui.log(f"Read {len(slugs)} item(s): {', '.join(slugs)}", dim=True)
        return "continue"

    def _handle_write_items(self, plan: dict, step_dir: Path) -> str:
        items = plan.get("items", [])
        if not items:
            self.tui.log("write_items but no items specified", color="yellow")
            return "continue"

        lean_feedback: list[str] = []
        lean_idx = 0
        for item in items:
            slug = item.get("slug", "")
            if not slug:
                continue
            content = item.get("content")
            fmt = item.get("format", "markdown")

            if fmt == "lean" and not self.lean_items:
                self.tui.log(f"[[{slug}]]: lean items not enabled", color="red")
                lean_feedback.append(
                    f"[[{slug}]]: Rejected — lean items are not enabled. "
                    f"Use --lean-items to enable."
                )
                continue

            if fmt == "lean" and content and self.lean_work_dir:
                path = self.lean_work_dir.make_file(slug, content)
                self.tui.log(f"Verifying [[{slug}]]...", dim=True)
                success, feedback = run_lean_check(path, self.lean_project_dir)

                lean_dir = step_dir / "lean"
                lean_dir.mkdir(exist_ok=True)
                (lean_dir / f"item_{lean_idx}_{slug}.lean").write_text(content)
                (lean_dir / f"result_{lean_idx}_{slug}.txt").write_text(
                    "OK" if success else feedback)
                lean_idx += 1

                if success:
                    self.repo.write_item(slug, content, fmt="lean")
                    self.tui.log(f"Wrote [[{slug}]] (lean, verified OK)", color="green")
                    logger.info("Lean item [[%s]] verified OK", slug)
                    lean_feedback.append(f"[[{slug}]]: Lean verification PASSED")
                else:
                    self.tui.log(f"[[{slug}]] lean verification failed — not saved",
                                 color="yellow")
                    logger.info("Lean item [[%s]] failed verification — not saved", slug)
                    lean_feedback.append(
                        f"[[{slug}]]: Lean verification FAILED — item was NOT saved "
                        f"to the repo.\n```\n{feedback}\n```"
                    )
            elif not content:
                self.repo.write_item(slug, content)
                self.tui.log(f"Deleted [[{slug}]]", color="yellow")
            else:
                self.repo.write_item(slug, content)
                first_line = content.split("\n", 1)[0]
                self.tui.log(f"Wrote [[{slug}]]: {first_line}", color="green")

        if lean_feedback:
            self._push_output(
                "## Lean Verification Results\n\n" + "\n\n".join(lean_feedback)
            )
        return "continue"

    def _handle_submit_lean_proof(self, plan: dict, step_dir: Path) -> str:
        if not self.lean_theorem:
            self.tui.log("submit_lean_proof requires --lean-theorem", color="red")
            self._push_output("submit_lean_proof rejected: no THEOREM.lean configured.")
            return "continue"

        blocks = plan.get("lean_blocks", [])
        context = plan.get("lean_context", "")
        logger.info("Submitting Lean proof (%d blocks)", len(blocks))

        if not blocks:
            self.tui.log("submit_lean_proof: no lean_blocks provided", color="red")
            self._push_output(
                f"submit_lean_proof rejected: no lean_blocks provided. "
                f"Expected {self.lean_theorem.num_sorries} block(s)."
            )
            return "continue"

        # Assemble the proof file
        try:
            proof_text = self.lean_theorem.assemble_proof(blocks, context)
        except ValueError as e:
            self.tui.log(f"Assembly error: {e}", color="red")
            logger.warning("Assembly error: %s", e)
            self._push_output(f"submit_lean_proof assembly error: {e}")
            return "continue"

        # Write to temp file and verify
        proof_path = self.lean_work_dir.make_file("proof-attempt", proof_text)
        self.tui.log(f"Verifying Lean proof: {proof_path.name}...", dim=True)

        success, feedback = run_lean_check(proof_path, self.lean_project_dir)

        # Archive lean I/O
        lean_dir = step_dir / "lean"
        lean_dir.mkdir(exist_ok=True)
        (lean_dir / "proof_attempt.lean").write_text(proof_text)
        (lean_dir / "proof_result.txt").write_text("OK" if success else feedback)

        if success:
            self.lean_work_dir.write_proof(proof_text)
            (self.work_dir / "PROOF.lean").write_text(proof_text)
            self.tui.log("Lean proof verified!", color="green", bold=True)
            self.tui.log(f"  {self.work_dir / 'PROOF.lean'}", dim=True)
            logger.info("Lean proof verified! PROOF.lean written")

            if self.mode == "formalize_only":
                return "stop"

            # prove_and_formalize: check if PROOF.md also exists
            if (self.work_dir / "PROOF.md").exists():
                self.tui.log("Both PROOF.md and PROOF.lean complete!",
                             color="green", bold=True)
                logger.info("Both PROOF.md and PROOF.lean complete")
                return "stop"

            self._push_output(
                "Lean proof verified! PROOF.lean has been written.\n"
                "Now produce the informal proof using submit_proof."
            )
            return "continue"
        else:
            self.tui.log("Lean verification failed", color="red")
            logger.info("Lean proof verification failed")
            self._push_output(
                f"submit_lean_proof verification FAILED.\n\n"
                f"Lean feedback:\n```\n{feedback}\n```\n\n"
                f"Fix the issues and try again."
            )
            return "continue"

    def _handle_read_theorem(self) -> str:
        parts = [f"## THEOREM.md\n\n{self.theorem_text}"]
        if self.lean_theorem_text:
            parts.append(
                f"\n\n## THEOREM.lean\n\n```lean\n{self.lean_theorem_text}\n```"
            )
            parts.append(
                f"\n\nNumber of `sorry` keywords to replace: "
                f"{self.lean_theorem.num_sorries}"
            )
        proof_md_path = self.work_dir / "PROOF.md"
        if proof_md_path.exists():
            parts.append(f"\n\n## PROOF.md\n\n{proof_md_path.read_text()}")
        proof_lean_path = self.work_dir / "PROOF.lean"
        if proof_lean_path.exists():
            parts.append(
                f"\n\n## PROOF.lean\n\n```lean\n{proof_lean_path.read_text()}\n```"
            )
        self._push_output("\n".join(parts))
        self.tui.log("Read theorem content", color="yellow")
        return "continue"

    def _handle_spawn(self, plan: dict, step_dir: Path,
                      planner_resp: dict | None = None) -> str:
        tasks = plan.get("tasks", [])
        if not tasks:
            self.tui.log("spawn but no tasks specified", color="yellow")
            self._save_step_meta(step_dir, status="ok", action="spawn",
                                 resp=planner_resp, error="No tasks specified")
            return "continue"

        # Limit to parallelism
        tasks = tasks[:self.parallelism]

        # Interactive confirmation
        if not self.autonomous:
            self.tui.show_proposal(plan)
            while True:
                user_resp = self.tui.get_confirmation()
                if user_resp == "":
                    break  # accept
                if user_resp == "q":
                    self.shutting_down = True
                    return "stop"
                if user_resp == "p":
                    return "pause"
                if user_resp == "a":
                    self.autonomous = True
                    self.tui.log("  autonomous mode", dim=True)
                    break
                # Feedback — set as prev_output and retry next step
                text = user_resp.strip()
                detail = (
                    f"Proposed step:\n"
                    f"{plan.get('action', '')} — {plan.get('summary', '')}".strip(" —")
                )
                self.tui.step_complete(
                    self.step_num,
                    self.max_steps,
                    plan.get("action", ""),
                    plan.get("summary", ""),
                    detail=detail,
                    rejected=True,
                    feedback=text,
                )
                self._save_step_meta(
                    step_dir,
                    status="rejected",
                    action=plan.get("action", ""),
                    resp=planner_resp,
                    error="Rejected by user feedback",
                    feedback=text,
                )
                self._push_output(f"Human feedback: {user_resp}")
                self.tui.show_replan_notice("Feedback noted — will replan next step")
                return "continue"

        self._apply_whiteboard_from_plan(plan)
        step_idx = self.tui.step_complete(
            self.step_num,
            self.max_steps,
            plan.get("action", ""),
            plan.get("summary", ""),
        )

        logger.info("Spawning %d worker(s)", len(tasks))

        # Create worker tabs
        worker_ids = []
        for i, task in enumerate(tasks):
            wid = f"worker_{self.step_num}_{i}"
            desc = task.get("description", "")
            label = f"Worker {i}"
            self.tui.add_worker_tab(wid, label, task_description=desc)
            worker_ids.append(wid)

        # Run workers
        workers_dir = step_dir / "workers"
        workers_dir.mkdir(exist_ok=True)
        worker_resps = [None] * len(tasks)
        n = len(tasks)
        done_count = 0
        self.tui.set_waiting_status(f"waiting for {n} worker(s) (0/{n} finished)")

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {}
            for i, task in enumerate(tasks):
                # Save task
                desc = task.get("description", "")
                (workers_dir / f"task_{i}.md").write_text(desc)

                archive = workers_dir / f"worker_{i}_call.json"
                future = pool.submit(self._run_worker, task, worker_ids[i], archive)
                futures[future] = i

            pending = set(futures.keys())
            while pending:
                done_set, pending = wait(pending, timeout=0.5,
                                         return_when=FIRST_COMPLETED)
                for future in done_set:
                    idx = futures[future]
                    try:
                        worker_resps[idx] = future.result()
                    except Exception as e:
                        worker_resps[idx] = {
                            "result": f"Worker error: {e}", "cost": 0.0,
                            "duration_ms": 0, "raw": {}, "error": str(e),
                        }
                    done_count += 1
                    logger.info("Worker %d/%d done", done_count, n)
                    self.tui.mark_worker_done(worker_ids[idx])
                    if done_count < n:
                        self.tui.set_waiting_status(
                            f"waiting for {n} worker(s) ({done_count}/{n} finished)"
                        )

        self.tui.set_waiting_status("")

        # Check if any workers were interrupted
        any_interrupted = any(
            w and w.get("error") == "interrupted" for w in worker_resps
        )
        if any_interrupted:
            self.planner_llm.clear_interrupt()
            self.worker_llm.clear_interrupt()
            self.tui.update_step_status(
                step_idx,
                interrupted=True,
                detail_append="Execution interrupted before all workers completed.",
            )
            if self.autonomous:
                self.autonomous = False
                self.tui.autonomous = False
                self.tui.log("Interrupted — switching to manual mode", color="yellow")

        # Save results and build prev_output
        parts = []
        for i, (task, wresp) in enumerate(zip(tasks, worker_resps)):
            desc = task.get("description", "")
            first_line = desc.split("\n")[0][:60] if desc else f"Worker {i}"
            result = wresp["result"] if wresp else ""
            parts.append(f"## Worker {i}: {first_line}\n\n{result}")
            (workers_dir / f"result_{i}.md").write_text(result or "")

        self._push_output("\n\n".join(parts))

        # Save step metadata with worker details
        status = "interrupted" if any_interrupted else "ok"
        self._save_step_meta(
            step_dir, status=status, action="spawn", resp=planner_resp,
            workers=[w for w in worker_resps if w],
        )

        # Store worker tab snapshots for history
        self.tui.snapshot_worker_tabs(self.step_num)

        return "continue"

    def _handle_literature_search(self, plan: dict, step_dir: Path,
                                   planner_resp: dict | None = None) -> str:
        if self.isolation:
            self.tui.log("Literature search not available (isolation mode)", color="yellow")
            self._push_output("Literature search is not available in isolation mode.")
            self._save_step_meta(step_dir, status="ok", action="literature_search",
                                 resp=planner_resp, error="Isolation mode")
            return "continue"

        query = plan.get("search_query", "")
        context = plan.get("search_context", "")
        if not query:
            self.tui.log("literature_search but no query", color="yellow")
            self._save_step_meta(step_dir, status="ok", action="literature_search",
                                 resp=planner_resp, error="No query specified")
            return "continue"

        # Interactive confirmation
        if not self.autonomous:
            self.tui.show_proposal(plan)
            while True:
                user_resp = self.tui.get_confirmation()
                if user_resp == "":
                    break  # accept
                if user_resp == "q":
                    self.shutting_down = True
                    return "stop"
                if user_resp == "p":
                    return "pause"
                if user_resp == "a":
                    self.autonomous = True
                    self.tui.log("  autonomous mode", dim=True)
                    break
                # Feedback — set as prev_output and retry next step
                text = user_resp.strip()
                detail = (
                    f"Proposed step:\n"
                    f"{plan.get('action', '')} — {plan.get('summary', '')}".strip(" —")
                )
                self.tui.step_complete(
                    self.step_num,
                    self.max_steps,
                    plan.get("action", ""),
                    plan.get("summary", ""),
                    detail=detail,
                    rejected=True,
                    feedback=text,
                )
                self._save_step_meta(
                    step_dir,
                    status="rejected",
                    action=plan.get("action", ""),
                    resp=planner_resp,
                    error="Rejected by user feedback",
                    feedback=text,
                )
                self._push_output(f"Human feedback: {user_resp}")
                self.tui.show_replan_notice("Feedback noted — will replan next step")
                return "continue"

        self._apply_whiteboard_from_plan(plan)
        step_idx = self.tui.step_complete(
            self.step_num,
            self.max_steps,
            plan.get("action", ""),
            plan.get("summary", ""),
        )

        logger.info("Literature search: %s", query)
        wid = f"search_{self.step_num}"
        task_desc = f"Query: {query}\n\nContext: {context}"
        self.tui.add_worker_tab(wid, "Search", task_description=task_desc)

        prompt = prompts.format_search_prompt(query, context)

        workers_dir = step_dir / "workers"
        workers_dir.mkdir(exist_ok=True)
        (workers_dir / "task_0.md").write_text(f"Query: {query}\n\nContext: {context}")

        self.tui.set_waiting_status("searching literature")
        self.tui.tab_log(wid, f"Query: {query}", dim=True)
        if context:
            self.tui.tab_log(wid, f"Context: {context}", dim=True)
        self.tui.tab_log(wid, "")
        self.tui.stream_start("searching", tab=wid)
        search_resp = None
        try:
            search_resp = self.worker_llm.call(
                prompt=prompt,
                system_prompt=prompts.SEARCH_SYSTEM_PROMPT,
                label=f"search_step_{self.step_num}",
                web_search=True,
                stream_callback=self._stream_cb(wid),
                archive_path=workers_dir / "search_call.json",
            )
            self.tui.stream_end(tab=wid)
            result = search_resp["result"]
            search_resp["error"] = ""
            self._push_output(result)
            (workers_dir / "result_0.md").write_text(result)
        except Interrupted:
            self.tui.stream_end(tab=wid)
            self.worker_llm.clear_interrupt()
            self.tui.update_step_status(
                step_idx,
                interrupted=True,
                detail_append="Execution interrupted before literature search completed.",
            )
            if self.autonomous:
                self.autonomous = False
                self.tui.autonomous = False
                self.tui.log("Interrupted — switching to manual mode", color="yellow")
            result = "(terminated by user)"
            self._push_output(result)
            search_resp = {"result": result, "cost": 0.0, "duration_ms": 0,
                           "raw": {}, "error": "interrupted"}
        except RuntimeError as e:
            self.tui.stream_end(tab=wid)
            result = f"Literature search failed: {e}"
            self.tui.log(f"Search error: {e}", color="red")
            self._push_output(result)
            search_resp = {"result": result, "cost": 0.0, "duration_ms": 0,
                           "raw": {}, "error": str(e)}
        self.tui.set_waiting_status("")
        self.tui.worker_output(wid, f"Result:\n\n{result}")

        status = "ok"
        if search_resp and search_resp.get("error") == "interrupted":
            status = "interrupted"
        self._save_step_meta(
            step_dir, status=status, action="literature_search",
            resp=planner_resp, workers=[search_resp] if search_resp else None,
        )

        self.tui.mark_worker_done(wid)
        self.tui.snapshot_worker_tabs(self.step_num)
        return "continue"

    def _run_worker(self, task: dict, worker_id: str,
                    archive_path: Path | None = None) -> dict:
        """Execute a single worker. Thread-safe.

        Returns dict with keys: result (str), cost, duration_ms, raw, error.
        """
        description = task.get("description", "")
        resolved_refs = self.repo.resolve_wikilinks(description)
        prompt = prompts.format_worker_prompt(description, resolved_refs)
        use_tools = self.lean_worker_actions and getattr(self.worker_llm, 'vllm', False)
        system_prompt = prompts.worker_system_prompt(lean_worker_actions=use_tools)

        if not use_tools:
            # Single-turn path (Claude CLI or non-vLLM)
            self.tui.stream_start("working...", tab=worker_id)
            try:
                resp = self.worker_llm.call(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    label=worker_id,
                    stream_callback=self._stream_cb(worker_id),
                    archive_path=archive_path,
                )
                self.tui.stream_end(tab=worker_id)

                # Phase 2 if truncated
                if resp.get("finish_reason") in ("length", "max_tokens"):
                    logger.info("[%s] truncated — Phase 2", worker_id)
                    self.tui.stream_start("forcing output...", tab=worker_id)
                    answer_reserve = getattr(self.worker_llm, 'answer_reserve', 4096)
                    phase2_prompt = (
                        f"{prompt}\n\n---\n\n"
                        f"Your previous reasoning was cut off. Continue with your final answer.\n\n"
                        f"Previous output (last 2000 chars):\n"
                        f"```\n{resp['result'][-2000:]}\n```"
                    )
                    resp2 = self.worker_llm.call(
                        prompt=phase2_prompt,
                        system_prompt=system_prompt,
                        label=f"{worker_id}_phase2",
                        stream_callback=self._stream_cb(worker_id),
                        archive_path=archive_path.parent / f"{archive_path.stem}_phase2.json" if archive_path else None,
                        max_tokens=answer_reserve,
                    )
                    self.tui.stream_end(tab=worker_id)
                    resp = {
                        "result": resp2["result"],
                        "thinking": resp["thinking"] + resp2.get("thinking", ""),
                        "cost": resp["cost"] + resp2["cost"],
                        "duration_ms": resp["duration_ms"] + resp2["duration_ms"],
                        "raw": resp2["raw"],
                        "finish_reason": resp2.get("finish_reason", "stop"),
                    }

                resp["error"] = ""
            except Interrupted:
                self.tui.stream_end(tab=worker_id)
                resp = {"result": "(terminated by user)", "cost": 0.0,
                        "duration_ms": 0, "raw": {}, "error": "interrupted"}
            except RuntimeError as e:
                self.tui.stream_end(tab=worker_id)
                resp = {"result": f"Worker error: {e}", "cost": 0.0,
                        "duration_ms": 0, "raw": {}, "error": str(e)}
            self.tui.worker_output(worker_id, resp["result"])
            return resp

        # Multi-turn tool-calling path (vLLM)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        total_cost = 0.0
        total_duration = 0
        call_idx = 0

        try:
            while True:
                self.tui.stream_start("working..." if call_idx == 0 else "continuing...", tab=worker_id)
                call_archive = (
                    archive_path.parent / f"{archive_path.stem}_{call_idx}.json"
                    if archive_path else None
                )
                resp = self.worker_llm.chat(
                    messages=messages,
                    tools=WORKER_TOOLS,
                    label=f"{worker_id}_turn_{call_idx}",
                    stream_callback=self._stream_cb(worker_id),
                    archive_path=call_archive,
                )
                self.tui.stream_end(tab=worker_id)
                total_cost += resp["cost"]
                total_duration += resp["duration_ms"]
                call_idx += 1

                finish = resp.get("finish_reason", "stop")

                if finish == "stop":
                    break

                if finish == "tool_calls" and resp.get("tool_calls"):
                    # Append assistant message with tool calls
                    assistant_msg = {"role": "assistant", "content": resp["result"] or None}
                    assistant_msg["tool_calls"] = resp["tool_calls"]
                    messages.append(assistant_msg)

                    # Execute each tool call
                    for tc in resp["tool_calls"]:
                        tool_name = tc["function"]["name"]
                        try:
                            tool_args = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError:
                            tool_args = {"raw": tc["function"]["arguments"]}

                        tool_result, tool_status = self._execute_worker_tool(
                            tool_name, tool_args, worker_id,
                        )

                        # Add to TUI
                        self.tui.add_worker_action(
                            worker_id, tool_name, tool_args,
                            tool_result, tool_status,
                        )

                        # Append tool result message
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_result,
                        })
                    continue

                if finish == "length":
                    # Phase 2: force output, no tools
                    logger.info("[%s] truncated — Phase 2", worker_id)
                    assistant_msg = {"role": "assistant", "content": resp["result"] or ""}
                    messages.append(assistant_msg)
                    messages.append({
                        "role": "user",
                        "content": "Your response was cut off. Continue with your final answer.",
                    })
                    self.tui.stream_start("forcing output...", tab=worker_id)
                    answer_reserve = getattr(self.worker_llm, 'answer_reserve', 4096)
                    resp = self.worker_llm.chat(
                        messages=messages,
                        tools=None,
                        max_tokens=answer_reserve,
                        label=f"{worker_id}_phase2",
                        stream_callback=self._stream_cb(worker_id),
                        archive_path=(
                            archive_path.parent / f"{archive_path.stem}_phase2.json"
                            if archive_path else None
                        ),
                    )
                    self.tui.stream_end(tab=worker_id)
                    total_cost += resp["cost"]
                    total_duration += resp["duration_ms"]
                    break

                # Unknown finish reason — treat as done
                break

            result = {
                "result": resp["result"],
                "thinking": resp.get("thinking", ""),
                "cost": total_cost,
                "duration_ms": total_duration,
                "raw": resp["raw"],
                "finish_reason": resp.get("finish_reason", "stop"),
                "error": "",
            }
        except Interrupted:
            self.tui.stream_end(tab=worker_id)
            result = {"result": "(terminated by user)", "cost": total_cost,
                      "duration_ms": total_duration, "raw": {}, "error": "interrupted"}
        except RuntimeError as e:
            self.tui.stream_end(tab=worker_id)
            result = {"result": f"Worker error: {e}", "cost": total_cost,
                      "duration_ms": total_duration, "raw": {}, "error": str(e)}

        self.tui.worker_output(worker_id, result["result"])
        return result

    def _execute_worker_tool(self, name: str, args: dict, worker_id: str) -> tuple[str, str]:
        """Execute a worker tool call. Returns (result_text, status)."""
        if name == "lean_verify":
            return self._tool_lean_verify(args, worker_id)
        if name == "lean_search":
            return self._tool_lean_search(args, worker_id)
        return (f"Unknown tool: {name}", "error")

    def _tool_lean_verify(self, args: dict, worker_id: str) -> tuple[str, str]:
        """Verify Lean code via lean_check."""
        code = args.get("code", "")
        if not code:
            return ("No code provided", "error")
        if not self.lean_work_dir:
            return ("Lean project not configured", "error")

        slug = f"worker_verify_{worker_id}"
        path = self.lean_work_dir.make_file(slug, code)
        success, feedback = run_lean_check(path, self.lean_project_dir)
        status = "ok" if success else "error"
        result = "OK — no errors" if success else feedback
        logger.info("[%s] lean_verify: %s", worker_id, status)
        return (result, status)

    def _tool_lean_search(self, args: dict, worker_id: str) -> tuple[str, str]:
        """Search Mathlib declarations."""
        import asyncio
        query = args.get("query", "")
        if not query:
            return ("No query provided", "error")
        if not self.lean_explore_service:
            return ("lean_search not available (lean_explore not installed)", "error")

        try:
            results = asyncio.run(
                self.lean_explore_service.search(query, limit=10)
            )
            if not results:
                return ("No results found", "ok")
            parts = []
            for r in results:
                name = getattr(r, 'name', str(r))
                doc = getattr(r, 'doc_string', '') or ''
                sig = getattr(r, 'signature', '') or ''
                entry = f"**{name}**"
                if sig:
                    entry += f"\n```lean\n{sig}\n```"
                if doc:
                    entry += f"\n{doc}"
                parts.append(entry)
            return ("\n\n".join(parts), "ok")
        except Exception as e:
            logger.warning("[%s] lean_search error: %s", worker_id, e)
            return (f"Search error: {e}", "error")

    # ── Saving & discussion ──────────────────────────────────

    def _save_step(self, step_dir: Path, plan: dict):
        # Save as TOML-like text (human readable)
        lines = [f'action = "{plan.get("action", "")}"']
        lines.append(f'summary = "{plan.get("summary", "")}"')
        if plan.get("whiteboard"):
            lines.append(f'whiteboard = """\n{plan["whiteboard"]}\n"""')
        # Save action-specific fields
        for key in ("proof", "search_query", "search_context", "lean_context"):
            if key in plan:
                val = plan[key]
                if isinstance(val, str) and "\n" in val:
                    lines.append(f'{key} = """\n{val}\n"""')
                else:
                    lines.append(f'{key} = "{val}"')
        if "read" in plan:
            lines.append(f'read = {json.dumps(plan["read"])}')
        if "items" in plan:
            for item in plan["items"]:
                lines.append("\n[[items]]")
                lines.append(f'slug = "{item.get("slug", "")}"')
                content = item.get("content")
                if content:
                    lines.append(f'content = """\n{content}\n"""')
        if "tasks" in plan:
            for task in plan["tasks"]:
                lines.append("\n[[tasks]]")
                desc = task.get("description", "")
                lines.append(f'description = """\n{desc}\n"""')
        if "lean_blocks" in plan:
            for block in plan["lean_blocks"]:
                lines.append("\n[[lean_blocks]]")
                lines.append(f'code = """\n{block}\n"""')
        (step_dir / "planner.toml").write_text("\n".join(lines) + "\n")

    @staticmethod
    def _extract_token_usage(resp: dict) -> dict:
        """Pull token counts from an LLM response dict."""
        raw = resp.get("raw") or {}
        # Claude CLI puts usage at top level; HF may have it in usage
        usage = raw.get("usage", {})
        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        }

    def _save_step_meta(self, step_dir: Path, *,
                        status: str,
                        action: str = "",
                        resp: dict | None = None,
                        error: str = "",
                        feedback: str = "",
                        workers: list[dict] | None = None):
        """Write step_meta.toml with structured metadata for the step."""
        lines = [
            f'timestamp = "{datetime.now(timezone.utc).isoformat()}"',
            f'step = {self.step_num}',
            f'status = "{status}"',
            f'action = "{action}"',
        ]

        if error:
            lines.append(f'error = """\n{error}\n"""')
        if feedback:
            lines.append(f'feedback = """\n{feedback}\n"""')

        # Planner call metadata
        if resp:
            tokens = self._extract_token_usage(resp)
            lines.append("")
            lines.append("[planner]")
            lines.append(f'cost_usd = {resp.get("cost", 0.0)}')
            lines.append(f'duration_ms = {resp.get("duration_ms", 0)}')
            lines.append(f'input_tokens = {tokens["input_tokens"]}')
            lines.append(f'output_tokens = {tokens["output_tokens"]}')
            lines.append(f'cache_creation_tokens = {tokens["cache_creation_tokens"]}')
            lines.append(f'cache_read_tokens = {tokens["cache_read_tokens"]}')
            raw = resp.get("raw") or {}
            lines.append(f'model = "{raw.get("model", self.planner_llm.model)}"')
            lines.append(f'stop_reason = "{raw.get("stop_reason", "")}"')

        # Worker metadata
        if workers:
            for i, w in enumerate(workers):
                lines.append("")
                lines.append(f"[[workers]]")
                lines.append(f"index = {i}")
                lines.append(f'cost_usd = {w.get("cost", 0.0)}')
                lines.append(f'duration_ms = {w.get("duration_ms", 0)}')
                tokens = self._extract_token_usage(w)
                lines.append(f'input_tokens = {tokens["input_tokens"]}')
                lines.append(f'output_tokens = {tokens["output_tokens"]}')
                lines.append(f'cache_creation_tokens = {tokens["cache_creation_tokens"]}')
                lines.append(f'cache_read_tokens = {tokens["cache_read_tokens"]}')
                if w.get("error"):
                    lines.append(f'error = "{w["error"]}"')

        (step_dir / "step_meta.toml").write_text("\n".join(lines) + "\n")

    def _write_discussion(self):
        logger.info("Writing discussion")
        repo_index = self.repo.list_summaries()
        prompt = prompts.format_discussion_prompt(
            theorem=self.theorem_text,
            whiteboard=self.whiteboard,
            repo_index=repo_index,
            steps_taken=self.step_num,
            max_steps=self.max_steps,
            proof=self.proof_text,
        )
        self.tui.stream_start("writing discussion", tab="planner")
        try:
            resp = self.planner_llm.call(
                prompt=prompt,
                system_prompt=prompts.planner_system_prompt(
                    isolation=self.isolation, lean_mode=self.mode,
                    lean_items=self.lean_items,
                ),
                label="discussion",
                stream_callback=self._stream_cb("planner"),
                archive_path=self.work_dir / "discussion_call.json",
            )
            self.tui.stream_end(tab="planner")
            (self.work_dir / "DISCUSSION.md").write_text(resp["result"])
            self.tui.log(f"  {self.work_dir / 'DISCUSSION.md'}", dim=True)
        except (Interrupted, RuntimeError) as e:
            self.tui.stream_end(tab="planner")
            if not isinstance(e, Interrupted):
                self.tui.log(f"Error generating discussion: {e}", color="red")
            (self.work_dir / "DISCUSSION.md").write_text(
                f"# Discussion\n\nSession ended after {self.step_num} steps.\n\n"
                f"## Final Whiteboard\n\n{self.whiteboard}\n"
            )
            self.tui.log(f"  {self.work_dir / 'DISCUSSION.md'}", dim=True)

    @property
    def is_finished(self) -> bool:
        """Check if this run is already finished (has discussion or proof)."""
        has_discussion = (self.work_dir / "DISCUSSION.md").exists()
        has_proof_md = (self.work_dir / "PROOF.md").exists()
        has_proof_lean = (self.work_dir / "PROOF.lean").exists()

        if self.mode == "formalize_only":
            return has_proof_lean or has_discussion
        elif self.mode == "prove_and_formalize":
            return (has_proof_md and has_proof_lean) or has_discussion
        else:
            return has_proof_md or has_discussion

    def inspect(self):
        """Enter inspect mode — browse historical run data without running steps."""
        self.tui.setup(
            theorem_name=self.theorem_name,
            work_dir=str(self.work_dir),
            step_num=self.step_num,
            max_steps=self.max_steps,
            model_name=self.model,
        )
        self.tui.autonomous = False
        self.tui.whiteboard = self.whiteboard
        self.tui.run_params = {
            "model": self.model,
            "max_steps": str(self.max_steps),
            "parallelism": str(self.parallelism),
            "give_up_after": f"{self.give_up_ratio:.0%}",
            "isolation": "on" if self.isolation else "off",
            "mode": self.mode,
        }

        self._load_history()

        # Show result status
        if (self.work_dir / "PROOF.md").exists():
            self.tui.log("Proof found!", color="green", bold=True)
        elif (self.work_dir / "DISCUSSION.md").exists():
            self.tui.log("Session ended.", dim=True)
        self.tui.log("Inspect mode — press q to quit", dim=True)

        self.tui.browse()

    def _load_history(self):
        """Load step history from disk for inspect mode."""
        steps_dir = self.work_dir / "steps"
        if not steps_dir.exists():
            return

        step_dirs = sorted(
            [d for d in steps_dir.iterdir()
             if d.is_dir() and d.name.startswith("step_")],
        )

        for idx, step_dir in enumerate(step_dirs):
            toml_file = step_dir / "planner.toml"
            if not toml_file.exists():
                continue

            plan = prompts.parse_planner_toml(toml_file.read_text())
            if plan is None:
                continue

            step_num = int(step_dir.name.removeprefix("step_"))
            action = plan.get("action", "")
            summary = plan.get("summary", "")
            meta = self._read_step_meta(step_dir)

            # Log step in planner tab
            step_idx = self.tui.step_complete(step_num, self.max_steps, action, summary)
            status = meta.get("status", "")
            feedback = meta.get("feedback", "")
            detail = ""
            if status == "rejected":
                detail = f"Proposed step:\n{action} — {summary}".strip(" —")
                self.tui.update_step_status(
                    step_idx,
                    rejected=True,
                    feedback=feedback,
                    detail_append=detail,
                )
            elif status == "interrupted":
                self.tui.update_step_status(
                    step_idx,
                    interrupted=True,
                    feedback=feedback,
                    detail_append="Execution was interrupted.",
                )

            # Load worker tabs for spawn/search steps
            workers_dir = step_dir / "workers"
            if not workers_dir.exists():
                self.tui.snapshot_worker_tabs(step_num)
                if idx < len(step_dirs) - 1:
                    self.tui.clear_worker_tabs()
                continue

            task_files = sorted(workers_dir.glob("task_*.md"))
            for task_file in task_files:
                tidx = task_file.stem.removeprefix("task_")
                result_file = workers_dir / f"result_{tidx}.md"
                desc = task_file.read_text()
                result = result_file.read_text() if result_file.exists() else ""

                if action == "literature_search":
                    tab_id = f"search_{step_num}"
                    label = "Search"
                else:
                    tab_id = f"worker_{step_num}_{tidx}"
                    label = f"Worker {tidx}"

                self.tui.add_worker_tab(tab_id, label, task_description=desc)
                if result:
                    self.tui.worker_output(tab_id, result)
                self.tui.mark_worker_done(tab_id)

            self.tui.snapshot_worker_tabs(step_num)
            if idx < len(step_dirs) - 1:
                self.tui.clear_worker_tabs()

    @staticmethod
    def _read_step_meta(step_dir: Path) -> dict[str, str]:
        meta_path = step_dir / "step_meta.toml"
        if not meta_path.exists():
            return {}
        text = meta_path.read_text()
        data: dict[str, str] = {}

        def extract_single(key: str) -> str:
            m = re.search(rf'^{key}\s*=\s*"([^"]*)"', text, flags=re.MULTILINE)
            return m.group(1) if m else ""

        def extract_block(key: str) -> str:
            m = re.search(
                rf'^{key}\s*=\s*"""\n(.*?)\n"""',
                text,
                flags=re.MULTILINE | re.DOTALL,
            )
            return m.group(1).strip() if m else ""

        data["status"] = extract_single("status")
        data["feedback"] = extract_block("feedback")
        data["error"] = extract_block("error")
        return data

    def request_interrupt(self):
        """Called by SIGINT handler. Kills active LLM call or nudges TUI."""
        self.planner_llm.interrupt()
        self.worker_llm.interrupt()
        self.tui.interrupt()  # in case we're in a confirmation prompt
