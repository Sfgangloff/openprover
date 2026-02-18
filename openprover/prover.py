"""Core proving loop for OpenProver — planner/worker architecture."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from . import prompts
from .llm import LLMClient
from .tui import TUI


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:50].strip("-")


class Repo:
    """Manages the repo/ directory of markdown items."""

    def __init__(self, repo_dir: Path):
        self.dir = repo_dir
        self.dir.mkdir(exist_ok=True)

    def list_summaries(self) -> str:
        """Return index: '- [[slug]]: summary' for each item."""
        entries = []
        for f in sorted(self.dir.glob("*.md")):
            first_line = f.read_text().split("\n", 1)[0]
            summary = first_line.removeprefix("Summary:").strip()
            entries.append(f"- [[{f.stem}]]: {summary}")
        return "\n".join(entries)

    def read_item(self, slug: str) -> str | None:
        path = self.dir / f"{slug}.md"
        return path.read_text() if path.exists() else None

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

    def write_item(self, slug: str, content: str | None):
        """Create/update item, or delete if content is None/empty."""
        path = self.dir / f"{slug}.md"
        if not content:
            path.unlink(missing_ok=True)
        else:
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


class Prover:
    def __init__(self, theorem_path: str | None, model: str, max_steps: int,
                 autonomous: bool, verbose: bool, tui: TUI,
                 isolation: bool = False, run_dir: str | None = None,
                 parallelism: int = 1):
        self.model = model
        self.max_steps = max_steps
        self.autonomous = autonomous
        self.verbose = verbose
        self.isolation = isolation
        self.tui = tui
        self.parallelism = parallelism
        self.shutting_down = False
        self.step_num = 0
        self.prev_output = ""
        self.proof_text = ""
        self.resumed = False

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
        (self.work_dir / "archive" / "calls").mkdir(parents=True, exist_ok=True)

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
            self.whiteboard = prompts.format_initial_whiteboard(self.theorem_text)
            whiteboard_path.write_text(self.whiteboard)

        # LLM client
        self.llm = LLMClient(model, self.work_dir / "archive" / "calls")

        # Derive theorem name for header
        lines = self.theorem_text.strip().splitlines()
        parts = []
        for line in lines:
            stripped = line.lstrip("#").strip()
            if stripped:
                parts.append(stripped)
        self.theorem_name = " ".join(parts)

    def run(self):
        self.tui.setup(
            theorem_name=self.theorem_name,
            work_dir=str(self.work_dir),
            step_num=self.step_num,
            max_steps=self.max_steps,
        )
        self.tui.autonomous = self.autonomous
        self.tui.whiteboard = self.whiteboard

        if self.resumed:
            self.tui.log(
                f"Resuming from step {self.step_num}/{self.max_steps}",
                color="cyan",
            )

        paused = False
        try:
            while self.step_num < self.max_steps and not self.shutting_down:
                self.step_num += 1
                result = self._do_step()
                if result == "stop":
                    break
                if result == "pause":
                    paused = True
                    self.tui.log("Paused.", color="yellow")
                    break
        except KeyboardInterrupt:
            self.shutting_down = True

        if not paused and self.tui.step_entries:
            self._write_discussion()

    def _do_step(self) -> str:
        """Execute one planner step. Returns 'continue', 'stop', 'pause'."""
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

        # Save step input
        step_dir = self.work_dir / "steps" / f"step_{self.step_num:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Build planner prompt
        repo_index = self.repo.list_summaries()
        prompt = prompts.format_planner_prompt(
            whiteboard=self.whiteboard,
            repo_index=repo_index,
            prev_output=self.prev_output,
            step_num=self.step_num,
            max_steps=self.max_steps,
            isolation=self.isolation,
        )
        self.prev_output = ""

        # Planner LLM call
        self.tui.stream_start("planning", tab="planner")
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.PLANNER_SYSTEM_PROMPT,
                label=f"planner_step_{self.step_num}",
                stream_callback=lambda t: self.tui.stream_text(t, tab="planner"),
            )
        except RuntimeError as e:
            self.tui.stream_end(tab="planner")
            self.tui.log(f"Error: {e}", color="red")
            return "continue"
        self.tui.stream_end(tab="planner")

        # Parse TOML decision
        plan = prompts.parse_planner_toml(resp["result"])
        if plan is None:
            self.tui.log("Failed to parse planner output — retrying...", color="red")
            return "continue"

        action = plan.get("action", "")
        summary = plan.get("summary", "")

        # Update whiteboard (always)
        if plan.get("whiteboard"):
            self.whiteboard = plan["whiteboard"]
            (self.work_dir / "WHITEBOARD.md").write_text(self.whiteboard)
            self.tui.whiteboard = self.whiteboard

        # Log step in planner tab
        self.tui.step_complete(
            self.step_num, self.max_steps, action, summary,
        )

        # Save planner output
        self._save_step(step_dir, plan)

        # Dispatch
        if action == "proof_found":
            return self._handle_proof_found(plan)
        if action == "give_up":
            return self._handle_give_up()
        if action == "read_items":
            return self._handle_read_items(plan)
        if action == "write_item":
            return self._handle_write_item(plan)
        if action == "spawn":
            return self._handle_spawn(plan, step_dir)
        if action == "literature_search":
            return self._handle_literature_search(plan, step_dir)

        self.tui.log(f"Unknown action: {action}", color="red")
        return "continue"

    # ── Action handlers ──────────────────────────────────────

    def _handle_proof_found(self, plan: dict) -> str:
        proof = plan.get("proof", "")
        if not proof:
            self.tui.log("proof_found but no proof text — continuing", color="red")
            self.prev_output = "proof_found rejected: no proof text provided."
            return "continue"
        self.proof_text = proof
        (self.work_dir / "PROOF.md").write_text(proof)
        self.tui.log("Proof found!", color="green", bold=True)
        self.tui.log(f"  {self.work_dir / 'PROOF.md'}", dim=True)
        return "stop"

    def _handle_give_up(self) -> str:
        if self.step_num < self.max_steps * 0.8:
            self.tui.log(
                f"Not giving up — only {self.step_num}/{self.max_steps} steps used",
                color="yellow",
            )
            self.prev_output = "give_up rejected: too many steps remaining. Keep trying."
            return "continue"
        self.tui.log("Stuck — no more ideas.", color="yellow")
        return "stop"

    def _handle_read_items(self, plan: dict) -> str:
        slugs = plan.get("read", [])
        if not slugs:
            self.tui.log("read_items but no slugs specified", color="yellow")
            return "continue"
        self.prev_output = self.repo.read_items(slugs)
        self.tui.log(f"Read {len(slugs)} item(s): {', '.join(slugs)}", dim=True)
        return "continue"

    def _handle_write_item(self, plan: dict) -> str:
        slug = plan.get("write_slug", "")
        if not slug:
            self.tui.log("write_item but no slug specified", color="yellow")
            return "continue"
        content = plan.get("write_content")
        self.repo.write_item(slug, content)
        if not content:
            self.tui.log(f"Deleted [[{slug}]]", color="yellow")
        else:
            first_line = content.split("\n", 1)[0]
            self.tui.log(f"Wrote [[{slug}]]: {first_line}", color="green")
        return "continue"

    def _handle_spawn(self, plan: dict, step_dir: Path) -> str:
        tasks = plan.get("tasks", [])
        if not tasks:
            self.tui.log("spawn but no tasks specified", color="yellow")
            return "continue"

        # Limit to parallelism
        tasks = tasks[:self.parallelism]

        # Interactive confirmation
        if not self.autonomous:
            self.tui.show_proposal(plan)
            while True:
                resp = self.tui.get_confirmation()
                if resp == "":
                    break  # accept
                if resp == "q":
                    self.shutting_down = True
                    return "stop"
                if resp == "p":
                    return "pause"
                if resp == "a":
                    self.autonomous = True
                    self.tui.log("  autonomous mode", dim=True)
                    break
                # Feedback — set as prev_output and retry next step
                self.prev_output = f"Human feedback: {resp}"
                self.tui.log("Feedback noted — will replan next step", color="yellow")
                return "continue"

        # Create worker tabs
        worker_ids = []
        for i, task in enumerate(tasks):
            wid = f"worker_{self.step_num}_{i}"
            desc = task.get("description", "")
            label = desc.split("\n")[0][:40] if desc else f"Worker {i}"
            self.tui.add_worker_tab(wid, label)
            worker_ids.append(wid)

        # Run workers
        workers_dir = step_dir / "workers"
        workers_dir.mkdir(exist_ok=True)
        results = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {}
            for i, task in enumerate(tasks):
                # Save task
                desc = task.get("description", "")
                (workers_dir / f"task_{i}.md").write_text(desc)

                future = pool.submit(self._run_worker, task, worker_ids[i])
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Worker error: {e}"
                self.tui.mark_worker_done(worker_ids[idx])

        # Save results and build prev_output
        parts = []
        for i, (task, result) in enumerate(zip(tasks, results)):
            desc = task.get("description", "")
            first_line = desc.split("\n")[0][:60] if desc else f"Worker {i}"
            parts.append(f"## Worker {i}: {first_line}\n\n{result}")
            (workers_dir / f"result_{i}.md").write_text(result or "")

        self.prev_output = "\n\n".join(parts)

        # Store worker tab snapshots for history
        self.tui.snapshot_worker_tabs(self.step_num)

        return "continue"

    def _handle_literature_search(self, plan: dict, step_dir: Path) -> str:
        if self.isolation:
            self.tui.log("Literature search not available (isolation mode)", color="yellow")
            self.prev_output = "Literature search is not available in isolation mode."
            return "continue"

        query = plan.get("search_query", "")
        context = plan.get("search_context", "")
        if not query:
            self.tui.log("literature_search but no query", color="yellow")
            return "continue"

        wid = f"search_{self.step_num}"
        self.tui.add_worker_tab(wid, f"search: {query[:30]}")

        prompt = prompts.format_search_prompt(query, context)

        workers_dir = step_dir / "workers"
        workers_dir.mkdir(exist_ok=True)
        (workers_dir / "task_0.md").write_text(f"Query: {query}\n\nContext: {context}")

        self.tui.stream_start("searching", tab=wid)
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.SEARCH_SYSTEM_PROMPT,
                label=f"search_step_{self.step_num}",
                web_search=True,
                stream_callback=lambda t: self.tui.stream_text(t, tab=wid),
            )
            self.tui.stream_end(tab=wid)
            self.prev_output = resp["result"]
            (workers_dir / "result_0.md").write_text(resp["result"])
        except RuntimeError as e:
            self.tui.stream_end(tab=wid)
            self.tui.log(f"Search error: {e}", color="red")
            self.prev_output = f"Literature search failed: {e}"

        self.tui.mark_worker_done(wid)
        self.tui.snapshot_worker_tabs(self.step_num)
        return "continue"

    def _run_worker(self, task: dict, worker_id: str) -> str:
        """Execute a single worker. Thread-safe."""
        description = task.get("description", "")
        resolved_refs = self.repo.resolve_wikilinks(description)
        prompt = prompts.format_worker_prompt(description, resolved_refs)

        self.tui.stream_start("working", tab=worker_id)
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.WORKER_SYSTEM_PROMPT,
                label=worker_id,
                stream_callback=lambda t: self.tui.stream_text(t, tab=worker_id),
            )
            self.tui.stream_end(tab=worker_id)
            return resp["result"]
        except RuntimeError as e:
            self.tui.stream_end(tab=worker_id)
            return f"Worker error: {e}"

    # ── Saving & discussion ──────────────────────────────────

    def _save_step(self, step_dir: Path, plan: dict):
        # Save as TOML-like text (human readable)
        lines = [f'action = "{plan.get("action", "")}"']
        lines.append(f'summary = "{plan.get("summary", "")}"')
        if plan.get("whiteboard"):
            lines.append(f'whiteboard = """\n{plan["whiteboard"]}\n"""')
        # Save action-specific fields
        for key in ("proof", "write_slug", "write_content", "search_query", "search_context"):
            if key in plan:
                val = plan[key]
                if isinstance(val, str) and "\n" in val:
                    lines.append(f'{key} = """\n{val}\n"""')
                else:
                    lines.append(f'{key} = "{val}"')
        if "read" in plan:
            lines.append(f'read = {json.dumps(plan["read"])}')
        if "tasks" in plan:
            for task in plan["tasks"]:
                lines.append("\n[[tasks]]")
                desc = task.get("description", "")
                lines.append(f'description = """\n{desc}\n"""')
        (step_dir / "planner.toml").write_text("\n".join(lines) + "\n")

    def _write_discussion(self):
        if self.shutting_down:
            self.tui.log(
                "Interrupted — writing discussion... (ctrl+c again to exit immediately)",
                color="yellow",
            )
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
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.PLANNER_SYSTEM_PROMPT,
                label="discussion",
                stream_callback=lambda t: self.tui.stream_text(t, tab="planner"),
            )
            self.tui.stream_end(tab="planner")
            (self.work_dir / "DISCUSSION.md").write_text(resp["result"])
            self.tui.log(f"  {self.work_dir / 'DISCUSSION.md'}", dim=True)
        except RuntimeError as e:
            self.tui.stream_end(tab="planner")
            self.tui.log(f"Error generating discussion: {e}", color="red")
            (self.work_dir / "DISCUSSION.md").write_text(
                f"# Discussion\n\nSession ended after {self.step_num} steps.\n\n"
                f"## Final Whiteboard\n\n{self.whiteboard}\n"
            )
            self.tui.log(f"  {self.work_dir / 'DISCUSSION.md'}", dim=True)

    def request_shutdown(self):
        self.shutting_down = True
        self.tui.interrupt()
