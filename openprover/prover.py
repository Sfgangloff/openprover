"""Core proving loop for OpenProver."""

import json
import re
import select
import shutil
import sys
from datetime import datetime
from pathlib import Path

from . import display, prompts
from .llm import LLMClient


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:50].strip("-")


class Prover:
    def __init__(self, theorem_path: str | None, model: str, max_steps: int,
                 autonomous: bool, verbose: bool, isolation: bool = False,
                 run_dir: str | None = None):
        self.model = model
        self.max_steps = max_steps
        self.autonomous = autonomous
        self.verbose = verbose
        self.isolation = isolation
        self.shutting_down = False
        self.step_num = 0
        self.verification_result = ""
        self.search_result = ""
        self.proof_text = ""
        self.resumed = False

        # Pick action list and schemas based on isolation mode
        if isolation:
            self.actions = prompts.ACTIONS_NO_SEARCH
        else:
            self.actions = prompts.ACTIONS
        self.plan_schema = prompts._make_plan_schema(self.actions)

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
        (self.work_dir / "lemmas").mkdir(exist_ok=True)
        (self.work_dir / "steps").mkdir(exist_ok=True)
        (self.work_dir / "archive" / "calls").mkdir(parents=True, exist_ok=True)

        # Check for resume (existing run with WHITEBOARD.md)
        whiteboard_path = self.work_dir / "WHITEBOARD.md"
        theorem_file = self.work_dir / "THEOREM.md"
        if whiteboard_path.exists() and theorem_file.exists():
            # Resume from existing run
            self.whiteboard = whiteboard_path.read_text()
            self.theorem_text = theorem_file.read_text()
            # Count existing steps to resume from correct position
            steps_dir = self.work_dir / "steps"
            existing = [d for d in steps_dir.iterdir()
                        if d.is_dir() and d.name.startswith("step_")]
            self.step_num = len(existing)
            self.resumed = True
        else:
            # Fresh start — theorem_path is required
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

    def run(self):
        display.header(self.theorem_text, str(self.work_dir))
        if self.resumed:
            display.resuming(self.step_num, self.max_steps)

        paused = False
        while self.step_num < self.max_steps and not self.shutting_down:
            self.step_num += 1
            action = self._do_step()
            if action == "stop":
                break
            if action == "pause":
                paused = True
                display.pausing()
                break
            if action == "restart":
                self._reset()
                continue

        if not paused:
            self._write_discussion()
        display.cost_summary(self.llm.total_cost, self.llm.call_count)

    def _do_step(self) -> str:
        """Execute one step. Returns: 'continue', 'stop', 'pause', 'restart'."""
        lemma_index = self._build_lemma_index()

        # Check for autonomous mode stdin interrupt
        if self.autonomous:
            self._check_stdin_interrupt()

        # Save input whiteboard for this step
        step_dir = self.work_dir / "steps" / f"step_{self.step_num:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "input.md").write_text(self.whiteboard)

        if not self.autonomous:
            # Interactive: plan first, then get approval
            plan = self._get_plan(lemma_index)
            if plan is None:
                if self.shutting_down:
                    return "stop"
                display.step_error()
                return "continue"

            display.proposal(self.step_num, self.max_steps, plan)

            # Feedback loop
            while True:
                user_input = display.interactive_prompt()
                cmd = user_input.lower()
                if cmd in ("q", "quit"):
                    self.shutting_down = True
                    return "stop"
                if cmd in ("p", "pause"):
                    return "pause"
                if cmd in ("r", "restart"):
                    return "restart"
                if cmd in ("s", "summarize"):
                    self._do_summary()
                    display.proposal(self.step_num, self.max_steps, plan)
                    continue
                if cmd in ("a", "auto"):
                    self.autonomous = True
                    display.mode_switch("autonomous")
                    break
                if user_input == "":
                    break  # accept plan
                # User gave feedback — replan
                display.feedback_note(user_input)
                plan = self._get_plan(lemma_index, human_feedback=user_input)
                if plan is None:
                    if self.shutting_down:
                        return "stop"
                    display.step_error()
                    return "continue"
                display.proposal(self.step_num, self.max_steps, plan)

            result = self._execute_step(lemma_index, plan=plan)
        else:
            # Autonomous: single combined call
            result = self._execute_step(lemma_index)

        if result is None:
            # Parse failure or error — skip this step, don't kill the run
            display.step_error()
            return "continue"

        action = result.get("action", "continue")
        display.step_start(self.step_num, self.max_steps, action, result.get("summary", ""))

        # Update whiteboard
        self.whiteboard = result.get("whiteboard", self.whiteboard)
        (self.work_dir / "WHITEBOARD.md").write_text(self.whiteboard)

        # Handle new/updated lemmas
        self._process_lemmas(result)

        # Save step data
        self._save_step(result)

        # Handle verification
        if action == "verify" and result.get("verify_content"):
            self._do_verification(result["verify_target"], result["verify_content"])

        # Handle literature search
        if action == "literature_search" and result.get("search_query"):
            self._do_literature_search(result["search_query"])

        # Handle terminal actions
        if action == "declare_proof":
            proof = result.get("proof", "")
            if not proof:
                display.proof_rejected("No proof text provided")
                return "continue"
            # Auto-verify before accepting
            passed = self._verify_proof(proof)
            if passed:
                self.proof_text = proof
                (self.work_dir / "PROOF.md").write_text(proof)
                display.proof_found()
                display.proof_written(str(self.work_dir / "PROOF.md"))
                return "stop"
            else:
                display.proof_rejected("Verification failed — continuing")
                return "continue"

        if action == "declare_stuck":
            # Only allow giving up after using 80% of steps
            if self.step_num < self.max_steps * 0.8:
                display.stuck_rejected(self.step_num, self.max_steps)
                return "continue"
            display.stuck()
            return "stop"

        return "continue"

    def _get_plan(self, lemma_index: str, human_feedback: str = "") -> dict | None:
        prompt = prompts.format_plan_prompt(
            theorem=self.theorem_text,
            whiteboard=self.whiteboard,
            lemma_index=lemma_index,
            step_num=self.step_num,
            max_steps=self.max_steps,
            human_feedback=human_feedback,
            verification_result=self.verification_result,
            search_result=self.search_result,
        )
        display.thinking()
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.SYSTEM_PROMPT,
                json_schema=self.plan_schema,
                label=f"plan_step_{self.step_num}",
            )
        except RuntimeError as e:
            print(f"\n  Error: {e}")
            return None
        display.thinking_done(resp["duration_ms"])

        if self.verbose:
            print(f"  [verbose] {resp['result'][:200]}")

        try:
            return json.loads(resp["result"])
        except json.JSONDecodeError:
            print(f"  Warning: failed to parse plan JSON, using fallback")
            return {"action": "continue", "summary": "Continue work", "reasoning": "Fallback due to parse error"}

    def _execute_step(self, lemma_index: str, plan: dict | None = None) -> dict | None:
        prompt = prompts.format_step_prompt(
            theorem=self.theorem_text,
            whiteboard=self.whiteboard,
            lemma_index=lemma_index,
            step_num=self.step_num,
            max_steps=self.max_steps,
            actions=self.actions,
            plan=plan,
            verification_result=self.verification_result,
            search_result=self.search_result,
        )
        # Clear pending results after using them
        self.verification_result = ""
        self.search_result = ""

        display.stream_start()
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.SYSTEM_PROMPT,
                label=f"step_{self.step_num}",
                stream_callback=display.stream_text,
            )
        except RuntimeError as e:
            display.stream_end()
            print(f"\n  Error: {e}")
            return None
        display.stream_end()

        if self.verbose:
            print(f"  [verbose] {resp['result'][:300]}")

        result = prompts.parse_step_output(resp["result"])
        if result is None:
            print(f"  Warning: failed to parse step output")
        return result

    @staticmethod
    def _parse_verdict(text: str) -> bool:
        """Parse verification result. Looks for 'VERDICT: CORRECT/INCORRECT' line."""
        for line in reversed(text.strip().splitlines()):
            line = line.strip().upper()
            if line == "VERDICT: CORRECT":
                return True
            if line == "VERDICT: INCORRECT":
                return False
        # Fallback: conservative — if no verdict line, assume failed
        return False

    def _verify_proof(self, proof: str) -> bool:
        """Auto-verify a declared proof. Returns True if it passes."""
        display.verification_start("declared proof")
        prompt = prompts.format_verify_prompt(self.theorem_text, proof)
        display.stream_start()
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.VERIFY_SYSTEM_PROMPT,
                label=f"verify_proof_step_{self.step_num}",
                stream_callback=display.stream_text,
            )
            display.stream_end()
            self.verification_result = resp["result"]
            passed = self._parse_verdict(resp["result"])
            display.verification_result(passed)
            return passed
        except RuntimeError as e:
            display.stream_end()
            print(f"  Verification error: {e}")
            self.verification_result = f"Verification failed due to error: {e}"
            return False

    def _do_verification(self, target: str, content: str):
        display.verification_start(target)
        prompt = prompts.format_verify_prompt(target, content)
        display.stream_start()
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.VERIFY_SYSTEM_PROMPT,
                label=f"verify_step_{self.step_num}",
                stream_callback=display.stream_text,
            )
            display.stream_end()
            self.verification_result = resp["result"]
            passed = self._parse_verdict(resp["result"])
            display.verification_result(passed)
        except RuntimeError as e:
            display.stream_end()
            print(f"  Verification error: {e}")
            self.verification_result = f"Verification failed due to error: {e}"

    def _do_literature_search(self, query: str):
        display.literature_search(query)
        prompt = prompts.format_literature_search_prompt(query, self.theorem_text)
        display.stream_start()
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.LITERATURE_SEARCH_SYSTEM_PROMPT,
                label=f"search_step_{self.step_num}",
                web_search=True,
                stream_callback=display.stream_text,
            )
            display.stream_end()
            self.search_result = resp["result"]
        except RuntimeError as e:
            display.stream_end()
            print(f"  Search error: {e}")
            self.search_result = f"Literature search failed: {e}"

    def _process_lemmas(self, result: dict):
        # Handle lemma from step output (text section format)
        if result.get("lemma_name") and result.get("lemma_content"):
            name = slugify(result["lemma_name"]) or "unnamed-lemma"
            lemma_dir = self.work_dir / "lemmas" / name
            lemma_dir.mkdir(parents=True, exist_ok=True)

            source = result.get("lemma_source", "")
            content = result["lemma_content"]

            header = f"# {result['lemma_name']}\n\n"
            if source:
                header += f"**Source**: {source}\n\n"
            (lemma_dir / "LEMMA.md").write_text(header + content + "\n")

            proof_text = ""
            if source:
                proof_text = f"**Source**: {source}\n\n"
            proof_text += content
            (lemma_dir / "PROOF.md").write_text(proof_text + "\n")

            display.lemma_stored(result["lemma_name"], source)

    def _save_step(self, result: dict):
        step_dir = self.work_dir / "steps" / f"step_{self.step_num:03d}"
        (step_dir / "output.md").write_text(result.get("whiteboard", ""))
        (step_dir / "action.json").write_text(json.dumps({
            "step": self.step_num,
            "action": result.get("action"),
            "summary": result.get("summary"),
        }, indent=2))

    def _build_lemma_index(self) -> str:
        lemmas_dir = self.work_dir / "lemmas"
        if not lemmas_dir.exists():
            return ""
        entries = []
        for d in sorted(lemmas_dir.iterdir()):
            if not d.is_dir():
                continue
            lemma_file = d / "LEMMA.md"
            if not lemma_file.exists():
                continue
            has_proof = (d / "PROOF.md").exists()
            status = "proven" if has_proof else "unproven"
            content = lemma_file.read_text().strip().split("\n")
            first_line = content[0] if content else d.name
            entries.append(f"- **{d.name}** [{status}]: {first_line}")
        return "\n".join(entries)

    def _write_discussion(self):
        if self.shutting_down:
            display.shutting_down()
        lemma_index = self._build_lemma_index()
        prompt = prompts.format_discussion_prompt(
            theorem=self.theorem_text,
            whiteboard=self.whiteboard,
            lemma_index=lemma_index,
            steps_taken=self.step_num,
            max_steps=self.max_steps,
            proof=self.proof_text,
        )
        display.stream_start()
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.SYSTEM_PROMPT,
                label="discussion",
                stream_callback=display.stream_text,
            )
            display.stream_end()
            (self.work_dir / "DISCUSSION.md").write_text(resp["result"])
            display.discussion_written(str(self.work_dir / "DISCUSSION.md"))
        except RuntimeError as e:
            display.stream_end()
            print(f"  Error generating discussion: {e}")
            (self.work_dir / "DISCUSSION.md").write_text(
                f"# Discussion\n\nSession ended after {self.step_num} steps.\n\n"
                f"## Final Whiteboard\n\n{self.whiteboard}\n"
            )
            display.discussion_written(str(self.work_dir / "DISCUSSION.md"))

    def _do_summary(self):
        """Generate and display a brief progress summary."""
        lemma_index = self._build_lemma_index()
        prompt = prompts.format_summary_prompt(
            theorem=self.theorem_text,
            whiteboard=self.whiteboard,
            lemma_index=lemma_index,
            step_num=self.step_num,
            max_steps=self.max_steps,
        )
        display.stream_start()
        try:
            self.llm.call(
                prompt=prompt,
                system_prompt=prompts.SYSTEM_PROMPT,
                label=f"summary_step_{self.step_num}",
                stream_callback=display.stream_text,
            )
            display.stream_end()
        except RuntimeError as e:
            display.stream_end()
            print(f"  Summary error: {e}")

    def _reset(self):
        """Reset prover state for a fresh start."""
        self.step_num = 0
        self.verification_result = ""
        self.search_result = ""
        self.proof_text = ""
        self.whiteboard = prompts.format_initial_whiteboard(self.theorem_text)
        (self.work_dir / "WHITEBOARD.md").write_text(self.whiteboard)
        # Clear lemmas
        lemmas_dir = self.work_dir / "lemmas"
        if lemmas_dir.exists():
            shutil.rmtree(lemmas_dir)
            lemmas_dir.mkdir()
        display.restarting()

    def _check_stdin_interrupt(self):
        """Non-blocking check if user typed something in autonomous mode."""
        if not sys.stdin.isatty():
            return
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if readable:
                sys.stdin.readline()
                self.autonomous = False
                display.mode_switch("interactive")
        except (OSError, ValueError):
            pass

    def request_shutdown(self):
        """Called by signal handler to request graceful shutdown."""
        self.shutting_down = True
