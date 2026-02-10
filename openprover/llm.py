"""Claude CLI wrapper for LLM calls."""

import json
import subprocess
import time
from pathlib import Path


class LLMClient:
    """Calls Claude via the CLI and archives all interactions."""

    def __init__(self, model: str, archive_dir: Path):
        self.model = model
        self.archive_dir = archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0
        self.total_cost = 0.0

    def call(
        self,
        prompt: str,
        system_prompt: str,
        json_schema: dict | None = None,
        label: str = "",
        web_search: bool = False,
        stream_callback=None,
    ) -> dict:
        """Make an LLM call and archive it.

        Args:
            web_search: If True, enable WebSearch tool instead of disabling all tools.
            stream_callback: If provided, called with text chunks as they arrive.
                Only works for non-schema calls. Signature: callback(text: str).

        Returns dict with keys: result (str), cost (float), duration_ms (int),
        raw (full JSON response).
        """
        self.call_count += 1
        call_num = self.call_count

        # Use streaming for non-schema calls when callback is provided
        use_streaming = stream_callback and not json_schema

        cmd = [
            "claude", "-p",
            "--model", self.model,
            "--system-prompt", system_prompt,
        ]

        if use_streaming:
            cmd.extend(["--output-format", "stream-json", "--verbose",
                         "--include-partial-messages"])
        else:
            cmd.extend(["--output-format", "json"])

        if web_search:
            cmd.extend(["--permission-mode", "bypassPermissions",
                         "--allowedTools", "WebSearch WebFetch"])
        else:
            cmd.extend(["--tools", ""])
        if json_schema:
            cmd.extend(["--json-schema", json.dumps(json_schema)])

        start = time.time()

        if use_streaming:
            return self._call_streaming(
                cmd, prompt, system_prompt, json_schema,
                call_num, label, start, stream_callback,
            )

        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
        )
        elapsed_ms = int((time.time() - start) * 1000)

        if proc.returncode != 0:
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, proc.stderr, elapsed_ms)
            raise RuntimeError(f"Claude CLI failed (exit {proc.returncode}): {proc.stderr[:500]}")

        try:
            raw = json.loads(proc.stdout)
        except json.JSONDecodeError:
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, proc.stdout[:1000], elapsed_ms)
            raise RuntimeError(f"Failed to parse Claude response as JSON: {proc.stdout[:500]}")

        cost = raw.get("total_cost_usd", 0.0)
        self.total_cost += cost

        # Check for structured output failures
        subtype = raw.get("subtype", "")
        if "error" in subtype:
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          raw, subtype, elapsed_ms)
            raise RuntimeError(f"Claude CLI error: {subtype}")

        # When using --json-schema, structured output is in 'structured_output'
        if json_schema and "structured_output" in raw:
            result_text = json.dumps(raw["structured_output"])
        else:
            result_text = raw.get("result", "")

        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      raw, None, elapsed_ms)

        return {
            "result": result_text,
            "cost": cost,
            "duration_ms": raw.get("duration_ms", elapsed_ms),
            "raw": raw,
        }

    def _call_streaming(self, cmd, prompt, system_prompt, json_schema,
                        call_num, label, start, callback):
        """Stream text deltas to callback, return final result."""
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        proc.stdin.write(prompt)
        proc.stdin.close()

        result_data = None
        # Use readline() instead of iterator — the iterator uses an internal
        # read-ahead buffer that defeats real-time streaming.
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "stream_event":
                event = msg.get("event", {})
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        callback(text)
            elif msg.get("type") == "result":
                result_data = msg

        proc.wait()
        elapsed_ms = int((time.time() - start) * 1000)

        if result_data is None:
            stderr = proc.stderr.read()
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, stderr, elapsed_ms)
            raise RuntimeError(f"No result from streaming call: {stderr[:500]}")

        cost = result_data.get("total_cost_usd", 0.0)
        self.total_cost += cost
        result_text = result_data.get("result", "")

        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      result_data, None, elapsed_ms)

        return {
            "result": result_text,
            "cost": cost,
            "duration_ms": result_data.get("duration_ms", elapsed_ms),
            "raw": result_data,
        }

    def _archive(self, call_num, label, prompt, system_prompt, json_schema,
                 response, error, elapsed_ms):
        record = {
            "call_num": call_num,
            "label": label,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "json_schema": json_schema,
            "response": response,
            "error": error,
            "elapsed_ms": elapsed_ms,
        }
        path = self.archive_dir / f"call_{call_num:03d}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
