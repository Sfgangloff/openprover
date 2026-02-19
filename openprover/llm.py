"""LLM client wrappers — Claude CLI and OpenAI-compatible HTTP."""

import json
import subprocess
import time
import urllib.request
import urllib.error
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
        archive_path: Path | None = None,
    ) -> dict:
        """Make an LLM call and archive it.

        Args:
            web_search: If True, enable WebSearch tool instead of disabling all tools.
            stream_callback: If provided, called with text chunks as they arrive.
                Signature: callback(text: str).

        Returns dict with keys: result (str), cost (float), duration_ms (int),
        raw (full JSON response).
        """
        self.call_count += 1
        call_num = self.call_count

        use_streaming = bool(stream_callback)

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
                call_num, label, start, stream_callback, archive_path,
            )

        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
        )
        elapsed_ms = int((time.time() - start) * 1000)

        if proc.returncode != 0:
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, proc.stderr, elapsed_ms, archive_path)
            raise RuntimeError(f"Claude CLI failed (exit {proc.returncode}): {proc.stderr[:500]}")

        try:
            raw = json.loads(proc.stdout)
        except json.JSONDecodeError:
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, proc.stdout[:1000], elapsed_ms, archive_path)
            raise RuntimeError(f"Failed to parse Claude response as JSON: {proc.stdout[:500]}")

        cost = raw.get("total_cost_usd", 0.0)
        self.total_cost += cost

        # Check for structured output failures
        subtype = raw.get("subtype", "")
        if "error" in subtype:
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          raw, subtype, elapsed_ms, archive_path)
            raise RuntimeError(f"Claude CLI error: {subtype}")

        # When using --json-schema, structured output is in 'structured_output'
        if json_schema and "structured_output" in raw:
            result_text = json.dumps(raw["structured_output"])
        else:
            result_text = raw.get("result", "")

        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      raw, None, elapsed_ms, archive_path)

        return {
            "result": result_text,
            "cost": cost,
            "duration_ms": raw.get("duration_ms", elapsed_ms),
            "raw": raw,
        }

    def _call_streaming(self, cmd, prompt, system_prompt, json_schema,
                        call_num, label, start, callback, archive_path=None):
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
                    text = delta.get("text", "") or delta.get("thinking", "")
                    if text:
                        callback(text)
            elif msg.get("type") == "result":
                result_data = msg

        proc.wait()
        elapsed_ms = int((time.time() - start) * 1000)

        if result_data is None:
            stderr = proc.stderr.read()
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, stderr, elapsed_ms, archive_path)
            raise RuntimeError(f"No result from streaming call: {stderr[:500]}")

        cost = result_data.get("total_cost_usd", 0.0)
        self.total_cost += cost
        if json_schema and "structured_output" in result_data:
            result_text = json.dumps(result_data["structured_output"])
        else:
            result_text = result_data.get("result", "")

        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      result_data, None, elapsed_ms, archive_path)

        return {
            "result": result_text,
            "cost": cost,
            "duration_ms": result_data.get("duration_ms", elapsed_ms),
            "raw": result_data,
        }

    def _archive(self, call_num, label, prompt, system_prompt, json_schema,
                 response, error, elapsed_ms, archive_path=None):
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
        if archive_path:
            path = archive_path
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path = self.archive_dir / f"call_{call_num:03d}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))


class HFClient:
    """Calls an OpenAI-compatible HTTP server (e.g. serve_hf.py) and archives interactions."""

    def __init__(self, model: str, archive_dir: Path, base_url: str = "http://localhost:8000"):
        self.model = model
        self.base_url = base_url.rstrip("/")
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
        archive_path: Path | None = None,
    ) -> dict:
        """Make an LLM call via HTTP and archive it.

        Same interface as LLMClient.call(). web_search and json_schema are ignored.
        """
        self.call_count += 1
        call_num = self.call_count

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 32768,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": bool(stream_callback),
        }

        start = time.time()

        try:
            if stream_callback:
                return self._call_streaming(
                    payload, prompt, system_prompt, json_schema,
                    call_num, label, start, stream_callback, archive_path,
                )
            else:
                return self._call_non_streaming(
                    payload, prompt, system_prompt, json_schema,
                    call_num, label, start, archive_path,
                )
        except (urllib.error.URLError, ConnectionError) as e:
            elapsed_ms = int((time.time() - start) * 1000)
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, str(e), elapsed_ms, archive_path)
            raise RuntimeError(f"HF server request failed: {e}")

    def _call_non_streaming(self, payload, prompt, system_prompt, json_schema,
                            call_num, label, start, archive_path):
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=600)
        raw = json.loads(resp.read())
        elapsed_ms = int((time.time() - start) * 1000)

        result_text = raw["choices"][0]["message"]["content"]
        usage = raw.get("usage", {})

        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      raw, None, elapsed_ms, archive_path)

        return {
            "result": result_text,
            "cost": 0.0,
            "duration_ms": elapsed_ms,
            "raw": raw,
        }

    def _call_streaming(self, payload, prompt, system_prompt, json_schema,
                        call_num, label, start, callback, archive_path):
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=600)

        collected = []
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                callback(content)
                collected.append(content)

        elapsed_ms = int((time.time() - start) * 1000)
        result_text = "".join(collected)

        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      {"result": result_text}, None, elapsed_ms, archive_path)

        return {
            "result": result_text,
            "cost": 0.0,
            "duration_ms": elapsed_ms,
            "raw": {"result": result_text},
        }

    def _archive(self, call_num, label, prompt, system_prompt, json_schema,
                 response, error, elapsed_ms, archive_path=None):
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
        if archive_path:
            path = archive_path
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path = self.archive_dir / f"call_{call_num:03d}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
