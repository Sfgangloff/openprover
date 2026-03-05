"""LLM client wrappers — Claude CLI and OpenAI-compatible HTTP."""

import json
import logging
import re
import subprocess
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger("openprover.llm")


class Interrupted(Exception):
    """Raised when an LLM call is cancelled via interrupt()."""
    pass


class StreamingUnavailable(RuntimeError):
    """Raised when HF server cannot stream in current configuration."""
    pass


def _split_think_tags(text: str) -> tuple[str, str]:
    """Split thinking from model output at </think> boundary.

    QED-Nano starts thinking immediately (no <think> tag) and ends with
    </think> before the answer. Also handles <think>...</think> format.

    Returns (result_text, thinking_text). If no </think> found,
    returns (original_text, "").
    """
    # Strip optional opening <think> tag
    stripped = text.lstrip()
    if stripped.startswith("<think>"):
        stripped = stripped[7:]
    else:
        stripped = text

    idx = stripped.find("</think>")
    if idx >= 0:
        thinking = stripped[:idx].strip()
        result = stripped[idx + 8:].strip()
        return (result or thinking, thinking)
    return (text, "")


class LLMClient:
    """Calls Claude via the CLI and archives all interactions."""

    def __init__(self, model: str, archive_dir: Path):
        self.model = model
        self.archive_dir = archive_dir
        self.call_count = 0
        self.total_cost = 0.0
        self._interrupted = threading.Event()
        self._active_procs: list[subprocess.Popen] = []
        self._procs_lock = threading.Lock()

    def interrupt(self):
        """Signal all active LLM calls to stop."""
        self._interrupted.set()
        with self._procs_lock:
            for proc in self._active_procs:
                if proc.poll() is None:
                    proc.kill()

    def clear_interrupt(self):
        """Reset the interrupt flag so new calls can proceed."""
        self._interrupted.clear()

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

        # Archive input immediately so it can be inspected while LLM runs
        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      None, None, 0, archive_path)

        use_streaming = bool(stream_callback)
        logger.info("[%s] calling %s%s", label, self.model,
                    " (streaming)" if use_streaming else "")

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

        duration = raw.get("duration_ms", elapsed_ms)
        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      raw, None, elapsed_ms, archive_path,
                      result_text=result_text)
        logger.info("[%s] done %dms $%.4f", label, duration, cost)

        return {
            "result": result_text,
            "thinking": "",
            "cost": cost,
            "duration_ms": duration,
            "raw": raw,
            "finish_reason": raw.get("stop_reason", "end_turn"),
        }

    def _call_streaming(self, cmd, prompt, system_prompt, json_schema,
                        call_num, label, start, callback, archive_path=None):
        """Stream text deltas to callback, return final result."""
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        with self._procs_lock:
            self._active_procs.append(proc)
        proc.stdin.write(prompt)
        proc.stdin.close()

        result_data = None
        thinking_parts = []
        interrupted = False
        # Use readline() instead of iterator — the iterator uses an internal
        # read-ahead buffer that defeats real-time streaming.
        try:
            while True:
                if self._interrupted.is_set():
                    interrupted = True
                    proc.kill()
                    break
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
                        thinking = delta.get("thinking", "")
                        text = delta.get("text", "")
                        if thinking:
                            thinking_parts.append(thinking)
                            callback(thinking, "thinking")
                        elif text:
                            callback(text, "text")
                elif msg.get("type") == "result":
                    result_data = msg
        finally:
            with self._procs_lock:
                if proc in self._active_procs:
                    self._active_procs.remove(proc)
            proc.wait()

        elapsed_ms = int((time.time() - start) * 1000)

        if interrupted:
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, "interrupted", elapsed_ms, archive_path)
            raise Interrupted()

        if result_data is None:
            stderr = proc.stderr.read()
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, stderr, elapsed_ms, archive_path)
            raise RuntimeError(f"No result from streaming call: {stderr[:500]}")

        subtype = result_data.get("subtype", "")
        if result_data.get("is_error") or "error" in subtype:
            err = result_data.get("result", "") or subtype or "streaming error"
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          result_data, err, elapsed_ms, archive_path)
            raise RuntimeError(f"Claude CLI streaming error: {err[:500]}")

        cost = result_data.get("total_cost_usd", 0.0)
        self.total_cost += cost
        if json_schema and "structured_output" in result_data:
            result_text = json.dumps(result_data["structured_output"])
        else:
            result_text = result_data.get("result", "")

        thinking_text = "".join(thinking_parts)
        duration = result_data.get("duration_ms", elapsed_ms)
        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      result_data, None, elapsed_ms, archive_path,
                      thinking=thinking_text, result_text=result_text)
        logger.info("[%s] done %dms $%.4f", label, duration, cost)

        return {
            "result": result_text,
            "thinking": thinking_text,
            "cost": cost,
            "duration_ms": duration,
            "raw": result_data,
            "finish_reason": result_data.get("stop_reason", "end_turn"),
        }

    def _archive(self, call_num, label, prompt, system_prompt, json_schema,
                 response, error, elapsed_ms, archive_path=None,
                 *, thinking="", result_text=""):
        record = {
            "call_num": call_num,
            "label": label,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "json_schema": json_schema,
            "result_text": result_text,
            "thinking": thinking,
            "response": response,
            "error": error,
            "elapsed_ms": elapsed_ms,
        }
        if archive_path:
            path = archive_path
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            path = self.archive_dir / f"call_{call_num:03d}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))


MODEL_CONTEXT_LENGTHS = {
    "lm-provers/QED-Nano": 49152,
    "Qwen/Qwen3-4B-Thinking-2507": 32768,
    "MiniMaxAI/MiniMax-M2.5": 196608,
}


class HFClient:
    """Calls an OpenAI-compatible HTTP server (e.g. serve_hf.py) and archives interactions."""

    def __init__(self, model: str, archive_dir: Path, base_url: str = "http://localhost:8000",
                 answer_reserve: int = 4096, vllm: bool = False):
        if model not in MODEL_CONTEXT_LENGTHS:
            raise ValueError(
                f"Unknown model {model!r}. "
                f"Known models: {', '.join(MODEL_CONTEXT_LENGTHS)}"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.archive_dir = archive_dir
        self.call_count = 0
        self.total_cost = 0.0
        self.vllm = vllm
        self._interrupted = threading.Event()
        self.max_context_length = MODEL_CONTEXT_LENGTHS[model]
        self.max_output_tokens = self.max_context_length
        self.answer_reserve = answer_reserve
        self.max_thinking_tokens = max(
            self.max_context_length - answer_reserve,
            self.max_context_length // 2,
        )
        self._check_server()

    def interrupt(self):
        """Signal the current LLM call to stop."""
        self._interrupted.set()

    def clear_interrupt(self):
        """Reset the interrupt flag so new calls can proceed."""
        self._interrupted.clear()

    def _check_server(self):
        """Verify the server is reachable. Fail fast with a clear error."""
        try:
            resp = urllib.request.urlopen(f"{self.base_url}/health", timeout=5)
            resp.read()  # drain response (may be empty for vLLM)
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            hint = "vllm serve ..." if self.vllm else "python scripts/serve_hf.py"
            raise SystemExit(
                f"Error: cannot reach server at {self.base_url}\n"
                f"  Start it with: {hint}\n"
                f"  ({e})"
            )

    def call(
        self,
        prompt: str,
        system_prompt: str,
        json_schema: dict | None = None,
        label: str = "",
        web_search: bool = False,
        stream_callback=None,
        archive_path: Path | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Make an LLM call via HTTP and archive it.

        Same interface as LLMClient.call(). web_search and json_schema are ignored.
        max_tokens overrides the default token budget for this call.
        """
        self.call_count += 1
        call_num = self.call_count

        # Archive input immediately so it can be inspected while LLM runs
        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      None, None, 0, archive_path)

        logger.info("[%s] calling %s%s", label, self.model,
                    " (streaming)" if stream_callback else "")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        effective_max_tokens = max_tokens if max_tokens is not None else (
            self.max_thinking_tokens + self.answer_reserve if self.vllm
            else self.max_output_tokens
        )
        if self.vllm:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": effective_max_tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "stream": bool(stream_callback),
            }
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_output_tokens": effective_max_tokens,
                "max_thinking_tokens": self.max_thinking_tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "stream": bool(stream_callback),
            }

        start = time.time()

        if self._interrupted.is_set():
            elapsed_ms = int((time.time() - start) * 1000)
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, "interrupted", elapsed_ms, archive_path)
            raise Interrupted()

        try:
            if stream_callback:
                try:
                    return self._call_streaming(
                        payload, prompt, system_prompt, json_schema,
                        call_num, label, start, stream_callback, archive_path,
                    )
                except StreamingUnavailable:
                    # Keep callers working when server runs with batch_size > 1.
                    # Fallback to non-streaming and emit the full text once.
                    out = self._call_non_streaming(
                        {**payload, "stream": False},
                        prompt, system_prompt, json_schema,
                        call_num, label, start, archive_path,
                    )
                    if out.get("result"):
                        stream_callback(out["result"], "text")
                    return out
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
        if self._interrupted.is_set():
            elapsed_ms = int((time.time() - start) * 1000)
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, "interrupted", elapsed_ms, archive_path)
            raise Interrupted()

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            raw = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            elapsed_ms = int((time.time() - start) * 1000)
            if e.code == 499:
                self._archive(call_num, label, prompt, system_prompt, json_schema,
                              None, "interrupted", elapsed_ms, archive_path)
                raise Interrupted()
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, f"HTTP {e.code}: {body}", elapsed_ms, archive_path)
            raise
        elapsed_ms = int((time.time() - start) * 1000)

        if self._interrupted.is_set():
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, "interrupted", elapsed_ms, archive_path)
            raise Interrupted()

        choice = raw["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")
        reasoning = msg.get("reasoning_content") or ""
        full_text = msg.get("content", "")
        if reasoning:
            result_text, thinking_text = full_text, reasoning
        else:
            result_text, thinking_text = _split_think_tags(full_text)

        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      raw, None, elapsed_ms, archive_path,
                      thinking=thinking_text, result_text=result_text)
        logger.info("[%s] done %dms", label, elapsed_ms)

        return {
            "result": result_text,
            "thinking": thinking_text,
            "cost": 0.0,
            "duration_ms": elapsed_ms,
            "raw": raw,
            "finish_reason": finish_reason,
        }

    def _call_streaming(self, payload, prompt, system_prompt, json_schema,
                        call_num, label, start, callback, archive_path):
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            elapsed_ms = int((time.time() - start) * 1000)
            if e.code == 499:
                self._archive(call_num, label, prompt, system_prompt, json_schema,
                              None, "interrupted", elapsed_ms, archive_path)
                raise Interrupted()
            if e.code == 400 and "streaming disabled in batched mode" in body.lower():
                # server uses batch_size > 1 and does not support streaming there
                raise StreamingUnavailable(
                    "HF streaming unavailable in batched mode "
                    "(set serve_hf --batch-size 1 for streaming)"
                )
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, f"HTTP {e.code}: {body}", elapsed_ms, archive_path)
            raise

        thinking_parts: list[str] = []
        output_parts: list[str] = []
        in_thinking = not self.vllm  # serve_hf starts in thinking; vLLM uses reasoning_content
        pending = ""         # buffer for partial </think> detection (serve_hf only)
        interrupted = False
        finish_reason = "stop"

        for raw_line in resp:
            if self._interrupted.is_set():
                interrupted = True
                resp.close()
                break
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
            choice = chunk.get("choices", [{}])[0]
            chunk_finish = choice.get("finish_reason")
            if chunk_finish:
                finish_reason = chunk_finish
            delta = choice.get("delta", {})

            if self.vllm:
                # vLLM with --reasoning-parser separates thinking/content
                reasoning = delta.get("reasoning_content", "")
                content = delta.get("content", "")
                if reasoning:
                    callback(reasoning, "thinking")
                    thinking_parts.append(reasoning)
                if content:
                    callback(content, "text")
                    output_parts.append(content)
            else:
                content = delta.get("content", "")
                if not content:
                    continue

                if in_thinking:
                    pending += content
                    # Look for </think> in pending buffer
                    end_idx = pending.find("</think>")
                    if end_idx >= 0:
                        think_part = pending[:end_idx]
                        if think_part:
                            callback(think_part, "thinking")
                            thinking_parts.append(think_part)
                        in_thinking = False
                        remainder = pending[end_idx + 8:]
                        pending = ""
                        if remainder.strip():
                            callback(remainder, "text")
                            output_parts.append(remainder)
                    else:
                        # Emit all but last 8 chars (could be partial </think>)
                        safe = len(pending) - 8
                        if safe > 0:
                            callback(pending[:safe], "thinking")
                            thinking_parts.append(pending[:safe])
                            pending = pending[safe:]
                else:
                    callback(content, "text")
                    output_parts.append(content)

        # Flush any remaining pending buffer (serve_hf only)
        if pending:
            kind = "thinking" if in_thinking else "text"
            callback(pending, kind)
            (thinking_parts if in_thinking else output_parts).append(pending)

        elapsed_ms = int((time.time() - start) * 1000)

        if interrupted:
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, "interrupted", elapsed_ms, archive_path)
            raise Interrupted()

        thinking_text = "".join(thinking_parts)
        result_text = "".join(output_parts)
        if not result_text and thinking_text:
            # No </think> found — treat everything as output
            result_text = thinking_text
            thinking_text = ""

        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      {"result": result_text}, None, elapsed_ms, archive_path,
                      thinking=thinking_text, result_text=result_text)
        logger.info("[%s] done %dms", label, elapsed_ms)

        return {
            "result": result_text,
            "thinking": thinking_text,
            "cost": 0.0,
            "duration_ms": elapsed_ms,
            "raw": {"result": result_text},
            "finish_reason": finish_reason,
        }

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        label: str = "",
        stream_callback=None,
        archive_path: Path | None = None,
    ) -> dict:
        """Multi-turn chat with optional tool calling (vLLM only).

        Args:
            messages: OpenAI-format message list.
            tools: OpenAI-format tool definitions, or None.
            max_tokens: Token budget for this call.
            stream_callback: If provided, called with (text, kind) chunks.
            archive_path: Override archive file path.

        Returns dict with keys: result, thinking, cost, duration_ms, raw,
        finish_reason, tool_calls.
        """
        if not self.vllm:
            raise RuntimeError("chat() is only supported for vLLM models")

        self.call_count += 1
        call_num = self.call_count

        effective_max_tokens = max_tokens if max_tokens is not None else (
            self.max_thinking_tokens + self.answer_reserve
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": bool(stream_callback),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Archive input
        prompt_text = json.dumps(messages, ensure_ascii=False)
        self._archive(call_num, label, prompt_text, "", None,
                      None, None, 0, archive_path)

        logger.info("[%s] chat %s%s", label, self.model,
                    " (streaming)" if stream_callback else "")
        start = time.time()

        if self._interrupted.is_set():
            elapsed_ms = int((time.time() - start) * 1000)
            self._archive(call_num, label, prompt_text, "", None,
                          None, "interrupted", elapsed_ms, archive_path)
            raise Interrupted()

        try:
            if stream_callback:
                return self._chat_streaming(
                    payload, prompt_text, call_num, label, start,
                    stream_callback, archive_path,
                )
            else:
                return self._chat_non_streaming(
                    payload, prompt_text, call_num, label, start, archive_path,
                )
        except (urllib.error.URLError, ConnectionError) as e:
            elapsed_ms = int((time.time() - start) * 1000)
            self._archive(call_num, label, prompt_text, "", None,
                          None, str(e), elapsed_ms, archive_path)
            raise RuntimeError(f"vLLM chat request failed: {e}")

    def _chat_non_streaming(self, payload, prompt_text, call_num, label,
                            start, archive_path):
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            raw = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            elapsed_ms = int((time.time() - start) * 1000)
            if e.code == 499:
                self._archive(call_num, label, prompt_text, "", None,
                              None, "interrupted", elapsed_ms, archive_path)
                raise Interrupted()
            self._archive(call_num, label, prompt_text, "", None,
                          None, f"HTTP {e.code}: {body}", elapsed_ms, archive_path)
            raise
        elapsed_ms = int((time.time() - start) * 1000)

        choice = raw["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")
        reasoning = msg.get("reasoning_content") or ""
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        self._archive(call_num, label, prompt_text, "", None,
                      raw, None, elapsed_ms, archive_path,
                      thinking=reasoning, result_text=content)
        logger.info("[%s] done %dms finish=%s tools=%d", label, elapsed_ms,
                    finish_reason, len(tool_calls) if tool_calls else 0)

        return {
            "result": content,
            "thinking": reasoning,
            "cost": 0.0,
            "duration_ms": elapsed_ms,
            "raw": raw,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }

    def _chat_streaming(self, payload, prompt_text, call_num, label,
                        start, callback, archive_path):
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            elapsed_ms = int((time.time() - start) * 1000)
            if e.code == 499:
                self._archive(call_num, label, prompt_text, "", None,
                              None, "interrupted", elapsed_ms, archive_path)
                raise Interrupted()
            self._archive(call_num, label, prompt_text, "", None,
                          None, f"HTTP {e.code}: {body}", elapsed_ms, archive_path)
            raise

        thinking_parts: list[str] = []
        output_parts: list[str] = []
        tool_call_acc: dict[int, dict] = {}  # index → {id, function: {name, arguments}}
        finish_reason = "stop"
        interrupted = False

        for raw_line in resp:
            if self._interrupted.is_set():
                interrupted = True
                resp.close()
                break
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

            choice = chunk.get("choices", [{}])[0]
            chunk_finish = choice.get("finish_reason")
            if chunk_finish:
                finish_reason = chunk_finish
            delta = choice.get("delta", {})

            # Reasoning content
            reasoning = delta.get("reasoning_content", "")
            if reasoning:
                callback(reasoning, "thinking")
                thinking_parts.append(reasoning)

            # Text content
            content = delta.get("content", "")
            if content:
                callback(content, "text")
                output_parts.append(content)

            # Tool call deltas
            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                if idx not in tool_call_acc:
                    tool_call_acc[idx] = {
                        "id": tc_delta.get("id", ""),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                acc = tool_call_acc[idx]
                if tc_delta.get("id"):
                    acc["id"] = tc_delta["id"]
                fn = tc_delta.get("function", {})
                if fn.get("name"):
                    acc["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    acc["function"]["arguments"] += fn["arguments"]

        elapsed_ms = int((time.time() - start) * 1000)

        if interrupted:
            self._archive(call_num, label, prompt_text, "", None,
                          None, "interrupted", elapsed_ms, archive_path)
            raise Interrupted()

        thinking_text = "".join(thinking_parts)
        result_text = "".join(output_parts)
        tool_calls = ([tool_call_acc[i] for i in sorted(tool_call_acc)]
                      if tool_call_acc else None)

        self._archive(call_num, label, prompt_text, "", None,
                      {"result": result_text, "tool_calls": tool_calls},
                      None, elapsed_ms, archive_path,
                      thinking=thinking_text, result_text=result_text)
        logger.info("[%s] done %dms finish=%s tools=%d", label, elapsed_ms,
                    finish_reason, len(tool_calls) if tool_calls else 0)

        return {
            "result": result_text,
            "thinking": thinking_text,
            "cost": 0.0,
            "duration_ms": elapsed_ms,
            "raw": {"result": result_text, "tool_calls": tool_calls},
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }

    def _archive(self, call_num, label, prompt, system_prompt, json_schema,
                 response, error, elapsed_ms, archive_path=None,
                 *, thinking="", result_text=""):
        record = {
            "call_num": call_num,
            "label": label,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "json_schema": json_schema,
            "result_text": result_text,
            "thinking": thinking,
            "response": response,
            "error": error,
            "elapsed_ms": elapsed_ms,
        }
        if archive_path:
            path = archive_path
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            path = self.archive_dir / f"call_{call_num:03d}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
