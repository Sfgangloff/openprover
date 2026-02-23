#!/usr/bin/env python3
"""Serve lm-provers/QED-Nano on multiple GPUs with simple load balancing.

Loads one model replica per GPU and exposes an OpenAI-compatible API
at /v1/chat/completions (streaming and non-streaming) and /v1/models.
"""

import argparse
import json
import logging
import os
import select
import socket
import threading
import time
import traceback
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import LogitsProcessor, StoppingCriteria, StoppingCriteriaList


class ThinkBudgetProcessor(LogitsProcessor):
    """Force </think> after a token budget to reserve room for the answer.

    Thinking models (e.g. QED-Nano) start generating reasoning tokens
    immediately and emit </think> before the answer.  If thinking exceeds
    `budget` new tokens, this processor forces the </think> token sequence
    so the model transitions to answer generation.
    """

    def __init__(self, end_think_ids: list[int], budget: int, prompt_len: int):
        self.end_ids = end_think_ids
        self.budget = budget
        self.prompt_len = prompt_len
        self.in_thinking = True
        self.force_idx = 0  # which token of </think> we're forcing next

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.in_thinking:
            return scores

        # Check if </think> was naturally generated
        n = len(self.end_ids)
        if input_ids.shape[1] >= n:
            last_n = input_ids[0, -n:].tolist()
            if last_n == self.end_ids:
                self.in_thinking = False
                return scores

        generated = input_ids.shape[1] - self.prompt_len

        # Budget exceeded — force </think> token by token
        if generated >= self.budget:
            if self.force_idx < len(self.end_ids):
                target = self.end_ids[self.force_idx]
                self.force_idx += 1
                forced = torch.full_like(scores, float('-inf'))
                forced[:, target] = 0
                return forced
            else:
                self.in_thinking = False

        return scores


class CancelBatchCriteria(StoppingCriteria):
    """Stop generation early when cancellation is requested."""

    def __init__(self, cancel_event: threading.Event):
        self.cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.cancel_event.is_set()


# Request batch item for batched generation
BatchItem = namedtuple("BatchItem", ["messages", "max_tokens", "temperature", "top_p", "thinking_budget"])


class Worker:
    """One model replica on one GPU."""

    _load_lock = threading.Lock()

    def __init__(self, gpu_id: int, model_name: str, dtype: torch.dtype, tokenizer):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.lock = threading.Lock()
        self.tokenizer = tokenizer
        self.busy = False  # Track if worker is processing a batch

        # Pre-compute </think> token IDs for thinking budget enforcement
        self.end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)

        print(f"  Loading on cuda:{gpu_id}...")
        # Serialize from_pretrained to avoid meta-tensor race conditions,
        # but .to(device) runs outside the lock so GPU transfers overlap.
        with Worker._load_lock:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model = model.to(self.device)
        self.model.eval()
        print(f"  cuda:{gpu_id} ready")

    def generate(self, messages, max_tokens, temperature, top_p,
                 stream=False, thinking_budget=None):
        """Single-request generation (legacy, for streaming)."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None

        # Thinking budget enforcement
        if thinking_budget is not None and self.end_think_ids:
            processor = ThinkBudgetProcessor(
                self.end_think_ids, thinking_budget, input_len,
            )
            gen_kwargs["logits_processor"] = [processor]

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True,
            )
            gen_kwargs["streamer"] = streamer
            result = {}

            def _run():
                with torch.inference_mode():
                    output = self.model.generate(**gen_kwargs)
                result["completion_tokens"] = output.shape[1] - input_len

            thread = threading.Thread(target=_run)
            thread.start()
            return streamer, input_len, thread, result
        else:
            with torch.inference_mode():
                output = self.model.generate(**gen_kwargs)
            new_tokens = output[0][input_len:]
            text_out = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return text_out, len(new_tokens), input_len

    def generate_batch(
        self,
        batch_items: list[BatchItem],
        cancel_event: threading.Event | None = None,
    ) -> list[dict]:
        """Batched non-streaming generation.
        
        Returns list of dicts with keys: text, completion_tokens, prompt_tokens
        """
        # Build prompts for all items
        texts = []
        for item in batch_items:
            text = self.tokenizer.apply_chat_template(
                item.messages, tokenize=False, add_generation_prompt=True,
            )
            texts.append(text)

        # Tokenize batch with padding
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        batch_size = inputs.input_ids.shape[0]
        input_lens = (inputs.attention_mask.sum(dim=1)).tolist()

        # All items in batch must have same generation params (enforced by scheduler)
        first = batch_items[0]
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=first.max_tokens,
            do_sample=first.temperature > 0,
        )
        if first.temperature > 0:
            gen_kwargs["temperature"] = first.temperature
            gen_kwargs["top_p"] = first.top_p
        else:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None

        if cancel_event is not None:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                CancelBatchCriteria(cancel_event),
            ])

        # Thinking budget enforcement (if specified, applies to all)
        # Note: ThinkBudgetProcessor is per-sample, so we'd need batch-aware version
        # For now, skip thinking_budget in batched mode (or implement batch processor)
        if first.thinking_budget is not None:
            # TODO: Implement batch-aware ThinkBudgetProcessor if needed
            pass

        # Generate batch
        with torch.inference_mode():
            output = self.model.generate(**gen_kwargs)

        # Decode per-sample and compute stats
        results = []
        for i in range(batch_size):
            input_len = input_lens[i]
            new_tokens = output[i][input_len:]
            text_out = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            completion_tokens = len(new_tokens)
            results.append({
                "text": text_out,
                "completion_tokens": completion_tokens,
                "prompt_tokens": input_len,
            })

        return results


class PendingRequest:
    """Pending request with completion synchronization and cancellation."""
    def __init__(self, batch_item, enqueue_time, completion_event):
        self.batch_item = batch_item
        self.enqueue_time = enqueue_time
        self.completion_event = completion_event
        self.result = None
        self.error = None
        self._cancelled = False
        self._cancel_lock = threading.Lock()
        self.batch_state = None  # Set when request is added to a batch

    def mark_cancelled(self):
        """Mark this request as cancelled."""
        with self._cancel_lock:
            if self._cancelled:
                return  # Already cancelled
            self._cancelled = True
            self.error = RuntimeError("cancelled")
            # Notify batch state if in an active batch
            if self.batch_state is not None:
                self.batch_state.on_request_cancelled()
            self.completion_event.set()

    def is_cancelled(self) -> bool:
        """Check if this request is cancelled."""
        with self._cancel_lock:
            return self._cancelled


class ActiveBatchState:
    """Track live request count for an active batch."""
    def __init__(self, requests: list[PendingRequest]):
        self.lock = threading.Lock()
        self.live_count = len(requests)
        self.cancel_event = threading.Event()
        # Link requests to this batch state
        for req in requests:
            req.batch_state = self

    def on_request_cancelled(self):
        """Called when a request in this batch is cancelled."""
        with self.lock:
            self.live_count -= 1
            if self.live_count <= 0:
                # All requests cancelled - signal stop generation
                self.cancel_event.set()


class BatchScheduler:
    """Dynamic batching scheduler with timeout and batch-size triggers."""

    def __init__(self, workers: list[Worker], batch_size: int, batch_timeout_s: float, verbose: bool = False):
        self.workers = workers
        self.batch_size = batch_size
        self.batch_timeout_s = batch_timeout_s
        self.verbose = verbose
        self.pending_queue = []
        self.queue_lock = threading.Condition()
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

    @staticmethod
    def _config_key(batch_item: BatchItem) -> tuple:
        """Get generation config key for grouping requests."""
        return (batch_item.max_tokens, batch_item.temperature,
                batch_item.top_p, batch_item.thinking_budget)

    def enqueue(self, batch_item: BatchItem) -> PendingRequest:
        """Enqueue a request and return a PendingRequest to wait on."""
        req = PendingRequest(
            batch_item=batch_item,
            enqueue_time=time.time(),
            completion_event=threading.Event(),
        )
        with self.queue_lock:
            self.pending_queue.append(req)
            self.queue_lock.notify()
        return req

    def _scheduler_loop(self):
        """Main scheduler loop: dispatch batches to free workers."""
        while self.running:
            with self.queue_lock:
                # Wait until we have requests or shutdown
                while self.running and len(self.pending_queue) == 0:
                    self.queue_lock.wait(timeout=0.1)

                if not self.running:
                    break

                # Drop canceled requests before scheduling.
                self.pending_queue = [req for req in self.pending_queue if not req.is_cancelled()]
                if not self.pending_queue:
                    continue

                # Find free workers
                free_workers = [w for w in self.workers if not w.busy]
                if not free_workers:
                    # No free workers, wait a bit before checking again
                    self.queue_lock.wait(timeout=0.05)
                    continue

                # Dispatch batches to free workers
                for worker in free_workers:
                    # Keep queue clean while looping workers.
                    self.pending_queue = [req for req in self.pending_queue if not req.is_cancelled()]
                    if not self.pending_queue:
                        break

                    # Recompute groups each dispatch so we never use stale references.
                    config_groups = {}
                    for req in self.pending_queue:
                        key = self._config_key(req.batch_item)
                        if key not in config_groups:
                            config_groups[key] = []
                        config_groups[key].append(req)

                    oldest_req = self.pending_queue[0]
                    oldest_wait = time.time() - oldest_req.enqueue_time

                    batch_requests = None

                    # Priority 1: Full batch available for any config
                    for key, group in config_groups.items():
                        if len(group) >= self.batch_size:
                            batch_requests = group[:self.batch_size]
                            break

                    # Priority 2: Timeout trigger - batch oldest request's config group
                    if not batch_requests and oldest_wait and oldest_wait >= self.batch_timeout_s:
                        oldest_key = self._config_key(oldest_req.batch_item)
                        if oldest_key in config_groups:
                            group = config_groups[oldest_key]
                            batch_requests = group[:min(self.batch_size, len(group))]

                    if not batch_requests:
                        continue

                    # Remove from queue
                    batch_items = [req.batch_item for req in batch_requests]
                    for req in batch_requests:
                        self.pending_queue.remove(req)

                    # Create batch state to track live requests
                    batch_state = ActiveBatchState(batch_requests)

                    # Dispatch to worker in background thread
                    worker.busy = True
                    if self.verbose:
                        print(f"[BATCH] Dispatching batch of size {len(batch_items)} to GPU {worker.gpu_id}")
                    threading.Thread(
                        target=self._process_batch,
                        args=(worker, batch_requests, batch_items, batch_state),
                        daemon=True
                    ).start()

    def _process_batch(self, worker: Worker, requests: list[PendingRequest],
                       batch_items: list[BatchItem],
                       batch_state: ActiveBatchState):
        """Process a batch on a worker and signal completion."""
        batch_size = len(batch_items)
        gpu_id = worker.gpu_id
        try:
            with worker.lock:
                results = worker.generate_batch(batch_items, cancel_event=batch_state.cancel_event)
            
            # Check if batch was cancelled (all requests cancelled)
            if batch_state.cancel_event.is_set():
                # Mark all requests as cancelled if not already
                for req in requests:
                    if not req.is_cancelled():
                        req.mark_cancelled()
                if self.verbose:
                    print(f"[BATCH] Cancelled batch of size {batch_size} on GPU {gpu_id} (all requests cancelled)")
                return

            # Keep index alignment with the original batch order.
            for idx, req in enumerate(requests):
                if req.is_cancelled():
                    continue
                if idx >= len(results):
                    req.error = RuntimeError("missing result")
                    req.completion_event.set()
                    continue
                req.result = results[idx]
                req.completion_event.set()

            if self.verbose:
                completed_count = sum(1 for req in requests if not req.is_cancelled() and req.result is not None)
                total_completion_tokens = sum(
                    req.result["completion_tokens"]
                    for req in requests
                    if req.result is not None and not req.is_cancelled()
                )
                print(f"[BATCH] Completed batch of size {batch_size} on GPU {gpu_id} "
                      f"({completed_count} live, {total_completion_tokens} completion tokens)")
        except Exception as e:
            # Propagate error to non-cancelled requests only
            for req in requests:
                if not req.is_cancelled():
                    req.error = e
                    req.completion_event.set()
            if self.verbose:
                print(f"[BATCH] ERROR: batch of size {batch_size} on GPU {gpu_id} failed: {e}")
        finally:
            worker.busy = False
            # Wake scheduler to check for more work
            with self.queue_lock:
                self.queue_lock.notify()

    def shutdown(self):
        """Stop scheduler and wait for completion."""
        self.running = False
        with self.queue_lock:
            self.queue_lock.notify_all()
        self.scheduler_thread.join(timeout=5.0)


class LoadBalancer:
    """Round-robin worker picker for direct mode."""

    def __init__(self, workers: list[Worker]):
        self.workers = workers
        self._counter = 0
        self._lock = threading.Lock()

    def get_worker(self) -> Worker:
        with self._lock:
            worker = self.workers[self._counter % len(self.workers)]
            self._counter += 1
        return worker


# Globals set in main
scheduler: BatchScheduler = None
lb: LoadBalancer = None
model_name_global: str = ""
batching_enabled: bool = True


class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # quiet

    def do_GET(self):
        if self.path == "/v1/models":
            self._json_response(200, {
                "object": "list",
                "data": [{"id": model_name_global, "object": "model"}],
            })
        elif self.path == "/health":
            self._json_response(200, {"status": "ok"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json_response(404, {"error": "not found"})
            return

        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 4096)
        temperature = body.get("temperature", 0.6)
        top_p = body.get("top_p", 0.95)
        stream = body.get("stream", False)
        thinking_budget = body.get("thinking_budget")  # None = no limit

        if not batching_enabled:
            self._handle_direct_request(
                messages, max_tokens, temperature, top_p, stream, thinking_budget,
            )
            return

        # Batching with batch_size > 1: non-streaming only.
        if stream:
            self._json_response(400, {
                "error": "streaming disabled in batched mode",
                "message": "Set --batch-size 1 to enable streaming",
            })
            return

        # Enqueue for batched processing
        batch_item = BatchItem(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking_budget=thinking_budget,
        )
        req = scheduler.enqueue(batch_item)

        # Wait for completion with disconnect detection
        # Poll in short intervals to check for client disconnect
        timeout_remaining = 600.0  # 10 min total timeout
        poll_interval = 0.1  # 100ms polling
        
        while timeout_remaining > 0:
            wait_time = min(poll_interval, timeout_remaining)
            if req.completion_event.wait(timeout=wait_time):
                # Request completed
                break
            
            timeout_remaining -= wait_time
            
            # Check if client disconnected
            if self._is_client_disconnected():
                req.mark_cancelled()
                return  # Client disconnected, don't send response
        
        if timeout_remaining <= 0:
            req.mark_cancelled()
            self._json_response(504, {"error": "request timeout"})
            return

        # Request completed - check if it was cancelled
        if req.is_cancelled():
            return  # Client disconnected, don't send response

        try:
            if req.error:
                error_text = str(req.error)
                if "cancelled" in error_text.lower():
                    # Already cancelled, don't send response
                    return
                else:
                    self._json_response(500, {"error": error_text})
                return

            result = req.result
            if result is None:
                # Shouldn't happen, but handle gracefully
                return
            
            t0 = int(time.time())
            self._json_response(200, {
                "id": f"chatcmpl-{t0}",
                "object": "chat.completion",
                "created": t0,
                "model": model_name_global,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                },
            })
        except BrokenPipeError:
            # Client disconnected during response write
            req.mark_cancelled()
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._json_response(500, {"error": str(e)})
            except BrokenPipeError:
                req.mark_cancelled()
                pass

    def _is_client_disconnected(self) -> bool:
        """Check if client connection is closed."""
        try:
            sock = self.connection
            if sock is None:
                return True

            # If socket is readable, peek one byte:
            # - b"" => peer closed
            # - BlockingIOError/no data => still connected
            readable, _, _ = select.select([sock], [], [], 0)
            if not readable:
                return False
            try:
                data = sock.recv(1, socket.MSG_PEEK)
                return len(data) == 0
            except BlockingIOError:
                return False
            except (ConnectionResetError, OSError):
                return True
        except Exception:
            # If we can't check, assume still connected (conservative)
            pass
        return False

    def _handle_direct_request(self, messages, max_tokens, temperature, top_p, stream, thinking_budget):
        worker = lb.get_worker()
        with worker.lock:
            try:
                if stream:
                    self._handle_streaming(worker, messages, max_tokens, temperature, top_p, thinking_budget)
                else:
                    self._handle_non_streaming(worker, messages, max_tokens, temperature, top_p, thinking_budget)
            except BrokenPipeError:
                pass
            except Exception as e:
                traceback.print_exc()
                try:
                    self._json_response(500, {"error": str(e)})
                except BrokenPipeError:
                    pass

    def _handle_non_streaming(self, worker, messages, max_tokens, temperature, top_p, thinking_budget=None):
        t0 = time.time()
        text, completion_tokens, prompt_tokens = worker.generate(
            messages,
            max_tokens,
            temperature,
            top_p,
            stream=False,
            thinking_budget=thinking_budget,
        )
        self._json_response(200, {
            "id": f"chatcmpl-{int(t0)}",
            "object": "chat.completion",
            "created": int(t0),
            "model": model_name_global,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(completion_tokens),
                "total_tokens": prompt_tokens + int(completion_tokens),
            },
        })

    def _handle_streaming(self, worker, messages, max_tokens, temperature, top_p, thinking_budget=None):
        streamer, input_len, gen_thread, result = worker.generate(
            messages,
            max_tokens,
            temperature,
            top_p,
            stream=True,
            thinking_budget=thinking_budget,
        )

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        chunk_id = f"chatcmpl-{int(time.time())}"

        for token_text in streamer:
            if not token_text:
                continue
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": model_name_global,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        gen_thread.join()
        completion_tokens = result["completion_tokens"]

        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": model_name_global,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": completion_tokens,
                "total_tokens": input_len + completion_tokens,
            },
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    global scheduler, lb, model_name_global, batching_enabled

    parser = argparse.ArgumentParser(
        description="Serve a HuggingFace model on multiple GPUs with dynamic batching",
    )
    parser.add_argument("--model", default="lm-provers/QED-Nano",
                        help="HuggingFace model name (default: lm-provers/QED-Nano)")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs / model replicas (default: 8)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (default: bfloat16)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for dynamic batching (default: 32)")
    parser.add_argument("--batch-timeout", type=float, default=1.0,
                        help="Batch timeout in seconds (default: 1.0)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print batch processing information")
    args = parser.parse_args()
    batching_enabled = args.batch_size > 1

    n_available = torch.cuda.device_count()
    if n_available < args.num_gpus:
        print(f"WARNING: requested {args.num_gpus} GPUs but only {n_available} available, using {n_available}")
        args.num_gpus = n_available
    if args.num_gpus == 0:
        print("ERROR: no GPUs available")
        return

    model_name_global = args.model
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"Loading {args.num_gpus} replica(s) of {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load first replica with full output (shows warnings like MISSING weights)
    workers = [Worker(0, args.model, dtype, tokenizer)]

    if args.num_gpus > 1:
        # Suppress noisy progress bars / load reports for remaining replicas
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        with ThreadPoolExecutor(max_workers=args.num_gpus - 1) as pool:
            workers += list(pool.map(
                lambda i: Worker(i, args.model, dtype, tokenizer),
                range(1, args.num_gpus),
            ))
    if batching_enabled:
        scheduler = BatchScheduler(workers, args.batch_size, args.batch_timeout, args.verbose)
    else:
        lb = LoadBalancer(workers)

    server = ThreadedHTTPServer((args.host, args.port), Handler)
    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"  Model: {args.model}")
    print(f"  Mode:  {'batched' if batching_enabled else 'direct'} (derived from batch_size)")
    print(f"  GPUs:  {args.num_gpus}")
    print(f"  dtype: {args.dtype}")
    if batching_enabled:
        print(f"  Batch size: {args.batch_size}")
        print(f"  Batch timeout: {args.batch_timeout}s")
    else:
        print("  Batch size: 1 (streaming/direct path enabled)")
    if args.verbose:
        print(f"  Verbose mode: enabled")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        if scheduler is not None:
            scheduler.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
