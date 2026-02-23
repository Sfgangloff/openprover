#!/usr/bin/env python3
"""Serve lm-provers/QED-Nano on multiple GPUs with simple load balancing.

Loads one model replica per GPU and exposes an OpenAI-compatible API
at /v1/chat/completions (streaming and non-streaming) and /v1/models.
"""

import argparse
import json
import logging
import os
import threading
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import LogitsProcessor


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

    def generate_batch(self, batch_items: list[BatchItem]) -> list[dict]:
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
    """Pending request with completion synchronization."""
    def __init__(self, batch_item, enqueue_time, completion_event):
        self.batch_item = batch_item
        self.enqueue_time = enqueue_time
        self.completion_event = completion_event
        self.result = None
        self.error = None


class BatchScheduler:
    """Dynamic batching scheduler with timeout and batch-size triggers."""

    def __init__(self, workers: list[Worker], batch_size: int, batch_timeout_s: float):
        self.workers = workers
        self.batch_size = batch_size
        self.batch_timeout_s = batch_timeout_s
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

                # Find free workers
                free_workers = [w for w in self.workers if not w.busy]
                if len(self.pending_queue) == 0:
                    continue
                if not free_workers:
                    # No free workers, wait a bit before checking again
                    self.queue_lock.wait(timeout=0.05)
                    continue

                # Group pending requests by generation config key
                config_groups = {}
                for req in self.pending_queue:
                    key = self._config_key(req.batch_item)
                    if key not in config_groups:
                        config_groups[key] = []
                    config_groups[key].append(req)

                # Check timeout trigger: oldest request waited too long
                now = time.time()
                oldest_wait = None
                oldest_req = None
                if self.pending_queue:
                    oldest_req = self.pending_queue[0]
                    oldest_wait = now - oldest_req.enqueue_time

                # Dispatch batches to free workers
                for worker in free_workers:
                    if len(self.pending_queue) == 0:
                        break

                    # Find a config group to batch
                    batch_requests = None
                    
                    # Priority 1: Full batch available for any config
                    for key, group in config_groups.items():
                        if len(group) >= self.batch_size:
                            batch_requests = group[:self.batch_size]
                            break
                    
                    # Priority 2: Timeout trigger - batch oldest request's config group
                    if not batch_requests and oldest_wait and oldest_wait >= self.batch_timeout_s:
                        if oldest_req:
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

                    # Dispatch to worker in background thread
                    worker.busy = True
                    threading.Thread(
                        target=self._process_batch,
                        args=(worker, batch_requests, batch_items),
                        daemon=True
                    ).start()

    def _process_batch(self, worker: Worker, requests: list[PendingRequest],
                       batch_items: list[BatchItem]):
        """Process a batch on a worker and signal completion."""
        try:
            with worker.lock:
                results = worker.generate_batch(batch_items)
            # Assign results to requests
            for req, result in zip(requests, results):
                req.result = result
                req.completion_event.set()
        except Exception as e:
            # Propagate error to all requests in batch
            for req in requests:
                req.error = e
                req.completion_event.set()
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


# Globals set in main
scheduler: BatchScheduler = None
model_name_global: str = ""


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

        # Reject streaming in batched mode
        if stream:
            self._json_response(400, {
                "error": "streaming disabled in batched mode",
                "message": "Set 'stream': false to use batched generation"
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

        # Wait for completion (with timeout)
        if not req.completion_event.wait(timeout=600):  # 10 min timeout
            self._json_response(504, {"error": "request timeout"})
            return

        try:
            if req.error:
                import traceback
                traceback.print_exc()
                self._json_response(500, {"error": str(req.error)})
                return

            result = req.result
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
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                self._json_response(500, {"error": str(e)})
            except BrokenPipeError:
                pass


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
    global scheduler, model_name_global

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
    args = parser.parse_args()

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
    scheduler = BatchScheduler(workers, args.batch_size, args.batch_timeout)

    server = ThreadedHTTPServer((args.host, args.port), Handler)
    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"  Model: {args.model}")
    print(f"  GPUs:  {args.num_gpus}")
    print(f"  dtype: {args.dtype}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batch timeout: {args.batch_timeout}s")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        scheduler.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
