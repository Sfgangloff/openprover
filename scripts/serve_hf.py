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


class Worker:
    """One model replica on one GPU."""

    _load_lock = threading.Lock()

    def __init__(self, gpu_id: int, model_name: str, dtype: torch.dtype, tokenizer):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.lock = threading.Lock()
        self.tokenizer = tokenizer

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


class LoadBalancer:
    """Round-robin across workers."""

    def __init__(self, workers: list[Worker]):
        self.workers = workers
        self._counter = 0
        self._lock = threading.Lock()

    def get_worker(self) -> Worker:
        with self._lock:
            w = self.workers[self._counter % len(self.workers)]
            self._counter += 1
        return w


# Globals set in main
lb: LoadBalancer = None
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

        worker = lb.get_worker()

        with worker.lock:
            try:
                if stream:
                    self._handle_streaming(worker, messages, max_tokens,
                                           temperature, top_p, thinking_budget)
                else:
                    self._handle_non_streaming(worker, messages, max_tokens,
                                               temperature, top_p, thinking_budget)
            except BrokenPipeError:
                pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    self._json_response(500, {"error": str(e)})
                except BrokenPipeError:
                    pass

    def _handle_non_streaming(self, worker, messages, max_tokens, temperature,
                              top_p, thinking_budget=None):
        t0 = time.time()
        text, completion_tokens, prompt_tokens = worker.generate(
            messages, max_tokens, temperature, top_p, stream=False,
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

    def _handle_streaming(self, worker, messages, max_tokens, temperature,
                          top_p, thinking_budget=None):
        streamer, input_len, gen_thread, result = worker.generate(
            messages, max_tokens, temperature, top_p, stream=True,
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
    global lb, model_name_global

    parser = argparse.ArgumentParser(
        description="Serve a HuggingFace model on multiple GPUs with load balancing",
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
    lb = LoadBalancer(workers)

    server = ThreadedHTTPServer((args.host, args.port), Handler)
    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"  Model: {args.model}")
    print(f"  GPUs:  {args.num_gpus}")
    print(f"  dtype: {args.dtype}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
