#!/usr/bin/env python3
"""Serve lm-provers/QED-Nano on multiple GPUs with simple load balancing.

Loads one model replica per GPU and exposes an OpenAI-compatible API
at /v1/chat/completions (streaming and non-streaming) and /v1/models.
"""

import argparse
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class Worker:
    """One model replica on one GPU."""

    def __init__(self, gpu_id: int, model_name: str, dtype: torch.dtype, tokenizer):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.lock = threading.Lock()
        self.tokenizer = tokenizer

        print(f"  Loading on cuda:{gpu_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        print(f"  cuda:{gpu_id} ready")

    def generate(self, messages, max_tokens, temperature, top_p, stream=False):
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

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True,
            )
            gen_kwargs["streamer"] = streamer

            def _run():
                with torch.inference_mode():
                    self.model.generate(**gen_kwargs)

            thread = threading.Thread(target=_run)
            thread.start()
            return streamer, input_len, thread
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

        worker = lb.get_worker()

        with worker.lock:
            try:
                if stream:
                    self._handle_streaming(worker, messages, max_tokens, temperature, top_p)
                else:
                    self._handle_non_streaming(worker, messages, max_tokens, temperature, top_p)
            except Exception as e:
                self._json_response(500, {"error": str(e)})

    def _handle_non_streaming(self, worker, messages, max_tokens, temperature, top_p):
        t0 = time.time()
        text, completion_tokens, prompt_tokens = worker.generate(
            messages, max_tokens, temperature, top_p, stream=False,
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

    def _handle_streaming(self, worker, messages, max_tokens, temperature, top_p):
        streamer, input_len, gen_thread = worker.generate(
            messages, max_tokens, temperature, top_p, stream=True,
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

        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": model_name_global,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

        gen_thread.join()

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
    workers = [Worker(i, args.model, dtype, tokenizer) for i in range(args.num_gpus)]
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
