#!/usr/bin/env python3
"""Benchmark tokens/second on a serve_hf.py server.

Measures single-request and concurrent throughput to exercise
multi-GPU load balancing.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


PROMPTS = {
    "short": "Prove that 1 + 1 = 2.",
    "medium": "Prove that the square root of 2 is irrational.",
    "long": (
        "Give a detailed, rigorous proof of the Bolzano-Weierstrass theorem: "
        "every bounded sequence in R^n has a convergent subsequence."
    ),
}


def get_model(base: str, model_arg: str | None) -> str:
    resp = urllib.request.urlopen(f"{base}/v1/models", timeout=5)
    data = json.loads(resp.read())
    available = [m["id"] for m in data.get("data", [])]
    if not available and not model_arg:
        print("ERROR: No models available and none specified", file=sys.stderr)
        sys.exit(1)
    model = model_arg or available[0]
    print(f"Model: {model}")
    return model


def single_request(base: str, model: str, prompt: str, max_tokens: int,
                   stream: bool) -> dict:
    """One request, return timing stats."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=600)

    if not stream:
        data = json.loads(resp.read())
        elapsed = time.perf_counter() - t0
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return {
            "elapsed_s": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "output_tok_per_s": completion_tokens / elapsed if elapsed > 0 else 0,
        }
    else:
        ttft = None
        chunks = 0
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line or not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1

        elapsed = time.perf_counter() - t0
        gen_time = elapsed - (ttft or 0)
        return {
            "elapsed_s": elapsed,
            "ttft_s": ttft or 0,
            "stream_chunks": chunks,
            "chunks_per_s": chunks / gen_time if gen_time > 0 else 0,
        }


def fmt(val: float, unit: str = "", decimals: int = 1) -> str:
    return f"{val:.{decimals}f}{unit}"


def run_bench(args):
    base = args.base_url.rstrip("/")

    try:
        model = get_model(base, args.model)
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach server at {base}: {e}", file=sys.stderr)
        sys.exit(1)

    prompt_text = PROMPTS.get(args.prompt, args.prompt)
    print(f"Prompt ({args.prompt}): {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
    print(f"Max tokens: {args.max_tokens}, Iterations: {args.iterations}, Concurrency: {args.concurrency}")
    print(f"Mode: {'streaming' if args.stream else 'non-streaming'}")
    print()

    # --- Sequential benchmark ---
    if args.concurrency <= 1:
        results = []
        for i in range(args.iterations):
            sys.stdout.write(f"  Run {i+1}/{args.iterations}...")
            sys.stdout.flush()
            try:
                r = single_request(base, model, prompt_text, args.max_tokens, args.stream)
                if args.stream:
                    print(f" {fmt(r['chunks_per_s'])} chunks/s, TTFT={fmt(r['ttft_s'], 's', 3)}, total={fmt(r['elapsed_s'], 's')}")
                else:
                    print(f" {fmt(r['output_tok_per_s'])} tok/s, {r['completion_tokens']} tokens in {fmt(r['elapsed_s'], 's')}")
                results.append(r)
            except Exception as e:
                print(f" ERROR: {e}")

        if not results:
            print("\nNo successful runs.")
            return
        _print_summary(results, args.stream)
        return

    # --- Concurrent benchmark ---
    print(f"Firing {args.concurrency} concurrent requests x {args.iterations} iteration(s)...\n")
    all_results = []

    for iteration in range(args.iterations):
        t_batch_start = time.perf_counter()
        batch_results = []

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(single_request, base, model, prompt_text, args.max_tokens, args.stream): j
                for j in range(args.concurrency)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    r = fut.result()
                    batch_results.append(r)
                except Exception as e:
                    print(f"  Request {idx}: ERROR: {e}")

        batch_elapsed = time.perf_counter() - t_batch_start

        if not args.stream:
            total_tokens = sum(r["completion_tokens"] for r in batch_results)
            agg_tok_s = total_tokens / batch_elapsed if batch_elapsed > 0 else 0
            print(f"  Batch {iteration+1}: {len(batch_results)}/{args.concurrency} ok, "
                  f"{total_tokens} tokens in {fmt(batch_elapsed, 's')}, "
                  f"aggregate {fmt(agg_tok_s)} tok/s")
        else:
            total_chunks = sum(r["stream_chunks"] for r in batch_results)
            print(f"  Batch {iteration+1}: {len(batch_results)}/{args.concurrency} ok, "
                  f"{total_chunks} chunks in {fmt(batch_elapsed, 's')}")

        all_results.extend(batch_results)

    if not all_results:
        print("\nNo successful runs.")
        return
    _print_summary(all_results, args.stream)


def _print_summary(results, stream):
    print(f"\n{'═' * 60}")
    print("SUMMARY")
    print(f"{'═' * 60}")

    if stream:
        ttfts = [r["ttft_s"] for r in results]
        rates = [r["chunks_per_s"] for r in results]
        print(f"Time to first token:  mean={fmt(statistics.mean(ttfts), 's', 3)}", end="")
        if len(ttfts) > 1:
            print(f"  stdev={fmt(statistics.stdev(ttfts), 's', 3)}", end="")
        print()
        print(f"Streaming chunks/s:   mean={fmt(statistics.mean(rates))}", end="")
        if len(rates) > 1:
            print(f"  stdev={fmt(statistics.stdev(rates))}", end="")
        print()
    else:
        out_rates = [r["output_tok_per_s"] for r in results]
        comp_tokens = [r["completion_tokens"] for r in results]
        print(f"Output tok/s:         mean={fmt(statistics.mean(out_rates))}", end="")
        if len(out_rates) > 1:
            print(f"  stdev={fmt(statistics.stdev(out_rates))}", end="")
        print()
        print(f"Completion tokens:    mean={fmt(statistics.mean(comp_tokens), decimals=0)}")

    elapsed_all = [r["elapsed_s"] for r in results]
    print(f"Wall time per req:    mean={fmt(statistics.mean(elapsed_all), 's')}", end="")
    if len(elapsed_all) > 1:
        print(f"  stdev={fmt(statistics.stdev(elapsed_all), 's')}", end="")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark tokens/second on a serve_hf.py server")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Server base URL (default: http://localhost:8000)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto-detect)")
    parser.add_argument("--prompt", default="medium", choices=list(PROMPTS.keys()),
                        help="Prompt preset (default: medium)")
    parser.add_argument("--custom-prompt", default=None,
                        help="Custom prompt text (overrides --prompt)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max output tokens (default: 512)")
    parser.add_argument("--iterations", "-n", type=int, default=3,
                        help="Number of iterations (default: 3)")
    parser.add_argument("--concurrency", "-c", type=int, default=1,
                        help="Concurrent requests per iteration (default: 1)")
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming mode (measures TTFT and chunk rate)")
    args = parser.parse_args()

    if args.custom_prompt:
        args.prompt = args.custom_prompt

    run_bench(args)


if __name__ == "__main__":
    main()
