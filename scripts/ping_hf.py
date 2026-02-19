#!/usr/bin/env python3
"""Test a local serve_hf.py server by sending a sample query and printing the response."""

import argparse
import json
import sys
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser(description="Test a local serve_hf.py server")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Server base URL (default: http://localhost:8000)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto-detect from /v1/models)")
    parser.add_argument("--prompt", default="Prove that the square root of 2 is irrational.",
                        help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    # Check server health
    print(f"Connecting to {base} ...")
    try:
        resp = urllib.request.urlopen(f"{base}/health", timeout=5)
        health = json.loads(resp.read())
        print(f"Health: {health.get('status', '?')}")
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach server at {base}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        resp = urllib.request.urlopen(f"{base}/v1/models", timeout=5)
        models_data = json.loads(resp.read())
        available = [m["id"] for m in models_data.get("data", [])]
        print(f"Available models: {available}")
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot list models: {e}", file=sys.stderr)
        sys.exit(1)

    model = args.model or available[0] if available else None
    if not model:
        print("ERROR: No model found and none specified with --model", file=sys.stderr)
        sys.exit(1)
    print(f"Using model: {model}\n")

    # Send chat completion request
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.6,
    }).encode()

    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"ERROR: Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results
    choice = data["choices"][0]
    message = choice["message"]["content"]
    usage = data.get("usage", {})

    print(f"Prompt: {args.prompt}")
    print(f"{'─' * 60}")
    print(message)
    print(f"{'─' * 60}")
    print(f"Finish reason: {choice.get('finish_reason', '?')}")
    print(f"Prompt tokens:     {usage.get('prompt_tokens', '?')}")
    print(f"Completion tokens: {usage.get('completion_tokens', '?')}")
    print(f"Total tokens:      {usage.get('total_tokens', '?')}")


if __name__ == "__main__":
    main()
