#!/usr/bin/env python3
"""Test a local vLLM server by sending a sample query and printing the response."""

import argparse
import json
import sys
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser(description="Test a local vLLM server")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="vLLM server base URL (default: http://localhost:8000)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto-detect from /v1/models)")
    parser.add_argument("--prompt", default="What is the fundamental theorem of algebra? State it precisely.",
                        help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Stream tokens to console (default: true)")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    # Check server health
    print(f"Connecting to {base} ...")
    try:
        resp = urllib.request.urlopen(f"{base}/v1/models", timeout=5)
        models_data = json.loads(resp.read())
        available = [m["id"] for m in models_data.get("data", [])]
        print(f"Available models: {available}")
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach server at {base}: {e}", file=sys.stderr)
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
        "temperature": 0.7,
        "stream": args.stream,
        **({"stream_options": {"include_usage": True}} if args.stream else {}),
    }).encode()

    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print(f"Prompt: {args.prompt}")
    print(f"{'─' * 60}")

    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"ERROR: HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"ERROR: Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.stream:
        finish_reason = None
        usage = {}
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            if "usage" in chunk:
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                sys.stdout.write(content)
                sys.stdout.flush()
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr
        print()
    else:
        data = json.loads(resp.read())
        choice = data["choices"][0]
        print(choice["message"]["content"])
        finish_reason = choice.get("finish_reason", "?")
        usage = data.get("usage", {})

    print(f"{'─' * 60}")
    print(f"Finish reason: {finish_reason or '?'}")
    print(f"Prompt tokens:     {usage.get('prompt_tokens', '?')}")
    print(f"Completion tokens: {usage.get('completion_tokens', '?')}")
    print(f"Total tokens:      {usage.get('total_tokens', '?')}")


if __name__ == "__main__":
    main()
