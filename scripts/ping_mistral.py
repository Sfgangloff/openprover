#!/usr/bin/env python3
"""Ping the Mistral API with a sample message and print the response."""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error


def build_example_calculator_tool():
    return {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic operation on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "Arithmetic operation to perform.",
                    },
                    "a": {"type": "number", "description": "First operand."},
                    "b": {"type": "number", "description": "Second operand."},
                },
                "required": ["operation", "a", "b"],
                "additionalProperties": False,
            },
        },
    }


def print_tool_calls(tool_calls):
    if not tool_calls:
        return
    print("\n\nTool calls:")
    for i, call in enumerate(tool_calls, start=1):
        print(f"  [{i}] {json.dumps(call, ensure_ascii=False)}")


def merge_stream_tool_calls(aggregated, delta_tool_calls):
    if not isinstance(delta_tool_calls, list):
        return
    for call in delta_tool_calls:
        if not isinstance(call, dict):
            continue
        idx = call.get("index", 0)
        entry = aggregated.setdefault(
            idx,
            {"id": "", "type": "", "function": {"name": "", "arguments": ""}},
        )
        if call.get("id"):
            entry["id"] = call["id"]
        if call.get("type"):
            entry["type"] = call["type"]
        fn = call.get("function")
        if isinstance(fn, dict):
            if fn.get("name"):
                entry["function"]["name"] = fn["name"]
            args_piece = fn.get("arguments")
            if isinstance(args_piece, str):
                entry["function"]["arguments"] += args_piece


def main():
    parser = argparse.ArgumentParser(description="Ping the Mistral API")
    parser.add_argument("--model", default="labs-leanstral-2603",
                        help="Model name (default: labs-leanstral-2603)")
    parser.add_argument("--prompt", default="Hello! What is the square root of 144?",
                        help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Stream tokens to console (default: true)")
    parser.add_argument("--print-reasoning", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Print reasoning/thinking tokens when present (default: true)")
    parser.add_argument("--example-tool", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Include a sample calculator tool in request (default: false)")
    parser.add_argument("--debug-stream", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Print raw SSE debug info to stderr (default: false)")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: MISTRAL_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    tools = [build_example_calculator_tool()] if args.example_tool else []

    payload = json.dumps({
        "model": args.model,
        "inputs": [{"role": "user", "content": args.prompt}],
        "tools": tools,
        "stream": args.stream,
        "completion_args": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": 1,
        },
        "instructions": "",
    }).encode()

    req = urllib.request.Request(
        "https://api.mistral.ai/v1/conversations",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    print(f"Model:  {args.model}")
    if args.example_tool:
        print("Tools:  calculator")
    print(f"Prompt: {args.prompt}")
    print("─" * 60)

    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"ERROR: HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"ERROR: Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    usage = {}

    if args.stream:
        saw_reasoning = False
        tool_calls_by_index = {}
        dim = "\033[2m" if sys.stdout.isatty() else ""
        reset = "\033[0m" if sys.stdout.isatty() else ""

        for line in resp:
            line = line.decode(errors="replace").strip()
            if args.debug_stream:
                print(f"[debug] raw: {line[:300]}", file=sys.stderr)
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].lstrip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError as e:
                if args.debug_stream:
                    print(f"[debug] JSON error: {e}", file=sys.stderr)
                continue
            if args.debug_stream:
                print(f"[debug] chunk keys: {list(chunk.keys())}", file=sys.stderr)

            if "usage" in chunk:
                usage = chunk["usage"]

            event_type = chunk.get("type", "")

            # message.output.delta: content/reasoning/tool_calls at top level
            if event_type == "message.output.delta":
                content = chunk.get("content", "")
                reasoning = (chunk.get("thinking") or chunk.get("reasoning")
                             or chunk.get("reasoning_content") or "")
                delta_tool_calls = chunk.get("tool_calls") or []
                merge_stream_tool_calls(tool_calls_by_index, delta_tool_calls)
            else:
                content = ""
                reasoning = ""

            if content:
                if saw_reasoning:
                    sys.stdout.write(reset)
                    saw_reasoning = False
                sys.stdout.write(content)
                sys.stdout.flush()
            elif args.print_reasoning and reasoning:
                if not saw_reasoning:
                    sys.stdout.write(dim)
                    saw_reasoning = True
                sys.stdout.write(reasoning)
                sys.stdout.flush()

        if saw_reasoning:
            sys.stdout.write(reset)
        print()

        if tool_calls_by_index:
            ordered = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
            print_tool_calls(ordered)
    else:
        data = json.loads(resp.read())
        if args.debug_stream:
            print(f"[debug] response keys: {list(data.keys())}", file=sys.stderr)
        usage = data.get("usage", {})
        outputs = data.get("outputs", [])
        tool_calls = []
        for entry in outputs:
            if entry.get("role") == "assistant":
                thinking = (entry.get("thinking") or entry.get("reasoning")
                            or entry.get("reasoning_content") or "")
                if args.print_reasoning and thinking:
                    dim = "\033[2m" if sys.stdout.isatty() else ""
                    reset = "\033[0m" if sys.stdout.isatty() else ""
                    print(f"{dim}{thinking}{reset}")
                print(entry.get("content", ""))
                tool_calls.extend(entry.get("tool_calls") or [])
        print_tool_calls(tool_calls)

    print("─" * 60)
    print(f"Prompt tokens:     {usage.get('prompt_tokens', '?')}")
    print(f"Completion tokens: {usage.get('completion_tokens', '?')}")
    print(f"Total tokens:      {usage.get('total_tokens', '?')}")


if __name__ == "__main__":
    main()
