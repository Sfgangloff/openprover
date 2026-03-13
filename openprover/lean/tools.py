"""Lean tool definitions and execution for vLLM worker tool-calling."""

import asyncio
import logging
import time

from .core import LeanWorkDir, run_lean_check

logger = logging.getLogger("openprover.lean")

WORKER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lean_verify",
            "description": "Verify Lean 4 code. Returns compiler output (errors/warnings or OK).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Lean 4 source code to verify.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lean_search",
            "description": "Search Mathlib and Lean 4 declarations by natural language query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def execute_worker_tool(
    name: str,
    args: dict,
    worker_id: str,
    lean_work_dir: LeanWorkDir | None,
    lean_project_dir,
    lean_explore_service,
) -> tuple[str, str]:
    """Execute a worker tool call. Returns (result_text, status)."""
    if name == "lean_verify":
        return _tool_lean_verify(args, worker_id, lean_work_dir, lean_project_dir)
    if name == "lean_search":
        return _tool_lean_search(args, worker_id, lean_explore_service)
    return (f"Unknown tool: {name}", "error")


def _tool_lean_verify(
    args: dict,
    worker_id: str,
    lean_work_dir: LeanWorkDir | None,
    lean_project_dir,
) -> tuple[str, str]:
    """Verify Lean code via lean_check."""
    code = args.get("code", "")
    if not code:
        return ("No code provided", "error")
    if not lean_work_dir:
        return ("Lean project not configured", "error")

    slug = f"worker_verify_{worker_id}"
    path = lean_work_dir.make_file(slug, code)
    success, feedback, _cmd_info = run_lean_check(path, lean_project_dir)
    status = "ok" if success else "error"
    result = "OK — no errors" if success else feedback
    logger.info("[%s] lean_verify: %s", worker_id, status)
    return (result, status)


def _tool_lean_search(
    args: dict,
    worker_id: str,
    lean_explore_service,
) -> tuple[str, str]:
    """Search Mathlib declarations."""
    import torch
    query = args.get("query", "")
    if not query:
        return ("No query provided", "error")
    if not lean_explore_service:
        return ("lean_search not available (lean_explore not installed)", "error")

    rerank = 25 if torch.cuda.is_available() else 0
    try:
        t0 = time.time()
        results = asyncio.run(
            lean_explore_service.search(query, limit=10, rerank_top=rerank)
        )
        elapsed = time.time() - t0
        logger.info("[%s] lean_search query=%r returned %d results in %.1fs",
                    worker_id, query, len(results) if results else 0, elapsed)
        if not results:
            return ("No results found", "ok")
        parts = []
        for r in results:
            name = getattr(r, 'name', str(r))
            doc = getattr(r, 'doc_string', '') or ''
            sig = getattr(r, 'signature', '') or ''
            entry = f"**{name}**"
            if sig:
                entry += f"\n```lean\n{sig}\n```"
            if doc:
                entry += f"\n{doc}"
            parts.append(entry)
        return ("\n\n".join(parts), "ok")
    except Exception as e:
        logger.warning("[%s] lean_search error: %s", worker_id, e)
        return (f"Search error: {e}", "error")
