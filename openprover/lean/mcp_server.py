"""MCP server exposing lean_verify and lean_search tools for Claude CLI.

Runs as a subprocess spawned by Claude CLI via --mcp-config.
Communicates over stdio using JSON-RPC (MCP protocol).

Environment variables:
    LEAN_PROJECT_DIR: Path to Lean project with lakefile (required for lean_verify)
    LEAN_WORK_DIR: Path to working directory for temporary Lean files
"""

import asyncio
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .core import LeanWorkDir, run_lean_check

mcp = FastMCP("lean_tools")

# Initialized lazily from environment variables
_project_dir: Path | None = None
_work_dir: LeanWorkDir | None = None
_search_service = None


def _get_project_dir() -> Path:
    global _project_dir
    if _project_dir is None:
        val = os.environ.get("LEAN_PROJECT_DIR")
        if not val:
            raise RuntimeError("LEAN_PROJECT_DIR not set")
        _project_dir = Path(val)
    return _project_dir


def _get_work_dir() -> LeanWorkDir:
    global _work_dir
    if _work_dir is None:
        val = os.environ.get("LEAN_WORK_DIR")
        if val:
            # Use the specified directory directly (already created by Prover)
            wd = LeanWorkDir.__new__(LeanWorkDir)
            wd.project_dir = _get_project_dir()
            wd.dir = Path(val)
            wd.dir.mkdir(parents=True, exist_ok=True)
            wd.random_id = "mcp"
            _work_dir = wd
        else:
            _work_dir = LeanWorkDir(_get_project_dir())
    return _work_dir


_has_gpu: bool | None = None

def _gpu_available() -> bool:
    global _has_gpu
    if _has_gpu is None:
        try:
            import torch
            _has_gpu = torch.cuda.is_available()
        except Exception:
            _has_gpu = False
    return _has_gpu


def _get_search_service():
    global _search_service
    if _search_service is None:
        from lean_explore.search import SearchEngine, Service
        engine = SearchEngine(use_local_data=False)
        _search_service = Service(engine=engine)
    return _search_service


@mcp.tool()
def lean_verify(code: str) -> str:
    """Verify Lean 4 code. Returns compiler output (errors/warnings or OK)."""
    if not code.strip():
        return "Error: no code provided"
    work_dir = _get_work_dir()
    project_dir = _get_project_dir()
    path = work_dir.make_file("mcp_verify", code)
    success, feedback, _cmd_info = run_lean_check(path, project_dir)
    return "OK — no errors" if success else feedback


@mcp.tool()
def lean_search(query: str) -> str:
    """Search Mathlib and Lean 4 declarations by natural language query."""
    if not query.strip():
        return "Error: no query provided"
    try:
        service = _get_search_service()
    except ImportError:
        return "Error: lean_explore not installed"
    except Exception as e:
        return f"Error initializing search: {e}"

    try:
        rerank = 25 if _gpu_available() else 0
        response = asyncio.run(service.search(query, limit=10, rerank_top=rerank))
        results = response.results
        if not results:
            return "No results found"
        parts = []
        for r in results:
            name = getattr(r, 'name', '')
            module = getattr(r, 'module', '') or ''
            source = getattr(r, 'source_text', '') or ''
            doc = getattr(r, 'docstring', '') or ''
            info = getattr(r, 'informalization', '') or ''
            header = f"**{name}**"
            if module:
                header += f"  ({module})"
            entry = header
            if source:
                entry += f"\n```lean\n{source.strip()}\n```"
            if doc:
                entry += f"\n{doc.strip()}"
            if info:
                entry += f"\nInformalization: {info.strip()}"
            parts.append(entry)
        return "\n\n".join(parts)
    except Exception as e:
        return f"Search error: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
