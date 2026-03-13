"""Shared utilities for LLM client modules."""

import json
from pathlib import Path


class Interrupted(Exception):
    """Raised when an LLM call is cancelled via interrupt()."""
    pass


class StreamingUnavailable(RuntimeError):
    """Raised when HF server cannot stream in current configuration."""
    pass


def archive(model, archive_dir, call_num, label, prompt, system_prompt,
            json_schema, response, error, elapsed_ms, archive_path=None,
            *, thinking="", result_text=""):
    """Archive an LLM call to a JSON file."""
    record = {
        "call_num": call_num,
        "label": label,
        "model": model,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "json_schema": json_schema,
        "result_text": result_text,
        "thinking": thinking,
        "response": response,
        "error": error,
        "elapsed_ms": elapsed_ms,
    }
    if archive_path:
        path = archive_path
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        archive_dir.mkdir(parents=True, exist_ok=True)
        path = archive_dir / f"call_{call_num:03d}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
