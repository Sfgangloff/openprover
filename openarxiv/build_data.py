#!/usr/bin/env python3
"""Convert TOML paper/problem data to a single JSON file for the web app."""

import json
import re
import tomllib
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_OUTPUT = Path(__file__).parent / "web" / "public" / "data.json"


def _fix_toml_strings(text: str) -> str:
    """Convert triple-double-quoted strings to triple-single-quoted (literal).

    The write_metadata_toml function uses triple-double-quotes for abstracts
    and titles, but these interpret backslashes as escapes. Convert to literal
    strings so backslashes are preserved verbatim.
    """
    # Match """\\\n...""" pattern (the \\\ is a line continuation)
    # Replace with '''\n...'''
    def replace_ml(m):
        content = m.group(1)
        # Remove leading backslash-newline (TOML line continuation)
        if content.startswith("\\\n"):
            content = content[2:]
        return "'''\n" + content + "'''"

    return re.sub(r'"""\\\n(.*?)"""', replace_ml, text, flags=re.DOTALL)


def parse_cost(cost_val) -> float | None:
    """Parse a cost value (string like '$0.20' or float) to a float.

    Returns None if cost is unknown/unreported.
    """
    if cost_val is None:
        return None
    if isinstance(cost_val, str):
        if cost_val in ("n/a", "unknown", ""):
            return None
        m = re.match(r"\$?([\d.]+)", cost_val)
        if not m:
            return None
        val = float(m.group(1))
        return val if val > 0 else None
    if isinstance(cost_val, (int, float)):
        return float(cost_val) if cost_val > 0 else None
    return None


def load_paper(paper_dir: Path) -> dict | None:
    """Load a single paper directory into a dict."""
    paper_toml = paper_dir / "paper.toml"
    if not paper_toml.exists():
        return None

    raw = paper_toml.read_text()
    try:
        meta = tomllib.loads(raw)
    except tomllib.TOMLDecodeError:
        # Some paper.toml files have backslashes in double-quoted strings
        # (e.g. LaTeX in abstracts). Fix the quoting style and retry.
        try:
            meta = tomllib.loads(_fix_toml_strings(raw))
        except tomllib.TOMLDecodeError as e:
            print(f"  WARNING: cannot parse {paper_toml}: {e}")
            return None

    # Extraction metadata
    extraction = None
    ext_toml = paper_dir / "extraction.toml"
    if ext_toml.exists():
        try:
            ext = tomllib.loads(ext_toml.read_text())
            if "error" not in ext:
                cost_raw = ext.get("cost", "")
                cost_float = parse_cost(cost_raw)
                extraction = {
                    "model": ext.get("model", ""),
                    "cost": (f"${cost_float:.2f}" if cost_float >= 0.01 else f"${cost_float:.4f}") if cost_float is not None else None,
                    "num_problems": ext.get("num_problems", 0),
                    "prompt_tokens": ext.get("prompt_tokens", 0),
                    "completion_tokens": ext.get("completion_tokens", 0),
                    "total_tokens": ext.get("total_tokens", 0),
                }
            else:
                extraction = {"error": ext["error"]}
        except Exception:
            pass

    # Problems
    problems = []
    problems_dir = paper_dir / "problems"
    if problems_dir.exists():
        for prob_path in sorted(problems_dir.glob("*.toml")):
            try:
                prob = tomllib.loads(prob_path.read_text())
                problems.append({
                    "name": prob.get("name", ""),
                    "id": prob.get("id", prob_path.stem),
                    "summary": prob.get("summary", ""),
                    "location": prob.get("location", ""),
                    "statement": prob.get("statement", ""),
                    "context": prob.get("context", ""),
                    "references": prob.get("references", []),
                })
            except Exception as e:
                print(f"  WARNING: failed to parse {prob_path}: {e}")

    return {
        "id": meta.get("id", paper_dir.name),
        "title": meta.get("title", ""),
        "authors": meta.get("authors", []),
        "abstract": meta.get("abstract", ""),
        "published": meta.get("published", ""),
        "updated": meta.get("updated", ""),
        "categories": meta.get("categories", []),
        "pdf_url": meta.get("pdf_url", ""),
        "extraction": extraction,
        "problems": problems,
    }


def build_data(data_dir: Path, output: Path):
    """Build the complete data.json from all paper directories."""
    papers = []
    for paper_dir in sorted(data_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        paper = load_paper(paper_dir)
        if paper:
            papers.append(paper)

    # Compute stats
    total_cost = 0.0
    papers_with_problems = 0
    total_problems = 0
    extracted_papers = 0
    for p in papers:
        ext = p.get("extraction")
        if ext and "error" not in ext:
            extracted_papers += 1
            cost_val = parse_cost(ext.get("cost", ""))
            if cost_val is not None:
                total_cost += cost_val
        if p["problems"]:
            papers_with_problems += 1
            total_problems += len(p["problems"])

    data = {
        "papers": papers,
        "stats": {
            "total_papers": len(papers),
            "extracted_papers": extracted_papers,
            "papers_with_problems": papers_with_problems,
            "total_problems": total_problems,
            "total_cost": f"${total_cost:.2f}",
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Wrote {output} ({len(papers)} papers, {total_problems} problems)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build data.json for web app")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    build_data(Path(args.data_dir), Path(args.output))
