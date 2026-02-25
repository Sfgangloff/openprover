#!/usr/bin/env python3
"""Download ArXiv combinatorics papers and extract open problems using an LLM."""

import argparse
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tomllib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

class _C:
    """ANSI color codes, disabled when not a TTY."""
    BOLD = RESET = DIM = RED = GREEN = YELLOW = CYAN = ""

if os.isatty(1):
    _C.BOLD = "\033[1m"
    _C.RESET = "\033[0m"
    _C.DIM = "\033[2m"
    _C.RED = "\033[31m"
    _C.GREEN = "\033[32m"
    _C.YELLOW = "\033[33m"
    _C.CYAN = "\033[36m"


ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_EPRINT = "https://arxiv.org/e-print"
ATOM_NS = "{http://www.w3.org/2005/Atom}"

DEFAULT_DATA_DIR = Path(__file__).parent / "data"

# Only include these file types in the LLM prompt.
SOURCE_EXTENSIONS = {".tex", ".bib", ".sty", ".bst"}

EXTRACT_SYSTEM_PROMPT = """\
You are a research mathematician.
Your task is to carefully read the full source of a combinatorics paper and extract all open problems that are introduced (posed/conjectured) but NOT solved.

We are looking specifically for open problems whose solution would be a mathematical proof - conjectures, open questions, unresolved problems.
Do not extract computational challenges, algorithmic open questions, or problems that ask for a construction/algorithm rather than a proof.

Rules:
- Do not extract solved theorems.
- For each problem, produce a TOML block with the fields described below.
- The "summary" is a very compact 1-2 sentence TLDR.
- The "location" describes where in the paper the open problem appears, e.g. "Section 3, Conjecture 4.2" or "Introduction, last paragraph".
- The "statement" is the conjecture/question itself. It should read as a standalone, compact, self-contained mathematical statement. A reader should be able to understand exactly what is being conjectured from the statement alone (with terms/notation defined in the context below). Written in **markdown with inline LaTeX**. Use `$...$` for inline math and `$$...$$` for display math. Use markdown formatting (**bold**, *italic*, `>` blockquotes, `###` headings, `-` lists) for text structure. Do NOT use LaTeX text-mode commands like \\textbf, \\emph, \\section, etc. Instead, write e.g. **Conjecture.** in markdown bold.
- CRITICAL: Do NOT use \\begin{...} or \\end{...} environments anywhere in statement or context. These do not render in MathJax. Specifically:
  - Instead of \\begin{conjecture}...\\end{conjecture}, write **Conjecture.** in markdown bold.
  - Instead of \\begin{theorem}...\\end{theorem}, write **Theorem.** in markdown bold. Same for lemma, proposition, question, problem, etc.
  - Instead of \\begin{enumerate}/\\begin{itemize}, use markdown lists (`1.`, `-`).
  - Instead of \\begin{cases}...\\end{cases}, use the LaTeX cases environment ONLY inside `$$...$$` math mode (i.e., `$$f(x) = \\begin{cases} ... \\end{cases}$$` is OK because MathJax renders it as math).
  - Instead of \\begin{align}...\\end{align} or \\begin{equation}, use `$$...$$` display math. For multi-line equations use `$$` with `\\\\` line breaks.
  - Instead of \\begin{bmatrix}/\\begin{pmatrix}, use the matrix command ONLY inside `$...$` or `$$...$$` math mode (these are OK since MathJax renders them as math).
  The rule is: \\begin/\\end may ONLY appear inside `$...$` or `$$...$$` delimiters for math-mode environments (cases, bmatrix, pmatrix, array, aligned). They must NEVER appear as top-level text-mode structural commands.
- The "context" provides ALL the background needed to fully understand the statement. Together, statement + context must be self-contained: a mathematician reading them without access to the paper should have everything needed to understand and attempt the problem. Include all relevant definitions, notation, prerequisite results, and motivating context from the paper. Use `###` subheadings to organize long context. Same format rules: markdown with inline LaTeX. Only include information explicitly mentioned in this paper.
- For references, write them as plain text with markdown formatting. E.g. "[AuthorYear]" as a plain citation marker, not \\cite{...}.
- Every reference mentioned in statement or context must appear in the references list. Conversely, every entry in references must actually be cited in the statement or context.
- Do NOT use the tilde character (~) for non-breaking spaces. Just use regular spaces everywhere.
- For MathJax compatibility: stick to standard math commands. Use \\operatorname{name} for custom operators. Avoid \\tikz, \\pstricks, or drawing commands. Use \\text{...} for text inside math mode.
- It is perfectly fine for a paper to have zero open problems. In that case, output an empty problems list.

Output format - a single TOML document.
CRITICAL TOML RULE: You MUST use TOML literal strings (triple single quotes '''...''') for ALL string values. NEVER use double-quoted strings ("..." or \"""...\"""). TOML double-quoted strings interpret backslashes as escape sequences, which corrupts LaTeX. Use '''...''' for everything - name, summary, location, statement, context, tag, text.

```toml
[[problems]]
name = '''Human-readable name of the problem'''
summary = '''Very brief 1-2 sentence description with $\\chi(G) \\le 4$.'''
location = '''Section 3, Conjecture 4.2'''
statement = '''
**Conjecture.** For every graph $G$ with $\\chi(G) \\le 4$, we have $f(G) \\le 100$.
'''
context = '''
Let $G = (V, E)$ be a simple undirected graph. The *chromatic number* $\\chi(G)$ is the minimum number of colors needed to color the vertices such that no two adjacent vertices share the same color.

### Key definitions

A graph is called *perfect* if for every induced subgraph $H$, we have $\\chi(H) = \\omega(H)$ where $\\omega(H)$ is the clique number.

The function $f(G)$ was introduced by [SmithJones2020] and is defined as:

$$f(G) = \\max_{S \\subseteq V} \\frac{|S|}{\\alpha(G[S])}$$
'''

[[problems.references]]
tag = '''SmithJones2020'''
text = '''J. Smith, A. Jones, *On chromatic bounds for perfect graphs*, J. Combin. Theory Ser. B, 2020.'''

[[problems.images]]
name = '''figurefile.png'''
```

If there are no open problems, output:
```toml
# No open problems found in this paper.
```

Now analyze the paper and output ONLY the TOML (inside a ```toml code fence)."""


# ---------------------------------------------------------------------------
# ArXiv download
# ---------------------------------------------------------------------------

def _urlopen(url: str, retries: int = 5, delay: float = 10.0):
    """urlopen with retries on 429/5xx."""
    import time
    req = urllib.request.Request(url, headers={"User-Agent": "openarxiv/0.1"})
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < retries - 1:
                print(f"    HTTP {e.code}, retrying in {delay:.0f}s... (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise


def fetch_arxiv_listing(category: str, max_results: int, start: int = 0) -> list[dict]:
    """Query ArXiv API and return list of paper metadata dicts."""
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": max_results,
    })
    url = f"{ARXIV_API}?{params}"
    with _urlopen(url) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id_url = entry.find(f"{ATOM_NS}id").text.strip()
        # e.g. "http://arxiv.org/abs/2506.12345v1" -> "2506.12345"
        arxiv_id = arxiv_id_url.split("/abs/")[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        title = entry.find(f"{ATOM_NS}title").text.strip()
        title = re.sub(r"\s+", " ", title)

        authors = []
        for author_el in entry.findall(f"{ATOM_NS}author"):
            name = author_el.find(f"{ATOM_NS}name").text.strip()
            authors.append(name)

        abstract = entry.find(f"{ATOM_NS}summary").text.strip()
        published = entry.find(f"{ATOM_NS}published").text.strip()
        updated = entry.find(f"{ATOM_NS}updated").text.strip()

        categories = []
        for cat_el in entry.findall(f"{ATOM_NS}category"):
            term = cat_el.get("term")
            if term:
                categories.append(term)

        pdf_url = ""
        for link_el in entry.findall(f"{ATOM_NS}link"):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href", "")

        papers.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "published": published,
            "updated": updated,
            "categories": categories,
            "pdf_url": pdf_url,
        })

    return papers


def write_metadata_toml(paper: dict, path: Path):
    """Write paper metadata as a TOML file."""
    lines = []
    lines.append(f'id = "{paper["id"]}"')
    lines.append(f'title = """\\\n{paper["title"]}"""')
    lines.append(f'authors = [{", ".join(repr(a) for a in paper["authors"])}]')
    lines.append(f'abstract = """\\\n{paper["abstract"]}"""')
    lines.append(f'published = "{paper["published"]}"')
    lines.append(f'updated = "{paper["updated"]}"')
    lines.append(f'categories = [{", ".join(repr(c) for c in paper["categories"])}]')
    lines.append(f'pdf_url = "{paper["pdf_url"]}"')
    path.write_text("\n".join(lines) + "\n")


def download_source(arxiv_id: str, source_dir: Path):
    """Download and extract the e-print source for a paper."""
    url = f"{ARXIV_EPRINT}/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "openarxiv/0.1"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()

    source_dir.mkdir(parents=True, exist_ok=True)

    # Try tar.gz first
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(path=source_dir, filter="data")
        return
    except (tarfile.TarError, gzip.BadGzipFile, EOFError):
        pass

    # Try plain tar
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
            tar.extractall(path=source_dir, filter="data")
        return
    except tarfile.TarError:
        pass

    # Try gzip (single file)
    try:
        decompressed = gzip.decompress(data)
        source_dir.joinpath("main.tex").write_bytes(decompressed)
        return
    except gzip.BadGzipFile:
        pass

    # Raw file (likely a single .tex)
    source_dir.joinpath("main.tex").write_bytes(data)


def cmd_download(args):
    """Download papers from ArXiv combinatorics section."""
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching up to {args.max} papers from math.CO...")
    papers = fetch_arxiv_listing("math.CO", args.max)
    print(f"Got {len(papers)} entries from ArXiv API.\n")

    for paper in papers:
        paper_dir = data_dir / paper["id"].replace("/", "_")
        if paper_dir.exists():
            print(f"  {_C.DIM}skip {paper['id']} (exists){_C.RESET}")
            continue

        print(f"  {_C.BOLD}{paper['id']}{_C.RESET}  {paper['title'][:72]}")
        paper_dir.mkdir(parents=True)
        write_metadata_toml(paper, paper_dir / "paper.toml")

        source_dir = paper_dir / "source"
        try:
            download_source(paper["id"], source_dir)
            n_files = sum(1 for _ in source_dir.rglob("*") if _.is_file())
            print(f"    {_C.GREEN}downloaded{_C.RESET}  "
                  f"{_C.DIM}{n_files} files{_C.RESET}")
        except Exception as e:
            print(f"    {_C.RED}download failed: {e}{_C.RESET}")


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def build_prompt(source_dir: Path) -> str:
    """Build the LLM prompt by concatenating whitelisted source files."""
    parts = ["Below are the source files of a combinatorics research paper.\n"]

    # Directory tree
    tree_result = subprocess.run(
        ["tree", "-a", "--noreport"],
        cwd=source_dir, capture_output=True, text=True, timeout=10,
    )
    if tree_result.returncode == 0:
        parts.append("Directory structure:")
        parts.append("```")
        parts.append(tree_result.stdout.rstrip())
        parts.append("```\n")

    files = sorted(source_dir.rglob("*"))
    included = 0
    for f in files:
        if not f.is_file():
            continue
        if f.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        rel = f.relative_to(source_dir)
        try:
            content = f.read_text(errors="replace")
        except OSError:
            continue
        parts.append(f"--- FILE: {rel} ---")
        parts.append(content)
        parts.append("--- END FILE ---\n")
        included += 1

    if included == 0:
        # Fallback: include any text file if no whitelisted files found
        for f in files:
            if not f.is_file():
                continue
            try:
                data = f.read_bytes()[:4096]
                data.decode("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = f.relative_to(source_dir)
            try:
                content = f.read_text(errors="replace")
            except OSError:
                continue
            parts.append(f"--- FILE: {rel} ---")
            parts.append(content)
            parts.append("--- END FILE ---\n")

    parts.append(
        "Now carefully analyze the paper above and extract all open problems "
        "(conjectures, open questions) that are posed but NOT solved in the "
        "paper. Output your answer as a single TOML document inside a "
        "```toml code fence, following the format described in your "
        "instructions."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call via OpenRouter
# ---------------------------------------------------------------------------

_model_context_cache: dict[str, int] = {}


def get_model_context_length(model: str) -> int | None:
    """Fetch the context length for a model from OpenRouter. Cached."""
    if model in _model_context_cache:
        return _model_context_cache[model]

    import httpx

    try:
        resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
        if resp.status_code == 200:
            for m in resp.json().get("data", []):
                mid = m.get("id", "")
                ctx = m.get("context_length")
                if ctx is not None:
                    _model_context_cache[mid] = int(ctx)
            return _model_context_cache.get(model)
    except Exception:
        pass
    return None


def call_llm(messages: list[dict], model: str) -> dict:
    """Call OpenRouter and return dict with content, reasoning, and usage.

    Returns {"content": str, "reasoning": str|None,
             "usage": {"model": str, "prompt_tokens": int,
             "completion_tokens": int, "total_tokens": int,
             "cost": float|None}}.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    import httpx

    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": messages},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()

    message = data["choices"][0]["message"]
    content = message["content"]
    reasoning = message.get("reasoning")
    usage_raw = data.get("usage", {})
    usage = {
        "model": data.get("model", model),
        "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
        "completion_tokens": int(usage_raw.get("completion_tokens", 0)),
        "total_tokens": int(usage_raw.get("total_tokens", 0)),
        "cost": usage_raw.get("cost"),
    }
    return {"content": content, "reasoning": reasoning, "usage": usage}


class TOMLParseError(Exception):
    """Raised when the LLM response contains TOML that fails to parse."""
    pass


# ---------------------------------------------------------------------------
# MathJax validation
# ---------------------------------------------------------------------------

VALIDATE_SCRIPT = Path(__file__).parent / "validate_mathjax.js"


def validate_mathjax(problems: list[dict],
                     save_dir: Path | None = None) -> list[str]:
    """Validate MathJax rendering of problem statement/context fields.

    Calls the Node.js validation script which mirrors the frontend's MathJax
    config.  Returns a list of human-readable error strings (empty = all OK).
    Degrades gracefully if node or the script is unavailable.

    If *save_dir* is given, stores input.json and output.json there.
    """
    if not problems:
        return []

    fields = []
    for i, prob in enumerate(problems):
        name = prob.get("name", f"problem {i}")
        for key in ("statement", "context"):
            text = prob.get(key, "")
            if text.strip():
                fields.append({"field": f"{name} → {key}", "text": text})
    if not fields:
        return []

    input_json = json.dumps(fields, indent=2)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "input.json").write_text(input_json)

    try:
        proc = subprocess.run(
            ["node", str(VALIDATE_SCRIPT)],
            input=input_json,
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        print(f"    {_C.YELLOW}node not found, skipping MathJax validation{_C.RESET}")
        return []
    except subprocess.TimeoutExpired:
        print(f"    {_C.YELLOW}MathJax validation timed out{_C.RESET}")
        return []

    if proc.returncode != 0:
        msg = proc.stderr.strip()[:200]
        print(f"    {_C.YELLOW}MathJax validation failed: {msg}{_C.RESET}")
        if save_dir is not None:
            (save_dir / "error.txt").write_text(msg)
        return []

    if save_dir is not None:
        (save_dir / "output.json").write_text(proc.stdout)

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"    {_C.YELLOW}MathJax validation returned invalid JSON{_C.RESET}")
        return []

    errors = []
    for err in data.get("errors", []):
        field = err.get("field", "?")
        msg = err.get("message", "?")
        expr = err.get("expression", "")
        # Truncate long expressions for readability
        if len(expr) > 80:
            expr = expr[:77] + "..."
        errors.append(f"[{field}] {msg}: {expr}")
    return errors


# Regex to strip math delimiters so bracket searches don't match inside math.
_MATH_RE = re.compile(r"\$\$[\s\S]*?\$\$|\$[^$\n]+?\$")
# Regex to find [bracketed] references in plain text.
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")


def check_references(problems: list[dict]) -> list[str]:
    """Check that references in statement/context match the references list.

    Mirrors the frontend's linkifyRefs logic: every [Tag] bracket in text must
    match (via substring) a tag in the references list, and every reference tag
    must be cited somewhere.  Returns a list of human-readable error strings.
    """
    errors = []
    for i, prob in enumerate(problems):
        name = prob.get("name", f"problem {i}")
        tags = {r.get("tag", "") for r in prob.get("references", []) if r.get("tag")}
        sorted_tags = sorted(tags, key=len, reverse=True)

        # Collect all bracket contents from statement + context, skipping math
        cited_tags: set[str] = set()
        unmatched_brackets: list[str] = []
        for key in ("statement", "context"):
            text = prob.get(key, "")
            if not text:
                continue
            # Strip math so we don't match brackets inside $...$ or $$...$$
            text_no_math = _MATH_RE.sub("", text)
            for m in _BRACKET_RE.finditer(text_no_math):
                bracket = m.group(1)
                hit = None
                for tag in sorted_tags:
                    if tag in bracket:
                        hit = tag
                        break
                if hit:
                    cited_tags.add(hit)
                else:
                    unmatched_brackets.append(bracket)

        # Report brackets that don't match any reference tag
        for bracket in unmatched_brackets:
            errors.append(
                f"{name}: unresolved reference [{bracket}] — "
                f"no matching tag in references list"
            )

        # Report reference tags never cited in text
        for tag in sorted(tags - cited_tags):
            errors.append(
                f"{name}: reference tag [{tag}] is in the references list "
                f"but never cited in statement or context"
            )
    return errors


def parse_toml_response(text: str) -> list[dict]:
    """Extract TOML from LLM response and parse the problems list.

    Returns a list of problem dicts (possibly empty if no problems found).
    Raises TOMLParseError if TOML is present but malformed.
    """
    # Find ```toml ... ``` block
    m = re.search(r"```toml\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        # Try without fence
        m = re.search(r"(\[\[problems\]\].*)", text, re.DOTALL)
    if not m:
        return []

    toml_text = m.group(1)
    try:
        data = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as e:
        raise TOMLParseError(f"{e}\n    Raw TOML:\n{toml_text[:500]}")

    return data.get("problems", [])


# ---------------------------------------------------------------------------
# TOML writing helpers
# ---------------------------------------------------------------------------

def write_problem_toml(problem: dict, path: Path):
    """Write a single problem as a TOML file."""
    lines = []
    lines.append(f'name = {_toml_str(problem.get("name", ""))}')
    lines.append(f'id = {_toml_str(problem.get("id", ""))}')
    lines.append(f'summary = {_toml_str(problem.get("summary", ""))}')
    lines.append(f'location = {_toml_str(problem.get("location", ""))}')
    lines.append(f"statement = '''\n{problem.get('statement', '').strip()}\n'''")
    lines.append(f"context = '''\n{problem.get('context', '').strip()}\n'''")

    refs = problem.get("references", [])
    if refs:
        for ref in refs:
            lines.append("")
            lines.append("[[references]]")
            lines.append(f'tag = {_toml_str(ref.get("tag", ""))}')
            lines.append(f'text = {_toml_str(ref.get("text", ""))}')

    images = problem.get("images", [])
    if images:
        for img in images:
            lines.append("")
            lines.append("[[images]]")
            name = img.get("name", img) if isinstance(img, dict) else img
            lines.append(f'name = {_toml_str(name)}')

    path.write_text("\n".join(lines) + "\n")


def _make_slug(name: str, existing: set[str]) -> str:
    """Derive a filesystem-safe slug from a problem name, deduplicating."""
    slug = re.sub(r"[^a-z0-9-]", "-", name.lower().strip())
    slug = re.sub(r"-+", "-", slug).strip("-") or "problem"
    # Truncate overly long slugs
    if len(slug) > 32:
        slug = slug[:32].rstrip("-")
    if slug not in existing:
        return slug
    for i in range(2, 1000):
        candidate = f"{slug}-{i}"
        if candidate not in existing:
            return candidate
    return f"{slug}-{len(existing)}"


def _toml_str(s: str) -> str:
    """Format a string as a TOML value. Uses literal strings for content with
    backslashes (LaTeX), basic strings otherwise."""
    if "\\" in s or "\n" in s:
        if "\n" in s:
            return f"'''\n{s}\n'''"
        return f"'{s}'"
    escaped = s.replace('"', '\\"')
    return f'"{escaped}"'


# ---------------------------------------------------------------------------
# Extract commands
# ---------------------------------------------------------------------------

def _fmt_cost(cost) -> str:
    """Format a cost value as USD string."""
    if cost is None or cost == 0:
        return "n/a"
    cost = float(cost)
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _write_extract_toml(paper_dir: Path, usage: dict | None = None,
                        error: str | None = None,
                        num_problems: int | None = None):
    """Write extraction.toml with extraction metadata, cost, and errors."""
    lines = []
    if usage:
        lines.append(f'model = "{usage["model"]}"')
        lines.append(f"prompt_tokens = {usage['prompt_tokens']}")
        lines.append(f"completion_tokens = {usage['completion_tokens']}")
        lines.append(f"total_tokens = {usage['total_tokens']}")
        lines.append(f'cost = "{_fmt_cost(usage.get("cost"))}"')
    if num_problems is not None:
        lines.append(f"num_problems = {num_problems}")
    if error:
        lines.append(f'error = """\\\n{error}"""')
    (paper_dir / "extraction.toml").write_text("\n".join(lines) + "\n")


def _save_llm_call(call_dir: Path, result: dict, prompt: str | None = None):
    """Save an LLM call's artifacts to a numbered subdirectory."""
    call_dir.mkdir(parents=True, exist_ok=True)
    if prompt is not None:
        (call_dir / "prompt.txt").write_text(prompt)
    (call_dir / "response.txt").write_text(result["content"])
    if result["reasoning"]:
        (call_dir / "reasoning.txt").write_text(result["reasoning"])


def _save_problems(problems: list[dict], problems_dir: Path):
    """Assign slugs and write TOML files for each problem."""
    used_slugs: set[str] = set()
    for prob in problems:
        slug = _make_slug(prob.get("name", "problem"), used_slugs)
        used_slugs.add(slug)
        prob["id"] = slug
        toml_path = problems_dir / f"{slug}.toml"
        write_problem_toml(prob, toml_path)
        print(f"  {_C.GREEN}OK{_C.RESET}   {slug}")


def extract_paper(paper_dir: Path, model: str, force: bool = False,
                   retries: int = 2):
    """Extract open problems from a single paper directory."""
    # Read paper title for display
    paper_toml = paper_dir / "paper.toml"
    title = ""
    if paper_toml.exists():
        try:
            meta = tomllib.loads(paper_toml.read_text())
            title = meta.get("title", "")
        except Exception:
            pass
    header = paper_dir.name
    if title:
        header += f"  {_C.DIM}{title[:60]}{_C.RESET}"
    print(f"\n{_C.BOLD}{_C.CYAN}{header}{_C.RESET}")

    extraction_toml = paper_dir / "extraction.toml"
    if extraction_toml.exists() and not force:
        print(f"  {_C.YELLOW}already extracted (use --force){_C.RESET}")
        return

    source_dir = paper_dir / "source"
    if not source_dir.exists():
        print(f"  {_C.YELLOW}no source/, skipping{_C.RESET}")
        return

    llm_calls_dir = paper_dir / "llm_calls"
    if force and llm_calls_dir.exists():
        shutil.rmtree(llm_calls_dir)
    llm_calls_dir.mkdir(exist_ok=True)
    mathjax_dir = paper_dir / "mathjax_checks"
    if force and mathjax_dir.exists():
        shutil.rmtree(mathjax_dir)

    prompt = build_prompt(source_dir)

    # Rough token estimate (~4 chars/token) and check against model context
    est_tokens = len(prompt) // 4 + len(EXTRACT_SYSTEM_PROMPT) // 4
    ctx_len = get_model_context_length(model)
    ctx_info = f"{_C.DIM}ctx: {ctx_len:,}{_C.RESET}" if ctx_len else ""
    print(f"  prompt ~{est_tokens:,} tokens  {ctx_info}")

    if ctx_len is not None and est_tokens > ctx_len:
        msg = f"prompt too large (~{est_tokens:,} > {ctx_len:,})"
        print(f"  {_C.RED}{msg}{_C.RESET}")
        _write_extract_toml(paper_dir, error=msg)
        raise RuntimeError(msg)

    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    max_attempts = 1 + retries
    last_error = None
    attempt = 0

    for attempt in range(max_attempts):
        call_dir = llm_calls_dir / f"{attempt:03d}"
        result = call_llm(messages, model)
        _save_llm_call(call_dir, result, prompt if attempt == 0 else None)

        try:
            problems = parse_toml_response(result["content"])
            last_error = None
            break
        except TOMLParseError as e:
            last_error = str(e)
            remaining = max_attempts - attempt - 1
            if remaining > 0:
                print(f"    {_C.YELLOW}TOML parse error (retrying, "
                      f"{remaining} attempt{'s' if remaining != 1 else ''} "
                      f"left): {e.args[0].splitlines()[0]}{_C.RESET}")
                # Feed the error back so the LLM can correct its output.
                messages.append({"role": "assistant", "content": result["content"]})
                messages.append({"role": "user", "content": (
                    f"TOML parse error: {e.args[0].splitlines()[0]}\n\n"
                    "This is almost certainly caused by using double-quoted "
                    "strings (\"...\") for values containing LaTeX backslashes. "
                    "TOML double-quoted strings interpret backslashes as escape "
                    "sequences (e.g. \\n becomes a newline, \\f becomes a form "
                    "feed), which corrupts the math content.\n\n"
                    "You MUST use TOML literal strings (triple single quotes "
                    "'''...''') for ALL string values. Please regenerate the "
                    "complete TOML output with '''...''' for every field."
                )})
            else:
                print(f"    {_C.RED}TOML parse error (no retries left): "
                      f"{e}{_C.RESET}")
                problems = []

    # Validation refinement: if problems parsed OK, validate MathJax
    # rendering and reference consistency, ask the LLM to fix errors
    # (up to 2 rounds).
    validation_retries = 2
    mathjax_checks_dir = paper_dir / "mathjax_checks"
    mathjax_check_num = 0
    if problems and not last_error:
        for _refine in range(validation_retries):
            # MathJax check
            check_dir = mathjax_checks_dir / f"{mathjax_check_num:03d}"
            mathjax_check_num += 1
            math_errors = validate_mathjax(problems, save_dir=check_dir)

            # Reference check
            ref_errors = check_references(problems)

            all_errors = math_errors + ref_errors
            if not all_errors:
                break

            if math_errors:
                print(f"    {_C.YELLOW}MathJax errors "
                      f"({len(math_errors)}):{_C.RESET}")
                for e in math_errors:
                    print(f"      {_C.DIM}{e}{_C.RESET}")
            if ref_errors:
                print(f"    {_C.YELLOW}Reference errors "
                      f"({len(ref_errors)}):{_C.RESET}")
                for e in ref_errors:
                    print(f"      {_C.DIM}{e}{_C.RESET}")

            # Build feedback message with all errors
            feedback_parts = []
            if math_errors:
                error_list = "\n".join(f"  - {e}" for e in math_errors)
                feedback_parts.append(
                    "MathJax rendering errors — these expressions fail "
                    "to render in the browser:\n"
                    f"{error_list}\n\n"
                    "Remember: \\begin/\\end environments like "
                    "\\begin{conjecture}, \\begin{enumerate}, \\begin{align} "
                    "must NOT appear as top-level text. Use markdown instead "
                    "(**Conjecture.**, lists, $$...$$). Math-mode environments "
                    "like \\begin{cases}, \\begin{bmatrix} are OK only inside "
                    "$...$ or $$...$$ delimiters."
                )
            if ref_errors:
                error_list = "\n".join(f"  - {e}" for e in ref_errors)
                feedback_parts.append(
                    "Reference errors — the references list must match "
                    "what is cited in the text:\n"
                    f"{error_list}\n\n"
                    "Every [Tag] bracket in statement/context must have a "
                    "matching entry in the references list, and every entry "
                    "in the references list must be cited as [Tag] somewhere "
                    "in the statement or context."
                )

            messages.append({"role": "assistant", "content": result["content"]})
            messages.append({"role": "user", "content": (
                "The extracted problems have validation errors.\n\n"
                + "\n\n".join(feedback_parts)
                + "\n\nPlease regenerate the COMPLETE corrected TOML output."
            )})

            attempt += 1
            call_dir = llm_calls_dir / f"{attempt:03d}"
            result = call_llm(messages, model)
            _save_llm_call(call_dir, result)

            try:
                problems = parse_toml_response(result["content"])
            except TOMLParseError as e:
                last_error = str(e)
                print(f"    {_C.RED}TOML parse error during "
                      f"refinement: {e.args[0].splitlines()[0]}{_C.RESET}")
                problems = []
                break

    n_prob = len(problems)
    cost_str = _fmt_cost(result["usage"].get("cost"))
    prob_color = _C.GREEN if n_prob > 0 else _C.DIM
    print(f"  {prob_color}{n_prob} problem(s){_C.RESET}  "
          f"{result['usage']['total_tokens']:,} tokens  {cost_str}")

    # Clean old problem files before writing new results
    problems_dir = paper_dir / "problems"
    if problems_dir.exists():
        for old in problems_dir.iterdir():
            old.unlink()
    problems_dir.mkdir(exist_ok=True)

    if problems:
        _save_problems(problems, problems_dir)

    _write_extract_toml(paper_dir, usage=result["usage"],
                        num_problems=len(problems),
                        error=f"TOML parse failed after {max_attempts} attempt(s): {last_error}" if last_error else None)

    if not problems and not last_error:
        (problems_dir / "none.txt").write_text(
            "No open problems found in this paper.\n"
        )


def cmd_extract(args):
    """Extract open problems from one or more papers."""
    data_dir = Path(args.data_dir)
    for arxiv_id in args.arxiv_id:
        paper_id = arxiv_id.replace("/", "_")
        paper_dir = data_dir / paper_id
        if not paper_dir.exists():
            print(f"Error: paper directory not found: {paper_dir}")
            continue
        try:
            extract_paper(paper_dir, args.model, force=args.force,
                          retries=args.retries)
        except Exception as e:
            print(f"  {_C.RED}ERROR: {e}{_C.RESET}")


def cmd_extract_all(args):
    """Extract open problems from all downloaded papers."""
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        return 1

    paper_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if args.max is not None:
        paper_dirs = paper_dirs[:args.max]
    print(f"{_C.BOLD}Extracting from {len(paper_dirs)} paper(s){_C.RESET}  "
          f"{_C.DIM}model: {args.model}{_C.RESET}")

    for paper_dir in paper_dirs:
        try:
            extract_paper(paper_dir, args.model, force=args.force,
                          retries=args.retries)
        except Exception as e:
            print(f"  {_C.RED}ERROR: {e}{_C.RESET}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download ArXiv combinatorics papers and extract open "
        "problems."
    )
    parser.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR),
        help=f"Data directory (default: {DEFAULT_DATA_DIR})",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # download
    dl = sub.add_parser("download", help="Download papers from ArXiv math.CO")
    dl.add_argument(
        "--max", type=int, default=10,
        help="Max papers to download (default: 10)",
    )

    # extract
    ex = sub.add_parser(
        "extract", help="Extract open problems from a single paper",
    )
    ex.add_argument("arxiv_id", nargs="+", help="ArXiv paper ID(s) (e.g. 2506.12345)")
    ex.add_argument(
        "--model", default="z-ai/glm-5",
        help="OpenRouter model (default: z-ai/glm-5)",
    )
    ex.add_argument(
        "--force", action="store_true",
        help="Overwrite existing extraction results",
    )
    ex.add_argument(
        "--retries", type=int, default=2,
        help="Max retries on TOML parse failure (default: 2)",
    )
    # extract-all
    ea = sub.add_parser(
        "extract-all",
        help="Extract open problems from all downloaded papers",
    )
    ea.add_argument(
        "--model", default="z-ai/glm-5",
        help="OpenRouter model (default: z-ai/glm-5)",
    )
    ea.add_argument(
        "--force", action="store_true",
        help="Overwrite existing extraction results",
    )
    ea.add_argument(
        "--retries", type=int, default=2,
        help="Max retries on TOML parse failure (default: 2)",
    )
    ea.add_argument(
        "--max", type=int, default=None,
        help="Max papers to extract (default: all)",
    )
    args = parser.parse_args()

    if args.action == "download":
        cmd_download(args)
    elif args.action == "extract":
        cmd_extract(args)
    elif args.action == "extract-all":
        cmd_extract_all(args)


if __name__ == "__main__":
    main()
