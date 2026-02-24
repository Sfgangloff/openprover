#!/usr/bin/env python3
"""Download ArXiv combinatorics papers and extract open problems using an LLM."""

import argparse
import gzip
import io
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
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

# LaTeX template for validating extracted problem snippets.
# The snippets (context + statement) are inserted at the %% SNIPPET %% marker.
# Packages chosen for broad math coverage AND MathJax compatibility.
TEMPLATE_TEX = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage{hyperref}
\usepackage{braket}

\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{conjecture}[theorem]{Conjecture}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{example}[theorem]{Example}
\newtheorem{question}[theorem]{Question}
\newtheorem{problem}[theorem]{Problem}
\theoremstyle{definition}
\newtheorem{openproblem}[theorem]{Open Problem}

\begin{document}

%% SNIPPET %%

\end{document}
"""

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
- The "statement" is the full LaTeX statement from the paper.  It must be a LaTeX snippet (not a full document). It must not contain any \\section or \\subsection commands. It should be self-contained math content that can be inserted into a document body.
- The "context" contains all relevant background from this paper needed to understand the statement. Only include context explicitly mentioned in this paper! It is also a LaTeX snippet. It may use \\subsection{...} to organize long context, but must not use \\section{...}. It should include definitions of key terms, notation, and any prerequisite results referenced in the statement.
- Both statement and context may reference \\cite{...}, \\citet{...}, or \\citep{...} - but only include truly relevant references.
- Every reference and every \\includegraphics image mentioned in statement or context must appear in the references/images lists. Conversely, every entry in references/images must actually be cited or included in the statement or context LaTeX.
- Clean up LaTeX: expand or define any custom macros. The snippets must compile standalone with standard packages (amsmath, amssymb, amsthm, mathtools, hyperref, braket).
- For MathJax web compatibility: avoid obscure packages. Stick to standard math commands. Prefer \\operatorname{name} over custom \\DeclareMathOperator. Avoid \\tikz, \\pstricks, or other drawing commands. Use \\text{...} for text in math mode.
- It is perfectly fine for a paper to have zero open problems. In that case, output an empty problems list.

The LaTeX snippets will be inserted into this template for validation:

```latex
""" + TEMPLATE_TEX + """\
```

Output format - a single TOML document.
IMPORTANT: For all fields that contain LaTeX (statement, context, text), you MUST use TOML literal strings (triple single quotes '''...''') so that backslashes are preserved verbatim.
Do NOT use basic strings (double quotes) for LaTeX content, as TOML interprets backslashes as escape sequences.

```toml
[[problems]]
name = "Human-readable name of the problem"
summary = "Very brief 1-2 sentence description."
location = "Section 3, Conjecture 4.2"
statement = '''
\\begin{conjecture}
For every graph $G$ with $\\chi(G) \\le 4$, we have...
\\end{conjecture}
'''
context = '''
Let $G = (V, E)$ be a simple undirected graph. The \\emph{chromatic
number} $\\chi(G)$ is the minimum number of colors needed...
'''

[[problems.references]]
tag = "AuthorYear"
text = '''Full reference string as it appears in the bibliography'''

[[problems.images]]
name = "figurefile.png"
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


def parse_toml_response(text: str) -> list[dict]:
    """Extract TOML from LLM response and parse the problems list."""
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
        print(f"    WARNING: TOML parse error: {e}")
        print(f"    Raw TOML:\n{toml_text[:500]}")
        return []

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
# LaTeX validation
# ---------------------------------------------------------------------------

def validate_latex(problem: dict, output_pdf: Path) -> bool:
    """Compile the problem's context + statement into a PDF using the template.

    Returns True if compilation succeeded (output_pdf created).
    On failure, writes the error log to output_pdf.with_suffix('.pdf.error').
    """
    statement = problem.get("statement", "").strip()
    context = problem.get("context", "").strip()

    snippet_parts = []
    if context:
        snippet_parts.append("\\section*{Context}\\setcounter{subsection}{0}")
        snippet_parts.append(context)
    if statement:
        snippet_parts.append("\\section*{Statement}\\setcounter{subsection}{0}")
        snippet_parts.append(statement)
    refs = problem.get("references", [])
    if refs:
        snippet_parts.append("\\begin{thebibliography}{99}")
        for ref in refs:
            tag = ref.get("tag", "?")
            text = ref.get("text", "")
            snippet_parts.append(f"\\bibitem{{{tag}}} {text}")
        snippet_parts.append("\\end{thebibliography}")
    snippet = "\n\n".join(snippet_parts)

    tex_content = TEMPLATE_TEX.replace("%% SNIPPET %%", snippet)

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "theorem.tex"
        tex_path.write_text(tex_content)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "theorem.tex"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        pdf_path = Path(tmpdir) / "theorem.pdf"
        if result.returncode == 0 and pdf_path.exists():
            shutil.copy2(pdf_path, output_pdf)
            # Remove stale error file if it exists
            error_path = output_pdf.parent / (output_pdf.name + ".error")
            error_path.unlink(missing_ok=True)
            return True
        else:
            error_path = output_pdf.parent / (output_pdf.name + ".error")
            log_path = Path(tmpdir) / "theorem.log"
            error_content = ""
            if log_path.exists():
                error_content = log_path.read_text(errors="replace")
            else:
                error_content = result.stdout + "\n" + result.stderr
            error_path.write_text(error_content)
            # Remove stale PDF if it exists
            output_pdf.unlink(missing_ok=True)
            return False


# ---------------------------------------------------------------------------
# Extract commands
# ---------------------------------------------------------------------------

def _fmt_cost(cost) -> str:
    """Format a cost value as USD string."""
    if cost is None:
        return "unknown"
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


def _validate_problems(problems: list[dict], problems_dir: Path
                       ) -> tuple[list[dict], list[dict]]:
    """Validate LaTeX for each problem. Returns (ok_problems, failed_problems).
    Each failed problem gets an 'error' key with the error text."""
    ok, failed = [], []
    used_slugs: set[str] = set()
    for prob in problems:
        slug = _make_slug(prob.get("name", "problem"), used_slugs)
        used_slugs.add(slug)
        prob["id"] = slug
        toml_path = problems_dir / f"{slug}.toml"
        write_problem_toml(prob, toml_path)

        pdf_path = problems_dir / f"{slug}.pdf"
        try:
            if validate_latex(prob, pdf_path):
                print(f"  {_C.GREEN}OK{_C.RESET}   {slug}")
                ok.append(prob)
            else:
                error_path = problems_dir / f"{slug}.pdf.error"
                err = error_path.read_text() if error_path.exists() else ""
                prob["_error"] = err
                failed.append(prob)
                print(f"  {_C.RED}ERR{_C.RESET}  {slug}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            error_path = problems_dir / f"{slug}.pdf.error"
            error_path.write_text(f"LaTeX compilation failed: {e}\n")
            prob["_error"] = str(e)
            failed.append(prob)
            print(f"  {_C.RED}ERR{_C.RESET}  {slug}")
    return ok, failed


def extract_paper(paper_dir: Path, model: str, force: bool = False,
                  max_retries: int = 2):
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
    llm_calls_dir.mkdir(exist_ok=True)

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

    # Initial LLM call
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    call_num = 0
    result = call_llm(messages, model)
    _save_llm_call(llm_calls_dir / f"{call_num:03d}", result, prompt)
    total_usage = dict(result["usage"])

    problems = parse_toml_response(result["content"])
    initial_problems = problems  # keep for fallback

    n_prob = len(problems)
    cost_str = _fmt_cost(total_usage.get("cost"))
    prob_color = _C.GREEN if n_prob > 0 else _C.DIM
    print(f"  {prob_color}{n_prob} problem(s){_C.RESET}  "
          f"{total_usage['total_tokens']:,} tokens  {cost_str}")

    problems_dir = paper_dir / "problems"
    problems_dir.mkdir(exist_ok=True)

    if problems:
        _, failed = _validate_problems(problems, problems_dir)

        # Retry loop: ask LLM to fix LaTeX errors
        messages.append({"role": "assistant", "content": result["content"]})
        retry = 0
        while failed and retry < max_retries:
            retry += 1
            call_num += 1
            error_summary = "\n\n".join(
                f"Problem \"{p.get('name', '?')}\":\n{p['_error'][:2000]}"
                for p in failed
            )
            fix_msg = (
                f"The following {len(failed)} problem(s) had LaTeX "
                f"compilation errors:\n\n{error_summary}\n\n"
                f"Please output a corrected COMPLETE TOML (all problems, "
                f"not just the broken ones) inside a ```toml code fence."
            )
            messages.append({"role": "user", "content": fix_msg})
            print(f"  {_C.YELLOW}retry {retry}/{max_retries}{_C.RESET}  "
                  f"fixing {len(failed)} LaTeX error(s)...")
            result = call_llm(messages, model)
            _save_llm_call(llm_calls_dir / f"{call_num:03d}", result, fix_msg)
            messages.append({"role": "assistant", "content": result["content"]})

            # Accumulate usage
            ru = result["usage"]
            total_usage["prompt_tokens"] += ru["prompt_tokens"]
            total_usage["completion_tokens"] += ru["completion_tokens"]
            total_usage["total_tokens"] += ru["total_tokens"]
            if ru.get("cost") is not None:
                prev = total_usage.get("cost")
                total_usage["cost"] = (prev or 0) + float(ru["cost"])

            new_problems = parse_toml_response(result["content"])
            if not new_problems:
                print(f"    {_C.RED}no problems parsed, stopping{_C.RESET}")
                break

            # Clean problems dir and re-validate
            for f in problems_dir.iterdir():
                f.unlink()
            _, failed = _validate_problems(new_problems, problems_dir)
            if not failed:
                problems = new_problems
                print(f"    {_C.GREEN}all fixed{_C.RESET}")
            else:
                problems = new_problems

        # If retries didn't fix all errors, restore initial output
        if failed and problems is not initial_problems:
            print(f"  {_C.YELLOW}reverting to initial extraction{_C.RESET}")
            for f in problems_dir.iterdir():
                f.unlink()
            _validate_problems(initial_problems, problems_dir)

        # Print totals if retries happened
        if call_num > 0:
            cost_str = _fmt_cost(total_usage.get("cost"))
            print(f"  {_C.DIM}total: {total_usage['total_tokens']:,} "
                  f"tokens  {cost_str}{_C.RESET}")

    _write_extract_toml(paper_dir, usage=total_usage,
                        num_problems=len(problems))

    if not problems:
        (problems_dir / "none.txt").write_text(
            "No open problems found in this paper.\n"
        )


def cmd_extract(args):
    """Extract open problems from a single paper."""
    data_dir = Path(args.data_dir)
    paper_id = args.arxiv_id.replace("/", "_")
    paper_dir = data_dir / paper_id
    if not paper_dir.exists():
        print(f"Error: paper directory not found: {paper_dir}")
        return 1
    extract_paper(paper_dir, args.model, force=args.force,
                  max_retries=args.retries)


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
                          max_retries=args.retries)
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
    ex.add_argument("arxiv_id", help="ArXiv paper ID (e.g. 2506.12345)")
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
        help="Max LLM retries to fix LaTeX errors (default: 2)",
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
        "--max", type=int, default=None,
        help="Max papers to extract (default: all)",
    )
    ea.add_argument(
        "--retries", type=int, default=2,
        help="Max LLM retries to fix LaTeX errors (default: 2)",
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
