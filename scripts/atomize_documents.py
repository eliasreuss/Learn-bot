import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv

# Load env for OPENAI_API_KEY if LLM refinement is used
load_dotenv()

# Optional LLM refinement
try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

# Loaders (lazy to avoid heavy deps if not needed)
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
try:
    from langchain_community.document_loaders import PyPDFLoader
    HAS_PDF = True
except Exception:
    HAS_PDF = False

SEPARATORS = [
    "\n\n## ", "\n\n# ", "\n\n### ", "\n\n- ", "\n\n* ", "\n\n1. ", "\n\n2. ", "\n\n",
]


def read_documents(input_path: Path) -> List[Tuple[str, str]]:
    files: List[Path] = []
    if input_path.is_file():
        files = [input_path]
    else:
        for ext in ("*.txt", "*.docx", "*.md", "*.pdf"):
            files.extend(sorted(input_path.rglob(ext)))

    docs: List[Tuple[str, str]] = []
    for fp in files:
        try:
            text = load_file_text(fp)
            if text.strip():
                docs.append((fp.name, text))
        except Exception as e:
            print(f"Skip {fp}: {e}")
    return docs


def load_file_text(fp: Path) -> str:
    suffix = fp.suffix.lower()
    if suffix == ".txt" or suffix == ".md":
        return TextLoader(str(fp), encoding="utf-8").load()[0].page_content
    if suffix == ".docx":
        return Docx2txtLoader(str(fp)).load()[0].page_content
    if suffix == ".pdf" and HAS_PDF:
        pages = PyPDFLoader(str(fp)).load()
        return "\n\n".join(page.page_content for page in pages)
    if suffix == ".pdf" and not HAS_PDF:
        raise RuntimeError("PDF support requires 'pypdf' (install via pip)")
    raise RuntimeError(f"Unsupported file type: {suffix}")


def heading_aware_split(text: str, max_len: int = 1000) -> List[str]:
    # First, split on likely headings and double newlines
    parts: List[str] = [text]
    for sep in SEPARATORS:
        parts = sum((p.split(sep) for p in parts), [])

    # Normalize whitespace
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts]
    parts = [p for p in parts if p]

    # Further split long parts softly on periods
    refined: List[str] = []
    for p in parts:
        if len(p) <= max_len:
            refined.append(p)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buf = []
            cur = 0
            for s in sentences:
                if cur + len(s) > max_len and buf:
                    refined.append(" ".join(buf).strip())
                    buf, cur = [], 0
                buf.append(s)
                cur += len(s) + 1
            if buf:
                refined.append(" ".join(buf).strip())
    return refined


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\-\_\s]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text[:80] or "chunk"


def llm_refine_titles(chunks: List[str], language: str) -> List[str]:
    if ChatOpenAI is None:
        return [f"chunk-{i+1:02d}" for i in range(len(chunks))]
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    prompt = (
        f"You will receive a list of paragraphs in {language}. "
        "Generate a concise kebab-case filename slug (max 8 words) for each paragraph that reflects its main intent. "
        "Only return the slugs as a newline-separated list, no numbering, no extra text."
    )
    joined = "\n\n".join(chunks)
    resp = llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": joined},
    ])
    lines = [l.strip() for l in resp.content.splitlines() if l.strip()]
    if len(lines) != len(chunks):
        # Fallback if counts mismatch
        return [f"chunk-{i+1:02d}" for i in range(len(chunks))]
    return [slugify(l) for l in lines]


def write_chunks(chunks: List[str], slugs: List[str], out_dir: Path, base_prefix: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for idx, (chunk, slug) in enumerate(zip(chunks, slugs), start=1):
        filename = f"{base_prefix}_{slug}.txt"
        target = out_dir / filename
        target.write_text(chunk.strip() + "\n", encoding="utf-8")
        written.append(target)
    return written


def process(input_path: Path, language: str, output_dir: Path, use_llm_titles: bool) -> None:
    docs = read_documents(input_path)
    if not docs:
        print("No documents found.")
        return

    for original_name, text in docs:
        base_prefix = Path(original_name).stem
        base_prefix = slugify(base_prefix)
        chunks = heading_aware_split(text)
        if not chunks:
            continue
        if use_llm_titles:
            slugs = llm_refine_titles(chunks, language)
        else:
            slugs = [f"chunk-{i:02d}" for i in range(1, len(chunks) + 1)]
        written = write_chunks(chunks, slugs, output_dir, base_prefix)
        print(f"Wrote {len(written)} chunks for {original_name} -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomize documents into knowledge chunks.")
    parser.add_argument("--input", required=True, help="File or directory containing source documents")
    parser.add_argument("--language", choices=["danish", "english"], default="danish")
    parser.add_argument(
        "--output-dir",
        default=str(Path("data/knowledge") / "danish"),
        help="Output directory for .txt chunks",
    )
    parser.add_argument("--use-llm-titles", action="store_true", help="Use OpenAI to create better slugs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    process(input_path, args.language, output_dir, args.use_llm_titles)


if __name__ == "__main__":
    main() 