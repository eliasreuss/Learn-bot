import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
try:
    from langchain_community.document_loaders import PyPDFLoader
    HAS_PDF = True
except Exception:
    HAS_PDF = False

load_dotenv()

HEADER_TMPL = (
    "Topic: {topic}\n"
    "Source article: {source}\n"
    "Keywords: {keywords}\n\n"
    "{content}\n"
)


def get_system_prompt(target_language: str | None) -> str:
    base = (
        "You are an expert technical editor and knowledge architect. Given an excerpt of a document, "
        "identify self-contained, high-utility chunks suitable for a retrieval-augmented chatbot.\n"
        "Return ONLY valid JSON (no prose) as a list named 'chunks'. Each item must have: \n"
        "- topic: short title line\n"
        "- source: short source title (e.g., document or section)\n"
        "- keywords: JSON array of 5-10 short strings (e.g., [\"inventory\", \"abc analysis\"])\n"
        "- content: a compact, standalone chunk (<= 1200 chars) with numbered steps if instructive.\n"
        "Do not include duplicate chunks across windows; prefer concise, unique entries."
    )
    if target_language:
        base += (
            "\nIMPORTANT: Write topic, keywords, and content in " + target_language + ". Translate faithfully if the excerpt is in another language."
        )
    else:
        base += ("\nWrite in the same language as the input excerpt.")
    return base

USER_PROMPT_TMPL = (
    "Document excerpt:\n\n"
    + '"""' + "{excerpt}" + '"""' + "\n\n"
    + "Produce the JSON now."
)


def load_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()[0].page_content
    if suf == ".docx":
        return Docx2txtLoader(str(path)).load()[0].page_content
    if suf == ".pdf" and HAS_PDF:
        pages = PyPDFLoader(str(path)).load()
        return "\n\n".join(p.page_content for p in pages)
    if suf == ".pdf" and not HAS_PDF:
        raise RuntimeError("PDF support requires pypdf. Install with: pip install pypdf")
    raise RuntimeError(f"Unsupported file type: {suf}")


def window(text: str, max_chars: int = 5000, overlap: int = 400) -> List[str]:
    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = j - overlap
        if i < 0:
            i = 0
    return chunks


def llm_extract(llm: ChatOpenAI, text: str, target_language: str | None) -> List[Dict]:
    resp = llm.invoke([
        {"role": "system", "content": get_system_prompt(target_language)},
        {"role": "user", "content": USER_PROMPT_TMPL.format(excerpt=text)}
    ])
    raw = resp.content.strip()
    # Try to pull JSON from code fences if present
    m = re.search(r"```json\s*(.*?)```", raw, re.S | re.I)
    if m:
        raw = m.group(1)
    try:
        data = json.loads(raw)
        chunks = data.get("chunks") if isinstance(data, dict) else data
        if not isinstance(chunks, list):
            raise ValueError("Expected a list under 'chunks'")
        return chunks
    except Exception as e:
        # Best-effort fallback: wrap as empty list
        print(f"Warning: JSON parse failed; skipping window. Error: {e}\nRaw: {raw[:400]}...")
        return []


def unique_slug(topic: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\-_\s]", "", topic).strip().lower()
    slug = re.sub(r"\s+", "-", slug)
    return slug[:80] or "chunk"


def curate(input_path: Path, out_dir: Path, language: str, model: str, target_language: str | None) -> List[Path]:
    text = load_text(input_path)
    llm = ChatOpenAI(temperature=0, model_name=model)

    def normalize_keywords(value) -> List[str]:
        # If already a list, flatten and split any embedded commas/semicolons
        def split_str(s: str) -> List[str]:
            # Try JSON array parse first
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed]
            except Exception:
                pass
            parts = re.split(r"[;,\n\u2022\u2023\u25E6\u2043\u2219]+", s)
            return [p.strip() for p in parts if p.strip()]

        items: List[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    items.extend(split_str(item))
                else:
                    items.append(str(item).strip())
        elif isinstance(value, str):
            items = split_str(value)
        # Deduplicate while preserving order
        seen = set()
        result: List[str] = []
        for it in items:
            if it and it not in seen:
                seen.add(it)
                result.append(it)
        return result[:10]

    outputs: List[Dict] = []
    for win in window(text):
        outputs.extend(llm_extract(llm, win, target_language))

    # Deduplicate by topic slug
    seen = set()
    written: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    source_title = input_path.stem.replace("_", " ")

    for item in outputs:
        topic = str(item.get("topic", "")).strip()
        content = str(item.get("content", "")).strip()
        keywords_list = item.get("keywords", [])
        if not topic or not content:
            continue
        slug = unique_slug(topic)
        if slug in seen:
            continue
        seen.add(slug)
        keywords = ", ".join(normalize_keywords(keywords_list))
        body = HEADER_TMPL.format(
            topic=topic,
            source=source_title,
            keywords=keywords,
            content=content,
        )
        target = out_dir / f"{unique_slug(source_title)}_{slug}.txt"
        target.write_text(body, encoding="utf-8")
        written.append(target)
    return written


def detect_language_llm(text: str) -> str:
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        prompt = (
            "You are a language detection helper. Is the text primarily in English or Danish? "
            "Answer with a single word: english or danish.\n\nText:\n" + text[:4000]
        )
        resp = llm.invoke(prompt)
        ans = resp.content.strip().lower()
        if "english" in ans:
            return "english"
    except Exception as e:
        print(f"Language detection failed, defaulting to danish: {e}")
    return "danish"


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-curate high-quality chatbot chunks with headers.")
    parser.add_argument("--input", required=True, help="Source file (txt, docx, md, pdf)")
    parser.add_argument("--language", choices=["danish", "english"], default="danish")
    parser.add_argument("--output-dir", default="data/knowledge/danish")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--auto-language", action="store_true", help="Auto-detect language and write to the corresponding data/knowledge/<lang>/ folder")
    parser.add_argument("--target-language", choices=["danish", "english"], help="If set, translate curated chunks to this language")
    args = parser.parse_args()

    input_path = Path(args.input)
    # If auto-language is requested, detect and override output dir accordingly
    if args.auto_language:
        text_for_lang = load_text(input_path)
        detected = detect_language_llm(text_for_lang)
        base_dir = Path("data/knowledge") / detected
        out_dir = base_dir
        print(f"Auto-detected language: {detected}. Writing curated chunks to {out_dir}")
    else:
        out_dir = Path(args.output_dir)

    written = curate(input_path, out_dir, args.language, args.model, args.target_language)
    print(f"Wrote {len(written)} curated chunks to {out_dir}")


if __name__ == "__main__":
    main() 