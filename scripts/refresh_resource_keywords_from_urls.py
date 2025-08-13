import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

RESOURCES_DIR = Path(__file__).resolve().parents[1] / "data" / "resources"

# Minimal Danish + English stopwords sufficient for keywording. Extend as needed.
STOPWORDS = set(
    [
        # Danish
        "og","i","jeg","det","at","en","den","til","er","som","på","de","med","han","af","for","ikke","der","var","mig","sig","men","et","har","om","vi","min","havde","ham","hun","nu","over","da","fra","du","ud","sin","dem","os","op","man","hans","hvor","eller","hvad","skal","selv","her","alle","vil","blev","kunne","ind","når","være","kom","så","nej","ja","mit","din","dine","dit","jer","jeres","vores","deres","ikke","også","meget","mere","mest","mange","få","flere","bliver","gør","gøre","hvordan","hvorfor","hvornår","hvem","hvilken","hvilke","hvilket","uden","igen","allerede","bare","sådan","kun","nok","således","al","andet",
        # Extra Danish fillers/common UI words
        "helt","går","kan","tilbage","inspirationssiden","inspiration","artikel","artikler","læs","mere","se","her",
        # English
        "the","and","to","of","in","a","is","it","that","for","on","as","are","with","this","be","or","by","an","from","at","was","we","you","your","our","their","have","has","had","can","will","not","but","if","they","he","she","them","there","here","what","how","why","when","which","who","where","into","about","also","more","most","many","much","very","just","only","than","then","so","do","does","did","done","been","being","may","might","should","would","could","over","under","between","within","across","per","via","using","use","used","using"
    ]
)

DOMAIN_BOOST: Dict[str, int] = {
    # Danish domain terms
    "action": 3,
    "actions": 3,
    "checklist": 3,
    "dashboard": 3,
    "dashboards": 3,
    "widget": 2,
    "widgets": 2,
    "insight": 3,
    "insights": 3,
    "master": 2,
    "data": 2,
    "abc": 2,
    "analyse": 2,
    "rapport": 2,
    "rapporter": 2,
    "subscription": 2,
    "subscriptions": 2,
    "admin": 2,
    "inact": 4,
}

HEADER_RE = {
    "type": re.compile(r"^\s*Type:\s*(.+)$", re.I | re.M),
    "title": re.compile(r"^\s*Title:\s*(.+)$", re.I | re.M),
    "url": re.compile(r"^\s*URL:\s*(.+)$", re.I | re.M),
    "desc": re.compile(r"^\s*(Description|Beskrivelse):\s*(.+)$", re.I | re.M),
    "keywords": re.compile(r"^\s*Keywords?:\s*(.+)$", re.I | re.M),
}

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
}

NOISE_PHRASES: List[str] = [
    "tilbage til inspirationssiden",
    "tilbage til",
]

def is_noise_term(term: str) -> bool:
    t = term.strip().lower()
    if not t:
        return True
    if t in STOPWORDS:
        return True
    if any(p in t for p in ["inspiration", "inspirationsside"]):
        return True
    if t in {"helt", "kan", "går", "tilbage"}:
        return True
    if t in NOISE_PHRASES:
        return True
    return False


def fetch_body_text(url: str, timeout: int = 20) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=timeout, headers=UA_HEADERS)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"Fetch failed: {url} -> {e}")
        return None

    soup = BeautifulSoup(html, "html.parser")
    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()
    # Prefer <article> or main content if present
    main = soup.find("article") or soup.find("main") or soup.body
    if not main:
        main = soup
    # Join paragraphs/headings
    texts: List[str] = []
    for el in main.find_all(["h1", "h2", "h3", "p", "li"]):
        t = (el.get_text(" ", strip=True) or "").strip()
        if t:
            texts.append(t)
    content = "\n".join(texts)
    return content if content.strip() else None


def tokenize(text: str) -> List[str]:
    # Keep Danish letters, fold to lowercase
    tokens = re.split(r"[^a-zA-ZæøåÆØÅ0-9]+", text.lower())
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    return tokens


def top_keywords(text: str, title: str, extra: List[str], k: int = 12) -> List[str]:
    toks = tokenize(text)
    if not toks:
        return []
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1 + DOMAIN_BOOST.get(t, 0)
    # Title words get a boost
    for t in tokenize(title):
        freq[t] = freq.get(t, 0) + 2 + DOMAIN_BOOST.get(t, 0)
    for t in extra:
        tt = t.strip().lower()
        if tt:
            freq[tt] = freq.get(tt, 0) + 3 + DOMAIN_BOOST.get(tt, 0)
    # Sort by score desc then alphabetically
    cand = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    singles = [w for w, _ in cand[: k + 8]]

    # Basic bigrams from consecutive tokens
    bigram_freq: Dict[Tuple[str, str], int] = {}
    for a, b in zip(toks, toks[1:]):
        if a in STOPWORDS or b in STOPWORDS:
            continue
        key = (a, b)
        bigram_freq[key] = bigram_freq.get(key, 0) + 1
    bigrams = [" ".join(pair) for pair, c in sorted(bigram_freq.items(), key=lambda kv: -kv[1]) if c >= 2]

    merged = []
    seen = set()
    for term in bigrams + singles:
        if is_noise_term(term):
            continue
        if term in seen:
            continue
        seen.add(term)
        merged.append(term)
        if len(merged) >= k:
            break
    return merged


def insert_or_replace_keywords(text: str, keywords: List[str]) -> str:
    line = f"Keywords: {', '.join(keywords)}"
    m = HEADER_RE["keywords"].search(text)
    if m:
        # Replace existing line
        start, end = m.span()
        return text[:start] + line + text[end:]
    # Insert after Description or URL
    m_desc = HEADER_RE["desc"].search(text)
    if m_desc:
        end = m_desc.end()
        before = text[:end]
        after = text[end:]
        if not before.endswith("\n"):
            before += "\n"
        return f"{before}{line}\n{after}"
    m_url = HEADER_RE["url"].search(text)
    if m_url:
        end = m_url.end()
        before = text[:end]
        after = text[end:]
        if not before.endswith("\n"):
            before += "\n"
        return f"{before}{line}\n{after}"
    if not text.endswith("\n"):
        text += "\n"
    return text + line + "\n"


def process_file(fp: Path, timeout: int, topk: int) -> bool:
    txt = fp.read_text(encoding="utf-8")
    title_m = HEADER_RE["title"].search(txt)
    url_m = HEADER_RE["url"].search(txt)
    desc_m = HEADER_RE["desc"].search(txt)
    if not url_m:
        return False
    url = url_m.group(1).strip()
    title = title_m.group(1).strip() if title_m else fp.stem
    desc = desc_m.group(2).strip() if desc_m else ""

    body = fetch_body_text(url, timeout=timeout)
    if not body:
        return False

    kws = top_keywords(body, title, extra=tokenize(desc), k=topk)
    if not kws:
        return False
    updated = insert_or_replace_keywords(txt, kws)
    if updated != txt:
        fp.write_text(updated, encoding="utf-8")
        print(f"Updated from URL: {fp.name} -> {len(kws)} keywords")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer and refresh Keywords for resources by scraping their URLs")
    parser.add_argument("--topk", type=int, default=12, help="Number of keywords to keep")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout seconds")
    parser.add_argument("--file", type=str, default=None, help="Process a single resource file (basename or path)")
    args = parser.parse_args()

    if args.file:
        fp = Path(args.file)
        if not fp.exists():
            fp = RESOURCES_DIR / args.file
        changed = process_file(fp, args.timeout, args.topk)
        print("Changed" if changed else "No change")
        return

    changed_any = False
    for fp in sorted(RESOURCES_DIR.glob("*.txt")):
        try:
            if process_file(fp, args.timeout, args.topk):
                changed_any = True
        except Exception as e:
            print(f"Failed {fp.name}: {e}")
    print("Done.")


if __name__ == "__main__":
    if not RESOURCES_DIR.exists():
        print(f"Resources directory not found: {RESOURCES_DIR}")
        sys.exit(1)
    main() 