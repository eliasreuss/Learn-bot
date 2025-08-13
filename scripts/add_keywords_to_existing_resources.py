import re
import sys
from pathlib import Path
from typing import List

RESOURCES_DIR = Path(__file__).resolve().parents[1] / "data" / "resources"

KEYWORDS_MAP_DANISH = {
    "action": ["action", "actions", "checklist", "create checklist", "opret action", "lav action", "alarm", "alarmer"],
    "dashboard": ["dashboard", "dashboards", "dashboard creator", "widget", "widgets", "visualisering", "overblik", "chart", "tabel"],
    "insight": ["insight", "insights", "analyse", "rapport", "rapporter", "view", "master data"],
    "master data": ["master data", "view", "kolonner", "filtre", "slice", "filtrering"],
    "subscription": ["subscription", "subscriptions", "planlagt", "e-mail", "levering i indbakken", "schedule"],
    "admin": ["admin", "admin panel", "brugere", "roller", "permissions", "rettigheder"],
    "abc": ["abc", "abc analyse", "dobbelt abc", "pareto", "kategori a", "kategori b", "kategori c"],
}

EXTRA_GENERAL = [
    "inact", "inact now", "hjælp", "help"
]

HEADER_RE = {
    "type": re.compile(r"^\s*Type:\s*(.+)$", re.I | re.M),
    "title": re.compile(r"^\s*Title:\s*(.+)$", re.I | re.M),
    "url": re.compile(r"^\s*URL:\s*(.+)$", re.I | re.M),
    "desc": re.compile(r"^\s*(Description|Beskrivelse):\s*(.+)$", re.I | re.M),
    "keywords": re.compile(r"^\s*Keywords?:\s*(.+)$", re.I | re.M),
}


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for it in items:
        t = it.strip()
        if not t:
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        result.append(t)
    return result


def infer_keywords(title: str, desc: str) -> List[str]:
    t = (title or "").lower()
    d = (desc or "").lower()
    keys: List[str] = []

    def hit(word: str) -> bool:
        return word in t or word in d

    if any(hit(w) for w in ["action", "actions", "checklist"]):
        keys += KEYWORDS_MAP_DANISH["action"]
    if any(hit(w) for w in ["dashboard", "dashboards", "widget", "widgets"]):
        keys += KEYWORDS_MAP_DANISH["dashboard"]
    if hit("insight"):
        keys += KEYWORDS_MAP_DANISH["insight"]
    if "master data" in t or "masterdata" in t or "master data" in d:
        keys += KEYWORDS_MAP_DANISH["master data"]
    if any(hit(w) for w in ["subscription", "subscriptions"]):
        keys += KEYWORDS_MAP_DANISH["subscription"]
    if any(hit(w) for w in ["admin", "admin panel"]):
        keys += KEYWORDS_MAP_DANISH["admin"]
    if "abc" in t or "abc" in d:
        keys += KEYWORDS_MAP_DANISH["abc"]

    # Add some tokens from title itself (split on spaces, keep 3+ chars)
    tokens = [w for w in re.split(r"[^a-zA-ZæøåÆØÅ0-9]+", t) if len(w) >= 4]
    keys += tokens[:8]
    keys += EXTRA_GENERAL
    return dedupe_preserve_order(keys)


def insert_keywords_block(text: str, keywords_line: str) -> str:
    # Insert Keywords after the Description/Beskrivelse line if present, else after the URL line, else at end
    m_desc = HEADER_RE["desc"].search(text)
    if m_desc:
        end = m_desc.end()
        before = text[:end]
        after = text[end:]
        if not before.endswith("\n"):
            before += "\n"
        return f"{before}{keywords_line}\n{after}"
    m_url = HEADER_RE["url"].search(text)
    if m_url:
        end = m_url.end()
        before = text[:end]
        after = text[end:]
        if not before.endswith("\n"):
            before += "\n"
        return f"{before}{keywords_line}\n{after}"
    # Fallback: append
    if not text.endswith("\n"):
        text += "\n"
    return text + keywords_line + "\n"


def process_file(fp: Path) -> bool:
    txt = fp.read_text(encoding="utf-8")
    if HEADER_RE["keywords"].search(txt):
        return False  # already has keywords
    title_m = HEADER_RE["title"].search(txt)
    desc_m = HEADER_RE["desc"].search(txt)
    title = title_m.group(1).strip() if title_m else fp.stem
    desc = desc_m.group(2).strip() if desc_m else ""
    keys = infer_keywords(title, desc)
    line = f"Keywords: {', '.join(keys)}"
    updated = insert_keywords_block(txt, line)
    fp.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    changed = 0
    for fp in sorted(RESOURCES_DIR.glob("*.txt")):
        try:
            if process_file(fp):
                changed += 1
                print(f"Updated keywords: {fp.name}")
        except Exception as e:
            print(f"Failed to update {fp}: {e}")
    print(f"Done. Updated {changed} files.")


if __name__ == "__main__":
    if not RESOURCES_DIR.exists():
        print(f"Resources directory not found: {RESOURCES_DIR}")
        sys.exit(1)
    main() 