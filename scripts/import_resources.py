import argparse
import re
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

RESOURCES_DIR = Path("data/resources")
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

HEADER_TMPL = (
    "Type: {type}\n"
    "Title: {title}\n"
    "URL: {url}\n"
    "Beskrivelse: {description}\n"
)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\-_\s]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text[:120] or "resource"


def fetch(url: str, timeout: int = 20) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
        })
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None


def extract_metadata(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    def meta(name: str) -> Optional[str]:
        tag = soup.find("meta", attrs={"name": name})
        return tag.get("content") if tag and tag.get("content") else None

    def og(name: str) -> Optional[str]:
        tag = soup.find("meta", attrs={"property": f"og:{name}"})
        return tag.get("content") if tag and tag.get("content") else None

    title = og("title") or (soup.title.string.strip() if soup.title and soup.title.string else None) or url
    description = og("description") or meta("description") or ""
    og_type = og("type") or ""

    detected_type = detect_type(url, title, og_type)

    return {
        "title": title.strip(),
        "description": description.strip(),
        "type": detected_type,
        "url": url,
    }


def detect_type(url: str, title: str, og_type: str) -> str:
    u = url.lower()
    t = (title or "").lower()
    if "webinar" in t or "webinar" in u:
        return "Webinar"
    if og_type.startswith("video") or any(x in u for x in ["youtube.com", "vimeo.com"]):
        return "Video"
    if any(x in t for x in ["case", "kundecase", "case study"]) or "case" in u:
        return "Case"
    return "Artikel"


def write_resource(meta: dict) -> Path:
    filename = f"{meta['type']}-{slugify(meta['title'])}.txt"
    path = RESOURCES_DIR / filename
    body = HEADER_TMPL.format(
        type=meta["type"],
        title=meta["title"],
        url=meta["url"],
        description=meta["description"] or "",
    )
    path.write_text(body, encoding="utf-8")
    return path


def process_urls(urls: List[str]) -> List[Path]:
    written: List[Path] = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        html = fetch(url)
        if not html:
            continue
        meta = extract_metadata(html, url)
        out = write_resource(meta)
        written.append(out)
        print(f"Saved: {out}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Create resource .txt files from URLs.")
    parser.add_argument("--urls", nargs="*", help="One or more URLs to import")
    parser.add_argument("--file", help="Path to a text file with one URL per line", default=None)
    args = parser.parse_args()

    urls: List[str] = []
    if args.file:
        p = Path(args.file)
        if p.exists():
            urls.extend([line.strip() for line in p.read_text(encoding="utf-8").splitlines()])
    if args.urls:
        urls.extend(args.urls)

    if not urls:
        print("No URLs provided. Use --urls or --file.")
        return

    process_urls(urls)


if __name__ == "__main__":
    main() 