import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

"""
Translate all Danish knowledge .txt files into English and write them to
`data/knowledge/english/` with the same filenames.

Requirements:
- Environment variable OPENAI_API_KEY must be set (e.g., via .env)
- langchain-openai installed

Usage:
  python scripts/translate_to_english.py
"""

load_dotenv()

SOURCE_DIR = Path("data/knowledge/danish")
TARGET_DIR = Path("data/knowledge/english")


def translate_text(text: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    system = (
        "You are a professional translator. Translate the provided Danish text to clear, concise English. "
        "Preserve line breaks, bullets, headings, and any code or UI labels. Do not add explanations."
    )
    # Use a single-turn chat call
    result = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ])
    return result.content.strip()


def main() -> None:
    if not SOURCE_DIR.exists():
        raise SystemExit(f"Source directory not found: {SOURCE_DIR}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(SOURCE_DIR.glob("*.txt"))
    if not txt_files:
        print("No .txt files found to translate.")
        return

    for path in txt_files:
        try:
            content = path.read_text(encoding="utf-8")
            translated = translate_text(content)
            target_path = TARGET_DIR / path.name
            target_path.write_text(translated, encoding="utf-8")
            print(f"Translated: {path.name} -> {target_path}")
        except Exception as e:
            print(f"Failed to translate {path.name}: {e}")


if __name__ == "__main__":
    main() 