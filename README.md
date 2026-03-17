# RAG Chatbot

A chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on your company data.

## Quick start (run locally)

**Prerequisites:** Python 3.11, Node.js, and an [OpenAI API key](https://platform.openai.com/api-keys).

1. **Clone and enter the project**
   ```bash
   git clone <repo-url>
   cd ts-gpt-chatbot
   ```

2. **Set OpenAI API key**
   ```bash
   cp env.example .env
   ```
   Open `.env` and replace `your-openai-api-key` with your actual key (e.g. `sk-proj-...`). Save the file.

3. **Backend** (one terminal window)
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux — on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn chatbot:app --host 0.0.0.0 --port 8000 --reload
   ```
   On first run, the vector database is built (1–2 min). Wait until you see "Ready."

4. **Frontend** (new terminal window)
   ```bash
   cd ui && npm install && npm run dev
   ```

5. **Open** [http://localhost:5173](http://localhost:5173)

---

## Architecture

- **Backend:** FastAPI + LangChain + ChromaDB + OpenAI (gpt-4o)
- **Frontend:** Vite + React + TypeScript + Bootstrap 4.6

## Requirements

- Python 3.11
- Node.js (for frontend)

## Setup

1. Place your `.txt` files with company knowledge in the `data/` folder.
2. Copy `env.example` to `.env` and set `OPENAI_API_KEY=...`
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the backend (builds the vector DB automatically on first run):
   ```bash
   uvicorn chatbot:app --host 0.0.0.0 --port 8000 --reload
   ```
5. Install and run the frontend:
   ```bash
   cd ui && npm install && npm run dev
   ```
6. Open [http://localhost:5173](http://localhost:5173)

7. Optional: To enable Supabase logging, add `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` to `.env`.

## Important notice

- This chatbot is a prototype/demo. Answers may be inaccurate or incomplete.
- Messages sent to the bot may be saved to help improve the assistant. The log is stored in `chromadb/chat_messages.jsonl`.

## Resources and recommendations

- Resource files live under `data/resources/*.txt` and support the following header fields:
  - `Type: Artikel|Video|Webinar|Case`
  - `Title: ...`
  - `URL: ...`
  - `Beskrivelse: ...` or `Description: ...`
  - `Keywords: comma, separated, keywords` (optional but recommended)
- You can also import resources from URLs and add keywords:
  ```bash
  python scripts/import_resources.py --urls "https://inact.io/blog/your-article" --keywords action, checklist, alarm
  ```
- The app augments resource queries with synonyms and indexes resource keywords into embeddings to improve recall.

## Data changes and DB rebuild

- The vector DB is stored in `chromadb/`. The backend keeps a fingerprint of files in `data/` and automatically rebuilds the DB when those files change.

## How it works

- Uses LangChain to load, split, and embed your documents.
- Stores embeddings in ChromaDB (local vector database).
- When you ask a question, retrieves the most relevant chunks and sends them to GPT for an answer.
- Supports Danish and English with automatic language detection.

## Add new knowledge quickly (atomize documents)

Use the helper script to convert long documents (txt, docx, md, pdf) into small, focused `.txt` chunks:

```bash
# Danish input -> write chunks under data/knowledge/danish
python scripts/atomize_documents.py --input ./my_docs --language danish --output-dir data/knowledge/danish

# English input -> write chunks under data/knowledge/english, generate slugs via LLM
python scripts/atomize_documents.py --input ./case_study.docx --language english --output-dir data/knowledge/english --use-llm-titles
```

The backend will pick up new files automatically on restart and index them into the vector DB.
