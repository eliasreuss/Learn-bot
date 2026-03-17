# RAG Chatbot

A chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on your company data.

## Architecture

- **Backend:** FastAPI + LangChain + ChromaDB + OpenAI (gpt-4o)
- **Frontend:** Vite + React + TypeScript + Bootstrap 4.6

---

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
   On first run, the vector database is built from `data/` (1–2 min). Wait until you see "Ready."

4. **Frontend** (new terminal window)
   ```bash
   cd ui && npm install && npm run dev
   ```

5. **Open** [http://localhost:5173](http://localhost:5173)

6. Optional: To enable Supabase logging, add `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` to `.env`.

---

## Important notice

- This chatbot is a prototype/demo. Answers may be inaccurate or incomplete.
- Messages sent to the bot may be saved to help improve the assistant. The log is stored in `chromadb/chat_messages.jsonl`.

---

## How it works

- Uses LangChain to load, split, and embed your `.txt` documents from `data/`.
- Stores embeddings in a local ChromaDB vector database.
- On each question, retrieves the most relevant chunks and sends them to GPT-4o for an answer.
- Supports Danish and English with automatic language detection.
- The vector DB is rebuilt automatically when files in `data/` change (fingerprint check).

---

## Resources

Resource files live under `data/resources/*.txt` and support the following header fields:

```
Type: Artikel|Video|Webinar|Case
Title: ...
URL: ...
Description: ...
Keywords: comma, separated, keywords
```

You can import resources from URLs:
```bash
python scripts/import_resources.py --urls "https://inact.io/blog/your-article" --keywords action, checklist, alarm
```

---

## Add new knowledge (atomize documents)

Convert long documents (txt, docx, md, pdf) into small focused `.txt` chunks:

```bash
# Danish
python scripts/atomize_documents.py --input ./my_docs --language danish --output-dir data/knowledge/danish

# English (with LLM-generated filenames)
python scripts/atomize_documents.py --input ./case_study.docx --language english --output-dir data/knowledge/english --use-llm-titles
```

The backend picks up new files automatically on restart.
