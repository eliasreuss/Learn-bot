# RAG Chatbot

A chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on your company data.

## Setup

1. Place your .txt, .png or .doxc files with company knowledge in the `data/` folder.
2. Add your OpenAI API key to the `.env` file.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ingest your data:
   ```bash
   python ingest_data.py
   ```
5. Start the chatbot:
   ```bash
   streamlit run app.py
   ```

## Important notice
- This chatbot is a prototype/demo. Answers may be inaccurate or incomplete.
- Messages sent to the bot may be saved to help improve the assistant. The simple text log is stored in `user_questions.txt`.

- GitHub: A scheduled workflow (`.github/workflows/keepalive.yml`) performs a daily empty commit to keep website active
  
## How it works
- Uses LangChain to load, split, and embed your documents.
- Stores embeddings in ChromaDB (local vector database).
- When you ask a question, retrieves the most relevant chunks and sends them to GPT for an answer.

## Add new knowledge quickly (atomize documents)
Use the helper script to convert long documents (txt, docx, md, pdf) into small, focused `.txt` chunks that match the existing style.

Examples:
```bash
# Danish input -> write chunks under data/knowledge/danish
python scripts/atomize_documents.py --input ./my_docs --language danish --output-dir data/knowledge/danish

# English input -> write chunks under data/knowledge/english, generate slugs via LLM
python scripts/atomize_documents.py --input ./case_study.docx --language english --output-dir data/knowledge/english --use-llm-titles
```
The app will pick up new files automatically on restart and index them into the vector DB.
