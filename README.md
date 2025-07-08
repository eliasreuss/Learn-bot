# RAG Chatbot

A chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on your company data.

## Setup

1. Place your `.txt`, '.png' or '.doxc' files with company knowledge in the `data/` folder.
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
   streamlit run chatbot.py
   ```

## How it works
- Uses LangChain to load, split, and embed your documents.
- Stores embeddings in ChromaDB (local vector database).
- When you ask a question, retrieves the most relevant chunks and sends them to GPT for an answer.
