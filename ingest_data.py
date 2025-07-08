# ingest_data.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def load_and_process_documents(directory):
    processed_docs = []
    
    # os.walk går igennem alle mapper og undermapper
    for root, _, files in os.walk(directory):
        # Bestem dokumenttypen baseret på mappenavnet
        doc_type = "knowledge" # Standard er viden
        if "resources" in root.lower(): # Tjek om 'resources' er i mappestien
            doc_type = "resource"
            
        for filename in files:
            if not filename.endswith(".txt"):
                continue # Spring over filer, der ikke er .txt

            filepath = os.path.join(root, filename)
            
            try:
                loader = TextLoader(filepath, encoding="utf-8")
                documents = loader.load()
            except Exception as e:
                print(f"Kunne ikke læse fil {filepath}: {e}")
                continue

            # Split dokumentet i chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)

            # Tilføj den korrekte metadata til hver chunk
            for chunk in chunks:
                if not isinstance(chunk.metadata, dict):
                    chunk.metadata = {}
                chunk.metadata['doc_type'] = doc_type
                chunk.metadata['source'] = filename
            
            processed_docs.extend(chunks)
            print(f"Behandlet {filename} som type: {doc_type}")
            
    return processed_docs

def embed_and_store(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    db = Chroma.from_documents(chunks, embeddings, persist_directory="chromadb")
    print("Database oprettet/overskrevet med ny mappestruktur.")
    return db

if __name__ == "__main__":
    # VIGTIGT: Slet din gamle 'chromadb' mappe før du kører denne
    data_directory = "data"
    all_chunks = load_and_process_documents(data_directory)
    db = embed_and_store(all_chunks)
    print(f"Data indlæst og gemt. I alt {len(all_chunks)} chunks.")