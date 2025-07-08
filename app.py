# app.py

import sys
import os
import streamlit as st
from dotenv import load_dotenv

# --- ChromaDB/SQLite3 Workaround for Streamlit Cloud ---
# This must be at the very top of the script
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully replaced sqlite3 with pysqlite3 for ChromaDB compatibility.")
except ImportError:
    print("pysqlite3 not found, using default sqlite3. This may cause issues with ChromaDB on Streamlit Cloud.")
# ---------------------------------------------------------

# --- Import necessary LangChain and OpenAI components ---
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

# Load environment variables (e.g., your OPENAI_API_KEY)
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="Inact Learn Chatbot", page_icon="💬")

# --- Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');
    
    html, body, [class*="st-"], [class*="css-"] {
        font-family: 'Poppins', sans-serif;
    }
    .custom-header {
        font-family: 'Poppins', sans-serif;
        color: #304642;
        font-size: 2rem;
        font-weight: 600;
        text-align: center;
        margin-top: 2.5rem;
        margin-bottom: 2.5rem;
    }
    .gertrud-orange { color: #FF5A00; }
    </style>
    """, unsafe_allow_html=True)


# --- Data Ingestion and Vector Store Logic (from ingest_data.py) ---
# This function will be called by get_vectorstore() if the database doesn't exist.
def load_and_process_documents(directory="data"):
    processed_docs = []
    
    for root, _, files in os.walk(directory):
        doc_type = "knowledge"
        if "resources" in root.lower():
            doc_type = "resource"
            
        for filename in files:
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(root, filename)
            
            try:
                loader = TextLoader(filepath, encoding="utf-8")
                documents = loader.load()
                
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(documents)

                for chunk in chunks:
                    if not isinstance(chunk.metadata, dict):
                        chunk.metadata = {}
                    chunk.metadata['doc_type'] = doc_type
                    chunk.metadata['source'] = filename
                
                processed_docs.extend(chunks)
                print(f"Processed {filename} as type: {doc_type}")
                
            except Exception as e:
                print(f"Could not read file {filepath}: {e}")
                continue
            
    return processed_docs

# --- Caching Resources (LLM and Vector Store) ---
DB_DIR = "chromadb_streamlit"

@st.cache_resource
def get_vectorstore():
    """
    Initializes the vector store. If it doesn't exist, it creates it 
    by processing documents from the 'data' folder.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    
    if os.path.exists(DB_DIR):
        # Load the existing database
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        print("Loaded existing ChromaDB database.")
    else:
        # Create the database if it doesn't exist
        with st.spinner("Første opstart: Opretter og indekserer databasen. Dette kan tage et øjeblik..."):
            all_chunks = load_and_process_documents()
            if not all_chunks:
                st.error("Ingen dokumenter fundet i 'data' mappen. Sørg for at dine .txt filer er i repository'et.")
                return None
            
            db = Chroma.from_documents(
                documents=all_chunks, 
                embedding=embeddings, 
                persist_directory=DB_DIR
            )
            print("Created and persisted new ChromaDB database.")
        st.success("Database oprettet. Du kan nu chatte med din data.")
        
    return db

@st.cache_resource
def get_llm():
    return ChatOpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-3.5-turbo")


# --- Main Application Logic ---

# Initialize Chat Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    "Du er en AI-assistent for Inact. Dit mål er at levere præcise svar og yderst relevante anbefalinger.\n\n"
    "### Opgave 1: Svar på spørgsmålet\n"
    "Brug 'VIDENSBASE KONTEKST' til at formulere et grundigt og klart svar på brugerens spørgsmål. Hvis der er en trin-for-trin guide, skal den præsenteres som en nummereret liste.\n\n"
    "### Opgave 2: Vurder og Anbefal Læringsressourcer\n"
    "Du har modtaget en liste af *potentielle* læringsressourcer i 'LÆRINGSRESSOURCE KONTEKST'. Din opgave er at agere som et intelligent filter.\n"
    "1.  **Vurder Relevans:** Gennemgå HVER ressource i 'LÆRINGSRESSOURCE KONTEKST' og vurder, om den er direkte relevant for brugerens specifikke spørgsmål.\n"
    "2.  **Lav Anbefalinger:** Hvis du finder en eller flere relevante ressourcer, SKAL du tilføje en sektion til sidst i dit svar med overskriften: '**Anbefalet Læring:**'\n"
    "3.  **Formater Kun De Relevante:** For KUN de ressourcer, du har vurderet som relevante, skal du formatere dem som et punkt med et link, f.eks.: `* Se vores video: [Titel på video](URL)`.\n"
    "4.  **Hvis Intet er Relevant:** Hvis INGEN af ressourcerne i 'LÆRINGSRESSOURCE KONTEKST' er relevante for spørgsmålet, skal du IKKE tilføje 'Anbefalet Læring'-sektionen.\n\n"
    "---"
    "\n\n**LÆRINGSRESSOURCE KONTEKST (Potentielle kandidater til anbefaling):**\n{resource_context}\n\n"
    "**VIDENSBASE KONTEKST (Til at formulere svar):**\n{knowledge_context}\n\n"
    "**Spørgsmål:** {question}\n\n"
    "**Svar:**"
)

# --- Initialization & Page Rendering ---

# Load resources
db = get_vectorstore()
llm = get_llm()

# Exit if database creation failed
if db is None:
    st.stop()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Page Header
st.markdown(
    """
    <div class="custom-header">Hello, <span class="gertrud-orange">Gertrud</span>. How can I help you today?</div>
    """,
    unsafe_allow_html=True
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_question := st.chat_input("How do I create an Insight in Inact Now?"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Tænker...") 

        # Retrieve relevant documents from both resource and knowledge types
        resource_retriever = db.as_retriever(search_kwargs={"k": 5, "filter": {"doc_type": "resource"}})
        resource_docs = resource_retriever.get_relevant_documents(user_question)
        resource_context = "\n\n---\n\n".join([doc.page_content for doc in resource_docs])

        knowledge_retriever = db.as_retriever(search_kwargs={"k": 3, "filter": {"doc_type": "knowledge"}})
        knowledge_docs = knowledge_retriever.get_relevant_documents(user_question)
        knowledge_context = "\n\n---\n\n".join([doc.page_content for doc in knowledge_docs])
        
        # Create and invoke the chain
        chain = prompt_template | llm
        response = chain.invoke({
            "resource_context": resource_context,
            "knowledge_context": knowledge_context,
            "question": user_question
        })
        answer = response.content
        
        # Display the answer and save it to session state
        message_placeholder.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})