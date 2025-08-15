# app.py

import sys
import os
# Force Chroma to use DuckDB+Parquet backend (works on Streamlit Cloud without modern sqlite)
os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")
import streamlit as st
from dotenv import load_dotenv
import logging
import uuid
from typing import Optional
import hashlib
from typing import List
import requests
import json
from datetime import datetime

# Prefer region-specific ingest URL if provided (copy from Better Stack "Ingesting host").
BETTERSTACK_INGEST_URL = os.environ.get("BETTERSTACK_INGEST_URL") or os.environ.get("LOGTAIL_ENDPOINT") or "https://in.logs.betterstack.com"

def send_log_manually(message: str = None, **fields):
    """Send a log directly to Better Stack via HTTP.
    Uses `BETTERSTACK_INGEST_URL`/`LOGTAIL_ENDPOINT` if set; otherwise falls back to the generic endpoint.
    Accepts arbitrary keyword fields to enable structured filtering in Better Stack.
    """
    token = os.environ.get("LOGTAIL_TOKEN")
    if not token:
        print("MANUAL LOG FAIL: No LOGTAIL_TOKEN found.")
        return

    url = BETTERSTACK_INGEST_URL

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "dt": datetime.utcnow().isoformat() + "Z",
        "message": message or "",
    }
    # Merge structured fields for easier querying (role, event, user_id, etc.)
    payload.update({k: v for k, v in fields.items() if v is not None})

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code // 100 != 2:
            print(f"MANUAL LOG ERROR: {response.status_code} - {response.text}")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"MANUAL LOG FAILED: Could not send log to Better Stack. Error: {e}")


def log_chat_message(role: str, user_id: str, text: str, event: str):
    """Convenience wrapper to emit a structured chat log entry."""
    send_log_manually(message=text, role=role, user_id=user_id, event=event, app="inact-bot")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "user_questions.log")

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
from langchain_chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain.prompts import ChatPromptTemplate
try:
    from chromadb.config import Settings
except Exception:
    Settings = None
import re

# Load environment variables (e.g., your OPENAI_API_KEY)
load_dotenv()

# --- Logging Configuration ---
# Create a custom logger
logger = logging.getLogger("user_questions")
logger.setLevel(logging.INFO)

# --- Better Stack Logging ---
# We will use our manual function instead of the handler for now.
# LOGTAIL_TOKEN = os.environ.get("LOGTAIL_TOKEN")
# if LOGTAIL_TOKEN:
#     handler = LogtailHandler(source_token=LOGTAIL_TOKEN)
#     logger.addHandler(handler)
# else:
# Fallback to local file logging if the token is not set
if not logger.handlers:
    f_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    f_format = logging.Formatter("%(asctime)s - %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

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
        
        # Detect language based on subfolder for knowledge
        language = None
        root_lower = root.lower()
        if "knowledge" in root_lower:
            if "danish" in root_lower:
                language = "danish"
            elif "english" in root_lower:
                language = "english"
        
        for filename in files:
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(root, filename)
            
            try:
                loader = TextLoader(filepath, encoding="utf-8")
                documents = loader.load()
                
                # For each document, first detect curated header format.
                curated_handled_any = False
                for doc in documents:
                    text = doc.page_content
                    # Curated knowledge with Topic/Source/Keywords
                    topic_m = re.search(r"^\s*Topic:\s*(.+)$", text, flags=re.I | re.M)
                    source_m = re.search(r"^\s*Source article:\s*(.+)$", text, flags=re.I | re.M)
                    keywords_m = re.search(r"^\s*Keywords:\s*(.+)$", text, flags=re.I | re.M)
                    # Resource header format
                    r_type_m = re.search(r"^\s*Type:\s*(.+)$", text, flags=re.I | re.M)
                    r_title_m = re.search(r"^\s*Title:\s*(.+)$", text, flags=re.I | re.M)
                    r_url_m = re.search(r"^\s*URL:\s*(.+)$", text, flags=re.I | re.M)
                    r_desc_m = re.search(r"^\s*(Description|Beskrivelse):\s*(.+)$", text, flags=re.I | re.M)
                    r_keywords_m = re.search(r"^\s*Keywords?:\s*(.+)$", text, flags=re.I | re.M)

                    if topic_m and source_m and keywords_m and doc_type == "knowledge":
                        # Strip header lines to keep only content
                        body = re.sub(r"^\s*Topic:.*\n\s*Source article:.*\n\s*Keywords:.*\n\n?", "", text, flags=re.I)
                        curated_doc = doc
                        curated_doc.page_content = body.strip()
                        if not isinstance(curated_doc.metadata, dict):
                            curated_doc.metadata = {}
                        curated_doc.metadata['doc_type'] = doc_type
                        curated_doc.metadata['source'] = filename
                        if language:
                            curated_doc.metadata['language'] = language
                        curated_doc.metadata['topic'] = topic_m.group(1).strip()
                        curated_doc.metadata['article_source'] = source_m.group(1).strip()
                        curated_keywords_list = [k.strip() for k in keywords_m.group(1).split(',') if k.strip()]
                        curated_doc.metadata['keywords'] = ", ".join(curated_keywords_list)
                        processed_docs.append(curated_doc)
                        curated_handled_any = True
                    elif r_type_m and r_title_m and r_url_m and doc_type == "resource":
                        # Resources: keep whole text as content for snippet, attach metadata
                        if not isinstance(doc.metadata, dict):
                            doc.metadata = {}
                        doc.metadata['doc_type'] = doc_type
                        doc.metadata['resource_type'] = r_type_m.group(1).strip()
                        doc.metadata['title'] = r_title_m.group(1).strip()
                        doc.metadata['url'] = r_url_m.group(1).strip()
                        if r_desc_m:
                            doc.metadata['description'] = r_desc_m.group(2).strip()
                        if r_keywords_m:
                            # Normalize and store parsed keywords for resources
                            kw_list = [k.strip() for k in r_keywords_m.group(1).split(',') if k.strip()]
                            if kw_list:
                                doc.metadata['keywords'] = ", ".join(kw_list)
                                # Ensure keywords influence embeddings by appending to content
                                doc.page_content = f"{doc.metadata['title']}\n{doc.metadata.get('description', '')}\nKeywords: {', '.join(kw_list)}".strip()
                        else:
                            # At least index title and description so retrieval works better
                            doc.page_content = f"{doc.metadata['title']}\n{doc.metadata.get('description', '')}".strip()
                        processed_docs.append(doc)
                        curated_handled_any = True
                
                if curated_handled_any:
                    print(f"Processed {filename} as curated {doc_type} language: {language or 'n/a'}")
                else:
                    # Fallback to standard splitting for non-curated files
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = text_splitter.split_documents(documents)

                    for chunk in chunks:
                        if not isinstance(chunk.metadata, dict):
                            chunk.metadata = {}
                        chunk.metadata['doc_type'] = doc_type
                        chunk.metadata['source'] = filename
                        if language:
                            chunk.metadata['language'] = language
                    
                    processed_docs.extend(chunks)
                    print(f"Processed {filename} as type: {doc_type} language: {language or 'n/a'}")
                
            except Exception as e:
                print(f"Could not read file {filepath}: {e}")
                continue
            
    return processed_docs

# --- Caching Resources (LLM and Vector Store) ---
DB_DIR = "chromadb_streamlit"
FINGERPRINT_FILE = os.path.join(DB_DIR, ".source_fingerprint")
CHROMA_SETTINGS = Settings(chroma_db_impl="duckdb+parquet", persist_directory=DB_DIR) if Settings else None
FAISS_DIR = "faiss_index"

def compute_data_fingerprint(directory: str = "data") -> str:
    hasher = hashlib.md5()
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            if not name.endswith('.txt'):
                continue
            path = os.path.join(root, name)
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue
            hasher.update(path.encode('utf-8'))
            hasher.update(str(stat.st_mtime_ns).encode('utf-8'))
            hasher.update(str(stat.st_size).encode('utf-8'))
    return hasher.hexdigest()

@st.cache_resource
def get_vectorstore():
    """
    Initializes the vector store. If it doesn't exist, it creates it 
    by processing documents from the 'data' folder.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    current_fp = compute_data_fingerprint()
    
    try:
        if os.path.exists(DB_DIR) and os.path.exists(FINGERPRINT_FILE):
            try:
                saved_fp = open(FINGERPRINT_FILE, 'r', encoding='utf-8').read().strip()
            except Exception:
                saved_fp = ""
            if saved_fp == current_fp:
                # Load the existing database
                db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings, client_settings=CHROMA_SETTINGS) if CHROMA_SETTINGS else Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                print("Loaded existing ChromaDB database.")
                return db
            else:
                print("Source data changed. Rebuilding ChromaDB...")
                try:
                    import shutil
                    shutil.rmtree(DB_DIR)
                except Exception as e:
                    print(f"Failed to remove existing DB dir: {e}")
        # Create the database if it doesn't exist or if source changed
        with st.spinner("Første opstart: Opretter og indekserer databasen. Dette kan tage et øjeblik..."):
            all_chunks = load_and_process_documents()
            if not all_chunks:
                st.error("Ingen dokumenter fundet i 'data' mappen. Sørg for at dine .txt filer er i repository'et.")
                return None
            
            db = Chroma.from_documents(
                documents=all_chunks, 
                embedding=embeddings, 
                persist_directory=DB_DIR,
                client_settings=CHROMA_SETTINGS
            )
            try:
                os.makedirs(DB_DIR, exist_ok=True)
                with open(FINGERPRINT_FILE, 'w', encoding='utf-8') as f:
                    f.write(current_fp)
            except Exception as e:
                print(f"Failed writing fingerprint file: {e}")
            print("Created and persisted new ChromaDB database.")
        return db
    except Exception as e:
        print(f"Chroma unavailable, falling back to FAISS: {e}")
        # FAISS fallback (no sqlite requirement)
        try:
            if os.path.exists(FAISS_DIR):
                db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                print("Loaded existing FAISS index.")
                return db
        except Exception as e2:
            print(f"Failed loading FAISS index: {e2}")
        with st.spinner("Indekserer dokumenter i FAISS (første opstart)..."):
            all_chunks = load_and_process_documents()
            if not all_chunks:
                st.error("Ingen dokumenter fundet i 'data' mappen. Sørg for at dine .txt filer er i repository'et.")
                return None
            try:
                db = FAISS.from_documents(all_chunks, embeddings)
                db.save_local(FAISS_DIR)
                print("Created and persisted new FAISS index.")
                return db
            except Exception as e3:
                print(f"FAISS creation failed, falling back to in-memory DocArray: {e3}")
                db = DocArrayInMemorySearch.from_documents(all_chunks, embeddings)
                print("Using DocArrayInMemorySearch (not persisted).")
                return db

@st.cache_resource
def get_llm():
    return ChatOpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-3.5-turbo")

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]) 

# --- Language Detection and Prompt Templates ---

@st.cache_data
def detect_language(text: str) -> str:
    """
    Uses the LLM to detect if the text is English or Danish.
    Defaults to Danish for ambiguous cases.
    """
    try:
        llm = get_llm()
        prompt = (
            "You are a language detection expert. "
            "Is the following text primarily in English or Danish? "
            "Answer with a single word: 'english' or 'danish'.\n\n"
            f"Text: '{text}'"
        )
        response = llm.invoke(prompt)
        result = response.content.strip().lower()
        if "english" in result:
            return "english"
    except Exception as e:
        print(f"Language detection failed: {e}")
    # Default to Danish if detection fails or is unclear
    return "danish"


def is_greeting_or_smalltalk(text: str) -> bool:
    """Return True if the text looks like a short greeting/small-talk, not a real question."""
    if not text:
        return False
    normalized = re.sub(r"[^a-zA-ZæøåÆØÅ\s]", "", text).strip().lower()
    if len(normalized) <= 25 and any(w in normalized for w in [
        "hi", "hello", "hey", "hi there", "hey there", "good morning", "good evening",
        "hej", "hej med dig", "hejsa", "godmorgen", "godaften", "god aften", "hej hej"
    ]):
        return True
    return False

def augment_question_for_resources(question: str, language: str) -> str:
    """Augment the resource retrieval query with domain synonyms to improve recall."""
    q = question or ""
    normalized = q.lower()
    synonyms_map_danish = {
        "action": ["actions", "alarm", "alarmer", "checklist", "create checklist", "opret action", "lav action"],
        "dashboard": ["dashboards", "dashboard creator", "widget", "widgets", "overblik", "visualisering"],
        "insight": ["insights", "analyse", "rapport", "rapporter", "view", "master data"],
    }
    synonyms_map_english = {
        "action": ["actions", "alert", "alerts", "checklist", "create checklist", "create action"],
        "dashboard": ["dashboards", "widgets", "charts", "visualization"],
        "insight": ["insights", "analysis", "report", "reports", "view"],
    }
    mapping = synonyms_map_danish if language == "danish" else synonyms_map_english
    additions: List[str] = []
    for key, syns in mapping.items():
        if key in normalized or any(s in normalized for s in syns):
            additions.extend(syns + [key])
    if not additions:
        return q
    # Deduplicate while preserving order
    seen = set()
    extras = " ".join([s for s in additions if not (s in seen or seen.add(s))])
    return f"{q} {extras}".strip()


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        na += a * a
        nb += b * b
    if na == 0.0 or nb == 0.0:
        return 0.0
    import math
    return dot / (math.sqrt(na) * math.sqrt(nb))


def is_semantically_related(new_question: str, last_answer: str, threshold: float = 0.50) -> bool:
    """Return True if new_question appears related to last_answer using embedding cosine similarity."""
    try:
        emb = get_embeddings()
        q_vec = emb.embed_query(new_question[:1000])
        a_vec = emb.embed_query((last_answer or "")[:1200])
        sim = cosine_similarity(q_vec, a_vec)
        return sim >= threshold
    except Exception as e:
        print(f"relatedness check failed: {e}")
        return False


def is_affirmative_reply(text: str) -> bool:
    """Heuristic: detect short affirmative/consent replies in Danish/English."""
    if not text:
        return False
    t = re.sub(r"\s+", " ", text.strip().lower())
    single_word = t.split()
    if len(t) <= 30:
        affirm = {
            # English
            "yes", "yes.", "yes!", "y", "yep", "yup", "sure", "ok", "okay", "please",
            "yes please", "go ahead", "do it", "continue", "sounds good", "great",
            # Danish
            "ja", "ja.", "ja!", "jep", "ok", "okay", "gerne", "ja tak", "meget gerne",
            "kør", "fortsæt", "gør det", "lyder godt",
        }
        if t in affirm:
            return True
        # Short patterns starting with yes/ja/ok/sure/gerne
        starts = ("yes", "ok", "okay", "sure", "please", "ja", "gerne")
        if any(t.startswith(s) for s in starts):
            return True
    return False


def parse_step_followup(text: str) -> Optional[int]:
    """Heuristic to find mentions of a step number in a short follow-up."""
    t = text.strip().lower()
    if len(t) > 60:  # Too long for a simple follow-up
        return None

    # Check for "step 4", "punkt 2", "nr. 3" etc.
    match = re.search(r"(?:step|punkt|item|number|nr\.?|trin)\s*(\d+)", t)
    if match:
        return int(match.group(1))

    # Check for number words if it's an affirmative reply
    if is_affirmative_reply(text):
        words_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "first": 1,
            "second": 2, "third": 3, "fourth": 4, "fifth": 5, "en": 1, "to": 2,
            "tre": 3, "fire": 4, "fem": 5, "første": 1, "anden": 2, "tredje": 3,
            "fjerde": 4, "femte": 5,
        }
        for word, num in words_to_num.items():
            if word in t:
                return num
        # Check for just a digit "4"
        numbers = re.findall(r"\d+", t)
        if len(numbers) == 1:
            return int(numbers[0])

    return None


def extract_step_content(answer_text: str, step_number: int) -> Optional[str]:
    """Extracts the text of a specific numbered step from a string."""
    if not answer_text:
        return None
    # This regex finds a number at the start of a line, followed by a dot.
    matches = re.findall(r"^\s*(\d+)\.\s+(.*)", answer_text, re.MULTILINE)
    for num_str, content in matches:
        if int(num_str) == step_number:
            return content.strip()
    return None


def extract_followup_question(text: str) -> Optional[str]:
    """Extract the last question sentence that looks like an offered follow-up."""
    if not text:
        return None
    # Split by question marks and keep the last clause containing '?'
    # We'll scan for typical openings
    openings = [
        "skal jeg", "vil du have", "skal jeg anbefale",  # Danish
        "would you like", "shall i", "do you want me"     # English
    ]
    candidates = []
    for part in re.split(r"(\?)+", text):
        pass
    # Simpler: find all sentences ending with '?'
    qs = re.findall(r"([^\n\r\?]{3,}?\?)", text, flags=re.IGNORECASE | re.DOTALL)
    for q in qs:
        q_clean = re.sub(r"\s+", " ", q.strip().lower())
        if any(q_clean.startswith(o) for o in openings):
            candidates.append(q.strip())
    if candidates:
        return candidates[-1]
    # Fallback to last question sentence
    if qs:
        return qs[-1].strip()
    return None

def build_recent_history(max_pairs: int = 1, char_limit: int = 1200) -> str:
    """Return a compact string of the last max_pairs user/assistant turns before the current input."""
    messages = st.session_state.get("messages", [])
    if not messages:
        return ""
    # Exclude the just-submitted user message if present at the end
    prior = messages[:-1] if messages and messages[-1].get("role") == "user" else messages
    if not prior:
        return ""
    recent = prior[-(2 * max_pairs):]
    lines = []
    for m in recent:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "").strip().replace("\n\n", "\n")
        lines.append(f"{role}: {content}")
    joined = "\n".join(lines)
    if len(joined) > char_limit:
        joined = joined[-char_limit:]
    return joined


PROMPT_TEMPLATES = {
    "danish": ChatPromptTemplate.from_template(
        "Du er en AI-assistent for brugere og kunder af Inact (Inact Now).\n"
        "Du hjælper både med tekniske spørgsmål om, hvordan man bruger Inact Now, og med mere teoretiske/situationelle spørgsmål om fx lagerstyring, dataanalyse, arbejdsmåder og kompleksitetsledelse.\n"
        "Skriv pragmatisk, venligt og uden marketing. Kombinér teknisk vejledning i Inact Now med relevant teori og best practice, når det giver mening. Brug viden fra 'VIDENSBASE KONTEKST' og vær specifik.\n\n"
        "Formatér dit svar sådan (skriv IKKE overskrifter eller tal som '1) Svar', 'Anbefalet Læring' eller 'Fortsæt/Spørgsmål' i selve svaret):\n"
        "- Start med et direkte, handlingsorienteret svar. Hvis relevant, korte, nummererede trin i Inact Now og nævn konkrete funktioner.\n"
        "- Hvis relevant, tilføj en kort uddybning, der forbinder til principper, trade-offs og faldgruber.\n"
        "- Respekter altid brugerens spørgsmålstype. Hvis spørgsmålet er 'hvorfor', så fokuser på formål/fordele/effekt (ikke trin). Hvis det er 'hvordan', så giv trin. Hvis det er 'hvad', så definer/beskriv.\n"
        "- Brug **LÆRINGSRESSOURCE KONTEKST** som baggrundsviden, men lad være med at liste eller linke til konkrete ressourcer. Applikationen tilføjer dem efter svaret.\n"
        "- Afslut med præcis ét kort opfølgende spørgsmål (kun hvis relevant), formuleret som et tilbud om at hjælpe videre. Brug formuleringer som: 'Skal jeg …', 'Vil du have, at jeg …' eller 'Skal jeg anbefale …?'. Ingen overskrift.\n\n"
        "(Tidligere udveksling – kun som kontekst, ignorer hvis ikke relevant)\n{history_context}\n\n"
        "---\n\n"
        "**LÆRINGSRESSOURCE KONTEKST (Potentielle kandidater):**\n{resource_context}\n\n"
        "**VIDENSBASE KONTEKST (Til at formulere svar):**\n{knowledge_context}\n\n"
        "**Spørgsmål:** {question}\n\n"
        "**Svar:**"
    ),
    "english": ChatPromptTemplate.from_template(
        "You are an AI assistant for users and customers of Inact (Inact Now).\n"
        "You help with technical 'how to use Inact Now' questions and with more theoretical/situated topics such as inventory management, analyzing your company, ways of working with data, and complexity management.\n"
        "Write pragmatically, kindly, and without marketing. Combine technical guidance in Inact Now with relevant theory and best practices when helpful. Ground recommendations in the 'KNOWLEDGE BASE CONTEXT' and be specific.\n\n"
        "Format your output like this (do NOT print headings or labels such as '1) Answer', 'Recommended Learning', or 'Continue/Question' in the answer):\n"
        "- Begin with a direct, actionable answer. Where relevant, include short, numbered steps in Inact Now and name specific features.\n"
        "- If helpful, add a brief paragraph connecting to principles, trade-offs, and common pitfalls.\n"
        "- Always match the user's intent. If the question is 'why', lead with purpose/benefits/outcomes (not steps). If it's 'how', give steps. If it's 'what', define/describe.\n"
        "- Use **LEARNING RESOURCE CONTEXT** only as background. Do not list or link any resources; the application will append them after your answer.\n"
        "- End with exactly one short follow-up question (only if relevant), phrased as an offer to help with the next step. Use openings like: 'Would you like me to …', 'Shall I …', or 'Do you want me to recommend …?'. No heading.\n\n"
        "(Recent exchange – context only; ignore if not relevant)\n{history_context}\n\n"
        "---\n\n"
        "**LEARNING RESOURCE CONTEXT (Potential candidates):**\n{resource_context}\n\n"
        "**KNOWLEDGE BASE CONTEXT (To formulate the answer):**\n{knowledge_context}\n\n"
        "**Question:** {question}\n\n"
        "**Answer:**"
    ),
}


def render_resources_section(resource_docs, language: str) -> str:
    items = []
    seen = set()
    for doc in resource_docs:
        meta = getattr(doc, "metadata", {}) or {}
        title = meta.get("title") or meta.get("source") or "Resource"
        url = meta.get("url")
        if not url:
            continue
        key = (title.strip(), url.strip())
        if key in seen:
            continue
        seen.add(key)
        items.append(f"- [{title}]({url})")
        if len(items) >= 3:
            break
    if not items:
        return ""
    header = (
        "Her er nogle relevante læringsressourcer, der muligvis kan hjælpe:" if language == "danish" 
        else "Here are some relevant learning resources that might help:"
    )
    return header + "\n" + "\n".join(items)

# --- Main Application Logic ---

# Initialize Chat Prompt Template
# This will be replaced dynamically based on language detection
prompt_template = PROMPT_TEMPLATES["danish"] 

# --- Initialization & Page Rendering ---

# Load resources
db = get_vectorstore()
llm = get_llm()

# Exit if database creation failed
if db is None:
    st.stop()

# Initialize session state for messages and user ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:4]
if "last_assistant_answer" not in st.session_state:
    st.session_state.last_assistant_answer = None

# If no history yet, show a disclaimer as the first assistant message
if not st.session_state.messages:
    disclaimer = (
        "Note: This chatbot is a prototype/demo. Answers may be inaccurate or incomplete. "
        "Messages you send may be saved to help improve the assistant."
    )
    st.session_state.messages.append({"role": "assistant", "content": disclaimer})
    log_chat_message("assistant", st.session_state.user_id, disclaimer, event="system")

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
    log_chat_message("user", st.session_state.user_id, user_question, event="question")
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("thinking...")

        # 1. Detect language and select prompt
        detected_language = detect_language(user_question)
        prompt_template = PROMPT_TEMPLATES[detected_language]
        print(f"Detected language: {detected_language}")

        # If it's a greeting/small talk, respond briefly without retrieval
        if is_greeting_or_smalltalk(user_question):
            if detected_language == "danish":
                answer = (
                    "Hej! Hvordan kan jeg hjælpe dig i Inact Now i dag? Skriv gerne den funktion eller opgave, du arbejder med."
                )
            else:
                answer = (
                    "Hi! How can I help you in Inact Now today? Tell me the feature or task you're working on."
                )
            st.session_state.last_assistant_answer = None
            message_placeholder.markdown(answer)
            log_chat_message("assistant", st.session_state.user_id, answer, event="answer")
        else:
            # Determine if this question relates to the last assistant answer
            last_answer = st.session_state.get("last_assistant_answer")
            use_history = False
            if last_answer:
                use_history = is_semantically_related(user_question, last_answer, threshold=0.50)

            # Build a small recent-history window only if related
            history_context = build_recent_history(max_pairs=1, char_limit=1200) if use_history else ""

            # Create an effective question; only apply follow-up logic if related
            effective_question = user_question
            if use_history:
                step_num_match = parse_step_followup(user_question)
                step_content = None
                if last_answer and step_num_match:
                    step_content = extract_step_content(last_answer, step_num_match)

                if step_content:
                    # User is following up on a specific step
                    if detected_language == "danish":
                        effective_question = f"Brugeren spørger ind til trin {step_num_match}: '{step_content}'. Uddyb venligst dette trin i detaljer, og forklar, hvordan det udføres i Inact Now."
                    else:
                        effective_question = f"The user is asking for details on step {step_num_match}: '{step_content}'. Please elaborate on this specific step, explaining how to perform it in Inact Now."
                elif last_answer and is_affirmative_reply(user_question):
                    # User gives a general affirmative reply, check for a follow-up question
                    follow_up_q = extract_followup_question(last_answer)
                    if follow_up_q:
                        if detected_language == "danish":
                            effective_question = f"Brugeren bekræfter opfølgning: '{follow_up_q}'. Fortsæt og leverer det efterspurgte i detaljer."
                        else:
                            effective_question = f"The user confirmed the follow-up: '{follow_up_q}'. Please proceed to deliver what was offered in detail."

            # Augment question for resource retrieval if it's a resource-related query
            if "resource" in effective_question.lower():
                effective_question = augment_question_for_resources(effective_question, detected_language)

            # 2. Build filters based on detected language
            resource_filter = {"doc_type": "resource"}
            knowledge_filter = {
                "$and": [
                    {"doc_type": "knowledge"},
                    {"language": detected_language},
                ]
            }

            # 3. Retrieve relevant documents (gate recommendations by score threshold)
            try:
                resource_retriever = db.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 5, "filter": resource_filter, "score_threshold": 0.4},
                )
            except Exception:
                # fall back if threshold search is unavailable
                resource_retriever = db.as_retriever(search_kwargs={"k": 5, "filter": resource_filter})

            resource_docs = resource_retriever.get_relevant_documents(effective_question)
            resource_context = "\n\n---\n\n".join([doc.page_content for doc in resource_docs])

            knowledge_retriever = db.as_retriever(search_kwargs={"k": 3, "filter": knowledge_filter})
            knowledge_docs = knowledge_retriever.get_relevant_documents(effective_question)
            knowledge_context = "\n\n---\n\n".join([doc.page_content for doc in knowledge_docs])

            # If nothing relevant is found, ask a clarifying question instead of generating a long answer
            if not knowledge_docs and not resource_docs:
                if detected_language == "danish":
                    answer = (
                        "Jeg kan hjælpe med Inact Now. Hvilken funktion eller opgave vil du gerne løse?"
                    )
                else:
                    answer = (
                        "I can help with Inact Now. Which feature or task would you like help with?"
                    )
                st.session_state.last_assistant_answer = None
                message_placeholder.markdown(answer)
                log_chat_message("assistant", st.session_state.user_id, answer, event="answer")
            else:
                # 4. Create and invoke the chain
                chain = prompt_template | llm
                response = chain.invoke({
                    "resource_context": resource_context,
                    "knowledge_context": knowledge_context,
                    "question": effective_question,
                    "history_context": history_context,
                })
                answer = response.content
                # Append curated resources strictly from retrieved docs (no hallucinated links)
                resources_md = render_resources_section(resource_docs, detected_language)
                if resources_md:
                    answer = f"{answer}\n\n{resources_md}"
                # Save last answer for next turn's context
                st.session_state.last_assistant_answer = answer
                # Display the answer and save it to session state
                message_placeholder.markdown(answer)
                log_chat_message("assistant", st.session_state.user_id, answer, event="answer")
        
    st.session_state.messages.append({"role": "assistant", "content": answer})