import os
import re
import sys
import json
import math
import uuid
import hashlib
import logging
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY is not set. Please create a .env file and add your API key.")

# --- ChromaDB / SQLite3 workaround (non-macOS) ---
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chromadb")
FINGERPRINT_FILE = os.path.join(DB_DIR, ".source_fingerprint")
LOG_FILE = os.path.join(DB_DIR, "chat_messages.jsonl")

logger = logging.getLogger("chatbot")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Document loading (ported from app.py)
# ---------------------------------------------------------------------------

def load_and_process_documents(directory: str = "data") -> list:
    processed_docs = []

    for root, _, files in os.walk(directory):
        doc_type = "resource" if "resources" in root.lower() else "knowledge"

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

                curated_handled_any = False
                for doc in documents:
                    text = doc.page_content
                    topic_m = re.search(r"^\s*Topic:\s*(.+)$", text, flags=re.I | re.M)
                    source_m = re.search(r"^\s*Source article:\s*(.+)$", text, flags=re.I | re.M)
                    keywords_m = re.search(r"^\s*Keywords:\s*(.+)$", text, flags=re.I | re.M)
                    r_type_m = re.search(r"^\s*Type:\s*(.+)$", text, flags=re.I | re.M)
                    r_title_m = re.search(r"^\s*Title:\s*(.+)$", text, flags=re.I | re.M)
                    r_url_m = re.search(r"^\s*URL:\s*(.+)$", text, flags=re.I | re.M)
                    r_desc_m = re.search(r"^\s*(Description|Beskrivelse):\s*(.+)$", text, flags=re.I | re.M)
                    r_keywords_m = re.search(r"^\s*Keywords?:\s*(.+)$", text, flags=re.I | re.M)

                    if topic_m and source_m and keywords_m and doc_type == "knowledge":
                        body = re.sub(
                            r"^\s*Topic:.*\n\s*Source article:.*\n\s*Keywords:.*\n\n?",
                            "", text, flags=re.I,
                        )
                        doc.page_content = body.strip()
                        doc.metadata = doc.metadata or {}
                        doc.metadata["doc_type"] = doc_type
                        doc.metadata["source"] = filename
                        if language:
                            doc.metadata["language"] = language
                        doc.metadata["topic"] = topic_m.group(1).strip()
                        doc.metadata["article_source"] = source_m.group(1).strip()
                        doc.metadata["keywords"] = ", ".join(
                            k.strip() for k in keywords_m.group(1).split(",") if k.strip()
                        )
                        processed_docs.append(doc)
                        curated_handled_any = True

                    elif r_type_m and r_title_m and r_url_m and doc_type == "resource":
                        doc.metadata = doc.metadata or {}
                        doc.metadata["doc_type"] = doc_type
                        doc.metadata["resource_type"] = r_type_m.group(1).strip()
                        doc.metadata["title"] = r_title_m.group(1).strip()
                        doc.metadata["url"] = r_url_m.group(1).strip()
                        if r_desc_m:
                            doc.metadata["description"] = r_desc_m.group(2).strip()
                        if r_keywords_m:
                            kw_list = [k.strip() for k in r_keywords_m.group(1).split(",") if k.strip()]
                            doc.metadata["keywords"] = ", ".join(kw_list)
                            doc.page_content = (
                                f"{doc.metadata['title']}\n"
                                f"{doc.metadata.get('description', '')}\n"
                                f"Keywords: {', '.join(kw_list)}"
                            ).strip()
                        else:
                            doc.page_content = (
                                f"{doc.metadata['title']}\n"
                                f"{doc.metadata.get('description', '')}"
                            ).strip()
                        processed_docs.append(doc)
                        curated_handled_any = True

                if not curated_handled_any:
                    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = splitter.split_documents(documents)
                    for chunk in chunks:
                        chunk.metadata = chunk.metadata or {}
                        chunk.metadata["doc_type"] = doc_type
                        chunk.metadata["source"] = filename
                        if language:
                            chunk.metadata["language"] = language
                    processed_docs.extend(chunks)

            except Exception as e:
                logger.warning(f"Could not read {filepath}: {e}")

    return processed_docs


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def compute_data_fingerprint(directory: str = "data") -> str:
    hasher = hashlib.md5()
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            if not name.endswith(".txt"):
                continue
            path = os.path.join(root, name)
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue
            hasher.update(path.encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
    return hasher.hexdigest()


def init_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    current_fp = compute_data_fingerprint()

    # Rebuild if fingerprint file is missing (e.g. DB was built by old ingest_data.py)
    if os.path.exists(DB_DIR) and not os.path.exists(FINGERPRINT_FILE):
        logger.info("Existing chromadb lacks fingerprint (old schema). Rebuilding…")
        try:
            shutil.rmtree(DB_DIR)
        except Exception as e:
            logger.warning(f"Could not remove old DB: {e}")

    if os.path.exists(DB_DIR) and os.path.exists(FINGERPRINT_FILE):
        try:
            saved_fp = open(FINGERPRINT_FILE, "r", encoding="utf-8").read().strip()
        except Exception:
            saved_fp = ""
        if saved_fp == current_fp:
            logger.info("Loading existing ChromaDB.")
            return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        else:
            logger.info("Data changed — rebuilding ChromaDB.")
            try:
                shutil.rmtree(DB_DIR)
            except Exception as e:
                logger.warning(f"Could not remove old DB: {e}")

    logger.info("Building ChromaDB from documents…")
    chunks = load_and_process_documents()
    if not chunks:
        raise RuntimeError("No documents found in 'data/'. Add .txt files and restart.")

    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)
    with open(FINGERPRINT_FILE, "w", encoding="utf-8") as f:
        f.write(current_fp)
    logger.info("ChromaDB built and persisted.")
    return db


# ---------------------------------------------------------------------------
# Language detection & helpers (ported from app.py)
# ---------------------------------------------------------------------------

def detect_language(text: str, llm: ChatOpenAI) -> str:
    try:
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
        logger.warning(f"Language detection failed: {e}")
    return "danish"


def is_greeting_or_smalltalk(text: str) -> bool:
    if not text:
        return False
    normalized = re.sub(r"[^a-zA-ZæøåÆØÅ\s]", "", text).strip().lower()
    if len(normalized) <= 25 and any(w in normalized for w in [
        "hi", "hello", "hey", "hi there", "hey there", "good morning", "good evening",
        "hej", "hej med dig", "hejsa", "godmorgen", "godaften", "god aften", "hej hej",
    ]):
        return True
    return False


def augment_question_for_resources(question: str, language: str) -> str:
    q = question or ""
    normalized = q.lower()
    synonyms_danish = {
        "action": ["actions", "alarm", "alarmer", "checklist", "create checklist", "opret action", "lav action"],
        "dashboard": ["dashboards", "dashboard creator", "widget", "widgets", "overblik", "visualisering"],
        "insight": ["insights", "analyse", "rapport", "rapporter", "view", "master data"],
    }
    synonyms_english = {
        "action": ["actions", "alert", "alerts", "checklist", "create checklist", "create action"],
        "dashboard": ["dashboards", "widgets", "charts", "visualization"],
        "insight": ["insights", "analysis", "report", "reports", "view"],
    }
    mapping = synonyms_danish if language == "danish" else synonyms_english
    additions: List[str] = []
    for key, syns in mapping.items():
        if key in normalized or any(s in normalized for s in syns):
            additions.extend(syns + [key])
    if not additions:
        return q
    seen: set = set()
    extras = " ".join(s for s in additions if not (s in seen or seen.add(s)))  # type: ignore[func-returns-value]
    return f"{q} {extras}".strip()


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(a * a for a in vec_a))
    nb = math.sqrt(sum(b * b for b in vec_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def is_semantically_related(
    new_question: str,
    last_answer: str,
    embeddings: OpenAIEmbeddings,
    threshold: float = 0.50,
) -> bool:
    try:
        q_vec = embeddings.embed_query(new_question[:1000])
        a_vec = embeddings.embed_query((last_answer or "")[:1200])
        return cosine_similarity(q_vec, a_vec) >= threshold
    except Exception as e:
        logger.warning(f"Relatedness check failed: {e}")
        return False


def is_affirmative_reply(text: str) -> bool:
    if not text:
        return False
    t = re.sub(r"\s+", " ", text.strip().lower())
    if len(t) <= 30:
        affirm = {
            "yes", "yes.", "yes!", "y", "yep", "yup", "sure", "ok", "okay", "please",
            "yes please", "go ahead", "do it", "continue", "sounds good", "great",
            "ja", "ja.", "ja!", "jep", "ok", "okay", "gerne", "ja tak", "meget gerne",
            "kør", "fortsæt", "gør det", "lyder godt",
        }
        if t in affirm:
            return True
        if any(t.startswith(s) for s in ("yes", "ok", "okay", "sure", "please", "ja", "gerne")):
            return True
    return False


def parse_step_followup(text: str) -> Optional[int]:
    t = text.strip().lower()
    if len(t) > 60:
        return None
    match = re.search(r"(?:step|punkt|item|number|nr\.?|trin)\s*(\d+)", t)
    if match:
        return int(match.group(1))
    if is_affirmative_reply(text):
        words_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "first": 1,
            "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "en": 1, "to": 2, "tre": 3, "fire": 4, "fem": 5,
            "første": 1, "anden": 2, "tredje": 3, "fjerde": 4, "femte": 5,
        }
        for word, num in words_to_num.items():
            if word in t:
                return num
        numbers = re.findall(r"\d+", t)
        if len(numbers) == 1:
            return int(numbers[0])
    return None


def extract_step_content(answer_text: str, step_number: int) -> Optional[str]:
    if not answer_text:
        return None
    matches = re.findall(r"^\s*(\d+)\.\s+(.*)", answer_text, re.MULTILINE)
    for num_str, content in matches:
        if int(num_str) == step_number:
            return content.strip()
    return None


def extract_followup_question(text: str) -> Optional[str]:
    if not text:
        return None
    openings = [
        "skal jeg", "vil du have", "skal jeg anbefale",
        "would you like", "shall i", "do you want me",
    ]
    qs = re.findall(r"([^\n\r\?]{3,}?\?)", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = [
        q.strip() for q in qs
        if any(re.sub(r"\s+", " ", q.strip().lower()).startswith(o) for o in openings)
    ]
    if candidates:
        return candidates[-1]
    if qs:
        return qs[-1].strip()
    return None


def build_history_context(history: list, max_pairs: int = 1, char_limit: int = 1200) -> str:
    """Build a compact string from the last max_pairs user/assistant turns."""
    if not history:
        return ""
    recent = history[-(2 * max_pairs):]
    lines = []
    for m in recent:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "").strip().replace("\n\n", "\n")
        lines.append(f"{role}: {content}")
    joined = "\n".join(lines)
    if len(joined) > char_limit:
        joined = joined[-char_limit:]
    return joined


# ---------------------------------------------------------------------------
# Prompt templates (ported from app.py)
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are the Inact Now assistant. Answer only from the provided context. "
    "Do not invent features or steps not mentioned in the context. "
    "If the context does not contain enough information to answer, say so and suggest what the user could clarify."
)

PROMPT_TEMPLATES = {
    "danish": ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("human",
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
    ]),
    "english": ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("human",
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
    ]),
}


def render_resources_section(resource_docs: list, language: str) -> str:
    items = []
    seen: set = set()
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
        items.append({"title": title, "url": url})
        if len(items) >= 3:
            break
    return items


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not create_client or not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        logger.warning(f"Supabase init failed: {e}")
        return None


def append_local_chat_log(role: str, user_id: str, text: str, event: str):
    try:
        os.makedirs(DB_DIR, exist_ok=True)
        record = {
            "dt": datetime.utcnow().isoformat() + "Z",
            "role": role,
            "user_id": user_id,
            "event": event,
            "message": text or "",
            "app": "inact-bot",
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Local chat log failed: {e}")


def persist_log_to_supabase(role: str, user_id: str, text: str, event: str):
    try:
        sb = get_supabase_client()
        if not sb:
            return
        payload = {
            "dt": datetime.utcnow().isoformat() + "Z",
            "role": role,
            "user_id": user_id,
            "event": event,
            "message": text or "",
        }
        sb.table("chat_logs").insert(payload).execute()
    except Exception as e:
        logger.warning(f"Supabase insert failed: {e}")


def log_chat_message(role: str, user_id: str, text: str, event: str):
    log_system = os.environ.get("LOG_SYSTEM_MESSAGES", "0").strip().lower() in {"1", "true", "yes"}
    if event == "system" and not log_system:
        return
    if event in {"question", "answer"}:
        append_local_chat_log(role, user_id, text, event)
        persist_log_to_supabase(role, user_id, text, event)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

db: Optional[Chroma] = None
llm: Optional[ChatOpenAI] = None
embeddings_model: Optional[OpenAIEmbeddings] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, llm, embeddings_model
    logger.info("Starting up — initialising models and vector store…")
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name="gpt-4o",
    )
    db = init_vectorstore()
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: List[HistoryMessage] = []
    user_id: str = ""
    last_answer: str = ""


class ResourceLink(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    language: str
    resources: List[ResourceLink] = []


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()
    history = [m.model_dump() for m in request.history]
    user_id = request.user_id or str(uuid.uuid4())[:4]
    last_answer = request.last_answer

    log_chat_message("user", user_id, question, event="question")

    try:
        return await _process_chat(question, history, user_id, last_answer)
    except Exception as e:
        logger.exception("Chat request failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Check backend logs for full traceback."},
        )


async def _process_chat(
    question: str,
    history: list,
    user_id: str,
    last_answer: str,
) -> ChatResponse:
    # 1. Detect language
    language = detect_language(question, llm)

    # 2. Handle greetings / small talk without retrieval
    if is_greeting_or_smalltalk(question):
        answer = (
            "Hej! Hvordan kan jeg hjælpe dig i Inact Now i dag? Skriv gerne den funktion eller opgave, du arbejder med."
            if language == "danish"
            else "Hi! How can I help you in Inact Now today? Tell me the feature or task you're working on."
        )
        log_chat_message("assistant", user_id, answer, event="answer")
        return ChatResponse(answer=answer, language=language, resources=[])

    # 3. History & follow-up logic
    use_history = False
    if last_answer:
        use_history = is_semantically_related(question, last_answer, embeddings_model)

    history_context = build_history_context(history, max_pairs=2) if use_history else ""

    effective_question = question
    if use_history:
        step_num = parse_step_followup(question)
        step_content = extract_step_content(last_answer, step_num) if last_answer and step_num else None

        if step_content:
            effective_question = (
                f"Brugeren spørger ind til trin {step_num}: '{step_content}'. Uddyb venligst dette trin i detaljer, og forklar, hvordan det udføres i Inact Now."
                if language == "danish"
                else f"The user is asking for details on step {step_num}: '{step_content}'. Please elaborate on this specific step, explaining how to perform it in Inact Now."
            )
        elif last_answer and is_affirmative_reply(question):
            follow_up_q = extract_followup_question(last_answer)
            if follow_up_q:
                effective_question = (
                    f"Brugeren bekræfter opfølgning: '{follow_up_q}'. Fortsæt og leverer det efterspurgte i detaljer."
                    if language == "danish"
                    else f"The user confirmed the follow-up: '{follow_up_q}'. Please proceed to deliver what was offered in detail."
                )

    # Augment for resource retrieval
    if "resource" in effective_question.lower():
        effective_question = augment_question_for_resources(effective_question, language)

    # 4. Retrieve documents
    resource_filter = {"doc_type": "resource"}
    knowledge_filter = {"$and": [{"doc_type": "knowledge"}, {"language": language}]}

    try:
        resource_retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "filter": resource_filter, "score_threshold": 0.4},
        )
    except Exception:
        resource_retriever = db.as_retriever(search_kwargs={"k": 5, "filter": resource_filter})

    resource_docs = resource_retriever.invoke(effective_question)
    resource_context = "\n\n---\n\n".join(doc.page_content for doc in resource_docs)

    knowledge_retriever = db.as_retriever(search_kwargs={"k": 5, "filter": knowledge_filter})
    knowledge_docs = knowledge_retriever.invoke(effective_question)
    knowledge_context = "\n\n---\n\n".join(doc.page_content for doc in knowledge_docs)

    # 5. Nothing relevant found
    if not knowledge_docs and not resource_docs:
        answer = (
            "Jeg kan hjælpe med Inact Now. Hvilken funktion eller opgave vil du gerne løse?"
            if language == "danish"
            else "I can help with Inact Now. Which feature or task would you like help with?"
        )
        log_chat_message("assistant", user_id, answer, event="answer")
        return ChatResponse(answer=answer, language=language, resources=[])

    # 6. Generate answer
    chain = PROMPT_TEMPLATES[language] | llm
    response = chain.invoke({
        "resource_context": resource_context,
        "knowledge_context": knowledge_context,
        "question": effective_question,
        "history_context": history_context,
    })
    answer = response.content

    # 7. Build resource links (returned as structured data, not embedded markdown)
    resource_links = render_resources_section(resource_docs, language)

    log_chat_message("assistant", user_id, answer, event="answer")

    return ChatResponse(
        answer=answer,
        language=language,
        resources=[ResourceLink(**r) for r in resource_links],
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
