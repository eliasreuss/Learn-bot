# streamlit_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="Inact Learn Chatbot", page_icon="💬")

#Font og farve
st.markdown("""
    <style>
    /* --- Importer og anvend Poppins font --- */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');
    
    html, body, [class*="st-"], [class*="css-"] {
        font-family: 'Poppins', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    return Chroma(persist_directory="chromadb", embedding_function=embeddings)

@st.cache_resource
def get_llm():
    return ChatOpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-3.5-turbo")

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

# Initialisering
db = get_vectorstore()
llm = get_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Brug st.markdown for at matche designet
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');
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
    <div class="custom-header">Hello, <span class="gertrud-orange">Gertrud</span>. How can I help you today?</div>
    """,
    unsafe_allow_html=True
)

# Chat logik med placeholder

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("How do i create an Insight in Inact Now?"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Tænker...") 

        resource_retriever = db.as_retriever(search_kwargs={"k": 5, "filter": {"doc_type": "resource"}})
        resource_docs = resource_retriever.get_relevant_documents(user_question)
        resource_context = "\n\n---\n\n".join([doc.page_content for doc in resource_docs])

        knowledge_retriever = db.as_retriever(search_kwargs={"k": 3, "filter": {"doc_type": "knowledge"}})
        knowledge_docs = knowledge_retriever.get_relevant_documents(user_question)
        knowledge_context = "\n\n---\n\n".join([doc.page_content for doc in knowledge_docs])
        
        chain = prompt_template | llm
        response = chain.invoke({
            "resource_context": resource_context,
            "knowledge_context": knowledge_context,
            "question": user_question
        })
        answer = response.content
        
        message_placeholder.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})