import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

load_dotenv()

app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175"]

# Update CORS configuration to allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,  # Allow credentials for cross-origin requests
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    question: str

def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    db = Chroma(persist_directory="chromadb", embedding_function=embeddings)
    return db

def load_chain(db):
    llm = OpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Du er en hjælpsom assistent for Inact. "
            "Svar på spørgesmålet ved hjælp af informationen i konteksten - det er fra Inacts help center. Giv step by step guides til hvordan de skal gøre det i Inact Now softwaren\n"
            "Kontekst:\n{context}\n\nSpørgsmål: {question}\nSvar:"
        ),
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

# Initialize the database and chain at startup
db = load_vectorstore()
chain = load_chain(db)

@app.post("/api/chat")
async def chat(query: Query):
    try:
        result = chain.run(query.question)
        return JSONResponse(
            content={"answer": result},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

@app.options("/api/chat")
async def options_handler(request: Request):
    print("OPTIONS request received")
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
