import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Setup paths and load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

print("--- System Startup Log ---")
print(f"Project Root Directory: {BASE_DIR}")

if not os.getenv("MONGO_URI") or not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: Missing keys inside your .env file!")
    sys.exit(1)

# 2. Establish connections to MongoDB Atlas
try:
    client = MongoClient(os.getenv("MONGO_URI"))
    # Force a real round-trip so connection/auth errors surface here instead of on first query
    client.admin.command("ping")
    DB_NAME = "RAG_Project"
    COLLECTION_NAME = "RAG_embeddings"
    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
    INDEX_NAME = "vector_index"
    print("✅ Successfully established MongoDB handshake link.")
except Exception as e:
    print(f"❌ MongoDB Connection Failure: {e}")
    sys.exit(1)

# 3. Synchronize vector embeddings structure
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
    index_name=INDEX_NAME
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 4. Bind the contextual chat logic loops with memory mapping arrays
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history. Do NOT answer it, just rewrite it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user query relying strictly on the book context below.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
rag_chain = create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt))

# 5. Initialize FastAPI Server Instance
app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
print("✅ Templates engine mounted cleanly.")

chat_history = []

class UserQuery(BaseModel):
    message: str

# HTTP Route 1: Serves the web-page dashboard UI panel safely
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    print("📥 GET request received for dashboard UI index panel.")
    return templates.TemplateResponse(request=request, name="index.html")

# HTTP Route 2: Receives chat inquiries, requests cloud vectors, and returns responses
@app.post("/chat")
async def chat_endpoint(payload: UserQuery):
    global chat_history
    print(f"📩 Chat API Request Received: '{payload.message}'")
    
    try:
        response = await rag_chain.ainvoke({
            "input": payload.message,
            "chat_history": chat_history
        })
        
        chat_history.append(HumanMessage(content=payload.message))
        chat_history.append(AIMessage(content=response["answer"]))
        
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
            
        return {"answer": response["answer"]}
    except Exception as e:
        print(f"❌ Error during RAG chain execution: {e}")
        return {"answer": f"An internal background tracking error occurred: {str(e)}"}
