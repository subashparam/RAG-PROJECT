import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# 1. Load configuration from your local hidden .env file
load_dotenv()


# 2. Establish connections to your MongoDB Atlas Cluster

client = MongoClient(os.environ["MONGO_URI"])
    
DB_NAME = "RAG_Project"
COLLECTION_NAME = "RAG_embeddings"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
INDEX_NAME = "vector_index"

# 3. Initialize the identical embedding model mapping layout
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 4. Connect straight to your existing cloud index maps (No text splitting or uploading!)
vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
    index_name=INDEX_NAME
)


# 5. Wire up the prompt template and retriever pipeline layers
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt = (
    "Answer the user query relying strictly on the book context below.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# 6. Interactive Terminal Chat Loop Setup
print("\n Connected to your Alice in Wonderland Database! Type your question below (or type 'quit' to exit):")

while True:
    user_query = input("\nYou: ")
    if user_query.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break
        
    if not user_query.strip():
        continue
        
    print("Searching database context and generating response...")
    response = rag_chain.invoke({"input": user_query})
    
    print("\nAI:", response["answer"])