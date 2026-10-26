import os
from dotenv import load_dotenv
from pymongo import MongoClient
#from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_unstructured import UnstructuredMarkdownLoader
#from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize MongoDB

client = MongoClient(os.environ["MONGO_URI"])
    
DB_NAME = "RAG_Project"
COLLECTION_NAME = "RAG_embeddings"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
INDEX_NAME = "vector_index"


# Load and Chunk the Markdown File
FILE_PATH = "../data/alice_in_wonderland.md"

loader = UnstructuredMarkdownLoader(FILE_PATH)
raw_documents = loader.load()
text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(raw_documents)

print(f"Total text chunks created: {len(docs)}")

        
   

