from fastapi import FastAPI
from langchain.agents import create_agent
from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()
app = FastAPI()


# Define the structure of the data your API expects from users
class ChatRequest(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {"status": 200}

@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    agent = create_agent(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant named Bob. You are only allowed to answer questions.",
    )

    user_msg = chat_request.message
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]}
    )

    print(result["messages"][-1].content_blocks)
    
    return {"response": result["messages"][-1].content_blocks}
