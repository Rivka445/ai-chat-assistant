from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── SYSTEM PROMPT ────────────────────────────────────────────

with open(os.path.join(os.path.dirname(__file__), "prompt.md"), encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


# ── DATA MODELS ─────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    products: list = []

# ── CHAT ENDPOINT ───────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):

    # Build conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Add history
    for m in req.history:
        messages.append({
            "role": m.role,
            "content": m.content
        })

    # Add current message
    messages.append({
        "role": "user",
        "content": req.message
    })

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=400,
        temperature=0.6
    )

    return {
        "reply": response.choices[0].message.content
    }
