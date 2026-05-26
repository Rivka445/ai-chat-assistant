from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

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

# ── EMBEDDING FUNCTIONS ──────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ── SEARCH ENDPOINT ──────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    products: list
    top_k: int = 5

@app.post('/search')
async def search(req: SearchRequest):
    if not req.products:
        return {'results': []}

    query_embedding = get_embedding(req.query)

    scored = []
    for p in req.products:
        name = p.get('name') or p.get('Name', '')
        desc = p.get('description') or p.get('Description', '')
        product_embedding = get_embedding(f"{name} {desc}")
        score = cosine_similarity(query_embedding, product_embedding)
        scored.append({**p, 'score': round(score, 3)})

    results = sorted(scored, key=lambda x: x['score'], reverse=True)
    return {'results': results[:req.top_k]}

# ── CHAT ENDPOINT ───────────────────────────────────────────

@app.post('/chat')
async def chat(req: ChatRequest):
    # Build a product catalog string from the list .NET sent
    if req.products:
        catalog_lines = []
        for p in req.products:
            name  = p.get('name')  or p.get('Name',  '')
            price = p.get('basePrice') or p.get('BasePrice', '')
            desc  = p.get('description') or p.get('Description', '')
            active = p.get('isActive') if 'isActive' in p else p.get('IsActive', True)
            stock = 'in stock' if active else 'out of stock'
            line = f"- {name} (${price}) [{stock}]: {desc}"
            catalog_lines.append(line)
        catalog = '\n'.join(catalog_lines)
        full_prompt = SYSTEM_PROMPT + f'\n\nAvailable products:\n{catalog}\n\nOnly recommend products from this list.'
    else:
        full_prompt = SYSTEM_PROMPT

    # Build conversation
    messages = [{'role': 'system', 'content': full_prompt}]
    for m in req.history:
        messages.append({'role': m.role, 'content': m.content})
    messages.append({'role': 'user', 'content': req.message})

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        max_tokens=400,
        temperature=0.5
    )
    return {'reply': response.choices[0].message.content}

