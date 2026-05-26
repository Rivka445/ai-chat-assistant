from fastapi import APIRouter
from openai import OpenAI
from models import ChatRequest
import os

router = APIRouter()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

with open(os.path.join(os.path.dirname(__file__), '..', 'prompt.md'), encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()

@router.post('/chat')
async def chat(req: ChatRequest):
    if req.products:
        catalog_lines = []
        for p in req.products:
            name   = p.get('name')      or p.get('Name',      '')
            price  = p.get('basePrice') or p.get('BasePrice', '')
            desc   = p.get('description') or p.get('Description', '')
            active = p.get('isActive') if 'isActive' in p else p.get('IsActive', True)
            stock  = 'in stock' if active else 'out of stock'
            catalog_lines.append(f'- {name} (${price}) [{stock}]: {desc}')
        catalog = '\n'.join(catalog_lines)
        full_prompt = SYSTEM_PROMPT + f'\n\nAvailable products:\n{catalog}\n\nOnly recommend products from this list.'
    else:
        full_prompt = SYSTEM_PROMPT

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
