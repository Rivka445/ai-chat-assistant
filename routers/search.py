from fastapi import APIRouter
from models import SearchRequest
from embeddings import get_embedding, cosine_similarity

router = APIRouter()

@router.post('/search')
async def search(req: SearchRequest):
    if not req.products:
        return {'results': []}

    query_embedding = get_embedding(req.query)

    scored = []
    for p in req.products:
        name = p.get('name') or p.get('Name', '')
        desc = p.get('description') or p.get('Description', '')
        product_embedding = get_embedding(f'{name} {desc}')
        score = cosine_similarity(query_embedding, product_embedding)
        scored.append({**p, 'score': round(score, 3)})

    results = sorted(scored, key=lambda x: x['score'], reverse=True)
    return {'results': results[:req.top_k]}
