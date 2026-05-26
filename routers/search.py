from fastapi import APIRouter, Depends
from models import SearchRequest
from embeddings import get_embedding, cosine_similarity
from rate_limiter import check_rate_limit
from logger import logger

router = APIRouter()

@router.post('/search')
async def search(req: SearchRequest, _=Depends(check_rate_limit)):
    if not req.products:
        return {'results': []}
    logger.info(f'Search request: "{req.query}"')

    query_embedding = get_embedding(req.query)

    products = req.products[:50]  # cap to avoid excessive API calls
    scored = []
    for p in products:
        name = p.get('name') or p.get('Name', '')
        desc = p.get('description') or p.get('Description', '')
        product_embedding = get_embedding(f'{name} {desc}')
        score = cosine_similarity(query_embedding, product_embedding)
        scored.append({**p, 'score': round(score, 3)})

    results = sorted(scored, key=lambda x: x['score'], reverse=True)
    return {'results': results[:req.top_k]}
