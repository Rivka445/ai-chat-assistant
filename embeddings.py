import numpy as np
import json
import hashlib
import redis
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

_redis = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

def get_embedding(text: str) -> list[float]:
    key = 'emb:' + hashlib.md5(text.encode()).hexdigest()
    cached = _redis.get(key)
    if cached:
        return json.loads(cached)

    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    embedding = response.data[0].embedding
    _redis.set(key, json.dumps(embedding), ex=60 * 60 * 24 * 7)  # 7 days
    return embedding

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
