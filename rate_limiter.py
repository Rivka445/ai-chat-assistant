import time
import redis
import os
from fastapi import HTTPException, Request

_redis = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

LIMIT = int(os.getenv('RATE_LIMIT', 20))   # requests
WINDOW = int(os.getenv('RATE_WINDOW', 60)) # seconds

def check_rate_limit(request: Request):
    ip = request.client.host
    key = f'rl:{ip}'
    now = time.time()

    pipe = _redis.pipeline()
    pipe.zremrangebyscore(key, 0, now - WINDOW)
    pipe.zadd(key, {str(now): now})
    pipe.zcard(key)
    pipe.expire(key, WINDOW)
    results = pipe.execute()

    count = results[2]
    if count > LIMIT:
        raise HTTPException(status_code=429, detail=f'Too many requests — limit is {LIMIT} per {WINDOW}s')
