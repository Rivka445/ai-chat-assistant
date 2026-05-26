from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routers import chat, search
from logger import logger
import time
import os

load_dotenv()

origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:4200').split(',')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.middleware('http')
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000)
    logger.info(f'{request.method} {request.url.path} → {response.status_code} ({duration}ms)')
    return response

app.include_router(chat.router)
app.include_router(search.router)
