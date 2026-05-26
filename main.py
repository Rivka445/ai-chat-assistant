from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routers import chat, search
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

app.include_router(chat.router)
app.include_router(search.router)
