from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router as predict_router
from app.config.db import db
from app.routes.upload import router as upload_router
from app.routes.history import router as history_router
from app.routes.explain import (
    router as explain_router
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(predict_router)
app.include_router(history_router)
app.include_router(explain_router)

@app.get("/")
async def home():
    collections = await db.list_collection_names()

    return {
        "message": "SkinScanAI Backend Running",
        "collections": collections
    }