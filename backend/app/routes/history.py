from fastapi import APIRouter
from app.config.db import db

router = APIRouter()

@router.get("/history")
async def get_history():

    predictions = []

    cursor = db.predictions.find().sort("_id", -1)

    async for document in cursor:

        document["_id"] = str(document["_id"])

        predictions.append(document)

    return predictions