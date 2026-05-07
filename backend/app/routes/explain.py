from fastapi import APIRouter

from pydantic import BaseModel

from app.services.gemini_service import (
    generate_explanation
)

router = APIRouter()

class ExplanationRequest(BaseModel):
    disease: str
    confidence: float

@router.post("/explain")

async def explain(
    data: ExplanationRequest
):

    explanation = generate_explanation(
            data.disease,
            data.confidence
        )

    return {
        "explanation": explanation
    }