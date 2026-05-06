from pydantic import BaseModel
from typing import Optional

class Prediction(BaseModel):
    image_url: str
    predicted_class: str
    confidence: float