from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.services.predict import predict_image

router = APIRouter()

UPLOAD_DIR = "temp"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(file_path)

    os.remove(file_path)

    return result