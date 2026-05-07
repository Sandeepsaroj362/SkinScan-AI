from fastapi import APIRouter, UploadFile, File
from app.services.gradcam import create_gradcam_image
import shutil
import os
import cloudinary.uploader

from app.services.predict import predict_image
from app.services.predict import (
    predict_image,
    model,
)
from app.services.save_prediction import save_prediction

router = APIRouter()

UPLOAD_DIR = "temp"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    # =========================
    # Save Temp File
    # =========================

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # =========================
    # Upload To Cloudinary
    # =========================

    cloudinary_result = cloudinary.uploader.upload(
        file_path
    )

    image_url = cloudinary_result["secure_url"]

    # =========================
    # ML Prediction
    # =========================

    prediction_result = predict_image(file_path)
    gradcam_image = create_gradcam_image(
    file_path,
    model
)
    # =========================
    # Save To MongoDB
    # =========================

    prediction_data = {
        "image_url": image_url,
        "predicted_class": prediction_result["predicted_class"],
        "confidence": prediction_result["confidence"],
        "gradcam_image": gradcam_image,
    }

    await save_prediction(prediction_data)

    # =========================
    # Cleanup Temp File
    # =========================

    os.remove(file_path)

    return prediction_data