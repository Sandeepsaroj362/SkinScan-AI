from fastapi import APIRouter, UploadFile, File
import cloudinary.uploader

from app.config.cloudinary import *

router = APIRouter()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    result = cloudinary.uploader.upload(file.file)

    return {
        "image_url": result["secure_url"]
    }