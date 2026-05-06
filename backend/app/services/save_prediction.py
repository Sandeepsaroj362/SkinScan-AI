from app.config.db import db

async def save_prediction(data):

    prediction_data = {
        "image_url": data["image_url"],
        "predicted_class": data["predicted_class"],
        "confidence": data["confidence"]
    }

    result = await db.predictions.insert_one(
        prediction_data
    )

    return str(result.inserted_id)