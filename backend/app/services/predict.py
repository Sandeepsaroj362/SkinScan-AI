import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================

model = load_model(
    "app/ml/skin_disease_model.keras"
)

# =========================
# LOAD CLASS NAMES
# =========================

with open("app/ml/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# =========================
# IMAGE SETTINGS
# =========================

IMG_SIZE = (224, 224)

# =========================
# PREDICT FUNCTION
# =========================

def predict_image(img_path):

    img = image.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_class = class_names[
        np.argmax(predictions[0])
    ]

    confidence = float(
        np.max(predictions[0]) * 100
    )

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2)
    }