import tensorflow as tf
import numpy as np
from preprocess import preprocess_image

# Load model only ONCE (important for performance)
model = tf.keras.models.load_model("best_model.keras")

conditions = ["Dryness", "Redness", "Acne", "Dark Circles", "Oily Zones"]

def predict_skin_condition(image_path: str):
    img = preprocess_image(image_path)

    if img is None:
        return {"error": "No face detected"}

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]

    results = {}
    active_conditions = []

    for i, prob in enumerate(prediction):
        results[conditions[i]] = float(round(prob, 2))
        if prob >= 0.5:
            active_conditions.append(conditions[i])

    return {
        "probabilities": results,
        "active_conditions": active_conditions
    }