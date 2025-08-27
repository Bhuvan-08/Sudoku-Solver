import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model("improved_digit_cnn.h5")  # Make sure your model is saved as .keras

def predict_digit(img28x28, confidence_threshold=0.85, return_confidence=False):
    """Predicts a digit from a 28x28 image. Optionally returns confidence."""
    img = img28x28.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence < confidence_threshold:
        predicted_class = 0

    return (predicted_class, confidence) if return_confidence else predicted_class