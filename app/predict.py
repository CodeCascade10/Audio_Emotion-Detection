import numpy as np
from app.model_loader import model, label_encoder
from app.utils import extract_features

def predict_audio(file_path):
    features = extract_features(file_path)

    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)

    label = label_encoder.inverse_transform([predicted_index])[0]

    return label