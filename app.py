import numpy as np
import librosa
import pickle
import os

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("sentiment_cnn_model.h5")
print("MODEL INPUT SHAPE:", model.input_shape)

# Load label encoder
with open("le.pkl", "rb") as f:
    le = pickle.load(f)


# Feature extraction (must match training time)
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["audio"]

        if file:
            file_path = "temp.wav"
            file.save(file_path)

            # Extract features
            features = extract_features(file_path)

            # Reshape according to CNN input
            features = features.reshape(1, 40, 1, 1)


            # Predict
            pred = model.predict(features)
            predicted_label = np.argmax(pred)
            emotion = le.inverse_transform([predicted_label])

            prediction = emotion[0]

            os.remove(file_path)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

