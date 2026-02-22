import tensorflow as tf
import pickle

MODEL_PATH = "models/sentiment_cnn_model.h5"
ENCODER_PATH = "models/le.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)