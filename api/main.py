import os
from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
IFOREST_PATH = os.path.join(MODEL_DIR, "iforest.pkl")
AUTOENCODER_PATH = os.path.join(MODEL_DIR, "autoencoder.keras")

print("Loading models from:", MODEL_DIR)


scaler = joblib.load(SCALER_PATH)
iforest = joblib.load(IFOREST_PATH)
autoencoder = tf.keras.models.load_model(AUTOENCODER_PATH)



@app.get("/")
def home():
    return {"status": "API is running"}


@app.post("/predict")
def predict(features: dict):
    x = np.array([list(features.values())], dtype=float)
    x_scaled = scaler.transform(x)

    if_label = int(iforest.predict(x_scaled)[0])

    x_recon = autoencoder.predict(x_scaled)
    ae_error = float(np.mean((x_scaled - x_recon) ** 2))
    ae_label = int(-1 if ae_error > 0.5 else 1)

    strong = (if_label == -1 and ae_label == -1)

    return {
        "iforest_label": if_label,
        "autoencoder_label": ae_label,
        "autoencoder_error": ae_error,
        "ensemble_strong": bool(strong)
    }