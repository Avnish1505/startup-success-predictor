import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "model" / "startup_model.pkl",
    BASE_DIR / "startup_model.pkl",
]

model = None
for model_path in MODEL_CANDIDATES:
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        break

if model is None:
    raise FileNotFoundError(
        "startup_model.pkl not found. Expected at model/startup_model.pkl or startup_model.pkl"
    )

def predict_startup(funding, team_size, experience, market):

    features = np.array([[funding, team_size, experience, market]])

    probability = model.predict_proba(features)[0][1]

    return round(probability * 100, 2)