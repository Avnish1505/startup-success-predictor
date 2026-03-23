import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "model" / "startup_model.pkl",
    BASE_DIR / "startup_model.pkl",
]

model = None
model_error = None

try:
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            model = joblib.load(model_path)
            break
    if model is None:
        model_error = "startup_model.pkl file not found."
except ModuleNotFoundError as e:
    model_error = f"Library missing: {e}. Please run 'pip install scikit-learn' in your terminal."
except Exception as e:
    model_error = f"Model load error: {e}"

def predict_startup(funding, team_size, experience, market):
    """
    Predicts the success probability of a startup using the trained ML model.
    
    Args:
        funding (float): Total funding secured.
        team_size (int): Number of team members.
        experience (int): Combined years of experience of the founders.
        market (int): Market size (0: Small, 1: Medium, 2: Large).
    Returns:
        float: Probability of success in percentage.
    """
    if model_error:
        raise RuntimeError(model_error)

    features = np.array([[funding, team_size, experience, market]])

    probability = model.predict_proba(features)[0][1]

    return round(probability * 100, 2)