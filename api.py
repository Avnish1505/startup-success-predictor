from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("startup_model.pkl")

@app.get("/")
def home():
    """Health check endpoint to verify the API is running."""
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(funding: float, experience: int, team: int, market: int):
    """
    Predicts the startup success probability.
    
    Args:
        funding (float): Total funding amount.
        experience (int): Total years of founders' experience.
        team (int): Size of the team.
        market (int): Market size indicator (0=Small, 1=Medium, 2=Large).
    """
    # Ensure correct feature order: funding, team size, experience, market
    data = np.array([[funding, team, experience, market]])
    prediction = model.predict_proba(data)[0][1]
    return {"success_probability": float(prediction)}