import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. Generate synthetic dataset (Features: Funding, Team Size, Experience, Market)
# Market encoding: 0=Small, 1=Medium, 2=Large
X = [
    [10000, 2, 1, 0],     # Low funding, small team, low exp -> Fail
    [500000, 5, 3, 1],    # Decent funding, optimal team, moderate exp -> Pass
    [2000000, 10, 5, 2],  # High funding, large team, high exp -> Pass
    [50000, 1, 0, 0],     # Low funding, solo founder, zero exp -> Fail
    [100000, 4, 2, 1],    # Average startup -> Fail
    [3000000, 15, 10, 2]  # Super startup (Unicorn traits) -> Pass
]

# 2. Define corresponding Labels (0 = Fail, 1 = Pass)
y = [0, 1, 1, 0, 0, 1]

# 3. Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 4. Save the newly trained model (overwrites existing file)
joblib.dump(model, "startup_model.pkl")
print("✅ Model trained and saved successfully as startup_model.pkl!")