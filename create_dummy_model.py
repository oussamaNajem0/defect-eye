import os
import joblib
import pandas as pd
from xgboost import XGBClassifier

# Create the models directory
os.makedirs("models", exist_ok=True)

# Create some dummy training data matching our API schema
df = pd.DataFrame({
    "loc": [10, 200, 50, 300],
    "cyclomatic_complexity": [1, 15, 5, 25],
    "halstead_volume": [100, 1000, 300, 1500],
    "defects": [0, 1, 0, 1]
})

X = df.drop("defects", axis=1)
y = df["defects"]

# Train and save the model
print("Training dummy model...")
model = XGBClassifier()
model.fit(X, y)

joblib.dump(model, "models/xgboost_model.pkl")
print("✅ Model successfully saved to models/xgboost_model.pkl!")